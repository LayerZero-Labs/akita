//! Canonical grinding-plan derivation from public schedule geometry.

use crate::{
    multilinear_point_loss_factor, nominal_challenge_capacity_bits,
    polynomial_identity_loss_factor, powers_batch_loss_factor, ring_switch_alpha_loss_factor,
    CommittedGroupParams, DigitRangePlan, FoldSchedule, GrindingPlan, GrindingRun, GrindingSite,
    OpeningClaimsLayout, PolynomialGroupLayout, SumcheckProtocol,
};
use akita_error::AkitaError;

/// Successor shape needed to price one planner edge.
#[doc(hidden)]
#[derive(Clone, Copy)]
pub enum GrindingPlanSuccessor<'a> {
    Recursive(&'a CommittedGroupParams),
    Terminal(&'a crate::TerminalFoldParams),
}

/// Derive the only accepted grinding plan from field metadata and public protocol shape.
pub fn derive_transcript_grinding_plan_from_public_shape(
    schedule: &FoldSchedule,
    root_layout: &OpeningClaimsLayout,
    modulus_bits: u32,
    extension_degree: usize,
) -> Result<GrindingPlan, AkitaError> {
    schedule.validate_structure()?;
    schedule.validate_nonterminal_opening_execution(extension_degree)?;
    derive_transcript_grinding_plan(schedule, root_layout, modulus_bits, extension_degree)
}

/// Price a planner fold sequence with the canonical query schedule.
///
/// A recursive suffix may legally start with a raw payload, so it is not a
/// standalone [`FoldSchedule`] and must not pass root-only structure checks.
/// The planner separately validates candidate geometry before calling this
/// pricing entry point.
#[doc(hidden)]
pub fn transcript_grinding_nonce_bits_for_planner_candidate(
    schedule: &FoldSchedule,
    root_layout: &OpeningClaimsLayout,
    modulus_bits: u32,
    extension_degree: usize,
) -> Result<usize, AkitaError> {
    Ok(
        derive_transcript_grinding_plan(schedule, root_layout, modulus_bits, extension_degree)?
            .total_nonce_bits(),
    )
}

/// Price one planner edge using the canonical query builders.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn transcript_grinding_nonce_bits_for_planner_edge(
    params: &CommittedGroupParams,
    output_witness_len: usize,
    layout: &OpeningClaimsLayout,
    successor: GrindingPlanSuccessor<'_>,
    modulus_bits: u32,
    extension_degree: usize,
    level: u32,
) -> Result<usize, AkitaError> {
    layout.check()?;
    if extension_degree == 0 || !extension_degree.is_power_of_two() {
        return Err(AkitaError::InvalidSetup(
            "grinding extension degree must be a nonzero power of two".into(),
        ));
    }
    let capacity = nominal_challenge_capacity_bits(modulus_bits, extension_degree)?;
    let mut runs = Vec::new();
    let rounds = append_nonterminal(
        &mut runs,
        capacity,
        extension_degree,
        level,
        params,
        output_witness_len,
        layout,
        successor,
    )?;
    if let GrindingPlanSuccessor::Terminal(terminal) = successor {
        append_terminal(
            &mut runs,
            capacity,
            extension_degree,
            level.checked_add(1).ok_or_else(|| {
                AkitaError::InvalidSetup("terminal grinding level overflow".into())
            })?,
            rounds,
            terminal,
        )?;
    }
    Ok(GrindingPlan::new(runs, capacity)?.total_nonce_bits())
}

fn derive_transcript_grinding_plan(
    schedule: &FoldSchedule,
    root_layout: &OpeningClaimsLayout,
    modulus_bits: u32,
    extension_degree: usize,
) -> Result<GrindingPlan, AkitaError> {
    root_layout.check()?;
    if extension_degree == 0 || !extension_degree.is_power_of_two() {
        return Err(AkitaError::InvalidSetup(
            "grinding extension degree must be a nonzero power of two".into(),
        ));
    }
    let capacity = nominal_challenge_capacity_bits(modulus_bits, extension_degree)?;
    let mut runs = Vec::new();

    let mut predecessor_rounds = append_nonterminal(
        &mut runs,
        capacity,
        extension_degree,
        0,
        &schedule.root.params,
        schedule.root.output_witness_len,
        root_layout,
        schedule.recursive_folds.first().map_or(
            GrindingPlanSuccessor::Terminal(&schedule.terminal),
            |step| GrindingPlanSuccessor::Recursive(&step.params),
        ),
    )?;

    for (index, fold) in schedule.recursive_folds.iter().enumerate() {
        let layout = recursive_layout(predecessor_rounds, &fold.params)?;
        let successor = schedule.recursive_folds.get(index + 1).map_or(
            GrindingPlanSuccessor::Terminal(&schedule.terminal),
            |step| GrindingPlanSuccessor::Recursive(&step.params),
        );
        predecessor_rounds = append_nonterminal(
            &mut runs,
            capacity,
            extension_degree,
            usize_to_u32(index + 1, "grinding level")?,
            &fold.params,
            fold.output_witness_len,
            &layout,
            successor,
        )?;
    }

    append_terminal(
        &mut runs,
        capacity,
        extension_degree,
        usize_to_u32(
            schedule.recursive_folds.len() + 1,
            "terminal grinding level",
        )?,
        predecessor_rounds,
        &schedule.terminal,
    )?;
    GrindingPlan::new(runs, capacity)
}

#[allow(clippy::too_many_arguments)]
fn append_nonterminal(
    runs: &mut Vec<GrindingRun>,
    capacity: u32,
    extension_degree: usize,
    level: u32,
    params: &CommittedGroupParams,
    output_witness_len: usize,
    layout: &OpeningClaimsLayout,
    successor: GrindingPlanSuccessor<'_>,
) -> Result<usize, AkitaError> {
    let opening_method = params.uniform_opening_method(layout)?;
    if opening_method.requires_extension_opening_reduction(extension_degree) {
        append_eor(runs, capacity, extension_degree, level, layout)?;
    }

    if layout.requires_row_batch_challenge() {
        runs.push(GrindingRun::proof_of_work(
            GrindingSite::EvaluationBatch { level },
            1,
            capacity,
        )?);
    }

    runs.push(GrindingRun::fold_response(level));
    append_fold_queries(runs, level, params, layout)?;

    let alpha_loss = (0..layout.num_groups()).try_fold(1u64, |largest, group_index| {
        let group = params.group_params(layout, group_index)?;
        Ok::<_, AkitaError>(largest.max(ring_switch_alpha_loss_factor(
            group.opening_method(),
            group.inner_commit_matrix_params().ring_dimension(),
        )?))
    })?;
    runs.push(GrindingRun::proof_of_work(
        GrindingSite::RingSwitchAlpha { level },
        alpha_loss,
        capacity,
    )?);

    let (successor_d, successor_opening_vars) = match successor {
        GrindingPlanSuccessor::Recursive(successor) => {
            (successor.d_a(), successor.recursive_opening_num_vars()?)
        }
        GrindingPlanSuccessor::Terminal(terminal) => {
            (terminal.d_a(), terminal.recursive_opening_num_vars()?)
        }
    };
    let tau0_width = params
        .relation_address_geometry(layout, extension_degree, successor_d, output_witness_len)?
        .relation_point_variable_count();
    if tau0_width > successor_opening_vars {
        return Err(AkitaError::InvalidSetup(
            "grinding Stage 2 point exceeds successor opening width".into(),
        ));
    }
    runs.push(GrindingRun::proof_of_work(
        GrindingSite::Tau0Point { level },
        multilinear_point_loss_factor(tau0_width)?,
        capacity,
    )?);
    runs.push(GrindingRun::proof_of_work(
        GrindingSite::Tau1Point { level },
        multilinear_point_loss_factor(params.relation_row_index_num_vars(layout)?)?,
        capacity,
    )?);

    let rounds = tau0_width;
    let basis = 1usize
        .checked_shl(params.open().digits.log_basis)
        .ok_or_else(|| AkitaError::InvalidSetup("digit-range basis exceeds usize".into()))?;
    let range = DigitRangePlan::new(basis)?;
    let (stages, norm) =
        range.proof_shapes_for_route(rounds, params.inner().matrix.security_route())?;
    for (stage_index, stage_shape) in stages.iter().enumerate() {
        let stage = usize_to_u32(stage_index, "Stage 1 grinding stage")?;
        let full_round_degree =
            stage_shape.sumcheck_proof.1.checked_add(1).ok_or_else(|| {
                AkitaError::InvalidSetup("Stage 1 full round degree overflow".into())
            })?;
        for round in 0..stage_shape.sumcheck_proof.0 {
            append_sumcheck(
                runs,
                capacity,
                SumcheckProtocol::Stage1,
                level,
                stage,
                round,
                full_round_degree,
            )?;
        }
        if stage_shape.child_claims > 0 {
            runs.push(GrindingRun::proof_of_work(
                GrindingSite::Stage1InterstageBatch { level, stage },
                powers_batch_loss_factor(stage_shape.child_claims)?,
                capacity,
            )?);
        }
    }
    if let Some(norm) = norm {
        if norm.subclaims > 0 {
            runs.push(GrindingRun::proof_of_work(
                GrindingSite::L2SubclaimBatch { level },
                powers_batch_loss_factor(norm.subclaims)?,
                capacity,
            )?);
        }
        runs.push(GrindingRun::proof_of_work(
            GrindingSite::L2NormMerge { level },
            1,
            capacity,
        )?);
        for (round, &degree) in norm.sumcheck.iter().enumerate() {
            append_sumcheck(
                runs,
                capacity,
                SumcheckProtocol::PhysicalL2,
                level,
                0,
                round,
                degree,
            )?;
        }
        runs.push(GrindingRun::proof_of_work(
            GrindingSite::L2VirtualBatch { level },
            powers_batch_loss_factor(norm.virtual_evaluations)?,
            capacity,
        )?);
    }
    if params.payload_mode.is_compressed() {
        runs.push(GrindingRun::proof_of_work(
            GrindingSite::CompressionBinary { level },
            1,
            capacity,
        )?);
    }
    runs.push(GrindingRun::proof_of_work(
        GrindingSite::Stage2Batch { level },
        1,
        capacity,
    )?);
    for round in 0..rounds {
        append_sumcheck(runs, capacity, SumcheckProtocol::Stage2, level, 0, round, 3)?;
    }
    if let GrindingPlanSuccessor::Recursive(successor) = successor {
        if let Some(prefix) = successor.setup_prefix() {
            for round in 0..setup_prefix_sumcheck_rounds(prefix)? {
                append_sumcheck(runs, capacity, SumcheckProtocol::Stage3, level, 0, round, 2)?;
            }
        }
    }
    Ok(rounds)
}

fn append_terminal(
    runs: &mut Vec<GrindingRun>,
    capacity: u32,
    extension_degree: usize,
    level: u32,
    predecessor_rounds: usize,
    terminal: &crate::TerminalFoldParams,
) -> Result<(), AkitaError> {
    let layout = OpeningClaimsLayout::new(predecessor_rounds, 1)?;
    if extension_degree > 1 {
        append_eor(runs, capacity, extension_degree, level, &layout)?;
    }
    runs.push(GrindingRun::fold_response(level));
    runs.push(GrindingRun::fold_challenge_group(
        level,
        0,
        usize_to_u64(terminal.blocks.live_blocks, "terminal fold coordinates")?,
    )?);
    Ok(())
}

fn append_fold_queries(
    runs: &mut Vec<GrindingRun>,
    level: u32,
    params: &CommittedGroupParams,
    layout: &OpeningClaimsLayout,
) -> Result<(), AkitaError> {
    for (group_index, group_layout) in layout.groups().iter().enumerate() {
        let group = usize_to_u32(group_index, "fold challenge group")?;
        let params = params.group_params(layout, group_index)?;
        let multiplicity = group_layout
            .num_polynomials()
            .checked_mul(params.num_live_blocks())
            .ok_or_else(|| AkitaError::InvalidSetup("fold coordinate count overflow".into()))?;
        runs.push(GrindingRun::fold_challenge_group(
            level,
            group,
            usize_to_u64(multiplicity, "fold coordinate count")?,
        )?);
    }
    Ok(())
}

fn append_eor(
    runs: &mut Vec<GrindingRun>,
    capacity: u32,
    extension_degree: usize,
    level: u32,
    layout: &OpeningClaimsLayout,
) -> Result<(), AkitaError> {
    let split_bits = extension_degree.trailing_zeros() as usize;
    if split_bits > layout.max_num_vars() {
        return Err(AkitaError::InvalidSetup(
            "extension-opening split exceeds opening arity".into(),
        ));
    }
    runs.push(GrindingRun::proof_of_work(
        GrindingSite::ExtensionOpeningPoint { level },
        multilinear_point_loss_factor(split_bits)?,
        capacity,
    )?);
    if layout.requires_row_batch_challenge() {
        runs.push(GrindingRun::proof_of_work(
            GrindingSite::ExtensionOpeningClaimBatch { level },
            1,
            capacity,
        )?);
    }
    for round in 0..layout.max_num_vars() - split_bits {
        append_sumcheck(
            runs,
            capacity,
            SumcheckProtocol::ExtensionOpeningReduction,
            level,
            0,
            round,
            2,
        )?;
    }
    Ok(())
}

fn append_sumcheck(
    runs: &mut Vec<GrindingRun>,
    capacity: u32,
    protocol: SumcheckProtocol,
    level: u32,
    stage: u32,
    round: usize,
    degree: usize,
) -> Result<(), AkitaError> {
    runs.push(GrindingRun::proof_of_work(
        GrindingSite::SumcheckRound {
            protocol,
            level,
            stage,
            round: usize_to_u32(round, "sumcheck grinding round")?,
        },
        polynomial_identity_loss_factor(degree)?,
        capacity,
    )?);
    Ok(())
}

fn recursive_layout(
    predecessor_rounds: usize,
    current: &CommittedGroupParams,
) -> Result<OpeningClaimsLayout, AkitaError> {
    let mut groups = Vec::with_capacity(2);
    if let Some(prefix) = current.setup_prefix() {
        groups.push(PolynomialGroupLayout::singleton(
            setup_prefix_sumcheck_rounds(prefix)?,
        ));
    }
    groups.push(PolynomialGroupLayout::singleton(predecessor_rounds));
    OpeningClaimsLayout::from_groups(groups)
}

/// Number of Stage 3 sumcheck rounds induced when a successor consumes a setup prefix.
pub fn setup_prefix_sumcheck_rounds(
    prefix: &crate::GroupOpenPhaseParams,
) -> Result<usize, AkitaError> {
    let n_prefix = prefix.n_prefix()?;
    let d_setup = prefix.d_setup();
    if d_setup == 0 || !n_prefix.is_multiple_of(d_setup) {
        return Err(AkitaError::InvalidSetup(
            "setup-prefix field length does not align with its ring dimension".into(),
        ));
    }
    let ring_len = n_prefix / d_setup;
    if ring_len == 0 {
        return Err(AkitaError::InvalidSetup(
            "setup-prefix ring length is zero".into(),
        ));
    }
    (d_setup.trailing_zeros() as usize)
        .checked_add(ring_len.next_power_of_two().trailing_zeros() as usize)
        .ok_or_else(|| AkitaError::InvalidSetup("setup-prefix round count overflow".into()))
}

fn usize_to_u32(value: usize, name: &str) -> Result<u32, AkitaError> {
    u32::try_from(value).map_err(|_| AkitaError::InvalidSetup(format!("{name} exceeds u32")))
}

fn usize_to_u64(value: usize, name: &str) -> Result<u64, AkitaError> {
    u64::try_from(value).map_err(|_| AkitaError::InvalidSetup(format!("{name} exceeds u64")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SisModulusProfileId;
    use akita_challenges::SparseChallengeConfig;

    fn params(ring_dimension: usize) -> CommittedGroupParams {
        CommittedGroupParams::params_only(
            SisModulusProfileId::Q128OffsetA7F7,
            ring_dimension,
            3,
            2,
            2,
            2,
            SparseChallengeConfig::pm1_only(3),
        )
        .with_decomp(1, 1, 2, 2, 2)
        .expect("test fold params")
    }

    #[test]
    fn stage_rounds_follow_successor_padded_relation_domain() {
        let current = params(64);
        let successor = params(128);
        let layout = OpeningClaimsLayout::new(6, 1).expect("opening layout");
        let output_witness_len = 64;
        let expected_rounds = current
            .relation_address_geometry(&layout, 1, successor.d_a(), output_witness_len)
            .expect("relation geometry")
            .relation_point_variable_count();
        assert_eq!(expected_rounds, 7);
        assert_ne!(
            expected_rounds,
            crate::sumcheck_rounds(current.d_a(), output_witness_len),
            "the fixture must distinguish successor padding from the old shortcut"
        );

        let mut runs = Vec::new();
        let rounds = append_nonterminal(
            &mut runs,
            128,
            1,
            0,
            &current,
            output_witness_len,
            &layout,
            GrindingPlanSuccessor::Recursive(&successor),
        )
        .expect("nonterminal grinding runs");
        assert_eq!(rounds, expected_rounds);
        assert_eq!(
            runs.iter()
                .filter(|run| {
                    matches!(
                        run.site(),
                        GrindingSite::SumcheckRound {
                            protocol: SumcheckProtocol::Stage2,
                            ..
                        }
                    )
                })
                .count(),
            expected_rounds
        );

        let recursive = recursive_layout(rounds, &successor).expect("recursive layout");
        assert_eq!(
            recursive
                .group_layout(recursive.root_final_group_index().expect("final group"))
                .expect("final layout")
                .num_vars(),
            expected_rounds
        );

        let terminal = crate::TerminalFoldParams::from_expanded_group(successor);
        let mut terminal_runs = Vec::new();
        append_terminal(&mut terminal_runs, 128, 4, 1, rounds, &terminal)
            .expect("terminal grinding runs");
        assert_eq!(
            terminal_runs
                .iter()
                .filter(|run| {
                    matches!(
                        run.site(),
                        GrindingSite::SumcheckRound {
                            protocol: SumcheckProtocol::ExtensionOpeningReduction,
                            ..
                        }
                    )
                })
                .count(),
            expected_rounds - 2
        );
    }
}
