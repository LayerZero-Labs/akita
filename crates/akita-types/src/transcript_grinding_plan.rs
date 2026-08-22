//! Canonical grinding-plan derivation from public schedule geometry.

use crate::{
    multilinear_point_loss_factor, nominal_challenge_capacity_bits,
    polynomial_identity_loss_factor, powers_batch_loss_factor, ring_switch_alpha_loss_factor,
    CommittedGroupParams, DigitRangePlan, FoldSchedule, GrindingPlan, GrindingRun, GrindingSite,
    OpeningClaimsLayout, PolynomialGroupLayout, SumcheckProtocol,
};
use akita_error::AkitaError;

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
    recursive_successor: Option<&CommittedGroupParams>,
    terminal_successor: Option<&crate::TerminalFoldParams>,
    modulus_bits: u32,
    extension_degree: usize,
    level: u32,
) -> Result<usize, AkitaError> {
    if recursive_successor.is_some() == terminal_successor.is_some() {
        return Err(AkitaError::InvalidSetup(
            "planner grinding edge requires exactly one successor".into(),
        ));
    }
    layout.check()?;
    if extension_degree == 0 || !extension_degree.is_power_of_two() {
        return Err(AkitaError::InvalidSetup(
            "grinding extension degree must be a nonzero power of two".into(),
        ));
    }
    let capacity = nominal_challenge_capacity_bits(modulus_bits, extension_degree)?;
    let mut runs = Vec::new();
    append_nonterminal(
        &mut runs,
        capacity,
        extension_degree,
        level,
        params,
        output_witness_len,
        layout,
        recursive_successor,
        terminal_successor,
    )?;
    if let Some(terminal) = terminal_successor {
        append_terminal(
            &mut runs,
            capacity,
            extension_degree,
            level.checked_add(1).ok_or_else(|| {
                AkitaError::InvalidSetup("terminal grinding level overflow".into())
            })?,
            params,
            output_witness_len,
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

    let mut predecessor = &schedule.root;
    append_nonterminal(
        &mut runs,
        capacity,
        extension_degree,
        0,
        &predecessor.params,
        predecessor.output_witness_len,
        root_layout,
        schedule.recursive_folds.first().map(|step| &step.params),
        schedule
            .recursive_folds
            .is_empty()
            .then_some(&schedule.terminal),
    )?;

    for (index, fold) in schedule.recursive_folds.iter().enumerate() {
        let layout = recursive_layout(
            &predecessor.params,
            predecessor.output_witness_len,
            &fold.params,
        )?;
        let successor = schedule
            .recursive_folds
            .get(index + 1)
            .map(|step| &step.params);
        append_nonterminal(
            &mut runs,
            capacity,
            extension_degree,
            usize_to_u32(index + 1, "grinding level")?,
            &fold.params,
            fold.output_witness_len,
            &layout,
            successor,
            successor.is_none().then_some(&schedule.terminal),
        )?;
        predecessor = fold;
    }

    append_terminal(
        &mut runs,
        capacity,
        extension_degree,
        usize_to_u32(
            schedule.recursive_folds.len() + 1,
            "terminal grinding level",
        )?,
        &predecessor.params,
        predecessor.output_witness_len,
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
    recursive_successor: Option<&CommittedGroupParams>,
    terminal_successor: Option<&crate::TerminalFoldParams>,
) -> Result<(), AkitaError> {
    let opening_method = params.uniform_opening_method(layout)?;
    if opening_method.requires_extension_opening_reduction(extension_degree) {
        append_eor(runs, capacity, extension_degree, level, layout)?;
    }

    runs.push(GrindingRun::proof_of_work(
        GrindingSite::EvaluationBatch,
        1,
        capacity,
    )?);

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

    let (successor_d, successor_opening_vars) = if let Some(successor) = recursive_successor {
        (successor.d_a(), successor.recursive_opening_num_vars()?)
    } else {
        let terminal = terminal_successor.ok_or_else(|| {
            AkitaError::InvalidSetup("nonterminal grinding level has no successor".into())
        })?;
        (terminal.d_a(), terminal.recursive_opening_num_vars()?)
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

    let rounds = crate::sumcheck_rounds(params.d_a(), output_witness_len);
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
    if let Some(successor) = recursive_successor {
        if let Some(prefix) = successor.setup_prefix() {
            for round in 0..setup_prefix_rounds(prefix)? {
                append_sumcheck(runs, capacity, SumcheckProtocol::Stage3, level, 0, round, 2)?;
            }
        }
    }
    Ok(())
}

fn append_terminal(
    runs: &mut Vec<GrindingRun>,
    capacity: u32,
    extension_degree: usize,
    level: u32,
    predecessor_params: &CommittedGroupParams,
    predecessor_output_witness_len: usize,
    terminal: &crate::TerminalFoldParams,
) -> Result<(), AkitaError> {
    let width = crate::sumcheck_rounds(predecessor_params.d_a(), predecessor_output_witness_len);
    let layout = OpeningClaimsLayout::new(width, 1)?;
    if extension_degree > 1 {
        append_eor(runs, capacity, extension_degree, level, &layout)?;
    }
    runs.push(GrindingRun::proof_of_work(
        GrindingSite::EvaluationBatch,
        1,
        capacity,
    )?);
    runs.push(GrindingRun::fold_response(level));
    runs.push(GrindingRun::fold_challenge_root(level, 0));
    runs.push(GrindingRun::fold_challenge_coordinates(
        level,
        0,
        usize_to_u64(terminal.blocks.live_blocks, "terminal fold coordinates")?,
    ));
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
        runs.push(GrindingRun::fold_challenge_root(level, group));
        runs.push(GrindingRun::fold_challenge_coordinates(
            level,
            group,
            usize_to_u64(multiplicity, "fold coordinate count")?,
        ));
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
        GrindingSite::ExtensionOpeningPoint,
        multilinear_point_loss_factor(split_bits)?,
        capacity,
    )?);
    runs.push(GrindingRun::proof_of_work(
        GrindingSite::ExtensionOpeningClaimBatch,
        1,
        capacity,
    )?);
    let encoded_level = if level == 0 { u32::MAX } else { level };
    for round in 0..layout.max_num_vars() - split_bits {
        append_sumcheck(
            runs,
            capacity,
            SumcheckProtocol::ExtensionOpeningReduction,
            encoded_level,
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
    predecessor_params: &CommittedGroupParams,
    predecessor_output_witness_len: usize,
    current: &CommittedGroupParams,
) -> Result<OpeningClaimsLayout, AkitaError> {
    let mut groups = Vec::with_capacity(2);
    if let Some(prefix) = current.setup_prefix() {
        groups.push(PolynomialGroupLayout::singleton(setup_prefix_rounds(
            prefix,
        )?));
    }
    groups.push(PolynomialGroupLayout::singleton(crate::sumcheck_rounds(
        predecessor_params.d_a(),
        predecessor_output_witness_len,
    )));
    OpeningClaimsLayout::from_groups(groups)
}

fn setup_prefix_rounds(prefix: &crate::GroupOpenPhaseParams) -> Result<usize, AkitaError> {
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
