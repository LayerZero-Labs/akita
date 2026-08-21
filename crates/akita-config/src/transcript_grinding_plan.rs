//! Canonical transcript-grinding plan derivation from public protocol shape.

use crate::CommitmentConfig;
use akita_error::AkitaError;
use akita_field::{CanonicalField, ExtField, FieldCore};
use akita_types::{
    multilinear_point_loss_factor, nominal_challenge_capacity_bits,
    polynomial_identity_loss_factor, powers_batch_loss_factor, ring_switch_alpha_loss_factor,
    tensor_opening_split, BasisMode, CommittedGroupParams, DigitRangePlan, FoldParams,
    FoldSchedule, GrindingPlan, GrindingRun, GrindingSite, OpeningClaimsLayout,
    PolynomialGroupLayout, SumcheckProtocol,
};

/// Derive the only accepted grinding plan for one effective schedule and call.
pub fn derive_transcript_grinding_plan<Cfg: CommitmentConfig>(
    schedule: &FoldSchedule,
    root_layout: &OpeningClaimsLayout,
    _basis: BasisMode,
) -> Result<GrindingPlan, AkitaError>
where
    Cfg::Field: CanonicalField,
    Cfg::ExtField: ExtField<Cfg::Field>,
{
    schedule.validate_structure()?;
    root_layout.check()?;
    let extension_degree = <Cfg::ExtField as ExtField<Cfg::Field>>::EXT_DEGREE;
    if Cfg::EXT_DEGREE != extension_degree {
        return Err(AkitaError::InvalidSetup(
            "grinding plan extension degree does not match the field tower".into(),
        ));
    }
    schedule.validate_nonterminal_opening_execution(extension_degree)?;
    let capacity = nominal_challenge_capacity_bits(
        <Cfg::Field as CanonicalField>::modulus_bits(),
        extension_degree,
    )?;
    let mut runs = vec![GrindingRun::proof_of_work(
        GrindingSite::EvaluationBatch,
        1,
        capacity,
    )?];

    let mut predecessor = &schedule.root;
    append_nonterminal::<Cfg>(
        &mut runs,
        capacity,
        0,
        predecessor,
        root_layout,
        schedule.recursive_folds.first().map(|step| &step.params),
        schedule
            .recursive_folds
            .is_empty()
            .then_some(&schedule.terminal),
    )?;

    for (index, fold) in schedule.recursive_folds.iter().enumerate() {
        let layout = recursive_layout(predecessor, &fold.params)?;
        let successor = schedule
            .recursive_folds
            .get(index + 1)
            .map(|step| &step.params);
        append_nonterminal::<Cfg>(
            &mut runs,
            capacity,
            usize_to_u32(index + 1, "grinding level")?,
            fold,
            &layout,
            successor,
            successor.is_none().then_some(&schedule.terminal),
        )?;
        predecessor = fold;
    }

    append_terminal::<Cfg>(
        &mut runs,
        capacity,
        usize_to_u32(
            schedule.recursive_folds.len() + 1,
            "terminal grinding level",
        )?,
        predecessor,
        schedule,
    )?;
    GrindingPlan::new(runs, capacity)
}

#[allow(clippy::too_many_arguments)]
fn append_nonterminal<Cfg: CommitmentConfig>(
    runs: &mut Vec<GrindingRun>,
    capacity: u32,
    level: u32,
    fold: &FoldParams,
    layout: &OpeningClaimsLayout,
    recursive_successor: Option<&CommittedGroupParams>,
    terminal_successor: Option<&akita_types::TerminalFoldParams>,
) -> Result<(), AkitaError>
where
    Cfg::Field: CanonicalField,
    Cfg::ExtField: ExtField<Cfg::Field>,
{
    let params = &fold.params;
    let opening_method = params.uniform_opening_method(layout)?;
    if opening_method.requires_extension_opening_reduction(Cfg::EXT_DEGREE) {
        append_eor::<Cfg>(runs, capacity, level, layout)?;
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

    let (successor_d, successor_opening_vars) = if let Some(successor) = recursive_successor {
        (successor.d_a(), successor.recursive_opening_num_vars()?)
    } else {
        let terminal = terminal_successor.ok_or_else(|| {
            AkitaError::InvalidSetup("nonterminal grinding level has no successor".into())
        })?;
        (terminal.d_a(), terminal.recursive_opening_num_vars()?)
    };
    let tau0_width = params
        .relation_address_geometry(
            layout,
            Cfg::EXT_DEGREE,
            successor_d,
            fold.output_witness_len,
        )?
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

    let rounds = akita_types::sumcheck_rounds(params.d_a(), fold.output_witness_len);
    let basis = 1usize
        .checked_shl(params.open().digits.log_basis)
        .ok_or_else(|| AkitaError::InvalidSetup("digit-range basis exceeds usize".into()))?;
    let range = DigitRangePlan::new(basis)?;
    let (stages, norm) =
        range.proof_shapes_for_route(rounds, params.inner().matrix.security_route())?;
    for (stage_index, stage_shape) in stages.iter().enumerate() {
        let stage = usize_to_u32(stage_index, "Stage 1 grinding stage")?;
        for round in 0..stage_shape.sumcheck_proof.0 {
            append_sumcheck(
                runs,
                capacity,
                SumcheckProtocol::Stage1,
                level,
                stage,
                round,
                stage_shape.sumcheck_proof.1,
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

fn append_terminal<Cfg: CommitmentConfig>(
    runs: &mut Vec<GrindingRun>,
    capacity: u32,
    level: u32,
    predecessor: &FoldParams,
    schedule: &FoldSchedule,
) -> Result<(), AkitaError>
where
    Cfg::Field: CanonicalField,
    Cfg::ExtField: ExtField<Cfg::Field>,
{
    let width =
        akita_types::sumcheck_rounds(predecessor.params.d_a(), predecessor.output_witness_len);
    let layout = OpeningClaimsLayout::new(width, 1)?;
    if Cfg::EXT_DEGREE > 1 {
        append_eor::<Cfg>(runs, capacity, level, &layout)?;
    }
    runs.push(GrindingRun::fold_response(level));
    runs.push(GrindingRun::fold_challenge_root(level, 0));
    runs.push(GrindingRun::fold_challenge_coordinates(
        level,
        0,
        usize_to_u64(
            schedule.terminal.blocks.live_blocks,
            "terminal fold coordinates",
        )?,
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

fn append_eor<Cfg: CommitmentConfig>(
    runs: &mut Vec<GrindingRun>,
    capacity: u32,
    level: u32,
    layout: &OpeningClaimsLayout,
) -> Result<(), AkitaError>
where
    Cfg::Field: FieldCore,
    Cfg::ExtField: ExtField<Cfg::Field>,
{
    let (split_bits, _) = tensor_opening_split::<Cfg::Field, Cfg::ExtField>()?;
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
    predecessor: &FoldParams,
    current: &CommittedGroupParams,
) -> Result<OpeningClaimsLayout, AkitaError> {
    let mut groups = Vec::with_capacity(2);
    if let Some(prefix) = current.setup_prefix() {
        groups.push(PolynomialGroupLayout::singleton(setup_prefix_rounds(
            prefix,
        )?));
    }
    groups.push(PolynomialGroupLayout::singleton(
        akita_types::sumcheck_rounds(predecessor.params.d_a(), predecessor.output_witness_len),
    ));
    OpeningClaimsLayout::from_groups(groups)
}

fn setup_prefix_rounds(prefix: &akita_types::GroupOpenPhaseParams) -> Result<usize, AkitaError> {
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
    use crate::proof_optimized::fp128;
    use akita_field::PseudoMersenneField;
    #[cfg(any(feature = "all-schedules", feature = "schedules-default"))]
    use akita_types::{GrindingQueryKind, GRINDING_NONCE_SLACK_BITS};

    #[cfg(feature = "schedules-default")]
    #[test]
    fn production_onehot_plan_is_canonical_and_fully_priced() {
        let layout = OpeningClaimsLayout::new(14, 1).expect("opening layout");
        let row = fp128::OneHot::resolve_catalog_row_for_opening(&layout)
            .expect("generated production row");
        let plan = derive_transcript_grinding_plan::<fp128::OneHot>(
            row.schedule(),
            &layout,
            BasisMode::Lagrange,
        )
        .expect("grinding plan");

        assert_eq!(
            plan.runs().first().unwrap().site(),
            GrindingSite::EvaluationBatch
        );
        for run in plan.runs() {
            if run.kind() == GrindingQueryKind::ProofOfWork {
                assert!(u128::from(run.loss_factor()) <= (1u128 << run.grind_bits()));
                assert_eq!(
                    run.nonce_bits(),
                    if run.grind_bits() == 0 {
                        0
                    } else {
                        run.grind_bits() + GRINDING_NONCE_SLACK_BITS
                    }
                );
            }
        }
        assert!(plan.runs().iter().any(|run| {
            matches!(run.site(), GrindingSite::FoldChallengeCoordinates { .. })
                && run.multiplicity() > 0
        }));
        assert_eq!(
            (
                plan.runs().len(),
                plan.expanded_query_count(),
                plan.total_nonce_bits(),
                plan.digest().unwrap(),
            ),
            (
                46,
                51,
                366,
                [
                    51, 160, 87, 176, 119, 237, 86, 152, 107, 131, 142, 46, 249, 143, 16, 50, 190,
                    49, 242, 207, 136, 30, 42, 249, 19, 22, 15, 17, 91, 211, 133, 102,
                ],
            )
        );
    }

    #[test]
    fn exact_field_orders_report_the_pseudo_mersenne_deficit_without_repricing() {
        fn exact_order<F: PseudoMersenneField>(extension_degree: usize) -> (u32, u128, usize) {
            (F::MODULUS_BITS, F::MODULUS_OFFSET, extension_degree)
        }

        // These tuples state the exact denominator symbolically as
        // `(2^bits - offset)^extension_degree`; the nominal pricing denominator
        // is `2^(bits * extension_degree)`.
        assert_eq!(
            exact_order::<fp128::Field>(1),
            (128, (1u128 << 32) - 22_537, 1)
        );
        assert_eq!(
            exact_order::<crate::proof_optimized::fp64::Field>(2),
            (64, 59, 2)
        );
        assert_eq!(
            exact_order::<crate::proof_optimized::fp32::Field>(4),
            (32, 99, 4)
        );
        for (bits, _, degree) in [
            exact_order::<fp128::Field>(1),
            exact_order::<crate::proof_optimized::fp64::Field>(2),
            exact_order::<crate::proof_optimized::fp32::Field>(4),
        ] {
            assert_eq!(nominal_challenge_capacity_bits(bits, degree).unwrap(), 128);
            assert_eq!(akita_types::grind_bits_for_loss(3, 128).unwrap(), 2);
        }
    }

    #[cfg(feature = "all-schedules")]
    #[test]
    fn every_generated_production_row_derives_a_complete_plan() {
        use crate::proof_optimized::{fp32, fp64};

        fn audit<Cfg: CommitmentConfig>()
        where
            Cfg::Field: CanonicalField,
            Cfg::ExtField: ExtField<Cfg::Field>,
        {
            let table = Cfg::schedule_catalog().expect("production catalog");
            for entry in table.entries {
                let row = Cfg::resolve_catalog_row_for_key(&entry.to_runtime_lookup_key())
                    .expect("admitted production row");
                let layout = row.profiles().opening_layout().expect("opening layout");
                let plan = derive_transcript_grinding_plan::<Cfg>(
                    row.schedule(),
                    &layout,
                    BasisMode::Lagrange,
                )
                .expect("complete grinding plan");
                let fold_responses = plan
                    .runs()
                    .iter()
                    .filter(|run| run.kind() == GrindingQueryKind::FoldResponse)
                    .count();
                let roots = plan
                    .runs()
                    .iter()
                    .filter(|run| run.kind() == GrindingQueryKind::FoldChallengeRoot)
                    .count();
                let coordinates = plan
                    .runs()
                    .iter()
                    .filter(|run| run.kind() == GrindingQueryKind::FoldChallengeCoordinates)
                    .count();
                assert_eq!(fold_responses, row.schedule().num_fold_levels());
                assert_eq!(roots, coordinates);
                assert!(roots >= fold_responses);
                assert!(plan.expanded_query_count() >= plan.runs().len() as u64);
            }
        }

        audit::<fp128::Dense>();
        audit::<fp128::DenseBounded>();
        audit::<fp128::DenseMultiChunk>();
        audit::<fp128::OneHot>();
        audit::<fp128::OneHotMultiChunk>();
        audit::<fp128::OneHotMultiChunkW2R2>();
        audit::<fp128::OneHotMultiChunkW4R2>();
        audit::<fp64::Dense>();
        audit::<fp64::OneHot>();
        audit::<fp32::Dense>();
        audit::<fp32::OneHot>();
        audit::<crate::RecursiveCommitmentConfig<fp128::OneHot>>();
        audit::<crate::RecursiveCommitmentConfig<fp128::OneHotMultiChunk>>();

        let has_eor = fp32::OneHot::schedule_catalog()
            .unwrap()
            .entries
            .iter()
            .any(|entry| {
                let row = fp32::OneHot::resolve_catalog_row_for_key(&entry.to_runtime_lookup_key())
                    .unwrap();
                let layout = row.profiles().opening_layout().unwrap();
                derive_transcript_grinding_plan::<fp32::OneHot>(
                    row.schedule(),
                    &layout,
                    BasisMode::Lagrange,
                )
                .unwrap()
                .runs()
                .iter()
                .any(|run| run.site() == GrindingSite::ExtensionOpeningPoint)
            });
        assert!(has_eor, "small-field production plans must cover EOR sites");

        type RecursiveOneHot = crate::RecursiveCommitmentConfig<fp128::OneHot>;
        let has_stage3 = RecursiveOneHot::schedule_catalog()
            .unwrap()
            .entries
            .iter()
            .any(|entry| {
                let row =
                    RecursiveOneHot::resolve_catalog_row_for_key(&entry.to_runtime_lookup_key())
                        .unwrap();
                let layout = row.profiles().opening_layout().unwrap();
                derive_transcript_grinding_plan::<RecursiveOneHot>(
                    row.schedule(),
                    &layout,
                    BasisMode::Lagrange,
                )
                .unwrap()
                .runs()
                .iter()
                .any(|run| {
                    matches!(
                        run.site(),
                        GrindingSite::SumcheckRound {
                            protocol: SumcheckProtocol::Stage3,
                            ..
                        }
                    )
                })
            });
        assert!(has_stage3, "recursive production plans must cover Stage 3");
    }
}
