use super::*;
use crate::backend::OperationCtx;
use crate::backend::{EvaluationTraceDescription, Stage2OpeningDescription};
use jolt_field::AdditiveGroup;

use akita_types::{
    batch_l2_virtual_evaluations, CoefficientPackingBatchSemantics, DigitRangeEqualityPoint,
    InnerCommitSecurityRoute, OpeningClaimsLayout, OpeningFamily, PhysicalResponsePlan,
    RelationRangeImagePlan,
};

pub(in crate::protocol::prove) struct PhysicalL2ProverReplay<E: Field> {
    plan: PhysicalResponsePlan,
    point: Vec<E>,
    virtual_evaluations: Vec<E>,
    batching: Vec<E>,
    claim: E,
}

struct Stage1ProveOutput<E: Field> {
    point: Vec<E>,
    range_image_evaluation: E,
    physical_l2: Option<PhysicalL2ProverReplay<E>>,
}

struct Stage2ProveOutput<E: Field> {
    challenges: Vec<E>,
    witness_evaluation: E,
}

pub(in crate::protocol::prove) struct PreparedFold<
    F: Field + CanonicalEncoding,
    E: Field,
    WitnessHandle,
> {
    pub(in crate::protocol::prove) instance: RingRelationInstance<F>,
    pub(in crate::protocol::prove) witness_handle: WitnessHandle,
    pub(in crate::protocol::prove) evaluation_trace_claim: E,
    pub(in crate::protocol::prove) relation_groups:
        Vec<crate::backend::PreparedRelationGroupPublic<F, E>>,
    pub(in crate::protocol::prove) evaluation_trace_claim_coefficients: Vec<E>,
    pub(in crate::protocol::prove) evaluation_trace_basis: BasisMode,
    pub(in crate::protocol::prove) row_coefficients: Option<Vec<E>>,
}

/// Bind scheduled opening messages and construct an opaque recursive witness.
#[allow(
    clippy::needless_lifetimes,
    clippy::too_many_arguments,
    clippy::type_complexity
)]
pub(super) fn prepare_fold<'claims, 'source, F, E, B>(
    backend: &B,
    block_claims: ProverOpeningData<
        'claims,
        E,
        OpeningSource<'source, B::CommitmentHandle, B::WitnessHandle>,
        F,
    >,
    commitment_material: Vec<B::CommitmentMaterialHandle>,
    pad_base_evals: bool,
    grinding: &mut akita_types::NativeProverGrinding<'_>,
    level: u32,
    level_params: &CommittedGroupParams,
    basis: BasisMode,
    session: &B::ProofSessionHandle,
    expected_witness_len: usize,
    commitment_ring_dimension: usize,
    run_eor: bool,
) -> Result<PreparedFold<F, E, B::WitnessHandle>, AkitaError>
where
    F: Field + CanonicalEncoding + AkitaSerialize + Ring + Unreduced + PseudoMersenne + 'static,
    <F as Unreduced>::Wide: From<F> + AdditiveGroup,
    E: FpExtEncoding<F>
        + ExtField<F>
        + Unreduced
        + Fold
        + Ring
        + MulBaseUnreduced<F>
        + AkitaSerialize
        + 'static,
    B: ProverBackend<F, E>,
{
    let trace_opening_batch = block_claims.opening_layout().clone();
    let trace_opening_batch = &trace_opening_batch;
    let context = backend.proof_context(session, level)?;
    let (protocol_points, reduction) = if run_eor {
        let groups = (0..trace_opening_batch.num_groups())
            .map(|index| {
                Ok(crate::backend::EorGroupRequest {
                    source: *block_claims.group(index)?,
                    point: block_claims.opening_claims().group_point(index)?,
                    ring_dimension: level_params
                        .group_role_dims(trace_opening_batch, index)?
                        .d_a(),
                })
            })
            .collect::<Result<Vec<_>, AkitaError>>()?;
        let expected = block_claims
            .opening_claims()
            .groups()
            .iter()
            .flat_map(|g| g.evaluations().iter().copied())
            .collect::<Vec<_>>();
        let reduced = prove_extension_opening_reduction::<F, E, B>(
            backend,
            session,
            &context,
            trace_opening_batch,
            &groups,
            grinding,
            level,
            &expected,
        )?;
        (reduced.protocol_points, Some(reduced.reduction))
    } else {
        (
            block_claims
                .opening_claims()
                .groups()
                .iter()
                .map(|g| g.point().to_vec())
                .collect(),
            None,
        )
    };

    let opening = &OperationCtx::new(backend, session, context);
    // A-role operation: prepare each group at its native A dimension,
    // fold-evaluate its claim polynomials, and derive scalar openings before
    // leaving the typed dispatch arm. Typed fold outputs cross the boundary
    // only through D-free `PreparedOpeningPoint` / `RingVec` carriers.
    let opening_batch = trace_opening_batch.clone();
    let opening_method = level_params.opening_method();
    if !matches!(opening_method, akita_types::OpeningMethod::EvaluationTrace) && reduction.is_some()
    {
        return Err(AkitaError::InvalidSetup(
            "coefficient packing cannot consume an extension-opening reduction".into(),
        ));
    }
    let final_group_index = level_params.validate_opening_batch(&opening_batch)?;
    let mut prepared_group_openings = Vec::with_capacity(opening_batch.num_groups());
    let mut scalar_openings = Vec::with_capacity(opening_batch.num_total_polynomials());
    for (group_index, group_lp) in level_params.groups().iter().enumerate() {
        let ring_dimension = group_lp.inner_commit_matrix_params().ring_dimension();
        let group_alpha_bits = ring_dimension.trailing_zeros() as usize;
        let group_protocol_point = protocol_points
            .get(group_index)
            .ok_or(AkitaError::InvalidProof)?;
        if matches!(
            group_lp.opening_method(),
            akita_types::OpeningMethod::EvaluationTrace
        ) {
            let target_len = akita_error::checked::sum([
                group_alpha_bits,
                group_lp.position_index_bits(),
                group_lp.block_index_bits(),
            ])
            .ok_or_else(|| {
                AkitaError::InvalidSetup("group opening point length overflow".into())
            })?;
            let allow_short_point = pad_base_evals && group_index == final_group_index;
            if group_protocol_point.len() > target_len
                || (!allow_short_point && group_protocol_point.len() != target_len)
            {
                return Err(AkitaError::InvalidPointDimension {
                    expected: target_len,
                    actual: group_protocol_point.len(),
                });
            }
        }
        let source = *block_claims.group(group_index)?;
        let logical_len = match source {
            OpeningSource::Commitment(h) => 1usize
                .checked_shl(
                    u32::try_from(h.metadata().num_vars()).map_err(|_| AkitaError::InvalidProof)?,
                )
                .ok_or(AkitaError::InvalidProof)?,
            OpeningSource::Witness(h) => h.manifest().logical_len(),
        };
        let plan = crate::backend::ValidatedRecursiveGroupOpeningPlan::new(
            group_protocol_point,
            basis,
            ring_dimension,
            group_lp.num_positions_per_block(),
            group_lp.num_live_blocks(),
            group_alpha_bits,
            group_lp.opening_method(),
            logical_len,
        );
        let prepared = backend.prepare_opening(
            opening.proof_session(),
            opening.for_group(group_index).proof_context(),
            source,
            &plan,
        )?;
        if prepared.scalar_openings().len()
            != opening_batch.group_layout(group_index)?.num_polynomials()
            || (reduction.is_none()
                && prepared.scalar_openings()
                    != block_claims
                        .opening_claims()
                        .groups()
                        .get(group_index)
                        .ok_or(AkitaError::InvalidProof)?
                        .evaluations())
        {
            return Err(AkitaError::InvalidProof);
        }
        if pad_base_evals {
            akita_transcript::public_native_extensions_prover::<F, E>(
                grinding.state_mut(),
                akita_transcript::ProtocolSiteId {
                    family: akita_transcript::SITE_FAMILY_FOLD_BINDING,
                    level,
                    stage: 1,
                    group: u32::try_from(group_index)
                        .map_err(|_| AkitaError::InvalidSetup("group index exceeds u32".into()))?,
                    ..akita_transcript::ProtocolSiteId::default()
                },
                group_protocol_point,
            )
            .map_err(|_| AkitaError::InvalidProof)?;
        }
        scalar_openings.extend_from_slice(prepared.scalar_openings());
        prepared_group_openings.push(prepared);
    }
    if reduction.is_none() {
        akita_transcript::public_native_extensions_prover::<F, E>(
            grinding.state_mut(),
            akita_transcript::ProtocolSiteId {
                family: akita_transcript::SITE_FAMILY_FOLD_BINDING,
                level,
                stage: 2,
                ..akita_transcript::ProtocolSiteId::default()
            },
            &scalar_openings,
        )
        .map_err(|_| AkitaError::InvalidProof)?;
    }
    let crate::protocol::ring_relation::PreparedRingRelationOutput {
        relation:
            crate::protocol::ring_relation::PreparedRingRelation {
                instance,
                witness_handle,
                groups: relation_groups,
            },
        trace_claim,
        row_coefficients,
    } = RingRelationProver::prepare::<F, E, B>(
        opening,
        prepared_group_openings,
        commitment_material,
        &block_claims,
        level_params.clone(),
        grinding,
        level,
        &reduction,
        &scalar_openings,
        trace_opening_batch,
        expected_witness_len,
        commitment_ring_dimension,
    )
    .map_err(|err| {
        AkitaError::InvalidInput(format!("ring relation preparation failed: {err:?}"))
    })?;
    let evaluation_trace_claim_coefficients = trace_claim.claim_coefficients;
    // Recursive suffixes still omit the public row coefficients from ring-switch
    // finalization. Evaluation-trace coefficients are normalized independently and
    // therefore do not inherit that path distinction.
    let clear_recursive_trace = pad_base_evals && !level_params.has_preceding_groups();
    let row_coefficients = if clear_recursive_trace {
        None
    } else {
        Some(row_coefficients)
    };
    Ok(PreparedFold {
        instance,
        witness_handle,
        evaluation_trace_claim: trace_claim.claimed_evaluation,
        relation_groups,
        evaluation_trace_claim_coefficients,
        evaluation_trace_basis: basis,
        row_coefficients,
    })
}

/// Typed commitment parameters for the witness produced by a non-terminal
/// fold. The terminal variant exposes only its inner commitment.
#[derive(Clone, Copy)]
pub(crate) enum FoldSuccessorParams<'a> {
    Recursive(&'a FoldParams),
    Terminal(&'a TerminalFoldParams),
}

impl<'a> FoldSuccessorParams<'a> {
    pub(crate) fn inner_ring_dimension(self) -> usize {
        match self {
            Self::Recursive(params) => params.params.d_a(),
            Self::Terminal(params) => params.d_a(),
        }
    }

    fn recursive(self) -> Option<&'a FoldParams> {
        match self {
            Self::Recursive(params) => Some(params),
            Self::Terminal(_) => None,
        }
    }

    fn setup_contribution_mode(self) -> SetupContributionMode {
        match self {
            Self::Recursive(params) => params.predecessor_setup_contribution_mode(),
            Self::Terminal(_) => SetupContributionMode::Direct,
        }
    }
}

struct CommittedNextWitness<F: Field, WitnessHandle, M> {
    committed_witness_len: usize,
    witness_handle: WitnessHandle,
    binding: NextWitnessState<F>,
    commitment_material_handle: M,
}

fn prepare_physical_l2_batch<F, E>(
    grinding: &mut akita_types::NativeProverGrinding<'_>,
    level: usize,
    physical_l2: Option<PhysicalL2ProverReplay<E>>,
) -> Result<Option<PhysicalL2ProverReplay<E>>, AkitaError>
where
    F: Field + CanonicalEncoding + AkitaSerialize,
    E: ExtField<F> + AkitaSerialize,
{
    let Some(mut replay) = physical_l2 else {
        return Ok(None);
    };
    let level = u32::try_from(level)
        .map_err(|_| AkitaError::InvalidSetup("fold level exceeds u32".into()))?;
    let eta = grinding
        .grinded_ext_challenge::<F, E>(akita_types::GrindingSite::L2VirtualBatch { level })?;
    (replay.claim, replay.batching) =
        batch_l2_virtual_evaluations(eta, &replay.virtual_evaluations);
    Ok(Some(replay))
}

fn prepare_stage2_compression<F, E, H>(
    grinding: &mut akita_types::NativeProverGrinding<'_>,
    level: usize,
    rs: &mut RingSwitchOutput<E, H>,
) -> Result<E, AkitaError>
where
    F: Field + CanonicalEncoding + AkitaSerialize,
    E: ExtField<F> + AkitaSerialize,
{
    if !rs.compressed {
        return Ok(E::zero());
    }
    let level = u32::try_from(level)
        .map_err(|_| AkitaError::InvalidSetup("fold level exceeds u32".into()))?;
    grinding.grinded_ext_challenge::<F, E>(akita_types::GrindingSite::CompressionBinary { level })
}

#[allow(clippy::too_many_arguments)]
fn prepare_relation_sumcheck<'a, F, E, H>(
    lp: &'a CommittedGroupParams,
    opening_batch: &'a OpeningClaimsLayout,
    opening_semantics: OpeningFamily<(), CoefficientPackingBatchSemantics<E>>,
    relation_groups: &[crate::backend::PreparedRelationGroupPublic<F, E>],
    evaluation_trace_claim_coefficients: &'a [E],
    evaluation_trace_claim: E,
    evaluation_trace_basis: BasisMode,
    relation_range_image_plan: &'a RelationRangeImagePlan,
    rs: &RingSwitchOutput<E, H>,
) -> Result<(Stage2OpeningDescription<'a, E>, E), AkitaError>
where
    F: Field + CanonicalEncoding + AkitaSerialize + Ring,
    E: FpExtEncoding<F> + ExtField<F> + Ring,
{
    let _span = tracing::info_span!(
        "stage2_opening_preparation",
        claims = opening_batch.num_total_polynomials(),
        groups = opening_batch.num_groups(),
        chunks = relation_range_image_plan.witness_layout().units().len(),
        coeff_count = rs
            .relation_address_geometry
            .relation_coefficient_block_len(),
    )
    .entered();
    match opening_semantics {
        OpeningFamily::SubringCoefficientPacking(batch) => {
            let mut authenticated_opening = E::zero();
            let mut weighted_opening_claim = E::zero();
            for semantics in batch.groups() {
                let group_index = semantics.group_index();
                let claim_range = semantics.stage2_terms().group_claim_range();
                let group = relation_groups
                    .get(group_index)
                    .ok_or(AkitaError::InvalidProof)?;
                let group_openings = group.scalar_openings();
                let claim_coefficients = evaluation_trace_claim_coefficients
                    .get(claim_range.clone())
                    .ok_or(AkitaError::InvalidProof)?;
                if group_openings.len() != claim_range.len() {
                    return Err(AkitaError::InvalidProof);
                }
                let group_opening = group_openings
                    .iter()
                    .zip(claim_coefficients)
                    .fold(E::zero(), |sum, (&opening, &coefficient)| {
                        sum + opening * coefficient
                    });
                authenticated_opening += group_opening;
                weighted_opening_claim +=
                    semantics.stage2_terms().scalar_claim_weight() * group_opening;
            }
            if authenticated_opening != evaluation_trace_claim {
                return Err(AkitaError::InvalidProof);
            }
            Ok((
                Stage2OpeningDescription::CoefficientPacking(batch),
                weighted_opening_claim,
            ))
        }
        OpeningFamily::EvaluationTrace(()) => {
            let evaluation_trace_row = lp.evaluation_trace_row_index(opening_batch)?;
            let evaluation_trace_weight = relation_row_weight(evaluation_trace_row, &rs.tau1)?;
            ensure_trace_stage2_supported(E::DEGREE)?;
            let evaluation_trace_points = relation_groups
                .iter()
                .map(|group| match group.kind() {
                    OpeningFamily::EvaluationTrace(point) => Ok(point.clone()),
                    OpeningFamily::SubringCoefficientPacking(_) => Err(AkitaError::InvalidProof),
                })
                .collect::<Result<Vec<_>, _>>()?;
            let semantic_trace = EvaluationTraceDescription::try_new(
                relation_range_image_plan.digit_witness_domain(),
                rs.relation_address_geometry
                    .relation_coefficient_block_len(),
                relation_range_image_plan.witness_layout(),
                lp,
                opening_batch,
                &evaluation_trace_points,
                evaluation_trace_claim_coefficients,
                evaluation_trace_basis,
            )?;
            Ok((
                Stage2OpeningDescription::EvaluationTrace {
                    trace: semantic_trace,
                    output_scale: evaluation_trace_weight,
                },
                evaluation_trace_weight * evaluation_trace_claim,
            ))
        }
    }
}

/// Build, commit, and transcript-bind the witness consumed by the successor.
/// The logical witness is computed once and retained for ring switching.
#[allow(clippy::too_many_arguments)]
fn commit_next_witness<F, E, B>(
    backend: &B,
    grinding: &mut akita_types::NativeProverGrinding<'_>,
    level: usize,
    next_params: FoldSuccessorParams<'_>,
    expected_output_witness_len: usize,
    next_witness_binding: akita_types::NextWitnessBindingPolicy,
    witness_handle: B::WitnessHandle,
) -> Result<CommittedNextWitness<F, B::WitnessHandle, B::CommitmentMaterialHandle>, AkitaError>
where
    F: Field + CanonicalEncoding + AkitaSerialize,
    E: Field,
    B: ProverBackend<F, E>,
{
    let next_opening_ring_dim = next_params.inner_ring_dimension();
    let committed_witness_len = akita_types::witness_commitment_domain_len(
        witness_handle.manifest().logical_len(),
        next_opening_ring_dim,
    )?;
    if witness_handle.manifest().logical_len() != expected_output_witness_len {
        return Err(AkitaError::InvalidSetup(format!(
            "scheduled fold level {level} produced unexpected next-w length: expected={expected_output_witness_len}, actual={}",
            witness_handle.manifest().logical_len()
        )));
    }
    let _span = tracing::info_span!("commit_w_level", level).entered();
    let (parameters, source_encoding) = match next_params {
        FoldSuccessorParams::Recursive(params) => (
            crate::backend::WitnessCommitmentParameters::Recursive(params.params.clone()),
            Some(params.params.source_encoding),
        ),
        FoldSuccessorParams::Terminal(params) => (
            crate::backend::WitnessCommitmentParameters::Terminal(params.clone()),
            None,
        ),
    };
    let commit_plan = crate::backend::ValidatedRecursiveWitnessCommitPlan::new(
        witness_handle.manifest().logical_len(),
        committed_witness_len,
        next_opening_ring_dim,
        parameters,
        source_encoding,
        next_witness_binding,
    );
    let commitment_output = backend.commit_witness(witness_handle, &commit_plan)?;
    let (message, witness_handle, commitment_material_handle) =
        commitment_output.into_commitment_parts();
    drop(_span);
    let binding = match (message, next_witness_binding) {
        (
            crate::backend::NextWitnessBindingMessage::OuterPayload(public_commitment),
            akita_types::NextWitnessBindingPolicy::OuterPayload,
        ) => {
            akita_transcript::send_native_field_group(
                grinding.state_mut(),
                akita_transcript::ProtocolSiteId {
                    family: akita_transcript::SITE_FAMILY_NEXT_WITNESS,
                    level: u32::try_from(level).map_err(|_| AkitaError::InvalidProof)?,
                    stage: 1,
                    ..akita_transcript::ProtocolSiteId::default()
                },
                public_commitment.coeffs(),
            )
            .map_err(|_| AkitaError::InvalidProof)?;
            NextWitnessState::OuterPayload(public_commitment)
        }
        (
            crate::backend::NextWitnessBindingMessage::TerminalInnerState(message),
            akita_types::NextWitnessBindingPolicy::TerminalInnerState,
        ) => {
            akita_transcript::send_native_field_group(
                grinding.state_mut(),
                akita_transcript::ProtocolSiteId {
                    family: akita_transcript::SITE_FAMILY_NEXT_WITNESS,
                    level: u32::try_from(level).map_err(|_| AkitaError::InvalidProof)?,
                    stage: 2,
                    ..akita_transcript::ProtocolSiteId::default()
                },
                message.fields(),
            )
            .map_err(|_| AkitaError::InvalidProof)?;
            NextWitnessState::TerminalInnerState
        }
        _ => return Err(AkitaError::InvalidProof),
    };
    Ok(CommittedNextWitness {
        committed_witness_len,
        witness_handle,
        binding,
        commitment_material_handle,
    })
}

/// Prove one recursive fold level after the caller has built its ring-relation
/// equation and selected the commitment policy for the next `w`.
///
/// This function owns prover mechanics: build `w`, commit it, finish ring
/// switching, run stage-1/stage-2 sumchecks, and produce the next recursive
/// state.
///
/// # Errors
///
/// Returns an error if ring switching, recursive commitment, or either
/// sumcheck prover fails.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub(in crate::protocol::prove) fn prove_fold<F, E, B>(
    expanded: &akita_types::AkitaSetupDescriptor,
    prefix_slots: &SetupPrefixProverRegistry<F, B::CommitmentHandle>,
    backend: &B,
    session: &B::ProofSessionHandle,
    grinding: &mut akita_types::NativeProverGrinding<'_>,
    level: usize,
    lp: &CommittedGroupParams,
    next_params: FoldSuccessorParams<'_>,
    expected_output_witness_len: usize,
    next_witness_binding: akita_types::NextWitnessBindingPolicy,
    prepared_fold: PreparedFold<F, E, B::WitnessHandle>,
) -> Result<ProveLevelOutput<F, E, B::CommitmentMaterialHandle, B::WitnessHandle>, AkitaError>
where
    F: Field + CanonicalEncoding + AkitaSerialize + Ring + Unreduced + PseudoMersenne + 'static,
    <F as Unreduced>::Wide: From<F> + AdditiveGroup,
    E: FpExtEncoding<F>
        + ExtField<F>
        + Unreduced
        + Fold
        + Ring
        + MulBaseUnreduced<F>
        + AkitaSerialize
        + 'static,
    B: ProverBackend<F, E>,
{
    let opening_batch = prepared_fold.instance.opening_batch().clone();
    let challenge_field_bits = F::MODULUS_BITS
        .checked_mul(u32::try_from(E::DEGREE).map_err(|_| AkitaError::InvalidProof)?)
        .ok_or(AkitaError::InvalidProof)?;
    let relation_geometry = lp.relation_address_geometry(
        &opening_batch,
        E::DEGREE,
        next_params.inner_ring_dimension(),
        expected_output_witness_len,
    )?;
    let level_layout = akita_types::native_nonterminal_level_layout(
        F::MODULUS_BITS,
        challenge_field_bits,
        lp,
        relation_geometry,
        next_params.recursive().map(|params| &params.params),
    )?;
    let PreparedFold {
        instance,
        witness_handle,
        evaluation_trace_claim,
        relation_groups,
        evaluation_trace_claim_coefficients,
        evaluation_trace_basis,
        row_coefficients,
    } = prepared_fold;
    let next_opening_ring_dim = next_params.inner_ring_dimension();
    let CommittedNextWitness {
        committed_witness_len,
        witness_handle: mut next_witness,
        binding: next_commitment_binding,
        commitment_material_handle,
    } = commit_next_witness::<F, E, B>(
        backend,
        grinding,
        level,
        next_params,
        expected_output_witness_len,
        next_witness_binding,
        witness_handle,
    )?;
    let fold_level = u32::try_from(level)
        .map_err(|_| AkitaError::InvalidSetup("fold level exceeds u32".into()))?;
    let consumer = backend;
    let consumer_ctx = OperationCtx::new(
        backend,
        session,
        backend.proof_context(session, fold_level)?,
    );
    let next_opening_source_len = committed_witness_len / next_opening_ring_dim;
    let ring_switch = ring_switch_finalize::<F, E, B>(
        &consumer_ctx,
        &instance,
        grinding,
        fold_level,
        &next_witness,
        lp,
        next_opening_source_len,
        next_opening_ring_dim,
        row_coefficients.as_deref(),
        &evaluation_trace_claim_coefficients,
        &relation_groups,
    )
    .map_err(|err| AkitaError::InvalidInput(format!("ring-switch finalize failed: {err:?}")))?;
    let mut rs = ring_switch.output;
    let (stage1_stages, stage1_norm) = akita_types::DigitRangePlan::new(rs.b)?
        .proof_shapes_for_route(
            relation_geometry.relation_point_variable_count(),
            lp.inner().matrix.security_route(),
        )?;
    if !stage1_stages
        .iter()
        .copied()
        .eq(level_layout.stage1_stages())
        || stage1_norm != level_layout.stage1_norm()
        || level_layout.stage2_sumcheck()
            != akita_sumcheck::NativeSumcheckShape::new(
                relation_geometry.relation_point_variable_count(),
                3,
            )?
    {
        return Err(AkitaError::InvalidSetup(
            "native level grammar disagrees with fold plan".into(),
        ));
    }
    let relation_range_image_plan = ring_switch.relation_plan;
    let opening_semantics = ring_switch.opening_semantics;

    let relation_rhs_layout = relation_range_image_plan
        .relation_witness_geometry()
        .rhs_layout();
    let relation_claim = relation_claim_from_compressed_rhs_extension::<F, E>(
        relation_rhs_layout,
        &rs.tau1,
        rs.alpha,
        instance.rhs(),
    )?;
    let Stage1ProveOutput {
        point: stage1_point,
        range_image_evaluation,
        physical_l2,
    } = prove_stage1::<F, E, _>(
        &consumer_ctx,
        grinding,
        fold_level,
        &mut rs,
        lp,
        &relation_range_image_plan,
    )?;
    let physical_l2 = prepare_physical_l2_batch::<F, E>(grinding, level, physical_l2)?;
    let compression = prepare_stage2_compression::<F, E, _>(grinding, level, &mut rs)?;
    let batching_coeff =
        grinding.grinded_ext_challenge::<F, E>(akita_types::GrindingSite::Stage2Batch {
            level: fold_level,
        })?;
    let (linear_terms, scalar_opening_claim) = prepare_relation_sumcheck::<F, E, _>(
        lp,
        &opening_batch,
        opening_semantics,
        &relation_groups,
        &evaluation_trace_claim_coefficients,
        evaluation_trace_claim,
        evaluation_trace_basis,
        &relation_range_image_plan,
        &rs,
    )?;
    let relation_address_geometry = rs.relation_address_geometry;
    let tau1 = rs.tau1.clone();
    let alpha = rs.alpha;
    let relation_coefficients = match relation_groups
        .first()
        .ok_or(AkitaError::InvalidProof)?
        .kind()
    {
        OpeningFamily::SubringCoefficientPacking(_) => evaluation_trace_claim_coefficients.clone(),
        OpeningFamily::EvaluationTrace(_) => row_coefficients
            .clone()
            .unwrap_or_else(|| instance.gamma().iter().copied().map(E::lift_base).collect()),
    };
    let Stage2ProveOutput {
        challenges: sumcheck_challenges,
        witness_evaluation: w_eval,
    } = prove_stage2::<F, E, _>(
        &consumer_ctx,
        level,
        grinding,
        batching_coeff,
        rs,
        &stage1_point,
        range_image_evaluation,
        relation_claim,
        compression,
        physical_l2,
        linear_terms,
        scalar_opening_claim,
        &relation_range_image_plan,
        crate::backend::RelationWeightRequest {
            relation: &instance,
            parameters: lp,
            alpha,
            tau1: &tau1,
            claim_coefficients: &relation_coefficients,
            opening_source_len: next_opening_source_len,
            opening_ring_dimension: next_opening_ring_dim,
            relation_plan: &relation_range_image_plan,
            groups: &relation_groups,
        },
    )
    .map_err(|err| AkitaError::InvalidInput(format!("stage-2 proving failed: {err:?}")))?;
    akita_types::native_stage2_prover_w_eval::<F, E>(grinding, fold_level, w_eval)?;
    let stage3_sumcheck_proof = match next_params.recursive() {
        Some(next_fold_params) => prove_stage3::<F, E, _>(
            consumer,
            session,
            level,
            next_params.setup_contribution_mode(),
            expanded,
            prefix_slots,
            lp,
            &next_fold_params.params,
            &instance,
            &tau1,
            alpha,
            &sumcheck_challenges,
            relation_address_geometry,
            grinding,
        )?,
        None => None,
    };
    let setup_prefix_opening =
        stage3_sumcheck_proof.map(|stage3| (stage3.setup_prefix_point, stage3.setup_prefix_eval));

    crate::backend::OpaqueWitnessCommitKernel::advance_witness_level(consumer, &mut next_witness)?;

    Ok(ProveLevelOutput {
        next_state: SuffixProverState {
            witness_handle: next_witness,
            binding: next_commitment_binding,
            commitment_material: commitment_material_handle,
            sumcheck_challenges,
            opening: w_eval,
            setup_prefix_opening,
        },
    })
}

mod sumcheck;
use sumcheck::{prove_stage1, prove_stage2, prove_stage3};
