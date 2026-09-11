mod extension_claim;
mod single_field;

use super::*;
use crate::commitment::TerminalBindingState;
use crate::compute::{
    ComputeBackendSetup, DigitRowsComputeBackend, ProverComputeStack, RuntimeRingSwitchProveBackend,
};
use crate::protocol::sumcheck::relation_range_image::{
    prepare_coefficient_packing_linear_terms, PreparedProverLinearTerms,
};
use crate::protocol::sumcheck::DigitRangeProver;
use akita_algebra::offset_eq::{materialize_eq_tensor_left, OffsetEqWindow};
use jolt_field::AdditiveGroup;

use akita_types::{
    CoefficientPackingBatchSemantics, DigitRangeEqualityPoint, InnerCommitSecurityRoute,
    OpeningClaimsLayout, OpeningFamily, PhysicalResponsePlan, RelationRangeImagePlan,
};

pub(in crate::protocol::core) struct PhysicalL2ProverReplay<E: Field> {
    plan: PhysicalResponsePlan,
    point: Vec<E>,
    virtual_evaluations: Vec<E>,
    batching: Vec<E>,
    claim: E,
}

struct Stage1ProveOutput<E: Field> {
    proof: AkitaStage1Proof<E>,
    point: Vec<E>,
    range_image_evaluation: E,
    physical_l2: Option<PhysicalL2ProverReplay<E>>,
}

struct Stage2ProveOutput<E: Field> {
    proof: SumcheckProof<E>,
    challenges: Vec<E>,
    prover: RelationRangeImageProver<E>,
}

pub(in crate::protocol::core) use extension_claim::{
    prepare_extension_claim_fold, ExtensionOpeningSource,
};
pub(in crate::protocol::core) use single_field::prepare_single_field_fold;

pub(in crate::protocol::core) struct PreparedFold<F: Field, E: Field> {
    pub(in crate::protocol::core) instance: RingRelationInstance<F>,
    pub(in crate::protocol::core) witness: RingRelationWitness<F>,
    pub(in crate::protocol::core) opening_payload: RingVec<F>,
    pub(in crate::protocol::core) extension_opening_reduction:
        Option<ExtensionOpeningReductionProof<E>>,
    pub(in crate::protocol::core) evaluation_trace_claim: E,
    pub(in crate::protocol::core) relation_groups:
        Vec<crate::protocol::ring_relation::PreparedRelationGroup<F, E>>,
    pub(in crate::protocol::core) evaluation_trace_claim_coefficients: Vec<E>,
    pub(in crate::protocol::core) evaluation_trace_basis: BasisMode,
    pub(in crate::protocol::core) row_coefficients: Option<Vec<E>>,
}

/// Evaluate folded claims, derive the trace target, and build the ring-relation
/// instance/witness for one borrowed source-view set `Q: RootOpeningSource`.
#[allow(clippy::needless_lifetimes, clippy::too_many_arguments)]
pub(super) fn prepare_fold_relation<'a, F, E, T, Q, S, O, TS, R, SP>(
    stack: &'a ProverComputeStack<'a, F, O, TS, R, SP>,
    block_claims: ProverOpeningData<'a, E, Q, F, S>,
    protocol_points: &[Vec<E>],
    reduction: Option<ExtensionOpeningReduction<E>>,
    trace_opening_batch: &OpeningClaimsLayout,
    level: u32,
    level_params: &CommittedGroupParams,
    basis: BasisMode,
    pad_base_evals: bool,
    transcript: &mut T,
) -> Result<PreparedFold<F, E>, AkitaError>
where
    F: Field
        + CanonicalEncoding
        + akita_serialization::AkitaSerialize
        + Ring
        + Field
        + Unreduced
        + Field
        + 'static,
    <F as Unreduced>::Wide: From<F> + AdditiveGroup,
    E: FpExtEncoding<F>
        + ExtField<F>
        + Unreduced
        + Fold
        + Ring
        + MulBaseUnreduced<F>
        + AkitaSerialize,
    T: akita_types::ProverTranscriptGrinding<F>,
    Q: RootProverGroupOpening<F, E, O>,
    S: crate::commitment::InnerRelationState<F> + crate::commitment::OuterCompressionState<F>,
    O: DigitRowsComputeBackend<F>,
    R: DigitRowsComputeBackend<F> + RuntimeRingSwitchProveBackend<F>,
    TS: ComputeBackendSetup<F>,
    SP: crate::commitment::CommitmentStatePolicy<F>,
{
    let opening = stack.opening();
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
        if pad_base_evals {
            for coordinate in group_protocol_point {
                append_ext_field::<F, E, T>(transcript, ABSORB_EVALUATION_CLAIMS, coordinate);
            }
        }
        let prepared = block_claims
            .group(group_index)?
            .prepare_opening(
                opening,
                ring_dimension,
                group_protocol_point,
                basis,
                group_lp.num_positions_per_block(),
                group_lp.num_live_blocks(),
                group_alpha_bits,
                group_lp.opening_method(),
            )
            .map_err(|err| {
                AkitaError::InvalidInput(format!(
                    "root opening preparation group {group_index} failed: {err:?}"
                ))
            })?;
        scalar_openings.extend_from_slice(&prepared.scalar_openings);
        prepared_group_openings.push(prepared);
    }
    if reduction.is_none() {
        append_claim_values_to_transcript::<F, E, T>(&scalar_openings, transcript);
    }
    let crate::protocol::ring_relation::PreparedRingRelationOutput {
        relation:
            crate::protocol::ring_relation::PreparedRingRelation {
                instance,
                witness,
                groups: relation_groups,
            },
        trace_claim,
        row_coefficients,
    } = RingRelationProver::prepare(
        opening,
        stack.ring_switch(),
        prepared_group_openings,
        block_claims,
        level_params.clone(),
        transcript,
        level,
        &reduction,
        &scalar_openings,
        trace_opening_batch,
    )
    .map_err(|err| {
        AkitaError::InvalidInput(format!("ring relation preparation failed: {err:?}"))
    })?;
    let opening_payload = if level_params.payload_mode.is_compressed() {
        witness.opening_payload()?
    } else {
        instance.v().clone()
    };
    let extension_opening_reduction = reduction.map(|reduction| reduction.proof);
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
        witness,
        opening_payload,
        extension_opening_reduction,
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
pub(in crate::protocol::core) enum FoldSuccessorParams<'a> {
    Recursive(&'a FoldParams),
    Terminal(&'a TerminalFoldParams),
}

impl<'a> FoldSuccessorParams<'a> {
    fn inner_ring_dimension(self) -> usize {
        match self {
            Self::Recursive(params) => params.params.d_a(),
            Self::Terminal(params) => params.d_a(),
        }
    }

    fn log_basis_inner(self) -> u32 {
        match self {
            Self::Recursive(params) => params.params.open().digits.log_basis,
            Self::Terminal(params) => params.inner.digits.log_basis,
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

struct CommittedNextWitness<F: Field, S> {
    logical_w: crate::backend::RecursiveWitnessFlat,
    committed_witness_len: usize,
    state: NextWitnessStateOutput<F, S>,
}

fn prepare_physical_l2_batch<F, E, T>(
    transcript: &mut T,
    level: usize,
    physical_l2: Option<PhysicalL2ProverReplay<E>>,
) -> Result<Option<PhysicalL2ProverReplay<E>>, AkitaError>
where
    F: Field + CanonicalEncoding + AkitaSerialize,
    E: ExtField<F> + AkitaSerialize,
    T: akita_types::ProverTranscriptGrinding<F>,
{
    let Some(mut replay) = physical_l2 else {
        return Ok(None);
    };
    let level = u32::try_from(level)
        .map_err(|_| AkitaError::InvalidSetup("fold level exceeds u32".into()))?;
    transcript.grind_query(akita_types::GrindingSite::L2VirtualBatch { level })?;
    let eta = sample_ext_challenge::<F, E, T>(
        transcript,
        akita_transcript::labels::CHALLENGE_L2_VIRTUAL_BATCH,
    );
    let mut power = E::one();
    replay.batching = Vec::with_capacity(replay.virtual_evaluations.len());
    replay.claim = E::zero();
    for &evaluation in &replay.virtual_evaluations {
        replay.batching.push(power);
        replay.claim += evaluation * power;
        power *= eta;
    }
    Ok(Some(replay))
}

fn prepare_stage2_compression<F, E, T>(
    transcript: &mut T,
    level: usize,
    rs: &mut RingSwitchOutput<E>,
) -> Result<stages::Stage2Compression<E>, AkitaError>
where
    F: Field + CanonicalEncoding + AkitaSerialize,
    E: ExtField<F> + AkitaSerialize,
    T: akita_types::ProverTranscriptGrinding<F>,
{
    let compression = std::mem::replace(
        &mut rs.compression,
        crate::protocol::ring_switch::RingSwitchCompression::Raw,
    );
    match compression {
        crate::protocol::ring_switch::RingSwitchCompression::Raw => {
            Ok(stages::Stage2Compression::Raw)
        }
        crate::protocol::ring_switch::RingSwitchCompression::QuotientLift { weights, support } => {
            let level = u32::try_from(level)
                .map_err(|_| AkitaError::InvalidSetup("fold level exceeds u32".into()))?;
            transcript.grind_query(akita_types::GrindingSite::CompressionBinary { level })?;
            Ok(stages::Stage2Compression::QuotientLift {
                weights,
                support,
                binary_batching: sample_ext_challenge::<F, E, T>(
                    transcript,
                    CHALLENGE_COMPRESSION_BINARY,
                ),
            })
        }
        crate::protocol::ring_switch::RingSwitchCompression::ReducedEvaluation { support } => {
            let level = u32::try_from(level)
                .map_err(|_| AkitaError::InvalidSetup("fold level exceeds u32".into()))?;
            transcript.grind_query(akita_types::GrindingSite::CompressionBinary { level })?;
            Ok(stages::Stage2Compression::ReducedEvaluation {
                support,
                binary_batching: sample_ext_challenge::<F, E, T>(
                    transcript,
                    CHALLENGE_COMPRESSION_BINARY,
                ),
            })
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn prepare_relation_sumcheck<F, E>(
    lp: &CommittedGroupParams,
    opening_batch: &OpeningClaimsLayout,
    opening_semantics: OpeningFamily<(), CoefficientPackingBatchSemantics<E>>,
    relation_groups: &[crate::protocol::ring_relation::PreparedRelationGroup<F, E>],
    evaluation_trace_claim_coefficients: &[E],
    evaluation_trace_claim: E,
    evaluation_trace_basis: BasisMode,
    relation_range_image_plan: &RelationRangeImagePlan,
    rs: &RingSwitchOutput<E>,
) -> Result<(PreparedProverLinearTerms<E>, E), AkitaError>
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
            let mut combined_terms: Option<PreparedProverLinearTerms<E>> = None;
            let mut authenticated_opening = E::zero();
            let mut weighted_opening_claim = E::zero();
            for semantics in batch.into_groups() {
                let group_index = semantics.group_index();
                let geometry = semantics.geometry();
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
                let prepared = prepare_coefficient_packing_linear_terms(semantics, group_opening)?;
                if prepared.group_index != group_index || prepared.geometry != geometry {
                    return Err(AkitaError::InvalidProof);
                }
                weighted_opening_claim += prepared.weighted_scalar_opening_claim;
                if let Some(combined) = combined_terms.as_mut() {
                    combined.merge(prepared.linear_terms)?;
                } else {
                    combined_terms = Some(prepared.linear_terms);
                }
            }
            if authenticated_opening != evaluation_trace_claim {
                return Err(AkitaError::InvalidProof);
            }
            Ok((
                combined_terms.ok_or(AkitaError::InvalidProof)?,
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
            let semantic_trace = build_evaluation_trace_weights::<F, E>(EvaluationTraceInputs {
                digit_witness_domain: relation_range_image_plan.digit_witness_domain(),
                relation_coefficient_block_len: rs
                    .relation_address_geometry
                    .relation_coefficient_block_len(),
                witness_layout: relation_range_image_plan.witness_layout(),
                level_params: lp,
                opening_batch,
                prepared_points: &evaluation_trace_points,
                claim_coefficients: evaluation_trace_claim_coefficients,
                basis: evaluation_trace_basis,
            })?;
            Ok((
                PreparedProverLinearTerms::from_evaluation_trace(
                    &semantic_trace,
                    rs.relation_address_geometry
                        .relation_coefficient_block_len(),
                    evaluation_trace_weight,
                )?,
                evaluation_trace_weight * evaluation_trace_claim,
            ))
        }
    }
}

/// Build, commit, and transcript-bind the witness consumed by the successor.
/// The logical witness is computed once and retained for ring switching.
#[allow(clippy::too_many_arguments)]
fn commit_next_witness<'stack, F, E, T, O, TS, R, SP, Cfg>(
    stack: &'stack ProverComputeStack<'stack, F, O, TS, R, SP>,
    transcript: &mut T,
    level: usize,
    lp: &CommittedGroupParams,
    next_params: FoldSuccessorParams<'_>,
    expected_output_witness_len: usize,
    instance: &RingRelationInstance<F>,
    witness: RingRelationWitness<F>,
) -> Result<CommittedNextWitness<F, SP::State>, AkitaError>
where
    F: Field + CanonicalEncoding + Unreduced + PseudoMersenne + AkitaSerialize,
    E: ExtField<F>,
    T: akita_types::ProverTranscriptGrinding<F>,
    O: ComputeBackendSetup<F>,
    TS: ComputeBackendSetup<F>,
    R: RuntimeRingSwitchProveBackend<F> + ComputeBackendSetup<F> + 'stack,
    <R as ComputeBackendSetup<F>>::PreparedSetup: 'stack,
    SP: crate::commitment::CommitmentStatePolicy<F>,
    SP::State: crate::commitment::TerminalBindingState<F>,
    Cfg: CommitmentConfig<Field = F, ExtField = E>,
{
    let next_opening_ring_dim = next_params.inner_ring_dimension();
    let logical_w = ring_switch_build_w::<F, R>(instance, witness, stack.ring_switch(), lp)
        .map_err(|err| {
            AkitaError::InvalidInput(format!("ring-switch witness build failed: {err:?}"))
        })?;
    let committed_witness_len = akita_types::witness_commitment_domain_len(
        logical_w.live_coeff_len(),
        next_opening_ring_dim,
    )?;
    if logical_w.live_coeff_len() != expected_output_witness_len {
        return Err(AkitaError::InvalidSetup(format!(
            "scheduled fold level {level} produced unexpected next-w length: expected={expected_output_witness_len}, actual={}",
            logical_w.live_coeff_len()
        )));
    }
    let logical_w = logical_w.align_for_commitment_ring_dim(next_opening_ring_dim)?;
    let _span = tracing::info_span!("commit_w_level", level).entered();
    let state = match next_params {
        FoldSuccessorParams::Recursive(params) => crate::commit_w::<Cfg, _>(
            &params.params,
            level
                .checked_add(1)
                .ok_or_else(|| AkitaError::InvalidSetup("fold level overflow".into()))?,
            stack.commitment(),
            &logical_w,
        )?,
        FoldSuccessorParams::Terminal(params) => {
            crate::commit_terminal_w::<Cfg, _>(params, stack.commitment(), &logical_w)?
        }
    };
    drop(_span);
    match &state.binding {
        NextWitnessState::OuterPayload(commitment) => {
            transcript.append_serde(ABSORB_NEXT_LEVEL_WITNESS_BINDING, commitment);
        }
        NextWitnessState::TerminalInnerState => {
            let message = state.prover_state.terminal_t_fields_message()?;
            transcript
                .absorb_and_record_bytes(ABSORB_NEXT_LEVEL_WITNESS_BINDING, message.as_bytes());
        }
    }
    Ok(CommittedNextWitness {
        logical_w,
        committed_witness_len,
        state,
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
pub(in crate::protocol::core) fn prove_fold<'stack, F, E, T, O, TS, R, SP, Cfg>(
    expanded: &Arc<AkitaExpandedSetup<F>>,
    prefix_slots: &SetupPrefixProverRegistry<F>,
    stack: &'stack ProverComputeStack<'stack, F, O, TS, R, SP>,
    transcript: &mut T,
    level: usize,
    lp: &CommittedGroupParams,
    next_params: FoldSuccessorParams<'_>,
    expected_output_witness_len: usize,
    prepared_fold: PreparedFold<F, E>,
) -> Result<ProveLevelOutput<F, E, SP::State>, AkitaError>
where
    F: Field
        + CanonicalEncoding
        + Field
        + Unreduced
        + Field
        + Field
        + PseudoMersenne
        + AkitaSerialize,
    E: ExtField<F>
        + FpExtEncoding<F>
        + Unreduced
        + Fold
        + Ring
        + MulBaseUnreduced<F>
        + AkitaSerialize,
    T: akita_types::ProverTranscriptGrinding<F>,
    O: ComputeBackendSetup<F>,
    TS: ComputeBackendSetup<F>,
    R: RuntimeRingSwitchProveBackend<F> + ComputeBackendSetup<F> + 'stack,
    <R as ComputeBackendSetup<F>>::PreparedSetup: 'stack,
    SP: crate::commitment::CommitmentStatePolicy<F>,
    SP::State: crate::commitment::TerminalBindingState<F>,
    Cfg: CommitmentConfig<Field = F, ExtField = E>,
{
    let opening_batch = prepared_fold.instance.opening_batch().clone();
    let PreparedFold {
        instance,
        witness,
        opening_payload,
        extension_opening_reduction,
        evaluation_trace_claim,
        relation_groups,
        evaluation_trace_claim_coefficients,
        evaluation_trace_basis,
        row_coefficients,
    } = prepared_fold;
    let next_opening_ring_dim = next_params.inner_ring_dimension();
    let CommittedNextWitness {
        logical_w,
        committed_witness_len,
        state: next_commitment,
    } = commit_next_witness::<F, E, T, O, TS, R, SP, Cfg>(
        stack,
        transcript,
        level,
        lp,
        next_params,
        expected_output_witness_len,
        &instance,
        witness,
    )?;
    let next_opening_source_len = committed_witness_len / next_opening_ring_dim;
    let ring_switch = ring_switch_finalize::<F, E, T>(
        &instance,
        expanded.as_ref(),
        transcript,
        u32::try_from(level)
            .map_err(|_| AkitaError::InvalidSetup("fold level exceeds u32".into()))?,
        &logical_w,
        lp,
        next_opening_source_len,
        next_opening_ring_dim,
        row_coefficients.as_deref(),
        &evaluation_trace_claim_coefficients,
        &relation_groups,
    )
    .map_err(|err| AkitaError::InvalidInput(format!("ring-switch finalize failed: {err:?}")))?;
    let mut rs = ring_switch.output;
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
        proof: stage1_proof,
        point: stage1_point,
        range_image_evaluation,
        physical_l2,
    } = prove_stage1::<F, E, T>(
        transcript,
        u32::try_from(level)
            .map_err(|_| AkitaError::InvalidSetup("fold level exceeds u32".into()))?,
        &mut rs,
        lp,
        &relation_range_image_plan,
    )?;
    transcript.append_serde(
        ABSORB_RANGE_IMAGE_EVALUATION,
        &stage1_proof.range_image_evaluation,
    );
    let physical_l2 = prepare_physical_l2_batch::<F, E, T>(transcript, level, physical_l2)?;
    let compression = prepare_stage2_compression::<F, E, T>(transcript, level, &mut rs)?;
    transcript.grind_query(akita_types::GrindingSite::Stage2Batch {
        level: u32::try_from(level)
            .map_err(|_| AkitaError::InvalidSetup("fold level exceeds u32".into()))?,
    })?;
    let batching_coeff: E = sample_ext_challenge::<F, E, T>(transcript, CHALLENGE_SUMCHECK_BATCH);
    let (linear_terms, scalar_opening_claim) = prepare_relation_sumcheck::<F, E>(
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
    let Stage2ProveOutput {
        proof: stage2_sumcheck_proof,
        challenges: sumcheck_challenges,
        prover: stage2_prover,
    } = prove_stage2::<F, E, T>(
        level,
        transcript,
        batching_coeff,
        rs,
        &stage1_point,
        range_image_evaluation,
        relation_claim,
        compression,
        physical_l2,
        linear_terms,
        scalar_opening_claim,
        relation_range_image_plan,
    )
    .map_err(|err| AkitaError::InvalidInput(format!("stage-2 proving failed: {err:?}")))?;
    let w_eval = {
        let _span = tracing::info_span!("multilinear_eval", level).entered();
        stage2_prover.final_w_eval()
    };
    let proof_w_eval = w_eval;
    transcript.append_serde(ABSORB_STAGE2_NEXT_W_EVAL, &proof_w_eval);
    let stage3_sumcheck_proof = match next_params.recursive() {
        Some(next_fold_params) => prove_stage3::<F, E, T>(
            level,
            next_params.setup_contribution_mode(),
            expanded.as_ref(),
            prefix_slots,
            lp,
            &next_fold_params.params,
            &instance,
            &tau1,
            alpha,
            &sumcheck_challenges,
            relation_address_geometry,
            transcript,
        )?,
        None => None,
    };
    let (stage3_sumcheck_proof, setup_prefix_opening) = if let Some(stage3) = stage3_sumcheck_proof
    {
        let setup_prefix_eval = stage3.proof.setup_prefix_eval;
        (
            Some(stage3.proof),
            Some((stage3.setup_prefix_point, setup_prefix_eval)),
        )
    } else {
        (None, None)
    };
    let NextWitnessStateOutput {
        witness: packed_witness,
        binding,
        prover_state: committed_state,
    } = next_commitment;
    let (proof_binding, next_binding) = match binding {
        NextWitnessState::OuterPayload(commitment) => (
            akita_types::NextWitnessBinding::OuterPayload(commitment.clone().into_compact()),
            NextWitnessState::OuterPayload(commitment),
        ),
        NextWitnessState::TerminalInnerState => (
            akita_types::NextWitnessBinding::TerminalInnerState,
            NextWitnessState::TerminalInnerState,
        ),
    };
    let level_proof = FoldLevelProof {
        extension_opening_reduction,
        opening_payload: opening_payload.into_compact(),
        stage1: stage1_proof,
        stage2: AkitaStage2Proof {
            sumcheck_proof: stage2_sumcheck_proof,
            next_witness_binding: proof_binding,
            next_w_eval: proof_w_eval,
        },
        stage3_sumcheck_proof,
    };

    let (committed_witness, logical_w) = match packed_witness {
        Some(packed_witness) => (packed_witness, Some(logical_w)),
        None => (logical_w, None),
    };

    Ok(ProveLevelOutput {
        level_proof,
        next_state: SuffixProverState {
            w: committed_witness,
            logical_w,
            binding: next_binding,
            prover_state: committed_state,
            log_basis: next_params.log_basis_inner(),
            sumcheck_challenges,
            opening: w_eval,
            setup_prefix_opening,
        },
    })
}

mod stages;
use stages::{prove_stage1, prove_stage2, prove_stage3};
