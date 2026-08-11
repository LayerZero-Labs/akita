mod extension_claim;
mod single_field;

use super::*;
use crate::compute::{
    ComputeBackendSetup, DigitRowsComputeBackend, ProverComputeStack, RuntimeCommitBackendFor,
    RuntimeRingSwitchProveBackend,
};
use crate::protocol::sumcheck::relation_range_image::PreparedProverEvaluationTrace;
use crate::protocol::sumcheck::DigitRangeProver;
use crate::RecursiveWitnessFlat;
use akita_field::unreduced::ReduceTo;
use akita_field::AdditiveGroup;

use akita_types::{
    dispatch_for_field, DigitRangeEqualityPoint, DigitRangePlan, OpeningClaimsLayout,
    RelationRangeImagePlan,
};

pub(in crate::protocol::core) use extension_claim::{
    prepare_extension_claim_fold, ExtensionOpeningSource,
};
pub(in crate::protocol::core) use single_field::prepare_single_field_fold;

pub(in crate::protocol::core) struct PreparedFold<F: FieldCore, E: FieldCore> {
    pub(in crate::protocol::core) instance: RingRelationInstance<F>,
    pub(in crate::protocol::core) witness: RingRelationWitness<F>,
    pub(in crate::protocol::core) opening_payload: RingVec<F>,
    pub(in crate::protocol::core) extension_opening_reduction:
        Option<ExtensionOpeningReductionProof<E>>,
    pub(in crate::protocol::core) evaluation_trace_claim: E,
    pub(in crate::protocol::core) evaluation_trace_points: Vec<PreparedOpeningPoint<F, E>>,
    pub(in crate::protocol::core) evaluation_trace_claim_coefficients: Vec<E>,
    pub(in crate::protocol::core) evaluation_trace_basis: BasisMode,
    pub(in crate::protocol::core) row_coefficients: Option<Vec<E>>,
}

pub(super) fn prepare_non_eor_opening<'a, F, E, P, V>(
    block_claims: &ProverOpeningData<'a, E, P, F>,
    opening_batch: &OpeningClaimsLayout,
    validate_non_eor: V,
) -> Result<Vec<Vec<E>>, AkitaError>
where
    F: FieldCore,
    E: ExtField<F>,
    P: RootProverGroupMeta<F>,
    V: FnOnce() -> Result<(), AkitaError>,
{
    validate_non_eor()?;
    (0..opening_batch.num_groups())
        .map(|group_index| {
            block_claims
                .opening_claims()
                .group_point(group_index)
                .map(<[E]>::to_vec)
        })
        .collect()
}

/// Borrowed/owned argument bundle for [`finish_prepared_fold`].
pub(super) struct FinishFoldArgs<'a, 'p, F, E, T, Q, C, O, TS, R>
where
    F: FieldCore + CanonicalField,
    E: FieldCore,
    C: ComputeBackendSetup<F>,
    O: ComputeBackendSetup<F>,
    TS: ComputeBackendSetup<F>,
    R: ComputeBackendSetup<F>,
{
    stack: &'a ProverComputeStack<'a, F, C, O, TS, R>,
    block_claims: ProverOpeningData<'a, E, Q, F>,
    protocol_points: &'a [Vec<E>],
    reduction: Option<ExtensionOpeningReduction<E>>,
    row_coefficients: Option<Vec<E>>,
    trace_opening_batch: &'a OpeningClaimsLayout,
    level_params: &'a CommittedGroupParams,
    basis: BasisMode,
    pad_base_evals: bool,
    transcript: &'p mut T,
}

/// Evaluate folded claims, derive the trace target, and build the ring-relation
/// instance/witness for one borrowed source-view set `Q: RootOpeningSource`.
#[allow(clippy::needless_lifetimes)]
pub(super) fn finish_prepared_fold<'a, 'p, F, E, T, Q, C, O, TS, R>(
    args: FinishFoldArgs<'a, 'p, F, E, T, Q, C, O, TS, R>,
) -> Result<PreparedFold<F, E>, AkitaError>
where
    F: FieldCore
        + CanonicalField
        + FromPrimitiveInt
        + HalvingField
        + HasWide
        + RandomSampling
        + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F> + AdditiveGroup,
    E: FpExtEncoding<F>
        + ExtField<F>
        + HasUnreducedOps
        + HasOptimizedFold
        + FromPrimitiveInt
        + MulBaseUnreduced<F>
        + AkitaSerialize,
    T: Transcript<F> + ProverTranscriptGrind<F>,
    Q: RootProverGroupOpening<F, E, O>,
    O: DigitRowsComputeBackend<F>,
    R: DigitRowsComputeBackend<F> + RuntimeRingSwitchProveBackend<F>,
    C: ComputeBackendSetup<F>,
    TS: ComputeBackendSetup<F>,
{
    let FinishFoldArgs {
        stack,
        block_claims,
        protocol_points,
        reduction,
        row_coefficients,
        trace_opening_batch,
        level_params,
        basis,
        pad_base_evals,
        transcript,
    } = args;
    let opening = stack.opening();
    // A-role operation: prepare each group at its native A dimension,
    // fold-evaluate its claim polynomials, and derive scalar openings before
    // leaving the typed dispatch arm. Typed fold outputs cross the boundary
    // only through D-free `PreparedOpeningPoint` / `RingVec` carriers.
    let opening_batch = block_claims
        .opening_claims()
        .layout()
        .map_err(|err| AkitaError::InvalidInput(format!("opening batch layout failed: {err:?}")))?;
    let final_group_index = opening_batch.root_final_group_index()?;
    let mut prepared_points = Vec::with_capacity(opening_batch.num_groups());
    let mut e_folded_by_claim = Vec::with_capacity(opening_batch.num_total_polynomials());
    let mut scalar_openings = Vec::with_capacity(opening_batch.num_total_polynomials());
    for group_index in 0..opening_batch.num_groups() {
        let group_lp = level_params
            .group_params(&opening_batch, group_index)
            .map_err(|err| {
                AkitaError::InvalidInput(format!("root group params {group_index} failed: {err:?}"))
            })?;
        let group_dims = level_params.group_role_dims(&opening_batch, group_index)?;
        let group_alpha_bits = group_dims.d_a().trailing_zeros() as usize;
        let target_len = group_alpha_bits
            .checked_add(group_lp.position_index_bits())
            .and_then(|n| n.checked_add(group_lp.block_index_bits()))
            .ok_or_else(|| {
                AkitaError::InvalidSetup("group opening point length overflow".to_string())
            })?;
        let group_protocol_point = protocol_points
            .get(group_index)
            .ok_or(AkitaError::InvalidProof)?;
        let point_width_is_valid = if pad_base_evals && group_index == final_group_index {
            group_protocol_point.len() <= target_len
        } else {
            group_protocol_point.len() == target_len
        };
        if !point_width_is_valid {
            return Err(AkitaError::InvalidPointDimension {
                expected: target_len,
                actual: group_protocol_point.len(),
            });
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
                group_dims.d_a(),
                group_protocol_point,
                basis,
                group_lp.num_positions_per_block(),
                group_lp.num_live_blocks(),
                group_alpha_bits,
            )
            .map_err(|err| {
                AkitaError::InvalidInput(format!(
                    "root opening preparation group {group_index} failed: {err:?}"
                ))
            })?;
        prepared_points.push(prepared.point);
        e_folded_by_claim.extend(prepared.folded_by_claim);
        scalar_openings.extend(prepared.scalar_openings);
    }
    let (trace_claim, row_coefficients) = prepare_evaluation_trace_claim::<F, E, T>(
        &reduction,
        &scalar_openings,
        trace_opening_batch,
        row_coefficients,
        transcript,
    )
    .map_err(|err| {
        AkitaError::InvalidInput(format!("prepare evaluation-trace claim failed: {err:?}"))
    })?;
    let row_coefficient_rings = dispatch_for_field!(
        ProtocolDispatchSlot::Role(RingRole::Inner),
        F,
        level_params.role_dims().d_a(),
        |D| {
            let row_coefficient_rings = row_coefficient_rings::<F, E, D>(&row_coefficients)
                .map_err(|err| {
                    AkitaError::InvalidInput(format!("row coefficient rings failed: {err:?}"))
                })?;
            Ok::<_, AkitaError>(RingVec::from_ring_elems(&row_coefficient_rings))
        }
    )
    .map_err(|err| {
        AkitaError::InvalidInput(format!("root row-coefficient preparation failed: {err:?}"))
    })?;
    let (instance, witness) = RingRelationProver::new(
        opening,
        stack.ring_switch(),
        prepared_points
            .iter()
            .map(|prepared| prepared.ring_multiplier_point.clone())
            .collect::<Vec<_>>(),
        block_claims,
        e_folded_by_claim,
        level_params.clone(),
        transcript,
        row_coefficient_rings,
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
    let clear_recursive_trace = pad_base_evals && !level_params.has_precommitted_groups();
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
        evaluation_trace_points: prepared_points,
        evaluation_trace_claim_coefficients,
        evaluation_trace_basis: basis,
        row_coefficients,
    })
}

/// Typed commitment parameters for the witness produced by a non-terminal
/// fold. The terminal variant exposes only its inner commitment.
#[derive(Clone, Copy)]
pub(in crate::protocol::core) enum FoldSuccessorParams<'a> {
    Recursive(&'a RecursiveFoldParams),
    Terminal(&'a TerminalCommittedGroupParams),
}

impl<'a> FoldSuccessorParams<'a> {
    fn inner_ring_dimension(self) -> usize {
        match self {
            Self::Recursive(params) => params.witness.d_a(),
            Self::Terminal(params) => params.d_a(),
        }
    }

    fn log_basis_inner(self) -> u32 {
        match self {
            Self::Recursive(params) => params.witness.log_basis_open,
            Self::Terminal(params) => params.log_basis_inner,
        }
    }

    fn recursive(self) -> Option<&'a RecursiveFoldParams> {
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
pub(in crate::protocol::core) fn prove_fold<'stack, F, E, T, C, O, TS, R, Cfg>(
    expanded: &Arc<AkitaExpandedSetup<F>>,
    prefix_slots: &SetupPrefixProverRegistry<F>,
    stack: &'stack ProverComputeStack<'stack, F, C, O, TS, R>,
    transcript: &mut T,
    level: usize,
    lp: &CommittedGroupParams,
    next_params: Option<FoldSuccessorParams<'_>>,
    expected_output_witness_len: Option<usize>,
    next_witness_binding: Option<akita_types::NextWitnessBindingPolicy>,
    prepared_fold: PreparedFold<F, E>,
) -> Result<ProveLevelOutput<F, E>, AkitaError>
where
    F: FieldCore
        + CanonicalField
        + RandomSampling
        + HasWide
        + HalvingField
        + Invertible
        + PseudoMersenneField
        + AkitaSerialize,
    E: ExtField<F>
        + FpExtEncoding<F>
        + HasUnreducedOps
        + HasOptimizedFold
        + FromPrimitiveInt
        + MulBaseUnreduced<F>
        + AkitaSerialize
        + crate::kernels::sumcheck::SumcheckTableOperations<F>,
    T: Transcript<F> + ProverTranscriptGrind<F>,
    C: RuntimeCommitBackendFor<F, RecursiveWitnessFlat> + ComputeBackendSetup<F> + 'stack,
    O: ComputeBackendSetup<F>,
    TS: ComputeBackendSetup<F>,
    R: RuntimeRingSwitchProveBackend<F> + ComputeBackendSetup<F> + 'stack,
    <C as ComputeBackendSetup<F>>::PreparedSetup: 'stack,
    <R as ComputeBackendSetup<F>>::PreparedSetup: 'stack,
    Cfg: CommitmentConfig<Field = F, ExtField = E>,
{
    let opening_batch = prepared_fold.instance.opening_batch();
    let fold_grind_nonce = prepared_fold.witness.fold_grind_nonce;
    let next_params = next_params.ok_or_else(|| {
        AkitaError::InvalidSetup("non-terminal fold is missing successor params".into())
    })?;
    let next_opening_ring_dim = next_params.inner_ring_dimension();
    let logical_w = ring_switch_build_w::<F, R>(
        &prepared_fold.instance,
        prepared_fold.witness,
        stack.ring_switch(),
        lp,
    )
    .map_err(|err| AkitaError::InvalidInput(format!("ring-switch witness build failed: {err:?}")))?
    .align_for_commitment_ring_dim(next_opening_ring_dim)?;
    let committed_witness_len = logical_w.committed_coeff_len()?;
    if Some(logical_w.live_coeff_len()) != expected_output_witness_len {
        return Err(AkitaError::InvalidSetup(format!(
            "scheduled fold level {level} produced unexpected next-w length: expected={expected_output_witness_len:?}, actual={}",
            logical_w.live_coeff_len()
        )));
    }
    let _span = tracing::info_span!("commit_w_level", level).entered();
    let next_commitment = match next_params {
        FoldSuccessorParams::Recursive(params) => {
            if next_witness_binding != Some(akita_types::NextWitnessBindingPolicy::OuterPayload) {
                return Err(AkitaError::InvalidSetup(
                    "recursive successor requires outer-payload binding".into(),
                ));
            }
            crate::commit_w::<Cfg, C>(&params.witness, expanded, stack.commit(), &logical_w)?
        }
        FoldSuccessorParams::Terminal(params) => {
            if next_witness_binding
                != Some(akita_types::NextWitnessBindingPolicy::TerminalInnerState)
            {
                return Err(AkitaError::InvalidSetup(
                    "terminal successor requires canonical inner-state binding".into(),
                ));
            }
            crate::commit_terminal_w::<Cfg, C>(params, expanded, stack.commit(), &logical_w)?
        }
    };
    drop(_span);
    match &next_commitment.binding {
        NextWitnessState::OuterPayload(commitment) => {
            transcript.append_serde(ABSORB_NEXT_LEVEL_WITNESS_BINDING, commitment);
        }
        NextWitnessState::TerminalInnerState => {
            let rows = next_commitment.hint.inner_rows();
            let t_state = match rows {
                [t_state] => t_state,
                _ => return Err(AkitaError::InvalidProof),
            };
            let bytes = akita_types::raw_field_segment_bytes(t_state)?;
            transcript.absorb_and_record_bytes(ABSORB_NEXT_LEVEL_WITNESS_BINDING, &bytes);
        }
    }
    let next_opening_source_len = committed_witness_len / next_opening_ring_dim;
    let mut rs = ring_switch_finalize::<F, E, T>(
        &prepared_fold.instance,
        expanded.as_ref(),
        transcript,
        &logical_w,
        lp,
        next_opening_source_len,
        next_opening_ring_dim,
        prepared_fold.row_coefficients.as_deref(),
    )
    .map_err(|err| AkitaError::InvalidInput(format!("ring-switch finalize failed: {err:?}")))?;

    let relation_range_image_plan = RelationRangeImagePlan::new(
        rs.relation_address_geometry,
        DigitRangePlan::new(rs.b)?,
        prepared_fold.instance.segment_layout(lp, None)?,
        prepared_fold.instance.opening_batch(),
    )?;

    let relation_rhs_layout = relation_rhs_layout_for(lp, prepared_fold.instance.opening_batch())?;
    let relation_claim = relation_claim_from_compressed_rhs_extension::<F, E>(
        &relation_rhs_layout,
        &rs.tau1,
        rs.alpha,
        prepared_fold.instance.rhs(),
    )?;
    let (stage1_proof, stage1_point, range_image_evaluation) =
        prove_stage1::<F, E, T>(transcript, &mut rs, &relation_range_image_plan)?;
    transcript.append_serde(
        ABSORB_RANGE_IMAGE_EVALUATION,
        &stage1_proof.range_image_evaluation,
    );
    let stage1_proof = Some(stage1_proof);
    let binary_batching = lp
        .payload_mode
        .is_compressed()
        .then(|| sample_ext_challenge::<F, E, T>(transcript, CHALLENGE_COMPRESSION_BINARY));
    let batching_coeff: E = sample_ext_challenge::<F, E, T>(transcript, CHALLENGE_SUMCHECK_BATCH);
    // EvaluationTrace is the last padded relation row: weight openings by
    // `eq(tau1, EvaluationTrace_row_index)`.
    let evaluation_trace_row = lp.evaluation_trace_row_index(opening_batch)?;
    let evaluation_trace_weight = evaluation_trace_row_weight(evaluation_trace_row, &rs.tau1)?;
    let trace_opening_claim = evaluation_trace_weight * prepared_fold.evaluation_trace_claim;
    ensure_trace_stage2_supported(E::EXT_DEGREE)?;
    let evaluation_trace_points = &prepared_fold.evaluation_trace_points;
    let trace_preparation_span = tracing::info_span!(
        "stage2_evaluation_trace_preparation",
        claims = opening_batch.num_total_polynomials(),
        groups = opening_batch.num_groups(),
        chunks = relation_range_image_plan.witness_layout().units().len(),
        coeff_count = rs
            .relation_address_geometry
            .relation_coefficient_block_len(),
    )
    .entered();
    let semantic_trace = build_evaluation_trace_weights::<F, E>(EvaluationTraceInputs {
        digit_witness_domain: relation_range_image_plan.digit_witness_domain(),
        relation_coefficient_block_len: rs
            .relation_address_geometry
            .relation_coefficient_block_len(),
        witness_layout: relation_range_image_plan.witness_layout(),
        level_params: lp,
        opening_batch,
        prepared_points: evaluation_trace_points,
        claim_coefficients: &prepared_fold.evaluation_trace_claim_coefficients,
        basis: prepared_fold.evaluation_trace_basis,
    })?;
    let evaluation_trace = PreparedProverEvaluationTrace::new(
        &semantic_trace,
        rs.relation_address_geometry
            .relation_coefficient_block_len(),
        evaluation_trace_weight,
    )?;
    drop(trace_preparation_span);
    let relation_address_geometry = rs.relation_address_geometry;
    let tau1 = rs.tau1.clone();
    let alpha = rs.alpha;
    let (stage2_sumcheck_proof, sumcheck_challenges, stage2_prover) = prove_stage2::<F, E, T>(
        level,
        transcript,
        batching_coeff,
        rs,
        &stage1_point,
        range_image_evaluation,
        relation_claim,
        binary_batching,
        evaluation_trace,
        trace_opening_claim,
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
            &next_fold_params.witness,
            &prepared_fold.instance,
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
    let stage1_proof = stage1_proof.ok_or_else(|| {
        AkitaError::InvalidInput("intermediate fold missing stage-1 proof".to_string())
    })?;
    let NextWitnessStateOutput {
        witness: packed_witness,
        binding,
        hint: committed_hint,
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
        extension_opening_reduction: prepared_fold.extension_opening_reduction,
        opening_payload: prepared_fold.opening_payload.into_compact(),
        fold_grind_nonce,
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
            hint: committed_hint,
            log_basis: next_params.log_basis_inner(),
            sumcheck_challenges,
            opening: w_eval,
            setup_prefix_opening,
        },
    })
}

#[allow(clippy::too_many_arguments)]
pub(in crate::protocol::core) fn prove_stage1<F, E, T>(
    transcript: &mut T,
    rs: &mut RingSwitchOutput<E>,
    plan: &RelationRangeImagePlan,
) -> Result<(AkitaStage1Proof<E>, Vec<E>, E), AkitaError>
where
    F: FieldCore + CanonicalField,
    E: ExtField<F>
        + HasUnreducedOps
        + HasOptimizedFold
        + FromPrimitiveInt
        + AkitaSerialize
        + crate::kernels::sumcheck::SumcheckTableOperations<F>,
    T: Transcript<F>,
{
    let _sumcheck_span = tracing::info_span!("stage1_sumcheck").entered();
    let domain = plan.digit_witness_domain();
    if plan.relation_address_geometry() != rs.relation_address_geometry
        || domain.live_len() != rs.w_evals_compact.len()
        || plan.digit_range_plan().basis() != rs.b
    {
        return Err(AkitaError::InvalidSetup(
            "ring-switch output disagrees with the relation/range-image plan".into(),
        ));
    }
    let digit_range_equality_col_bits = rs
        .tau0
        .len()
        .checked_sub(rs.digit_range_equality_low_variable_count)
        .ok_or_else(|| AkitaError::InvalidSetup("digit-range equality width overflow".into()))?;
    let equality_point = DigitRangeEqualityPoint::from_column_then_ring_challenges(
        &rs.tau0,
        digit_range_equality_col_bits,
        rs.digit_range_equality_low_variable_count,
    )?;
    let stage1_prover = DigitRangeProver::new(
        std::sync::Arc::clone(&rs.w_evals_compact),
        plan.digit_range_plan(),
        domain,
        equality_point,
    )?;
    let (stage1_proof, stage1_point) = stage1_prover.prove::<F, T>(transcript)?;
    let range_image_evaluation = stage1_proof.range_image_evaluation;
    Ok((stage1_proof, stage1_point, range_image_evaluation))
}

#[allow(clippy::too_many_arguments)]
fn prove_stage2<F, E, T>(
    level: usize,
    transcript: &mut T,
    batching_coeff: E,
    rs: RingSwitchOutput<E>,
    stage1_point: &[E],
    range_image_evaluation: E,
    relation_claim: E,
    binary_batching: Option<E>,
    evaluation_trace: PreparedProverEvaluationTrace<E>,
    trace_opening_claim: E,
    plan: RelationRangeImagePlan,
) -> Result<RelationRangeImageProveResult<F, E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: ExtField<F>
        + HasUnreducedOps
        + HasOptimizedFold
        + FromPrimitiveInt
        + AkitaSerialize
        + crate::kernels::sumcheck::SumcheckTableOperations<F>,
    T: Transcript<F>,
{
    let _sumcheck_span = tracing::info_span!("stage2_sumcheck").entered();
    let domain = plan.digit_witness_domain();
    let geometry = rs.relation_address_geometry;
    let live_relation_lane_count = geometry.live_relation_lane_count();
    let relation_lane_variable_count = geometry.relation_lane_variable_count();
    let relation_coefficient_variable_count = geometry.relation_coefficient_variable_count();
    if plan.relation_address_geometry() != geometry
        || domain.live_len() != rs.w_evals_compact.len()
        || plan.digit_range_plan().basis() != rs.b
    {
        return Err(AkitaError::InvalidSetup(
            "ring-switch output disagrees with the relation/range-image plan".into(),
        ));
    }
    let (common_alpha_factor, relation_lane_weights) = rs
        .relation_weight_factorization
        .into_common_alpha_factor_and_relation_lane_weights();
    let expected_factor_len = geometry.relation_coefficient_block_len();
    if common_alpha_factor.len() != expected_factor_len {
        return Err(AkitaError::InvalidSetup(format!(
            "common alpha factor has length {}, expected {expected_factor_len}",
            common_alpha_factor.len(),
        )));
    }
    let additional_relation_terms = rs
        .compression_relation_weights
        .map(|weights| {
            let compression_domain_len = weights.physical_field_len();
            let binary_support =
                NegativeBinarySupport::new(plan.witness_layout(), compression_domain_len)?;
            AdditionalRelationTerms::new(
                rs.w_evals_compact.as_ref(),
                compression_domain_len,
                weights.into_sparse_entries()?,
                binary_support.intervals(),
                stage1_point,
                binary_batching.ok_or(AkitaError::InvalidProof)?,
            )
        })
        .transpose()?;
    let ordinary_relation_claim = relation_claim
        - additional_relation_terms
            .as_ref()
            .map_or_else(E::zero, AdditionalRelationTerms::input_claim);
    let mut stage2_prover = RelationRangeImageProver::new(
        batching_coeff,
        rs.w_evals_compact,
        stage1_point,
        range_image_evaluation,
        plan.digit_range_plan().basis(),
        common_alpha_factor,
        relation_lane_weights,
        live_relation_lane_count,
        relation_lane_variable_count,
        relation_coefficient_variable_count,
        ordinary_relation_claim,
        evaluation_trace,
        trace_opening_claim,
        additional_relation_terms,
    )
    .map_err(|err| {
        AkitaError::InvalidInput(format!(
            "stage-2 prover initialization failed at fold level {level}: {err}"
        ))
    })?;
    let (stage2_sumcheck_proof, sumcheck_challenges, final_claim) = stage2_prover
        .prove::<F, T, _>(transcript, |tr| {
            sample_ext_challenge::<F, E, T>(tr, CHALLENGE_SUMCHECK_ROUND)
        })?;
    if final_claim != stage2_prover.expected_final_claim()? {
        return Err(AkitaError::InvalidInput(
            "stage-2 prover final claim disagrees with its folded oracle".into(),
        ));
    }
    Ok((stage2_sumcheck_proof, sumcheck_challenges, stage2_prover))
}

#[allow(clippy::too_many_arguments)]
pub(in crate::protocol::core) fn prove_stage3<F, E, T>(
    level: usize,
    setup_contribution_mode: SetupContributionMode,
    expanded: &AkitaExpandedSetup<F>,
    prefix_slots: &SetupPrefixProverRegistry<F>,
    lp: &CommittedGroupParams,
    next_level_params: &CommittedGroupParams,
    instance: &RingRelationInstance<F>,
    tau1: &[E],
    alpha: E,
    sumcheck_challenges: &[E],
    relation_address_geometry: akita_types::RelationAddressGeometry,
    transcript: &mut T,
) -> Result<Option<Stage3ProveOutput<E>>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FpExtEncoding<F>
        + FromPrimitiveInt
        + LiftBase<F>
        + AkitaSerialize
        + akita_field::unreduced::HasUnreducedOps
        + akita_field::MulBaseUnreduced<F>
        + crate::kernels::sumcheck::SumcheckTableOperations<F>,
    T: Transcript<F>,
{
    match setup_contribution_mode {
        SetupContributionMode::Recursive => {
            let _stage3_span = tracing::info_span!(
                "stage3_sumcheck",
                level,
                stage2_rounds = sumcheck_challenges.len(),
                d_a = lp.d_a(),
            )
            .entered();
            let mut stage3_prover = {
                let _prepare_span = tracing::info_span!("stage3_prover_prepare").entered();
                AkitaStage3Prover::new::<T>(
                    expanded,
                    prefix_slots,
                    lp,
                    next_level_params,
                    instance,
                    tau1,
                    alpha,
                    sumcheck_challenges,
                    relation_address_geometry,
                    transcript,
                )?
            };
            let output = stage3_prover.prove::<T, _>(transcript, |tr| {
                sample_ext_challenge::<F, E, T>(tr, CHALLENGE_SUMCHECK_ROUND)
            })?;
            Ok(Some(Stage3ProveOutput {
                proof: SetupSumcheckProof {
                    claim: output.setup_product_claim,
                    setup_prefix_eval: output.setup_prefix_eval,
                    sumcheck: output.sumcheck,
                },
                setup_prefix_point: output.setup_prefix_point,
            }))
        }
        SetupContributionMode::Direct => Ok(None),
    }
}
