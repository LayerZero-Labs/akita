use super::*;
use crate::compute::{
    tensor_root_projection, CommitmentComputeBackend, ComputeBackendSetup, DigitRowsComputeBackend,
    ProverComputeStack, RootOpeningSource, RootPolyMeta, RuntimeOpeningProveBackendFor,
    RuntimeRingSwitchProveBackend, RuntimeRootProvePoly, RuntimeTensorBackendFor,
};
use crate::protocol::sumcheck::relation_range_image::PreparedProverEvaluationTrace;
use crate::protocol::sumcheck::DigitRangeProver;
use crate::RootTensorProjectionPoly;
use akita_field::unreduced::ReduceTo;
use akita_field::AdditiveGroup;

use akita_types::{
    dispatch_for_field, DigitRangeEqualityPoint, DigitRangePlan, FlatBooleanDomain,
    OpeningClaimsLayout, RelationRangeImagePlan,
};

pub(in crate::protocol::core) struct PreparedFold<F: FieldCore, E: FieldCore> {
    pub(in crate::protocol::core) commitment: RingVec<F>,
    pub(in crate::protocol::core) instance: RingRelationInstance<F>,
    pub(in crate::protocol::core) witness: RingRelationWitness<F>,
    pub(in crate::protocol::core) extension_opening_reduction:
        Option<ExtensionOpeningReductionProof<E>>,
    pub(in crate::protocol::core) evaluation_trace_claim: E,
    pub(in crate::protocol::core) evaluation_trace_points: Vec<PreparedOpeningPoint<F, E>>,
    pub(in crate::protocol::core) evaluation_trace_claim_coefficients: Vec<E>,
    pub(in crate::protocol::core) evaluation_trace_basis: BasisMode,
    pub(in crate::protocol::core) row_coefficients: Option<Vec<E>>,
}

#[allow(clippy::too_many_arguments)]
pub(in crate::protocol::core) fn prepare_fold_inner<'a, F, E, T, P, V, C, O, TS, R>(
    stack: &ProverComputeStack<'_, F, C, O, TS, R>,
    needs_extension_reduction: bool,
    block_claims: ProverOpeningData<'a, E, P, F>,
    eor_polys: &[&P],
    eor_opening_batch: &OpeningClaims<'_, E>,
    pad_base_evals: bool,
    transcript: &mut T,
    non_eor_protocol_point: Vec<E>,
    validate_non_eor: V,
    level_params: &CommittedGroupParams,
    alpha_bits: usize,
    basis: BasisMode,
) -> Result<PreparedFold<F, E>, AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide,
    E: FpExtEncoding<F>
        + ExtField<F>
        + HasUnreducedOps
        + HasOptimizedFold
        + FromPrimitiveInt
        + MulBaseUnreduced<F>
        + AkitaSerialize,
    T: Transcript<F> + ProverTranscriptGrind<F>,
    F: RandomSampling + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F> + AdditiveGroup,
    P: RuntimeRootProvePoly<F>,
    V: FnOnce() -> Result<(), AkitaError>,
    TS: RuntimeTensorBackendFor<F, P, E>,
    C: ComputeBackendSetup<F>,
    O: DigitRowsComputeBackend<F>
        + RuntimeOpeningProveBackendFor<F, P>
        + RuntimeOpeningProveBackendFor<F, RootTensorProjectionPoly<F>>,
    R: DigitRowsComputeBackend<F>,
{
    let opening_batch = block_claims
        .opening_claims()
        .layout()
        .map_err(|err| AkitaError::InvalidInput(format!("opening batch layout failed: {err:?}")))?;
    let fold_polys = block_claims.flat_polys();
    let tensor = stack.tensor();
    // A-role fold dimension: the EOR sumcheck and tensor projection operate on
    // the claim polynomials at this level's fold ring.
    let ring_d = level_params.role_dims().d_a();
    let (protocol_point, row_coefficients, reduction) = if needs_extension_reduction {
        let proved = dispatch_for_field!(
            ProtocolDispatchSlot::Role(RingRole::Inner),
            F,
            ring_d,
            |D| {
                prove_extension_opening_reduction::<F, E, T, P, TS, D>(
                    tensor.backend(),
                    Some(tensor.prepared()),
                    eor_polys,
                    eor_opening_batch,
                    pad_base_evals,
                    transcript,
                    if pad_base_evals { "recursive" } else { "root" },
                )
            }
        )
        .map_err(|err| {
            AkitaError::InvalidInput(format!("root opening preparation failed: {err:?}"))
        })?;
        (
            proved.protocol_point,
            Some(proved.row_coefficients),
            Some(proved.reduction),
        )
    } else {
        validate_non_eor()?;
        let row_coefficients = if pad_base_evals {
            Some(vec![E::one(); opening_batch.num_total_polynomials()])
        } else {
            None
        };
        (non_eor_protocol_point, row_coefficients, None)
    };

    if needs_extension_reduction {
        if pad_base_evals {
            finish_prepared_fold::<F, E, T, P, C, O, TS, R>(FinishFoldArgs {
                stack,
                block_claims,
                protocol_point: &protocol_point,
                reduction,
                row_coefficients,
                trace_opening_batch: &opening_batch,
                level_params,
                alpha_bits,
                basis,
                pad_base_evals,
                transcript,
            })
            .map_err(|err| {
                AkitaError::InvalidInput(format!("finish prepared fold failed: {err:?}"))
            })
        } else {
            let transformed: Vec<RootTensorProjectionPoly<F>> = {
                let _span =
                    tracing::info_span!("extension_transform_polys", num_claims = fold_polys.len())
                        .entered();
                dispatch_for_field!(
                    ProtocolDispatchSlot::Role(RingRole::Inner),
                    F,
                    ring_d,
                    |D| {
                        cfg_iter!(fold_polys)
                            .map(|poly| {
                                tensor_root_projection::<F, P, E, TS, D>(
                                    tensor.backend(),
                                    Some(tensor.prepared()),
                                    *poly,
                                )
                            })
                            .collect::<Result<Vec<_>, _>>()
                    }
                )?
            };
            let fold_refs = transformed.iter().collect::<Vec<_>>();
            let transformed_block_claims = block_claims.regroup_polynomial_refs(&fold_refs)?;
            finish_prepared_fold::<F, E, T, RootTensorProjectionPoly<F>, C, O, TS, R>(
                FinishFoldArgs {
                    stack,
                    block_claims: transformed_block_claims,
                    protocol_point: &protocol_point,
                    reduction,
                    row_coefficients,
                    trace_opening_batch: &opening_batch,
                    level_params,
                    alpha_bits,
                    basis,
                    pad_base_evals,
                    transcript,
                },
            )
        }
    } else {
        finish_prepared_fold::<F, E, T, P, C, O, TS, R>(FinishFoldArgs {
            stack,
            block_claims,
            protocol_point: &protocol_point,
            reduction,
            row_coefficients,
            trace_opening_batch: &opening_batch,
            level_params,
            alpha_bits,
            basis,
            pad_base_evals,
            transcript,
        })
    }
}

/// Borrowed/owned argument bundle for [`finish_prepared_fold`].
struct FinishFoldArgs<'a, 'p, F, E, T, Q, C, O, TS, R>
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
    protocol_point: &'a [E],
    reduction: Option<ExtensionOpeningReduction<E>>,
    row_coefficients: Option<Vec<E>>,
    trace_opening_batch: &'a OpeningClaimsLayout,
    level_params: &'a CommittedGroupParams,
    alpha_bits: usize,
    basis: BasisMode,
    pad_base_evals: bool,
    transcript: &'p mut T,
}

/// Evaluate folded claims, derive the trace target, and build the ring-relation
/// instance/witness for one borrowed source-view set `Q: RootOpeningSource`.
#[allow(clippy::needless_lifetimes)]
fn finish_prepared_fold<'a, 'p, F, E, T, Q, C, O, TS, R>(
    args: FinishFoldArgs<'a, 'p, F, E, T, Q, C, O, TS, R>,
) -> Result<PreparedFold<F, E>, AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide + RandomSampling + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F> + AdditiveGroup,
    E: FpExtEncoding<F>
        + ExtField<F>
        + HasUnreducedOps
        + HasOptimizedFold
        + FromPrimitiveInt
        + MulBaseUnreduced<F>
        + AkitaSerialize,
    T: Transcript<F> + ProverTranscriptGrind<F>,
    Q: RootOpeningSource<F, 32>
        + RootOpeningSource<F, 64>
        + RootOpeningSource<F, 128>
        + RootOpeningSource<F, 256>
        + RootPolyMeta<F>,
    O: DigitRowsComputeBackend<F> + RuntimeOpeningProveBackendFor<F, Q>,
    R: DigitRowsComputeBackend<F>,
    C: ComputeBackendSetup<F>,
    TS: ComputeBackendSetup<F>,
{
    let FinishFoldArgs {
        stack,
        block_claims,
        protocol_point,
        reduction,
        row_coefficients,
        trace_opening_batch,
        level_params,
        alpha_bits,
        basis,
        pad_base_evals,
        transcript,
    } = args;
    let opening = stack.opening();
    // Extracted level numbers for the A-role claims-evaluation operation; the
    // kernels below must not read schedule types.
    let ring_d = level_params.role_dims().d_a();
    // A-role operation: prepare the typed opening point, fold-evaluate every
    // claim polynomial at it, and derive the trace target. Typed outputs are
    // converted to D-free carriers (`PreparedOpeningPoint`, `RingVec`) inside
    // the arm.
    let opening_batch = block_claims
        .opening_claims()
        .layout()
        .map_err(|err| AkitaError::InvalidInput(format!("opening batch layout failed: {err:?}")))?;
    let (prepared_points, e_folded_by_claim, trace_claim, row_coefficients, row_coefficient_rings) =
        dispatch_for_field!(
            ProtocolDispatchSlot::Role(RingRole::Inner),
            F,
            ring_d,
            |D| {
                let mut prepared_points = Vec::with_capacity(opening_batch.num_groups());
                let mut folded_rings = Vec::with_capacity(opening_batch.num_total_polynomials());
                let mut e_folded_by_claim =
                    Vec::with_capacity(opening_batch.num_total_polynomials());
                for group_index in 0..opening_batch.num_groups() {
                    let group_lp = level_params
                        .group_params(&opening_batch, group_index)
                        .map_err(|err| {
                            AkitaError::InvalidInput(format!(
                                "root group params {group_index} failed: {err:?}"
                            ))
                        })?;
                    let target_len = alpha_bits
                        .checked_add(group_lp.position_index_bits())
                        .and_then(|n| n.checked_add(group_lp.block_index_bits()))
                        .ok_or_else(|| {
                            AkitaError::InvalidSetup(
                                "group opening point length overflow".to_string(),
                            )
                        })?;
                    let point_vars = block_claims
                        .opening_claims()
                        .group_point_vars(group_index)?;
                    if point_vars.num_vars() != target_len {
                        return Err(AkitaError::InvalidPointDimension {
                            expected: target_len,
                            actual: point_vars.num_vars(),
                        });
                    }
                    let group_protocol_point = point_vars
                        .indices()
                        .iter()
                        .map(|&idx| {
                            protocol_point
                                .get(idx)
                                .copied()
                                .ok_or(AkitaError::InvalidProof)
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    let group_polys = block_claims.group_polys(group_index).map_err(|err| {
                        AkitaError::InvalidInput(format!(
                            "root group polynomials {group_index} failed: {err:?}"
                        ))
                    })?;
                    let (prepared_point, (group_folded_rings, group_e_folded_by_claim)) =
                        prepare_and_evaluate_opening_group::<F, E, T, Q, O, D>(
                            opening.backend(),
                            Some(opening.prepared()),
                            group_polys,
                            &group_protocol_point,
                            basis,
                            group_lp.num_positions_per_block(),
                            group_lp.num_live_blocks(),
                            alpha_bits,
                            transcript,
                        )
                        .map_err(|err| {
                            AkitaError::InvalidInput(format!(
                                "evaluate claims group {group_index} failed: {err:?}"
                            ))
                        })?;
                    e_folded_by_claim.extend(
                        group_e_folded_by_claim
                            .iter()
                            // The opening kernel emits A-role rings. The ring
                            // relation reinterprets the unchanged coefficients
                            // at its D-role dimension.
                            .map(|rows| RingVec::from_ring_elems(rows).into_compact()),
                    );
                    folded_rings.extend(group_folded_rings);
                    prepared_points.push(prepared_point);
                }

                let (trace_claim, row_coefficients) = prepare_evaluation_trace_claim::<F, E, T, D>(
                    &reduction,
                    &folded_rings,
                    &prepared_points,
                    protocol_point,
                    alpha_bits,
                    basis,
                    trace_opening_batch,
                    row_coefficients,
                    transcript,
                )
                .map_err(|err| {
                    AkitaError::InvalidInput(format!(
                        "prepare evaluation-trace claim failed: {err:?}"
                    ))
                })?;
                let row_coefficient_rings = row_coefficient_rings::<F, E, D>(&row_coefficients)
                    .map_err(|err| {
                        AkitaError::InvalidInput(format!("row coefficient rings failed: {err:?}"))
                    })?;
                Ok::<_, AkitaError>((
                    prepared_points,
                    e_folded_by_claim,
                    trace_claim,
                    row_coefficients,
                    RingVec::from_ring_elems(&row_coefficient_rings),
                ))
            }
        )
        .map_err(|err| {
            AkitaError::InvalidInput(format!("root opening preparation failed: {err:?}"))
        })?;
    let commitment = block_claims.fold_commitment(level_params).map_err(|err| {
        AkitaError::InvalidInput(format!("fold commitment preparation failed: {err:?}"))
    })?;
    let (instance, witness) = RingRelationProver::new(
        opening,
        stack.ring_switch(),
        prepared_points
            .iter()
            .map(|prepared| prepared.ring_opening_point.clone())
            .collect::<Vec<_>>(),
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
        commitment,
        instance,
        witness,
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
        + AkitaSerialize,
    T: Transcript<F> + ProverTranscriptGrind<F>,
    C: CommitmentComputeBackend<F> + ComputeBackendSetup<F> + 'stack,
    O: ComputeBackendSetup<F>,
    TS: ComputeBackendSetup<F>,
    R: RuntimeRingSwitchProveBackend<F> + ComputeBackendSetup<F> + 'stack,
    <C as ComputeBackendSetup<F>>::PreparedSetup: 'stack,
    <R as ComputeBackendSetup<F>>::PreparedSetup: 'stack,
    Cfg: CommitmentConfig<Field = F, ExtField = E>,
{
    let ring_d = prepared_fold.instance.role_dims().d_a();
    let fold_grind_nonce = prepared_fold.witness.fold_grind_nonce;
    let logical_w = ring_switch_build_w::<F, R>(
        &prepared_fold.instance,
        prepared_fold.witness,
        stack.ring_switch(),
        lp,
    )
    .map_err(|err| {
        AkitaError::InvalidInput(format!("ring-switch witness build failed: {err:?}"))
    })?;
    let next_params = next_params.ok_or_else(|| {
        AkitaError::InvalidSetup("non-terminal fold is missing successor params".into())
    })?;
    if Some(logical_w.len()) != expected_output_witness_len {
        return Err(AkitaError::InvalidSetup(format!(
            "scheduled fold level {level} produced unexpected next-w length: expected={expected_output_witness_len:?}, actual={}",
            logical_w.len()
        )));
    }
    let _span = tracing::info_span!("commit_w_level", level).entered();
    let next_commitment = match next_params {
        FoldSuccessorParams::Recursive(params) => {
            if next_witness_binding != Some(akita_types::NextWitnessBindingPolicy::OuterCommitment)
            {
                return Err(AkitaError::InvalidSetup(
                    "recursive successor requires outer-commitment binding".into(),
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
        NextWitnessState::OuterCommitment(commitment) => {
            transcript.append_serde(ABSORB_NEXT_LEVEL_WITNESS_BINDING, commitment);
        }
        NextWitnessState::TerminalInnerState { t_state } => {
            let bytes = akita_types::raw_field_segment_bytes(t_state)?;
            transcript.absorb_and_record_bytes(ABSORB_NEXT_LEVEL_WITNESS_BINDING, &bytes);
        }
    }
    let next_opening_ring_dim = next_params.inner_ring_dimension();
    if !logical_w.len().is_multiple_of(next_opening_ring_dim) {
        return Err(AkitaError::InvalidProof);
    }
    let next_opening_source_len = logical_w.len() / next_opening_ring_dim;
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

    let digit_witness_num_vars = rs
        .col_bits
        .checked_add(rs.ring_bits)
        .ok_or_else(|| AkitaError::InvalidInput("digit witness domain width overflow".into()))?;
    let relation_range_image_plan = RelationRangeImagePlan::new(
        FlatBooleanDomain::new(rs.w_evals_compact.len(), digit_witness_num_vars)?,
        DigitRangePlan::new(rs.b)?,
        prepared_fold.instance.segment_layout(lp, None)?,
        prepared_fold.instance.opening_batch(),
        prepared_fold.instance.role_dims(),
        next_opening_ring_dim,
    )?;

    let relation_rhs_layout = relation_rhs_layout_for(lp, prepared_fold.instance.opening_batch())?;
    let relation_claim = relation_claim_from_layout_extension::<F, E>(
        prepared_fold.instance.role_dims(),
        &relation_rhs_layout,
        &rs.tau1,
        rs.alpha,
        prepared_fold.instance.v(),
        &prepared_fold.commitment,
    )?;
    let (stage1_proof, stage1_point, range_image_evaluation) =
        prove_stage1::<F, E, T>(transcript, &mut rs, &relation_range_image_plan)?;
    transcript.append_serde(
        ABSORB_RANGE_IMAGE_EVALUATION,
        &stage1_proof.range_image_evaluation,
    );
    let stage1_proof = Some(stage1_proof);
    let batching_coeff: E = sample_ext_challenge::<F, E, T>(transcript, CHALLENGE_SUMCHECK_BATCH);
    // EvaluationTrace is the last padded relation row: weight openings by
    // `eq(tau1, EvaluationTrace_row_index)`.
    let opening_batch = prepared_fold.instance.opening_batch();
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
        source_ring_dimension = ring_d,
        coeff_count = 1usize << rs.ring_bits,
    )
    .entered();
    let evaluation_trace = dispatch_for_field!(
        ProtocolDispatchSlot::Role(RingRole::Inner),
        F,
        ring_d,
        |D| {
            let semantic_trace =
                build_evaluation_trace_weights::<F, E, D>(EvaluationTraceInputs {
                    digit_witness_domain: relation_range_image_plan.digit_witness_domain(),
                    witness_layout: relation_range_image_plan.witness_layout(),
                    role_dims: relation_range_image_plan.role_dims(),
                    level_params: lp,
                    opening_batch,
                    prepared_points: evaluation_trace_points,
                    claim_coefficients: &prepared_fold.evaluation_trace_claim_coefficients,
                    basis: prepared_fold.evaluation_trace_basis,
                })?;
            PreparedProverEvaluationTrace::new(
                &semantic_trace,
                1usize << rs.ring_bits,
                evaluation_trace_weight,
            )
        }
    )?;
    drop(trace_preparation_span);
    let ring_bits = rs.ring_bits;
    let col_bits = rs.col_bits;
    let live_x_cols = rs.live_x_cols;
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
            w_eval,
            logical_w.as_i8_digits(),
            live_x_cols,
            col_bits,
            ring_bits,
            transcript,
        )?,
        None => None,
    };
    let (stage3_sumcheck_proof, next_opening_point, next_opening, setup_prefix_opening) =
        if let Some(stage3) = stage3_sumcheck_proof {
            (
                Some(stage3.proof),
                stage3.next_w_point,
                stage3.next_w_eval,
                Some((stage3.setup_prefix_point, stage3.setup_prefix_eval)),
            )
        } else {
            (None, sumcheck_challenges, w_eval, None)
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
        NextWitnessState::OuterCommitment(commitment) => (
            akita_types::NextWitnessBinding::OuterCommitment(commitment.clone().into_compact()),
            NextWitnessState::OuterCommitment(commitment),
        ),
        NextWitnessState::TerminalInnerState { t_state } => (
            akita_types::NextWitnessBinding::TerminalInnerState,
            NextWitnessState::TerminalInnerState { t_state },
        ),
    };
    let level_proof = FoldLevelProof {
        extension_opening_reduction: prepared_fold.extension_opening_reduction,
        v: prepared_fold.instance.v().clone().into_compact(),
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
            sumcheck_challenges: next_opening_point,
            opening: next_opening,
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
    E: ExtField<F> + HasUnreducedOps + HasOptimizedFold + FromPrimitiveInt + AkitaSerialize,
    T: Transcript<F>,
{
    let _sumcheck_span = tracing::info_span!("stage1_sumcheck").entered();
    let domain = plan.digit_witness_domain();
    if domain.live_len() != rs.w_evals_compact.len() || plan.digit_range_plan().basis() != rs.b {
        return Err(AkitaError::InvalidSetup(
            "ring-switch output disagrees with the relation/range-image plan".into(),
        ));
    }
    let derived_live_x_cols = domain.live_block_count(rs.ring_bits)?;
    if derived_live_x_cols != rs.live_x_cols {
        return Err(AkitaError::InvalidSize {
            expected: derived_live_x_cols,
            actual: rs.live_x_cols,
        });
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
    evaluation_trace: PreparedProverEvaluationTrace<E>,
    trace_opening_claim: E,
    plan: RelationRangeImagePlan,
) -> Result<RelationRangeImageProveResult<E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: ExtField<F> + HasUnreducedOps + HasOptimizedFold + FromPrimitiveInt + AkitaSerialize,
    T: Transcript<F>,
{
    let _sumcheck_span = tracing::info_span!("stage2_sumcheck").entered();
    let domain = plan.digit_witness_domain();
    let derived_live_x_cols = domain.live_block_count(rs.ring_bits)?;
    let derived_col_bits = domain
        .num_vars()
        .checked_sub(rs.ring_bits)
        .ok_or_else(|| AkitaError::InvalidSetup("stage-2 ring width exceeds domain".into()))?;
    if domain.live_len() != rs.w_evals_compact.len()
        || plan.digit_range_plan().basis() != rs.b
        || derived_live_x_cols != rs.live_x_cols
        || derived_col_bits != rs.col_bits
    {
        return Err(AkitaError::InvalidSetup(
            "ring-switch output disagrees with the relation/range-image plan".into(),
        ));
    }
    let (common_alpha_factor, relation_lane_weights) = rs
        .relation_weight_factorization
        .into_common_alpha_factor_and_relation_lane_weights();
    let expected_factor_len = 1usize << rs.ring_bits;
    if common_alpha_factor.len() != expected_factor_len {
        return Err(AkitaError::InvalidSetup(format!(
            "common alpha factor has length {}, expected {expected_factor_len}",
            common_alpha_factor.len(),
        )));
    }
    let mut stage2_prover = RelationRangeImageProver::new(
        batching_coeff,
        rs.w_evals_compact,
        stage1_point,
        range_image_evaluation,
        plan.digit_range_plan().basis(),
        common_alpha_factor,
        relation_lane_weights,
        derived_live_x_cols,
        derived_col_bits,
        rs.ring_bits,
        relation_claim,
        evaluation_trace,
        trace_opening_claim,
    )
    .map_err(|err| {
        AkitaError::InvalidInput(format!(
            "stage-2 prover initialization failed at fold level {level}: {err}"
        ))
    })?;
    let (stage2_sumcheck_proof, sumcheck_challenges, _) = stage2_prover
        .prove::<F, T, _>(transcript, |tr| {
            sample_ext_challenge::<F, E, T>(tr, CHALLENGE_SUMCHECK_ROUND)
        })?;
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
    stage2_next_w_eval: E,
    logical_w: &[i8],
    live_x_cols: usize,
    col_bits: usize,
    ring_bits: usize,
    transcript: &mut T,
) -> Result<Option<Stage3ProveOutput<E>>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FpExtEncoding<F> + FromPrimitiveInt + LiftBase<F> + AkitaSerialize,
    T: Transcript<F>,
{
    match setup_contribution_mode {
        SetupContributionMode::Recursive => {
            let eta = sample_ext_challenge::<F, E, T>(transcript, CHALLENGE_SUMCHECK_BATCH);
            let mut stage3_prover = AkitaStage3Prover::new::<F, T>(
                expanded,
                prefix_slots,
                lp,
                next_level_params,
                instance,
                tau1,
                alpha,
                sumcheck_challenges,
                stage2_next_w_eval,
                logical_w,
                live_x_cols,
                col_bits,
                ring_bits,
                level,
                eta,
                transcript,
            )?;
            let output = stage3_prover.prove::<F, T, _>(transcript, |tr| {
                sample_ext_challenge::<F, E, T>(tr, CHALLENGE_SUMCHECK_ROUND)
            })?;
            transcript.append_serde(ABSORB_STAGE3_NEXT_W_EVAL, &output.next_w_eval);
            Ok(Some(Stage3ProveOutput {
                proof: SetupSumcheckProof {
                    claim: output.setup_product_claim,
                    setup_prefix_eval: output.setup_prefix_eval,
                    next_w_eval: output.next_w_eval,
                    sumcheck: output.sumcheck,
                },
                next_w_point: output.next_w_point,
                setup_prefix_point: output.setup_prefix_point,
                setup_prefix_eval: output.setup_prefix_eval,
                next_w_eval: output.next_w_eval,
            }))
        }
        SetupContributionMode::Direct => Ok(None),
    }
}
