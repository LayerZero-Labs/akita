use super::*;
use crate::compute::{
    tensor_root_projection, CommitmentComputeBackend, ComputeBackendSetup, DigitRowsComputeBackend,
    ProverComputeStack, RootOpeningSource, RootPolyMeta, RuntimeOpeningProveBackendFor,
    RuntimeRingSwitchProveBackend, RuntimeRootProvePoly, RuntimeTensorBackendFor,
};
use crate::RootTensorProjectionPoly;
use akita_field::unreduced::ReduceTo;
use akita_field::AdditiveGroup;

use crate::protocol::ring_switch::RingSwitchTerminalArtifacts;
use akita_types::build_segment_typed_witness_from_groups;
use akita_types::dispatch_for_field;
use akita_types::CleartextWitnessShape;
use akita_types::OpeningClaimsLayout;
use akita_types::SegmentTypedWitnessGroupParts;

fn trace_layout_for_instance<F: FieldCore + CanonicalField>(
    lp: &LevelParams,
    instance: &RingRelationInstance<F>,
    col_bits: usize,
    ring_bits: usize,
    num_trace_blocks: usize,
) -> Result<
    (
        akita_types::WitnessChunkLayout,
        akita_types::TraceWeightLayout,
    ),
    AkitaError,
> {
    let witness_layout = instance.segment_layout(lp, None)?;
    let segment = *witness_layout
        .chunks
        .first()
        .ok_or_else(|| AkitaError::InvalidSetup("empty witness layout".to_string()))?;
    let layout = trace_weight_layout_from_segment(
        lp,
        &witness_layout,
        col_bits,
        ring_bits,
        num_trace_blocks,
    )?;
    Ok((segment, layout))
}

pub(in crate::protocol::core) struct PreparedFold<F: FieldCore, E: FieldCore> {
    pub(in crate::protocol::core) commitment: RingVec<F>,
    pub(in crate::protocol::core) instance: RingRelationInstance<F>,
    pub(in crate::protocol::core) witness: RingRelationWitness<F>,
    pub(in crate::protocol::core) extension_opening_reduction:
        Option<ExtensionOpeningReductionProof<E>>,
    pub(in crate::protocol::core) trace_eval_target: E,
    pub(in crate::protocol::core) trace_prepared_points: Option<Vec<PreparedOpeningPoint<F, E>>>,
    pub(in crate::protocol::core) trace_claim_scales: Option<Vec<E>>,
    pub(in crate::protocol::core) trace_scale: E,
    pub(in crate::protocol::core) row_coefficients: Option<Vec<E>>,
}

#[allow(clippy::too_many_arguments)]
pub(in crate::protocol::core) fn prepare_fold_inner<'a, F, E, T, P, V, C, O, TS, R>(
    stack: &ProverComputeStack<'_, F, C, O, TS, R>,
    needs_extension_reduction: bool,
    fold_claims: ProverOpeningData<'a, E, P, F>,
    eor_polys: &[&P],
    eor_opening_batch: &OpeningClaims<'_, E>,
    pad_base_evals: bool,
    transcript: &mut T,
    non_eor_protocol_point: Vec<E>,
    validate_non_eor: V,
    level_params: &LevelParams,
    alpha_bits: usize,
    basis: BasisMode,
    block_order: BlockOrder,
    relation_matrix_row_layout: RelationMatrixRowLayout,
    terminal_tail_t_vectors: Option<usize>,
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
    let opening_batch = fold_claims.opening_claims().layout()?;
    let fold_polys = fold_claims.flat_polys();
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
        )?;
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
                fold_claims,
                protocol_point: &protocol_point,
                reduction,
                row_coefficients,
                trace_opening_batch: &opening_batch,
                level_params,
                alpha_bits,
                basis,
                block_order,
                pad_base_evals,
                transcript,
                relation_matrix_row_layout,
                terminal_tail_t_vectors,
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
            let transformed_fold_claims = fold_claims.regroup_polynomial_refs(&fold_refs)?;
            finish_prepared_fold::<F, E, T, RootTensorProjectionPoly<F>, C, O, TS, R>(
                FinishFoldArgs {
                    stack,
                    fold_claims: transformed_fold_claims,
                    protocol_point: &protocol_point,
                    reduction,
                    row_coefficients,
                    trace_opening_batch: &opening_batch,
                    level_params,
                    alpha_bits,
                    basis,
                    block_order,
                    pad_base_evals,
                    transcript,
                    relation_matrix_row_layout,
                    terminal_tail_t_vectors,
                },
            )
        }
    } else {
        finish_prepared_fold::<F, E, T, P, C, O, TS, R>(FinishFoldArgs {
            stack,
            fold_claims,
            protocol_point: &protocol_point,
            reduction,
            row_coefficients,
            trace_opening_batch: &opening_batch,
            level_params,
            alpha_bits,
            basis,
            block_order,
            pad_base_evals,
            transcript,
            relation_matrix_row_layout,
            terminal_tail_t_vectors,
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
    fold_claims: ProverOpeningData<'a, E, Q, F>,
    protocol_point: &'a [E],
    reduction: Option<ExtensionOpeningReduction<E>>,
    row_coefficients: Option<Vec<E>>,
    trace_opening_batch: &'a OpeningClaimsLayout,
    level_params: &'a LevelParams,
    alpha_bits: usize,
    basis: BasisMode,
    block_order: BlockOrder,
    pad_base_evals: bool,
    transcript: &'p mut T,
    relation_matrix_row_layout: RelationMatrixRowLayout,
    terminal_tail_t_vectors: Option<usize>,
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
        fold_claims,
        protocol_point,
        reduction,
        row_coefficients,
        trace_opening_batch,
        level_params,
        alpha_bits,
        basis,
        block_order,
        pad_base_evals,
        transcript,
        relation_matrix_row_layout,
        terminal_tail_t_vectors,
    } = args;
    let opening = stack.opening();
    // Extracted level numbers for the A-role claims-evaluation operation; the
    // kernels below must not read schedule types.
    let ring_d = level_params.role_dims().d_a();
    // A-role operation: prepare the typed opening point, fold-evaluate every
    // claim polynomial at it, and derive the trace target. Typed outputs are
    // converted to D-free carriers (`PreparedOpeningPoint`, `RingVec`) inside
    // the arm.
    let opening_batch = fold_claims.opening_claims().layout()?;
    let (prepared_points, e_folded_by_claim, trace_target, row_coefficients, row_coefficient_rings) =
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
                    let group_lp = level_params.root_group_params(&opening_batch, group_index)?;
                    let target_len = alpha_bits
                        .checked_add(group_lp.m_vars())
                        .and_then(|n| n.checked_add(group_lp.r_vars()))
                        .ok_or_else(|| {
                            AkitaError::InvalidSetup(
                                "group opening point length overflow".to_string(),
                            )
                        })?;
                    let point_vars = fold_claims.opening_claims().group_point_vars(group_index)?;
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
                    let prepared_point = prepare_opening_point::<F, E, D>(
                        &group_protocol_point,
                        basis,
                        group_lp.m_vars(),
                        group_lp.r_vars(),
                        alpha_bits,
                        block_order,
                    )?;
                    let group_polys = fold_claims.group_polys(group_index)?;
                    let (group_folded_rings, group_e_folded_by_claim) =
                        evaluate_claims_at_prepared_point(
                            opening.backend(),
                            Some(opening.prepared()),
                            group_polys,
                            &prepared_point,
                            group_lp.block_len(),
                        )?;
                    for pt in &prepared_point.padded_point {
                        append_ext_field::<F, E, T>(transcript, ABSORB_EVALUATION_CLAIMS, pt);
                    }
                    e_folded_by_claim.extend(
                        group_e_folded_by_claim
                            .iter()
                            .map(|rows| RingVec::from_ring_elems(rows)),
                    );
                    folded_rings.extend(group_folded_rings);
                    prepared_points.push(prepared_point);
                }

                let (trace_target, row_coefficients) = compute_trace_target::<F, E, T, D>(
                    &reduction,
                    &folded_rings,
                    &prepared_points,
                    protocol_point,
                    alpha_bits,
                    basis,
                    trace_opening_batch,
                    row_coefficients,
                    transcript,
                )?;
                let row_coefficient_rings = row_coefficient_rings::<F, E, D>(&row_coefficients)?;
                Ok::<_, AkitaError>((
                    prepared_points,
                    e_folded_by_claim,
                    trace_target,
                    row_coefficients,
                    RingVec::from_ring_elems(&row_coefficient_rings),
                ))
            }
        )?;
    let commitment = fold_claims.fold_commitment(level_params)?;
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
        fold_claims,
        e_folded_by_claim,
        level_params.clone(),
        transcript,
        row_coefficient_rings,
        relation_matrix_row_layout,
        terminal_tail_t_vectors,
    )?;
    let extension_opening_reduction = reduction.map(|reduction| reduction.proof);
    // §6 invariant (#239 HIGH) — suffix `PreparedFold` trace-table layout vs
    // `pad_base_evals`. `row_coefficients` and `trace_claim_scales` MUST be
    // cleared together on the `pad_base_evals` (recursive-suffix) path and
    // written together otherwise; `prove_fold` selects the root vs recursive
    // `build_*_stage2_trace_table` branch on `row_coefficients.is_some()`, so a
    // split here would silently scale the trace table incorrectly. Preserve the exact
    // branch wiring and assert the two stay coupled to `pad_base_evals`.
    let clear_recursive_trace = pad_base_evals && !level_params.has_precommitted_groups();
    let row_coefficients = if clear_recursive_trace {
        None
    } else {
        Some(row_coefficients)
    };
    let trace_claim_scales = if clear_recursive_trace {
        None
    } else {
        trace_target.trace_claim_scales
    };
    debug_assert_eq!(
        clear_recursive_trace,
        row_coefficients.is_none(),
        "suffix trace layout: row_coefficients must be cleared iff pad_base_evals"
    );
    debug_assert!(
        !clear_recursive_trace || trace_claim_scales.is_none(),
        "suffix trace layout: trace_claim_scales must be cleared when pad_base_evals"
    );
    Ok(PreparedFold {
        commitment,
        instance,
        witness,
        extension_opening_reduction,
        trace_eval_target: trace_target.trace_eval_target,
        trace_scale: trace_target.trace_scale,
        trace_prepared_points: Some(prepared_points),
        trace_claim_scales,
        row_coefficients,
    })
}

pub(in crate::protocol::core) type TerminalFoldResult<F, E> = TerminalLevelProof<F, E>;

pub(in crate::protocol::core) enum FoldProveOutput<F: FieldCore, E: FieldCore> {
    Intermediate(Box<ProveLevelOutput<F, E>>),
    Terminal(Box<TerminalFoldResult<F, E>>),
}

impl<F: FieldCore, E: FieldCore> FoldProveOutput<F, E> {
    pub(in crate::protocol::core) fn get_intermediate(
        self,
    ) -> Result<ProveLevelOutput<F, E>, AkitaError> {
        match self {
            Self::Intermediate(out) => Ok(*out),
            Self::Terminal(_) => Err(AkitaError::InvalidInput(
                "intermediate fold unexpectedly returned terminal proof".to_string(),
            )),
        }
    }

    pub(in crate::protocol::core) fn get_terminal(
        self,
    ) -> Result<TerminalFoldResult<F, E>, AkitaError> {
        match self {
            Self::Terminal(terminal) => Ok(*terminal),
            Self::Intermediate(_) => Err(AkitaError::InvalidInput(
                "terminal fold unexpectedly returned intermediate proof".to_string(),
            )),
        }
    }
}
type BoundNextWitness<F> = (
    Option<NextWitnessCommitment<F>>,
    Option<CleartextWitnessProof<F>>,
);
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
    scheduled: &ExecutionSchedule,
    prepared_fold: PreparedFold<F, E>,
    _setup_contribution_mode: SetupContributionMode,
    is_terminal_fold: bool,
    terminal_direct_witness_shape: Option<&CleartextWitnessShape>,
) -> Result<FoldProveOutput<F, E>, AkitaError>
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
    let lp = &scheduled.params;
    let ring_d = prepared_fold.instance.role_dims().d_a();
    let fold_grind_nonce = prepared_fold.witness.fold_grind_nonce;
    let build_output = ring_switch_build_w::<F, R>(
        &prepared_fold.instance,
        prepared_fold.witness,
        stack.ring_switch(),
        lp,
        is_terminal_fold,
    )?;
    let logical_w = build_output.w;
    scheduled.validate_next_w_len(logical_w.len())?;
    let next_commitment = if is_terminal_fold {
        None
    } else {
        let _span = tracing::info_span!("commit_w_level", level).entered();
        Some(crate::commit_w::<Cfg, C>(
            &scheduled.next_params,
            expanded,
            stack.commit(),
            &logical_w,
        )?)
    };
    let (next_commitment, final_witness) = bind_next_witness_for_ring_switch::<F, T>(
        transcript,
        is_terminal_fold,
        lp,
        next_commitment,
        if is_terminal_fold {
            Some(scheduled.next_params.log_basis)
        } else {
            None
        },
        build_output.terminal_artifacts,
        terminal_direct_witness_shape,
        prepared_fold.instance.opening_batch(),
    )?;
    let relation_matrix_row_layout = if is_terminal_fold {
        RelationMatrixRowLayout::WithoutDBlock
    } else {
        RelationMatrixRowLayout::WithDBlock
    };
    let rs = ring_switch_finalize::<F, E, T>(
        &prepared_fold.instance,
        expanded.as_ref(),
        transcript,
        &logical_w,
        lp,
        prepared_fold.row_coefficients.as_deref(),
        relation_matrix_row_layout,
    )?;

    let relation_rhs_layout = relation_rhs_layout_for(
        lp,
        prepared_fold.instance.opening_batch(),
        prepared_fold.instance.relation_matrix_row_layout(),
    )?;
    let relation_claim = relation_claim_from_layout_extension::<F, E>(
        prepared_fold.instance.role_dims(),
        &relation_rhs_layout,
        &rs.tau1,
        rs.alpha,
        prepared_fold.instance.v(),
        &prepared_fold.commitment,
    )?;
    let (stage1_proof, stage1_point, s_claim) = if is_terminal_fold {
        (None, vec![E::zero(); rs.col_bits + rs.ring_bits], E::zero())
    } else {
        let (stage1_proof, stage1_point, s_claim) = prove_stage1::<F, E, T>(transcript, &rs)?;
        transcript.append_serde(ABSORB_SUMCHECK_S_CLAIM, &stage1_proof.s_claim);
        (Some(stage1_proof), stage1_point, s_claim)
    };
    let batching_coeff: E = if is_terminal_fold {
        E::zero()
    } else {
        sample_ext_challenge::<F, E, T>(transcript, CHALLENGE_SUMCHECK_BATCH)
    };
    // EvaluationTrace is the last padded relation row: weight openings by
    // `eq(tau1, EvaluationTrace_row_index)`.
    let opening_batch = prepared_fold.instance.opening_batch();
    let evaluation_trace_row =
        lp.evaluation_trace_row_index_for_layout(relation_matrix_row_layout, opening_batch)?;
    let evaluation_trace_weight = evaluation_trace_row_weight(evaluation_trace_row, &rs.tau1)?;
    let trace_opening_claim = evaluation_trace_weight * prepared_fold.trace_eval_target;
    ensure_trace_stage2_supported(E::EXT_DEGREE)?;
    let trace_compact = if let Some(row_coefficients) = prepared_fold.row_coefficients.as_ref() {
        if lp.has_precommitted_groups() {
            Some(akita_types::build_multi_group_root_stage2_trace_table::<
                F,
                E,
            >(
                ring_d,
                lp,
                prepared_fold.instance.opening_batch(),
                prepared_fold
                    .trace_prepared_points
                    .as_ref()
                    .ok_or(AkitaError::InvalidProof)?,
                row_coefficients,
                prepared_fold.trace_claim_scales.as_deref(),
                evaluation_trace_weight,
                rs.live_x_cols,
            )?)
        } else {
            let num_trace_blocks = prepared_fold
                .instance
                .opening_batch()
                .num_total_polynomials()
                .checked_mul(lp.num_blocks)
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("trace block count overflow".to_string())
                })?;
            let (_, layout) = trace_layout_for_instance(
                lp,
                &prepared_fold.instance,
                rs.col_bits,
                rs.ring_bits,
                num_trace_blocks,
            )?;
            Some(build_root_stage2_trace_table::<F, E>(
                ring_d,
                lp.num_blocks,
                &layout,
                prepared_fold.instance.opening_batch(),
                prepared_fold
                    .trace_prepared_points
                    .as_ref()
                    .and_then(|points| points.first())
                    .ok_or(AkitaError::InvalidProof)?,
                row_coefficients,
                prepared_fold.trace_claim_scales.as_deref(),
                evaluation_trace_weight,
                rs.live_x_cols,
            )?)
        }
    } else if let Some(prepared) = prepared_fold
        .trace_prepared_points
        .as_ref()
        .and_then(|points| points.first())
    {
        let (_, layout) = trace_layout_for_instance(
            lp,
            &prepared_fold.instance,
            rs.col_bits,
            rs.ring_bits,
            lp.num_blocks,
        )?;
        Some(build_recursive_stage2_trace_table::<F, E>(
            ring_d,
            &layout,
            prepared,
            prepared_fold.trace_scale,
            evaluation_trace_weight,
            rs.live_x_cols,
        )?)
    } else {
        None
    };
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
        s_claim,
        relation_claim,
        trace_compact,
        trace_opening_claim,
    )?;
    if is_terminal_fold {
        let final_witness = final_witness.ok_or_else(|| {
            AkitaError::InvalidInput("terminal fold did not bind a final witness".to_string())
        })?;
        let proof = TerminalLevelProof::new_with_extension_opening_reduction(
            prepared_fold.extension_opening_reduction,
            stage2_sumcheck_proof,
            final_witness,
            fold_grind_nonce,
        );
        Ok(FoldProveOutput::Terminal(Box::new(proof)))
    } else {
        let w_eval = {
            let _span = tracing::info_span!("multilinear_eval", level).entered();
            stage2_prover.final_w_eval()
        };
        let proof_w_eval = w_eval;
        transcript.append_serde(ABSORB_STAGE2_NEXT_W_EVAL, &proof_w_eval);
        let stage3_sumcheck_proof = prove_stage3::<F, E, T>(
            level,
            lp.setup_contribution_mode,
            expanded.as_ref(),
            prefix_slots,
            lp,
            &scheduled.next_params,
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
        )?;
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
        let NextWitnessCommitment {
            witness: packed_witness,
            commitment: committed_commitment,
            hint: committed_hint,
        } = next_commitment.ok_or_else(|| {
            AkitaError::InvalidInput("intermediate fold did not bind a next commitment".to_string())
        })?;
        let w_commitment_proof = committed_commitment.clone();
        let level_proof = AkitaLevelProof::Intermediate {
            extension_opening_reduction: prepared_fold.extension_opening_reduction,
            v: prepared_fold.instance.v().clone().into_compact(),
            fold_grind_nonce,
            stage1: stage1_proof,
            stage2: AkitaStage2Proof::Intermediate(AkitaIntermediateStage2Proof {
                sumcheck_proof: stage2_sumcheck_proof,
                next_w_commitment: w_commitment_proof.into_compact(),
                next_w_eval: proof_w_eval,
            }),
            stage3_sumcheck_proof,
        };

        let (committed_witness, logical_w) = match packed_witness {
            Some(packed_witness) => (packed_witness, Some(logical_w)),
            None => (logical_w, None),
        };

        Ok(FoldProveOutput::Intermediate(Box::new(ProveLevelOutput {
            level_proof,
            next_state: SuffixProverState {
                w: committed_witness,
                logical_w,
                commitment: committed_commitment,
                hint: committed_hint,
                log_basis: scheduled.next_params.log_basis,
                sumcheck_challenges: next_opening_point,
                opening: next_opening,
                setup_prefix_opening,
            },
        })))
    }
}

#[allow(clippy::too_many_arguments)]
pub(in crate::protocol::core) fn bind_next_witness_for_ring_switch<F, T>(
    transcript: &mut T,
    is_terminal_fold: bool,
    lp: &LevelParams,
    next_commitment: Option<NextWitnessCommitment<F>>,
    final_log_basis: Option<u32>,
    terminal_artifacts: Option<RingSwitchTerminalArtifacts<F>>,
    terminal_direct_witness_shape: Option<&CleartextWitnessShape>,
    opening_batch: &OpeningClaimsLayout,
) -> Result<BoundNextWitness<F>, AkitaError>
where
    F: FieldCore + CanonicalField + HalvingField + AkitaSerialize,
    T: Transcript<F>,
{
    if is_terminal_fold {
        final_log_basis.ok_or_else(|| {
            AkitaError::InvalidInput("terminal fold missing final witness basis".to_string())
        })?;
        if let Some(artifacts) = terminal_artifacts {
            let CleartextWitnessShape::SegmentTyped(scheduled_shape) =
                terminal_direct_witness_shape.ok_or_else(|| {
                    AkitaError::InvalidSetup(
                        "terminal fold missing scheduled segment-typed witness shape".to_string(),
                    )
                })?
            else {
                return Err(AkitaError::InvalidSetup(
                    "terminal fold expected segment-typed witness shape".to_string(),
                ));
            };
            let group_parts = artifacts
                .groups
                .iter()
                .enumerate()
                .map(|(layout_index, group)| {
                    let params = lp.root_group_params(opening_batch, group.group_index)?;
                    let (num_w_vectors, num_t_vectors, num_z_segments) =
                        akita_types::tail_segment_multiplicities_from_layout_for_params(
                            params,
                            lp.ring_dimension,
                            &scheduled_shape.layout,
                            layout_index,
                        )?;
                    Ok(SegmentTypedWitnessGroupParts {
                        params,
                        num_w_vectors,
                        num_t_vectors,
                        num_z_segments,
                        e_folded: &group.e_folded,
                        recomposed_inner_rows: &group.recomposed_inner_rows,
                        z_folded_centered_flat: group.z_folded_centered_flat(),
                    })
                })
                .collect::<Result<Vec<_>, AkitaError>>()?;
            let segment = build_segment_typed_witness_from_groups::<F>(
                artifacts.ring_dim(),
                &group_parts,
                &artifacts.r,
                lp,
                opening_batch.num_groups(),
            )?;
            if segment.layout != scheduled_shape.layout {
                return Err(AkitaError::InvalidSetup(
                    "segment-typed witness layout does not match schedule".to_string(),
                ));
            }
            let parts = segment.terminal_transcript_parts()?;
            transcript.absorb_and_record_bytes(ABSORB_TERMINAL_W_REMAINDER, &parts.remainder);
            return Ok((None, Some(CleartextWitnessProof::SegmentTyped(segment))));
        }
        return Err(AkitaError::InvalidSetup(
            "terminal fold missing segment-typed witness artifacts".to_string(),
        ));
    }

    let next_commitment = next_commitment.ok_or_else(|| {
        AkitaError::InvalidInput("intermediate fold missing next commitment".to_string())
    })?;
    transcript.append_serde(
        ABSORB_NEXT_LEVEL_WITNESS_BINDING,
        &next_commitment.commitment,
    );
    Ok((Some(next_commitment), None))
}

#[allow(clippy::too_many_arguments)]
pub(in crate::protocol::core) fn prove_stage1<F, E, T>(
    transcript: &mut T,
    rs: &RingSwitchOutput<E>,
) -> Result<(AkitaStage1Proof<E>, Vec<E>, E), AkitaError>
where
    F: FieldCore + CanonicalField,
    E: ExtField<F> + HasUnreducedOps + HasOptimizedFold + FromPrimitiveInt + AkitaSerialize,
    T: Transcript<F>,
{
    let _sumcheck_span = tracing::info_span!("stage1_sumcheck").entered();
    let tau0_reordered = reorder_stage1_coords(&rs.tau0, rs.col_bits, rs.ring_bits);
    let stage1_prover = AkitaStage1Prover::new(
        &rs.w_evals_compact,
        &tau0_reordered,
        rs.b,
        rs.live_x_cols,
        rs.col_bits,
        rs.ring_bits,
    )?;
    let (stage1_proof, stage1_point) = stage1_prover.prove::<F, T>(transcript)?;
    let s_claim = stage1_proof.s_claim;
    Ok((stage1_proof, stage1_point, s_claim))
}

#[allow(clippy::too_many_arguments)]
fn prove_stage2<F, E, T>(
    level: usize,
    transcript: &mut T,
    batching_coeff: E,
    rs: RingSwitchOutput<E>,
    stage1_point: &[E],
    s_claim: E,
    relation_claim: E,
    trace_compact: Option<TraceTable<E>>,
    trace_opening_claim: E,
) -> Result<Stage2ProveResult<E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: ExtField<F> + HasUnreducedOps + HasOptimizedFold + FromPrimitiveInt + AkitaSerialize,
    T: Transcript<F>,
{
    let _sumcheck_span = tracing::info_span!("stage2_sumcheck").entered();
    let mut stage2_prover = AkitaStage2Prover::new(
        batching_coeff,
        rs.w_evals_compact,
        stage1_point,
        s_claim,
        rs.b,
        rs.alpha_evals_y,
        rs.relation_matrix_col_evals,
        rs.live_x_cols,
        rs.col_bits,
        rs.ring_bits,
        relation_claim,
        trace_compact.clone(),
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
    lp: &LevelParams,
    next_level_params: &LevelParams,
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

#[cfg(all(test, feature = "logging-transcript"))]
mod transcript_schedule_tests {
    use super::*;
    use akita_field::{Fp32, FpExt2, NegOneNr};
    use akita_transcript::{
        is_ext_limb_label, labels, AkitaTranscript, LoggingTranscript, Transcript, TranscriptEvent,
    };

    type F = Fp32<251>;
    type E = FpExt2<F, NegOneNr>;

    fn sample_stage2_batching_coeff<T: Transcript<F>>(
        transcript: &mut T,
        is_terminal_fold: bool,
    ) -> E {
        if is_terminal_fold {
            E::zero()
        } else {
            sample_ext_challenge::<F, E, T>(transcript, CHALLENGE_SUMCHECK_BATCH)
        }
    }

    fn squeezes_logical_label(events: &[TranscriptEvent], base: &[u8]) -> bool {
        events.iter().any(|event| {
            matches!(event, TranscriptEvent::Squeeze { label, .. }
                if label.as_slice() == base || is_ext_limb_label(label, base))
        })
    }

    #[test]
    fn terminal_fold_skips_stage2_batch_challenge() {
        let mut transcript = LoggingTranscript::wrap(AkitaTranscript::<F>::new(b"fold/terminal"));
        let batching = sample_stage2_batching_coeff(&mut transcript, true);
        assert!(batching.is_zero());
        assert!(
            !squeezes_logical_label(transcript.events(), labels::CHALLENGE_SUMCHECK_BATCH),
            "terminal fold must not squeeze stage-2 batch challenge for trace weighting"
        );
    }

    #[test]
    fn intermediate_fold_squeezes_stage2_batch_challenge() {
        let mut transcript =
            LoggingTranscript::wrap(AkitaTranscript::<F>::new(b"fold/intermediate"));
        let batching = sample_stage2_batching_coeff(&mut transcript, false);
        assert!(!batching.is_zero());
        assert!(
            squeezes_logical_label(transcript.events(), labels::CHALLENGE_SUMCHECK_BATCH),
            "intermediate fold must squeeze stage-2 batch challenge before trace weighting"
        );
    }
}
