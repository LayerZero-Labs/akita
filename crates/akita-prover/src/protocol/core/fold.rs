use super::*;
use crate::compute::{
    tensor_root_projection, CommitmentComputeBackend, ComputeBackendSetup, DigitRowsComputeBackend,
    ProverComputeStack, RootOpeningSource, RootPolyMeta, RuntimeOpeningProveBackendFor,
    RuntimeRingSwitchProveBackend, RuntimeRootProvePoly, RuntimeTensorBackendFor,
};
use crate::protocol::sumcheck::DigitRangeProver;
use crate::RootTensorProjectionPoly;
use akita_field::unreduced::ReduceTo;
use akita_field::AdditiveGroup;

use crate::protocol::ring_switch::RingSwitchTerminalArtifacts;
use akita_types::{
    build_segment_typed_witness_from_groups, dispatch_for_field, DigitRangeEqualityPoint,
    DigitRangePlan, FlatBooleanDomain, OpeningClaimsLayout, RelationRangeImagePlan,
    SegmentTypedWitnessGroupParts, SegmentTypedWitnessShape,
};

fn trace_layout_for_instance<F: FieldCore + CanonicalField>(
    lp: &LevelParams,
    instance: &RingRelationInstance<F>,
    opening_source_len: usize,
    col_bits: usize,
    ring_bits: usize,
    num_trace_blocks: usize,
) -> Result<akita_types::TraceWeightLayout, AkitaError> {
    let witness_layout = instance.segment_layout(lp, None)?;
    trace_weight_layout_from_segment(
        lp,
        &witness_layout,
        opening_source_len,
        col_bits,
        ring_bits,
        num_trace_blocks,
    )
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
    /// Canonical terminal `t` state already bound by the predecessor.
    pub(in crate::protocol::core) terminal_t_state: Option<RingVec<F>>,
    /// Per-block terminal `t` rows retained by the predecessor's inner commit.
    pub(in crate::protocol::core) terminal_recomposed_inner_rows: Option<Vec<RingVec<F>>>,
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
    level_params: &LevelParams,
    alpha_bits: usize,
    basis: BasisMode,
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
                relation_matrix_row_layout,
                terminal_tail_t_vectors,
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
                    relation_matrix_row_layout,
                    terminal_tail_t_vectors,
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
    block_claims: ProverOpeningData<'a, E, Q, F>,
    protocol_point: &'a [E],
    reduction: Option<ExtensionOpeningReduction<E>>,
    row_coefficients: Option<Vec<E>>,
    trace_opening_batch: &'a OpeningClaimsLayout,
    level_params: &'a LevelParams,
    alpha_bits: usize,
    basis: BasisMode,
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
    let opening_batch = block_claims
        .opening_claims()
        .layout()
        .map_err(|err| AkitaError::InvalidInput(format!("opening batch layout failed: {err:?}")))?;
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
                    let prepared_point = prepare_opening_point::<F, E, D>(
                        &group_protocol_point,
                        basis,
                        group_lp.num_positions_per_block(),
                        group_lp.num_live_blocks(),
                        alpha_bits,
                    )
                    .map_err(|err| {
                        AkitaError::InvalidInput(format!(
                            "prepare opening point group {group_index} failed: {err:?}"
                        ))
                    })?;
                    let group_polys = block_claims.group_polys(group_index).map_err(|err| {
                        AkitaError::InvalidInput(format!(
                            "root group polynomials {group_index} failed: {err:?}"
                        ))
                    })?;
                    let (group_folded_rings, group_e_folded_by_claim) =
                        evaluate_claims_at_prepared_point(
                            opening.backend(),
                            Some(opening.prepared()),
                            group_polys,
                            &prepared_point,
                            group_lp.num_positions_per_block(),
                        )
                        .map_err(|err| {
                            AkitaError::InvalidInput(format!(
                                "evaluate claims group {group_index} failed: {err:?}"
                            ))
                        })?;
                    for pt in &prepared_point.padded_point {
                        append_ext_field::<F, E, T>(transcript, ABSORB_EVALUATION_CLAIMS, pt);
                    }
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
                )
                .map_err(|err| {
                    AkitaError::InvalidInput(format!("compute trace target failed: {err:?}"))
                })?;
                let row_coefficient_rings = row_coefficient_rings::<F, E, D>(&row_coefficients)
                    .map_err(|err| {
                        AkitaError::InvalidInput(format!("row coefficient rings failed: {err:?}"))
                    })?;
                Ok::<_, AkitaError>((
                    prepared_points,
                    e_folded_by_claim,
                    trace_target,
                    row_coefficients,
                    RingVec::from_ring_elems(&row_coefficient_rings),
                ))
            }
        )
        .map_err(|err| {
            AkitaError::InvalidInput(format!("root opening preparation failed: {err:?}"))
        })?;
    let commitment = if LevelParams::has_commitment_block(relation_matrix_row_layout) {
        block_claims.fold_commitment(level_params).map_err(|err| {
            AkitaError::InvalidInput(format!("fold commitment preparation failed: {err:?}"))
        })?
    } else {
        RingVec::from_coeffs(Vec::new())
    };
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
        relation_matrix_row_layout,
        terminal_tail_t_vectors,
    )
    .map_err(|err| {
        AkitaError::InvalidInput(format!("ring relation preparation failed: {err:?}"))
    })?;
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
        terminal_t_state: None,
        terminal_recomposed_inner_rows: None,
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
    is_terminal_fold: bool,
    terminal_direct_witness_shape: Option<&SegmentTypedWitnessShape>,
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
        prepared_fold.terminal_recomposed_inner_rows.as_deref(),
    )
    .map_err(|err| {
        AkitaError::InvalidInput(format!("ring-switch witness build failed: {err:?}"))
    })?;
    if is_terminal_fold {
        let RingSwitchBuildOutput::Terminal(terminal_artifacts) = build_output else {
            return Err(AkitaError::InvalidProof);
        };
        let final_witness = bind_terminal_witness::<F, T>(
            transcript,
            lp,
            terminal_artifacts,
            terminal_direct_witness_shape,
            prepared_fold.instance.opening_batch(),
            prepared_fold
                .terminal_t_state
                .as_ref()
                .ok_or(AkitaError::InvalidProof)?,
        )
        .map_err(|err| {
            AkitaError::InvalidInput(format!("terminal witness binding failed: {err:?}"))
        })?;
        let proof = TerminalLevelProof {
            extension_opening_reduction: prepared_fold.extension_opening_reduction,
            fold_grind_nonce,
            final_witness,
        };
        return Ok(FoldProveOutput::Terminal(Box::new(proof)));
    }
    let RingSwitchBuildOutput::Intermediate(logical_w) = build_output else {
        return Err(AkitaError::InvalidProof);
    };
    let next_params = scheduled.next_params.as_ref().ok_or_else(|| {
        AkitaError::InvalidSetup("non-terminal fold is missing successor params".into())
    })?;
    scheduled.validate_next_w_len(logical_w.len())?;
    let _span = tracing::info_span!("commit_w_level", level).entered();
    let next_commitment = crate::commit_w::<Cfg, C>(
        next_params,
        expanded,
        stack.commit(),
        &logical_w,
        scheduled.next_witness_binding.ok_or_else(|| {
            AkitaError::InvalidSetup("non-terminal fold is missing its outgoing binding".into())
        })?,
    )?;
    drop(_span);
    match &next_commitment.binding {
        NextWitnessState::OuterCommitment(commitment) => {
            transcript.append_serde(ABSORB_NEXT_LEVEL_WITNESS_BINDING, commitment);
        }
        NextWitnessState::TerminalInnerState { t_state, .. } => {
            let bytes = akita_types::raw_field_segment_bytes(t_state)?;
            transcript.absorb_and_record_bytes(ABSORB_NEXT_LEVEL_WITNESS_BINDING, &bytes);
        }
    }
    let relation_matrix_row_layout = RelationMatrixRowLayout::WithDBlock;
    let next_opening_ring_dim = next_params.d_a();
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
        relation_matrix_row_layout,
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
    let evaluation_trace_row =
        lp.evaluation_trace_row_index_for_layout(relation_matrix_row_layout, opening_batch)?;
    let evaluation_trace_weight = evaluation_trace_row_weight(evaluation_trace_row, &rs.tau1)?;
    let trace_opening_claim = evaluation_trace_weight * prepared_fold.trace_eval_target;
    ensure_trace_stage2_supported(E::EXT_DEGREE)?;
    let trace_witness_layout = relation_range_image_plan.witness_layout();
    let trace_opening_source_len = trace_witness_layout.total_len();
    let trace_x_cols = akita_types::opening_domain_len(trace_opening_source_len)?;
    let trace_col_bits = trace_x_cols.trailing_zeros() as usize;
    let trace_ring_bits = ring_d.trailing_zeros() as usize;
    let trace_compact = if let Some(row_coefficients) = prepared_fold.row_coefficients.as_ref() {
        if lp.has_precommitted_groups() {
            Some(akita_types::build_multi_group_root_stage2_trace_table::<
                F,
                E,
            >(
                ring_d,
                trace_witness_layout,
                trace_opening_source_len,
                lp,
                prepared_fold.instance.opening_batch(),
                prepared_fold
                    .trace_prepared_points
                    .as_ref()
                    .ok_or(AkitaError::InvalidProof)?,
                row_coefficients,
                prepared_fold.trace_claim_scales.as_deref(),
                evaluation_trace_weight,
                trace_opening_source_len,
            )?)
        } else {
            let num_trace_blocks = prepared_fold
                .instance
                .opening_batch()
                .num_total_polynomials()
                .checked_mul(lp.num_live_blocks)
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("trace block count overflow".to_string())
                })?;
            let layout = trace_layout_for_instance(
                lp,
                &prepared_fold.instance,
                trace_opening_source_len,
                trace_col_bits,
                trace_ring_bits,
                num_trace_blocks,
            )?;
            Some(build_root_stage2_trace_table::<F, E>(
                ring_d,
                lp.num_live_blocks,
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
                trace_opening_source_len,
            )?)
        }
    } else if let Some(prepared) = prepared_fold
        .trace_prepared_points
        .as_ref()
        .and_then(|points| points.first())
    {
        let layout = trace_layout_for_instance(
            lp,
            &prepared_fold.instance,
            trace_opening_source_len,
            trace_col_bits,
            trace_ring_bits,
            lp.num_live_blocks,
        )?;
        Some(build_recursive_stage2_trace_table::<F, E>(
            ring_d,
            &layout,
            prepared,
            prepared_fold.trace_scale,
            evaluation_trace_weight,
            trace_opening_source_len,
        )?)
    } else {
        None
    }
    .map(|table| {
        remap_trace_table(
            table,
            trace_opening_source_len,
            ring_d,
            next_opening_source_len,
            next_opening_ring_dim,
            logical_w.len(),
        )
    })
    .transpose()?;
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
        trace_compact,
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
    let stage3_sumcheck_proof = prove_stage3::<F, E, T>(
        level,
        lp.setup_contribution_mode,
        expanded.as_ref(),
        prefix_slots,
        lp,
        next_params,
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
        NextWitnessState::TerminalInnerState {
            t_state,
            recomposed_inner_rows,
        } => (
            akita_types::NextWitnessBinding::TerminalInnerState,
            NextWitnessState::TerminalInnerState {
                t_state,
                recomposed_inner_rows,
            },
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

    Ok(FoldProveOutput::Intermediate(Box::new(ProveLevelOutput {
        level_proof,
        next_state: SuffixProverState {
            w: committed_witness,
            logical_w,
            binding: next_binding,
            hint: committed_hint,
            log_basis: next_params.log_basis,
            sumcheck_challenges: next_opening_point,
            opening: next_opening,
            setup_prefix_opening,
        },
    })))
}

#[allow(clippy::too_many_arguments)]
pub(in crate::protocol::core) fn bind_terminal_witness<F, T>(
    transcript: &mut T,
    lp: &LevelParams,
    artifacts: RingSwitchTerminalArtifacts<F>,
    terminal_direct_witness_shape: Option<&SegmentTypedWitnessShape>,
    opening_batch: &OpeningClaimsLayout,
    bound_t_state: &RingVec<F>,
) -> Result<SegmentTypedWitness<F>, AkitaError>
where
    F: FieldCore + CanonicalField + HalvingField + AkitaSerialize,
    T: Transcript<F>,
{
    let scheduled_shape = terminal_direct_witness_shape.ok_or_else(|| {
        AkitaError::InvalidSetup("terminal fold missing scheduled witness shape".to_string())
    })?;
    let group_parts = artifacts
        .groups
        .iter()
        .enumerate()
        .map(|(layout_index, group)| {
            let params = lp.group_params(opening_batch, group.group_index)?;
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
    let segment =
        build_segment_typed_witness_from_groups::<F>(artifacts.ring_dim(), &group_parts, lp)?;
    if segment.layout != scheduled_shape.layout {
        return Err(AkitaError::InvalidSetup(
            "segment-typed witness layout does not match schedule".to_string(),
        ));
    }
    let parts = segment.terminal_transcript_parts()?;
    if segment.t_fields.coeffs() != bound_t_state.coeffs() {
        return Err(AkitaError::InvalidProof);
    }
    transcript.absorb_and_record_bytes(ABSORB_TERMINAL_W_REMAINDER, &parts.response);
    Ok(segment)
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
    let equality_point = DigitRangeEqualityPoint::from_column_then_ring_challenges(
        &rs.tau0,
        rs.col_bits,
        rs.ring_bits,
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

fn remap_trace_table<E: FieldCore>(
    table: TraceTable<E>,
    source_len: usize,
    source_ring_dim: usize,
    destination_len: usize,
    destination_ring_dim: usize,
    physical_field_len: usize,
) -> Result<TraceTable<E>, AkitaError> {
    let source_capacity = source_len
        .checked_mul(source_ring_dim)
        .ok_or_else(|| AkitaError::InvalidSetup("trace source capacity overflow".into()))?;
    let destination_capacity = destination_len
        .checked_mul(destination_ring_dim)
        .ok_or_else(|| AkitaError::InvalidSetup("trace destination capacity overflow".into()))?;
    if physical_field_len > source_capacity
        || physical_field_len > destination_capacity
        || source_ring_dim == 0
        || destination_ring_dim == 0
    {
        return Err(AkitaError::InvalidProof);
    }
    // Fast path: when the source and destination expose the same Boolean
    // source and ring geometry, the remap is the identity over the complete
    // live prefix, so keep the existing representation without materializing
    // or allocating anything.
    if source_ring_dim == destination_ring_dim
        && source_len == destination_len
        && physical_field_len == source_capacity
    {
        return Ok(table);
    }
    // Stage 2 represents the padded Boolean suffix implicitly. Allocate only
    // the exact destination prefix and read the source in place, so a remap
    // never holds either source or destination Boolean capacity densely.
    let mut destination = vec![E::zero(); physical_field_len];
    for physical in 0..physical_field_len {
        let source_col =
            akita_types::checked_opening_source_index(source_len, physical / source_ring_dim)?;
        let source_coeff = physical % source_ring_dim;
        let destination_col = akita_types::checked_opening_source_index(
            destination_len,
            physical / destination_ring_dim,
        )?;
        let destination_index =
            destination_col * destination_ring_dim + physical % destination_ring_dim;
        destination[destination_index] = table.get(source_col, source_coeff, source_ring_dim);
    }
    Ok(TraceTable::ring_dense(destination))
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
    trace_compact: Option<TraceTable<E>>,
    trace_opening_claim: E,
    plan: RelationRangeImagePlan,
) -> Result<Stage2ProveResult<E>, AkitaError>
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
    // `alpha(y)` powers over the inner ring domain. For the flattened fallback
    // (`ring_bits == 0`) this is `[1]`; for uniform geometry it is
    // `[1, alpha, ..., alpha^(2^ring_bits - 1)]`, supplying the per-coefficient
    // spread that the compact per-column relation table `M(x)` omits.
    let alpha_evals_y = akita_algebra::ring::scalar_powers(rs.alpha, 1usize << rs.ring_bits);
    let mut stage2_prover = AkitaStage2Prover::new(
        batching_coeff,
        rs.w_evals_compact,
        stage1_point,
        range_image_evaluation,
        plan.digit_range_plan().basis(),
        alpha_evals_y,
        rs.relation_weight_evals,
        derived_live_x_cols,
        derived_col_bits,
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

#[cfg(test)]
mod trace_remap_tests {
    use super::*;
    use akita_field::Fp32;

    type F = Fp32<251>;

    #[test]
    fn nonidentity_trace_remap_keeps_only_live_prefix() {
        let source = (0..12).map(F::from_u64).collect::<Vec<_>>();
        let remapped = remap_trace_table(TraceTable::ring_dense(source.clone()), 3, 4, 6, 2, 12)
            .expect("valid trace remap")
            .into_ring_dense()
            .expect("dense trace");

        assert_eq!(remapped, source);
        assert_eq!(remapped.len(), 12);
    }
}
