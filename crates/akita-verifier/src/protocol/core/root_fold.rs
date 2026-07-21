use akita_field::{
    AkitaError, CanonicalField, ExtField, FieldCore, FrobeniusExtField, FromPrimitiveInt,
    HalvingField, MulBaseUnreduced, RandomSampling,
};
use akita_serialization::AkitaSerialize;
use akita_transcript::labels::{ABSORB_COMMITMENT, ABSORB_EVALUATION_CLAIMS};
use akita_transcript::{append_ext_field, Transcript};
use akita_types::{
    append_claim_values_to_transcript, dispatch_for_field, prepare_opening_point,
    ring_subfield_packed_extension_opening_point, root_trace_block_opening,
    sample_public_row_coefficients, AkitaVerifierSetup, BasisMode, Commitment,
    ExtensionOpeningReductionProof, FoldLevelProof, FpExtEncoding, LevelParams, OpeningClaims,
    OpeningClaimsLayout, RelationMatrixRowLayout, RingVec, RingView, SetupSumcheckProof,
};

use super::{
    verify_fold, verify_fold_eor, FoldVerifyOutput, PreparedFoldPayload, PreparedFoldReplay,
    PreparedNextWitness, TracePreparation,
};

/// Verify the folded root proof payload.
///
/// This replays the canonical root transcript layout: batch-shape header,
/// commitments, padded opening points, per-claim field openings, row
/// coefficients, EOR if present, y-rings, ring switch, stage-1 when present,
/// stage-2, and stage-3 setup sumcheck when required by the intermediate branch.
///
/// # Errors
///
/// Returns an error if the proof shape is inconsistent, any public trace check
/// fails, ring-switch replay fails, or a sumcheck verifier rejects.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub(super) fn verify_root<F, E, T>(
    proof: &FoldLevelProof<F, E>,
    setup: &AkitaVerifierSetup<F>,
    transcript: &mut T,
    claims: &OpeningClaims<'_, E, &Commitment<F>>,
    basis: BasisMode,
    root_lp: &LevelParams,
    next_fold_level_params: &LevelParams,
    next_t_state: Option<&[u8]>,
) -> Result<FoldVerifyOutput<E>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling + HalvingField,
    E: FpExtEncoding<F>
        + ExtField<F>
        + FrobeniusExtField<F>
        + FromPrimitiveInt
        + AkitaSerialize
        + MulBaseUnreduced<F>,
    T: Transcript<F>,
{
    let relation_matrix_row_layout = RelationMatrixRowLayout::WithDBlock;
    let extension_opening_reduction = proof.extension_opening_reduction();
    let stage3_sumcheck_proof = proof
        .stage3_for_mode(
            root_lp.setup_contribution_mode,
            Some(next_fold_level_params),
        )?
        .map(|(proof, _)| proof);
    let next_witness = match (proof.next_w_commitment(), next_t_state) {
        (Some(commitment), None) => PreparedNextWitness::Commitment {
            commitment,
            ring_dim: next_fold_level_params.role_dims().d_b(),
        },
        (None, Some(t_state)) if !t_state.is_empty() => PreparedNextWitness::TerminalT(t_state),
        _ => return Err(AkitaError::InvalidProof),
    };
    let openings = claims.flat_evaluations();
    let opening_batch = claims.layout().map_err(|_| AkitaError::InvalidProof)?;
    let shared_opening_point = claims.point();
    let num_claims = opening_batch.num_total_polynomials();
    if openings.len() != num_claims {
        return Err(AkitaError::InvalidProof);
    }
    let ring_dim = root_lp.role_dims().d_b();

    // Transcript binding, D-free and byte-identical to the prover's absorb
    // (`ProverOpeningData::append_to_transcript`): batch shape header, then each
    // group commitment's flat coefficients under `ring_dim` in `OpeningClaims`
    // order, then the shared opening point. Each group's committed row count is
    // validated against its (final vs frozen-precommit) params before the
    // absorb, so a swapped/truncated group commitment rejects here.
    opening_batch.append_batch_shape_to_transcript::<F, T>(transcript)?;
    for group_index in 0..opening_batch.num_groups() {
        let commitment = claims.group_commitment(group_index)?;
        let expected_rows = root_lp.group_commitment_rows(&opening_batch, group_index)?;
        let commitment_view = RingView::new(commitment.rows().coeffs(), ring_dim)?;
        if commitment_view.num_rings() != expected_rows {
            return Err(AkitaError::InvalidProof);
        }
        commitment_view.append_flat_to_transcript::<T>(ABSORB_COMMITMENT, transcript)?;
    }
    for coord in shared_opening_point {
        append_ext_field::<F, E, T>(transcript, ABSORB_EVALUATION_CLAIMS, coord);
    }

    // D-free root replay: typed kernels dispatch inside `verify_fold` and
    // `verify_fold_eor` on per-role dimensions. Grouped roots (`G > 1`) never
    // collapse into a synthetic single group.
    if opening_batch.num_groups() > 1 {
        return verify_multi_group_root_inner::<F, E, T>(
            proof,
            setup,
            transcript,
            claims,
            &openings,
            &opening_batch,
            shared_opening_point,
            extension_opening_reduction,
            stage3_sumcheck_proof,
            next_fold_level_params,
            basis,
            root_lp,
            next_witness,
        );
    }
    let commitment = claims
        .single_group_commitment()
        .copied()
        .ok_or(AkitaError::InvalidProof)?;
    verify_root_inner::<F, E, T>(
        proof,
        setup,
        transcript,
        commitment.rows(),
        &openings,
        &opening_batch,
        shared_opening_point,
        relation_matrix_row_layout,
        extension_opening_reduction,
        stage3_sumcheck_proof,
        next_fold_level_params,
        basis,
        root_lp,
        next_witness,
    )
}

/// Root-fold replay orchestrator (D-free).
///
/// Reached from [`verify_root`]; per-role typed kernels dispatch inside
/// [`verify_fold`] and [`verify_fold_eor`].
#[allow(clippy::too_many_arguments)]
fn verify_root_inner<F, E, T>(
    proof: &FoldLevelProof<F, E>,
    setup: &AkitaVerifierSetup<F>,
    transcript: &mut T,
    commitment: &RingVec<F>,
    openings: &[E],
    opening_batch: &OpeningClaimsLayout,
    shared_opening_point: &[E],
    relation_matrix_row_layout: RelationMatrixRowLayout,
    extension_opening_reduction: Option<&ExtensionOpeningReductionProof<E>>,
    stage3_sumcheck_proof: Option<&SetupSumcheckProof<E>>,
    next_fold_level_params: &LevelParams,
    basis: BasisMode,
    root_lp: &LevelParams,
    next_witness: PreparedNextWitness<'_, F>,
) -> Result<FoldVerifyOutput<E>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling + HalvingField,
    E: FpExtEncoding<F>
        + ExtField<F>
        + FrobeniusExtField<F>
        + FromPrimitiveInt
        + AkitaSerialize
        + MulBaseUnreduced<F>,
    T: Transcript<F>,
{
    let role_dims = root_lp.role_dims();
    let d_a = role_dims.d_a();
    let v_storage = proof.v.clone();

    let prepared_without_eor = if extension_opening_reduction.is_none() {
        let prepared_point =
            dispatch_for_field!(ProtocolDispatchSlot::Role(RingRole::Inner), F, d_a, |D| {
                prepare_opening_point::<F, E, D>(
                    shared_opening_point,
                    basis,
                    root_lp.num_positions_per_block,
                    root_lp.num_live_blocks,
                    d_a.trailing_zeros() as usize,
                )
            })?;
        for pt in &prepared_point.padded_point {
            append_ext_field::<F, E, T>(transcript, ABSORB_EVALUATION_CLAIMS, pt);
        }
        Some(prepared_point)
    } else {
        None
    };
    append_claim_values_to_transcript::<F, E, T>(openings, transcript);
    let row_coefficients = sample_public_row_coefficients::<F, E, T>(opening_batch, transcript)?;
    let root_eor = verify_fold_eor::<F, E, T>(
        extension_opening_reduction,
        shared_opening_point,
        openings,
        &row_coefficients,
        opening_batch,
        basis,
        root_lp,
        false,
        transcript,
    )?;
    let reduction_check = root_eor.reduction_challenges;
    let prepared_points = root_eor.prepared_points;
    let eor_trace_final = root_eor.final_relation;
    let prepared_point = if let Some(prepared) = prepared_without_eor.as_ref() {
        prepared
    } else {
        prepared_points.first().ok_or(AkitaError::InvalidProof)?
    };
    if extension_opening_reduction.is_some() {
        for pt in &prepared_point.padded_point {
            append_ext_field::<F, E, T>(transcript, ABSORB_EVALUATION_CLAIMS, pt);
        }
    }
    let trace_block_opening: Vec<E> = if let Some(rho) = &reduction_check {
        let protocol_point =
            dispatch_for_field!(ProtocolDispatchSlot::Role(RingRole::Inner), F, d_a, |D| {
                ring_subfield_packed_extension_opening_point::<F, E, D>(rho.len(), rho)
            })?;
        root_trace_block_opening::<E>(
            &protocol_point,
            root_lp.num_positions_per_block,
            root_lp.num_live_blocks,
            d_a.trailing_zeros() as usize,
        )?
    } else {
        root_trace_block_opening::<E>(
            shared_opening_point,
            root_lp.num_positions_per_block,
            root_lp.num_live_blocks,
            d_a.trailing_zeros() as usize,
        )?
    };
    let ordinary_trace_eval_target =
        opening_batch.batched_eval_target(&row_coefficients, openings)?;
    let trace_eval_target = eor_trace_final
        .as_ref()
        .map(|(final_claim, _)| *final_claim)
        .unwrap_or(ordinary_trace_eval_target);
    let trace_claim_scales = eor_trace_final
        .as_ref()
        .map(|(_, factors_by_point)| {
            let shared_factor = *factors_by_point.first().ok_or(AkitaError::InvalidProof)?;
            Ok(vec![shared_factor; opening_batch.num_total_polynomials()])
        })
        .transpose()?;

    // Chunked levels commit a wider (replicated-ẑ) next witness; size it
    // with the per-level chunk count (`num_chunks = 1` is unchanged).
    let w_len = akita_types::intermediate_w_ring_element_count_for_chunks(
        F::modulus_bits(),
        root_lp,
        opening_batch.num_total_polynomials(),
        root_lp.witness_chunk.num_chunks,
    )?
    .checked_mul(d_a)
    .ok_or_else(|| AkitaError::InvalidSetup("next witness length overflow".to_string()))?;
    let fold_grind_nonce = proof.fold_grind_nonce;
    // Scalar root: the sole commitment's rows are the whole M-row commitment
    // block.
    let commitment_rows = RingVec::from_coeffs(commitment.coeffs().to_vec());
    let next_witness_ring_dim = next_fold_level_params.role_dims().d_a();
    if !w_len.is_multiple_of(next_witness_ring_dim) {
        return Err(AkitaError::InvalidProof);
    }
    let prepared = PreparedFoldReplay {
        lp: root_lp,
        relation_matrix_row_layout,
        fold_grind_nonce,
        v: v_storage,
        opening_shape: opening_batch.clone(),
        commitment_rows,
        row_coefficients,
        group_ring_opening_points: vec![prepared_point.ring_opening_point.clone()],
        group_ring_multiplier_points: vec![prepared_point.ring_multiplier_point.clone()],
        w_len,
        payload: PreparedFoldPayload::Recursive {
            stage1: &proof.stage1,
            stage2: &proof.stage2,
            next_witness,
            next_witness_ring_dim,
            next_opening_source_len: w_len / next_witness_ring_dim,
            stage3: stage3_sumcheck_proof.map(|proof| (proof, next_fold_level_params)),
        },
        trace: TracePreparation {
            prepared_points: Some(vec![prepared_point.clone()]),
            block_opening: Some(trace_block_opening),
            eval_target: trace_eval_target,
            eval_scale: E::one(),
            claim_scales: trace_claim_scales,
            basis,
        },
    };
    verify_fold::<F, E, T>(setup, transcript, prepared)
}

/// Grouped folded-root replay (`G > 1`): preserve real per-group public claims,
/// commitments, and opening geometry rather than collapsing into a synthetic
/// single group.
///
/// The supported grouped shape is a degree-one one-hot same-point fold that
/// hands off to a singleton recursive suffix, so it never uses extension-opening
/// reduction and never terminates at the root. This builds one prepared opening
/// point per group (mirroring the prover's `finish_prepared_fold` loop and its
/// per-group padded-point absorbs), concatenates the group commitment rows in
/// relation-matrix row (final-first) order, sizes the next witness from the grouped witness
/// layout, and hands a per-group `PreparedFoldReplay` to [`verify_fold`].
///
/// # Errors
///
/// Returns [`AkitaError::InvalidProof`] for an extension-opening reduction, a
/// non-fold root, or any malformed group shape, and propagates layout/replay
/// errors.
#[allow(clippy::too_many_arguments)]
fn verify_multi_group_root_inner<F, E, T>(
    proof: &FoldLevelProof<F, E>,
    setup: &AkitaVerifierSetup<F>,
    transcript: &mut T,
    claims: &OpeningClaims<'_, E, &Commitment<F>>,
    openings: &[E],
    opening_batch: &OpeningClaimsLayout,
    shared_opening_point: &[E],
    extension_opening_reduction: Option<&ExtensionOpeningReductionProof<E>>,
    stage3_sumcheck_proof: Option<&SetupSumcheckProof<E>>,
    next_fold_level_params: &LevelParams,
    basis: BasisMode,
    root_lp: &LevelParams,
    next_witness: PreparedNextWitness<'_, F>,
) -> Result<FoldVerifyOutput<E>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling + HalvingField,
    E: FpExtEncoding<F>
        + ExtField<F>
        + FrobeniusExtField<F>
        + FromPrimitiveInt
        + AkitaSerialize
        + akita_field::MulBaseUnreduced<F>,
    T: Transcript<F>,
{
    // Grouped roots are degree-one one-hot same-point folds: extension-opening
    // reduction is a scalar-only path and must not appear here.
    if extension_opening_reduction.is_some() {
        return Err(AkitaError::InvalidProof);
    }
    let relation_matrix_row_layout = RelationMatrixRowLayout::WithDBlock;
    let role_dims = root_lp.role_dims();
    let d_a = role_dims.d_a();
    let alpha_bits = d_a.trailing_zeros() as usize;

    // One prepared opening point per group from the shared point, absorbing each
    // group's padded point in `OpeningClaims` order — byte-identical to the
    // prover's per-group absorb in `finish_prepared_fold`.
    let mut prepared_points = Vec::with_capacity(opening_batch.num_groups());
    for group_index in 0..opening_batch.num_groups() {
        let group_lp = root_lp.group_params(opening_batch, group_index)?;
        let target_len = alpha_bits
            .checked_add(group_lp.position_index_bits())
            .and_then(|n| n.checked_add(group_lp.block_index_bits()))
            .ok_or_else(|| {
                AkitaError::InvalidSetup("group opening point length overflow".to_string())
            })?;
        let point_vars = claims.group_point_vars(group_index)?;
        if point_vars.num_vars() != target_len {
            return Err(AkitaError::InvalidProof);
        }
        let group_point = point_vars
            .indices()
            .iter()
            .map(|&index| {
                shared_opening_point
                    .get(index)
                    .copied()
                    .ok_or(AkitaError::InvalidProof)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let prepared =
            dispatch_for_field!(ProtocolDispatchSlot::Role(RingRole::Inner), F, d_a, |D| {
                prepare_opening_point::<F, E, D>(
                    &group_point,
                    basis,
                    group_lp.num_positions_per_block(),
                    group_lp.num_live_blocks(),
                    alpha_bits,
                )
            })?;
        for pt in &prepared.padded_point {
            append_ext_field::<F, E, T>(transcript, ABSORB_EVALUATION_CLAIMS, pt);
        }
        prepared_points.push(prepared);
    }
    append_claim_values_to_transcript::<F, E, T>(openings, transcript);
    let row_coefficients = sample_public_row_coefficients::<F, E, T>(opening_batch, transcript)?;
    let trace_eval_target = opening_batch.batched_eval_target(&row_coefficients, openings)?;

    // Concatenate group commitment rows in relation-matrix row (final-first) order, matching
    // the prover's `RingRelationProver` commitment-row concatenation and
    // `relation_rhs_layout_for` block order.
    let order = opening_batch.root_group_order()?;
    let mut commitment_coeffs = Vec::new();
    for &group_index in &order {
        let commitment = claims.group_commitment(group_index)?;
        commitment_coeffs.extend_from_slice(commitment.rows().coeffs());
    }
    let commitment_rows = RingVec::from_coeffs(commitment_coeffs);

    let w_len = root_lp.next_w_len::<F>(opening_batch, relation_matrix_row_layout)?;
    let fold_grind_nonce = proof.fold_grind_nonce;
    let v_storage = proof.v.clone();
    // Routes `verify_fold` to the multi-group-root trace path; inert for the dense
    // trace-weight table that multi-group roots evaluate.
    let trace_block_opening = root_trace_block_opening::<E>(
        shared_opening_point,
        root_lp.num_positions_per_block,
        root_lp.num_live_blocks,
        alpha_bits,
    )?;

    let group_ring_opening_points = prepared_points
        .iter()
        .map(|prepared| prepared.ring_opening_point.clone())
        .collect::<Vec<_>>();
    let group_ring_multiplier_points = prepared_points
        .iter()
        .map(|prepared| prepared.ring_multiplier_point.clone())
        .collect::<Vec<_>>();

    let next_witness_ring_dim = next_fold_level_params.role_dims().d_a();
    if !w_len.is_multiple_of(next_witness_ring_dim) {
        return Err(AkitaError::InvalidProof);
    }
    let prepared = PreparedFoldReplay {
        lp: root_lp,
        relation_matrix_row_layout,
        fold_grind_nonce,
        v: v_storage,
        opening_shape: opening_batch.clone(),
        commitment_rows,
        row_coefficients,
        group_ring_opening_points,
        group_ring_multiplier_points,
        w_len,
        payload: PreparedFoldPayload::Recursive {
            stage1: &proof.stage1,
            stage2: &proof.stage2,
            next_witness,
            next_witness_ring_dim,
            next_opening_source_len: w_len / next_witness_ring_dim,
            stage3: stage3_sumcheck_proof.map(|proof| (proof, next_fold_level_params)),
        },
        trace: TracePreparation {
            prepared_points: Some(prepared_points),
            block_opening: Some(trace_block_opening),
            eval_target: trace_eval_target,
            eval_scale: E::one(),
            claim_scales: None,
            basis,
        },
    };
    verify_fold::<F, E, T>(setup, transcript, prepared)
}
