//! Root and recursive level verifier replay for Akita proofs.
//!
//! This module owns the transcript and algebra checks for an already selected
//! root or fold level. Schedule/config dispatch stays with the scheme crate
//! until the verifier-facing config boundary is extracted.

use crate::protocol::ring_switch::ring_switch_verifier;
use crate::stages::stage1::{derive_stage1_challenges, AkitaStage1Verifier};
use crate::stages::stage2::{AkitaStage2Verifier, Stage2RowEvalSource};
use akita_algebra::CyclotomicRing;
use akita_field::{
    AkitaError, CanonicalField, ExtField, FieldCore, FromPrimitiveInt, RandomSampling,
};
use akita_serialization::AkitaSerialize;
use akita_sumcheck::{verify_sumcheck, SumcheckInstanceVerifier};
use akita_transcript::labels::{
    ABSORB_COMMITMENT, ABSORB_EVALUATION_CLAIMS, ABSORB_SUMCHECK_S_CLAIM, CHALLENGE_EVAL_BATCH,
    CHALLENGE_SUMCHECK_BATCH, CHALLENGE_SUMCHECK_ROUND,
};
use akita_transcript::{append_ext_field, sample_ext_challenge, Transcript};
use akita_types::{
    append_batched_commitments_to_transcript, append_claim_incidence_shape_to_transcript,
    append_claim_points_to_transcript, append_claim_values_to_transcript,
    dispatch_trace_inner_product_check, flatten_batched_commitment_rows,
    prepare_recursive_opening_point_ext, prepare_root_opening_point_ext,
    relation_claim_from_batched_root_rows_extension, relation_claim_from_rows_extension,
    reorder_stage1_coords, schedule_num_fold_levels, w_ring_element_count,
    w_ring_element_count_with_counts, AkitaBatchedProof, AkitaLevelProof, AkitaProofStep,
    AkitaStage1Proof, AkitaStage2Proof, AkitaVerifierSetup, BasisMode, BlockOrder,
    ClaimIncidenceSummary, DirectWitnessProof, FlatRingVec, HachiSubfieldEncoding, LevelParams,
    PreparedRootOpeningPoint, RingCommitment, RingOpeningPoint, Schedule, Step,
};

/// Verifier state carried between recursive fold levels.
pub(crate) struct RecursiveVerifierState<'a, F: FieldCore, L: FieldCore> {
    /// Current opening point for the committed recursive witness.
    pub opening_point: Vec<L>,
    /// Claimed opening value for the current commitment.
    pub opening: L,
    /// Current recursive witness commitment.
    pub commitment: &'a FlatRingVec<F>,
    /// Basis used to interpret the current opening point.
    pub basis: BasisMode,
    /// Current recursive witness length in field elements.
    pub w_len: usize,
    /// Current digit basis, as `log2(b)`.
    pub log_basis: u32,
}

/// Verify the root proof payload for singleton and multi-point batched proofs.
///
/// This replays the canonical root transcript layout: batch-shape header,
/// commitments, padded opening points, per-claim field openings, one gamma
/// challenge per claim, and gamma-combined per-point y-rings.
///
/// # Errors
///
/// Returns an error if the proof shape is inconsistent, any public trace check
/// fails, ring-switch replay fails, or either sumcheck verifier rejects.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub(crate) fn verify_root_level<F, E, C, T, const D: usize>(
    y_rings_flat: &FlatRingVec<F>,
    v_flat: &FlatRingVec<F>,
    stage1: &AkitaStage1Proof<C>,
    stage2: &AkitaStage2Proof<F, C>,
    setup: &AkitaVerifierSetup<F>,
    transcript: &mut T,
    claim_points: &[&[E]],
    prepared_points: &[PreparedRootOpeningPoint<F, D>],
    openings: &[E],
    commitments: &[RingCommitment<F, D>],
    incidence_summary: &ClaimIncidenceSummary,
    root_lp: &LevelParams,
    batched_lp: &LevelParams,
    is_last: bool,
    final_w: Option<&DirectWitnessProof<F>>,
) -> Result<Vec<C>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    E: ExtField<F>,
    C: HachiSubfieldEncoding<F> + ExtField<E> + FromPrimitiveInt + AkitaSerialize,
    T: Transcript<F>,
{
    let y_rings = y_rings_flat.as_ring_slice::<D>()?;
    let v_typed = v_flat.as_ring_slice::<D>()?;
    let num_claims = incidence_summary.num_claims;
    let num_points = prepared_points.len();
    if num_points == 0
        || num_points != incidence_summary.num_points
        || claim_points.len() != incidence_summary.num_points
        || y_rings.len() != num_points
        || openings.len() != num_claims
        || commitments.len() != incidence_summary.num_groups
        || incidence_summary.claim_to_point.len() != num_claims
    {
        return Err(AkitaError::InvalidProof);
    }
    if incidence_summary
        .claim_to_point
        .iter()
        .any(|&point_idx| point_idx >= num_points)
    {
        return Err(AkitaError::InvalidProof);
    }
    if commitments
        .iter()
        .any(|commitment| commitment.u.len() != root_lp.b_key.row_len())
    {
        return Err(AkitaError::InvalidProof);
    }
    // Mirror the prover's commitment-rows optimization: avoid a clone when
    // there is only a single commitment.
    let commitment_rows_owned: Option<Vec<CyclotomicRing<F, D>>> = if commitments.len() == 1 {
        None
    } else {
        Some(flatten_batched_commitment_rows(commitments))
    };
    let commitment_rows: &[CyclotomicRing<F, D>] = match &commitment_rows_owned {
        Some(v) => v.as_slice(),
        None => commitments[0].u.as_slice(),
    };

    append_claim_incidence_shape_to_transcript::<F, T>(incidence_summary, transcript);
    append_batched_commitments_to_transcript(commitments, transcript);
    append_claim_points_to_transcript::<F, E, T>(claim_points, transcript);
    append_claim_values_to_transcript::<F, E, T>(openings, transcript);
    let gamma: Vec<C> = if openings.len() == 1 {
        vec![C::one()]
    } else {
        (0..openings.len())
            .map(|_| sample_ext_challenge::<F, C, T>(transcript, CHALLENGE_EVAL_BATCH))
            .collect()
    };
    for y_ring in y_rings {
        transcript.append_serde(ABSORB_EVALUATION_CLAIMS, y_ring);
    }

    // Per-point trace check: for each opening point `j`, verify the Hachi
    // trace inner-product identity
    // `trace_h(y_j · σ_{-1}(v_j)) == (D / K) · embed_subfield(opening_j)` in
    // R_q, where `opening_j = Σ_{ι: point(ι)=j} γ_ι · opening_ι` is the
    // batched per-point opening, computed in `Cfg::ClaimField` directly. Each
    // opening point carries its own inner reduction `v_j`, which may differ
    // across the batch. K is dispatched at runtime from the claim-field
    // extension degree.
    let mut batched_openings_per_point: Vec<C> = vec![C::zero(); num_points];
    if <C as ExtField<F>>::EXT_DEGREE == 1 {
        for (claim_idx, (opening, &g)) in openings.iter().zip(gamma.iter()).enumerate() {
            let point_idx = incidence_summary.claim_to_point[claim_idx];
            batched_openings_per_point[point_idx] += g * C::lift_base(*opening);
        }
    } else {
        let mut seen_point = vec![false; num_points];
        for (claim_idx, opening) in openings.iter().enumerate() {
            let point_idx = incidence_summary.claim_to_point[claim_idx];
            if seen_point[point_idx] {
                return Err(AkitaError::InvalidProof);
            }
            seen_point[point_idx] = true;
            batched_openings_per_point[point_idx] = C::lift_base(*opening);
        }
    }
    for (point_idx, (y_ring, batched_opening)) in y_rings
        .iter()
        .zip(batched_openings_per_point.iter())
        .enumerate()
    {
        let v = &prepared_points[point_idx].inner_reduction;
        let trace_input = *y_ring * v.sigma_m1();
        let coords = batched_opening.to_hachi_subfield_coords();
        if !dispatch_trace_inner_product_check::<F, { D }>(
            &trace_input,
            &coords,
            AkitaError::InvalidProof,
        )? {
            return Err(AkitaError::InvalidProof);
        }
    }

    let total_blocks = root_lp
        .num_blocks
        .checked_mul(num_claims)
        .ok_or_else(|| AkitaError::InvalidSetup("batched root block count overflow".to_string()))?;
    let stage1_challenges =
        derive_stage1_challenges::<F, T, D>(transcript, v_typed, total_blocks, batched_lp)?;

    let w_len = if is_last {
        final_w.map_or(0, DirectWitnessProof::num_elems)
    } else {
        w_ring_element_count_with_counts::<F>(
            batched_lp,
            num_claims,
            incidence_summary.group_poly_counts.len(),
            num_points,
        ) * D
    };

    let ring_opening_points: Vec<RingOpeningPoint<F>> = prepared_points
        .iter()
        .map(|prepared_point| prepared_point.ring_opening_point.clone())
        .collect();
    let rs = ring_switch_verifier::<F, C, T, { D }>(
        &ring_opening_points,
        &incidence_summary.claim_to_point,
        &stage1_challenges,
        w_len,
        &stage2.next_w_commitment,
        transcript,
        batched_lp,
        &incidence_summary.group_poly_counts,
        &incidence_summary.claim_to_group,
        &incidence_summary.claim_poly_indices,
        &gamma,
        num_points,
    )?;
    let relation_claim = if <C as ExtField<F>>::EXT_DEGREE == 1 {
        relation_claim_from_rows_extension::<F, C, D>(
            &rs.tau1,
            rs.alpha,
            v_typed,
            commitment_rows,
            y_rings,
        )
    } else {
        relation_claim_from_batched_root_rows_extension::<F, C, D>(
            &rs.tau1,
            rs.alpha,
            v_typed,
            commitment_rows,
            y_rings,
            &incidence_summary.claim_to_point,
            &gamma,
        )
    };
    let tau0_reordered = reorder_stage1_coords(&rs.tau0, rs.col_bits, rs.ring_bits);
    let stage1_verifier = AkitaStage1Verifier::new(tau0_reordered, rs.b);
    let r_stage1 = {
        let _sumcheck_span = tracing::info_span!("stage1_sumcheck").entered();
        stage1_verifier.verify::<F, T>(stage1, transcript)?
    };
    transcript.append_serde(ABSORB_SUMCHECK_S_CLAIM, &stage1.s_claim);
    let batching_coeff: C = sample_ext_challenge::<F, C, T>(transcript, CHALLENGE_SUMCHECK_BATCH);
    let stage2_input_claim = batching_coeff * stage1.s_claim + relation_claim;
    let row_eval_source = Stage2RowEvalSource::new(rs.prepared_row_eval);
    let stage2_verifier = if is_last {
        let fw = final_w.ok_or(AkitaError::InvalidProof)?;
        AkitaStage2Verifier::new_with_direct_witness(
            batching_coeff,
            stage1.s_claim,
            fw,
            r_stage1.clone(),
            rs.alpha_evals_y,
            row_eval_source,
            &setup.expanded,
            &ring_opening_points,
            &rs.tau1,
            v_typed,
            commitment_rows,
            y_rings,
            Some(relation_claim),
            rs.alpha,
            rs.col_bits,
            rs.ring_bits,
        )
    } else {
        AkitaStage2Verifier::new_with_claimed_w_eval(
            batching_coeff,
            stage1.s_claim,
            stage2.next_w_eval,
            r_stage1.clone(),
            rs.alpha_evals_y,
            row_eval_source,
            &setup.expanded,
            &ring_opening_points,
            &rs.tau1,
            v_typed,
            commitment_rows,
            y_rings,
            Some(relation_claim),
            rs.alpha,
            rs.col_bits,
            rs.ring_bits,
        )
    };
    if stage2_input_claim != SumcheckInstanceVerifier::input_claim(&stage2_verifier) {
        return Err(AkitaError::InvalidProof);
    }
    let sumcheck_challenges = {
        let _sumcheck_span = tracing::info_span!("stage2_sumcheck").entered();
        verify_sumcheck::<F, _, C, _, _>(&stage2.sumcheck, &stage2_verifier, transcript, |tr| {
            sample_ext_challenge::<F, C, T>(tr, CHALLENGE_SUMCHECK_ROUND)
        })?
    };

    Ok(sumcheck_challenges)
}

/// Verify one recursive fold level.
///
/// At the final level, `final_w` is provided and the verifier checks `w_val`
/// from it directly. At intermediate levels, `level_proof.next_w_eval()` is
/// used. The returned challenges become the opening point for the next level.
///
/// # Errors
///
/// Returns an error if the level proof shape is inconsistent, the public trace
/// check fails, ring-switch replay fails, or either sumcheck verifier rejects.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
#[tracing::instrument(skip_all, name = "verify_one_level")]
pub(crate) fn verify_one_level<F, L, T, const D: usize>(
    level_proof: &AkitaLevelProof<F, L>,
    setup: &AkitaVerifierSetup<F>,
    transcript: &mut T,
    current_state: &RecursiveVerifierState<'_, F, L>,
    is_last: bool,
    final_w: Option<&DirectWitnessProof<F>>,
    lp: &LevelParams,
    block_order: BlockOrder,
) -> Result<Vec<L>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    L: HachiSubfieldEncoding<F> + FromPrimitiveInt + AkitaSerialize,
    T: Transcript<F>,
{
    let y_ring = level_proof.y_ring.as_single_ring::<D>()?;
    let v_typed = level_proof.v.as_ring_slice::<D>()?;
    let commitment_u = current_state.commitment.as_ring_slice::<D>()?;

    let alpha_bits = lp.ring_dimension.trailing_zeros() as usize;
    let prepared_point = prepare_recursive_opening_point_ext::<F, L, D>(
        &current_state.opening_point,
        current_state.basis,
        lp,
        alpha_bits,
        block_order,
    )?;

    current_state
        .commitment
        .append_as_ring_slice::<T, D>(ABSORB_COMMITMENT, transcript)?;
    for pt in &prepared_point.padded_point {
        append_ext_field::<F, L, T>(transcript, ABSORB_EVALUATION_CLAIMS, pt);
    }
    transcript.append_serde(ABSORB_EVALUATION_CLAIMS, y_ring);

    let trace_input = *y_ring * prepared_point.inner_reduction.sigma_m1();
    let opening_coords = current_state.opening.to_hachi_subfield_coords();
    if !dispatch_trace_inner_product_check::<F, { D }>(
        &trace_input,
        &opening_coords,
        AkitaError::InvalidProof,
    )? {
        return Err(AkitaError::InvalidProof);
    }

    let ring_opening_point = prepared_point.ring_opening_point;
    let stage1_challenges =
        derive_stage1_challenges::<F, T, D>(transcript, v_typed, lp.num_blocks, lp)?;

    let w_len = if is_last {
        final_w.map_or(0, DirectWitnessProof::num_elems)
    } else {
        w_ring_element_count::<F>(lp) * D
    };
    tracing::debug!(w_len, is_last, "verify ring_switch");

    let rs = ring_switch_verifier::<F, L, T, { D }>(
        std::slice::from_ref(&ring_opening_point),
        &[0usize],
        &stage1_challenges,
        w_len,
        level_proof.next_w_commitment(),
        transcript,
        lp,
        &[1usize],
        &[0usize],
        &[0usize],
        &[L::one()],
        1,
    )?;
    let relation_claim = relation_claim_from_rows_extension::<F, L, D>(
        &rs.tau1,
        rs.alpha,
        v_typed,
        commitment_u,
        std::slice::from_ref(y_ring),
    );
    let stage1 = &level_proof.stage1;
    let stage2 = &level_proof.stage2;
    let tau0_reordered = reorder_stage1_coords(&rs.tau0, rs.col_bits, rs.ring_bits);
    let stage1_verifier = AkitaStage1Verifier::new(tau0_reordered, rs.b);
    let r_stage1 = {
        let _sumcheck_span = tracing::info_span!("stage1_sumcheck").entered();
        stage1_verifier.verify::<F, T>(stage1, transcript)?
    };

    transcript.append_serde(ABSORB_SUMCHECK_S_CLAIM, &stage1.s_claim);
    let batching_coeff: L = sample_ext_challenge::<F, L, T>(transcript, CHALLENGE_SUMCHECK_BATCH);
    let stage2_input_claim = batching_coeff * stage1.s_claim + relation_claim;
    let row_eval_source = Stage2RowEvalSource::new(rs.prepared_row_eval);
    let ring_opening_points_slice = std::slice::from_ref(&ring_opening_point);

    let y_rings_slice = std::slice::from_ref(y_ring);
    let stage2_verifier = if is_last {
        let fw = final_w.ok_or(AkitaError::InvalidProof)?;
        AkitaStage2Verifier::new_with_direct_witness(
            batching_coeff,
            stage1.s_claim,
            fw,
            r_stage1.clone(),
            rs.alpha_evals_y,
            row_eval_source,
            &setup.expanded,
            ring_opening_points_slice,
            &rs.tau1,
            v_typed,
            commitment_u,
            y_rings_slice,
            Some(relation_claim),
            rs.alpha,
            rs.col_bits,
            rs.ring_bits,
        )
    } else {
        AkitaStage2Verifier::new_with_claimed_w_eval(
            batching_coeff,
            stage1.s_claim,
            stage2.next_w_eval,
            r_stage1.clone(),
            rs.alpha_evals_y,
            row_eval_source,
            &setup.expanded,
            ring_opening_points_slice,
            &rs.tau1,
            v_typed,
            commitment_u,
            y_rings_slice,
            Some(relation_claim),
            rs.alpha,
            rs.col_bits,
            rs.ring_bits,
        )
    };
    if stage2_input_claim != SumcheckInstanceVerifier::input_claim(&stage2_verifier) {
        return Err(AkitaError::InvalidProof);
    }

    let challenges = {
        let _sumcheck_span = tracing::info_span!("stage2_sumcheck").entered();
        verify_sumcheck::<F, _, L, _, _>(&stage2.sumcheck, &stage2_verifier, transcript, |tr| {
            sample_ext_challenge::<F, L, T>(tr, CHALLENGE_SUMCHECK_ROUND)
        })?
    };

    Ok(challenges)
}

fn scheduled_recursive_verify_level<F: FieldCore, L: FieldCore>(
    schedule: &Schedule,
    level: usize,
    current_state: &RecursiveVerifierState<'_, F, L>,
) -> Result<(LevelParams, usize, Option<LevelParams>), AkitaError> {
    let Some(Step::Fold(step)) = schedule.steps.get(level) else {
        return Err(AkitaError::InvalidSetup(format!(
            "schedule is missing fold step at level {level}"
        )));
    };
    if step.current_w_len != current_state.w_len || step.params.log_basis != current_state.log_basis
    {
        return Err(AkitaError::InvalidSetup(
            "scheduled recursive level did not match runtime state".to_string(),
        ));
    }
    let next_level_params = match schedule.steps.get(level + 1) {
        Some(Step::Fold(next_step)) => Some(next_step.params.clone()),
        Some(Step::Direct(_)) => None,
        None => {
            return Err(AkitaError::InvalidSetup(
                "schedule is missing successor step".to_string(),
            ))
        }
    };
    Ok((step.params.clone(), step.next_w_len, next_level_params))
}

#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn dispatch_verify_level<F, L, T>(
    level_d: usize,
    level_proof: &AkitaLevelProof<F, L>,
    setup: &AkitaVerifierSetup<F>,
    transcript: &mut T,
    current_state: &RecursiveVerifierState<'_, F, L>,
    is_last: bool,
    final_w: Option<&DirectWitnessProof<F>>,
    lp: &LevelParams,
    block_order: BlockOrder,
) -> Result<Vec<L>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    L: HachiSubfieldEncoding<F> + FromPrimitiveInt + AkitaSerialize,
    T: Transcript<F>,
{
    match level_d {
        32 => verify_one_level::<F, L, T, 32>(
            level_proof,
            setup,
            transcript,
            current_state,
            is_last,
            final_w,
            lp,
            block_order,
        ),
        64 => verify_one_level::<F, L, T, 64>(
            level_proof,
            setup,
            transcript,
            current_state,
            is_last,
            final_w,
            lp,
            block_order,
        ),
        128 => verify_one_level::<F, L, T, 128>(
            level_proof,
            setup,
            transcript,
            current_state,
            is_last,
            final_w,
            lp,
            block_order,
        ),
        256 => verify_one_level::<F, L, T, 256>(
            level_proof,
            setup,
            transcript,
            current_state,
            is_last,
            final_w,
            lp,
            block_order,
        ),
        512 => verify_one_level::<F, L, T, 512>(
            level_proof,
            setup,
            transcript,
            current_state,
            is_last,
            final_w,
            lp,
            block_order,
        ),
        1024 => verify_one_level::<F, L, T, 1024>(
            level_proof,
            setup,
            transcript,
            current_state,
            is_last,
            final_w,
            lp,
            block_order,
        ),
        _ => Err(AkitaError::InvalidProof),
    }
}

/// Verify all recursive fold levels after the root proof.
///
/// The supplied `schedule` is the already-selected public schedule for this
/// proof shape. This function checks that each proof level matches that
/// schedule, dispatches to the corresponding ring dimension, and threads the
/// verifier state to the next recursive commitment.
///
/// # Errors
///
/// Returns an error if the schedule is malformed for the supplied proof,
/// decoded proof dimensions do not match, any fold-level verifier rejects, or
/// the recursive witness handoff has the wrong shape.
pub(crate) fn verify_batched_recursive_suffix<'a, F, L, T, const D: usize>(
    proof: &'a AkitaBatchedProof<F, L>,
    setup: &AkitaVerifierSetup<F>,
    transcript: &mut T,
    schedule: &Schedule,
    mut current_state: RecursiveVerifierState<'a, F, L>,
    final_w: Option<&DirectWitnessProof<F>>,
) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    L: HachiSubfieldEncoding<F> + FromPrimitiveInt + AkitaSerialize,
    T: Transcript<F>,
{
    let num_levels = proof.num_fold_levels();
    for (offset, level_proof) in proof.fold_levels().enumerate() {
        let level_index = offset + 1;
        let is_last = offset == num_levels - 1;
        let (current_lp, next_w_len, scheduled_next_params) =
            scheduled_recursive_verify_level(schedule, level_index, &current_state)?;
        let level_d = current_lp.ring_dimension;
        if !current_state.commitment.can_decode_vec(level_d)
            || !level_proof.y_ring.can_decode_single(level_d)
            || !level_proof.v.can_decode_vec(level_d)
        {
            return Err(AkitaError::InvalidProof);
        }

        let challenges = if level_d == D {
            verify_one_level::<F, L, T, D>(
                level_proof,
                setup,
                transcript,
                &current_state,
                is_last,
                if is_last { final_w } else { None },
                &current_lp,
                BlockOrder::ColumnMajor,
            )?
        } else {
            dispatch_verify_level::<F, L, T>(
                level_d,
                level_proof,
                setup,
                transcript,
                &current_state,
                is_last,
                if is_last { final_w } else { None },
                &current_lp,
                BlockOrder::ColumnMajor,
            )?
        };

        if !is_last {
            let scheduled_next_params = scheduled_next_params.ok_or(AkitaError::InvalidProof)?;
            let next_level_d = scheduled_next_params.ring_dimension;
            if next_level_d == 0 || !level_proof.next_w_commitment().can_decode_vec(next_level_d) {
                return Err(AkitaError::InvalidProof);
            }
            let computed_next_w_len = w_ring_element_count::<F>(&current_lp) * level_d;
            if computed_next_w_len != next_w_len {
                return Err(AkitaError::InvalidProof);
            }
            current_state = RecursiveVerifierState {
                opening_point: challenges,
                opening: level_proof.next_w_eval(),
                commitment: level_proof.next_w_commitment(),
                basis: BasisMode::Lagrange,
                w_len: next_w_len,
                log_basis: scheduled_next_params.log_basis,
            };
        }
    }

    Ok(())
}

/// Verify the folded-root branch of a batched opening proof.
///
/// The caller owns config-backed schedule selection and passes the derived
/// root verifier layout plus the first recursive-level params. This function
/// owns the fold-root proof-shape checks, root opening preparation, root
/// transcript replay, and recursive suffix handoff.
///
/// # Errors
///
/// Returns an error if the proof is not a folded-root proof, the schedule does
/// not match the proof shape, the root proof rejects, or a recursive suffix
/// level rejects.
#[allow(clippy::too_many_arguments)]
pub(crate) fn verify_fold_batched_proof<F, E, C, T, const D: usize>(
    proof: &AkitaBatchedProof<F, C>,
    setup: &AkitaVerifierSetup<F>,
    transcript: &mut T,
    opening_points: &[&[E]],
    openings: &[E],
    commitments: &[RingCommitment<F, D>],
    incidence_summary: &ClaimIncidenceSummary,
    basis: BasisMode,
    schedule: &Schedule,
    root_lp: &LevelParams,
    next_level_params: &LevelParams,
) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    E: ExtField<F>,
    C: HachiSubfieldEncoding<F> + ExtField<E> + FromPrimitiveInt + AkitaSerialize,
    T: Transcript<F>,
{
    let Some(Step::Fold(root_step)) = schedule.steps.first() else {
        return Err(AkitaError::InvalidProof);
    };
    let fold_root = proof.root.as_fold().ok_or(AkitaError::InvalidProof)?;
    let expected_recursive_levels = schedule_num_fold_levels(schedule)
        .checked_sub(1)
        .ok_or(AkitaError::InvalidProof)?;
    if proof.num_fold_levels() != expected_recursive_levels {
        return Err(AkitaError::InvalidProof);
    }

    let y_coeff_len = fold_root.y_rings.coeff_len();
    if !y_coeff_len.is_multiple_of(D) {
        return Err(AkitaError::InvalidProof);
    }
    // One public y-ring per distinct opening point.
    if y_coeff_len / D != opening_points.len() {
        return Err(AkitaError::InvalidProof);
    }

    let final_w = proof
        .steps
        .last()
        .and_then(AkitaProofStep::as_direct)
        .ok_or(AkitaError::InvalidProof)?;
    let final_w = Some(final_w);
    let alpha_bits = root_lp.ring_dimension.trailing_zeros() as usize;
    let prepared_points = opening_points
        .iter()
        .map(|opening_point| {
            prepare_root_opening_point_ext::<F, E, C, D>(opening_point, basis, root_lp, alpha_bits)
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| AkitaError::InvalidProof)?;

    let has_recursive_levels = proof.num_fold_levels() > 0;
    let root_challenges = verify_root_level::<F, E, C, T, D>(
        &fold_root.y_rings,
        &fold_root.v,
        &fold_root.stage1,
        &fold_root.stage2,
        setup,
        transcript,
        opening_points,
        &prepared_points,
        openings,
        commitments,
        incidence_summary,
        root_lp,
        &root_step.params,
        !has_recursive_levels,
        if has_recursive_levels { None } else { final_w },
    )?;

    if has_recursive_levels {
        let first_level_d = next_level_params.ring_dimension;
        if !fold_root
            .stage2
            .next_w_commitment
            .can_decode_vec(first_level_d)
        {
            return Err(AkitaError::InvalidProof);
        }

        let current_state = RecursiveVerifierState {
            opening_point: root_challenges,
            opening: fold_root.stage2.next_w_eval,
            commitment: &fold_root.stage2.next_w_commitment,
            basis: BasisMode::Lagrange,
            w_len: root_step.next_w_len,
            log_basis: next_level_params.log_basis,
        };
        verify_batched_recursive_suffix::<F, C, T, D>(
            proof,
            setup,
            transcript,
            schedule,
            current_state,
            final_w,
        )?;
    }

    Ok(())
}
