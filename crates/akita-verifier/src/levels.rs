//! Root and recursive level verifier replay for Akita proofs.
//!
//! This module owns the transcript and algebra checks for an already selected
//! root or fold level. Schedule/config dispatch stays with the scheme crate
//! until the verifier-facing config boundary is extracted.

use crate::{
    derive_stage1_challenges, relation_claim_from_rows, ring_switch_verifier, HachiStage1Verifier,
    HachiStage2Verifier, Stage2MEvalSource,
};
use akita_algebra::ring::trace;
use akita_algebra::CyclotomicRing;
use akita_field::{CanonicalField, FieldCore, FieldSampling, HachiError};
use akita_sumcheck::{verify_sumcheck, SumcheckInstanceVerifier};
use akita_transcript::labels::{
    ABSORB_COMMITMENT, ABSORB_EVALUATION_CLAIMS, ABSORB_EVAL_OPENINGS_FIELD,
    ABSORB_SUMCHECK_S_CLAIM, CHALLENGE_EVAL_BATCH, CHALLENGE_SUMCHECK_BATCH,
    CHALLENGE_SUMCHECK_ROUND,
};
use akita_transcript::Transcript;
use akita_types::{
    append_batch_shape_to_transcript, append_batched_commitments_to_transcript,
    checked_total_claims, flatten_batched_commitment_rows, reduce_inner_opening_to_ring_element,
    reorder_stage1_coords, ring_opening_point_from_field, w_ring_element_count,
    w_ring_element_count_with_claim_groups, BasisMode, BlockOrder, DirectWitnessProof, FlatRingVec,
    HachiBatchedProof, HachiLevelProof, HachiStage1Proof, HachiStage2Proof, HachiVerifierSetup,
    LevelParams, MultiPointBatchShape, PreparedRootOpeningPoint, RingCommitment, RingOpeningPoint,
    Schedule, Step,
};

/// Verifier state carried between recursive fold levels.
pub struct RecursiveVerifierState<'a, F: FieldCore> {
    /// Current opening point for the committed recursive witness.
    pub opening_point: Vec<F>,
    /// Claimed opening value for the current commitment.
    pub opening: F,
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
pub fn verify_root_level<F, T, const D: usize>(
    y_rings_flat: &FlatRingVec<F>,
    v_flat: &FlatRingVec<F>,
    stage1: &HachiStage1Proof<F>,
    stage2: &HachiStage2Proof<F>,
    setup: &HachiVerifierSetup<F>,
    transcript: &mut T,
    prepared_points: &[PreparedRootOpeningPoint<F, D>],
    openings: &[F],
    commitments: &[RingCommitment<F, D>],
    batch_shape: &MultiPointBatchShape,
    root_lp: &LevelParams,
    batched_lp: &LevelParams,
    is_last: bool,
    final_w: Option<&DirectWitnessProof<F>>,
) -> Result<Vec<F>, HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling,
    T: Transcript<F>,
{
    let y_rings = y_rings_flat.as_ring_slice::<D>()?;
    let v_typed = v_flat.as_ring_slice::<D>()?;
    let num_claims = checked_total_claims(&batch_shape.claim_group_sizes, "batched_verify")
        .map_err(|_| HachiError::InvalidProof)?;
    let num_points = prepared_points.len();
    if num_points == 0
        || y_rings.len() != num_points
        || openings.len() != num_claims
        || commitments.len() != batch_shape.claim_group_sizes.len()
        || batch_shape.claim_to_point.len() != num_claims
    {
        return Err(HachiError::InvalidProof);
    }
    if commitments
        .iter()
        .any(|commitment| commitment.u.len() != root_lp.b_key.row_len())
    {
        return Err(HachiError::InvalidProof);
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

    append_batch_shape_to_transcript::<F, T>(
        &batch_shape.point_group_sizes,
        &batch_shape.claim_group_sizes,
        transcript,
    );
    append_batched_commitments_to_transcript(commitments, transcript);
    for prepared_point in prepared_points {
        for pt in &prepared_point.padded_point {
            transcript.append_field(ABSORB_EVALUATION_CLAIMS, pt);
        }
    }
    for opening in openings {
        transcript.append_field(ABSORB_EVAL_OPENINGS_FIELD, opening);
    }
    let gamma: Vec<F> = (0..openings.len())
        .map(|_| transcript.challenge_scalar(CHALLENGE_EVAL_BATCH))
        .collect();
    for y_ring in y_rings {
        transcript.append_serde(ABSORB_EVALUATION_CLAIMS, y_ring);
    }

    // Per-point trace check: for each opening point `j`, verify
    // `trace(y_j · σ_{-1}(v_j)) = d · Σ_{ι: point(ι)=j} γ_ι · opening_ι`.
    // Each opening point carries its own inner reduction `v_j`, which may
    // differ across the batch.
    let d_field = F::from_u64(root_lp.ring_dimension as u64);
    let mut batched_openings_per_point = vec![F::zero(); num_points];
    for (claim_idx, (&opening, &g)) in openings.iter().zip(gamma.iter()).enumerate() {
        let point_idx = batch_shape.claim_to_point[claim_idx];
        batched_openings_per_point[point_idx] += g * opening;
    }
    for (point_idx, (y_ring, &batched_opening)) in y_rings
        .iter()
        .zip(batched_openings_per_point.iter())
        .enumerate()
    {
        let v = &prepared_points[point_idx].inner_reduction;
        let trace_lhs = trace::<F, { D }>(&(*y_ring * v.sigma_m1()));
        let trace_rhs = d_field * batched_opening;
        if trace_lhs != trace_rhs {
            return Err(HachiError::InvalidProof);
        }
    }

    let total_blocks = root_lp
        .num_blocks
        .checked_mul(num_claims)
        .ok_or_else(|| HachiError::InvalidSetup("batched root block count overflow".to_string()))?;
    let stage1_challenges =
        derive_stage1_challenges::<F, T, D>(transcript, v_typed, total_blocks, batched_lp)?;

    let w_len = if is_last {
        final_w.map_or(0, DirectWitnessProof::num_elems)
    } else {
        w_ring_element_count_with_claim_groups::<F>(
            batched_lp,
            &batch_shape.claim_group_sizes,
            num_points,
        ) * D
    };

    let ring_opening_points: Vec<RingOpeningPoint<F>> = prepared_points
        .iter()
        .map(|prepared_point| prepared_point.ring_opening_point.clone())
        .collect();
    let rs = ring_switch_verifier::<F, T, { D }>(
        &ring_opening_points,
        &batch_shape.claim_to_point,
        &stage1_challenges,
        w_len,
        &stage2.next_w_commitment,
        transcript,
        batched_lp,
        &batch_shape.claim_group_sizes,
        &gamma,
        num_points,
    )?;
    let relation_claim =
        relation_claim_from_rows(&rs.tau1, rs.alpha, v_typed, commitment_rows, y_rings);
    let tau0_reordered = reorder_stage1_coords(&rs.tau0, rs.col_bits, rs.ring_bits);
    let stage1_verifier = HachiStage1Verifier::new(tau0_reordered, rs.b);
    let r_stage1 = {
        let _sumcheck_span = tracing::info_span!("stage1_sumcheck").entered();
        stage1_verifier.verify(stage1, transcript)?
    };
    transcript.append_serde(ABSORB_SUMCHECK_S_CLAIM, &stage1.s_claim);
    let batching_coeff: F = transcript.challenge_scalar(CHALLENGE_SUMCHECK_BATCH);
    let stage2_input_claim = batching_coeff * stage1.s_claim + relation_claim;
    let m_eval_source = Stage2MEvalSource::new(rs.prepared_m_eval);
    let stage2_verifier = if is_last {
        let fw = final_w.ok_or(HachiError::InvalidProof)?;
        HachiStage2Verifier::new_with_direct_witness(
            batching_coeff,
            stage1.s_claim,
            fw,
            r_stage1.clone(),
            rs.alpha_evals_y,
            m_eval_source,
            &setup.expanded,
            &ring_opening_points,
            &rs.tau1,
            v_typed,
            commitment_rows,
            y_rings,
            rs.alpha,
            rs.col_bits,
            rs.ring_bits,
        )
    } else {
        HachiStage2Verifier::new_with_claimed_w_eval(
            batching_coeff,
            stage1.s_claim,
            stage2.next_w_eval,
            r_stage1.clone(),
            rs.alpha_evals_y,
            m_eval_source,
            &setup.expanded,
            &ring_opening_points,
            &rs.tau1,
            v_typed,
            commitment_rows,
            y_rings,
            rs.alpha,
            rs.col_bits,
            rs.ring_bits,
        )
    };
    if stage2_input_claim != SumcheckInstanceVerifier::input_claim(&stage2_verifier) {
        return Err(HachiError::InvalidProof);
    }
    let sumcheck_challenges = {
        let _sumcheck_span = tracing::info_span!("stage2_sumcheck").entered();
        verify_sumcheck::<F, _, F, _, _>(&stage2.sumcheck, &stage2_verifier, transcript, |tr| {
            tr.challenge_scalar(CHALLENGE_SUMCHECK_ROUND)
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
pub fn verify_one_level<F, T, const D: usize>(
    level_proof: &HachiLevelProof<F>,
    setup: &HachiVerifierSetup<F>,
    transcript: &mut T,
    current_state: &RecursiveVerifierState<'_, F>,
    is_last: bool,
    final_w: Option<&DirectWitnessProof<F>>,
    lp: &LevelParams,
    block_order: BlockOrder,
) -> Result<Vec<F>, HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling,
    T: Transcript<F>,
{
    let y_ring = level_proof.y_ring.as_single_ring::<D>()?;
    let v_typed = level_proof.v.as_ring_slice::<D>()?;
    let commitment_u = current_state.commitment.as_ring_slice::<D>()?;

    let alpha_bits = lp.ring_dimension.trailing_zeros() as usize;
    if current_state.opening_point.len() < alpha_bits {
        return Err(HachiError::InvalidSetup(
            "opening point length underflow".to_string(),
        ));
    }
    let target_num_vars = lp.m_vars + lp.r_vars + alpha_bits;
    let mut padded_point = current_state.opening_point.clone();
    padded_point.resize(target_num_vars, F::zero());
    let inner_point = &padded_point[..alpha_bits];
    let reduced_opening_point = &padded_point[alpha_bits..];

    current_state
        .commitment
        .append_as_ring_slice::<T, D>(ABSORB_COMMITMENT, transcript)?;
    for pt in &padded_point {
        transcript.append_field(ABSORB_EVALUATION_CLAIMS, pt);
    }
    transcript.append_serde(ABSORB_EVALUATION_CLAIMS, y_ring);

    let v = reduce_inner_opening_to_ring_element::<F, { D }>(inner_point, current_state.basis)?;
    let d = F::from_u64(lp.ring_dimension as u64);
    let trace_lhs = trace::<F, { D }>(&(*y_ring * v.sigma_m1()));
    let trace_rhs = d * current_state.opening;
    if trace_lhs != trace_rhs {
        return Err(HachiError::InvalidProof);
    }

    let ring_opening_point = ring_opening_point_from_field::<F>(
        reduced_opening_point,
        lp.r_vars,
        lp.m_vars,
        current_state.basis,
        block_order,
    )?;
    let stage1_challenges =
        derive_stage1_challenges::<F, T, D>(transcript, v_typed, lp.num_blocks, lp)?;

    let w_len = if is_last {
        final_w.map_or(0, DirectWitnessProof::num_elems)
    } else {
        w_ring_element_count::<F>(lp) * D
    };
    tracing::debug!(w_len, is_last, "verify ring_switch");

    let rs = ring_switch_verifier::<F, T, { D }>(
        std::slice::from_ref(&ring_opening_point),
        &[0usize],
        &stage1_challenges,
        w_len,
        level_proof.next_w_commitment(),
        transcript,
        lp,
        &[1usize],
        &[F::one()],
        1,
    )?;
    let relation_claim = relation_claim_from_rows(
        &rs.tau1,
        rs.alpha,
        v_typed,
        commitment_u,
        std::slice::from_ref(y_ring),
    );
    let stage1 = &level_proof.stage1;
    let stage2 = &level_proof.stage2;
    let tau0_reordered = reorder_stage1_coords(&rs.tau0, rs.col_bits, rs.ring_bits);
    let stage1_verifier = HachiStage1Verifier::new(tau0_reordered, rs.b);
    let r_stage1 = {
        let _sumcheck_span = tracing::info_span!("stage1_sumcheck").entered();
        stage1_verifier.verify(stage1, transcript)?
    };

    transcript.append_serde(ABSORB_SUMCHECK_S_CLAIM, &stage1.s_claim);
    let batching_coeff: F = transcript.challenge_scalar(CHALLENGE_SUMCHECK_BATCH);
    let stage2_input_claim = batching_coeff * stage1.s_claim + relation_claim;
    let m_eval_source = Stage2MEvalSource::new(rs.prepared_m_eval);
    let ring_opening_points_slice = std::slice::from_ref(&ring_opening_point);

    let y_rings_slice = std::slice::from_ref(y_ring);
    let stage2_verifier = if is_last {
        let fw = final_w.ok_or(HachiError::InvalidProof)?;
        HachiStage2Verifier::new_with_direct_witness(
            batching_coeff,
            stage1.s_claim,
            fw,
            r_stage1.clone(),
            rs.alpha_evals_y,
            m_eval_source,
            &setup.expanded,
            ring_opening_points_slice,
            &rs.tau1,
            v_typed,
            commitment_u,
            y_rings_slice,
            rs.alpha,
            rs.col_bits,
            rs.ring_bits,
        )
    } else {
        HachiStage2Verifier::new_with_claimed_w_eval(
            batching_coeff,
            stage1.s_claim,
            stage2.next_w_eval,
            r_stage1.clone(),
            rs.alpha_evals_y,
            m_eval_source,
            &setup.expanded,
            ring_opening_points_slice,
            &rs.tau1,
            v_typed,
            commitment_u,
            y_rings_slice,
            rs.alpha,
            rs.col_bits,
            rs.ring_bits,
        )
    };
    if stage2_input_claim != SumcheckInstanceVerifier::input_claim(&stage2_verifier) {
        return Err(HachiError::InvalidProof);
    }

    let challenges = {
        let _sumcheck_span = tracing::info_span!("stage2_sumcheck").entered();
        verify_sumcheck::<F, _, F, _, _>(&stage2.sumcheck, &stage2_verifier, transcript, |tr| {
            tr.challenge_scalar(CHALLENGE_SUMCHECK_ROUND)
        })?
    };

    Ok(challenges)
}

fn scheduled_recursive_verify_level<F: FieldCore>(
    schedule: &Schedule,
    level: usize,
    current_state: &RecursiveVerifierState<'_, F>,
) -> Result<(LevelParams, usize, Option<LevelParams>), HachiError> {
    let Some(Step::Fold(step)) = schedule.steps.get(level) else {
        return Err(HachiError::InvalidSetup(format!(
            "schedule is missing fold step at level {level}"
        )));
    };
    if step.current_w_len != current_state.w_len || step.params.log_basis != current_state.log_basis
    {
        return Err(HachiError::InvalidSetup(
            "scheduled recursive level did not match runtime state".to_string(),
        ));
    }
    let next_level_params = match schedule.steps.get(level + 1) {
        Some(Step::Fold(next_step)) => Some(next_step.params.clone()),
        Some(Step::Direct(_)) => None,
        None => {
            return Err(HachiError::InvalidSetup(
                "schedule is missing successor step".to_string(),
            ))
        }
    };
    Ok((step.params.clone(), step.next_w_len, next_level_params))
}

#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn dispatch_verify_level<F, T>(
    level_d: usize,
    level_proof: &HachiLevelProof<F>,
    setup: &HachiVerifierSetup<F>,
    transcript: &mut T,
    current_state: &RecursiveVerifierState<'_, F>,
    is_last: bool,
    final_w: Option<&DirectWitnessProof<F>>,
    lp: &LevelParams,
    block_order: BlockOrder,
) -> Result<Vec<F>, HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling,
    T: Transcript<F>,
{
    match level_d {
        32 => verify_one_level::<F, T, 32>(
            level_proof,
            setup,
            transcript,
            current_state,
            is_last,
            final_w,
            lp,
            block_order,
        ),
        64 => verify_one_level::<F, T, 64>(
            level_proof,
            setup,
            transcript,
            current_state,
            is_last,
            final_w,
            lp,
            block_order,
        ),
        128 => verify_one_level::<F, T, 128>(
            level_proof,
            setup,
            transcript,
            current_state,
            is_last,
            final_w,
            lp,
            block_order,
        ),
        256 => verify_one_level::<F, T, 256>(
            level_proof,
            setup,
            transcript,
            current_state,
            is_last,
            final_w,
            lp,
            block_order,
        ),
        512 => verify_one_level::<F, T, 512>(
            level_proof,
            setup,
            transcript,
            current_state,
            is_last,
            final_w,
            lp,
            block_order,
        ),
        1024 => verify_one_level::<F, T, 1024>(
            level_proof,
            setup,
            transcript,
            current_state,
            is_last,
            final_w,
            lp,
            block_order,
        ),
        _ => Err(HachiError::InvalidProof),
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
pub fn verify_batched_recursive_suffix<'a, F, T, const D: usize>(
    proof: &'a HachiBatchedProof<F>,
    setup: &HachiVerifierSetup<F>,
    transcript: &mut T,
    schedule: &Schedule,
    mut current_state: RecursiveVerifierState<'a, F>,
    final_w: Option<&DirectWitnessProof<F>>,
) -> Result<(), HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling,
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
            return Err(HachiError::InvalidProof);
        }

        let challenges = if level_d == D {
            verify_one_level::<F, T, D>(
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
            dispatch_verify_level::<F, T>(
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
            let scheduled_next_params = scheduled_next_params.ok_or(HachiError::InvalidProof)?;
            let next_level_d = scheduled_next_params.ring_dimension;
            if next_level_d == 0 || !level_proof.next_w_commitment().can_decode_vec(next_level_d) {
                return Err(HachiError::InvalidProof);
            }
            let computed_next_w_len = w_ring_element_count::<F>(&current_lp) * level_d;
            if computed_next_w_len != next_w_len {
                return Err(HachiError::InvalidProof);
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
