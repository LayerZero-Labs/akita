//! Root and recursive level verifier replay for Akita proofs.
//!
//! This module owns the transcript and algebra checks for an already selected
//! root or fold level. Schedule/config dispatch stays with the scheme crate
//! until the verifier-facing config boundary is extracted.

use crate::{
    derive_stage1_challenges, ring_switch_verifier, verify_stage2_with_setup_claim_reduction,
    AkitaStage1Verifier, AkitaStage2Verifier, Stage2MEvalSource,
};
use akita_algebra::ring::trace;
use akita_algebra::CyclotomicRing;
use akita_field::{AkitaError, CanonicalField, FieldCore, RandomSampling};
use akita_sumcheck::{verify_sumcheck, SumcheckInstanceVerifier};
use akita_transcript::labels::{
    ABSORB_COMMITMENT, ABSORB_EVALUATION_CLAIMS, ABSORB_EVAL_OPENINGS_FIELD,
    ABSORB_SUMCHECK_S_CLAIM, CHALLENGE_EVAL_BATCH, CHALLENGE_SUMCHECK_BATCH,
    CHALLENGE_SUMCHECK_ROUND,
};
use akita_transcript::Transcript;
use akita_types::{
    append_batch_shape_to_transcript, append_batched_commitments_to_transcript,
    checked_total_claims, flatten_batched_commitment_rows, prepare_root_opening_point,
    reduce_inner_opening_to_ring_element, relation_claim_from_rows, reorder_stage1_coords,
    ring_opening_point_from_field, schedule_num_fold_levels, w_ring_element_count,
    w_ring_element_count_with_claim_groups, AkitaBatchedProof, AkitaLevelProof, AkitaProofStep,
    AkitaStage1Proof, AkitaStage2Proof, AkitaVerifierSetup, BasisMode, BlockOrder,
    DirectWitnessProof, FlatRingVec, LevelParams, MultiPointBatchShape, PreparedRootOpeningPoint,
    RecursiveOpeningClaim, RingCommitment, RingOpeningPoint, Schedule, Step,
};

/// Verifier state carried between recursive fold levels.
///
/// Each entry of `claims` is one polynomial opening that the next fold
/// level must discharge. The single-poly recursive path uses
/// `claims.len() == 1`; Phase D-full slice F adds an additional claim
/// for the shared setup polynomial `S` so the next level discharges
/// the deferred `S(r_setup) = y_setup` claim alongside the folded
/// witness via multi-claim batched Hachi.
pub struct RecursiveVerifierState<'a, F: FieldCore> {
    /// Recursive opening claims to discharge at the next fold level.
    pub claims: Vec<RecursiveOpeningClaim<'a, F>>,
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
    stage1: &AkitaStage1Proof<F>,
    stage2: &AkitaStage2Proof<F>,
    setup: &AkitaVerifierSetup<F>,
    transcript: &mut T,
    prepared_points: &[PreparedRootOpeningPoint<F, D>],
    openings: &[F],
    commitments: &[RingCommitment<F, D>],
    batch_shape: &MultiPointBatchShape,
    root_lp: &LevelParams,
    batched_lp: &LevelParams,
    is_last: bool,
    final_w: Option<&DirectWitnessProof<F>>,
) -> Result<Vec<F>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    T: Transcript<F>,
{
    let y_rings = y_rings_flat.as_ring_slice::<D>()?;
    let v_typed = v_flat.as_ring_slice::<D>()?;
    let num_claims = checked_total_claims(&batch_shape.claim_group_sizes, "batched_verify")
        .map_err(|_| AkitaError::InvalidProof)?;
    let num_points = prepared_points.len();
    if num_points == 0
        || y_rings.len() != num_points
        || openings.len() != num_claims
        || commitments.len() != batch_shape.claim_group_sizes.len()
        || batch_shape.claim_to_point.len() != num_claims
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
            return Err(AkitaError::InvalidProof);
        }
    }

    let stage1_challenges = derive_stage1_challenges::<F, T, D>(
        transcript,
        v_typed,
        root_lp.num_blocks,
        num_claims,
        batched_lp,
    )?;

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
    let stage1_verifier = AkitaStage1Verifier::new(tau0_reordered, rs.b);
    let r_stage1 = {
        let _sumcheck_span = tracing::info_span!("stage1_sumcheck").entered();
        stage1_verifier.verify(stage1, transcript)?
    };
    transcript.append_serde(ABSORB_SUMCHECK_S_CLAIM, &stage1.s_claim);
    let batching_coeff: F = transcript.challenge_scalar(CHALLENGE_SUMCHECK_BATCH);
    let stage2_input_claim = batching_coeff * stage1.s_claim + relation_claim;
    let m_eval_source = Stage2MEvalSource::new(rs.prepared_m_eval);
    let stage2_verifier = if is_last {
        let fw = final_w.ok_or(AkitaError::InvalidProof)?;
        AkitaStage2Verifier::new_with_direct_witness(
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
        AkitaStage2Verifier::new_with_claimed_w_eval(
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
        return Err(AkitaError::InvalidProof);
    }
    let sumcheck_challenges = {
        let _sumcheck_span = tracing::info_span!("stage2_sumcheck").entered();
        if let Some(payload) = stage2.setup_claim_reduction.as_ref() {
            let (stage2_challenges, _r_setup, _s_opening_value) =
                verify_stage2_with_setup_claim_reduction::<F, _, D>(
                    &stage2.sumcheck,
                    payload,
                    &stage2_verifier,
                    transcript,
                )?;
            // Phase D-full v2 seam: `(_r_setup, _s_opening_value)` will
            // be threaded into the next-level state via the split-
            // commitment recursive opening (book §5.3 lines 627-660,
            // implemented in slices D-G of
            // `specs/phase-d-full-design.md`). Today the transitional
            // `mle` check inside `verify_setup_claim_reduction` keeps
            // the protocol sound; the deferred claim is otherwise
            // dropped on the floor.
            stage2_challenges
        } else {
            verify_sumcheck::<F, _, F, _, _>(
                &stage2.sumcheck,
                &stage2_verifier,
                transcript,
                |tr| tr.challenge_scalar(CHALLENGE_SUMCHECK_ROUND),
            )?
        }
    };

    Ok(sumcheck_challenges)
}

/// Verify one recursive fold level.
///
/// Drives multi-claim verification: `current_state.claims` may carry one
/// or more recursive opening claims, and `level_proof.y_ring` is decoded
/// as a per-point flat ring vector aligned to those claims under the
/// inference rule "one claim per opening point, one commitment per
/// claim".
///
/// At the final level, `final_w` is provided and the verifier checks
/// `w_val` from it directly. At intermediate levels,
/// `level_proof.next_w_eval()` is used. The returned challenges become
/// the opening point for the next level.
///
/// # Errors
///
/// Returns an error if the level proof shape is inconsistent, the public trace
/// check fails, ring-switch replay fails, or either sumcheck verifier rejects.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
#[tracing::instrument(skip_all, name = "verify_one_level")]
pub fn verify_one_level<F, T, const D: usize>(
    level_proof: &AkitaLevelProof<F>,
    setup: &AkitaVerifierSetup<F>,
    transcript: &mut T,
    current_state: &RecursiveVerifierState<'_, F>,
    is_last: bool,
    final_w: Option<&DirectWitnessProof<F>>,
    lp: &LevelParams,
    block_order: BlockOrder,
) -> Result<Vec<F>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    T: Transcript<F>,
{
    let claims = current_state.claims.as_slice();
    let num_claims = claims.len();
    if num_claims == 0 {
        return Err(AkitaError::InvalidProof);
    }
    let num_eval_rows = num_claims;
    let claim_to_point: Vec<usize> = (0..num_claims).collect();
    let claim_group_sizes: Vec<usize> = vec![1usize; num_claims];

    // Slice E shape check (mirror of the prover side): each per-claim
    // LP override must agree with the shared `lp` until slice F lifts
    // the homogeneous restriction alongside the multi-group commit
    // kernel hookup and the heterogeneous prepare_m_eval / stage-2
    // extensions.
    for (i, claim) in claims.iter().enumerate() {
        if let Some(claim_lp) = &claim.per_claim_lp {
            if claim_lp != lp {
                return Err(AkitaError::InvalidSetup(format!(
                    "verify_one_level: per-claim LP override at index {i} disagrees \
                     with the shared level params; heterogeneous per-claim LP support \
                     lands in slice F"
                )));
            }
        }
    }

    let y_rings = level_proof.y_ring.as_ring_slice::<D>()?;
    if y_rings.len() != num_eval_rows {
        return Err(AkitaError::InvalidProof);
    }
    let v_typed = level_proof.v.as_ring_slice::<D>()?;

    let alpha_bits = lp.ring_dimension.trailing_zeros() as usize;
    let target_num_vars = lp.m_vars + lp.r_vars + alpha_bits;

    let mut padded_points: Vec<Vec<F>> = Vec::with_capacity(num_claims);
    let mut inner_reductions: Vec<CyclotomicRing<F, D>> = Vec::with_capacity(num_claims);
    let mut ring_opening_points: Vec<akita_types::RingOpeningPoint<F>> =
        Vec::with_capacity(num_claims);
    for claim in claims.iter() {
        if claim.opening_point.len() < alpha_bits {
            return Err(AkitaError::InvalidSetup(
                "opening point length underflow".to_string(),
            ));
        }
        let mut padded_point = claim.opening_point.clone();
        padded_point.resize(target_num_vars, F::zero());
        let inner_point = &padded_point[..alpha_bits];
        let reduced_opening_point = &padded_point[alpha_bits..];

        let inner_reduction =
            reduce_inner_opening_to_ring_element::<F, { D }>(inner_point, claim.basis)?;
        let ring_opening_point = ring_opening_point_from_field::<F>(
            reduced_opening_point,
            lp.r_vars,
            lp.m_vars,
            claim.basis,
            block_order,
        )?;
        padded_points.push(padded_point);
        inner_reductions.push(inner_reduction);
        ring_opening_points.push(ring_opening_point);
    }

    // Transcript layout. For N == 1 we keep today's recursive wire shape
    // (one commitment + padded point + y-ring, no `gamma`). For N > 1 we
    // mirror the root multi-claim layout: append all commitments and
    // padded points, then openings, sample `gamma`, then append the per-
    // point `gamma`-combined y-rings.
    for claim in claims.iter() {
        claim
            .commitment
            .append_as_ring_slice::<T, D>(ABSORB_COMMITMENT, transcript)?;
    }
    for padded_point in &padded_points {
        for pt in padded_point {
            transcript.append_field(ABSORB_EVALUATION_CLAIMS, pt);
        }
    }
    let gamma: Vec<F> = if num_claims > 1 {
        for claim in claims.iter() {
            transcript.append_field(ABSORB_EVAL_OPENINGS_FIELD, &claim.opening);
        }
        (0..num_claims)
            .map(|_| transcript.challenge_scalar(CHALLENGE_EVAL_BATCH))
            .collect()
    } else {
        vec![F::one()]
    };
    for y_ring in y_rings {
        transcript.append_serde(ABSORB_EVALUATION_CLAIMS, y_ring);
    }

    // Per-point trace check: the wire `y_rings` carry the per-point
    // `gamma`-combination of each claim landing on that point. With the
    // 1-claim-per-point inference rule each point has exactly one
    // contribution, so the per-point batched opening is just `gamma[i]
    // * claim[i].opening`.
    let d_field = F::from_u64(lp.ring_dimension as u64);
    let mut batched_openings_per_point = vec![F::zero(); num_eval_rows];
    for (claim_idx, (claim, &g)) in claims.iter().zip(gamma.iter()).enumerate() {
        let point_idx = claim_to_point[claim_idx];
        batched_openings_per_point[point_idx] += g * claim.opening;
    }
    for (point_idx, (y_ring, &batched_opening)) in y_rings
        .iter()
        .zip(batched_openings_per_point.iter())
        .enumerate()
    {
        let inner_reduction = &inner_reductions[point_idx];
        let trace_lhs = trace::<F, { D }>(&(*y_ring * inner_reduction.sigma_m1()));
        let trace_rhs = d_field * batched_opening;
        if trace_lhs != trace_rhs {
            return Err(AkitaError::InvalidProof);
        }
    }

    let stage1_challenges =
        derive_stage1_challenges::<F, T, D>(transcript, v_typed, lp.num_blocks, num_claims, lp)?;

    let w_len = if is_last {
        final_w.map_or(0, DirectWitnessProof::num_elems)
    } else if num_claims == 1 {
        w_ring_element_count::<F>(lp) * D
    } else {
        w_ring_element_count_with_claim_groups::<F>(lp, &claim_group_sizes, num_eval_rows) * D
    };
    tracing::debug!(w_len, is_last, num_claims, "verify ring_switch");

    let commitment_rows_owned: Option<Vec<CyclotomicRing<F, D>>> = if num_claims == 1 {
        None
    } else {
        let mut rows = Vec::with_capacity(num_claims * lp.b_key.row_len());
        for claim in claims.iter() {
            rows.extend_from_slice(claim.commitment.as_ring_slice::<D>()?);
        }
        Some(rows)
    };
    let commitment_u: &[CyclotomicRing<F, D>] = match &commitment_rows_owned {
        Some(rows) => rows.as_slice(),
        None => claims[0].commitment.as_ring_slice::<D>()?,
    };

    let rs = ring_switch_verifier::<F, T, { D }>(
        &ring_opening_points,
        &claim_to_point,
        &stage1_challenges,
        w_len,
        level_proof.next_w_commitment(),
        transcript,
        lp,
        &claim_group_sizes,
        &gamma,
        num_eval_rows,
    )?;
    let relation_claim =
        relation_claim_from_rows(&rs.tau1, rs.alpha, v_typed, commitment_u, y_rings);
    let stage1 = &level_proof.stage1;
    let stage2 = &level_proof.stage2;
    let tau0_reordered = reorder_stage1_coords(&rs.tau0, rs.col_bits, rs.ring_bits);
    let stage1_verifier = AkitaStage1Verifier::new(tau0_reordered, rs.b);
    let r_stage1 = {
        let _sumcheck_span = tracing::info_span!("stage1_sumcheck").entered();
        stage1_verifier.verify(stage1, transcript)?
    };

    transcript.append_serde(ABSORB_SUMCHECK_S_CLAIM, &stage1.s_claim);
    let batching_coeff: F = transcript.challenge_scalar(CHALLENGE_SUMCHECK_BATCH);
    let stage2_input_claim = batching_coeff * stage1.s_claim + relation_claim;
    let m_eval_source = Stage2MEvalSource::new(rs.prepared_m_eval);

    let stage2_verifier = if is_last {
        let fw = final_w.ok_or(AkitaError::InvalidProof)?;
        AkitaStage2Verifier::new_with_direct_witness(
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
            commitment_u,
            y_rings,
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
            m_eval_source,
            &setup.expanded,
            &ring_opening_points,
            &rs.tau1,
            v_typed,
            commitment_u,
            y_rings,
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
        if let Some(payload) = stage2.setup_claim_reduction.as_ref() {
            let (stage2_challenges, _r_setup, _s_opening_value) =
                verify_stage2_with_setup_claim_reduction::<F, _, D>(
                    &stage2.sumcheck,
                    payload,
                    &stage2_verifier,
                    transcript,
                )?;
            // Phase D-full v2 seam: see `verify_root_level` for the
            // routing-slice hand-off comment. The same applies here.
            stage2_challenges
        } else {
            verify_sumcheck::<F, _, F, _, _>(
                &stage2.sumcheck,
                &stage2_verifier,
                transcript,
                |tr| tr.challenge_scalar(CHALLENGE_SUMCHECK_ROUND),
            )?
        }
    };

    Ok(challenges)
}

fn scheduled_recursive_verify_level<F: FieldCore>(
    schedule: &Schedule,
    level: usize,
    current_state: &RecursiveVerifierState<'_, F>,
) -> Result<(LevelParams, usize, Option<LevelParams>), AkitaError> {
    let Some(Step::Fold(step)) = schedule.steps.get(level) else {
        return Err(AkitaError::InvalidSetup(format!(
            "schedule is missing fold step at level {level}"
        )));
    };
    let claim = &current_state.claims[0];
    if step.current_w_len != claim.w_len || step.params.log_basis != claim.log_basis {
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
fn dispatch_verify_level<F, T>(
    level_d: usize,
    level_proof: &AkitaLevelProof<F>,
    setup: &AkitaVerifierSetup<F>,
    transcript: &mut T,
    current_state: &RecursiveVerifierState<'_, F>,
    is_last: bool,
    final_w: Option<&DirectWitnessProof<F>>,
    lp: &LevelParams,
    block_order: BlockOrder,
) -> Result<Vec<F>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
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
pub fn verify_batched_recursive_suffix<'a, F, T, const D: usize>(
    proof: &'a AkitaBatchedProof<F>,
    setup: &AkitaVerifierSetup<F>,
    transcript: &mut T,
    schedule: &Schedule,
    mut current_state: RecursiveVerifierState<'a, F>,
    final_w: Option<&DirectWitnessProof<F>>,
) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    T: Transcript<F>,
{
    let num_levels = proof.num_fold_levels();
    for (offset, level_proof) in proof.fold_levels().enumerate() {
        let level_index = offset + 1;
        let is_last = offset == num_levels - 1;
        let (current_lp, next_w_len, scheduled_next_params) =
            scheduled_recursive_verify_level(schedule, level_index, &current_state)?;
        let level_d = current_lp.ring_dimension;
        // Multi-ring shape check: the level proof's y_ring carries one
        // ring element per opening point at this level. With the
        // 1-claim-per-point inference rule, that count equals the
        // recursive state's claim count.
        let expected_num_y_rings = current_state.claims.len();
        if !current_state
            .claims
            .iter()
            .all(|claim| claim.commitment.can_decode_vec(level_d))
            || !level_proof
                .y_ring
                .can_decode_count(level_d, expected_num_y_rings)
            || !level_proof.v.can_decode_vec(level_d)
        {
            return Err(AkitaError::InvalidProof);
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
            let scheduled_next_params = scheduled_next_params.ok_or(AkitaError::InvalidProof)?;
            let next_level_d = scheduled_next_params.ring_dimension;
            if next_level_d == 0 || !level_proof.next_w_commitment().can_decode_vec(next_level_d) {
                return Err(AkitaError::InvalidProof);
            }
            // Account for multi-claim w_ring sizing on cascade levels.
            // The runtime witness produced by this level has size
            // `w_ring(current_lp, num_claims_at_level) * level_d`. We
            // already know `num_claims_at_level == current_state.claims.len()`.
            let computed_next_w_len = if expected_num_y_rings == 1 {
                w_ring_element_count::<F>(&current_lp) * level_d
            } else {
                let claim_group_sizes: Vec<usize> = vec![1usize; expected_num_y_rings];
                w_ring_element_count_with_claim_groups::<F>(
                    &current_lp,
                    &claim_group_sizes,
                    expected_num_y_rings,
                ) * level_d
            };
            if computed_next_w_len != next_w_len {
                return Err(AkitaError::InvalidProof);
            }
            current_state = RecursiveVerifierState {
                claims: vec![RecursiveOpeningClaim {
                    opening_point: challenges,
                    opening: level_proof.next_w_eval(),
                    commitment: level_proof.next_w_commitment(),
                    basis: BasisMode::Lagrange,
                    w_len: next_w_len,
                    log_basis: scheduled_next_params.log_basis,
                    per_claim_lp: None,
                }],
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
pub fn verify_fold_batched_proof<F, T, const D: usize>(
    proof: &AkitaBatchedProof<F>,
    setup: &AkitaVerifierSetup<F>,
    transcript: &mut T,
    opening_points: &[&[F]],
    openings: &[F],
    commitments: &[RingCommitment<F, D>],
    batch_shape: &MultiPointBatchShape,
    basis: BasisMode,
    schedule: &Schedule,
    root_lp: &LevelParams,
    next_level_params: &LevelParams,
) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
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
            prepare_root_opening_point::<F, D>(opening_point, basis, root_lp, alpha_bits)
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| AkitaError::InvalidProof)?;

    let has_recursive_levels = proof.num_fold_levels() > 0;
    let root_challenges = verify_root_level::<F, T, D>(
        &fold_root.y_rings,
        &fold_root.v,
        &fold_root.stage1,
        &fold_root.stage2,
        setup,
        transcript,
        &prepared_points,
        openings,
        commitments,
        batch_shape,
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
            claims: vec![RecursiveOpeningClaim {
                opening_point: root_challenges,
                opening: fold_root.stage2.next_w_eval,
                commitment: &fold_root.stage2.next_w_commitment,
                basis: BasisMode::Lagrange,
                w_len: root_step.next_w_len,
                log_basis: next_level_params.log_basis,
                per_claim_lp: None,
            }],
        };
        verify_batched_recursive_suffix::<F, T, D>(
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
