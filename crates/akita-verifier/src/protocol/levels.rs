//! Root and recursive level verifier replay for Akita proofs.
//!
//! This module owns the transcript and algebra checks for an already selected
//! root or fold level. Schedule/config dispatch stays with the scheme crate
//! until the verifier-facing config boundary is extracted.

use crate::{
    derive_stage1_challenges, ring_switch_verifier, AkitaStage1Verifier, AkitaStage2Verifier,
    Stage2RowEvalSource,
};
#[cfg(feature = "zk")]
use akita_algebra::eq_poly::EqPolynomial;
use akita_algebra::ring::trace;
use akita_algebra::CyclotomicRing;
use akita_field::{AkitaError, CanonicalField, ExtField, FieldCore, RandomSampling};
#[cfg(feature = "zk")]
use akita_r1cs::{ZkR1csLinearCombination, ZkR1csTerm, ZkR1csVariable, ZkRelationAccumulator};
use akita_sumcheck::SumcheckInstanceVerifier;
#[cfg(not(feature = "zk"))]
use akita_sumcheck::SumcheckInstanceVerifierExt;
#[cfg(feature = "zk")]
use akita_sumcheck::ZkSumcheckInstanceVerifierExt;
#[cfg(feature = "zk")]
use akita_transcript::labels::ABSORB_ZK_HIDING_COMMITMENT;
use akita_transcript::labels::{
    ABSORB_COMMITMENT, ABSORB_EVALUATION_CLAIMS, ABSORB_STAGE2_NEXT_W_EVAL,
    ABSORB_SUMCHECK_S_CLAIM, CHALLENGE_EVAL_BATCH, CHALLENGE_SUMCHECK_BATCH,
    CHALLENGE_SUMCHECK_ROUND,
};
use akita_transcript::Transcript;
use akita_types::{
    append_batched_commitments_to_transcript, append_claim_incidence_shape_to_transcript,
    append_claim_points_to_transcript, append_claim_values_to_transcript, claim_points_to_base,
    claim_values_to_base, flatten_batched_commitment_rows, prepare_root_opening_point,
    reduce_inner_opening_to_ring_element, relation_claim_from_rows, reorder_stage1_coords,
    ring_opening_point_from_field, schedule_num_fold_levels, w_ring_element_count,
    w_ring_element_count_with_counts, AkitaBatchedProof, AkitaLevelProof, AkitaProofStep,
    AkitaStage1Proof, AkitaStage2Proof, AkitaVerifierSetup, BasisMode, BlockOrder,
    ClaimIncidenceSummary, DegreeOneChallengeSampler, DirectWitnessProof, FlatRingVec, LevelParams,
    PreparedRootOpeningPoint, RingCommitment, RingOpeningPoint, Schedule, Step,
};

#[cfg(feature = "zk")]
fn zk_add_scaled_lc<F: FieldCore>(
    target: &mut ZkR1csLinearCombination<F>,
    scale: F,
    source: &ZkR1csLinearCombination<F>,
) {
    target.constant += scale * source.constant;
    target
        .terms
        .extend(source.terms.iter().cloned().map(|term| ZkR1csTerm {
            variable: term.variable,
            coeff: scale * term.coeff,
        }));
}

#[cfg(feature = "zk")]
fn zk_y_relation_mask_lc<F: FieldCore, const D: usize>(
    mask_start: usize,
    row_weight: F,
    alpha: F,
) -> ZkR1csLinearCombination<F> {
    let mut out = ZkR1csLinearCombination::zero();
    let mut alpha_power = F::one();
    for coeff_idx in 0..D {
        zk_add_scaled_lc(
            &mut out,
            row_weight * alpha_power,
            &ZkR1csLinearCombination::variable(
                ZkR1csVariable::HiddenWitness(mask_start + coeff_idx),
                F::one(),
            ),
        );
        alpha_power *= alpha;
    }
    out
}

/// Verifier state carried between recursive fold levels.
pub struct RecursiveVerifierState<'a, F: FieldCore> {
    /// Current opening point for the committed recursive witness.
    pub opening_point: Vec<F>,
    /// Claimed opening value for the current commitment.
    pub opening: F,
    /// Hiding-witness slot masking the current recursive opening value.
    #[cfg(feature = "zk")]
    opening_mask_index: usize,
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
pub fn verify_root_level<F, E, C, T, const D: usize>(
    y_rings_flat: &FlatRingVec<F>,
    v_flat: &FlatRingVec<F>,
    stage1: &AkitaStage1Proof<F>,
    stage2: &AkitaStage2Proof<F>,
    setup: &AkitaVerifierSetup<F>,
    transcript: &mut T,
    claim_points: &[&[E]],
    prepared_points: &[PreparedRootOpeningPoint<F, D>],
    openings: &[E],
    commitments: &[RingCommitment<F, D>],
    incidence_summary: &ClaimIncidenceSummary,
    #[cfg(feature = "zk")] zk_hiding_cursor: &mut usize,
    root_lp: &LevelParams,
    batched_lp: &LevelParams,
    is_last: bool,
    final_w: Option<&DirectWitnessProof<F>>,
    #[cfg(feature = "zk")] zk_relations: &mut ZkRelationAccumulator<F>,
) -> Result<Vec<F>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    E: ExtField<F>,
    C: ExtField<F>,
    T: Transcript<F>,
{
    let challenge_sampler = DegreeOneChallengeSampler::<F, C>::new(AkitaError::InvalidProof)?;
    let y_rings = y_rings_flat.as_ring_slice::<D>()?;
    let v_typed = v_flat.as_ring_slice::<D>()?;
    let num_claims = incidence_summary.num_claims;
    let num_points = prepared_points.len();
    let base_openings =
        claim_values_to_base::<F, E>(openings, AkitaError::InvalidProof, AkitaError::InvalidProof)?;
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
    let gamma: Vec<F> = (0..num_claims)
        .map(|_| challenge_sampler.sample(transcript, CHALLENGE_EVAL_BATCH))
        .collect();
    for y_ring in y_rings {
        transcript.append_serde(ABSORB_EVALUATION_CLAIMS, y_ring);
    }
    #[cfg(feature = "zk")]
    let mut y_mask_starts = Vec::with_capacity(num_points);

    // Per-point trace check: for each opening point `j`, verify
    // `trace(y_j · σ_{-1}(v_j)) = d · Σ_{ι: point(ι)=j} γ_ι · opening_ι`.
    // Each opening point carries its own inner reduction `v_j`, which may
    // differ across the batch.
    let d_field = F::from_u64(root_lp.ring_dimension as u64);
    let mut batched_openings_per_point = vec![F::zero(); num_points];
    for (claim_idx, (&opening, &g)) in base_openings.iter().zip(gamma.iter()).enumerate() {
        let point_idx = incidence_summary.claim_to_point[claim_idx];
        batched_openings_per_point[point_idx] += g * opening;
    }
    for (point_idx, (y_ring, &batched_opening)) in y_rings
        .iter()
        .zip(batched_openings_per_point.iter())
        .enumerate()
    {
        let v = &prepared_points[point_idx].inner_reduction;
        #[cfg(not(feature = "zk"))]
        let trace_lhs = trace::<F, { D }>(&(*y_ring * v.sigma_m1()));
        let trace_rhs = d_field * batched_opening;
        #[cfg(not(feature = "zk"))]
        if trace_lhs != trace_rhs {
            return Err(AkitaError::InvalidProof);
        }
        #[cfg(feature = "zk")]
        {
            let y_mask_start = *zk_hiding_cursor;
            y_mask_starts.push(y_mask_start);
            *zk_hiding_cursor = (*zk_hiding_cursor)
                .checked_add(D)
                .ok_or(AkitaError::InvalidProof)?;
            let mut trace_lc = ZkR1csLinearCombination::constant(-trace_rhs);
            let sigma_v = v.sigma_m1();
            for coeff_idx in 0..D {
                let mask_lc = ZkR1csLinearCombination::variable(
                    ZkR1csVariable::HiddenWitness(y_mask_start + coeff_idx),
                    F::one(),
                );
                let true_y_lc =
                    ZkRelationAccumulator::unmask_lc(y_ring.coeffs[coeff_idx], &mask_lc);
                let basis = CyclotomicRing::<F, D>::from_coefficients(std::array::from_fn(|idx| {
                    if idx == coeff_idx {
                        F::one()
                    } else {
                        F::zero()
                    }
                }));
                zk_add_scaled_lc(&mut trace_lc, trace::<F, D>(&(basis * sigma_v)), &true_y_lc);
            }
            zk_relations.push_r1cs(
                "root y-ring trace pin",
                trace_lc,
                ZkR1csLinearCombination::one(),
                ZkR1csLinearCombination::zero(),
            );
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
    let rs = ring_switch_verifier::<F, F, T, { D }>(
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
    let relation_claim =
        relation_claim_from_rows(&rs.tau1, rs.alpha, v_typed, commitment_rows, y_rings);
    #[cfg(feature = "zk")]
    let relation_claim_mask = {
        let eq_tau1 = EqPolynomial::evals(&rs.tau1);
        let mut mask = ZkR1csLinearCombination::zero();
        for (point_idx, &mask_start) in y_mask_starts.iter().enumerate() {
            if 1 + point_idx < eq_tau1.len() {
                let point_mask =
                    zk_y_relation_mask_lc::<F, D>(mask_start, eq_tau1[1 + point_idx], rs.alpha);
                zk_add_scaled_lc(&mut mask, F::one(), &point_mask);
            }
        }
        mask
    };
    let tau0_reordered = reorder_stage1_coords(&rs.tau0, rs.col_bits, rs.ring_bits);
    let stage1_verifier = AkitaStage1Verifier::new(tau0_reordered, rs.b);
    #[cfg(feature = "zk")]
    let (r_stage1, stage1_s_claim_mask) = {
        let _sumcheck_span = tracing::info_span!("stage1_sumcheck").entered();
        stage1_verifier.verify(stage1, transcript, zk_relations, zk_hiding_cursor)?
    };
    #[cfg(not(feature = "zk"))]
    let r_stage1 = {
        let _sumcheck_span = tracing::info_span!("stage1_sumcheck").entered();
        stage1_verifier.verify(stage1, transcript)?
    };
    transcript.append_serde(ABSORB_SUMCHECK_S_CLAIM, &stage1.s_claim);
    let batching_coeff: F = challenge_sampler.sample(transcript, CHALLENGE_SUMCHECK_BATCH);
    let stage2_input_claim = batching_coeff * stage1.s_claim + relation_claim;
    let row_eval_source = Stage2RowEvalSource::new(rs.prepared_row_eval);
    #[cfg(feature = "zk")]
    let root_output_mask_variable = ZkR1csVariable::HiddenWitness(
        (*zk_hiding_cursor)
            .checked_add((rs.col_bits + rs.ring_bits) * 4)
            .ok_or(AkitaError::InvalidProof)?,
    );
    let stage2_verifier = if is_last {
        let fw = final_w.ok_or(AkitaError::InvalidProof)?;
        AkitaStage2Verifier::new_with_direct_witness(
            batching_coeff,
            stage1.s_claim,
            #[cfg(feature = "zk")]
            stage1_s_claim_mask.clone(),
            #[cfg(feature = "zk")]
            relation_claim_mask.clone(),
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
            rs.alpha,
            rs.col_bits,
            rs.ring_bits,
        )
    } else {
        AkitaStage2Verifier::new_with_claimed_w_eval(
            batching_coeff,
            stage1.s_claim,
            #[cfg(feature = "zk")]
            stage1_s_claim_mask.clone(),
            #[cfg(feature = "zk")]
            relation_claim_mask.clone(),
            stage2.next_w_eval(),
            #[cfg(feature = "zk")]
            root_output_mask_variable,
            r_stage1.clone(),
            rs.alpha_evals_y,
            row_eval_source,
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
        #[cfg(feature = "zk")]
        {
            stage2_verifier.verify_zk::<F, _, _>(
                &stage2.sumcheck_proof_masked,
                transcript,
                zk_relations,
                zk_hiding_cursor,
                |tr| challenge_sampler.sample(tr, CHALLENGE_SUMCHECK_ROUND),
            )?
        }
        #[cfg(not(feature = "zk"))]
        {
            stage2_verifier.verify::<F, _, _>(&stage2.sumcheck_proof, transcript, |tr| {
                challenge_sampler.sample(tr, CHALLENGE_SUMCHECK_ROUND)
            })?
        }
    };
    #[cfg(feature = "zk")]
    if !is_last {
        *zk_hiding_cursor = (*zk_hiding_cursor)
            .checked_add(1)
            .ok_or(AkitaError::InvalidProof)?;
    }
    transcript.append_serde(ABSORB_STAGE2_NEXT_W_EVAL, &stage2.next_w_eval());

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
    level_proof: &AkitaLevelProof<F>,
    setup: &AkitaVerifierSetup<F>,
    transcript: &mut T,
    current_state: &RecursiveVerifierState<'_, F>,
    is_last: bool,
    final_w: Option<&DirectWitnessProof<F>>,
    lp: &LevelParams,
    block_order: BlockOrder,
    #[cfg(feature = "zk")] zk_hiding_cursor: &mut usize,
    #[cfg(feature = "zk")] zk_relations: &mut ZkRelationAccumulator<F>,
) -> Result<Vec<F>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    T: Transcript<F>,
{
    let y_ring = level_proof.y_ring.as_single_ring::<D>()?;
    let v_typed = level_proof.v.as_ring_slice::<D>()?;
    let commitment_u = current_state.commitment.as_ring_slice::<D>()?;

    let alpha_bits = lp.ring_dimension.trailing_zeros() as usize;
    if current_state.opening_point.len() < alpha_bits {
        return Err(AkitaError::InvalidSetup(
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
    #[cfg(not(feature = "zk"))]
    let trace_lhs = trace::<F, { D }>(&(*y_ring * v.sigma_m1()));
    #[cfg(not(feature = "zk"))]
    let trace_rhs = d * current_state.opening;
    #[cfg(not(feature = "zk"))]
    if trace_lhs != trace_rhs {
        return Err(AkitaError::InvalidProof);
    }
    #[cfg(feature = "zk")]
    let current_y_mask_index = *zk_hiding_cursor;
    #[cfg(feature = "zk")]
    {
        *zk_hiding_cursor = (*zk_hiding_cursor)
            .checked_add(D)
            .ok_or(AkitaError::InvalidProof)?;
        let opening_mask_lc = ZkR1csLinearCombination::variable(
            ZkR1csVariable::HiddenWitness(current_state.opening_mask_index),
            F::one(),
        );
        let true_opening_lc =
            ZkRelationAccumulator::unmask_lc(current_state.opening, &opening_mask_lc);
        let mut trace_lc = ZkR1csLinearCombination::zero();
        let sigma_v = v.sigma_m1();
        for coeff_idx in 0..D {
            let mask_lc = ZkR1csLinearCombination::variable(
                ZkR1csVariable::HiddenWitness(current_y_mask_index + coeff_idx),
                F::one(),
            );
            let true_y_lc = ZkRelationAccumulator::unmask_lc(y_ring.coeffs[coeff_idx], &mask_lc);
            let basis = CyclotomicRing::<F, D>::from_coefficients(std::array::from_fn(|idx| {
                if idx == coeff_idx {
                    F::one()
                } else {
                    F::zero()
                }
            }));
            zk_add_scaled_lc(&mut trace_lc, trace::<F, D>(&(basis * sigma_v)), &true_y_lc);
        }
        zk_add_scaled_lc(&mut trace_lc, -d, &true_opening_lc);
        zk_relations.push_r1cs(
            "recursive y-ring trace pin",
            trace_lc,
            ZkR1csLinearCombination::one(),
            ZkR1csLinearCombination::zero(),
        );
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

    let rs = ring_switch_verifier::<F, F, T, { D }>(
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
    #[cfg(feature = "zk")]
    let relation_claim_mask = {
        let eq_tau1 = EqPolynomial::evals(&rs.tau1);
        if eq_tau1.len() > 1 {
            zk_y_relation_mask_lc::<F, D>(current_y_mask_index, eq_tau1[1], rs.alpha)
        } else {
            ZkR1csLinearCombination::zero()
        }
    };
    let stage1 = &level_proof.stage1;
    let stage2 = &level_proof.stage2;
    let tau0_reordered = reorder_stage1_coords(&rs.tau0, rs.col_bits, rs.ring_bits);
    let stage1_verifier = AkitaStage1Verifier::new(tau0_reordered, rs.b);
    #[cfg(feature = "zk")]
    let (r_stage1, stage1_s_claim_mask) = {
        let _sumcheck_span = tracing::info_span!("stage1_sumcheck").entered();
        stage1_verifier.verify(stage1, transcript, zk_relations, zk_hiding_cursor)?
    };
    #[cfg(not(feature = "zk"))]
    let r_stage1 = {
        let _sumcheck_span = tracing::info_span!("stage1_sumcheck").entered();
        stage1_verifier.verify(stage1, transcript)?
    };

    transcript.append_serde(ABSORB_SUMCHECK_S_CLAIM, &stage1.s_claim);
    let batching_coeff: F = transcript.challenge_scalar(CHALLENGE_SUMCHECK_BATCH);
    let stage2_input_claim = batching_coeff * stage1.s_claim + relation_claim;
    let row_eval_source = Stage2RowEvalSource::new(rs.prepared_row_eval);
    #[cfg(feature = "zk")]
    let output_w_eval_mask_variable = ZkR1csVariable::HiddenWitness(
        (*zk_hiding_cursor)
            .checked_add((rs.col_bits + rs.ring_bits) * 4)
            .ok_or(AkitaError::InvalidProof)?,
    );
    let ring_opening_points_slice = std::slice::from_ref(&ring_opening_point);

    let y_rings_slice = std::slice::from_ref(y_ring);
    let stage2_verifier = if is_last {
        let fw = final_w.ok_or(AkitaError::InvalidProof)?;
        AkitaStage2Verifier::new_with_direct_witness(
            batching_coeff,
            stage1.s_claim,
            #[cfg(feature = "zk")]
            stage1_s_claim_mask.clone(),
            #[cfg(feature = "zk")]
            relation_claim_mask.clone(),
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
            rs.alpha,
            rs.col_bits,
            rs.ring_bits,
        )
    } else {
        AkitaStage2Verifier::new_with_claimed_w_eval(
            batching_coeff,
            stage1.s_claim,
            #[cfg(feature = "zk")]
            stage1_s_claim_mask.clone(),
            #[cfg(feature = "zk")]
            relation_claim_mask.clone(),
            stage2.next_w_eval(),
            #[cfg(feature = "zk")]
            output_w_eval_mask_variable,
            r_stage1.clone(),
            rs.alpha_evals_y,
            row_eval_source,
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
        return Err(AkitaError::InvalidProof);
    }
    let challenges = {
        let _sumcheck_span = tracing::info_span!("stage2_sumcheck").entered();
        #[cfg(feature = "zk")]
        {
            stage2_verifier.verify_zk::<F, _, _>(
                &stage2.sumcheck_proof_masked,
                transcript,
                zk_relations,
                zk_hiding_cursor,
                |tr| tr.challenge_scalar(CHALLENGE_SUMCHECK_ROUND),
            )?
        }
        #[cfg(not(feature = "zk"))]
        {
            stage2_verifier.verify::<F, _, _>(&stage2.sumcheck_proof, transcript, |tr| {
                tr.challenge_scalar(CHALLENGE_SUMCHECK_ROUND)
            })?
        }
    };
    #[cfg(feature = "zk")]
    if !is_last {
        *zk_hiding_cursor = (*zk_hiding_cursor)
            .checked_add(1)
            .ok_or(AkitaError::InvalidProof)?;
    }
    transcript.append_serde(ABSORB_STAGE2_NEXT_W_EVAL, &stage2.next_w_eval());

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
    #[cfg(feature = "zk")] zk_hiding_cursor: &mut usize,
    #[cfg(feature = "zk")] zk_relations: &mut ZkRelationAccumulator<F>,
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
            #[cfg(feature = "zk")]
            zk_hiding_cursor,
            #[cfg(feature = "zk")]
            zk_relations,
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
            #[cfg(feature = "zk")]
            zk_hiding_cursor,
            #[cfg(feature = "zk")]
            zk_relations,
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
            #[cfg(feature = "zk")]
            zk_hiding_cursor,
            #[cfg(feature = "zk")]
            zk_relations,
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
            #[cfg(feature = "zk")]
            zk_hiding_cursor,
            #[cfg(feature = "zk")]
            zk_relations,
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
            #[cfg(feature = "zk")]
            zk_hiding_cursor,
            #[cfg(feature = "zk")]
            zk_relations,
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
            #[cfg(feature = "zk")]
            zk_hiding_cursor,
            #[cfg(feature = "zk")]
            zk_relations,
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
#[allow(clippy::too_many_arguments)]
pub fn verify_batched_recursive_suffix<'a, F, T, const D: usize>(
    proof: &'a AkitaBatchedProof<F>,
    setup: &AkitaVerifierSetup<F>,
    transcript: &mut T,
    schedule: &Schedule,
    mut current_state: RecursiveVerifierState<'a, F>,
    final_w: Option<&DirectWitnessProof<F>>,
    #[cfg(feature = "zk")] zk_hiding_cursor: &mut usize,
    #[cfg(feature = "zk")] zk_relations: &mut ZkRelationAccumulator<F>,
) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    T: Transcript<F>,
{
    let fold_levels = proof.fold_levels().collect::<Vec<_>>();
    let num_levels = fold_levels.len();
    for (offset, level_proof) in fold_levels.iter().copied().enumerate() {
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
            verify_one_level::<F, T, D>(
                level_proof,
                setup,
                transcript,
                &current_state,
                is_last,
                if is_last { final_w } else { None },
                &current_lp,
                BlockOrder::ColumnMajor,
                #[cfg(feature = "zk")]
                zk_hiding_cursor,
                #[cfg(feature = "zk")]
                zk_relations,
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
                #[cfg(feature = "zk")]
                zk_hiding_cursor,
                #[cfg(feature = "zk")]
                zk_relations,
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
                #[cfg(feature = "zk")]
                opening_mask_index: (*zk_hiding_cursor)
                    .checked_sub(1)
                    .ok_or(AkitaError::InvalidProof)?,
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
pub fn verify_fold_batched_proof<F, E, C, T, const D: usize>(
    proof: &AkitaBatchedProof<F>,
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
    C: ExtField<F>,
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
    let base_opening_points = claim_points_to_base::<F, E>(
        opening_points,
        AkitaError::InvalidProof,
        AkitaError::InvalidProof,
    )?;
    let base_opening_point_slices = base_opening_points
        .iter()
        .map(Vec::as_slice)
        .collect::<Vec<_>>();
    let prepared_points = base_opening_point_slices
        .iter()
        .map(|opening_point| {
            prepare_root_opening_point::<F, D>(opening_point, basis, root_lp, alpha_bits)
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| AkitaError::InvalidProof)?;

    #[cfg(feature = "zk")]
    let mut zk_relations = ZkRelationAccumulator::new();
    #[cfg(feature = "zk")]
    {
        if proof.zk_hiding.u_blind.is_empty() || proof.zk_hiding.hiding_witness.is_empty() {
            return Err(AkitaError::InvalidProof);
        }
        transcript.append_serde(ABSORB_ZK_HIDING_COMMITMENT, &proof.zk_hiding.u_blind);
    }
    #[cfg(feature = "zk")]
    let mut zk_hiding_cursor = 0usize;
    let root_openings = openings.to_vec();
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
        &root_openings,
        commitments,
        incidence_summary,
        #[cfg(feature = "zk")]
        &mut zk_hiding_cursor,
        root_lp,
        &root_step.params,
        !has_recursive_levels,
        if has_recursive_levels { None } else { final_w },
        #[cfg(feature = "zk")]
        &mut zk_relations,
    )?;

    if has_recursive_levels {
        #[cfg(feature = "zk")]
        let root_opening_mask_index = zk_hiding_cursor
            .checked_sub(1)
            .ok_or(AkitaError::InvalidProof)?;
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
            opening: fold_root.stage2.next_w_eval(),
            #[cfg(feature = "zk")]
            opening_mask_index: root_opening_mask_index,
            commitment: &fold_root.stage2.next_w_commitment,
            basis: BasisMode::Lagrange,
            w_len: root_step.next_w_len,
            log_basis: next_level_params.log_basis,
        };
        verify_batched_recursive_suffix::<F, T, D>(
            proof,
            setup,
            transcript,
            schedule,
            current_state,
            final_w,
            #[cfg(feature = "zk")]
            &mut zk_hiding_cursor,
            #[cfg(feature = "zk")]
            &mut zk_relations,
        )?;
    }

    #[cfg(feature = "zk")]
    {
        let expected_hiding_witness_len = zk_hiding_cursor
            .checked_add(1)
            .ok_or(AkitaError::InvalidProof)?;
        if expected_hiding_witness_len != proof.zk_hiding.hiding_witness.len() {
            tracing::error!(
                consumed = zk_hiding_cursor,
                supplied = proof.zk_hiding.hiding_witness.len(),
                "ZK hiding witness cursor did not match supplied witness length"
            );
            return Err(AkitaError::InvalidProof);
        }
        zk_relations.verify_all(&proof.zk_hiding.hiding_witness)?;
    }

    Ok(())
}
