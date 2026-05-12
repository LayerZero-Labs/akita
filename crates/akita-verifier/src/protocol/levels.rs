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
    canonical_frobenius_thetas, solve_frobenius_moore, AkitaError, CanonicalField, ExtField,
    FieldCore, FrobeniusExtField, FromPrimitiveInt, PseudoMersenneField, RandomSampling,
};
use akita_serialization::AkitaSerialize;
use akita_sumcheck::{verify_sumcheck, SumcheckInstanceVerifier};
use akita_transcript::labels::{
    ABSORB_COMMITMENT, ABSORB_EVALUATION_CLAIMS, ABSORB_SUMCHECK_S_CLAIM, CHALLENGE_SUMCHECK_BATCH,
    CHALLENGE_SUMCHECK_ROUND,
};
use akita_transcript::{append_ext_field, sample_ext_challenge, Transcript};
use akita_types::{
    append_batched_commitments_to_transcript, append_claim_incidence_shape_to_transcript,
    append_claim_points_to_transcript, append_claim_values_to_transcript, basis_weights,
    dispatch_trace_inner_product_check, flatten_batched_commitment_rows,
    prepare_recursive_opening_point_ext, prepare_root_opening_point_ext,
    recover_ring_subfield_inner_product, relation_claim_from_rows_extension, reorder_stage1_coords,
    sample_public_row_coefficients, schedule_num_fold_levels, w_ring_element_count_with_counts,
    AkitaBatchedProof, AkitaLevelProof, AkitaProofStep, AkitaStage1Proof, AkitaStage2Proof,
    AkitaVerifierSetup, BasisMode, BlockOrder, ClaimIncidenceSummary, DirectWitnessProof,
    FlatRingVec, LevelParams, PreparedRootOpeningPoint, RingCommitment, RingOpeningPoint,
    RingSubfieldEncoding, Schedule, Step,
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

fn ring_subfield_packed_extension_opening_point<F, E, const D: usize>(
    extension_num_vars: usize,
    point: &[E],
) -> Result<Vec<E>, AkitaError>
where
    F: FieldCore,
    E: ExtField<F>,
{
    let k = E::EXT_DEGREE;
    if k == 1 {
        return Ok(point.to_vec());
    }
    if !k.is_power_of_two() || D % k != 0 {
        return Err(AkitaError::InvalidInput(
            "extension degree must be a power of two dividing D".to_string(),
        ));
    }
    if point.len() != extension_num_vars {
        return Err(AkitaError::InvalidPointDimension {
            expected: extension_num_vars,
            actual: point.len(),
        });
    }
    let alpha_bits = D.trailing_zeros() as usize;
    let kappa_bits = k.trailing_zeros() as usize;
    let packed_inner_bits = alpha_bits.checked_sub(kappa_bits).ok_or_else(|| {
        AkitaError::InvalidInput("extension degree exceeds ring dimension".to_string())
    })?;
    if extension_num_vars < packed_inner_bits {
        return Err(AkitaError::InvalidPointDimension {
            expected: packed_inner_bits,
            actual: extension_num_vars,
        });
    }

    let mut transformed = Vec::with_capacity(extension_num_vars + kappa_bits);
    transformed.extend_from_slice(&point[..packed_inner_bits]);
    transformed.resize(alpha_bits, E::zero());
    transformed.extend_from_slice(&point[packed_inner_bits..]);
    Ok(transformed)
}

fn frobenius_protocol_points<F, L, const D: usize>(
    logical_point: &[L],
) -> Result<(usize, Vec<Vec<L>>), AkitaError>
where
    F: FieldCore,
    L: FrobeniusExtField<F>,
{
    let split_bits = L::EXT_DEGREE.trailing_zeros() as usize;
    let width = 1usize
        .checked_shl(split_bits as u32)
        .ok_or(AkitaError::InvalidProof)?;
    if width != L::EXT_DEGREE || !L::EXT_DEGREE.is_power_of_two() {
        return Err(AkitaError::InvalidProof);
    }
    if split_bits > logical_point.len() {
        return Err(AkitaError::InvalidProof);
    }
    let extension_num_vars = logical_point.len() - split_bits;
    let tail_point = &logical_point[split_bits..];
    let mut protocol_points = Vec::with_capacity(width);
    for power in 0..width {
        let conjugate_tail = tail_point
            .iter()
            .copied()
            .map(|coord| L::frobenius_pow(coord, power))
            .collect::<Vec<_>>();
        protocol_points.push(ring_subfield_packed_extension_opening_point::<F, L, D>(
            extension_num_vars,
            &conjugate_tail,
        )?);
    }
    Ok((split_bits, protocol_points))
}

fn reconstruct_frobenius_opening<F, L>(
    logical_point: &[L],
    split_bits: usize,
    internal_claims: &[L],
) -> Result<L, AkitaError>
where
    F: PseudoMersenneField + FromPrimitiveInt,
    L: FrobeniusExtField<F>,
{
    let width = 1usize
        .checked_shl(split_bits as u32)
        .ok_or(AkitaError::InvalidProof)?;
    if internal_claims.len() != width || logical_point.len() < split_bits {
        return Err(AkitaError::InvalidProof);
    }
    let thetas = canonical_frobenius_thetas::<F, L>(width).map_err(|_| AkitaError::InvalidProof)?;
    let r = internal_claims
        .iter()
        .enumerate()
        .map(|(idx, &claim)| L::frobenius_inv_pow(claim, idx))
        .collect::<Vec<_>>();
    let z = solve_frobenius_moore::<F, L>(&thetas, &r).map_err(|_| AkitaError::InvalidProof)?;
    let head_weights = basis_weights(&logical_point[..split_bits], BasisMode::Lagrange);
    Ok(head_weights
        .into_iter()
        .zip(z)
        .fold(L::zero(), |acc, (weight, z_h)| acc + weight * z_h))
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
    final_w_len: Option<usize>,
) -> Result<Vec<C>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    E: RingSubfieldEncoding<F>,
    C: RingSubfieldEncoding<F> + ExtField<E> + FromPrimitiveInt + AkitaSerialize,
    T: Transcript<F>,
{
    let y_rings = y_rings_flat.as_ring_slice::<D>()?;
    let v_typed = v_flat.as_ring_slice::<D>()?;
    let num_claims = incidence_summary.num_claims;
    let num_points = prepared_points.len();
    if num_points == 0
        || num_points != incidence_summary.num_points
        || claim_points.len() != incidence_summary.num_points
        || y_rings.len() != incidence_summary.num_public_rows
        || openings.len() != num_claims
        || commitments.len() != incidence_summary.num_groups
        || incidence_summary.claim_to_point.len() != num_claims
        || incidence_summary.claim_to_public_row.len() != num_claims
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
    let row_coefficients =
        sample_public_row_coefficients::<F, C, T>(incidence_summary, transcript)?;
    for y_ring in y_rings {
        transcript.append_serde(ABSORB_EVALUATION_CLAIMS, y_ring);
    }

    // Per-row trace check: for each public row `r`, verify the ring-subfield
    // trace inner-product identity
    // `trace_h(Y_r · σ_{-1}(v_{point(r)})) == (D / K) · embed_subfield(opening_r)`
    // in R_q, where `opening_r = Σ_{c in row(r)} γ_{r,c} · opening_c`.
    let mut batched_openings_per_row: Vec<C> = vec![C::zero(); incidence_summary.num_public_rows];
    for (row_idx, row) in incidence_summary.public_rows.iter().enumerate() {
        if row.point_idx >= prepared_points.len() || row.claim_indices.is_empty() {
            return Err(AkitaError::InvalidProof);
        }
        for &claim_idx in &row.claim_indices {
            if claim_idx >= openings.len()
                || incidence_summary.claim_to_public_row[claim_idx] != row_idx
                || incidence_summary.claim_to_point[claim_idx] != row.point_idx
            {
                return Err(AkitaError::InvalidProof);
            }
            batched_openings_per_row[row_idx] +=
                row_coefficients[claim_idx] * C::lift_base(openings[claim_idx]);
        }
    }
    for (row, (y_ring, batched_opening)) in incidence_summary
        .public_rows
        .iter()
        .zip(y_rings.iter().zip(batched_openings_per_row.iter()))
    {
        let v = &prepared_points[row.point_idx].inner_reduction;
        let trace_input = *y_ring * v.sigma_m1();
        let coords = batched_opening.to_ring_subfield_coords();
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
        final_w_len.ok_or(AkitaError::InvalidProof)?
    } else {
        w_ring_element_count_with_counts::<F>(
            batched_lp,
            incidence_summary.group_poly_counts.len(),
            incidence_summary.group_poly_counts.iter().sum(),
            num_claims,
            incidence_summary.num_public_rows,
        ) * D
    };

    let ring_opening_points: Vec<RingOpeningPoint<F>> = incidence_summary
        .public_rows
        .iter()
        .map(|row| prepared_points[row.point_idx].ring_opening_point.clone())
        .collect();
    let ring_multiplier_points: Vec<_> = incidence_summary
        .public_rows
        .iter()
        .map(|row| prepared_points[row.point_idx].ring_multiplier_point.clone())
        .collect();
    let rs = ring_switch_verifier::<F, C, T, { D }>(
        &ring_opening_points,
        &ring_multiplier_points,
        &incidence_summary.claim_to_public_row,
        &stage1_challenges,
        w_len,
        &stage2.next_w_commitment,
        transcript,
        batched_lp,
        &incidence_summary.group_poly_counts,
        &incidence_summary.claim_to_group,
        &incidence_summary.claim_poly_indices,
        &row_coefficients,
        incidence_summary.num_public_rows,
    )?;
    let relation_claim = relation_claim_from_rows_extension::<F, C, D>(
        &rs.tau1,
        rs.alpha,
        v_typed,
        commitment_rows,
        y_rings,
    );
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
            w_len,
            r_stage1.clone(),
            rs.alpha_evals_y,
            row_eval_source,
            &setup.expanded,
            &ring_opening_points,
            &ring_multiplier_points,
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
            &ring_multiplier_points,
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
    final_w_len: Option<usize>,
    lp: &LevelParams,
    block_order: BlockOrder,
) -> Result<Vec<L>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling + PseudoMersenneField,
    L: RingSubfieldEncoding<F> + FrobeniusExtField<F> + FromPrimitiveInt + AkitaSerialize,
    T: Transcript<F>,
{
    let y_rings = level_proof.try_y_rings_typed::<D>()?;
    let v_typed = level_proof.v.as_ring_slice::<D>()?;
    let commitment_u = current_state.commitment.as_ring_slice::<D>()?;

    let alpha_bits = lp.ring_dimension.trailing_zeros() as usize;
    let (split_bits, protocol_points) =
        frobenius_protocol_points::<F, L, D>(&current_state.opening_point)?;
    if y_rings.len() != protocol_points.len() {
        return Err(AkitaError::InvalidProof);
    }
    let prepared_points = protocol_points
        .iter()
        .map(|point| {
            prepare_recursive_opening_point_ext::<F, L, D>(
                point,
                current_state.basis,
                lp,
                alpha_bits,
                block_order,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;

    current_state
        .commitment
        .append_as_ring_slice::<T, D>(ABSORB_COMMITMENT, transcript)?;
    for prepared_point in &prepared_points {
        for pt in &prepared_point.padded_point {
            append_ext_field::<F, L, T>(transcript, ABSORB_EVALUATION_CLAIMS, pt);
        }
    }
    for y_ring in &y_rings {
        transcript.append_serde(ABSORB_EVALUATION_CLAIMS, y_ring);
    }

    let internal_claims = y_rings
        .iter()
        .zip(prepared_points.iter())
        .map(|(y_ring, prepared_point)| {
            recover_ring_subfield_inner_product::<F, L, D>(y_ring, &prepared_point.inner_reduction)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let reconstructed = reconstruct_frobenius_opening::<F, L>(
        &current_state.opening_point,
        split_bits,
        &internal_claims,
    )?;
    if reconstructed != current_state.opening {
        return Err(AkitaError::InvalidProof);
    }

    let ring_opening_points = prepared_points
        .iter()
        .map(|prepared_point| prepared_point.ring_opening_point.clone())
        .collect::<Vec<_>>();
    let ring_multiplier_points = prepared_points
        .iter()
        .map(|prepared_point| prepared_point.ring_multiplier_point.clone())
        .collect::<Vec<_>>();
    let num_claims = y_rings.len();
    let stage1_challenges =
        derive_stage1_challenges::<F, T, D>(transcript, v_typed, lp.num_blocks * num_claims, lp)?;

    let w_len = if is_last {
        final_w_len.ok_or(AkitaError::InvalidProof)?
    } else {
        w_ring_element_count_with_counts::<F>(lp, 1, 1, num_claims, num_claims) * D
    };
    let claim_to_point = (0..num_claims).collect::<Vec<_>>();
    let claim_to_group = vec![0usize; num_claims];
    let claim_poly_indices = vec![0usize; num_claims];
    let gamma = vec![L::one(); num_claims];

    let rs = ring_switch_verifier::<F, L, T, { D }>(
        &ring_opening_points,
        &ring_multiplier_points,
        &claim_to_point,
        &stage1_challenges,
        w_len,
        level_proof.next_w_commitment(),
        transcript,
        lp,
        &[1usize],
        &claim_to_group,
        &claim_poly_indices,
        &gamma,
        num_claims,
    )?;
    let relation_claim = relation_claim_from_rows_extension::<F, L, D>(
        &rs.tau1,
        rs.alpha,
        v_typed,
        commitment_u,
        &y_rings,
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
    let stage2_verifier = if is_last {
        let fw = final_w.ok_or(AkitaError::InvalidProof)?;
        AkitaStage2Verifier::new_with_direct_witness(
            batching_coeff,
            stage1.s_claim,
            fw,
            w_len,
            r_stage1.clone(),
            rs.alpha_evals_y,
            row_eval_source,
            &setup.expanded,
            &ring_opening_points,
            &ring_multiplier_points,
            &rs.tau1,
            v_typed,
            commitment_u,
            &y_rings,
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
            &ring_multiplier_points,
            &rs.tau1,
            v_typed,
            commitment_u,
            &y_rings,
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
    final_w_len: Option<usize>,
    lp: &LevelParams,
    block_order: BlockOrder,
) -> Result<Vec<L>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling + PseudoMersenneField,
    L: RingSubfieldEncoding<F> + FrobeniusExtField<F> + FromPrimitiveInt + AkitaSerialize,
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
            final_w_len,
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
            final_w_len,
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
            final_w_len,
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
            final_w_len,
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
            final_w_len,
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
            final_w_len,
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
    F: FieldCore + CanonicalField + RandomSampling + PseudoMersenneField,
    L: RingSubfieldEncoding<F> + FrobeniusExtField<F> + FromPrimitiveInt + AkitaSerialize,
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
            || !level_proof.y_ring.can_decode_vec(level_d)
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
                if is_last { Some(next_w_len) } else { None },
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
                if is_last { Some(next_w_len) } else { None },
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
            let y_ring_count = level_proof.y_ring.coeff_len() / level_d;
            let computed_next_w_len = w_ring_element_count_with_counts::<F>(
                &current_lp,
                1,
                1,
                y_ring_count,
                y_ring_count,
            ) * level_d;
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
    F: FieldCore + CanonicalField + RandomSampling + PseudoMersenneField,
    E: RingSubfieldEncoding<F>,
    C: RingSubfieldEncoding<F>
        + ExtField<E>
        + FrobeniusExtField<F>
        + FromPrimitiveInt
        + AkitaSerialize,
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
        tracing::debug!(
            proof_levels = proof.num_fold_levels(),
            expected_recursive_levels,
            "folded proof recursive level count mismatch"
        );
        return Err(AkitaError::InvalidProof);
    }

    let y_coeff_len = fold_root.y_rings.coeff_len();
    if !y_coeff_len.is_multiple_of(D) {
        return Err(AkitaError::InvalidProof);
    }
    // One public y-ring per distinct opening point.
    if y_coeff_len / D != opening_points.len() {
        tracing::debug!(
            y_rings = y_coeff_len / D,
            opening_points = opening_points.len(),
            "folded root y-ring count mismatch"
        );
        return Err(AkitaError::InvalidProof);
    }

    let final_w = proof
        .steps
        .last()
        .and_then(AkitaProofStep::as_direct)
        .ok_or(AkitaError::InvalidProof)?;
    let terminal_direct = schedule
        .steps
        .last()
        .and_then(|step| match step {
            Step::Direct(direct) => Some(direct),
            Step::Fold(_) => None,
        })
        .ok_or(AkitaError::InvalidProof)?;
    if final_w.shape() != terminal_direct.witness_shape {
        tracing::debug!(
            actual_shape = ?final_w.shape(),
            expected_shape = ?terminal_direct.witness_shape,
            "folded proof terminal witness shape mismatch"
        );
        return Err(AkitaError::InvalidProof);
    }
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
        if has_recursive_levels {
            None
        } else {
            Some(root_step.next_w_len)
        },
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
