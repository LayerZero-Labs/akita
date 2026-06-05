//! Terminal direct ring-relation verifier.

use super::*;
use akita_algebra::CyclotomicRing;
use akita_challenges::IntegerChallenge;
use akita_types::CleartextWitnessProof;

#[derive(Clone)]
struct TerminalDirectSegments<F: FieldCore, const D: usize> {
    w_folded: Vec<CyclotomicRing<F, D>>,
    t_digits: Vec<CyclotomicRing<F, D>>,
    t_recomposed_rows: Vec<Vec<CyclotomicRing<F, D>>>,
    z_pre: Vec<CyclotomicRing<F, D>>,
}

/// Verify terminal direct ring relations without sampling ring-switch or
/// stage-2 challenges.
///
/// The caller must have already bound the descriptor-selected terminal
/// `w_hat` slice and sampled the stage-1 folding challenges. This function
/// binds the remaining final-witness bytes, decodes the transparent terminal
/// witness, and checks the reduced M-row equations directly.
#[cfg(not(feature = "zk"))]
#[allow(clippy::too_many_arguments)]
#[tracing::instrument(skip_all, name = "verify_terminal_direct_ring_relations")]
pub(crate) fn verify_terminal_direct_ring_relations<F, E, T, const D: usize>(
    opening_points: &[RingOpeningPoint<F>],
    ring_multiplier_points: &[RingMultiplierOpeningPoint<F, D>],
    claim_to_point: &[usize],
    challenges: &Challenges,
    final_w_len: usize,
    final_witness: &CleartextWitnessProof<F>,
    transcript: &mut T,
    terminal_parts: &TerminalWitnessTranscriptParts,
    setup: &AkitaExpandedSetup<F>,
    lp: &LevelParams,
    num_polys_per_point: &[usize],
    claim_to_point_poly: &[usize],
    claim_poly_indices: &[usize],
    gamma: &[E],
    commitment_rows: &[CyclotomicRing<F, D>],
    y_rings: &[CyclotomicRing<F, D>],
    num_public_rows: usize,
) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt,
    E: RingSubfieldEncoding<F> + FromPrimitiveInt,
    T: Transcript<F>,
{
    transcript.record_wire_bytes(ABSORB_TERMINAL_W_REMAINDER, &terminal_parts.remainder);
    transcript.append_bytes(ABSORB_TERMINAL_W_REMAINDER, &terminal_parts.remainder);
    verify_terminal_direct_relation_rows::<F, E, D>(
        opening_points,
        ring_multiplier_points,
        claim_to_point,
        challenges,
        final_w_len,
        final_witness,
        setup,
        lp,
        num_polys_per_point,
        claim_to_point_poly,
        claim_poly_indices,
        gamma,
        commitment_rows,
        y_rings,
        num_public_rows,
    )
}

#[cfg(not(feature = "zk"))]
#[allow(clippy::too_many_arguments)]
pub(super) fn verify_terminal_direct_relation_rows<F, E, const D: usize>(
    opening_points: &[RingOpeningPoint<F>],
    ring_multiplier_points: &[RingMultiplierOpeningPoint<F, D>],
    claim_to_point: &[usize],
    challenges: &Challenges,
    final_w_len: usize,
    final_witness: &CleartextWitnessProof<F>,
    setup: &AkitaExpandedSetup<F>,
    lp: &LevelParams,
    num_polys_per_point: &[usize],
    claim_to_point_poly: &[usize],
    claim_poly_indices: &[usize],
    gamma: &[E],
    commitment_rows: &[CyclotomicRing<F, D>],
    y_rings: &[CyclotomicRing<F, D>],
    num_public_rows: usize,
) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt,
    E: RingSubfieldEncoding<F> + FromPrimitiveInt,
{
    validate_terminal_direct_shape::<F, D>(
        opening_points,
        ring_multiplier_points,
        claim_to_point,
        challenges,
        lp,
        num_polys_per_point,
        claim_to_point_poly,
        claim_poly_indices,
        commitment_rows,
        y_rings,
        num_public_rows,
    )?;
    if gamma.len() != claim_to_point.len() || final_witness.num_elems() != final_w_len {
        return Err(AkitaError::InvalidProof);
    }

    let challenge_rings = materialize_challenge_rings::<F, D>(challenges)?;
    let segments = decode_terminal_direct_segments::<F, D>(
        final_witness,
        final_w_len,
        lp,
        claim_to_point.len(),
        num_polys_per_point,
        num_public_rows,
    )?;
    let row_coefficient_rings = gamma
        .iter()
        .copied()
        .map(|coefficient| {
            embed_ring_subfield_scalar::<F, E, D>(coefficient, AkitaError::InvalidProof)
        })
        .collect::<Result<Vec<_>, _>>()?;

    check_terminal_direct_consistency_row::<F, D>(
        &segments.w_folded,
        &segments.z_pre,
        &challenge_rings,
        ring_multiplier_points,
        lp,
        num_public_rows,
    )?;
    check_terminal_direct_public_rows::<F, D>(
        &segments.w_folded,
        &row_coefficient_rings,
        ring_multiplier_points,
        claim_to_point,
        y_rings,
        lp.num_blocks,
    )?;
    check_terminal_direct_b_rows::<F, D>(
        setup,
        &segments.t_digits,
        lp,
        num_polys_per_point,
        commitment_rows,
    )?;
    check_terminal_direct_a_rows::<F, D>(
        setup,
        &segments.t_recomposed_rows,
        &segments.z_pre,
        &challenge_rings,
        lp,
        num_polys_per_point,
        claim_to_point_poly,
        claim_poly_indices,
        num_public_rows,
    )
}

#[cfg(not(feature = "zk"))]
#[allow(clippy::too_many_arguments)]
fn validate_terminal_direct_shape<F, const D: usize>(
    opening_points: &[RingOpeningPoint<F>],
    ring_multiplier_points: &[RingMultiplierOpeningPoint<F, D>],
    claim_to_point: &[usize],
    challenges: &Challenges,
    lp: &LevelParams,
    num_polys_per_point: &[usize],
    claim_to_point_poly: &[usize],
    claim_poly_indices: &[usize],
    commitment_rows: &[CyclotomicRing<F, D>],
    y_rings: &[CyclotomicRing<F, D>],
    num_public_rows: usize,
) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField,
{
    validate_level_dispatch::<D>(lp)?;
    validate_log_basis(lp.log_basis)?;
    let num_claims = claim_to_point.len();
    let num_points = num_polys_per_point.len();
    if num_claims == 0
        || num_points == 0
        || num_public_rows != num_points
        || y_rings.len() != num_public_rows
        || num_polys_per_point.contains(&0)
    {
        return Err(AkitaError::InvalidProof);
    }
    validate_opening_points_for_claims(opening_points, claim_to_point, lp, num_claims)?;
    if opening_points.len() != num_points || ring_multiplier_points.len() != num_public_rows {
        return Err(AkitaError::InvalidProof);
    }
    if ring_multiplier_points
        .iter()
        .any(|point| point.a_len() < lp.block_len || point.b_len() != lp.num_blocks)
    {
        return Err(AkitaError::InvalidProof);
    }
    if claim_to_point_poly.len() != num_claims
        || claim_poly_indices.len() != num_claims
        || challenges.logical_len()
            != num_claims
                .checked_mul(lp.num_blocks)
                .ok_or(AkitaError::InvalidProof)?
    {
        return Err(AkitaError::InvalidProof);
    }
    for claim_idx in 0..num_claims {
        let point_idx = *claim_to_point_poly
            .get(claim_idx)
            .ok_or(AkitaError::InvalidProof)?;
        let poly_idx = *claim_poly_indices
            .get(claim_idx)
            .ok_or(AkitaError::InvalidProof)?;
        let point_poly_count = *num_polys_per_point
            .get(point_idx)
            .ok_or(AkitaError::InvalidProof)?;
        if poly_idx >= point_poly_count {
            return Err(AkitaError::InvalidProof);
        }
    }
    let expected_commitment_rows = lp
        .b_key
        .row_len()
        .checked_mul(num_points)
        .ok_or(AkitaError::InvalidProof)?;
    if commitment_rows.len() != expected_commitment_rows {
        return Err(AkitaError::InvalidProof);
    }
    let num_t_vectors = num_polys_per_point.iter().try_fold(0usize, |acc, &count| {
        acc.checked_add(count).ok_or(AkitaError::InvalidProof)
    })?;
    let num_digits_fold = lp
        .num_digits_fold(num_t_vectors, F::modulus_bits())
        .map_err(|_| AkitaError::InvalidSetup("terminal direct fold depth invalid".to_string()))?;
    if lp.num_blocks == 0
        || !lp.num_blocks.is_power_of_two()
        || lp.block_len == 0
        || lp.num_digits_open == 0
        || lp.num_digits_commit == 0
        || num_digits_fold == 0
    {
        return Err(AkitaError::InvalidSetup(
            "terminal direct verifier layout has zero width".to_string(),
        ));
    }
    Ok(())
}

#[cfg(not(feature = "zk"))]
fn materialize_challenge_rings<F, const D: usize>(
    challenges: &Challenges,
) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>
where
    F: FieldCore + FromPrimitiveInt,
{
    match challenges {
        Challenges::Sparse {
            challenges: sparse, ..
        } => sparse
            .iter()
            .map(sparse_challenge_to_ring::<F, D>)
            .collect(),
        Challenges::Tensor { factored } => factored
            .expand_integer::<D>()?
            .iter()
            .map(integer_challenge_to_ring::<F, D>)
            .collect(),
    }
}

#[cfg(not(feature = "zk"))]
fn sparse_challenge_to_ring<F, const D: usize>(
    challenge: &akita_challenges::SparseChallenge,
) -> Result<CyclotomicRing<F, D>, AkitaError>
where
    F: FieldCore + FromPrimitiveInt,
{
    challenge.validate::<D>()?;
    let mut coeffs = [F::zero(); D];
    for (&pos, &coeff) in challenge.positions.iter().zip(challenge.coeffs.iter()) {
        let slot = coeffs
            .get_mut(pos as usize)
            .ok_or(AkitaError::InvalidProof)?;
        *slot += F::from_i64(i64::from(coeff));
    }
    Ok(CyclotomicRing::from_coefficients(coeffs))
}

#[cfg(not(feature = "zk"))]
fn integer_challenge_to_ring<F, const D: usize>(
    challenge: &IntegerChallenge,
) -> Result<CyclotomicRing<F, D>, AkitaError>
where
    F: FieldCore + FromPrimitiveInt,
{
    if challenge.positions.len() != challenge.coeffs.len() {
        return Err(AkitaError::InvalidInput(
            "integer challenge positions/coeffs length mismatch".to_string(),
        ));
    }
    let mut coeffs = [F::zero(); D];
    for (&pos, &coeff) in challenge.positions.iter().zip(challenge.coeffs.iter()) {
        if coeff == 0 {
            return Err(AkitaError::InvalidInput(
                "integer challenge coefficients must be non-zero".to_string(),
            ));
        }
        let slot = coeffs.get_mut(pos as usize).ok_or_else(|| {
            AkitaError::InvalidInput(format!(
                "integer challenge position {pos} out of range for D={D}"
            ))
        })?;
        *slot += F::from_i64(i64::from(coeff));
    }
    Ok(CyclotomicRing::from_coefficients(coeffs))
}

#[cfg(not(feature = "zk"))]
fn decode_terminal_direct_segments<F, const D: usize>(
    final_witness: &CleartextWitnessProof<F>,
    final_w_len: usize,
    lp: &LevelParams,
    num_claims: usize,
    num_polys_per_point: &[usize],
    num_public_rows: usize,
) -> Result<TerminalDirectSegments<F, D>, AkitaError>
where
    F: FieldCore + CanonicalField,
{
    validate_log_basis(lp.log_basis)?;
    let digits = final_witness.packed_i8_digits()?;
    if digits.len() != final_w_len || !digits.len().is_multiple_of(D) {
        return Err(AkitaError::InvalidProof);
    }
    let planes = digits
        .chunks_exact(D)
        .map(|chunk| {
            let mut plane = [0i8; D];
            plane.copy_from_slice(chunk);
            plane
        })
        .collect::<Vec<_>>();
    let layout =
        terminal_direct_plane_layout::<F>(lp, num_claims, num_polys_per_point, num_public_rows)?;
    if planes.len() != layout.total_planes {
        return Err(AkitaError::InvalidProof);
    }
    let w_folded = decode_terminal_direct_w_folded::<F, D>(&planes, &layout, lp)?;
    let (t_digits, t_recomposed_rows) =
        decode_terminal_direct_t::<F, D>(&planes, &layout, lp, num_polys_per_point)?;
    let z_pre = decode_terminal_direct_z::<F, D>(&planes, &layout, lp, num_public_rows)?;
    Ok(TerminalDirectSegments {
        w_folded,
        t_digits,
        t_recomposed_rows,
        z_pre,
    })
}

#[cfg(not(feature = "zk"))]
#[derive(Clone, Copy)]
struct TerminalDirectPlaneLayout {
    offset_w: usize,
    offset_t: usize,
    offset_z: usize,
    num_digits_fold: usize,
    total_blocks: usize,
    t_total_blocks: usize,
    total_planes: usize,
}

#[cfg(not(feature = "zk"))]
fn terminal_direct_plane_layout<F: FieldCore + CanonicalField>(
    lp: &LevelParams,
    num_claims: usize,
    num_polys_per_point: &[usize],
    num_public_rows: usize,
) -> Result<TerminalDirectPlaneLayout, AkitaError> {
    let total_blocks = lp
        .num_blocks
        .checked_mul(num_claims)
        .ok_or_else(|| AkitaError::InvalidSetup("terminal direct W width overflow".to_string()))?;
    let total_poly_slots = num_polys_per_point.iter().try_fold(0usize, |acc, &count| {
        acc.checked_add(count)
            .ok_or_else(|| AkitaError::InvalidSetup("terminal direct T width overflow".to_string()))
    })?;
    let t_total_blocks = lp.num_blocks.checked_mul(total_poly_slots).ok_or_else(|| {
        AkitaError::InvalidSetup("terminal direct T block count overflow".to_string())
    })?;
    let w_len = lp
        .num_digits_open
        .checked_mul(total_blocks)
        .ok_or_else(|| AkitaError::InvalidSetup("terminal direct W width overflow".to_string()))?;
    let t_len = lp
        .num_digits_open
        .checked_mul(lp.a_key.row_len())
        .and_then(|len| len.checked_mul(t_total_blocks))
        .ok_or_else(|| AkitaError::InvalidSetup("terminal direct T width overflow".to_string()))?;
    let num_digits_fold = lp
        .num_digits_fold(total_poly_slots, F::modulus_bits())
        .map_err(|_| AkitaError::InvalidSetup("terminal direct fold depth invalid".to_string()))?;
    let z_len = num_digits_fold
        .checked_mul(lp.inner_width())
        .and_then(|len| len.checked_mul(num_public_rows))
        .ok_or_else(|| AkitaError::InvalidSetup("terminal direct Z width overflow".to_string()))?;
    let z_first = akita_types::ring_column_z_first(lp);
    let offset_z = if z_first { 0 } else { w_len + t_len };
    let offset_w = if z_first { z_len } else { 0 };
    let offset_t = if z_first {
        z_len.checked_add(w_len).ok_or_else(|| {
            AkitaError::InvalidSetup("terminal direct T offset overflow".to_string())
        })?
    } else {
        w_len
    };
    let total_planes = w_len
        .checked_add(t_len)
        .and_then(|len| len.checked_add(z_len))
        .ok_or_else(|| AkitaError::InvalidSetup("terminal direct width overflow".to_string()))?;
    Ok(TerminalDirectPlaneLayout {
        offset_w,
        offset_t,
        offset_z,
        num_digits_fold,
        total_blocks,
        t_total_blocks,
        total_planes,
    })
}

#[cfg(not(feature = "zk"))]
fn plane_at<const D: usize>(planes: &[[i8; D]], idx: usize) -> Result<[i8; D], AkitaError> {
    planes.get(idx).copied().ok_or(AkitaError::InvalidProof)
}

#[cfg(not(feature = "zk"))]
fn recompose_i8_planes<F, const D: usize>(
    planes: &[[i8; D]],
    log_basis: u32,
) -> CyclotomicRing<F, D>
where
    F: FieldCore + CanonicalField,
{
    CyclotomicRing::gadget_recompose_pow2_i8(planes, log_basis)
}

#[cfg(not(feature = "zk"))]
fn decode_terminal_direct_w_folded<F, const D: usize>(
    planes: &[[i8; D]],
    layout: &TerminalDirectPlaneLayout,
    lp: &LevelParams,
) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>
where
    F: FieldCore + CanonicalField,
{
    let mut out = Vec::with_capacity(layout.total_blocks);
    for block_idx in 0..layout.total_blocks {
        let digits = (0..lp.num_digits_open)
            .map(|digit_idx| {
                layout
                    .offset_w
                    .checked_add(
                        digit_idx
                            .checked_mul(layout.total_blocks)
                            .and_then(|offset| offset.checked_add(block_idx))
                            .ok_or(AkitaError::InvalidProof)?,
                    )
                    .ok_or(AkitaError::InvalidProof)
                    .and_then(|idx| plane_at(planes, idx))
            })
            .collect::<Result<Vec<_>, _>>()?;
        out.push(recompose_i8_planes::<F, D>(&digits, lp.log_basis));
    }
    Ok(out)
}

#[cfg(not(feature = "zk"))]
type TerminalDirectTDigits<F, const D: usize> =
    (Vec<CyclotomicRing<F, D>>, Vec<Vec<CyclotomicRing<F, D>>>);

#[cfg(not(feature = "zk"))]
fn decode_terminal_direct_t<F, const D: usize>(
    planes: &[[i8; D]],
    layout: &TerminalDirectPlaneLayout,
    lp: &LevelParams,
    num_polys_per_point: &[usize],
) -> Result<TerminalDirectTDigits<F, D>, AkitaError>
where
    F: FieldCore + CanonicalField,
{
    let n_a = lp.a_key.row_len();
    let planes_per_block = n_a
        .checked_mul(lp.num_digits_open)
        .ok_or(AkitaError::InvalidProof)?;
    let t_digit_len = layout
        .t_total_blocks
        .checked_mul(planes_per_block)
        .ok_or(AkitaError::InvalidProof)?;
    let mut t_digits = vec![CyclotomicRing::<F, D>::zero(); t_digit_len];
    let mut recomposed = Vec::with_capacity(layout.t_total_blocks);
    for flat_block_idx in 0..layout.t_total_blocks {
        let mut rows = Vec::with_capacity(n_a);
        for a_idx in 0..n_a {
            let digits = (0..lp.num_digits_open)
                .map(|digit_idx| {
                    let compound_digit = a_idx
                        .checked_mul(lp.num_digits_open)
                        .and_then(|offset| offset.checked_add(digit_idx))
                        .ok_or(AkitaError::InvalidProof)?;
                    let source_idx = layout
                        .offset_t
                        .checked_add(
                            compound_digit
                                .checked_mul(layout.t_total_blocks)
                                .and_then(|offset| offset.checked_add(flat_block_idx))
                                .ok_or(AkitaError::InvalidProof)?,
                        )
                        .ok_or(AkitaError::InvalidProof)?;
                    plane_at(planes, source_idx)
                })
                .collect::<Result<Vec<_>, _>>()?;
            let ring = recompose_i8_planes::<F, D>(&digits, lp.log_basis);
            rows.push(ring);
            for (digit_idx, plane) in digits.into_iter().enumerate() {
                let row_digit_idx = a_idx
                    .checked_mul(lp.num_digits_open)
                    .and_then(|idx| idx.checked_add(digit_idx))
                    .ok_or(AkitaError::InvalidProof)?;
                let target_idx = flat_block_idx
                    .checked_mul(planes_per_block)
                    .and_then(|idx| idx.checked_add(row_digit_idx))
                    .ok_or(AkitaError::InvalidProof)?;
                let slot = t_digits
                    .get_mut(target_idx)
                    .ok_or(AkitaError::InvalidProof)?;
                *slot = recompose_i8_planes::<F, D>(&[plane], lp.log_basis);
            }
        }
        recomposed.push(rows);
    }

    let expected_blocks = num_polys_per_point
        .iter()
        .try_fold(0usize, |acc, &count| {
            acc.checked_add(count).ok_or(AkitaError::InvalidProof)
        })?
        .checked_mul(lp.num_blocks)
        .ok_or(AkitaError::InvalidProof)?;
    if recomposed.len() != expected_blocks || expected_blocks != layout.t_total_blocks {
        return Err(AkitaError::InvalidProof);
    }
    Ok((t_digits, recomposed))
}

#[cfg(not(feature = "zk"))]
fn decode_terminal_direct_z<F, const D: usize>(
    planes: &[[i8; D]],
    layout: &TerminalDirectPlaneLayout,
    lp: &LevelParams,
    num_public_rows: usize,
) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>
where
    F: FieldCore + CanonicalField,
{
    let inner_width = lp.inner_width();
    let expected = num_public_rows
        .checked_mul(inner_width)
        .ok_or(AkitaError::InvalidProof)?;
    let mut out = Vec::with_capacity(expected);
    for point_idx in 0..num_public_rows {
        for block_idx in 0..lp.block_len {
            for commit_digit_idx in 0..lp.num_digits_commit {
                let digits = (0..layout.num_digits_fold)
                    .map(|fold_digit_idx| {
                        let source_idx = layout
                            .offset_z
                            .checked_add(
                                commit_digit_idx
                                    .checked_mul(layout.num_digits_fold)
                                    .and_then(|idx| idx.checked_add(fold_digit_idx))
                                    .and_then(|idx| idx.checked_mul(num_public_rows))
                                    .and_then(|idx| idx.checked_mul(lp.block_len))
                                    .and_then(|idx| {
                                        point_idx
                                            .checked_mul(lp.block_len)
                                            .and_then(|offset| offset.checked_add(block_idx))
                                            .and_then(|offset| idx.checked_add(offset))
                                    })
                                    .ok_or(AkitaError::InvalidProof)?,
                            )
                            .ok_or(AkitaError::InvalidProof)?;
                        plane_at(planes, source_idx)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                out.push(recompose_i8_planes::<F, D>(&digits, lp.log_basis));
            }
        }
    }
    if out.len() != expected {
        return Err(AkitaError::InvalidProof);
    }
    Ok(out)
}

#[cfg(not(feature = "zk"))]
fn ring_dot<F, const D: usize>(
    row: &[CyclotomicRing<F, D>],
    input: &[CyclotomicRing<F, D>],
) -> Result<CyclotomicRing<F, D>, AkitaError>
where
    F: FieldCore,
{
    if input.len() > row.len() {
        return Err(AkitaError::InvalidProof);
    }
    Ok(row
        .iter()
        .zip(input.iter())
        .fold(CyclotomicRing::<F, D>::zero(), |acc, (lhs, rhs)| {
            acc + (*lhs * *rhs)
        }))
}

#[cfg(not(feature = "zk"))]
fn check_terminal_direct_consistency_row<F, const D: usize>(
    w_folded: &[CyclotomicRing<F, D>],
    z_pre: &[CyclotomicRing<F, D>],
    challenge_rings: &[CyclotomicRing<F, D>],
    ring_multiplier_points: &[RingMultiplierOpeningPoint<F, D>],
    lp: &LevelParams,
    num_public_rows: usize,
) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField,
{
    let g_commit = gadget_row_scalars::<F>(lp.num_digits_commit, lp.log_basis);
    let mut folded = CyclotomicRing::<F, D>::zero();
    for (challenge, w) in challenge_rings.iter().zip(w_folded.iter()) {
        folded += *challenge * *w;
    }
    let inner_width = lp.inner_width();
    let mut z_reduced = CyclotomicRing::<F, D>::zero();
    for point_idx in 0..num_public_rows {
        let point = ring_multiplier_points
            .get(point_idx)
            .ok_or(AkitaError::InvalidProof)?;
        for block_idx in 0..lp.block_len {
            let mut z_block = CyclotomicRing::<F, D>::zero();
            for digit_idx in 0..lp.num_digits_commit {
                let z_idx = point_idx
                    .checked_mul(inner_width)
                    .and_then(|idx| {
                        block_idx
                            .checked_mul(lp.num_digits_commit)
                            .and_then(|offset| offset.checked_add(digit_idx))
                            .and_then(|offset| idx.checked_add(offset))
                    })
                    .ok_or(AkitaError::InvalidProof)?;
                let z = z_pre.get(z_idx).ok_or(AkitaError::InvalidProof)?;
                let gadget = g_commit
                    .get(digit_idx)
                    .copied()
                    .ok_or(AkitaError::InvalidProof)?;
                z_block += z.scale(&gadget);
            }
            if let Some(scalar) = point.a_constant_coeff(block_idx) {
                z_reduced += z_block.scale(&scalar);
            } else {
                let multiplier = point
                    .a_rings()
                    .and_then(|rings| rings.get(block_idx))
                    .copied()
                    .ok_or(AkitaError::InvalidProof)?;
                z_reduced += multiplier * z_block;
            }
        }
    }
    if folded != z_reduced {
        return Err(AkitaError::InvalidProof);
    }
    Ok(())
}

#[cfg(not(feature = "zk"))]
fn check_terminal_direct_public_rows<F, const D: usize>(
    w_folded: &[CyclotomicRing<F, D>],
    row_coefficient_rings: &[CyclotomicRing<F, D>],
    ring_multiplier_points: &[RingMultiplierOpeningPoint<F, D>],
    claim_to_point: &[usize],
    y_rings: &[CyclotomicRing<F, D>],
    blocks_per_claim: usize,
) -> Result<(), AkitaError>
where
    F: FieldCore,
{
    for (target_point_idx, expected) in y_rings.iter().enumerate() {
        let mut actual = CyclotomicRing::<F, D>::zero();
        for (claim_idx, &point_idx) in claim_to_point.iter().enumerate() {
            if point_idx != target_point_idx {
                continue;
            }
            let point = ring_multiplier_points
                .get(point_idx)
                .ok_or(AkitaError::InvalidProof)?;
            let coefficient_ring = row_coefficient_rings
                .get(claim_idx)
                .ok_or(AkitaError::InvalidProof)?;
            for block_idx in 0..blocks_per_claim {
                let folded_idx = claim_idx
                    .checked_mul(blocks_per_claim)
                    .and_then(|idx| idx.checked_add(block_idx))
                    .ok_or(AkitaError::InvalidProof)?;
                let folded = w_folded.get(folded_idx).ok_or(AkitaError::InvalidProof)?;
                let weighted_multiplier = if let Some(scalar) = point.b_constant_coeff(block_idx) {
                    coefficient_ring.scale(&scalar)
                } else {
                    let b_ring = point
                        .b_rings()
                        .and_then(|rings| rings.get(block_idx))
                        .copied()
                        .ok_or(AkitaError::InvalidProof)?;
                    *coefficient_ring * b_ring
                };
                actual += weighted_multiplier * *folded;
            }
        }
        if &actual != expected {
            return Err(AkitaError::InvalidProof);
        }
    }
    Ok(())
}

#[cfg(not(feature = "zk"))]
fn check_terminal_direct_b_rows<F, const D: usize>(
    setup: &AkitaExpandedSetup<F>,
    t_digits: &[CyclotomicRing<F, D>],
    lp: &LevelParams,
    num_polys_per_point: &[usize],
    commitment_rows: &[CyclotomicRing<F, D>],
) -> Result<(), AkitaError>
where
    F: FieldCore,
{
    let n_b = lp.b_key.row_len();
    let n_a = lp.a_key.row_len();
    let per_poly_cols = lp
        .num_blocks
        .checked_mul(n_a)
        .and_then(|cols| cols.checked_mul(lp.num_digits_open))
        .ok_or(AkitaError::InvalidProof)?;
    let max_point_poly_count = num_polys_per_point.iter().copied().max().unwrap_or(0);
    let b_stride = max_point_poly_count
        .checked_mul(per_poly_cols)
        .ok_or(AkitaError::InvalidProof)?;
    let b_view = setup.shared_matrix().ring_view::<D>(n_b, b_stride)?;
    let mut group_offset = 0usize;
    for (point_idx, &group_size) in num_polys_per_point.iter().enumerate() {
        let group_len = group_size
            .checked_mul(per_poly_cols)
            .ok_or(AkitaError::InvalidProof)?;
        let start = group_offset
            .checked_mul(per_poly_cols)
            .ok_or(AkitaError::InvalidProof)?;
        let end = start
            .checked_add(group_len)
            .ok_or(AkitaError::InvalidProof)?;
        let group_t = t_digits.get(start..end).ok_or(AkitaError::InvalidProof)?;
        for b_idx in 0..n_b {
            let actual = ring_dot(b_view.row(b_idx)?, group_t)?;
            let commitment_idx = point_idx
                .checked_mul(n_b)
                .and_then(|idx| idx.checked_add(b_idx))
                .ok_or(AkitaError::InvalidProof)?;
            if commitment_rows
                .get(commitment_idx)
                .ok_or(AkitaError::InvalidProof)?
                != &actual
            {
                return Err(AkitaError::InvalidProof);
            }
        }
        group_offset = group_offset
            .checked_add(group_size)
            .ok_or(AkitaError::InvalidProof)?;
    }
    Ok(())
}

#[cfg(not(feature = "zk"))]
#[allow(clippy::too_many_arguments)]
fn check_terminal_direct_a_rows<F, const D: usize>(
    setup: &AkitaExpandedSetup<F>,
    t_recomposed_rows: &[Vec<CyclotomicRing<F, D>>],
    z_pre: &[CyclotomicRing<F, D>],
    challenge_rings: &[CyclotomicRing<F, D>],
    lp: &LevelParams,
    num_polys_per_point: &[usize],
    claim_to_point_poly: &[usize],
    claim_poly_indices: &[usize],
    num_public_rows: usize,
) -> Result<(), AkitaError>
where
    F: FieldCore,
{
    let n_a = lp.a_key.row_len();
    let inner_width = lp.inner_width();
    let a_view = setup.shared_matrix().ring_view::<D>(n_a, inner_width)?;
    let mut group_offsets = Vec::with_capacity(num_polys_per_point.len());
    let mut offset = 0usize;
    for &count in num_polys_per_point {
        group_offsets.push(offset);
        offset = offset.checked_add(count).ok_or(AkitaError::InvalidProof)?;
    }
    for a_idx in 0..n_a {
        let mut lhs = CyclotomicRing::<F, D>::zero();
        for (challenge_idx, challenge) in challenge_rings.iter().enumerate() {
            let claim_idx = challenge_idx
                .checked_div(lp.num_blocks)
                .ok_or(AkitaError::InvalidProof)?;
            let block_idx = challenge_idx % lp.num_blocks;
            let point_idx = *claim_to_point_poly
                .get(claim_idx)
                .ok_or(AkitaError::InvalidProof)?;
            let poly_idx = *claim_poly_indices
                .get(claim_idx)
                .ok_or(AkitaError::InvalidProof)?;
            let poly_slot = group_offsets
                .get(point_idx)
                .and_then(|base| base.checked_add(poly_idx))
                .ok_or(AkitaError::InvalidProof)?;
            let inner_idx = poly_slot
                .checked_mul(lp.num_blocks)
                .and_then(|idx| idx.checked_add(block_idx))
                .ok_or(AkitaError::InvalidProof)?;
            let row_value = t_recomposed_rows
                .get(inner_idx)
                .and_then(|rows| rows.get(a_idx))
                .copied()
                .ok_or(AkitaError::InvalidProof)?;
            lhs += *challenge * row_value;
        }
        let mut rhs = CyclotomicRing::<F, D>::zero();
        let row = a_view.row(a_idx)?;
        for point_idx in 0..num_public_rows {
            let start = point_idx
                .checked_mul(inner_width)
                .ok_or(AkitaError::InvalidProof)?;
            let end = start
                .checked_add(inner_width)
                .ok_or(AkitaError::InvalidProof)?;
            let z_segment = z_pre.get(start..end).ok_or(AkitaError::InvalidProof)?;
            rhs += ring_dot(row, z_segment)?;
        }
        if lhs != rhs {
            return Err(AkitaError::InvalidProof);
        }
    }
    Ok(())
}
