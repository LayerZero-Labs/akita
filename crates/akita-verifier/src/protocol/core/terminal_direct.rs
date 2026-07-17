//! Deterministic terminal checks over the revealed segment-typed witness.

use super::direct_ring_arithmetic::decompose_rows_i8;
use akita_algebra::CyclotomicRing;
use akita_challenges::{Challenges, SparseChallenge};
use akita_field::{
    AkitaError, CanonicalField, ExtField, FieldCore, FromPrimitiveInt, HalvingField,
};
use akita_types::{
    decode_terminal_z_golomb_payload_with_cap, dispatch_for_field,
    recover_ring_subfield_inner_product, AkitaVerifierSetup, CleartextWitnessProof, FpExtEncoding,
    LevelParams, LevelParamsLike, PreparedOpeningPoint, RelationMatrixRowLayout,
    RingRelationInstance, RingVec,
};

fn sparse_challenge_ring<F, const D: usize>(
    challenge: &SparseChallenge,
) -> Result<CyclotomicRing<F, D>, AkitaError>
where
    F: FieldCore + FromPrimitiveInt,
{
    challenge.validate::<D>()?;
    let mut coeffs = [F::zero(); D];
    for (&position, &coefficient) in challenge.positions.iter().zip(&challenge.coeffs) {
        let slot = coeffs
            .get_mut(position as usize)
            .ok_or(AkitaError::InvalidProof)?;
        *slot += F::from_i64(i64::from(coefficient));
    }
    Ok(CyclotomicRing::from_coefficients(coeffs))
}

fn challenge_rings<F, const D: usize>(
    challenges: &Challenges,
) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>
where
    F: FieldCore + FromPrimitiveInt,
{
    match challenges {
        Challenges::Sparse { challenges, .. } => challenges
            .iter()
            .map(sparse_challenge_ring::<F, D>)
            .collect(),
        Challenges::Tensor { factored } => {
            factored.validate::<D>()?;
            (0..factored.total_blocks()?)
                .map(|index| {
                    let (_, _, high, low) = factored.factors_for_logical_block(index)?;
                    Ok(sparse_challenge_ring::<F, D>(high)? * sparse_challenge_ring::<F, D>(low)?)
                })
                .collect()
        }
    }
}

fn ring_dot<F, const D: usize>(
    row: &[CyclotomicRing<F, D>],
    input: &[CyclotomicRing<F, D>],
) -> Result<CyclotomicRing<F, D>, AkitaError>
where
    F: FieldCore,
{
    if row.len() != input.len() {
        return Err(AkitaError::InvalidProof);
    }
    Ok(row
        .iter()
        .zip(input)
        .fold(CyclotomicRing::zero(), |sum, (lhs, rhs)| {
            sum + (*lhs * *rhs)
        }))
}

fn decode_rings<F, const D: usize>(coeffs: &[F]) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>
where
    F: FieldCore,
{
    if !coeffs.len().is_multiple_of(D) {
        return Err(AkitaError::InvalidProof);
    }
    Ok(coeffs
        .chunks_exact(D)
        .map(CyclotomicRing::from_slice)
        .collect())
}

fn decode_centered_rings<F, const D: usize>(
    coeffs: &[i64],
) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>
where
    F: FieldCore + FromPrimitiveInt,
{
    if !coeffs.len().is_multiple_of(D) {
        return Err(AkitaError::InvalidProof);
    }
    Ok(coeffs
        .chunks_exact(D)
        .map(|chunk| {
            CyclotomicRing::from_coefficients(std::array::from_fn(|index| {
                F::from_i64(chunk[index])
            }))
        })
        .collect())
}

#[tracing::instrument(skip_all, name = "terminal_direct_a_rows")]
fn check_a_rows<F, const D: usize>(
    setup: &AkitaVerifierSetup<F>,
    t: &[CyclotomicRing<F, D>],
    z: &[[i64; D]],
    challenges: &[CyclotomicRing<F, D>],
    params: &dyn LevelParamsLike,
    prepared_prefix_len: usize,
) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt,
{
    let n_a = params.a_rows_len();
    if t.len()
        != challenges
            .len()
            .checked_mul(n_a)
            .ok_or(AkitaError::InvalidProof)?
    {
        return Err(AkitaError::InvalidProof);
    }
    let rhs = super::terminal_ntt::centered_rows(setup, n_a, z, prepared_prefix_len)?;
    for row_index in 0..n_a {
        let lhs = challenges
            .iter()
            .zip(t.chunks_exact(n_a))
            .try_fold(CyclotomicRing::zero(), |sum, (challenge, rows)| {
                Ok::<_, AkitaError>(sum + (*challenge * rows[row_index]))
            })?;
        if lhs != *rhs.get(row_index).ok_or(AkitaError::InvalidProof)? {
            return Err(AkitaError::InvalidProof);
        }
    }
    Ok(())
}

#[tracing::instrument(skip_all, name = "terminal_direct_b_rows")]
fn check_b_rows<F, const D_A: usize, const D_B: usize>(
    setup: &AkitaVerifierSetup<F>,
    t: &[CyclotomicRing<F, D_A>],
    expected: &[CyclotomicRing<F, D_B>],
    params: &dyn LevelParamsLike,
    prepared_prefix_len: usize,
) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField,
{
    let t_digits = decompose_rows_i8(t, params.num_digits_open(), params.log_basis());
    let flat = t_digits.as_flattened();
    if !flat.len().is_multiple_of(D_B) {
        return Err(AkitaError::InvalidProof);
    }
    let (outer_digits, remainder) = flat.as_chunks::<D_B>();
    if !remainder.is_empty() || outer_digits.len() != params.b_col_len() {
        return Err(AkitaError::InvalidProof);
    }
    let actual = super::terminal_ntt::digit_rows(
        setup,
        params.b_rows_len(),
        outer_digits,
        params.log_basis(),
        prepared_prefix_len,
    )?;
    (actual.as_slice() == expected)
        .then_some(())
        .ok_or(AkitaError::InvalidProof)
}

/// Check reduced consistency, A, and B rows for a quotient-free terminal witness.
#[tracing::instrument(skip_all, name = "terminal_direct_ring_relations")]
pub(super) fn verify_terminal_ring_relations<F>(
    setup: &AkitaVerifierSetup<F>,
    relation: &RingRelationInstance<F>,
    lp: &LevelParams,
    commitment_rows: &RingVec<F>,
    final_witness: &CleartextWitnessProof<F>,
) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HalvingField,
{
    let witness = final_witness
        .as_segment_typed()
        .ok_or(AkitaError::InvalidProof)?;
    if witness.layout.ring_dimension != relation.role_dims().d_a()
        || witness.layout.groups.len() != relation.opening_batch().num_groups()
    {
        return Err(AkitaError::InvalidProof);
    }
    let order = relation.opening_batch().root_group_order()?;
    let mut max_a_prefix_len = 0usize;
    let mut max_b_prefix_len = 0usize;
    for &group_index in &order {
        let params = lp.group_params(relation.opening_batch(), group_index)?;
        max_a_prefix_len = max_a_prefix_len.max(
            params
                .a_rows_len()
                .checked_mul(params.a_col_len())
                .ok_or(AkitaError::InvalidProof)?,
        );
        max_b_prefix_len = max_b_prefix_len.max(
            params
                .b_rows_len()
                .checked_mul(params.b_col_len())
                .ok_or(AkitaError::InvalidProof)?,
        );
    }
    let (prepared_a_prefix_len, prepared_b_prefix_len) =
        if relation.role_dims().d_a() == relation.role_dims().d_b() {
            let shared = max_a_prefix_len.max(max_b_prefix_len);
            (shared, shared)
        } else {
            (max_a_prefix_len, max_b_prefix_len)
        };
    let mut e_offset = 0usize;
    let mut t_offset = 0usize;
    let mut commitment_offset = 0usize;
    dispatch_for_field!(
        akita_types::ProtocolDispatchSlot::Role(akita_types::RingRole::Inner),
        F,
        relation.role_dims().d_a(),
        |D_A| {
            let mut consistency_lhs = CyclotomicRing::<F, D_A>::zero();
            let mut consistency_rhs = CyclotomicRing::<F, D_A>::zero();
            for (layout_index, &group_index) in order.iter().enumerate() {
                let params = lp.group_params(relation.opening_batch(), group_index)?;
                let group_layout = witness
                    .layout
                    .groups
                    .get(layout_index)
                    .ok_or(AkitaError::InvalidProof)?;
                let num_polynomials = relation
                    .opening_batch()
                    .group_layout(group_index)?
                    .num_polynomials();
                let a_range = lp.a_row_range(
                    relation.opening_batch(),
                    group_index,
                    RelationMatrixRowLayout::WithoutDBlock,
                )?;
                let b_range = lp.commitment_row_range(
                    relation.opening_batch(),
                    group_index,
                    RelationMatrixRowLayout::WithoutDBlock,
                )?;
                if a_range.len() != params.a_rows_len()
                    || b_range.len() != params.b_rows_len()
                    || b_range.start != a_range.end
                {
                    return Err(AkitaError::InvalidSetup(
                        "terminal direct row ranges do not match group matrix heights".to_string(),
                    ));
                }
                let num_blocks = num_polynomials
                    .checked_mul(params.num_live_blocks())
                    .ok_or(AkitaError::InvalidProof)?;
                let e_end = e_offset
                    .checked_add(group_layout.e_field_elems)
                    .ok_or(AkitaError::InvalidProof)?;
                let t_end = t_offset
                    .checked_add(group_layout.t_field_elems)
                    .ok_or(AkitaError::InvalidProof)?;
                let e = decode_rings::<F, D_A>(
                    witness
                        .e_fields
                        .coeffs()
                        .get(e_offset..e_end)
                        .ok_or(AkitaError::InvalidProof)?,
                )?;
                let t = decode_rings::<F, D_A>(
                    witness
                        .t_fields
                        .coeffs()
                        .get(t_offset..t_end)
                        .ok_or(AkitaError::InvalidProof)?,
                )?;
                let cap = lp.fold_witness_linf_cap_for_params(
                    params,
                    num_polynomials,
                    F::modulus_bits(),
                )?;
                let z_values = decode_terminal_z_golomb_payload_with_cap(
                    witness
                        .z_payloads
                        .get(layout_index)
                        .ok_or(AkitaError::InvalidProof)?,
                    group_layout.z_coords,
                    cap,
                    Some(group_layout.z_payload_bytes),
                )?;
                let z_centered = {
                    if !z_values.len().is_multiple_of(D_A) {
                        return Err(AkitaError::InvalidProof);
                    }
                    let (rings, remainder) = z_values.as_chunks::<D_A>();
                    if !remainder.is_empty() {
                        return Err(AkitaError::InvalidProof);
                    }
                    rings
                };
                let z = decode_centered_rings::<F, D_A>(&z_values)?;
                let challenges = challenge_rings::<F, D_A>(
                    relation
                        .group_challenges()
                        .get(group_index)
                        .ok_or(AkitaError::InvalidProof)?,
                )?;
                let expected_t_len = num_blocks
                    .checked_mul(params.a_rows_len())
                    .ok_or(AkitaError::InvalidProof)?;
                if e.len() != num_blocks || t.len() != expected_t_len {
                    return Err(AkitaError::InvalidProof);
                }
                let (folded, reduced) = {
                    let _span = tracing::info_span!(
                        "terminal_direct_consistency",
                        group_index,
                        num_blocks,
                        num_positions = params.num_positions_per_block()
                    )
                    .entered();
                    let multiplier = relation.group_ring_multiplier_point(group_index)?;
                    let folded = ring_dot(&challenges, &e)?;
                    let gadget = akita_types::gadget_row_scalars::<F>(
                        params.num_digits_commit(),
                        params.log_basis(),
                    );
                    let mut reduced = CyclotomicRing::zero();
                    for position in 0..params.num_positions_per_block() {
                        let start = position
                            .checked_mul(params.num_digits_commit())
                            .ok_or(AkitaError::InvalidProof)?;
                        let mut z_value = CyclotomicRing::zero();
                        for digit in 0..params.num_digits_commit() {
                            let index = start.checked_add(digit).ok_or(AkitaError::InvalidProof)?;
                            z_value += z
                                .get(index)
                                .ok_or(AkitaError::InvalidProof)?
                                .scale(gadget.get(digit).ok_or(AkitaError::InvalidProof)?);
                        }
                        if let Some(scale) = multiplier.position_constant_coeff(position) {
                            reduced += z_value.scale(&scale);
                        } else {
                            reduced += *multiplier
                                .position_rings_trusted::<D_A>()?
                                .ok_or(AkitaError::InvalidProof)?
                                .get(position)
                                .ok_or(AkitaError::InvalidProof)?
                                * z_value;
                        }
                    }
                    (folded, reduced)
                };
                consistency_lhs += folded;
                consistency_rhs += reduced;
                check_a_rows::<F, D_A>(
                    setup,
                    &t,
                    z_centered,
                    &challenges,
                    params,
                    prepared_a_prefix_len,
                )?;

                dispatch_for_field!(
                    akita_types::ProtocolDispatchSlot::Role(akita_types::RingRole::Outer),
                    F,
                    relation.role_dims().d_b(),
                    |D_B| {
                        let commitment_coeffs = b_range
                            .len()
                            .checked_mul(D_B)
                            .ok_or(AkitaError::InvalidProof)?;
                        let end = commitment_offset
                            .checked_add(commitment_coeffs)
                            .ok_or(AkitaError::InvalidProof)?;
                        let expected = decode_rings::<F, D_B>(
                            commitment_rows
                                .coeffs()
                                .get(commitment_offset..end)
                                .ok_or(AkitaError::InvalidProof)?,
                        )?;
                        check_b_rows::<F, D_A, D_B>(
                            setup,
                            &t,
                            &expected,
                            params,
                            prepared_b_prefix_len,
                        )?;
                        commitment_offset = end;
                        Ok::<(), AkitaError>(())
                    }
                )?;
                e_offset = e_end;
                t_offset = t_end;
            }
            if consistency_lhs != consistency_rhs {
                return Err(AkitaError::InvalidProof);
            }
            Ok::<(), AkitaError>(())
        }
    )?;
    if e_offset != witness.e_fields.coeff_len()
        || t_offset != witness.t_fields.coeff_len()
        || commitment_offset != commitment_rows.coeff_len()
    {
        return Err(AkitaError::InvalidProof);
    }
    Ok(())
}

/// Check the public opening directly against the revealed folded `e` segment.
#[allow(clippy::too_many_arguments)]
#[tracing::instrument(skip_all, name = "terminal_direct_trace")]
pub(super) fn verify_terminal_trace<F, E>(
    relation: &RingRelationInstance<F>,
    lp: &LevelParams,
    final_witness: &CleartextWitnessProof<F>,
    prepared_points: &[PreparedOpeningPoint<F, E>],
    row_coefficients: &[E],
    claim_scales: Option<&[E]>,
    global_scale: E,
    target: E,
) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt,
    E: ExtField<F> + FpExtEncoding<F>,
{
    let witness = final_witness
        .as_segment_typed()
        .ok_or(AkitaError::InvalidProof)?;
    if prepared_points.len() != relation.opening_batch().num_groups()
        || row_coefficients.len() != relation.opening_batch().num_total_polynomials()
        || claim_scales.is_some_and(|scales| scales.len() != row_coefficients.len())
    {
        return Err(AkitaError::InvalidProof);
    }
    let order = relation.opening_batch().root_group_order()?;
    let mut e_offset = 0usize;
    let mut actual = E::zero();
    dispatch_for_field!(
        akita_types::ProtocolDispatchSlot::Role(akita_types::RingRole::Inner),
        F,
        relation.role_dims().d_a(),
        |D| {
            for (layout_index, &group_index) in order.iter().enumerate() {
                let params = lp.group_params(relation.opening_batch(), group_index)?;
                let group_layout = witness
                    .layout
                    .groups
                    .get(layout_index)
                    .ok_or(AkitaError::InvalidProof)?;
                let end = e_offset
                    .checked_add(group_layout.e_field_elems)
                    .ok_or(AkitaError::InvalidProof)?;
                let e = decode_rings::<F, D>(
                    witness
                        .e_fields
                        .coeffs()
                        .get(e_offset..end)
                        .ok_or(AkitaError::InvalidProof)?,
                )?;
                let claim_range = relation
                    .opening_batch()
                    .root_group_claim_range(group_index)?;
                let multiplier = relation.group_ring_multiplier_point(group_index)?;
                let packed_inner = prepared_points
                    .get(group_index)
                    .ok_or(AkitaError::InvalidProof)?
                    .packed_inner_trusted::<D>()?;
                for (local_claim, claim_index) in claim_range.enumerate() {
                    let start = local_claim
                        .checked_mul(params.num_live_blocks())
                        .ok_or(AkitaError::InvalidProof)?;
                    let end = start
                        .checked_add(params.num_live_blocks())
                        .ok_or(AkitaError::InvalidProof)?;
                    let claim_e = e.get(start..end).ok_or(AkitaError::InvalidProof)?;
                    let mut outer_eval = CyclotomicRing::zero();
                    for (block, value) in claim_e.iter().enumerate() {
                        if let Some(scale) = multiplier.fold_constant_coeff(block) {
                            outer_eval += value.scale(&scale);
                        } else {
                            outer_eval += *multiplier
                                .fold_rings_trusted::<D>()?
                                .ok_or(AkitaError::InvalidProof)?
                                .get(block)
                                .ok_or(AkitaError::InvalidProof)?
                                * *value;
                        }
                    }
                    let opening =
                        recover_ring_subfield_inner_product::<F, E, D>(&outer_eval, packed_inner)?;
                    let scale = claim_scales
                        .and_then(|scales| scales.get(claim_index))
                        .copied()
                        .unwrap_or(global_scale);
                    actual += *row_coefficients
                        .get(claim_index)
                        .ok_or(AkitaError::InvalidProof)?
                        * scale
                        * opening;
                }
                e_offset = end;
            }
            Ok::<(), AkitaError>(())
        }
    )?;
    if e_offset != witness.e_fields.coeff_len() || actual != target {
        return Err(AkitaError::InvalidProof);
    }
    Ok(())
}
