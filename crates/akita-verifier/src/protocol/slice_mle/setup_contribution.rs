use akita_algebra::eq_poly::EqPolynomial;
use akita_algebra::offset_eq::eq_eval_at_index;
use akita_algebra::ring::eval_ring_at_pows;
use akita_field::parallel::*;
use akita_field::{AkitaError, CanonicalField, ExtField, FieldCore};
use akita_types::AkitaExpandedSetup;

use super::structured_slice::POSSIBLE_CARRIES;
#[cfg(test)]
use crate::protocol::ring_switch::PreparedChallengeEvals;
use crate::protocol::ring_switch::RingSwitchDeferredRowEval;

/// Translate a D-column (D-physical order `[digit, block, claim]`) into
/// the M-layout `(low_block_eq_idx, high_eq_idx)` pair.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn get_eq_indices_for_d(
    current_index: usize,
    num_digits: usize,
    num_blocks: usize,
    num_claims: usize,
    blocks_per_claim_w: usize,
    block_offset_low: usize,
    block_mask: usize,
    block_bits: usize,
) -> (usize, usize) {
    let digit_idx = current_index % num_digits;
    let block_idx = (current_index / num_digits) % num_blocks;
    let claim_idx = current_index / blocks_per_claim_w;
    let m_layout_high_idx = digit_idx * num_claims + claim_idx;
    let block_sum = block_offset_low + block_idx;
    let low_eq_idx = block_sum & block_mask;
    let block_carry = block_sum >> block_bits;
    let high_eq_idx = m_layout_high_idx + block_carry;
    (low_eq_idx, high_eq_idx)
}

/// Translate a B-column (B-physical order `[digit, a_row, block, t_vector]`)
/// into `(low_block_eq_idx, high_eq_idx)`. `flat_t_vector` resolves the
/// per-group polynomial slot to the global t-vector index used by the high
/// index.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn get_eq_indices_for_b(
    current_index: usize,
    flat_t_vector: usize,
    num_digits: usize,
    n_a: usize,
    num_blocks: usize,
    num_t_vectors: usize,
    stride_t: usize,
    block_offset_low: usize,
    block_mask: usize,
    block_bits: usize,
) -> (usize, usize) {
    let digit_idx = current_index % num_digits;
    let a_row_idx = (current_index / num_digits) % n_a;
    let block_idx = (current_index / stride_t) % num_blocks;
    let m_layout_high_idx =
        flat_t_vector + num_t_vectors * digit_idx + num_t_vectors * num_digits * a_row_idx;
    let block_sum = block_offset_low + block_idx;
    let low_eq_idx = block_sum & block_mask;
    let block_carry = block_sum >> block_bits;
    let high_eq_idx = m_layout_high_idx + block_carry;
    (low_eq_idx, high_eq_idx)
}

/// Translate an A-column (A-physical order `[dc, block]`) into the
/// `(low_block_eq_idx, dc_idx, block_carry)` triple used to index
/// `z_block_low_eq` and the precomputed `s_per_dc_per_carry` table.
#[inline(always)]
fn get_eq_indices_for_a(
    current_index: usize,
    depth_commit: usize,
    z_offset_low: usize,
    z_block_mask: usize,
    z_offset_low_bits: usize,
) -> (usize, usize, usize) {
    let block_idx = current_index / depth_commit;
    let depth_commit_idx = current_index % depth_commit;
    let block_sum = z_offset_low + block_idx;
    let low_eq_idx = block_sum & z_block_mask;
    let block_carry = block_sum >> z_offset_low_bits;
    (low_eq_idx, depth_commit_idx, block_carry)
}

/// Sum `Σ r_eval[c] · (Σ_{p ∈ active} weight_p · pattern_p[c])` over one
/// contiguous column slice. The const generics select which of `{W, T, Z}`
/// is active — the compiler strips the inactive arms at monomorphisation.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn slice_inner_sum<F, E, const HAS_W: bool, const HAS_T: bool, const HAS_Z: bool>(
    range: std::ops::Range<usize>,
    r_eval: &[E],
    d_w: E,
    w_eq: &[E],
    b_w_for_groups: &[E],
    t_eq_per_group: &[Vec<E>],
    num_groups: usize,
    a_w: E,
    z_eq: &[E],
) -> E
where
    F: FieldCore,
    E: ExtField<F>,
{
    cfg_into_iter!(range)
        .map(|c| {
            let mut m = E::zero();
            if HAS_W {
                m += d_w * w_eq[c];
            }
            if HAS_T {
                for g in 0..num_groups {
                    m += b_w_for_groups[g] * t_eq_per_group[g][c];
                }
            }
            if HAS_Z {
                m += a_w * z_eq[c];
            }
            r_eval[c] * m
        })
        .sum()
}

/// Compute the packed setup-matrix contribution `D · ŵ + B · t̂ + A · ẑ`
/// as one scalar over the shared SIS matrix prefix. The D, B, and A roles use
/// their natural packed widths; physical aliasing between role prefixes is
/// represented by summing the role contributions that land on the same raw
/// setup entry.
#[allow(clippy::too_many_arguments)]
pub(crate) fn compute_setup_contribution<F, E, const D: usize>(
    prepared: &RingSwitchDeferredRowEval<E>,
    full_vec_randomness: &[E],
    setup: &AkitaExpandedSetup<F>,
    eq_low: &[E],
    z_block_low_eq: &[E],
    alpha_pows: &[E],
    fold_gadget: &[F],
    offset_w: usize,
    offset_t: usize,
    offset_z: usize,
) -> Result<E, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: ExtField<F>,
{
    if prepared.num_blocks == 0 || !prepared.num_blocks.is_power_of_two() {
        return Err(AkitaError::InvalidSetup(
            "num_blocks must be a non-zero power of two".to_string(),
        ));
    }
    if prepared.block_len == 0 || prepared.depth_commit == 0 {
        return Err(AkitaError::InvalidSetup(
            "Z layout requires non-zero block length and commit depth".to_string(),
        ));
    }
    let block_bits = prepared.num_blocks.trailing_zeros() as usize;
    if block_bits > full_vec_randomness.len() {
        return Err(AkitaError::InvalidSize {
            expected: block_bits,
            actual: full_vec_randomness.len(),
        });
    }
    let block_mask = prepared.num_blocks - 1;
    let block_offset_low = offset_w & block_mask;
    let w_offset_high = offset_w >> block_bits;
    let t_offset_high = offset_t >> block_bits;
    let high_challenges = &full_vec_randomness[block_bits..];

    let z_offset_low_bits = prepared.block_len.trailing_zeros() as usize;
    if z_offset_low_bits > full_vec_randomness.len() {
        return Err(AkitaError::InvalidSize {
            expected: z_offset_low_bits,
            actual: full_vec_randomness.len(),
        });
    }
    let z_offset_low = offset_z & prepared.block_len.saturating_sub(1);
    let z_range = prepared.inner_width;
    let z_used = prepared.n_a > 0 && z_range > 0;
    let z_dims_pow2 = prepared.block_len.is_power_of_two();

    let n_d_active = prepared.n_d_active();
    let d_start = 1usize
        .checked_add(prepared.num_public_rows)
        .ok_or_else(|| AkitaError::InvalidSetup("D row start overflow".to_string()))?;
    let b_start = d_start
        .checked_add(n_d_active)
        .ok_or_else(|| AkitaError::InvalidSetup("B row start overflow".to_string()))?;
    let a_start = b_start
        .checked_add(
            prepared
                .n_b
                .checked_mul(prepared.num_points)
                .ok_or_else(|| AkitaError::InvalidSetup("B row width overflow".to_string()))?,
        )
        .ok_or_else(|| AkitaError::InvalidSetup("A row start overflow".to_string()))?;
    if d_start
        .checked_add(n_d_active)
        .is_none_or(|end| end > prepared.eq_tau1.len())
        || a_start > prepared.rows
        || prepared.rows > prepared.eq_tau1.len()
    {
        return Err(AkitaError::InvalidSetup(
            "M-row weights are inconsistent with verifier layout".to_string(),
        ));
    }
    let d_weights = &prepared.eq_tau1[d_start..(d_start + n_d_active)];
    let a_weights = &prepared.eq_tau1[a_start..prepared.rows];

    let stride_t = prepared
        .n_a
        .checked_mul(prepared.depth_open)
        .ok_or_else(|| AkitaError::InvalidSetup("T stride overflow".to_string()))?;
    let cols_per_poly_t = stride_t
        .checked_mul(prepared.num_blocks)
        .ok_or_else(|| AkitaError::InvalidSetup("T column width overflow".to_string()))?;
    let b_per_claim_w = prepared
        .num_blocks
        .checked_mul(prepared.depth_open)
        .ok_or_else(|| AkitaError::InvalidSetup("W claim width overflow".to_string()))?;
    let n_cols_w = prepared
        .num_claims
        .checked_mul(b_per_claim_w)
        .ok_or_else(|| AkitaError::InvalidSetup("W column width overflow".to_string()))?;

    // T's row weight is group-dependent and its c-axis indexes `poly_idx`
    // within the group. Its M-layout high index, however, is the global
    // t-vector slot `Σ_{h<g} num_polys_per_point[h] + poly_idx`, so sizing
    // follows `num_polys_per_point` rather than the number of opened claims.
    let max_group_poly_count = prepared
        .num_polys_per_point
        .iter()
        .copied()
        .max()
        .unwrap_or(0);
    let mut group_offsets = Vec::with_capacity(prepared.num_polys_per_point.len());
    let mut next_offset = 0usize;
    for &group_poly_count in &prepared.num_polys_per_point {
        group_offsets.push(next_offset);
        next_offset += group_poly_count;
    }
    let n_cols_t = max_group_poly_count
        .checked_mul(cols_per_poly_t)
        .ok_or_else(|| AkitaError::InvalidSetup("T column width overflow".to_string()))?;

    if (n_d_active == 0 || n_cols_w == 0)
        && (prepared.n_b == 0 || n_cols_t == 0)
        && (!z_used || prepared.n_a == 0)
    {
        return Err(AkitaError::InvalidSetup(
            "matrix-row pattern evaluation requires at least one active setup role".to_string(),
        ));
    }

    let w_hi_len = prepared
        .num_claims
        .checked_mul(prepared.depth_open)
        .ok_or_else(|| AkitaError::InvalidSetup("W high-eq width overflow".to_string()))?;
    let t_hi_len = prepared
        .num_t_vectors
        .checked_mul(prepared.depth_open)
        .and_then(|len| len.checked_mul(prepared.n_a))
        .ok_or_else(|| AkitaError::InvalidSetup("T high-eq width overflow".to_string()))?;
    let eq_hi_w_table: Vec<E> = (0..=w_hi_len)
        .map(|k| eq_eval_at_index(high_challenges, w_offset_high + k))
        .collect();
    let eq_hi_t_table: Vec<E> = (0..=t_hi_len)
        .map(|k| eq_eval_at_index(high_challenges, t_offset_high + k))
        .collect();
    let w_eq_slice: Vec<E> = cfg_into_iter!(0..n_cols_w)
        .map(|current_index| {
            let (low_eq_idx, high_eq_idx) = get_eq_indices_for_d(
                current_index,
                prepared.depth_open,
                prepared.num_blocks,
                prepared.num_claims,
                b_per_claim_w,
                block_offset_low,
                block_mask,
                block_bits,
            );
            eq_low[low_eq_idx] * eq_hi_w_table[high_eq_idx]
        })
        .collect();

    let t_eq_slice_per_group: Vec<Vec<E>> = (0..prepared.num_points)
        .map(|g| {
            let group_size = prepared.num_polys_per_point[g];
            cfg_into_iter!(0..n_cols_t)
                .map(|c| {
                    let poly_idx = c / cols_per_poly_t;
                    if poly_idx >= group_size {
                        return E::zero();
                    }
                    let flat_t_vector = group_offsets[g] + poly_idx;
                    let (low_eq_idx, high_eq_idx) = get_eq_indices_for_b(
                        c,
                        flat_t_vector,
                        prepared.depth_open,
                        prepared.n_a,
                        prepared.num_blocks,
                        prepared.num_t_vectors,
                        stride_t,
                        block_offset_low,
                        block_mask,
                        block_bits,
                    );
                    eq_low[low_eq_idx] * eq_hi_t_table[high_eq_idx]
                })
                .collect()
        })
        .collect();

    // `z_eq_slice[c]` — column-only Z pattern. Length `z_range`, empty
    // when `!z_used`. Pow2: peeled-block lookup `z_block_low_eq[low] ·
    // S_per_dc_per_carry[dc][carry]`. Non-pow2: dense aggregation over
    // `(pt, df)` with a one-shot peeled eq cache so per-cell cost stays
    // O(P · DF).
    let z_eq_slice: Vec<E> = if !z_used {
        Vec::new()
    } else if z_dims_pow2 {
        // `S_per_dc_per_carry[dc][carry] = -Σ_{pt, df} fold_gadget[df]
        //   · eq_hi_z[z_offset_high + (pt + P·df + P·DF·dc) + carry]`
        let z_offset_high = offset_z >> z_offset_low_bits;
        let z_block_mask = prepared.block_len.wrapping_sub(1);
        let s_per_dc_per_carry: Vec<[E; POSSIBLE_CARRIES]> = {
            let z_high_challenges = &full_vec_randomness[z_offset_low_bits..];
            let num_q_z = prepared.num_points * prepared.depth_fold * prepared.depth_commit;
            let eq_hi_z_table: Vec<E> = (0..=num_q_z)
                .map(|k| eq_eval_at_index(z_high_challenges, z_offset_high + k))
                .collect();
            (0..prepared.depth_commit)
                .map(|dc| {
                    let mut s = [E::zero(); POSSIBLE_CARRIES];
                    for (carry_slot, slot) in s.iter_mut().enumerate() {
                        let mut acc = E::zero();
                        for (df, &fg) in fold_gadget.iter().enumerate().take(prepared.depth_fold) {
                            for pt in 0..prepared.num_points {
                                let k = pt
                                    + prepared.num_points * df
                                    + prepared.num_points * prepared.depth_fold * dc
                                    + carry_slot;
                                acc += eq_hi_z_table[k].mul_base(fg);
                            }
                        }
                        *slot = -acc;
                    }
                    s
                })
                .collect()
        };
        cfg_into_iter!(0..z_range)
            .map(|c| {
                let (low_eq_idx, depth_commit_idx, block_carry) = get_eq_indices_for_a(
                    c,
                    prepared.depth_commit,
                    z_offset_low,
                    z_block_mask,
                    z_offset_low_bits,
                );
                z_block_low_eq[low_eq_idx] * s_per_dc_per_carry[depth_commit_idx][block_carry]
            })
            .collect()
    } else {
        // Build a peeled eq cache so each per-cell `eq(r, offset_z +
        // j_M^Z)` is O(1) instead of O(|r|).
        let z_total_blocks_dense = prepared
            .block_len
            .checked_mul(prepared.num_points)
            .ok_or_else(|| AkitaError::InvalidSetup("dense Z block width overflow".to_string()))?;
        let z_len_dense = prepared
            .depth_fold
            .checked_mul(prepared.depth_commit)
            .and_then(|len| len.checked_mul(z_total_blocks_dense))
            .ok_or_else(|| AkitaError::InvalidSetup("dense Z length overflow".to_string()))?;
        let n_rand = full_vec_randomness.len();
        let k = z_len_dense
            .saturating_sub(1)
            .checked_next_power_of_two()
            .map(|p| p.trailing_zeros() as usize)
            .unwrap_or(0)
            .max(1)
            .min(n_rand);
        let mask = 1usize
            .checked_shl(u32::try_from(k).map_err(|_| AkitaError::InvalidSize {
                expected: usize::BITS as usize,
                actual: k,
            })?)
            .ok_or_else(|| AkitaError::InvalidSetup("dense Z eq width overflow".to_string()))?
            - 1;
        let offset_z_dense_low = offset_z & mask;
        let offset_z_dense_high = offset_z >> k;
        let eq_low_z_dense = EqPolynomial::evals(&full_vec_randomness[..k])?;
        let max_high = offset_z
            .checked_add(z_len_dense)
            .and_then(|end| end.checked_sub(1))
            .ok_or_else(|| {
                AkitaError::InvalidSetup("dense Z high-eq bound overflow".to_string())
            })?
            >> k;
        let n_high = max_high - offset_z_dense_high + 1;
        let eq_high_z_dense: Vec<E> = (0..n_high)
            .map(|h| eq_eval_at_index(&full_vec_randomness[k..], offset_z_dense_high + h))
            .collect();

        cfg_into_iter!(0..z_range)
            .map(|c| {
                let dc = c % prepared.depth_commit;
                let blk = c / prepared.depth_commit;
                let mut acc = E::zero();
                for pt in 0..prepared.num_points {
                    for (df, &fg) in fold_gadget.iter().enumerate().take(prepared.depth_fold) {
                        // j_M^Z(c, pt, df) = blk + B·pt + B·P·df + B·P·DF·dc
                        let x = blk
                            + prepared.block_len * pt
                            + prepared.block_len * prepared.num_points * df
                            + prepared.block_len * prepared.num_points * prepared.depth_fold * dc;
                        let sum = offset_z_dense_low + x;
                        let low_idx = sum & mask;
                        let high_idx = sum >> k;
                        let eq_val = eq_low_z_dense[low_idx] * eq_high_z_dense[high_idx];
                        acc += eq_val.mul_base(fg);
                    }
                }
                -acc
            })
            .collect()
    };

    let b_weights_by_row: Vec<Vec<E>> = (0..prepared.n_b)
        .map(|row| {
            (0..prepared.num_points)
                .map(|g| prepared.eq_tau1[b_start + g * prepared.n_b + row])
                .collect()
        })
        .collect();

    let d_part = if n_d_active == 0 {
        E::zero()
    } else {
        let d_view = setup.shared_matrix.ring_view::<D>(n_d_active, n_cols_w)?;
        let d_width = d_view.num_cols();
        let d_flat = d_view.as_slice();
        cfg_into_iter!(0..n_d_active)
            .map(|row| {
                let row_start = row * d_width;
                let row_slice = &d_flat[row_start..row_start + d_width];
                let r_eval: Vec<E> = cfg_into_iter!(0..n_cols_w)
                    .map(|c| eval_ring_at_pows(&row_slice[c], alpha_pows))
                    .collect();
                slice_inner_sum::<F, E, true, false, false>(
                    0..n_cols_w,
                    &r_eval,
                    d_weights[row],
                    &w_eq_slice,
                    &[],
                    &[],
                    0,
                    E::zero(),
                    &[],
                )
            })
            .sum()
    };

    let b_part = if prepared.n_b == 0 {
        E::zero()
    } else {
        let b_view = setup.shared_matrix.ring_view::<D>(prepared.n_b, n_cols_t)?;
        let b_width = b_view.num_cols();
        let b_flat = b_view.as_slice();
        cfg_into_iter!(0..prepared.n_b)
            .map(|row| {
                let row_start = row * b_width;
                let row_slice = &b_flat[row_start..row_start + b_width];
                let r_eval: Vec<E> = cfg_into_iter!(0..n_cols_t)
                    .map(|c| eval_ring_at_pows(&row_slice[c], alpha_pows))
                    .collect();
                slice_inner_sum::<F, E, false, true, false>(
                    0..n_cols_t,
                    &r_eval,
                    E::zero(),
                    &[],
                    &b_weights_by_row[row],
                    &t_eq_slice_per_group,
                    prepared.num_points,
                    E::zero(),
                    &[],
                )
            })
            .sum()
    };

    let a_part = if !z_used || prepared.n_a == 0 {
        E::zero()
    } else {
        let a_view = setup.shared_matrix.ring_view::<D>(prepared.n_a, z_range)?;
        let a_width = a_view.num_cols();
        let a_flat = a_view.as_slice();
        cfg_into_iter!(0..prepared.n_a)
            .map(|row| {
                let row_start = row * a_width;
                let row_slice = &a_flat[row_start..row_start + a_width];
                let r_eval: Vec<E> = cfg_into_iter!(0..z_range)
                    .map(|c| eval_ring_at_pows(&row_slice[c], alpha_pows))
                    .collect();
                slice_inner_sum::<F, E, false, false, true>(
                    0..z_range,
                    &r_eval,
                    E::zero(),
                    &[],
                    &[],
                    &[],
                    0,
                    a_weights[row],
                    &z_eq_slice,
                )
            })
            .sum()
    };

    Ok(d_part + b_part + a_part)
}

#[cfg(test)]
mod tests {
    use super::*;

    use akita_algebra::ring::scalar_powers;
    use akita_algebra::CyclotomicRing;
    use akita_field::Prime128OffsetA7F7;
    use akita_types::{
        gadget_row_scalars, AkitaSetupSeed, FlatMatrix, MRowLayout, SetupRoleDimensions,
    };

    type F = Prime128OffsetA7F7;
    const D: usize = 32;

    fn f(value: u128) -> F {
        F::from_canonical_u128_reduced(value)
    }

    #[test]
    fn setup_contribution_handles_nonidentity_multigroup_routing() {
        // `nv = 32` in `fp128_d32_onehot.rs` includes repeated compact
        // recursive levels with this real D=32 shape.
        let num_blocks = 8usize;
        let num_claims = 3usize;
        let depth_open = 26usize;
        let depth_commit = 1usize;
        let depth_fold = 4usize;
        let block_len = 512usize;
        let inner_width = block_len * depth_commit;
        let log_basis = 5u32;
        let n_a = 2usize;
        let n_d = 2usize;
        let n_b = 2usize;
        let num_polys_per_point = vec![2usize, 1usize];
        let num_public_rows = 2usize;
        let num_points = num_polys_per_point.len();
        let total_blocks = num_blocks * num_claims;
        let rows = 1 + num_public_rows + n_d + n_b * num_points + n_a;

        // Claims deliberately do not follow group-local polynomial order.
        let claim_to_point_poly = vec![(0usize, 1usize), (1, 0), (0, 0)];

        let w_len = depth_open * total_blocks;
        let t_len = depth_open * n_a * total_blocks;
        let z_len = depth_fold * depth_commit * num_points * block_len;
        let offset_w = 0usize;
        let offset_t = w_len;
        let offset_z = w_len + t_len;
        let total_len = offset_z + z_len;
        let bits = total_len.next_power_of_two().trailing_zeros() as usize;

        let stride_t = n_a * depth_open;
        let cols_per_poly_t = stride_t * num_blocks;
        let n_cols_w = num_claims * num_blocks * depth_open;
        let n_cols_t = num_polys_per_point.iter().copied().max().unwrap() * cols_per_poly_t;
        let dimensions = SetupRoleDimensions {
            n_a,
            n_b,
            n_d,
            a_setup_width: inner_width,
            b_setup_width: n_cols_t,
            d_setup_width: n_cols_w,
        };
        let max_setup_len = dimensions.max_footprint().unwrap();

        let matrix_entries: Vec<CyclotomicRing<F, D>> = (0..max_setup_len)
            .map(|idx| {
                CyclotomicRing::from_coefficients(std::array::from_fn(|coeff| {
                    f(1_000 + (idx * D + coeff) as u128)
                }))
            })
            .collect();
        let setup = AkitaExpandedSetup::from_parts(
            AkitaSetupSeed {
                max_num_vars: 32,
                max_num_batched_polys: num_polys_per_point.iter().sum(),
                max_num_points: num_points,
                max_setup_len: matrix_entries.len(),
                public_matrix_seed: [7u8; 32],
                zk_blinding_seed: [8u8; 32],
            },
            FlatMatrix::from_ring_slice::<D>(&matrix_entries),
        )
        .unwrap();

        let eq_tau1: Vec<F> = (0..rows.next_power_of_two())
            .map(|idx| f(11 + idx as u128))
            .collect();
        let prepared = RingSwitchDeferredRowEval {
            c_alphas: PreparedChallengeEvals::Flat(
                (0..total_blocks).map(|idx| f(41 + idx as u128)).collect(),
            ),
            eq_tau1: eq_tau1.clone(),
            total_blocks,
            num_t_vectors: num_polys_per_point.iter().sum(),
            num_blocks,
            num_claims,
            depth_open,
            depth_commit,
            depth_fold,
            #[cfg(feature = "zk")]
            d_blinding_segment_len: 0,
            #[cfg(feature = "zk")]
            b_blinding_digit_planes_per_point: 0,
            #[cfg(feature = "zk")]
            b_blinding_segment_len: 0,
            block_len,
            inner_width,
            log_basis,
            n_a,
            n_d,
            m_row_layout: MRowLayout::Intermediate,
            n_b,
            num_points,
            rows,
            z_first: false,
            claim_to_point_poly: claim_to_point_poly.clone(),
            num_polys_per_point: num_polys_per_point.clone(),
            num_public_rows,
            gamma: vec![F::one(); num_claims],
            claim_to_point: vec![1, 0, 1],
        };

        let full_vec_randomness: Vec<F> = (0..bits).map(|idx| f(101 + idx as u128)).collect();
        let eq: Vec<F> = (0..(1usize << bits))
            .map(|idx| eq_eval_at_index(&full_vec_randomness, idx))
            .collect();
        let alpha = f(19);
        let alpha_pows = scalar_powers(alpha, D);
        let fold_gadget = gadget_row_scalars::<F>(depth_fold, log_basis);
        let block_bits = num_blocks.trailing_zeros() as usize;
        let eq_low = EqPolynomial::evals(&full_vec_randomness[..block_bits]).unwrap();
        let z_offset_low_bits = block_len.trailing_zeros() as usize;
        let z_block_low_eq =
            EqPolynomial::evals(&full_vec_randomness[..z_offset_low_bits]).unwrap();

        let got = compute_setup_contribution::<F, F, D>(
            &prepared,
            &full_vec_randomness,
            &setup,
            &eq_low,
            &z_block_low_eq,
            &alpha_pows,
            &fold_gadget,
            offset_w,
            offset_t,
            offset_z,
        )
        .unwrap();

        let d_view = setup.shared_matrix.ring_view::<D>(n_d, n_cols_w).unwrap();
        let b_view = setup.shared_matrix.ring_view::<D>(n_b, n_cols_t).unwrap();
        let a_view = setup
            .shared_matrix
            .ring_view::<D>(n_a, inner_width)
            .unwrap();
        let d_rows: Vec<_> = d_view.rows().collect();
        let b_rows: Vec<_> = b_view.rows().collect();
        let a_rows: Vec<_> = a_view.rows().collect();
        let d_start = 1 + num_public_rows;
        let b_start = d_start + n_d;
        let a_start = b_start + n_b * num_points;

        let mut expected = F::zero();

        for row in 0..n_d {
            let weight = eq_tau1[d_start + row];
            for claim in 0..num_claims {
                for block in 0..num_blocks {
                    for digit in 0..depth_open {
                        let d_phys_col = (claim * num_blocks + block) * depth_open + digit;
                        let m_idx = block + num_blocks * (claim + num_claims * digit);
                        expected += weight
                            * eval_ring_at_pows(&d_rows[row][d_phys_col], &alpha_pows)
                            * eq[offset_w + m_idx];
                    }
                }
            }
        }

        let num_t_vectors: usize = num_polys_per_point.iter().sum();
        let mut flat_t_vector = 0usize;
        for (point_idx, &group_poly_count) in num_polys_per_point.iter().enumerate() {
            for poly_idx in 0..group_poly_count {
                for row in 0..n_b {
                    let weight = eq_tau1[b_start + point_idx * n_b + row];
                    for a_idx in 0..n_a {
                        for digit in 0..depth_open {
                            for block in 0..num_blocks {
                                let phys_claim_offset =
                                    block * stride_t + a_idx * depth_open + digit;
                                let local_col = poly_idx * cols_per_poly_t + phys_claim_offset;
                                let m_idx = block
                                    + num_blocks
                                        * (flat_t_vector
                                            + num_t_vectors * digit
                                            + num_t_vectors * depth_open * a_idx);
                                expected += weight
                                    * eval_ring_at_pows(&b_rows[row][local_col], &alpha_pows)
                                    * eq[offset_t + m_idx];
                            }
                        }
                    }
                }
                flat_t_vector += 1;
            }
        }

        for row in 0..n_a {
            let weight = eq_tau1[a_start + row];
            for dc in 0..depth_commit {
                for (df, &fold_weight) in fold_gadget.iter().enumerate() {
                    for point in 0..num_points {
                        for block in 0..block_len {
                            let local_col = block * depth_commit + dc;
                            let m_idx = block
                                + block_len
                                    * (point + num_points * df + num_points * depth_fold * dc);
                            expected -= weight
                                * eval_ring_at_pows(&a_rows[row][local_col], &alpha_pows)
                                * fold_weight
                                * eq[offset_z + m_idx];
                        }
                    }
                }
            }
        }

        assert_eq!(
            got, expected,
            "fused setup contribution must follow num_polys_per_point and claim_poly_indices"
        );
    }
}
