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

#[derive(Copy, Clone)]
struct PackedInterval {
    start: usize,
    end: usize,
}

impl PackedInterval {
    #[inline(always)]
    fn contains(self, start: usize, end: usize) -> bool {
        self.start <= start && end <= self.end
    }
}

#[inline(always)]
fn packed_interval(
    active: bool,
    row: usize,
    stride: usize,
    width: usize,
    name: &'static str,
) -> Result<Option<PackedInterval>, AkitaError> {
    if !active || width == 0 {
        return Ok(None);
    }
    let start = row
        .checked_mul(stride)
        .ok_or_else(|| AkitaError::InvalidSetup(format!("packed {name} row offset overflow")))?;
    let end = start
        .checked_add(width)
        .ok_or_else(|| AkitaError::InvalidSetup(format!("packed {name} row end overflow")))?;
    Ok(Some(PackedInterval { start, end }))
}

/// Sum a contiguous absolute slice of the packed setup prefix after alpha
/// evaluation. The const generics select which of the row-local D/B/A
/// intervals cover this slice, so inactive arms disappear after
/// monomorphisation.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn packed_slice_inner_sum<E, const HAS_D: bool, const HAS_B: bool, const HAS_A: bool>(
    range: std::ops::Range<usize>,
    setup_alpha: &[E],
    d_start: usize,
    d_weight: E,
    w_eq: &[E],
    b_start: usize,
    b_weights: &[E],
    t_eq_per_group: &[Vec<E>],
    a_start: usize,
    a_weight: E,
    z_eq: &[E],
) -> E
where
    E: FieldCore,
{
    cfg_into_iter!(range)
        .map(|lambda| {
            let mut weight = E::zero();
            if HAS_D {
                weight += d_weight * w_eq[lambda - d_start];
            }
            if HAS_B {
                for (g, t_eq_slice) in t_eq_per_group.iter().enumerate() {
                    weight += b_weights[g] * t_eq_slice[lambda - b_start];
                }
            }
            if HAS_A {
                weight += a_weight * z_eq[lambda - a_start];
            }
            if weight.is_zero() {
                E::zero()
            } else {
                setup_alpha[lambda] * weight
            }
        })
        .sum()
}

/// Compute the fused setup-matrix contribution `D · ŵ + B · t̂ + A · ẑ`
/// over packed role-local A/B/D setup views. The three role views overlap as
/// prefixes of the same raw setup vector but use their natural row widths.
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
    if prepared.n_a == 0 || z_range == 0 {
        return Err(AkitaError::InvalidSetup(
            "A/Z layout requires non-zero A rows and Z width".to_string(),
        ));
    }
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
    let d_stride = n_cols_w;

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
    let b_stride = n_cols_t;

    // Row range covers every SIS row that any of W/T/Z touch. Z extends it to
    // `n_a`, so Z-only rows participate inside the loop; no separate
    // post-loop matrix-A scan is needed.
    let r_max = n_d_active.max(prepared.n_b).max(prepared.n_a);
    let n_cols_total = n_cols_w.max(n_cols_t).max(z_range);
    if n_cols_total == 0 {
        return Err(AkitaError::InvalidSetup(
            "matrix-row pattern evaluation requires at least one SIS column".to_string(),
        ));
    }
    if r_max == 0 {
        return Err(AkitaError::InvalidSetup(
            "matrix-row pattern evaluation requires at least one SIS row".to_string(),
        ));
    }
    let setup_len = setup.shared_matrix().total_ring_elements_at::<D>()?;
    let d_required = n_d_active
        .checked_mul(d_stride)
        .ok_or_else(|| AkitaError::InvalidSetup("D setup footprint overflow".to_string()))?;
    let b_required = prepared
        .n_b
        .checked_mul(b_stride)
        .ok_or_else(|| AkitaError::InvalidSetup("B setup footprint overflow".to_string()))?;
    let a_required = prepared
        .n_a
        .checked_mul(z_range)
        .ok_or_else(|| AkitaError::InvalidSetup("A setup footprint overflow".to_string()))?;
    let required = d_required.max(b_required).max(a_required);
    if required > setup_len {
        return Err(AkitaError::InvalidSetup(
            "shared matrix is too small for selected verifier layout".to_string(),
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

    // `z_eq_slice[c]` — column-only Z pattern. Pow2: peeled-block lookup
    // `z_block_low_eq[low] · S_per_dc_per_carry[dc][carry]`. Non-pow2:
    // dense aggregation over `(pt, df)` with a one-shot peeled eq cache so
    // per-cell cost stays O(P · DF).
    let z_eq_slice: Vec<E> = if z_dims_pow2 {
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

    let setup_view = setup.shared_matrix().ring_view::<D>(1, setup_len)?;
    let setup_flat = setup_view.as_slice();
    let setup_alpha: Vec<E> = cfg_into_iter!(0..required)
        .map(|lambda| eval_ring_at_pows(&setup_flat[lambda], alpha_pows))
        .collect();
    let b_weights_by_row: Vec<Vec<E>> = (0..prepared.n_b)
        .map(|row| {
            (0..prepared.num_points)
                .map(|g| prepared.eq_tau1[b_start + g * prepared.n_b + row])
                .collect()
        })
        .collect();

    let row_contribs: Vec<E> = cfg_into_iter!(0..r_max)
        .map(|row| -> Result<E, AkitaError> {
            let d_interval = packed_interval(row < n_d_active, row, d_stride, n_cols_w, "D")?;
            let b_interval = packed_interval(row < prepared.n_b, row, b_stride, n_cols_t, "B")?;
            let a_interval = packed_interval(row < prepared.n_a, row, z_range, z_range, "A")?;

            let mut endpoints = [0usize; 6];
            let mut n_endpoints = 0usize;
            for interval in [d_interval, b_interval, a_interval].into_iter().flatten() {
                endpoints[n_endpoints] = interval.start;
                endpoints[n_endpoints + 1] = interval.end;
                n_endpoints += 2;
            }
            if n_endpoints == 0 {
                return Ok(E::zero());
            }
            let endpoints = &mut endpoints[..n_endpoints];
            endpoints.sort_unstable();
            let mut dedup_len = 0usize;
            for idx in 0..endpoints.len() {
                if dedup_len == 0 || endpoints[idx] != endpoints[dedup_len - 1] {
                    endpoints[dedup_len] = endpoints[idx];
                    dedup_len += 1;
                }
            }

            let d_start = d_interval.map_or(0, |interval| interval.start);
            let b_start_abs = b_interval.map_or(0, |interval| interval.start);
            let a_start = a_interval.map_or(0, |interval| interval.start);
            let d_weight = if row < n_d_active {
                d_weights[row]
            } else {
                E::zero()
            };
            let b_weights: &[E] = if row < prepared.n_b {
                &b_weights_by_row[row]
            } else {
                &[]
            };
            let a_weight = if row < prepared.n_a {
                a_weights[row]
            } else {
                E::zero()
            };

            let mut acc = E::zero();
            macro_rules! segment_sum {
                ($lo:expr, $hi:expr, $has_d:literal, $has_b:literal, $has_a:literal) => {
                    packed_slice_inner_sum::<E, $has_d, $has_b, $has_a>(
                        $lo..$hi,
                        &setup_alpha,
                        d_start,
                        d_weight,
                        &w_eq_slice,
                        b_start_abs,
                        b_weights,
                        &t_eq_slice_per_group,
                        a_start,
                        a_weight,
                        &z_eq_slice,
                    )
                };
            }
            for idx in 0..dedup_len.saturating_sub(1) {
                let lo = endpoints[idx];
                let hi = endpoints[idx + 1];
                if lo == hi {
                    continue;
                }
                let has_d = d_interval.is_some_and(|interval| interval.contains(lo, hi));
                let has_b = b_interval.is_some_and(|interval| interval.contains(lo, hi));
                let has_a = a_interval.is_some_and(|interval| interval.contains(lo, hi));
                acc += match (has_d, has_b, has_a) {
                    (true, true, true) => segment_sum!(lo, hi, true, true, true),
                    (true, true, false) => segment_sum!(lo, hi, true, true, false),
                    (true, false, true) => segment_sum!(lo, hi, true, false, true),
                    (false, true, true) => segment_sum!(lo, hi, false, true, true),
                    (true, false, false) => segment_sum!(lo, hi, true, false, false),
                    (false, true, false) => segment_sum!(lo, hi, false, true, false),
                    (false, false, true) => segment_sum!(lo, hi, false, false, true),
                    (false, false, false) => E::zero(),
                };
            }
            Ok(acc)
        })
        .collect::<Result<Vec<_>, AkitaError>>()?;

    Ok(row_contribs.into_iter().sum())
}

#[cfg(test)]
mod tests {
    use super::*;

    use akita_algebra::ring::scalar_powers;
    use akita_algebra::CyclotomicRing;
    use akita_challenges::SparseChallengeConfig;
    use akita_field::Prime128OffsetA7F7;
    use akita_types::{
        gadget_row_scalars, ring_relation_segment_layout_for_opening_shape, AkitaSetupSeed,
        FlatMatrix, LevelParams, MRowLayout, SisModulusFamily,
    };

    type F = Prime128OffsetA7F7;
    const D: usize = 32;

    fn f(value: u128) -> F {
        F::from_canonical_u128_reduced(value)
    }

    fn fixture_lp() -> LevelParams {
        LevelParams::params_only(
            SisModulusFamily::Q128,
            D,
            5,
            2,
            2,
            2,
            SparseChallengeConfig::Uniform {
                weight: 1,
                nonzero_coeffs: vec![1],
            },
        )
        .with_decomp(2, 3, 1, 26, 4, 512 * 8)
        .expect("setup contribution fixture lp")
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

        let lp = fixture_lp();
        let witness_segment_layout = ring_relation_segment_layout_for_opening_shape::<F, D>(
            &lp,
            MRowLayout::WithDBlock,
            &num_polys_per_point,
        )
        .expect("witness segment layout");
        let offset_w = witness_segment_layout.offset_w;
        let offset_t = witness_segment_layout.offset_t;
        let offset_z = witness_segment_layout.offset_z;
        let total_len = witness_segment_layout.offset_r;
        let bits = total_len.next_power_of_two().trailing_zeros() as usize;

        let stride_t = n_a * depth_open;
        let cols_per_poly_t = stride_t * num_blocks;
        let n_cols_w = num_claims * num_blocks * depth_open;
        let n_cols_t = num_polys_per_point.iter().copied().max().unwrap() * cols_per_poly_t;
        let max_setup_len = (n_d * n_cols_w).max(n_b * n_cols_t).max(n_a * inner_width);

        let matrix_entries: Vec<CyclotomicRing<F, D>> = (0..max_setup_len)
            .map(|idx| {
                CyclotomicRing::from_coefficients(std::array::from_fn(|coeff| {
                    f(1_000 + (idx * D + coeff) as u128)
                }))
            })
            .collect();
        let setup = AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
            AkitaSetupSeed {
                max_num_vars: 32,
                max_num_batched_polys: num_polys_per_point.iter().sum(),
                max_num_points: num_points,
                gen_ring_dim: D,
                max_setup_len,
                #[cfg(feature = "zk")]
                max_zk_b_len: 1,
                #[cfg(feature = "zk")]
                max_zk_d_len: 1,
                public_matrix_seed: [7u8; 32],
            },
            FlatMatrix::from_ring_slice::<D>(&matrix_entries),
            #[cfg(feature = "zk")]
            FlatMatrix::from_flat_data(vec![F::zero(); D], D),
            #[cfg(feature = "zk")]
            FlatMatrix::from_flat_data(vec![F::zero(); D], D),
        );

        let eq_tau1: Vec<F> = (0..rows.next_power_of_two())
            .map(|idx| f(11 + idx as u128))
            .collect();
        let prepared = RingSwitchDeferredRowEval {
            c_alphas: PreparedChallengeEvals::Flat(
                (0..total_blocks).map(|idx| f(41 + idx as u128)).collect(),
            ),
            eq_tau1: eq_tau1.clone(),
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
            m_row_layout: MRowLayout::WithDBlock,
            n_b,
            num_points,
            rows,
            claim_to_point_poly: claim_to_point_poly.clone(),
            num_polys_per_point: num_polys_per_point.clone(),
            num_public_rows,
            gamma: vec![F::one(); num_claims],
            claim_to_point: vec![1, 0, 1],
            witness_segment_layout,
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

        let d_view = setup.shared_matrix().ring_view::<D>(n_d, n_cols_w).unwrap();
        let d_rows: Vec<_> = d_view.rows().collect();
        let b_view = setup.shared_matrix().ring_view::<D>(n_b, n_cols_t).unwrap();
        let b_rows: Vec<_> = b_view.rows().collect();
        let a_view = setup
            .shared_matrix()
            .ring_view::<D>(n_a, inner_width)
            .unwrap();
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
