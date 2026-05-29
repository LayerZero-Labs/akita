use akita_algebra::eq_poly::EqPolynomial;
use akita_algebra::offset_eq::eq_eval_at_index;
use akita_algebra::ring::eval_ring_at_pows;
use akita_algebra::CyclotomicRing;
use akita_field::parallel::*;
use akita_field::{AkitaError, CanonicalField, ExtField, FieldCore};
use akita_types::AkitaExpandedSetup;

use super::structured_slice::POSSIBLE_CARRIES;
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

#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
#[inline(always)]
fn jolt_cycle_marker(marker_id_str: &str, event_type: u32) {
    const JOLT_CYCLE_TRACK_CALL_ID: u32 = 0xC7C1E;
    let marker_id = marker_id_str.as_ptr() as usize as u32;
    let marker_len = marker_id_str.len() as u32;
    unsafe {
        core::arch::asm!(
            ".insn i 0x5B, 2, x0, x0, 0",
            in("x10") JOLT_CYCLE_TRACK_CALL_ID,
            in("x11") marker_id,
            in("x12") marker_len,
            in("x13") event_type,
            options(nostack, preserves_flags)
        );
    }
}

#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
#[inline(always)]
fn jolt_start_cycle_tracking(marker_id: &str) {
    jolt_cycle_marker(marker_id, 1);
}

#[cfg(not(any(target_arch = "riscv32", target_arch = "riscv64")))]
#[inline(always)]
fn jolt_start_cycle_tracking(_marker_id: &str) {}

#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
#[inline(always)]
fn jolt_end_cycle_tracking(marker_id: &str) {
    jolt_cycle_marker(marker_id, 2);
}

#[cfg(not(any(target_arch = "riscv32", target_arch = "riscv64")))]
#[inline(always)]
fn jolt_end_cycle_tracking(_marker_id: &str) {}

#[inline(always)]
fn push_role_boundaries(
    endpoints: &mut Vec<usize>,
    rows: usize,
    stride: usize,
    name: &'static str,
) -> Result<(), AkitaError> {
    if rows == 0 || stride == 0 {
        return Ok(());
    }
    let mut boundary = 0usize;
    for _ in 0..rows {
        boundary = boundary
            .checked_add(stride)
            .ok_or_else(|| AkitaError::InvalidSetup(format!("packed {name} boundary overflow")))?;
        endpoints.push(boundary);
    }
    Ok(())
}

/// Sum a contiguous absolute slice of the packed setup prefix.
///
/// The packed interval split first combines every active D/B/A contribution to
/// `bar_omega(λ)`, then contracts the same `<S, omega_S>` value as
/// `eval_alpha(S[λ]) * bar_omega(λ)`. This keeps the direct verifier scalar in
/// the hot loop while the test oracle still materializes
/// `omega_S(λ, y) = bar_omega(λ) * α^y`.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn packed_slice_inner_sum<
    F,
    E,
    const D: usize,
    const HAS_D: bool,
    const HAS_B: bool,
    const HAS_A: bool,
>(
    range: std::ops::Range<usize>,
    setup_flat: &[CyclotomicRing<F, D>],
    alpha_pows: &[E],
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
    F: FieldCore,
    E: ExtField<F>,
{
    jolt_start_cycle_tracking("setup_packed_slice_inner_sum");
    let result = cfg_fold_reduce!(
        range,
        E::zero,
        |mut acc, lambda| {
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
            if !weight.is_zero() {
                acc += eval_ring_at_pows(&setup_flat[lambda], alpha_pows) * weight;
            }
            acc
        },
        |lhs, rhs| lhs + rhs
    );
    jolt_end_cycle_tracking("setup_packed_slice_inner_sum");
    result
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
    if alpha_pows.len() != D {
        return Err(AkitaError::InvalidSize {
            expected: D,
            actual: alpha_pows.len(),
        });
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
    jolt_start_cycle_tracking("setup_hi_eq_tables");
    let eq_hi_w_table: Vec<E> = (0..=w_hi_len)
        .map(|k| eq_eval_at_index(high_challenges, w_offset_high + k))
        .collect();
    let eq_hi_t_table: Vec<E> = (0..=t_hi_len)
        .map(|k| eq_eval_at_index(high_challenges, t_offset_high + k))
        .collect();
    jolt_end_cycle_tracking("setup_hi_eq_tables");

    jolt_start_cycle_tracking("setup_w_eq_slice");
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
    jolt_end_cycle_tracking("setup_w_eq_slice");

    jolt_start_cycle_tracking("setup_t_eq_slices");
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
    jolt_end_cycle_tracking("setup_t_eq_slices");

    // `z_eq_slice[c]` — column-only Z pattern. Pow2: peeled-block lookup
    // `z_block_low_eq[low] · S_per_dc_per_carry[dc][carry]`. Non-pow2:
    // dense aggregation over `(pt, df)` with a one-shot peeled eq cache so
    // per-cell cost stays O(P · DF).
    jolt_start_cycle_tracking("setup_z_eq_slice");
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
    jolt_end_cycle_tracking("setup_z_eq_slice");

    let setup_view = setup.shared_matrix().ring_view::<D>(1, setup_len)?;
    let setup_flat = setup_view.as_slice();
    jolt_start_cycle_tracking("setup_b_weights");
    let b_weights_by_row: Vec<Vec<E>> = (0..prepared.n_b)
        .map(|row| {
            (0..prepared.num_points)
                .map(|g| prepared.eq_tau1[b_start + g * prepared.n_b + row])
                .collect()
        })
        .collect();
    jolt_end_cycle_tracking("setup_b_weights");

    jolt_start_cycle_tracking("setup_inner_product_segments");
    let mut endpoints = Vec::with_capacity(n_d_active + prepared.n_b + prepared.n_a + 2);
    endpoints.push(0);
    endpoints.push(required);
    push_role_boundaries(&mut endpoints, n_d_active, d_stride, "D")?;
    push_role_boundaries(&mut endpoints, prepared.n_b, b_stride, "B")?;
    push_role_boundaries(&mut endpoints, prepared.n_a, z_range, "A")?;
    endpoints.sort_unstable();
    endpoints.dedup();

    let segment_sums: Vec<E> = cfg_into_iter!(0..endpoints.len().saturating_sub(1))
        .map(|idx| -> Result<E, AkitaError> {
            let lo = endpoints[idx];
            let hi = endpoints[idx + 1];
            if lo == hi {
                return Ok(E::zero());
            }

            let has_d = d_stride != 0 && lo < d_required;
            let d_row = if has_d { lo / d_stride } else { 0 };
            let d_start_abs = if has_d {
                d_row.checked_mul(d_stride).ok_or_else(|| {
                    AkitaError::InvalidSetup("D segment start overflow".to_string())
                })?
            } else {
                0
            };
            let d_weight = if has_d { d_weights[d_row] } else { E::zero() };

            let has_b = b_stride != 0 && lo < b_required;
            let b_row = if has_b { lo / b_stride } else { 0 };
            let b_start_abs = if has_b {
                b_row.checked_mul(b_stride).ok_or_else(|| {
                    AkitaError::InvalidSetup("B segment start overflow".to_string())
                })?
            } else {
                0
            };
            let b_weights: &[E] = if has_b { &b_weights_by_row[b_row] } else { &[] };

            let has_a = z_range != 0 && lo < a_required;
            let a_row = if has_a { lo / z_range } else { 0 };
            let a_start_abs = if has_a {
                a_row.checked_mul(z_range).ok_or_else(|| {
                    AkitaError::InvalidSetup("A segment start overflow".to_string())
                })?
            } else {
                0
            };
            let a_weight = if has_a { a_weights[a_row] } else { E::zero() };

            macro_rules! segment_sum {
                ($has_d:literal, $has_b:literal, $has_a:literal) => {
                    packed_slice_inner_sum::<F, E, D, $has_d, $has_b, $has_a>(
                        lo..hi,
                        setup_flat,
                        alpha_pows,
                        d_start_abs,
                        d_weight,
                        &w_eq_slice,
                        b_start_abs,
                        b_weights,
                        &t_eq_slice_per_group,
                        a_start_abs,
                        a_weight,
                        &z_eq_slice,
                    )
                };
            }

            Ok(match (has_d, has_b, has_a) {
                (true, true, true) => segment_sum!(true, true, true),
                (true, true, false) => segment_sum!(true, true, false),
                (true, false, true) => segment_sum!(true, false, true),
                (false, true, true) => segment_sum!(false, true, true),
                (true, false, false) => segment_sum!(true, false, false),
                (false, true, false) => segment_sum!(false, true, false),
                (false, false, true) => segment_sum!(false, false, true),
                (false, false, false) => E::zero(),
            })
        })
        .collect::<Result<Vec<_>, AkitaError>>()?;
    jolt_end_cycle_tracking("setup_inner_product_segments");

    Ok(segment_sums.into_iter().sum())
}

#[cfg(test)]
mod fixtures;
#[cfg(test)]
mod tests;
