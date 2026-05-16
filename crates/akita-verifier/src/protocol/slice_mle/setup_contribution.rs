use akita_algebra::eq_poly::EqPolynomial;
use akita_algebra::offset_eq::eq_eval_at_index;
use akita_algebra::ring::eval_ring_at_pows;
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

/// Translate a B-column (B-physical order `[digit, a_row, block, claim]`)
/// into `(low_block_eq_idx, high_eq_idx)`. `flat_claim` resolves the
/// per-group claim index to the global flat claim used by the high index.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn get_eq_indices_for_b(
    current_index: usize,
    flat_claim: usize,
    num_digits: usize,
    n_a: usize,
    num_blocks: usize,
    num_claims: usize,
    stride_t: usize,
    block_offset_low: usize,
    block_mask: usize,
    block_bits: usize,
) -> (usize, usize) {
    let digit_idx = current_index % num_digits;
    let a_row_idx = (current_index / num_digits) % n_a;
    let block_idx = (current_index / stride_t) % num_blocks;
    let m_layout_high_idx =
        flat_claim + num_claims * digit_idx + num_claims * num_digits * a_row_idx;
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

/// Compute the fused setup-matrix contribution `D · ŵ + B · t̂ + A · ẑ`
/// as a single `<M_Flat, Eval>` over the shared SIS matrix. W, T, and Z
/// share `r_eval[c] = M_Flat[row, c]` for every row that participates in
/// more than one half. Per-row, the column axis is partitioned into three
/// contiguous slices sorted by each pattern's endpoint; the active subset
/// of `{W, T, Z}` is constant inside each slice and selected at the type
/// level via `slice_inner_sum`'s const generics. See
/// `specs/optimized_verifier.md` for the full derivation.
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

    let d_start = 1usize
        .checked_add(prepared.num_public_eval_rows)
        .ok_or_else(|| AkitaError::InvalidSetup("D row start overflow".to_string()))?;
    let b_start = d_start
        .checked_add(prepared.n_d)
        .ok_or_else(|| AkitaError::InvalidSetup("B row start overflow".to_string()))?;
    let a_start = b_start
        .checked_add(
            prepared
                .n_b
                .checked_mul(prepared.num_commitment_groups)
                .ok_or_else(|| AkitaError::InvalidSetup("B row width overflow".to_string()))?,
        )
        .ok_or_else(|| AkitaError::InvalidSetup("A row start overflow".to_string()))?;
    if d_start
        .checked_add(prepared.n_d)
        .is_none_or(|end| end > prepared.eq_tau1.len())
        || a_start > prepared.rows
        || prepared.rows > prepared.eq_tau1.len()
    {
        return Err(AkitaError::InvalidSetup(
            "M-row weights are inconsistent with verifier layout".to_string(),
        ));
    }
    let d_weights = &prepared.eq_tau1[d_start..(d_start + prepared.n_d)];
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

    // Invert `claim_to_group`: T's row weight is group-dependent and its
    // c-axis indexes `poly_idx` within the group (the polynomial slot in
    // the committed group, *not* a claim-within-group counter). The
    // SIS-matrix T section for group `g` has `group_poly_counts[g] *
    // cols_per_poly_t` columns — one column block per polynomial slot —
    // so sizing must follow `group_poly_counts`, not the number of
    // claims that open polynomials in `g`. Polynomial slots that no
    // claim opens contribute zero (and stay `None` here).
    let max_group_poly_count = prepared
        .group_poly_counts
        .iter()
        .copied()
        .max()
        .unwrap_or(0);
    let mut flat_claim_for_group: Vec<Vec<Option<usize>>> = prepared
        .group_poly_counts
        .iter()
        .map(|&n| vec![None; n])
        .collect();
    for (flat_idx, &(g, poly_idx)) in prepared.claim_to_group.iter().enumerate() {
        flat_claim_for_group[g][poly_idx] = Some(flat_idx);
    }
    let n_cols_t = max_group_poly_count
        .checked_mul(cols_per_poly_t)
        .ok_or_else(|| AkitaError::InvalidSetup("T column width overflow".to_string()))?;

    // Row range covers every SIS row that any of W/T/Z touch. Z extends
    // it to `n_a` when active, so Z-only rows participate inside the loop
    // — no separate post-loop matrix-A scan.
    let r_max = if z_used {
        prepared.n_d.max(prepared.n_b).max(prepared.n_a)
    } else {
        prepared.n_d.max(prepared.n_b)
    };
    let n_cols_total = n_cols_w.max(n_cols_t).max(if z_used { z_range } else { 0 });
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
    if n_cols_total > setup.seed.max_stride {
        return Err(AkitaError::InvalidSetup(
            "shared matrix stride is too small for selected verifier layout".to_string(),
        ));
    }

    let w_hi_len = prepared
        .num_claims
        .checked_mul(prepared.depth_open)
        .ok_or_else(|| AkitaError::InvalidSetup("W high-eq width overflow".to_string()))?;
    let t_hi_len = w_hi_len
        .checked_mul(prepared.n_a)
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

    let t_eq_slice_per_group: Vec<Vec<E>> = (0..prepared.num_commitment_groups)
        .map(|g| {
            let group_size = prepared.group_poly_counts[g];
            cfg_into_iter!(0..n_cols_t)
                .map(|c| {
                    let poly_idx = c / cols_per_poly_t;
                    if poly_idx >= group_size {
                        return E::zero();
                    }
                    match flat_claim_for_group[g][poly_idx] {
                        Some(flat_claim) => {
                            let (low_eq_idx, high_eq_idx) = get_eq_indices_for_b(
                                c,
                                flat_claim,
                                prepared.depth_open,
                                prepared.n_a,
                                prepared.num_blocks,
                                prepared.num_claims,
                                stride_t,
                                block_offset_low,
                                block_mask,
                                block_bits,
                            );
                            eq_low[low_eq_idx] * eq_hi_t_table[high_eq_idx]
                        }
                        None => E::zero(),
                    }
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

    // Per-row inner products. Each row's column axis splits into three
    // contiguous slices sorted by which pattern's endpoint comes first;
    // the active subset of {W, T, Z} is constant inside each slice and
    // dispatched via `slice_inner_sum`'s const generics. The B / D / A
    // sub-matrices alias the same backing storage.
    #[derive(Copy, Clone)]
    enum Pat {
        W,
        T,
        Z,
    }
    let shared_view = setup
        .shared_matrix
        .ring_view::<D>(r_max, setup.seed.max_stride)?;
    let shared_stride = shared_view.num_cols();
    let shared_flat = shared_view.as_slice();
    let b_weights_by_row: Vec<Vec<E>> = (0..prepared.n_b)
        .map(|row| {
            (0..prepared.num_commitment_groups)
                .map(|g| prepared.eq_tau1[b_start + g * prepared.n_b + row])
                .collect()
        })
        .collect();

    let row_contribs: Vec<E> = cfg_into_iter!(0..r_max)
        .map(|row| {
            let row_start = row * shared_stride;
            let row_slice = &shared_flat[row_start..row_start + shared_stride];

            let e_w = if row < prepared.n_d { n_cols_w } else { 0 };
            let e_t = if row < prepared.n_b { n_cols_t } else { 0 };
            let e_z = if row < prepared.n_a && z_used {
                z_range
            } else {
                0
            };

            let mut ends = [(e_w, Pat::W), (e_t, Pat::T), (e_z, Pat::Z)];
            ends.sort_by_key(|&(e, _)| e);
            let [(e1, k1), (e2, _), (e3, k3)] = ends;
            if e3 == 0 {
                return E::zero();
            }

            let r_eval: Vec<E> = cfg_into_iter!(0..e3)
                .map(|c| eval_ring_at_pows(&row_slice[c], alpha_pows))
                .collect();

            // `b_w_for_groups` is only read when `HAS_T = true`, which
            let b_w_for_groups: &[E] = if row < prepared.n_b {
                &b_weights_by_row[row]
            } else {
                &[]
            };

            let s1 = if e1 > 0 {
                slice_inner_sum::<F, E, true, true, true>(
                    0..e1,
                    &r_eval,
                    d_weights[row],
                    &w_eq_slice,
                    b_w_for_groups,
                    &t_eq_slice_per_group,
                    prepared.num_commitment_groups,
                    a_weights[row],
                    &z_eq_slice,
                )
            } else {
                E::zero()
            };

            let s2 = if e2 > e1 {
                match k1 {
                    Pat::W => slice_inner_sum::<F, E, false, true, true>(
                        e1..e2,
                        &r_eval,
                        E::zero(),
                        &w_eq_slice,
                        b_w_for_groups,
                        &t_eq_slice_per_group,
                        prepared.num_commitment_groups,
                        a_weights[row],
                        &z_eq_slice,
                    ),
                    Pat::T => slice_inner_sum::<F, E, true, false, true>(
                        e1..e2,
                        &r_eval,
                        d_weights[row],
                        &w_eq_slice,
                        b_w_for_groups,
                        &t_eq_slice_per_group,
                        prepared.num_commitment_groups,
                        a_weights[row],
                        &z_eq_slice,
                    ),
                    Pat::Z => slice_inner_sum::<F, E, true, true, false>(
                        e1..e2,
                        &r_eval,
                        d_weights[row],
                        &w_eq_slice,
                        b_w_for_groups,
                        &t_eq_slice_per_group,
                        prepared.num_commitment_groups,
                        E::zero(),
                        &z_eq_slice,
                    ),
                }
            } else {
                E::zero()
            };

            let s3 = if e3 > e2 {
                match k3 {
                    Pat::W => slice_inner_sum::<F, E, true, false, false>(
                        e2..e3,
                        &r_eval,
                        d_weights[row],
                        &w_eq_slice,
                        b_w_for_groups,
                        &t_eq_slice_per_group,
                        prepared.num_commitment_groups,
                        E::zero(),
                        &z_eq_slice,
                    ),
                    Pat::T => slice_inner_sum::<F, E, false, true, false>(
                        e2..e3,
                        &r_eval,
                        E::zero(),
                        &w_eq_slice,
                        b_w_for_groups,
                        &t_eq_slice_per_group,
                        prepared.num_commitment_groups,
                        E::zero(),
                        &z_eq_slice,
                    ),
                    Pat::Z => slice_inner_sum::<F, E, false, false, true>(
                        e2..e3,
                        &r_eval,
                        E::zero(),
                        &w_eq_slice,
                        b_w_for_groups,
                        &t_eq_slice_per_group,
                        prepared.num_commitment_groups,
                        a_weights[row],
                        &z_eq_slice,
                    ),
                }
            } else {
                E::zero()
            };

            s1 + s2 + s3
        })
        .collect();

    Ok(row_contribs.into_iter().sum::<E>())
}

#[cfg(test)]
mod tests {
    use super::*;

    use akita_algebra::ring::scalar_powers;
    use akita_algebra::CyclotomicRing;
    use akita_field::Prime128OffsetA7F7;
    use akita_types::{gadget_row_scalars, AkitaSetupSeed, FlatMatrix};

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
        let group_poly_counts = vec![2usize, 1usize];
        let num_commitment_groups = group_poly_counts.len();
        let num_public_eval_rows = 2usize;
        let num_points = 2usize;
        let total_blocks = num_blocks * num_claims;
        let rows = 1 + num_public_eval_rows + n_d + n_b * num_commitment_groups + n_a;

        // Claims deliberately do not follow group-local polynomial order.
        let claim_to_group = vec![(0usize, 1usize), (1, 0), (0, 0)];

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
        let n_cols_t = group_poly_counts.iter().copied().max().unwrap() * cols_per_poly_t;
        let max_stride = n_cols_w.max(n_cols_t).max(inner_width);
        let r_max = n_d.max(n_b).max(n_a);

        let matrix_entries: Vec<CyclotomicRing<F, D>> = (0..(r_max * max_stride))
            .map(|idx| {
                CyclotomicRing::from_coefficients(std::array::from_fn(|coeff| {
                    f(1_000 + (idx * D + coeff) as u128)
                }))
            })
            .collect();
        let setup = AkitaExpandedSetup {
            seed: AkitaSetupSeed {
                max_num_vars: 32,
                max_num_batched_polys: group_poly_counts.iter().sum(),
                max_num_points: num_points,
                max_stride,
                public_matrix_seed: [7u8; 32],
            },
            shared_matrix: FlatMatrix::from_ring_slice::<D>(&matrix_entries),
        };

        let eq_tau1: Vec<F> = (0..rows.next_power_of_two())
            .map(|idx| f(11 + idx as u128))
            .collect();
        let prepared = RingSwitchDeferredRowEval {
            c_alphas: (0..total_blocks).map(|idx| f(41 + idx as u128)).collect(),
            eq_tau1: eq_tau1.clone(),
            total_blocks,
            num_blocks,
            num_claims,
            depth_open,
            depth_commit,
            depth_fold,
            #[cfg(feature = "zk")]
            d_blinding_segment_len: 0,
            #[cfg(feature = "zk")]
            b_blinding_digit_planes_per_group: 0,
            #[cfg(feature = "zk")]
            b_blinding_segment_len: 0,
            block_len,
            inner_width,
            log_basis,
            n_a,
            n_d,
            n_b,
            num_commitment_groups,
            rows,
            z_first: false,
            claim_to_group: claim_to_group.clone(),
            group_poly_counts: group_poly_counts.clone(),
            num_points,
            num_public_eval_rows,
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

        let shared_view = setup
            .shared_matrix
            .ring_view::<D>(r_max, max_stride)
            .unwrap();
        let shared_rows: Vec<_> = shared_view.rows().collect();
        let d_start = 1 + num_public_eval_rows;
        let b_start = d_start + n_d;
        let a_start = b_start + n_b * num_commitment_groups;

        let mut expected = F::zero();

        for row in 0..n_d {
            let weight = eq_tau1[d_start + row];
            for claim in 0..num_claims {
                for block in 0..num_blocks {
                    for digit in 0..depth_open {
                        let d_phys_col = (claim * num_blocks + block) * depth_open + digit;
                        let m_idx = block + num_blocks * (claim + num_claims * digit);
                        expected += weight
                            * eval_ring_at_pows(&shared_rows[row][d_phys_col], &alpha_pows)
                            * eq[offset_w + m_idx];
                    }
                }
            }
        }

        for (flat_claim, &(group_idx, poly_idx)) in claim_to_group.iter().enumerate() {
            for row in 0..n_b {
                let weight = eq_tau1[b_start + group_idx * n_b + row];
                for a_idx in 0..n_a {
                    for digit in 0..depth_open {
                        for block in 0..num_blocks {
                            let phys_claim_offset = block * stride_t + a_idx * depth_open + digit;
                            let local_col = poly_idx * cols_per_poly_t + phys_claim_offset;
                            let m_idx = block
                                + num_blocks
                                    * (flat_claim
                                        + num_claims * digit
                                        + num_claims * depth_open * a_idx);
                            expected += weight
                                * eval_ring_at_pows(&shared_rows[row][local_col], &alpha_pows)
                                * eq[offset_t + m_idx];
                        }
                    }
                }
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
                                * eval_ring_at_pows(&shared_rows[row][local_col], &alpha_pows)
                                * fold_weight
                                * eq[offset_z + m_idx];
                        }
                    }
                }
            }
        }

        assert_eq!(
            got, expected,
            "fused setup contribution must follow group_poly_counts and claim_poly_indices"
        );
    }
}
