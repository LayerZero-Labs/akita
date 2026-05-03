//! Prover-owned helpers for the Akita ring-switch handoff.

use crate::RecursiveWitnessFlat;
use akita_algebra::eq_poly::EqPolynomial;
use akita_algebra::ring::cyclotomic::BalancedDecomposePow2I8Params;
use akita_algebra::ring::eval_ring_at_pows;
use akita_algebra::{CyclotomicRing, SparseChallenge};
use akita_challenges::eval_sparse_challenge_at_pows;
use akita_field::parallel::*;
use akita_field::{CanonicalField, FieldCore, HachiError};
use akita_types::{
    checked_num_claims_from_group_sizes, gadget_row_scalars, r_decomp_levels,
    validate_opening_points_for_claims, FlatDigitBlocks, FlatRingVec, HachiExpandedSetup,
    LevelParams, RingOpeningPoint,
};

/// D-agnostic output of the ring switch protocol, containing everything
/// needed for sumchecks and level chaining.
pub struct RingSwitchOutput<F: FieldCore> {
    /// The witness vector w as balanced digits in `[-b/2, b/2)`.
    pub w: RecursiveWitnessFlat,
    /// Runtime commitment to w.
    pub w_commitment: Option<FlatRingVec<F>>,
    /// Runtime-only prover hint cache for the w-commitment.
    pub w_hint: Option<crate::RecursiveCommitmentHintCache<F>>,
    /// Compact evaluation table of w, stored as x-outer/y-inner slices.
    pub w_evals_compact: Vec<i8>,
    /// Physical x width before zero-extension to the next power of two.
    pub live_x_cols: usize,
    /// Evaluation table of M_alpha(x) (tau1-weighted).
    pub m_evals_x: Vec<F>,
    /// Evaluation table of alpha powers (y dimension).
    pub alpha_evals_y: Vec<F>,
    /// Number of upper variable bits.
    pub col_bits: usize,
    /// Number of lower variable bits.
    pub ring_bits: usize,
    /// Challenge tau0 for F_0 sumcheck.
    pub tau0: Vec<F>,
    /// Challenge tau1 for F_alpha sumcheck.
    pub tau1: Vec<F>,
    /// Basis size b = 2^LOG_BASIS.
    pub b: usize,
    /// Ring-switch challenge alpha.
    pub alpha: F,
}

/// Produce the compact `Vec<i8>` eval table of `w` for the fused prover.
///
/// The compact witness stays in the raw [`build_w_coeffs`] order:
/// `w[x * y_len + y]`, with x outer and y inner.
///
/// # Errors
///
/// Returns an error if the witness length is not divisible by the ring
/// dimension.
pub fn build_w_evals_compact(w: &[i8], d: usize) -> Result<(Vec<i8>, usize, usize), HachiError> {
    if !w.len().is_multiple_of(d) {
        return Err(HachiError::InvalidSize {
            expected: d,
            actual: w.len(),
        });
    }
    let ring_bits = d.trailing_zeros() as usize;
    let live_x_cols = w.len() / d;
    let col_bits = live_x_cols.next_power_of_two().trailing_zeros() as usize;
    Ok((w.to_vec(), col_bits, ring_bits))
}

/// Unified M-table evaluation for the batched CWSS protocol.
///
/// `opening_points` holds the distinct ring-level opening points used by the
/// batch, `claim_to_point` maps each flattened claim index to its opening-point
/// index, and `gamma` provides the per-claim random linear-combination
/// coefficients. The matrix carries one public y-row per distinct opening
/// point (`num_eval_rows = opening_points.len()`).
///
/// # Errors
///
/// Returns an error if the batch shape, opening-point layout, challenge count,
/// or expanded matrix dimensions are inconsistent.
#[allow(clippy::too_many_arguments)]
#[tracing::instrument(skip_all, name = "compute_m_evals_x_batched")]
pub fn compute_m_evals_x<F: FieldCore + CanonicalField, const D: usize>(
    setup: &HachiExpandedSetup<F>,
    opening_points: &[RingOpeningPoint<F>],
    claim_to_point: &[usize],
    challenges: &[SparseChallenge],
    alpha: F,
    alpha_pows: &[F],
    lp: &LevelParams,
    tau1: &[F],
    claim_group_sizes: &[usize],
    gamma: &[F],
    num_eval_rows: usize,
) -> Result<Vec<F>, HachiError> {
    if alpha_pows.len() != D {
        return Err(HachiError::InvalidSize {
            expected: D,
            actual: alpha_pows.len(),
        });
    }
    let num_claims = checked_num_claims_from_group_sizes(claim_group_sizes)?;
    validate_opening_points_for_claims(opening_points, claim_to_point, lp, num_claims)?;
    let num_commitment_groups = claim_group_sizes.len();

    let depth_commit = lp.num_digits_commit;
    let depth_open = lp.num_digits_open;
    let depth_fold = lp.num_digits_fold;
    let log_basis = lp.log_basis;
    let num_blocks = lp.num_blocks;
    let total_blocks = num_blocks
        .checked_mul(num_claims)
        .ok_or_else(|| HachiError::InvalidSetup("batched block count overflow".to_string()))?;
    if challenges.len() != total_blocks {
        return Err(HachiError::InvalidSize {
            expected: total_blocks,
            actual: challenges.len(),
        });
    }
    let block_len = lp.block_len;
    let w_len = depth_open * total_blocks;
    let n_a = lp.a_key.row_len();
    let n_b = lp.b_key.row_len();
    let n_d = lp.d_key.row_len();
    let t_len = depth_open * n_a * total_blocks;
    let inner_width = block_len * depth_commit;
    let z_base_len = opening_points
        .len()
        .checked_mul(inner_width)
        .ok_or_else(|| HachiError::InvalidSetup("batched z width overflow".to_string()))?;
    let z_len = depth_fold
        .checked_mul(z_base_len)
        .ok_or_else(|| HachiError::InvalidSetup("batched z width overflow".to_string()))?;
    let rows = lp.m_row_count(num_commitment_groups, num_eval_rows);
    let levels = r_decomp_levels::<F>(log_basis);
    let total_cols = w_len
        .checked_add(t_len)
        .and_then(|cols| cols.checked_add(z_len))
        .and_then(|cols| cols.checked_add(rows.checked_mul(levels)?))
        .ok_or_else(|| HachiError::InvalidSetup("expanded M width overflow".to_string()))?;

    let eq_tau1 = EqPolynomial::evals(tau1);
    if eq_tau1.len() < rows {
        return Err(HachiError::InvalidSize {
            expected: rows,
            actual: eq_tau1.len(),
        });
    }

    let g1_open = gadget_row_scalars::<F>(depth_open, log_basis);
    let g1_commit = gadget_row_scalars::<F>(depth_commit, log_basis);
    let fold_gadget = gadget_row_scalars::<F>(depth_fold, log_basis);
    let r_gadget = gadget_row_scalars::<F>(levels, log_basis);
    let x_len = total_cols.next_power_of_two();
    let mut out = Vec::with_capacity(x_len);

    let c_alphas: Vec<F> = challenges
        .iter()
        .map(|challenge| eval_sparse_challenge_at_pows::<F, D>(challenge, alpha_pows))
        .collect::<Result<_, _>>()?;

    let stride = setup.seed.max_stride;
    let d_view = setup.shared_matrix.ring_view::<D>(n_d, stride);
    let b_view = setup.shared_matrix.ring_view::<D>(n_b, stride);
    let a_view = setup.shared_matrix.ring_view::<D>(n_a, stride);

    // Row layout: consistency (1) | public (num_eval_rows) | D (n_d) |
    //             B (n_b * num_commitment_groups) | A (n_a)
    let commitment_row_count = n_b * num_commitment_groups;
    let consistency_weight = eq_tau1[0];
    let public_weights = &eq_tau1[1..(1 + num_eval_rows)];
    let d_start = 1 + num_eval_rows;
    let b_start = d_start + n_d;
    let a_start = b_start + commitment_row_count;
    let a_weights = &eq_tau1[a_start..rows];
    let claim_to_group: Vec<(usize, usize)> = claim_group_sizes
        .iter()
        .enumerate()
        .flat_map(|(group_idx, &group_size)| {
            (0..group_size).map(move |within_group| (group_idx, within_group))
        })
        .collect();

    let t_compound_per_block = n_a * depth_open;

    let w_segment: Vec<F> = cfg_into_iter!(0..w_len)
        .map(|x| {
            let dig = x / total_blocks;
            let blk = x % total_blocks;
            let claim_idx = blk / num_blocks;
            let block_idx = blk % num_blocks;
            let d_phys_col = blk * depth_open + dig;
            let point_idx = claim_to_point[claim_idx];
            let opening_point = &opening_points[point_idx];
            // The public row weight is per-point: each opening point
            // contributes its own public y-row (one row per point).
            let mut acc =
                (public_weights[point_idx] * gamma[claim_idx] * opening_point.b[block_idx]
                    + consistency_weight * c_alphas[blk])
                    * g1_open[dig];
            for (di, eq_i) in eq_tau1[d_start..(d_start + n_d)].iter().enumerate() {
                if !eq_i.is_zero() {
                    acc += *eq_i * eval_ring_at_pows(&d_view.row(di)[d_phys_col], alpha_pows);
                }
            }
            acc
        })
        .collect();

    let t_cols_per_claim = t_compound_per_block * num_blocks;
    let t_segment: Vec<F> = cfg_into_iter!(0..t_len)
        .map(|x| {
            let compound_dig = x / total_blocks;
            let blk = x % total_blocks;
            let a_idx = compound_dig / depth_open;
            let digit_idx = compound_dig % depth_open;
            let claim_idx = blk / num_blocks;
            let block_idx = blk % num_blocks;
            let (group_idx, claim_idx_within_group) = claim_to_group[claim_idx];
            let phys_claim_offset =
                block_idx * t_compound_per_block + a_idx * depth_open + digit_idx;
            let local_col = claim_idx_within_group * t_cols_per_claim + phys_claim_offset;
            let commitment_weights =
                &eq_tau1[(b_start + group_idx * n_b)..(b_start + (group_idx + 1) * n_b)];
            let mut acc = a_weights[a_idx] * c_alphas[blk] * g1_open[digit_idx];
            for (row_idx, eq_i) in commitment_weights.iter().enumerate() {
                if !eq_i.is_zero() {
                    acc += *eq_i * eval_ring_at_pows(&b_view.row(row_idx)[local_col], alpha_pows);
                }
            }
            acc
        })
        .collect();

    let z_base: Vec<F> = cfg_into_iter!(0..z_base_len)
        .map(|k| {
            let point_idx = k / inner_width;
            let local_k = k % inner_width;
            let block_idx = local_k / depth_commit;
            let digit_idx = local_k % depth_commit;
            let opening_point = &opening_points[point_idx];
            let mut acc = consistency_weight * opening_point.a[block_idx] * g1_commit[digit_idx];
            for (a_idx, eq_i) in a_weights.iter().enumerate() {
                if !eq_i.is_zero() {
                    acc += *eq_i * eval_ring_at_pows(&a_view.row(a_idx)[local_k], alpha_pows);
                }
            }
            acc
        })
        .collect();

    let num_points = opening_points.len();
    let z_total_blocks = num_points * block_len;
    let z_segment: Vec<F> = cfg_into_iter!(0..z_len)
        .map(|x| {
            let compound_dig = x / z_total_blocks;
            let global_blk = x % z_total_blocks;
            let dc = compound_dig / depth_fold;
            let df = compound_dig % depth_fold;
            let point_idx = global_blk / block_len;
            let blk = global_blk % block_len;
            let phys_k = point_idx * inner_width + blk * depth_commit + dc;
            -(z_base[phys_k] * fold_gadget[df])
        })
        .collect();

    let alpha_pow_d = alpha_pows[D - 1] * alpha;
    let denom = alpha_pow_d + F::one();
    let r_tail_len = rows * levels;
    let r_tail: Vec<F> = cfg_into_iter!(0..r_tail_len)
        .map(|idx| {
            let row_idx = idx / levels;
            let level_idx = idx % levels;
            -(eq_tau1[row_idx] * denom * r_gadget[level_idx])
        })
        .collect();

    let z_first = lp.m_vars >= lp.r_vars;
    if z_first {
        out.extend(z_segment);
        out.extend(w_segment);
        out.extend(t_segment);
    } else {
        out.extend(w_segment);
        out.extend(t_segment);
        out.extend(z_segment);
    }
    out.extend(r_tail);
    out.resize(x_len, F::zero());
    Ok(out)
}

fn balanced_decompose_centered_i32_i8_into<const D: usize>(
    centered: &[i32; D],
    out: &mut [[i8; D]],
    log_basis: u32,
) {
    let levels = out.len();
    assert!(
        log_basis > 0 && log_basis <= 6,
        "log_basis must be in 1..=6 for i8 output"
    );
    assert!(
        (levels as u32).saturating_mul(log_basis) <= 128 + log_basis,
        "levels * log_basis must be <= 128 + log_basis"
    );

    let half_b = 1i128 << (log_basis - 1);
    let b = half_b << 1;
    let mask = b - 1;

    for coeff_idx in 0..D {
        let mut c = centered[coeff_idx] as i128;
        for plane in out.iter_mut() {
            let d = c & mask;
            let balanced = if d >= half_b { d - b } else { d };
            c = (c - balanced) >> log_basis;
            plane[coeff_idx] = balanced as i8;
        }
    }
}

/// Transpose block-major digit planes to digit-major order (block index
/// innermost): for each compound digit index, emit all blocks in order.
fn emit_planes_block_inner<const D: usize>(
    out: &mut Vec<i8>,
    flat: &[[i8; D]],
    total_blocks: usize,
    planes_per_block: usize,
) {
    debug_assert_eq!(
        flat.len(),
        total_blocks * planes_per_block,
        "emit_planes_block_inner: flat.len()={} != total_blocks({}) * planes_per_block({})",
        flat.len(),
        total_blocks,
        planes_per_block
    );
    for compound_dig in 0..planes_per_block {
        for blk in 0..total_blocks {
            out.extend_from_slice(&flat[blk * planes_per_block + compound_dig]);
        }
    }
}

/// Decompose z_pre elements and emit in digit-major order.
///
/// z_pre has `num_points * block_len * depth_commit` elements indexed as
/// `z[point * inner_width + blk * depth_commit + dc]`. Each decomposes into
/// `num_digits_fold` planes.
///
/// Output order: for each `(dc, df)`, emit all `(point, blk)` pairs with
/// the global block index `point * block_len + blk` innermost.
fn emit_z_pre_block_inner<const D: usize>(
    out: &mut Vec<i8>,
    z_pre_centered: &[[i32; D]],
    block_len: usize,
    depth_commit: usize,
    num_digits_fold: usize,
    log_basis: u32,
) {
    let total_elems = z_pre_centered.len();
    let inner_width = block_len * depth_commit;
    debug_assert_eq!(
        total_elems % inner_width,
        0,
        "z_pre length {total_elems} not divisible by inner_width {inner_width}",
    );
    let num_points = total_elems / inner_width;

    let mut all_planes = vec![[0i8; D]; total_elems * num_digits_fold];
    for (k, z_j) in z_pre_centered.iter().enumerate() {
        balanced_decompose_centered_i32_i8_into(
            z_j,
            &mut all_planes[k * num_digits_fold..(k + 1) * num_digits_fold],
            log_basis,
        );
    }

    for dc in 0..depth_commit {
        for df in 0..num_digits_fold {
            for pt in 0..num_points {
                for blk in 0..block_len {
                    let k = pt * inner_width + blk * depth_commit + dc;
                    out.extend_from_slice(&all_planes[k * num_digits_fold + df]);
                }
            }
        }
    }
}

/// Build the committed witness polynomial from ring-domain digit planes.
///
/// Emits field-domain coefficients in digit-major order (block index innermost)
/// with adaptive segment ordering: the segment whose block dimension is the
/// larger power of two comes first.
///
/// Segment ordering:
/// - If `m_vars >= r_vars`: z-hat (`2^m` blocks), e-hat + t-hat (`2^r` blocks), r-hat
/// - If `m_vars < r_vars`: e-hat + t-hat (`2^r` blocks), z-hat (`2^m` blocks), r-hat
///
/// Within each segment, the power-of-2 block index is the fastest-varying
/// (innermost) dimension.
///
/// `FlatDigitBlocks` stores ring-domain data in block-major order (all digit
/// planes for one block contiguously), which is natural for ring-domain matvec
/// and recomposition. This function transposes to digit-major at the
/// ring-to-field boundary. An alternative would be propagating digit-major
/// throughout `FlatDigitBlocks`, eliminating this transposition but requiring
/// restructured producers and block-level operations.
pub fn build_w_coeffs<F: CanonicalField, const D: usize>(
    w_hat: &FlatDigitBlocks<D>,
    t_hat: &FlatDigitBlocks<D>,
    z_pre_centered: &[[i32; D]],
    r: &[CyclotomicRing<F, D>],
    lp: &LevelParams,
) -> RecursiveWitnessFlat {
    let log_basis = lp.log_basis;
    let num_digits_fold = lp.num_digits_fold;
    let depth_open = lp.num_digits_open;
    let depth_commit = lp.num_digits_commit;
    let block_len = lp.block_len;
    let levels = r_decomp_levels::<F>(log_basis);

    let w_hat_planes = w_hat.flat_digits().len();
    let t_hat_planes = t_hat.flat_digits().len();
    let z_count = w_hat_planes + t_hat_planes + z_pre_centered.len() * num_digits_fold;
    let r_hat_count = r.len() * levels;
    let z_first = lp.m_vars >= lp.r_vars;
    tracing::debug!(
        w_hat_planes,
        t_hat_planes,
        z_pre_elems = z_pre_centered.len(),
        z_pre_planes = z_pre_centered.len() * num_digits_fold,
        r_elems = r.len(),
        r_planes = r_hat_count,
        total_ring = z_count + r_hat_count,
        total_field = (z_count + r_hat_count) * D,
        z_first,
        "build_w_coeffs"
    );
    let total_planes = z_count + r_hat_count;
    let total_elems = total_planes * D;

    let mut out = Vec::with_capacity(total_elems);

    let total_blocks_et = if depth_open > 0 {
        w_hat_planes / depth_open
    } else {
        0
    };
    let t_planes_per_block = if total_blocks_et > 0 {
        t_hat_planes / total_blocks_et
    } else {
        0
    };

    if z_first {
        emit_z_pre_block_inner(
            &mut out,
            z_pre_centered,
            block_len,
            depth_commit,
            num_digits_fold,
            log_basis,
        );
        emit_planes_block_inner(&mut out, w_hat.flat_digits(), total_blocks_et, depth_open);
        emit_planes_block_inner(
            &mut out,
            t_hat.flat_digits(),
            total_blocks_et,
            t_planes_per_block,
        );
    } else {
        emit_planes_block_inner(&mut out, w_hat.flat_digits(), total_blocks_et, depth_open);
        emit_planes_block_inner(
            &mut out,
            t_hat.flat_digits(),
            total_blocks_et,
            t_planes_per_block,
        );
        emit_z_pre_block_inner(
            &mut out,
            z_pre_centered,
            block_len,
            depth_commit,
            num_digits_fold,
            log_basis,
        );
    }

    let mut r_planes = vec![[0i8; D]; levels];
    let q = (-F::one()).to_canonical_u128() + 1;
    let decompose_params = BalancedDecomposePow2I8Params::new(levels, log_basis, q);
    for ri in r {
        r_planes.fill([0i8; D]);
        ri.balanced_decompose_pow2_i8_into_with_params(&mut r_planes, &decompose_params);
        for plane in &r_planes {
            out.extend_from_slice(plane);
        }
    }
    RecursiveWitnessFlat::from_i8_digits(out)
}

#[cfg(test)]
mod tests {
    use super::balanced_decompose_centered_i32_i8_into;
    use akita_algebra::{CyclotomicRing, Prime128OffsetA7F7};
    use akita_field::FromSmallInt;
    use std::array::from_fn;

    #[test]
    fn centered_i32_decompose_matches_ring_decompose() {
        type F = Prime128OffsetA7F7;
        const D: usize = 128;

        let centered = from_fn(|i| ((37 * i as i32 + 11) % 95) - 47);
        let ring =
            CyclotomicRing::<F, D>::from_coefficients(from_fn(|i| F::from_i64(centered[i] as i64)));

        for (num_digits, log_basis) in [
            (7usize, 3u32),
            (10usize, 2u32),
            (5usize, 5u32),
            (4usize, 6u32),
        ] {
            let mut got = vec![[0i8; D]; num_digits];
            balanced_decompose_centered_i32_i8_into(&centered, &mut got, log_basis);

            let mut expected = vec![[0i8; D]; num_digits];
            ring.balanced_decompose_pow2_i8_into(&mut expected, log_basis);
            assert_eq!(
                got, expected,
                "centered i32 decomposition mismatch for num_digits={num_digits} log_basis={log_basis}"
            );
        }
    }
}
