use super::*;
use crate::validation::{is_i8_log_basis, validate_i8_input_log_basis};

#[inline]
pub(super) fn accumulate_pointwise_product_into<W: PrimeWidth, const K: usize, const D: usize>(
    acc: &mut CyclotomicCrtNtt<W, K, D>,
    lhs: &CyclotomicCrtNtt<W, K, D>,
    rhs: &CyclotomicCrtNtt<W, K, D>,
    params: &CrtNttParamSet<W, K, D>,
) {
    acc.add_assign_pointwise_mul(lhs, rhs, params);
}

#[inline]
pub(super) fn is_zero_plane<const D: usize>(plane: &[i8; D]) -> bool {
    plane.iter().all(|&d| d == 0)
}

#[inline]
pub(super) fn is_zero_centered_row<const D: usize>(row: &[i32; D]) -> bool {
    row.iter().all(|&d| d == 0)
}

pub(super) fn quotient_from_cyclic_and_negacyclic<F: FieldCore + HalvingField, const D: usize>(
    cyclic: &CyclotomicRing<F, D>,
    negacyclic: &CyclotomicRing<F, D>,
) -> CyclotomicRing<F, D> {
    let cyc = cyclic.coefficients();
    let neg = negacyclic.coefficients();
    CyclotomicRing::from_coefficients(from_fn(|k| (cyc[k] - neg[k]).half()))
}

pub(super) fn add_cyclic_product_into<F: FieldCore, const D: usize>(
    acc: &mut CyclotomicRing<F, D>,
    lhs: &CyclotomicRing<F, D>,
    rhs: &CyclotomicRing<F, D>,
) {
    for (i, &a) in lhs.coefficients().iter().enumerate() {
        if a.is_zero() {
            continue;
        }
        for (j, &b) in rhs.coefficients().iter().enumerate() {
            if !b.is_zero() {
                acc.coefficients_mut()[(i + j) % D] += a * b;
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
pub(super) const TARGET_L2_CACHE_BYTES: usize = 4 * 1024 * 1024;
#[cfg(not(target_arch = "aarch64"))]
pub(super) const TARGET_L2_CACHE_BYTES: usize = 1024 * 1024;
pub(super) const CENTERED_LUT_MAX_ABS: u32 = (1 << 16) - 1;
// Row-count ceiling for the block-parallel matvec. Commitments up to `n_a == 7`
// still parallelize over blocks through the generic accumulator loop instead of
// falling back to the column-tiled path, which has too few tiles to scale at
// high nv. The block-parallel and column-tiled paths produce identical ring
// output (per-step `reduce_range` accumulation + canonicalizing `to_ring`), so
// raising the cap is a pure performance change.
pub(super) const SMALL_ROW_BLOCK_PARALLEL_MAX_ROWS: usize = 7;
pub(super) const SMALL_ROW_BLOCK_PARALLEL_MIN_BLOCKS: usize = 16;

#[inline]
pub(super) fn validate_i8_log_basis(log_basis: u32) -> Result<(), AkitaError> {
    validate_i8_input_log_basis(log_basis, "for i8 NTT kernels")
}

#[inline]
pub(super) fn balanced_digit_abs_bound(log_basis: u32) -> u64 {
    debug_assert!(is_i8_log_basis(log_basis));
    1u64 << (log_basis - 1)
}

/// Whether every coefficient across `blocks` (fold-major) is a balanced gadget
/// digit for `log_basis`, i.e. lies in `[-2^(log_basis-1), 2^(log_basis-1))`.
///
/// A `num_digits_inner == 1` recursive witness is a raw signed-i8 coefficient
/// stream: degree-one fields yield balanced digits (the fast predecomposed
/// digit commit applies), but extension-field tensor base-lift packing sums
/// gadget digits and can exceed this range, requiring the general raw ring
/// mat-vec. This predicate selects between the two.
#[inline]
pub(crate) fn digit_blocks_are_balanced<const D: usize>(
    blocks: &[&[[i8; D]]],
    num_cols: usize,
    log_basis: u32,
) -> bool {
    if !is_i8_log_basis(log_basis) {
        return false;
    }
    let bound = balanced_digit_abs_bound(log_basis);
    blocks
        .iter()
        .all(|block| digit_rows_within_digit_bound(block, num_cols.min(block.len()), bound))
}

#[inline]
pub(super) fn digit_rows_within_digit_bound<const D: usize>(
    rows: &[[i8; D]],
    len: usize,
    digit_bound: u64,
) -> bool {
    let bound = i16::try_from(digit_bound).unwrap_or(i16::MAX);
    rows.iter()
        .take(len)
        .flat_map(|row| row.iter())
        .all(|&coeff| (-bound..bound).contains(&i16::from(coeff)))
}

#[inline]
pub(super) fn validate_digit_rows_for_log_basis<const D: usize>(
    rows: &[[i8; D]],
    len: usize,
    log_basis: u32,
    context: &str,
) -> Result<(), AkitaError> {
    let bound = 1i16 << (log_basis - 1);
    if rows
        .iter()
        .take(len)
        .flat_map(|row| row.iter())
        .all(|&coeff| (-bound..bound).contains(&i16::from(coeff)))
    {
        Ok(())
    } else {
        let offending = rows
            .iter()
            .take(len)
            .enumerate()
            .flat_map(|(row, coeffs)| {
                coeffs
                    .iter()
                    .enumerate()
                    .map(move |(col, &coeff)| (row, col, coeff))
            })
            .find(|&(_, _, coeff)| !(-bound..bound).contains(&i16::from(coeff)));
        Err(AkitaError::InvalidInput(format!(
            "predecomposed digit row contains a coefficient outside the balanced log_basis range {context}: log_basis={log_basis}, offending={offending:?}"
        )))
    }
}

#[inline]
pub(super) fn aligned_i8_tile_width(
    raw_width: usize,
    inner_width: usize,
    num_digits: usize,
) -> usize {
    debug_assert!(inner_width > 0);
    debug_assert!(num_digits > 0);

    if inner_width <= num_digits {
        return inner_width;
    }

    let clamped = raw_width.min(inner_width).max(num_digits);
    ((clamped / num_digits).max(1)) * num_digits
}

#[inline]
pub(super) fn capacity_safe_i8_chunk_width(
    safe_width: usize,
    inner_width: usize,
    num_digits: usize,
) -> usize {
    debug_assert!(safe_width > 0);
    debug_assert!(inner_width > 0);
    debug_assert!(num_digits > 0);

    if safe_width < num_digits {
        safe_width.min(inner_width)
    } else {
        aligned_i8_tile_width(safe_width, inner_width, num_digits).min(safe_width)
    }
}

#[inline]
pub(super) fn add_ntt_into<W: PrimeWidth, const K: usize, const D: usize>(
    acc: &mut CyclotomicCrtNtt<W, K, D>,
    other: &CyclotomicCrtNtt<W, K, D>,
    params: &CrtNttParamSet<W, K, D>,
) {
    acc.add_assign_reduced(other, params);
}
