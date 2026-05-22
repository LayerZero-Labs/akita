//! AArch64 NEON kernel for sparse-multiply-accumulate in the decompose-fold
//! pipeline.
//!
//! Called from [`crate::backend::poly_helpers::integer_mul_acc`] when NEON is
//! available and challenge coefficients have magnitude ≤ 2.  Rotates an i8
//! digit plane by each challenge position and accumulates into an i64
//! accumulator.

/// NEON integer-multiply-accumulate.
///
/// For each challenge term `(pos, coeff)`, rotates the `digit_plane` by `pos`
/// positions in the negacyclic ring (X^D + 1) and adds or subtracts the
/// widened i8 values into the i64 `acc`. Small magnitudes like `+/-2` reuse
/// the unit add/sub kernel multiple times. Callers MUST gate on
/// `|coeff| <= 2` for every term before invoking this entry point; the
/// |coeff| > 2 fallback lives on the scalar path (see
/// [`crate::backend::poly_helpers::integer_mul_acc_scalar`]).
///
/// # Safety
///
/// - `digit_plane` must point to at least `d` valid i8 values.
/// - `acc` must point to at least `d` valid i64 values.
/// - `d` must be a multiple of 16.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn integer_mul_acc_neon(
    digit_plane: *const i8,
    acc: *mut i64,
    d: usize,
    positions: &[u32],
    coeffs: &[i32],
) {
    debug_assert!(d.is_multiple_of(16));

    for (&pos, &coeff) in positions.iter().zip(coeffs.iter()) {
        let p = pos as usize;
        let split = d - p;

        match coeff {
            1 => acc_rotated_add(digit_plane, acc, d, p, split),
            -1 => acc_rotated_sub(digit_plane, acc, d, p, split),
            2 => {
                acc_rotated_add(digit_plane, acc, d, p, split);
                acc_rotated_add(digit_plane, acc, d, p, split);
            }
            -2 => {
                acc_rotated_sub(digit_plane, acc, d, p, split);
                acc_rotated_sub(digit_plane, acc, d, p, split);
            }
            _ => debug_assert!(false, "caller must gate large coeffs to scalar path"),
        }
    }
}

/// Add rotated digit plane: acc[i+p] += digits[i] for i in [0, split),
/// acc[i-split] -= digits[i] for i in [split, D) (negacyclic wrap).
#[inline(always)]
unsafe fn acc_rotated_add(digits: *const i8, acc: *mut i64, d: usize, p: usize, split: usize) {
    // First segment: digits[0..split] -> acc[p..D], ADD
    acc_segment_add(digits, acc.add(p), split);
    // Second segment: digits[split..D] -> acc[0..p], SUB (negacyclic)
    if p > 0 {
        acc_segment_sub(digits.add(split), acc, p);
    }
    let _ = d;
}

/// Sub rotated digit plane: acc[i+p] -= digits[i] for i in [0, split),
/// acc[i-split] += digits[i] for i in [split, D) (negacyclic wrap).
#[inline(always)]
unsafe fn acc_rotated_sub(digits: *const i8, acc: *mut i64, d: usize, p: usize, split: usize) {
    // First segment: digits[0..split] -> acc[p..D], SUB
    acc_segment_sub(digits, acc.add(p), split);
    // Second segment: digits[split..D] -> acc[0..p], ADD (negacyclic)
    if p > 0 {
        acc_segment_add(digits.add(split), acc, p);
    }
    let _ = d;
}

/// Widen i8 source values to i64 and ADD into accumulator.
#[inline(always)]
unsafe fn acc_segment_add(src: *const i8, dst: *mut i64, len: usize) {
    for i in 0..len {
        *dst.add(i) += i64::from(*src.add(i));
    }
}

/// Widen i8 source values to i64 and SUB from accumulator.
#[inline(always)]
unsafe fn acc_segment_sub(src: *const i8, dst: *mut i64, len: usize) {
    for i in 0..len {
        *dst.add(i) -= i64::from(*src.add(i));
    }
}
