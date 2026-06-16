//! AArch64 NEON row projection kernel.
//!
//! Uses the shared `SIGNS_FOR_BYTE` LUT (one load per packed byte) and a
//! 2-byte inner loop (eight coefficients per iteration) before a single
//! `vaddvq_s32` reduction.

use std::arch::aarch64::*;

use super::scalar::{SIGNS_FOR_BYTE, SIGN_LUT};

#[inline]
unsafe fn dot_byte_neon(byte: u8, coeff_ptr: *const i32) -> int32x4_t {
    let signs = vld1q_s32(SIGNS_FOR_BYTE[byte as usize].as_ptr());
    let coeffs = vld1q_s32(coeff_ptr);
    vmulq_s32(signs, coeffs)
}

#[inline]
unsafe fn dot_2bytes_neon(b0: u8, b1: u8, coeff_ptr: *const i32) -> int32x4_t {
    let partial0 = dot_byte_neon(b0, coeff_ptr);
    let partial1 = dot_byte_neon(b1, coeff_ptr.add(4));
    vaddq_s32(partial0, partial1)
}

/// # Safety
///
/// Caller must ensure NEON is available and the scalar fast-path contract holds.
#[target_feature(enable = "neon")]
pub(super) unsafe fn project_row_neon(row: &[u8], digits: &[i32], cols: usize) -> i32 {
    debug_assert_eq!(digits.len(), cols);

    let full_bytes = cols >> 2;
    let remainder = cols & 0b11;
    let pair_bytes = full_bytes >> 1;
    let tail_byte = full_bytes & 1;
    let mut coeff_idx = 0usize;
    let mut acc = vdupq_n_s32(0);

    for chunk in 0..pair_bytes {
        let byte_base = chunk * 2;
        acc = vaddq_s32(
            acc,
            dot_2bytes_neon(
                row[byte_base],
                row[byte_base + 1],
                digits.as_ptr().add(coeff_idx),
            ),
        );
        coeff_idx += 8;
    }

    if tail_byte != 0 {
        acc = vaddq_s32(
            acc,
            dot_byte_neon(row[pair_bytes * 2], digits.as_ptr().add(coeff_idx)),
        );
        coeff_idx += 4;
    }

    let mut sum = vaddvq_s32(acc);
    if remainder > 0 {
        let byte = row[full_bytes];
        for lane in 0..remainder {
            let pair = (byte >> (lane << 1)) & 0b11;
            sum += SIGN_LUT[pair as usize] * digits[coeff_idx];
            coeff_idx += 1;
        }
    }
    debug_assert_eq!(coeff_idx, cols);
    sum
}
