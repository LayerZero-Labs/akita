//! AArch64 NEON row projection kernel (stable intrinsics).
//!
//! Signs are `i8 {-1, +1}` and digits are `i8` (`|d| <= MAX_JL_DIGIT = 32`),
//! so the hot operands are one byte each. Each step widens sixteen `i8` signs
//! and sixteen `i8` digits to `i16` and accumulates with `vmlal_s16` into two
//! `i32x4` lanes (broken into two accumulators for instruction-level
//! parallelism), then reduces with `vaddvq_s32`.
//!
//! `vdotq_s32` (the Armv8.2 dot-product instruction, sixteen MACs per op) would
//! be ~2x denser here, but its intrinsic is still unstable (`stdarch_neon_dotprod`)
//! on the pinned toolchain; reaching it would need inline asm. Deferred.

use std::arch::aarch64::*;

use super::scalar::{BINARY_SIGNS_FOR_BYTE, SIGN_LUT};

/// Gather the sixteen binary signs for two packed row bytes into one vector.
#[inline]
unsafe fn signs_for_2bytes(row: &[u8], byte_base: usize) -> int8x16_t {
    let mut signs = [0i8; 16];
    signs[0..8].copy_from_slice(&BINARY_SIGNS_FOR_BYTE[row[byte_base] as usize]);
    signs[8..16].copy_from_slice(&BINARY_SIGNS_FOR_BYTE[row[byte_base + 1] as usize]);
    vld1q_s8(signs.as_ptr())
}

/// # Safety
///
/// Caller must ensure NEON is available and the scalar fast-path contract holds
/// (`digits.len() == cols`, `|digits[i]| <= MAX_JL_DIGIT`).
#[target_feature(enable = "neon")]
pub(super) unsafe fn project_row_neon(row: &[u8], digits: &[i8], cols: usize) -> i32 {
    debug_assert_eq!(digits.len(), cols);

    let full_bytes = cols >> 3;
    let remainder = cols & 0b111;
    let chunk2 = full_bytes >> 1;
    let tail_start_byte = chunk2 << 1;
    let mut coeff_idx = 0usize;
    let mut acc_a = vdupq_n_s32(0);
    let mut acc_b = vdupq_n_s32(0);

    for chunk in 0..chunk2 {
        let signs = signs_for_2bytes(row, chunk << 1);
        let coeffs = vld1q_s8(digits.as_ptr().add(coeff_idx));

        let s_lo = vmovl_s8(vget_low_s8(signs));
        let s_hi = vmovl_s8(vget_high_s8(signs));
        let d_lo = vmovl_s8(vget_low_s8(coeffs));
        let d_hi = vmovl_s8(vget_high_s8(coeffs));

        acc_a = vmlal_s16(acc_a, vget_low_s16(s_lo), vget_low_s16(d_lo));
        acc_b = vmlal_s16(acc_b, vget_high_s16(s_lo), vget_high_s16(d_lo));
        acc_a = vmlal_s16(acc_a, vget_low_s16(s_hi), vget_low_s16(d_hi));
        acc_b = vmlal_s16(acc_b, vget_high_s16(s_hi), vget_high_s16(d_hi));
        coeff_idx += 16;
    }

    let mut sum = vaddvq_s32(vaddq_s32(acc_a, acc_b));

    for byte_idx in tail_start_byte..full_bytes {
        let signs = &BINARY_SIGNS_FOR_BYTE[row[byte_idx] as usize];
        sum += i32::from(signs[0]) * i32::from(digits[coeff_idx])
            + i32::from(signs[1]) * i32::from(digits[coeff_idx + 1])
            + i32::from(signs[2]) * i32::from(digits[coeff_idx + 2])
            + i32::from(signs[3]) * i32::from(digits[coeff_idx + 3])
            + i32::from(signs[4]) * i32::from(digits[coeff_idx + 4])
            + i32::from(signs[5]) * i32::from(digits[coeff_idx + 5])
            + i32::from(signs[6]) * i32::from(digits[coeff_idx + 6])
            + i32::from(signs[7]) * i32::from(digits[coeff_idx + 7]);
        coeff_idx += 8;
    }

    if remainder > 0 {
        let byte = row[full_bytes];
        for lane in 0..remainder {
            let bit = (byte >> lane) & 1;
            sum += SIGN_LUT[bit as usize] * i32::from(digits[coeff_idx]);
            coeff_idx += 1;
        }
    }
    debug_assert_eq!(coeff_idx, cols);
    sum
}
