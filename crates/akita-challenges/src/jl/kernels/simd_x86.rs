//! x86_64 SIMD row projection kernels (AVX2 and AVX-512).
//!
//! Signs are `i8 {-1, 0, +1}` and digits are `i8` (`|d| <= MAX_JL_DIGIT = 32`).
//! Both are sign-extended to `i16` and fed to `madd_epi16`, which multiplies
//! `i16` pairs and horizontally adds them into `i32`: AVX2 fuses sixteen
//! coefficients (four packed bytes) per step, AVX-512 fuses thirty-two (eight
//! packed bytes).

use std::arch::x86_64::*;

use super::scalar::{SIGNS_FOR_BYTE, SIGN_LUT};

/// Gather the ternary signs for `n` packed row bytes (`n <= 8`) into the low
/// `4 * n` lanes of an `i8` buffer.
#[inline]
fn signs_for_bytes(row: &[u8], byte_base: usize, n: usize) -> [i8; 32] {
    let mut signs = [0i8; 32];
    for k in 0..n {
        signs[k * 4..k * 4 + 4].copy_from_slice(&SIGNS_FOR_BYTE[row[byte_base + k] as usize]);
    }
    signs
}

#[inline]
unsafe fn horizontal_sum_i32_128(v: __m128i) -> i32 {
    let shuf = _mm_shuffle_epi32(v, 0b01_00_11_10);
    let sums = _mm_add_epi32(v, shuf);
    let shuf2 = _mm_shuffle_epi32(sums, 0b00_01_00_01);
    let sums2 = _mm_add_epi32(sums, shuf2);
    _mm_cvtsi128_si32(sums2)
}

#[inline]
unsafe fn horizontal_sum_i32_256(v: __m256i) -> i32 {
    let lo = _mm256_castsi256_si128(v);
    let hi = _mm256_extracti128_si256(v, 1);
    horizontal_sum_i32_128(_mm_add_epi32(lo, hi))
}

#[inline]
pub(crate) fn avx512_available() -> bool {
    std::is_x86_feature_detected!("avx512f")
        && std::is_x86_feature_detected!("avx512dq")
        && std::is_x86_feature_detected!("avx512bw")
}

#[inline]
unsafe fn project_row_tail_scalar(
    row: &[u8],
    digits: &[i8],
    full_bytes: usize,
    tail_start_byte: usize,
    remainder: usize,
    mut coeff_idx: usize,
    mut sum: i32,
) -> i32 {
    for byte_idx in tail_start_byte..full_bytes {
        let signs = &SIGNS_FOR_BYTE[row[byte_idx] as usize];
        sum += i32::from(signs[0]) * i32::from(digits[coeff_idx])
            + i32::from(signs[1]) * i32::from(digits[coeff_idx + 1])
            + i32::from(signs[2]) * i32::from(digits[coeff_idx + 2])
            + i32::from(signs[3]) * i32::from(digits[coeff_idx + 3]);
        coeff_idx += 4;
    }
    if remainder > 0 {
        let byte = row[full_bytes];
        for lane in 0..remainder {
            let pair = (byte >> (lane << 1)) & 0b11;
            sum += SIGN_LUT[pair as usize] * i32::from(digits[coeff_idx]);
            coeff_idx += 1;
        }
    }
    sum
}

/// # Safety
///
/// Caller must ensure AVX2 is available and the scalar fast-path contract holds.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn project_row_avx2(row: &[u8], digits: &[i8], cols: usize) -> i32 {
    debug_assert_eq!(digits.len(), cols);

    let full_bytes = cols >> 2;
    let remainder = cols & 0b11;
    let chunk4 = full_bytes >> 2;
    let tail_start_byte = chunk4 << 2;
    let mut coeff_idx = 0usize;
    let mut acc = _mm256_setzero_si256();

    for chunk in 0..chunk4 {
        let signs = signs_for_bytes(row, chunk << 2, 4);
        let signs16 = _mm256_cvtepi8_epi16(_mm_loadu_si128(signs.as_ptr() as *const __m128i));
        let coeffs16 = _mm256_cvtepi8_epi16(_mm_loadu_si128(
            digits.as_ptr().add(coeff_idx) as *const __m128i
        ));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(signs16, coeffs16));
        coeff_idx += 16;
    }

    let sum = horizontal_sum_i32_256(acc);
    project_row_tail_scalar(
        row,
        digits,
        full_bytes,
        tail_start_byte,
        remainder,
        coeff_idx,
        sum,
    )
}

/// # Safety
///
/// Caller must ensure AVX-512F/DQ/BW are available and the scalar fast-path
/// contract holds.
#[target_feature(enable = "avx512f,avx512dq,avx512bw")]
pub(super) unsafe fn project_row_avx512(row: &[u8], digits: &[i8], cols: usize) -> i32 {
    debug_assert_eq!(digits.len(), cols);

    let full_bytes = cols >> 2;
    let remainder = cols & 0b11;
    let chunk8 = full_bytes >> 3;
    let tail_start_byte = chunk8 << 3;
    let mut coeff_idx = 0usize;
    let mut acc = _mm512_setzero_si512();

    for chunk in 0..chunk8 {
        let signs = signs_for_bytes(row, chunk << 3, 8);
        let signs16 = _mm512_cvtepi8_epi16(_mm256_loadu_si256(signs.as_ptr() as *const __m256i));
        let coeffs16 = _mm512_cvtepi8_epi16(_mm256_loadu_si256(
            digits.as_ptr().add(coeff_idx) as *const __m256i
        ));
        acc = _mm512_add_epi32(acc, _mm512_madd_epi16(signs16, coeffs16));
        coeff_idx += 32;
    }

    let sum = _mm512_reduce_add_epi32(acc);
    project_row_tail_scalar(
        row,
        digits,
        full_bytes,
        tail_start_byte,
        remainder,
        coeff_idx,
        sum,
    )
}
