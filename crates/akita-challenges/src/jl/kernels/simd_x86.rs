//! x86_64 SIMD row projection kernels (AVX2 and AVX-512).
//!
//! Each packed row byte encodes four ternary signs; AVX2 fuses one byte (four
//! `i32` lanes) per step, AVX-512 fuses four bytes (sixteen lanes).

use std::arch::x86_64::*;

use super::scalar::{SIGNS_FOR_BYTE, SIGN_LUT};

#[inline]
unsafe fn dot_byte_sse(byte: u8, coeff_ptr: *const i32) -> __m128i {
    let signs = _mm_loadu_si128(SIGNS_FOR_BYTE[byte as usize].as_ptr() as *const __m128i);
    let coeffs = _mm_loadu_si128(coeff_ptr as *const __m128i);
    _mm_mullo_epi32(signs, coeffs)
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
unsafe fn dot_4bytes_avx512(bytes: &[u8; 4], coeff_ptr: *const i32) -> __m512i {
    let mut signs = [0i32; 16];
    for (idx, &byte) in bytes.iter().enumerate() {
        signs[idx * 4..idx * 4 + 4].copy_from_slice(&SIGNS_FOR_BYTE[byte as usize]);
    }
    let signs_v = _mm512_loadu_si512(signs.as_ptr() as *const __m512i);
    let coeffs = _mm512_loadu_si512(coeff_ptr as *const __m512i);
    _mm512_mullo_epi32(signs_v, coeffs)
}

#[inline]
pub(crate) fn avx512_available() -> bool {
    std::is_x86_feature_detected!("avx512f")
        && std::is_x86_feature_detected!("avx512dq")
        && std::is_x86_feature_detected!("avx512bw")
}

#[inline]
unsafe fn project_row_remainder_scalar(
    row: &[u8],
    digits: &[i32],
    full_bytes: usize,
    mut coeff_idx: usize,
    remainder: usize,
    mut sum: i32,
) -> i32 {
    if remainder > 0 {
        let byte = row[full_bytes];
        for lane in 0..remainder {
            let pair = (byte >> (lane << 1)) & 0b11;
            sum += SIGN_LUT[pair as usize] * digits[coeff_idx];
            coeff_idx += 1;
        }
    }
    sum
}

/// # Safety
///
/// Caller must ensure AVX2 is available and the scalar fast-path contract holds.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn project_row_avx2(row: &[u8], digits: &[i32], cols: usize) -> i32 {
    debug_assert_eq!(digits.len(), cols);

    let full_bytes = cols >> 2;
    let remainder = cols & 0b11;
    let mut coeff_idx = 0usize;
    let mut acc = _mm_setzero_si128();

    for &byte in row.iter().take(full_bytes) {
        acc = _mm_add_epi32(acc, dot_byte_sse(byte, digits.as_ptr().add(coeff_idx)));
        coeff_idx += 4;
    }

    let mut sum = horizontal_sum_i32_128(acc);
    sum = project_row_remainder_scalar(row, digits, full_bytes, coeff_idx, remainder, sum);
    debug_assert_eq!(coeff_idx + remainder, cols);
    sum
}

/// # Safety
///
/// Caller must ensure AVX-512F/DQ/BW are available and the scalar fast-path
/// contract holds.
#[target_feature(enable = "avx512f,avx512dq,avx512bw")]
pub(super) unsafe fn project_row_avx512(row: &[u8], digits: &[i32], cols: usize) -> i32 {
    debug_assert_eq!(digits.len(), cols);

    let full_bytes = cols >> 2;
    let remainder = cols & 0b11;
    let chunk_bytes = full_bytes >> 2;
    let tail_bytes = full_bytes & 0b11;
    let mut coeff_idx = 0usize;
    let mut acc = _mm512_setzero_si512();

    for chunk in 0..chunk_bytes {
        let byte_base = chunk * 4;
        let bytes = [
            row[byte_base],
            row[byte_base + 1],
            row[byte_base + 2],
            row[byte_base + 3],
        ];
        acc = _mm512_add_epi32(
            acc,
            dot_4bytes_avx512(&bytes, digits.as_ptr().add(coeff_idx)),
        );
        coeff_idx += 16;
    }

    let mut sum = _mm512_reduce_add_epi32(acc);
    for &byte in row.iter().skip(chunk_bytes * 4).take(tail_bytes) {
        let partial = dot_byte_sse(byte, digits.as_ptr().add(coeff_idx));
        sum += horizontal_sum_i32_128(partial);
        coeff_idx += 4;
    }

    sum = project_row_remainder_scalar(row, digits, full_bytes, coeff_idx, remainder, sum);
    debug_assert_eq!(coeff_idx + remainder, cols);
    sum
}
