//! x86-64 AVX2 dense projection kernels.

use std::arch::x86_64::*;

pub(super) fn dot_i8_avx2_dispatch(weights: &[i8], input: &[i8]) -> i64 {
    // SAFETY: the caller selected this function only after detecting AVX2.
    unsafe { dot_i8_avx2(weights, input) }
}

pub(super) fn dot_i16_avx2_dispatch(weights: &[i8], input: &[i16]) -> i64 {
    // SAFETY: the caller selected this function only after detecting AVX2.
    unsafe { dot_i16_avx2(weights, input) }
}

pub(super) fn dot_i32_avx2_dispatch(weights: &[i8], input: &[i32]) -> i64 {
    // SAFETY: the caller selected this function only after detecting AVX2.
    unsafe { dot_i32_avx2(weights, input) }
}

pub(super) fn dot_i64_avx2_dispatch(weights: &[i8], input: &[i64]) -> i64 {
    // SAFETY: the caller selected this function only after detecting AVX2.
    unsafe { dot_i64_avx2(weights, input) }
}

#[target_feature(enable = "avx2")]
unsafe fn dot_i8_avx2(weights: &[i8], input: &[i8]) -> i64 {
    let len = weights.len().min(input.len());
    let mut low_sum = _mm256_setzero_si256();
    let mut high_sum = _mm256_setzero_si256();
    let mut index = 0;
    while index + 16 <= len {
        let w8 = _mm_loadu_si128(weights.as_ptr().add(index).cast());
        let x8 = _mm_loadu_si128(input.as_ptr().add(index).cast());
        let pairs = _mm256_madd_epi16(_mm256_cvtepi8_epi16(w8), _mm256_cvtepi8_epi16(x8));
        low_sum = _mm256_add_epi64(
            low_sum,
            _mm256_cvtepi32_epi64(_mm256_castsi256_si128(pairs)),
        );
        high_sum = _mm256_add_epi64(
            high_sum,
            _mm256_cvtepi32_epi64(_mm256_extracti128_si256::<1>(pairs)),
        );
        index += 16;
    }
    let mut result = horizontal_sum(_mm256_add_epi64(low_sum, high_sum));
    while index < len {
        result += i64::from(*weights.get_unchecked(index)) * i64::from(*input.get_unchecked(index));
        index += 1;
    }
    result
}

#[target_feature(enable = "avx2")]
unsafe fn dot_i16_avx2(weights: &[i8], input: &[i16]) -> i64 {
    let len = weights.len().min(input.len());
    let mut low_sum = _mm256_setzero_si256();
    let mut high_sum = _mm256_setzero_si256();
    let mut index = 0;
    while index + 16 <= len {
        let w8 = _mm_loadu_si128(weights.as_ptr().add(index).cast());
        let x = _mm256_loadu_si256(input.as_ptr().add(index).cast());
        let pairs = _mm256_madd_epi16(_mm256_cvtepi8_epi16(w8), x);
        low_sum = _mm256_add_epi64(
            low_sum,
            _mm256_cvtepi32_epi64(_mm256_castsi256_si128(pairs)),
        );
        high_sum = _mm256_add_epi64(
            high_sum,
            _mm256_cvtepi32_epi64(_mm256_extracti128_si256::<1>(pairs)),
        );
        index += 16;
    }
    let mut result = horizontal_sum(_mm256_add_epi64(low_sum, high_sum));
    while index < len {
        result += i64::from(*weights.get_unchecked(index)) * i64::from(*input.get_unchecked(index));
        index += 1;
    }
    result
}

#[target_feature(enable = "avx2")]
unsafe fn dot_i32_avx2(weights: &[i8], input: &[i32]) -> i64 {
    let len = weights.len().min(input.len());
    let mut even_sum = _mm256_setzero_si256();
    let mut odd_sum = _mm256_setzero_si256();
    let mut index = 0;
    while index + 8 <= len {
        let w8 = _mm_loadl_epi64(weights.as_ptr().add(index).cast());
        let w = _mm256_cvtepi8_epi32(w8);
        let x = _mm256_loadu_si256(input.as_ptr().add(index).cast());
        even_sum = _mm256_add_epi64(even_sum, _mm256_mul_epi32(w, x));
        odd_sum = _mm256_add_epi64(
            odd_sum,
            _mm256_mul_epi32(_mm256_srli_epi64::<32>(w), _mm256_srli_epi64::<32>(x)),
        );
        index += 8;
    }
    let mut result = horizontal_sum(_mm256_add_epi64(even_sum, odd_sum));
    while index < len {
        result += i64::from(*weights.get_unchecked(index)) * i64::from(*input.get_unchecked(index));
        index += 1;
    }
    result
}

#[target_feature(enable = "avx2")]
unsafe fn dot_i64_avx2(weights: &[i8], input: &[i64]) -> i64 {
    let len = weights.len().min(input.len());
    let zero = _mm256_setzero_si256();
    let mut sum = zero;
    let mut index = 0;
    while index + 4 <= len {
        let packed = std::ptr::read_unaligned(weights.as_ptr().add(index).cast::<i32>());
        let w = _mm256_cvtepi8_epi64(_mm_cvtsi32_si128(packed));
        let x = _mm256_loadu_si256(input.as_ptr().add(index).cast());
        let positive = _mm256_cmpgt_epi64(w, zero);
        let negative = _mm256_cmpgt_epi64(zero, w);
        sum = _mm256_add_epi64(sum, _mm256_and_si256(positive, x));
        sum = _mm256_sub_epi64(sum, _mm256_and_si256(negative, x));
        index += 4;
    }
    let mut result = horizontal_sum(sum);
    while index < len {
        result = match *weights.get_unchecked(index) {
            -1 => result.wrapping_sub(*input.get_unchecked(index)),
            1 => result.wrapping_add(*input.get_unchecked(index)),
            _ => result,
        };
        index += 1;
    }
    result
}

#[target_feature(enable = "avx2")]
unsafe fn horizontal_sum(value: __m256i) -> i64 {
    let mut lanes = [0i64; 4];
    _mm256_storeu_si256(lanes.as_mut_ptr().cast(), value);
    lanes.into_iter().fold(0i64, i64::wrapping_add)
}
