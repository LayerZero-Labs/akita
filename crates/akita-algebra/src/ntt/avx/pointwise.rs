#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::montgomery::{
    mont_mul_16x_i16_avx2, mont_mul_16x_i32_avx512, mont_mul_8x_i32_avx2,
    reduce_range_16x_i16_avx2, reduce_range_16x_i32_avx512, reduce_range_8x_i32_avx2,
};
use crate::ntt::prime::{MontCoeff, NttPrime};

/// AVX2 pointwise multiply-accumulate for one `i32` CRT limb.
///
/// Computes `acc[i] = reduce_range(acc[i] + mont_mul(lhs[i], rhs[i]))`.
///
/// # Safety
///
/// The caller must ensure AVX2 is available. `acc`, `lhs`, and `rhs` must be
/// valid for `d` `i32` elements. `acc` must be writable and must not alias in
/// a way that violates Rust's mutable-reference rules.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn pointwise_mul_acc_i32(
    acc: *mut i32,
    lhs: *const i32,
    rhs: *const i32,
    d: usize,
    p: i32,
    pinv: i32,
) {
    let p_v = _mm256_set1_epi32(p);
    let pinv_v = _mm256_set1_epi32(pinv);
    let mut i = 0;
    while i + 8 <= d {
        // SAFETY: guaranteed by this function's safety contract and loop bound.
        unsafe {
            let a = _mm256_loadu_si256(acc.add(i) as *const __m256i);
            let l = _mm256_loadu_si256(lhs.add(i) as *const __m256i);
            let r = _mm256_loadu_si256(rhs.add(i) as *const __m256i);
            let prod = mont_mul_8x_i32_avx2(l, r, p_v, pinv_v);
            let sum = _mm256_add_epi32(a, prod);
            _mm256_storeu_si256(
                acc.add(i) as *mut __m256i,
                reduce_range_8x_i32_avx2(sum, p_v),
            );
        }
        i += 8;
    }
    if i < d {
        let prime = NttPrime::compute(p);
        while i < d {
            // SAFETY: guaranteed by this function's safety contract and loop bound.
            unsafe {
                let prod = prime.mul(
                    MontCoeff::from_raw(*lhs.add(i)),
                    MontCoeff::from_raw(*rhs.add(i)),
                );
                let sum = MontCoeff::from_raw((*acc.add(i)).wrapping_add(prod.raw()));
                *acc.add(i) = prime.reduce_range(sum).raw();
            }
            i += 1;
        }
    }
}

/// AVX-512 pointwise multiply-accumulate for one `i32` CRT limb.
///
/// Computes `acc[i] = reduce_range(acc[i] + mont_mul(lhs[i], rhs[i]))`.
///
/// # Safety
///
/// The caller must ensure AVX-512F/DQ/BW are available. `acc`, `lhs`, and
/// `rhs` must be valid for `d` `i32` elements. `acc` must be writable and must
/// not alias in a way that violates Rust's mutable-reference rules.
#[target_feature(enable = "avx512f,avx512dq,avx512bw")]
pub(crate) unsafe fn pointwise_mul_acc_i32_avx512(
    acc: *mut i32,
    lhs: *const i32,
    rhs: *const i32,
    d: usize,
    p: i32,
    pinv: i32,
) {
    let p_v = _mm512_set1_epi32(p);
    let pinv_v = _mm512_set1_epi32(pinv);
    let mut i = 0;
    while i + 16 <= d {
        // SAFETY: guaranteed by this function's safety contract and loop bound.
        unsafe {
            let a = _mm512_loadu_si512(acc.add(i) as *const __m512i);
            let l = _mm512_loadu_si512(lhs.add(i) as *const __m512i);
            let r = _mm512_loadu_si512(rhs.add(i) as *const __m512i);
            let prod = mont_mul_16x_i32_avx512(l, r, p_v, pinv_v);
            let sum = _mm512_add_epi32(a, prod);
            _mm512_storeu_si512(
                acc.add(i) as *mut __m512i,
                reduce_range_16x_i32_avx512(sum, p_v),
            );
        }
        i += 16;
    }
    if i < d {
        let prime = NttPrime::compute(p);
        while i < d {
            // SAFETY: guaranteed by this function's safety contract and loop bound.
            unsafe {
                let prod = prime.mul(
                    MontCoeff::from_raw(*lhs.add(i)),
                    MontCoeff::from_raw(*rhs.add(i)),
                );
                let sum = MontCoeff::from_raw((*acc.add(i)).wrapping_add(prod.raw()));
                *acc.add(i) = prime.reduce_range(sum).raw();
            }
            i += 1;
        }
    }
}

/// AVX2 add-and-reduce for one `i32` CRT limb.
///
/// Computes `acc[i] = reduce_range(acc[i] + other[i])`.
///
/// # Safety
///
/// The caller must ensure AVX2 is available. `acc` and `other` must be valid
/// for `d` `i32` elements. `acc` must be writable and must not alias in a way
/// that violates Rust's mutable-reference rules.
#[target_feature(enable = "avx2")]
pub unsafe fn add_reduce_i32(acc: *mut i32, other: *const i32, d: usize, p: i32) {
    let p_v = _mm256_set1_epi32(p);
    let mut i = 0;
    while i + 8 <= d {
        // SAFETY: guaranteed by this function's safety contract and loop bound.
        unsafe {
            let a = _mm256_loadu_si256(acc.add(i) as *const __m256i);
            let b = _mm256_loadu_si256(other.add(i) as *const __m256i);
            _mm256_storeu_si256(
                acc.add(i) as *mut __m256i,
                reduce_range_8x_i32_avx2(_mm256_add_epi32(a, b), p_v),
            );
        }
        i += 8;
    }
    if i < d {
        let prime = NttPrime::compute(p);
        while i < d {
            // SAFETY: guaranteed by this function's safety contract and loop bound.
            unsafe {
                let sum = MontCoeff::from_raw((*acc.add(i)).wrapping_add(*other.add(i)));
                *acc.add(i) = prime.reduce_range(sum).raw();
            }
            i += 1;
        }
    }
}

/// AVX-512 add-and-reduce for one `i32` CRT limb.
///
/// Computes `acc[i] = reduce_range(acc[i] + other[i])`.
///
/// # Safety
///
/// The caller must ensure AVX-512F/DQ/BW are available. `acc` and `other` must
/// be valid for `d` `i32` elements. `acc` must be writable and must not alias in
/// a way that violates Rust's mutable-reference rules.
#[target_feature(enable = "avx512f,avx512dq,avx512bw")]
pub unsafe fn add_reduce_i32_avx512(acc: *mut i32, other: *const i32, d: usize, p: i32) {
    let p_v = _mm512_set1_epi32(p);
    let mut i = 0;
    while i + 16 <= d {
        // SAFETY: guaranteed by this function's safety contract and loop bound.
        unsafe {
            let a = _mm512_loadu_si512(acc.add(i) as *const __m512i);
            let b = _mm512_loadu_si512(other.add(i) as *const __m512i);
            _mm512_storeu_si512(
                acc.add(i) as *mut __m512i,
                reduce_range_16x_i32_avx512(_mm512_add_epi32(a, b), p_v),
            );
        }
        i += 16;
    }
    if i < d {
        let prime = NttPrime::compute(p);
        while i < d {
            // SAFETY: guaranteed by this function's safety contract and loop bound.
            unsafe {
                let sum = MontCoeff::from_raw((*acc.add(i)).wrapping_add(*other.add(i)));
                *acc.add(i) = prime.reduce_range(sum).raw();
            }
            i += 1;
        }
    }
}

/// AVX2 pointwise multiply-accumulate for one `i16` CRT limb.
///
/// Computes `acc[i] = reduce_range(acc[i] + mont_mul(lhs[i], rhs[i]))`.
///
/// # Safety
///
/// The caller must ensure AVX2 is available. `acc`, `lhs`, and `rhs` must be
/// valid for `d` `i16` elements. `acc` must be writable and must not alias in
/// a way that violates Rust's mutable-reference rules.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn pointwise_mul_acc_i16(
    acc: *mut i16,
    lhs: *const i16,
    rhs: *const i16,
    d: usize,
    p: i16,
    pinv: i16,
) {
    let p_v = _mm256_set1_epi16(p);
    let pinv_v = _mm256_set1_epi16(pinv);
    let mut i = 0;
    while i + 16 <= d {
        // SAFETY: guaranteed by this function's safety contract and loop bound.
        unsafe {
            let a = _mm256_loadu_si256(acc.add(i) as *const __m256i);
            let l = _mm256_loadu_si256(lhs.add(i) as *const __m256i);
            let r = _mm256_loadu_si256(rhs.add(i) as *const __m256i);
            let prod = mont_mul_16x_i16_avx2(l, r, p_v, pinv_v);
            let sum = _mm256_add_epi16(a, prod);
            _mm256_storeu_si256(
                acc.add(i) as *mut __m256i,
                reduce_range_16x_i16_avx2(sum, p_v),
            );
        }
        i += 16;
    }
    if i < d {
        let prime = NttPrime::compute(p);
        while i < d {
            // SAFETY: guaranteed by this function's safety contract and loop bound.
            unsafe {
                let prod = prime.mul(
                    MontCoeff::from_raw(*lhs.add(i)),
                    MontCoeff::from_raw(*rhs.add(i)),
                );
                let sum = MontCoeff::from_raw((*acc.add(i)).wrapping_add(prod.raw()));
                *acc.add(i) = prime.reduce_range(sum).raw();
            }
            i += 1;
        }
    }
}

/// AVX2 add-and-reduce for one `i16` CRT limb.
///
/// Computes `acc[i] = reduce_range(acc[i] + other[i])`.
///
/// # Safety
///
/// The caller must ensure AVX2 is available. `acc` and `other` must be valid
/// for `d` `i16` elements. `acc` must be writable and must not alias in a way
/// that violates Rust's mutable-reference rules.
#[target_feature(enable = "avx2")]
pub unsafe fn add_reduce_i16(acc: *mut i16, other: *const i16, d: usize, p: i16) {
    let p_v = _mm256_set1_epi16(p);
    let mut i = 0;
    while i + 16 <= d {
        // SAFETY: guaranteed by this function's safety contract and loop bound.
        unsafe {
            let a = _mm256_loadu_si256(acc.add(i) as *const __m256i);
            let b = _mm256_loadu_si256(other.add(i) as *const __m256i);
            _mm256_storeu_si256(
                acc.add(i) as *mut __m256i,
                reduce_range_16x_i16_avx2(_mm256_add_epi16(a, b), p_v),
            );
        }
        i += 16;
    }
    if i < d {
        let prime = NttPrime::compute(p);
        while i < d {
            // SAFETY: guaranteed by this function's safety contract and loop bound.
            unsafe {
                let sum = MontCoeff::from_raw((*acc.add(i)).wrapping_add(*other.add(i)));
                *acc.add(i) = prime.reduce_range(sum).raw();
            }
            i += 1;
        }
    }
}
