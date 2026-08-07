#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::montgomery::{
    caddp_8x_i32_avx2, mont_mul_16x_i16_avx2, mont_mul_16x_i32_avx512, mont_mul_8x_i32_avx2,
    mont_reduce_i16_dot_avx512, reduce_range_16x_i16_avx2, reduce_range_16x_i32_avx512,
    reduce_range_8x_i32_avx2,
};
use crate::ntt::prime::{MontCoeff, NttPrime, I16_VNNI_DOT_BATCH, I32_LAZY_DOT_BATCH};

/// AVX2 pointwise dot-product accumulation for up to six i32 CRT entries.
///
/// Raw signed products are accumulated in i64 lanes and Montgomery-reduced
/// once per batch. For `B <= 6` and `p < 2^30`, the reduction numerator is
/// bounded by `B*2^60 + 2^61 < 2^63`; the reduced result therefore fits i32.
///
/// # Safety
///
/// The caller must ensure AVX2 is available. `acc` must be valid for `d`
/// writable i32 elements. Each of the first `count` pointers in `lhs` and
/// `rhs` must be valid for `d` readable i32 elements. The pointed-to ranges
/// must obey Rust's aliasing rules with `acc`.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn pointwise_dot_acc_i32(
    acc: *mut i32,
    lhs: *const *const i32,
    rhs: *const *const i32,
    count: usize,
    d: usize,
    p: i32,
    pinv: i32,
) {
    debug_assert!(count <= I32_LAZY_DOT_BATCH);
    if count == 0 {
        return;
    }
    let p_v = _mm256_set1_epi32(p);
    let pinv_v = _mm256_set1_epi32(pinv);
    let mut i = 0;
    while i + 8 <= d {
        let mut even_sum = _mm256_setzero_si256();
        let mut odd_sum = _mm256_setzero_si256();
        for product in 0..count {
            // SAFETY: guaranteed by this function's pointer contract and loop bounds.
            unsafe {
                let l = _mm256_loadu_si256((*lhs.add(product)).add(i).cast());
                let r = _mm256_loadu_si256((*rhs.add(product)).add(i).cast());
                even_sum = _mm256_add_epi64(even_sum, _mm256_mul_epi32(l, r));
                odd_sum = _mm256_add_epi64(
                    odd_sum,
                    _mm256_mul_epi32(_mm256_srli_epi64::<32>(l), _mm256_srli_epi64::<32>(r)),
                );
            }
        }
        // SAFETY: the six-product bound above keeps both signed reductions exact.
        unsafe {
            let even = super::montgomery::mont_reduce_i32_products_avx2(even_sum, p_v, pinv_v);
            let odd = super::montgomery::mont_reduce_i32_products_avx2(odd_sum, p_v, pinv_v);
            let batch = _mm256_or_si256(even, _mm256_slli_epi64::<32>(odd));
            let batch = reduce_range_8x_i32_avx2(batch, p_v);
            let accumulator_pointer = acc.add(i);
            let accumulator = _mm256_loadu_si256(accumulator_pointer.cast());
            _mm256_storeu_si256(
                accumulator_pointer.cast(),
                reduce_range_8x_i32_avx2(_mm256_add_epi32(accumulator, batch), p_v),
            );
        }
        i += 8;
    }
    if i < d {
        let prime = NttPrime::compute(p);
        while i < d {
            let mut raw_sum = 0_i64;
            for product in 0..count {
                // SAFETY: guaranteed by this function's pointer contract and tail bound.
                unsafe {
                    raw_sum += i64::from(*(*lhs.add(product)).add(i))
                        * i64::from(*(*rhs.add(product)).add(i));
                }
            }
            let correction = (raw_sum as i32).wrapping_mul(pinv);
            let reduced = ((raw_sum - i64::from(correction) * i64::from(p)) >> 32) as i32;
            let reduced = prime.reduce_range(MontCoeff::from_raw(reduced));
            // SAFETY: guaranteed by this function's pointer contract.
            unsafe {
                let sum = MontCoeff::from_raw((*acc.add(i)).wrapping_add(reduced.raw()));
                *acc.add(i) = prime.reduce_range(sum).raw();
            }
            i += 1;
        }
    }
}

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
/// Computes `out[i] = reduce_range(lhs[i] + rhs[i])`.
///
/// # Safety
///
/// The caller must ensure AVX2 is available. All pointers must be valid for
/// `d` `i32` elements and `out` must be writable. `out` may equal `lhs`.
#[target_feature(enable = "avx2")]
pub unsafe fn add_reduce_i32(out: *mut i32, lhs: *const i32, rhs: *const i32, d: usize, p: i32) {
    let p_v = _mm256_set1_epi32(p);
    let mut i = 0;
    while i + 8 <= d {
        // SAFETY: guaranteed by this function's safety contract and loop bound.
        unsafe {
            let a = _mm256_loadu_si256(lhs.add(i) as *const __m256i);
            let b = _mm256_loadu_si256(rhs.add(i) as *const __m256i);
            _mm256_storeu_si256(
                out.add(i) as *mut __m256i,
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
                let sum = MontCoeff::from_raw((*lhs.add(i)).wrapping_add(*rhs.add(i)));
                *out.add(i) = prime.reduce_range(sum).raw();
            }
            i += 1;
        }
    }
}

/// AVX-512 add-and-reduce for one `i32` CRT limb.
///
/// Computes `out[i] = reduce_range(lhs[i] + rhs[i])`.
///
/// # Safety
///
/// The caller must ensure AVX-512F/DQ/BW are available. All pointers must be
/// valid for `d` `i32` elements and `out` must be writable. `out` may equal
/// `lhs`.
#[target_feature(enable = "avx512f,avx512dq,avx512bw")]
pub unsafe fn add_reduce_i32_avx512(
    out: *mut i32,
    lhs: *const i32,
    rhs: *const i32,
    d: usize,
    p: i32,
) {
    let p_v = _mm512_set1_epi32(p);
    let mut i = 0;
    while i + 16 <= d {
        // SAFETY: guaranteed by this function's safety contract and loop bound.
        unsafe {
            let a = _mm512_loadu_si512(lhs.add(i) as *const __m512i);
            let b = _mm512_loadu_si512(rhs.add(i) as *const __m512i);
            _mm512_storeu_si512(
                out.add(i) as *mut __m512i,
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
                let sum = MontCoeff::from_raw((*lhs.add(i)).wrapping_add(*rhs.add(i)));
                *out.add(i) = prime.reduce_range(sum).raw();
            }
            i += 1;
        }
    }
}

/// AVX2 subtract-and-reduce for one `i32` CRT limb.
///
/// # Safety
///
/// The caller must ensure AVX2 is available and all pointers are valid for
/// `d` elements. `out` must be writable and may equal `lhs`.
#[target_feature(enable = "avx2")]
pub unsafe fn sub_reduce_i32(out: *mut i32, lhs: *const i32, rhs: *const i32, d: usize, p: i32) {
    let p_v = _mm256_set1_epi32(p);
    let mut i = 0;
    while i + 8 <= d {
        unsafe {
            let a = _mm256_loadu_si256(lhs.add(i) as *const __m256i);
            let b = _mm256_loadu_si256(rhs.add(i) as *const __m256i);
            _mm256_storeu_si256(
                out.add(i) as *mut __m256i,
                reduce_range_8x_i32_avx2(_mm256_sub_epi32(a, b), p_v),
            );
        }
        i += 8;
    }
    if i < d {
        let prime = NttPrime::compute(p);
        while i < d {
            unsafe {
                let diff = MontCoeff::from_raw((*lhs.add(i)).wrapping_sub(*rhs.add(i)));
                *out.add(i) = prime.reduce_range(diff).raw();
            }
            i += 1;
        }
    }
}

/// AVX-512 subtract-and-reduce for one `i32` CRT limb.
///
/// # Safety
///
/// The caller must ensure AVX-512F/DQ/BW are available and all pointers are
/// valid for `d` elements. `out` must be writable and may equal `lhs`.
#[target_feature(enable = "avx512f,avx512dq,avx512bw")]
pub unsafe fn sub_reduce_i32_avx512(
    out: *mut i32,
    lhs: *const i32,
    rhs: *const i32,
    d: usize,
    p: i32,
) {
    let p_v = _mm512_set1_epi32(p);
    let mut i = 0;
    while i + 16 <= d {
        unsafe {
            let a = _mm512_loadu_si512(lhs.add(i) as *const __m512i);
            let b = _mm512_loadu_si512(rhs.add(i) as *const __m512i);
            _mm512_storeu_si512(
                out.add(i) as *mut __m512i,
                reduce_range_16x_i32_avx512(_mm512_sub_epi32(a, b), p_v),
            );
        }
        i += 16;
    }
    if i < d {
        let prime = NttPrime::compute(p);
        while i < d {
            unsafe {
                let diff = MontCoeff::from_raw((*lhs.add(i)).wrapping_sub(*rhs.add(i)));
                *out.add(i) = prime.reduce_range(diff).raw();
            }
            i += 1;
        }
    }
}

/// AVX2 negate-and-reduce for one `i32` CRT limb.
///
/// # Safety
///
/// The caller must ensure AVX2 is available. `out` must be writable and `input`
/// must be readable for `d` elements. The pointers may be equal.
#[target_feature(enable = "avx2")]
pub unsafe fn neg_reduce_i32(out: *mut i32, input: *const i32, d: usize, p: i32) {
    let p_v = _mm256_set1_epi32(p);
    let zero = _mm256_setzero_si256();
    let mut i = 0;
    while i + 8 <= d {
        unsafe {
            let a = _mm256_loadu_si256(input.add(i) as *const __m256i);
            _mm256_storeu_si256(
                out.add(i) as *mut __m256i,
                reduce_range_8x_i32_avx2(_mm256_sub_epi32(zero, a), p_v),
            );
        }
        i += 8;
    }
    if i < d {
        let prime = NttPrime::compute(p);
        while i < d {
            unsafe {
                let neg = MontCoeff::from_raw((*input.add(i)).wrapping_neg());
                *out.add(i) = prime.reduce_range(neg).raw();
            }
            i += 1;
        }
    }
}

/// AVX-512 negate-and-reduce for one `i32` CRT limb.
///
/// # Safety
///
/// The caller must ensure AVX-512F/DQ/BW are available. `out` must be writable
/// and `input` readable for `d` elements. The pointers may be equal.
#[target_feature(enable = "avx512f,avx512dq,avx512bw")]
pub unsafe fn neg_reduce_i32_avx512(out: *mut i32, input: *const i32, d: usize, p: i32) {
    let p_v = _mm512_set1_epi32(p);
    let zero = _mm512_setzero_si512();
    let mut i = 0;
    while i + 16 <= d {
        unsafe {
            let a = _mm512_loadu_si512(input.add(i) as *const __m512i);
            _mm512_storeu_si512(
                out.add(i) as *mut __m512i,
                reduce_range_16x_i32_avx512(_mm512_sub_epi32(zero, a), p_v),
            );
        }
        i += 16;
    }
    if i < d {
        let prime = NttPrime::compute(p);
        while i < d {
            unsafe {
                let neg = MontCoeff::from_raw((*input.add(i)).wrapping_neg());
                *out.add(i) = prime.reduce_range(neg).raw();
            }
            i += 1;
        }
    }
}

/// AVX2 pointwise Montgomery multiplication for one `i32` CRT limb.
///
/// # Safety
///
/// The caller must ensure AVX2 is available. All pointers must be valid for
/// `d` elements, and `out` must be writable.
#[target_feature(enable = "avx2")]
pub unsafe fn pointwise_mul_i32(
    out: *mut i32,
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
        unsafe {
            let l = _mm256_loadu_si256(lhs.add(i) as *const __m256i);
            let r = _mm256_loadu_si256(rhs.add(i) as *const __m256i);
            let prod = mont_mul_8x_i32_avx2(l, r, p_v, pinv_v);
            _mm256_storeu_si256(out.add(i) as *mut __m256i, caddp_8x_i32_avx2(prod, p_v));
        }
        i += 8;
    }
    if i < d {
        let prime = NttPrime::compute(p);
        while i < d {
            unsafe {
                *out.add(i) = prime
                    .caddp(prime.mul(
                        MontCoeff::from_raw(*lhs.add(i)),
                        MontCoeff::from_raw(*rhs.add(i)),
                    ))
                    .raw();
            }
            i += 1;
        }
    }
}

/// AVX-512 pointwise Montgomery multiplication for one `i32` CRT limb.
///
/// # Safety
///
/// The caller must ensure AVX-512F/DQ/BW are available. All pointers must be
/// valid for `d` elements, and `out` must be writable.
#[target_feature(enable = "avx512f,avx512dq,avx512bw")]
pub unsafe fn pointwise_mul_i32_avx512(
    out: *mut i32,
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
        unsafe {
            let l = _mm512_loadu_si512(lhs.add(i) as *const __m512i);
            let r = _mm512_loadu_si512(rhs.add(i) as *const __m512i);
            let prod = mont_mul_16x_i32_avx512(l, r, p_v, pinv_v);
            _mm512_storeu_si512(
                out.add(i) as *mut __m512i,
                reduce_range_16x_i32_avx512(prod, p_v),
            );
        }
        i += 16;
    }
    if i < d {
        let prime = NttPrime::compute(p);
        while i < d {
            unsafe {
                *out.add(i) = prime
                    .reduce_range(prime.mul(
                        MontCoeff::from_raw(*lhs.add(i)),
                        MontCoeff::from_raw(*rhs.add(i)),
                    ))
                    .raw();
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

/// Pair-pack exactly six `i16` CRT entries in place for AVX-512VNNI dots.
///
/// Each adjacent source pair becomes an even-coefficient pair stream followed
/// by an odd-coefficient pair stream in the same two backing arrays. Packing
/// once lets every matrix row reuse the transformed rhs without repeating the
/// interleave instructions in the pointwise kernel.
///
/// # Safety
///
/// The caller must ensure AVX-512F/BW are available. `rhs` must point to six
/// distinct writable arrays of `d` `i16` elements. `d` must be even.
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn pack_i16_dot_rhs_6_avx512(rhs: *const *mut i16, d: usize) {
    const EVEN_WORDS: __mmask32 = 0x5555_5555;

    let mut i = 0usize;
    while i + 32 <= d {
        for pair in 0..(I16_VNNI_DOT_BATCH / 2) {
            // SAFETY: guaranteed by the pointer contract and loop bound.
            unsafe {
                let even_ptr = *rhs.add(pair * 2);
                let odd_ptr = *rhs.add(pair * 2 + 1);
                let rhs0 = _mm512_loadu_si512(even_ptr.add(i).cast());
                let rhs1 = _mm512_loadu_si512(odd_ptr.add(i).cast());
                let even = _mm512_mask_mov_epi16(_mm512_slli_epi32::<16>(rhs1), EVEN_WORDS, rhs0);
                let odd = _mm512_mask_mov_epi16(rhs1, EVEN_WORDS, _mm512_srli_epi32::<16>(rhs0));
                _mm512_storeu_si512(even_ptr.add(i).cast(), even);
                _mm512_storeu_si512(odd_ptr.add(i).cast(), odd);
            }
        }
        i += 32;
    }

    if i < d {
        debug_assert_eq!((d - i) % 2, 0);
        for pair in 0..(I16_VNNI_DOT_BATCH / 2) {
            let mut rhs0 = [0_i16; 32];
            let mut rhs1 = [0_i16; 32];
            let remaining = d - i;
            // SAFETY: guaranteed by the pointer contract and tail bound.
            unsafe {
                let even_ptr = *rhs.add(pair * 2);
                let odd_ptr = *rhs.add(pair * 2 + 1);
                for offset in 0..remaining {
                    rhs0[offset] = *even_ptr.add(i + offset);
                    rhs1[offset] = *odd_ptr.add(i + offset);
                }
                for offset in 0..(remaining / 2) {
                    let dst = i + offset * 2;
                    *even_ptr.add(dst) = rhs0[offset * 2];
                    *even_ptr.add(dst + 1) = rhs1[offset * 2];
                    *odd_ptr.add(dst) = rhs0[offset * 2 + 1];
                    *odd_ptr.add(dst + 1) = rhs1[offset * 2 + 1];
                }
            }
        }
    }
}

/// AVX-512VNNI pointwise dot accumulation for exactly six `i16` CRT entries.
///
/// Adjacent lhs columns are interleaved in registers. The corresponding rhs
/// arrays must already have been pair-packed by [`pack_i16_dot_rhs_6_avx512`].
/// Three `vpdpwssd` instructions cover the six products for each parity, then
/// one Montgomery reduction returns each frequency to the canonical range.
///
/// # Safety
///
/// The caller must ensure AVX2 and AVX-512F/DQ/BW/VNNI are available. `acc` must be valid
/// for `d` writable `i16` elements. Every pointer in the six-entry `lhs` and
/// `rhs` arrays must be valid for `d` readable `i16` elements and obey Rust's
/// aliasing rules with `acc`. The rhs arrays must have the packed layout above.
/// All values must lie in `(-p, p)` for `p < 2^14`.
#[target_feature(enable = "avx2,avx512f,avx512dq,avx512bw,avx512vnni")]
pub(crate) unsafe fn pointwise_dot_acc_6_i16_avx512vnni(
    acc: *mut i16,
    lhs: *const *const i16,
    rhs: *const *const i16,
    d: usize,
    p: i16,
    pinv: i16,
) {
    const EVEN_WORDS: __mmask32 = 0x5555_5555;

    let p_i32 = _mm512_set1_epi32(i32::from(p));
    let pinv_i32 = _mm512_set1_epi32(i32::from(pinv));
    let p_i16 = _mm256_set1_epi16(p);
    let mut i = 0usize;
    while i + 32 <= d {
        let mut even = _mm512_setzero_si512();
        let mut odd = _mm512_setzero_si512();
        for pair in 0..(I16_VNNI_DOT_BATCH / 2) {
            let column = pair * 2;
            // SAFETY: guaranteed by the pointer contract and loop bound.
            unsafe {
                let lhs0 = _mm512_loadu_si512((*lhs.add(column)).add(i).cast());
                let lhs1 = _mm512_loadu_si512((*lhs.add(column + 1)).add(i).cast());
                let rhs_even = _mm512_loadu_si512((*rhs.add(column)).add(i).cast());
                let rhs_odd = _mm512_loadu_si512((*rhs.add(column + 1)).add(i).cast());

                let lhs_even =
                    _mm512_mask_mov_epi16(_mm512_slli_epi32::<16>(lhs1), EVEN_WORDS, lhs0);
                even = _mm512_dpwssd_epi32(even, lhs_even, rhs_even);

                let lhs_odd =
                    _mm512_mask_mov_epi16(lhs1, EVEN_WORDS, _mm512_srli_epi32::<16>(lhs0));
                odd = _mm512_dpwssd_epi32(odd, lhs_odd, rhs_odd);
            }
        }

        let even =
            reduce_range_16x_i32_avx512(mont_reduce_i16_dot_avx512(even, p_i32, pinv_i32), p_i32);
        let odd =
            reduce_range_16x_i32_avx512(mont_reduce_i16_dot_avx512(odd, p_i32, pinv_i32), p_i32);
        let even = _mm512_cvtepi32_epi16(even);
        let odd = _mm512_cvtepi32_epi16(odd);
        let split_lo = _mm256_unpacklo_epi16(even, odd);
        let split_hi = _mm256_unpackhi_epi16(even, odd);
        let batch_lo = _mm256_permute2x128_si256::<0x20>(split_lo, split_hi);
        let batch_hi = _mm256_permute2x128_si256::<0x31>(split_lo, split_hi);

        // SAFETY: guaranteed by the pointer contract and loop bound.
        unsafe {
            let acc_lo_ptr = acc.add(i);
            let acc_hi_ptr = acc.add(i + 16);
            let acc_lo = _mm256_loadu_si256(acc_lo_ptr.cast());
            let acc_hi = _mm256_loadu_si256(acc_hi_ptr.cast());
            _mm256_storeu_si256(
                acc_lo_ptr.cast(),
                reduce_range_16x_i16_avx2(_mm256_add_epi16(acc_lo, batch_lo), p_i16),
            );
            _mm256_storeu_si256(
                acc_hi_ptr.cast(),
                reduce_range_16x_i16_avx2(_mm256_add_epi16(acc_hi, batch_hi), p_i16),
            );
        }
        i += 32;
    }

    if i < d {
        let prime = NttPrime::compute(p);
        while i < d {
            let mut products = 0_i32;
            for pair in 0..(I16_VNNI_DOT_BATCH / 2) {
                let packed = if i.is_multiple_of(2) {
                    // Even coefficients occupy the first array in each pair.
                    unsafe { (*rhs.add(pair * 2)).add(i) }
                } else {
                    // Odd coefficients occupy the second array in each pair.
                    unsafe { (*rhs.add(pair * 2 + 1)).add(i - 1) }
                };
                // SAFETY: guaranteed by the pointer contract and tail bound.
                unsafe {
                    products += i32::from(*(*lhs.add(pair * 2)).add(i)) * i32::from(*packed);
                    products +=
                        i32::from(*(*lhs.add(pair * 2 + 1)).add(i)) * i32::from(*packed.add(1));
                }
            }
            let correction = (products as i16).wrapping_mul(pinv);
            let reduced = ((products - i32::from(correction) * i32::from(p)) >> 16) as i16;
            let reduced = prime.reduce_range(MontCoeff::from_raw(reduced));
            // SAFETY: guaranteed by the pointer contract.
            unsafe {
                let sum = MontCoeff::from_raw((*acc.add(i)).wrapping_add(reduced.raw()));
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
