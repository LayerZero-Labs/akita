#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[target_feature(enable = "avx2")]
pub(super) unsafe fn mont_mul_8x_i32_avx2(
    a: __m256i,
    b: __m256i,
    p: __m256i,
    pinv: __m256i,
) -> __m256i {
    let even_products = _mm256_mul_epi32(a, b);
    let a_odd = _mm256_srli_epi64::<32>(a);
    let b_odd = _mm256_srli_epi64::<32>(b);
    let odd_products = _mm256_mul_epi32(a_odd, b_odd);

    let even = mont_reduce_i32_products_avx2(even_products, p, pinv);
    let odd = mont_reduce_i32_products_avx2(odd_products, p, pinv);
    _mm256_or_si256(even, _mm256_slli_epi64::<32>(odd))
}

#[target_feature(enable = "avx2")]
pub(super) unsafe fn mont_reduce_i32_products_avx2(
    c: __m256i,
    p: __m256i,
    pinv: __m256i,
) -> __m256i {
    let t = _mm256_mullo_epi32(c, pinv);
    let tp = _mm256_mul_epi32(t, p);
    let diff = _mm256_sub_epi64(c, tp);
    // Keep the high 32-bit two's-complement pattern from each 64-bit lane.
    // AVX2 has no arithmetic i64 shift, but the low half after this logical
    // shift is exactly the scalar `(diff >> 32) as i32` bit pattern.
    _mm256_srli_epi64::<32>(diff)
}

#[target_feature(enable = "avx2")]
pub(super) unsafe fn reduce_range_8x_i32_avx2(a: __m256i, p: __m256i) -> __m256i {
    let one = _mm256_set1_epi32(1);
    let p_minus_one = _mm256_sub_epi32(p, one);
    let ge_mask = _mm256_cmpgt_epi32(a, p_minus_one);
    let after_sub = _mm256_sub_epi32(a, _mm256_and_si256(p, ge_mask));

    let zero = _mm256_setzero_si256();
    let lt_mask = _mm256_cmpgt_epi32(zero, after_sub);
    _mm256_add_epi32(after_sub, _mm256_and_si256(p, lt_mask))
}

#[target_feature(enable = "avx2")]
pub(super) unsafe fn mont_mul_4x_i32_avx2(
    a: __m128i,
    b: __m128i,
    p: __m128i,
    pinv: __m128i,
) -> __m128i {
    let even_products = _mm_mul_epi32(a, b);
    let a_odd = _mm_srli_epi64::<32>(a);
    let b_odd = _mm_srli_epi64::<32>(b);
    let odd_products = _mm_mul_epi32(a_odd, b_odd);

    let even = mont_reduce_i32_products_128_avx2(even_products, p, pinv);
    let odd = mont_reduce_i32_products_128_avx2(odd_products, p, pinv);
    _mm_or_si128(even, _mm_slli_epi64::<32>(odd))
}

#[target_feature(enable = "avx2")]
pub(super) unsafe fn mont_reduce_i32_products_128_avx2(
    c: __m128i,
    p: __m128i,
    pinv: __m128i,
) -> __m128i {
    let t = _mm_mullo_epi32(c, pinv);
    let tp = _mm_mul_epi32(t, p);
    let diff = _mm_sub_epi64(c, tp);
    _mm_srli_epi64::<32>(diff)
}

#[target_feature(enable = "avx2")]
pub(super) unsafe fn reduce_range_4x_i32_avx2(a: __m128i, p: __m128i) -> __m128i {
    let one = _mm_set1_epi32(1);
    let p_minus_one = _mm_sub_epi32(p, one);
    let ge_mask = _mm_cmpgt_epi32(a, p_minus_one);
    let after_sub = _mm_sub_epi32(a, _mm_and_si128(p, ge_mask));

    let zero = _mm_setzero_si128();
    let lt_mask = _mm_cmpgt_epi32(zero, after_sub);
    _mm_add_epi32(after_sub, _mm_and_si128(p, lt_mask))
}

/// 4-wide `caddp` for i32: add `p` where negative, mapping `(-p, p)` → `[0, p)`.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn caddp_4x_i32_avx2(a: __m128i, p: __m128i) -> __m128i {
    let zero = _mm_setzero_si128();
    let lt_mask = _mm_cmpgt_epi32(zero, a);
    _mm_add_epi32(a, _mm_and_si128(p, lt_mask))
}

/// Transpose a 4×4 matrix of `i32` held in four `__m128i` row registers.
#[inline(always)]
unsafe fn transpose4_epi32(
    r0: __m128i,
    r1: __m128i,
    r2: __m128i,
    r3: __m128i,
) -> (__m128i, __m128i, __m128i, __m128i) {
    let t0 = _mm_unpacklo_epi32(r0, r1);
    let t1 = _mm_unpacklo_epi32(r2, r3);
    let t2 = _mm_unpackhi_epi32(r0, r1);
    let t3 = _mm_unpackhi_epi32(r2, r3);
    (
        _mm_unpacklo_epi64(t0, t1),
        _mm_unpackhi_epi64(t0, t1),
        _mm_unpacklo_epi64(t2, t3),
        _mm_unpackhi_epi64(t2, t3),
    )
}

/// Vectorized final two DIF stages (`len = 2`, then `len = 1`) for forward i32 NTTs.
///
/// Mirrors the AArch64 `neon::forward_dif_tail_i32` kernel: a 4×4 coefficient
/// transpose via SSE unpacks lands four independent size-4 sub-DFTs across lanes,
/// both remaining stages run 4-wide, and the closing `caddp` folds the transform's
/// final `reduce_range` pass into the last stage outputs.
///
/// Requires `D` divisible by 16.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn forward_dif_tail_i32_avx2<const D: usize>(
    a_ptr: *mut i32,
    fwd_twiddles: *const i32,
    p: __m128i,
    pinv: __m128i,
) {
    let tw0 = _mm_set1_epi32(*fwd_twiddles);
    let tw1 = _mm_set1_epi32(*fwd_twiddles.add(1));
    let tw2 = _mm_set1_epi32(*fwd_twiddles.add(2));

    let mut base = 0usize;
    while base < D {
        let x0 = _mm_loadu_si128(a_ptr.add(base) as *const __m128i);
        let x1 = _mm_loadu_si128(a_ptr.add(base + 4) as *const __m128i);
        let x2 = _mm_loadu_si128(a_ptr.add(base + 8) as *const __m128i);
        let x3 = _mm_loadu_si128(a_ptr.add(base + 12) as *const __m128i);
        let (r0, r1, r2, r3) = transpose4_epi32(x0, x1, x2, x3);

        let s0 = reduce_range_4x_i32_avx2(_mm_add_epi32(r0, r2), p);
        let d0 = mont_mul_4x_i32_avx2(_mm_sub_epi32(r0, r2), tw1, p, pinv);
        let s1 = reduce_range_4x_i32_avx2(_mm_add_epi32(r1, r3), p);
        let d1 = mont_mul_4x_i32_avx2(_mm_sub_epi32(r1, r3), tw2, p, pinv);

        let o0 = caddp_4x_i32_avx2(reduce_range_4x_i32_avx2(_mm_add_epi32(s0, s1), p), p);
        let o1 = caddp_4x_i32_avx2(mont_mul_4x_i32_avx2(_mm_sub_epi32(s0, s1), tw0, p, pinv), p);
        let o2 = caddp_4x_i32_avx2(reduce_range_4x_i32_avx2(_mm_add_epi32(d0, d1), p), p);
        let o3 = caddp_4x_i32_avx2(mont_mul_4x_i32_avx2(_mm_sub_epi32(d0, d1), tw0, p, pinv), p);

        let (y0, y1, y2, y3) = transpose4_epi32(o0, o1, o2, o3);
        _mm_storeu_si128(a_ptr.add(base) as *mut __m128i, y0);
        _mm_storeu_si128(a_ptr.add(base + 4) as *mut __m128i, y1);
        _mm_storeu_si128(a_ptr.add(base + 8) as *mut __m128i, y2);
        _mm_storeu_si128(a_ptr.add(base + 12) as *mut __m128i, y3);
        base += 16;
    }
}

#[target_feature(enable = "avx512f,avx512dq,avx512bw")]
pub(super) unsafe fn mont_mul_16x_i32_avx512(
    a: __m512i,
    b: __m512i,
    p: __m512i,
    pinv: __m512i,
) -> __m512i {
    let even_products = _mm512_mul_epi32(a, b);
    let a_odd = _mm512_srli_epi64::<32>(a);
    let b_odd = _mm512_srli_epi64::<32>(b);
    let odd_products = _mm512_mul_epi32(a_odd, b_odd);

    let even = mont_reduce_i32_products_avx512(even_products, p, pinv);
    let odd = mont_reduce_i32_products_avx512(odd_products, p, pinv);
    _mm512_or_si512(even, _mm512_slli_epi64::<32>(odd))
}

#[target_feature(enable = "avx512f,avx512dq,avx512bw")]
pub(super) unsafe fn mont_reduce_i32_products_avx512(
    c: __m512i,
    p: __m512i,
    pinv: __m512i,
) -> __m512i {
    let t = _mm512_mullo_epi32(c, pinv);
    let tp = _mm512_mul_epi32(t, p);
    let diff = _mm512_sub_epi64(c, tp);
    _mm512_srli_epi64::<32>(diff)
}

#[target_feature(enable = "avx512f,avx512dq,avx512bw")]
pub(super) unsafe fn reduce_range_16x_i32_avx512(a: __m512i, p: __m512i) -> __m512i {
    let one = _mm512_set1_epi32(1);
    let p_minus_one = _mm512_sub_epi32(p, one);
    let ge_mask = _mm512_cmpgt_epi32_mask(a, p_minus_one);
    let after_sub = _mm512_mask_sub_epi32(a, ge_mask, a, p);

    let zero = _mm512_setzero_si512();
    let lt_mask = _mm512_cmplt_epi32_mask(after_sub, zero);
    _mm512_mask_add_epi32(after_sub, lt_mask, after_sub, p)
}

#[target_feature(enable = "avx2")]
pub(super) unsafe fn mont_mul_16x_i16_avx2(
    a: __m256i,
    b: __m256i,
    p: __m256i,
    pinv: __m256i,
) -> __m256i {
    // Signed Montgomery reduction with R = 2^16:
    //   c = a*b, t = low(c)*pinv mod R, out = (c - t*p)/R.
    // `mulhi` exposes the signed high half directly, while `mullo` computes
    // the two low halves modulo R. Hence high(c) - high(t*p) is exactly the
    // desired quotient, with no lane widening or packing.
    let c_lo = _mm256_mullo_epi16(a, b);
    let c_hi = _mm256_mulhi_epi16(a, b);
    let t = _mm256_mullo_epi16(c_lo, pinv);
    let tp_hi = _mm256_mulhi_epi16(t, p);
    _mm256_sub_epi16(c_hi, tp_hi)
}

#[target_feature(enable = "avx2")]
pub(super) unsafe fn reduce_range_16x_i16_avx2(a: __m256i, p: __m256i) -> __m256i {
    let one = _mm256_set1_epi16(1);
    let p_minus_one = _mm256_sub_epi16(p, one);
    let ge_mask = _mm256_cmpgt_epi16(a, p_minus_one);
    let after_sub = _mm256_sub_epi16(a, _mm256_and_si256(p, ge_mask));

    let zero = _mm256_setzero_si256();
    let lt_mask = _mm256_cmpgt_epi16(zero, after_sub);
    _mm256_add_epi16(after_sub, _mm256_and_si256(p, lt_mask))
}
