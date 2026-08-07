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

/// Add `p` to negative lanes, mapping `(-p, p)` into `[0, p)`.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn caddp_8x_i32_avx2(a: __m256i, p: __m256i) -> __m256i {
    let negative = _mm256_cmpgt_epi32(_mm256_setzero_si256(), a);
    _mm256_add_epi32(a, _mm256_and_si256(p, negative))
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

/// Vectorized first two inverse DIT stages (`len = 1`, then `len = 2`).
///
/// A 4×4 transpose runs four independent size-4 sub-transforms across SSE
/// lanes, avoiding scalar small-stride butterflies. Requires `D` divisible by
/// 16.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn inverse_dit_head_i32_avx2<const D: usize>(
    a_ptr: *mut i32,
    inv_twiddles: *const i32,
    p: __m128i,
    pinv: __m128i,
) {
    let tw0 = _mm_set1_epi32(*inv_twiddles);
    let tw1 = _mm_set1_epi32(*inv_twiddles.add(1));
    let tw2 = _mm_set1_epi32(*inv_twiddles.add(2));

    let mut base = 0usize;
    while base < D {
        let x0 = _mm_loadu_si128(a_ptr.add(base) as *const __m128i);
        let x1 = _mm_loadu_si128(a_ptr.add(base + 4) as *const __m128i);
        let x2 = _mm_loadu_si128(a_ptr.add(base + 8) as *const __m128i);
        let x3 = _mm_loadu_si128(a_ptr.add(base + 12) as *const __m128i);
        let (r0, r1, r2, r3) = transpose4_epi32(x0, x1, x2, x3);

        let v1 = mont_mul_4x_i32_avx2(r1, tw0, p, pinv);
        let s0 = reduce_range_4x_i32_avx2(_mm_add_epi32(r0, v1), p);
        let d0 = reduce_range_4x_i32_avx2(_mm_sub_epi32(r0, v1), p);
        let v3 = mont_mul_4x_i32_avx2(r3, tw0, p, pinv);
        let s1 = reduce_range_4x_i32_avx2(_mm_add_epi32(r2, v3), p);
        let d1 = reduce_range_4x_i32_avx2(_mm_sub_epi32(r2, v3), p);

        let v2 = mont_mul_4x_i32_avx2(s1, tw1, p, pinv);
        let o0 = reduce_range_4x_i32_avx2(_mm_add_epi32(s0, v2), p);
        let o2 = reduce_range_4x_i32_avx2(_mm_sub_epi32(s0, v2), p);
        let v3 = mont_mul_4x_i32_avx2(d1, tw2, p, pinv);
        let o1 = reduce_range_4x_i32_avx2(_mm_add_epi32(d0, v3), p);
        let o3 = reduce_range_4x_i32_avx2(_mm_sub_epi32(d0, v3), p);

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

/// Reduce sixteen signed sums of up to six `i16` Montgomery products.
///
/// Each input lane is `sum(a_j * b_j)` in a signed `i32`. For `p < 2^14`
/// and at most six products, both the dot product and the Montgomery
/// correction fit in `i32` without saturation.
#[target_feature(enable = "avx512f,avx512bw")]
pub(super) unsafe fn mont_reduce_i16_dot_avx512(
    products: __m512i,
    p: __m512i,
    pinv: __m512i,
) -> __m512i {
    // Signed Montgomery reduction with R = 2^16:
    //   t = low16(products) * pinv mod R
    //   out = (products - t*p) / R.
    // Sign-extending the low half after the i32 multiply gives the same
    // centered i16 `t` used by the scalar and AVX2 implementations.
    let t = _mm512_mullo_epi32(products, pinv);
    let t = _mm512_srai_epi32::<16>(_mm512_slli_epi32::<16>(t));
    _mm512_srai_epi32::<16>(_mm512_sub_epi32(products, _mm512_mullo_epi32(t, p)))
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

#[inline(always)]
unsafe fn forward_dif_butterfly_i16_avx2(
    values: __m256i,
    paired: __m256i,
    twiddles: __m256i,
    p: __m256i,
    pinv: __m256i,
) -> (__m256i, __m256i) {
    let sums = reduce_range_16x_i16_avx2(_mm256_add_epi16(values, paired), p);
    // In an upper lane `paired = u` and `values = v`, so this is `(u - v)w`.
    let differences = mont_mul_16x_i16_avx2(_mm256_sub_epi16(paired, values), twiddles, p, pinv);
    (sums, differences)
}

#[inline(always)]
unsafe fn inverse_dit_butterfly_i16_avx2(
    u: __m256i,
    vw: __m256i,
    p: __m256i,
) -> (__m256i, __m256i) {
    let sums = reduce_range_16x_i16_avx2(_mm256_add_epi16(u, vw), p);
    let differences = reduce_range_16x_i16_avx2(_mm256_sub_epi16(u, vw), p);
    (sums, differences)
}

/// Vectorized final four DIF stages (`len = 8, 4, 2, 1`) for forward i16 NTTs.
///
/// Each YMM register holds one independent size-16 transform. Lane shuffles
/// exchange butterfly halves while masks retain sums in the lower half and
/// Montgomery-scaled differences in the upper half. The final range reduction
/// is folded into the register-resident kernel. Requires `D` divisible by 16.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn forward_dif_tail_i16_avx2<const D: usize>(
    a_ptr: *mut i16,
    fwd_twiddles: *const i16,
    p: __m256i,
    pinv: __m256i,
) {
    let tw8 = _mm256_broadcastsi128_si256(_mm_loadu_si128(fwd_twiddles.add(7) as *const __m128i));
    let tw4_half = _mm_loadl_epi64(fwd_twiddles.add(3) as *const __m128i);
    let tw4 = _mm256_broadcastsi128_si256(_mm_unpacklo_epi64(tw4_half, tw4_half));
    let tw2 = _mm256_set1_epi32(std::ptr::read_unaligned(fwd_twiddles.add(1) as *const i32));
    let tw1 = _mm256_set1_epi16(*fwd_twiddles);

    let mut base = 0usize;
    while base < D {
        let mut values = _mm256_loadu_si256(a_ptr.add(base) as *const __m256i);

        let paired = _mm256_permute2x128_si256::<0x01>(values, values);
        let (sums, differences) = forward_dif_butterfly_i16_avx2(values, paired, tw8, p, pinv);
        values = _mm256_permute2x128_si256::<0x30>(sums, differences);

        let paired = _mm256_permute4x64_epi64::<0xb1>(values);
        let (sums, differences) = forward_dif_butterfly_i16_avx2(values, paired, tw4, p, pinv);
        values = _mm256_blend_epi16::<0xf0>(sums, differences);

        let paired = _mm256_shuffle_epi32::<0xb1>(values);
        let (sums, differences) = forward_dif_butterfly_i16_avx2(values, paired, tw2, p, pinv);
        values = _mm256_blend_epi16::<0xcc>(sums, differences);

        let paired = _mm256_shufflehi_epi16::<0xb1>(_mm256_shufflelo_epi16::<0xb1>(values));
        let (sums, differences) = forward_dif_butterfly_i16_avx2(values, paired, tw1, p, pinv);
        values = _mm256_blend_epi16::<0xaa>(sums, differences);

        _mm256_storeu_si256(
            a_ptr.add(base) as *mut __m256i,
            reduce_range_16x_i16_avx2(values, p),
        );
        base += 16;
    }
}

/// Vectorized first four DIT stages (`len = 1, 2, 4, 8`) for inverse i16 NTTs.
///
/// Requires `D` divisible by 16.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn inverse_dit_head_i16_avx2<const D: usize>(
    a_ptr: *mut i16,
    inv_twiddles: *const i16,
    p: __m256i,
    pinv: __m256i,
) {
    let tw8 = _mm256_broadcastsi128_si256(_mm_loadu_si128(inv_twiddles.add(7) as *const __m128i));
    let tw4_half = _mm_loadl_epi64(inv_twiddles.add(3) as *const __m128i);
    let tw4 = _mm256_broadcastsi128_si256(_mm_unpacklo_epi64(tw4_half, tw4_half));
    let tw2 = _mm256_set1_epi32(std::ptr::read_unaligned(inv_twiddles.add(1) as *const i32));
    let tw1 = _mm256_set1_epi16(*inv_twiddles);

    let mut base = 0usize;
    while base < D {
        let mut values = _mm256_loadu_si256(a_ptr.add(base) as *const __m256i);

        let paired = _mm256_shufflehi_epi16::<0xb1>(_mm256_shufflelo_epi16::<0xb1>(values));
        let products = mont_mul_16x_i16_avx2(paired, tw1, p, pinv);
        let u = _mm256_shufflehi_epi16::<0xa0>(_mm256_shufflelo_epi16::<0xa0>(values));
        let vw = _mm256_shufflehi_epi16::<0xa0>(_mm256_shufflelo_epi16::<0xa0>(products));
        let (sums, differences) = inverse_dit_butterfly_i16_avx2(u, vw, p);
        values = _mm256_blend_epi16::<0xaa>(sums, differences);

        let paired = _mm256_shuffle_epi32::<0xb1>(values);
        let products = mont_mul_16x_i16_avx2(paired, tw2, p, pinv);
        let u = _mm256_shuffle_epi32::<0xa0>(values);
        let vw = _mm256_shuffle_epi32::<0xa0>(products);
        let (sums, differences) = inverse_dit_butterfly_i16_avx2(u, vw, p);
        values = _mm256_blend_epi16::<0xcc>(sums, differences);

        let paired = _mm256_permute4x64_epi64::<0xb1>(values);
        let products = mont_mul_16x_i16_avx2(paired, tw4, p, pinv);
        let u = _mm256_permute4x64_epi64::<0xa0>(values);
        let vw = _mm256_permute4x64_epi64::<0xa0>(products);
        let (sums, differences) = inverse_dit_butterfly_i16_avx2(u, vw, p);
        values = _mm256_blend_epi16::<0xf0>(sums, differences);

        let paired = _mm256_permute2x128_si256::<0x01>(values, values);
        let products = mont_mul_16x_i16_avx2(paired, tw8, p, pinv);
        let u = _mm256_permute2x128_si256::<0x00>(values, values);
        let vw = _mm256_permute2x128_si256::<0x00>(products, products);
        let (sums, differences) = inverse_dit_butterfly_i16_avx2(u, vw, p);
        values = _mm256_permute2x128_si256::<0x30>(sums, differences);

        _mm256_storeu_si256(a_ptr.add(base) as *mut __m256i, values);
        base += 16;
    }
}
