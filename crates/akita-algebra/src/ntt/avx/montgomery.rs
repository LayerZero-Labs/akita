#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Montgomery-reduce the signed 64-bit lanes of `c` whose low halves are
/// arbitrary, leaving the reduced `i32` in the low half of each lane. The
/// caller keeps `|c| < 2^31 p` so the result fits.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn mont_reduce_i32_products_avx2(
    c: __m256i,
    p: __m256i,
    pinv: __m256i,
) -> __m256i {
    // Only the low 32 bits of `t` matter, so the unsigned even-lane product
    // replaces a `mullo`.
    let t = _mm256_mul_epu32(c, pinv);
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
