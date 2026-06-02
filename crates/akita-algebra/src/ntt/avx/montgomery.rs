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
    let a_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(a));
    let b_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(b));
    let a_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(a));
    let b_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(b));

    let prod_lo = mont_mul_8x_i16_as_i32_avx2(a_lo, b_lo, p, pinv);
    let prod_hi = mont_mul_8x_i16_as_i32_avx2(a_hi, b_hi, p, pinv);
    let packed = _mm256_packs_epi32(prod_lo, prod_hi);
    _mm256_permute4x64_epi64::<0xd8>(packed)
}

#[target_feature(enable = "avx2")]
pub(super) unsafe fn mont_mul_8x_i16_as_i32_avx2(
    a: __m256i,
    b: __m256i,
    p: __m256i,
    pinv: __m256i,
) -> __m256i {
    let c = _mm256_mullo_epi32(a, b);
    let t_wrapped = _mm256_mullo_epi32(c, pinv);
    let t = _mm256_srai_epi32::<16>(_mm256_slli_epi32::<16>(t_wrapped));
    let tp = _mm256_mullo_epi32(t, p);
    _mm256_srai_epi32::<16>(_mm256_sub_epi32(c, tp))
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
