#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::BalancedDecomposePow2Params;

#[inline(always)]
unsafe fn cmpgt_epu32_avx2(lhs: __m256i, rhs: __m256i) -> __m256i {
    let sign = _mm256_set1_epi32(i32::MIN);
    _mm256_cmpgt_epi32(_mm256_xor_si256(lhs, sign), _mm256_xor_si256(rhs, sign))
}

#[inline(always)]
unsafe fn store_i8_digits_avx2(digits: __m256i, dst: *mut i8) {
    let low = _mm256_castsi256_si128(digits);
    let high = _mm256_extracti128_si256::<1>(digits);
    let words = _mm_packs_epi32(low, high);
    let bytes = _mm_packs_epi16(words, _mm_setzero_si128());
    std::ptr::write_unaligned(dst.cast::<u64>(), _mm_cvtsi128_si64(bytes) as u64);
}

/// AVX2 balanced decomposition of canonical fp32 representatives.
///
/// The first quotient handles negative centered representatives through their
/// unsigned magnitude. This is required by full-width asymmetric centering,
/// whose smallest representative can be slightly below `i32::MIN`. After one
/// digit, every quotient fits an i32 lane and uses signed arithmetic shifts.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn balanced_decompose_canonical_u32_pow2_i8_avx2(
    canonical: &[u32],
    out: &mut [i8],
    params: &BalancedDecomposePow2Params,
) {
    debug_assert!(canonical.len().is_multiple_of(8));
    debug_assert_eq!(out.len(), canonical.len() * params.levels);
    debug_assert!(params.levels > 0);
    debug_assert!(params.log_basis <= 8);
    debug_assert!(params.q <= u32::MAX.into());

    let width = canonical.len();
    let q = params.q as u32;
    let threshold = params.threshold as u32;
    let b = 1u32 << params.log_basis;
    let half_b = b >> 1;
    let mask = b - 1;

    let q_v = _mm256_set1_epi32(q as i32);
    let threshold_v = _mm256_set1_epi32(threshold as i32);
    let b_v = _mm256_set1_epi32(b as i32);
    let half_b_minus_one_v = _mm256_set1_epi32((half_b - 1) as i32);
    let mask_v = _mm256_set1_epi32(mask as i32);
    let shift_v = _mm256_set1_epi32(params.log_basis as i32);
    let zero = _mm256_setzero_si256();

    let mut base = 0usize;
    while base < width {
        let values = _mm256_loadu_si256(canonical.as_ptr().add(base).cast());
        let negative = cmpgt_epu32_avx2(values, threshold_v);
        let centered_low = _mm256_sub_epi32(values, _mm256_and_si256(q_v, negative));
        let raw_digit = _mm256_and_si256(centered_low, mask_v);
        let high_digit = _mm256_cmpgt_epi32(raw_digit, half_b_minus_one_v);
        let digit = _mm256_sub_epi32(raw_digit, _mm256_and_si256(b_v, high_digit));

        store_i8_digits_avx2(digit, out.as_mut_ptr().add(base));

        let positive_numerator = _mm256_sub_epi32(values, digit);
        let positive_quotient = _mm256_srlv_epi32(positive_numerator, shift_v);
        let negative_magnitude = _mm256_add_epi32(_mm256_sub_epi32(q_v, values), digit);
        let negative_quotient =
            _mm256_sub_epi32(zero, _mm256_srlv_epi32(negative_magnitude, shift_v));
        let mut quotient = _mm256_blendv_epi8(positive_quotient, negative_quotient, negative);

        for level in 1..params.levels {
            let raw_digit = _mm256_and_si256(quotient, mask_v);
            let high_digit = _mm256_cmpgt_epi32(raw_digit, half_b_minus_one_v);
            let digit = _mm256_sub_epi32(raw_digit, _mm256_and_si256(b_v, high_digit));
            quotient = _mm256_srav_epi32(_mm256_sub_epi32(quotient, digit), shift_v);
            store_i8_digits_avx2(digit, out.as_mut_ptr().add(level * width + base));
        }

        base += 8;
    }
}
