use std::arch::aarch64::*;

use super::BalancedDecomposePow2Params;

#[inline(always)]
unsafe fn store_i8_digits_neon(digits: int32x4_t, dst: *mut i8) {
    let words = vmovn_s32(digits);
    let bytes = vmovn_s16(vcombine_s16(words, vdup_n_s16(0)));
    vst1_lane_u32(dst.cast(), vreinterpret_u32_s8(bytes), 0);
}

/// NEON balanced decomposition of canonical fp32 representatives.
///
/// The first quotient handles negative centered representatives through their
/// unsigned magnitude. This is required by full-width asymmetric centering,
/// whose smallest representative can be slightly below `i32::MIN`. After one
/// digit, every quotient fits an i32 lane and uses signed arithmetic shifts.
#[target_feature(enable = "neon")]
pub(super) unsafe fn balanced_decompose_canonical_u32_pow2_i8_neon(
    canonical: &[u32],
    out: &mut [i8],
    params: &BalancedDecomposePow2Params,
) {
    debug_assert!(canonical.len().is_multiple_of(4));
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

    let q_v = vdupq_n_u32(q);
    let threshold_v = vdupq_n_u32(threshold);
    let b_v = vdupq_n_u32(b);
    let half_b_minus_one_v = vdupq_n_u32(half_b - 1);
    let mask_v = vdupq_n_u32(mask);
    let shift_v = vdupq_n_s32(-(params.log_basis as i32));
    let zero = vdupq_n_s32(0);

    let mut base = 0usize;
    while base < width {
        let values = vld1q_u32(canonical.as_ptr().add(base));
        let negative = vcgtq_u32(values, threshold_v);
        let centered_low = vsubq_u32(values, vandq_u32(q_v, negative));
        let raw_digit = vandq_u32(centered_low, mask_v);
        let high_digit = vcgtq_u32(raw_digit, half_b_minus_one_v);
        let digit = vreinterpretq_s32_u32(vsubq_u32(raw_digit, vandq_u32(b_v, high_digit)));

        store_i8_digits_neon(digit, out.as_mut_ptr().add(base));

        let positive_numerator = vsubq_u32(values, vreinterpretq_u32_s32(digit));
        let positive_quotient = vreinterpretq_s32_u32(vshlq_u32(positive_numerator, shift_v));
        let negative_magnitude = vaddq_u32(vsubq_u32(q_v, values), vreinterpretq_u32_s32(digit));
        let negative_quotient = vsubq_s32(
            zero,
            vreinterpretq_s32_u32(vshlq_u32(negative_magnitude, shift_v)),
        );
        let mut quotient = vbslq_s32(negative, negative_quotient, positive_quotient);

        for level in 1..params.levels {
            let raw_digit = vandq_u32(vreinterpretq_u32_s32(quotient), mask_v);
            let high_digit = vcgtq_u32(raw_digit, half_b_minus_one_v);
            let digit = vreinterpretq_s32_u32(vsubq_u32(raw_digit, vandq_u32(b_v, high_digit)));
            quotient = vshlq_s32(vsubq_s32(quotient, digit), shift_v);
            store_i8_digits_neon(digit, out.as_mut_ptr().add(level * width + base));
        }

        base += 4;
    }
}
