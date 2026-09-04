//! AArch64 NEON dense projection kernels.

use std::arch::aarch64::*;
use std::arch::asm;

pub(super) fn selected_i8_dot() -> fn(&[i8], &[i8]) -> i64 {
    if std::arch::is_aarch64_feature_detected!("dotprod") {
        dot_i8_dotprod_dispatch
    } else {
        dot_i8_neon_dispatch
    }
}

fn dot_i8_dotprod_dispatch(weights: &[i8], input: &[i8]) -> i64 {
    // SAFETY: dispatch checks `dotprod`.
    unsafe { dot_i8_dotprod(weights, input) }
}

fn dot_i8_neon_dispatch(weights: &[i8], input: &[i8]) -> i64 {
    // SAFETY: Advanced SIMD is mandatory on AArch64.
    unsafe { dot_i8_neon(weights, input) }
}

pub(super) fn dot_i16_neon_dispatch(weights: &[i8], input: &[i16]) -> i64 {
    // SAFETY: Advanced SIMD is mandatory on AArch64.
    unsafe { dot_i16_neon(weights, input) }
}

pub(super) fn dot_i32_neon_dispatch(weights: &[i8], input: &[i32]) -> i64 {
    // SAFETY: Advanced SIMD is mandatory on AArch64.
    unsafe { dot_i32_neon(weights, input) }
}

pub(super) fn dot_i64_neon_dispatch(weights: &[i8], input: &[i64]) -> i64 {
    // SAFETY: Advanced SIMD is mandatory on AArch64.
    unsafe { dot_i64_neon(weights, input) }
}

#[target_feature(enable = "dotprod")]
unsafe fn dot_i8_dotprod(weights: &[i8], input: &[i8]) -> i64 {
    let len = weights.len().min(input.len());
    let mut sum = vdupq_n_s64(0);
    let mut index = 0;
    while index + 16 <= len {
        let w = vld1q_s8(weights.as_ptr().add(index));
        let x = vld1q_s8(input.as_ptr().add(index));
        let mut products = vdupq_n_s32(0);
        asm!(
            "sdot {products:v}.4s, {weights:v}.16b, {input:v}.16b",
            products = inout(vreg) products,
            weights = in(vreg) w,
            input = in(vreg) x,
            options(pure, nomem, nostack),
        );
        sum = vpadalq_s32(sum, products);
        index += 16;
    }
    let mut result = vaddvq_s64(sum);
    while index < len {
        result += i64::from(*weights.get_unchecked(index)) * i64::from(*input.get_unchecked(index));
        index += 1;
    }
    result
}

#[target_feature(enable = "neon")]
unsafe fn dot_i8_neon(weights: &[i8], input: &[i8]) -> i64 {
    let len = weights.len().min(input.len());
    let mut sum = vdupq_n_s64(0);
    let mut index = 0;
    while index + 16 <= len {
        let w = vld1q_s8(weights.as_ptr().add(index));
        let x = vld1q_s8(input.as_ptr().add(index));
        let low = vmull_s8(vget_low_s8(w), vget_low_s8(x));
        let high = vmull_high_s8(w, x);
        sum = vpadalq_s32(sum, vpaddlq_s16(low));
        sum = vpadalq_s32(sum, vpaddlq_s16(high));
        index += 16;
    }
    let mut result = vaddvq_s64(sum);
    while index < len {
        result += i64::from(*weights.get_unchecked(index)) * i64::from(*input.get_unchecked(index));
        index += 1;
    }
    result
}

#[target_feature(enable = "neon")]
unsafe fn dot_i16_neon(weights: &[i8], input: &[i16]) -> i64 {
    let len = weights.len().min(input.len());
    let mut sum = vdupq_n_s64(0);
    let mut index = 0;
    while index + 8 <= len {
        let w = vmovl_s8(vld1_s8(weights.as_ptr().add(index)));
        let x = vld1q_s16(input.as_ptr().add(index));
        sum = vpadalq_s32(sum, vmull_s16(vget_low_s16(w), vget_low_s16(x)));
        sum = vpadalq_s32(sum, vmull_high_s16(w, x));
        index += 8;
    }
    let mut result = vaddvq_s64(sum);
    while index < len {
        result += i64::from(*weights.get_unchecked(index)) * i64::from(*input.get_unchecked(index));
        index += 1;
    }
    result
}

#[target_feature(enable = "neon")]
unsafe fn dot_i32_neon(weights: &[i8], input: &[i32]) -> i64 {
    let len = weights.len().min(input.len());
    let mut sums = [vdupq_n_s64(0); 4];
    let mut index = 0;
    while index + 8 <= len {
        let w16 = vmovl_s8(vld1_s8(weights.as_ptr().add(index)));
        let w0 = vmovl_s16(vget_low_s16(w16));
        let w1 = vmovl_high_s16(w16);
        let x0 = vld1q_s32(input.as_ptr().add(index));
        let x1 = vld1q_s32(input.as_ptr().add(index + 4));
        sums[0] = vaddq_s64(sums[0], vmull_s32(vget_low_s32(w0), vget_low_s32(x0)));
        sums[1] = vaddq_s64(sums[1], vmull_high_s32(w0, x0));
        sums[2] = vaddq_s64(sums[2], vmull_s32(vget_low_s32(w1), vget_low_s32(x1)));
        sums[3] = vaddq_s64(sums[3], vmull_high_s32(w1, x1));
        index += 8;
    }
    let mut result = vaddvq_s64(vaddq_s64(
        vaddq_s64(sums[0], sums[1]),
        vaddq_s64(sums[2], sums[3]),
    ));
    while index < len {
        result += i64::from(*weights.get_unchecked(index)) * i64::from(*input.get_unchecked(index));
        index += 1;
    }
    result
}

#[target_feature(enable = "neon")]
unsafe fn dot_i64_neon(weights: &[i8], input: &[i64]) -> i64 {
    let len = weights.len().min(input.len());
    let zero = vdupq_n_s64(0);
    let mut sums = [zero; 4];
    let mut index = 0;
    while index + 8 <= len {
        let w16 = vmovl_s8(vld1_s8(weights.as_ptr().add(index)));
        let w32 = [vmovl_s16(vget_low_s16(w16)), vmovl_high_s16(w16)];
        let w = [
            vmovl_s32(vget_low_s32(w32[0])),
            vmovl_high_s32(w32[0]),
            vmovl_s32(vget_low_s32(w32[1])),
            vmovl_high_s32(w32[1]),
        ];
        for lane in 0..4 {
            let x = vld1q_s64(input.as_ptr().add(index + lane * 2));
            let positive = vcgtq_s64(w[lane], zero);
            let negative = vcltq_s64(w[lane], zero);
            sums[lane] = vaddq_s64(sums[lane], vandq_s64(vreinterpretq_s64_u64(positive), x));
            sums[lane] = vsubq_s64(sums[lane], vandq_s64(vreinterpretq_s64_u64(negative), x));
        }
        index += 8;
    }
    let mut result = vaddvq_s64(vaddq_s64(
        vaddq_s64(sums[0], sums[1]),
        vaddq_s64(sums[2], sums[3]),
    ));
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
