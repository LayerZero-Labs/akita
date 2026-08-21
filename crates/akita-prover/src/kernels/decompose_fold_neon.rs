//! AArch64 NEON kernel for sparse-multiply-accumulate in the decompose-fold
//! pipeline.
//!
//! Called from [`crate::backend::poly_helpers::sparse_mul_acc`] when NEON is available and
//! challenge coefficients have magnitude ≤ 2.  Rotates an i8 digit plane by
//! each challenge position and accumulates into an i32 accumulator using
//! widening add/sub (`SADDW` / `SSUBW`).

use std::arch::aarch64::*;

/// NEON sparse-multiply-accumulate.
///
/// For each challenge term `(pos, coeff)`, rotates the `digit_plane` by `pos`
/// positions in the negacyclic ring (X^D + 1) and adds or subtracts the
/// widened i8 values into the i32 `acc`. Coefficients `+/-2` use one widened,
/// doubled add/sub pass so two-magnitude challenge families stay on the NEON
/// fast path without repeating a full rotation.
///
/// # Safety
///
/// - `digit_plane` must point to at least `d` valid i8 values.
/// - `acc` must point to at least `d` valid i32 values.
/// - `d` must be a multiple of 16.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn sparse_mul_acc_neon(
    digit_plane: *const i8,
    acc: *mut i32,
    d: usize,
    positions: &[u32],
    coeffs: &[i8],
) {
    debug_assert!(d.is_multiple_of(16));

    for (&pos, &coeff) in positions.iter().zip(coeffs.iter()) {
        let p = pos as usize;
        let split = d - p;

        match coeff {
            1 => acc_rotated_add(digit_plane, acc, d, p, split),
            -1 => acc_rotated_sub(digit_plane, acc, d, p, split),
            2 => acc_rotated_add_twice(digit_plane, acc, p, split),
            -2 => acc_rotated_sub_twice(digit_plane, acc, p, split),
            _ => {
                for _ in 0..coeff.unsigned_abs() {
                    if coeff > 0 {
                        acc_rotated_add(digit_plane, acc, d, p, split);
                    } else {
                        acc_rotated_sub(digit_plane, acc, d, p, split);
                    }
                }
            }
        }
    }
}

/// Branch-free NEON kernel for a prepared ±1-only challenge.
///
/// # Safety
///
/// `digit_plane` and `acc` must each address `d` valid elements, every
/// position must be below `d`, and `d` must be a multiple of 16.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn sparse_mul_acc_pm1_neon(
    digit_plane: *const i8,
    acc: *mut i32,
    d: usize,
    positive: &[u32],
    negative: &[u32],
) {
    debug_assert!(d.is_multiple_of(16));
    for &position in positive {
        let position = position as usize;
        acc_rotated_add(digit_plane, acc, d, position, d - position);
    }
    for &position in negative {
        let position = position as usize;
        acc_rotated_sub(digit_plane, acc, d, position, d - position);
    }
}

/// Signed-i16 variant used by large inner decomposition bases.
///
/// # Safety
///
/// - `digit_plane` must point to at least `d` valid `i16` values.
/// - `acc` must point to at least `d` valid `i32` values.
/// - Every challenge position must be less than `d`.
/// - `positions` and `coeffs` must have equal lengths.
/// - `d` must be a multiple of 8.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn sparse_mul_acc_i16_neon(
    digit_plane: *const i16,
    acc: *mut i32,
    d: usize,
    positions: &[u32],
    coeffs: &[i8],
) {
    debug_assert!(d.is_multiple_of(8));
    for (&pos, &coeff) in positions.iter().zip(coeffs.iter()) {
        let p = pos as usize;
        let split = d - p;
        match coeff {
            1 => {
                acc_segment_i16_add(digit_plane, acc.add(p), split);
                if p > 0 {
                    acc_segment_i16_sub(digit_plane.add(split), acc, p);
                }
            }
            -1 => {
                acc_segment_i16_sub(digit_plane, acc.add(p), split);
                if p > 0 {
                    acc_segment_i16_add(digit_plane.add(split), acc, p);
                }
            }
            2 => {
                acc_segment_i16_add_twice(digit_plane, acc.add(p), split);
                if p > 0 {
                    acc_segment_i16_sub_twice(digit_plane.add(split), acc, p);
                }
            }
            -2 => {
                acc_segment_i16_sub_twice(digit_plane, acc.add(p), split);
                if p > 0 {
                    acc_segment_i16_add_twice(digit_plane.add(split), acc, p);
                }
            }
            _ => {
                for _ in 0..coeff.unsigned_abs() {
                    if coeff > 0 {
                        acc_segment_i16_add(digit_plane, acc.add(p), split);
                        if p > 0 {
                            acc_segment_i16_sub(digit_plane.add(split), acc, p);
                        }
                    } else {
                        acc_segment_i16_sub(digit_plane, acc.add(p), split);
                        if p > 0 {
                            acc_segment_i16_add(digit_plane.add(split), acc, p);
                        }
                    }
                }
            }
        }
    }
}

/// Branch-free NEON i16 kernel for a prepared ±1-only challenge.
///
/// # Safety
///
/// `digit_plane` and `acc` must each address `d` valid elements, every
/// position must be below `d`, and `d` must be a multiple of 8.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn sparse_mul_acc_i16_pm1_neon(
    digit_plane: *const i16,
    acc: *mut i32,
    d: usize,
    positive: &[u32],
    negative: &[u32],
) {
    debug_assert!(d.is_multiple_of(8));
    for &position in positive {
        let position = position as usize;
        let split = d - position;
        acc_segment_i16_add(digit_plane, acc.add(position), split);
        if position > 0 {
            acc_segment_i16_sub(digit_plane.add(split), acc, position);
        }
    }
    for &position in negative {
        let position = position as usize;
        let split = d - position;
        acc_segment_i16_sub(digit_plane, acc.add(position), split);
        if position > 0 {
            acc_segment_i16_add(digit_plane.add(split), acc, position);
        }
    }
}

#[inline(always)]
unsafe fn acc_segment_i16_add(src: *const i16, dst: *mut i32, len: usize) {
    let chunks = len / 8;
    for i in 0..chunks {
        let offset = i * 8;
        let values = vld1q_s16(src.add(offset));
        let low = vmovl_s16(vget_low_s16(values));
        let high = vmovl_s16(vget_high_s16(values));
        let dst_low = vld1q_s32(dst.add(offset));
        let dst_high = vld1q_s32(dst.add(offset + 4));
        vst1q_s32(dst.add(offset), vaddq_s32(dst_low, low));
        vst1q_s32(dst.add(offset + 4), vaddq_s32(dst_high, high));
    }
    for i in chunks * 8..len {
        *dst.add(i) += i32::from(*src.add(i));
    }
}

#[inline(always)]
unsafe fn acc_segment_i16_sub(src: *const i16, dst: *mut i32, len: usize) {
    let chunks = len / 8;
    for i in 0..chunks {
        let offset = i * 8;
        let values = vld1q_s16(src.add(offset));
        let low = vmovl_s16(vget_low_s16(values));
        let high = vmovl_s16(vget_high_s16(values));
        let dst_low = vld1q_s32(dst.add(offset));
        let dst_high = vld1q_s32(dst.add(offset + 4));
        vst1q_s32(dst.add(offset), vsubq_s32(dst_low, low));
        vst1q_s32(dst.add(offset + 4), vsubq_s32(dst_high, high));
    }
    for i in chunks * 8..len {
        *dst.add(i) -= i32::from(*src.add(i));
    }
}

#[inline(always)]
unsafe fn acc_segment_i16_add_twice(src: *const i16, dst: *mut i32, len: usize) {
    let chunks = len / 8;
    for i in 0..chunks {
        let offset = i * 8;
        let values = vld1q_s16(src.add(offset));
        let low = vshlq_n_s32::<1>(vmovl_s16(vget_low_s16(values)));
        let high = vshlq_n_s32::<1>(vmovl_s16(vget_high_s16(values)));
        let dst_low = vld1q_s32(dst.add(offset));
        let dst_high = vld1q_s32(dst.add(offset + 4));
        vst1q_s32(dst.add(offset), vaddq_s32(dst_low, low));
        vst1q_s32(dst.add(offset + 4), vaddq_s32(dst_high, high));
    }
    for i in chunks * 8..len {
        *dst.add(i) += 2 * i32::from(*src.add(i));
    }
}

#[inline(always)]
unsafe fn acc_segment_i16_sub_twice(src: *const i16, dst: *mut i32, len: usize) {
    let chunks = len / 8;
    for i in 0..chunks {
        let offset = i * 8;
        let values = vld1q_s16(src.add(offset));
        let low = vshlq_n_s32::<1>(vmovl_s16(vget_low_s16(values)));
        let high = vshlq_n_s32::<1>(vmovl_s16(vget_high_s16(values)));
        let dst_low = vld1q_s32(dst.add(offset));
        let dst_high = vld1q_s32(dst.add(offset + 4));
        vst1q_s32(dst.add(offset), vsubq_s32(dst_low, low));
        vst1q_s32(dst.add(offset + 4), vsubq_s32(dst_high, high));
    }
    for i in chunks * 8..len {
        *dst.add(i) -= 2 * i32::from(*src.add(i));
    }
}

/// Add rotated digit plane: acc[i+p] += digits[i] for i in [0, split),
/// acc[i-split] -= digits[i] for i in [split, D) (negacyclic wrap).
#[inline(always)]
unsafe fn acc_rotated_add(digits: *const i8, acc: *mut i32, d: usize, p: usize, split: usize) {
    // First segment: digits[0..split] -> acc[p..D], ADD
    acc_segment_add(digits, acc.add(p), split);
    // Second segment: digits[split..D] -> acc[0..p], SUB (negacyclic)
    if p > 0 {
        acc_segment_sub(digits.add(split), acc, p);
    }
    let _ = d;
}

/// Sub rotated digit plane: acc[i+p] -= digits[i] for i in [0, split),
/// acc[i-split] += digits[i] for i in [split, D) (negacyclic wrap).
#[inline(always)]
unsafe fn acc_rotated_sub(digits: *const i8, acc: *mut i32, d: usize, p: usize, split: usize) {
    // First segment: digits[0..split] -> acc[p..D], SUB
    acc_segment_sub(digits, acc.add(p), split);
    // Second segment: digits[split..D] -> acc[0..p], ADD (negacyclic)
    if p > 0 {
        acc_segment_add(digits.add(split), acc, p);
    }
    let _ = d;
}

#[inline(always)]
unsafe fn acc_rotated_add_twice(digits: *const i8, acc: *mut i32, p: usize, split: usize) {
    acc_segment_add_twice(digits, acc.add(p), split);
    if p > 0 {
        acc_segment_sub_twice(digits.add(split), acc, p);
    }
}

#[inline(always)]
unsafe fn acc_rotated_sub_twice(digits: *const i8, acc: *mut i32, p: usize, split: usize) {
    acc_segment_sub_twice(digits, acc.add(p), split);
    if p > 0 {
        acc_segment_add_twice(digits.add(split), acc, p);
    }
}

/// Widen i8 source values to i32 and ADD into accumulator.
/// Handles arbitrary length (processes 16 at a time, then remainder).
#[inline(always)]
unsafe fn acc_segment_add(src: *const i8, dst: *mut i32, len: usize) {
    let chunks = len / 16;
    let rem = len % 16;

    for i in 0..chunks {
        let offset = i * 16;
        let v = vld1q_s8(src.add(offset));

        let lo8 = vget_low_s8(v);
        let hi8 = vget_high_s8(v);
        let lo16 = vmovl_s8(lo8);
        let hi16 = vmovl_s8(hi8);

        let s0 = vmovl_s16(vget_low_s16(lo16));
        let s1 = vmovl_s16(vget_high_s16(lo16));
        let s2 = vmovl_s16(vget_low_s16(hi16));
        let s3 = vmovl_s16(vget_high_s16(hi16));

        let d0 = vld1q_s32(dst.add(offset));
        let d1 = vld1q_s32(dst.add(offset + 4));
        let d2 = vld1q_s32(dst.add(offset + 8));
        let d3 = vld1q_s32(dst.add(offset + 12));

        vst1q_s32(dst.add(offset), vaddq_s32(d0, s0));
        vst1q_s32(dst.add(offset + 4), vaddq_s32(d1, s1));
        vst1q_s32(dst.add(offset + 8), vaddq_s32(d2, s2));
        vst1q_s32(dst.add(offset + 12), vaddq_s32(d3, s3));
    }

    let base = chunks * 16;
    for i in 0..rem {
        let val = *src.add(base + i) as i32;
        *dst.add(base + i) += val;
    }
}

/// Widen i8 source values to i32 and SUB from accumulator.
/// Handles arbitrary length (processes 16 at a time, then remainder).
#[inline(always)]
unsafe fn acc_segment_sub(src: *const i8, dst: *mut i32, len: usize) {
    let chunks = len / 16;
    let rem = len % 16;

    for i in 0..chunks {
        let offset = i * 16;
        let v = vld1q_s8(src.add(offset));

        let lo8 = vget_low_s8(v);
        let hi8 = vget_high_s8(v);
        let lo16 = vmovl_s8(lo8);
        let hi16 = vmovl_s8(hi8);

        let s0 = vmovl_s16(vget_low_s16(lo16));
        let s1 = vmovl_s16(vget_high_s16(lo16));
        let s2 = vmovl_s16(vget_low_s16(hi16));
        let s3 = vmovl_s16(vget_high_s16(hi16));

        let d0 = vld1q_s32(dst.add(offset));
        let d1 = vld1q_s32(dst.add(offset + 4));
        let d2 = vld1q_s32(dst.add(offset + 8));
        let d3 = vld1q_s32(dst.add(offset + 12));

        vst1q_s32(dst.add(offset), vsubq_s32(d0, s0));
        vst1q_s32(dst.add(offset + 4), vsubq_s32(d1, s1));
        vst1q_s32(dst.add(offset + 8), vsubq_s32(d2, s2));
        vst1q_s32(dst.add(offset + 12), vsubq_s32(d3, s3));
    }

    let base = chunks * 16;
    for i in 0..rem {
        let val = *src.add(base + i) as i32;
        *dst.add(base + i) -= val;
    }
}

#[inline(always)]
unsafe fn acc_segment_add_twice(src: *const i8, dst: *mut i32, len: usize) {
    let chunks = len / 16;
    for i in 0..chunks {
        let offset = i * 16;
        let values = vld1q_s8(src.add(offset));
        let low = vmovl_s8(vget_low_s8(values));
        let high = vmovl_s8(vget_high_s8(values));
        let lanes = [
            vshlq_n_s32::<1>(vmovl_s16(vget_low_s16(low))),
            vshlq_n_s32::<1>(vmovl_s16(vget_high_s16(low))),
            vshlq_n_s32::<1>(vmovl_s16(vget_low_s16(high))),
            vshlq_n_s32::<1>(vmovl_s16(vget_high_s16(high))),
        ];
        for (lane, values) in lanes.into_iter().enumerate() {
            let lane_dst = dst.add(offset + lane * 4);
            vst1q_s32(lane_dst, vaddq_s32(vld1q_s32(lane_dst), values));
        }
    }
    for i in chunks * 16..len {
        *dst.add(i) += 2 * i32::from(*src.add(i));
    }
}

#[inline(always)]
unsafe fn acc_segment_sub_twice(src: *const i8, dst: *mut i32, len: usize) {
    let chunks = len / 16;
    for i in 0..chunks {
        let offset = i * 16;
        let values = vld1q_s8(src.add(offset));
        let low = vmovl_s8(vget_low_s8(values));
        let high = vmovl_s8(vget_high_s8(values));
        let lanes = [
            vshlq_n_s32::<1>(vmovl_s16(vget_low_s16(low))),
            vshlq_n_s32::<1>(vmovl_s16(vget_high_s16(low))),
            vshlq_n_s32::<1>(vmovl_s16(vget_low_s16(high))),
            vshlq_n_s32::<1>(vmovl_s16(vget_high_s16(high))),
        ];
        for (lane, values) in lanes.into_iter().enumerate() {
            let lane_dst = dst.add(offset + lane * 4);
            vst1q_s32(lane_dst, vsubq_s32(vld1q_s32(lane_dst), values));
        }
    }
    for i in chunks * 16..len {
        *dst.add(i) -= 2 * i32::from(*src.add(i));
    }
}
