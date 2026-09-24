//! Binary startup, followed by the shared NEON DIF stages.

use std::arch::aarch64::*;

use super::i32_kernels::{forward_ntt_i32_from_twisted, mont_mul_4x_i32, reduce_range_4x_i32};
use crate::ntt::{BinaryNttStrategy, MontCoeff, NttPrime, NttTwiddles};

/// The caller validates binary digits, D >= 16, and all table slice lengths.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn forward_binary_i32<const D: usize>(
    out: &mut [MontCoeff<i32>; D],
    digits: &[i8; D],
    prime: NttPrime<i32>,
    tw: &NttTwiddles<i32, D>,
    compact: &[i32],
    factors: &[i32],
    pairs: &[i32],
    positional: Option<&[[i32; 16]]>,
    selected4: Option<&[i32]>,
    split_twiddles: &[i32],
    split_companions: &[i32],
    split8: Option<&[i32]>,
    strategy: BinaryNttStrategy,
) {
    let dst = out.as_mut_ptr().cast::<i32>();
    let p = vdupq_n_s32(prime.p);
    let pinv = vdupq_n_s32(prime.pinv);
    let len = match strategy {
        BinaryNttStrategy::Split8 => {
            if let Some(table) = split8 {
                super::split_binary::forward_split8(
                    out,
                    digits,
                    prime,
                    table,
                    split_twiddles,
                    split_companions,
                );
            }
            return;
        }
        BinaryNttStrategy::Split4 => {
            super::split_binary::forward_split4(
                out,
                digits,
                prime,
                compact,
                split_twiddles,
                split_companions,
            );
            return;
        }
        BinaryNttStrategy::Mask => {
            for j in (0..D).step_by(16) {
                let bytes = vld1q_s8(digits.as_ptr().add(j));
                let lo = vmovl_s8(vget_low_s8(bytes));
                let hi = vmovl_high_s8(bytes);
                let words = [
                    vmovl_s16(vget_low_s16(lo)),
                    vmovl_high_s16(lo),
                    vmovl_s16(vget_low_s16(hi)),
                    vmovl_high_s16(hi),
                ];
                for (chunk, word) in words.into_iter().enumerate() {
                    let offset = j + 4 * chunk;
                    let psi = vld1q_s32(tw.psi_pows.as_ptr().cast::<i32>().add(offset));
                    vst1q_s32(dst.add(offset), vandq_s32(vnegq_s32(word), psi));
                }
            }
            D / 2
        }
        BinaryNttStrategy::SelectedPairs => {
            for j in (0..D / 2).step_by(4) {
                let a = binary_four_mask(digits.as_ptr().add(j));
                let b = binary_four_mask(digits.as_ptr().add(j + D / 2));
                for segment in 0..2 {
                    let index = segment * (D / 2) + j;
                    let u = vandq_s32(a, vld1q_s32(pairs.as_ptr().add(index)));
                    let v = vandq_s32(b, vld1q_s32(pairs.as_ptr().add(D + index)));
                    vst1q_s32(dst.add(index), reduce_range_4x_i32(vaddq_s32(u, v), p));
                }
            }
            D / 4
        }
        BinaryNttStrategy::Selected4 => {
            if let Some(factors) = selected4 {
                for j in (0..D / 4).step_by(4) {
                    let masks = [
                        binary_four_mask(digits.as_ptr().add(j)),
                        binary_four_mask(digits.as_ptr().add(j + D / 4)),
                        binary_four_mask(digits.as_ptr().add(j + D / 2)),
                        binary_four_mask(digits.as_ptr().add(j + 3 * (D / 4))),
                    ];
                    for segment in 0..4 {
                        let index = segment * (D / 4) + j;
                        let a = vandq_s32(masks[0], vld1q_s32(factors.as_ptr().add(index)));
                        let b = vandq_s32(masks[1], vld1q_s32(factors.as_ptr().add(D + index)));
                        let c = vandq_s32(masks[2], vld1q_s32(factors.as_ptr().add(2 * D + index)));
                        let d = vandq_s32(masks[3], vld1q_s32(factors.as_ptr().add(3 * D + index)));
                        // Each factor is centered. Four terms total less than
                        // 2p in absolute value, hence cannot overflow i32.
                        let sum = vaddq_s32(vaddq_s32(a, b), vaddq_s32(c, d));
                        vst1q_s32(dst.add(index), reduce_range_4x_i32(sum, p));
                    }
                }
            }
            D / 8
        }
        BinaryNttStrategy::Compact4 => {
            let byte_offsets = vcreate_u8(0x03020100_03020100);
            let offsets = vcombine_u8(byte_offsets, byte_offsets);
            for j in (0..D / 4).step_by(4) {
                let indices = nibble_four(digits, j, D / 4);
                // Four equal bytes per nibble, then add [0,1,2,3] to address
                // the four bytes of its 32-bit Montgomery table entry.
                let packed = vmulq_n_u32(indices, 0x04040404);
                let lookup = vaddq_u8(vreinterpretq_u8_u32(packed), offsets);
                for segment in 0..4 {
                    let ptr = compact.as_ptr().add(segment * 16).cast::<u8>();
                    let table = uint8x16x4_t(
                        vld1q_u8(ptr),
                        vld1q_u8(ptr.add(16)),
                        vld1q_u8(ptr.add(32)),
                        vld1q_u8(ptr.add(48)),
                    );
                    let value = vreinterpretq_s32_u8(vqtbl4q_u8(table, lookup));
                    let index = segment * (D / 4) + j;
                    let factor = vld1q_s32(factors.as_ptr().add(index));
                    vst1q_s32(dst.add(index), mont_mul_4x_i32(value, factor, p, pinv));
                }
            }
            D / 8
        }
        BinaryNttStrategy::Positional4 => {
            if let Some(table) = positional {
                for j in 0..D / 4 {
                    let nibble = (digits[j] as usize)
                        | ((digits[j + D / 4] as usize) << 1)
                        | ((digits[j + D / 2] as usize) << 2)
                        | ((digits[j + 3 * (D / 4)] as usize) << 3);
                    for segment in 0..4 {
                        let index = segment * (D / 4) + j;
                        *dst.add(index) = table[index][nibble];
                    }
                }
            }
            D / 8
        }
    };
    forward_ntt_i32_from_twisted(out, prime, tw, len);
}

/// Load exactly four bytes (never eight at the end of a source quarter).
#[inline(always)]
unsafe fn binary_four_mask(ptr: *const i8) -> int32x4_t {
    let packed = std::ptr::read_unaligned(ptr.cast::<u32>());
    let bytes = vreinterpret_s8_u64(vcreate_u64(u64::from(packed)));
    vnegq_s32(vmovl_s16(vget_low_s16(vmovl_s8(bytes))))
}

#[inline(always)]
pub(super) unsafe fn nibble_four<const D: usize>(
    digits: &[i8; D],
    j: usize,
    stride: usize,
) -> uint32x4_t {
    let a = vnegq_s32(binary_four_mask(digits.as_ptr().add(j)));
    let b = vnegq_s32(binary_four_mask(digits.as_ptr().add(j + stride)));
    let c = vnegq_s32(binary_four_mask(digits.as_ptr().add(j + 2 * stride)));
    let d = vnegq_s32(binary_four_mask(digits.as_ptr().add(j + 3 * stride)));
    vreinterpretq_u32_s32(vorrq_s32(
        vorrq_s32(a, vshlq_n_s32::<1>(b)),
        vorrq_s32(vshlq_n_s32::<2>(c), vshlq_n_s32::<3>(d)),
    ))
}
