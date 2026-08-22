//! NEON packed-digit extraction and byte-wise sign extension.

use core::arch::aarch64::*;

use super::DIGITS_PER_BLOCK;

#[derive(Clone, Copy)]
struct GatherTable {
    low: [u8; 16],
    high: [u8; 16],
    right: [i8; 16],
    left: [i8; 16],
}

const fn gather_table(bit_width: u8) -> GatherTable {
    let mut low = [0u8; 16];
    let mut high = [0u8; 16];
    let mut right = [0i8; 16];
    let mut left = [0i8; 16];
    let mut lane = 0;
    while lane < 16 {
        let bit = lane * bit_width as usize;
        let shift = (bit % 8) as u8;
        low[lane] = (bit / 8) as u8;
        high[lane] = low[lane] + 1;
        right[lane] = -(shift as i8);
        left[lane] = (8 - shift) as i8;
        lane += 1;
    }
    GatherTable {
        low,
        high,
        right,
        left,
    }
}

const GATHER_TABLES: [GatherTable; 9] = [
    gather_table(0),
    gather_table(1),
    gather_table(2),
    gather_table(3),
    gather_table(4),
    gather_table(5),
    gather_table(6),
    gather_table(7),
    gather_table(8),
];

#[target_feature(enable = "neon")]
pub(super) unsafe fn decode_full_block(
    encoded: &[u8],
    bit_width: u8,
    output: &mut [i8; DIGITS_PER_BLOCK],
) {
    debug_assert!(encoded.len() >= usize::from(bit_width) * 8 + 16);
    match bit_width {
        2 => unsafe { decode_l2(encoded, output) },
        4 => unsafe { decode_l4(encoded, output) },
        8 => unsafe { decode_l8(encoded, output) },
        _ => unsafe { decode_gather(encoded, bit_width, output) },
    }
}

#[target_feature(enable = "neon")]
unsafe fn decode_l2(encoded: &[u8], output: &mut [i8; DIGITS_PER_BLOCK]) {
    let source = unsafe { vld1q_u8(encoded.as_ptr()) };
    let mask = vdupq_n_u8(3);
    let sign = vdupq_n_u8(2);
    let d0 = unsafe { sign_extend(vandq_u8(source, mask), sign) };
    let d1 = unsafe { sign_extend(vandq_u8(vshrq_n_u8::<2>(source), mask), sign) };
    let d2 = unsafe { sign_extend(vandq_u8(vshrq_n_u8::<4>(source), mask), sign) };
    let d3 = unsafe { sign_extend(vshrq_n_u8::<6>(source), sign) };
    unsafe {
        vst4q_u8(output.as_mut_ptr().cast(), uint8x16x4_t(d0, d1, d2, d3));
    }
}

#[target_feature(enable = "neon")]
unsafe fn decode_l4(encoded: &[u8], output: &mut [i8; DIGITS_PER_BLOCK]) {
    let mask = vdupq_n_u8(15);
    let sign = vdupq_n_u8(8);
    for half in 0..2 {
        let source = unsafe { vld1q_u8(encoded.as_ptr().add(half * 16)) };
        let low = unsafe { sign_extend(vandq_u8(source, mask), sign) };
        let high = unsafe { sign_extend(vshrq_n_u8::<4>(source), sign) };
        let first = vzip1q_u8(low, high);
        let second = vzip2q_u8(low, high);
        unsafe {
            vst1q_u8(output.as_mut_ptr().add(half * 32).cast(), first);
            vst1q_u8(output.as_mut_ptr().add(half * 32 + 16).cast(), second);
        }
    }
}

#[target_feature(enable = "neon")]
unsafe fn decode_l8(encoded: &[u8], output: &mut [i8; DIGITS_PER_BLOCK]) {
    for chunk in 0..4 {
        let source = unsafe { vld1q_u8(encoded.as_ptr().add(chunk * 16)) };
        unsafe { vst1q_u8(output.as_mut_ptr().add(chunk * 16).cast(), source) };
    }
}

#[target_feature(enable = "neon")]
unsafe fn decode_gather(encoded: &[u8], bit_width: u8, output: &mut [i8; DIGITS_PER_BLOCK]) {
    let table = &GATHER_TABLES[usize::from(bit_width)];
    let low_indices = unsafe { vld1q_u8(table.low.as_ptr()) };
    let high_indices = unsafe { vld1q_u8(table.high.as_ptr()) };
    let right = unsafe { vld1q_s8(table.right.as_ptr()) };
    let left = unsafe { vld1q_s8(table.left.as_ptr()) };
    let mask = vdupq_n_u8((1u8 << bit_width) - 1);
    let sign = vdupq_n_u8(1u8 << (bit_width - 1));
    let chunk_bytes = usize::from(bit_width) * 2;

    for chunk in 0..4 {
        let source = unsafe { vld1q_u8(encoded.as_ptr().add(chunk * chunk_bytes)) };
        let low = vqtbl1q_u8(source, low_indices);
        let high = vqtbl1q_u8(source, high_indices);
        let raw = vandq_u8(vorrq_u8(vshlq_u8(low, right), vshlq_u8(high, left)), mask);
        let signed = unsafe { sign_extend(raw, sign) };
        unsafe { vst1q_u8(output.as_mut_ptr().add(chunk * 16).cast(), signed) };
    }
}

#[target_feature(enable = "neon")]
unsafe fn sign_extend(raw: uint8x16_t, sign: uint8x16_t) -> uint8x16_t {
    vsubq_u8(veorq_u8(raw, sign), sign)
}
