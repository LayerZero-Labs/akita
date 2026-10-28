//! Four u64 lanes per NEON `umlal` pair over packed q24 slots.

use super::super::super::image::{IMAGE_OFFSET, Q24};
use super::{TrinomialQ24PackedSlots, DEGREE, Q24_PACKED_SLOT_BYTES};
use std::arch::aarch64::*;

const GROUPS: usize = DEGREE / 4;
const LAST: usize = GROUPS - 1;
// A full 16-byte load at 12 * group stays inside the slot bytes up to group 160.
const _: () = assert!(12 * (LAST - 1) + 16 <= Q24_PACKED_SLOT_BYTES);
// The last group reloads the final 16 bytes; its words start 4 bytes in.
const LAST_LOAD: usize = Q24_PACKED_SLOT_BYTES - 16;
const _: () = assert!(12 * LAST - LAST_LOAD == 4);

/// Table indices that zero-extend four three-byte words starting at `skip`.
const fn word_table(skip: u8) -> [u8; 16] {
    let mut table = [0xff; 16];
    let mut position = 0;
    while position < 16 {
        if position % 4 != 3 {
            table[position] = skip + (3 * (position / 4) + position % 4) as u8;
        }
        position += 1;
    }
    table
}

const WORD_TABLE: [u8; 16] = word_table(0);
const LAST_WORD_TABLE: [u8; 16] = word_table(4);

/// Decode four consecutive three-byte words into their canonical residues.
#[inline(always)]
fn residues<const LAST_GROUP: bool>(
    bytes: &[u8; Q24_PACKED_SLOT_BYTES],
    group: usize,
) -> uint32x4_t {
    // SAFETY: NEON is statically available on this target. For group < 161
    // the 16-byte load starts at 12 * group <= 1920 and ends at or before
    // 1936 <= 1944; the last group loads bytes 1928..1944. Table index 0xff
    // is out of range and yields zero, so each word is zero-extended to u32.
    // Every word is below 2^24, so x = word - offset lies in (-q, q), and one
    // masked addition of q gives the residue in 0..q.
    unsafe {
        let (start, table) = if LAST_GROUP {
            (LAST_LOAD, LAST_WORD_TABLE)
        } else {
            (12 * group, WORD_TABLE)
        };
        let raw = vld1q_u8(bytes.as_ptr().add(start));
        let words = vreinterpretq_s32_u8(vqtbl1q_u8(raw, vld1q_u8(table.as_ptr())));
        let x = vsubq_s32(words, vdupq_n_s32(IMAGE_OFFSET));
        let q = vdupq_n_s32(Q24 as i32);
        vreinterpretq_u32_s32(vaddq_s32(x, vandq_s32(vshrq_n_s32::<31>(x), q)))
    }
}

/// Accumulate one group of four slots across the whole batch.
#[inline(always)]
fn group<const LAST_GROUP: bool>(
    sums: &mut [u64; DEGREE],
    lhs: &[TrinomialQ24PackedSlots],
    rhs: &[TrinomialQ24PackedSlots],
    index: usize,
) {
    // SAFETY: index < 162, so both two-lane sum loads/stores at 4 * index lie
    // inside `sums`. Residues are below 2^24, so each u32 x u32 product is
    // exact in u64. The caller's reduction schedule keeps every lane below
    // 2^63, so no lane wraps.
    unsafe {
        let ptr = sums.as_mut_ptr().add(4 * index);
        let mut low = vld1q_u64(ptr);
        let mut high = vld1q_u64(ptr.add(2));
        for (a, b) in lhs.iter().zip(rhs) {
            let a = residues::<LAST_GROUP>(&a.bytes, index);
            let b = residues::<LAST_GROUP>(&b.bytes, index);
            low = vmlal_u32(low, vget_low_u32(a), vget_low_u32(b));
            high = vmlal_high_u32(high, a, b);
        }
        vst1q_u64(ptr, low);
        vst1q_u64(ptr.add(2), high);
    }
}

/// The caller stops the batch at the next `2^15`-product reduction.
pub(super) fn accumulate_batch(
    sums: &mut [u64; DEGREE],
    lhs: &[TrinomialQ24PackedSlots],
    rhs: &[TrinomialQ24PackedSlots],
) {
    for index in 0..LAST {
        group::<false>(sums, lhs, rhs, index);
    }
    group::<true>(sums, lhs, rhs, LAST);
}
