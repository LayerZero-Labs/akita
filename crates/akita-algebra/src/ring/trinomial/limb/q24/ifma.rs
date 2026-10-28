//! Eight u64 lanes per AVX-512 IFMA multiply-add over packed q24 slots.

use super::super::super::image::{IMAGE_OFFSET, Q24};
use super::{TrinomialQ24PackedSlots, DEGREE, Q24_PACKED_SLOT_BYTES};
use std::arch::x86_64::*;

const GROUPS: usize = DEGREE / 8;
const LAST: usize = GROUPS - 1;
// A full 32-byte load at 24 * group stays inside the slot bytes up to group 79.
const _: () = assert!(24 * (LAST - 1) + 32 <= Q24_PACKED_SLOT_BYTES);
// The last group reloads the final 32 bytes; its words start 8 bytes in.
const LAST_LOAD: usize = Q24_PACKED_SLOT_BYTES - 32;
const _: () = assert!(24 * LAST - LAST_LOAD == 8);

/// Byte shuffle that zero-extends four three-byte words per 128-bit lane.
const fn word_shuffle() -> [i8; 32] {
    let mut table = [-128; 32];
    let mut position = 0;
    while position < 32 {
        let lane_position = position % 16;
        if lane_position % 4 != 3 {
            table[position] = (3 * (lane_position / 4) + lane_position % 4) as i8;
        }
        position += 1;
    }
    table
}

const WORD_SHUFFLE: [i8; 32] = word_shuffle();

/// Decode eight consecutive three-byte words into their canonical residues.
#[inline(always)]
unsafe fn residues<const LAST_GROUP: bool>(
    bytes: &[u8; Q24_PACKED_SLOT_BYTES],
    group: usize,
) -> __m512i {
    // SAFETY: The caller detects AVX2, AVX-512F and AVX-512 IFMA. For
    // group < 80 the 32-byte load starts at 24 * group <= 1896 and ends at or
    // before 1928 <= 1944. The last group loads bytes 1912..1944. Dword
    // selection puts words 0..3 at the start of the low 128-bit lane and
    // words 4..7 at the start of the high lane; the byte shuffle zero-extends
    // each three-byte word to u32. Every word is below 2^24, so
    // x = word - offset lies in (-q, q), and one masked addition of q gives
    // the residue in 0..q.
    unsafe {
        let (start, select) = if LAST_GROUP {
            (LAST_LOAD, _mm256_setr_epi32(2, 3, 4, 5, 5, 6, 7, 7))
        } else {
            (24 * group, _mm256_setr_epi32(0, 1, 2, 3, 3, 4, 5, 6))
        };
        let raw = _mm256_loadu_si256(bytes.as_ptr().add(start).cast());
        let lanes = _mm256_permutevar8x32_epi32(raw, select);
        let shuffle = _mm256_loadu_si256(WORD_SHUFFLE.as_ptr().cast());
        let words = _mm256_shuffle_epi8(lanes, shuffle);
        let x = _mm256_sub_epi32(words, _mm256_set1_epi32(IMAGE_OFFSET));
        let q = _mm256_set1_epi32(Q24 as i32);
        let residue = _mm256_add_epi32(x, _mm256_and_si256(_mm256_srai_epi32::<31>(x), q));
        _mm512_cvtepu32_epi64(residue)
    }
}

/// Accumulate one group of eight slots across the whole batch.
#[inline(always)]
unsafe fn group<const LAST_GROUP: bool>(
    sums: &mut [u64; DEGREE],
    lhs: &[TrinomialQ24PackedSlots],
    rhs: &[TrinomialQ24PackedSlots],
    index: usize,
) {
    // SAFETY: index < 81, so the eight-lane sum load/store at 8 * index lies
    // inside `sums`. Residues are below 2^24, so the 52-bit low product is the
    // full product, and IFMA adds it to the whole 64-bit lane. The caller's
    // reduction schedule keeps every lane below 2^63, so no lane wraps.
    unsafe {
        let ptr = sums.as_mut_ptr().add(8 * index);
        let mut sum = _mm512_loadu_si512(ptr.cast_const().cast());
        for (a, b) in lhs.iter().zip(rhs) {
            let a = residues::<LAST_GROUP>(&a.bytes, index);
            let b = residues::<LAST_GROUP>(&b.bytes, index);
            sum = _mm512_madd52lo_epu64(sum, a, b);
        }
        _mm512_storeu_si512(ptr.cast(), sum);
    }
}

/// Caller must detect AVX2, AVX-512F/DQ and AVX-512 IFMA, and stop the batch
/// at the next `2^15`-product reduction.
#[target_feature(enable = "avx2,avx512f,avx512dq,avx512ifma")]
pub(super) unsafe fn accumulate_batch(
    sums: &mut [u64; DEGREE],
    lhs: &[TrinomialQ24PackedSlots],
    rhs: &[TrinomialQ24PackedSlots],
) {
    // SAFETY: Eighty-one groups of eight lanes cover DEGREE exactly, and this
    // function's target features cover every intrinsic in `group`.
    unsafe {
        for index in 0..LAST {
            group::<false>(sums, lhs, rhs, index);
        }
        group::<true>(sums, lhs, rhs, LAST);
    }
}
