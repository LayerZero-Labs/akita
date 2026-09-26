//! AArch64 NEON kernel for field-switch partial rows.

use std::arch::aarch64::{
    uint8x16_t, vandq_u8, vdupq_n_u8, veorq_u8, vld1q_u8, vqtbl1q_u8, vshrq_n_u8, vst1q_u8,
};

use super::{bit_matrix::transpose8, SwitchField};

const TILE: usize = 64;
const BLOCK: usize = 8;
const MAX_HOST_BYTES: usize = 192 / 8;
const MAX_SOURCE_BYTES: usize = 128 / 8;

/// Images of all four-bit subsets of four source bytes.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn nibble_table(bytes: [u8; 4]) -> uint8x16_t {
    let mut entries = [0u8; 16];
    for mask in 1usize..16 {
        let bit = mask.trailing_zeros() as usize;
        entries[mask] = entries[mask & (mask - 1)] ^ bytes[bit];
    }
    // SAFETY: `entries` contains a complete 16-byte vector.
    unsafe { vld1q_u8(entries.as_ptr()) }
}

/// Compute ordered partial rows from whole 64-entry tiles.
///
/// Eight adjacent source bytes form two four-bit XOR-subset tables. Each host
/// row supplies an eight-bit source selector; NEON table lookup evaluates its
/// low and high nibbles for sixteen rows at a time. The output remains in the
/// profile's low-coordinate-first order, with zero padding for F192.
///
/// # Safety
///
/// The caller must establish NEON support. Both slices must have equal lengths
/// divisible by 64, and `H` must be one of the sealed F128/F128 or F64/F192
/// profiles. Their host fields contain two or three 64-bit words, respectively.
#[target_feature(enable = "neon")]
pub(super) unsafe fn partials<H: SwitchField>(source: &[H::Source], weights: &[H]) -> [u128; 192] {
    debug_assert_eq!(source.len(), weights.len());
    debug_assert_eq!(source.len() % TILE, 0);
    let host_bytes = H::ROWS / 8;
    let source_bytes = std::mem::size_of::<H::Source>();
    debug_assert!(matches!(host_bytes, 16 | 24));
    debug_assert!(matches!(source_bytes, 8 | 16));

    // One vector holds the same source-byte position for sixteen host rows.
    // Accumulating here avoids extracting SIMD lanes in every eight-source
    // block; the final extraction is independent of the domain size.
    let zero = vdupq_n_u8(0);
    let mut accumulators = [[zero; MAX_SOURCE_BYTES]; MAX_HOST_BYTES / 2];
    let nibble_mask = vdupq_n_u8(15);

    for tile in (0..source.len()).step_by(TILE) {
        for block in (tile..tile + TILE).step_by(BLOCK) {
            let mut host_words = [[0u64; BLOCK]; 3];
            for (lane, &host) in weights[block..block + BLOCK].iter().enumerate() {
                let words = host.coordinates();
                for (dst, word) in host_words.iter_mut().zip(words) {
                    dst[lane] = word;
                }
            }

            let mut selectors = [[0u8; 16]; MAX_HOST_BYTES / 2];
            for host_byte in 0..host_bytes {
                let shift = 8 * (host_byte % 8);
                let mut packed = 0u64;
                for (lane, &word) in host_words[host_byte / 8].iter().enumerate() {
                    packed |= ((word >> shift) & 0xff) << (8 * lane);
                }
                let row_selectors = transpose8(packed).to_le_bytes();
                let pair = host_byte / 2;
                let offset = 8 * (host_byte % 2);
                selectors[pair][offset..offset + 8].copy_from_slice(&row_selectors);
            }
            let mut low_selectors = [zero; MAX_HOST_BYTES / 2];
            let mut high_selectors = [zero; MAX_HOST_BYTES / 2];
            for pair in 0..host_bytes / 2 {
                // SAFETY: each selector array contains sixteen valid bytes.
                let all = unsafe { vld1q_u8(selectors[pair].as_ptr()) };
                low_selectors[pair] = vandq_u8(all, nibble_mask);
                high_selectors[pair] = vshrq_n_u8::<4>(all);
            }

            let mut source_values = [0u128; BLOCK];
            for (dst, &value) in source_values.iter_mut().zip(&source[block..block + BLOCK]) {
                *dst = value.into();
            }
            // Pair-major accumulator storage benchmarks about 30% faster on
            // M4 Max than source-major storage for both sealed profiles.
            #[allow(clippy::needless_range_loop)]
            for source_byte in 0..source_bytes {
                let shift = 8 * source_byte;
                let mut bytes = [0u8; BLOCK];
                for (dst, &value) in bytes.iter_mut().zip(&source_values) {
                    *dst = (value >> shift) as u8;
                }
                // SAFETY: this function runs only with the caller-detected NEON feature.
                let lo_table = unsafe { nibble_table([bytes[0], bytes[1], bytes[2], bytes[3]]) };
                let hi_table = unsafe { nibble_table([bytes[4], bytes[5], bytes[6], bytes[7]]) };
                for pair in 0..host_bytes / 2 {
                    let low = vqtbl1q_u8(lo_table, low_selectors[pair]);
                    let high = vqtbl1q_u8(hi_table, high_selectors[pair]);
                    accumulators[pair][source_byte] =
                        veorq_u8(accumulators[pair][source_byte], veorq_u8(low, high));
                }
            }
        }
    }

    let mut output = [0u128; 192];
    for (pair, source_columns) in accumulators[..host_bytes / 2].iter().enumerate() {
        for (source_byte, &column) in source_columns[..source_bytes].iter().enumerate() {
            let mut bytes = [0u8; 16];
            // SAFETY: `bytes` contains a complete 16-byte vector destination.
            unsafe { vst1q_u8(bytes.as_mut_ptr(), column) };
            for (row, &value) in bytes.iter().enumerate() {
                output[16 * pair + row] ^= u128::from(value) << (8 * source_byte);
            }
        }
    }
    output
}
