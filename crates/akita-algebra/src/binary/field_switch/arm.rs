//! AArch64 NEON bit-matrix kernels for the binary field switch.

use std::arch::aarch64::{
    vandq_u8, vdupq_n_u8, veorq_u8, vld1q_u8, vqtbl1q_u8, vshrq_n_u8, vst1q_u8,
};

use super::{SwitchField, F};

const TILE: usize = 64;
const LANES: usize = 16;
const MAX_HOST_BYTES: usize = 192 / 8;
const OUTPUT_BYTES: usize = 21;
const LIMB_BYTES: [usize; 3] = [8, 8, 5];

/// Map host equality weights through F162 row weights into three SoA limbs.
///
/// Each low or high nibble selects one of 16 precomputed images. NEON table
/// lookup applies the same binary-linear map to 16 independent host values.
///
/// # Safety
///
/// The caller must establish NEON support. The input length must be divisible
/// by 64, all output slices must match it, and the row weights must have been
/// validated for the sealed profile (including zero padding).
#[target_feature(enable = "neon")]
pub(super) unsafe fn coefficients<H: SwitchField>(
    weights: &[H],
    rows: &[F; 256],
    output: [&mut [u64]; 3],
) {
    debug_assert_eq!(weights.len() % TILE, 0);
    debug_assert!(output.iter().all(|limb| limb.len() == weights.len()));
    let host_bytes = H::ROWS / 8;
    debug_assert!(host_bytes <= MAX_HOST_BYTES);

    // The table is independent of domain size and reused across all tiles.
    // `tables[host_byte][output_byte][nibble][selection]` is the image byte.
    let mut tables = [[[[0u8; 16]; 2]; OUTPUT_BYTES]; MAX_HOST_BYTES];
    for (host_byte, byte_tables) in tables[..host_bytes].iter_mut().enumerate() {
        for (output_byte, nibble_tables) in byte_tables.iter_mut().enumerate() {
            let mut basis = [0u8; 8];
            for (input_bit, basis_byte) in basis.iter_mut().enumerate() {
                let words = rows[8 * host_byte + input_bit].to_words();
                *basis_byte = (words[output_byte / 8] >> (8 * (output_byte % 8))) as u8;
            }
            for (nibble, table) in nibble_tables.iter_mut().enumerate() {
                for selection in 1usize..16 {
                    let bit = selection.trailing_zeros() as usize;
                    table[selection] = table[selection & (selection - 1)] ^ basis[4 * nibble + bit];
                }
            }
        }
    }

    let [low, high, top] = output;
    for tile in (0..weights.len()).step_by(TILE) {
        for block in (0..TILE).step_by(LANES) {
            let mut host_planes = [[0u8; LANES]; MAX_HOST_BYTES];
            for (lane, &host) in weights[tile + block..tile + block + LANES]
                .iter()
                .enumerate()
            {
                let words = host.coordinates();
                for (host_byte, plane) in host_planes[..host_bytes].iter_mut().enumerate() {
                    plane[lane] = (words[host_byte / 8] >> (8 * (host_byte % 8))) as u8;
                }
            }

            let mut mapped_bytes = [[0u8; LANES]; OUTPUT_BYTES];
            for (output_byte, result) in mapped_bytes.iter_mut().enumerate() {
                let mut mapped = vdupq_n_u8(0);
                for (host_byte, plane) in host_planes[..host_bytes].iter().enumerate() {
                    // SAFETY: every plane and nibble table contains 16 bytes.
                    let input = unsafe { vld1q_u8(plane.as_ptr()) };
                    let lo = vandq_u8(input, vdupq_n_u8(15));
                    let hi = vshrq_n_u8::<4>(input);
                    let tables = &tables[host_byte][output_byte];
                    let lo_table = unsafe { vld1q_u8(tables[0].as_ptr()) };
                    let hi_table = unsafe { vld1q_u8(tables[1].as_ptr()) };
                    mapped = veorq_u8(mapped, vqtbl1q_u8(lo_table, lo));
                    mapped = veorq_u8(mapped, vqtbl1q_u8(hi_table, hi));
                }
                // SAFETY: the result has exactly 16 bytes.
                unsafe { vst1q_u8(result.as_mut_ptr(), mapped) };
            }

            let range = tile + block..tile + block + LANES;
            for (lane, ((lo, hi), top_word)) in low[range.clone()]
                .iter_mut()
                .zip(high[range.clone()].iter_mut())
                .zip(top[range].iter_mut())
                .enumerate()
            {
                let mut limbs = [0u64; 3];
                for limb in 0..3 {
                    for byte in 0..LIMB_BYTES[limb] {
                        limbs[limb] |= u64::from(mapped_bytes[8 * limb + byte][lane]) << (8 * byte);
                    }
                }
                *lo = limbs[0];
                *hi = limbs[1];
                *top_word = limbs[2];
            }
        }
    }
}
