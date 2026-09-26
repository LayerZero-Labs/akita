//! AArch64 NEON bit-matrix kernels for the binary field switch.

use std::arch::aarch64::{
    uint64x2_t, uint8x16_t, vandq_u8, vdupq_n_u8, veorq_u8, vld1q_u8, vqtbl1q_u8,
    vreinterpretq_u16_u8, vreinterpretq_u32_u16, vreinterpretq_u64_u32, vshrq_n_u8, vst1q_u64,
    vzip1q_u16, vzip1q_u32, vzip1q_u8, vzip2q_u16, vzip2q_u32, vzip2q_u8,
};

use super::{SwitchField, F};

const TILE: usize = 64;
const LANES: usize = 16;
const MAX_HOST_BYTES: usize = 192 / 8;
const OUTPUT_BYTES: usize = 21;
const LIMB_BYTES: [usize; 3] = [8, 8, 5];

/// Transpose eight 16-byte planes into sixteen contiguous little-endian words.
/// Each zip stage doubles the number of adjacent bytes from one output word.
#[inline]
#[target_feature(enable = "neon")]
fn transpose_output_bytes(planes: [uint8x16_t; 8]) -> [uint64x2_t; 8] {
    let pairs = [
        vzip1q_u8(planes[0], planes[1]),
        vzip2q_u8(planes[0], planes[1]),
        vzip1q_u8(planes[2], planes[3]),
        vzip2q_u8(planes[2], planes[3]),
        vzip1q_u8(planes[4], planes[5]),
        vzip2q_u8(planes[4], planes[5]),
        vzip1q_u8(planes[6], planes[7]),
        vzip2q_u8(planes[6], planes[7]),
    ];
    let quads = [
        vzip1q_u16(
            vreinterpretq_u16_u8(pairs[0]),
            vreinterpretq_u16_u8(pairs[2]),
        ),
        vzip2q_u16(
            vreinterpretq_u16_u8(pairs[0]),
            vreinterpretq_u16_u8(pairs[2]),
        ),
        vzip1q_u16(
            vreinterpretq_u16_u8(pairs[1]),
            vreinterpretq_u16_u8(pairs[3]),
        ),
        vzip2q_u16(
            vreinterpretq_u16_u8(pairs[1]),
            vreinterpretq_u16_u8(pairs[3]),
        ),
        vzip1q_u16(
            vreinterpretq_u16_u8(pairs[4]),
            vreinterpretq_u16_u8(pairs[6]),
        ),
        vzip2q_u16(
            vreinterpretq_u16_u8(pairs[4]),
            vreinterpretq_u16_u8(pairs[6]),
        ),
        vzip1q_u16(
            vreinterpretq_u16_u8(pairs[5]),
            vreinterpretq_u16_u8(pairs[7]),
        ),
        vzip2q_u16(
            vreinterpretq_u16_u8(pairs[5]),
            vreinterpretq_u16_u8(pairs[7]),
        ),
    ];
    let quad = |index| vreinterpretq_u32_u16(quads[index]);
    [
        vreinterpretq_u64_u32(vzip1q_u32(quad(0), quad(4))),
        vreinterpretq_u64_u32(vzip2q_u32(quad(0), quad(4))),
        vreinterpretq_u64_u32(vzip1q_u32(quad(1), quad(5))),
        vreinterpretq_u64_u32(vzip2q_u32(quad(1), quad(5))),
        vreinterpretq_u64_u32(vzip1q_u32(quad(2), quad(6))),
        vreinterpretq_u64_u32(vzip2q_u32(quad(2), quad(6))),
        vreinterpretq_u64_u32(vzip1q_u32(quad(3), quad(7))),
        vreinterpretq_u64_u32(vzip2q_u32(quad(3), quad(7))),
    ]
}

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
    let zero = vdupq_n_u8(0);
    let mut tables = [[[zero; 2]; OUTPUT_BYTES]; MAX_HOST_BYTES];
    for (host_byte, byte_tables) in tables[..host_bytes].iter_mut().enumerate() {
        for (output_byte, nibble_tables) in byte_tables.iter_mut().enumerate() {
            let mut basis = [0u8; 8];
            for (input_bit, basis_byte) in basis.iter_mut().enumerate() {
                let words = rows[8 * host_byte + input_bit].to_words();
                *basis_byte = (words[output_byte / 8] >> (8 * (output_byte % 8))) as u8;
            }
            for (nibble, table) in nibble_tables.iter_mut().enumerate() {
                let mut entries = [0u8; 16];
                for selection in 1usize..16 {
                    let bit = selection.trailing_zeros() as usize;
                    entries[selection] =
                        entries[selection & (selection - 1)] ^ basis[4 * nibble + bit];
                }
                // SAFETY: `entries` contains exactly one full table vector.
                *table = unsafe { vld1q_u8(entries.as_ptr()) };
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

            let mut low_nibbles = [zero; MAX_HOST_BYTES];
            let mut high_nibbles = [zero; MAX_HOST_BYTES];
            for (host_byte, plane) in host_planes[..host_bytes].iter().enumerate() {
                // SAFETY: each input plane contains exactly 16 bytes.
                let input = unsafe { vld1q_u8(plane.as_ptr()) };
                low_nibbles[host_byte] = vandq_u8(input, vdupq_n_u8(15));
                high_nibbles[host_byte] = vshrq_n_u8::<4>(input);
            }

            for (limb, words) in [&mut *low, &mut *high, &mut *top].into_iter().enumerate() {
                let mut mapped_bytes = [zero; 8];
                for host_byte in 0..host_bytes {
                    let low_nibble = low_nibbles[host_byte];
                    let high_nibble = high_nibbles[host_byte];
                    for (byte, mapped) in mapped_bytes[..LIMB_BYTES[limb]].iter_mut().enumerate() {
                        let output_byte = 8 * limb + byte;
                        let [lo_table, hi_table] = tables[host_byte][output_byte];
                        *mapped = veorq_u8(*mapped, vqtbl1q_u8(lo_table, low_nibble));
                        *mapped = veorq_u8(*mapped, vqtbl1q_u8(hi_table, high_nibble));
                    }
                }
                // The last three top-limb planes remain zero, so every word
                // has canonical F162 padding without a separate mask/store.
                let packed = transpose_output_bytes(mapped_bytes);
                for (group, &pair) in packed.iter().enumerate() {
                    // SAFETY: one vector stores two complete output values.
                    unsafe { vst1q_u64(words.as_mut_ptr().add(tile + block + 2 * group), pair) };
                }
            }
        }
    }
}
