//! Scalar layout shared by the 256-bit and 512-bit GFNI kernels.

use super::F;

pub(super) const TILE: usize = 64;
pub(super) const MAX_HOST_BYTES: usize = 192 / 8;
pub(super) const MAX_SOURCE_BYTES: usize = 128 / 8;
pub(super) const LIMB_BYTES: [usize; 3] = [8, 8, 5];

/// Transpose one 8-by-8 bit matrix stored as eight row bytes.
#[inline(always)]
pub(super) fn transpose8(mut value: u64) -> u64 {
    let mut swap = (value ^ (value >> 7)) & 0x00aa_00aa_00aa_00aa;
    value ^= swap ^ (swap << 7);
    swap = (value ^ (value >> 14)) & 0x0000_cccc_0000_cccc;
    value ^= swap ^ (swap << 14);
    swap = (value ^ (value >> 28)) & 0x0000_0000_f0f0_f0f0;
    value ^ swap ^ (swap << 28)
}

/// Build the GFNI matrix mapping one host-coordinate byte into one output byte.
pub(super) fn coefficient_matrix(
    host_byte: usize,
    output_bit_offset: usize,
    rows: &[F; 256],
) -> u64 {
    let mut matrix = 0u64;
    for input_bit in 0..8 {
        let words = rows[8 * host_byte + input_bit].to_words();
        for output_bit in 0..8 {
            let bit_index = output_bit_offset + output_bit;
            let bit = (words[bit_index / 64] >> (bit_index % 64)) & 1;
            // GFNI computes result bit i from matrix byte 7-i.
            matrix |= bit << (8 * (7 - output_bit) + input_bit);
        }
    }
    matrix
}
