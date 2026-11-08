//! Eight 81-bit pieces gathered with an eight-by-eight delta-swap transpose.

#[inline(always)]
pub(super) fn indices(bits: &[u64; 11], out: &mut [u8; 81]) {
    // Fixed column offsets make all word indices, shifts and cross-word cases
    // compile-time constants, including on the default x86_64 target.
    out[0..8].copy_from_slice(&column::<0>(bits).to_le_bytes());
    out[8..16].copy_from_slice(&column::<1>(bits).to_le_bytes());
    out[16..24].copy_from_slice(&column::<2>(bits).to_le_bytes());
    out[24..32].copy_from_slice(&column::<3>(bits).to_le_bytes());
    out[32..40].copy_from_slice(&column::<4>(bits).to_le_bytes());
    out[40..48].copy_from_slice(&column::<5>(bits).to_le_bytes());
    out[48..56].copy_from_slice(&column::<6>(bits).to_le_bytes());
    out[56..64].copy_from_slice(&column::<7>(bits).to_le_bytes());
    out[64..72].copy_from_slice(&column::<8>(bits).to_le_bytes());
    out[72..80].copy_from_slice(&column::<9>(bits).to_le_bytes());
    // The final column has only bit 80 of each piece. Its positions are
    // 80+81*j; moving each directly to bit j avoids a mostly empty transpose.
    out[80] = (((bits[1] >> 16) & 1)
        | ((bits[2] >> 32) & 2)
        | ((bits[3] >> 48) & 4)
        | (bits[5] & 8)
        | ((bits[6] >> 16) & 16)
        | ((bits[7] >> 32) & 32)
        | ((bits[8] >> 48) & 64)
        | (bits[10] & 128)) as u8;
}

#[inline(always)]
fn column<const COLUMN: usize>(bits: &[u64; 11]) -> u64 {
    let mut square = byte(bits, 8 * COLUMN)
        | (byte(bits, 81 + 8 * COLUMN) << 8)
        | (byte(bits, 162 + 8 * COLUMN) << 16)
        | (byte(bits, 243 + 8 * COLUMN) << 24)
        | (byte(bits, 324 + 8 * COLUMN) << 32)
        | (byte(bits, 405 + 8 * COLUMN) << 40)
        | (byte(bits, 486 + 8 * COLUMN) << 48)
        | (byte(bits, 567 + 8 * COLUMN) << 56);
    let t = (square ^ (square >> 7)) & 0x00aa_00aa_00aa_00aa;
    square ^= t ^ (t << 7);
    let t = (square ^ (square >> 14)) & 0x0000_cccc_0000_cccc;
    square ^= t ^ (t << 14);
    let t = (square ^ (square >> 28)) & 0x0000_0000_f0f0_f0f0;
    square ^ t ^ (t << 28)
}

#[inline(always)]
fn byte(bits: &[u64; 11], position: usize) -> u64 {
    let word = position / 64;
    let shift = position % 64;
    let low = bits[word] >> shift;
    let high = if shift > 56 {
        bits[word + 1] << (64 - shift)
    } else {
        0
    };
    (low | high) & 255
}
