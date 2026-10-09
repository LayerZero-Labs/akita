//! Eight 81-bit pieces gathered with an eight-by-eight delta-swap transpose.

pub(super) fn indices(bits: &[u64], out: &mut [u8; 81]) {
    for block in 0..11 {
        let mut square = 0u64;
        for j in 0..8 {
            let position = 81 * j + 8 * block;
            // Last block has only bit 80. Masking avoids reading bit 648.
            let byte = if block == 10 {
                (bits[position / 64] >> (position % 64)) & 1
            } else {
                let word = position / 64;
                let shift = position % 64;
                let low = bits[word] >> shift;
                let high = if shift > 56 {
                    bits[word + 1] << (64 - shift)
                } else {
                    0
                };
                (low | high) & 255
            };
            square |= byte << (8 * j);
        }
        let t = (square ^ (square >> 7)) & 0x00aa_00aa_00aa_00aa;
        square ^= t ^ (t << 7);
        let t = (square ^ (square >> 14)) & 0x0000_cccc_0000_cccc;
        square ^= t ^ (t << 14);
        let t = (square ^ (square >> 28)) & 0x0000_0000_f0f0_f0f0;
        square ^= t ^ (t << 28);
        let count = if block == 10 { 1 } else { 8 };
        for i in 0..count {
            out[8 * block + i] = (square >> (8 * i)) as u8;
        }
    }
}
