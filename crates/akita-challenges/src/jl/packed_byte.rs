//! Shared packed-byte decode tables for binary-sign JL rows.

const SIGN_LUT_I8: [i8; 2] = [-1, 1];

/// Map a packed bit to its binary sign: `0 -> -1`, `1 -> +1`.
#[inline]
pub(crate) fn bit_to_sign(bit: u8) -> i8 {
    SIGN_LUT_I8[(bit & 1) as usize]
}

const fn signs_for_byte(byte: u8) -> [i8; 8] {
    [
        SIGN_LUT_I8[(byte & 1) as usize],
        SIGN_LUT_I8[((byte >> 1) & 1) as usize],
        SIGN_LUT_I8[((byte >> 2) & 1) as usize],
        SIGN_LUT_I8[((byte >> 3) & 1) as usize],
        SIGN_LUT_I8[((byte >> 4) & 1) as usize],
        SIGN_LUT_I8[((byte >> 5) & 1) as usize],
        SIGN_LUT_I8[((byte >> 6) & 1) as usize],
        SIGN_LUT_I8[((byte >> 7) & 1) as usize],
    ]
}

const fn build_signs_for_byte_lut() -> [[i8; 8]; 256] {
    let mut lut = [[0i8; 8]; 256];
    let mut byte = 0u8;
    loop {
        lut[byte as usize] = signs_for_byte(byte);
        if byte == 255 {
            break;
        }
        byte += 1;
    }
    lut
}

/// Pre-decoded binary signs for every packed row byte (`256 × 8` `i8`s).
pub(crate) static BINARY_SIGNS_FOR_BYTE: [[i8; 8]; 256] = build_signs_for_byte_lut();

/// Binary sign for a packed bit as `i32` (scalar remainder tail).
pub(crate) const SIGN_LUT_I32: [i32; 2] = [-1, 1];

const fn bit_lanes_for_byte(byte: u8) -> ([u8; 8], usize) {
    let mut lanes = [0u8; 8];
    let mut count = 0usize;
    let mut lane = 0u8;
    while lane < 8 {
        if ((byte >> lane) & 1) != 0 {
            lanes[count] = lane;
            count += 1;
        }
        lane += 1;
    }
    (lanes, count)
}

const fn build_bit_lanes_for_byte() -> [([u8; 8], usize); 256] {
    let mut out = [([0u8; 8], 0usize); 256];
    let mut byte = 0u16;
    while byte < 256 {
        out[byte as usize] = bit_lanes_for_byte(byte as u8);
        byte += 1;
    }
    out
}

/// Set bit positions for each packed matrix byte (scatter fast path).
pub(crate) static BIT_LANES_FOR_BYTE: [([u8; 8], usize); 256] = build_bit_lanes_for_byte();
