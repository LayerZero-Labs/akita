//! Canonical scalar packed-digit codec.

use super::DIGITS_PER_BLOCK;

#[inline]
pub(super) fn encode_at(storage: &mut [u8], index: usize, bit_width: u8, digit: i8) {
    let bit_offset = index * usize::from(bit_width);
    let byte_offset = bit_offset / 8;
    let shift = bit_offset % 8;
    let mask = bit_mask(bit_width);
    let raw = (digit as u8) & mask;
    storage[byte_offset] |= raw << shift;
    if shift + usize::from(bit_width) > 8 {
        storage[byte_offset + 1] |= raw >> (8 - shift);
    }
}

#[inline]
pub(super) fn decode_at(storage: &[u8], index: usize, bit_width: u8) -> i8 {
    let bit_offset = index * usize::from(bit_width);
    let byte_offset = bit_offset / 8;
    let shift = bit_offset % 8;
    let word = u16::from(storage[byte_offset]) | (u16::from(storage[byte_offset + 1]) << 8);
    let raw = ((word >> shift) as u8) & bit_mask(bit_width);
    sign_extend(raw, bit_width)
}

pub(super) fn decode_full_block(
    encoded: &[u8],
    bit_width: u8,
    output: &mut [i8; DIGITS_PER_BLOCK],
) {
    for (index, slot) in output.iter_mut().enumerate() {
        *slot = decode_at(encoded, index, bit_width);
    }
}

#[inline]
const fn bit_mask(bit_width: u8) -> u8 {
    if bit_width == 8 {
        u8::MAX
    } else {
        (1u8 << bit_width) - 1
    }
}

#[inline]
const fn sign_extend(raw: u8, bit_width: u8) -> i8 {
    if bit_width == 8 {
        return raw as i8;
    }
    let sign = 1u8 << (bit_width - 1);
    ((raw ^ sign).wrapping_sub(sign)) as i8
}
