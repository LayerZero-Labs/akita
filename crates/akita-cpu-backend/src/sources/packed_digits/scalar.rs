//! Canonical scalar signed-digit codec.

use super::DIGITS_PER_BLOCK;

/// Byte offset and in-byte shift of digit `index`.
///
/// Splits the index by eight before scaling, like `encoded_byte_len`: the
/// byte offset is at most `index` for every supported width, so an absolute
/// index near `usize::MAX` cannot overflow the way `index * bit_width` does.
#[inline]
pub(super) fn bit_position(index: usize, bit_width: u8) -> (usize, usize) {
    let width = usize::from(bit_width);
    let tail_bits = index % 8 * width;
    (index / 8 * width + tail_bits / 8, tail_bits % 8)
}

#[inline]
pub(super) fn encode_at(storage: &mut [u8], index: usize, bit_width: u8, digit: i8) {
    let (byte_offset, shift) = bit_position(index, bit_width);
    let mask = bit_mask(bit_width);
    let raw = (digit as u8) & mask;
    storage[byte_offset] |= raw << shift;
    if shift + usize::from(bit_width) > 8 {
        storage[byte_offset + 1] |= raw >> (8 - shift);
    }
}

/// Encode one byte-aligned block of at most 64 digits.
///
/// Sixty-four digits always occupy an integral number of bytes for every
/// supported width, so callers can encode separate blocks concurrently.
#[inline]
pub(super) fn encode_block(source: &[i8], bit_width: u8, output: &mut [u8]) {
    debug_assert!(source.len() <= DIGITS_PER_BLOCK);
    debug_assert_eq!(
        output.len(),
        (source.len() * usize::from(bit_width)).div_ceil(8)
    );
    if bit_width == 8 {
        for (slot, &digit) in output.iter_mut().zip(source) {
            *slot = digit as u8;
        }
        return;
    }
    for (index, &digit) in source.iter().enumerate() {
        encode_at(output, index, bit_width, digit);
    }
}

#[inline]
pub(super) fn decode_at(storage: &[u8], index: usize, bit_width: u8) -> i8 {
    let (byte_offset, shift) = bit_position(index, bit_width);
    let word = u16::from(storage[byte_offset]) | (u16::from(storage[byte_offset + 1]) << 8);
    let raw = ((word >> shift) as u8) & bit_mask(bit_width);
    sign_extend(raw, bit_width)
}

#[inline]
pub(super) fn decode_at_zero_padded(storage: &[u8], index: usize, bit_width: u8) -> i8 {
    let (byte_offset, shift) = bit_position(index, bit_width);
    let low = storage.get(byte_offset).copied().unwrap_or(0);
    let high = storage.get(byte_offset + 1).copied().unwrap_or(0);
    let word = u16::from(low) | (u16::from(high) << 8);
    let raw = ((word >> shift) as u8) & bit_mask(bit_width);
    sign_extend(raw, bit_width)
}

#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
pub(super) fn decode_full_block(
    encoded: &[u8],
    bit_width: u8,
    output: &mut [i8; DIGITS_PER_BLOCK],
) {
    for (index, slot) in output.iter_mut().enumerate() {
        *slot = decode_at(encoded, index, bit_width);
    }
}

pub(super) fn decode_full_block_zero_padded(
    encoded: &[u8],
    bit_width: u8,
    output: &mut [i8; DIGITS_PER_BLOCK],
) {
    for (index, slot) in output.iter_mut().enumerate() {
        *slot = decode_at_zero_padded(encoded, index, bit_width);
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
