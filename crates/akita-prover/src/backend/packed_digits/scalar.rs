//! Canonical scalar signed-digit codec.

use super::DIGITS_PER_BLOCK;

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
    if bit_width == 6 && source.len().is_multiple_of(4) {
        encode_l6(source, output);
        return;
    }
    if bit_width == 7 && source.len().is_multiple_of(8) {
        encode_l7(source, output);
        return;
    }
    let mask = bit_mask(bit_width);
    let bytes_per_group = usize::from(bit_width);
    for (source, encoded) in source.chunks(8).zip(output.chunks_mut(bytes_per_group)) {
        let mut word = 0u64;
        for (index, &digit) in source.iter().enumerate() {
            word |= u64::from((digit as u8) & mask) << (index * usize::from(bit_width));
        }
        let live_bytes = (source.len() * usize::from(bit_width)).div_ceil(8);
        encoded[..live_bytes].copy_from_slice(&word.to_le_bytes()[..live_bytes]);
    }
}

#[inline(always)]
fn encode_l6(source: &[i8], output: &mut [u8]) {
    let mut encoded = output.chunks_exact_mut(3);
    for digits in source.chunks_exact(4) {
        let dst = encoded
            .next()
            .expect("six-bit output has three bytes per four digits");
        let d0 = (digits[0] as u8) & 0x3f;
        let d1 = (digits[1] as u8) & 0x3f;
        let d2 = (digits[2] as u8) & 0x3f;
        let d3 = (digits[3] as u8) & 0x3f;
        dst[0] = d0 | (d1 << 6);
        dst[1] = (d1 >> 2) | (d2 << 4);
        dst[2] = (d2 >> 4) | (d3 << 2);
    }
    debug_assert!(source.len().is_multiple_of(4));
    debug_assert!(encoded.into_remainder().is_empty());
}

#[inline(always)]
fn encode_l7(source: &[i8], output: &mut [u8]) {
    let mut encoded = output.chunks_exact_mut(7);
    for digits in source.chunks_exact(8) {
        let dst = encoded
            .next()
            .expect("seven-bit output has seven bytes per eight digits");
        let d0 = (digits[0] as u8) & 0x7f;
        let d1 = (digits[1] as u8) & 0x7f;
        let d2 = (digits[2] as u8) & 0x7f;
        let d3 = (digits[3] as u8) & 0x7f;
        let d4 = (digits[4] as u8) & 0x7f;
        let d5 = (digits[5] as u8) & 0x7f;
        let d6 = (digits[6] as u8) & 0x7f;
        let d7 = (digits[7] as u8) & 0x7f;
        dst[0] = d0 | (d1 << 7);
        dst[1] = (d1 >> 1) | (d2 << 6);
        dst[2] = (d2 >> 2) | (d3 << 5);
        dst[3] = (d3 >> 3) | (d4 << 4);
        dst[4] = (d4 >> 4) | (d5 << 3);
        dst[5] = (d5 >> 5) | (d6 << 2);
        dst[6] = (d6 >> 6) | (d7 << 1);
    }
    debug_assert!(source.len().is_multiple_of(8));
    debug_assert!(encoded.into_remainder().is_empty());
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
