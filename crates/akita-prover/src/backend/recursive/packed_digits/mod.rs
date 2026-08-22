//! Exact packed storage for bounded signed recursive-witness digits.

mod scalar;

#[cfg(target_arch = "aarch64")]
mod aarch64;
#[cfg(target_arch = "x86_64")]
mod x86_64;

use std::sync::Arc;

use akita_error::{checked, AkitaError};

const DIGITS_PER_BLOCK: usize = 64;
const VECTOR_LOAD_PADDING: usize = 16;

/// Exact signed extrema observed while packing a digit buffer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct SignedDigitBounds {
    negative_abs_max: u8,
    positive_max: u8,
}

impl SignedDigitBounds {
    pub(crate) fn negative_abs_max(self) -> u8 {
        self.negative_abs_max
    }

    pub(crate) fn positive_max(self) -> u8 {
        self.positive_max
    }
}

/// Immutable exact-width two's-complement packed signed digits.
///
/// Every group of 64 digits starts on a byte boundary because a block occupies
/// exactly `8 * bit_width` bytes. The zero suffix belongs to storage safety,
/// not to the encoded payload: architecture decoders may issue bounded word
/// vector loads that extend past the final payload byte.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct PackedSignedDigits {
    storage: Arc<[u8]>,
    encoded_len: usize,
    len: usize,
    bit_width: u8,
    bounds: SignedDigitBounds,
}

impl PackedSignedDigits {
    pub(crate) fn from_i8_digits(digits: Vec<i8>, bit_width: u8) -> Result<Self, AkitaError> {
        validate_bit_width(bit_width)?;
        let encoded_len = encoded_byte_len(digits.len(), bit_width)?;
        let storage_len = checked::sum([encoded_len, VECTOR_LOAD_PADDING]).ok_or_else(|| {
            AkitaError::InvalidInput("packed signed-digit storage length overflow".into())
        })?;
        let mut storage = vec![0u8; storage_len];
        let mut negative_abs_max = 0u8;
        let mut positive_max = 0u8;

        for (index, &digit) in digits.iter().enumerate() {
            validate_digit(digit, bit_width)?;
            if digit < 0 {
                negative_abs_max = negative_abs_max.max(digit.unsigned_abs());
            } else {
                positive_max = positive_max.max(digit as u8);
            }
            scalar::encode_at(&mut storage, index, bit_width, digit);
        }

        Ok(Self {
            storage: storage.into(),
            encoded_len,
            len: digits.len(),
            bit_width,
            bounds: SignedDigitBounds {
                negative_abs_max,
                positive_max,
            },
        })
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub(crate) fn bit_width(&self) -> u8 {
        self.bit_width
    }

    pub(crate) fn bounds(&self) -> SignedDigitBounds {
        self.bounds
    }

    pub(crate) fn encoded_bytes(&self) -> &[u8] {
        &self.storage[..self.encoded_len]
    }

    pub(crate) fn get(&self, index: usize) -> Option<i8> {
        (index < self.len).then(|| scalar::decode_at(&self.storage, index, self.bit_width))
    }

    pub(crate) fn zero_padded(&self, len: usize) -> Result<PackedSignedDigitView<'_>, AkitaError> {
        if len < self.len {
            return Err(AkitaError::InvalidSize {
                expected: self.len,
                actual: len,
            });
        }
        Ok(PackedSignedDigitView { digits: self, len })
    }

    pub(crate) fn decode(&self) -> Vec<i8> {
        let mut decoded = vec![0i8; self.len];
        self.decode_into(&mut decoded)
            .expect("fresh output has the exact packed digit length");
        decoded
    }

    pub(crate) fn decode_into(&self, output: &mut [i8]) -> Result<(), AkitaError> {
        if output.len() != self.len {
            return Err(AkitaError::InvalidSize {
                expected: self.len,
                actual: output.len(),
            });
        }
        decode_prefix(self, output);
        Ok(())
    }
}

/// A logical zero-padded view without a second allocation of the witness.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PackedSignedDigitView<'a> {
    digits: &'a PackedSignedDigits,
    len: usize,
}

impl PackedSignedDigitView<'_> {
    pub(crate) fn len(self) -> usize {
        self.len
    }

    pub(crate) fn block_count(self) -> usize {
        self.len.div_ceil(DIGITS_PER_BLOCK)
    }

    pub(crate) fn decode_block(
        self,
        block_index: usize,
        output: &mut [i8; DIGITS_PER_BLOCK],
    ) -> Result<usize, AkitaError> {
        let start = checked::product([block_index, DIGITS_PER_BLOCK]).ok_or_else(|| {
            AkitaError::InvalidInput("packed signed-digit block offset overflow".into())
        })?;
        if start >= self.len {
            return Err(AkitaError::InvalidSize {
                expected: self.block_count(),
                actual: block_index,
            });
        }

        output.fill(0);
        let logical_end = self.digits.len.min(self.len);
        if start >= logical_end {
            return Ok(0);
        }
        let live = (logical_end - start).min(DIGITS_PER_BLOCK);
        if live == DIGITS_PER_BLOCK {
            decode_full_block(self.digits, block_index, output);
        } else {
            for (offset, slot) in output.iter_mut().take(live).enumerate() {
                *slot =
                    scalar::decode_at(&self.digits.storage, start + offset, self.digits.bit_width);
            }
        }
        Ok(live)
    }
}

fn decode_prefix(digits: &PackedSignedDigits, output: &mut [i8]) {
    let full_blocks = output.len() / DIGITS_PER_BLOCK;
    for (block_index, block) in output
        .chunks_exact_mut(DIGITS_PER_BLOCK)
        .take(full_blocks)
        .enumerate()
    {
        let block: &mut [i8; DIGITS_PER_BLOCK] = block.try_into().expect("exact chunk length");
        decode_full_block(digits, block_index, block);
    }
    for (index, slot) in output
        .iter_mut()
        .enumerate()
        .skip(full_blocks * DIGITS_PER_BLOCK)
    {
        *slot = scalar::decode_at(&digits.storage, index, digits.bit_width);
    }
}

#[inline]
fn decode_full_block(
    digits: &PackedSignedDigits,
    block_index: usize,
    output: &mut [i8; DIGITS_PER_BLOCK],
) {
    let byte_offset = block_index * usize::from(digits.bit_width) * 8;
    let encoded = &digits.storage[byte_offset..];
    debug_assert!(encoded.len() >= usize::from(digits.bit_width) * 8 + VECTOR_LOAD_PADDING);

    #[cfg(target_arch = "x86_64")]
    if x86_64::try_decode_full_block(encoded, digits.bit_width, output) {
        return;
    }
    #[cfg(target_arch = "aarch64")]
    {
        // NEON is part of the baseline AArch64 architecture.
        unsafe { aarch64::decode_full_block(encoded, digits.bit_width, output) };
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    scalar::decode_full_block(encoded, digits.bit_width, output);

    #[cfg(target_arch = "x86_64")]
    scalar::decode_full_block(encoded, digits.bit_width, output);
}

fn encoded_byte_len(len: usize, bit_width: u8) -> Result<usize, AkitaError> {
    let bit_len = checked::product([len, usize::from(bit_width)]).ok_or_else(|| {
        AkitaError::InvalidInput("packed signed-digit bit length overflow".into())
    })?;
    checked::div_ceil(bit_len, 8)
        .ok_or_else(|| AkitaError::InvalidInput("invalid packed signed-digit width".into()))
}

fn validate_bit_width(bit_width: u8) -> Result<(), AkitaError> {
    if !(1..=8).contains(&bit_width) {
        return Err(AkitaError::InvalidInput(format!(
            "packed signed-digit width must be in 1..=8, got {bit_width}"
        )));
    }
    Ok(())
}

fn validate_digit(digit: i8, bit_width: u8) -> Result<(), AkitaError> {
    let half = 1i16 << (bit_width - 1);
    let digit = i16::from(digit);
    if (-half..half).contains(&digit) {
        return Ok(());
    }
    Err(AkitaError::InvalidInput(format!(
        "digit {digit} does not fit signed {bit_width}-bit storage"
    )))
}

#[cfg(test)]
mod tests;
