//! Exact packed storage for bounded signed prover digits.

mod scalar;

#[cfg(target_arch = "aarch64")]
mod aarch64;
#[cfg(target_arch = "x86_64")]
mod x86_64;

use std::mem::MaybeUninit;
use std::sync::Arc;
use std::{iter::FusedIterator, ops::Range};

use akita_error::{checked, AkitaError};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

const DIGITS_PER_BLOCK: usize = 64;
const VECTOR_LOAD_PADDING: usize = 16;
/// Avoid Rayon scheduling for the small recursive tails where serial packing
/// is cheaper. Large ring-switch outputs cross this threshold by orders of
/// magnitude and encode independent 64-digit blocks in parallel.
#[cfg(feature = "parallel")]
const PARALLEL_ENCODE_THRESHOLD: usize = 1 << 16;

/// Exact signed extrema observed while packing a digit buffer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct SignedDigitBounds {
    negative_abs_max: u8,
    positive_max: u8,
}

impl SignedDigitBounds {
    const ZERO: Self = Self {
        negative_abs_max: 0,
        positive_max: 0,
    };

    pub(crate) fn negative_abs_max(self) -> u8 {
        self.negative_abs_max
    }

    pub(crate) fn positive_max(self) -> u8 {
        self.positive_max
    }

    /// Whether every observed digit lies in the balanced base-`2^log_basis`
    /// interval `[-2^(log_basis - 1), 2^(log_basis - 1) - 1]`.
    pub(crate) fn fits_balanced_log_basis(self, log_basis: u32) -> bool {
        let Some(abs_bound) = akita_types::balanced_signed_digit_abs_bound(log_basis) else {
            return false;
        };
        u64::from(self.negative_abs_max) <= abs_bound && u64::from(self.positive_max) < abs_bound
    }

    fn include_bounds(&mut self, bounds: Self) {
        self.negative_abs_max = self.negative_abs_max.max(bounds.negative_abs_max);
        self.positive_max = self.positive_max.max(bounds.positive_max);
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

impl Default for PackedSignedDigits {
    fn default() -> Self {
        Self::from_i8_digits(Vec::new(), 1).expect("one-bit empty digit storage is valid")
    }
}

#[cfg(test)]
impl From<Arc<[i8]>> for PackedSignedDigits {
    fn from(digits: Arc<[i8]>) -> Self {
        Self::from_i8_digits_auto(digits.as_ref().to_vec())
    }
}

impl PackedSignedDigits {
    pub(crate) fn from_i8_digits_auto(digits: Vec<i8>) -> Self {
        let bit_width = minimum_signed_bit_width(signed_digit_bounds(&digits));
        Self::from_i8_digits(digits, bit_width)
            .expect("the derived signed width and storage length are valid")
    }

    pub(crate) fn from_i8_digits(digits: Vec<i8>, bit_width: u8) -> Result<Self, AkitaError> {
        let mut writer = PackedSignedDigitWriter::new(digits.len(), bit_width)?;
        writer.write_at(0, &digits)?;
        writer.finish()
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

    pub(crate) fn iter(&self) -> impl ExactSizeIterator<Item = i8> + '_ {
        self.view().iter()
    }

    pub(crate) fn view(&self) -> PackedSignedDigitView<'_> {
        PackedSignedDigitView {
            storage: &self.storage,
            stored_len: self.len,
            bit_width: self.bit_width,
            bounds: self.bounds,
            vector_safe: true,
            start: 0,
            len: self.len,
        }
    }

    pub(crate) fn zero_padded(&self, len: usize) -> Result<PackedSignedDigitView<'_>, AkitaError> {
        if len < self.len {
            return Err(AkitaError::InvalidSize {
                expected: self.len,
                actual: len,
            });
        }
        Ok(PackedSignedDigitView {
            storage: &self.storage,
            stored_len: self.len,
            bit_width: self.bit_width,
            bounds: self.bounds,
            vector_safe: true,
            start: 0,
            len,
        })
    }

    #[cfg(test)]
    pub(crate) fn decode(&self) -> Vec<i8> {
        let mut decoded = vec![0i8; self.len];
        self.decode_into(&mut decoded)
            .expect("fresh output has the exact packed digit length");
        decoded
    }

    #[cfg(test)]
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

/// Builder for a packed digit stream emitted in physical order.
///
/// Writes must be monotonic. Gaps are zeroes, which the zeroed storage already
/// holds, so alignment padding costs nothing. Whole 64-digit blocks encode
/// directly into storage; only the block split by the current position waits
/// in `pending` until a later write or [`Self::finish`] completes it.
pub(crate) struct PackedSignedDigitWriter {
    storage: Arc<[u8]>,
    encoded_len: usize,
    len: usize,
    bit_width: u8,
    position: usize,
    /// Digits of the block containing `position`; those at or past `position`
    /// are zero.
    pending: [i8; DIGITS_PER_BLOCK],
    bounds: SignedDigitBounds,
}

impl PackedSignedDigitWriter {
    pub(crate) fn new(len: usize, bit_width: u8) -> Result<Self, AkitaError> {
        validate_bit_width(bit_width)?;
        let encoded_len = encoded_byte_len(len, bit_width)?;
        let storage_len = checked::sum([encoded_len, VECTOR_LOAD_PADDING]).ok_or_else(|| {
            AkitaError::InvalidInput("packed signed-digit storage length overflow".into())
        })?;
        // Touch every page here: faulting fresh pages from parallel encoders
        // costs far more than one serial fill.
        let mut storage = Arc::<[u8]>::new_uninit_slice(storage_len);
        Arc::get_mut(&mut storage)
            .expect("fresh packed storage is uniquely owned")
            .fill(MaybeUninit::new(0));
        // SAFETY: every slot was initialized immediately above.
        let storage = unsafe { storage.assume_init() };
        Ok(Self {
            storage,
            encoded_len,
            len,
            bit_width,
            position: 0,
            pending: [0; DIGITS_PER_BLOCK],
            bounds: SignedDigitBounds::ZERO,
        })
    }

    pub(crate) fn position(&self) -> usize {
        self.position
    }

    pub(crate) fn write_at(&mut self, start: usize, mut digits: &[i8]) -> Result<(), AkitaError> {
        if start < self.position {
            return Err(AkitaError::InvalidInput(
                "packed signed-digit writes must be in physical order".into(),
            ));
        }
        let end = start.checked_add(digits.len()).ok_or_else(|| {
            AkitaError::InvalidInput("packed signed-digit write length overflow".into())
        })?;
        if end > self.len {
            return Err(AkitaError::InvalidSize {
                expected: self.len,
                actual: end,
            });
        }
        // Skip the gap, completing the split block first if the write starts
        // past it.
        let split = self.position % DIGITS_PER_BLOCK;
        let split_start = self.position - split;
        if split != 0 && start >= split_start + DIGITS_PER_BLOCK {
            self.flush_pending(split_start)?;
        }
        self.position = start;

        let offset = self.position % DIGITS_PER_BLOCK;
        if offset != 0 {
            let take = (DIGITS_PER_BLOCK - offset).min(digits.len());
            self.pending[offset..offset + take].copy_from_slice(&digits[..take]);
            self.position += take;
            digits = &digits[take..];
            if self.position.is_multiple_of(DIGITS_PER_BLOCK) {
                self.flush_pending(self.position - DIGITS_PER_BLOCK)?;
            }
        }
        let whole = digits.len() - digits.len() % DIGITS_PER_BLOCK;
        if whole != 0 {
            let encoded_start = encoded_byte_len(self.position, self.bit_width)?;
            let encoded_end = encoded_start + encoded_byte_len(whole, self.bit_width)?;
            let storage = Arc::get_mut(&mut self.storage)
                .expect("streaming packed storage remains uniquely owned");
            let bounds = encode_digits(
                &digits[..whole],
                self.bit_width,
                storage
                    .get_mut(encoded_start..encoded_end)
                    .ok_or(AkitaError::InvalidProof)?,
            );
            self.bounds.include_bounds(bounds);
            self.position += whole;
            digits = &digits[whole..];
        }
        self.pending[..digits.len()].copy_from_slice(digits);
        self.position += digits.len();
        Ok(())
    }

    pub(crate) fn finish(mut self) -> Result<PackedSignedDigits, AkitaError> {
        let split = self.position % DIGITS_PER_BLOCK;
        if split != 0 {
            self.flush_pending(self.position - split)?;
        }
        validate_bounds(self.bounds, self.bit_width)?;
        Ok(PackedSignedDigits {
            storage: self.storage,
            encoded_len: self.encoded_len,
            len: self.len,
            bit_width: self.bit_width,
            bounds: self.bounds,
        })
    }

    /// Encode the pending block starting at digit `block_start` and clear it.
    fn flush_pending(&mut self, block_start: usize) -> Result<(), AkitaError> {
        let block_len = (self.len - block_start).min(DIGITS_PER_BLOCK);
        let encoded_start = encoded_byte_len(block_start, self.bit_width)?;
        let encoded_end = encoded_start + encoded_byte_len(block_len, self.bit_width)?;
        let storage = Arc::get_mut(&mut self.storage)
            .expect("streaming packed storage remains uniquely owned");
        let bounds = encode_digits(
            &self.pending[..block_len],
            self.bit_width,
            storage
                .get_mut(encoded_start..encoded_end)
                .ok_or(AkitaError::InvalidProof)?,
        );
        self.bounds.include_bounds(bounds);
        self.pending = [0; DIGITS_PER_BLOCK];
        Ok(())
    }
}

impl akita_types::WitnessCoefficientSink for PackedSignedDigitWriter {
    fn write_coefficients(&mut self, start: usize, coefficients: &[i8]) -> Result<(), AkitaError> {
        self.write_at(start, coefficients)
    }
}

/// A logical zero-padded view without a second allocation of the witness.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PackedSignedDigitView<'a> {
    storage: &'a [u8],
    stored_len: usize,
    bit_width: u8,
    bounds: SignedDigitBounds,
    vector_safe: bool,
    start: usize,
    len: usize,
}

impl<'a> PackedSignedDigitView<'a> {
    #[inline]
    fn decode_stored(self, index: usize) -> i8 {
        if self.vector_safe {
            scalar::decode_at(self.storage, index, self.bit_width)
        } else {
            scalar::decode_at_zero_padded(self.storage, index, self.bit_width)
        }
    }

    #[cfg(test)]
    pub(crate) fn from_encoded(
        storage: &'a [u8],
        live_len: usize,
        stored_len: usize,
        physical_len: usize,
        bit_width: u8,
        negative_abs_max: u8,
        positive_max: u8,
    ) -> Result<Self, AkitaError> {
        validate_bit_width(bit_width)?;
        if live_len == 0 || live_len > stored_len || stored_len > physical_len {
            return Err(AkitaError::InvalidInput(
                "packed signed-digit view has inconsistent extents".into(),
            ));
        }
        let stored_bits =
            checked::product([stored_len, usize::from(bit_width)]).ok_or_else(|| {
                AkitaError::InvalidInput("packed signed-digit stored length overflow".into())
            })?;
        let expected_bytes = checked::div_ceil(stored_bits, 8).ok_or_else(|| {
            AkitaError::InvalidInput("packed signed-digit byte length overflow".into())
        })?;
        if storage.len() != expected_bytes {
            return Err(AkitaError::InvalidSize {
                expected: expected_bytes,
                actual: storage.len(),
            });
        }
        let used_final_bits = stored_bits % 8;
        if used_final_bits != 0 {
            let unused_mask = !((1u8 << used_final_bits) - 1);
            if storage.last().is_some_and(|byte| byte & unused_mask != 0) {
                return Err(AkitaError::InvalidInput(
                    "packed signed-digit payload has non-canonical trailing bits".into(),
                ));
            }
        }
        let bounds = SignedDigitBounds {
            negative_abs_max,
            positive_max,
        };
        validate_bounds(bounds, bit_width)?;
        let mut decoded_bounds = SignedDigitBounds {
            negative_abs_max: 0,
            positive_max: 0,
        };
        for index in 0..stored_len {
            let digit = scalar::decode_at_zero_padded(storage, index, bit_width);
            if digit < 0 {
                decoded_bounds.negative_abs_max =
                    decoded_bounds.negative_abs_max.max(digit.unsigned_abs());
            } else {
                decoded_bounds.positive_max = decoded_bounds.positive_max.max(digit as u8);
            }
        }
        if decoded_bounds != bounds {
            return Err(AkitaError::InvalidInput(format!(
                "packed signed-digit bounds [-{}, {}] disagree with decoded stored bounds [-{}, {}]",
                bounds.negative_abs_max,
                bounds.positive_max,
                decoded_bounds.negative_abs_max,
                decoded_bounds.positive_max,
            )));
        }
        Ok(Self {
            storage,
            stored_len,
            bit_width,
            bounds,
            vector_safe: false,
            start: 0,
            len: physical_len,
        })
    }

    pub(crate) fn len(self) -> usize {
        self.len
    }

    #[cfg(test)]
    pub(crate) fn block_count(self) -> usize {
        self.len.div_ceil(DIGITS_PER_BLOCK)
    }

    #[cfg(test)]
    pub(crate) fn uses_vector_safe_storage(self) -> bool {
        self.vector_safe
    }

    pub(crate) fn bounds(self) -> SignedDigitBounds {
        self.bounds
    }

    pub(crate) fn get(self, index: usize) -> Option<i8> {
        if index >= self.len {
            return None;
        }
        let source_index = self.start + index;
        Some(if source_index < self.stored_len {
            self.decode_stored(source_index)
        } else {
            0
        })
    }

    #[inline(always)]
    pub(crate) fn at(self, index: usize) -> i8 {
        self.get(index).expect("packed digit index is in bounds")
    }

    pub(crate) fn slice(self, range: Range<usize>) -> Result<Self, AkitaError> {
        if range.start > range.end || range.end > self.len {
            return Err(AkitaError::InvalidSize {
                expected: self.len,
                actual: range.end,
            });
        }
        Ok(Self {
            storage: self.storage,
            stored_len: self.stored_len,
            bit_width: self.bit_width,
            bounds: self.bounds,
            vector_safe: self.vector_safe,
            start: self.start + range.start,
            len: range.len(),
        })
    }

    pub(crate) fn iter(self) -> PackedSignedDigitIter<'a> {
        PackedSignedDigitIter {
            view: self,
            position: 0,
            decoded_start: 0,
            decoded_len: 0,
            decoded: [0; DIGITS_PER_BLOCK],
        }
    }

    pub(crate) fn decode_array<const N: usize>(self, start: usize) -> Result<[i8; N], AkitaError> {
        let mut output = [0i8; N];
        self.decode_range(start, &mut output)?;
        Ok(output)
    }

    pub(crate) fn decode_rings<const D: usize>(
        self,
        start_ring: usize,
        count: usize,
    ) -> Result<Vec<[i8; D]>, AkitaError> {
        let start = checked::product([start_ring, D]).ok_or_else(|| {
            AkitaError::InvalidInput("packed signed-digit ring offset overflow".into())
        })?;
        let mut output = vec![[0i8; D]; count];
        self.decode_range(start, output.as_flattened_mut())?;
        Ok(output)
    }

    pub(crate) fn decode_range(self, start: usize, output: &mut [i8]) -> Result<usize, AkitaError> {
        let end = start.checked_add(output.len()).ok_or_else(|| {
            AkitaError::InvalidInput("packed signed-digit decode range overflow".into())
        })?;
        if end > self.len {
            return Err(AkitaError::InvalidSize {
                expected: self.len,
                actual: end,
            });
        }

        output.fill(0);
        let source_start = self.start + start;
        let source_end = self.start + end;
        let live_end = self.stored_len.min(source_end);
        if source_start >= live_end {
            return Ok(0);
        }
        let live = live_end - source_start;
        let scalar_prefix =
            (DIGITS_PER_BLOCK - source_start % DIGITS_PER_BLOCK).min(live) % DIGITS_PER_BLOCK;
        for (offset, slot) in output.iter_mut().take(scalar_prefix).enumerate() {
            *slot = self.decode_stored(source_start + offset);
        }
        let block_start = source_start + scalar_prefix;
        let full_blocks = (live - scalar_prefix) / DIGITS_PER_BLOCK;
        for (offset, block) in output[scalar_prefix..]
            .chunks_exact_mut(DIGITS_PER_BLOCK)
            .take(full_blocks)
            .enumerate()
        {
            decode_full_block_view(
                self.storage,
                self.bit_width,
                self.vector_safe,
                block_start / DIGITS_PER_BLOCK + offset,
                block.try_into().expect("exact packed decode block"),
            );
        }
        let decoded = scalar_prefix + full_blocks * DIGITS_PER_BLOCK;
        for (offset, slot) in output
            .iter_mut()
            .skip(decoded)
            .take(live - decoded)
            .enumerate()
        {
            *slot = self.decode_stored(source_start + decoded + offset);
        }
        Ok(live)
    }

    #[cfg(test)]
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

        self.decode_range(start, output)
    }
}

pub(crate) struct PackedSignedDigitIter<'a> {
    view: PackedSignedDigitView<'a>,
    position: usize,
    decoded_start: usize,
    decoded_len: usize,
    decoded: [i8; DIGITS_PER_BLOCK],
}

impl Iterator for PackedSignedDigitIter<'_> {
    type Item = i8;

    fn next(&mut self) -> Option<Self::Item> {
        if self.position >= self.view.len() {
            return None;
        }
        if self.position < self.decoded_start
            || self.position >= self.decoded_start + self.decoded_len
        {
            self.refill();
        }
        let value = self.decoded[self.position - self.decoded_start];
        self.position += 1;
        Some(value)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.view.len() - self.position;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for PackedSignedDigitIter<'_> {}
impl FusedIterator for PackedSignedDigitIter<'_> {}

impl PackedSignedDigitIter<'_> {
    #[inline(always)]
    fn refill(&mut self) {
        self.decoded_start = self.position;
        let source_position = self.view.start + self.position;
        let until_aligned =
            (DIGITS_PER_BLOCK - source_position % DIGITS_PER_BLOCK) % DIGITS_PER_BLOCK;
        let batch_len = if until_aligned == 0 {
            DIGITS_PER_BLOCK
        } else {
            until_aligned
        };
        self.decoded_len = (self.view.len() - self.position).min(batch_len);
        self.view
            .decode_range(self.decoded_start, &mut self.decoded[..self.decoded_len])
            .expect("packed iterator range is in bounds");
    }

    #[inline(always)]
    pub(crate) fn next_array<const N: usize>(&mut self) -> Option<[i8; N]> {
        let end = self.position.checked_add(N)?;
        if end > self.view.len() {
            return None;
        }
        if self.position < self.decoded_start || end > self.decoded_start + self.decoded_len {
            self.refill();
        }
        let local_start = self.position - self.decoded_start;
        if local_start + N > self.decoded_len {
            let values = self
                .view
                .decode_array::<N>(self.position)
                .expect("packed iterator array is in bounds");
            self.position = end;
            return Some(values);
        }
        let values = std::array::from_fn(|offset| self.decoded[local_start + offset]);
        self.position = end;
        Some(values)
    }
}

#[cfg(test)]
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
#[cfg(test)]
fn decode_full_block(
    digits: &PackedSignedDigits,
    block_index: usize,
    output: &mut [i8; DIGITS_PER_BLOCK],
) {
    decode_full_block_view(&digits.storage, digits.bit_width, true, block_index, output);
}

#[inline]
fn decode_full_block_view(
    storage: &[u8],
    bit_width: u8,
    vector_safe: bool,
    block_index: usize,
    output: &mut [i8; DIGITS_PER_BLOCK],
) {
    let byte_offset = block_index * usize::from(bit_width) * 8;
    let encoded = &storage[byte_offset..];
    if !vector_safe {
        scalar::decode_full_block_zero_padded(encoded, bit_width, output);
        return;
    }
    debug_assert!(encoded.len() >= usize::from(bit_width) * 8 + VECTOR_LOAD_PADDING);

    #[cfg(target_arch = "x86_64")]
    if x86_64::try_decode_full_block(encoded, bit_width, output) {
        return;
    }
    #[cfg(target_arch = "aarch64")]
    {
        // NEON is part of the baseline AArch64 architecture.
        unsafe { aarch64::decode_full_block(encoded, bit_width, output) };
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    scalar::decode_full_block(encoded, bit_width, output);

    #[cfg(target_arch = "x86_64")]
    scalar::decode_full_block(encoded, bit_width, output);
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

fn validate_bounds(bounds: SignedDigitBounds, bit_width: u8) -> Result<(), AkitaError> {
    let half = 1i16 << (bit_width - 1);
    if i16::from(bounds.negative_abs_max) <= half && i16::from(bounds.positive_max) < half {
        return Ok(());
    }
    Err(AkitaError::InvalidInput(format!(
        "digit bounds [-{}, {}] do not fit signed {bit_width}-bit storage",
        bounds.negative_abs_max, bounds.positive_max,
    )))
}

fn signed_digit_bounds(digits: &[i8]) -> SignedDigitBounds {
    // Folding from zero makes the minimum's magnitude the largest negative
    // magnitude and the maximum the largest nonnegative digit.
    let (min, max) = digits.iter().fold((0i8, 0i8), |(min, max), &digit| {
        (min.min(digit), max.max(digit))
    });
    SignedDigitBounds {
        negative_abs_max: min.unsigned_abs(),
        positive_max: max as u8,
    }
}

fn minimum_signed_bit_width(bounds: SignedDigitBounds) -> u8 {
    (1..=8)
        .find(|&bit_width| {
            let half = 1u16 << (bit_width - 1);
            u16::from(bounds.negative_abs_max) <= half && u16::from(bounds.positive_max) < half
        })
        .expect("every i8 value fits signed eight-bit storage")
}

/// Encode `digits` from a block boundary and return their bounds.
fn encode_digits(digits: &[i8], bit_width: u8, output: &mut [u8]) -> SignedDigitBounds {
    debug_assert_eq!(
        output.len(),
        encoded_byte_len(digits.len(), bit_width).expect("validated packed length")
    );
    let block_bytes = usize::from(bit_width) * 8;
    let encode = |output: &mut [u8], digits: &[i8]| {
        output
            .chunks_mut(block_bytes)
            .zip(digits.chunks(DIGITS_PER_BLOCK))
            .for_each(|(encoded, source)| scalar::encode_block(source, bit_width, encoded));
        signed_digit_bounds(digits)
    };

    #[cfg(feature = "parallel")]
    if digits.len() >= PARALLEL_ENCODE_THRESHOLD {
        const TASK_BLOCKS: usize = 64;
        return output
            .par_chunks_mut(block_bytes * TASK_BLOCKS)
            .zip(digits.par_chunks(DIGITS_PER_BLOCK * TASK_BLOCKS))
            .map(|(output, digits)| encode(output, digits))
            .reduce(
                || SignedDigitBounds::ZERO,
                |mut bounds, task| {
                    bounds.include_bounds(task);
                    bounds
                },
            );
    }

    encode(output, digits)
}

#[cfg(test)]
mod tests;
