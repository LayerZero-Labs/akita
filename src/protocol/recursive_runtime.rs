//! Runtime-only caches for recursive Hachi prove levels.
//!
//! These structures sit between the recursive `w` witness and the verifier-
//! facing proof wire. They preserve the commitment-side prover caches that the
//! next recursive level needs, without forcing the prover to round-trip through
//! the proof-oriented flat adapters each time.

use crate::protocol::proof::HachiCommitmentHint;
use crate::FieldCore;
use akita_algebra::CyclotomicRing;
use akita_field::HachiError;

/// D-erased prover cache for a recursive commitment hint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RecursiveCommitmentHintCache<F: FieldCore> {
    inner_opening_digits: Vec<i8>,
    inner_opening_block_sizes: Vec<usize>,
    t_coeffs: Vec<F>,
    t_block_sizes: Vec<usize>,
    ring_dim: usize,
}

impl<F: FieldCore> RecursiveCommitmentHintCache<F> {
    /// Flatten a typed prover hint into a runtime cache that preserves both the
    /// digit planes and the recomposed `t` rows.
    pub(crate) fn from_typed<const D: usize>(
        hint: HachiCommitmentHint<F, D>,
    ) -> Result<Self, HachiError> {
        let (flat_hint_digits, t) = hint.into_flat_parts();
        let inner_opening_block_sizes = flat_hint_digits.block_sizes().to_vec();
        let total_digit_planes: usize = flat_hint_digits.flat_digits().len();
        let mut inner_opening_digits = Vec::with_capacity(total_digit_planes * D);
        for plane in flat_hint_digits.flat_digits() {
            inner_opening_digits.extend_from_slice(plane);
        }

        let t = t.ok_or_else(|| {
            HachiError::InvalidInput(
                "missing recomposed t rows in recursive commitment hint".to_string(),
            )
        })?;
        let t_block_sizes: Vec<usize> = t.iter().map(Vec::len).collect();
        let total_t_rings: usize = t_block_sizes.iter().sum();
        let mut t_coeffs = Vec::with_capacity(total_t_rings * D);
        for block in &t {
            for ring in block {
                t_coeffs.extend_from_slice(ring.coefficients());
            }
        }

        Ok(Self {
            inner_opening_digits,
            inner_opening_block_sizes,
            t_coeffs,
            t_block_sizes,
            ring_dim: D,
        })
    }

    /// Reconstruct the typed prover hint without recomputing `t`.
    pub(crate) fn to_typed<const D: usize>(&self) -> Result<HachiCommitmentHint<F, D>, HachiError> {
        if self.ring_dim != D {
            return Err(HachiError::InvalidInput(format!(
                "recursive hint cache D mismatch: cache={}, requested={D}",
                self.ring_dim
            )));
        }
        if self.inner_opening_block_sizes.len() != self.t_block_sizes.len() {
            return Err(HachiError::InvalidInput(
                "recursive hint cache block metadata mismatch".to_string(),
            ));
        }

        let (flat_digits, digit_remainder) = self.inner_opening_digits.as_chunks::<D>();
        if !digit_remainder.is_empty() {
            return Err(HachiError::InvalidSize {
                expected: D,
                actual: self.inner_opening_digits.len(),
            });
        }
        let (flat_t, t_remainder) = self.t_coeffs.as_chunks::<D>();
        if !t_remainder.is_empty() {
            return Err(HachiError::InvalidSize {
                expected: D,
                actual: self.t_coeffs.len(),
            });
        }

        let mut digit_offset = 0usize;
        let mut t_offset = 0usize;
        let mut inner_opening_digits = Vec::with_capacity(flat_digits.len());
        let mut t = Vec::with_capacity(self.t_block_sizes.len());

        for (&digit_block_size, &t_block_size) in self
            .inner_opening_block_sizes
            .iter()
            .zip(self.t_block_sizes.iter())
        {
            let digit_end = digit_offset + digit_block_size;
            let t_end = t_offset + t_block_size;
            if digit_end > flat_digits.len() || t_end > flat_t.len() {
                return Err(HachiError::InvalidInput(
                    "recursive hint cache block data is truncated".to_string(),
                ));
            }

            inner_opening_digits.extend_from_slice(&flat_digits[digit_offset..digit_end]);
            t.push(
                flat_t[t_offset..t_end]
                    .iter()
                    .map(|coeffs| CyclotomicRing::from_coefficients(*coeffs))
                    .collect(),
            );
            digit_offset = digit_end;
            t_offset = t_end;
        }

        if digit_offset != flat_digits.len() || t_offset != flat_t.len() {
            return Err(HachiError::InvalidInput(
                "recursive hint cache has trailing block data".to_string(),
            ));
        }

        let inner_opening_digits = crate::protocol::proof::FlatDigitBlocks::new(
            inner_opening_digits,
            self.inner_opening_block_sizes.clone(),
        )?;
        Ok(HachiCommitmentHint::singleton_with_t(
            inner_opening_digits,
            t,
        ))
    }
}
