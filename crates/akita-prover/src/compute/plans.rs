use akita_algebra::CyclotomicRing;
use akita_error::{checked, AkitaError};
use akita_field::FieldCore;

use crate::backend::packed_digits::{PackedSignedDigitBlock, PackedSignedDigitView};

/// Checked block geometry over one packed dense decomposition.
#[derive(Clone, Copy)]
pub(crate) struct PackedDenseCommitInput<'a> {
    digits: PackedSignedDigitView<'a>,
    num_rings: usize,
    num_positions_per_block: usize,
    num_digits_inner: usize,
}

impl<'a> PackedDenseCommitInput<'a> {
    pub(crate) fn new<const D: usize>(
        digits: PackedSignedDigitView<'a>,
        num_rings: usize,
        num_positions_per_block: usize,
        num_digits_inner: usize,
    ) -> Result<Self, AkitaError> {
        if num_positions_per_block == 0 || num_digits_inner == 0 {
            return Err(AkitaError::InvalidSetup(
                "packed dense commitment geometry must be nonzero".into(),
            ));
        }
        let expected = checked::product([num_rings, num_digits_inner, D])
            .ok_or_else(|| AkitaError::InvalidSetup("packed dense digit length overflow".into()))?;
        if digits.len() != expected {
            return Err(AkitaError::InvalidSize {
                expected,
                actual: digits.len(),
            });
        }
        Ok(Self {
            digits,
            num_rings,
            num_positions_per_block,
            num_digits_inner,
        })
    }

    pub(crate) fn row_width(self) -> Result<usize, AkitaError> {
        checked::product([self.num_positions_per_block, self.num_digits_inner])
            .ok_or_else(|| AkitaError::InvalidSetup("dense row width overflow".into()))
    }

    pub(crate) fn num_live_blocks(self) -> usize {
        self.num_rings.div_ceil(self.num_positions_per_block)
    }

    /// Borrow block slices when each stored byte is already one signed digit.
    pub(crate) fn borrowed_blocks<const D: usize>(
        self,
    ) -> Result<Option<Vec<&'a [[i8; D]]>>, AkitaError> {
        let Some(bytes) = self.digits.as_i8_slice() else {
            return Ok(None);
        };
        let (rings, remainder) = bytes.as_chunks::<D>();
        if !remainder.is_empty() {
            return Err(AkitaError::InvalidSetup(
                "dense byte digits are not ring aligned".into(),
            ));
        }
        let mut blocks = Vec::with_capacity(self.num_live_blocks());
        for block_index in 0..self.num_live_blocks() {
            let start_ring = checked::product([block_index, self.num_positions_per_block])
                .ok_or_else(|| AkitaError::InvalidSetup("dense block offset overflow".into()))?;
            let live_rings = (self.num_rings - start_ring).min(self.num_positions_per_block);
            let start = checked::product([start_ring, self.num_digits_inner]).ok_or_else(|| {
                AkitaError::InvalidSetup("dense digit block offset overflow".into())
            })?;
            let len = checked::product([live_rings, self.num_digits_inner]).ok_or_else(|| {
                AkitaError::InvalidSetup("dense digit block length overflow".into())
            })?;
            let end = checked::sum([start, len])
                .ok_or_else(|| AkitaError::InvalidSetup("dense digit extent overflow".into()))?;
            blocks.push(rings.get(start..end).ok_or(AkitaError::InvalidProof)?);
        }
        Ok(Some(blocks))
    }

    pub(crate) fn decode_block<const D: usize>(
        self,
        block_index: usize,
    ) -> Result<PackedSignedDigitBlock<'a, D>, AkitaError> {
        if block_index >= self.num_live_blocks() {
            return Err(AkitaError::InvalidSize {
                expected: self.num_live_blocks(),
                actual: block_index,
            });
        }
        let start_ring = checked::product([block_index, self.num_positions_per_block])
            .ok_or_else(|| AkitaError::InvalidSetup("dense block offset overflow".into()))?;
        let live_rings = (self.num_rings - start_ring).min(self.num_positions_per_block);
        let start_digit_ring = checked::product([start_ring, self.num_digits_inner])
            .ok_or_else(|| AkitaError::InvalidSetup("dense digit block offset overflow".into()))?;
        let digit_rings = checked::product([live_rings, self.num_digits_inner])
            .ok_or_else(|| AkitaError::InvalidSetup("dense digit block length overflow".into()))?;
        let start = checked::product([start_digit_ring, D])
            .ok_or_else(|| AkitaError::InvalidSetup("dense digit offset overflow".into()))?;
        let len = checked::product([digit_rings, D])
            .ok_or_else(|| AkitaError::InvalidSetup("dense digit length overflow".into()))?;
        let end = checked::sum([start, len])
            .ok_or_else(|| AkitaError::InvalidSetup("dense digit extent overflow".into()))?;
        let block = self.digits.slice(start..end)?;
        if let Some(bytes) = block.as_i8_slice() {
            let (rings, remainder) = bytes.as_chunks::<D>();
            debug_assert!(remainder.is_empty());
            return Ok(PackedSignedDigitBlock::Borrowed(rings));
        }
        Ok(PackedSignedDigitBlock::Decoded(
            self.digits
                .decode_rings::<D>(start_digit_ring, digit_rings)?,
        ))
    }
}

/// Internal dense CPU commit representation.
pub(crate) enum DenseCommitInput<'a, F: FieldCore, const D: usize> {
    /// Balanced digit planes are already packed by the polynomial.
    PackedDigits {
        /// Checked packed source and block geometry.
        source: PackedDenseCommitInput<'a>,
        /// Logarithm of the gadget basis used to produce the cached digits.
        log_basis_inner: u32,
    },
    /// Ring coefficients need backend-side digit decomposition.
    CoeffBlocks {
        /// Per-block coefficient slices.
        block_slices: Vec<&'a [CyclotomicRing<F, D>]>,
        /// Number of balanced digits used for the A-side commit.
        num_digits_inner: usize,
        /// Logarithm of the gadget basis.
        log_basis_inner: u32,
    },
}

/// Named ring-switch relation rows returned by a backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RingSwitchRelationRows<F: FieldCore, const D: usize> {
    /// D-side negacyclic reduced rows used as the transcript-visible `v`.
    pub d_negacyclic: Vec<CyclotomicRing<F, D>>,
    /// D-side cyclic rows.
    pub d_cyclic: Vec<CyclotomicRing<F, D>>,
    /// B-side cyclic rows.
    pub b_cyclic: Vec<CyclotomicRing<F, D>>,
    /// A-side quotient rows.
    pub a_quotients: Vec<CyclotomicRing<F, D>>,
}
