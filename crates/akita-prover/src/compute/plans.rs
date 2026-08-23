use akita_algebra::CyclotomicRing;
use akita_error::{checked, AkitaError};
use akita_field::FieldCore;

use crate::backend::packed_digits::PackedSignedDigitView;

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

    pub(crate) fn decode_block<const D: usize>(
        self,
        block_index: usize,
    ) -> Result<Vec<[i8; D]>, AkitaError> {
        if block_index >= self.num_live_blocks() {
            return Err(AkitaError::InvalidSize {
                expected: self.num_live_blocks(),
                actual: block_index,
            });
        }
        let start_ring = checked::product([block_index, self.num_positions_per_block])
            .ok_or_else(|| AkitaError::InvalidSetup("dense block offset overflow".into()))?;
        let live_rings = (self.num_rings - start_ring).min(self.num_positions_per_block);
        self.digits.decode_rings::<D>(
            start_ring * self.num_digits_inner,
            live_rings * self.num_digits_inner,
        )
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
