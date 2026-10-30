//! Exact source addressing, including committed padding and virtual MLE slots.

use akita_error::{checked, AkitaError};

/// Location of a bit in the signed degree-648 source ring.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BinaryRingPosition {
    /// Commitment column (the low carrier-address bits).
    pub column: usize,
    /// Degree-648 ring element within that column.
    pub ring_row: usize,
    /// Coefficient slot, including virtual slots through 1023.
    pub coefficient: usize,
    /// Signed packing multiplier, either 1 or -1.
    pub sign: i8,
}

/// Location in the scalar-coordinate MLE layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BinaryScalarPosition {
    /// F128 carrier address in canonical low-column-bit order.
    pub carrier: usize,
    /// Scalar coordinate, including virtual coordinates through 255.
    pub coordinate: usize,
}

/// Whether a source position is data, constrained padding, or definitionally zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinaryPositionClass {
    /// One bit of a logical word.
    Payload {
        /// Logical word address.
        word: usize,
        /// Low-bit-first coordinate within the logical word.
        bit: usize,
    },
    /// A real committed coefficient that the agreement contract requires zero.
    CommittedPadding,
    /// An MLE slot outside the source ring, zero by definition.
    Virtual,
}

/// Low-bit-first variable split for logical word claims.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BinaryClaimVariableSplit {
    /// Low logical-address variables selecting a word inside one F128 carrier.
    pub intra_carrier: usize,
    /// Remaining variables selecting a carrier, including committed padding.
    pub carrier_address: usize,
    /// Bit variables selecting a coordinate inside the logical word.
    pub word_bit: usize,
}

/// Validated immutable geometry for dense binary words and signed source rings.
///
/// A carrier stores `128 / ell` words. Canonical carrier address `i` has column
/// `i % C`, scalar row `i / C`, component `a = (i / C) % 4`, and ring row
/// `i / (4*C)`. Scalar coordinate `s` occupies coefficient `4*s+a`, with sign
/// `(-1)^s`. The existing source packing kernel instead stores whole columns
/// contiguously; [`Self::kernel_carrier_index`] publishes that permutation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BinaryIndexMap {
    ell: usize,
    logical_len: usize,
    padded_carriers: usize,
    columns: usize,
}

impl BinaryIndexMap {
    /// Validate a nonempty source and power-of-two, full-`4*C` carrier geometry.
    ///
    /// The logical length counts words; the padded length counts F128 carriers.
    /// Unsupported word widths, insufficient padding, and overflowing capacities
    /// are rejected before any index operation or allocation.
    pub fn new(
        ell: usize,
        logical_len: usize,
        padded_carriers: usize,
        columns: usize,
    ) -> Result<Self, AkitaError> {
        if !ell.is_power_of_two() || ell > 128 {
            return Err(AkitaError::InvalidInput(
                "binary word width must be a power of two from 1 through 128".into(),
            ));
        }
        let block = checked::product([4, columns])
            .ok_or_else(|| AkitaError::InvalidInput("binary carrier block size overflow".into()))?;
        if logical_len == 0
            || !columns.is_power_of_two()
            || !padded_carriers.is_power_of_two()
            || padded_carriers < block
            || checked::exact_div(padded_carriers, block).is_none()
        {
            return Err(AkitaError::InvalidInput(
                "binary source requires a nonempty power-of-two full-block layout".into(),
            ));
        }
        let capacity = checked::product([padded_carriers, 128 / ell])
            .ok_or_else(|| AkitaError::InvalidInput("binary word capacity overflow".into()))?;
        // Check the entire padded MLE address space at the admission boundary.
        checked::product([padded_carriers, 256])
            .ok_or_else(|| AkitaError::InvalidInput("binary MLE size overflow".into()))?;
        if logical_len > capacity {
            return Err(AkitaError::InvalidSize {
                expected: capacity,
                actual: logical_len,
            });
        }
        Ok(Self {
            ell,
            logical_len,
            padded_carriers,
            columns,
        })
    }

    /// Logical word width in bits.
    pub fn word_width(&self) -> usize {
        self.ell
    }

    /// Number of logical words before committed padding.
    pub fn logical_len(&self) -> usize {
        self.logical_len
    }

    /// Number of F128 carriers, including full-block committed padding.
    pub fn padded_carriers(&self) -> usize {
        self.padded_carriers
    }

    /// Number of commitment columns.
    pub fn columns(&self) -> usize {
        self.columns
    }

    /// Publish the low-bit-first split used to form field-switch weights.
    pub fn claim_variable_split(&self) -> BinaryClaimVariableSplit {
        BinaryClaimVariableSplit {
            intra_carrier: (128 / self.ell).trailing_zeros() as usize,
            carrier_address: self.padded_carriers.trailing_zeros() as usize,
            word_bit: self.ell.trailing_zeros() as usize,
        }
    }

    /// Map one logical payload bit to its signed source coefficient.
    pub fn logical_to_ring(
        &self,
        word: usize,
        bit: usize,
    ) -> Result<BinaryRingPosition, AkitaError> {
        if word >= self.logical_len || bit >= self.ell {
            return Err(AkitaError::InvalidInput(
                "binary logical bit address out of range".into(),
            ));
        }
        let words_per_carrier = 128 / self.ell;
        let coordinate = checked::mul_add(word % words_per_carrier, self.ell, bit)
            .ok_or_else(|| AkitaError::InvalidInput("binary coordinate overflow".into()))?;
        self.scalar_to_ring(word / words_per_carrier, coordinate)
    }

    /// Map a scalar MLE slot to its signed ring slot, including virtual slots.
    pub fn scalar_to_ring(
        &self,
        carrier: usize,
        coordinate: usize,
    ) -> Result<BinaryRingPosition, AkitaError> {
        if carrier >= self.padded_carriers || coordinate >= 256 {
            return Err(AkitaError::InvalidInput(
                "binary scalar address out of range".into(),
            ));
        }
        let scalar_row = carrier / self.columns;
        let coefficient = checked::mul_add(4, coordinate, scalar_row % 4)
            .ok_or_else(|| AkitaError::InvalidInput("binary coefficient overflow".into()))?;
        Ok(BinaryRingPosition {
            column: carrier % self.columns,
            ring_row: scalar_row / 4,
            coefficient,
            sign: if coordinate.is_multiple_of(2) { 1 } else { -1 },
        })
    }

    /// Invert the canonical ring layout, including coefficient slots 648..1023.
    ///
    /// The sign is determined by the returned scalar coordinate; it is never a
    /// caller-controlled degree of freedom.
    pub fn ring_to_scalar(
        &self,
        column: usize,
        ring_row: usize,
        coefficient: usize,
    ) -> Result<BinaryScalarPosition, AkitaError> {
        if column >= self.columns
            || ring_row >= (self.padded_carriers / self.columns) / 4
            || coefficient >= 1024
        {
            return Err(AkitaError::InvalidInput(
                "binary ring address out of range".into(),
            ));
        }
        let scalar_row = checked::mul_add(ring_row, 4, coefficient % 4)
            .ok_or_else(|| AkitaError::InvalidInput("binary scalar row overflow".into()))?;
        let carrier = checked::mul_add(scalar_row, self.columns, column)
            .ok_or_else(|| AkitaError::InvalidInput("binary carrier address overflow".into()))?;
        Ok(BinaryScalarPosition {
            carrier,
            coordinate: coefficient / 4,
        })
    }

    /// Classify a scalar position and recover its logical payload bit, if any.
    ///
    /// Coordinates 128..161 and unused low coordinates are committed padding;
    /// coordinates 162..255 are virtual, even in an otherwise unused carrier.
    pub fn classify_scalar(
        &self,
        carrier: usize,
        coordinate: usize,
    ) -> Result<BinaryPositionClass, AkitaError> {
        if carrier >= self.padded_carriers || coordinate >= 256 {
            return Err(AkitaError::InvalidInput(
                "binary scalar address out of range".into(),
            ));
        }
        if coordinate >= 162 {
            return Ok(BinaryPositionClass::Virtual);
        }
        if coordinate >= 128 {
            return Ok(BinaryPositionClass::CommittedPadding);
        }
        let word = checked::mul_add(carrier, 128 / self.ell, coordinate / self.ell)
            .ok_or_else(|| AkitaError::InvalidInput("binary word address overflow".into()))?;
        Ok(if word < self.logical_len {
            BinaryPositionClass::Payload {
                word,
                bit: coordinate % self.ell,
            }
        } else {
            BinaryPositionClass::CommittedPadding
        })
    }

    /// Classify a ring position and invert payload positions to `(word, bit)`.
    pub fn ring_to_logical(
        &self,
        column: usize,
        ring_row: usize,
        coefficient: usize,
    ) -> Result<BinaryPositionClass, AkitaError> {
        let scalar = self.ring_to_scalar(column, ring_row, coefficient)?;
        self.classify_scalar(scalar.carrier, scalar.coordinate)
    }

    /// Convert canonical carrier order to the existing column-contiguous kernel.
    ///
    /// The permutation is `i -> (i % C)*(N/C) + i/C`. It applies to source tables
    /// and point coordinates consistently; packing four adjacent kernel entries
    /// still uses exactly coefficient `4*s+a` and sign `(-1)^s`.
    pub fn kernel_carrier_index(&self, carrier: usize) -> Result<usize, AkitaError> {
        if carrier >= self.padded_carriers {
            return Err(AkitaError::InvalidInput(
                "binary carrier address out of range".into(),
            ));
        }
        checked::mul_add(
            carrier % self.columns,
            self.padded_carriers / self.columns,
            carrier / self.columns,
        )
        .ok_or_else(|| AkitaError::InvalidInput("binary kernel address overflow".into()))
    }

    /// Invert the published column-contiguous kernel permutation.
    pub fn carrier_from_kernel_index(&self, index: usize) -> Result<usize, AkitaError> {
        if index >= self.padded_carriers {
            return Err(AkitaError::InvalidInput(
                "binary kernel address out of range".into(),
            ));
        }
        let rows = self.padded_carriers / self.columns;
        checked::mul_add(index % rows, self.columns, index / rows)
            .ok_or_else(|| AkitaError::InvalidInput("binary carrier address overflow".into()))
    }
}

#[cfg(test)]
#[path = "index_tests.rs"]
mod tests;
