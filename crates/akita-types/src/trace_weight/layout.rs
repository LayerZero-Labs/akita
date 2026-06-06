use akita_field::AkitaError;

/// Geometry of the opening-digit column segment inside the stage-2 witness.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TraceWeightLayout {
    pub ring_bits: usize,
    pub col_bits: usize,
    pub opening_digit_offset: usize,
    pub num_blocks: usize,
    pub num_digits_open: usize,
    pub r_vars: usize,
    pub log_basis: u32,
}

impl TraceWeightLayout {
    pub fn opening_digit_col_count(&self) -> usize {
        self.num_blocks * self.num_digits_open
    }

    pub fn ring_len(&self) -> usize {
        1usize << self.ring_bits
    }

    pub fn table_len(&self) -> Result<usize, AkitaError> {
        1usize
            .checked_shl((self.col_bits + self.ring_bits) as u32)
            .ok_or_else(|| {
                AkitaError::InvalidInput("trace-weight table length overflow".to_string())
            })
    }

    pub fn opening_digit_col_index(&self, block: usize, plane: usize) -> usize {
        self.opening_digit_offset + plane * self.num_blocks + block
    }

    pub fn witness_index(&self, col: usize, ring_coord: usize) -> usize {
        col * self.ring_len() + ring_coord
    }

    pub(crate) fn plane_bits(&self) -> usize {
        self.num_digits_open.trailing_zeros() as usize
    }

    pub(crate) fn validate_ring_dimension<const D: usize>(&self) -> Result<(), AkitaError> {
        if self.ring_bits != D.trailing_zeros() as usize {
            return Err(AkitaError::InvalidInput(
                "ring_bits does not match ring dimension".to_string(),
            ));
        }
        Ok(())
    }

    pub(crate) fn validate_opening_digit_segment(&self) -> Result<(), AkitaError> {
        let end = self
            .opening_digit_offset
            .checked_add(self.opening_digit_col_count())
            .ok_or_else(|| {
                AkitaError::InvalidInput("opening-digit segment overflow".to_string())
            })?;
        if end > 1usize << self.col_bits {
            return Err(AkitaError::InvalidInput(
                "opening-digit segment exceeds column hypercube".to_string(),
            ));
        }
        Ok(())
    }

    pub(crate) fn validate_closed_form_eval_point(
        &self,
        ring_point_len: usize,
        col_point_len: usize,
    ) -> Result<usize, AkitaError> {
        if ring_point_len != self.ring_bits || col_point_len != self.col_bits {
            return Err(AkitaError::InvalidSize {
                expected: self.col_bits + self.ring_bits,
                actual: col_point_len + ring_point_len,
            });
        }
        self.validate_opening_digit_segment()?;
        let plane_bits = self.plane_bits();
        if !self.num_digits_open.is_power_of_two() || plane_bits + self.r_vars > self.col_bits {
            return Err(AkitaError::InvalidInput(
                "trace-weight closed-form eval requires power-of-two open digits and aligned column split"
                    .to_string(),
            ));
        }
        Ok(plane_bits)
    }
}
