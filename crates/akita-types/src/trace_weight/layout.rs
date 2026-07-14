use akita_field::AkitaError;

use crate::{OpeningBatchWitnessLayout, SemanticGroupId};

/// Geometry of the opening-digit columns used by the stage-2 trace term.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceWeightLayout {
    pub ring_bits: usize,
    pub col_bits: usize,
    pub live_fold_count: usize,
    pub num_digits_open: usize,
    pub fold_bits: usize,
    pub log_basis: u32,
    pub witness_layout: OpeningBatchWitnessLayout,
    pub opening_source_len: usize,
    pub group_id: SemanticGroupId,
}

impl TraceWeightLayout {
    pub fn opening_digit_col_count(&self) -> usize {
        self.live_fold_count * self.num_digits_open
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

    pub fn opening_digit_col_index(&self, block: usize, digit: usize) -> Result<usize, AkitaError> {
        let group = self.witness_layout.group(self.group_id)?;
        if block >= self.live_fold_count || digit >= self.num_digits_open {
            return Err(AkitaError::InvalidInput(
                "trace opening-digit index out of range".to_string(),
            ));
        }
        let claim = block / group.live_fold_count;
        let global_block = block % group.live_fold_count;
        let unit = self
            .witness_layout
            .unit_for_block(self.group_id, global_block)?;
        let physical_index = self
            .witness_layout
            .e_index(unit, claim, global_block, digit)?;
        crate::checked_opening_source_index(self.opening_source_len, physical_index)
    }

    pub fn witness_index(&self, col: usize, ring_coord: usize) -> usize {
        col * self.ring_len() + ring_coord
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
        let group = self.witness_layout.group(self.group_id)?;
        if self.live_fold_count == 0
            || self.num_digits_open != group.depth_open
            || self.live_fold_count
                != group
                    .num_claims
                    .checked_mul(group.live_fold_count)
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup("trace block count overflow".to_string())
                    })?
        {
            return Err(AkitaError::InvalidSetup(
                "trace geometry disagrees with witness layout".to_string(),
            ));
        }
        let column_bound = 1usize.checked_shl(self.col_bits as u32).ok_or_else(|| {
            AkitaError::InvalidInput("opening-digit column bound overflow".to_string())
        })?;
        for block in 0..self.live_fold_count {
            for digit in 0..self.num_digits_open {
                if self.opening_digit_col_index(block, digit)? >= column_bound {
                    return Err(AkitaError::InvalidInput(
                        "opening-digit segment exceeds column hypercube".to_string(),
                    ));
                }
            }
        }
        Ok(())
    }

    pub(crate) fn validate_trace_term_block_range(
        &self,
        block_offset: usize,
        block_span: usize,
    ) -> Result<(), AkitaError> {
        let end = block_offset.checked_add(block_span).ok_or_else(|| {
            AkitaError::InvalidInput("trace term block range overflow".to_string())
        })?;
        if end > self.live_fold_count {
            return Err(AkitaError::InvalidInput(
                "trace term exceeds layout block count".to_string(),
            ));
        }
        Ok(())
    }
}
