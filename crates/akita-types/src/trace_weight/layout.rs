use akita_field::AkitaError;

use crate::WitnessLayout;

/// Geometry of the opening-digit columns used by the stage-2 trace term.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceWeightLayout {
    pub ring_bits: usize,
    pub col_bits: usize,
    pub num_blocks: usize,
    pub num_digits_open: usize,
    pub block_bits: usize,
    pub log_basis: u32,
    pub witness_layout: WitnessLayout,
    pub opening_source_len: usize,
    pub group_id: usize,
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

    pub fn opening_digit_col_index(&self, block: usize, digit: usize) -> Result<usize, AkitaError> {
        if block >= self.num_blocks || digit >= self.num_digits_open {
            return Err(AkitaError::InvalidInput(
                "trace opening-digit index out of range".to_string(),
            ));
        }
        let group_live_block_count = self.witness_layout.group_live_block_count(self.group_id)?;
        let num_claims = self
            .num_blocks
            .checked_div(group_live_block_count)
            .filter(|_| self.num_blocks.is_multiple_of(group_live_block_count))
            .ok_or_else(|| {
                AkitaError::InvalidSetup("trace claim axis disagrees with witness layout".into())
            })?;
        let claim = block / group_live_block_count;
        let global_block = block % group_live_block_count;
        let unit = self
            .witness_layout
            .unit_for_block(self.group_id, global_block)?;
        let physical_index = self.witness_layout.e_index(
            unit,
            num_claims,
            self.num_digits_open,
            claim,
            global_block,
            digit,
        )?;
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
        let group_live_block_count = self.witness_layout.group_live_block_count(self.group_id)?;
        if self.num_blocks == 0
            || group_live_block_count == 0
            || !self.num_blocks.is_multiple_of(group_live_block_count)
        {
            return Err(AkitaError::InvalidSetup(
                "trace geometry disagrees with witness layout".to_string(),
            ));
        }
        let column_bound = 1usize.checked_shl(self.col_bits as u32).ok_or_else(|| {
            AkitaError::InvalidInput("opening-digit column bound overflow".to_string())
        })?;
        if self.num_digits_open == 0 {
            return Err(AkitaError::InvalidSetup(
                "trace layout requires an opening digit".to_string(),
            ));
        }
        let num_claims = self.num_blocks / group_live_block_count;
        for claim in 0..num_claims {
            let claim_start = claim.checked_mul(group_live_block_count).ok_or_else(|| {
                AkitaError::InvalidSetup("trace claim offset overflow".to_string())
            })?;
            for unit in self.witness_layout.units_for_group(self.group_id)? {
                let last_global = unit
                    .global_block_start()
                    .checked_add(unit.live_block_count().checked_sub(1).ok_or_else(|| {
                        AkitaError::InvalidSetup("trace unit has no live blocks".to_string())
                    })?)
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup("trace unit block range overflow".to_string())
                    })?;
                let first_block = claim_start
                    .checked_add(unit.global_block_start())
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup("trace block offset overflow".to_string())
                    })?;
                let last_block = claim_start.checked_add(last_global).ok_or_else(|| {
                    AkitaError::InvalidSetup("trace block offset overflow".to_string())
                })?;
                let first = self.opening_digit_col_index(first_block, 0)?;
                let last = self.opening_digit_col_index(last_block, self.num_digits_open - 1)?;
                if first >= column_bound || last >= column_bound {
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
        if end > self.num_blocks {
            return Err(AkitaError::InvalidInput(
                "trace term exceeds layout block count".to_string(),
            ));
        }
        Ok(())
    }
}
