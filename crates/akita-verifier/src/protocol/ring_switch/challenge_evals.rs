use akita_field::{AkitaError, FieldCore};

/// Alpha evaluations of the fold challenges in claim-major block order.
#[derive(Clone)]
pub(crate) struct PreparedChallengeEvals<F: FieldCore>(pub(crate) Vec<F>);

/// One claim's logical fold weights, padded to a power of two.
pub(crate) struct PreparedAffineFactors<F> {
    pub(crate) low: Vec<F>,
}

impl<F: FieldCore> PreparedChallengeEvals<F> {
    pub(crate) fn affine_factors(
        &self,
        claim: usize,
        num_live_blocks: usize,
    ) -> Result<PreparedAffineFactors<F>, AkitaError> {
        if num_live_blocks == 0 {
            return Err(AkitaError::InvalidSetup(
                "challenge factors require num_live_blocks > 0".into(),
            ));
        }
        let start = claim
            .checked_mul(num_live_blocks)
            .ok_or_else(|| AkitaError::InvalidSetup("challenge factor offset overflow".into()))?;
        let end = start
            .checked_add(num_live_blocks)
            .ok_or_else(|| AkitaError::InvalidSetup("challenge factor end overflow".into()))?;
        let values = self.0.get(start..end).ok_or(AkitaError::InvalidSize {
            expected: end,
            actual: self.0.len(),
        })?;
        let low_len = num_live_blocks
            .checked_next_power_of_two()
            .ok_or_else(|| AkitaError::InvalidSetup("challenge factor length overflow".into()))?;
        let mut low = vec![F::zero(); low_len];
        low[..num_live_blocks].copy_from_slice(values);
        Ok(PreparedAffineFactors { low })
    }
}
