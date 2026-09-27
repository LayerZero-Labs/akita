use super::*;
use crate::wire_limits::{checked_shape_len, checked_shape_sequence_len};
use crate::OpeningClaimsLayout;
use akita_sumcheck::SumcheckProofShape;

/// Public shape of the native extension-opening-reduction messages.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtensionOpeningReductionShape {
    /// Number of partial evaluations serialized before the sumcheck.
    pub partials: usize,
    /// Number of individual terminal claims serialized after the sumcheck.
    pub final_claims: usize,
    /// One compact coefficient count per round of the batched reduction.
    pub sumcheck: SumcheckProofShape,
}

impl ExtensionOpeningReductionShape {
    /// Construct the standard degree-two reduction shape.
    pub fn standard(partials: usize, num_rounds: usize, num_claims: usize) -> Self {
        Self {
            partials,
            final_claims: num_claims,
            sumcheck: uniform_sumcheck_shape(num_rounds, EXTENSION_OPENING_REDUCTION_DEGREE),
        }
    }
}

/// Derive the only accepted extension-opening reduction shape for one opening batch.
pub fn canonical_extension_opening_reduction_shape(
    opening_layout: &OpeningClaimsLayout,
    extension_degree: usize,
) -> Result<ExtensionOpeningReductionShape, AkitaError> {
    if extension_degree <= 1 || !extension_degree.is_power_of_two() {
        return Err(AkitaError::InvalidSetup(
            "extension opening degree must be a power of two greater than one".to_string(),
        ));
    }
    opening_layout.check()?;
    let split_bits = extension_degree.trailing_zeros() as usize;
    let num_rounds = opening_layout
        .max_num_vars()
        .checked_sub(split_bits)
        .ok_or_else(|| {
            AkitaError::InvalidSetup(
                "extension opening split exceeds the opening arity".to_string(),
            )
        })?;
    let num_claims = opening_layout.num_total_polynomials();
    let partials = extension_degree.checked_mul(num_claims).ok_or_else(|| {
        AkitaError::InvalidSetup("extension opening partial count overflow".to_string())
    })?;
    Ok(ExtensionOpeningReductionShape::standard(
        partials, num_rounds, num_claims,
    ))
}

impl Valid for ExtensionOpeningReductionShape {
    fn check(&self) -> Result<(), SerializationError> {
        checked_shape_len(self.partials)?;
        checked_shape_len(self.final_claims)?;
        checked_shape_sequence_len(self.sumcheck.len())?;
        if self.final_claims == 0 {
            return Err(SerializationError::InvalidData(
                "extension opening reduction shape must contain terminal claims".to_string(),
            ));
        }
        for &degree in &self.sumcheck {
            checked_shape_len(degree)?;
            if degree != EXTENSION_OPENING_REDUCTION_DEGREE {
                return Err(SerializationError::InvalidData(format!(
                    "extension opening reduction degree {} does not match expected degree {}",
                    degree, EXTENSION_OPENING_REDUCTION_DEGREE
                )));
            }
        }
        Ok(())
    }
}
