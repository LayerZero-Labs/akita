//! Focused geometry helpers for setup-prefix commitments.

use crate::{
    CompressionChainPlan, OpeningClaimsLayout, PolynomialGroupLayout, PrecommittedLevelParams,
};
use akita_field::AkitaError;
use akita_serialization::SerializationError;

use super::padded_setup_prefix_len;

/// Build the opening layout for one recursive suffix witness.
///
/// An incoming setup prefix is a separate singleton group. Its Boolean domain
/// uses the padded prefix length, while the witness keeps its own arity.
pub fn suffix_opening_layout(
    current_witness_len: usize,
    incoming_setup_prefix: Option<usize>,
) -> Result<OpeningClaimsLayout, AkitaError> {
    fn power_of_two_vars(field_len: usize, context: &'static str) -> Result<usize, AkitaError> {
        if field_len == 0 {
            return Err(AkitaError::InvalidSetup(format!(
                "{context} must be nonzero"
            )));
        }
        let padded = field_len.checked_next_power_of_two().ok_or_else(|| {
            AkitaError::InvalidSetup(format!("{context} power-of-two padding overflow"))
        })?;
        Ok(padded.trailing_zeros() as usize)
    }

    let witness_vars = power_of_two_vars(current_witness_len, "suffix witness length")?;
    let witness_group = PolynomialGroupLayout::singleton(witness_vars);
    match incoming_setup_prefix {
        Some(natural_len) => {
            let n_prefix = padded_setup_prefix_len(natural_len);
            if n_prefix == 0 || !n_prefix.is_power_of_two() {
                return Err(AkitaError::InvalidSetup(
                    "incoming setup prefix length must be a nonzero power of two".to_string(),
                ));
            }
            let prefix_vars = power_of_two_vars(n_prefix, "incoming setup prefix length")?;
            OpeningClaimsLayout::from_groups(vec![
                PolynomialGroupLayout::singleton(prefix_vars),
                witness_group,
            ])
        }
        None => OpeningClaimsLayout::from_groups(vec![witness_group]),
    }
}

pub(super) fn setup_prefix_compression_plan(
    params: &PrecommittedLevelParams,
) -> Result<CompressionChainPlan, SerializationError> {
    let matrix = &params.layout.outer_commit_matrix;
    let source_coefficients = matrix
        .output_rank()
        .checked_mul(matrix.ring_dimension())
        .ok_or_else(|| {
            SerializationError::InvalidData(
                "setup prefix compression source length overflow".into(),
            )
        })?;
    CompressionChainPlan::for_complete_source(
        matrix.sis_table_key().modulus_profile,
        source_coefficients,
    )
    .map_err(|error| SerializationError::InvalidData(error.to_string()))
}
