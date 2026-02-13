//! Prover placeholders for upcoming commitment proof flow.

use crate::error::HachiError;
use crate::protocol::commitment::{RingOpenProof, RingOpening};
use crate::protocol::transcript::Transcript;
use crate::{CanonicalField, FieldCore};

/// Placeholder prover hook for open-check proof generation.
///
/// # Errors
///
/// Always returns `HachiError::InvalidInput` until the prover path is implemented.
pub fn prove_opening_stub<T, F, const D: usize>(
    _transcript: &mut T,
    _opening: &RingOpening<F, D>,
) -> Result<RingOpenProof<F, D>, HachiError>
where
    T: Transcript<F>,
    F: FieldCore + CanonicalField,
{
    Err(HachiError::InvalidInput(
        "prover open-check stub is not implemented yet".to_string(),
    ))
}
