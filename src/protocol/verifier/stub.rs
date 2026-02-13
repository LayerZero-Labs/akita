//! Verifier placeholders for upcoming commitment proof flow.

use crate::error::HachiError;
use crate::protocol::commitment::{RingCommitment, RingOpenProof};
use crate::protocol::transcript::Transcript;
use crate::{CanonicalField, FieldCore};

/// Placeholder verifier hook for open-check proof verification.
///
/// # Errors
///
/// Always returns `HachiError::InvalidInput` until the verifier path is implemented.
pub fn verify_opening_stub<T, F, const D: usize>(
    _transcript: &mut T,
    _commitment: &RingCommitment<F, D>,
    _proof: &RingOpenProof<F, D>,
) -> Result<(), HachiError>
where
    T: Transcript<F>,
    F: FieldCore + CanonicalField,
{
    Err(HachiError::InvalidInput(
        "verifier open-check stub is not implemented yet".to_string(),
    ))
}
