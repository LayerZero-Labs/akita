//! Verifier

use crate::algebra::ring::CyclotomicRing;
use crate::error::HachiError;
use crate::protocol::opening_point::RingOpeningPoint;
use crate::protocol::proof::HachiProof;
use crate::FieldCore;

/// Verify a Hachi opening proof against the claimed evaluation.
///
/// # Errors
///
/// Returns an error if verification fails.
pub fn verify<F: FieldCore, const D: usize>(
    _point: &RingOpeningPoint<F, D>,
    _evaluation: &CyclotomicRing<F, D>,
    _proof: &HachiProof<F, D>,
) -> Result<(), HachiError> {
    unimplemented!()
}
