//! Protocol-level Fiat–Shamir challenge samplers.
//!
//! These utilities derive structured challenges (e.g. sparse ring elements) from
//! the transcript while keeping the low-level representations in the algebra layer.

pub(crate) mod bounded_l1;
pub mod sparse;
pub mod sparse_challenge;

pub use sparse_challenge::{SparseChallenge, SparseChallengeConfig};

use akita_field::AkitaError;
use akita_field::{CanonicalField, FieldCore};

/// Evaluate a sparse ring challenge against precomputed scalar powers.
///
/// # Errors
///
/// Returns an error when `alpha_pows` does not have length `D`.
pub fn eval_sparse_challenge_at_pows<F: FieldCore + CanonicalField, const D: usize>(
    challenge: &SparseChallenge,
    alpha_pows: &[F],
) -> Result<F, AkitaError> {
    if alpha_pows.len() != D {
        return Err(AkitaError::InvalidSize {
            expected: D,
            actual: alpha_pows.len(),
        });
    }

    debug_assert_eq!(challenge.positions.len(), challenge.coeffs.len());

    let mut acc = F::zero();
    for (&pos, &coeff) in challenge.positions.iter().zip(challenge.coeffs.iter()) {
        let idx = pos as usize;
        debug_assert!(idx < D);
        debug_assert_ne!(coeff, 0);
        acc += F::from_i64(coeff as i64) * alpha_pows[idx];
    }
    Ok(acc)
}
