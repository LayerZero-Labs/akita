//! The [`SparseChallenge`] data type.
//!
//! A [`SparseChallenge`] is a sparse polynomial in `F[X] / (X^D + 1)` represented
//! by its non-zero terms. It is the unified output type for every sampling family
//! in [`crate::SparseChallengeConfig`], so downstream arithmetic can stay uniform
//! regardless of how a challenge was sampled.
//!
//! Production challenges are expected to come from [`crate::sample_sparse_challenges`],
//! which constructs values satisfying the invariants below. Methods on this
//! type check cheap shape/range errors needed for memory safety, but they do
//! not re-validate every sampler invariant on the hot path.
//!
//! This module deliberately depends only on `akita-field`; it does not pull in
//! the transcript layer or the sampler. Most consumers of this crate
//! (`akita-types`, `akita-config`, `akita-planner`, `akita-prover`/
//! `akita-verifier` ring-switching, etc.) only ever touch this type and never
//! run the sampler.

use akita_field::{AkitaError, CanonicalField, FieldCore};

/// Sparse polynomial in `F[X]/(X^D+1)` represented by its non-zero terms.
///
/// Sampler invariants:
/// - `positions.len() == coeffs.len()`
/// - all positions are `< D`
/// - positions are unique
/// - all coeffs are non-zero
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SparseChallenge {
    /// Coefficient indices (powers of `X`) where the polynomial is non-zero.
    pub positions: Vec<u32>,
    /// Small integer coefficients at the corresponding positions. Stored
    /// as `i8` since every shipping sampling family caps `|coeff| <= 8`.
    pub coeffs: Vec<i8>,
}

impl SparseChallenge {
    /// Evaluate this challenge against precomputed scalar powers
    /// `alpha_pows = [1, α, α^2, ..., α^{D-1}]`, returning
    /// `Σ_i coeffs[i] · alpha_pows[positions[i]]` in `F`.
    ///
    /// # Errors
    ///
    /// Returns an error if `alpha_pows` does not have length `D`, or if a
    /// term would index outside the supplied powers. This method assumes the
    /// challenge came from [`crate::sample_sparse_challenges`] and therefore
    /// does not re-check uniqueness of positions on the hot path.
    pub fn eval_at_pows<F: FieldCore + CanonicalField, const D: usize>(
        &self,
        alpha_pows: &[F],
    ) -> Result<F, AkitaError> {
        if alpha_pows.len() != D {
            return Err(AkitaError::InvalidSize {
                expected: D,
                actual: alpha_pows.len(),
            });
        }
        if self.positions.len() != self.coeffs.len() {
            return Err(AkitaError::InvalidInput(
                "sparse challenge positions/coeffs length mismatch".to_string(),
            ));
        }

        let mut acc = F::zero();
        for (&pos, &coeff) in self.positions.iter().zip(self.coeffs.iter()) {
            let idx = pos as usize;
            if idx >= D {
                return Err(AkitaError::InvalidInput(format!(
                    "sparse challenge position {idx} out of range for D={D}"
                )));
            }
            if coeff == 0 {
                return Err(AkitaError::InvalidInput(
                    "sparse challenge coefficients must be non-zero".to_string(),
                ));
            }
            acc += F::from_i64(coeff as i64) * alpha_pows[idx];
        }
        Ok(acc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_field::Prime128Offset275;

    type F = Prime128Offset275;
    const D: usize = 4;

    fn alpha_pows() -> [F; D] {
        [
            F::from_u64(1),
            F::from_u64(3),
            F::from_u64(9),
            F::from_u64(27),
        ]
    }

    #[test]
    fn eval_at_pows_evaluates_sparse_terms() {
        let challenge = SparseChallenge {
            positions: vec![0, 2],
            coeffs: vec![1, -2],
        };

        let got = challenge.eval_at_pows::<F, D>(&alpha_pows()).unwrap();
        let expected = F::from_u64(1) + F::from_i64(-2) * F::from_u64(9);

        assert_eq!(got, expected);
    }

    #[test]
    fn eval_at_pows_rejects_wrong_power_count() {
        let challenge = SparseChallenge {
            positions: vec![0],
            coeffs: vec![1],
        };

        let err = challenge
            .eval_at_pows::<F, D>(&alpha_pows()[..D - 1])
            .unwrap_err();

        assert_eq!(
            err,
            AkitaError::InvalidSize {
                expected: D,
                actual: D - 1
            }
        );
    }

    #[test]
    fn eval_at_pows_rejects_out_of_range_position() {
        let challenge = SparseChallenge {
            positions: vec![D as u32],
            coeffs: vec![1],
        };

        let err = challenge.eval_at_pows::<F, D>(&alpha_pows()).unwrap_err();

        assert!(matches!(err, AkitaError::InvalidInput(msg) if msg.contains("out of range")));
    }

    #[test]
    fn eval_at_pows_rejects_mismatched_terms() {
        let challenge = SparseChallenge {
            positions: vec![0, 1],
            coeffs: vec![1],
        };

        let err = challenge.eval_at_pows::<F, D>(&alpha_pows()).unwrap_err();

        assert!(matches!(err, AkitaError::InvalidInput(msg) if msg.contains("length mismatch")));
    }
}
