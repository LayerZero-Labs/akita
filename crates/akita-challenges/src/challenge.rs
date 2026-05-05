//! The [`SparseChallenge`] data type.
//!
//! A [`SparseChallenge`] is a sparse polynomial in `F[X] / (X^D + 1)` represented
//! by its non-zero terms. It is the unified output type for every sampling family
//! in [`crate::SparseChallengeConfig`], so downstream arithmetic can stay uniform
//! regardless of how a challenge was sampled.
//!
//! This module deliberately depends only on `akita-algebra` and `akita-field`;
//! it does not pull in the transcript layer or the sampler. Most consumers of
//! this crate (`akita-types`, `akita-config`, `akita-planner`,
//! `akita-prover`/`akita-verifier` ring-switching, etc.) only ever touch this
//! type and never run the sampler.

use akita_algebra::ring::CyclotomicRing;
use akita_field::AkitaError;
use akita_field::{CanonicalField, FieldCore};

/// Sparse polynomial in `F[X]/(X^D+1)` represented by its non-zero terms.
///
/// Invariants:
/// - `positions.len() == coeffs.len()`
/// - all positions are `< D`
/// - positions are unique
/// - all coeffs are non-zero
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SparseChallenge {
    /// Coefficient indices (powers of `X`) where the polynomial is non-zero.
    pub positions: Vec<u32>,
    /// Small integer coefficients at the corresponding positions. Held as
    /// `i8` since every supported [`crate::SparseChallengeConfig`] family
    /// caps `|coeff|` at `8` (uniform), `2` (exact-shell), or `M = 8`
    /// (bounded-`L1`); all fit comfortably in `[-128, 127]`.
    pub coeffs: Vec<i8>,
}

impl SparseChallenge {
    /// Convert to a dense ring element by placing coefficients in the canonical
    /// coefficient basis.
    ///
    /// # Errors
    ///
    /// Returns an error if the sparse representation violates structural
    /// invariants: mismatched `positions`/`coeffs` lengths, a zero coefficient,
    /// an out-of-range position, or a duplicate position.
    pub fn to_dense<F: FieldCore + CanonicalField, const D: usize>(
        &self,
    ) -> Result<CyclotomicRing<F, D>, &'static str> {
        if self.positions.len() != self.coeffs.len() {
            return Err("positions and coeffs must have same length");
        }
        let mut out = [F::zero(); D];
        let mut seen = vec![false; D];
        for (&pos, &c) in self.positions.iter().zip(self.coeffs.iter()) {
            if c == 0 {
                return Err("coeffs must not contain 0");
            }
            let p = pos as usize;
            if p >= D {
                return Err("position out of range");
            }
            if seen[p] {
                return Err("positions must be unique");
            }
            seen[p] = true;
            out[p] += F::from_i64(c as i64);
        }
        Ok(CyclotomicRing::from_coefficients(out))
    }

    /// Evaluate this challenge against precomputed scalar powers
    /// `alpha_pows = [1, α, α^2, ..., α^{D-1}]`, returning
    /// `Σ_i coeffs[i] · alpha_pows[positions[i]]` in `F`.
    ///
    /// # Errors
    ///
    /// Returns an error when `alpha_pows` does not have length `D`.
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

        debug_assert_eq!(self.positions.len(), self.coeffs.len());

        let mut acc = F::zero();
        for (&pos, &coeff) in self.positions.iter().zip(self.coeffs.iter()) {
            let idx = pos as usize;
            debug_assert!(idx < D);
            debug_assert_ne!(coeff, 0);
            acc += F::from_i64(coeff as i64) * alpha_pows[idx];
        }
        Ok(acc)
    }
}
