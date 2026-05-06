//! The [`SparseChallenge`] data type.
//!
//! A [`SparseChallenge`] is a sparse polynomial in `F[X] / (X^D + 1)` represented
//! by its non-zero terms. It is the unified output type for every sampling family
//! in [`crate::SparseChallengeConfig`], so downstream arithmetic can stay uniform
//! regardless of how a challenge was sampled.
//!
//! This module deliberately depends only on `akita-field`; it does not pull in
//! the transcript layer or the sampler. Most consumers of this crate
//! (`akita-types`, `akita-config`, `akita-planner`, `akita-prover`/
//! `akita-verifier` ring-switching, etc.) only ever touch this type and never
//! run the sampler.

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
    /// Small integer coefficients at the corresponding positions. Stored
    /// as `i8` since every shipping sampling family caps `|coeff| <= 8`.
    pub coeffs: Vec<i8>,
}

impl SparseChallenge {
    /// Evaluate this challenge against precomputed scalar powers
    /// `alpha_pows = [1, α, α^2, ..., α^{D-1}]`, returning
    /// `Σ_i coeffs[i] · alpha_pows[positions[i]]` in `F`.
    pub fn eval_at_pows<F: FieldCore + CanonicalField, const D: usize>(
        &self,
        alpha_pows: &[F],
    ) -> F {
        debug_assert_eq!(alpha_pows.len(), D);
        debug_assert_eq!(self.positions.len(), self.coeffs.len());

        let mut acc = F::zero();
        for (&pos, &coeff) in self.positions.iter().zip(self.coeffs.iter()) {
            let idx = pos as usize;
            debug_assert!(idx < D);
            debug_assert_ne!(coeff, 0);
            acc += F::from_i64(coeff as i64) * alpha_pows[idx];
        }
        acc
    }
}
