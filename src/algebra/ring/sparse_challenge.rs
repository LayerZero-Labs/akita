//! Sparse ring challenges for cyclotomic protocols.
//!
//! Many lattice protocols sample "short/sparse" ring challenges whose coefficients
//! are mostly zero and whose non-zero coefficients come from a tiny integer alphabet
//! (e.g. `{±1}` or `{±1,±2}`), with a fixed Hamming weight `ω`.
//!
//! This module provides a minimal representation that is:
//! - independent of any specific protocol (Hachi/Greyhound/SuperNeo, etc.),
//! - easy to sample deterministically from Fiat–Shamir at the protocol layer,
//! - and efficient to evaluate at a point `α` using precomputed powers.

use super::CyclotomicRing;
use crate::{CanonicalField, FieldCore};

/// Configuration for sampling a sparse challenge.
///
/// This intentionally avoids redundant knobs: the distribution is determined by:
/// - exact `weight` (Hamming weight),
/// - and a list of allowed **non-zero** integer coefficients.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SparseChallengeConfig {
    /// Exact Hamming weight ω.
    pub weight: usize,
    /// Allowed non-zero coefficients (small signed integers).
    ///
    /// Examples:
    /// - `{±1}`: `vec![-1, 1]`
    /// - `{±1,±2}`: `vec![-2, -1, 1, 2]`
    pub nonzero_coeffs: Vec<i16>,
}

impl SparseChallengeConfig {
    /// Validate basic invariants for a given ring degree `D`.
    pub fn validate<const D: usize>(&self) -> Result<(), &'static str> {
        if self.weight > D {
            return Err("weight must be <= ring degree D");
        }
        if self.nonzero_coeffs.is_empty() {
            return Err("nonzero_coeffs must be non-empty");
        }
        if self.nonzero_coeffs.iter().any(|&c| c == 0) {
            return Err("nonzero_coeffs must not contain 0");
        }
        Ok(())
    }
}

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
    /// Small integer coefficients at the corresponding positions.
    pub coeffs: Vec<i16>,
}

impl SparseChallenge {
    /// Construct an empty (all-zero) challenge.
    #[inline]
    pub fn zero() -> Self {
        Self {
            positions: Vec::new(),
            coeffs: Vec::new(),
        }
    }

    /// Number of non-zero coefficients (Hamming weight).
    #[inline]
    pub fn hamming_weight(&self) -> usize {
        debug_assert_eq!(self.positions.len(), self.coeffs.len());
        self.positions.len()
    }

    /// ℓ₁ norm over integers: `Σ |coeff_i|`.
    #[inline]
    pub fn l1_norm(&self) -> u64 {
        self.coeffs
            .iter()
            .map(|&c| (c as i32).unsigned_abs() as u64)
            .sum()
    }

    /// Validate structural invariants for a ring degree `D`.
    pub fn validate<const D: usize>(&self) -> Result<(), &'static str> {
        if self.positions.len() != self.coeffs.len() {
            return Err("positions and coeffs must have same length");
        }
        // Check coeffs are non-zero and positions are in range + unique.
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
        }
        Ok(())
    }

    /// Convert to a dense ring element by placing coefficients in the canonical
    /// coefficient basis.
    pub fn to_dense<F: FieldCore + CanonicalField, const D: usize>(
        &self,
    ) -> Result<CyclotomicRing<F, D>, &'static str> {
        self.validate::<D>()?;
        let mut out = [F::zero(); D];
        for (&pos, &c) in self.positions.iter().zip(self.coeffs.iter()) {
            out[pos as usize] = out[pos as usize] + F::from_i64(c as i64);
        }
        Ok(CyclotomicRing::from_coefficients(out))
    }

    /// Evaluate this sparse polynomial at `α` in `E`, given precomputed powers
    /// `[α^0, α^1, ..., α^{D-1}]`.
    ///
    /// This is `O(weight)` and is intended to be used for verifier-side oracles
    /// where `D` may be large but `weight` is small.
    pub fn eval_at_alpha<F, E, const D: usize>(&self, alpha_pows: &[E]) -> Result<E, &'static str>
    where
        F: FieldCore + CanonicalField,
        E: FieldCore + crate::algebra::fields::LiftBase<F>,
    {
        self.validate::<D>()?;
        if alpha_pows.len() != D {
            return Err("alpha_pows length mismatch");
        }
        let mut acc = E::zero();
        for (&pos, &c) in self.positions.iter().zip(self.coeffs.iter()) {
            let coeff_f = F::from_i64(c as i64);
            acc = acc + (E::lift_base(coeff_f) * alpha_pows[pos as usize]);
        }
        Ok(acc)
    }
}
