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
use rand_core::RngCore;

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
    ///
    /// # Errors
    ///
    /// Returns an error if `weight > D`, if `nonzero_coeffs` is empty, or if it
    /// contains `0`.
    pub fn validate<const D: usize>(&self) -> Result<(), &'static str> {
        if self.weight > D {
            return Err("weight must be <= ring degree D");
        }
        if self.nonzero_coeffs.is_empty() {
            return Err("nonzero_coeffs must be non-empty");
        }
        if self.nonzero_coeffs.contains(&0) {
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
    ///
    /// # Errors
    ///
    /// Returns an error if lengths mismatch, if any coefficient is zero, if any
    /// position is out of range, or if positions contain duplicates.
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
    ///
    /// # Errors
    ///
    /// Returns an error if the sparse representation violates structural invariants.
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
    ///
    /// # Errors
    ///
    /// Returns an error if structural invariants fail or if `alpha_pows.len() != D`.
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

/// Sample a dense ternary ring element with coefficients in `{-1, 0, 1}`.
///
/// Distribution matches Labrador C's ternary nibble LUT (`0xA815`), yielding
/// probabilities `5/16, 6/16, 5/16` for `-1, 0, 1` respectively.
pub fn sample_ternary<F: FieldCore + CanonicalField, R: RngCore, const D: usize>(
    rng: &mut R,
) -> CyclotomicRing<F, D> {
    const LUT: u16 = 0xA815;
    let mut coeffs = [F::zero(); D];
    let mut i = 0usize;
    while i < D {
        let byte = (rng.next_u32() & 0xFF) as u8;
        let lo = (((LUT >> (byte & 0x0F)) & 0x3) as i16) - 1;
        coeffs[i] = F::from_i64(lo as i64);
        i += 1;
        if i < D {
            let hi = (((LUT >> (byte >> 4)) & 0x3) as i16) - 1;
            coeffs[i] = F::from_i64(hi as i64);
            i += 1;
        }
    }
    CyclotomicRing::from_coefficients(coeffs)
}

/// Sample a dense quaternary ring element with coefficients in `{-2, -1, 0, 1}`.
///
/// Coefficients are sampled uniformly from two-bit chunks and shifted by `-2`.
pub fn sample_quaternary<F: FieldCore + CanonicalField, R: RngCore, const D: usize>(
    rng: &mut R,
) -> CyclotomicRing<F, D> {
    let mut coeffs = [F::zero(); D];
    let mut i = 0usize;
    while i < D {
        let bits = rng.next_u32();
        for lane in 0..16 {
            if i >= D {
                break;
            }
            let val = (((bits >> (2 * lane)) & 0x3) as i16) - 2;
            coeffs[i] = F::from_i64(val as i64);
            i += 1;
        }
    }
    CyclotomicRing::from_coefficients(coeffs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::fields::Fp64;
    use rand::{rngs::StdRng, SeedableRng};

    type F = Fp64<4294967197>;
    const D: usize = 64;

    fn centered(v: F) -> i64 {
        let q = (-F::one()).to_canonical_u128() + 1;
        let c = v.to_canonical_u128();
        if c > q / 2 {
            -((q - c) as i64)
        } else {
            c as i64
        }
    }

    #[test]
    fn ternary_sampler_range() {
        let mut rng = StdRng::seed_from_u64(123);
        let sample = sample_ternary::<F, _, D>(&mut rng);
        for &c in sample.coefficients().iter() {
            assert!(matches!(centered(c), -1..=1));
        }
    }

    #[test]
    fn quaternary_sampler_range() {
        let mut rng = StdRng::seed_from_u64(456);
        let sample = sample_quaternary::<F, _, D>(&mut rng);
        for &c in sample.coefficients().iter() {
            assert!(matches!(centered(c), -2..=1));
        }
    }
}
