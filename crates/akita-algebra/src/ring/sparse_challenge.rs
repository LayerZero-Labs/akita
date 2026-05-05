//! Sparse ring challenges for cyclotomic protocols.
//!
//! Many lattice protocols sample "short/sparse" ring challenges whose coefficients
//! are mostly zero and whose non-zero coefficients come from a tiny integer alphabet
//! (e.g. `{±1}` or `{±1,±2}`).
//!
//! This module provides a minimal representation that is:
//! - independent of any specific protocol (Akita/Greyhound/SuperNeo, etc.),
//! - easy to sample deterministically from Fiat–Shamir at the protocol layer,
//! - and efficient to evaluate at a point `α` using precomputed powers.

use super::CyclotomicRing;
use crate::{CanonicalField, FieldCore};
use akita_field::fields::LiftBase;
use rand_core::RngCore;

/// Specifies the distribution from which sparse ring challenges are sampled.
///
/// A sparse challenge is a "short" element of the cyclotomic ring
/// `F[X]/(X^D + 1)` with few non-zero coefficients drawn from a small integer
/// alphabet. Different families trade off challenge entropy against the
/// resulting coefficient mass, which in turn affects the folded witness bounds
/// used by the protocol.
///
/// All families produce the same sampled output type [`SparseChallenge`], so the
/// downstream arithmetic is uniform regardless of which family was used.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SparseChallengeConfig {
    /// Uniform sparse challenge over the full ring.
    ///
    /// Sampling chooses `weight` distinct positions from `0..D`, then assigns
    /// each position a coefficient drawn uniformly from `nonzero_coeffs`.
    ///
    /// The worst-case L1 mass is
    /// `weight * max(|c| for c in nonzero_coeffs)`.
    Uniform {
        /// Exact number of non-zero coefficients.
        weight: usize,
        /// Allowed small non-zero coefficients, e.g. `[-1, 1]` or
        /// `[-2, -1, 1, 2]`.
        nonzero_coeffs: Vec<i16>,
    },

    /// Split-ring sparse challenge with an independent budget in each parity
    /// half.
    ///
    /// The `D` coefficient positions are partitioned into the even indices and
    /// the odd indices, each of size `D / 2`. In each half, sampling chooses
    /// `half_weight` distinct positions, assigns each a random sign, and then
    /// chooses a shell with between `0` and `max_mag2_per_half` magnitude-2
    /// entries uniformly over the full union of shells
    /// `C_{half_weight,<=max_mag2_per_half}` before upgrading that many
    /// positions from magnitude 1 to magnitude 2. The two halves are
    /// interleaved back into one ring element.
    ///
    /// The worst-case L1 mass is `2 * (half_weight + max_mag2_per_half)`.
    SplitRing {
        /// Number of active positions in each parity half.
        half_weight: usize,
        /// Maximum number of magnitude-2 coefficients in each parity half.
        max_mag2_per_half: usize,
    },

    /// Exact-shell sparse challenge over the full ring.
    ///
    /// Sampling chooses `count_mag1 + count_mag2` distinct positions from
    /// `0..D`, assigns `count_mag1` of them a random sign with magnitude 1, and
    /// assigns the remaining `count_mag2` a random sign with magnitude 2.
    ///
    /// The L1 mass is exact: `count_mag1 + 2 * count_mag2`.
    ExactShell {
        /// Number of coefficients with magnitude 1.
        count_mag1: usize,
        /// Number of coefficients with magnitude 2.
        count_mag2: usize,
    },

    /// Exactly uniform sample over the bounded-`L1` ball
    /// `{ c in Z^D : ||c||_inf <= max_abs_coeff and ||c||_1 <= l1_bound }`.
    ///
    /// Unlike the fixed-shape families, the realized Hamming weight is variable;
    /// the worst-case nonzero count is `min(D, l1_bound)`. The challenge keeps
    /// the dense `L_inf` bound `max_abs_coeff` and uses `l1_bound` as the true
    /// worst-case coefficient `L1` mass for protocol sizing.
    ///
    /// The bounded-`L1` family is sampled via an exact streaming DP-decoder over
    /// suffix counts, see `akita-challenges` for the concrete sampler.
    BoundedL1Ball {
        /// Coefficient `L_inf` bound `M`. Each conceptual dense coefficient is
        /// constrained to `[-M, M]`.
        max_abs_coeff: u8,
        /// Coefficient `L1` bound `B`. The sampled dense vector satisfies
        /// `sum_i |c_i| <= B`.
        l1_bound: u16,
    },
}

fn validate_uniform_coeffs(nonzero_coeffs: &[i16]) -> Result<(), &'static str> {
    if nonzero_coeffs.is_empty() {
        return Err("nonzero_coeffs must be non-empty");
    }
    if nonzero_coeffs.contains(&0) {
        return Err("nonzero_coeffs must not contain 0");
    }
    Ok(())
}

impl SparseChallengeConfig {
    /// Worst-case sum of absolute values of the sampled coefficients.
    #[inline]
    pub fn l1_mass(&self) -> usize {
        match self {
            Self::Uniform {
                weight,
                nonzero_coeffs,
            } => {
                let max_coeff = nonzero_coeffs
                    .iter()
                    .map(|c| c.unsigned_abs() as usize)
                    .max()
                    .unwrap_or(0);
                weight.saturating_mul(max_coeff)
            }
            Self::SplitRing {
                half_weight,
                max_mag2_per_half,
            } => 2 * (half_weight + max_mag2_per_half),
            Self::ExactShell {
                count_mag1,
                count_mag2,
            } => count_mag1 + 2 * count_mag2,
            Self::BoundedL1Ball { l1_bound, .. } => *l1_bound as usize,
        }
    }

    /// Largest absolute value of any coefficient that can appear in a
    /// challenge sampled from this family.
    #[inline]
    pub fn max_abs_coeff(&self) -> u32 {
        match self {
            Self::Uniform { nonzero_coeffs, .. } => nonzero_coeffs
                .iter()
                .map(|c| c.unsigned_abs() as u32)
                .max()
                .unwrap_or(0),
            Self::SplitRing {
                max_mag2_per_half, ..
            } => {
                if *max_mag2_per_half > 0 {
                    2
                } else {
                    1
                }
            }
            Self::ExactShell { count_mag2, .. } => {
                if *count_mag2 > 0 {
                    2
                } else {
                    1
                }
            }
            Self::BoundedL1Ball { max_abs_coeff, .. } => *max_abs_coeff as u32,
        }
    }

    /// Total number of non-zero coefficients sampled from this family, when
    /// the family fixes the Hamming weight.
    ///
    /// # Panics
    ///
    /// Panics for [`SparseChallengeConfig::BoundedL1Ball`], because that family
    /// has variable realized Hamming weight; use
    /// [`SparseChallengeConfig::max_hamming_weight`] for a tight degree-aware
    /// bound.
    #[inline]
    pub fn hamming_weight(&self) -> usize {
        match self {
            Self::Uniform { weight, .. } => *weight,
            Self::SplitRing { half_weight, .. } => 2 * half_weight,
            Self::ExactShell {
                count_mag1,
                count_mag2,
            } => count_mag1 + count_mag2,
            Self::BoundedL1Ball { .. } => {
                panic!(
                    "SparseChallengeConfig::hamming_weight is not defined for the variable-weight \
                     BoundedL1Ball family; use max_hamming_weight::<D>() for a degree-aware bound"
                )
            }
        }
    }

    /// Tight worst-case nonzero coefficient count for ring degree `D`.
    ///
    /// For fixed-shape families this matches [`SparseChallengeConfig::hamming_weight`].
    /// For the variable-weight [`SparseChallengeConfig::BoundedL1Ball`] family,
    /// returns `min(D, l1_bound)`.
    #[inline]
    pub fn max_hamming_weight<const D: usize>(&self) -> usize {
        match self {
            Self::Uniform { weight, .. } => *weight,
            Self::SplitRing { half_weight, .. } => 2 * half_weight,
            Self::ExactShell {
                count_mag1,
                count_mag2,
            } => count_mag1 + count_mag2,
            Self::BoundedL1Ball { l1_bound, .. } => D.min(*l1_bound as usize),
        }
    }

    /// Canonical byte encoding used for transcript domain separation.
    #[inline]
    pub fn domain_separator_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        match self {
            Self::Uniform {
                weight,
                nonzero_coeffs,
            } => {
                out.push(0);
                out.extend_from_slice(&(*weight as u64).to_le_bytes());
                out.extend_from_slice(&(nonzero_coeffs.len() as u64).to_le_bytes());
                for &c in nonzero_coeffs {
                    out.extend_from_slice(&c.to_le_bytes());
                }
            }
            Self::SplitRing {
                half_weight,
                max_mag2_per_half,
            } => {
                out.push(1);
                out.extend_from_slice(&(*half_weight as u64).to_le_bytes());
                out.extend_from_slice(&(*max_mag2_per_half as u64).to_le_bytes());
            }
            Self::ExactShell {
                count_mag1,
                count_mag2,
            } => {
                out.push(2);
                out.extend_from_slice(&(*count_mag1 as u64).to_le_bytes());
                out.extend_from_slice(&(*count_mag2 as u64).to_le_bytes());
            }
            Self::BoundedL1Ball {
                max_abs_coeff,
                l1_bound,
            } => {
                out.push(3);
                out.extend_from_slice(&(*max_abs_coeff as u64).to_le_bytes());
                out.extend_from_slice(&(*l1_bound as u64).to_le_bytes());
            }
        }
        out
    }

    /// Validate basic invariants for a given ring degree `D`.
    ///
    /// # Errors
    ///
    /// Returns an error if the family parameters are inconsistent with ring
    /// degree `D`.
    pub fn validate<const D: usize>(&self) -> Result<(), &'static str> {
        match self {
            Self::Uniform {
                weight,
                nonzero_coeffs,
            } => {
                if *weight > D {
                    return Err("weight must be <= ring degree D");
                }
                validate_uniform_coeffs(nonzero_coeffs)
            }
            Self::SplitRing {
                half_weight,
                max_mag2_per_half,
            } => {
                if D == 0 || !D.is_multiple_of(2) {
                    return Err("split-ring family requires an even ring degree");
                }
                if *half_weight > D / 2 {
                    return Err("half_weight must be <= D / 2");
                }
                if *max_mag2_per_half > *half_weight {
                    return Err("max_mag2_per_half must be <= half_weight");
                }
                Ok(())
            }
            Self::ExactShell {
                count_mag1,
                count_mag2,
            } => {
                if count_mag1
                    .checked_add(*count_mag2)
                    .is_none_or(|weight| weight > D)
                {
                    return Err("count_mag1 + count_mag2 must be <= ring degree D");
                }
                Ok(())
            }
            Self::BoundedL1Ball {
                max_abs_coeff,
                l1_bound,
            } => {
                if *max_abs_coeff < 1 {
                    return Err("BoundedL1Ball: max_abs_coeff must be >= 1");
                }
                if *l1_bound < 1 {
                    return Err("BoundedL1Ball: l1_bound must be >= 1");
                }
                let max_l1 = (D as u64).saturating_mul(*max_abs_coeff as u64);
                if (*l1_bound as u64) > max_l1 {
                    return Err("BoundedL1Ball: l1_bound must be <= D * max_abs_coeff");
                }
                Ok(())
            }
        }
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
            out[pos as usize] += F::from_i64(c as i64);
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
        E: FieldCore + LiftBase<F>,
    {
        self.validate::<D>()?;
        if alpha_pows.len() != D {
            return Err("alpha_pows length mismatch");
        }
        let mut acc = E::zero();
        for (&pos, &c) in self.positions.iter().zip(self.coeffs.iter()) {
            let coeff_f = F::from_i64(c as i64);
            acc += E::lift_base(coeff_f) * alpha_pows[pos as usize];
        }
        Ok(acc)
    }
}

/// Sample a dense ternary ring element with coefficients in `{-1, 0, 1}`.
///
/// Distribution matches the ternary nibble LUT (`0xA815`), yielding
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
