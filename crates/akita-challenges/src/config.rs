//! Sampling-family configuration for [`crate::SparseChallenge`].
//!
//! Many lattice protocols sample "short/sparse" ring challenges whose coefficients
//! are mostly zero and whose non-zero coefficients come from a tiny integer
//! alphabet (e.g. `{±1}` or `{±1,±2}`). [`SparseChallengeConfig`] enumerates the
//! supported families and exposes the policy questions that the rest of the
//! workspace asks of a configured challenge:
//!
//! - [`SparseChallengeConfig::l1_mass`] — worst-case integer `L1` mass; drives
//!   folded-witness bounds in `akita-config` / `akita-planner`.
//! - [`SparseChallengeConfig::max_abs_coeff`] — largest `|c|` that can appear.
//! - [`SparseChallengeConfig::domain_separator_bytes`] — canonical byte tag
//!   absorbed by the sampler before drawing the PRG seed.
//! - [`SparseChallengeConfig::validate`] — sanity-check the family parameters
//!   against a chosen ring degree `D`.
//!
//! All families produce the same sampled output type [`crate::SparseChallenge`].
//! The actual sampler that turns this config plus a transcript into challenges
//! lives in [`crate::sampler`]; this file is policy-only and has no transcript
//! or PRG dependency.

/// Specifies the distribution from which sparse ring challenges are sampled.
///
/// A sparse challenge is a "short" element of the cyclotomic ring
/// `F[X]/(X^D + 1)` with few non-zero coefficients drawn from a small integer
/// alphabet. Different families trade off challenge entropy against the
/// resulting coefficient mass, which in turn affects the folded witness bounds
/// used by the protocol.
///
/// All families produce the same sampled output type [`crate::SparseChallenge`],
/// so the downstream arithmetic is uniform regardless of which family was used.
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
        /// `[-2, -1, 1, 2]`. Each entry must satisfy `|c| <= 127`; in
        /// practice every shipping preset uses values with `|c| <= 8`.
        nonzero_coeffs: Vec<i8>,
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

    /// Truncated-`2^128` sample over the bounded-`L1` ball
    /// `{ c in Z^D : ||c||_inf <= max_abs_coeff and ||c||_1 <= l1_bound }`.
    ///
    /// Unlike the fixed-shape families, the realized Hamming weight is variable;
    /// the worst-case nonzero count is `min(D, l1_bound)`. The challenge keeps
    /// the dense `L_inf` bound `max_abs_coeff` and uses `l1_bound` as the true
    /// worst-case coefficient `L1` mass for protocol sizing.
    ///
    /// The bounded-`L1` family is sampled via the truncated-`2^128`
    /// rank-unranking decoder in [`crate::sampler::bounded_l1`].
    BoundedL1Ball {
        /// Coefficient `L_inf` bound `M`. Each conceptual dense coefficient is
        /// constrained to `[-M, M]`.
        max_abs_coeff: u8,
        /// Coefficient `L1` bound `B`. The sampled dense vector satisfies
        /// `sum_i |c_i| <= B`.
        l1_bound: u16,
    },
}

fn validate_uniform_coeffs(nonzero_coeffs: &[i8]) -> Result<(), &'static str> {
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
            Self::ExactShell {
                count_mag1,
                count_mag2,
            } => {
                out.push(1);
                out.extend_from_slice(&(*count_mag1 as u64).to_le_bytes());
                out.extend_from_slice(&(*count_mag2 as u64).to_le_bytes());
            }
            Self::BoundedL1Ball {
                max_abs_coeff,
                l1_bound,
            } => {
                out.push(2);
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
