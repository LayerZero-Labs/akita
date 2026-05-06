//! Sampling-family configuration for [`crate::SparseChallenge`].
//!
//! Many lattice protocols sample "short/sparse" ring challenges whose coefficients
//! are mostly zero and whose non-zero coefficients come from a tiny integer
//! alphabet (e.g. `{±1}` or `{±1,±2}`).
//!
//! All families produce the same output type [`crate::SparseChallenge`].
//! The actual sampler that turns this config plus a transcript into challenges
//! lives in [`crate::sampler`]; this file is policy-only and has no transcript
//! or PRG dependency.

use crate::sampler::bounded_l1::{COEFFS_BOUND_32, D_32, MAX_L1_NORM_32};

/// Specifies the distribution from which sparse ring challenges are sampled.
///
/// Different families trade off challenge entropy against the
/// resulting coefficient mass, which in turn affects the folded witness bounds
/// used by the protocol.
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

    /// Bounded-`L1` production preset for `D = 32`.
    ///
    /// The preset fixes the coefficient bound `||c||_inf <= 8`, then chooses
    /// the smallest `L1` bound whose challenge space has at least 128 bits of
    /// support. For `D = 32`, that minimum is `||c||_1 <= 121`.
    /// Sampling draws a 128-bit rank and maps it into this bounded challenge
    /// space with the crate-internal `sampler::bounded_l1` decoder.
    ///
    /// This sampler is slower than the fixed-shape `Uniform` and `ExactShell`
    /// families, so it should be used only when the `L1` mass in this appraoch
    /// is smaller than the other two.
    BoundedL1Norm,
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
    /// Worst-case `L1` norm of the sampled coefficients.
    #[inline]
    pub fn l1_norm(&self) -> usize {
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
            Self::BoundedL1Norm => MAX_L1_NORM_32,
        }
    }

    /// Worst-case `L_infinity` norm of the sampled coefficients.
    #[inline]
    pub fn infinity_norm(&self) -> u32 {
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
            Self::BoundedL1Norm => COEFFS_BOUND_32 as u32,
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
            Self::BoundedL1Norm => {
                out.push(2);
                out.extend_from_slice(&(COEFFS_BOUND_32 as u64).to_le_bytes());
                out.extend_from_slice(&(MAX_L1_NORM_32 as u64).to_le_bytes());
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
            Self::BoundedL1Norm => {
                if D != D_32 {
                    return Err("BoundedL1Norm: only D = 32 is supported");
                }
                Ok(())
            }
        }
    }
}
