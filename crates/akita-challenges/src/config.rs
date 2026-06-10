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

/// Minimum min-entropy (bits) for every stage-1 sparse-challenge transcript draw.
///
/// Flat folds sample one such draw per logical block. Tensor folds sample
/// independent left and right factor vectors; each factor is one draw and is
/// reused across many logical blocks (`c_{p,q} = left_p · right_q`). Soundness
/// therefore requires **each draw** to clear this floor, not merely the product
/// `left ⊗ right` summed to 128 bits (a 64+64 split would pass a sum rule but
/// leave each factor brute-forceable).
pub const MIN_FOLD_CHALLENGE_ENTROPY_BITS: u32 = 128;

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
    /// families, so it should be used only when the `L1` mass in this approach
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

    /// `log2` of the number of distinct challenges this family can emit for ring
    /// degree `D` — the (raw) min-entropy of a single sampled challenge.
    ///
    /// Knowledge soundness of the fold needs the challenge set a prover must
    /// guess against to be large: a single sparse challenge with only a handful
    /// of nonzero coordinates has a small support, and `validate::<D>` alone does
    /// **not** rule that out. Use this together with [`Self::validate_min_entropy`]
    /// to enforce a floor.
    ///
    /// Counting (each is a product of independent choices):
    /// - `Uniform { weight, nonzero_coeffs }`:
    ///   `C(D, weight) · |nonzero_coeffs|^weight`.
    /// - `ExactShell { count_mag1, count_mag2 }` with `w = count_mag1 + count_mag2`:
    ///   `C(D, w) · C(w, count_mag1) · 2^w` (place the two magnitude classes, then
    ///   pick a sign per nonzero).
    /// - `BoundedL1Norm`: the `D = 32` preset is *defined* as the smallest L1 cap
    ///   whose support reaches `2^128` (see `sampler::bounded_l1_support`), so we
    ///   report that proven floor rather than re-running the DP here.
    ///
    /// For tensor-shaped folds, this is the support of **one** left or right
    /// factor draw. A logical block challenge `left_p · right_q` multiplies the
    /// supports, but the security floor is applied per draw (see
    /// [`MIN_FOLD_CHALLENGE_ENTROPY_BITS`]).
    pub fn log2_support_bits<const D: usize>(&self) -> f64 {
        fn log2_binom(n: usize, k: usize) -> f64 {
            if k > n {
                return f64::NEG_INFINITY;
            }
            // C(n,k) = prod_{i=1..k} (n-k+i)/i, summed in log space to avoid
            // overflow for the large binomials that arise at production D.
            (1..=k)
                .map(|i| ((n - k + i) as f64 / i as f64).log2())
                .sum()
        }
        match self {
            Self::Uniform {
                weight,
                nonzero_coeffs,
            } => {
                if *weight > D || nonzero_coeffs.is_empty() {
                    return f64::NEG_INFINITY;
                }
                log2_binom(D, *weight) + (*weight as f64) * (nonzero_coeffs.len() as f64).log2()
            }
            Self::ExactShell {
                count_mag1,
                count_mag2,
            } => {
                let w = count_mag1 + count_mag2;
                if w > D {
                    return f64::NEG_INFINITY;
                }
                log2_binom(D, w) + log2_binom(w, *count_mag1) + w as f64
            }
            // The preset is the minimal L1 cap with >= 128-bit support for D=32.
            Self::BoundedL1Norm => 128.0,
        }
    }

    /// Reject challenge families whose single-draw support is below
    /// `required_bits` of min-entropy for ring degree `D`.
    ///
    /// This is intentionally **not** folded into [`Self::validate`]: that check
    /// is a structural well-formedness gate also exercised at tiny test degrees,
    /// whereas the entropy floor is a security-parameter policy callers apply at
    /// config-selection time. The same per-draw floor applies to flat blocks and
    /// to each tensor left/right factor.
    ///
    /// # Errors
    ///
    /// Returns an error when `log2_support_bits::<D>() < required_bits`.
    pub fn validate_min_entropy<const D: usize>(
        &self,
        required_bits: u32,
    ) -> Result<(), &'static str> {
        if self.log2_support_bits::<D>() < f64::from(required_bits) {
            return Err("sparse challenge family has insufficient min-entropy for security floor");
        }
        Ok(())
    }

    /// Runtime ring-dimension dispatch for [`Self::validate_min_entropy`].
    ///
    /// # Errors
    ///
    /// Returns an error when `ring_dim` is unsupported or the per-draw floor fails.
    pub fn validate_min_entropy_for_ring_dim(
        &self,
        ring_dim: usize,
        required_bits: u32,
    ) -> Result<(), &'static str> {
        match ring_dim {
            32 => self.validate_min_entropy::<32>(required_bits),
            64 => self.validate_min_entropy::<64>(required_bits),
            128 => self.validate_min_entropy::<128>(required_bits),
            256 => self.validate_min_entropy::<256>(required_bits),
            _ => Err("unsupported ring dimension for fold-challenge entropy audit"),
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

#[cfg(test)]
mod entropy_tests {
    use super::*;

    #[test]
    fn tiny_shell_is_rejected_at_128_bits() {
        // A near-empty shell has a trivially small support and must be caught.
        let tiny = SparseChallengeConfig::ExactShell {
            count_mag1: 2,
            count_mag2: 0,
        };
        assert!(tiny.log2_support_bits::<32>() < 128.0);
        assert!(tiny.validate_min_entropy::<32>(128).is_err());
    }

    #[test]
    fn tiny_uniform_is_rejected_at_128_bits() {
        let tiny = SparseChallengeConfig::Uniform {
            weight: 1,
            nonzero_coeffs: vec![-1, 1],
        };
        assert!(tiny.validate_min_entropy::<32>(128).is_err());
    }

    #[test]
    fn full_shell_clears_128_bits() {
        // The canonical d=64 shell (writeup App. C) clears the floor.
        let shell = SparseChallengeConfig::ExactShell {
            count_mag1: 30,
            count_mag2: 12,
        };
        assert!(shell.log2_support_bits::<64>() >= 128.0);
        assert!(shell.validate_min_entropy::<64>(128).is_ok());
    }

    #[test]
    fn bounded_l1_preset_clears_128_bits() {
        let preset = SparseChallengeConfig::BoundedL1Norm;
        assert!(preset.validate_min_entropy::<32>(128).is_ok());
    }

    #[test]
    fn tensor_floor_is_per_draw_not_product_budget() {
        // One mag-1 coeff at D=4 has ~3 bits per draw. A sum rule would accept
        // 64+64; production requires each draw (and each tensor factor) to clear
        // 128 bits independently.
        let weak = SparseChallengeConfig::ExactShell {
            count_mag1: 1,
            count_mag2: 0,
        };
        let per_draw = weak.log2_support_bits::<4>();
        assert!(per_draw < 128.0);
        assert!(weak.validate_min_entropy::<4>(128).is_err());
        // Logical-block product entropy would be 2 * per_draw (informational).
        assert!((per_draw + per_draw - 2.0 * per_draw).abs() < 1e-9);
    }

    #[test]
    fn log2_support_matches_small_closed_form() {
        // ExactShell, D=4, one mag-1 coeff: C(4,1)*C(1,1)*2^1 = 8 -> 3 bits.
        let cfg = SparseChallengeConfig::ExactShell {
            count_mag1: 1,
            count_mag2: 0,
        };
        assert!((cfg.log2_support_bits::<4>() - 3.0).abs() < 1e-9);
        // Uniform, D=4, weight 2, alphabet {-1,1}: C(4,2)*2^2 = 24.
        let uni = SparseChallengeConfig::Uniform {
            weight: 2,
            nonzero_coeffs: vec![-1, 1],
        };
        assert!((uni.log2_support_bits::<4>() - 24.0_f64.log2()).abs() < 1e-9);
    }
}
