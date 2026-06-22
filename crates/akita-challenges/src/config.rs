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

/// Production D=64 exact shell `(31, 11)` with operator-norm cap `Gamma = 18`.
pub const D64_PRODUCTION_EXACT_SHELL_MAG1: usize = 31;
pub const D64_PRODUCTION_EXACT_SHELL_MAG2: usize = 11;
pub const D64_PRODUCTION_OPERATOR_NORM_THRESHOLD: u32 = 18;

/// Certified floor on `Pr[gamma(c) <= 18]` for the production `(31, 11)` shell at
/// `D = 64`. Source: `lattice-jolt/experiments/operator-norm-acceptance/
/// cert_d64_a31_b11_gamma18.json` (`p0 ≈ 0.2349106543`). The rational `117/500`
/// undershoots `p0` so tail-bound `ln(1/p)` sizing stays conservative.
pub const D64_EXACT_SHELL_OP_NORM_ACCEPT_NUM: u128 = 117;
pub const D64_EXACT_SHELL_OP_NORM_ACCEPT_DEN: u128 = 500;

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

    /// Exact-shell sparse challenge over the full ring, optionally paired with
    /// operator-norm rejection by the sampler caller.
    ///
    /// Sampling chooses `count_mag1 + count_mag2` distinct positions from
    /// `0..D`, assigns `count_mag1` of them a random sign with magnitude 1, and
    /// assigns the remaining `count_mag2` a random sign with magnitude 2.
    ///
    /// The L1 mass is exact: `count_mag1 + 2 * count_mag2`. When a caller enables
    /// operator-norm rejection, a sampled candidate is retained only if its
    /// negacyclic operator norm satisfies `gamma_D(c) <= operator_norm_threshold`
    /// (the crate-internal certified `OpNormTable` predicate); otherwise it is
    /// rejected and the next candidate is drawn from the same transcript-derived
    /// stream. When rejection is disabled, the sampler draws from the full shell
    /// and only the deterministic `||c||_1` cap is guaranteed.
    ExactShell {
        /// Number of coefficients with magnitude 1.
        count_mag1: usize,
        /// Number of coefficients with magnitude 2.
        count_mag2: usize,
        /// Operator-norm acceptance cap `T`: a candidate is kept only when the
        /// certified predicate proves `gamma_D(c) <= T`. Setting
        /// `T >= count_mag1 + 2 * count_mag2` (the exact `||c||_1`) disables
        /// rejection, since `gamma_D(c) <= ||c||_1` always holds; the family
        /// then degrades to the deterministic `||c||_1` operator-norm bound.
        operator_norm_threshold: u32,
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
                ..
            } => count_mag1 + 2 * count_mag2,
            Self::BoundedL1Norm => MAX_L1_NORM_32,
        }
    }

    /// Per-challenge operator-norm cap `Gamma` guaranteed on every accepted draw.
    ///
    /// For [`Self::ExactShell`] the sampler rejects any candidate with
    /// `gamma_D(c) > T`, so the cap is `min(T, ||c||_1)` (the `min` makes a
    /// `T >= ||c||_1` setting collapse cleanly to the always-true deterministic
    /// bound `gamma_D(c) <= ||c||_1`, i.e. no rejection). A-role collision
    /// sizing reads [`crate::ChallengeShape::effective_operator_norm_cap`]
    /// (flat `Gamma`, tensor `Gamma^2`). Fold-digit `beta_inf` still uses
    /// [`Self::l1_norm`] / [`crate::ChallengeShape::effective_l1_mass`].
    ///
    /// To revert the operator-norm policy for a preset, set its `ExactShell`
    /// threshold to `>= ||c||_1`; the cap then becomes `||c||_1` and sampling
    /// performs no rejection.
    #[inline]
    pub fn operator_norm_cap(&self) -> u32 {
        match self {
            Self::ExactShell {
                count_mag1,
                count_mag2,
                operator_norm_threshold,
            } => {
                let l1 = (count_mag1 + 2 * count_mag2) as u32;
                (*operator_norm_threshold).min(l1)
            }
            Self::Uniform { .. } | Self::BoundedL1Norm => self.l1_norm() as u32,
        }
    }

    /// `true` when [`Self::ExactShell`] rejection is binding (`T < ||c||_1`).
    ///
    /// Shared by the sampler oracle and tail-bound acceptance-probability lookup.
    #[inline]
    pub fn operator_norm_rejection_binds(&self) -> bool {
        matches!(self, Self::ExactShell { .. }) && self.operator_norm_cap() < self.l1_norm() as u32
    }

    /// Rational lower bound on `Pr[gamma(c) <= T]` for tail-bound sizing.
    ///
    /// Returns `(1, 1)` when [`Self::operator_norm_rejection_binds`] is false.
    /// When binding, returns a preset-specific certified floor (no live oracle).
    ///
    /// # Errors
    ///
    /// Returns an error when `ring_dim` is unsupported for a binding preset.
    pub fn operator_norm_acceptance_prob(
        &self,
        ring_dim: usize,
    ) -> Result<(u128, u128), &'static str> {
        if !self.operator_norm_rejection_binds() {
            return Ok((1, 1));
        }
        let binds_production = matches!(
            self,
            Self::ExactShell {
                count_mag1: D64_PRODUCTION_EXACT_SHELL_MAG1,
                count_mag2: D64_PRODUCTION_EXACT_SHELL_MAG2,
                operator_norm_threshold: D64_PRODUCTION_OPERATOR_NORM_THRESHOLD,
            }
        );
        if binds_production && ring_dim == 64 {
            return Ok((
                D64_EXACT_SHELL_OP_NORM_ACCEPT_NUM,
                D64_EXACT_SHELL_OP_NORM_ACCEPT_DEN,
            ));
        }
        Err("unsupported binding exact-shell preset for operator-norm acceptance probability")
    }

    /// Worst-case squared ℓ₂ norm `max ‖c‖_2²` over the challenge family.
    ///
    /// Used by the folded-witness `‖z‖_inf` tail bound (`t*`) in
    /// `akita-types::sis::fold_witness_linf_tail_bound_sq`. Exact integers for every
    /// shipping preset; see `specs/fold-linf-rejection.md`.
    #[inline]
    #[must_use]
    pub fn challenge_l2_sq_max(&self) -> u128 {
        match self {
            Self::Uniform {
                weight,
                nonzero_coeffs,
            } => {
                let max_coeff_sq = nonzero_coeffs
                    .iter()
                    .map(|c| i128::from(*c).pow(2) as u128)
                    .max()
                    .unwrap_or(1);
                (*weight as u128).saturating_mul(max_coeff_sq)
            }
            Self::ExactShell {
                count_mag1,
                count_mag2,
                ..
            } => (*count_mag1 as u128).saturating_add(4u128.saturating_mul(*count_mag2 as u128)),
            Self::BoundedL1Norm => {
                // Safe upper bound `M·B` with `M = 8`, `B = 121` (exact max is 961).
                (COEFFS_BOUND_32 as u128).saturating_mul(MAX_L1_NORM_32 as u128)
            }
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
                ..
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
                operator_norm_threshold,
            } => {
                out.push(1);
                out.extend_from_slice(&(*count_mag1 as u64).to_le_bytes());
                out.extend_from_slice(&(*count_mag2 as u64).to_le_bytes());
                out.extend_from_slice(&operator_norm_threshold.to_le_bytes());
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
                ..
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
            operator_norm_threshold: 2,
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
        let shell = SparseChallengeConfig::ExactShell {
            count_mag1: D64_PRODUCTION_EXACT_SHELL_MAG1,
            count_mag2: D64_PRODUCTION_EXACT_SHELL_MAG2,
            operator_norm_threshold: D64_PRODUCTION_OPERATOR_NORM_THRESHOLD,
        };
        assert!(shell.log2_support_bits::<64>() >= 128.0);
        assert!(shell.validate_min_entropy::<64>(128).is_ok());
    }

    #[test]
    fn production_shell_binding_and_acceptance_prob() {
        let production = SparseChallengeConfig::ExactShell {
            count_mag1: D64_PRODUCTION_EXACT_SHELL_MAG1,
            count_mag2: D64_PRODUCTION_EXACT_SHELL_MAG2,
            operator_norm_threshold: D64_PRODUCTION_OPERATOR_NORM_THRESHOLD,
        };
        assert!(production.operator_norm_rejection_binds());
        assert_eq!(production.operator_norm_cap(), 18);
        assert_eq!(production.l1_norm(), 53);
        assert_eq!(
            production.operator_norm_acceptance_prob(64).unwrap(),
            (
                D64_EXACT_SHELL_OP_NORM_ACCEPT_NUM,
                D64_EXACT_SHELL_OP_NORM_ACCEPT_DEN
            )
        );

        let non_binding = SparseChallengeConfig::ExactShell {
            count_mag1: D64_PRODUCTION_EXACT_SHELL_MAG1,
            count_mag2: D64_PRODUCTION_EXACT_SHELL_MAG2,
            operator_norm_threshold: 54,
        };
        assert!(!non_binding.operator_norm_rejection_binds());
        assert_eq!(non_binding.operator_norm_cap(), 53);
        assert_eq!(
            non_binding.operator_norm_acceptance_prob(64).unwrap(),
            (1, 1)
        );

        let legacy = SparseChallengeConfig::ExactShell {
            count_mag1: 30,
            count_mag2: 12,
            operator_norm_threshold: 54,
        };
        assert!(!legacy.operator_norm_rejection_binds());
        assert_eq!(legacy.operator_norm_cap(), 54);
    }

    #[test]
    fn operator_norm_acceptance_prob_rejects_unknown_binding_shells() {
        let production = SparseChallengeConfig::ExactShell {
            count_mag1: D64_PRODUCTION_EXACT_SHELL_MAG1,
            count_mag2: D64_PRODUCTION_EXACT_SHELL_MAG2,
            operator_norm_threshold: D64_PRODUCTION_OPERATOR_NORM_THRESHOLD,
        };
        assert!(production.operator_norm_acceptance_prob(32).is_err());

        let binding_other = SparseChallengeConfig::ExactShell {
            count_mag1: 31,
            count_mag2: 11,
            operator_norm_threshold: 16,
        };
        assert!(binding_other.operator_norm_rejection_binds());
        assert!(binding_other.operator_norm_acceptance_prob(64).is_err());
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
            operator_norm_threshold: 1,
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
            operator_norm_threshold: 1,
        };
        assert!((cfg.log2_support_bits::<4>() - 3.0).abs() < 1e-9);
        // Uniform, D=4, weight 2, alphabet {-1,1}: C(4,2)*2^2 = 24.
        let uni = SparseChallengeConfig::Uniform {
            weight: 2,
            nonzero_coeffs: vec![-1, 1],
        };
        assert!((uni.log2_support_bits::<4>() - 24.0_f64.log2()).abs() < 1e-9);
    }

    #[test]
    fn challenge_l2_sq_max_matches_spec_table() {
        let shell = SparseChallengeConfig::ExactShell {
            count_mag1: D64_PRODUCTION_EXACT_SHELL_MAG1,
            count_mag2: D64_PRODUCTION_EXACT_SHELL_MAG2,
            operator_norm_threshold: D64_PRODUCTION_OPERATOR_NORM_THRESHOLD,
        };
        assert_eq!(shell.challenge_l2_sq_max(), 75);

        let uni128 = SparseChallengeConfig::Uniform {
            weight: 31,
            nonzero_coeffs: vec![-1, 1],
        };
        assert_eq!(uni128.challenge_l2_sq_max(), 31);

        let uni256 = SparseChallengeConfig::Uniform {
            weight: 23,
            nonzero_coeffs: vec![-1, 1],
        };
        assert_eq!(uni256.challenge_l2_sq_max(), 23);

        let bounded = SparseChallengeConfig::BoundedL1Norm;
        assert_eq!(bounded.challenge_l2_sq_max(), 8 * 121);
    }
}
