//! Single-step Gärtner rejection rule.
//!
//! Implements the `f_v / g_v` acceptance functions of Corollary 1 from Joel
//! Gärtner's "Compact Lattice Signatures via Iterative Rejection Sampling"
//! (Gaertner_Iterative_Rejection_Sampling.pdf). This module exposes the
//! single-step variant of the rule, suitable as a drop-in replacement for the
//! BLISS bimodal acceptance test.
//!
//! This is a **measurement-only** policy. The Gärtner rule samples a hidden
//! sign `b in {-1, +1}` and emits `z = y + b v`. To remain zero-knowledge the
//! sign must be hidden by the protocol. Akita's Ajtai relation does not have
//! the BLISS-style structural pieces (mod `2 q`, `A s = q j`) needed for the
//! sign to be hidden algebraically. The protocol variant in
//! `crate::protocols::opening` therefore emits the sign publicly, breaking
//! ZK. Acceptance probabilities and proof-size estimates measured with this
//! module are still useful for comparing rejection-rule families, but the
//! resulting transcripts cannot be claimed to be ZK.

use crate::error::ZkResult;
use crate::norm::field_modulus;
use akita_challenges::SparseChallengeConfig;
use akita_field::{AkitaError, PseudoMersenneField};

const LN_2: f64 = core::f64::consts::LN_2;
const SQRT_2_PI: f64 = 2.506_628_274_631_000_7_f64;

/// Cap on the number of terms summed when truncating `S_v(y)`.
///
/// The series decays super-geometrically once `k > alpha`, so this is a hard
/// cutoff for pathological inputs. Real evaluations break out of the loop
/// earlier via the relative threshold below.
pub const SV_MAX_TERMS: usize = 1024;

/// Stop the alternating-sum truncation when the absolute term value drops
/// below this fraction of the running sum's magnitude. `1e-18` is just below
/// `f64` mantissa precision.
pub const SV_RELATIVE_THRESHOLD: f64 = 1.0e-18;

/// Parameters for the single-step Gärtner rejection policy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GaertnerRejectionParams {
    /// Bound `B_s` on witness coefficients.
    pub witness_bound: u128,
    /// Worst-case challenge coefficient `L1` mass.
    pub challenge_l1_bound: usize,
    /// Bound `beta >= ||c s||_infty` derived from `L1 * witness_bound`.
    pub beta: u128,
    /// Number of revealed coefficients `m * D`.
    pub revealed_coefficients: usize,
    /// Ratio `alpha = sigma / shift_l2_bound` used to compute `M`.
    pub width_factor: f64,
    /// Statistical ZK error budget in bits, used to size the response box.
    pub zk_error_bits: u32,
    /// Tail-truncation error budget in bits for the Banaszczyk cutoff.
    pub tail_error_bits: u32,
    /// Worst-case shift `||c s||_2` bound.
    pub shift_l2_bound: f64,
    /// Discrete Gaussian standard deviation used for masking.
    pub sigma: f64,
    /// Public response infinity bound.
    pub response_bound: u128,
    /// Internal mask infinity bound for avoiding modular wrap.
    pub mask_bound: u128,
    /// Repetition rate `M_alpha` from Corollary 1.
    pub rejection_m: f64,
}

impl GaertnerRejectionParams {
    /// Derive Gärtner rejection parameters from the worst-case `L2` shift bound.
    ///
    /// # Errors
    ///
    /// Returns an error if inputs are invalid or if derived parameters are not
    /// finite.
    pub fn for_l2_bound(
        witness_len: usize,
        ring_degree: usize,
        challenge_cfg: &SparseChallengeConfig,
        witness_bound: u128,
        width_factor: f64,
        zk_error_bits: u32,
        tail_error_bits: u32,
    ) -> ZkResult<Self> {
        if witness_len == 0 {
            return Err(AkitaError::InvalidInput(
                "witness_len must be non-zero".to_string(),
            ));
        }
        if ring_degree == 0 {
            return Err(AkitaError::InvalidInput(
                "ring_degree must be non-zero".to_string(),
            ));
        }
        if witness_bound == 0 {
            return Err(AkitaError::InvalidInput(
                "witness_bound must be non-zero".to_string(),
            ));
        }
        if !width_factor.is_finite() || width_factor <= 0.0 {
            return Err(AkitaError::InvalidInput(
                "width_factor must be finite and positive".to_string(),
            ));
        }

        let challenge_l1_bound = challenge_cfg.l1_norm();
        let beta = (challenge_l1_bound as u128)
            .checked_mul(witness_bound)
            .ok_or_else(|| AkitaError::InvalidInput("beta overflow".to_string()))?;
        let revealed_coefficients = witness_len
            .checked_mul(ring_degree)
            .ok_or_else(|| AkitaError::InvalidInput("revealed coefficient overflow".to_string()))?;
        let shift_l2_bound = beta as f64 * (revealed_coefficients as f64).sqrt();
        let sigma = width_factor * shift_l2_bound;
        if !sigma.is_finite() || sigma <= 0.0 {
            return Err(AkitaError::InvalidInput(
                "derived sigma must be finite and positive".to_string(),
            ));
        }

        let tail_log = (2.0 * revealed_coefficients as f64).ln() + tail_error_bits as f64 * LN_2;
        let tail_cutoff = (2.0 * tail_log).sqrt();
        let response_bound = ceil_f64_to_u128(sigma * tail_cutoff)?;
        let mask_bound = response_bound
            .checked_add(beta)
            .ok_or_else(|| AkitaError::InvalidInput("mask bound overflow".to_string()))?;

        let rejection_m = gaertner_repetition_rate(width_factor);
        if !rejection_m.is_finite() || rejection_m < 1.0 {
            return Err(AkitaError::InvalidInput(
                "derived rejection constant must be finite".to_string(),
            ));
        }

        Ok(Self {
            witness_bound,
            challenge_l1_bound,
            beta,
            revealed_coefficients,
            width_factor,
            zk_error_bits,
            tail_error_bits,
            shift_l2_bound,
            sigma,
            response_bound,
            mask_bound,
            rejection_m,
        })
    }

    /// Estimated lower bound on the non-abort probability `1 / M`.
    #[must_use]
    pub fn estimated_acceptance_probability(&self) -> f64 {
        1.0 / self.rejection_m
    }

    /// Validate that sampled masks and accepted responses cannot wrap modulo
    /// the field.
    ///
    /// # Errors
    ///
    /// Returns an error if the modulus metadata is unsupported or if
    /// `mask_bound >= q/2`.
    pub fn validate_no_modular_wrap<F>(&self) -> ZkResult<()>
    where
        F: PseudoMersenneField,
    {
        let q = field_modulus::<F>()?;
        if self.mask_bound >= q / 2 {
            return Err(AkitaError::InvalidInput(format!(
                "gaertner parameters allow modular wrap: mask_bound = {}, q/2 = {}",
                self.mask_bound,
                q / 2
            )));
        }
        Ok(())
    }
}

/// Gärtner's repetition rate `M_alpha` from Corollary 1.
///
/// `M_alpha = 1 + (2 alpha sqrt(2 pi) rho(pi alpha))
///                / (rho_alpha(1) * (1 - rho(2 pi alpha)))`,
/// using `rho_s(x) = exp(-x^2 / (2 s^2))` and `rho(x) = rho_1(x)`.
#[must_use]
pub fn gaertner_repetition_rate(alpha: f64) -> f64 {
    let pi = core::f64::consts::PI;
    let pi_alpha_sq = (pi * alpha) * (pi * alpha);
    let two_pi_alpha_sq = (2.0 * pi * alpha) * (2.0 * pi * alpha);
    let rho_pi_alpha = (-pi_alpha_sq / 2.0).exp();
    let rho_alpha_one = (-1.0 / (2.0 * alpha * alpha)).exp();
    let rho_two_pi_alpha = (-two_pi_alpha_sq / 2.0).exp();
    let denom = rho_alpha_one * (1.0 - rho_two_pi_alpha);
    if denom == 0.0 {
        return f64::INFINITY;
    }
    1.0 + (2.0 * alpha * SQRT_2_PI * rho_pi_alpha) / denom
}

/// Truncated evaluation of `S_v(y) = sum_{k>=0} (-1)^k * rho_r(y + k v) / rho_r(y)`.
///
/// Uses `||y + k v||^2 - ||y||^2 = 2 k <y, v> + k^2 ||v||^2`, so the ratio in
/// each term is `exp(-(2 k <y, v> + k^2 ||v||^2) / (2 sigma^2))`. Stops once a
/// term's magnitude falls below `SV_RELATIVE_THRESHOLD`, capped at
/// `SV_MAX_TERMS`.
#[must_use]
pub fn s_v_truncated(inner_y_v: f64, v_l2_squared: f64, sigma: f64) -> f64 {
    if !sigma.is_finite() || sigma <= 0.0 {
        return f64::NAN;
    }
    let two_sigma_sq = 2.0 * sigma * sigma;
    let mut sum = 1.0;
    let mut sign = -1.0;
    for k in 1..=SV_MAX_TERMS {
        let kf = k as f64;
        let exponent = -(2.0 * kf * inner_y_v + kf * kf * v_l2_squared) / two_sigma_sq;
        let mag = exponent.exp();
        sum += sign * mag;
        if mag < SV_RELATIVE_THRESHOLD {
            break;
        }
        sign = -sign;
    }
    sum
}

/// Returns `(f_v(y), g_v(y))` for the Gärtner acceptance rule from Theorem 1.
///
/// `M` should be supplied by [`gaertner_repetition_rate`] or higher. Outputs
/// are clamped to `[0, 1]` to absorb truncation noise.
#[must_use]
pub fn gaertner_acceptance(inner_y_v: f64, v_l2_squared: f64, sigma: f64, m: f64) -> (f64, f64) {
    let sv_y = s_v_truncated(inner_y_v, v_l2_squared, sigma);
    let sv_neg_y = s_v_truncated(-inner_y_v, v_l2_squared, sigma);

    let f_v_raw = if inner_y_v >= v_l2_squared {
        sv_y / m
    } else {
        (1.0 - sv_neg_y) / m
    };
    let g_v_raw = if inner_y_v >= -v_l2_squared {
        (1.0 - sv_y) / m
    } else {
        sv_neg_y / m
    };

    let f_v = clamp_unit(f_v_raw);
    let g_v = clamp_unit(g_v_raw);
    let total = f_v + g_v;
    if total > 1.0 {
        // Should not happen under exact arithmetic by Theorem 1 but can occur
        // under truncation. Renormalize to keep `f + g <= 1`.
        (f_v / total, g_v / total)
    } else {
        (f_v, g_v)
    }
}

/// Outcome of a single Gärtner acceptance roll.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GaertnerOutcome {
    /// Accept with sign `-1`, response is `y - v`.
    SignNegative,
    /// Accept with sign `+1`, response is `y + v`.
    SignPositive,
    /// Abort and resample.
    Abort,
}

/// Roll the Gärtner acceptance with a uniform `[0, 1)` sample `u`.
#[must_use]
pub fn gaertner_roll(u: f64, f_v: f64, g_v: f64) -> GaertnerOutcome {
    if u < f_v {
        GaertnerOutcome::SignNegative
    } else if u < f_v + g_v {
        GaertnerOutcome::SignPositive
    } else {
        GaertnerOutcome::Abort
    }
}

fn clamp_unit(x: f64) -> f64 {
    if x.is_nan() {
        0.0
    } else {
        x.clamp(0.0, 1.0)
    }
}

fn ceil_f64_to_u128(value: f64) -> ZkResult<u128> {
    if !value.is_finite() || value < 0.0 || value > u128::MAX as f64 {
        return Err(AkitaError::InvalidInput(
            "cannot convert derived f64 bound to u128".to_string(),
        ));
    }
    Ok(value.ceil() as u128)
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_challenges::SparseChallengeConfig;

    #[test]
    fn repetition_rate_matches_paper_table_landmarks() {
        // Paper Section 3.4 / Figure 2 landmarks. The exact numerical values
        // are not given as a table, but the qualitative shape is:
        //   alpha = 1.0 -> M close to 1.06 (a few percent rejection).
        //   alpha = 4.0 -> M - 1 < 2^-100 (rejection-free regime).
        //   alpha = 0.5 -> M dominated by exp-style growth.
        let m_at_1 = gaertner_repetition_rate(1.0);
        assert!(m_at_1 > 1.0 && m_at_1 < 1.10, "M(1) = {m_at_1}");

        let m_at_4 = gaertner_repetition_rate(4.0);
        assert!(m_at_4 - 1.0 < 1e-30, "M(4) - 1 = {}", m_at_4 - 1.0);

        let m_at_half = gaertner_repetition_rate(0.5);
        assert!(m_at_half > 5.0 && m_at_half < 8.0, "M(0.5) = {m_at_half}");
    }

    #[test]
    fn beats_bliss_bimodal_at_moderate_alpha() {
        // BLISS bimodal: M = exp(1/(2 alpha^2)).
        let bliss = |alpha: f64| (1.0 / (2.0 * alpha * alpha)).exp();
        for alpha in [0.5, 0.7, 1.0, 1.5, 2.0, 4.0] {
            let m_bliss = bliss(alpha);
            let m_gaertner = gaertner_repetition_rate(alpha);
            assert!(
                m_gaertner < m_bliss + 1e-12,
                "alpha = {alpha}: gaertner = {m_gaertner}, bliss = {m_bliss}"
            );
        }
    }

    #[test]
    fn s_v_truncation_converges() {
        // For alpha = 1, the series at k = 0..40 should fully converge.
        let sigma = 1.0;
        let v_l2 = 1.0;
        for inner in [-1.0, 0.0, 0.5, 1.0] {
            let s = s_v_truncated(inner, v_l2, sigma);
            assert!(s.is_finite(), "S_v({inner}) = {s}");
        }
    }

    #[test]
    fn acceptance_lemma_average_equals_one_over_m() {
        // Numerically integrate `f_v + g_v` against the Gaussian density of y
        // and confirm it equals 1/M to within a small tolerance. This checks
        // Lemma 1 and Theorem 1 hold for our truncation. Use a simple 1d slice
        // since S_v depends only on <y, v> and ||v||^2.
        let alpha = 1.0_f64;
        let v_norm = 1.0_f64;
        let v_l2 = v_norm * v_norm;
        let sigma = alpha * v_norm;
        let m = gaertner_repetition_rate(alpha);

        // Numerically estimate E[f_v(y) + g_v(y)] over y ~ N(0, sigma).
        // Use a uniform grid in [-6 sigma, 6 sigma] with trapezoid rule.
        let n = 4001;
        let lo = -6.0 * sigma;
        let hi = 6.0 * sigma;
        let dx = (hi - lo) / (n as f64 - 1.0);
        let mut accum = 0.0;
        for i in 0..n {
            let y = lo + dx * i as f64;
            let inner_y_v = y * v_norm;
            let (f, g) = gaertner_acceptance(inner_y_v, v_l2, sigma, m);
            let weight = (-y * y / (2.0 * sigma * sigma)).exp() / (sigma * SQRT_2_PI);
            let trapz = if i == 0 || i == n - 1 { 0.5 } else { 1.0 };
            accum += trapz * (f + g) * weight * dx;
        }
        let expected = 1.0 / m;
        assert!(
            (accum - expected).abs() < 5.0e-3,
            "accum = {accum}, 1/M = {expected}",
        );
    }

    #[test]
    fn for_l2_bound_yields_acceptance_close_to_one_over_m() {
        let cfg = SparseChallengeConfig::Uniform {
            weight: 31,
            nonzero_coeffs: vec![-1, 1],
        };
        let params =
            GaertnerRejectionParams::for_l2_bound(406, 128, &cfg, 16, 16.0, 128, 128).unwrap();
        let p = params.estimated_acceptance_probability();
        assert!(
            p > 0.999,
            "alpha = 16 should give near-rejection-free regime, got {p}",
        );
    }
}
