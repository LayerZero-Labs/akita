//! Experimental Gaussian rejection parameters and samplers.
//!
//! This module is for protocol-shape and proof-size experiments. It uses
//! floating-point sampling and acceptance tests, so it is not a production
//! constant-time Gaussian engine.

use crate::error::ZkResult;
use crate::norm::{field_from_centered_i128, field_modulus};
use crate::util::{ceil_f64_to_u128, open_unit_f64};
use akita_algebra::CyclotomicRing;
use akita_challenges::SparseChallengeConfig;
use akita_field::{AkitaError, CanonicalField, FieldCore, PseudoMersenneField};
use rand_core::RngCore;

const LN_2: f64 = core::f64::consts::LN_2;

/// Parameters for the experimental discrete-Gaussian rejection policy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GaussianRejectionParams {
    /// Bound `B_s` on witness coefficients.
    pub witness_bound: u128,
    /// Worst-case challenge coefficient `L1` mass.
    pub challenge_l1_bound: usize,
    /// Bound `beta >= ||c s||_infty`.
    pub beta: u128,
    /// Number of revealed coefficients `m * D`.
    pub revealed_coefficients: usize,
    /// Ratio between the Gaussian width and the L2 shift bound.
    pub width_factor: f64,
    /// Statistical error target used in the rejection lemma.
    pub zk_error_bits: u32,
    /// Tail error target used to derive the public response bound.
    pub tail_error_bits: u32,
    /// Worst-case L2 shift bound used for this parameter set.
    pub shift_l2_bound: f64,
    /// Discrete Gaussian standard deviation used for masking.
    pub sigma: f64,
    /// Public response infinity bound.
    pub response_bound: u128,
    /// Internal mask infinity bound for avoiding modular wrap.
    pub mask_bound: u128,
    /// Rejection constant `M`.
    pub rejection_m: f64,
}

impl GaussianRejectionParams {
    /// Derive Gaussian rejection parameters from the worst-case L2 shift bound.
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

        let a_kappa = (2.0 * (zk_error_bits as f64 + 1.0) * LN_2).sqrt();
        let rejection_m =
            (a_kappa / width_factor + 1.0 / (2.0 * width_factor * width_factor)).exp();
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

    /// Estimated lower bound on the non-abort probability.
    #[must_use]
    pub fn estimated_acceptance_probability(&self) -> f64 {
        (1.0 - 2.0_f64.powf(-(self.zk_error_bits as f64))) / self.rejection_m
    }

    /// Validate that sampled masks and accepted responses cannot wrap modulo
    /// the field.
    ///
    /// # Errors
    ///
    /// Returns an error if the modulus metadata is unsupported or if
    /// `mask_bound + beta >= q/2`.
    pub fn validate_no_modular_wrap<F>(&self) -> ZkResult<()>
    where
        F: PseudoMersenneField,
    {
        let q = field_modulus::<F>()?;
        let max_abs = self
            .mask_bound
            .checked_add(self.beta)
            .ok_or_else(|| AkitaError::InvalidInput("mask_bound + beta overflow".to_string()))?;
        if max_abs >= q / 2 {
            return Err(AkitaError::InvalidInput(format!(
                "gaussian parameters allow modular wrap: mask_bound + beta = {max_abs}, q/2 = {}",
                q / 2
            )));
        }
        Ok(())
    }
}

/// Sample a ring vector from a rounded, tail-truncated Gaussian.
///
/// # Errors
///
/// Returns an error if the sampler parameters are invalid.
pub fn sample_ring_vec_discrete_gaussian<F, R, const D: usize>(
    rng: &mut R,
    len: usize,
    sigma: f64,
    bound: u128,
) -> ZkResult<Vec<CyclotomicRing<F, D>>>
where
    F: FieldCore + CanonicalField,
    R: RngCore + ?Sized,
{
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        out.push(sample_ring_discrete_gaussian(rng, sigma, bound)?);
    }
    Ok(out)
}

/// Acceptance probability for shifting a Gaussian sample from `y` to `z`.
#[must_use]
pub fn gaussian_rejection_acceptance(
    mask_l2_squared: f64,
    response_l2_squared: f64,
    m: f64,
    sigma: f64,
) -> f64 {
    let exponent = (mask_l2_squared - response_l2_squared) / (2.0 * sigma * sigma);
    (exponent.exp() / m).clamp(0.0, 1.0)
}

fn sample_ring_discrete_gaussian<F, R, const D: usize>(
    rng: &mut R,
    sigma: f64,
    bound: u128,
) -> ZkResult<CyclotomicRing<F, D>>
where
    F: FieldCore + CanonicalField,
    R: RngCore + ?Sized,
{
    if !sigma.is_finite() || sigma <= 0.0 {
        return Err(AkitaError::InvalidInput(
            "sigma must be finite and positive".to_string(),
        ));
    }
    let bound_i128 = i128::try_from(bound)
        .map_err(|_| AkitaError::InvalidInput("gaussian sampler bound exceeds i128".to_string()))?;
    let mut coeffs = [F::zero(); D];
    for coeff in &mut coeffs {
        let value = sample_discrete_gaussian_i128(rng, sigma, bound_i128);
        *coeff = field_from_centered_i128(value)?;
    }
    Ok(CyclotomicRing::from_coefficients(coeffs))
}

fn sample_discrete_gaussian_i128<R>(rng: &mut R, sigma: f64, bound: i128) -> i128
where
    R: RngCore + ?Sized,
{
    loop {
        let value = (sigma * standard_normal(rng)).round() as i128;
        if value.abs() <= bound {
            return value;
        }
    }
}

fn standard_normal<R>(rng: &mut R) -> f64
where
    R: RngCore + ?Sized,
{
    let u1 = open_unit_f64(rng);
    let u2 = open_unit_f64(rng);
    (-2.0 * u1.ln()).sqrt() * (2.0 * core::f64::consts::PI * u2).cos()
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_challenges::SparseChallengeConfig;

    #[test]
    fn derives_tail_shape_gaussian_bound() {
        let cfg = SparseChallengeConfig::Uniform {
            weight: 31,
            nonzero_coeffs: vec![-1, 1],
        };
        let params =
            GaussianRejectionParams::for_l2_bound(406, 128, &cfg, 16, 16.0, 128, 128).unwrap();

        assert_eq!(params.beta, 496);
        assert_eq!(params.revealed_coefficients, 51_968);
        assert_eq!(params.response_bound, 25_620_030);
        assert!(params.estimated_acceptance_probability() > 0.40);
    }
}
