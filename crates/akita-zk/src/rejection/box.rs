//! Exact uniform-box rejection parameters.

use crate::error::ZkResult;
use crate::norm::field_modulus;
use akita_challenges::SparseChallengeConfig;
use akita_field::{AkitaError, PseudoMersenneField};

/// Parameters for exact box rejection in the Ajtai opening protocol.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BoxRejectionParams {
    /// Bound `B_s` on witness coefficients.
    pub witness_bound: u128,
    /// Worst-case challenge coefficient `L1` mass.
    pub challenge_l1_bound: usize,
    /// Bound `beta >= ||c s||_infty`.
    pub beta: u128,
    /// Masking box radius.
    pub gamma: u128,
    /// Accepted response box radius `gamma - beta`.
    pub response_bound: u128,
    /// Number of revealed coefficients `m * D`.
    pub revealed_coefficients: usize,
}

impl BoxRejectionParams {
    /// Derive parameters for a target acceptance probability.
    ///
    /// # Errors
    ///
    /// Returns an error if the inputs are invalid or arithmetic overflows.
    pub fn for_target_acceptance(
        witness_len: usize,
        ring_degree: usize,
        challenge_cfg: &SparseChallengeConfig,
        witness_bound: u128,
        target_acceptance: f64,
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
        if !target_acceptance.is_finite() || target_acceptance <= 0.0 || target_acceptance >= 1.0 {
            return Err(AkitaError::InvalidInput(
                "target_acceptance must be in (0, 1)".to_string(),
            ));
        }

        let challenge_l1_bound = challenge_cfg.l1_norm();
        let beta = (challenge_l1_bound as u128)
            .checked_mul(witness_bound)
            .ok_or_else(|| AkitaError::InvalidInput("beta overflow".to_string()))?;
        let revealed_coefficients = witness_len
            .checked_mul(ring_degree)
            .ok_or_else(|| AkitaError::InvalidInput("revealed coefficient overflow".to_string()))?;
        let gamma = minimal_gamma(beta, revealed_coefficients, target_acceptance)?;
        let response_bound = gamma.checked_sub(beta).ok_or_else(|| {
            AkitaError::InvalidInput("gamma must be larger than beta".to_string())
        })?;

        Ok(Self {
            witness_bound,
            challenge_l1_bound,
            beta,
            gamma,
            response_bound,
            revealed_coefficients,
        })
    }

    /// Derive parameters for acceptance probability at least `1/2`.
    ///
    /// # Errors
    ///
    /// Returns an error if the inputs are invalid or arithmetic overflows.
    pub fn for_half_acceptance(
        witness_len: usize,
        ring_degree: usize,
        challenge_cfg: &SparseChallengeConfig,
        witness_bound: u128,
    ) -> ZkResult<Self> {
        Self::for_target_acceptance(witness_len, ring_degree, challenge_cfg, witness_bound, 0.5)
    }

    /// Exact acceptance probability, evaluated as `f64`.
    pub fn acceptance_probability(&self) -> f64 {
        acceptance_probability(self.beta, self.gamma, self.revealed_coefficients)
    }

    /// Validate that centered additions cannot wrap modulo the field.
    ///
    /// # Errors
    ///
    /// Returns an error if the field modulus metadata is unsupported or if
    /// `gamma + beta >= q/2`.
    pub fn validate_no_modular_wrap<F>(&self) -> ZkResult<()>
    where
        F: PseudoMersenneField,
    {
        let q = field_modulus::<F>()?;
        let max_abs = self
            .gamma
            .checked_add(self.beta)
            .ok_or_else(|| AkitaError::InvalidInput("gamma + beta overflow".to_string()))?;
        if max_abs >= q / 2 {
            return Err(AkitaError::InvalidInput(format!(
                "box parameters allow modular wrap: gamma + beta = {max_abs}, q/2 = {}",
                q / 2
            )));
        }
        Ok(())
    }
}

fn minimal_gamma(beta: u128, revealed_coefficients: usize, target: f64) -> ZkResult<u128> {
    if beta == 0 {
        return Err(AkitaError::InvalidInput(
            "beta must be non-zero".to_string(),
        ));
    }
    if revealed_coefficients == 0 {
        return Err(AkitaError::InvalidInput(
            "revealed_coefficients must be non-zero".to_string(),
        ));
    }

    let beta_f = beta as f64;
    let root = target.powf(1.0 / revealed_coefficients as f64);
    let estimate = ((2.0 * beta_f - 1.0 + root) / (2.0 * (1.0 - root))).ceil();
    if !estimate.is_finite() || estimate < 0.0 {
        return Err(AkitaError::InvalidInput(
            "failed to derive finite gamma estimate".to_string(),
        ));
    }
    let mut gamma = estimate as u128;
    if gamma <= beta {
        gamma = beta + 1;
    }
    while acceptance_probability(beta, gamma, revealed_coefficients) < target {
        gamma = gamma
            .checked_add(1)
            .ok_or_else(|| AkitaError::InvalidInput("gamma overflow".to_string()))?;
    }
    while gamma > beta + 1
        && acceptance_probability(beta, gamma - 1, revealed_coefficients) >= target
    {
        gamma -= 1;
    }
    Ok(gamma)
}

fn acceptance_probability(beta: u128, gamma: u128, revealed_coefficients: usize) -> f64 {
    if gamma <= beta {
        return 0.0;
    }
    let numerator = 2.0 * (gamma - beta) as f64 + 1.0;
    let denominator = 2.0 * gamma as f64 + 1.0;
    (numerator / denominator).powf(revealed_coefficients as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derives_half_acceptance_parameters_from_existing_challenges() {
        let d32 = SparseChallengeConfig::BoundedL1Norm;
        let d64 = SparseChallengeConfig::ExactShell {
            count_mag1: 30,
            count_mag2: 12,
        };
        let d128 = SparseChallengeConfig::Uniform {
            weight: 31,
            nonzero_coeffs: vec![-1, 1],
        };

        let cases = [
            (&d32, 32, 1, 1936, 90_349, 88_413),
            (&d64, 64, 1, 864, 80_208, 79_344),
            (&d128, 128, 1, 496, 91_842, 91_346),
            (&d32, 32, 2, 1936, 179_725, 177_789),
            (&d64, 64, 2, 864, 159_983, 159_119),
            (&d128, 128, 2, 496, 183_436, 182_940),
            (&d32, 32, 4, 1936, 358_480, 356_544),
            (&d64, 64, 4, 864, 319_533, 318_669),
            (&d128, 128, 4, 496, 366_623, 366_127),
        ];

        for (cfg, degree, witness_len, beta, gamma, response_bound) in cases {
            let params =
                BoxRejectionParams::for_half_acceptance(witness_len, degree, cfg, 16).unwrap();
            assert_eq!(params.beta, beta);
            assert_eq!(params.gamma, gamma);
            assert_eq!(params.response_bound, response_bound);
            assert!(params.acceptance_probability() >= 0.5);
        }
    }

    #[test]
    fn acceptance_probability_handles_large_gamma_without_integer_overflow() {
        let beta = 1;
        let gamma = u128::MAX;
        let p = acceptance_probability(beta, gamma, 1);
        assert!(p.is_finite());
        assert!(p > 0.0 && p <= 1.0);
    }

    #[test]
    fn target_acceptance_excludes_zero() {
        let cfg = SparseChallengeConfig::Uniform {
            weight: 31,
            nonzero_coeffs: vec![-1, 1],
        };
        assert!(BoxRejectionParams::for_target_acceptance(1, 128, &cfg, 16, 0.0).is_err());
    }
}
