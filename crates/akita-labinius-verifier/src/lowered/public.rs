use akita_algebra::{
    binary::BinaryField162,
    fft::SmoothFftField,
    ring::{embed_scalar, TrinomialModulus},
};
use akita_challenges::BinaryChallenge;
use akita_error::{checked, AkitaError};
use akita_types::RelationPolynomial;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::{horner, powers, signed_field, zero_vec, LoweredRootLayout};
use crate::{
    endpoint::verify_left_expansion,
    source::{challenge_scalar, equality_weights, scalar_from_binary},
    BinaryClearSetup, BinaryEvaluationClaim,
};

/// Explicit evaluation and row-batching challenges; no sampling occurs here.
#[derive(Clone, Copy, Debug)]
pub struct LoweredChallenges<F> {
    pub alpha: F,
    pub xi: F,
    pub gamma: F,
}

/// Validated public data and cached evaluations for one lowered relation.
#[derive(Clone, Debug)]
pub struct LoweredPublic<F> {
    pub(crate) layout: LoweredRootLayout,
    pub(crate) challenges: LoweredChallenges<F>,
    pub(crate) abar: Vec<F>,
    pub(crate) binary_rows: Vec<F>,
    pub(crate) embedded_challenges: Vec<F>,
    pub(crate) alpha_powers: Vec<F>,
    pub(crate) xi_powers: Vec<F>,
    pub(crate) gamma_powers: Vec<F>,
    pub(crate) digit_powers: Vec<F>,
    pub(crate) a_quotient_evaluations: Vec<F>,
    pub(crate) parity_rhs: F,
    c_pub: F,
}

impl<F: SmoothFftField> LoweredPublic<F> {
    /// Check public geometry, fold challenges, left expansion and clear ranges.
    #[allow(clippy::too_many_arguments)]
    pub fn new<const D: usize, M: TrinomialModulus>(
        layout: &LoweredRootLayout,
        setup: &BinaryClearSetup<F, D, M>,
        claim: &BinaryEvaluationClaim,
        u: &[BinaryField162],
        fold_challenges: &[BinaryChallenge],
        a_quotients: &[Vec<F>],
        parity_quotient: &[i128],
        parity_carry: &[i128],
        challenges: LoweredChallenges<F>,
    ) -> Result<Self, AkitaError> {
        layout.validate_setup(setup)?;
        if claim.point.len() != setup.num_vars() || u.len() != layout.columns() {
            return Err(AkitaError::InvalidInput(
                "lowered public claim geometry mismatch".into(),
            ));
        }
        verify_left_expansion(setup, claim, u)?;
        let quotient_len = layout.polynomial().quotient_coefficient_len()?;
        if fold_challenges.len() != layout.columns()
            || a_quotients.len() != layout.n_a()
            || a_quotients.iter().any(|row| row.len() != quotient_len)
            || parity_quotient.len() != 161
            || parity_carry.len() != 162
        {
            return Err(AkitaError::InvalidProof);
        }
        for challenge in fold_challenges {
            challenge
                .validate(setup.profile())
                .map_err(|_| AkitaError::InvalidProof)?;
        }
        for (values, range) in [
            (parity_quotient, layout.encoding().quotient()),
            (parity_carry, layout.encoding().carry()),
        ] {
            let (lower, upper) = range.interval();
            if values.iter().any(|&v| v < lower || v > upper) {
                return Err(AkitaError::InvalidProof);
            }
        }
        let alpha_powers = powers(challenges.alpha, D)?;
        let xi_powers = powers(challenges.xi, 162)?;
        let gamma_len = layout
            .n_a()
            .checked_add(1)
            .ok_or(AkitaError::InvalidProof)?;
        let gamma_powers = powers(challenges.gamma, gamma_len)?;
        let digit_base = F::from_u64(
            1u64.checked_shl(layout.encoding().base().bits())
                .ok_or(AkitaError::InvalidProof)?,
        );
        let digit_powers = powers(digit_base, layout.response_layout().digit_depth())?;
        // Dense A-row weights intentionally use direct Horner evaluation.
        let mut abar = zero_vec(layout.m())?;
        #[cfg(feature = "parallel")]
        {
            // Borrow field slices rather than the ring's modulus marker, which
            // need not implement Sync. The validated matrix bounds this scratch.
            let mut matrix_coefficients: Vec<&[F]> = Vec::new();
            matrix_coefficients
                .try_reserve_exact(setup.matrix().len())
                .map_err(|_| AkitaError::InvalidProof)?;
            matrix_coefficients.extend(
                setup
                    .matrix()
                    .iter()
                    .map(|element| element.coefficients().as_slice()),
            );
            abar.par_iter_mut().enumerate().try_for_each(
                |(column, destination)| -> Result<(), AkitaError> {
                    for (row, &weight) in gamma_powers.iter().take(layout.n_a()).enumerate() {
                        let index = checked::mul_add(row, layout.m(), column)
                            .ok_or(AkitaError::InvalidProof)?;
                        let coefficients = matrix_coefficients
                            .get(index)
                            .ok_or(AkitaError::InvalidProof)?;
                        *destination += weight * horner(coefficients, challenges.alpha);
                    }
                    Ok(())
                },
            )?;
        }
        #[cfg(not(feature = "parallel"))]
        for (row, &weight) in gamma_powers.iter().take(layout.n_a()).enumerate() {
            for (column, destination) in abar.iter_mut().enumerate() {
                let index =
                    checked::mul_add(row, layout.m(), column).ok_or(AkitaError::InvalidProof)?;
                let element = setup.matrix().get(index).ok_or(AkitaError::InvalidProof)?;
                *destination += weight * horner(element.coefficients(), challenges.alpha);
            }
        }
        let row_point = claim
            .point
            .get(..setup.row_vars())
            .ok_or(AkitaError::InvalidProof)?;
        let binary_weights = equality_weights(row_point)?;
        let mut binary_rows = zero_vec(layout.scalar_rows())?;
        #[cfg(feature = "parallel")]
        binary_rows
            .par_iter_mut()
            .zip(&binary_weights)
            .try_for_each(|(destination, &weight)| -> Result<(), AkitaError> {
                *destination = horner(
                    scalar_from_binary::<F>(weight)?.coefficients(),
                    challenges.xi,
                );
                Ok(())
            })?;
        #[cfg(not(feature = "parallel"))]
        for (destination, &weight) in binary_rows.iter_mut().zip(&binary_weights) {
            *destination = horner(
                scalar_from_binary::<F>(weight)?.coefficients(),
                challenges.xi,
            );
        }
        let mut embedded_challenges = zero_vec(layout.columns())?;
        let mut u_challenge_sum = F::zero();
        for ((destination, challenge), &u_value) in
            embedded_challenges.iter_mut().zip(fold_challenges).zip(u)
        {
            let scalar = challenge_scalar::<F>(challenge).map_err(|_| AkitaError::InvalidProof)?;
            let embedded =
                embed_scalar::<F, 162, D, M>(&scalar).map_err(|_| AkitaError::InvalidProof)?;
            *destination = horner(embedded.coefficients(), challenges.alpha);
            u_challenge_sum += horner(
                scalar_from_binary::<F>(u_value)?.coefficients(),
                challenges.xi,
            ) * horner(scalar.coefficients(), challenges.xi);
        }
        let mut a_quotient_evaluations = zero_vec(layout.n_a())?;
        for (destination, row) in a_quotient_evaluations.iter_mut().zip(a_quotients) {
            *destination = horner(row, challenges.alpha);
        }
        let integer_evaluation = |values: &[i128]| {
            values.iter().rev().fold(F::zero(), |acc, &value| {
                acc * challenges.xi + signed_field::<F>(value)
            })
        };
        let parity_rhs = u_challenge_sum
            + RelationPolynomial::plus_trinomial(162)?.evaluate_modulus_at(challenges.xi)?
                * integer_evaluation(parity_quotient)
            + F::from_u64(2) * integer_evaluation(parity_carry);
        let mut result = Self {
            layout: layout.clone(),
            challenges,
            abar,
            binary_rows,
            embedded_challenges,
            alpha_powers,
            xi_powers,
            gamma_powers,
            digit_powers,
            a_quotient_evaluations,
            parity_rhs,
            c_pub: F::zero(),
        };
        #[cfg(feature = "parallel")]
        {
            result.c_pub = (0..layout.m())
                .into_par_iter()
                .map(|j| -> Result<F, AkitaError> {
                    let mut sum = F::zero();
                    for t in 0..D {
                        sum +=
                            signed_field::<F>(layout.off(t)?) * result.coefficient_weight(j, t)?;
                    }
                    Ok(sum)
                })
                .try_reduce(F::zero, |left, right| Ok(left + right))?;
        }
        #[cfg(not(feature = "parallel"))]
        for j in 0..layout.m() {
            for t in 0..D {
                result.c_pub +=
                    signed_field::<F>(layout.off(t)?) * result.coefficient_weight(j, t)?;
            }
        }
        let modulus = layout.polynomial().evaluate_modulus_at(challenges.alpha)?;
        for (&weight, &quotient) in result
            .gamma_powers
            .iter()
            .zip(&result.a_quotient_evaluations)
        {
            result.c_pub += weight * modulus * quotient;
        }
        result.c_pub += result.g()? * result.parity_rhs;
        Ok(result)
    }
    pub fn c_pub(&self) -> F {
        self.c_pub
    }
    /// Digit factors of the response weights: entry `l` is `(2^b)^l` for digit
    /// width `b`, one entry per response digit, lowest digit first.
    pub fn digit_powers(&self) -> &[F] {
        &self.digit_powers
    }
    pub fn challenges(&self) -> LoweredChallenges<F> {
        self.challenges
    }
    pub(crate) fn validate_layout(&self, layout: &LoweredRootLayout) -> Result<(), AkitaError> {
        if &self.layout != layout {
            return Err(AkitaError::InvalidInput(
                "public weights belong to a different layout".into(),
            ));
        }
        Ok(())
    }
    pub(crate) fn g(&self) -> Result<F, AkitaError> {
        self.gamma_powers
            .get(self.layout.n_a())
            .copied()
            .ok_or(AkitaError::InvalidProof)
    }
    /// Weight of coefficient `t` of response row `j` before the digit factor:
    /// `Abar_j * alpha^t + g * sigma(t) * xi^(t / k) * B_(j * k + t % k)(xi)`.
    ///
    /// The response weight at digit `l` is this value times `digit_powers()[l]`.
    /// Requires `j < m` and `t < degree`; any other index is
    /// [`AkitaError::InvalidProof`]. Padded coefficients `t >= degree` have
    /// weight zero and are not served here, so a table over the padded
    /// coefficient domain must supply those zeros itself.
    pub fn coefficient_weight(&self, j: usize, t: usize) -> Result<F, AkitaError> {
        let k = self.layout.k();
        let row = checked::mul_add(j, k, t % k).ok_or(AkitaError::InvalidProof)?;
        Ok(*self.abar.get(j).ok_or(AkitaError::InvalidProof)?
            * *self.alpha_powers.get(t).ok_or(AkitaError::InvalidProof)?
            + self.g()?
                * signed_field::<F>(self.layout.sigma(t)?)
                * *self.xi_powers.get(t / k).ok_or(AkitaError::InvalidProof)?
                * *self.binary_rows.get(row).ok_or(AkitaError::InvalidProof)?)
    }
    pub(crate) fn image_entry_weight(&self, e: usize) -> Result<F, AkitaError> {
        Ok(-*self
            .gamma_powers
            .get(e % self.layout.n_a())
            .ok_or(AkitaError::InvalidProof)?
            * *self
                .embedded_challenges
                .get(e / self.layout.n_a())
                .ok_or(AkitaError::InvalidProof)?)
    }
}
