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

use super::{horner, powers, remainder_evaluations, signed_field, zero_vec, LoweredRootLayout};
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

/// Validated public data for one lowered relation; construction never scans A.
#[derive(Clone, Debug)]
pub struct LoweredPublic<F> {
    pub(crate) layout: LoweredRootLayout,
    pub(crate) challenges: LoweredChallenges<F>,
    pub(crate) binary_rows: Vec<F>,
    pub(crate) embedded_challenges: Vec<Vec<F>>,
    pub(crate) alpha_powers: Vec<F>,
    pub(crate) remainder_evaluations: Vec<F>,
    pub(crate) xi_powers: Vec<F>,
    pub(crate) gamma_powers: Vec<F>,
    pub(crate) digit_powers: Vec<F>,
    pub(crate) a_carry_terms: Vec<F>,
    pub(crate) parity_rhs: F,
    c_pub: F,
}

impl<F: SmoothFftField> LoweredPublic<F> {
    /// Check public geometry, fold challenges, left expansion and clear ranges.
    /// Cached matrix-bound offset remainders supply the matrix constant.
    #[allow(clippy::too_many_arguments)]
    pub fn new<const D: usize, M: TrinomialModulus>(
        layout: &LoweredRootLayout,
        setup: &BinaryClearSetup<F, D, M>,
        claim: &BinaryEvaluationClaim,
        u: &[BinaryField162],
        fold_challenges: &[BinaryChallenge],
        a_carry: &[i128],
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
        if fold_challenges.len() != layout.columns()
            || a_carry.len() != layout.encoding().a_carry_len()
            || parity_quotient.len() != 161
            || parity_carry.len() != 162
        {
            return Err(AkitaError::InvalidProof);
        }
        if let Some(range) = layout.encoding().a_carry() {
            let (lower, upper) = range.interval();
            if a_carry.iter().any(|&value| value < lower || value > upper) {
                return Err(AkitaError::InvalidProof);
            }
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
        let remainder_evaluations = remainder_evaluations(layout.polynomial(), challenges.alpha)?;
        let xi_powers = powers(challenges.xi, 162)?;
        let gamma_len = checked::sum([layout.n_a(), 1]).ok_or(AkitaError::InvalidProof)?;
        let gamma_powers = powers(challenges.gamma, gamma_len)?;
        let digit_base = F::from_u64(
            1u64.checked_shl(layout.encoding().base().bits())
                .ok_or(AkitaError::InvalidProof)?,
        );
        let digit_powers = powers(digit_base, layout.response_layout().digit_depth())?;
        let row_point = claim
            .point
            .get(..setup.row_vars())
            .ok_or(AkitaError::InvalidProof)?;
        let binary_weights = equality_weights(row_point)?;
        let mut binary_rows = zero_vec(layout.scalar_rows())?;
        let binary_row = |weight| -> Result<F, AkitaError> {
            Ok(horner(
                scalar_from_binary::<F>(weight)?.coefficients(),
                challenges.xi,
            ))
        };
        #[cfg(feature = "parallel")]
        binary_rows
            .par_iter_mut()
            .zip(&binary_weights)
            .try_for_each(|(destination, &weight)| -> Result<(), AkitaError> {
                *destination = binary_row(weight)?;
                Ok(())
            })?;
        #[cfg(not(feature = "parallel"))]
        for (destination, &weight) in binary_rows.iter_mut().zip(&binary_weights) {
            *destination = binary_row(weight)?;
        }
        let mut embedded_challenges = Vec::new();
        embedded_challenges
            .try_reserve_exact(layout.columns())
            .map_err(|_| AkitaError::InvalidProof)?;
        let mut u_challenge_sum = F::zero();
        for (challenge, &u_value) in fold_challenges.iter().zip(u) {
            let scalar = challenge_scalar::<F>(challenge).map_err(|_| AkitaError::InvalidProof)?;
            let embedded =
                embed_scalar::<F, 162, D, M>(&scalar).map_err(|_| AkitaError::InvalidProof)?;
            let mut coefficients = zero_vec(D)?;
            coefficients.copy_from_slice(embedded.coefficients());
            embedded_challenges.push(coefficients);
            u_challenge_sum += horner(
                scalar_from_binary::<F>(u_value)?.coefficients(),
                challenges.xi,
            ) * horner(scalar.coefficients(), challenges.xi);
        }
        let mut a_carry_terms = Vec::new();
        if let Some(q0) = setup.commitment_modulus().small_modulus() {
            a_carry_terms = zero_vec(layout.n_a())?;
            for (destination, row) in a_carry_terms.iter_mut().zip(a_carry.chunks_exact(D)) {
                *destination = F::from_u64(u64::from(q0))
                    * row.iter().rev().fold(F::zero(), |acc, &value| {
                        acc * challenges.alpha + signed_field::<F>(value)
                    });
            }
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
            binary_rows,
            embedded_challenges,
            alpha_powers,
            remainder_evaluations,
            xi_powers,
            gamma_powers,
            digit_powers,
            a_carry_terms,
            parity_rhs,
            c_pub: F::zero(),
        };
        for (&weight, h) in result.gamma_powers.iter().zip(setup.a_offset_remainders()) {
            result.c_pub += weight * horner(h.coefficients(), challenges.alpha);
        }
        // Offset and sign depend on the scalar coefficient, not its component.
        // The unchanged parity offset therefore factors across all binary rows.
        let mut scalar_offset = F::zero();
        for (s, &power) in result.xi_powers.iter().enumerate() {
            let t = checked::product([s, layout.k()]).ok_or(AkitaError::InvalidProof)?;
            scalar_offset +=
                signed_field::<F>(layout.off(t)?) * signed_field::<F>(layout.sigma(t)?) * power;
        }
        let row_sum = result
            .binary_rows
            .iter()
            .copied()
            .fold(F::zero(), |sum, value| sum + value);
        result.c_pub += result.g()? * scalar_offset * row_sum;
        for (&weight, &carry) in result.gamma_powers.iter().zip(&result.a_carry_terms) {
            result.c_pub += weight * carry;
        }
        result.c_pub += result.g()? * result.parity_rhs;
        Ok(result)
    }
    pub fn c_pub(&self) -> F {
        self.c_pub
    }
    /// Digit factors `(2^b)^l`, lowest digit first.
    pub fn digit_powers(&self) -> &[F] {
        &self.digit_powers
    }
    /// Functional coefficient factors `alpha^t`, for all natural coefficients.
    pub fn alpha_powers(&self) -> &[F] {
        &self.alpha_powers
    }
    /// Row factors `gamma^i`, including the parity factor at index `n_A`.
    pub fn gamma_powers(&self) -> &[F] {
        &self.gamma_powers
    }
    /// Canonical coefficient rows of the embedded fold challenges, one per column.
    pub fn embedded_challenge_coefficients(&self) -> &[Vec<F>] {
        &self.embedded_challenges
    }
    pub fn challenges(&self) -> LoweredChallenges<F> {
        self.challenges
    }
    /// Reject public weights prepared for a different admitted table layout.
    pub fn validate_layout(&self, layout: &LoweredRootLayout) -> Result<(), AkitaError> {
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
    /// Parity part `g * sigma(t) * xi^(t/k) * B_(j*k+t%k)(xi)` of a
    /// coefficient weight. The caller adds the matrix multiplication adjoint.
    /// Requires `j < m`, `t < degree`; padded coefficients have zero weight
    /// and are supplied by the table owner, not by this accessor.
    pub fn parity_coefficient_weight(&self, j: usize, t: usize) -> Result<F, AkitaError> {
        if j >= self.layout.m() {
            return Err(AkitaError::InvalidProof);
        }
        let k = self.layout.k();
        let row = checked::mul_add(j, k, t % k).ok_or(AkitaError::InvalidProof)?;
        Ok(self.g()?
            * signed_field::<F>(self.layout.sigma(t)?)
            * *self.xi_powers.get(t / k).ok_or(AkitaError::InvalidProof)?
            * *self.binary_rows.get(row).ok_or(AkitaError::InvalidProof)?)
    }
}
