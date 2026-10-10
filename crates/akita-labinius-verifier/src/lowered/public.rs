use akita_algebra::{
    binary::BinaryField162,
    ring::{embed_scalar, TrinomialModulus},
    EqPolynomial,
};
use akita_challenges::BinaryChallenge;
use akita_error::{checked, AkitaError};
use akita_params::sis::labinius::LABINIUS_BALANCED_LOG_BASIS;
use akita_types::RelationPolynomial;
use jolt_field::Field;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::{horner, powers, signed_field, zero_vec, LoweredRootLayout};
use crate::{
    endpoint::verify_left_expansion,
    source::{challenge_scalar, equality_weights, scalar_from_binary},
    BinaryClearSetup, BinaryEvaluationClaim,
};

/// Explicit evaluation and row-batching challenges of the commitment rows; no
/// sampling occurs here.
#[derive(Clone, Copy, Debug)]
pub struct LoweredChallenges<F> {
    pub alpha: F,
    pub gamma: F,
}

/// Public inputs of the parity row, present exactly with a binary claim: the
/// frontend's claim, the binary left opening, the clear quotient and carry,
/// and the row's evaluation point.
#[derive(Clone, Copy, Debug)]
pub struct LoweredParity<'a, F> {
    pub claim: &'a BinaryEvaluationClaim,
    pub u: &'a [BinaryField162],
    pub quotient: &'a [i128],
    pub carry: &'a [i128],
    pub xi: F,
}

/// Public inputs of the prime row, present exactly with a prime claim: the
/// claim's ring point and the row's batching scalar.
#[derive(Clone, Copy, Debug)]
pub struct LoweredPrime<'a, F> {
    pub ring_point: &'a [F],
    pub eta: F,
}

/// The parity row at `xi`, scaled by `factor = gamma^(n_A)`.
#[derive(Clone, Debug)]
pub(super) struct ParityRow<F> {
    /// `B_r(xi)`, one per scalar row `r = j * k + c`.
    pub(super) binary_rows: Vec<F>,
    pub(super) xi_powers: Vec<F>,
    pub(super) factor: F,
}

/// The prime row at `alpha`, scaled by `eta`.
#[derive(Clone, Debug)]
pub(super) struct PrimeRow<F> {
    pub(super) ring_point: Vec<F>,
    /// `eq(ring_point, j)`, one per ring element.
    pub(super) ring_weights: Vec<F>,
    /// `alpha^t` for `t < D`.
    pub(super) alpha_powers: Vec<F>,
    pub(super) eta: F,
}

/// Validated public data for one lowered relation; construction never scans A.
#[derive(Clone, Debug)]
pub struct LoweredPublic<F> {
    pub(crate) layout: LoweredRootLayout,
    pub(crate) challenges: LoweredChallenges<F>,
    pub(crate) embedded_challenges: Vec<Vec<F>>,
    pub(crate) gamma_powers: Vec<F>,
    pub(crate) digit_powers: Vec<F>,
    pub(crate) image_digit_powers: Vec<F>,
    pub(super) parity: Option<ParityRow<F>>,
    pub(super) prime: Option<PrimeRow<F>>,
    c_pub: F,
}

fn integer_evaluation<F: Field>(values: &[i128], point: F) -> F {
    values.iter().rev().fold(F::zero(), |acc, &value| {
        acc * point + signed_field::<F>(value)
    })
}

impl<F: Field> ParityRow<F> {
    /// Check the left expansion and the clear ranges, and return the row with
    /// its part of the constant: `factor` times the response offsets seen
    /// through the row plus its right-hand side.
    fn new<const D: usize, M: TrinomialModulus>(
        layout: &LoweredRootLayout,
        setup: &BinaryClearSetup<D, M>,
        fold_challenges: &[BinaryChallenge],
        input: LoweredParity<'_, F>,
        factor: F,
    ) -> Result<(Self, F), AkitaError> {
        let LoweredParity {
            claim,
            u,
            quotient,
            carry,
            xi,
        } = input;
        if claim.point.len() != setup.num_vars() || u.len() != layout.columns() {
            return Err(AkitaError::InvalidInput(
                "lowered public claim geometry mismatch".into(),
            ));
        }
        verify_left_expansion(setup, claim, u)?;
        if quotient.len() != 161 || carry.len() != 162 {
            return Err(AkitaError::InvalidProof);
        }
        for (values, range) in [
            (quotient, layout.encoding().quotient()),
            (carry, layout.encoding().carry()),
        ] {
            let (lower, upper) = range.interval();
            if values.iter().any(|&v| v < lower || v > upper) {
                return Err(AkitaError::InvalidProof);
            }
        }
        let xi_powers = powers(xi, 162)?;
        let row_point = claim
            .point
            .get(..setup.row_vars())
            .ok_or(AkitaError::InvalidProof)?;
        let binary_weights = equality_weights(row_point)?;
        let mut binary_rows = zero_vec(layout.scalar_rows())?;
        let binary_row = |weight| -> Result<F, AkitaError> {
            Ok(horner(scalar_from_binary::<F>(weight)?.coefficients(), xi))
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
        let mut u_challenge_sum = F::zero();
        for (challenge, &u_value) in fold_challenges.iter().zip(u) {
            let scalar = challenge_scalar::<F>(challenge).map_err(|_| AkitaError::InvalidProof)?;
            u_challenge_sum += horner(scalar_from_binary::<F>(u_value)?.coefficients(), xi)
                * horner(scalar.coefficients(), xi);
        }
        let rhs = u_challenge_sum
            + RelationPolynomial::plus_trinomial(162)?.evaluate_modulus_at(xi)?
                * integer_evaluation(quotient, xi)
            + F::from_u64(2) * integer_evaluation(carry, xi);
        // Offset and sign depend on the scalar coefficient, not its component.
        // The unchanged parity offset therefore factors across all binary rows.
        let mut scalar_offset = F::zero();
        for (s, &power) in xi_powers.iter().enumerate() {
            let t = checked::product([s, layout.k()]).ok_or(AkitaError::InvalidProof)?;
            scalar_offset +=
                signed_field::<F>(layout.off(t)?) * signed_field::<F>(layout.sigma(t)?) * power;
        }
        let row_sum = binary_rows
            .iter()
            .copied()
            .fold(F::zero(), |sum, value| sum + value);
        let row = Self {
            binary_rows,
            xi_powers,
            factor,
        };
        Ok((row, factor * (scalar_offset * row_sum + rhs)))
    }

    /// Parity part `factor * sigma(t) * xi^(t/k) * B_(j*k+t%k)(xi)` of the
    /// weight of coefficient `t < D` of ring element `j < m`.
    pub(super) fn weight(
        &self,
        layout: &LoweredRootLayout,
        j: usize,
        t: usize,
    ) -> Result<F, AkitaError> {
        if j >= layout.m() {
            return Err(AkitaError::InvalidProof);
        }
        let k = layout.k();
        let row = checked::mul_add(j, k, t % k).ok_or(AkitaError::InvalidProof)?;
        Ok(self.factor
            * signed_field::<F>(layout.sigma(t)?)
            * *self.xi_powers.get(t / k).ok_or(AkitaError::InvalidProof)?
            * *self.binary_rows.get(row).ok_or(AkitaError::InvalidProof)?)
    }
}

impl<F: Field> PrimeRow<F> {
    /// The row with its part of the constant, `eta * sum_t alpha^t * off(t)`:
    /// the response offsets seen through the row, whose ring weights
    /// `eq(ring_point, j)` sum to one.
    fn new(
        layout: &LoweredRootLayout,
        input: LoweredPrime<'_, F>,
        alpha: F,
    ) -> Result<(Self, F), AkitaError> {
        if checked::pow2(input.ring_point.len()) != Some(layout.m()) {
            return Err(AkitaError::InvalidInput(
                "prime ring point dimension mismatch".into(),
            ));
        }
        let ring_weights = EqPolynomial::evals_serial(input.ring_point, None)?;
        let alpha_powers = powers(alpha, layout.degree())?;
        let mut offset = F::zero();
        for (t, &power) in alpha_powers.iter().enumerate() {
            offset += signed_field::<F>(layout.off(t)?) * power;
        }
        let mut ring_point = zero_vec(input.ring_point.len())?;
        ring_point.copy_from_slice(input.ring_point);
        let row = Self {
            ring_point,
            ring_weights,
            alpha_powers,
            eta: input.eta,
        };
        Ok((row, input.eta * offset))
    }
}

impl<F: Field> LoweredPublic<F> {
    /// Check public geometry, fold challenges and clear ranges, and the left
    /// expansion when the relation has the parity row.
    /// Cached matrix-bound offset remainders supply the matrix constant.
    ///
    /// The relation is the commitment rows at `alpha` batched by powers of
    /// `gamma`, plus the parity row at `xi` with factor `gamma^(n_A)` when
    /// `parity` is given, plus the prime row at `alpha` with factor `eta` when
    /// `prime` is given.
    ///
    /// `F` is the field the challenges live in. Every public integer is lifted
    /// into it; nothing here needs more than field arithmetic.
    ///
    /// The constant is taken on the stored digit tables: the response offsets
    /// and the image offset are moved into it, so both linear terms are inner
    /// products of stored digits with `weight * 16^l`.
    pub fn new<const D: usize, M: TrinomialModulus>(
        layout: &LoweredRootLayout,
        setup: &BinaryClearSetup<D, M>,
        fold_challenges: &[BinaryChallenge],
        a_carry: &[i128],
        challenges: LoweredChallenges<F>,
        parity: Option<LoweredParity<'_, F>>,
        prime: Option<LoweredPrime<'_, F>>,
    ) -> Result<Self, AkitaError> {
        layout.validate_setup(setup)?;
        if fold_challenges.len() != layout.columns()
            || a_carry.len() != layout.encoding().a_carry_len()
        {
            return Err(AkitaError::InvalidProof);
        }
        for challenge in fold_challenges {
            challenge
                .validate(setup.profile())
                .map_err(|_| AkitaError::InvalidProof)?;
        }
        let (lower, upper) = layout.encoding().a_carry().interval();
        if a_carry.iter().any(|&v| v < lower || v > upper) {
            return Err(AkitaError::InvalidProof);
        }
        let gamma_powers = powers(challenges.gamma, layout.n_a())?;
        // Padding slots of either digit axis carry no weight.
        let digit_base = F::from_u64(1 << LABINIUS_BALANCED_LOG_BASIS);
        let digit_axis = |slots: usize, count: usize| -> Result<Vec<F>, AkitaError> {
            let mut axis = zero_vec(slots)?;
            let weighted = axis.get_mut(..count).ok_or(AkitaError::InvalidProof)?;
            weighted.copy_from_slice(&powers(digit_base, count)?);
            Ok(axis)
        };
        let digit_powers = digit_axis(
            layout.encoding().response_digit_slots(),
            layout.encoding().response_digit_count(),
        )?;
        let image_digit_powers = digit_axis(
            layout.encoding().image_digit_slots(),
            layout.encoding().image_digit_count(),
        )?;
        let mut embedded_challenges = Vec::new();
        embedded_challenges
            .try_reserve_exact(layout.columns())
            .map_err(|_| AkitaError::InvalidProof)?;
        for challenge in fold_challenges {
            let scalar = challenge_scalar::<F>(challenge).map_err(|_| AkitaError::InvalidProof)?;
            let embedded =
                embed_scalar::<F, 162, D, M>(&scalar).map_err(|_| AkitaError::InvalidProof)?;
            let mut coefficients = zero_vec(D)?;
            coefficients.copy_from_slice(embedded.coefficients());
            embedded_challenges.push(coefficients);
        }
        let mut c_pub = F::zero();
        let q = F::from_u64(u64::from(setup.modulus()));
        for ((&weight, h), carry) in gamma_powers
            .iter()
            .zip(setup.a_offset_remainders().chunks_exact(D))
            .zip(a_carry.chunks_exact(D))
        {
            c_pub += weight
                * (integer_evaluation(h, challenges.alpha)
                    + q * integer_evaluation(carry, challenges.alpha));
        }
        let parity = match parity {
            Some(input) => {
                // The parity row follows the commitment rows in the powers of
                // `gamma`.
                let factor = gamma_powers
                    .last()
                    .map_or(F::one(), |&last| last * challenges.gamma);
                let (row, constant) =
                    ParityRow::new(layout, setup, fold_challenges, input, factor)?;
                c_pub += constant;
                Some(row)
            }
            None => None,
        };
        let prime = match prime {
            Some(input) => {
                let (row, constant) = PrimeRow::new(layout, input, challenges.alpha)?;
                c_pub += constant;
                Some(row)
            }
            None => None,
        };
        let mut result = Self {
            layout: layout.clone(),
            challenges,
            embedded_challenges,
            gamma_powers,
            digit_powers,
            image_digit_powers,
            parity,
            prime,
            c_pub,
        };
        // A stored image coefficient is `T + image_offset`, so the offset
        // times the sum of the image weights moves to the constant.
        result.c_pub += F::from_u128(layout.encoding().image_offset())
            * super::weights::image_weight_sum(&result)?;
        Ok(result)
    }

    pub fn c_pub(&self) -> F {
        self.c_pub
    }
    /// Linear claim of the response instance: `c_pub - y_Y`, plus
    /// `eta * y_P` exactly when the relation has the prime row, where `y_P`
    /// is the row's inner product with the prime left opening.
    pub fn response_claim(&self, y_y: F, y_p: Option<F>) -> Result<F, AkitaError> {
        match (&self.prime, y_p) {
            (Some(prime), Some(y_p)) => Ok(self.c_pub - y_y + prime.eta * y_p),
            (None, None) => Ok(self.c_pub - y_y),
            _ => Err(AkitaError::InvalidInput(
                "prime row claim does not match the lowered relation".into(),
            )),
        }
    }
    /// Response digit factors `16^l`, lowest digit first, then zero for each
    /// padding slot of the digit axis.
    pub fn digit_powers(&self) -> &[F] {
        &self.digit_powers
    }
    /// Image digit factors, in the same form over the image digit axis.
    pub fn image_digit_powers(&self) -> &[F] {
        &self.image_digit_powers
    }
    /// Row factors `gamma^i` of the `n_A` commitment rows.
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
}
