//! Public linear weights of the lowered relation, by the shift recurrence.
//!
//! Every remainder weight is `g_t = (B * Y^t mod Phi)(alpha)` for a public
//! polynomial `B`: a gamma-batched matrix column for the response table, an
//! embedded fold challenge for the image table and for the prime left
//! opening, and the equality polynomial for the structured terminals. All `D`
//! values cost `O(D)` field operations and need no root of unity in the
//! challenge field.
//!
//! The response and image tables hold stored base-16 digits with the digit
//! axis innermost. A table weight is a coefficient weight times the digit
//! factor `16^l`; the coefficient weights are the compact tables built here.
//! The prime left opening holds field elements and has no digit axis.

use super::{signed_field, zero_vec, LoweredPublic, LoweredRootLayout};
use crate::BinaryClearSetup;
use akita_algebra::{
    offset_eq::eq_eval_at_index,
    ring::{shifted_remainder_evaluations, MinusTrinomial, PlusTrinomial, TrinomialModulus},
    EqPolynomial,
};
use akita_error::{checked, AkitaError};
use akita_types::{RelationPolynomialKind, TrinomialSign};
use jolt_field::Field;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// `output[t] = (B * Y^t mod Phi)(alpha)` for the layout's commitment trinomial.
fn shifted_evaluations<F: Field>(
    layout: &LoweredRootLayout,
    coefficients: &[F],
    alpha: F,
    output: &mut [F],
) -> Result<(), AkitaError> {
    match layout.polynomial().kind() {
        RelationPolynomialKind::Trinomial(TrinomialSign::Minus) => {
            shifted_remainder_evaluations::<F, MinusTrinomial>(coefficients, alpha, output)
        }
        RelationPolynomialKind::Trinomial(TrinomialSign::Plus) => {
            shifted_remainder_evaluations::<F, PlusTrinomial>(coefficients, alpha, output)
        }
        _ => {
            return Err(AkitaError::InvalidSetup(
                "remainder weights require a trinomial".into(),
            ))
        }
    }
    .map_err(|error| AkitaError::InvalidInput(error.to_string()))
}

/// `z[s] = (e * Y^s mod Phi)(alpha)` for `e[t] = eq(rho_t, t)`, `t < D`.
///
/// For any `B`, `sum_t e[t] * (B * Y^t mod Phi)(alpha) = sum_s B[s] * z[s]`:
/// both sides are `(B * e mod Phi)(alpha)`.
fn equality_shift<F: Field>(
    layout: &LoweredRootLayout,
    rho_t: &[F],
    alpha: F,
) -> Result<Vec<F>, AkitaError> {
    let mut e = zero_vec(layout.degree())?;
    for (t, value) in e.iter_mut().enumerate() {
        *value = eq_eval_at_index(rho_t, t);
    }
    let mut z = zero_vec(layout.degree())?;
    shifted_evaluations(layout, &e, alpha, &mut z)?;
    Ok(z)
}

/// `<iota(Ch), z>` over the nonzero coefficients of an embedded fold
/// challenge.
fn challenge_contraction<F: Field>(challenge: &[F], z: &[F]) -> F {
    let mut evaluation = F::zero();
    for (&coefficient, &shift) in challenge.iter().zip(z) {
        if !coefficient.is_zero() {
            evaluation += coefficient * shift;
        }
    }
    evaluation
}

/// Response-weight factor in coefficient-low, ring-element-high order.
///
/// Column `j` holds `g_t(sum_i gamma^i * A_ij)` in its first `D` entries, plus
/// the parity weight when the relation has the parity row and
/// `eta * eq(r_ring, j) * alpha^t` when it has the prime row; coefficient
/// padding stays zero. Multiplying this table by `public.digit_powers()` in
/// digit-innermost order gives the response weights.
pub fn coefficient_weights<F: Field, const D: usize, M: TrinomialModulus>(
    layout: &LoweredRootLayout,
    public: &LoweredPublic<F>,
    setup: &BinaryClearSetup<D, M>,
) -> Result<Vec<F>, AkitaError> {
    public.validate_layout(layout)?;
    layout.validate_setup(setup)?;
    let padded = layout.padded_coefficients();
    let len = checked::product([layout.m(), padded]).ok_or(AkitaError::InvalidProof)?;
    if padded < D {
        return Err(AkitaError::InvalidProof);
    }
    let mut weights = zero_vec(len)?;
    let alpha = public.challenges.alpha;
    let fill = |(j, column): (usize, &mut [F])| -> Result<(), AkitaError> {
        let mut batched = zero_vec(D)?;
        for (i, &gamma) in public.gamma_powers.iter().enumerate() {
            let start = checked::mul_add(i, layout.m(), j)
                .and_then(|entry| checked::product([entry, D]))
                .ok_or(AkitaError::InvalidProof)?;
            let a = setup
                .matrix()
                .get(checked::range(start, D).ok_or(AkitaError::InvalidProof)?)
                .ok_or(AkitaError::InvalidProof)?;
            for (destination, &coefficient) in batched.iter_mut().zip(a) {
                *destination += gamma * F::from_u64(u64::from(coefficient));
            }
        }
        let live = column.get_mut(..D).ok_or(AkitaError::InvalidProof)?;
        shifted_evaluations(layout, &batched, alpha, live)?;
        if let Some(parity) = &public.parity {
            for (t, weight) in live.iter_mut().enumerate() {
                *weight += parity.weight(layout, j, t)?;
            }
        }
        if let Some(prime) = &public.prime {
            // The packed response coefficient is not reduced: its degree is
            // below `D`.
            let ring = prime.eta * *prime.ring_weights.get(j).ok_or(AkitaError::InvalidProof)?;
            for (weight, &power) in live.iter_mut().zip(&prime.alpha_powers) {
                *weight += ring * power;
            }
        }
        Ok(())
    };
    #[cfg(feature = "parallel")]
    weights
        .par_chunks_mut(padded)
        .enumerate()
        .try_for_each(fill)?;
    #[cfg(not(feature = "parallel"))]
    weights.chunks_mut(padded).enumerate().try_for_each(fill)?;
    Ok(weights)
}

/// Image-weight factor in coefficient-low, image-entry-high order: entry
/// `(col, i)` holds `-gamma^i * g_t(iota(Ch_col))` in its first `D`
/// coefficients. Coefficient tails and entry padding are zero. Multiplying
/// this table by `public.image_digit_powers()` in digit-innermost order gives
/// the image weights.
pub fn image_weights<F: Field>(
    layout: &LoweredRootLayout,
    public: &LoweredPublic<F>,
) -> Result<Vec<F>, AkitaError> {
    public.validate_layout(layout)?;
    let degree = layout.degree();
    let padded = layout.padded_coefficients();
    let column_len = checked::product([layout.n_a(), padded]).ok_or(AkitaError::InvalidProof)?;
    if public.embedded_challenges.len() != layout.columns() || padded < degree || column_len == 0 {
        return Err(AkitaError::InvalidProof);
    }
    let len = checked::exact_div(layout.image_len(), public.image_digit_powers.len())
        .ok_or(AkitaError::InvalidProof)?;
    let mut weights = zero_vec(len)?;
    let alpha = public.challenges.alpha;
    let fill = |(entries, challenge): (&mut [F], &Vec<F>)| -> Result<(), AkitaError> {
        let mut shifted = zero_vec(degree)?;
        shifted_evaluations(layout, challenge, alpha, &mut shifted)?;
        for (entry, &gamma) in entries.chunks_exact_mut(padded).zip(&public.gamma_powers) {
            for (destination, &value) in entry.iter_mut().zip(&shifted) {
                *destination = -gamma * value;
            }
        }
        Ok(())
    };
    #[cfg(feature = "parallel")]
    weights
        .par_chunks_mut(column_len)
        .zip(public.embedded_challenges.par_iter())
        .try_for_each(fill)?;
    #[cfg(not(feature = "parallel"))]
    weights
        .chunks_mut(column_len)
        .zip(public.embedded_challenges.iter())
        .try_for_each(fill)?;
    Ok(weights)
}

/// Structured response terminal. One matrix pass accumulates
/// `Abar_i = sum_j eq(rho_j, j) * A_ij` in `n_A * D` field slots, which are
/// then contracted against `gamma^i * z[s]`; the response weights are never
/// materialized. The parity row and the prime row add one product of
/// per-axis sums each.
pub fn witness_weight_mle<F: Field, const D: usize, M: TrinomialModulus>(
    layout: &LoweredRootLayout,
    public: &LoweredPublic<F>,
    setup: &BinaryClearSetup<D, M>,
    rho: &[F],
) -> Result<F, AkitaError> {
    public.validate_layout(layout)?;
    layout.validate_setup(setup)?;
    if rho.len() != layout.witness_log_len() {
        return Err(AkitaError::InvalidInput(
            "response MLE point dimension mismatch".into(),
        ));
    }
    let digit_bits = layout.response_layout().digit_depth().trailing_zeros() as usize;
    let coefficient_bits = layout.padded_coefficients().trailing_zeros() as usize;
    let component_bits = layout.k().trailing_zeros() as usize;
    let (rho_l, rest) = rho
        .split_at_checked(digit_bits)
        .ok_or(AkitaError::InvalidProof)?;
    let (rho_t, rho_j) = rest
        .split_at_checked(coefficient_bits)
        .ok_or(AkitaError::InvalidProof)?;
    let gadget = public
        .digit_powers
        .iter()
        .enumerate()
        .fold(F::zero(), |acc, (l, &digit)| {
            acc + eq_eval_at_index(rho_l, l) * digit
        });
    let z = equality_shift(layout, rho_t, public.challenges.alpha)?;
    if checked::pow2(rho_j.len()) != Some(layout.m()) {
        return Err(AkitaError::InvalidProof);
    }
    let row_len = checked::product([layout.m(), D]).ok_or(AkitaError::InvalidProof)?;
    let mut contracted = zero_vec::<F>(D)?;
    let mut a_part = F::zero();
    for (row, &gamma) in setup
        .matrix()
        .chunks_exact(row_len)
        .zip(&public.gamma_powers)
    {
        contracted.fill(F::zero());
        for (j, element) in row.chunks_exact(D).enumerate() {
            let weight = eq_eval_at_index(rho_j, j);
            for (destination, &coefficient) in contracted.iter_mut().zip(element) {
                *destination += weight * F::from_u64(u64::from(coefficient));
            }
        }
        let mut row_sum = F::zero();
        for (&coefficient, &shift) in contracted.iter().zip(&z) {
            row_sum += coefficient * shift;
        }
        a_part += gamma * row_sum;
    }
    let mut result = a_part * gadget;
    if let Some(parity) = &public.parity {
        let (rho_c, rho_s) = rho_t
            .split_at_checked(component_bits)
            .ok_or(AkitaError::InvalidProof)?;
        let mut scalar = F::zero();
        for (s, &power) in parity.xi_powers.iter().enumerate() {
            let t = checked::product([s, layout.k()]).ok_or(AkitaError::InvalidProof)?;
            scalar += eq_eval_at_index(rho_s, s) * signed_field::<F>(layout.sigma(t)?) * power;
        }
        let mut binary = F::zero();
        for (r, &value) in parity.binary_rows.iter().enumerate() {
            binary += eq_eval_at_index(rho_c, r % layout.k())
                * eq_eval_at_index(rho_j, r / layout.k())
                * value;
        }
        result += parity.factor * gadget * scalar * binary;
    }
    if let Some(prime) = &public.prime {
        let mut coefficients = F::zero();
        for (t, &power) in prime.alpha_powers.iter().enumerate() {
            coefficients += eq_eval_at_index(rho_t, t) * power;
        }
        result += prime.eta * gadget * EqPolynomial::mle(rho_j, &prime.ring_point)? * coefficients;
    }
    Ok(result)
}

/// Sum of all image weights `-gamma^i * g_t(iota(Ch_col))` over `col`,
/// `i < n_A` and `t < D`, by the identity of `equality_shift` with `e` the
/// all-ones vector: `sum_t g_t(B) = <B, z>`.
pub(super) fn image_weight_sum<F: Field>(public: &LoweredPublic<F>) -> Result<F, AkitaError> {
    let layout = &public.layout;
    let mut ones = zero_vec(layout.degree())?;
    ones.fill(F::one());
    let mut z = zero_vec(layout.degree())?;
    shifted_evaluations(layout, &ones, public.challenges.alpha, &mut z)?;
    let challenges = public
        .embedded_challenges
        .iter()
        .fold(F::zero(), |sum, challenge| {
            sum + challenge_contraction(challenge, &z)
        });
    let rows = public
        .gamma_powers
        .iter()
        .fold(F::zero(), |sum, &gamma| sum + gamma);
    Ok(-rows * challenges)
}

/// Structured image terminal in digit-low, coefficient, image-entry-high order:
/// `-gadget(rho_l) * sum_(col,i) eq(rho_e, col*n_A + i) * gamma^i * <iota(Ch_col), z>`
/// with `gadget(rho_l) = sum_l eq(rho_l, l) * 16^l` over the weighted digits.
pub fn image_weight_mle<F: Field>(
    layout: &LoweredRootLayout,
    public: &LoweredPublic<F>,
    rho_y: &[F],
) -> Result<F, AkitaError> {
    public.validate_layout(layout)?;
    if rho_y.len() != layout.image_log_len() {
        return Err(AkitaError::InvalidInput(
            "image MLE point dimension mismatch".into(),
        ));
    }
    let (rho_l, rest) = rho_y
        .split_at_checked(public.image_digit_powers.len().trailing_zeros() as usize)
        .ok_or(AkitaError::InvalidProof)?;
    let (rho_t, rho_e) = rest
        .split_at_checked(layout.padded_coefficients().trailing_zeros() as usize)
        .ok_or(AkitaError::InvalidProof)?;
    let gadget = public
        .image_digit_powers
        .iter()
        .enumerate()
        .fold(F::zero(), |acc, (l, &digit)| {
            acc + eq_eval_at_index(rho_l, l) * digit
        });
    let z = equality_shift(layout, rho_t, public.challenges.alpha)?;
    let mut result = F::zero();
    for (col, challenge) in public.embedded_challenges.iter().enumerate() {
        let evaluation = challenge_contraction(challenge, &z);
        for (i, &gamma) in public.gamma_powers.iter().enumerate() {
            let entry = checked::mul_add(col, layout.n_a(), i).ok_or(AkitaError::InvalidProof)?;
            result -= eq_eval_at_index(rho_e, entry) * gamma * evaluation;
        }
    }
    Ok(gadget * result)
}

/// Weights of the prime row on the prime left opening, coefficient-low and
/// column-high: column `col` holds `g_t(iota(Ch_col))` in its first `D`
/// coefficients and zero on the coefficient tail. There is no factor of
/// `gamma` or `eta`: the row's batching scalar multiplies the claim `y_P`.
pub fn prime_row_weights<F: Field>(
    layout: &LoweredRootLayout,
    public: &LoweredPublic<F>,
) -> Result<Vec<F>, AkitaError> {
    public.validate_layout(layout)?;
    let degree = layout.degree();
    let padded = layout.padded_coefficients();
    if public.embedded_challenges.len() != layout.columns() || padded < degree || padded == 0 {
        return Err(AkitaError::InvalidProof);
    }
    let mut weights = zero_vec(layout.prime_len())?;
    let alpha = public.challenges.alpha;
    let fill = |(column, challenge): (&mut [F], &Vec<F>)| -> Result<(), AkitaError> {
        let live = column.get_mut(..degree).ok_or(AkitaError::InvalidProof)?;
        shifted_evaluations(layout, challenge, alpha, live)
    };
    #[cfg(feature = "parallel")]
    weights
        .par_chunks_mut(padded)
        .zip(public.embedded_challenges.par_iter())
        .try_for_each(fill)?;
    #[cfg(not(feature = "parallel"))]
    weights
        .chunks_mut(padded)
        .zip(public.embedded_challenges.iter())
        .try_for_each(fill)?;
    Ok(weights)
}

/// Structured terminal of [`prime_row_weights`] at `rho = (rho_t, rho_col)`:
/// `sum_col eq(rho_col, col) * <iota(Ch_col), z>`.
pub fn prime_row_weight_mle<F: Field>(
    layout: &LoweredRootLayout,
    public: &LoweredPublic<F>,
    rho: &[F],
) -> Result<F, AkitaError> {
    public.validate_layout(layout)?;
    if rho.len() != layout.prime_log_len() {
        return Err(AkitaError::InvalidInput(
            "prime opening MLE point dimension mismatch".into(),
        ));
    }
    let (rho_t, rho_col) = rho
        .split_at_checked(layout.padded_coefficients().trailing_zeros() as usize)
        .ok_or(AkitaError::InvalidProof)?;
    let z = equality_shift(layout, rho_t, public.challenges.alpha)?;
    Ok(public
        .embedded_challenges
        .iter()
        .enumerate()
        .fold(F::zero(), |sum, (col, challenge)| {
            sum + eq_eval_at_index(rho_col, col) * challenge_contraction(challenge, &z)
        }))
}
