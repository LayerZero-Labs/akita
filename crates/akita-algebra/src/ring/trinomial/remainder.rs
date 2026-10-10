//! Evaluations of the shifted remainders `B * Y^t mod Phi` at one point.
//!
//! Write `Phi = Y^D - c*Y^h + 1` with `h = D/2`, where `c = 1` for
//! [`MinusTrinomial`](super::MinusTrinomial) and `c = -1` for
//! [`PlusTrinomial`](super::PlusTrinomial). Multiplying a reduced polynomial
//! `X` by `Y` gives `Y*X = (Y*X mod Phi) + top(X) * Phi` exactly, where
//! `top(X)` is the coefficient of `Y^(D-1)`. Evaluating at `alpha` turns this
//! into a one-term recurrence, so all `D` values need `O(D)` field operations
//! and no root of unity.

use jolt_field::Field;

use super::{TrinomialError, TrinomialModulus};
use crate::fft::field_pow;

/// Write `g_t = (B * Y^t mod Phi)(alpha)` for every `0 <= t < D`.
///
/// `coefficients` holds `B` lowest degree first and fixes `D`; `output` must
/// have the same positive even length. With `rho(n) = (Y^n mod Phi)(alpha)`,
/// the result equals the dense definition `g_t = sum_s b[s] * rho(s + t)`.
///
/// ```text
/// top_t   = b[D-1-t] + (c * top_(t-h) if t >= h else 0)
/// g_0     = B(alpha)
/// g_(t+1) = alpha * g_t - top_t * Phi(alpha)
/// ```
///
/// # Errors
///
/// Returns [`TrinomialError::InvalidDegree`] for an empty or odd-length `B`,
/// and [`TrinomialError::CoefficientLength`] when `output` has another length.
pub fn shifted_remainder_evaluations<F: Field, M: TrinomialModulus>(
    coefficients: &[F],
    alpha: F,
    output: &mut [F],
) -> Result<(), TrinomialError> {
    let degree = coefficients.len();
    if degree == 0 || !degree.is_multiple_of(2) {
        return Err(TrinomialError::InvalidDegree { degree });
    }
    if output.len() != degree {
        return Err(TrinomialError::CoefficientLength {
            expected: degree,
            actual: output.len(),
        });
    }
    let half = degree / 2;
    let half_exponent =
        u64::try_from(half).map_err(|_| TrinomialError::InvalidDegree { degree })?;
    // `Phi = Y^D + middle*Y^h + 1`, so `c = -middle`.
    let minus = M::MIDDLE_COEFFICIENT < 0;

    // First pass: `output[t] = top_t`. A unit injected at `Y^h` by one
    // reduction reaches `Y^(D-1)` exactly `h` shifts later; the unit injected
    // at `Y^0` would need `D` shifts and never arrives within the range.
    let (low_coefficients, high_coefficients) = coefficients.split_at(half);
    let (low_tops, high_tops) = output.split_at_mut(half);
    for (top, &coefficient) in low_tops.iter_mut().zip(high_coefficients.iter().rev()) {
        *top = coefficient;
    }
    for ((top, &coefficient), &earlier) in high_tops
        .iter_mut()
        .zip(low_coefficients.iter().rev())
        .zip(low_tops.iter())
    {
        *top = if minus {
            coefficient + earlier
        } else {
            coefficient - earlier
        };
    }

    let alpha_half = field_pow(alpha, half_exponent);
    let modulus_at_alpha = if minus {
        alpha_half * alpha_half - alpha_half + F::one()
    } else {
        alpha_half * alpha_half + alpha_half + F::one()
    };
    // Second pass: replace each top by its evaluation and step the recurrence.
    let mut evaluation = coefficients
        .iter()
        .rev()
        .fold(F::zero(), |accumulator, &coefficient| {
            accumulator * alpha + coefficient
        });
    for slot in output.iter_mut() {
        let top = *slot;
        *slot = evaluation;
        evaluation = alpha * evaluation - top * modulus_at_alpha;
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::indexing_slicing, clippy::unwrap_used)]
mod tests {
    use jolt_field::{Field, Prime128Offset275, Prime31Offset19, Zero};
    use rand::{rngs::StdRng, RngCore, SeedableRng};

    use super::shifted_remainder_evaluations;
    use crate::ring::trinomial::{MinusTrinomial, PlusTrinomial, TrinomialError, TrinomialModulus};

    fn middle<F: Field, M: TrinomialModulus>() -> F {
        if M::MIDDLE_COEFFICIENT < 0 {
            -F::one()
        } else {
            F::one()
        }
    }

    fn horner<F: Field>(coefficients: &[F], alpha: F) -> F {
        coefficients
            .iter()
            .rev()
            .fold(F::zero(), |acc, &c| acc * alpha + c)
    }

    /// Dense definition: `g_t = sum_s b[s] * rho(s+t)`, with
    /// `rho(n) = alpha^n` below `D` and `Y^n = -middle*Y^(n-h) - Y^(n-D)` above.
    fn dense<F: Field, M: TrinomialModulus>(b: &[F], alpha: F) -> Vec<F> {
        let degree = b.len();
        let mut rho = vec![F::one(); 2 * degree - 1];
        for n in 1..rho.len() {
            rho[n] = if n < degree {
                rho[n - 1] * alpha
            } else {
                -middle::<F, M>() * rho[n - degree / 2] - rho[n - degree]
            };
        }
        (0..degree)
            .map(|t| {
                b.iter()
                    .enumerate()
                    .fold(F::zero(), |sum, (s, &c)| sum + c * rho[s + t])
            })
            .collect()
    }

    /// Multiply by `Y` and reduce coefficient-wise, then evaluate by Horner.
    fn stepwise<F: Field, M: TrinomialModulus>(b: &[F], alpha: F) -> Vec<F> {
        let degree = b.len();
        let mut current = b.to_vec();
        let mut values = Vec::with_capacity(degree);
        for _ in 0..degree {
            values.push(horner(&current, alpha));
            let top = current[degree - 1];
            current.rotate_right(1);
            current[0] = -top;
            current[degree / 2] -= middle::<F, M>() * top;
        }
        values
    }

    fn check<F: Field + core::fmt::Debug, M: TrinomialModulus>(b: &[F], alpha: F) {
        let mut fast = vec![F::zero(); b.len()];
        shifted_remainder_evaluations::<F, M>(b, alpha, &mut fast).unwrap();
        assert_eq!(fast, dense::<F, M>(b, alpha));
        assert_eq!(fast, stepwise::<F, M>(b, alpha));
    }

    fn differential<F: Field + core::fmt::Debug, M: TrinomialModulus>(seed: u64) {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut random =
            || F::from_u128(u128::from(rng.next_u64()) << 64 | u128::from(rng.next_u64()));
        for degree in [2, 4, 6, 162, 324, 648] {
            let b: Vec<F> = (0..degree).map(|_| random()).collect();
            check::<F, M>(&b, random());
            // Evaluation points where `Phi(alpha)` or the shift itself degenerates.
            check::<F, M>(&b, F::zero());
            check::<F, M>(&b, F::one());
            check::<F, M>(&b, -F::one());
            // Every monomial: the top coefficient fires after `D-1-s` shifts.
            let alpha = random();
            for s in [0, 1, degree / 2 - 1, degree / 2, degree - 1] {
                let mut monomial = vec![F::zero(); degree];
                monomial[s] = F::one();
                check::<F, M>(&monomial, alpha);
            }
            check::<F, M>(&vec![F::zero(); degree], alpha);
        }
    }

    #[test]
    fn recurrence_matches_dense_and_stepwise_definitions() {
        differential::<Prime128Offset275, MinusTrinomial>(0x275);
        differential::<Prime128Offset275, PlusTrinomial>(0x276);
        differential::<Prime31Offset19, MinusTrinomial>(0x19);
        differential::<Prime31Offset19, PlusTrinomial>(0x20);
    }

    #[test]
    fn malformed_lengths_are_rejected() {
        type F = Prime31Offset19;
        let zero = F::zero();
        let mut output = [zero; 4];
        assert_eq!(
            shifted_remainder_evaluations::<F, MinusTrinomial>(&[], zero, &mut []),
            Err(TrinomialError::InvalidDegree { degree: 0 })
        );
        assert_eq!(
            shifted_remainder_evaluations::<F, MinusTrinomial>(&[zero; 3], zero, &mut output[..3]),
            Err(TrinomialError::InvalidDegree { degree: 3 })
        );
        assert_eq!(
            shifted_remainder_evaluations::<F, PlusTrinomial>(&[zero; 2], zero, &mut output),
            Err(TrinomialError::CoefficientLength {
                expected: 2,
                actual: 4
            })
        );
    }
}
