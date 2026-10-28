//! Checked integer products and remainders in signed trinomial rings.
//!
//! Coefficients are lowest degree first. Arithmetic is over the integers;
//! this module never reduces modulo a coefficient prime. In particular,
//! `D = 648` and [`super::MinusTrinomial`] give
//! `Psi = Y^648 - Y^324 + 1`.

use akita_error::{checked, AkitaError};

use super::TrinomialModulus;

/// Reduce an integer product modulo `X^D + middle * X^(D/2) + 1`.
///
/// Accepts at most `2D - 1` coefficients, including the empty polynomial.
/// Rejects zero or odd `D`, excessive input length, and any intermediate
/// coefficient that overflows `i128`. For the two sealed modulus shapes,
/// each monomial below degree `2D - 1` reduces to at most two monomials.
pub fn signed_trinomial_remainder<const D: usize, M: TrinomialModulus>(
    coefficients: &[i128],
) -> Result<[i128; D], AkitaError> {
    if D == 0 || !D.is_multiple_of(2) {
        return Err(AkitaError::InvalidInput(
            "signed trinomial degree must be positive and even".into(),
        ));
    }
    let maximum = checked::product([2, D])
        .and_then(|value| value.checked_sub(1))
        .ok_or_else(|| AkitaError::InvalidInput("signed trinomial degree overflow".into()))?;
    if coefficients.len() > maximum {
        return Err(AkitaError::InvalidInput(
            "signed trinomial polynomial exceeds product degree".into(),
        ));
    }
    let half = D / 2;
    let mut remainder = [0i128; D];
    for (exponent, &coefficient) in coefficients.iter().enumerate() {
        // For h = D/2 and middle = s in {-1, 1}:
        // X^(D+j) = -X^j - s X^(h+j) for j < h;
        // X^(3h+j) = s X^j for j < h-1.
        let (position, value) = if exponent < D {
            (exponent, coefficient)
        } else if exponent - D < half {
            let low = remainder
                .get_mut(exponent - D)
                .ok_or_else(|| AkitaError::InvalidInput("signed reduction index".into()))?;
            *low = low.checked_sub(coefficient).ok_or_else(|| {
                AkitaError::InvalidInput("signed trinomial coefficient overflow".into())
            })?;
            let high_value = if M::MIDDLE_COEFFICIENT < 0 {
                coefficient
            } else {
                coefficient.checked_neg().ok_or_else(|| {
                    AkitaError::InvalidInput("signed trinomial coefficient overflow".into())
                })?
            };
            (exponent - half, high_value)
        } else {
            let value = if M::MIDDLE_COEFFICIENT < 0 {
                coefficient.checked_neg().ok_or_else(|| {
                    AkitaError::InvalidInput("signed trinomial coefficient overflow".into())
                })?
            } else {
                coefficient
            };
            (exponent - D - half, value)
        };
        let target = remainder
            .get_mut(position)
            .ok_or_else(|| AkitaError::InvalidInput("signed reduction index".into()))?;
        *target = target.checked_add(value).ok_or_else(|| {
            AkitaError::InvalidInput("signed trinomial coefficient overflow".into())
        })?;
    }
    Ok(remainder)
}

/// Multiply two integer coefficient vectors and reduce in the trinomial ring.
///
/// Both inputs must have exactly `D` coefficients. Signed centered matrix
/// coefficients, balanced `i8` digits, and signed integer sources can be
/// supplied directly. All products, convolution sums, and reduction sums use
/// checked `i128` arithmetic; overflow or invalid shape returns [`AkitaError`].
pub fn signed_trinomial_product<const D: usize, M: TrinomialModulus>(
    lhs: &[impl Copy + Into<i128>],
    rhs: &[impl Copy + Into<i128>],
) -> Result<[i128; D], AkitaError> {
    for actual in [lhs.len(), rhs.len()] {
        if actual != D {
            return Err(AkitaError::InvalidSize {
                expected: D,
                actual,
            });
        }
    }
    if D == 0 || !D.is_multiple_of(2) {
        return Err(AkitaError::InvalidInput(
            "signed trinomial degree must be positive and even".into(),
        ));
    }
    let length = checked::product([2, D])
        .and_then(|value| value.checked_sub(1))
        .ok_or_else(|| AkitaError::InvalidInput("signed trinomial degree overflow".into()))?;
    let mut product = vec![0i128; length];
    for (left_index, &left) in lhs.iter().enumerate() {
        for (right_index, &right) in rhs.iter().enumerate() {
            let coefficient = left.into().checked_mul(right.into()).ok_or_else(|| {
                AkitaError::InvalidInput("signed trinomial coefficient overflow".into())
            })?;
            let target = product.get_mut(left_index + right_index).ok_or_else(|| {
                AkitaError::InvalidInput("signed trinomial convolution index".into())
            })?;
            *target = target.checked_add(coefficient).ok_or_else(|| {
                AkitaError::InvalidInput("signed trinomial coefficient overflow".into())
            })?;
        }
    }
    signed_trinomial_remainder::<D, M>(&product)
}

#[cfg(test)]
mod tests;
