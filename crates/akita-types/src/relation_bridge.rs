//! Checked polynomial identities and padded coefficient layouts for relation bridges.
//!
//! Existing [`crate::RelationRowGeometry`] remains the power-of-two negacyclic
//! fast path. These descriptors let a caller name a trinomial identity and its
//! padded table layout without reinterpreting that existing geometry.

use core::ops::Range;

use akita_algebra::fft::field_pow;
use akita_error::{checked, AkitaError};
use jolt_field::Field;

/// Sign of the middle term in `X^D +/- X^(D/2) + 1`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TrinomialSign {
    Plus,
    Minus,
}

/// Polynomial family named by a relation identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RelationPolynomialKind {
    /// `X^D + 1`, retained by the existing power-of-two relation path.
    Negacyclic,
    /// `X^D +/- X^(D/2) + 1`.
    Trinomial(TrinomialSign),
}

/// Checked monic polynomial used by a relation row.
///
/// This value is the complete polynomial identity: equality compares both the
/// family/sign and the natural degree.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RelationPolynomial {
    kind: RelationPolynomialKind,
    degree: usize,
}

impl RelationPolynomial {
    /// Construct the current native relation polynomial `X^D + 1`.
    ///
    /// Negacyclic relation rows remain restricted to nonzero power-of-two
    /// degrees so this constructor cannot widen existing dispatch geometry.
    pub fn negacyclic(degree: usize) -> Result<Self, AkitaError> {
        if !degree.is_power_of_two() {
            return Err(AkitaError::InvalidSetup(
                "negacyclic relation degree must be a nonzero power of two".into(),
            ));
        }
        Ok(Self {
            kind: RelationPolynomialKind::Negacyclic,
            degree,
        })
    }

    /// Construct `X^D + X^(D/2) + 1`.
    pub fn plus_trinomial(degree: usize) -> Result<Self, AkitaError> {
        Self::trinomial(degree, TrinomialSign::Plus)
    }

    /// Construct `X^D - X^(D/2) + 1`.
    pub fn minus_trinomial(degree: usize) -> Result<Self, AkitaError> {
        Self::trinomial(degree, TrinomialSign::Minus)
    }

    fn trinomial(degree: usize, sign: TrinomialSign) -> Result<Self, AkitaError> {
        if degree == 0 || !degree.is_multiple_of(2) {
            return Err(AkitaError::InvalidSetup(
                "trinomial relation degree must be positive and even".into(),
            ));
        }
        Ok(Self {
            kind: RelationPolynomialKind::Trinomial(sign),
            degree,
        })
    }

    #[must_use]
    pub const fn kind(self) -> RelationPolynomialKind {
        self.kind
    }

    #[must_use]
    pub const fn degree(self) -> usize {
        self.degree
    }

    /// Maximum coefficient count of a product of two degree-`D` elements.
    pub fn product_coefficient_len(self) -> Result<usize, AkitaError> {
        self.degree
            .checked_mul(2)
            .and_then(|value| value.checked_sub(1))
            .ok_or_else(|| AkitaError::InvalidSetup("relation product length overflow".into()))
    }

    /// Canonical coefficient count of the monic quotient for such a product.
    pub fn quotient_coefficient_len(self) -> Result<usize, AkitaError> {
        self.degree
            .checked_sub(1)
            .ok_or_else(|| AkitaError::InvalidSetup("relation quotient length underflow".into()))
    }

    /// Validate a natural, unpadded coefficient slice length for one role.
    pub fn validate_coefficient_len(
        self,
        role: RelationCoefficientRole,
        actual: usize,
    ) -> Result<(), AkitaError> {
        let expected = role.natural_len(self)?;
        if actual != expected {
            return Err(AkitaError::InvalidSize { expected, actual });
        }
        Ok(())
    }

    /// Evaluate this monic polynomial at `alpha`.
    pub fn evaluate_modulus_at<F: Field>(self, alpha: F) -> Result<F, AkitaError> {
        let degree = u64::try_from(self.degree)
            .map_err(|_| AkitaError::InvalidSetup("relation degree exceeds u64".into()))?;
        let leading = field_pow(alpha, degree);
        let value = match self.kind {
            RelationPolynomialKind::Negacyclic => leading + F::one(),
            RelationPolynomialKind::Trinomial(sign) => {
                let middle = field_pow(alpha, degree / 2);
                match sign {
                    TrinomialSign::Plus => leading + middle + F::one(),
                    TrinomialSign::Minus => leading - middle + F::one(),
                }
            }
        };
        Ok(value)
    }

    /// Evaluate from cached powers `[1, alpha, ..., alpha^(D-1)]`.
    ///
    /// Exact length validation prevents a padded table from changing the
    /// polynomial identity. The leading power is formed as
    /// `powers[D - 1] * alpha`.
    pub fn evaluate_modulus_with_powers<F: Field>(
        self,
        alpha: F,
        powers: &[F],
    ) -> Result<F, AkitaError> {
        if powers.len() != self.degree {
            return Err(AkitaError::InvalidSize {
                expected: self.degree,
                actual: powers.len(),
            });
        }
        let leading = powers
            .get(self.degree - 1)
            .copied()
            .ok_or(AkitaError::InvalidProof)?
            * alpha;
        let value = match self.kind {
            RelationPolynomialKind::Negacyclic => leading + F::one(),
            RelationPolynomialKind::Trinomial(sign) => {
                let middle = powers
                    .get(self.degree / 2)
                    .copied()
                    .ok_or(AkitaError::InvalidProof)?;
                match sign {
                    TrinomialSign::Plus => leading + middle + F::one(),
                    TrinomialSign::Minus => leading - middle + F::one(),
                }
            }
        };
        Ok(value)
    }
}

/// Natural coefficient role represented by a padded table segment.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RelationCoefficientRole {
    /// One canonical quotient-ring element, with `D` coefficients.
    Element,
    /// The quotient of a product identity, with exactly `D - 1` coefficients.
    Quotient,
}

impl RelationCoefficientRole {
    fn natural_len(self, polynomial: RelationPolynomial) -> Result<usize, AkitaError> {
        match self {
            Self::Element => Ok(polynomial.degree()),
            Self::Quotient => polynomial.quotient_coefficient_len(),
        }
    }
}

/// Checked natural and padded coefficient geometry for one polynomial role.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RelationCoefficientLayout {
    polynomial: RelationPolynomial,
    role: RelationCoefficientRole,
    natural_len: usize,
    padded_len: usize,
}

impl RelationCoefficientLayout {
    /// Construct an explicitly padded coefficient layout.
    pub fn new(
        polynomial: RelationPolynomial,
        role: RelationCoefficientRole,
        padded_len: usize,
    ) -> Result<Self, AkitaError> {
        let natural_len = role.natural_len(polynomial)?;
        if !padded_len.is_power_of_two() || padded_len < natural_len {
            return Err(AkitaError::InvalidSetup(
                "relation coefficient padding must be a power of two covering the natural length"
                    .into(),
            ));
        }
        Ok(Self {
            polynomial,
            role,
            natural_len,
            padded_len,
        })
    }

    #[must_use]
    pub const fn polynomial(self) -> RelationPolynomial {
        self.polynomial
    }

    #[must_use]
    pub const fn role(self) -> RelationCoefficientRole {
        self.role
    }

    #[must_use]
    pub const fn natural_len(self) -> usize {
        self.natural_len
    }

    #[must_use]
    pub const fn padded_len(self) -> usize {
        self.padded_len
    }

    /// Address one natural coefficient relative to a table base.
    pub fn coefficient_address(self, base: usize, coefficient: usize) -> Result<usize, AkitaError> {
        if coefficient >= self.natural_len {
            return Err(AkitaError::InvalidInput(
                "relation coefficient index lies in padding".into(),
            ));
        }
        base.checked_add(coefficient)
            .ok_or_else(|| AkitaError::InvalidSetup("relation coefficient address overflow".into()))
    }

    pub fn natural_range(self, base: usize) -> Result<Range<usize>, AkitaError> {
        checked::range(base, self.natural_len)
            .ok_or_else(|| AkitaError::InvalidSetup("relation natural range overflow".into()))
    }

    pub fn padded_range(self, base: usize) -> Result<Range<usize>, AkitaError> {
        checked::range(base, self.padded_len)
            .ok_or_else(|| AkitaError::InvalidSetup("relation padded range overflow".into()))
    }

    pub fn tail_range(self, base: usize) -> Result<Range<usize>, AkitaError> {
        let start = base
            .checked_add(self.natural_len)
            .ok_or_else(|| AkitaError::InvalidSetup("relation tail offset overflow".into()))?;
        checked::range(start, self.padded_len - self.natural_len)
            .ok_or_else(|| AkitaError::InvalidSetup("relation tail range overflow".into()))
    }

    /// Validate one complete padded segment and require a zero tail.
    pub fn validate_padded_coefficients<F: Field>(
        self,
        coefficients: &[F],
    ) -> Result<(), AkitaError> {
        if coefficients.len() != self.padded_len {
            return Err(AkitaError::InvalidSize {
                expected: self.padded_len,
                actual: coefficients.len(),
            });
        }
        if coefficients[self.natural_len..]
            .iter()
            .any(|coefficient| !coefficient.is_zero())
        {
            return Err(AkitaError::InvalidInput(
                "relation coefficient padding must be zero".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_algebra::ring::scalar_powers;
    use jolt_field::{One, Prime128OffsetA7F7, Ring, Zero};

    type F = Prime128OffsetA7F7;

    #[test]
    fn polynomial_evaluation_matches_cached_powers_for_all_families() {
        let alpha = F::from_u64(17);
        for polynomial in [
            RelationPolynomial::negacyclic(256).unwrap(),
            RelationPolynomial::plus_trinomial(162).unwrap(),
            RelationPolynomial::minus_trinomial(648).unwrap(),
        ] {
            let powers = scalar_powers(alpha, polynomial.degree());
            assert_eq!(
                polynomial.evaluate_modulus_at(alpha).unwrap(),
                polynomial
                    .evaluate_modulus_with_powers(alpha, &powers)
                    .unwrap()
            );
            assert!(polynomial
                .evaluate_modulus_with_powers(alpha, &powers[..powers.len() - 1])
                .is_err());
        }
    }

    #[test]
    fn polynomial_constructors_preserve_fast_path_boundary() {
        assert!(RelationPolynomial::negacyclic(162).is_err());
        assert!(RelationPolynomial::negacyclic(0).is_err());
        assert!(RelationPolynomial::plus_trinomial(0).is_err());
        assert!(RelationPolynomial::minus_trinomial(163).is_err());
    }

    #[test]
    fn degree_648_layout_keeps_natural_data_and_padding_distinct() {
        let polynomial = RelationPolynomial::minus_trinomial(648).unwrap();
        let element =
            RelationCoefficientLayout::new(polynomial, RelationCoefficientRole::Element, 1024)
                .unwrap();
        let quotient =
            RelationCoefficientLayout::new(polynomial, RelationCoefficientRole::Quotient, 1024)
                .unwrap();
        assert_eq!(element.natural_len(), 648);
        assert_eq!(quotient.natural_len(), 647);
        assert_eq!(element.coefficient_address(31, 647).unwrap(), 678);
        assert!(element.coefficient_address(31, 648).is_err());
        assert_eq!(element.tail_range(31).unwrap(), 679..1055);

        let mut padded = vec![F::zero(); 1024];
        padded[647] = F::from_u64(9);
        element.validate_padded_coefficients(&padded).unwrap();
        padded[648] = F::one();
        assert!(element.validate_padded_coefficients(&padded).is_err());
        assert!(
            RelationCoefficientLayout::new(polynomial, RelationCoefficientRole::Element, 648)
                .is_err()
        );
    }
}
