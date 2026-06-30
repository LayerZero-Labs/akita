//! SIS parameter types and validation.

use num_bigint::BigUint;
use num_traits::{One, Zero};

use crate::error::{EstimatorError, Result};

/// SIS norm family.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SisNorm {
    /// Euclidean `l2` norm.
    Euclidean,
    /// Infinity `l∞` norm.
    Infinity,
}

/// Length-bound representation accepted by the public estimator API.
#[derive(Clone, Debug, PartialEq)]
pub enum Bound {
    /// Exact integer bound.
    Integer(BigUint),
    /// Floating-point bound for compatibility paths.
    Float(f64),
    /// Exact rational bound.
    Rational {
        /// Positive numerator.
        numerator: BigUint,
        /// Positive denominator.
        denominator: BigUint,
    },
}

impl Bound {
    /// Create an integer bound from a `u64`.
    #[must_use]
    pub fn from_u64(value: u64) -> Self {
        Self::Integer(BigUint::from(value))
    }

    /// Create an integer bound from a `u128`.
    #[must_use]
    pub fn from_u128(value: u128) -> Self {
        Self::Integer(BigUint::from(value))
    }

    /// Validate that the bound is positive and numerically well formed.
    ///
    /// # Errors
    ///
    /// Returns an error if the bound is zero, negative, non-finite, or has a
    /// zero rational denominator.
    pub fn validate(&self) -> Result<()> {
        match self {
            Self::Integer(value) if value.is_zero() => Err(EstimatorError::InvalidParameter {
                field: "length_bound",
                reason: "integer bound must be positive".to_string(),
            }),
            Self::Integer(_) => Ok(()),
            Self::Float(value) if !value.is_finite() || *value <= 0.0 => {
                Err(EstimatorError::InvalidParameter {
                    field: "length_bound",
                    reason: "float bound must be finite and positive".to_string(),
                })
            }
            Self::Float(_) => Ok(()),
            Self::Rational {
                numerator,
                denominator,
            } if numerator.is_zero() || denominator.is_zero() => {
                Err(EstimatorError::InvalidParameter {
                    field: "length_bound",
                    reason: "rational numerator and denominator must be positive".to_string(),
                })
            }
            Self::Rational { .. } => Ok(()),
        }
    }
}

/// SIS lattice estimator parameters.
#[derive(Clone, Debug, PartialEq)]
pub struct SisParameters {
    /// Number of SIS equations.
    pub n: u32,
    /// Modulus.
    pub q: BigUint,
    /// Number of SIS columns, or `None` to let a later estimator path choose
    /// the lattice-estimator-compatible default.
    pub m: Option<u32>,
    /// Short-vector length bound.
    pub length_bound: Bound,
    /// Norm used to interpret `length_bound`.
    pub norm: SisNorm,
    /// Optional caller tag propagated to estimator output.
    pub tag: Option<String>,
}

impl SisParameters {
    /// Create checked SIS parameters.
    ///
    /// # Errors
    ///
    /// Returns an error when any parameter is malformed.
    pub fn try_new(
        n: u32,
        q: BigUint,
        m: Option<u32>,
        length_bound: Bound,
        norm: SisNorm,
    ) -> Result<Self> {
        let params = Self {
            n,
            q,
            m,
            length_bound,
            norm,
            tag: None,
        };
        params.validate()?;
        Ok(params)
    }

    /// Return a copy with a caller-visible tag.
    #[must_use]
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tag = Some(tag.into());
        self
    }

    /// Return a copy with a different column count.
    #[must_use]
    pub const fn with_m(mut self, m: Option<u32>) -> Self {
        self.m = m;
        self
    }

    /// Return a validated copy with selected fields replaced.
    ///
    /// # Errors
    ///
    /// Returns an error if the updated parameters are malformed.
    pub fn updated(&self, update: SisParameterUpdate) -> Result<Self> {
        let params = Self {
            n: update.n.unwrap_or(self.n),
            q: update.q.unwrap_or_else(|| self.q.clone()),
            m: update.m.unwrap_or(self.m),
            length_bound: update
                .length_bound
                .unwrap_or_else(|| self.length_bound.clone()),
            norm: update.norm.unwrap_or(self.norm),
            tag: update.tag.unwrap_or_else(|| self.tag.clone()),
        };
        params.validate()?;
        Ok(params)
    }

    /// Validate all public parameter invariants.
    ///
    /// # Errors
    ///
    /// Returns an error when a dimension is zero, `q <= 1`, or the length bound
    /// is malformed.
    pub fn validate(&self) -> Result<()> {
        if self.n == 0 {
            return Err(EstimatorError::InvalidParameter {
                field: "n",
                reason: "n must be positive".to_string(),
            });
        }
        if self.q <= BigUint::one() {
            return Err(EstimatorError::InvalidParameter {
                field: "q",
                reason: "q must be greater than 1".to_string(),
            });
        }
        if self.m == Some(0) {
            return Err(EstimatorError::InvalidParameter {
                field: "m",
                reason: "m must be positive when present".to_string(),
            });
        }
        self.length_bound.validate()
    }
}

/// Field replacements for [`SisParameters::updated`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SisParameterUpdate {
    /// Replacement `n`.
    pub n: Option<u32>,
    /// Replacement `q`.
    pub q: Option<BigUint>,
    /// Replacement `m`; `Some(None)` clears `m`.
    pub m: Option<Option<u32>>,
    /// Replacement length bound.
    pub length_bound: Option<Bound>,
    /// Replacement norm.
    pub norm: Option<SisNorm>,
    /// Replacement tag; `Some(None)` clears the tag.
    pub tag: Option<Option<String>>,
}

/// Representative modulus for Akita q32 tables: `2^32 - 99`.
#[must_use]
pub fn akita_q32() -> BigUint {
    BigUint::from(4_294_967_197u64)
}

/// Representative modulus for Akita q64 tables: `2^64 - 59`.
#[must_use]
pub fn akita_q64() -> BigUint {
    (BigUint::one() << 64usize) - BigUint::from(59u32)
}

/// Representative modulus for Akita q128 tables:
/// `2^128 - (2^32 - 22537)`.
#[must_use]
pub fn akita_q128() -> BigUint {
    (BigUint::one() << 128usize) - ((BigUint::one() << 32usize) - BigUint::from(22_537u32))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructors_validate_core_fields() {
        assert!(SisParameters::try_new(
            1,
            akita_q32(),
            Some(2),
            Bound::from_u64(1),
            SisNorm::Infinity
        )
        .is_ok());
        assert!(matches!(
            SisParameters::try_new(
                0,
                akita_q32(),
                Some(2),
                Bound::from_u64(1),
                SisNorm::Infinity
            ),
            Err(EstimatorError::InvalidParameter { field: "n", .. })
        ));
        assert!(matches!(
            SisParameters::try_new(
                1,
                BigUint::one(),
                Some(2),
                Bound::from_u64(1),
                SisNorm::Infinity
            ),
            Err(EstimatorError::InvalidParameter { field: "q", .. })
        ));
        assert!(matches!(
            SisParameters::try_new(
                1,
                akita_q32(),
                Some(0),
                Bound::from_u64(1),
                SisNorm::Infinity
            ),
            Err(EstimatorError::InvalidParameter { field: "m", .. })
        ));
    }

    #[test]
    fn bounds_reject_zero_and_nonfinite_values() {
        assert!(Bound::from_u64(1).validate().is_ok());
        assert!(Bound::from_u64(0).validate().is_err());
        assert!(Bound::Float(2.5).validate().is_ok());
        assert!(Bound::Float(f64::INFINITY).validate().is_err());
        assert!(Bound::Rational {
            numerator: BigUint::from(1u32),
            denominator: BigUint::from(2u32),
        }
        .validate()
        .is_ok());
        assert!(Bound::Rational {
            numerator: BigUint::from(1u32),
            denominator: BigUint::zero(),
        }
        .validate()
        .is_err());
    }

    #[test]
    fn updated_replaces_selected_fields_and_validates_result() {
        let params = SisParameters::try_new(
            32,
            akita_q32(),
            Some(64),
            Bound::from_u64(15),
            SisNorm::Infinity,
        )
        .unwrap()
        .with_tag("base");
        let updated = params
            .updated(SisParameterUpdate {
                m: Some(None),
                length_bound: Some(Bound::from_u64(255)),
                tag: Some(Some("updated".to_string())),
                ..SisParameterUpdate::default()
            })
            .unwrap();
        assert_eq!(updated.n, 32);
        assert_eq!(updated.m, None);
        assert_eq!(updated.tag.as_deref(), Some("updated"));
        assert_eq!(updated.length_bound, Bound::from_u64(255));

        assert!(params
            .updated(SisParameterUpdate {
                n: Some(0),
                ..SisParameterUpdate::default()
            })
            .is_err());
    }

    #[test]
    fn representative_moduli_match_golden_families() {
        assert_eq!(akita_q32(), BigUint::from(4_294_967_197u64));
        assert_eq!(akita_q64(), BigUint::from(u64::MAX) - BigUint::from(58u32));
        assert_eq!(
            akita_q128().to_string(),
            "340282366920938463463374607427473266697"
        );
    }

    #[test]
    fn debug_output_is_serialization_free_and_informative() {
        let params = SisParameters::try_new(
            32,
            akita_q32(),
            Some(64),
            Bound::from_u64(15),
            SisNorm::Infinity,
        )
        .unwrap()
        .with_tag("smoke");
        let debug = format!("{params:?}");
        assert!(debug.contains("SisParameters"));
        assert!(debug.contains("Infinity"));
        assert!(debug.contains("smoke"));
    }
}
