//! Numeric policy types shared by estimator formulas and golden comparisons.

use crate::error::{EstimatorError, Result};

/// Probability in the closed interval `(0, 1]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Probability {
    value: f64,
}

impl Probability {
    /// Create a checked probability.
    ///
    /// # Errors
    ///
    /// Returns an error if `value` is not finite or is outside `(0, 1]`.
    pub fn new(value: f64) -> Result<Self> {
        if !value.is_finite() || value <= 0.0 || value > 1.0 {
            return Err(EstimatorError::InvalidConfig {
                field: "probability",
                reason: "probability must be finite and in (0, 1]".to_string(),
            });
        }
        Ok(Self { value })
    }

    /// Return the probability as `f64`.
    #[must_use]
    pub const fn get(self) -> f64 {
        self.value
    }
}

impl Default for Probability {
    fn default() -> Self {
        Self { value: 0.99 }
    }
}

/// Numeric backend requested by the caller.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum NumericBackend {
    /// Fast `f64` arithmetic.
    #[default]
    F64,
    /// Checked high-precision arithmetic, parameterized by precision bits.
    HighPrecision {
        /// Requested precision in bits.
        bits: u32,
    },
}

/// Numeric policy for estimator execution and Sage parity checks.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NumericConfig {
    /// Arithmetic backend.
    pub backend: NumericBackend,
    /// Absolute tolerance for Sage log-space parity checks.
    pub sage_abs_tolerance: f64,
    /// Relative tolerance for Sage log-space parity checks.
    pub sage_rel_tolerance: f64,
}

impl NumericConfig {
    /// Validate numeric settings.
    ///
    /// # Errors
    ///
    /// Returns an error if tolerances are negative or non-finite, or if a
    /// high-precision backend asks for too few bits to exceed `f64`.
    pub fn validate(&self) -> Result<()> {
        if !self.sage_abs_tolerance.is_finite() || self.sage_abs_tolerance < 0.0 {
            return Err(EstimatorError::InvalidConfig {
                field: "numeric.sage_abs_tolerance",
                reason: "absolute tolerance must be finite and nonnegative".to_string(),
            });
        }
        if !self.sage_rel_tolerance.is_finite() || self.sage_rel_tolerance < 0.0 {
            return Err(EstimatorError::InvalidConfig {
                field: "numeric.sage_rel_tolerance",
                reason: "relative tolerance must be finite and nonnegative".to_string(),
            });
        }
        if let NumericBackend::HighPrecision { bits } = self.backend {
            if bits <= 64 {
                return Err(EstimatorError::InvalidConfig {
                    field: "numeric.backend",
                    reason: "high-precision backend must request more than 64 bits".to_string(),
                });
            }
        }
        Ok(())
    }
}

impl Default for NumericConfig {
    fn default() -> Self {
        Self {
            backend: NumericBackend::F64,
            sage_abs_tolerance: 1e-6,
            sage_rel_tolerance: 1e-12,
        }
    }
}

/// Trust status for a golden cell.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GoldenTrust {
    /// The upstream reference cell is stable enough for exact parity tests.
    Trusted,
    /// The upstream reference cell is recorded but excluded from hard parity.
    Fragile {
        /// Reason the cell is excluded.
        reason: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probability_accepts_only_open_closed_unit_interval() {
        assert_eq!(Probability::new(1.0).unwrap().get(), 1.0);
        assert!(Probability::new(0.0).is_err());
        assert!(Probability::new(f64::NAN).is_err());
        assert!(Probability::new(1.01).is_err());
    }

    #[test]
    fn numeric_config_validates_tolerances_and_precision() {
        assert!(NumericConfig::default().validate().is_ok());
        assert!(NumericConfig {
            backend: NumericBackend::HighPrecision { bits: 64 },
            ..NumericConfig::default()
        }
        .validate()
        .is_err());
        assert!(NumericConfig {
            sage_abs_tolerance: -1.0,
            ..NumericConfig::default()
        }
        .validate()
        .is_err());
    }
}
