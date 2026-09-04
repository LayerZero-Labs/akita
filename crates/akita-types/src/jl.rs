//! Kernel-checked JL parameter records used by protocol planning.
//!
//! These constants pin the exact theorem outputs from CertifiedJL. They do not
//! choose a protocol schedule: later planner work must compose per-layer and
//! per-stem failure budgets explicitly.

/// CertifiedJL revision from which the bound records were transcribed.
pub const CERTIFIED_JL_SOURCE_REVISION: &str = "8ac6eda09c6f8b6fe38770f78489af610eb05023";

/// Identifier for the independent row distribution used by a JL matrix.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum JlRowLawId {
    /// Entries are independent with probabilities `P(0)=1/2` and
    /// `P(-1)=P(+1)=1/4`.
    BalancedTernary,
}

/// Nonnegative rational constant represented without floating-point rounding.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RationalBound {
    /// Rational numerator.
    numerator: u32,
    /// Positive rational denominator.
    denominator: u32,
}

impl RationalBound {
    /// Rational numerator.
    #[must_use]
    pub const fn numerator(self) -> u32 {
        self.numerator
    }

    /// Positive rational denominator.
    #[must_use]
    pub const fn denominator(self) -> u32 {
        self.denominator
    }
}

/// Certified lower- and upper-tail constants for squared Euclidean energy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CertifiedJlL2Bounds {
    /// `P(||Jw||_q^2 > upper * ||w||_2^2) < 2^-upper_failure_bits`.
    upper_output_energy_per_input_energy: RationalBound,
    /// Negative binary logarithm of the upper-tail failure probability.
    upper_failure_bits: u16,
    /// Under the threshold theorem's hypotheses,
    /// `P(||Jw||_q^2 < lower * b^2) < 2^-lower_failure_bits` whenever
    /// `b^2 <= ||w||_2^2`.
    lower_output_energy_per_input_threshold_sq: RationalBound,
    /// Negative binary logarithm of the lower-tail failure probability.
    lower_failure_bits: u16,
    /// Required no-wrap hypothesis `lower_modulus_margin * b <= q`.
    lower_modulus_margin: u32,
}

impl CertifiedJlL2Bounds {
    /// Upper output-energy multiplier.
    #[must_use]
    pub const fn upper_output_energy_per_input_energy(self) -> RationalBound {
        self.upper_output_energy_per_input_energy
    }

    /// Negative binary logarithm of the upper-tail failure probability.
    #[must_use]
    pub const fn upper_failure_bits(self) -> u16 {
        self.upper_failure_bits
    }

    /// Lower output-energy multiplier relative to the squared input threshold.
    #[must_use]
    pub const fn lower_output_energy_per_input_threshold_sq(self) -> RationalBound {
        self.lower_output_energy_per_input_threshold_sq
    }

    /// Negative binary logarithm of the lower-tail failure probability.
    #[must_use]
    pub const fn lower_failure_bits(self) -> u16 {
        self.lower_failure_bits
    }

    /// Multiplier in the required no-wrap condition against the modulus.
    #[must_use]
    pub const fn lower_modulus_margin(self) -> u32 {
        self.lower_modulus_margin
    }
}

/// Certified lower- and upper-tail constants for maximum-coordinate magnitude.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CertifiedJlLinfBounds {
    /// `P(||Jw||_{q,infinity} > upper * ||w||_2) < 2^-upper_failure_bits`.
    upper_output_coordinate_per_input_norm: RationalBound,
    /// Negative binary logarithm of the upper-tail failure probability.
    upper_failure_bits: u16,
    /// Under the threshold theorem's hypotheses,
    /// `P(||Jw||_{q,infinity} <= lower * b) < 2^-lower_failure_bits` whenever
    /// `b^2 <= ||w||_2^2`.
    lower_output_coordinate_per_input_threshold: RationalBound,
    /// Negative binary logarithm of the lower-tail failure probability.
    lower_failure_bits: u16,
    /// Required no-wrap hypothesis `lower_modulus_margin * b <= q`.
    lower_modulus_margin: u32,
}

impl CertifiedJlLinfBounds {
    /// Upper maximum-coordinate multiplier relative to the input norm.
    #[must_use]
    pub const fn upper_output_coordinate_per_input_norm(self) -> RationalBound {
        self.upper_output_coordinate_per_input_norm
    }

    /// Negative binary logarithm of the upper-tail failure probability.
    #[must_use]
    pub const fn upper_failure_bits(self) -> u16 {
        self.upper_failure_bits
    }

    /// Lower maximum-coordinate multiplier relative to the input threshold.
    #[must_use]
    pub const fn lower_output_coordinate_per_input_threshold(self) -> RationalBound {
        self.lower_output_coordinate_per_input_threshold
    }

    /// Negative binary logarithm of the lower-tail failure probability.
    #[must_use]
    pub const fn lower_failure_bits(self) -> u16 {
        self.lower_failure_bits
    }

    /// Multiplier in the required no-wrap condition against the modulus.
    #[must_use]
    pub const fn lower_modulus_margin(self) -> u32 {
        self.lower_modulus_margin
    }
}

/// One row law and output dimension with both available tail routes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CertifiedJlBounds {
    /// Independent matrix-entry law.
    row_law: JlRowLawId,
    /// Number of projected rows.
    rows: u16,
    /// Squared-Euclidean tail results.
    l2: CertifiedJlL2Bounds,
    /// Maximum-coordinate tail results.
    linf: CertifiedJlLinfBounds,
}

impl CertifiedJlBounds {
    /// Independent matrix-entry law.
    #[must_use]
    pub const fn row_law(self) -> JlRowLawId {
        self.row_law
    }

    /// Number of projected rows.
    #[must_use]
    pub const fn rows(self) -> u16 {
        self.rows
    }

    /// Squared-Euclidean tail results.
    #[must_use]
    pub const fn l2(self) -> CertifiedJlL2Bounds {
        self.l2
    }

    /// Maximum-coordinate tail results.
    #[must_use]
    pub const fn linf(self) -> CertifiedJlLinfBounds {
        self.linf
    }
}

/// Pinned 256-row CertifiedJL foundation record for 128-bit schedule work.
///
/// The L2 theorems are
/// `CertifiedJL.Results.Rows256Bits128.ternaryL2Upper338` and
/// `CertifiedJL.Results.Rows256Bits128.ternaryL2ThresholdLower29`. The Linf
/// theorems are
/// `CertifiedJL.Results.Rows256Bits128.ternaryLInfUpper39Over4` and
/// `CertifiedJL.Results.Rows256Bits130.ternaryLInfThresholdLower21Over50`.
/// The latter gives 130 lower-tail bits, which safely exceeds the 128-bit
/// target without changing the 256-row geometry.
pub const BALANCED_TERNARY_256_ROWS_128_BITS: CertifiedJlBounds = CertifiedJlBounds {
    row_law: JlRowLawId::BalancedTernary,
    rows: 256,
    l2: CertifiedJlL2Bounds {
        upper_output_energy_per_input_energy: RationalBound {
            numerator: 338,
            denominator: 1,
        },
        upper_failure_bits: 128,
        lower_output_energy_per_input_threshold_sq: RationalBound {
            numerator: 29,
            denominator: 1,
        },
        lower_failure_bits: 128,
        lower_modulus_margin: 3,
    },
    linf: CertifiedJlLinfBounds {
        upper_output_coordinate_per_input_norm: RationalBound {
            numerator: 39,
            denominator: 4,
        },
        upper_failure_bits: 128,
        lower_output_coordinate_per_input_threshold: RationalBound {
            numerator: 21,
            denominator: 50,
        },
        lower_failure_bits: 130,
        lower_modulus_margin: 2,
    },
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn foundation_record_pins_the_certifiedjl_catalog() {
        let bounds = BALANCED_TERNARY_256_ROWS_128_BITS;
        assert_eq!(bounds.row_law(), JlRowLawId::BalancedTernary);
        assert_eq!(bounds.rows(), 256);
        assert_eq!(
            bounds.l2().upper_output_energy_per_input_energy(),
            RationalBound {
                numerator: 338,
                denominator: 1
            }
        );
        assert_eq!(
            bounds.l2().lower_output_energy_per_input_threshold_sq(),
            RationalBound {
                numerator: 29,
                denominator: 1
            }
        );
        assert_eq!(bounds.l2().lower_modulus_margin(), 3);
        assert_eq!(
            bounds.linf().upper_output_coordinate_per_input_norm(),
            RationalBound {
                numerator: 39,
                denominator: 4
            }
        );
        assert_eq!(
            bounds.linf().lower_output_coordinate_per_input_threshold(),
            RationalBound {
                numerator: 21,
                denominator: 50
            }
        );
        assert_eq!(bounds.linf().lower_modulus_margin(), 2);
        assert!(bounds.linf().lower_failure_bits() >= 128);
    }
}
