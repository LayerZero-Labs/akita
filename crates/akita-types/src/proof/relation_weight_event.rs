use std::ops::Range;

use akita_field::{AkitaError, FieldCore};

/// Whether one relation event belongs to the protocol constraint or setup matrix.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RelationWeightContribution {
    /// Consistency, A-row, opening, and quotient-denominator arithmetic.
    Constraint,
    /// D/B/A setup-matrix arithmetic replaceable by one offloaded setup claim.
    SetupMatrix,
}

/// One aligned consecutive-alpha contribution to the flat relation weight table.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RelationWeightEvent<E: FieldCore> {
    physical_coefficients: Range<usize>,
    alpha_exponent_start: usize,
    scalar: E,
    contribution: RelationWeightContribution,
}

impl<E: FieldCore> RelationWeightEvent<E> {
    /// Construct one nonempty power-of-two event interval.
    pub fn new(
        physical_coefficients: Range<usize>,
        alpha_exponent_start: usize,
        scalar: E,
        contribution: RelationWeightContribution,
    ) -> Result<Self, AkitaError> {
        let coefficient_count = physical_coefficients.len();
        if coefficient_count == 0 || !coefficient_count.is_power_of_two() {
            return Err(AkitaError::InvalidSetup(
                "relation event length must be a nonzero power of two".into(),
            ));
        }
        Ok(Self {
            physical_coefficients,
            alpha_exponent_start,
            scalar,
            contribution,
        })
    }

    /// Flat physical coefficient interval receiving this contribution.
    #[must_use]
    pub fn physical_coefficients(&self) -> Range<usize> {
        self.physical_coefficients.clone()
    }

    /// Alpha exponent attached to the first coefficient in the interval.
    #[must_use]
    pub const fn alpha_exponent_start(&self) -> usize {
        self.alpha_exponent_start
    }

    /// Scalar multiplying the consecutive alpha powers.
    #[must_use]
    pub const fn scalar(&self) -> E {
        self.scalar
    }

    /// Whether this is constraint or setup-matrix arithmetic.
    #[must_use]
    pub const fn contribution(&self) -> RelationWeightContribution {
        self.contribution
    }
}
