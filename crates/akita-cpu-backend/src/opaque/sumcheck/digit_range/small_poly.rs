use akita_error::AkitaError;
use jolt_field::Field;

/// A range-leaf polynomial whose degree is structurally at most four.
pub(super) enum SmallPoly<E: Field> {
    Zero,
    Constant([E; 1]),
    Linear([E; 2]),
    Quadratic([E; 3]),
    Cubic([E; 4]),
    Quartic([E; 5]),
}

impl<E: Field> SmallPoly<E> {
    pub(super) fn new(coefficients: &[E]) -> Result<Self, AkitaError> {
        match coefficients {
            [] => Ok(Self::Zero),
            [c0] => Ok(Self::Constant([*c0])),
            [c0, c1] => Ok(Self::Linear([*c0, *c1])),
            [c0, c1, c2] => Ok(Self::Quadratic([*c0, *c1, *c2])),
            [c0, c1, c2, c3] => Ok(Self::Cubic([*c0, *c1, *c2, *c3])),
            [c0, c1, c2, c3, c4] => Ok(Self::Quartic([*c0, *c1, *c2, *c3, *c4])),
            _ => Err(AkitaError::Internal(format!(
                "range-leaf polynomial coefficient count: expected 5, actual {}",
                coefficients.len(),
            ))),
        }
    }

    pub(super) fn coefficients(&self) -> &[E] {
        match self {
            Self::Zero => &[],
            Self::Constant(c) => c,
            Self::Linear(c) => c,
            Self::Quadratic(c) => c,
            Self::Cubic(c) => c,
            Self::Quartic(c) => c,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::{Prime128Offset275 as F, Zero};

    #[test]
    fn rejects_coefficients_beyond_quartic() {
        assert!(matches!(
            SmallPoly::new(&[F::zero(); 6]),
            Err(AkitaError::Internal(message))
                if message == "range-leaf polynomial coefficient count: expected 5, actual 6"
        ));
    }
}
