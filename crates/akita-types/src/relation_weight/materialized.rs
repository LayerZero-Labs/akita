//! Prover-side materialized relation-weight evaluations.

use akita_field::{AkitaError, FieldCore};

/// Errors specific to [`RelationWeightPolynomial`] validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RelationWeightPolynomialError {
    LengthMismatch { expected: usize, actual: usize },
}

impl std::fmt::Display for RelationWeightPolynomialError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LengthMismatch { expected, actual } => {
                write!(
                    f,
                    "relation weight length mismatch (expected {expected}, got {actual})"
                )
            }
        }
    }
}

impl std::error::Error for RelationWeightPolynomialError {}

fn relation_weight_error(err: RelationWeightPolynomialError) -> AkitaError {
    match err {
        RelationWeightPolynomialError::LengthMismatch { expected, actual } => {
            AkitaError::InvalidSize { expected, actual }
        }
    }
}

/// Materialized evaluations of the relation-weight multilinear polynomial.
///
/// Evaluations are stored only for the live flat next-witness coefficient range.
/// The surrounding Boolean hypercube padding is part of the protocol's
/// zero-extension convention and is not materialized here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelationWeightPolynomial<E: FieldCore> {
    evals: Vec<E>,
}

impl<E: FieldCore> RelationWeightPolynomial<E> {
    /// Construct from materialized evaluations over the live coefficient range.
    pub fn from_live_evals(evals: Vec<E>, live_len: usize) -> Result<Self, AkitaError> {
        if live_len == 0 || evals.len() != live_len {
            return Err(relation_weight_error(
                RelationWeightPolynomialError::LengthMismatch {
                    expected: live_len,
                    actual: evals.len(),
                },
            ));
        }
        Ok(Self { evals })
    }

    #[must_use]
    pub fn evals(&self) -> &[E] {
        &self.evals
    }

    #[must_use]
    pub fn evals_mut(&mut self) -> &mut [E] {
        &mut self.evals
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.evals.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.evals.is_empty()
    }

    /// Pair of relation-weight evaluations for one sumcheck fold step.
    pub fn pair_flat(&self, idx0: usize, idx1: usize) -> (E, E) {
        let p0 = self.evals.get(idx0).copied().unwrap_or_else(E::zero);
        let p1 = self.evals.get(idx1).copied().unwrap_or_else(E::zero);
        (p0, p1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_field::{FpExt2, NegOneNr, Prime128Offset275};

    type F = Prime128Offset275;
    type E = FpExt2<F, NegOneNr>;

    #[test]
    fn rejects_length_mismatch() {
        let evals = vec![E::zero(); 3];
        let err =
            RelationWeightPolynomial::<E>::from_live_evals(evals, 4).expect_err("length mismatch");
        assert!(matches!(
            err,
            AkitaError::InvalidSize {
                expected: 4,
                actual: 3
            }
        ));
    }

    #[test]
    fn pair_flat_zero_extends_out_of_range_reads() {
        let relation_weight =
            RelationWeightPolynomial::from_live_evals(vec![E::from_u64(7)], 1).unwrap();

        let (p0, p1) = relation_weight.pair_flat(0, 1);
        assert_eq!(p0, E::from_u64(7));
        assert_eq!(p1, E::zero());

        let (p0, p1) = relation_weight.pair_flat(2, 3);
        assert_eq!(p0, E::zero());
        assert_eq!(p1, E::zero());
    }
}
