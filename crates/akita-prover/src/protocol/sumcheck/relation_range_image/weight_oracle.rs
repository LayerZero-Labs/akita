//! Typed relation-weight states at the ring-switch and Stage-2 boundaries.

use crate::protocol::ring_switch::RelationWeightFactorization;
use akita_error::AkitaError;
use jolt_field::Field;

/// Complete padded relation weights for a quotient-free Stage-2 instance.
pub(crate) struct DenseRelationWeights<E: Field> {
    evaluations: Vec<E>,
    live_len: usize,
}

impl<E: Field> DenseRelationWeights<E> {
    pub(crate) fn new(evaluations: Vec<E>, live_len: usize) -> Result<Self, AkitaError> {
        if evaluations.is_empty()
            || !evaluations.len().is_power_of_two()
            || live_len == 0
            || live_len > evaluations.len()
        {
            return Err(AkitaError::InvalidSize {
                expected: evaluations.len(),
                actual: live_len,
            });
        }
        Ok(Self {
            evaluations,
            live_len,
        })
    }

    pub(crate) fn evaluations(&self) -> &[E] {
        &self.evaluations
    }

    pub(crate) const fn live_len(&self) -> usize {
        self.live_len
    }

    pub(crate) fn bind(&mut self, challenge: E)
    where
        E: jolt_field::Fold,
    {
        akita_sumcheck::fold_evals_in_place(&mut self.evaluations, challenge);
        self.live_len = self.live_len.div_ceil(2);
    }

    pub(crate) fn terminal_weight(&self) -> Result<E, AkitaError> {
        if self.live_len == 1 {
            if let [weight] = self.evaluations.as_slice() {
                return Ok(*weight);
            }
        }
        Err(AkitaError::InvalidProof)
    }
}

/// Canonical primary relation-weight state owned and folded by Stage 2.
pub(crate) enum RelationWeightOracle<E: Field> {
    QuotientFactored(RelationWeightFactorization<E>),
    ReducedDense(DenseRelationWeights<E>),
}
