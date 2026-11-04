//! Typed relation-weight states at the ring-switch and Stage-2 boundaries.

use crate::opaque::relation_weights::RelationWeightFactorization;
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
            return Err(AkitaError::Internal(format!(
                "dense relation table shape or live length: expected {}, actual {live_len}",
                evaluations.len(),
            )));
        }
        Ok(Self {
            evaluations,
            live_len,
        })
    }

    pub(crate) fn evaluations(&self) -> &[E] {
        &self.evaluations
    }

    pub(crate) fn take_evaluations(&mut self) -> Vec<E> {
        std::mem::take(&mut self.evaluations)
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
        Err(AkitaError::Internal(
            "terminal dense relation weights are not fully folded".into(),
        ))
    }
}

/// Complete relation weights kept in a caller's structured form.
///
/// Stage 2 reads the pair `(q(left), q(left + 1))` in each coefficient round
/// and binds the lowest live variable after it. At the lane transition the
/// oracle hands over its folded dense table. Every value must equal the
/// corresponding entry of the dense table folded by the same challenges.
pub(crate) trait StructuredRelationWeights<E: Field>: Send + Sync {
    /// Length of the current, partially bound table.
    fn len(&self) -> usize;
    /// Weights at the even address `left` and its neighbour.
    fn pair(&self, left: usize) -> (E, E);
    /// Bind the lowest live variable to `challenge`.
    fn bind(&mut self, challenge: E);
    /// Take the current table in dense form, leaving the oracle empty.
    fn take_dense(&mut self) -> Vec<E>;
}

/// Canonical primary relation-weight state owned and folded by Stage 2.
pub(crate) enum RelationWeightOracle<E: Field> {
    QuotientFactored(RelationWeightFactorization<E>),
    ReducedDense(DenseRelationWeights<E>),
    #[cfg_attr(not(any(test, feature = "labinius-trinomial")), allow(dead_code))]
    Structured(Box<dyn StructuredRelationWeights<E>>),
}
