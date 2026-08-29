//! Typed relation-weight states at the ring-switch and Stage-2 boundaries.

use crate::protocol::ring_switch::RelationWeightFactorization;
use akita_error::AkitaError;
use akita_field::FieldCore;

/// Complete padded relation weights for a quotient-free Stage-2 instance.
pub(crate) struct DenseRelationWeights<E: FieldCore> {
    evaluations: Vec<E>,
}

impl<E: FieldCore> DenseRelationWeights<E> {
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
        Ok(Self { evaluations })
    }

    pub(crate) fn evaluations(&self) -> &[E] {
        &self.evaluations
    }

    pub(crate) fn bind(&mut self, challenge: E)
    where
        E: akita_field::unreduced::HasOptimizedFold,
    {
        akita_sumcheck::fold_evals_in_place(&mut self.evaluations, challenge);
    }

    pub(crate) fn terminal_weight(&self) -> Result<E, AkitaError> {
        if let [weight] = self.evaluations.as_slice() {
            Ok(*weight)
        } else {
            Err(AkitaError::InvalidProof)
        }
    }
}

/// Transcript-complete relation weights before payload-specific sparse terms
/// are separated at the Stage-2 boundary.
pub(crate) enum CompiledRelationWeights<E: FieldCore> {
    QuotientLift(RelationWeightFactorization<E>),
    ReducedEvaluation(DenseRelationWeights<E>),
}

/// Canonical primary relation-weight state owned and folded by Stage 2.
pub(crate) enum RelationWeightOracle<E: FieldCore> {
    QuotientFactored(RelationWeightFactorization<E>),
    ReducedDense(DenseRelationWeights<E>),
}

impl<E: FieldCore> CompiledRelationWeights<E> {
    pub(crate) fn into_stage2(self) -> RelationWeightOracle<E> {
        match self {
            Self::QuotientLift(factorization) => {
                RelationWeightOracle::QuotientFactored(factorization)
            }
            Self::ReducedEvaluation(dense) => RelationWeightOracle::ReducedDense(dense),
        }
    }
}

impl<E: FieldCore> RelationWeightOracle<E> {
    pub(super) fn terminal_weight(&self) -> Result<E, AkitaError> {
        match self {
            Self::QuotientFactored(factorization)
                if factorization.common_alpha_factor().len() == 1
                    && factorization.relation_lane_weights().len() == 1 =>
            {
                Ok(factorization.common_alpha_factor()[0]
                    * factorization.relation_lane_weights()[0])
            }
            Self::QuotientFactored(_) => Err(AkitaError::InvalidProof),
            Self::ReducedDense(dense) => dense.terminal_weight(),
        }
    }
}
