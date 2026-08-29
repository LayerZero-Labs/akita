//! Typed relation-weight states at the ring-switch and Stage-2 boundaries.

use akita_error::AkitaError;
use akita_field::FieldCore;
use akita_types::CompressionRelationWeights;

use crate::protocol::ring_switch::RelationWeightFactorization;

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

    pub(crate) fn into_evaluations(self) -> Vec<E> {
        self.evaluations
    }
}

/// Transcript-complete relation weights before payload-specific sparse terms
/// are separated at the Stage-2 boundary.
pub(crate) enum CompiledRelationWeights<E: FieldCore> {
    QuotientLift {
        ordinary: RelationWeightFactorization<E>,
        compression: Option<CompressionRelationWeights<E>>,
    },
    ReducedEvaluation(DenseRelationWeights<E>),
}

/// Canonical primary relation-weight state owned and folded by Stage 2.
pub(crate) enum RelationWeightOracle<E: FieldCore> {
    QuotientFactored {
        common_alpha_factor: Vec<E>,
        relation_lane_weights: Vec<E>,
    },
    ReducedDense {
        lane_evaluations: Vec<E>,
    },
}

impl<E: FieldCore> CompiledRelationWeights<E> {
    pub(crate) fn into_stage2(
        self,
    ) -> (
        RelationWeightOracle<E>,
        Option<CompressionRelationWeights<E>>,
    ) {
        match self {
            Self::QuotientLift {
                ordinary,
                compression,
            } => {
                let (common_alpha_factor, relation_lane_weights) =
                    ordinary.into_common_alpha_factor_and_relation_lane_weights();
                (
                    RelationWeightOracle::QuotientFactored {
                        common_alpha_factor,
                        relation_lane_weights,
                    },
                    compression,
                )
            }
            Self::ReducedEvaluation(dense) => (
                RelationWeightOracle::ReducedDense {
                    lane_evaluations: dense.into_evaluations(),
                },
                None,
            ),
        }
    }
}

impl<E: FieldCore> RelationWeightOracle<E> {
    pub(super) fn is_reduced_dense(&self) -> bool {
        matches!(self, Self::ReducedDense { .. })
    }

    pub(super) fn dense_evaluations(&self) -> Option<&[E]> {
        match self {
            Self::QuotientFactored { .. } => None,
            Self::ReducedDense { lane_evaluations } => Some(lane_evaluations),
        }
    }

    pub(super) fn bind_dense(&mut self, challenge: E) -> Result<(), AkitaError>
    where
        E: akita_field::unreduced::HasOptimizedFold,
    {
        match self {
            Self::QuotientFactored { .. } => Err(AkitaError::InvalidProof),
            Self::ReducedDense { lane_evaluations } => {
                akita_sumcheck::fold_evals_in_place(lane_evaluations, challenge);
                Ok(())
            }
        }
    }

    pub(super) fn terminal_weight(&self) -> Result<E, AkitaError> {
        match self {
            Self::QuotientFactored {
                common_alpha_factor,
                relation_lane_weights,
            } if common_alpha_factor.len() == 1 && relation_lane_weights.len() == 1 => {
                Ok(common_alpha_factor[0] * relation_lane_weights[0])
            }
            Self::ReducedDense { lane_evaluations } if lane_evaluations.len() == 1 => {
                Ok(lane_evaluations[0])
            }
            _ => Err(AkitaError::InvalidProof),
        }
    }
}
