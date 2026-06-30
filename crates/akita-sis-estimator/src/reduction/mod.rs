//! Reduction cost models.

pub mod adps16;
pub mod bdgl16;
pub mod delta;
pub mod short_vectors;

pub use adps16::{adps16_log2_cost, adps16_short_vectors, log2_to_cost_value};
pub use bdgl16::{bdgl16_log2_cost, bdgl16_short_vectors};
pub use delta::delta;
pub use short_vectors::ShortVectors;

use crate::{
    config::ReductionCostModel,
    error::{EstimatorError, Result},
};

/// Validate that the configured reduction model is implemented on the infinity path.
pub fn validate_infinity_reduction(model: ReductionCostModel) -> Result<()> {
    match model {
        ReductionCostModel::Adps16 { .. } | ReductionCostModel::Bdgl16 => Ok(()),
        ReductionCostModel::Matzov { .. } => Err(EstimatorError::Unsupported {
            feature: "red_cost_model::Matzov",
        }),
        ReductionCostModel::Gj21 { .. } => Err(EstimatorError::Unsupported {
            feature: "red_cost_model::Gj21",
        }),
        ReductionCostModel::Kyber { .. } => Err(EstimatorError::Unsupported {
            feature: "red_cost_model::Kyber",
        }),
    }
}

/// BKZ preprocessing cost in log₂ space for the configured reduction model.
pub fn log2_bkz_cost(
    model: ReductionCostModel,
    beta: u32,
    effective_dimension: u32,
) -> Result<f64> {
    validate_infinity_reduction(model)?;
    Ok(match model {
        ReductionCostModel::Adps16 { mode } => adps16_log2_cost(beta, mode),
        ReductionCostModel::Bdgl16 => bdgl16_log2_cost(beta, effective_dimension),
        ReductionCostModel::Matzov { .. }
        | ReductionCostModel::Gj21 { .. }
        | ReductionCostModel::Kyber { .. } => unreachable!("validated above"),
    })
}

/// Short-vector sieve output for the configured reduction model.
pub fn short_vectors_for(
    model: ReductionCostModel,
    beta: u32,
    effective_dimension: u32,
) -> Result<ShortVectors> {
    validate_infinity_reduction(model)?;
    Ok(match model {
        ReductionCostModel::Adps16 { mode } => {
            adps16_short_vectors(beta, effective_dimension, mode)
        }
        ReductionCostModel::Bdgl16 => bdgl16_short_vectors(beta, effective_dimension),
        ReductionCostModel::Matzov { .. }
        | ReductionCostModel::Gj21 { .. }
        | ReductionCostModel::Kyber { .. } => unreachable!("validated above"),
    })
}
