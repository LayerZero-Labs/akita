//! Reduction cost models.

pub mod adps16;
pub mod bcss23;
pub mod bdgl16;
pub mod delta;
pub mod short_vectors;

pub use adps16::{adps16_log2_cost, adps16_short_vectors, log2_to_cost_value};
pub use bcss23::{
    bcss23_idealized_log2_cost, bcss23_idealized_short_vectors, BCSS23_IDEALIZED_EXPONENT,
};
pub use bdgl16::{bdgl16_log2_cost, bdgl16_short_vectors};
pub use delta::{beta, delta};
pub use short_vectors::ShortVectors;

use crate::{
    config::ReductionCostModel,
    error::{EstimatorError, Result},
};

/// Validate that the configured reduction model is implemented on the infinity path.
pub fn validate_infinity_reduction(model: ReductionCostModel) -> Result<()> {
    match model {
        ReductionCostModel::Adps16 { .. }
        | ReductionCostModel::Bcss23Idealized
        | ReductionCostModel::Bdgl16 => Ok(()),
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

/// Validate that the configured reduction model is implemented on the
/// Euclidean SIS path.
pub fn validate_euclidean_reduction(model: ReductionCostModel) -> Result<()> {
    match model {
        ReductionCostModel::Bdgl16 => Ok(()),
        ReductionCostModel::Adps16 { .. } => Err(EstimatorError::Unsupported {
            feature: "euclidean red_cost_model::ADPS16",
        }),
        ReductionCostModel::Bcss23Idealized => Err(EstimatorError::Unsupported {
            feature: "euclidean red_cost_model::BCSS23 idealized",
        }),
        ReductionCostModel::Matzov { .. } => Err(EstimatorError::Unsupported {
            feature: "euclidean red_cost_model::Matzov",
        }),
        ReductionCostModel::Gj21 { .. } => Err(EstimatorError::Unsupported {
            feature: "euclidean red_cost_model::Gj21",
        }),
        ReductionCostModel::Kyber { .. } => Err(EstimatorError::Unsupported {
            feature: "euclidean red_cost_model::Kyber",
        }),
    }
}

/// BKZ preprocessing cost in log₂ space for the configured reduction model.
pub fn log2_bkz_cost(
    model: ReductionCostModel,
    beta: u32,
    effective_dimension: u64,
) -> Result<f64> {
    validate_infinity_reduction(model)?;
    Ok(match model {
        ReductionCostModel::Adps16 { mode } => {
            let _ = effective_dimension;
            adps16_log2_cost(beta, mode)
        }
        ReductionCostModel::Bcss23Idealized => {
            let _ = effective_dimension;
            bcss23_idealized_log2_cost(beta)
        }
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
    effective_dimension: u64,
) -> Result<ShortVectors> {
    validate_infinity_reduction(model)?;
    Ok(match model {
        ReductionCostModel::Adps16 { mode } => {
            let dimension = u32::try_from(effective_dimension).unwrap_or(u32::MAX);
            adps16_short_vectors(beta, dimension, mode)
        }
        ReductionCostModel::Bcss23Idealized => {
            let _ = effective_dimension;
            bcss23_idealized_short_vectors(beta)
        }
        ReductionCostModel::Bdgl16 => bdgl16_short_vectors(beta, effective_dimension),
        ReductionCostModel::Matzov { .. }
        | ReductionCostModel::Gj21 { .. }
        | ReductionCostModel::Kyber { .. } => unreachable!("validated above"),
    })
}
