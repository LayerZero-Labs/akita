//! Reduction cost models.

pub mod adps16;
pub mod delta;
pub mod short_vectors;

pub use adps16::{adps16_log2_cost, adps16_short_vectors, log2_to_cost_value};
pub use delta::delta;
pub use short_vectors::ShortVectors;

use crate::{
    config::{Adps16Mode, ReductionCostModel},
    error::{EstimatorError, Result},
};

/// Validate that the configured reduction model is implemented on the infinity path.
pub fn validate_infinity_reduction(model: ReductionCostModel) -> Result<()> {
    match model {
        ReductionCostModel::Adps16 { .. } => Ok(()),
        ReductionCostModel::Bdgl16 => Err(EstimatorError::Unsupported {
            feature: "red_cost_model::Bdgl16",
        }),
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

/// BKZ cost `log2(2^(c * beta))` for the configured reduction model.
pub fn log2_bkz_cost(model: ReductionCostModel, beta: u32) -> Result<f64> {
    validate_infinity_reduction(model)?;
    Ok(adps16_log2_cost(beta, adps16_mode(model)))
}

/// Short-vector sieve output for the configured reduction model.
pub fn short_vectors_for(
    model: ReductionCostModel,
    beta: u32,
    effective_dimension: u32,
) -> Result<ShortVectors> {
    validate_infinity_reduction(model)?;
    Ok(adps16_short_vectors(
        beta,
        effective_dimension,
        adps16_mode(model),
    ))
}

fn adps16_mode(model: ReductionCostModel) -> Adps16Mode {
    match model {
        ReductionCostModel::Adps16 { mode } => mode,
        _ => Adps16Mode::Classical,
    }
}
