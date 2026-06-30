//! Reduced-basis shape simulators.

pub mod lgsa;
pub mod profile;

pub use lgsa::lgsa_squared_norms;
pub use profile::ShapeProfile;

use num_bigint::BigUint;

use crate::{
    config::ShapeModel,
    error::{EstimatorError, Result},
};

/// Validate that the configured shape model is implemented on the infinity path.
pub fn validate_infinity_shape(model: ShapeModel) -> Result<()> {
    match model {
        ShapeModel::Lgsa => Ok(()),
        ShapeModel::Gsa => Err(EstimatorError::Unsupported {
            feature: "red_shape_model::GSA",
        }),
        ShapeModel::Zgsa => Err(EstimatorError::Unsupported {
            feature: "red_shape_model::ZGSA",
        }),
        ShapeModel::Cn11 => Err(EstimatorError::Unsupported {
            feature: "red_shape_model::CN11",
        }),
        ShapeModel::Cn11NoQary => Err(EstimatorError::Unsupported {
            feature: "red_shape_model::CN11_NQ",
        }),
    }
}

/// Squared-GSO profile for the configured shape model.
pub fn infinity_shape_profile(
    model: ShapeModel,
    effective_dimension: u32,
    identity_vectors: i64,
    q: &BigUint,
    beta: u32,
) -> Result<ShapeProfile> {
    validate_infinity_shape(model)?;
    let squared_norms = match model {
        ShapeModel::Lgsa => lgsa_squared_norms(effective_dimension, identity_vectors, q, beta)?,
        ShapeModel::Gsa | ShapeModel::Zgsa | ShapeModel::Cn11 | ShapeModel::Cn11NoQary => {
            unreachable!("validated above")
        }
    };
    Ok(ShapeProfile::from_squared_norms(squared_norms))
}
