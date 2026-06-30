//! Offline SIS lattice estimator API.
//!
//! This crate is intentionally offline-only. It exposes a typed surface that
//! mirrors the SIS inputs accepted by `lattice-estimator`, while later slices
//! fill in the estimator formulas and optimizer implementations.

#![warn(missing_docs)]
#![warn(unreachable_pub)]

pub mod akita;
pub mod config;
pub mod cost;
pub mod error;
pub mod lattice;
pub mod math;
pub mod numeric;
pub mod params;
pub mod probability;
pub mod reduction;
pub mod simulator;

pub use akita::{scalar_sis_from_ring, AkitaModulusFamily};
pub use config::{
    Adps16Mode, EstimateConfig, NearestNeighborModel, OptimizerConfig, ReductionCostModel,
    SearchMode, ShapeModel,
};
pub use cost::{CostValue, EstimateTag, LatticeCost, LogCost};
pub use error::{EstimatorError, Result};
pub use numeric::{GoldenTrust, NumericBackend, NumericConfig, Probability};
pub use params::{
    akita_q128, akita_q32, akita_q64, Bound, SisNorm, SisParameterUpdate, SisParameters,
};

/// Estimate the cheapest SIS lattice attack for the configured optimizer.
///
/// # Errors
///
/// Returns validation errors for malformed inputs. The actual estimator math is
/// implemented in later slices, so valid inputs currently return
/// [`EstimatorError::Unsupported`].
pub fn estimate(params: &SisParameters, config: &EstimateConfig) -> Result<LatticeCost> {
    params.validate()?;
    config.validate()?;
    Err(EstimatorError::Unsupported {
        feature: "estimate",
    })
}

/// Evaluate a fixed-beta, fixed-zeta infinity-norm SIS lattice cost.
///
/// # Errors
///
/// Returns validation errors for malformed inputs. The actual estimator math is
/// implemented for the fixed ADPS16 + LGSA target profile in this slice. Other
/// profiles return [`EstimatorError::Unsupported`].
pub fn cost_infinity(
    beta: u32,
    params: &SisParameters,
    zeta: u32,
    config: &EstimateConfig,
) -> Result<LatticeCost> {
    params.validate()?;
    config.validate()?;
    validate_beta_zeta(beta, zeta)?;
    if params.norm != SisNorm::Infinity {
        return Err(EstimatorError::InvalidParameter {
            field: "norm",
            reason: "cost_infinity requires SisNorm::Infinity".to_string(),
        });
    }
    lattice::cost_infinity_fixed(beta, params, zeta, config)
}

/// Evaluate the best beta for one fixed zeta.
///
/// # Errors
///
/// Returns validation errors for malformed inputs. The actual estimator math is
/// implemented in later slices, so valid inputs currently return
/// [`EstimatorError::Unsupported`].
pub fn cost_zeta(
    zeta: u32,
    params: &SisParameters,
    config: &EstimateConfig,
) -> Result<LatticeCost> {
    params.validate()?;
    config.validate()?;
    let _ = zeta;
    Err(EstimatorError::Unsupported {
        feature: "cost_zeta",
    })
}

/// Evaluate the Euclidean-norm SIS lattice cost.
///
/// # Errors
///
/// Returns validation errors for malformed inputs. The actual estimator math is
/// implemented in later slices, so valid inputs currently return
/// [`EstimatorError::Unsupported`].
pub fn cost_euclidean(params: &SisParameters, config: &EstimateConfig) -> Result<LatticeCost> {
    params.validate()?;
    config.validate()?;
    if params.norm != SisNorm::Euclidean {
        return Err(EstimatorError::InvalidParameter {
            field: "norm",
            reason: "cost_euclidean requires SisNorm::Euclidean".to_string(),
        });
    }
    Err(EstimatorError::Unsupported {
        feature: "cost_euclidean",
    })
}

fn validate_beta_zeta(beta: u32, zeta: u32) -> Result<()> {
    if beta < 2 {
        return Err(EstimatorError::InvalidParameter {
            field: "beta",
            reason: "beta must be at least 2".to_string(),
        });
    }
    let _ = zeta;
    Ok(())
}

#[cfg(test)]
mod tests {
    use num_bigint::BigUint;

    use super::*;

    fn sample_params(norm: SisNorm) -> SisParameters {
        SisParameters::try_new(
            32,
            BigUint::from(4_294_967_197u64),
            Some(64),
            Bound::from_u64(15),
            norm,
        )
        .unwrap()
    }

    #[test]
    fn optimizer_entry_point_is_explicitly_unsupported_for_now() {
        let params = sample_params(SisNorm::Infinity);
        let config = EstimateConfig::default();
        assert!(matches!(
            estimate(&params, &config),
            Err(EstimatorError::Unsupported {
                feature: "estimate"
            })
        ));
        assert!(cost_infinity(64, &params, 0, &config).is_ok());
    }

    #[test]
    fn norm_specific_entry_points_reject_wrong_norm() {
        let config = EstimateConfig::default();
        assert!(matches!(
            cost_infinity(64, &sample_params(SisNorm::Euclidean), 1, &config),
            Err(EstimatorError::InvalidParameter { field: "norm", .. })
        ));
        assert!(matches!(
            cost_euclidean(&sample_params(SisNorm::Infinity), &config),
            Err(EstimatorError::InvalidParameter { field: "norm", .. })
        ));
    }
}
