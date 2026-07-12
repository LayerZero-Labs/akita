//! Estimator configuration and optimizer selection types.

use crate::{
    error::{EstimatorError, Result},
    numeric::{NumericConfig, Probability},
};

/// BKZ/reduction cost model.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ReductionCostModel {
    /// Albrecht-Ducas-Pöppelmann-Schwabe 2016 model.
    Adps16 {
        /// Cost mode.
        mode: Adps16Mode,
    },
    /// BCSS23 idealized quantum-sieving model with writable QRAQM.
    Bcss23Idealized,
    /// Becker-Ducas-Gama-Laarhoven 2016 model.
    Bdgl16,
    /// MATZOV model.
    Matzov {
        /// Nearest-neighbor model used by the short-vector backend.
        nearest_neighbor: NearestNeighborModel,
    },
    /// GJ21 model.
    Gj21 {
        /// Nearest-neighbor model used by the short-vector backend.
        nearest_neighbor: NearestNeighborModel,
    },
    /// Kyber estimator model.
    Kyber {
        /// Nearest-neighbor model used by the short-vector backend.
        nearest_neighbor: NearestNeighborModel,
    },
}

/// A hard lower bound attached to one independently optimized reduction model.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SisSecurityConstraint {
    /// Reduction model to optimize.
    pub reduction_model: ReductionCostModel,
    /// Minimum accepted `log2(rop)` score.
    pub minimum_log2_rop: f64,
}

/// A non-gating diagnostic attached to one independently optimized model.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SisSecurityDiagnostic {
    /// Reduction model to optimize.
    pub reduction_model: ReductionCostModel,
    /// Score below which a generated boundary requires manual review.
    pub review_below_log2_rop: f64,
}

/// Versioned SIS security policy understood by the offline estimator.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SisSecurityPolicy {
    /// Classical 138-bit and conventional-quantum 128-bit hard constraints,
    /// with one non-gating idealized BCSS23 diagnostic.
    Classical138Quantum128WithIdealizedBcssV1,
}

impl SisSecurityPolicy {
    /// Stable descriptive label shared with generated runtime policy identity.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Classical138Quantum128WithIdealizedBcssV1 => {
                "Classical138Quantum128WithIdealizedBcssV1"
            }
        }
    }

    /// Hard ADPS16 classical constraint.
    #[must_use]
    pub const fn classical_constraint(self) -> SisSecurityConstraint {
        match self {
            Self::Classical138Quantum128WithIdealizedBcssV1 => SisSecurityConstraint {
                reduction_model: ReductionCostModel::Adps16 {
                    mode: Adps16Mode::Classical,
                },
                minimum_log2_rop: 138.0,
            },
        }
    }

    /// Hard conventional ADPS16 quantum constraint.
    #[must_use]
    pub const fn conventional_quantum_constraint(self) -> SisSecurityConstraint {
        match self {
            Self::Classical138Quantum128WithIdealizedBcssV1 => SisSecurityConstraint {
                reduction_model: ReductionCostModel::Adps16 {
                    mode: Adps16Mode::Quantum,
                },
                minimum_log2_rop: 128.0,
            },
        }
    }

    /// Non-gating idealized BCSS23 diagnostic and manual-review line.
    #[must_use]
    pub const fn idealized_bcss_diagnostic(self) -> SisSecurityDiagnostic {
        match self {
            Self::Classical138Quantum128WithIdealizedBcssV1 => SisSecurityDiagnostic {
                reduction_model: ReductionCostModel::Bcss23Idealized,
                review_below_log2_rop: 124.0,
            },
        }
    }
}

impl Default for ReductionCostModel {
    fn default() -> Self {
        Self::Adps16 {
            mode: Adps16Mode::Classical,
        }
    }
}

/// ADPS16 cost mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Adps16Mode {
    /// Classical cost.
    Classical,
    /// Quantum cost.
    Quantum,
    /// Paranoid cost.
    Paranoid,
}

/// Nearest-neighbor backend used by several lattice-estimator reduction models.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum NearestNeighborModel {
    /// Classical nearest-neighbor model.
    #[default]
    Classical,
    /// Quantum nearest-neighbor model.
    Quantum,
    /// Conservative/paranoid nearest-neighbor model.
    Paranoid,
}

/// Reduced-basis shape model.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ShapeModel {
    /// Geometric series assumption.
    Gsa,
    /// Z-shaped q-ary profile.
    Zgsa,
    /// L-shaped rerandomized q-ary profile.
    #[default]
    Lgsa,
    /// Chen-Nguyen simulator.
    Cn11,
    /// Chen-Nguyen simulator ignoring q-ary structure.
    Cn11NoQary,
}

/// Optimizer configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OptimizerConfig {
    /// Evaluate one fixed beta and zeta.
    Fixed {
        /// BKZ block size.
        beta: u32,
        /// Number of zeroed coordinates.
        zeta: u64,
    },
    /// Optimize beta while keeping zeta fixed.
    OptimizeBeta {
        /// Fixed zeta.
        zeta: u64,
        /// Beta search strategy.
        beta: SearchMode,
    },
    /// Optimize beta and zeta.
    OptimizeZeta {
        /// Beta search strategy.
        beta: SearchMode,
        /// Zeta search strategy.
        zeta: SearchMode,
    },
}

impl OptimizerConfig {
    /// Validate optimizer parameters.
    ///
    /// # Errors
    ///
    /// Returns an error when a fixed beta is zero.
    pub fn validate(&self) -> Result<()> {
        match *self {
            Self::Fixed { beta, zeta: _ } => validate_min_beta("optimizer.beta", beta),
            Self::OptimizeBeta { zeta: _, .. } => Ok(()),
            Self::OptimizeZeta { .. } => Ok(()),
        }
    }
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self::OptimizeZeta {
            beta: SearchMode::PythonLocalMinimum,
            zeta: SearchMode::PythonLocalMinimum,
        }
    }
}

/// Search strategy for beta or zeta.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SearchMode {
    /// Match lattice-estimator's local-minimum search shape.
    PythonLocalMinimum,
    /// Exhaustively scan the configured search interval.
    Exhaustive,
    /// Exhaustively scan in parallel.
    ExhaustiveParallel,
    /// Future pruned search with a proof that skipped cells cannot win.
    ProvenPruned,
}

/// Top-level estimator configuration.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct EstimateConfig {
    /// Reduction cost model.
    pub red_cost_model: ReductionCostModel,
    /// Reduced-basis shape model.
    pub red_shape_model: ShapeModel,
    /// Optimizer mode.
    pub optimizer: OptimizerConfig,
    /// Target success probability.
    pub success_probability: Probability,
    /// Optional lattice dimension override matching lattice-estimator's `d`
    /// argument on fixed-cost calls.
    pub lattice_dimension: Option<u64>,
    /// Numeric precision and tolerance policy.
    pub numeric: NumericConfig,
}

impl EstimateConfig {
    /// Akita infinity table generation profile: ADPS16 classical + LGSA with
    /// exhaustive beta and zeta search.
    #[must_use]
    pub fn akita_infinity_table() -> Self {
        Self {
            optimizer: OptimizerConfig::OptimizeZeta {
                beta: SearchMode::Exhaustive,
                zeta: SearchMode::Exhaustive,
            },
            ..Self::default()
        }
    }

    /// Akita Euclidean table generation profile: BDGL16 with the Euclidean
    /// SIS lattice path used by the shipped 128-bit L2 table.
    #[must_use]
    pub fn akita_euclidean_table() -> Self {
        Self {
            red_cost_model: ReductionCostModel::Bdgl16,
            ..Self::default()
        }
    }

    /// Lattice-estimator parity profile: ADPS16 classical + LGSA with Python's
    /// local-minimum beta and zeta search.
    #[must_use]
    pub fn lattice_estimator_parity() -> Self {
        Self::default()
    }

    /// Validate all configuration fields.
    ///
    /// # Errors
    ///
    /// Returns an error when an optimizer or numeric setting is malformed.
    pub fn validate(&self) -> Result<()> {
        self.optimizer.validate()?;
        if self.lattice_dimension == Some(0) {
            return Err(EstimatorError::InvalidConfig {
                field: "lattice_dimension",
                reason: "lattice dimension override must be positive".to_string(),
            });
        }
        self.numeric.validate()
    }
}

fn validate_min_beta(field: &'static str, beta: u32) -> Result<()> {
    if beta < 2 {
        return Err(EstimatorError::InvalidConfig {
            field,
            reason: "beta must be at least 2".to_string(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_profile_is_akita_infinity_target_shape() {
        let config = EstimateConfig::default();
        assert_eq!(
            config.red_cost_model,
            ReductionCostModel::Adps16 {
                mode: Adps16Mode::Classical
            }
        );
        assert_eq!(config.red_shape_model, ShapeModel::Lgsa);
        assert_eq!(
            config.optimizer,
            OptimizerConfig::OptimizeZeta {
                beta: SearchMode::PythonLocalMinimum,
                zeta: SearchMode::PythonLocalMinimum
            }
        );
        assert!(config.validate().is_ok());
    }

    #[test]
    fn optimizer_validation_rejects_beta_below_two_for_fixed_mode() {
        assert!(OptimizerConfig::Fixed { beta: 0, zeta: 1 }
            .validate()
            .is_err());
        assert!(OptimizerConfig::Fixed { beta: 1, zeta: 1 }
            .validate()
            .is_err());
        assert!(OptimizerConfig::Fixed { beta: 64, zeta: 0 }
            .validate()
            .is_ok());
    }

    #[test]
    fn optimizer_validation_allows_zero_zeta_for_non_fixed_modes() {
        assert!(OptimizerConfig::OptimizeBeta {
            zeta: 0,
            beta: SearchMode::PythonLocalMinimum,
        }
        .validate()
        .is_ok());
    }

    #[test]
    fn lattice_dimension_override_must_be_positive_when_present() {
        assert!(EstimateConfig {
            lattice_dimension: Some(1),
            ..EstimateConfig::default()
        }
        .validate()
        .is_ok());
        assert!(EstimateConfig {
            lattice_dimension: Some(0),
            ..EstimateConfig::default()
        }
        .validate()
        .is_err());
    }

    #[test]
    fn enum_surface_covers_specified_models() {
        let reduction_models = [
            ReductionCostModel::Adps16 {
                mode: Adps16Mode::Classical,
            },
            ReductionCostModel::Adps16 {
                mode: Adps16Mode::Quantum,
            },
            ReductionCostModel::Adps16 {
                mode: Adps16Mode::Paranoid,
            },
            ReductionCostModel::Bcss23Idealized,
            ReductionCostModel::Bdgl16,
            ReductionCostModel::Matzov {
                nearest_neighbor: NearestNeighborModel::Classical,
            },
            ReductionCostModel::Gj21 {
                nearest_neighbor: NearestNeighborModel::Quantum,
            },
            ReductionCostModel::Kyber {
                nearest_neighbor: NearestNeighborModel::Paranoid,
            },
        ];
        let shape_models = [
            ShapeModel::Gsa,
            ShapeModel::Zgsa,
            ShapeModel::Lgsa,
            ShapeModel::Cn11,
            ShapeModel::Cn11NoQary,
        ];
        let search_modes = [
            SearchMode::PythonLocalMinimum,
            SearchMode::Exhaustive,
            SearchMode::ExhaustiveParallel,
            SearchMode::ProvenPruned,
        ];

        assert_eq!(reduction_models.len(), 8);
        assert_eq!(shape_models.len(), 5);
        assert_eq!(search_modes.len(), 4);
    }

    #[test]
    fn descriptive_security_policy_pins_models_and_thresholds() {
        let policy = SisSecurityPolicy::Classical138Quantum128WithIdealizedBcssV1;
        assert_eq!(
            policy.classical_constraint(),
            SisSecurityConstraint {
                reduction_model: ReductionCostModel::Adps16 {
                    mode: Adps16Mode::Classical,
                },
                minimum_log2_rop: 138.0,
            }
        );
        assert_eq!(
            policy.conventional_quantum_constraint(),
            SisSecurityConstraint {
                reduction_model: ReductionCostModel::Adps16 {
                    mode: Adps16Mode::Quantum,
                },
                minimum_log2_rop: 128.0,
            }
        );
        assert_eq!(
            policy.idealized_bcss_diagnostic(),
            SisSecurityDiagnostic {
                reduction_model: ReductionCostModel::Bcss23Idealized,
                review_below_log2_rop: 124.0,
            }
        );
    }
}
