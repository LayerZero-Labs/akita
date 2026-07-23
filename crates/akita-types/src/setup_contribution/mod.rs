//! Shared setup-contribution planning for prover and verifier.
//!
//! This module owns challenge-free geometry (`geometry.rs`), pure layout/weight
//! derivation for the stage-3 setup product, and evaluation planning. The
//! prover consumes the materialized setup-index weight vector: one scalar weight
//! per packed setup position. The recursive stage-3 verifier evaluates the
//! multilinear extension of that weight vector directly at the setup-index
//! challenge point, while the direct verifier scans the packed setup with the
//! same segment partition.

mod geometry;
mod plan;
#[cfg(test)]
mod weights;

#[cfg(test)]
mod tests;

pub use geometry::{ensure_setup_envelope, SetupProjectionGeometry};
#[cfg(test)]
pub(crate) use plan::get_d_col_range;
#[cfg(test)]
pub(crate) use plan::validate_setup_inputs;
pub use plan::SetupContributionPlan;
