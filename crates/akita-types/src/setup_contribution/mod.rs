//! Shared setup-contribution planning for prover and verifier.
//!
//! This module owns the pure layout/weight derivation for the stage-3 setup
//! product. The prover consumes the materialized `bar_omega` vector, while the
//! verifier can evaluate the same plan directly against the packed setup.

mod bounds;
mod inputs;
mod plan;
mod weights;

#[cfg(test)]
mod tests;

pub use inputs::SetupContributionPlanInputs;
pub use plan::{SetupContributionGroupInputs, SetupContributionPlan, SetupContributionStatic};

pub(crate) use bounds::{checked_add, checked_mul, checked_slice, push_role_boundaries};
