//! Shared setup-contribution planning for prover and verifier.
//!
//! This module owns challenge-free geometry (`geometry.rs`), pure layout/weight
//! derivation for the stage-3 setup product, and evaluation planning. The
//! prover consumes the materialized setup-index weight vector: one scalar weight
//! per packed setup position. The recursive stage-3 verifier evaluates the
//! multilinear extension of that weight vector directly at the setup-index
//! challenge point, while the direct verifier scans the packed setup with the
//! same segment partition.

use akita_field::AkitaError;

mod geometry;
mod inputs;
mod plan;
mod setup_index_weight_evaluator;
mod weights;

#[cfg(test)]
mod tests;

pub use geometry::{
    ensure_setup_envelope, setup_active_ring_elems_at, setup_active_ring_elems_for_fold,
    setup_required_for_inputs, SetupProjectionGeometry,
};
pub use inputs::SetupContributionPlanInputs;
pub use plan::{
    SetupContributionGroupInputs, SetupContributionPlan, SetupContributionStatic,
    SingleGroupSetupContributionLayout,
};
pub use setup_index_weight_evaluator::SetupIndexWeightEvaluator;

pub(crate) fn push_role_boundaries(
    endpoints: &mut Vec<usize>,
    rows: usize,
    stride: usize,
    name: &'static str,
) -> Result<(), AkitaError> {
    if rows == 0 || stride == 0 {
        return Ok(());
    }
    let mut boundary = 0usize;
    for _ in 0..rows {
        boundary = boundary
            .checked_add(stride)
            .ok_or_else(|| AkitaError::InvalidSetup(format!("packed {name} boundary overflow")))?;
        endpoints.push(boundary);
    }
    Ok(())
}

#[inline(always)]
pub(crate) fn checked_slice<'a, T>(
    slice: &'a [T],
    start: usize,
    len: usize,
    context: &'static str,
) -> Result<&'a [T], AkitaError> {
    let end = start
        .checked_add(len)
        .ok_or_else(|| AkitaError::InvalidSetup(format!("{context} overflow")))?;
    slice.get(start..end).ok_or(AkitaError::InvalidProof)
}
