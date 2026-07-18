//! Shared setup-contribution planning for prover and verifier.
//!
//! This module owns challenge-free geometry (`geometry.rs`), pure layout/weight
//! derivation for the stage-3 setup product, and evaluation planning. The
//! prover consumes the materialized setup-index weight vector: one scalar weight
//! per packed setup position. The recursive stage-3 verifier evaluates the
//! multilinear extension of that weight vector directly at the setup-index
//! challenge point, while the direct verifier scans the packed setup with the
//! same segment partition.

use crate::{LevelParams, OpeningClaimsLayout};
use akita_field::{AkitaError, CanonicalField, FieldCore};

mod geometry;
mod plan;
mod setup_index_weight_evaluator;
mod weights;

#[cfg(test)]
mod tests;

pub use geometry::{ensure_setup_envelope, SetupProjectionGeometry};
pub(crate) use plan::get_d_col_range;
#[cfg(test)]
pub(crate) use plan::validate_setup_inputs;
pub use plan::{SetupContributionGroupInputs, SetupContributionPlan};
pub use setup_index_weight_evaluator::SetupIndexWeightEvaluator;

/// Shared fold gadget when every setup-contribution group uses the same basis.
///
/// Groups may have different fold depths: each group uses the prefix
/// `gadget[..group.depth_fold]`. Return `None` only when the basis differs and
/// callers must derive per-group gadgets.
pub fn shared_setup_fold_gadget<F: FieldCore + CanonicalField>(
    level_params: &LevelParams,
    opening_batch: &OpeningClaimsLayout,
    groups: &[SetupContributionGroupInputs],
) -> Option<Vec<F>> {
    let first = groups.first()?;
    let first_log_basis = first.log_basis(level_params, opening_batch).ok()?;
    if !groups.iter().all(|group| {
        group
            .log_basis(level_params, opening_batch)
            .is_ok_and(|log_basis| log_basis == first_log_basis)
    }) {
        return None;
    }
    let max_depth = groups
        .iter()
        .map(|group| group.depth_fold)
        .max()
        .unwrap_or(first.depth_fold);
    Some(crate::gadget_row_scalars::<F>(max_depth, first_log_basis))
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
