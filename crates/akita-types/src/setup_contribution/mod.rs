//! Shared setup-contribution planning for prover and verifier.
//!
//! This module owns the challenge-free envelope guard (`geometry.rs`) and the pure
//! layout/weight derivation for the stage-3 setup product. The prover consumes
//! the materialized setup-index weight vector: one scalar weight per packed
//! setup position. Verifier-only evaluation of the same plan (the direct
//! setup scan and closed-form structured groups live in `akita-verifier`.
//! Coefficient functionals and the structured setup-index MLE live here so
//! external accumulation uses exactly the same arithmetic definitions.

use akita_error::{checked, AkitaError};
use akita_params::{CommittedGroupParams, OpeningClaimsLayout};
use jolt_field::{CanonicalEncoding, Field};

mod coefficient_functional;
mod geometry;
mod plan;
pub(crate) mod setup_index_weight;

#[cfg(test)]
mod tests;

pub use geometry::ensure_setup_envelope;
pub use plan::{
    factor_aligned_role_tensors, project_role_tensors, role_projection_evaluation,
    role_tensors_are_aligned, PhysicalBSetupPlan, PhysicalBWeightSegment, PhysicalBWeightTerm,
    PreparedRelationAddress, SetupContributionGroupInputs, SetupContributionGroupPlan,
    SetupContributionPlan,
};

/// Shared fold gadget when every setup-contribution group uses the same basis.
///
/// Groups may have different fold depths: each group uses the prefix
/// `gadget[..group.depth_fold]`. All fresh folded-response digits use the root
/// opening basis.
pub fn shared_setup_fold_gadget<F: Field + CanonicalEncoding>(
    level_params: &CommittedGroupParams,
    opening_batch: &OpeningClaimsLayout,
    groups: &[SetupContributionGroupInputs],
) -> Option<Vec<F>> {
    let first = groups.first()?;
    let max_depth = groups
        .iter()
        .map(|group| group.depth_fold)
        .max()
        .unwrap_or(first.depth_fold);
    let _ = opening_batch;
    Some(akita_params::gadget_row_scalars::<F>(
        max_depth,
        level_params.open().digits.log_basis,
    ))
}

/// Borrow `slice[start..start + len]`, rejecting overflow and short slices.
#[inline(always)]
pub fn checked_slice<'a, T>(
    slice: &'a [T],
    start: usize,
    len: usize,
    context: &'static str,
) -> Result<&'a [T], AkitaError> {
    let range = checked::range(start, len)
        .ok_or_else(|| AkitaError::InvalidSetup(format!("{context} overflow")))?;
    slice.get(range).ok_or_else(|| {
        AkitaError::InvalidInput(format!("{context} range exceeds the supplied slice"))
    })
}

pub use coefficient_functional::{
    materialize_reduced_role_tensor_weights, prepare_reduced_role_coefficient_state,
    PreparedCoefficientFunctional, ReducedRoleCoefficientState,
};
pub use setup_index_weight::SetupIndexWeightMle;
