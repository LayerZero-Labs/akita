//! Shared setup-contribution planning for prover and verifier.
//!
//! This module owns challenge-free geometry (`geometry.rs`) and the pure
//! layout/weight derivation for the stage-3 setup product. The prover consumes
//! the materialized setup-index weight vector: one scalar weight per packed
//! setup position. Verifier-only evaluation of the same plan (the direct
//! setup scan, closed-form structured groups, and the setup-index weight MLE
//! at the stage-3 challenge point) lives in `akita-verifier`.

use crate::{CommittedGroupParams, OpeningClaimsLayout};
use akita_error::{checked, AkitaError};
use jolt_field::{CanonicalEncoding, Field};

mod geometry;
mod plan;

#[cfg(test)]
mod tests;

pub(crate) use geometry::SetupProjectionGroupGeometry;
pub use geometry::{ensure_setup_envelope, SetupProjectionGeometry};
pub use plan::{
    factor_aligned_role_tensors, project_role_tensors, role_projection_evaluation,
    role_tensors_are_aligned, PhysicalBSetupPlan, PhysicalBWeightSegment, PhysicalBWeightTerm,
    PreparedRelationAddress, SetupContributionGroupInputs, SetupContributionGroupPlan,
    SetupContributionPlan, SetupUnitRange,
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
    Some(crate::gadget_row_scalars::<F>(
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
    slice.get(range).ok_or(AkitaError::InvalidProof)
}
