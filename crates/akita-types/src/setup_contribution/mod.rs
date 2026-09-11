//! Shared setup-contribution planning for prover and verifier.
//!
//! This module owns challenge-free geometry (`geometry.rs`), pure layout/weight
//! derivation for the stage-3 setup product, and evaluation planning. The
//! prover consumes the materialized setup-index weight vector: one scalar weight
//! per packed setup position. The recursive stage-3 verifier evaluates the
//! multilinear extension of that weight vector directly at the setup-index
//! challenge point, while the direct verifier scans the packed setup with the
//! same segment partition.

use crate::{CommittedGroupParams, OpeningClaimsLayout};
use akita_error::{checked, AkitaError};
use jolt_field::{CanonicalEncoding, Field};

mod geometry;
mod plan;
#[allow(dead_code)]
#[cfg(test)]
mod test_oracle_weights;

#[cfg(test)]
mod tests;

pub(crate) use geometry::SetupProjectionGroupGeometry;
pub use geometry::{ensure_setup_envelope, SetupProjectionGeometry};
#[cfg(test)]
pub(crate) use plan::validate_setup_inputs;
pub use plan::{
    PreparedCoefficientFunctional, PreparedRelationAddress, SetupContributionGroupInputs,
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
    Some(crate::gadget_row_scalars::<F>(
        max_depth,
        level_params.open().digits.log_basis,
    ))
}

#[inline(always)]
pub(crate) fn checked_slice<'a, T>(
    slice: &'a [T],
    start: usize,
    len: usize,
    context: &'static str,
) -> Result<&'a [T], AkitaError> {
    let range = checked::range(start, len)
        .ok_or_else(|| AkitaError::InvalidSetup(format!("{context} overflow")))?;
    slice.get(range).ok_or(AkitaError::InvalidProof)
}
