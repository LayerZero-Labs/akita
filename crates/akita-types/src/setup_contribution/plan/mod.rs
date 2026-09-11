//! Setup-contribution planning and evaluation.
//!
//! The public API has two protocol-facing operations:
//! prepare the plan from verifier/prover-local inputs, and evaluate the
//! resulting setup contribution. Internally, the shape is:
//!
//! - `prepare`: static and challenge-dependent plan construction.
//! - `segments`: the packed D/B/A partition used by the specialized
//!   single-group direct scanner.
//! - `setup_index_weight`: the setup-index weight polynomial used by the
//!   recursive stage-3 setup-product sumcheck.
//! - `scan`: direct verifier evaluation of the setup matrix. Multi-group scans
//!   add every group's weight before evaluating each shared setup ring once.
//!
//! The direct scanner and `setup_index_weight` implement the same additive
//! setup-position weight. Direct setup evaluation always projects role
//! dimensions onto one base ring dimension. A singleton retains the specialized
//! segment hot loop; a multi-group evaluation fuses overlapping group views into
//! one base-dimension scan.

mod kernels;
mod physical_b;
mod prepare;
mod reduced_role;
mod scan;
mod segments;
mod setup_index_weight;
mod structured;
mod structured_reduced;
#[cfg(test)]
mod test_oracle;
mod types;

pub(crate) use types::validate_setup_inputs;
pub(crate) use types::ReducedRoleCoefficientState;
pub(crate) use types::{
    DirectScanState, DirectScanWeights, PhysicalBSetupPlan, ReducedDirectScanWeights,
    SetupContributionGroupPlan, SetupUnitRange,
};
pub(super) use types::{PhysicalBWeightSegment, PhysicalBWeightTerm};
pub use types::{
    PreparedCoefficientFunctional, PreparedRelationAddress, SetupContributionGroupInputs,
    SetupContributionPlan,
};

use super::geometry::SetupProjectionGroupGeometry;
use super::{checked_slice, SetupProjectionGeometry};
use crate::dispatch_for_field;
use crate::layout::{CommittedGroupParams, RingMatrixView};
use crate::proof::AkitaExpandedSetup;
use crate::{OpeningClaimsLayout, RelationAddressGeometry, WitnessLayout};
use akita_error::{checked, AkitaError};
use jolt_field::solinas::parallel::*;
use jolt_field::{CanonicalEncoding, ExtField, Field, MulBaseUnreduced};

fn divide_aligned(
    value: usize,
    divisor: usize,
    context: &'static str,
) -> Result<usize, AkitaError> {
    checked::exact_div(value, divisor).ok_or_else(|| AkitaError::InvalidSetup(context.into()))
}

fn extension_gadget<F, E>(depth: usize, log_basis: u32) -> Vec<E>
where
    F: Field + CanonicalEncoding,
    E: Field + ExtField<F>,
{
    crate::gadget_row_scalars::<F>(depth, log_basis)
        .into_iter()
        .map(|weight| E::one().mul_base(weight))
        .collect()
}

#[cfg(test)]
use kernels::evaluate_weighted_setup_row;
use kernels::{
    base_ring_segment_inner_sum_typed, dispatch_segment_roles,
    for_each_base_ring_segment_weight_typed, role_projection, GroupSetupSegment, RoleProjection,
};
