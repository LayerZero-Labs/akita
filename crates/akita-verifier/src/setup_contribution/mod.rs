//! Verifier-only evaluation of the shared setup-contribution plan.
//!
//! `akita-types` owns the functional-free [`akita_types::SetupContributionPlan`]
//! (geometry, canonical relation-column tensors, and the dense setup-index
//! weights the prover consumes). This module owns every verifier-only
//! evaluation of that plan:
//!
//! - `direct_scan`: [`DirectScan`], the owned coefficient-functional state of
//!   one direct scan (per-group column weights and packed segment partitions).
//!   Only the direct verifier path builds it; the deferred path evaluates
//!   structured groups in closed form from the plan alone.
//! - `segments`: the packed D/B/A partition used by the specialized
//!   single-group direct scanner.
//! - `scan`: direct evaluation of the setup matrix. Multi-group scans add
//!   every group's weight before evaluating each shared setup ring once.
//! - `structured` / `structured_reduced`: closed-form structured E/T/Z
//!   contractions for the deferred, cached-lifted, and reduced paths.
//! - [`SetupIndexWeightMle`] and [`PreparedCoefficientFunctional`] are shared
//!   with external accumulation through `akita-types`.
//!
//! The direct scanner and the setup-index weight implement the same additive
//! setup-position weight. Direct setup evaluation always projects role
//! dimensions onto one base ring dimension. A singleton retains the specialized
//! segment hot loop; a multi-group evaluation fuses overlapping group views into
//! one base-dimension scan.

mod direct_scan;
mod kernels;
mod scan;
mod segments;
mod structured;
mod structured_reduced;
#[cfg(test)]
mod test_oracle;
#[allow(dead_code)]
#[cfg(test)]
mod test_oracle_weights;
#[cfg(test)]
mod tests;

pub use akita_types::{PreparedCoefficientFunctional, SetupIndexWeightMle};
pub use direct_scan::DirectScan;
pub(crate) use direct_scan::{
    DirectScanMode, DirectScanWeights, GroupScanPartition, ReducedDirectScanWeights,
};
pub(crate) use structured::evaluate_structured_group;

use akita_error::{checked, AkitaError};
use akita_params::{dispatch_for_field, RingMatrixView, SetupProjectionGeometry};
use akita_types::{
    checked_slice, AkitaExpandedSetup, PhysicalBWeightSegment, PhysicalBWeightTerm,
    SetupContributionGroupPlan, SetupContributionPlan,
};
use jolt_field::solinas::parallel::*;
use jolt_field::{CanonicalEncoding, ExtField, Field, MulBaseUnreduced};

fn extension_gadget<F, E>(depth: usize, log_basis: u32) -> Vec<E>
where
    F: Field + CanonicalEncoding,
    E: Field + ExtField<F>,
{
    akita_params::gadget_row_scalars::<F>(depth, log_basis)
        .into_iter()
        .map(|weight| E::one().mul_base(weight))
        .collect()
}

use akita_types::{
    materialize_reduced_role_tensor_weights, prepare_reduced_role_coefficient_state,
    ReducedRoleCoefficientState,
};
#[cfg(test)]
use kernels::evaluate_weighted_setup_row;
use kernels::{
    base_ring_segment_inner_sum_typed, dispatch_segment_roles,
    for_each_base_ring_segment_weight_typed, role_projection, GroupSetupSegment, RoleProjection,
};
use segments::scan_partition;

#[cfg(feature = "external-setup-accumulation")]
pub(crate) use direct_scan::materialize_reduced_direct_scan_weights;
#[cfg(feature = "external-setup-accumulation")]
pub(crate) use structured_reduced::evaluate_reduced_structured_group;
