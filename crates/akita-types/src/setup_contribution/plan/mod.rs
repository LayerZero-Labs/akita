//! Setup-contribution planning and evaluation.
//!
//! The public API has two protocol-facing operations:
//! prepare the plan from verifier/prover-local inputs, and evaluate the
//! resulting setup contribution. Internally, the shape is:
//!
//! - `prepare`: functional-free plan construction from the relation address.
//! - `direct_scan`: [`DirectScan`], the owned coefficient-functional state of
//!   one direct scan (per-group column weights and packed segment partitions).
//!   Only the direct verifier path builds it; the deferred path evaluates
//!   structured groups in closed form from the plan alone.
//! - `segments`: the packed D/B/A partition used by the specialized
//!   single-group direct scanner.
//! - `setup_index_weight`: the setup-index weight used by the recursive
//!   stage-3 setup-product sumcheck. The prover materializes it densely from
//!   the plan; the verifier evaluates it through [`SetupIndexWeightMle`],
//!   whose paired-equality tensors the plan does not store.
//! - `scan`: direct verifier evaluation of the setup matrix. Multi-group scans
//!   add every group's weight before evaluating each shared setup ring once.
//!
//! The direct scanner and `setup_index_weight` implement the same additive
//! setup-position weight. Direct setup evaluation always projects role
//! dimensions onto one base ring dimension. A singleton retains the specialized
//! segment hot loop; a multi-group evaluation fuses overlapping group views into
//! one base-dimension scan.

mod direct_scan;
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

pub use direct_scan::DirectScan;
pub(crate) use direct_scan::{DirectScanMode, GroupScanPartition};
pub use setup_index_weight::SetupIndexWeightMle;
pub use setup_index_weight::{
    factor_aligned_role_tensors, project_role_tensors, role_projection_evaluation,
    role_tensors_are_aligned,
};
pub(crate) use types::validate_setup_inputs;
pub(crate) use types::ReducedRoleCoefficientState;
pub(crate) use types::{DirectScanWeights, ReducedDirectScanWeights};
pub use types::{
    PhysicalBSetupPlan, PhysicalBWeightSegment, PhysicalBWeightTerm, PreparedCoefficientFunctional,
    PreparedRelationAddress, SetupContributionGroupInputs, SetupContributionGroupPlan,
    SetupContributionPlan, SetupUnitRange,
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
