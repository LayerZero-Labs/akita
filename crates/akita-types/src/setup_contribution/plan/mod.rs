//! Setup-contribution planning and evaluation.
//!
//! The public API has three protocol-facing operations:
//! prepare the challenge-free plan data, finish it after challenges are known,
//! and evaluate the resulting setup contribution. Internally, the shape is:
//!
//! - `prepare`: static and challenge-dependent plan construction.
//! - `segments`: the single packed D/B/A segment partition used by every
//!   evaluator.
//! - `setup_index_weight`: the setup-index weight polynomial used by the
//!   recursive stage-3 setup-product sumcheck.
//! - `scan`: direct verifier evaluation of the setup matrix against those same
//!   segment weights.
//!
//! The important invariant is that `setup_index_weight` and `scan` both use the
//! same cached [`GroupSetupSegment`] partition. Direct setup evaluation always projects
//! role dimensions onto one base ring dimension; the ratio-1 case keeps a
//! segment-based hot loop, but it is an optimization inside that single
//! base-dimension scan rather than a separate product definition.

mod kernels;
mod prepare;
mod scan;
mod segments;
mod setup_index_weight;
#[cfg(test)]
mod test_oracle;
mod types;

pub use types::{
    SetupContributionGroupInputs, SetupContributionPlan, SetupContributionStatic,
    SingleGroupSetupContributionLayout,
};
pub(crate) use types::{SetupContributionGroupPlan, SetupContributionGroupStatic};

use super::weights::{setup_e_col_weights, setup_t_col_weights, setup_z_col_weights};
use super::{checked_slice, push_role_boundaries, SetupContributionPlanInputs};
use crate::dispatch_for_field;
use crate::layout::RingMatrixView;
use crate::proof::AkitaExpandedSetup;
use akita_field::parallel::*;
use akita_field::{AkitaError, CanonicalField, ExtField, FieldCore, MulBase, MulBaseUnreduced};

#[cfg(test)]
use kernels::evaluate_weighted_setup_row;
use kernels::{
    base_ring_segment_inner_sum_typed, dispatch_segment_roles,
    identity_base_ring_segment_inner_sum_typed, role_projection, GroupSetupSegment,
    ProjectedRoleWeights, RoleProjection,
};
use segments::{build_packed_segments, validate_group_chunk_layout};
