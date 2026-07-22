//! Setup-contribution planning and evaluation.
//!
//! The public API has two protocol-facing operations:
//! prepare the plan from verifier/prover-local inputs, and evaluate the
//! resulting setup contribution. Internally, the shape is:
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

pub(crate) use types::SetupContributionGroupPlan;
pub(crate) use types::{get_d_col_range, get_total_d, validate_setup_inputs};
pub use types::{SetupContributionGroupInputs, SetupContributionPlan};

use super::geometry::SetupProjectionGroupGeometry;
use super::weights::{setup_e_col_weights, setup_t_col_weights, setup_z_col_weights};
use super::{checked_slice, SetupProjectionGeometry};
use crate::dispatch_for_field;
use crate::layout::{CommitmentRingDims, CommittedGroupParams, RingMatrixView};
use crate::proof::AkitaExpandedSetup;
use crate::{OpeningClaimsLayout, WitnessLayout};
use akita_field::parallel::*;
use akita_field::{AkitaError, CanonicalField, ExtField, FieldCore, MulBase, MulBaseUnreduced};

#[cfg(test)]
use kernels::evaluate_weighted_setup_row;
use kernels::{
    base_ring_segment_inner_sum_typed, dispatch_segment_roles, role_projection, GroupSetupSegment,
    RoleProjection,
};
