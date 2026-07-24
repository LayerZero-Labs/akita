//! Setup-contribution planning and evaluation.
//!
//! The public API has two protocol-facing operations:
//! prepare the plan from verifier/prover-local inputs, and evaluate the
//! resulting setup contribution. Internally, the shape is:
//!
//! - `prepare`: static and challenge-dependent plan construction.
//! - `segments`: a path-local packed D/B/A partition compiled by the direct
//!   scanner.
//! - `setup_index_weight`: the setup-index weight polynomial used by the
//!   recursive stage-3 setup-product sumcheck.
//! - `scan`: direct verifier evaluation of the setup matrix. Multi-group scans
//!   add every group's weight before evaluating each shared setup ring once.
//!
//! The direct scanner and `setup_index_weight` implement the same additive
//! setup-position weight. Direct setup evaluation always projects role
//! dimensions onto one base ring dimension, and overlapping groups are fused
//! into one base-dimension scan.

mod kernels;
mod prepare;
mod scan;
mod segments;
mod setup_index_weight;
mod structured;
#[cfg(test)]
mod test_oracle;
mod types;

pub use types::SetupContributionPlan;
pub(crate) use types::{d_active_cols, get_d_col_range, get_total_d, validate_setup_inputs};
pub(crate) use types::{SetupContributionGroupPlan, SetupContributionSpan};

use super::geometry::SetupProjectionGroupGeometry;
use super::SetupProjectionGeometry;
use crate::dispatch_for_field;
use crate::layout::{CommitmentRingDims, CommittedGroupParams};
use crate::proof::AkitaExpandedSetup;
use crate::{OpeningClaimsLayout, WitnessLayout};
use akita_field::parallel::*;
use akita_field::{AkitaError, CanonicalField, ExtField, FieldCore, MulBase, MulBaseUnreduced};

use kernels::{
    dispatch_segment_roles, for_each_base_ring_segment_weight_typed, role_projection,
    GroupSetupSegment, RoleProjection,
};
