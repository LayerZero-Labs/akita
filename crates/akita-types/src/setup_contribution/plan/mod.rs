//! Setup-contribution planning and evaluation.
//!
//! The public API has two protocol-facing operations:
//! prepare the plan from verifier/prover-local inputs, and evaluate the
//! resulting setup contribution. Internally, the shape is:
//!
//! - `prepare`: static and challenge-dependent plan construction.
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

#[allow(dead_code, unused_macros)]
mod kernels;
mod prepare;
mod scan;
mod setup_index_weight;
#[cfg(test)]
mod test_oracle;
mod types;

pub(crate) use types::SetupContributionGroupPlan;
pub(crate) use types::{get_d_col_range, get_total_d, validate_setup_inputs};
pub use types::{SetupContributionGroupInputs, SetupContributionPlan};

use super::geometry::SetupProjectionGroupGeometry;
use super::{checked_slice, SetupProjectionGeometry};
use crate::dispatch_for_field;
use crate::layout::{CommitmentRingDims, CommittedGroupParams};
use crate::proof::AkitaExpandedSetup;
use crate::{OpeningClaimsLayout, WitnessLayout};
use akita_field::parallel::*;
use akita_field::{AkitaError, CanonicalField, ExtField, FieldCore, MulBase, MulBaseUnreduced};

use kernels::{role_projection, RoleProjection};
