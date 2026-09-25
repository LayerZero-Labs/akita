//! Functional-free setup-contribution planning shared by prover and verifier.
//!
//! - `prepare`: functional-free plan construction from the relation address,
//!   including each group's canonical D/B/A relation-column tensors.
//! - `physical_b`: the physical B matrix and its logical sliced image.
//! - `setup_index_weight`: the dense setup-index weight vector the prover's
//!   stage-3 setup-product sumcheck consumes, plus the role-tensor helpers the
//!   verifier's setup evaluation shares with it.
//!
//! Verifier-only evaluation of the plan (direct scans, closed-form structured
//! groups, and the setup-index weight MLE) lives in `akita-verifier`.

mod physical_b;
mod prepare;
mod setup_index_weight;
mod types;

pub use setup_index_weight::{
    factor_aligned_role_tensors, project_role_tensors, role_projection_evaluation,
    role_tensors_are_aligned,
};
use types::validate_setup_inputs;
pub use types::{
    PhysicalBSetupPlan, PhysicalBWeightSegment, PhysicalBWeightTerm, PreparedRelationAddress,
    SetupContributionGroupInputs, SetupContributionGroupPlan, SetupContributionPlan,
};

use super::geometry::SetupProjectionGroupGeometry;
use super::{checked_slice, SetupProjectionGeometry};
use crate::layout::CommittedGroupParams;
use crate::{OpeningClaimsLayout, RelationAddressGeometry, WitnessLayout};
use akita_error::{checked, AkitaError};
use jolt_field::solinas::parallel::*;
use jolt_field::{CanonicalEncoding, ExtField, Field};
