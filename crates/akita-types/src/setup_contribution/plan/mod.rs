//! Setup-contribution planning and evaluation.
//!
//! The public API has three protocol-facing operations:
//! prepare the challenge-free plan data, finish it after challenges are known,
//! and evaluate the resulting setup contribution. Internally, the shape is:
//!
//! - `prepare`: static and challenge-dependent plan construction.
//! - `segments`: the single packed D/B/A segment partition used by every
//!   evaluator.
//! - `omega`: the dense setup-weight vector used by the recursive stage-3
//!   setup-product sumcheck.
//! - `scan`: direct verifier evaluation of the setup matrix against those same
//!   segment weights.
//!
//! The important invariant is that `omega` and `scan` both walk the cached
//! [`GroupSetupSegment`] partition. Uniform, divisible, and mixed-ring scans are
//! performance specializations of that same product, not separate definitions.

mod kernels;
mod omega;
mod prepare;
mod scan;
mod segments;
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
use akita_algebra::ring::eval_ring_at_pows_fast;
use akita_field::parallel::*;
use akita_field::{AkitaError, CanonicalField, ExtField, FieldCore, MulBase, MulBaseUnreduced};

#[cfg(test)]
use kernels::evaluate_weighted_setup_row;
use kernels::{
    alpha_chunk_scales, dispatch_segment_roles, group_bar_omega_segment_eval,
    packed_group_slice_inner_sum_typed, packed_uniform_group_slice_inner_sum_typed,
    scaled_row_weights, validate_typed_packed_scan_access, AlphaChunkScales, GroupSetupSegment,
};
use segments::{build_packed_segments, validate_group_chunk_layout};
