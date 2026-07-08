mod segment;
mod types;

pub use types::{SetupContributionGroupInputs, SetupContributionPlan, SetupContributionStatic};
pub(crate) use types::{SetupContributionGroupPlan, SetupContributionGroupStatic};

use super::weights::{setup_e_col_weights, setup_t_col_weights, setup_z_col_weights};
use super::{
    checked_add, checked_mul, checked_slice, push_role_boundaries, SetupContributionPlanInputs,
};
use crate::dispatch_for_field;
use crate::layout::{RelationMatrixRowLayout, RingMatrixView};
use crate::proof::AkitaExpandedSetup;
use akita_algebra::ring::eval_ring_at_pows_fast;
use akita_field::parallel::*;
use akita_field::{AkitaError, CanonicalField, ExtField, FieldCore, MulBase, MulBaseUnreduced};

#[cfg(test)]
use segment::evaluate_weighted_setup_row;
use segment::{
    alpha_chunk_scales, group_bar_omega_segment_eval, packed_group_slice_inner_sum_typed,
    packed_uniform_group_slice_inner_sum_typed, push_group_d_boundaries, scaled_row_weights,
    validate_group_chunk_layout, validate_typed_packed_scan_access, AlphaChunkScales,
    GroupSetupSegment,
};

include!("inherent.rs");
