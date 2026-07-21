//! Layout, parameter, opening-point, and proof-size helpers.
//!
//! Pure data and pure verifier-reachable helpers only. The recursion layout is
//! owned by the schedule: the planner builds each fold level's `LevelParams`
//! (`akita_planner::schedule_from_entry` / `find_group_batch_schedule`, using the
//! digit-math `optimal_block_geometry_split` sweep), and prover/verifier read those params
//! directly. This module retains the layout glue the replay path reaches
//! through `CommitmentConfig`.

pub mod digit_math;
pub mod flat_matrix;
pub mod opening_point;
pub mod params;
pub mod proof_size;
pub mod ring_dims;

pub use digit_math::{gadget_row_scalars, isqrt_ceil};
pub use flat_matrix::{FlatMatrix, RingMatrixView};
pub use opening_point::{
    basis_weights, basis_weights_prefix, block_rings_at_opening, checked_opening_source_index,
    lagrange_weights, monomial_weights, opening_domain_len, reduce_inner_opening_to_ring_element,
    ring_opening_point_from_field, BasisMode, RingOpeningPoint,
};
pub use params::{
    shared_d_digit_log_basis, AjtaiKeyParams, LevelParams, LevelParamsLike,
    PrecommittedLevelParams, RelationMatrixRowLayout, SisModulusProfileId,
};
pub use proof_size::{
    extension_opening_reduction_level_bytes, extension_opening_reduction_proof_bytes, field_bytes,
    packed_digits_bytes, padded_boolean_opening_vars, planned_next_w_len,
    planned_w_ring_element_count, proof_ring_vec_bytes, segment_typed_witness_bytes,
    sumcheck_rounds,
};
pub use ring_dims::{
    validate_role_dims, validate_schedule_ring_dims, CommitmentRingDims, RingRole, MAX_FOLD_LEVELS,
    MIN_A_ROLE_FOLD_CHALLENGE_RING_D, SUPPORTED_CHALLENGE_RING_DIMS, SUPPORTED_RING_DIMS,
};
