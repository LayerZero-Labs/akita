//! Layout, parameter, opening-point, and proof-size helpers.
//!
//! Pure data and pure verifier-reachable helpers only. The recursion layout is
//! owned by the schedule: runtime expands catalog rows through
//! `akita_schedules::schedule_from_entry`, while the offline planner builds new
//! candidates with `akita_planner::find_schedule`. Prover/verifier
//! read those params directly.
//! This module retains the layout glue the replay path reaches through
//! `CommitmentConfig`.

pub mod digit_math;
pub mod flat_matrix;
pub mod geometry;
pub mod opening_point;
pub mod params;
pub mod proof_size;
pub mod ring_dims;

pub use digit_math::{gadget_row_scalars, isqrt_ceil};
pub use flat_matrix::{FlatMatrix, RingMatrixView};
pub use geometry::{BlockGeometry, GadgetDigits};
pub use opening_point::{
    basis_weights, basis_weights_prefix, block_rings_at_opening, checked_opening_source_index,
    lagrange_weights, monomial_weights, opening_domain_len, reduce_inner_opening_to_ring_element,
    ring_opening_point_from_field, witness_commitment_domain_len, BasisMode, RingOpeningPoint,
};
pub use params::{
    opening_d_segment_width, shared_d_digit_log_basis, CommittedGroupParams, GroupOpeningPlan,
    InnerCommitMatrixParams, LevelParamsLike, OpenCommitMatrixParams, OpeningFamily, OpeningMethod,
    OuterCommitMatrixParams, PrecommittedGroupAdmissionPolicy, PrecommittedLevelParams,
    SisModulusProfileId,
};
pub use proof_size::{
    extension_opening_reduction_level_bytes, extension_opening_reduction_proof_bytes, field_bytes,
    packed_digits_bytes, padded_boolean_opening_vars, proof_ring_vec_bytes, sumcheck_rounds,
    terminal_response_bytes, terminal_response_planner_bytes,
    try_extension_opening_reduction_level_bytes,
};
pub use ring_dims::{
    validate_role_dims, validate_schedule_ring_dims, CommitmentRingDims, RingRole, MAX_FOLD_LEVELS,
    MIN_A_ROLE_FOLD_CHALLENGE_RING_D, SUPPORTED_CHALLENGE_RING_DIMS,
    SUPPORTED_COMMITMENT_RING_DIMS,
};
