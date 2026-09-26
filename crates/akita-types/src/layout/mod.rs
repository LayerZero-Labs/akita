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
pub mod digit_range;
pub mod flat_matrix;
pub mod geometry;
pub mod opening_layout;
pub mod opening_point;
pub mod params;
pub mod proof_size;
pub mod relation_address;
pub mod relation_layout;
pub(crate) mod relation_rhs_layout;
pub mod ring_dims;
pub mod setup_envelope;
pub mod setup_prefix_slots;
pub mod setup_projection;
pub mod tail_segments;

pub use digit_math::{gadget_row_scalars, isqrt_ceil};
pub use digit_range::{
    AkitaStage1StageShape, DigitRangePlan, FlatBooleanDomain, PhysicalL2NormProofWireShape,
};
pub use flat_matrix::{FlatMatrix, RingMatrixView};
pub use geometry::{
    BlockGeometry, GadgetDigits, InnerRoleParams, OpenRoleParams, OuterRoleParams, RoleParams,
};
pub use opening_layout::{OpeningClaimsLayout, PolynomialGroupLayout};
pub use opening_point::{
    basis_weights, basis_weights_prefix, checked_opening_source_index, lagrange_weights,
    monomial_weights, opening_domain_len, reduce_inner_opening_to_ring_element,
    ring_opening_point_from_field, witness_commitment_domain_len, BasisMode, RingOpeningPoint,
};
pub use params::{
    opening_d_segment_width, shared_d_digit_log_basis, CommittedGroupParams, GroupOpenPhaseParams,
    GroupOpeningPlan, InnerCommitMatrixParams, OpenCommitMatrixParams, OpeningMethod,
    OuterCommitMatrixParams, PrecommittedGroupAdmissionPolicy, SisModulusProfileId,
};
pub use proof_size::{
    extension_opening_reduction_level_bytes, extension_opening_reduction_proof_bytes, field_bytes,
    native_terminal_response_max_bytes, native_terminal_response_planner_bytes,
    padded_boolean_opening_vars, sumcheck_rounds, terminal_response_bytes,
    try_extension_opening_reduction_level_bytes, EXTENSION_OPENING_REDUCTION_DEGREE,
    SETUP_SUMCHECK_DEGREE,
};
pub use relation_address::{CompressionRelationAddressGeometry, RelationAddressGeometry};
pub use relation_layout::{
    RelationGroupRows, RelationRhsLayout, RelationRowFamily, RelationRowGeometry,
    RelationWitnessGeometry,
};
pub use ring_dims::{
    validate_role_dims, validate_schedule_ring_dims, CommitmentRingDims, RingRole, MAX_FOLD_LEVELS,
    MIN_A_ROLE_FOLD_CHALLENGE_RING_D, SUPPORTED_CHALLENGE_RING_DIMS,
    SUPPORTED_COMMITMENT_RING_DIMS,
};
pub use setup_envelope::{
    accumulate_matrix_field_elements_for_level, accumulate_terminal_matrix_field_elements,
    commit_only_setup_field_elements, commitment_execution_setup_field_elements,
    setup_matrix_capacity_for_schedule, setup_matrix_field_elements_for_schedule,
    setup_prefix_slot_field_elements, verifier_setup_matrix_capacity_for_schedule,
    CommitmentSetupMatrixShape, SetupMatrixCapacity,
};
pub use setup_prefix_slots::{
    active_setup_field_len, padded_setup_prefix_len, scheduled_setup_prefix,
    setup_prefix_precommitted_params, suffix_opening_layout, validate_setup_prefix_domain,
    SetupPrefixSlotId, SETUP_PREFIX_CONTENT_TAG,
};
pub use setup_projection::SetupProjectionGeometry;
pub use tail_segments::{
    terminal_response_upper_bound_bytes, TailSegmentGroupLayout, TailSegmentLayout,
    TerminalResponseShape,
};
