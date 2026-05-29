//! Layout, parameter, opening-point, and proof-size helpers.
//!
//! Pure data and pure verifier-reachable helpers only. Search and
//! SIS-derivation loops (`sis_secure_level_params`,
//! `sis_derived_*_for_layout`, `derived_root_commitment_layout_from_params`)
//! live in `akita_planner::derivation`; the digit-math search loop
//! (`optimal_m_r_split` callers, the (m, r) sweep) lives in
//! `akita_planner::schedule_params`. This module retains the layout glue
//! the verifier replay path reaches through `CommitmentConfig`
//! materializers and `akita_planner::schedule_plan_from_table`.

pub mod digit_math;
pub mod flat_matrix;
pub mod opening_point;
pub mod params;
pub mod proof_size;
pub mod sis_derivation;

pub use digit_math::gadget_row_scalars;
pub use flat_matrix::{FlatMatrix, RingMatrixView};
pub use opening_point::{
    basis_weights, lagrange_weights, monomial_weights, reduce_inner_opening_to_ring_element,
    ring_opening_point_from_field, BasisMode, BlockOrder, RingOpeningPoint,
};
pub use params::{AjtaiKeyParams, LevelParams, MRowLayout, SisModulusFamily};
pub use proof_size::{
    direct_witness_bytes, extension_opening_reduction_proof_bytes, field_bytes,
    packed_digits_bytes, planned_next_w_len, planned_w_ring_element_count, proof_ring_vec_bytes,
    root_extension_opening_partials, sumcheck_rounds,
};
pub use sis_derivation::{
    decomp_depths, level_layout_from_params, recursive_level_layout_from_params,
};
