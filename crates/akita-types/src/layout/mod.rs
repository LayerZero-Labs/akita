//! Layout, parameter, opening-point, and proof-size helpers.

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
pub use params::{AjtaiKeyParams, LevelParams};
pub use proof_size::{
    direct_witness_bytes, field_bytes, level_proof_bytes, packed_digits_bytes, planned_next_w_len,
    planned_w_ring_element_count, proof_ring_vec_bytes, recursive_level_proof_bytes,
    sumcheck_rounds,
};
pub use sis_derivation::{
    decomp_depths, derived_root_commitment_layout_from_params, level_layout_from_params,
    recursive_level_decomposition_from_root, recursive_level_layout_from_params,
    sis_derived_recursive_params_for_layout, sis_derived_root_params_for_layout,
    sis_secure_level_params, SisRoleWidths,
};
