//! Layout, parameter, opening-point, and proof-size helpers.

pub mod digit_math;
pub mod flat_matrix;
pub mod opening_point;
pub mod params;
pub mod proof_size;
pub mod sis_derivation;

pub use digit_math::gadget_row_scalars;
pub use flat_matrix::{FlatMatrix, RingMatrixView, SetupMatrixPolynomialView};
pub use opening_point::{
    basis_weights, lagrange_weights, monomial_weights, reduce_inner_opening_to_ring_element,
    ring_opening_point_from_field, BasisMode, BlockOrder, RingOpeningPoint,
};
pub use params::{
    setup_polynomial_padded_dims_inner, stage1_accumulator_bound,
    validate_stage1_accumulator_headroom, AjtaiKeyParams, GroupLayout, GroupSpec, LevelParams,
    MRowLayout, SetupPolynomialDimsOuter, Stage1SisExtractionReport,
};
pub use proof_size::{
    derive_chunk_sis_ranks_from_widths, direct_witness_bytes, field_bytes, level_proof_bytes,
    packed_digits_bytes, planned_joint_next_w_len_with_setup_group,
    planned_joint_next_w_len_with_setup_group_tiered, planned_joint_w_ring_with_setup_group,
    planned_joint_w_ring_with_setup_group_tiered, planned_next_w_len,
    planned_next_w_len_with_claims, planned_setup_claim_reduction_rounds, planned_setup_field_len,
    planned_setup_padded_dims, planned_w_ring_element_count,
    planned_w_ring_element_count_with_claims, proof_ring_vec_bytes, recursive_level_proof_bytes,
    sumcheck_rounds, tiered_setup_chunk_index_map, tiered_setup_chunk_opening_point,
    tiered_setup_group_lp, tiered_setup_group_lp_from_dims, untiered_setup_group_lp,
};
pub use sis_derivation::{
    decomp_depths, derived_root_commitment_layout_from_params, level_layout_from_params,
    recursive_level_decomposition_from_root, recursive_level_layout_from_params,
    sis_derived_recursive_params_for_layout, sis_derived_root_params_for_layout,
    sis_secure_level_params, validate_stored_sis_ranks, SisRoleWidths,
};
