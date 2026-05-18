//! Shared Akita protocol data shapes.
//!
//! This crate contains proof objects, commitment/opening wrappers, opening
//! point reductions, per-level parameter shapes, commitment API contracts, and
//! generated schedule/SIS data shared by prover, verifier, and planner code.

pub mod config;
pub mod field_reduction;
pub mod generated;
pub mod layout;
pub mod proof;
pub mod schedule;
pub mod transcript;
#[cfg(feature = "zk")]
pub mod zk;

pub use config::{AjtaiRole, CommitmentEnvelope, DecompositionParams};
pub use field_reduction::{
    check_trace_inner_product, dispatch_trace_inner_product_check, embed_ring_subfield_scalar,
    embed_ring_subfield_vector, embed_subfield, pack_tensor_base_lift_i8_digits, psi_embed,
    recover_ring_subfield_inner_product, trace_h, validate_ring_subfield_role,
    RingSubfieldEncoding, SubfieldParams,
};
pub use layout::{
    basis_weights, decomp_depths, derived_root_commitment_layout_from_params, direct_witness_bytes,
    extension_opening_reduction_proof_bytes, field_bytes, gadget_row_scalars, lagrange_weights,
    level_layout_from_params, level_proof_bytes, monomial_weights, packed_digits_bytes,
    planned_next_w_len, planned_w_ring_element_count, proof_ring_vec_bytes,
    recursive_level_decomposition_from_root, recursive_level_layout_from_params,
    reduce_inner_opening_to_ring_element, ring_opening_point_from_field,
    sis_derived_recursive_params_for_layout, sis_derived_root_params_for_layout,
    sis_secure_level_params, sumcheck_rounds, terminal_level_proof_bytes, AjtaiKeyParams,
    BasisMode, BlockOrder, FlatMatrix, LevelParams, MRowLayout, RingMatrixView, RingOpeningPoint,
    SisModulusFamily, SisRoleWidths,
};
pub use proof::{
    absorb_interstage_claims, combine_polys, eval_poly, linear_combination,
    range_check_eval_from_s, reorder_stage1_coords, stage1_interstage_batch_weights,
    stage1_leaf_coeffs, stage1_stage_count, stage1_tree_product_stage_arities,
    stage1_tree_stage_shapes, validate_stage1_tree_basis,
};
pub use proof::{
    append_batched_commitments_to_transcript, append_claim_incidence_shape_to_transcript,
    append_claim_points_to_transcript, append_claim_values_to_transcript,
    append_prepared_root_opening_point, checked_total_claims, flatten_batched_commitment_rows,
    folded_root_supports_opening_shape, prepare_recursive_opening_point_ext,
    prepare_root_opening_point, prepare_root_opening_point_ext, relation_claim_from_rows,
    relation_claim_from_rows_extension, ring_inner_product_with_extension_weights,
    ring_subfield_packed_extension_opening_point, root_tensor_projection_enabled,
    sample_public_row_coefficients, validate_batched_inputs, verifier_claims_to_incidence,
    AkitaBatchedFoldRoot, AkitaBatchedProof, AkitaBatchedProofShape, AkitaBatchedRootProof,
    AkitaCommitment, AkitaCommitmentHint, AkitaExpandedSetup, AkitaLevelProof, AkitaProofStep,
    AkitaProofStepShape, AkitaSetupSeed, AkitaStage1Proof, AkitaStage1StageProof,
    AkitaStage1StageShape, AkitaStage2Proof, AkitaVerifierSetup, ClaimIncidence,
    ClaimIncidenceLimits, ClaimIncidenceSummary, CommitmentVerifier, CommittedOpenings,
    DirectWitnessProof, DirectWitnessShape, DummyProof, ExtensionOpeningReductionProof,
    ExtensionOpeningReductionShape, FlatDigitBlockIter, FlatDigitBlocks, FlatRingVec,
    IncidenceClaim, LevelProofShape, OpeningPoints, PackedDigits, PreparedRecursiveOpeningPoint,
    PreparedRootOpeningPoint, PublicMatrixSeed, PublicOpeningRow, RingCommitment,
    RingMultiplierOpeningPoint, RingSliceSerializer, TerminalLevelProof, TerminalLevelProofShape,
    VerifierClaims,
};
pub use schedule::{
    detect_field_modulus, exact_planned_level_execution, generated_schedule_lookup_key,
    generated_schedule_plan_from_table, planned_log_basis_at_level_from_schedule,
    planned_schedule_key_from_schedule, r_decomp_levels, root_current_w_len, root_direct_schedule,
    scale_batched_root_layout, schedule_from_plan, schedule_is_root_direct,
    schedule_num_fold_levels, schedule_plan_from_generated_entry, schedule_root_fold_params,
    schedule_root_fold_step, scheduled_fold_execution, scheduled_next_level_params,
    split_batched_root_params, split_batched_root_params_from_schedule_plan,
    validate_opening_points_for_claims, w_ring_element_count, w_ring_element_count_with_counts,
    w_ring_element_count_with_counts_for_layout, AkitaPlannedDirectStep, AkitaPlannedLevel,
    AkitaPlannedLevelExecution, AkitaPlannedState, AkitaPlannedStep, AkitaScheduleInputs,
    AkitaScheduleLookupKey, AkitaSchedulePlan, DirectStep, FoldStep, GeneratedSchedulePlanPolicy,
    Schedule, ScheduleProvider, Step,
};
pub use transcript::AppendToTranscript;
