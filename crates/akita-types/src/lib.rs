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

pub use config::{AjtaiRole, CommitmentEnvelope, DecompositionParams};
pub use field_reduction::{psi_pack, trace_h, SubfieldParams};
pub use layout::{
    basis_weights, decomp_depths, derived_root_commitment_layout_from_params, direct_witness_bytes,
    field_bytes, gadget_row_scalars, lagrange_weights, level_layout_from_params, level_proof_bytes,
    monomial_weights, packed_digits_bytes, planned_next_w_len, planned_w_ring_element_count,
    proof_ring_vec_bytes, recursive_level_decomposition_from_root,
    recursive_level_layout_from_params, recursive_level_proof_bytes,
    reduce_inner_opening_to_ring_element, ring_opening_point_from_field,
    sis_derived_recursive_params_for_layout, sis_derived_root_params_for_layout,
    sis_secure_level_params, sumcheck_rounds, AjtaiKeyParams, BasisMode, BlockOrder, FlatMatrix,
    LevelParams, RingMatrixView, RingOpeningPoint, SisRoleWidths,
};
pub use proof::{
    absorb_interstage_claims, combine_polys, eval_poly, linear_combination,
    range_check_eval_from_s, reorder_stage1_coords, stage1_interstage_batch_weights,
    stage1_leaf_coeffs, stage1_stage_count, stage1_tree_product_stage_arities,
    stage1_tree_stage_shapes, validate_stage1_tree_basis,
};
pub use proof::{
    append_batched_commitments_to_transcript, append_claim_incidence_shape_to_transcript,
    append_prepared_root_opening_point, checked_total_claims, checked_total_groups,
    flatten_batched_commitment_rows, prepare_root_opening_point, relation_claim_from_rows,
    relation_claim_from_rows_extension, validate_batched_inputs, verifier_claims_to_incidence,
    AkitaBatchedFoldRoot, AkitaBatchedProof, AkitaBatchedProofShape, AkitaBatchedRootProof,
    AkitaCommitment, AkitaCommitmentHint, AkitaExpandedSetup, AkitaLevelProof, AkitaOpeningClaim,
    AkitaOpeningPoint, AkitaProofStep, AkitaProofStepShape, AkitaSetupSeed, AkitaStage1Proof,
    AkitaStage1StageProof, AkitaStage1StageShape, AkitaStage2Proof, AkitaVerifierSetup,
    ClaimIncidence, ClaimIncidenceLimits, ClaimIncidenceSummary, CommitmentVerifier,
    CommittedOpenings, DirectWitnessProof, DirectWitnessShape, DummyProof, FlatDigitBlockIter,
    FlatDigitBlocks, FlatRingVec, IncidenceClaim, IncidenceGroup, LevelProofShape, OpeningPoints,
    PackedDigits, PreparedRootOpeningPoint, PublicMatrixSeed, RingCommitment, RingSliceSerializer,
    VerifierClaims,
};
pub use schedule::{
    checked_num_claims_from_group_sizes, detect_field_modulus, exact_planned_level_execution,
    generated_schedule_lookup_key, generated_schedule_plan_from_table,
    planned_log_basis_at_level_from_schedule, planned_schedule_key_from_schedule, r_decomp_levels,
    root_current_w_len, scale_batched_root_layout, schedule_from_plan, schedule_is_root_direct,
    schedule_num_fold_levels, schedule_plan_from_generated_entry, scheduled_fold_execution,
    scheduled_next_level_params, split_batched_root_params,
    split_batched_root_params_from_schedule_plan, validate_opening_points_for_claims,
    w_ring_element_count, w_ring_element_count_with_batch_summary,
    w_ring_element_count_with_claim_groups, w_ring_element_count_with_num_claims,
    AkitaPlannedDirectStep, AkitaPlannedLevel, AkitaPlannedLevelExecution, AkitaPlannedState,
    AkitaPlannedStep, AkitaRootBatchSummary, AkitaScheduleInputs, AkitaScheduleLookupKey,
    AkitaSchedulePlan, DirectStep, FoldStep, Schedule, ScheduleProvider, Step, WitnessShape,
};
pub use transcript::AppendToTranscript;
