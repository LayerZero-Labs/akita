//! Shared Akita protocol data shapes.
//!
//! This crate contains proof objects, commitment/opening wrappers, opening
//! point reductions, per-level parameter shapes, commitment API contracts, and
//! generated schedule/SIS data shared by prover, verifier, and planner code.

pub mod batch;
pub mod commitment;
pub mod config;
pub mod digit_math;
pub mod flat_matrix;
pub mod generated;
pub mod opening_point;
pub mod params;
pub mod proof;
pub mod proof_size;
pub mod relation;
pub mod schedule;
pub mod scheme;
pub mod setup;
pub mod sis_derivation;
pub mod stage1;
pub mod transcript_append;

pub use batch::{
    append_batch_shape_to_transcript, append_batched_commitments_to_transcript,
    append_prepared_root_opening_point, checked_total_claims, checked_total_groups,
    flatten_batched_commitment_rows, prepare_root_opening_point, validate_batched_inputs,
    MultiPointBatchShape, PreparedRootOpeningPoint,
};
pub use commitment::{
    AkitaCommitment, AkitaOpeningClaim, AkitaOpeningPoint, DummyProof, RingCommitment,
};
pub use config::{AjtaiRole, CommitmentEnvelope, DecompositionParams};
pub use digit_math::gadget_row_scalars;
pub use flat_matrix::{FlatMatrix, RingMatrixView};
pub use opening_point::{
    basis_weights, lagrange_weights, monomial_weights, reduce_inner_opening_to_ring_element,
    ring_opening_point_from_field, BasisMode, BlockOrder, RingOpeningPoint,
};
pub use params::{AjtaiKeyParams, LevelParams};
pub use proof::{
    AkitaBatchedFoldRoot, AkitaBatchedProof, AkitaBatchedProofShape, AkitaBatchedRootProof,
    AkitaCommitmentHint, AkitaLevelProof, AkitaProofStep, AkitaProofStepShape, AkitaStage1Proof,
    AkitaStage1StageProof, AkitaStage1StageShape, AkitaStage2Proof, DirectWitnessProof,
    DirectWitnessShape, FlatDigitBlockIter, FlatDigitBlocks, FlatRingVec, LevelProofShape,
    PackedDigits, RingSliceSerializer,
};
pub use proof_size::{
    direct_witness_bytes, field_bytes, level_proof_bytes, packed_digits_bytes, planned_next_w_len,
    planned_w_ring_element_count, proof_ring_vec_bytes, recursive_level_proof_bytes,
    sumcheck_rounds,
};
pub use relation::relation_claim_from_rows;
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
pub use scheme::{CommitmentVerifier, CommittedOpenings, OpeningPoints, VerifierClaims};
pub use setup::{AkitaExpandedSetup, AkitaSetupSeed, AkitaVerifierSetup, PublicMatrixSeed};
pub use sis_derivation::{
    decomp_depths, derived_root_commitment_layout_from_params, level_layout_from_params,
    recursive_level_decomposition_from_root, recursive_level_layout_from_params,
    sis_derived_recursive_params_for_layout, sis_derived_root_params_for_layout,
    sis_secure_level_params, SisRoleWidths,
};
pub use stage1::{
    absorb_interstage_claims, combine_polys, eval_poly, linear_combination,
    range_check_eval_from_s, reorder_stage1_coords, stage1_interstage_batch_weights,
    stage1_leaf_coeffs, stage1_stage_count, stage1_tree_product_stage_arities,
    stage1_tree_stage_shapes, validate_stage1_tree_basis,
};
pub use transcript_append::AppendToTranscript;
