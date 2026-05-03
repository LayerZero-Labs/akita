//! Shared Akita protocol data shapes.
//!
//! This crate contains proof objects, commitment/opening wrappers, opening
//! point reductions, per-level parameter shapes, and generated schedule/SIS
//! data shared by prover, verifier, and planner code.

pub mod commitment;
pub mod digit_math;
pub mod flat_matrix;
pub mod generated;
pub mod opening_point;
pub mod params;
pub mod proof;
pub mod schedule;
pub mod setup;
pub mod stage1;
pub mod transcript_append;

pub use commitment::{
    DummyProof, HachiCommitment, HachiOpeningClaim, HachiOpeningPoint, RingCommitment,
};
pub use digit_math::gadget_row_scalars;
pub use flat_matrix::{FlatMatrix, RingMatrixView};
pub use opening_point::{
    basis_weights, lagrange_weights, monomial_weights, reduce_inner_opening_to_ring_element,
    ring_opening_point_from_field, BasisMode, BlockOrder, RingOpeningPoint,
};
pub use params::{AjtaiKeyParams, LevelParams};
pub use proof::{
    DirectWitnessProof, DirectWitnessShape, FlatDigitBlockIter, FlatDigitBlocks, FlatRingVec,
    HachiBatchedFoldRoot, HachiBatchedProof, HachiBatchedProofShape, HachiBatchedRootProof,
    HachiCommitmentHint, HachiLevelProof, HachiProofStep, HachiProofStepShape, HachiStage1Proof,
    HachiStage1StageProof, HachiStage1StageShape, HachiStage2Proof, LevelProofShape, PackedDigits,
    RingSliceSerializer,
};
pub use schedule::{
    checked_num_claims_from_group_sizes, detect_field_modulus, generated_schedule_lookup_key,
    r_decomp_levels, validate_opening_points_for_claims, w_ring_element_count,
    w_ring_element_count_with_batch_summary, w_ring_element_count_with_claim_groups,
    w_ring_element_count_with_num_claims, DirectStep, FoldStep, HachiPlannedDirectStep,
    HachiPlannedLevel, HachiPlannedLevelExecution, HachiPlannedState, HachiPlannedStep,
    HachiRootBatchSummary, HachiScheduleInputs, HachiScheduleLookupKey, HachiSchedulePlan,
    Schedule, ScheduleProvider, Step, WitnessShape,
};
pub use setup::{HachiExpandedSetup, HachiSetupSeed, HachiVerifierSetup, PublicMatrixSeed};
pub use stage1::{
    absorb_interstage_claims, combine_polys, eval_poly, linear_combination,
    range_check_eval_from_s, reorder_stage1_coords, stage1_interstage_batch_weights,
    stage1_leaf_coeffs, stage1_stage_count, stage1_tree_product_stage_arities,
    stage1_tree_stage_shapes, validate_stage1_tree_basis,
};
pub use transcript_append::AppendToTranscript;
