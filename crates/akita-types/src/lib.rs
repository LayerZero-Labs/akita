//! Shared Akita protocol data shapes.
//!
//! This crate contains proof objects, commitment/opening wrappers, opening
//! point reductions, per-level parameter shapes, commitment API contracts, and
//! generated schedule/SIS data shared by prover, verifier, and planner code.

pub mod config;
pub(crate) mod descriptor_bytes;
pub mod field_reduction;
pub mod generated;
pub mod instance_descriptor;
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
pub use instance_descriptor::{
    digest_effective_schedule, digest_incidence, digest_level_params, digest_serializable,
    setup_seed_digest, AkitaInstanceDescriptor, AlgebraSection, CallSection, PlanSection,
    ProtocolFeatureSet, SetupSection,
};
pub use layout::{
    basis_weights, decomp_depths, direct_witness_bytes, extension_opening_reduction_proof_bytes,
    field_bytes, gadget_row_scalars, lagrange_weights, level_layout_from_params, level_proof_bytes,
    monomial_weights, packed_digits_bytes, planned_next_w_len, planned_w_ring_element_count,
    proof_ring_vec_bytes, recursive_level_layout_from_params, reduce_inner_opening_to_ring_element,
    ring_opening_point_from_field, root_extension_opening_partials, sumcheck_rounds,
    terminal_level_proof_bytes, AjtaiKeyParams, BasisMode, BlockOrder, FlatMatrix, LevelParams,
    MRowLayout, RingMatrixView, RingOpeningPoint, SisModulusFamily,
};
#[cfg(feature = "zk")]
pub use proof::ZkHidingProof;
pub use proof::{
    absorb_interstage_claims, combine_polys, eval_poly, linear_combination,
    range_check_eval_from_s, reorder_stage1_coords, stage1_interstage_batch_weights,
    stage1_leaf_coeffs, stage1_stage_count, stage1_tree_product_stage_arities,
    stage1_tree_stage_shapes, validate_stage1_tree_basis,
};
pub use proof::{
    append_batched_commitments_to_transcript, append_carried_opening_batch_to_transcript,
    append_claim_incidence_shape_to_transcript, append_claim_points_to_transcript,
    append_claim_values_to_transcript, append_prepared_root_opening_point,
    carried_opening_incidence_summary, checked_total_claims, derive_public_matrix_flat,
    flatten_batched_commitment_rows, folded_root_supports_opening_shape, i8_digits_to_bytes,
    prepare_recursive_opening_point_ext, prepare_root_opening_point,
    prepare_root_opening_point_ext, relation_claim_from_rows, relation_claim_from_rows_extension,
    ring_inner_product_with_extension_weights, ring_subfield_packed_extension_opening_point,
    root_tensor_projection_enabled, sample_public_matrix_seed, sample_public_row_coefficients,
    terminal_w_hat_bytes_from_blocks, terminal_witness_segment_layout,
    terminal_witness_segment_layout_from_counts, terminal_witness_transcript_parts,
    validate_batched_inputs, validate_carried_opening_batch, validate_public_matrix_matches_seed,
    verifier_claims_to_incidence, AkitaBatchedFoldRoot, AkitaBatchedProof, AkitaBatchedProofShape,
    AkitaBatchedRootProof, AkitaCommitment, AkitaCommitmentHint, AkitaExpandedSetup,
    AkitaLevelProof, AkitaProofStep, AkitaProofStepShape, AkitaSetupSeed, AkitaStage1Proof,
    AkitaStage1StageProof, AkitaStage1StageShape, AkitaStage2Proof, AkitaVerifierSetup,
    CarriedOpeningClaim, CarriedOpeningKind, ClaimIncidence, ClaimIncidenceLimits,
    ClaimIncidenceSummary, CommitmentVerifier, CommittedOpenings, DirectWitnessProof,
    DirectWitnessShape, DummyProof, ExtensionOpeningReductionProof, ExtensionOpeningReductionShape,
    FlatDigitBlockIter, FlatDigitBlocks, FlatRingVec, IncidenceClaim, LevelProofShape,
    OpeningPoints, PackedDigits, PreparedRecursiveOpeningPoint, PreparedRootOpeningPoint,
    PublicMatrixSeed, PublicOpeningRow, RelationOnlyStage2Inputs, RingCommitment,
    RingMultiplierOpeningPoint, RingSliceSerializer, SetupMatrixEnvelope, TerminalLevelProof,
    TerminalLevelProofShape, TerminalWitnessSegmentLayout, TerminalWitnessTranscriptParts,
    VerifierClaims, EXTENSION_OPENING_REDUCTION_DEGREE, MAX_SETUP_MATRIX_FIELD_ELEMENTS,
};
#[cfg(feature = "zk")]
pub use proof::{derive_zk_b_matrix, derive_zk_d_matrix};
pub use schedule::{
    detect_field_modulus, exact_planned_level_execution, generated_schedule_lookup_key,
    planned_schedule_key_from_schedule, r_decomp_levels, root_current_w_len, root_direct_schedule,
    scale_batched_root_layout, schedule_from_plan, schedule_is_root_direct,
    schedule_num_fold_levels, schedule_root_fold_step, schedule_terminal_direct_witness_shape,
    scheduled_fold_execution, scheduled_next_level_params, split_batched_root_params,
    split_batched_root_params_from_schedule_plan, validate_opening_points_for_claims,
    w_ring_element_count, w_ring_element_count_with_counts, w_ring_element_count_with_counts_bits,
    w_ring_element_count_with_counts_for_layout, w_ring_element_count_with_counts_for_layout_bits,
    AkitaPlannedDirectStep, AkitaPlannedLevel, AkitaPlannedLevelExecution, AkitaPlannedState,
    AkitaPlannedStep, AkitaScheduleInputs, AkitaScheduleLookupKey, AkitaSchedulePlan, DirectStep,
    FoldStep, Schedule, Step,
};
pub use transcript::AppendToTranscript;
