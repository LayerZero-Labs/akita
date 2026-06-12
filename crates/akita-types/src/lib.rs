//! Shared Akita protocol data shapes.
//!
//! This crate contains proof objects, commitment/opening wrappers, opening
//! point reductions, per-level parameter shapes, commitment API contracts, and
//! generated schedule/SIS data shared by prover, verifier, and planner code.

pub mod config;
pub(crate) mod descriptor_bytes;
pub mod extension_opening_reduction;
pub mod field_reduction;
pub mod instance_descriptor;
pub mod layout;
pub mod proof;
pub mod proof_size;
pub mod schedule;
pub mod setup_contribution;
pub mod sis;
pub mod trace_weight;
pub mod transcript;
#[cfg(feature = "zk")]
pub mod zk;

pub use config::{DecompositionParams, SetupContributionMode};
pub use extension_opening_reduction::{
    check_extension_opening_reduction_output, check_tensor_extension_opening_claim,
    checked_table_len, extension_opening_reduction_claim,
    extension_opening_reduction_eval_at_point, num_rounds_from_table_len,
    project_tensor_factor_value, tensor_column_partials_from_base_evals,
    tensor_column_partials_split_fold, tensor_equality_factor_eval_at_point,
    tensor_equality_factor_evals, tensor_logical_claim_from_partials, tensor_opening_split,
    tensor_packed_witness_evals, tensor_partials_from_base_evals, tensor_reduction_claim_from_rows,
    tensor_row_partials_from_columns, validate_reduction_tables, ExtensionOpeningFactorTerm,
    ExtensionOpeningReductionFactor, ExtensionOpeningReductionRoundResult,
    ExtensionOpeningTensorPartials, FlatColumnSource, TensorColumnSource,
    EXTENSION_OPENING_REDUCTION_DEGREE,
};
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
    basis_weights, block_rings_at_opening, direct_witness_bytes,
    extension_opening_reduction_proof_bytes, field_bytes, gadget_row_scalars, lagrange_weights,
    monomial_weights, packed_digits_bytes, planned_next_w_len, planned_w_ring_element_count,
    proof_ring_vec_bytes, reduce_inner_opening_to_ring_element, ring_opening_point_from_field,
    root_extension_opening_partials, sumcheck_rounds, BasisMode, BlockOrder, FlatMatrix,
    LevelParams, MRowLayout, RingMatrixView, RingOpeningPoint,
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
    active_setup_field_len, append_batched_commitments_to_transcript,
    append_claim_incidence_shape_to_transcript, append_claim_points_to_transcript,
    append_claim_values_to_transcript, checked_total_claims, derive_public_matrix_flat,
    flatten_batched_commitment_rows, folded_root_supports_opening_shape, generate_y,
    i8_digits_to_bytes, padded_setup_prefix_len, prepare_opening_point, relation_claim_from_rows,
    relation_claim_from_rows_extension, ring_column_z_first,
    ring_relation_segment_layout_for_opening_shape, ring_subfield_packed_extension_opening_point,
    root_tensor_projection_enabled, sample_public_matrix_seed, sample_public_row_coefficients,
    select_setup_prefix_slot, setup_prefix_level_params, setup_prefix_slot_id,
    terminal_e_hat_bytes_from_blocks, terminal_witness_segment_layout,
    terminal_witness_segment_layout_from_counts, terminal_witness_transcript_parts,
    validate_batched_inputs, validate_public_matrix_matches_seed, verifier_claims_to_incidence,
    AkitaBatchedFoldRoot, AkitaBatchedProof, AkitaBatchedProofShape, AkitaBatchedRootProof,
    AkitaCommitment, AkitaCommitmentHint, AkitaExpandedSetup, AkitaLevelProof, AkitaProofStep,
    AkitaProofStepShape, AkitaSetupSeed, AkitaStage1Proof, AkitaStage1StageProof,
    AkitaStage1StageShape, AkitaStage2Proof, AkitaVerifierSetup, ClaimIncidence,
    ClaimIncidenceLimits, ClaimIncidenceSummary, CleartextWitnessProof, CleartextWitnessShape,
    CommitmentRouting, CommitmentVerifier, CommittedOpenings, DummyProof,
    ExtensionOpeningReductionProof, ExtensionOpeningReductionShape, FlatDigitBlockIter,
    FlatDigitBlocks, FlatRingVec, IncidenceClaim, LevelProofShape, OpeningPoints, PackedDigits,
    PreparedOpeningPoint, PublicMatrixSeed, PublicOpeningRow, RelationOnlyStage2Inputs,
    RingCommitment, RingMultiplierOpeningPoint, RingRelationInstance, RingRelationSegmentLayout,
    RingSliceSerializer, RootLevelRawOutput, SetupMatrixEnvelope, SetupPrefixProverRegistry,
    SetupPrefixPublicCommitment, SetupPrefixSlot, SetupPrefixSlotId, SetupPrefixVerifierRegistry,
    SetupPrefixVerifierSlot, SetupProductSumcheckShape, SetupSumcheckProof, TerminalLevelProof,
    TerminalLevelProofShape, TerminalWitnessSegmentLayout, TerminalWitnessTranscriptParts,
    VerifierClaims, MAX_SETUP_MATRIX_FIELD_ELEMENTS, SETUP_OFFLOAD_D_SETUP, SETUP_SUMCHECK_DEGREE,
};
#[cfg(feature = "zk")]
pub use proof::{derive_zk_b_matrix, derive_zk_d_matrix};
pub use proof_size::level_proof_bytes;
pub use schedule::{
    detect_field_modulus, r_decomp_levels, root_current_w_len, root_direct_schedule,
    schedule_is_root_direct, schedule_num_fold_levels, schedule_root_fold_step,
    schedule_terminal_direct_witness_shape, scheduled_fold_execution, scheduled_next_level_params,
    validate_opening_points_for_claims, w_ring_element_count, w_ring_element_count_with_counts,
    w_ring_element_count_with_counts_bits, w_ring_element_count_with_counts_for_layout,
    w_ring_element_count_with_counts_for_layout_bits, AkitaScheduleInputs, AkitaScheduleLookupKey,
    DirectStep, FoldStep, Schedule, Step,
};
pub use setup_contribution::{SetupContributionPlan, SetupContributionPlanInputs};
pub use sis::{AjtaiKeyParams, SisModulusFamily};
pub use trace_weight::{
    batched_eval_target_from_incidence, build_trace_stage2_compact,
    build_trace_stage2_compact_scaled, build_trace_table_scaled,
    build_trace_weight_table_field_block_weights, build_trace_weight_table_field_terms,
    build_trace_weight_table_ring_block_weights, build_trace_weight_table_ring_terms,
    ensure_trace_stage2_supported, eval_trace_claim, eval_trace_terms_closed,
    eval_trace_weight_at_point, root_trace_block_opening, trace_block_weights_k1,
    trace_input_claim, trace_public_weights_field_terms, trace_public_weights_k1,
    trace_public_weights_recursive, trace_public_weights_ring, trace_public_weights_ring_terms,
    trace_public_weights_root_terms, trace_stage2_supported, trace_terms_recursive,
    trace_terms_root, trace_weight_evals_for_witness, trace_weight_layout_from_segment, TraceClaim,
    TraceFieldBlockOpening, TraceOpeningAtPoint, TracePublicWeights, TraceRingBlockOpening,
    TraceSparseColumn, TraceTable, TraceTerm, TraceWeightLayout,
};
pub use transcript::AppendToTranscript;
