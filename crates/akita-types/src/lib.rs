//! Shared Akita protocol data shapes.
//!
//! This crate contains proof objects, commitment/opening wrappers, opening
//! point reductions, per-level parameter shapes, commitment API contracts, and
//! generated schedule/SIS data shared by prover, verifier, and planner code.

pub mod config;
pub(crate) mod descriptor_bytes;
pub mod dispatch;
pub mod extension_opening_reduction;
pub mod field_reduction;
pub mod golomb_rice;
pub mod instance_descriptor;
pub mod jl;
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
    check_extension_opening_reduction_output, checked_table_len,
    derive_tensor_extension_opening_claim, derive_tensor_extension_opening_claim_from_partials,
    extension_opening_reduction_claim, extension_opening_reduction_eval_at_point,
    num_rounds_from_table_len, project_tensor_factor_value, tensor_column_partials_from_base_evals,
    tensor_column_partials_split_fold, tensor_equality_factor_eval_at_point,
    tensor_equality_factor_evals, tensor_opening_split, tensor_packed_witness_evals,
    tensor_reduction_claim_from_rows, tensor_row_partials_from_columns, validate_reduction_tables,
    ExtensionOpeningFactorTerm, ExtensionOpeningReductionFactor,
    ExtensionOpeningReductionRoundResult, ExtensionOpeningTensorPartials, FlatColumnSource,
    TensorColumnSource, EXTENSION_OPENING_REDUCTION_DEGREE,
};
pub use field_reduction::{
    check_trace_inner_product, dispatch_trace_inner_product_check, embed_ring_subfield_scalar,
    embed_ring_subfield_vector, embed_subfield, pack_tensor_base_lift_i8_digits, psi_embed,
    recover_ring_subfield_inner_product, trace_h, validate_ring_subfield_role, FpExtEncoding,
    SubfieldParams,
};
pub use golomb_rice::ZFoldEncodingStats;
pub use instance_descriptor::{
    digest_effective_schedule, digest_level_params, digest_opening_batch, digest_serializable,
    setup_seed_digest, AkitaInstanceDescriptor, AlgebraSection, CallSection,
    FoldLinfProtocolBinding, PlanSection, ProtocolFeatureSet, SetupSection,
    FOLD_GRIND_PROBE_ORDER_SEQUENTIAL_MIN, FOLD_GRIND_PROBE_ORDER_TRANSCRIPT_SHUFFLE,
};
pub use jl::{
    absorb_jl_image, embed_jl_image_coords, embed_signed_i32, field_modulus, jl_image_claim,
    padded_live_table, sample_jl_row_point, validate_layout_for_matrix_mle, JlWitnessLayout,
    JL_CONSISTENCY_DEGREE,
};
pub use layout::{
    basis_weights, block_rings_at_opening, direct_witness_bytes,
    extension_opening_reduction_proof_bytes, field_bytes, gadget_row_scalars, lagrange_weights,
    monomial_weights, packed_digits_bytes, planned_next_w_len, planned_w_ring_element_count,
    proof_ring_vec_bytes, reduce_inner_opening_to_ring_element, ring_opening_point_from_field,
    sumcheck_rounds, BasisMode, BlockOrder, FlatMatrix, LevelParams, MRowLayout, RingMatrixView,
    RingOpeningPoint,
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
    append_claim_values_to_transcript, batched_eval_target_from_opening_batch,
    build_segment_typed_witness, checked_total_claims, derive_public_matrix_flat,
    e_folded_segment_bytes, emit_witness_planes_block_inner, emit_witness_z_folded_planes_inner,
    expand_segment_typed_to_i8_digits, folded_root_supports_opening_shape, generate_y,
    i8_digits_to_bytes, padded_scalar_batch_num_vars, padded_setup_prefix_len,
    prepare_opening_point, relation_claim_from_rows, relation_claim_from_rows_extension,
    ring_relation_segment_lengths, ring_subfield_packed_extension_opening_point,
    root_tensor_projection_enabled, sample_public_matrix_seed, sample_public_row_coefficients,
    segment_typed_witness_shape, segment_typed_witness_upper_bound_bytes,
    segment_typed_z_payload_bytes, select_setup_prefix_slot, setup_prefix_level_params,
    setup_prefix_slot_id, tail_golomb_rice_z_params, tail_segment_layout,
    tail_segment_multiplicities_from_layout, terminal_direct_witness_shape,
    terminal_direct_witness_shape_for_key, terminal_e_hat_bytes_from_blocks,
    terminal_fold_segment_counts, terminal_witness_segment_layout,
    terminal_witness_segment_layout_from_counts, terminal_witness_transcript_parts,
    validate_batched_inputs, validate_public_matrix_matches_seed,
    validate_scalar_point_matches_poly_arity, validate_segment_typed_z_payload,
    z_fold_decoded_from_segment, z_fold_encoding_stats_from_segment, AkitaBatchedFoldRoot,
    AkitaBatchedProof, AkitaBatchedProofShape, AkitaBatchedRootProof, AkitaCommitment,
    AkitaCommitmentHint, AkitaExpandedSetup, AkitaIntermediateStage2Proof, AkitaLevelProof,
    AkitaProofStepShape, AkitaSetupSeed, AkitaStage1Proof, AkitaStage1StageProof,
    AkitaStage1StageShape, AkitaStage2Proof, AkitaTerminalStage2Proof, AkitaVerifierSetup,
    CleartextWitnessProof, CleartextWitnessShape, CommitmentGroup, CommitmentVerifier, DummyProof,
    ExtensionOpeningReductionProof, ExtensionOpeningReductionShape, FlatDigitBlockIter,
    FlatDigitBlocks, FlatRingVec, LevelProofShape, OpeningBatchLimits, OpeningBatchShape,
    OpeningGroupShape, OpeningPoints, PackedDigits, PointVariableSelection, PreparedOpeningPoint,
    ProverCommitmentRows, PublicMatrixSeed, RelationOnlyStage2Inputs, RingCommitment,
    RingMultiplierOpeningPoint, RingRelationInstance, RingRelationOpeningCounts,
    RingRelationSegmentLayout, RingRelationSegmentLengths, RingSliceSerializer,
    SegmentTypedWitness, SegmentTypedWitnessShape, SetupMatrixEnvelope, SetupPrefixProverRegistry,
    SetupPrefixPublicCommitment, SetupPrefixSlot, SetupPrefixSlotId, SetupPrefixVerifierRegistry,
    SetupPrefixVerifierSlot, SetupProductSumcheckShape, SetupSumcheckProof, TailSegmentLayout,
    TerminalLevelProof, TerminalLevelProofShape, TerminalWitnessSegmentLayout,
    TerminalWitnessTranscriptParts, VerifierOpeningBatch, GROUPED_ROOT_DENSE_UNSUPPORTED,
    GROUPED_ROOT_RECURSIVE_SETUP_UNSUPPORTED, GROUPED_ROOT_TIERED_UNSUPPORTED,
    GROUPED_ROOT_UNSUPPORTED, MAX_SETUP_MATRIX_FIELD_ELEMENTS, SETUP_OFFLOAD_D_SETUP,
    SETUP_SUMCHECK_DEGREE,
};
#[cfg(feature = "zk")]
pub use proof::{derive_zk_b_matrix, derive_zk_d_matrix};
pub use proof_size::{level_proof_bytes, FOLD_GRIND_NONCE_BYTES};
pub use schedule::{
    detect_field_modulus, r_decomp_levels, root_current_w_len, root_direct_schedule,
    schedule_is_root_direct, schedule_num_fold_levels, schedule_root_fold_step,
    schedule_terminal_direct_witness_shape, scheduled_next_level_params, w_ring_element_count,
    w_ring_element_count_with_counts, w_ring_element_count_with_counts_bits,
    w_ring_element_count_with_counts_for_layout, w_ring_element_count_with_counts_for_layout_bits,
    AkitaScheduleInputs, AkitaScheduleLookupKey, DirectStep, ExecutionSchedule, FoldStep, Schedule,
    Step,
};
pub use setup_contribution::{SetupContributionPlan, SetupContributionPlanInputs};
pub use sis::{AjtaiKeyParams, SisModulusFamily};
pub use trace_weight::{
    build_trace_claim_root, build_trace_table_scaled, ensure_trace_stage2_supported,
    eval_trace_terms_closed, root_trace_block_opening, stage2_trace_coeff,
    trace_public_weights_recursive, trace_public_weights_root_terms, trace_terms_recursive,
    trace_terms_root, trace_weight_layout_from_segment, TraceClaim, TraceFieldBlockOpening,
    TraceOpeningAtPoint, TracePublicWeights, TraceRingBlockOpening, TraceSparseColumn, TraceTable,
    TraceTerm, TraceWeightLayout,
};
pub use transcript::AppendToTranscript;
