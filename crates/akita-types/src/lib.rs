//! Shared Akita protocol data shapes.
//!
//! This crate contains proof objects, commitment/opening wrappers, opening
//! point reductions, per-level parameter shapes, commitment API contracts, and
//! generated schedule/SIS data shared by prover, verifier, and planner code.

pub mod config;
pub(crate) mod descriptor_bytes;
pub mod dispatch;
pub use dispatch::{
    field_modulus, ntt_max_ring_d, ntt_min_ring_d, ntt_ring_degree_supported_for_field,
    ntt_ring_degree_supported_for_tier, outer_opening_min_ring_d, protocol_dispatch_tier,
    validate_ring_dispatch, validate_role_dims_for_field, validate_role_dispatch,
    ProtocolDispatchSlot, ProtocolRingDispatchTierId,
};
pub mod extension_opening_reduction;
pub mod field_reduction;
pub mod golomb_rice;
pub mod instance_descriptor;
pub mod layout;
pub mod lhl_blinding;
pub mod ntt_cache;
pub mod opening_claims;
pub mod proof;
pub mod proof_size;
pub mod schedule;
pub mod setup_contribution;
pub mod sis;
pub mod stage3_geometry;
pub mod tail_golomb_rice_low_bits;
pub mod trace_weight;
pub mod transcript;
pub mod witness;

pub use akita_challenges::TensorChallengeShape;
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
    embed_ring_subfield_scalar_flat, embed_ring_subfield_vector, embed_subfield,
    pack_tensor_base_lift_i8_digits, psi_embed, recover_ring_subfield_inner_product, trace_h,
    validate_ring_subfield_role, FpExtEncoding, SubfieldParams,
};
pub use golomb_rice::{
    golomb_rice_flat_rows_admit_terminal_wire, golomb_rice_max_quotient_for_cap,
    golomb_rice_rows_admit_terminal_wire, golomb_rice_rows_encodable_at_wire_low_bits,
    golomb_rice_total_wire_bits, ZFoldEncodingStats, TAIL_Z_PLANNER_CAP_LOW_BITS_PLUS_TWO,
};
pub use instance_descriptor::{
    digest_effective_schedule, digest_level_params, digest_serializable, setup_seed_digest,
    AkitaInstanceDescriptor, AlgebraSection, CallSection, FoldLinfProtocolBinding, PlanSection,
    ProtocolFeatureSet, SetupSection, FOLD_GRIND_PROBE_ORDER_SEQUENTIAL_MIN,
    FOLD_GRIND_PROBE_ORDER_TRANSCRIPT_SHUFFLE,
};
pub use layout::{
    basis_weights, basis_weights_prefix, block_rings_at_opening, checked_opening_source_index,
    extension_opening_reduction_level_bytes, extension_opening_reduction_proof_bytes, field_bytes,
    gadget_row_scalars, lagrange_weights, monomial_weights, opening_domain_len,
    packed_digits_bytes, padded_boolean_opening_vars, planned_next_w_len,
    planned_w_ring_element_count, proof_ring_vec_bytes, reduce_inner_opening_to_ring_element,
    ring_opening_point_from_field, segment_typed_witness_bytes, shared_d_digit_log_basis,
    sumcheck_rounds, validate_role_dims, validate_schedule_ring_dims, BasisMode,
    CommitmentRingDims, FlatMatrix, LevelParams, LevelParamsLike, PrecommittedLevelParams,
    RelationMatrixRowLayout, RingMatrixView, RingOpeningPoint, RingRole, MAX_FOLD_LEVELS,
    MIN_A_ROLE_FOLD_CHALLENGE_RING_D, SUPPORTED_CHALLENGE_RING_DIMS, SUPPORTED_RING_DIMS,
};
pub use ntt_cache::{
    build_negacyclic_and_cyclic_ntt_slot, build_negacyclic_ntt_slot,
    max_safe_crt_accumulation_width, select_crt_ntt_params, NttCacheKey, PreparedNttSlot,
    PreparedNttSlotAny, ProtocolCrtNttParams,
};
pub use proof::{
    active_setup_field_len, append_batched_commitments_to_transcript,
    append_claim_values_to_transcript, assemble_relation_rhs,
    build_segment_typed_witness_from_groups, decode_terminal_z_golomb_payload,
    decode_terminal_z_golomb_payload_with_cap, derive_public_matrix_flat, emit_witness_e_planes,
    emit_witness_r_planes, emit_witness_t_planes, emit_witness_z_planes,
    folded_root_supports_opening_shape, generate_relation_rhs, padded_scalar_batch_num_vars,
    padded_setup_prefix_len, prepare_opening_point, raw_field_segment_bytes,
    relation_claim_from_layout_extension, relation_claim_from_rows,
    relation_claim_from_rows_extension, relation_rhs_coeff_len, relation_rhs_layout_for,
    relation_rhs_row_count, ring_relation_segment_lengths,
    ring_subfield_packed_extension_opening_point, root_tensor_projection_enabled,
    sample_public_matrix_seed, sample_public_row_coefficients,
    segment_typed_witness_upper_bound_bytes, segment_typed_z_payload_bytes,
    select_setup_prefix_slot, setup_prefix_precommitted_params, setup_prefix_slot_id,
    should_reject_multi_group_root, tail_golomb_rice_z_params,
    tail_segment_multiplicities_from_layout, tail_segment_multiplicities_from_layout_for_params,
    terminal_golomb_grind_tail_t_vectors, validate_batched_inputs,
    validate_public_matrix_matches_seed, validate_scalar_point_matches_poly_arity,
    validate_segment_typed_z_payload, z_fold_decoded_from_segment,
    z_fold_encoding_stats_from_segment, AkitaBatchedProof, AkitaBatchedProofShape, AkitaCommitment,
    AkitaCommitmentHint, AkitaExpandedSetup, AkitaSetupSeed, AkitaStage1Proof,
    AkitaStage1StageProof, AkitaStage1StageShape, AkitaStage2Proof, AkitaVerifierSetup, Commitment,
    CommitmentVerifier, DigitBlockIter, DigitBlocks, DummyProof, ExtensionOpeningReductionProof,
    ExtensionOpeningReductionShape, FoldLevelProof, LevelProofShape, NextWitnessBinding,
    NextWitnessBindingShape, OpeningClaims, OpeningClaimsLayout, OpeningPoints,
    PointVariableSelection, PolynomialGroupClaims, PolynomialGroupLayout, PreparedOpeningPoint,
    ProverCommitmentRows, PublicMatrixSeed, RelationGroupRows, RelationRangeImageGroupPlan,
    RelationRangeImagePlan, RelationRhsLayout, RingCommitment, RingMultiplierOpeningPoint,
    RingRelationInstance, RingRelationOpeningCounts, RingRelationSegmentLengths, RingVec, RingView,
    SegmentTypedWitness, SegmentTypedWitnessGroupParts, SegmentTypedWitnessShape,
    SetupMatrixEnvelope, SetupPrefixProverRegistry, SetupPrefixPublicCommitment, SetupPrefixSlot,
    SetupPrefixSlotId, SetupPrefixVerifierRegistry, SetupPrefixVerifierSlot,
    SetupProductSumcheckShape, SetupSumcheckProof, TailSegmentGroupLayout, TailSegmentLayout,
    TerminalLevelProof, TerminalLevelProofShape, TerminalWitnessTranscriptParts,
    MAX_SETUP_MATRIX_FIELD_ELEMENTS, MULTI_GROUP_ROOT_DENSE_UNSUPPORTED, SETUP_OFFLOAD_D_SETUP,
    SETUP_OFFLOAD_MIN_PREFIX_FIELD_LEN, SETUP_SUMCHECK_DEGREE,
};
pub use proof::{
    append_digit_range_child_claims, DigitRangeEqualityPoint, DigitRangePlan, FlatBooleanDomain,
};
pub use proof_size::{level_proof_bytes, FOLD_GRIND_NONCE_BYTES};
pub use schedule::{
    detect_field_modulus, intermediate_w_ring_element_count_for_chunks,
    intermediate_w_ring_element_count_with_counts,
    intermediate_w_ring_element_count_with_counts_bits, r_decomp_levels, root_current_w_len,
    AkitaScheduleInputs, AkitaScheduleLookupKey, ExecutionSchedule, FoldStep,
    NextWitnessBindingPolicy, PrecommittedGroupParams, Schedule, ScheduleKeyPrecommitSource,
    TerminalWitnessPlan,
};
pub use setup_contribution::{
    ensure_setup_envelope, shared_setup_fold_gadget, SetupContributionGroupInputs,
    SetupContributionPlan, SetupIndexWeightEvaluator, SetupProjectionGeometry,
};
pub use sis::{
    AjtaiKeyParams, ScalarCutoff, SisMatrixRole, SisModulusProfileId, SisRoleCell,
    SisSecurityPolicyId, SisTableDigest, SisTableKey, DEFAULT_SIS_SECURITY_POLICY,
};
pub use stage3_geometry::BatchedStage3Geometry;
pub use tail_golomb_rice_low_bits::{
    cap_rice_low_bits, wire_rice_low_bits, wire_rice_low_bits_from_rule, WIRE_RICE_LOW_BITS_DELTA,
    WIRE_RICE_LOW_BITS_RULE_SECURITY_MINUS_DELTA,
};
pub use trace_weight::{
    ensure_trace_stage2_supported, prepare_evaluation_trace_group_parameters,
    scale_evaluation_trace_claim_coefficients, EvaluationTraceGroupParameters,
    EvaluationTraceInputs,
};
pub use transcript::AppendToTranscript;
pub use witness::{
    ChunkedWitnessCfg, MultiChunkProfileId, WitnessLayout, WitnessUnitLayout, MAX_WITNESS_CHUNKS,
};
