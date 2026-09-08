//! Shared Akita protocol data shapes.
//!
//! This crate contains proof objects, commitment/opening wrappers, opening
//! point reductions, per-level parameter shapes, commitment API contracts, and
//! generated schedule/SIS data shared by prover, verifier, and planner code.

pub mod commitment_slicing;
pub mod compression;
pub mod config;
pub(crate) mod descriptor_bytes;
pub mod dispatch;
mod subring_coefficient_packing;
pub use dispatch::{
    compression_ring_dim_supported_for_tier, field_modulus, field_modulus_be_bytes, ntt_max_ring_d,
    ntt_min_ring_d, ntt_ring_degree_supported_for_field, ntt_ring_degree_supported_for_tier,
    outer_opening_min_ring_d, protocol_dispatch_tier, protocol_dispatch_tier_for_sis_profile,
    validate_ring_dispatch, validate_role_dims_for_field, validate_role_dispatch,
    ProtocolDispatchSlot, ProtocolRingDispatchTierId,
};
pub mod extension_opening_reduction;
pub mod field_reduction;
pub mod golomb_rice;
pub mod instance_descriptor;
pub mod layout;
pub mod lhl_blinding;
pub(crate) mod narrowing;
pub mod ntt_cache;
pub mod opening_claims;
pub mod proof;
pub mod proof_size;
mod ring_relation_mode;
pub mod schedule;
pub mod schedule_selection;
pub mod setup_contribution;
pub mod signed_digit;
pub mod sis;
pub mod tail_golomb_rice_low_bits;
pub mod trace_weight;
pub mod transcript;
mod transcript_grinding;
mod transcript_grinding_plan;
pub mod witness;

pub use commitment_slicing::{
    CommitmentSliceCount, CommitmentSliceGeometry, MAX_COMMITMENT_SLICES,
};
pub use compression::{
    compression_ring_dimensions, CommitmentPayloadGeometry, CommitmentPayloadMode,
    CommitmentPayloadPhase, CompressionChainPlan, CompressionChainWitness, CompressionMapPlan,
    CompressionPolicyId, CompressionTerminalPayload, PackedNegativeBinary, COMPRESSION_MAP_COUNT,
    COMPRESSION_POLICY, COMPRESSION_TARGET_BYTES, MAX_COMPRESSION_INPUT_BYTES,
};
pub use config::{DecompositionParams, SetupContributionMode};
pub use extension_opening_reduction::{
    check_extension_opening_reduction_output, derive_tensor_extension_opening_claim,
    derive_tensor_extension_opening_claim_from_partials, extension_opening_reduction_claim,
    extension_opening_reduction_eval_at_point, num_rounds_from_table_len,
    project_tensor_factor_value, reduction_table_len, tensor_column_partials_from_base_evals,
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
    FpExtEncoding, SubfieldParams,
};
pub use golomb_rice::{
    golomb_rice_flat_admit_terminal_wire, golomb_rice_flat_rows_admit_terminal_wire,
    golomb_rice_max_quotient_for_cap, golomb_rice_rows_admit_terminal_wire,
    golomb_rice_rows_encodable_at_wire_low_bits, golomb_rice_total_wire_bits,
    golomb_rice_values_within_cap, golomb_rice_zigzag_width, ZFoldEncodingStats,
};
pub use instance_descriptor::{
    digest_descriptor_bytes, digest_effective_schedule, digest_level_params, digest_serializable,
    setup_seed_digest, AkitaInstanceDescriptor, AlgebraSection, CallSection, PlanSection,
    ProtocolFeatureSet, SetupSection, TranscriptGrindingBinding,
};
pub use layout::{
    basis_weights, basis_weights_prefix, block_rings_at_opening, checked_opening_source_index,
    extension_opening_reduction_level_bytes, extension_opening_reduction_proof_bytes, field_bytes,
    gadget_row_scalars, lagrange_weights, monomial_weights, opening_d_segment_width,
    opening_domain_len, packed_digits_bytes, padded_boolean_opening_vars, proof_ring_vec_bytes,
    reduce_inner_opening_to_ring_element, ring_opening_point_from_field, shared_d_digit_log_basis,
    sumcheck_rounds, terminal_response_bytes, terminal_response_planner_bytes,
    try_extension_opening_reduction_level_bytes, validate_role_dims, validate_schedule_ring_dims,
    witness_commitment_domain_len, BasisMode, BlockGeometry, CommitmentRingDims,
    CommittedGroupParams, FlatMatrix, GadgetDigits, GroupOpenPhaseParams, GroupOpeningPlan,
    InnerRoleParams, OpenRoleParams, OpeningFamily, OpeningMethod, OuterRoleParams,
    PrecommittedGroupAdmissionPolicy, RingMatrixView, RingOpeningPoint, RingRole, RoleParams,
    MAX_FOLD_LEVELS, MIN_A_ROLE_FOLD_CHALLENGE_RING_D, SUPPORTED_CHALLENGE_RING_DIMS,
    SUPPORTED_COMMITMENT_RING_DIMS,
};
pub use ntt_cache::{
    build_riscv64_scalar_q128_cache_artifact, centered_quotient_requires_i16_tail,
    centered_quotient_requires_i16_tail_for_field, dense_i8_commit_prefers_exact_ifma52,
    ntt_cache_requires_exactness_tail, planned_exact_ntt_cache_bytes,
    prepare_compression_ntt_cache, prepare_ntt_cache, prepare_reduced_compression_ntt_cache,
    prepared_verifier_ntt_cache_metadata, select_compression_crt_ntt_params, select_crt_ntt_params,
    NttCacheKey, NttCacheMode, NttPrefixRequirement, NttTransformDomain, PreparedNttCache,
    PreparedNttTailPairView, PreparedVerifierNttCacheBinding, PreparedVerifierNttCacheMetadata,
    ProtocolCrtNttParams, PREPARED_VERIFIER_NTT_CACHE_MAX_BYTES,
};
pub use proof::{
    accumulate_matrix_field_elements_for_level, accumulate_terminal_matrix_field_elements,
    active_setup_field_len, append_batched_commitments_to_transcript,
    append_claim_values_to_transcript, assemble_compressed_relation_rhs, assemble_relation_rhs,
    build_compression_relation_weights, build_reduced_compression_relation_weights,
    build_terminal_response, build_terminal_response_from_groups,
    canonical_extension_opening_reduction_shape, canonical_proof_shape,
    commit_only_setup_field_elements, compression_relation_claim_from_rhs_extension,
    decode_terminal_z_golomb_payload, derive_public_matrix_prefix, draw_group_fold_challenges,
    emit_witness_e_planes, emit_witness_r_planes, emit_witness_t_planes, emit_witness_z_planes,
    evaluate_reduced_compression_map, folded_root_supports_opening_shape, generate_relation_rhs,
    padded_setup_prefix_len, prepare_coefficient_packing_batch_semantics,
    prepare_coefficient_packing_verifier_batch_semantics, prepare_opening_point,
    raw_field_segment_bytes, relation_claim_from_compressed_rhs_extension,
    relation_claim_from_layout_extension, relation_claim_from_rows,
    relation_claim_from_rows_extension, relation_rhs_coeff_len, relation_rhs_row_count,
    ring_relation_segment_lengths, ring_subfield_packed_extension_opening_point,
    sample_akita_setup_seed, sample_row_coefficients, scheduled_setup_prefix,
    setup_matrix_capacity_for_schedule, setup_matrix_field_elements_for_schedule,
    setup_prefix_coverage_eval_len, setup_prefix_precommitted_params,
    setup_prefix_slot_field_elements, suffix_opening_layout,
    tail_segment_multiplicities_from_layout, tail_segment_multiplicities_from_layout_for_params,
    terminal_response_upper_bound_bytes, terminal_response_z_payload_bytes,
    validate_batched_inputs, validate_public_matrix_matches_seed, validate_setup_prefix_domain,
    validate_terminal_response_z_payload, verifier_setup_matrix_capacity_for_schedule,
    AkitaBatchedProof, AkitaBatchedProofShape, AkitaCommitment, AkitaCommitmentHint,
    AkitaExpandedSetup, AkitaSetupDescriptor, AkitaSetupSeed, AkitaStage1Proof,
    AkitaStage1StageProof, AkitaStage1StageShape, AkitaStage2Proof, AkitaVerifierSetup,
    CoefficientPackingBatchSemanticInputs, CoefficientPackingBatchSemantics,
    CoefficientPackingChallenges, CoefficientPackingGroupSemantics, CoefficientPackingStage2Source,
    CoefficientPackingStage2Terms, CoefficientPackingVerifierBatchSemantics,
    CoefficientPackingVerifierGroupSemantics, Commitment, CommitmentVerifier, CommittedGroup,
    CompressionRelationAddressGeometry, CompressionRelationWeights, DigitBlockIter, DigitBlocks,
    DummyProof, ExtensionOpeningReductionProof, ExtensionOpeningReductionShape, FoldLevelProof,
    GroupBatchStatement, GroupFoldChallenges, LevelProofShape, NegativeBinarySupport,
    NextWitnessBinding, NextWitnessBindingShape, OpeningClaims, OpeningClaimsLayout, OpeningPoints,
    PhysicalL2NormProof, PhysicalResponsePlan, PolynomialGroupClaims, PolynomialGroupLayout,
    PreparedOpeningPoint, PreparedRingMultiplier, ProverCommitmentRows, PublicMatrixDerivation,
    ReducedCoefficientFunctional, ReducedCompressionRelationWeights, RelationAddressGeometry,
    RelationGroupRows, RelationRangeImageGroupPlan, RelationRangeImagePlan, RelationRhsLayout,
    RelationRowFamily, RelationRowGeometry, RelationWeightContribution, RelationWeightEvent,
    RelationWitnessGeometry, RingCommitment, RingMultiplierOpeningPoint, RingRelationGroupOpening,
    RingRelationGroupOpeningView, RingRelationInstance, RingRelationOpeningCounts,
    RingRelationSegmentLengths, RingVec, RingView, SetupMatrixCapacity, SetupPrefixProverRegistry,
    SetupPrefixPublicCommitment, SetupPrefixSlot, SetupPrefixSlotId, SetupPrefixVerifierRegistry,
    SetupPrefixVerifierSlot, SetupProductSumcheckShape, SetupSumcheckProof,
    SubfieldMultiplierOpeningPoint, TailSegmentGroupLayout, TailSegmentLayout, TerminalLevelProof,
    TerminalLevelProofShape, TerminalResponse, TerminalResponseGroupParts, TerminalResponseShape,
    TerminalWitnessTranscriptParts, WitnessCoefficientSink,
    MAX_GENERIC_SETUP_DECODE_FIELD_ELEMENTS, MAX_UNTRUSTED_COMMITMENT_COEFFICIENTS,
    SETUP_PREFIX_CONTENT_TAG, SETUP_SUMCHECK_DEGREE,
};
pub use proof::{
    append_digit_range_child_claims, reconstruct_l2_sq_from_gram, DigitRangeEqualityPoint,
    DigitRangePlan, FlatBooleanDomain,
};
pub use proof_size::level_proof_bytes;
pub use ring_relation_mode::{RelationCandidateTopology, RingRelationMode, RingRelationPhase};
pub use schedule::{
    detect_field_modulus, r_decomp_levels, root_input_witness_len, AkitaScheduleInputs,
    AkitaScheduleLookupKey, AkitaScheduleLookupOrderKey, CommittedGroupBatchProfile,
    CommittedSourceEncoding, FoldParams, FoldSchedule, FoldScheduleDescriptorStep,
    FoldScheduleEstimate, FoldSuccessor, GroupCommitPhaseParams, NextWitnessBindingPolicy,
    PlannedFoldSchedule, PrecommittedGroupProfiles, ScheduleSisBound, ScheduleSisOccurrence,
    ScheduleSisRole, TerminalFoldParams, TERMINAL_RESPONSE_MIN_TARGET_RETAIN_DEN,
    TERMINAL_RESPONSE_MIN_TARGET_RETAIN_NUM,
};
pub use schedule_selection::{schedule_row_digest, OpeningScheduleSelection, ScheduleRowDigest};
pub use setup_contribution::{
    ensure_setup_envelope, shared_setup_fold_gadget, PreparedCoefficientFunctional,
    PreparedRelationAddress, SetupContributionGroupInputs, SetupContributionPlan,
    SetupProjectionGeometry,
};
pub use signed_digit::{
    balanced_signed_digit_abs_bound, SignedDigitKernel, MAX_I16_LOG_BASIS, MAX_I8_LOG_BASIS,
    MIN_SIGNED_DIGIT_LOG_BASIS,
};
pub use sis::{
    InnerCommitMatrixParams, InnerCommitSecurityRoute, OpenCommitMatrixParams,
    OuterCommitMatrixParams, PhysicalL2NormProofShape, ScalarCutoff, SisL2TableDigest,
    SisL2TableKey, SisMatrixRole, SisModulusProfileId, SisRoleCell, SisSecurityPolicyId,
    SisTableDigest, SisTableKey, DEFAULT_SIS_SECURITY_POLICY,
};
#[cfg(any(test, feature = "test-support"))]
pub use subring_coefficient_packing::coefficient_packing_partials;
pub use subring_coefficient_packing::{
    coefficient_packing_scalar_opening, fold_coefficient_packing_partials,
    CoefficientPackingFoldProduct, PreparedSubringCoefficientPackingPoint,
    SubringCoefficientPackingGeometry,
};
pub use tail_golomb_rice_low_bits::{cap_rice_low_bits, wire_rice_low_bits};
pub use trace_weight::{
    build_multi_group_root_stage2_trace_table, build_trace_claim_multi_group_root,
    build_trace_claim_root, build_trace_table_scaled, ensure_trace_stage2_supported,
    eval_dense_trace_table, eval_trace_terms_closed, prepare_evaluation_trace_group_parameters,
    root_trace_block_opening, scale_evaluation_trace_claim_coefficients,
    trace_public_weights_recursive, trace_public_weights_root_terms, trace_terms_recursive,
    trace_terms_root, trace_weight_layout_from_segment, EvaluationTraceGroupParameters,
    EvaluationTraceInputs, TraceClaim, TraceFieldBlockOpening, TraceOpeningAtPoint,
    TracePublicWeights, TraceRingBlockOpening, TraceSparseColumn, TraceTable, TraceTerm,
    TraceTermBatch, TraceWeightLayout,
};
pub use transcript::AppendToTranscript;
pub use transcript_grinding::{
    grind_bits_for_loss, multilinear_point_loss_factor, nominal_challenge_capacity_bits,
    polynomial_identity_loss_factor, powers_batch_loss_factor, ring_switch_alpha_loss_factor,
    sample_grinded_sumcheck_challenge, GrindingPlan, GrindingQueryKind, GrindingRun, GrindingSite,
    ProverGrindingTranscript, ProverTranscriptGrinding, SumcheckProtocol, TranscriptGrinding,
    TranscriptNonceReader, TranscriptNonceStream, TranscriptNonceWriter,
    VerifierGrindingTranscript, VerifierTranscriptGrinding, FOLD_COORDINATE_ORACLE_REVISION,
    FOLD_RESPONSE_ATTEMPTS, FOLD_RESPONSE_NONCE_BITS, GRINDING_ENCODING_VERSION,
    GRINDING_LITTLE_ENDIAN_BIT_ORDER, GRINDING_NONCE_SLACK_BITS, GRINDING_PREDICATE_BYTES,
    GRINDING_QUERY_POLICY_REVISION, MAX_GRINDING_BITS, TRANSCRIPT_SECURITY_BITS,
};
pub use transcript_grinding_plan::{
    derive_transcript_grinding_plan_from_public_shape,
    transcript_grinding_nonce_bits_for_planner_candidate,
    transcript_grinding_nonce_bits_for_planner_edge,
};
pub use witness::{
    dyadic_block_ranges, grouped_witness_body_coefficients, ChunkedWitnessCfg,
    CompressionWitnessLayerLayout, CompressionWitnessSpan, MultiChunkProfileId,
    QuotientCoefficientBreakdown, RelationQuotientLayout, RelationQuotientPlan, WitnessLayout,
    WitnessQuotientRowLayout, WitnessUnitLayout, MAX_WITNESS_CHUNKS,
};
