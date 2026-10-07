//! Akita parameter geometry, sizing, schedules, and security policy.

pub mod commitment_slicing;
pub mod compression;
pub mod config;
pub mod descriptor_bytes;
pub mod dispatch;
pub use dispatch::{
    compression_ring_dim_supported_for_tier, field_modulus, field_modulus_be_bytes, ntt_max_ring_d,
    ntt_min_ring_d, ntt_ring_degree_supported_for_field, ntt_ring_degree_supported_for_tier,
    outer_opening_min_ring_d, protocol_dispatch_tier, protocol_dispatch_tier_for_sis_profile,
    validate_role_dims_for_field, ProtocolDispatchSlot, ProtocolRingDispatchTierId,
};
pub mod golomb_rice;
pub mod layout;
pub mod proof_size;
mod ring_relation_mode;
pub mod schedule;
pub mod schedule_selection;
pub mod signed_digit;
pub mod sis;
pub mod tail_golomb_rice_low_bits;
mod transcript_grinding;
#[path = "transcript_grinding/plan.rs"]
mod transcript_grinding_plan;
pub mod wire_limits;
pub mod witness;
pub use commitment_slicing::{CommitmentSliceCount, CommitmentSliceGeometry};
pub use compression::{
    compression_ring_dimensions, CommitmentPayloadGeometry, CommitmentPayloadMode,
    CommitmentPayloadPhase, CompressionChainPlan, CompressionChainWitness, CompressionMapPlan,
    CompressionPolicyId, CompressionTerminalPayload, PackedNegativeBinary, COMPRESSION_MAP_COUNT,
    COMPRESSION_POLICY, COMPRESSION_TARGET_BYTES, MAX_COMPRESSION_INPUT_BYTES,
};
pub use config::{DecompositionParams, SetupContributionMode};
pub use descriptor_bytes::{
    digest_descriptor_bytes, DescriptorDigest, AKITA_INSTANCE_DESCRIPTOR_VERSION,
};
pub use golomb_rice::{
    golomb_rice_encode_vec, golomb_rice_max_quotient_for_cap, golomb_rice_total_wire_bits,
    golomb_rice_values_within_cap, golomb_rice_zigzag_width,
};
pub use layout::setup_prefix_slots::setup_prefix_compression_plan;
pub use layout::{
    accumulate_matrix_field_elements_for_level, accumulate_terminal_matrix_field_elements,
    active_setup_field_len, basis_weights, basis_weights_prefix, checked_opening_source_index,
    commit_only_setup_field_elements, commitment_execution_setup_field_elements,
    extension_opening_reduction_level_bytes, extension_opening_reduction_proof_bytes, field_bytes,
    gadget_row_scalars, lagrange_weights, monomial_weights, opening_d_segment_width,
    opening_domain_len, padded_boolean_opening_vars, padded_setup_prefix_len,
    reduce_inner_opening_to_ring_element, ring_opening_point_from_field, scheduled_setup_prefix,
    setup_matrix_capacity_for_schedule, setup_matrix_field_elements_for_schedule,
    setup_prefix_precommitted_params, setup_prefix_slot_field_elements, shared_d_digit_log_basis,
    suffix_opening_layout, sumcheck_rounds, terminal_response_bytes, terminal_response_max_bytes,
    terminal_response_planner_bytes, terminal_response_upper_bound_bytes,
    try_extension_opening_reduction_level_bytes, validate_role_dims, validate_schedule_ring_dims,
    validate_setup_prefix_domain, verifier_setup_matrix_capacity_for_schedule,
    witness_commitment_domain_len, BasisMode, BlockGeometry, CommitmentRingDims,
    CommitmentSetupMatrixShape, CommittedGroupParams, CompressionRelationAddressGeometry,
    DigitRangePlan, FlatBooleanDomain, FlatMatrix, GadgetDigits, GroupOpenPhaseParams,
    GroupOpeningPlan, InnerRoleParams, OpenRoleParams, OpeningClaimsLayout, OpeningMethod,
    OuterRoleParams, PolynomialGroupLayout, PrecommittedGroupAdmissionPolicy,
    RelationAddressGeometry, RelationGroupRows, RelationRhsLayout, RelationRowFamily,
    RelationRowGeometry, RelationWitnessGeometry, RingMatrixView, RingOpeningPoint, RingRole,
    RoleParams, SetupMatrixCapacity, SetupPrefixSlotId, SetupProjectionGeometry, Stage1StageShape,
    SubringCoefficientPackingGeometry, TailSegmentGroupLayout, TailSegmentLayout,
    TerminalResponseShape, EXTENSION_OPENING_REDUCTION_DEGREE, MAX_FOLD_LEVELS,
    MIN_A_ROLE_FOLD_CHALLENGE_RING_D, SETUP_PREFIX_CONTENT_TAG, SETUP_SUMCHECK_DEGREE,
    SUPPORTED_CHALLENGE_RING_DIMS, SUPPORTED_COMMITMENT_RING_DIMS,
};
pub use proof_size::{nonterminal_level_layout, NonterminalLevelLayout};
pub use ring_relation_mode::{RelationCandidateTopology, RingRelationMode, RingRelationPhase};
pub use schedule::{
    detect_field_modulus, r_decomp_levels, root_input_witness_len, CommittedGroupBatchProfile,
    CommittedSourceEncoding, FoldParams, FoldSchedule, FoldScheduleDescriptorStep,
    FoldScheduleEstimate, FoldSuccessor, GroupCommitPhaseParams, NextWitnessBindingPolicy,
    PlannedFoldSchedule, PrecommittedGroupProfiles, ScheduleLookupKey, ScheduleLookupOrderKey,
    ScheduleSisBound, ScheduleSisOccurrence, ScheduleSisRole, TerminalFoldParams,
    TERMINAL_RESPONSE_MIN_TARGET_RETAIN_DEN, TERMINAL_RESPONSE_MIN_TARGET_RETAIN_NUM,
};
pub use schedule_selection::{schedule_row_digest, OpeningScheduleSelection, ScheduleRowDigest};
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
pub use tail_golomb_rice_low_bits::{rice_low_bits_for_cap, wire_rice_low_bits};
pub use transcript_grinding::{
    grind_bits_for_loss, independent_batch_loss_factor, multilinear_point_loss_factor,
    polynomial_identity_loss_factor, powers_batch_loss_factor, ring_switch_alpha_loss_factor,
    ChallengeFieldOrder, GrindingPlan, GrindingQueryKind, GrindingRun, GrindingSite,
    SumcheckProtocol, TranscriptGrindingCost, FOLD_COORDINATE_ORACLE_REVISION,
    FOLD_RESPONSE_ATTEMPTS, FOLD_RESPONSE_NONCE_BITS, GRINDING_ENCODING_VERSION,
    GRINDING_LITTLE_ENDIAN_BIT_ORDER, GRINDING_NONCE_SLACK_BITS, GRINDING_PREDICATE_BYTES,
    GRINDING_QUERY_POLICY_REVISION, MAX_GRINDING_BITS, TRANSCRIPT_GRINDING_QUERY_LIMIT,
    TRANSCRIPT_SECURITY_BITS,
};
pub use transcript_grinding_plan::{
    derive_transcript_grinding_plan_from_public_shape, transcript_grinding_cost_for_planner_edge,
};
pub use witness::{
    dyadic_block_ranges, grouped_witness_body_coefficients, ChunkedWitnessCfg,
    CompressionWitnessLayerLayout, CompressionWitnessSpan, MultiChunkProfileId,
    QuotientCoefficientBreakdown, RelationQuotientLayout, RelationQuotientPlan, WitnessChunkShape,
    WitnessLayout, WitnessQuotientRowLayout, WitnessUnitLayout, MAX_WITNESS_CHUNKS,
};

#[cfg(any(test, feature = "test-support"))]
pub mod test_fixtures;
