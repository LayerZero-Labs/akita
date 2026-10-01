//! Shared Akita wire values and protocol math.

mod eor_sites;
pub mod extension_opening_reduction;
pub mod field_reduction;
mod fold_sites;
pub mod instance_descriptor;
mod l2_sites;
pub mod ntt_cache;
pub mod opening_claims;
pub mod proof;
pub mod setup_contribution;
mod stage1_sites;
mod stage2_sites;
mod stage3_sites;
mod subring_coefficient_packing;
pub mod trace_weight;
mod transcript_grinding;
pub use eor_sites::{eor_final_claims, eor_prefix, EorPrefix, EOR_SUMCHECK_INVOCATION};
pub use extension_opening_reduction::{
    derive_tensor_extension_opening_claim_from_partials, tensor_equality_factor_eval_at_point,
    tensor_opening_split, tensor_reduction_claim_from_rows, tensor_row_partials_from_columns,
};
pub use field_reduction::{
    check_trace_inner_product, dispatch_trace_inner_product_check, embed_ring_subfield_scalar,
    embed_ring_subfield_vector, embed_subfield, pack_tensor_base_lift_i8_digits, psi_embed,
    recover_ring_subfield_inner_product, trace_h, FpExtEncoding, SubfieldParams,
};
pub use fold_sites::FoldSite;
pub use instance_descriptor::{
    digest_effective_schedule, digest_serializable, setup_seed_digest, AkitaInstanceDescriptor,
    AlgebraSection, CallSection, PlanSection, ProtocolFeatureSet, SetupSection,
    TranscriptGrindingBinding,
};
pub use l2_sites::{l2_prefix, l2_virtual_evaluations};
pub use ntt_cache::{
    build_riscv64_scalar_q128_cache_artifact, centered_quotient_requires_i16_tail,
    centered_quotient_requires_i16_tail_for_field, decode_riscv64_scalar_q128_cache,
    dense_i8_commit_prefers_exact_ifma52, ntt_cache_requires_exactness_tail,
    prepare_compression_ntt_cache, prepare_joined_exact_ntt_cache, prepare_ntt_cache,
    prepare_reduced_compression_ntt_cache, prepared_verifier_ntt_cache_metadata,
    select_compression_crt_ntt_params, select_crt_ntt_params, NttCacheKey, NttCacheMode,
    NttPrefixRequirement, NttTransformDomain, PreparedNttCache, PreparedNttTailPairView,
    PreparedVerifierNttCacheBinding, PreparedVerifierNttCacheMetadata, ProtocolCrtNttParams,
    PREPARED_VERIFIER_NTT_CACHE_MAX_BYTES,
};
pub use proof::{
    assemble_compressed_relation_rhs, assemble_relation_rhs, build_compression_relation_weights,
    build_reduced_compression_relation_weights, build_terminal_response_from_payload,
    canonical_extension_opening_reduction_shape, coefficient_packing_relation_events,
    decode_terminal_z_golomb_payload, derive_public_matrix_prefix, draw_group_fold_challenges,
    emit_witness_e_planes, emit_witness_t_planes, generate_relation_rhs,
    prepare_coefficient_packing_batch_semantics, prepare_opening_point,
    relation_claim_from_compressed_rhs_extension, relation_claim_from_layout_extension,
    relation_claim_from_rows, relation_claim_from_rows_extension, relation_rhs_coeff_len,
    relation_rhs_row_count, ring_subfield_packed_extension_opening_point, row_coefficients,
    sample_akita_setup_seed, setup_prefix_coverage_eval_len,
    validate_coefficient_packing_batch_groups, validate_public_matrix_matches_seed,
    AkitaExpandedSetup, AkitaSetupDescriptor, AkitaSetupSeed, AkitaVerifierSetup,
    CoefficientPackingBatchSemanticInputs, CoefficientPackingBatchSemantics,
    CoefficientPackingChallenges, CoefficientPackingGroupSemantics, Commitment, CommittedGroup,
    CompressionRelationWeights, DigitBlockIter, DigitBlocks, ExtensionOpeningReductionShape,
    GroupBatchStatement, GroupFoldChallenges, NegativeBinarySupport, OpeningClaims, OpeningFamily,
    OpeningPoints, PhysicalResponsePlan, PolynomialGroupClaims, PreparedOpeningPoint,
    PreparedRingMultiplier, PublicMatrixDerivation, ReducedCompressionRelationWeights,
    RelationRangeImageGroupPlan, RelationRangeImagePlan, RelationWeightContribution,
    RelationWeightEvent, RingMultiplierOpeningPoint, RingRelationGroupOpening,
    RingRelationGroupOpeningView, RingRelationInstance, RingVec, RingView,
    SetupPrefixPublicCommitment, SetupPrefixVerifierRegistry, SetupPrefixVerifierSlot,
    SubfieldMultiplierOpeningPoint, TerminalResponse, ValidatedCoefficientPackingGroup,
    WitnessCoefficientSink, MAX_GENERIC_SETUP_DECODE_FIELD_ELEMENTS,
    MAX_UNTRUSTED_COMMITMENT_COEFFICIENTS,
};
pub use proof::{
    batch_l2_virtual_evaluations, reconstruct_l2_sq_from_gram, DigitRangeEqualityPoint,
};
#[cfg(any(test, feature = "test-support"))]
pub use proof::{
    coefficient_packing_fixture, coefficient_packing_multigroup_fixture, CoefficientPackingFixture,
    CoefficientPackingMultigroupFixture,
};
#[cfg(test)]
pub(crate) use proof::{
    AkitaStage1Proof, AkitaStage1StageProof, AkitaStage2Proof, ExtensionOpeningReductionProof,
    FoldLevelProof, NextWitnessBinding, PhysicalL2NormProof, SetupSumcheckProof,
    TerminalLevelProof,
};
pub use setup_contribution::{
    checked_slice, ensure_setup_envelope, factor_aligned_role_tensors, project_role_tensors,
    role_projection_evaluation, role_tensors_are_aligned, shared_setup_fold_gadget,
    PhysicalBSetupPlan, PhysicalBWeightSegment, PhysicalBWeightTerm, PreparedRelationAddress,
    SetupContributionGroupInputs, SetupContributionGroupPlan, SetupContributionPlan,
};
pub use stage1_sites::{stage1_child_claims, stage1_range_image};
pub use stage2_sites::stage2_w_eval;
pub use stage3_sites::{stage3_claim, stage3_prefix_eval, stage3_public_slot};
pub use subring_coefficient_packing::PreparedSubringCoefficientPackingPoint;
pub use trace_weight::{
    ensure_trace_stage2_supported, prepare_evaluation_trace_group_parameters,
    EvaluationTraceGroupParameters, EvaluationTraceInputs,
};
pub use transcript_grinding::{
    GrindingReplay, GrindingSumcheckProver, GrindingSumcheckVerifier, ProofAcceptance,
    ProverGrinding, VerifierGrinding,
};

#[cfg(test)]
mod schedule_tests;

#[cfg(test)]
mod proof_size_tests;

#[cfg(test)]
mod witness_scalar_len_tests;

#[cfg(test)]
mod sis_compression_tests;

#[cfg(test)]
mod params_tests;
