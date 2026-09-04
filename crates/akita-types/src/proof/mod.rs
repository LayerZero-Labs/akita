//! Proof structures for the Akita protocol.
//!
//! Opening-side notation (paper §§3--5): pre-digit ring openings are `e_folded`;
//! per-block opening digits are `e_hat` (`e_i = ⟨a, f_i⟩`, `ê_i = G^{-1}(e_i)`).
//! The full next-level recursive witness stays `w` (`next_w_commitment`,
//! `terminal_response`, `num_w_vectors`, `build_w_coeffs`).

//! Proof, commitment, setup, and claim data shapes.

pub mod batch;
mod coefficient_functional;
mod coefficient_packing_relation;
pub mod commitment;
pub mod compression_relation_weights;
mod fold_challenges;
pub mod relation;
pub mod relation_address;
pub mod relation_range_image;
mod relation_weight_event;
pub mod ring_relation;
pub mod scheme;
pub mod setup;
pub mod setup_envelope;
pub mod setup_prefix;
pub mod stage1;
pub mod terminal_witness;

mod containers;
mod hints;
mod levels;
mod shapes;
mod tail_segments;
#[cfg(test)]
mod tests;
mod wire;
mod witness_emission;

/// Maximum coefficients accepted from a self-describing commitment artifact.
///
/// This guards generic untrusted allocation. It is not a bound on the public
/// setup stream or on a caller-validated setup package.
pub const MAX_UNTRUSTED_COMMITMENT_COEFFICIENTS: usize = 1 << 26;

pub use crate::opening_claims::{
    sample_row_coefficients, GroupBatchStatement, OpeningClaims, OpeningClaimsLayout,
    PolynomialGroupClaims, PolynomialGroupLayout,
};
pub use batch::{
    append_batched_commitments_to_transcript, append_claim_values_to_transcript,
    folded_root_supports_opening_shape, prepare_opening_point,
    ring_subfield_packed_extension_opening_point, validate_batched_inputs, PreparedOpeningPoint,
    PreparedRingMultiplier, RingMultiplierOpeningPoint, SubfieldMultiplierOpeningPoint,
};
pub use coefficient_functional::ReducedCoefficientFunctional;
pub use coefficient_packing_relation::{
    prepare_coefficient_packing_batch_semantics,
    prepare_coefficient_packing_verifier_batch_semantics, CoefficientPackingBatchSemanticInputs,
    CoefficientPackingBatchSemantics, CoefficientPackingCompactFactors,
    CoefficientPackingGroupSemantics, CoefficientPackingStage2Segment,
    CoefficientPackingStage2Source, CoefficientPackingStage2Term, CoefficientPackingStage2Terms,
    CoefficientPackingVerifierBatchSemantics, CoefficientPackingVerifierGroupSemantics,
};
pub use commitment::{
    AkitaCommitment, Commitment, CommittedGroup, DummyProof, ProverCommitmentRows, RingCommitment,
};
pub use compression_relation_weights::{
    build_compression_relation_weights, build_reduced_compression_relation_weights,
    evaluate_reduced_compression_map, CompressionRelationWeights, NegativeBinarySupport,
    ReducedCompressionRelationWeights,
};
pub use containers::{
    append_flat_coefficients, DigitBlockIter, DigitBlocks, FlatCoeffSerializer, RingVec, RingView,
};
pub use fold_challenges::{draw_group_fold_challenges, GroupFoldChallenges};
pub use hints::AkitaCommitmentHint;
pub use levels::{
    AkitaBatchedProof, AkitaStage1Proof, AkitaStage1StageProof, AkitaStage2Proof,
    ExtensionOpeningReductionProof, FoldLevelProof, NextWitnessBinding, PhysicalL2NormProof,
    SetupSumcheckProof, TerminalLevelProof,
};
pub use relation::{
    assemble_compressed_relation_rhs, assemble_relation_rhs,
    compression_relation_claim_from_rhs_extension, generate_relation_rhs,
    relation_claim_from_compressed_rhs_extension, relation_claim_from_layout_extension,
    relation_claim_from_rows, relation_claim_from_rows_extension, relation_rhs_coeff_len,
    relation_rhs_row_count, relation_row_weight, RelationGroupRows, RelationRhsLayout,
    RelationRowFamily, RelationRowGeometry, RelationWitnessGeometry,
};
pub use relation_address::{CompressionRelationAddressGeometry, RelationAddressGeometry};
pub use relation_range_image::{
    reconstruct_l2_sq_from_gram, PhysicalResponsePlan, RelationRangeImageGroupPlan,
    RelationRangeImagePlan,
};
pub use relation_weight_event::{RelationWeightContribution, RelationWeightEvent};
pub use ring_relation::{
    ring_relation_segment_lengths, CoefficientPackingChallenges, RingRelationGroupOpening,
    RingRelationGroupOpeningView, RingRelationInstance, RingRelationOpeningCounts,
    RingRelationSegmentLengths,
};
pub use scheme::{CommitmentVerifier, OpeningPoints};
pub use setup::{
    derive_public_matrix_prefix, sample_akita_setup_seed, validate_public_matrix_matches_seed,
    AkitaExpandedSetup, AkitaSetupDescriptor, AkitaSetupSeed, AkitaVerifierSetup,
    PublicMatrixDerivation, SetupMatrixCapacity, MAX_GENERIC_SETUP_DECODE_FIELD_ELEMENTS,
};
pub use setup_envelope::{
    accumulate_matrix_field_elements_for_level, accumulate_terminal_matrix_field_elements,
    commit_only_setup_field_elements, setup_matrix_capacity_for_schedule,
    setup_matrix_field_elements_for_schedule, setup_prefix_slot_field_elements,
    verifier_setup_matrix_capacity_for_schedule,
};
pub use setup_prefix::{
    active_setup_field_len, padded_setup_prefix_len, scheduled_setup_prefix,
    setup_prefix_coverage_eval_len, setup_prefix_precommitted_params, suffix_opening_layout,
    validate_setup_prefix_domain, SetupPrefixProverRegistry, SetupPrefixPublicCommitment,
    SetupPrefixSlot, SetupPrefixSlotId, SetupPrefixVerifierRegistry, SetupPrefixVerifierSlot,
    SETUP_PREFIX_CONTENT_TAG,
};
pub use shapes::{
    canonical_extension_opening_reduction_shape, canonical_proof_shape, AkitaBatchedProofShape,
    AkitaStage1StageShape, ExtensionOpeningReductionShape, LevelProofShape,
    NextWitnessBindingShape, PhysicalL2NormProofWireShape, SetupProductSumcheckShape,
    TerminalLevelProofShape, SETUP_SUMCHECK_DEGREE,
};
pub use stage1::{
    append_digit_range_child_claims, DigitRangeEqualityPoint, DigitRangePlan, FlatBooleanDomain,
};
pub use tail_segments::{
    build_terminal_response, build_terminal_response_from_groups, decode_terminal_z_golomb_payload,
    raw_field_segment_bytes, tail_segment_multiplicities_from_layout,
    tail_segment_multiplicities_from_layout_for_params, terminal_response_upper_bound_bytes,
    terminal_response_z_payload_bytes, validate_terminal_response_z_payload,
    TailSegmentGroupLayout, TailSegmentLayout, TerminalResponse, TerminalResponseGroupParts,
    TerminalResponseShape,
};
pub use terminal_witness::TerminalWitnessTranscriptParts;
pub use witness_emission::{
    emit_witness_e_planes, emit_witness_r_planes, emit_witness_t_planes, emit_witness_z_planes,
    WitnessCoefficientSink,
};

use crate::EXTENSION_OPENING_REDUCTION_DEGREE;
use akita_algebra::CyclotomicRing;
use akita_error::AkitaError;
use akita_serialization::{AkitaDeserialize, AkitaSerialize, DEFAULT_MAX_SEQUENCE_LEN};
use akita_serialization::{Compress, SerializationError};
use akita_serialization::{Valid, Validate};
use akita_sumcheck::EqFactoredSumcheckProof;
use akita_sumcheck::{
    uniform_sumcheck_shape, EqFactoredSumcheckProofShape, SumcheckProof, SumcheckProofShape,
};
use akita_transcript::Transcript;
use jolt_field::{CanonicalEncoding, ExtField, Field};
use std::io::{Read, Write};

pub(super) const MAX_PROOF_SHAPE_SEQUENCE_LEN: usize = 1 << 12;

pub(super) fn checked_shape_len(len: usize) -> Result<(), SerializationError> {
    if len > DEFAULT_MAX_SEQUENCE_LEN {
        return Err(SerializationError::LengthLimitExceeded {
            len: u64::try_from(len).unwrap_or(u64::MAX),
            max: DEFAULT_MAX_SEQUENCE_LEN,
        });
    }
    Ok(())
}

pub(super) fn checked_shape_sequence_len(len: usize) -> Result<(), SerializationError> {
    if len > MAX_PROOF_SHAPE_SEQUENCE_LEN {
        return Err(SerializationError::LengthLimitExceeded {
            len: u64::try_from(len).unwrap_or(u64::MAX),
            max: MAX_PROOF_SHAPE_SEQUENCE_LEN,
        });
    }
    Ok(())
}

pub(super) fn reserve_shape_len<T>(vec: &mut Vec<T>, len: usize) -> Result<(), SerializationError> {
    checked_shape_len(len)?;
    vec.try_reserve_exact(len)
        .map_err(|_| SerializationError::InvalidData("shape-backed allocation failed".to_string()))
}
