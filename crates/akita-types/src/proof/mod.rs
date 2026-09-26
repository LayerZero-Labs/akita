//! Proof structures for the Akita protocol.
//!
//! Opening-side notation (paper §§3--5): pre-digit ring openings are `e_folded`;
//! per-block opening digits are `e_hat` (`e_i = ⟨a, f_i⟩`, `ê_i = G^{-1}(e_i)`).
//! The full next-level recursive witness stays `w` (`next_w_commitment`,
//! `terminal_response`, `num_w_vectors`, `build_w_coeffs`).

//! Proof, commitment, setup, and claim data shapes.

pub mod batch;
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

mod containers;

#[cfg(test)]
mod levels;
mod shapes;
mod tail_segments;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod wire;
mod witness_emission;

/// Maximum coefficients accepted from a self-describing commitment artifact.
///
/// This guards generic untrusted allocation. It is not a bound on the public
/// setup stream or on a caller-validated setup package.
pub const MAX_UNTRUSTED_COMMITMENT_COEFFICIENTS: usize = 1 << 26;

pub use crate::opening_claims::{
    sample_row_coefficients_native, verify_row_coefficients_native, GroupBatchStatement,
    OpeningClaims, OpeningClaimsLayout, PolynomialGroupClaims, PolynomialGroupLayout,
};
pub use batch::{
    prepare_opening_point, ring_subfield_packed_extension_opening_point, PreparedOpeningPoint,
    PreparedRingMultiplier, RingMultiplierOpeningPoint, SubfieldMultiplierOpeningPoint,
};
#[cfg(any(test, feature = "test-support"))]
pub use coefficient_packing_relation::{
    coefficient_packing_fixture, coefficient_packing_multigroup_fixture, CoefficientPackingFixture,
    CoefficientPackingMultigroupFixture,
};
pub use coefficient_packing_relation::{
    prepare_coefficient_packing_batch_semantics, validate_coefficient_packing_batch_groups,
    CoefficientPackingBatchSemanticInputs, CoefficientPackingBatchSemantics,
    CoefficientPackingGroupSemantics, CoefficientPackingStage2Segment,
    CoefficientPackingStage2Source, CoefficientPackingStage2Term, CoefficientPackingStage2Terms,
    ValidatedCoefficientPackingGroup,
};
pub use commitment::{Commitment, CommittedGroup};
pub use compression_relation_weights::{
    build_compression_relation_weights, build_reduced_compression_relation_weights,
    CompressionRelationWeights, NegativeBinarySupport, ReducedCompressionRelationWeights,
};
pub use containers::{DigitBlockIter, DigitBlocks, RingVec, RingView};
pub use fold_challenges::{draw_group_fold_challenges, GroupFoldChallenges, OpeningFamily};

#[cfg(test)]
pub(crate) use levels::{
    AkitaStage1Proof, AkitaStage1StageProof, AkitaStage2Proof, ExtensionOpeningReductionProof,
    FoldLevelProof, NextWitnessBinding, PhysicalL2NormProof, SetupSumcheckProof,
    TerminalLevelProof,
};
pub use relation::{
    assemble_compressed_relation_rhs, assemble_relation_rhs, generate_relation_rhs,
    relation_claim_from_compressed_rhs_extension, relation_claim_from_layout_extension,
    relation_claim_from_rows, relation_claim_from_rows_extension, relation_rhs_coeff_len,
    relation_rhs_row_count, relation_row_weight, RelationGroupRows, RelationRhsLayout,
    RelationRowFamily, RelationRowGeometry, RelationWitnessGeometry,
};
pub use relation_address::{CompressionRelationAddressGeometry, RelationAddressGeometry};
pub use relation_range_image::{
    batch_l2_virtual_evaluations, reconstruct_l2_sq_from_gram, PhysicalResponsePlan,
    RelationRangeImageGroupPlan, RelationRangeImagePlan,
};
pub use relation_weight_event::{RelationWeightContribution, RelationWeightEvent};
pub use ring_relation::{
    CoefficientPackingChallenges, RingRelationGroupOpening, RingRelationGroupOpeningView,
    RingRelationInstance,
};
pub use scheme::OpeningPoints;
pub use setup::{
    derive_public_matrix_prefix, sample_akita_setup_seed, validate_public_matrix_matches_seed,
    AkitaExpandedSetup, AkitaSetupDescriptor, AkitaSetupSeed, AkitaVerifierSetup,
    PublicMatrixDerivation, SetupMatrixCapacity, MAX_GENERIC_SETUP_DECODE_FIELD_ELEMENTS,
};
pub use setup_envelope::{
    accumulate_matrix_field_elements_for_level, accumulate_terminal_matrix_field_elements,
    commit_only_setup_field_elements, commitment_execution_setup_field_elements,
    setup_matrix_capacity_for_schedule, setup_matrix_field_elements_for_schedule,
    setup_prefix_slot_field_elements, verifier_setup_matrix_capacity_for_schedule,
    CommitmentSetupMatrixShape,
};
pub use setup_prefix::{
    active_setup_field_len, padded_setup_prefix_len, scheduled_setup_prefix,
    setup_prefix_coverage_eval_len, setup_prefix_precommitted_params, suffix_opening_layout,
    validate_setup_prefix_domain, SetupPrefixPublicCommitment, SetupPrefixSlotId,
    SetupPrefixVerifierRegistry, SetupPrefixVerifierSlot, SETUP_PREFIX_CONTENT_TAG,
};
pub use shapes::{
    canonical_extension_opening_reduction_shape, AkitaStage1StageShape,
    ExtensionOpeningReductionShape, PhysicalL2NormProofWireShape, SETUP_SUMCHECK_DEGREE,
};
pub use stage1::{DigitRangeEqualityPoint, DigitRangePlan, FlatBooleanDomain};
pub use tail_segments::{
    build_terminal_response_from_payload, decode_terminal_z_golomb_payload,
    terminal_response_upper_bound_bytes, TailSegmentGroupLayout, TailSegmentLayout,
    TerminalResponse, TerminalResponseShape,
};
pub use witness_emission::{emit_witness_e_planes, emit_witness_t_planes, WitnessCoefficientSink};

use crate::EXTENSION_OPENING_REDUCTION_DEGREE;
use akita_algebra::CyclotomicRing;
use akita_error::AkitaError;
use akita_serialization::{AkitaDeserialize, AkitaSerialize, DEFAULT_MAX_SEQUENCE_LEN};
use akita_serialization::{Compress, SerializationError};
use akita_serialization::{Valid, Validate};
use akita_sumcheck::uniform_sumcheck_shape;
#[cfg(test)]
use jolt_field::CanonicalEncoding;
use jolt_field::Field;
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
