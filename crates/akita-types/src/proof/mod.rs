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
pub mod relation_range_image;
mod relation_weight_event;
pub mod ring_relation;
pub mod scheme;
pub mod setup;
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
    OpeningClaims, PolynomialGroupClaims,
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
    relation_rhs_row_count, relation_row_weight,
};
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
    PublicMatrixDerivation, MAX_GENERIC_SETUP_DECODE_FIELD_ELEMENTS,
};
pub use setup_prefix::{
    setup_prefix_coverage_eval_len, SetupPrefixPublicCommitment, SetupPrefixVerifierRegistry,
    SetupPrefixVerifierSlot,
};
pub use shapes::{canonical_extension_opening_reduction_shape, ExtensionOpeningReductionShape};
pub use stage1::DigitRangeEqualityPoint;
pub use tail_segments::{
    build_terminal_response_from_payload, decode_terminal_z_golomb_payload, TerminalResponse,
};
pub use witness_emission::{emit_witness_e_planes, emit_witness_t_planes, WitnessCoefficientSink};

use crate::EXTENSION_OPENING_REDUCTION_DEGREE;
use akita_algebra::CyclotomicRing;
use akita_error::AkitaError;
use akita_serialization::{AkitaDeserialize, AkitaSerialize};
use akita_serialization::{Compress, SerializationError};
use akita_serialization::{Valid, Validate};
use akita_sumcheck::uniform_sumcheck_shape;
#[cfg(test)]
use jolt_field::CanonicalEncoding;
use jolt_field::Field;
use std::io::{Read, Write};
