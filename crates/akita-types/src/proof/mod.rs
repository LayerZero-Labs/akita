//! Proof structures for the Akita protocol.

//! Proof, commitment, setup, and claim data shapes.

pub mod batch;
pub mod commitment;
pub mod incidence;
pub mod relation;
pub mod ring_relation;
pub mod scheme;
pub mod setup;
pub mod stage1;
pub mod terminal_witness;

mod containers;
mod direct_witness;
mod hints;
mod levels;
mod shapes;
#[cfg(test)]
mod tests;
mod wire;

pub use batch::{
    append_batched_commitments_to_transcript, append_claim_points_to_transcript,
    append_claim_values_to_transcript, append_prepared_root_opening_point, checked_total_claims,
    flatten_batched_commitment_rows, folded_root_supports_opening_shape,
    prepare_recursive_opening_point_ext, prepare_root_opening_point,
    prepare_root_opening_point_ext, ring_inner_product_with_extension_weights,
    ring_subfield_packed_extension_opening_point, root_tensor_projection_enabled,
    validate_batched_inputs, PreparedRecursiveOpeningPoint, PreparedRootOpeningPoint,
    RingMultiplierOpeningPoint,
};
pub use commitment::{AkitaCommitment, DummyProof, RingCommitment};
#[cfg(feature = "zk")]
pub use containers::ZkHidingProof;
pub use containers::{FlatDigitBlockIter, FlatDigitBlocks, FlatRingVec, RingSliceSerializer};
pub use direct_witness::{CleartextWitnessProof, CleartextWitnessShape, PackedDigits};
pub use hints::AkitaCommitmentHint;
pub use incidence::{
    append_claim_incidence_shape_to_transcript, sample_public_row_coefficients,
    verifier_claims_to_incidence, ClaimIncidence, ClaimIncidenceLimits, ClaimIncidenceSummary,
    CommitmentRouting, IncidenceClaim, PublicOpeningRow,
};
pub use levels::{
    AkitaBatchedFoldRoot, AkitaBatchedProof, AkitaBatchedRootProof, AkitaLevelProof,
    AkitaProofStep, AkitaStage1Proof, AkitaStage1StageProof, AkitaStage2Proof,
    ExtensionOpeningReductionProof, TerminalLevelProof,
};
pub use relation::{relation_claim_from_rows, relation_claim_from_rows_extension};
pub use ring_relation::{
    ring_column_z_first, ring_relation_segment_layout_for_opening_shape, RingRelationInstance,
    RingRelationSegmentLayout,
};
pub use scheme::{CommitmentVerifier, CommittedOpenings, OpeningPoints, VerifierClaims};
pub use setup::{
    derive_public_matrix_flat, sample_public_matrix_seed, validate_public_matrix_matches_seed,
    AkitaExpandedSetup, AkitaSetupSeed, AkitaVerifierSetup, PublicMatrixSeed, SetupMatrixEnvelope,
    MAX_SETUP_MATRIX_FIELD_ELEMENTS,
};
#[cfg(feature = "zk")]
pub use setup::{derive_zk_b_matrix, derive_zk_d_matrix};
pub use shapes::{
    AkitaBatchedProofShape, AkitaProofStepShape, AkitaStage1StageShape,
    ExtensionOpeningReductionShape, LevelProofShape, TerminalLevelProofShape,
};
pub use stage1::{
    absorb_interstage_claims, combine_polys, eval_poly, linear_combination,
    range_check_eval_from_s, reorder_stage1_coords, stage1_interstage_batch_weights,
    stage1_leaf_coeffs, stage1_stage_count, stage1_tree_product_stage_arities,
    stage1_tree_stage_shapes, validate_stage1_tree_basis,
};
pub use terminal_witness::{
    i8_digits_to_bytes, terminal_w_hat_bytes_from_blocks, terminal_witness_segment_layout,
    terminal_witness_segment_layout_from_counts, terminal_witness_transcript_parts,
    RelationOnlyStage2Inputs, TerminalWitnessSegmentLayout, TerminalWitnessTranscriptParts,
};

use akita_algebra::CyclotomicRing;
use akita_field::AkitaError;
use akita_field::{CanonicalField, FieldCore, FromPrimitiveInt};
use akita_serialization::{AkitaDeserialize, AkitaSerialize, DEFAULT_MAX_SEQUENCE_LEN};
use akita_serialization::{Compress, SerializationError};
use akita_serialization::{Valid, Validate};
pub use akita_sumcheck::EXTENSION_OPENING_REDUCTION_DEGREE;
use akita_sumcheck::{uniform_sumcheck_shape, EqFactoredSumcheckProofShape, SumcheckProofShape};
#[cfg(not(feature = "zk"))]
use akita_sumcheck::{EqFactoredSumcheckProof, SumcheckProof};
#[cfg(feature = "zk")]
use akita_sumcheck::{EqFactoredSumcheckProofMasked, SumcheckProofMasked};
use akita_transcript::Transcript;
use std::io::{Read, Write};
use std::marker::PhantomData;

pub(super) fn checked_shape_len(len: usize) -> Result<(), SerializationError> {
    if len > DEFAULT_MAX_SEQUENCE_LEN {
        return Err(SerializationError::LengthLimitExceeded {
            len: u64::try_from(len).unwrap_or(u64::MAX),
            max: DEFAULT_MAX_SEQUENCE_LEN,
        });
    }
    Ok(())
}

pub(super) fn reserve_shape_len<T>(vec: &mut Vec<T>, len: usize) -> Result<(), SerializationError> {
    checked_shape_len(len)?;
    vec.try_reserve_exact(len)
        .map_err(|_| SerializationError::InvalidData("shape-backed allocation failed".to_string()))
}
