//! Proof structures for the Akita protocol.
//!
//! Opening-side notation (paper §§3--5): pre-digit ring openings are `e_folded`;
//! per-block opening digits are `e_hat` (`e_i = ⟨a, f_i⟩`, `ê_i = G^{-1}(e_i)`).
//! The full next-level recursive witness stays `w` (`next_w_commitment`,
//! `final_witness`, `num_w_vectors`, `build_w_coeffs`).

//! Proof, commitment, setup, and claim data shapes.

pub mod batch;
pub mod commitment;
pub mod relation;
pub mod ring_relation;
pub mod scheme;
pub mod setup;
pub mod setup_prefix;
pub mod stage1;
pub mod terminal_witness;

mod containers;
mod direct_witness;
mod hints;
mod levels;
mod shapes;
mod tail_segments;
#[cfg(test)]
mod tests;
mod wire;

pub use crate::opening_claims::{
    sample_public_row_coefficients, OpeningClaims, OpeningClaimsLayout, PointVariableSelection,
    PolynomialGroupClaims, PolynomialGroupLayout, GROUPED_ROOT_DENSE_UNSUPPORTED,
    GROUPED_ROOT_RECURSIVE_SETUP_UNSUPPORTED, GROUPED_ROOT_TIERED_UNSUPPORTED,
    GROUPED_ROOT_UNSUPPORTED,
};
pub use batch::{
    append_batched_commitments_to_transcript, append_claim_values_to_transcript,
    folded_root_supports_opening_shape, padded_scalar_batch_num_vars, prepare_opening_point,
    ring_subfield_packed_extension_opening_point, root_tensor_projection_enabled,
    validate_scalar_point_matches_poly_arity, PreparedOpeningPoint, RingMultiplierOpeningPoint,
};
pub use commitment::{AkitaCommitment, DummyProof, ProverCommitmentRows, RingCommitment};
pub use containers::{FlatDigitBlockIter, FlatDigitBlocks, FlatRingVec, RingSliceSerializer};
pub use direct_witness::{
    segment_typed_witness_shape, CleartextWitnessProof, CleartextWitnessShape,
};
pub use hints::AkitaCommitmentHint;
pub use levels::{
    AkitaBatchedFoldRoot, AkitaBatchedProof, AkitaBatchedRootProof, AkitaIntermediateStage2Proof,
    AkitaLevelProof, AkitaStage1Proof, AkitaStage1StageProof, AkitaStage2Proof,
    AkitaTerminalStage2Proof, ExtensionOpeningReductionProof, SetupSumcheckProof,
    TerminalLevelProof,
};
pub use relation::{generate_y, relation_claim_from_rows, relation_claim_from_rows_extension};
pub use ring_relation::{
    ring_relation_segment_lengths, RingRelationInstance, RingRelationOpeningCounts,
    RingRelationSegmentLengths,
};
pub use scheme::{CommitmentVerifier, OpeningPoints};
pub use setup::{
    derive_public_matrix_flat, sample_public_matrix_seed, validate_public_matrix_matches_seed,
    AkitaExpandedSetup, AkitaSetupSeed, AkitaVerifierSetup, PublicMatrixSeed, SetupMatrixEnvelope,
    MAX_SETUP_MATRIX_FIELD_ELEMENTS,
};
pub use setup_prefix::{
    active_setup_field_len, padded_setup_prefix_len, select_setup_prefix_slot,
    setup_prefix_level_params, setup_prefix_slot_id, SetupPrefixProverRegistry,
    SetupPrefixPublicCommitment, SetupPrefixSlot, SetupPrefixSlotId, SetupPrefixVerifierRegistry,
    SetupPrefixVerifierSlot, SETUP_OFFLOAD_D_SETUP,
};
pub use shapes::{
    AkitaBatchedProofShape, AkitaProofStepShape, AkitaStage1StageShape,
    ExtensionOpeningReductionShape, LevelProofShape, SetupProductSumcheckShape,
    TerminalLevelProofShape, SETUP_SUMCHECK_DEGREE,
};
pub use stage1::{
    absorb_interstage_claims, combine_polys, eval_poly, linear_combination,
    range_check_eval_from_s, reorder_stage1_coords, stage1_interstage_batch_weights,
    stage1_leaf_coeffs, stage1_stage_count, stage1_tree_product_stage_arities,
    stage1_tree_stage_shapes, validate_stage1_tree_basis,
};
pub use tail_segments::{
    build_segment_typed_witness, decode_terminal_z_golomb_payload, e_folded_segment_bytes,
    emit_witness_planes_block_inner, emit_witness_z_folded_planes_inner,
    expand_segment_typed_to_i8_digits, segment_typed_witness_upper_bound_bytes,
    segment_typed_z_payload_bytes, tail_golomb_rice_z_params, tail_segment_layout,
    tail_segment_multiplicities_from_layout, terminal_golomb_grind_tail_t_vectors,
    validate_segment_typed_z_payload, z_fold_decoded_from_segment,
    z_fold_encoding_stats_from_segment, SegmentTypedWitness, SegmentTypedWitnessShape,
    TailSegmentLayout,
};
pub use terminal_witness::{
    i8_digits_to_bytes, terminal_e_hat_bytes_from_blocks, terminal_witness_segment_layout,
    terminal_witness_segment_layout_from_counts, terminal_witness_transcript_parts,
    RelationOnlyStage2Inputs, TerminalWitnessSegmentLayout, TerminalWitnessTranscriptParts,
};

use crate::EXTENSION_OPENING_REDUCTION_DEGREE;
use akita_algebra::CyclotomicRing;
use akita_field::AkitaError;
use akita_field::{CanonicalField, FieldCore, HalvingField};
use akita_serialization::{AkitaDeserialize, AkitaSerialize, DEFAULT_MAX_SEQUENCE_LEN};
use akita_serialization::{Compress, SerializationError};
use akita_serialization::{Valid, Validate};
use akita_sumcheck::EqFactoredSumcheckProof;
use akita_sumcheck::{
    uniform_sumcheck_shape, EqFactoredSumcheckProofShape, SumcheckProof, SumcheckProofShape,
};
use akita_transcript::Transcript;
use std::io::{Read, Write};
use std::marker::PhantomData;

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
