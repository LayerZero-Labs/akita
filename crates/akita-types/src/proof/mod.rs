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
pub mod relation_matrix_cols;
pub mod ring_relation;
pub mod scheme;
pub mod setup;
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
pub use crate::opening_claims::{
    sample_public_row_coefficients, should_reject_multi_group_root, OpeningClaims,
    OpeningClaimsLayout, PointVariableSelection, PolynomialGroupClaims, PolynomialGroupLayout,
    MULTI_GROUP_ROOT_DENSE_UNSUPPORTED,
};
pub use batch::{
    append_batched_commitments_to_transcript, append_claim_values_to_transcript,
    folded_root_supports_opening_shape, padded_scalar_batch_num_vars, prepare_opening_point,
    ring_subfield_packed_extension_opening_point, root_tensor_projection_enabled,
    validate_batched_inputs, validate_scalar_point_matches_poly_arity, PreparedOpeningPoint,
    RingMultiplierOpeningPoint,
};
pub use commitment::{
    AkitaCommitment, Commitment, DummyProof, ProverCommitmentRows, RingCommitment,
};
pub use containers::{
    append_flat_coefficients, DigitBlockIter, DigitBlocks, FlatCoeffSerializer, RingVec, RingView,
};
pub use hints::AkitaCommitmentHint;
pub use levels::{
    AkitaBatchedProof, AkitaStage1Proof, AkitaStage1StageProof, AkitaStage2Proof,
    ExtensionOpeningReductionProof, FoldLevelProof, NextWitnessBinding, SetupSumcheckProof,
    TerminalLevelProof,
};
pub use relation::{
    assemble_relation_rhs, evaluation_trace_row_weight, generate_relation_rhs,
    relation_claim_from_layout_extension, relation_claim_from_rows,
    relation_claim_from_rows_extension, relation_rhs_coeff_len, relation_rhs_layout_for,
    relation_rhs_row_count, RelationGroupRows, RelationRhsLayout,
};
pub use relation_matrix_cols::{
    compute_relation_matrix_col_evals, compute_relation_weight_evals, eval_relation_weight_at_point,
};
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
    setup_prefix_precommitted_params, setup_prefix_slot_id, SetupPrefixProverRegistry,
    SetupPrefixPublicCommitment, SetupPrefixSlot, SetupPrefixSlotId, SetupPrefixVerifierRegistry,
    SetupPrefixVerifierSlot, SETUP_OFFLOAD_D_SETUP, SETUP_OFFLOAD_MIN_PREFIX_FIELD_LEN,
};
pub use shapes::{
    AkitaBatchedProofShape, AkitaStage1StageShape, ExtensionOpeningReductionShape, LevelProofShape,
    NextWitnessBindingShape, SetupProductSumcheckShape, TerminalLevelProofShape,
    SETUP_SUMCHECK_DEGREE,
};
pub use stage1::{
    absorb_interstage_claims, combine_polys, eval_poly, linear_combination,
    range_check_eval_from_s, reorder_stage1_coords, stage1_interstage_batch_weights,
    stage1_leaf_coeffs, stage1_stage_count, stage1_tree_product_stage_arities,
    stage1_tree_stage_shapes, validate_stage1_tree_basis,
};
pub use tail_segments::{
    build_segment_typed_witness_from_groups, decode_terminal_z_golomb_payload,
    decode_terminal_z_golomb_payload_with_cap, emit_witness_e_planes, emit_witness_r_planes,
    emit_witness_t_planes, emit_witness_z_planes, raw_field_segment_bytes,
    segment_typed_witness_upper_bound_bytes, segment_typed_z_payload_bytes,
    tail_golomb_rice_z_params, tail_segment_multiplicities_from_layout,
    tail_segment_multiplicities_from_layout_for_params, terminal_golomb_grind_tail_t_vectors,
    validate_segment_typed_z_payload, z_fold_decoded_from_segment,
    z_fold_encoding_stats_from_segment, SegmentTypedWitness, SegmentTypedWitnessGroupParts,
    SegmentTypedWitnessShape, TailSegmentGroupLayout, TailSegmentLayout,
};
pub use terminal_witness::TerminalWitnessTranscriptParts;

use crate::EXTENSION_OPENING_REDUCTION_DEGREE;
use akita_algebra::CyclotomicRing;
use akita_field::AkitaError;
use akita_field::{CanonicalField, FieldCore};
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
