//! Root and suffix fold verifier replay for Akita proofs.
//!
//! This module owns the shared per-fold replay engine plus path-specific prep
//! in `verify`, `root_fold`, and `suffix`. Schedule/config dispatch stays with
//! the scheme crate until the verifier-facing config boundary is extracted.

use super::validate_level_dispatch;
#[cfg(not(feature = "zk"))]
mod extension_opening_reduction;
mod verify;
#[cfg(feature = "zk")]
mod zk;
use crate::protocol::ring_switch::{
    ring_switch_verifier, ring_switch_verifier_terminal, RingSwitchReplay, RingSwitchVerifyOutput,
};
use crate::stages::stage1::{
    derive_stage1_challenges, validate_fold_grind_nonce, AkitaStage1Verifier,
};
use crate::stages::stage2::{stage2_cleartext_oracle, AkitaStage2Verifier, Stage2WitnessOracle};
use crate::stages::SetupSumcheckVerifier;
use akita_algebra::CyclotomicRing;
#[cfg(feature = "zk")]
use akita_algebra::EqPolynomial;
use akita_field::{
    AkitaError, CanonicalField, ExtField, FieldCore, FrobeniusExtField, FromPrimitiveInt,
    HalvingField, PseudoMersenneField, RandomSampling,
};
#[cfg(feature = "zk")]
use akita_r1cs::{
    lift_hiding_witness, zk_ext_mask_lc, zk_ext_mask_lc_at, zk_masked_compressed_round_claim_mask,
    zk_push_linear_zero, zk_row_masks_from_column_masks, ZkR1csLinearCombination,
    ZkRelationAccumulator,
};
use akita_serialization::AkitaSerialize;
#[cfg(not(feature = "zk"))]
use akita_sumcheck::SumcheckInstanceVerifierExt;
#[cfg(feature = "zk")]
use akita_sumcheck::ZkSumcheckInstanceVerifierExt;
use akita_transcript::labels::{
    ABSORB_COMMITMENT, ABSORB_EVALUATION_CLAIMS, ABSORB_STAGE2_NEXT_W_EVAL,
    ABSORB_SUMCHECK_S_CLAIM, ABSORB_TERMINAL_E_HAT, CHALLENGE_SUMCHECK_BATCH,
    CHALLENGE_SUMCHECK_ROUND,
};
#[cfg(feature = "zk")]
use akita_transcript::labels::{ABSORB_SUMCHECK_CLAIM, ABSORB_ZK_HIDING_COMMITMENT};
use akita_transcript::{append_ext_field, sample_ext_challenge, Transcript};
#[cfg(not(feature = "zk"))]
use akita_types::derive_tensor_extension_opening_claim_from_partials;
#[cfg(feature = "zk")]
use akita_types::EXTENSION_OPENING_REDUCTION_DEGREE;
use akita_types::{
    append_batched_commitments_to_transcript, append_claim_values_to_transcript,
    append_opening_batch_shape_to_transcript, batched_eval_target_from_opening_batch,
    build_trace_claim_root, ensure_trace_stage2_supported, flatten_batched_commitment_rows,
    generate_y, prepare_opening_point, relation_claim_from_rows_extension, reorder_stage1_coords,
    ring_subfield_packed_extension_opening_point, root_trace_block_opening,
    sample_public_row_coefficients, schedule_num_fold_levels, scheduled_next_level_params,
    stage2_trace_coeff, tensor_equality_factor_eval_at_point, trace_terms_recursive,
    trace_weight_layout_from_segment, w_ring_element_count_with_counts, AkitaBatchedRootProof,
    AkitaLevelProof, AkitaStage1Proof, AkitaStage2Proof, AkitaVerifierSetup, BasisMode, BlockOrder,
    CleartextWitnessProof, ExecutionSchedule, ExtensionOpeningReductionProof, FlatRingVec,
    FoldLinfProtocolBinding, FpExtEncoding, LevelParams, MRowLayout, OpeningBatch,
    PreparedOpeningPoint, RelationOnlyStage2Inputs, RingCommitment, RingMultiplierOpeningPoint,
    RingOpeningPoint, RingRelationInstance, Schedule, SetupContributionMode, SetupSumcheckProof,
    TerminalWitnessSegmentLayout, TerminalWitnessTranscriptParts, TraceClaim,
};
use akita_types::{
    tensor_opening_split, tensor_reduction_claim_from_rows, tensor_row_partials_from_columns,
};
#[cfg(not(feature = "zk"))]
use extension_opening_reduction::verify_extension_opening_reduction_sumcheck;
#[cfg(feature = "zk")]
use zk::verify_zk_hiding_commitment;

mod fold;
mod root_fold;
mod suffix;
use root_fold::verify_root;

pub use verify::{batched_verify, batched_verify_shaped, batched_verify_shaped_root_direct};

pub(in crate::protocol::core) use fold::{
    verify_fold, verify_fold_eor, FoldEorReplay, PreparedFoldReplay,
};

fn prepare_terminal_witness_replay<F, T>(
    transcript: &mut T,
    final_witness: &CleartextWitnessProof<F>,
    final_w_len: usize,
    layout: TerminalWitnessSegmentLayout,
) -> Result<TerminalWitnessTranscriptParts, AkitaError>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
{
    if final_witness.num_elems() != final_w_len {
        return Err(AkitaError::InvalidProof);
    }
    let parts = final_witness.terminal_transcript_parts(layout)?;
    transcript.absorb_and_record_bytes(ABSORB_TERMINAL_E_HAT, &parts.e_hat);
    Ok(parts)
}
