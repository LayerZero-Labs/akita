//! Root and suffix fold verifier replay for Akita proofs.
//!
//! This module owns the shared per-fold replay engine plus path-specific prep
//! in `verify`, `root_fold`, and `suffix`. Schedule/config dispatch stays with
//! the scheme crate until the verifier-facing config boundary is extracted.

use super::validate_level_dispatch;
mod extension_opening_reduction;
mod verify;
use crate::protocol::ring_switch::{
    ring_switch_verifier, ring_switch_verifier_terminal, RingSwitchReplay, RingSwitchVerifyOutput,
};
use crate::stages::stage1::{
    derive_stage1_challenges, validate_fold_grind_nonce, AkitaStage1Verifier,
};
use crate::stages::stage2::{stage2_cleartext_oracle, AkitaStage2Verifier, Stage2WitnessOracle};
use crate::stages::SetupSumcheckVerifier;
use akita_algebra::CyclotomicRing;
use akita_field::{
    AkitaError, CanonicalField, ExtField, FieldCore, FrobeniusExtField, FromPrimitiveInt,
    HalvingField, PseudoMersenneField, RandomSampling,
};
use akita_serialization::AkitaSerialize;
use akita_sumcheck::SumcheckInstanceVerifierExt;
use akita_transcript::labels::{
    ABSORB_COMMITMENT, ABSORB_EVALUATION_CLAIMS, ABSORB_STAGE2_NEXT_W_EVAL,
    ABSORB_STAGE3_NEXT_W_EVAL, ABSORB_SUMCHECK_S_CLAIM, ABSORB_TERMINAL_E_HAT,
    CHALLENGE_SUMCHECK_BATCH, CHALLENGE_SUMCHECK_ROUND,
};
use akita_transcript::{append_ext_field, sample_ext_challenge, Transcript};
use akita_types::derive_tensor_extension_opening_claim_from_partials;
use akita_types::{
    append_claim_values_to_transcript, assemble_relation_y, build_trace_claim_root,
    ensure_trace_stage2_supported, prepare_opening_point,
    relation_claim_from_rows_extension_at_dims, reorder_stage1_coords,
    ring_subfield_packed_extension_opening_point, root_trace_block_opening,
    sample_public_row_coefficients, schedule_num_fold_levels, scheduled_next_level_params,
    stage2_trace_coeff, tensor_equality_factor_eval_at_point, trace_terms_recursive,
    trace_weight_layout_from_segment, w_ring_element_count_with_counts_for_layout,
    AkitaBatchedRootProof, AkitaLevelProof, AkitaStage1Proof, AkitaStage2Proof, AkitaVerifierSetup,
    BasisMode, BlockOrder, CleartextWitnessProof, ExecutionSchedule,
    ExtensionOpeningReductionProof, FoldLinfProtocolBinding, FpExtEncoding, LevelParams,
    MRowLayout, OpeningClaims, OpeningClaimsLayout, PointVariableSelection, PolynomialGroupClaims,
    PreparedOpeningPoint, RelationOnlyStage2Inputs, RelationYLayout, RingCommitment,
    RingMultiplierOpeningPoint, RingOpeningPoint, RingRelationInstance, RingVec, Schedule,
    SetupContributionMode, SetupSumcheckProof, TerminalWitnessSegmentLayout,
    TerminalWitnessTranscriptParts, TraceClaim,
};
use akita_types::{
    tensor_opening_split, tensor_reduction_claim_from_rows, tensor_row_partials_from_columns,
};
use extension_opening_reduction::verify_extension_opening_reduction_sumcheck;

mod fold;
mod root_fold;
mod suffix;
use root_fold::verify_root;

pub use verify::batched_verify;

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
