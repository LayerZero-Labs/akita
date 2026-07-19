//! Root and suffix fold verifier replay for Akita proofs.
//!
//! This module owns the shared per-fold replay engine plus path-specific prep
//! in `verify`, `root_fold`, and `suffix`. Schedule/config dispatch stays with
//! the scheme crate until the verifier-facing config boundary is extracted.

mod extension_opening_reduction;
mod verify;
use crate::protocol::ring_switch::{
    ring_switch_verifier, RingSwitchReplay, RingSwitchVerifyOutput,
};
use crate::stages::stage1::{derive_multi_group_stage1_challenges, AkitaStage1Verifier};
use crate::stages::stage2::AkitaStage2Verifier;
use crate::stages::SetupSumcheckVerifier;
use akita_field::{
    AkitaError, CanonicalField, ExtField, FieldCore, FrobeniusExtField, FromPrimitiveInt,
    HalvingField, MulBaseUnreduced, PseudoMersenneField, RandomSampling,
};
use akita_serialization::AkitaSerialize;
use akita_sumcheck::SumcheckInstanceVerifierExt;
use akita_transcript::labels::{
    ABSORB_COMMITMENT, ABSORB_EVALUATION_CLAIMS, ABSORB_NEXT_LEVEL_WITNESS_BINDING,
    ABSORB_RANGE_IMAGE_EVALUATION, ABSORB_STAGE2_NEXT_W_EVAL, ABSORB_STAGE3_NEXT_W_EVAL,
    ABSORB_TERMINAL_E_HAT, ABSORB_TERMINAL_W_REMAINDER, CHALLENGE_SUMCHECK_BATCH,
    CHALLENGE_SUMCHECK_ROUND,
};
use akita_transcript::{append_ext_field, sample_ext_challenge, Transcript};
use akita_types::derive_tensor_extension_opening_claim_from_partials;
use akita_types::{
    append_claim_values_to_transcript, assemble_relation_rhs, build_trace_claim_multi_group_root,
    build_trace_claim_root, build_trace_table_scaled, dispatch_for_field,
    ensure_trace_stage2_supported, prepare_opening_point,
    proof::relation::evaluation_trace_row_weight, raw_field_segment_bytes,
    relation_claim_from_layout_extension, relation_rhs_layout_for,
    ring_subfield_packed_extension_opening_point, root_trace_block_opening,
    sample_public_row_coefficients, tensor_equality_factor_eval_at_point,
    trace_public_weights_recursive, trace_public_weights_root_terms, trace_terms_recursive,
    trace_weight_layout_from_segment, AkitaStage1Proof, AkitaStage2Proof, AkitaVerifierSetup,
    BasisMode, ExecutionSchedule, ExtensionOpeningReductionProof, FoldLevelProof,
    FoldLinfProtocolBinding, FpExtEncoding, LevelParams, OpeningClaims, OpeningClaimsLayout,
    PointVariableSelection, PolynomialGroupClaims, PreparedOpeningPoint, RelationMatrixRowLayout,
    RingMultiplierOpeningPoint, RingOpeningPoint, RingRelationInstance, RingVec, Schedule,
    SegmentTypedWitness, SetupSumcheckProof, TerminalLevelProof, TerminalWitnessTranscriptParts,
    TraceClaim,
};
use akita_types::{
    tensor_opening_split, tensor_reduction_claim_from_rows, tensor_row_partials_from_columns,
};
use extension_opening_reduction::verify_extension_opening_reduction_sumcheck;

mod fold;
mod root_fold;
mod suffix;
mod terminal_direct;
mod terminal_ntt;
use root_fold::verify_root;

pub use verify::batched_verify;

pub(in crate::protocol::core) type SetupPrefixOpening<E> = (Vec<E>, E);
pub(in crate::protocol::core) type FoldVerifyOutput<E> = (Vec<E>, Option<SetupPrefixOpening<E>>);

pub(in crate::protocol::core) use fold::{
    verify_fold, verify_fold_eor, FoldEorReplay, PreparedFoldPayload, PreparedFoldReplay,
    PreparedNextWitness,
};

fn prepare_terminal_witness_replay<F, T>(
    transcript: &mut T,
    final_witness: &SegmentTypedWitness<F>,
    final_w_len: usize,
) -> Result<TerminalWitnessTranscriptParts, AkitaError>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
{
    if final_witness.num_elems() != final_w_len {
        return Err(AkitaError::InvalidProof);
    }
    let parts = final_witness.terminal_transcript_parts()?;
    transcript.absorb_and_record_bytes(ABSORB_TERMINAL_E_HAT, &parts.e_folded);
    Ok(parts)
}
