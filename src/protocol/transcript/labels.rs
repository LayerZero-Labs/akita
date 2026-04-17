//! Hachi-native transcript labels.
//!
//! These constants are the single source of truth for protocol transcript
//! labels in Hachi core. External integrations should translate at adapter
//! boundaries instead of introducing foreign label names here.

/// Top-level protocol domain label.
pub const DOMAIN_HACHI_PROTOCOL: &[u8] = b"hachi/protocol";

/// Absorb commitment object(s) (paper §4.1).
pub const ABSORB_COMMITMENT: &[u8] = b"hachi/absorb/commitment";
/// Absorb claimed openings/evaluations before relation reduction (paper §4.2).
pub const ABSORB_EVALUATION_CLAIMS: &[u8] = b"hachi/absorb/evaluation-claims";
/// Absorb the public batch nesting shape for grouped/multipoint batching.
pub const ABSORB_BATCH_SHAPE: &[u8] = b"hachi/absorb/batch-shape";
/// Challenge for the evaluation-to-linear-relation reduction (paper §4.2).
pub const CHALLENGE_LINEAR_RELATION: &[u8] = b"hachi/challenge/linear-relation";
/// Absorb ring-switch relation messages (paper §4.3).
pub const ABSORB_RING_SWITCH_MESSAGE: &[u8] = b"hachi/absorb/ring-switch-message";
/// Challenge used by ring-switching relation checks (paper §4.3).
pub const CHALLENGE_RING_SWITCH: &[u8] = b"hachi/challenge/ring-switch";
/// Absorb sparse-challenge sampling context (e.g. for short/sparse ring `c`).
pub const ABSORB_SPARSE_CHALLENGE: &[u8] = b"hachi/absorb/sparse-challenge";
/// Challenge bytes used to sample sparse challenges (e.g. ring `c` with weight ω).
pub const CHALLENGE_SPARSE_CHALLENGE: &[u8] = b"hachi/challenge/sparse-challenge";
/// Absorb the initial sumcheck claim before round messages begin.
pub const ABSORB_SUMCHECK_CLAIM: &[u8] = b"hachi/absorb/sumcheck-claim";
/// Absorb per-round sumcheck messages (paper §4.3).
pub const ABSORB_SUMCHECK_ROUND: &[u8] = b"hachi/absorb/sumcheck-round";
/// Challenge sampled per sumcheck round (paper §4.3).
pub const CHALLENGE_SUMCHECK_ROUND: &[u8] = b"hachi/challenge/sumcheck-round";
/// Absorb the stage-1 final `s_claim` before the batching challenge.
pub const ABSORB_SUMCHECK_S_CLAIM: &[u8] = b"hachi/absorb/sumcheck-s-claim";
/// Absorb stage-1 inter-stage claims before batching them into the next stage.
pub const ABSORB_SUMCHECK_INTERSTAGE_CLAIM: &[u8] = b"hachi/absorb/sumcheck-interstage-claim";
/// Challenge for batching relation into fused Stage 1 (T2 protocol).
pub const CHALLENGE_FUSED_RELATION_BATCH: &[u8] = b"hachi/challenge/fused-relation-batch";
/// Absorb `w_eval = w(r_stage1)` after fused Stage 1.
pub const ABSORB_FUSED_W_EVAL: &[u8] = b"hachi/absorb/fused-w-eval";
/// Absorb `claimed_setup_val` after fused Stage 1.
pub const ABSORB_FUSED_SETUP_VAL: &[u8] = b"hachi/absorb/fused-setup-val";
/// Absorb `shared_matrix_eval` after the batched fused Stage 2 (setup-claim
/// instance at `r_setup`).
pub const ABSORB_FUSED_SHARED_MATRIX_EVAL: &[u8] = b"hachi/absorb/fused-shared-matrix-eval";
/// Challenge for batched sumcheck coefficient sampling.
pub const CHALLENGE_SUMCHECK_BATCH: &[u8] = b"hachi/challenge/sumcheck-batch";
/// Challenge for batching stage-1 inter-stage claims into the next tree stage.
pub const CHALLENGE_SUMCHECK_INTERSTAGE_BATCH: &[u8] = b"hachi/challenge/sumcheck-interstage-batch";
/// Absorb recursion/stop-condition message payloads (paper §4.5).
pub const ABSORB_STOP_CONDITION: &[u8] = b"hachi/absorb/stop-condition";
/// Challenge sampled for recursion stop-condition checks (paper §4.5).
pub const CHALLENGE_STOP_CONDITION: &[u8] = b"hachi/challenge/stop-condition";

/// Absorb the prover's stage-1 message `v = D · ŵ` (paper §4.2, Figure 3).
pub const ABSORB_PROVER_V: &[u8] = b"hachi/absorb/prover-stage1-v";
/// Challenge label for stage-1 fold (sampling sparse `c_i`).
pub const CHALLENGE_STAGE1_FOLD: &[u8] = b"hachi/challenge/stage1-fold";

/// Absorb field-element evaluation claims for γ-batching.
pub const ABSORB_EVAL_OPENINGS_FIELD: &[u8] = b"hachi/absorb/eval-openings-field";
/// Challenge for γ-batching evaluation claims at the same point.
pub const CHALLENGE_EVAL_BATCH: &[u8] = b"hachi/challenge/eval-batch";

/// Absorb the `w` coefficient vector before sumcheck (paper §4.3).
pub const ABSORB_SUMCHECK_W: &[u8] = b"hachi/absorb/sumcheck-w";
/// Challenge for sampling `τ₀` (F_0 range-check batching point, paper §4.3).
pub const CHALLENGE_TAU0: &[u8] = b"hachi/challenge/tau0";
/// Challenge for sampling `τ₁` (F_α evaluation-relation batching point, paper §4.3).
pub const CHALLENGE_TAU1: &[u8] = b"hachi/challenge/tau1";

/// Return all Hachi-core transcript labels.
pub fn all_labels() -> &'static [&'static [u8]] {
    &[
        DOMAIN_HACHI_PROTOCOL,
        ABSORB_COMMITMENT,
        ABSORB_EVALUATION_CLAIMS,
        ABSORB_BATCH_SHAPE,
        CHALLENGE_LINEAR_RELATION,
        ABSORB_RING_SWITCH_MESSAGE,
        CHALLENGE_RING_SWITCH,
        ABSORB_SPARSE_CHALLENGE,
        CHALLENGE_SPARSE_CHALLENGE,
        ABSORB_SUMCHECK_CLAIM,
        ABSORB_SUMCHECK_ROUND,
        CHALLENGE_SUMCHECK_ROUND,
        ABSORB_SUMCHECK_S_CLAIM,
        ABSORB_SUMCHECK_INTERSTAGE_CLAIM,
        CHALLENGE_FUSED_RELATION_BATCH,
        ABSORB_FUSED_W_EVAL,
        ABSORB_FUSED_SETUP_VAL,
        ABSORB_FUSED_SHARED_MATRIX_EVAL,
        CHALLENGE_SUMCHECK_BATCH,
        CHALLENGE_SUMCHECK_INTERSTAGE_BATCH,
        ABSORB_STOP_CONDITION,
        CHALLENGE_STOP_CONDITION,
        ABSORB_PROVER_V,
        CHALLENGE_STAGE1_FOLD,
        ABSORB_EVAL_OPENINGS_FIELD,
        CHALLENGE_EVAL_BATCH,
        ABSORB_SUMCHECK_W,
        CHALLENGE_TAU0,
        CHALLENGE_TAU1,
    ]
}
