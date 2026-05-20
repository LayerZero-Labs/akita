//! Akita-native transcript labels.
//!
//! These constants are the single source of truth for protocol transcript
//! labels in Akita core. External integrations should translate at adapter
//! boundaries instead of introducing foreign label names here.
//!
//! Production transcripts are positional: these byte strings are diagnostics
//! for logging builds, tests, and schedule inspection, not bytes absorbed by
//! [`AkitaTranscript`](crate::AkitaTranscript).
//!
/// Top-level protocol domain label.
pub const DOMAIN_AKITA_PROTOCOL: &[u8] = b"ak/p";

/// Absorb commitment object(s) (paper §4.1).
pub const ABSORB_COMMITMENT: &[u8] = b"ak/a/cm";
/// Absorb claimed openings/evaluations before relation reduction (paper §4.2).
pub const ABSORB_EVALUATION_CLAIMS: &[u8] = b"ak/a/ec";
/// Absorb the public batch nesting shape for grouped/multipoint batching.
pub const ABSORB_BATCH_SHAPE: &[u8] = b"ak/a/bs";
/// Challenge for the evaluation-to-linear-relation reduction (paper §4.2).
pub const CHALLENGE_LINEAR_RELATION: &[u8] = b"ak/c/lr";
/// Absorb ring-switch relation messages (paper §4.3).
pub const ABSORB_RING_SWITCH_MESSAGE: &[u8] = b"ak/a/rs";
/// Challenge used by ring-switching relation checks (paper §4.3).
pub const CHALLENGE_RING_SWITCH: &[u8] = b"ak/c/rs";
/// Absorb sparse-challenge sampling context (e.g. for short/sparse ring `c`).
pub const ABSORB_SPARSE_CHALLENGE: &[u8] = b"ak/a/sp";
/// Challenge bytes used to sample sparse challenges (e.g. ring `c` with weight ω).
pub const CHALLENGE_SPARSE_CHALLENGE: &[u8] = b"ak/c/sp";
/// Absorb the initial sumcheck claim before round messages begin.
pub const ABSORB_SUMCHECK_CLAIM: &[u8] = b"ak/a/sc";
/// Absorb per-round sumcheck messages (paper §4.3).
pub const ABSORB_SUMCHECK_ROUND: &[u8] = b"ak/a/scr";
/// Challenge sampled per sumcheck round (paper §4.3).
pub const CHALLENGE_SUMCHECK_ROUND: &[u8] = b"ak/c/scr";
/// Absorb the stage-1 final `s_claim` before the batching challenge.
pub const ABSORB_SUMCHECK_S_CLAIM: &[u8] = b"ak/a/scs";
/// Absorb stage-1 inter-stage claims before batching them into the next stage.
pub const ABSORB_SUMCHECK_INTERSTAGE_CLAIM: &[u8] = b"ak/a/sci";
/// Challenge for batched sumcheck coefficient sampling.
pub const CHALLENGE_SUMCHECK_BATCH: &[u8] = b"ak/c/scb";
/// Challenge for batching stage-1 inter-stage claims into the next tree stage.
pub const CHALLENGE_SUMCHECK_INTERSTAGE_BATCH: &[u8] = b"ak/c/scib";
/// Absorb recursion/stop-condition message payloads (paper §4.5).
pub const ABSORB_STOP_CONDITION: &[u8] = b"ak/a/st";
/// Challenge sampled for recursion stop-condition checks (paper §4.5).
pub const CHALLENGE_STOP_CONDITION: &[u8] = b"ak/c/st";

/// Absorb the prover's stage-1 message `v = D · ŵ` (paper §4.2, Figure 3).
pub const ABSORB_PROVER_V: &[u8] = b"ak/a/v";
/// Challenge label for stage-1 fold (sampling sparse `c_i`).
pub const CHALLENGE_STAGE1_FOLD: &[u8] = b"ak/c/s1f";

/// Absorb field-element evaluation claims for γ-batching.
pub const ABSORB_EVAL_OPENINGS_FIELD: &[u8] = b"ak/a/eof";
/// Challenge for γ-batching evaluation claims at the same point.
pub const CHALLENGE_EVAL_BATCH: &[u8] = b"ak/c/eb";

/// Absorb the `w` coefficient vector before sumcheck (paper §4.3).
pub const ABSORB_SUMCHECK_W: &[u8] = b"ak/a/w";
/// Challenge for sampling `τ₀` (F_0 range-check batching point, paper §4.3).
pub const CHALLENGE_TAU0: &[u8] = b"ak/c/t0";
/// Challenge for sampling `τ₁` (F_α evaluation-relation batching point, paper §4.3).
pub const CHALLENGE_TAU1: &[u8] = b"ak/c/t1";

/// All Akita-core transcript labels.
pub const ALL_LABELS: &[&[u8]] = &[
    DOMAIN_AKITA_PROTOCOL,
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
];

/// Return all Akita-core transcript labels.
pub fn all_labels() -> &'static [&'static [u8]] {
    ALL_LABELS
}
