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
/// Challenge for batched sumcheck coefficient sampling.
pub const CHALLENGE_SUMCHECK_BATCH: &[u8] = b"hachi/challenge/sumcheck-batch";
/// Absorb recursion/stop-condition message payloads (paper §4.5).
pub const ABSORB_STOP_CONDITION: &[u8] = b"hachi/absorb/stop-condition";
/// Challenge sampled for recursion stop-condition checks (paper §4.5).
pub const CHALLENGE_STOP_CONDITION: &[u8] = b"hachi/challenge/stop-condition";

/// Absorb the prover's stage-1 message `v = D · ŵ` (paper §4.2, Figure 3).
pub const ABSORB_PROVER_V: &[u8] = b"hachi/absorb/prover-stage1-v";
/// Challenge label for stage-1 fold (sampling sparse `c_i`).
pub const CHALLENGE_STAGE1_FOLD: &[u8] = b"hachi/challenge/stage1-fold";

/// Absorb the `w` coefficient vector before sumcheck (paper §4.3).
pub const ABSORB_SUMCHECK_W: &[u8] = b"hachi/absorb/sumcheck-w";
/// Challenge for sampling `τ₀` (F_0 range-check batching point, paper §4.3).
pub const CHALLENGE_TAU0: &[u8] = b"hachi/challenge/tau0";
/// Challenge for sampling `τ₁` (F_α evaluation-relation batching point, paper §4.3).
pub const CHALLENGE_TAU1: &[u8] = b"hachi/challenge/tau1";

/// Labrador protocol domain label (used for recursive reduction stages).
pub const DOMAIN_LABRADOR_PROTOCOL: &[u8] = b"hachi/labrador/protocol";
/// Greyhound evaluation-reduction domain label.
pub const DOMAIN_GREYHOUND_EVAL: &[u8] = b"hachi/greyhound/eval";
/// Absorb canonical Greyhound evaluation context bytes (dimensions/backend id).
pub const ABSORB_GREYHOUND_EVAL_CONTEXT: &[u8] = b"hachi/absorb/greyhound-eval-context";
/// Absorb canonicalized evaluation-point coordinates for Greyhound reduction.
pub const ABSORB_GREYHOUND_EVAL_POINT: &[u8] = b"hachi/absorb/greyhound-eval-point";
/// Absorb the claimed evaluation value for Greyhound reduction.
pub const ABSORB_GREYHOUND_EVAL_VALUE: &[u8] = b"hachi/absorb/greyhound-eval-value";
/// Absorb the Greyhound second outer commitment `u2`.
pub const ABSORB_GREYHOUND_U2: &[u8] = b"hachi/absorb/greyhound-u2";
/// Challenge for Greyhound column-fold coefficients.
pub const CHALLENGE_GREYHOUND_FOLD: &[u8] = b"hachi/challenge/greyhound-fold";
/// Absorb canonical Labrador level metadata (shape/config/tail/backend id).
pub const ABSORB_LABRADOR_LEVEL_CONTEXT: &[u8] = b"hachi/absorb/labrador-level-context";
/// Absorb Labrador JL projection vector `p`.
pub const ABSORB_LABRADOR_JL_PROJECTION: &[u8] = b"hachi/absorb/labrador-jl-projection";
/// Absorb Labrador JL nonce.
pub const ABSORB_LABRADOR_JL_NONCE: &[u8] = b"hachi/absorb/labrador-jl-nonce";
/// Challenge for Labrador aggregation/lift stage.
pub const CHALLENGE_LABRADOR_AGGREGATION: &[u8] = b"hachi/challenge/labrador-aggregation";
/// Absorb Labrador inner commitment u1 at each recursion level.
pub const ABSORB_LABRADOR_U1: &[u8] = b"hachi/absorb/labrador-u1";
/// Absorb Labrador outer commitment u2 at each recursion level.
pub const ABSORB_LABRADOR_U2: &[u8] = b"hachi/absorb/labrador-u2";
/// Absorb Labrador lift polynomials (constant-term-removed).
pub const ABSORB_LABRADOR_BB: &[u8] = b"hachi/absorb/labrador-bb";
/// Absorb Labrador squared norm bound at each level.
pub const ABSORB_LABRADOR_NORM: &[u8] = b"hachi/absorb/labrador-norm";
/// Challenge for Labrador amortization fold (ring-element challenges).
pub const CHALLENGE_LABRADOR_AMORTIZE: &[u8] = b"hachi/challenge/labrador-amortize";

/// Return all Hachi-core transcript labels.
pub fn all_labels() -> &'static [&'static [u8]] {
    &[
        DOMAIN_HACHI_PROTOCOL,
        ABSORB_COMMITMENT,
        ABSORB_EVALUATION_CLAIMS,
        CHALLENGE_LINEAR_RELATION,
        ABSORB_RING_SWITCH_MESSAGE,
        CHALLENGE_RING_SWITCH,
        ABSORB_SPARSE_CHALLENGE,
        CHALLENGE_SPARSE_CHALLENGE,
        ABSORB_SUMCHECK_CLAIM,
        ABSORB_SUMCHECK_ROUND,
        CHALLENGE_SUMCHECK_ROUND,
        CHALLENGE_SUMCHECK_BATCH,
        ABSORB_STOP_CONDITION,
        CHALLENGE_STOP_CONDITION,
        ABSORB_PROVER_V,
        CHALLENGE_STAGE1_FOLD,
        ABSORB_SUMCHECK_W,
        CHALLENGE_TAU0,
        CHALLENGE_TAU1,
        DOMAIN_LABRADOR_PROTOCOL,
        DOMAIN_GREYHOUND_EVAL,
        ABSORB_GREYHOUND_EVAL_CONTEXT,
        ABSORB_GREYHOUND_EVAL_POINT,
        ABSORB_GREYHOUND_EVAL_VALUE,
        ABSORB_GREYHOUND_U2,
        CHALLENGE_GREYHOUND_FOLD,
        ABSORB_LABRADOR_LEVEL_CONTEXT,
        ABSORB_LABRADOR_JL_PROJECTION,
        ABSORB_LABRADOR_JL_NONCE,
        CHALLENGE_LABRADOR_AGGREGATION,
        ABSORB_LABRADOR_U1,
        ABSORB_LABRADOR_U2,
        ABSORB_LABRADOR_BB,
        ABSORB_LABRADOR_NORM,
        CHALLENGE_LABRADOR_AMORTIZE,
    ]
}
