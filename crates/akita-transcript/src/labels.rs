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
/// Absorb the public batch nesting shape for multi-group single-point batching.
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
/// Absorb the selected setup-prefix slot id before delegated setup sumcheck replay.
pub const ABSORB_SETUP_PREFIX_SLOT: &[u8] = b"ak/a/sps";
/// Absorb per-round sumcheck messages (paper §4.3).
pub const ABSORB_SUMCHECK_ROUND: &[u8] = b"ak/a/scr";
/// Challenge sampled per sumcheck round (paper §4.3).
pub const CHALLENGE_SUMCHECK_ROUND: &[u8] = b"ak/c/scr";
/// Absorb the stage-1 final `s_claim` before the batching challenge.
pub const ABSORB_SUMCHECK_S_CLAIM: &[u8] = b"ak/a/scs";
/// Absorb the stage-2 next-witness evaluation handoff before recursion continues.
pub const ABSORB_STAGE2_NEXT_W_EVAL: &[u8] = b"ak/a/s2w";
/// Absorb the stage-3 carried next-witness evaluation before recursion continues.
pub const ABSORB_STAGE3_NEXT_W_EVAL: &[u8] = b"ak/a/s3w";
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

/// Absorb the prover's folded witness message `v = D · ŵ` before sampling fold
/// challenges (paper §4.2, Figure 3).
pub const ABSORB_PROVER_V: &[u8] = b"ak/a/v";
/// Challenge label for witness-fold sparse ring elements `c_i` (flat shape).
///
/// Prefixes the sparse-challenge Fiat–Shamir absorb buffer for one draw batch.
/// The buffer is appended under [`ABSORB_SPARSE_CHALLENGE`]; this string is not
/// absorbed by itself into the positional production sponge.
pub const CHALLENGE_WITNESS_FOLD: &[u8] = b"ak/c/wf";
/// Challenge label for the left factor `α` in a tensor-shaped fold round.
///
/// Tensor folds sample `√N` left and `√N` right sparse challenges per claim and
/// use `c_{p,q} = α_p · β_q`. This prefixes the absorb buffer for the **left**
/// draw batch (under [`ABSORB_SPARSE_CHALLENGE`]). After the left challenges are
/// expanded, [`ABSORB_TENSOR_FOLD_LEFT`] commits a digest of the left vector
/// before the right batch is drawn.
pub const CHALLENGE_TENSOR_FOLD_LEFT: &[u8] = b"ak/c/wfl";
/// Digest of the sampled left tensor factor, appended between left and right draws.
///
/// Canonical hash of the left sparse-challenge vector (`tensor_left_digest`).
/// Prevents choosing the right factor `β` adaptively after seeing `α`. This is
/// a real transcript append (positional sponge); it is not the challenges themselves.
pub const ABSORB_TENSOR_FOLD_LEFT: &[u8] = b"ak/a/wtl";
/// Challenge label for the right factor `β` in a tensor-shaped fold round.
///
/// Prefixes the absorb buffer for the **right** draw batch (under
/// [`ABSORB_SPARSE_CHALLENGE`]), after [`ABSORB_TENSOR_FOLD_LEFT`]. There is no
/// symmetric digest absorb for the right vector.
pub const CHALLENGE_TENSOR_FOLD_RIGHT: &[u8] = b"ak/c/wfr";

/// Absorb field-element evaluation claims for γ-batching.
pub const ABSORB_EVAL_OPENINGS_FIELD: &[u8] = b"ak/a/eof";
/// Challenge for γ-batching evaluation claims at the same point.
pub const CHALLENGE_EVAL_BATCH: &[u8] = b"ak/c/eb";

/// Binds the next-level witness at this fold step. Intermediate folds absorb
/// the Ajtai commitment `u'` to the next-level witness `w` (`next_w_commitment`);
/// the terminal fold absorbs the cleartext `final_witness` (packed `w`) in the
/// same wire position. Diagnostic label only; sponge bytes are positional.
pub const ABSORB_NEXT_LEVEL_WITNESS_BINDING: &[u8] = b"ak/a/w";
/// Absorb terminal logical `e_hat` digits before sparse-challenge sampling.
pub const ABSORB_TERMINAL_E_HAT: &[u8] = b"ak/a/twh";
/// Absorb terminal final-witness digits outside logical `e_hat`.
pub const ABSORB_TERMINAL_W_REMAINDER: &[u8] = b"ak/a/twr";
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
    ABSORB_SETUP_PREFIX_SLOT,
    ABSORB_SUMCHECK_ROUND,
    CHALLENGE_SUMCHECK_ROUND,
    ABSORB_SUMCHECK_S_CLAIM,
    ABSORB_STAGE2_NEXT_W_EVAL,
    ABSORB_STAGE3_NEXT_W_EVAL,
    ABSORB_SUMCHECK_INTERSTAGE_CLAIM,
    CHALLENGE_SUMCHECK_BATCH,
    CHALLENGE_SUMCHECK_INTERSTAGE_BATCH,
    ABSORB_STOP_CONDITION,
    CHALLENGE_STOP_CONDITION,
    ABSORB_PROVER_V,
    CHALLENGE_WITNESS_FOLD,
    CHALLENGE_TENSOR_FOLD_LEFT,
    ABSORB_TENSOR_FOLD_LEFT,
    CHALLENGE_TENSOR_FOLD_RIGHT,
    ABSORB_EVAL_OPENINGS_FIELD,
    CHALLENGE_EVAL_BATCH,
    ABSORB_NEXT_LEVEL_WITNESS_BINDING,
    ABSORB_TERMINAL_E_HAT,
    ABSORB_TERMINAL_W_REMAINDER,
    CHALLENGE_TAU0,
    CHALLENGE_TAU1,
];

/// Return all Akita-core transcript labels.
pub fn all_labels() -> &'static [&'static [u8]] {
    ALL_LABELS
}
