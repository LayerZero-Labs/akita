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
/// Absorb per-round sumcheck messages (paper §4.3).
pub const ABSORB_SUMCHECK_ROUND: &[u8] = b"hachi/absorb/sumcheck-round";
/// Challenge sampled per sumcheck round (paper §4.3).
pub const CHALLENGE_SUMCHECK_ROUND: &[u8] = b"hachi/challenge/sumcheck-round";
/// Absorb recursion/stop-condition message payloads (paper §4.5).
pub const ABSORB_STOP_CONDITION: &[u8] = b"hachi/absorb/stop-condition";
/// Challenge sampled for recursion stop-condition checks (paper §4.5).
pub const CHALLENGE_STOP_CONDITION: &[u8] = b"hachi/challenge/stop-condition";

/// Return all Hachi-core transcript labels.
pub fn all_labels() -> &'static [&'static [u8]] {
    &[
        DOMAIN_HACHI_PROTOCOL,
        ABSORB_COMMITMENT,
        ABSORB_EVALUATION_CLAIMS,
        CHALLENGE_LINEAR_RELATION,
        ABSORB_RING_SWITCH_MESSAGE,
        CHALLENGE_RING_SWITCH,
        ABSORB_SUMCHECK_ROUND,
        CHALLENGE_SUMCHECK_ROUND,
        ABSORB_STOP_CONDITION,
        CHALLENGE_STOP_CONDITION,
    ]
}
