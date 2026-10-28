//! Backend-private source handles paired with the public root witness contract.

use super::{BinaryCommittedGroupManifest, BinaryRootWitnessPlan, BinaryTerminalClaim};
use akita_params::GroupCommitPhaseParams;

/// Typed request consumed only by an explicitly selected binary root backend.
/// The ordinary handle, when present, supplies a full commitment preimage to the
/// same outgoing witness. It does not introduce another opening or transcript.
pub struct BinaryRootWitnessRequest<H> {
    pub manifest: BinaryCommittedGroupManifest,
    pub claim: BinaryTerminalClaim,
    pub plan: BinaryRootWitnessPlan,
    pub ordinary: Option<(GroupCommitPhaseParams, H)>,
}
