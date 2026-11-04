//! Backend-private source handles paired with the public root witness contract.

use super::{BinaryCommittedGroupManifest, BinaryRootWitnessPlan, BinaryTerminalClaim};
use akita_params::GroupCommitPhaseParams;

/// Typed request consumed only by an explicitly selected binary root backend.
/// The ordinary handle, when present, supplies a full commitment preimage to the
/// same outgoing witness. It does not introduce another opening or transcript.
pub struct BinaryRootWitnessRequest<H> {
    pub manifest: BinaryCommittedGroupManifest,
    pub claim: Option<BinaryTerminalClaim>,
    pub prime: super::PrimePlaneManifest,
    pub plan: BinaryRootWitnessPlan,
    pub ordinary: Option<(GroupCommitPhaseParams, H)>,
}

/// Request canonical padded carrier words for the field-switch prover.
/// The backend validates the source owner and the complete commitment manifest.
pub struct BinaryFieldSwitchSourceRequest {
    pub manifest: BinaryCommittedGroupManifest,
    pub polynomial: [u8; 32],
}

/// Request private exact integer residuals after the outgoing witness commitment.
/// The caller supplies the accepted response challenges from the same transcript.
/// The backend replays the fixed response from its authenticated source handle;
/// projection rows never enter response selection or nonce grinding.
pub struct BinaryProjectionRequest {
    pub challenges: akita_challenges::Challenges,
}

/// Private result of a [`BinaryProjectionRequest`].
///
/// `matrix` is the source handle's A, shared rather than copied, so the
/// prover compiles the relation and its typed prefix without deriving A again.
pub struct BinaryProjectionWitness {
    pub residuals: Vec<[i128; 648]>,
    pub matrix: std::sync::Arc<Vec<Vec<[i128; 648]>>>,
}
