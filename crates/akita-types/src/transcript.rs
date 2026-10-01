//! Akita's protocol identity and diagnostic site coordinates on the shared
//! `jolt-transcript` proof channel.

use akita_error::AkitaError;
use jolt_field::Field;
use jolt_transcript::{ProtocolId, SiteId};

/// Sponge used by Akita's standalone proving and verification API.
///
/// Composed protocols choose their own sponge: the core entry points are
/// generic over it and run on the caller's transcript.
pub type AkitaSponge = jolt_transcript::Blake2b512;

/// Protocol identity of a standalone Akita proof stream.
pub const PROOF_STREAM_PROTOCOL: ProtocolId =
    ProtocolId::new::<AkitaSponge>("akita-pcs/native-proof-stream/v8");

/// Stable family identifier for standard and batched sumcheck sites.
pub const SITE_FAMILY_SUMCHECK: u32 = 1;

/// Stable family identifier for extension-opening reduction messages.
pub const SITE_FAMILY_EXTENSION_OPENING_REDUCTION: u32 = 2;

/// Stable family identifier for stage-2 terminal claims.
pub const SITE_FAMILY_STAGE2: u32 = 3;

/// Stable family identifier for stage-1 late oracle claims.
pub const SITE_FAMILY_STAGE1: u32 = 4;

/// Stable family identifier for physical-L2 proof values.
pub const SITE_FAMILY_PHYSICAL_L2: u32 = 5;

/// Stable family identifier for recursive setup-product stage 3.
pub const SITE_FAMILY_STAGE3: u32 = 6;

/// Stable family identifier for indexed sparse fold-challenge roots.
pub const SITE_FAMILY_FOLD_CHALLENGE: u32 = 7;

/// Stable family identifier for ring-relation opening payloads.
pub const SITE_FAMILY_OPENING_PAYLOAD: u32 = 8;

/// Stable family identifier for derived fold-opening values.
pub const SITE_FAMILY_FOLD_BINDING: u32 = 9;

/// Stable family identifier for the successor witness binding.
pub const SITE_FAMILY_NEXT_WITNESS: u32 = 10;

/// Stable family identifier for root public commitments and opening points.
pub const SITE_FAMILY_ROOT_STATEMENT: u32 = 11;

/// Stable family identifier for terminal response messages.
pub const SITE_FAMILY_TERMINAL: u32 = 12;

/// Canonical 32-byte identity of one protocol site.
///
/// Sites are diagnostic only: they tag transcript events under the
/// `jolt-transcript/logging` feature and never reach the sponge. Unused
/// coordinates are zero. Callers derive every coordinate from validated public
/// schedule state; proof input never selects a site.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ProtocolSiteId {
    /// Protocol family, such as sumcheck or folding.
    pub family: u32,
    /// Invocation within the enclosing public schedule.
    pub invocation: u32,
    /// Recursive fold level.
    pub level: u32,
    /// Stage within a level.
    pub stage: u32,
    /// Round within a stage.
    pub round: u32,
    /// Commitment group within a round.
    pub group: u32,
    /// Family-specific public discriminator, such as grinding widths.
    pub detail: u32,
}

impl ProtocolSiteId {
    /// Encode this site identity as seven little-endian `u32` coordinates in
    /// declaration order, followed by four zero bytes.
    #[must_use]
    pub fn to_bytes(self) -> [u8; 32] {
        let mut encoded = [0u8; 32];
        for (chunk, coordinate) in encoded.chunks_exact_mut(4).zip(self.coordinates()) {
            chunk.copy_from_slice(&coordinate.to_le_bytes());
        }
        encoded
    }

    /// Decode the layout produced by [`Self::to_bytes`].
    ///
    /// This inverse supports typed diagnostics and tests; it is not part of
    /// proof parsing.
    #[must_use]
    pub fn from_bytes(encoded: [u8; 32]) -> Self {
        let mut coordinates = [0u32; 7];
        for (coordinate, chunk) in coordinates.iter_mut().zip(encoded.chunks_exact(4)) {
            *coordinate = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        let [family, invocation, level, stage, round, group, detail] = coordinates;
        Self {
            family,
            invocation,
            level,
            stage,
            round,
            group,
            detail,
        }
    }

    const fn coordinates(self) -> [u32; 7] {
        [
            self.family,
            self.invocation,
            self.level,
            self.stage,
            self.round,
            self.group,
            self.detail,
        ]
    }
}

impl From<ProtocolSiteId> for SiteId {
    fn from(site: ProtocolSiteId) -> Self {
        Self(site.to_bytes())
    }
}

/// Allocate `count` zeroed extension slots for an in-place exchange.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidProof`] when the allocation cannot be reserved.
pub fn extension_slots<E: Field>(count: usize) -> Result<Vec<E>, AkitaError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| AkitaError::InvalidProof)?;
    values.resize(count, E::zero());
    Ok(values)
}

#[cfg(test)]
pub(crate) mod test_transcripts {
    use super::{AkitaSponge, PROOF_STREAM_PROTOCOL};
    use jolt_transcript::{ProverTranscript, VerifierTranscript};

    pub(crate) fn prover(session: &[u8]) -> ProverTranscript<AkitaSponge> {
        ProverTranscript::new(&PROOF_STREAM_PROTOCOL, session)
    }

    pub(crate) fn verifier<'proof>(
        session: &[u8],
        proof: &'proof [u8],
    ) -> VerifierTranscript<'proof, AkitaSponge> {
        VerifierTranscript::new(&PROOF_STREAM_PROTOCOL, session, proof)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn site_layout_round_trips() {
        let site = ProtocolSiteId {
            family: 1,
            invocation: 2,
            level: 3,
            stage: 4,
            round: 5,
            group: 6,
            detail: u32::MAX,
        };
        let bytes = site.to_bytes();
        assert_eq!(bytes[..4], 1u32.to_le_bytes());
        assert_eq!(bytes[24..28], u32::MAX.to_le_bytes());
        assert_eq!(bytes[28..], [0; 4]);
        assert_eq!(ProtocolSiteId::from_bytes(bytes), site);
    }
}
