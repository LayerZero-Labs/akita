//! Akita's protocol identity on the shared `jolt-transcript` proof channel.

use akita_error::AkitaError;
use jolt_field::Field;
use jolt_transcript::ProtocolId;

/// Sponge used by Akita's standalone proving and verification API.
///
/// Composed protocols choose their own sponge: the core entry points are
/// generic over it and run on the caller's transcript.
pub type AkitaSponge = jolt_transcript::Blake2b512;

/// Protocol identity of a standalone Akita proof stream.
pub const PROOF_STREAM_PROTOCOL: ProtocolId =
    ProtocolId::new::<AkitaSponge>("akita-pcs/native-proof-stream/v8");

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
