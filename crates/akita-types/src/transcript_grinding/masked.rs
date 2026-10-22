//! The masked channels Akita's grinding replay runs over at the levels up to a
//! zero-knowledge cut (`specs/whiteout-akita.md` §4.5): jolt-zk's
//! [`MaskedProver`] and [`MaskedVerifier`]. Their proof-of-work nonces and
//! fold-response counters are public; nothing else leaves them unmasked.

use super::{ProverGrindingChannel, VerifierGrindingChannel};
use akita_error::AkitaError;
use jolt_algebra::{CanonicalEncoding, Field};
use jolt_transcript::{Sponge, TranscriptError};
use jolt_zk::masking::{MaskedProver, MaskedVerifier};

fn terminal_only() -> AkitaError {
    AkitaError::UnsupportedSchedule("a masked level sends no terminal byte message".into())
}

impl<P: Field + CanonicalEncoding, O, H: Sponge> ProverGrindingChannel
    for MaskedProver<'_, P, O, H>
{
    fn grind(&mut self, bits: u8) -> Result<u32, TranscriptError> {
        Self::grind(self, bits)
    }

    fn send_nonce(&mut self, nonce: u32) {
        Self::send_nonce(self, nonce);
    }

    fn narg_len(&self) -> usize {
        Self::narg_len(self)
    }

    fn send_public_bytes(&mut self, _: &[u8]) -> Result<(), AkitaError> {
        Err(terminal_only())
    }

    fn send_bounded_public_bytes(&mut self, _: &[u8], _: usize) -> Result<(), AkitaError> {
        Err(terminal_only())
    }
}

impl<H: Sponge> VerifierGrindingChannel for MaskedVerifier<'_, '_, H> {
    fn check_grind(&mut self, bits: u8) -> Result<u32, TranscriptError> {
        Self::check_grind(self, bits)
    }

    fn receive_nonce(&mut self, nonce_bits: u8) -> Result<u32, TranscriptError> {
        Self::receive_nonce(self, nonce_bits)
    }

    fn remaining(&self) -> usize {
        Self::remaining(self)
    }
}
