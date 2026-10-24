//! Role-generic access to one end of the proof channel.

use super::{
    invalid_proof, prover_context, receive_atoms, reject, send_atoms, verifier_context,
    word_padded, ProtocolContextRecord,
};
use crate::{ProverChannel, VerifierChannel};
use akita_error::AkitaError;
use jolt_transcript::{ChallengeCodec, ProverTranscript, VerifierTranscript};
use spongefish::{Encoding, FromNarg, Transcript};

/// One party's end of the Akita proof channel.
///
/// Protocol-site code written against this trait runs unchanged for the prover
/// and the verifier. A proof message is exchanged in place: the prover emits
/// the values it holds and leaves them unchanged, and the verifier overwrites
/// the slots with the decoded atoms.
pub trait ProofChannel {
    /// Record diagnostic metadata for the next message or challenge group.
    fn context(&mut self, record: ProtocolContextRecord);

    /// Absorb public bytes that both sides already hold, zero-padded to a
    /// word. Their length must come from bound data.
    fn public(&mut self, bytes: &[u8]);

    /// Emit (prover) or read (verifier) one proof message of `messages.len()`
    /// canonical atoms, absorbed as their wire bytes zero-padded to a word.
    /// The count must come from bound data.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidProof`] when the verifier cannot decode the
    /// next message. The prover never fails.
    fn exchange<T: Encoding + FromNarg + Copy>(
        &mut self,
        messages: &mut [T],
    ) -> Result<(), AkitaError>;

    /// Draw one challenge message of `len` field elements.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] on the prover, or
    /// [`AkitaError::InvalidProof`] on the verifier, when the codec does not
    /// support `len` or its rejection pool is exhausted (probability at most
    /// `2^-161`). A verifier failure poisons the transcript.
    fn challenges<E: ChallengeCodec>(&mut self, len: usize) -> Result<Vec<E>, AkitaError>;

    /// Squeeze one challenge message of exactly `len` bytes and decode it.
    ///
    /// The length must be fixed by public data, never by squeezed values.
    ///
    /// # Errors
    ///
    /// Returns the error of `decode` on the prover. On the verifier, returns
    /// [`AkitaError::InvalidProof`] when the transcript is already rejected or
    /// `decode` fails; a failure poisons the transcript.
    fn challenge_message<T>(
        &mut self,
        len: usize,
        decode: impl FnOnce(&[u8]) -> Result<T, AkitaError>,
    ) -> Result<T, AkitaError>;

    /// Return `result` unchanged. On the verifier, an error first rejects the
    /// transcript, so a caller that drops the error still cannot accept.
    ///
    /// # Errors
    ///
    /// Returns the error in `result`.
    fn guard<T>(&mut self, result: Result<T, AkitaError>) -> Result<T, AkitaError>;
}

impl ProofChannel for ProverChannel {
    #[inline(always)]
    fn context(&mut self, record: ProtocolContextRecord) {
        prover_context(self, record);
    }

    fn public(&mut self, bytes: &[u8]) {
        self.public_message_as(bytes, word_padded);
    }

    fn exchange<T: Encoding + FromNarg + Copy>(
        &mut self,
        messages: &mut [T],
    ) -> Result<(), AkitaError> {
        send_atoms(self, messages.iter().copied());
        Ok(())
    }

    fn challenges<E: ChallengeCodec>(&mut self, len: usize) -> Result<Vec<E>, AkitaError> {
        ProverTranscript::field_challenges(self, len).map_err(|_| {
            AkitaError::InvalidSetup(
                "field challenge codec rejected the message length or exhausted its pool".into(),
            )
        })
    }

    fn challenge_message<T>(
        &mut self,
        len: usize,
        decode: impl FnOnce(&[u8]) -> Result<T, AkitaError>,
    ) -> Result<T, AkitaError> {
        self.verifier_message_as(len, decode)
    }

    fn guard<T>(&mut self, result: Result<T, AkitaError>) -> Result<T, AkitaError> {
        result
    }
}

impl ProofChannel for VerifierChannel<'_> {
    #[inline(always)]
    fn context(&mut self, record: ProtocolContextRecord) {
        verifier_context(self, record);
    }

    fn public(&mut self, bytes: &[u8]) {
        self.public_message_as(bytes, word_padded);
    }

    fn exchange<T: Encoding + FromNarg + Copy>(
        &mut self,
        messages: &mut [T],
    ) -> Result<(), AkitaError> {
        let received = receive_atoms::<T>(self, messages.len())?;
        for (slot, value) in messages.iter_mut().zip(received) {
            *slot = value;
        }
        Ok(())
    }

    fn challenges<E: ChallengeCodec>(&mut self, len: usize) -> Result<Vec<E>, AkitaError> {
        VerifierTranscript::field_challenges(self, len).map_err(invalid_proof)
    }

    fn challenge_message<T>(
        &mut self,
        len: usize,
        decode: impl FnOnce(&[u8]) -> Result<T, AkitaError>,
    ) -> Result<T, AkitaError> {
        Transcript::check(self, || true).map_err(invalid_proof)?;
        let result = self.verifier_message_as(len, decode);
        Transcript::check(self, || result.is_ok()).map_err(invalid_proof)?;
        result
    }

    fn guard<T>(&mut self, result: Result<T, AkitaError>) -> Result<T, AkitaError> {
        if result.is_err() {
            let _ = reject(self);
        }
        result
    }
}
