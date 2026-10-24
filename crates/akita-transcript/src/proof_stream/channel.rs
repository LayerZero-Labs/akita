//! Role-generic access to one end of the proof channel.

use super::{
    invalid_proof, prover_context, receive_atoms, reject, send_atoms, verifier_context,
    ProtocolContextRecord,
};
use crate::{ProverChannel, VerifierChannel, FOLD_CHALLENGE_SEED_LEN};
use akita_error::AkitaError;
use jolt_transcript::{ChallengeCodec, FieldTranscript};
use spongefish::{Encoding, FromNarg, Transcript};

/// One party's end of the Akita proof channel.
///
/// Protocol-site code written against this trait runs unchanged for the prover
/// and the verifier. A proof message is exchanged in place: the prover emits
/// the value it holds and leaves it unchanged, and the verifier overwrites the
/// slot with the decoded atom.
pub trait ProofChannel {
    /// Record diagnostic metadata for the next message or challenge group.
    fn context(&mut self, record: ProtocolContextRecord);

    /// Absorb public bytes that both sides already hold.
    fn public(&mut self, bytes: &[u8]);

    /// Emit (prover) or read and absorb (verifier) one proof message.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidProof`] when the verifier cannot decode the
    /// next message. The prover never fails.
    fn exchange<T: Encoding + FromNarg>(&mut self, message: &mut T) -> Result<(), AkitaError>;

    /// Emit (prover) or read and absorb (verifier) a schedule-fixed group of
    /// atoms as one proof message. An empty group makes no absorb call.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidProof`] when the verifier cannot decode the
    /// group. The prover never fails.
    fn exchange_group<T: Encoding + FromNarg>(&mut self, atoms: &mut [T])
        -> Result<(), AkitaError>;

    /// Draw one challenge message of `len` field elements.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] on the prover, or
    /// [`AkitaError::InvalidProof`] on the verifier, when the codec does not
    /// support `len` or its rejection pool is exhausted (probability at most
    /// `2^-161`). A verifier failure poisons the transcript.
    fn challenges<E: ChallengeCodec>(&mut self, len: usize) -> Result<Vec<E>, AkitaError>;

    /// Squeeze one fold-root seed.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidProof`] when the verifier transcript is
    /// already rejected. The prover never fails.
    fn seed(&mut self) -> Result<[u8; FOLD_CHALLENGE_SEED_LEN], AkitaError>;

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
        self.public_message_as(bytes, |bytes| bytes);
    }

    fn exchange<T: Encoding + FromNarg>(&mut self, message: &mut T) -> Result<(), AkitaError> {
        self.prover_message(message);
        Ok(())
    }

    fn exchange_group<T: Encoding + FromNarg>(
        &mut self,
        atoms: &mut [T],
    ) -> Result<(), AkitaError> {
        send_atoms(self, atoms);
        Ok(())
    }

    fn challenges<E: ChallengeCodec>(&mut self, len: usize) -> Result<Vec<E>, AkitaError> {
        FieldTranscript::<E>::challenge_vector(self, len).map_err(|_| {
            AkitaError::InvalidSetup(
                "field challenge codec rejected the message length or exhausted its pool".into(),
            )
        })
    }

    fn seed(&mut self) -> Result<[u8; FOLD_CHALLENGE_SEED_LEN], AkitaError> {
        Ok(self.verifier_message())
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
        self.public_message_as(bytes, |bytes| bytes);
    }

    fn exchange<T: Encoding + FromNarg>(&mut self, message: &mut T) -> Result<(), AkitaError> {
        *message = self.prover_message::<T>().map_err(invalid_proof)?;
        Ok(())
    }

    fn exchange_group<T: Encoding + FromNarg>(
        &mut self,
        atoms: &mut [T],
    ) -> Result<(), AkitaError> {
        let received = receive_atoms::<T>(self, atoms.len())?;
        for (slot, atom) in atoms.iter_mut().zip(received) {
            *slot = atom;
        }
        Ok(())
    }

    fn challenges<E: ChallengeCodec>(&mut self, len: usize) -> Result<Vec<E>, AkitaError> {
        FieldTranscript::<E>::challenge_vector(self, len).map_err(invalid_proof)
    }

    fn seed(&mut self) -> Result<[u8; FOLD_CHALLENGE_SEED_LEN], AkitaError> {
        Transcript::check(self, || true).map_err(invalid_proof)?;
        Ok(self.verifier_message())
    }

    fn guard<T>(&mut self, result: Result<T, AkitaError>) -> Result<T, AkitaError> {
        if result.is_err() {
            let _ = reject(self);
        }
        result
    }
}
