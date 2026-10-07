//! Role-generic access to one end of the proof channel.

use super::{
    prover_context, prover_field_challenge, verifier_context, verifier_field_challenge,
    ProtocolContextRecord, ProverChannel, VerifierChannel,
};
use akita_error::AkitaError;
use jolt_field::CanonicalEncoding;
use spongefish::{Encoding, NargDeserialize};

/// One party's end of the Akita proof channel.
///
/// The prover writes proof bytes into the Fiat-Shamir state and the verifier
/// reads them back, so the two ends are not interchangeable the way a
/// symmetric absorb/squeeze transcript is.
///
/// Protocol-site code written against this trait runs unchanged for the prover
/// and the verifier. A proof message is exchanged in place: the prover emits
/// the value it holds and leaves it unchanged, and the verifier overwrites the
/// slot with the decoded atom.
pub trait ProofChannel {
    /// Record diagnostic metadata for the next message or challenge group.
    fn context(&mut self, record: ProtocolContextRecord);

    /// Absorb a public message that both sides already hold.
    fn public<T: Encoding<[u8]> + ?Sized>(&mut self, message: &T);

    /// Emit (prover) or read and absorb (verifier) one proof message.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidProof`] when the verifier cannot decode the
    /// next message. The prover never fails.
    fn exchange<T: Encoding<[u8]> + NargDeserialize>(
        &mut self,
        message: &mut T,
    ) -> Result<(), AkitaError>;

    /// Draw one exactly uniform base-field challenge.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] when `F` is not certified for exact
    /// sampling, or [`AkitaError::InvalidProof`] when the retry counter is
    /// exhausted or the verifier state is already invalid.
    fn field_challenge<F: CanonicalEncoding>(&mut self) -> Result<F, AkitaError>;
}

impl ProofChannel for ProverChannel {
    #[inline(always)]
    fn context(&mut self, record: ProtocolContextRecord) {
        prover_context(self, record);
    }

    fn public<T: Encoding<[u8]> + ?Sized>(&mut self, message: &T) {
        self.public_message(message);
    }

    fn exchange<T: Encoding<[u8]> + NargDeserialize>(
        &mut self,
        message: &mut T,
    ) -> Result<(), AkitaError> {
        self.prover_message(message);
        Ok(())
    }

    fn field_challenge<F: CanonicalEncoding>(&mut self) -> Result<F, AkitaError> {
        prover_field_challenge(self)
    }
}

impl ProofChannel for VerifierChannel<'_> {
    #[inline(always)]
    fn context(&mut self, record: ProtocolContextRecord) {
        verifier_context(self, record);
    }

    fn public<T: Encoding<[u8]> + ?Sized>(&mut self, message: &T) {
        self.public_message(message);
    }

    fn exchange<T: Encoding<[u8]> + NargDeserialize>(
        &mut self,
        message: &mut T,
    ) -> Result<(), AkitaError> {
        *message = self.prover_message::<T>()?;
        Ok(())
    }

    fn field_challenge<F: CanonicalEncoding>(&mut self) -> Result<F, AkitaError> {
        verifier_field_challenge(self)
    }
}
