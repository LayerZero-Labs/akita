use crate::TranscriptSponge;
use akita_error::AkitaError;
use spongefish::{Decoding, Encoding, NargDeserialize, VerifierState};

/// Akita's fail-closed native Spongefish verifier state.
///
/// Receipt failures poison this owner permanently. Keeping the underlying
/// Spongefish state private prevents protocol callers from consuming malformed
/// input and later obtaining a successful EOF result by ignoring the error.
pub struct NativeVerifierState<'proof> {
    inner: VerifierState<'proof, TranscriptSponge>,
    invalid: bool,
}

impl<'proof> NativeVerifierState<'proof> {
    pub(super) const fn new(inner: VerifierState<'proof, TranscriptSponge>) -> Self {
        Self {
            inner,
            invalid: false,
        }
    }

    /// Read and absorb one prover message, recording every decoding failure.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidProof`] when an earlier receipt failed or
    /// the next message is missing or noncanonical.
    pub fn prover_message<T>(&mut self) -> Result<T, AkitaError>
    where
        T: Encoding<[u8]> + NargDeserialize,
    {
        if self.invalid {
            return Err(AkitaError::InvalidProof);
        }
        let result = self
            .inner
            .prover_message::<T>()
            .map_err(|_| AkitaError::InvalidProof);
        self.invalid |= result.is_err();
        result
    }

    /// Absorb one public message without consuming proof bytes.
    pub fn public_message<T: Encoding<[u8]> + ?Sized>(&mut self, message: &T) {
        if self.invalid {
            return;
        }
        self.inner.public_message(message);
    }

    /// Draw one verifier message from the native transcript.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidProof`] when an earlier receipt failed.
    pub fn verifier_message<T: Decoding<[u8]>>(&mut self) -> Result<T, AkitaError> {
        if self.invalid {
            return Err(AkitaError::InvalidProof);
        }
        Ok(self.inner.verifier_message())
    }

    /// Finish only if no earlier receipt failed and no proof bytes remain.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidProof`] when an earlier receipt failed or
    /// proof bytes remain.
    pub fn check_eof(self) -> Result<(), AkitaError> {
        if self.invalid {
            return Err(AkitaError::InvalidProof);
        }
        self.inner.check_eof().map_err(|_| AkitaError::InvalidProof)
    }

    pub(super) fn sponge_mut(&mut self) -> &mut TranscriptSponge {
        &mut self.inner.duplex_sponge_state
    }

    pub(super) const fn is_invalid(&self) -> bool {
        self.invalid
    }

    /// Permanently reject this transcript after a semantic protocol failure.
    pub fn invalidate(&mut self) {
        self.invalid = true;
    }
}
