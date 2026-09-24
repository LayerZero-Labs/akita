use crate::TranscriptSponge;
use spongefish::{Decoding, Encoding, NargDeserialize, VerificationError, VerifierState};

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
    pub fn prover_message<T>(&mut self) -> Result<T, VerificationError>
    where
        T: Encoding<[u8]> + NargDeserialize,
    {
        if self.invalid {
            return Err(VerificationError);
        }
        let result = self.inner.prover_message::<T>();
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
    pub fn verifier_message<T: Decoding<[u8]>>(&mut self) -> Result<T, VerificationError> {
        if self.invalid {
            return Err(VerificationError);
        }
        Ok(self.inner.verifier_message())
    }

    /// Finish only if no earlier receipt failed and no proof bytes remain.
    pub fn check_eof(self) -> Result<(), VerificationError> {
        if self.invalid {
            return Err(VerificationError);
        }
        self.inner.check_eof()
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
