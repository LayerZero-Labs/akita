use crate::TranscriptSponge;
use akita_error::AkitaError;
use spongefish::{Decoding, Encoding, NargDeserialize, VerifierState};
use std::ops::{Deref, DerefMut};

/// Akita's fail-closed Spongefish verifier state.
///
/// Receipt failures poison this owner permanently. Keeping the underlying
/// Spongefish state private prevents protocol callers from consuming malformed
/// input and later obtaining a successful EOF result by ignoring the error.
/// This view cannot construct, replace, or finish the caller's session.
/// Its unsized tail also prevents swapping owner storage through this view.
///
/// ```compile_fail
/// use akita_transcript::{new_verifier_channel, VerifierSession};
/// fn swap(session: &mut VerifierSession<'_>) {
///     let mut replacement = new_verifier_channel(b"reset", b"", &[]).unwrap();
///     std::mem::swap(session, &mut *replacement);
///     replacement.check_eof().unwrap();
/// }
/// ```
///
/// ```compile_fail
/// use akita_transcript::VerifierSession;
/// fn finish(session: &mut VerifierSession<'_>) {
///     session.check_eof().unwrap();
/// }
/// ```
pub struct VerifierSession<'proof, T: ?Sized = [()]> {
    inner: VerifierState<'proof, TranscriptSponge>,
    invalid: bool,
    _unsized: T,
}

impl<'proof> VerifierSession<'proof> {
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

    /// Draw one verifier message from the proof channel.
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

/// Owner of a verifier session; only this owner may require final EOF.
pub struct VerifierChannel<'proof> {
    session: VerifierSession<'proof, [(); 0]>,
}

impl<'proof> VerifierChannel<'proof> {
    pub(super) const fn new(inner: VerifierState<'proof, TranscriptSponge>) -> Self {
        Self {
            session: VerifierSession {
                inner,
                invalid: false,
                _unsized: [],
            },
        }
    }

    /// Finish only if no earlier receipt failed and no proof bytes remain.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidProof`] when an earlier receipt failed or
    /// proof bytes remain.
    pub fn check_eof(self) -> Result<(), AkitaError> {
        if self.session.invalid {
            return Err(AkitaError::InvalidProof);
        }
        self.session
            .inner
            .check_eof()
            .map_err(|_| AkitaError::InvalidProof)
    }
}

impl<'proof> Deref for VerifierChannel<'proof> {
    type Target = VerifierSession<'proof>;
    fn deref(&self) -> &Self::Target {
        &self.session
    }
}

impl DerefMut for VerifierChannel<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.session
    }
}
