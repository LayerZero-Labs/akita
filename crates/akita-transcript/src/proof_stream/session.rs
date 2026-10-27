//! Restricted session capabilities and public opening domain records.

use crate::TranscriptSponge;
use spongefish::{Decoding, Encoding, NargSerialize, ProverState};
use std::ops::{Deref, DerefMut};

/// Owner of a prover session and its completed argument bytes.
/// Borrow it as [`ProverSession`] to compose openings.
pub struct ProverChannel {
    session: ProverSession<[(); 0]>,
}

/// Restricted prover session view: messages and challenges, without a
/// constructor, reset, clone, or finish operation.
/// The default unsized tail prevents moving or swapping the owner's storage.
/// Sized instantiations are private storage details and cannot be constructed.
///
/// ```compile_fail
/// use akita_transcript::{new_prover_channel, ProverSession};
/// fn swap(session: &mut ProverSession) {
///     let mut replacement = new_prover_channel(b"reset", b"").unwrap();
///     std::mem::swap(session, &mut *replacement);
/// }
/// ```
///
/// ```compile_fail
/// use akita_transcript::{new_prover_channel, ProverSession};
/// fn reset(session: &mut ProverSession) {
///     *session = new_prover_channel(b"reset", b"").unwrap();
/// }
/// ```
pub struct ProverSession<T: ?Sized = [()]> {
    pub(super) inner: ProverState<TranscriptSponge>,
    _unsized: T,
}

impl ProverChannel {
    pub(super) const fn new(inner: ProverState<TranscriptSponge>) -> Self {
        Self {
            session: ProverSession {
                inner,
                _unsized: [],
            },
        }
    }

    /// The authoritative argument bytes emitted by this owned session.
    pub fn narg_string(&self) -> &[u8] {
        self.session.inner.narg_string()
    }
}

impl Deref for ProverChannel {
    type Target = ProverSession;
    fn deref(&self) -> &ProverSession {
        &self.session
    }
}

impl DerefMut for ProverChannel {
    fn deref_mut(&mut self) -> &mut ProverSession {
        &mut self.session
    }
}

impl ProverSession {
    /// Absorb public data without emitting proof bytes.
    pub fn public_message<T: Encoding<[u8]> + ?Sized>(&mut self, message: &T) {
        self.inner.public_message(message);
    }

    /// Emit and absorb a canonical proof message.
    pub fn prover_message<T: Encoding<[u8]> + NargSerialize + ?Sized>(&mut self, message: &T) {
        self.inner.prover_message(message);
    }

    /// Draw a challenge from the live session.
    pub fn verifier_message<T: Decoding<[u8]>>(&mut self) -> T {
        self.inner.verifier_message()
    }
}

use super::{proof_stream_protocol_id, FramedBytes, ProofChannel};
use akita_error::AkitaError;

/// Absorb a framed public subprotocol record before its first challenge.
///
/// Unlike diagnostic `context` metadata, every byte here enters the sponge.
/// The native suite, protocol identifier, protocol version, and canonical
/// setup/profile/catalog descriptor bind the opening to its public context.
/// Standalone entry points retain their existing session domains instead.
///
/// # Errors
/// Returns an error if a framed length cannot be represented as `u64`.
pub fn absorb_subprotocol_context<S: ProofChannel + ?Sized>(
    state: &mut S,
    protocol_id: &[u8],
    version: u32,
    identity: &[u8],
) -> Result<(), AkitaError> {
    let protocol = FramedBytes::new(protocol_id)?;
    let identity = FramedBytes::new(identity)?;
    state.public(b"akita/borrowed-opening/context/v1".as_slice());
    state.public(&proof_stream_protocol_id());
    state.public(&protocol);
    state.public(&version);
    state.public(&identity);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{new_prover_channel, new_verifier_channel, FieldAtom};
    use jolt_field::Prime32Offset99 as Fr;

    #[test]
    fn context_is_framed_and_absorbed_and_session_continues() {
        let challenge = |protocol: &[u8], version, identity: &[u8]| {
            let mut state = new_prover_channel(b"aggregate", b"header").unwrap();
            absorb_subprotocol_context(&mut state, protocol, version, identity).unwrap();
            state.field_challenge::<Fr>().unwrap()
        };
        let expected = challenge(b"pcs", 1, b"setup");
        assert_ne!(expected, challenge(b"root", 1, b"setup"));
        assert_ne!(expected, challenge(b"pcs", 2, b"setup"));
        assert_ne!(expected, challenge(b"pcs", 1, b"other"));
        assert_ne!(challenge(b"a", 1, b"bc"), challenge(b"ab", 1, b"c"));
        let mut prover = new_prover_channel(b"aggregate", b"header").unwrap();
        absorb_subprotocol_context(&mut prover, b"pcs", 1, b"setup").unwrap();
        let sampled = prover.field_challenge::<Fr>().unwrap();
        prover.prover_message(&FieldAtom::new(sampled));
        prover.public(b"suffix".as_slice());
        let end = prover.field_challenge::<Fr>().unwrap();
        let proof = prover.narg_string();
        let mut verifier = new_verifier_channel(b"aggregate", b"header", proof).unwrap();
        absorb_subprotocol_context(&mut verifier, b"pcs", 1, b"setup").unwrap();
        assert_eq!(sampled, verifier.field_challenge::<Fr>().unwrap());
        assert_eq!(
            sampled,
            verifier
                .prover_message::<FieldAtom<Fr>>()
                .unwrap()
                .into_inner()
        );
        verifier.public(b"suffix".as_slice());
        assert_eq!(end, verifier.field_challenge::<Fr>().unwrap());
        verifier.check_eof().unwrap();
    }
}
