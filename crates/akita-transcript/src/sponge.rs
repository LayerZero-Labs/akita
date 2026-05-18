//! Spongefish-backed Akita transcript substrate.

use crate::Label;
use crate::Transcript;
use akita_field::{CanonicalBytes, CanonicalField, FieldCore, TranscriptChallenge};
use akita_serialization::AkitaSerialize;
use spongefish::{
    DomainSeparator, DuplexSpongeInterface, Encoding, ProverState, VerifierState, WithoutInstance,
};
use std::marker::PhantomData;

#[cfg(all(feature = "transcript-blake2b", feature = "transcript-keccak"))]
compile_error!("enable exactly one transcript backend: transcript-blake2b or transcript-keccak");

#[cfg(not(any(feature = "transcript-blake2b", feature = "transcript-keccak")))]
compile_error!("enable exactly one transcript backend: transcript-blake2b or transcript-keccak");

/// Sponge backend selected by the active transcript feature.
#[cfg(feature = "transcript-blake2b")]
pub type TranscriptSponge = spongefish::instantiations::Blake2b512;

/// Sponge backend selected by the active transcript feature.
#[cfg(feature = "transcript-keccak")]
pub type TranscriptSponge = spongefish::instantiations::Keccak;

/// Backend-specific 64-byte protocol tag for spongefish domain separation.
#[cfg(feature = "transcript-blake2b")]
pub const PROTOCOL_TAG: &[u8; 64] =
    b"akita-pcs/transcript/v1/blake2b\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0";

/// Backend-specific 64-byte protocol tag for spongefish domain separation.
#[cfg(feature = "transcript-keccak")]
pub const PROTOCOL_TAG: &[u8; 64] =
    b"akita-pcs/transcript/v1/keccak\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0";

const SQUEEZE_CHUNK_LEN: usize = 32;

enum TranscriptState<S>
where
    S: DuplexSpongeInterface,
{
    Prover(Box<ProverState<S>>),
    Verifier(Box<VerifierState<'static, S>>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TranscriptSide {
    Prover,
    Verifier,
}

/// Thin Akita transcript wrapper over spongefish prover/verifier states.
pub struct AkitaTranscript<F, S = TranscriptSponge>
where
    S: DuplexSpongeInterface<U = u8>,
{
    session_tag: [u8; 64],
    side: TranscriptSide,
    state: Option<TranscriptState<S>>,
    _field: PhantomData<fn() -> F>,
}

impl<F> AkitaTranscript<F, TranscriptSponge>
where
    F: FieldCore + CanonicalField + CanonicalBytes + TranscriptChallenge,
{
    /// Construct a prover-side transcript with the selected backend.
    pub fn prover(session_label: &[u8], instance_bytes: &[u8]) -> Self {
        Self::new_prover(session_label, instance_bytes)
    }

    /// Construct a prover-side transcript under a session label.
    ///
    /// The scheme-level prover/verifier paths re-bind this transcript to the
    /// actual instance descriptor before replay. Direct unit tests that only
    /// exercise lower-level transcript consumers use this deterministic
    /// placeholder instance.
    pub fn new(session_label: &[u8]) -> Self {
        Self::new_prover(session_label, b"akita/default-instance")
    }

    /// Reset this transcript under a new session label and placeholder
    /// instance.
    ///
    /// Scheme-level callers re-bind the actual descriptor before replay.
    pub fn reset(&mut self, session_label: &[u8]) {
        *self = Self::new(session_label);
    }

    /// Construct a verifier-side transcript with the selected backend.
    pub fn verifier(session_label: &[u8], instance_bytes: &[u8]) -> Self {
        Self::new_verifier(session_label, instance_bytes)
    }

    /// Construct a prover-side transcript that will be instance-bound later.
    pub fn unbound_prover(session_label: &[u8]) -> Self {
        Self::unbound(session_label, TranscriptSide::Prover)
    }

    /// Construct a verifier-side transcript that will be instance-bound later.
    pub fn unbound_verifier(session_label: &[u8]) -> Self {
        Self::unbound(session_label, TranscriptSide::Verifier)
    }
}

impl<F, S> AkitaTranscript<F, S>
where
    F: FieldCore + CanonicalField + CanonicalBytes + TranscriptChallenge,
    S: Default + DuplexSpongeInterface<U = u8>,
{
    /// Construct a prover-side transcript from canonical instance bytes.
    ///
    /// `instance_bytes` must be `AkitaInstanceDescriptor::canonical_bytes()`
    /// from `akita-types`.
    pub fn new_prover(session_label: &[u8], instance_bytes: &[u8]) -> Self {
        let mut transcript = Self::unbound(session_label, TranscriptSide::Prover);
        transcript.bind_instance_bytes(instance_bytes);
        transcript
    }

    /// Construct a verifier-side transcript from canonical instance bytes.
    ///
    /// `instance_bytes` must be `AkitaInstanceDescriptor::canonical_bytes()`
    /// from `akita-types`.
    pub fn new_verifier(session_label: &[u8], instance_bytes: &[u8]) -> Self {
        let mut transcript = Self::unbound(session_label, TranscriptSide::Verifier);
        transcript.bind_instance_bytes(instance_bytes);
        transcript
    }

    fn unbound(session_label: &[u8], side: TranscriptSide) -> Self {
        Self {
            session_tag: session_tag(session_label),
            side,
            state: None,
            _field: PhantomData,
        }
    }

    /// Bind or re-bind the transcript to canonical instance bytes.
    ///
    /// `instance_bytes` must be `AkitaInstanceDescriptor::canonical_bytes()`
    /// from `akita-types`, and this method must be called before any absorb or
    /// squeeze operation for the proof being replayed.
    pub fn bind_instance_bytes(&mut self, instance_bytes: &[u8]) {
        let domain = domain_separator_from_tag(self.session_tag, instance_bytes);
        self.state = Some(match self.side {
            TranscriptSide::Prover => {
                TranscriptState::Prover(Box::new(domain.to_prover(S::default())))
            }
            TranscriptSide::Verifier => {
                TranscriptState::Verifier(Box::new(domain.to_verifier(S::default(), &[])))
            }
        });
    }
}

impl<F, S> AkitaTranscript<F, S>
where
    F: FieldCore + CanonicalField + CanonicalBytes + TranscriptChallenge,
    S: DuplexSpongeInterface<U = u8>,
{
    fn state_mut(&mut self) -> &mut TranscriptState<S> {
        self.state
            .as_mut()
            .expect("AkitaTranscript must be instance-bound before use")
    }
}

impl<F, S> AkitaTranscript<F, S>
where
    F: FieldCore + CanonicalField + CanonicalBytes + TranscriptChallenge,
    S: DuplexSpongeInterface<U = u8>,
{
    /// Absorb prefix-free bytes into the transcript.
    pub fn absorb_bytes(&mut self, _label: Label, bytes: &[u8]) {
        let framed = FramedBytes { bytes };
        match self.state_mut() {
            TranscriptState::Prover(state) => state.public_message(&framed),
            TranscriptState::Verifier(state) => state.public_message(&framed),
        }
    }

    /// Absorb a field element using its canonical little-endian bytes.
    pub fn absorb_field(&mut self, label: Label, value: &F) {
        let mut bytes = vec![0u8; F::NUM_BYTES];
        value.to_bytes_le(&mut bytes);
        self.absorb_bytes(label, &bytes);
    }

    /// Absorb an Akita-serializable value using compressed serialization.
    ///
    /// # Panics
    ///
    /// Panics if serialization fails while writing to an in-memory buffer.
    pub fn absorb_serde<T: AkitaSerialize>(&mut self, label: Label, value: &T) {
        let mut bytes = Vec::new();
        value
            .serialize_compressed(&mut bytes)
            .expect("AkitaSerialize should not fail for transcript absorb");
        self.absorb_bytes(label, &bytes);
    }

    /// Squeeze a base-field scalar challenge.
    pub fn squeeze_scalar(&mut self, label: Label) -> F {
        let bytes = self.squeeze_bytes(label, 2 * F::NUM_BYTES);
        F::from_challenge_bytes(&bytes)
    }

    /// Squeeze challenge bytes.
    pub fn squeeze_bytes(&mut self, _label: Label, len: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(len);
        while out.len() < len {
            let chunk: [u8; SQUEEZE_CHUNK_LEN] = match self.state_mut() {
                TranscriptState::Prover(state) => state.verifier_message(),
                TranscriptState::Verifier(state) => state.verifier_message(),
            };
            let take = (len - out.len()).min(chunk.len());
            out.extend_from_slice(&chunk[..take]);
        }
        out
    }
}

impl<F, S> Transcript<F> for AkitaTranscript<F, S>
where
    F: FieldCore + CanonicalField + CanonicalBytes + TranscriptChallenge + 'static,
    S: Default + DuplexSpongeInterface<U = u8> + Send + 'static,
{
    fn new(domain_label: &[u8]) -> Self {
        Self::new_prover(domain_label, b"akita/default-instance")
    }

    fn bind_instance_bytes(&mut self, instance_bytes: &[u8]) {
        AkitaTranscript::bind_instance_bytes(self, instance_bytes);
    }

    fn append_bytes(&mut self, _label: &[u8], bytes: &[u8]) {
        self.absorb_bytes(crate::label!("compat_absorb_bytes"), bytes);
    }

    fn append_field(&mut self, _label: &[u8], x: &F) {
        self.absorb_field(crate::label!("compat_absorb_field"), x);
    }

    fn append_serde<T: AkitaSerialize>(&mut self, _label: &[u8], s: &T) {
        self.absorb_serde(crate::label!("compat_absorb_serde"), s);
    }

    fn challenge_scalar(&mut self, _label: &[u8]) -> F {
        self.squeeze_scalar(crate::label!("compat_squeeze_scalar"))
    }

    fn challenge_bytes(&mut self, _label: &[u8], len: usize) -> Vec<u8> {
        self.squeeze_bytes(crate::label!("compat_squeeze_bytes"), len)
    }
}

#[derive(Clone, Copy)]
struct FramedBytes<'a> {
    bytes: &'a [u8],
}

impl Encoding<[u8]> for FramedBytes<'_> {
    fn encode(&self) -> impl AsRef<[u8]> {
        let len = u64::try_from(self.bytes.len()).expect("transcript payload length overflows u64");
        let mut out = Vec::with_capacity(8 + self.bytes.len());
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(self.bytes);
        out
    }
}

#[inline]
fn domain_separator_from_tag<'a>(
    session_tag: [u8; 64],
    instance_bytes: &'a [u8],
) -> DomainSeparator<spongefish::WithInstance<FramedBytes<'a>>, spongefish::WithSession<[u8; 64]>> {
    DomainSeparator::<WithoutInstance>::new(*PROTOCOL_TAG)
        .session(session_tag)
        .instance(FramedBytes {
            bytes: instance_bytes,
        })
}

#[inline]
fn session_tag(session_label: &[u8]) -> [u8; 64] {
    let mut tag = [0u8; 64];
    assert!(
        session_label.len() <= tag.len(),
        "transcript session labels must fit in 64 bytes"
    );
    tag[..session_label.len()].copy_from_slice(session_label);
    tag
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_field::Prime128Offset275;

    type F = Prime128Offset275;

    #[test]
    fn preamble_bytes_affect_first_challenge() {
        let mut left = AkitaTranscript::<F>::prover(b"test/session", b"instance-a");
        let mut right = AkitaTranscript::<F>::prover(b"test/session", b"instance-b");

        assert_ne!(
            left.squeeze_scalar(crate::label!("challenge")),
            right.squeeze_scalar(crate::label!("challenge"))
        );
    }

    #[test]
    fn prover_and_verifier_agree_on_public_transcript() {
        let mut prover = AkitaTranscript::<F>::prover(b"test/session", b"same-instance");
        let mut verifier = AkitaTranscript::<F>::verifier(b"test/session", b"same-instance");
        let value = F::from_u64(42);

        prover.absorb_field(crate::label!("absorbed"), &value);
        verifier.absorb_field(crate::label!("absorbed"), &value);

        assert_eq!(
            prover.squeeze_scalar(crate::label!("challenge")),
            verifier.squeeze_scalar(crate::label!("challenge"))
        );
    }

    #[cfg(not(feature = "logging-transcript"))]
    #[test]
    fn labels_do_not_enter_production_sponge() {
        let mut left = AkitaTranscript::<F>::prover(b"test/session", b"same-instance");
        let mut right = AkitaTranscript::<F>::prover(b"test/session", b"same-instance");
        let value = F::from_u64(7);

        left.absorb_field(crate::label!("left_label"), &value);
        right.absorb_field(crate::label!("right_label"), &value);

        assert_eq!(
            left.squeeze_scalar(crate::label!("left_challenge")),
            right.squeeze_scalar(crate::label!("right_challenge"))
        );
    }
}
