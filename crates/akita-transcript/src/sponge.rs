//! Spongefish-backed Akita transcript substrate.

use crate::Label;
use crate::Transcript;
use akita_field::{CanonicalBytes, CanonicalField, FieldCore, TranscriptChallenge};
use akita_serialization::AkitaSerialize;
use spongefish::{
    DomainSeparator, DuplexSpongeInterface, Encoding, ProverState, VerifierState, WithoutInstance,
};
use std::marker::PhantomData;

/// Sponge backend selected by the active transcript feature.
///
/// Exactly one transcript backend feature must be active in the complete PCS graph.
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
    session_label: Vec<u8>,
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
            session_label: session_label.to_vec(),
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
        let domain = domain_separator_from_label(&self.session_label, instance_bytes);
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

    /// Preview the final 32-byte challenge after hypothetical absorbs.
    pub fn preview_challenge_block(
        &self,
        absorb_payloads: &[&[u8]],
    ) -> [u8; crate::TRANSCRIPT_CHALLENGE_BLOCK_LEN] {
        let TranscriptState::Prover(state) = self
            .state
            .as_ref()
            .expect("AkitaTranscript must be instance-bound before use")
        else {
            panic!("preview_challenge_block requires a prover transcript");
        };
        let mut sponge = state.duplex_sponge_state.clone();
        let mut out = [0u8; crate::TRANSCRIPT_CHALLENGE_BLOCK_LEN];
        for &absorb in absorb_payloads {
            let framed = FramedBytes { bytes: absorb };
            sponge.absorb(framed.encode().as_ref());
            sponge.squeeze(out.as_mut());
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

    // The `Transcript` trait keeps semantic labels for logging wrappers and
    // callsite readability. The production `AkitaTranscript` sponge is
    // intentionally positional: labels are not absorbed, and protocol/instance
    // separation comes from the spongefish domain separator plus replay order.
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

struct SessionBoundInstance<'a> {
    session_label: &'a [u8],
    instance_bytes: &'a [u8],
}

impl Encoding<[u8]> for SessionBoundInstance<'_> {
    fn encode(&self) -> impl AsRef<[u8]> {
        let label_len = u64::try_from(self.session_label.len())
            .expect("transcript session-label length overflows u64");
        let instance_len = u64::try_from(self.instance_bytes.len())
            .expect("transcript instance length overflows u64");
        let mut out = Vec::with_capacity(16 + self.session_label.len() + self.instance_bytes.len());
        out.extend_from_slice(&label_len.to_le_bytes());
        out.extend_from_slice(self.session_label);
        out.extend_from_slice(&instance_len.to_le_bytes());
        out.extend_from_slice(self.instance_bytes);
        out
    }
}

const fn session_domain_tag() -> [u8; 64] {
    let source = b"akita-pcs/session-label/v1";
    let mut tag = [0u8; 64];
    let mut index = 0usize;
    while index < source.len() {
        tag[index] = source[index];
        index += 1;
    }
    tag
}

const SESSION_DOMAIN_TAG: [u8; 64] = session_domain_tag();

#[inline]
fn domain_separator_from_label<'a>(
    session_label: &'a [u8],
    instance_bytes: &'a [u8],
) -> DomainSeparator<
    spongefish::WithInstance<SessionBoundInstance<'a>>,
    spongefish::WithSession<[u8; 64]>,
> {
    DomainSeparator::<WithoutInstance>::new(*PROTOCOL_TAG)
        .session(SESSION_DOMAIN_TAG)
        .instance(SessionBoundInstance {
            session_label,
            instance_bytes,
        })
}

impl<F, S> crate::TranscriptChallengePreview for AkitaTranscript<F, S>
where
    F: FieldCore + CanonicalField + CanonicalBytes + TranscriptChallenge,
    S: Default + DuplexSpongeInterface<U = u8> + Send + 'static,
{
    fn preview_challenge_block(
        &self,
        absorb_payloads: &[&[u8]],
    ) -> [u8; crate::TRANSCRIPT_CHALLENGE_BLOCK_LEN] {
        Self::preview_challenge_block(self, absorb_payloads)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        grinding_predicate_accepts, preview_grinding_predicate, search_grinding_nonce,
        GRINDING_NONCE_SLACK_BITS,
    };
    use akita_field::Prime128Offset275;
    use std::num::NonZeroU8;

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
    fn backend_protocol_tag_and_first_challenge_are_stable() {
        let mut transcript = AkitaTranscript::<F>::prover(b"backend-test", b"instance");
        let challenge = transcript.squeeze_scalar(crate::label!("challenge"));
        #[cfg(feature = "transcript-blake2b")]
        {
            assert_eq!(&PROTOCOL_TAG[..31], b"akita-pcs/transcript/v1/blake2b");
            assert_eq!(
                challenge.to_canonical_u128(),
                313_598_626_200_370_843_239_849_985_198_023_824_966
            );
        }
        #[cfg(feature = "transcript-keccak")]
        {
            assert_eq!(&PROTOCOL_TAG[..30], b"akita-pcs/transcript/v1/keccak");
            assert_eq!(
                challenge.to_canonical_u128(),
                23_462_597_902_952_977_795_780_514_374_913_799_469
            );
        }
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

    #[test]
    fn framed_bytes_encoding_is_prefix_free() {
        let short = FramedBytes { bytes: b"abc" }.encode();
        let long = FramedBytes { bytes: b"abcdef" }.encode();

        assert_eq!(&short.as_ref()[..8], &3u64.to_le_bytes());
        assert_eq!(&long.as_ref()[..8], &6u64.to_le_bytes());
        assert!(!long.as_ref().starts_with(short.as_ref()));
    }

    #[test]
    fn session_labels_are_prefix_free_and_unbounded() {
        let labels: [&[u8]; 5] = [b"", b"\0", b"x", b"x\0", &[7u8; 65]];
        let challenges = labels.map(|label| {
            let mut transcript = AkitaTranscript::<F>::prover(label, b"same-instance");
            transcript.squeeze_scalar(crate::label!("challenge"))
        });

        for (index, challenge) in challenges.iter().enumerate() {
            assert!(
                challenges[index + 1..]
                    .iter()
                    .all(|other| other != challenge),
                "distinct session labels must derive distinct transcript states"
            );
        }
    }

    #[test]
    fn session_and_instance_encoding_is_exactly_framed() {
        let encoded = SessionBoundInstance {
            session_label: b"x\0",
            instance_bytes: b"instance",
        }
        .encode()
        .as_ref()
        .to_vec();
        let mut expected = Vec::new();
        expected.extend_from_slice(&2u64.to_le_bytes());
        expected.extend_from_slice(b"x\0");
        expected.extend_from_slice(&8u64.to_le_bytes());
        expected.extend_from_slice(b"instance");
        assert_eq!(encoded, expected);
    }

    #[test]
    fn grinding_preview_live_and_verifier_replay_agree() {
        let mut prover = AkitaTranscript::<F>::prover(b"grinding-test", b"instance");
        let mut verifier = AkitaTranscript::<F>::verifier(b"grinding-test", b"instance");
        prover.absorb_bytes(crate::label!("context"), b"fixed history");
        verifier.absorb_bytes(crate::label!("context"), b"fixed history");

        let grind_bits = 4;
        let nonce_bits = grind_bits + GRINDING_NONCE_SLACK_BITS;
        let nonce = search_grinding_nonce(&prover, grind_bits, nonce_bits).unwrap();
        let preview = preview_grinding_predicate(&prover, grind_bits, nonce_bits, nonce).unwrap();
        let live = Transcript::grinding_predicate(
            &mut prover,
            crate::labels::CHALLENGE_TAU0,
            grind_bits,
            nonce_bits,
            nonce,
        )
        .unwrap();
        let replay = Transcript::grinding_predicate(
            &mut verifier,
            crate::labels::CHALLENGE_TAU0,
            grind_bits,
            nonce_bits,
            nonce,
        )
        .unwrap();

        assert_eq!(preview, live);
        assert_eq!(live, replay);
        assert!(grinding_predicate_accepts(
            &live,
            NonZeroU8::new(grind_bits).unwrap()
        ));

        let prover_challenge =
            prover.challenge_bytes(crate::labels::CHALLENGE_TAU0, SQUEEZE_CHUNK_LEN);
        let verifier_challenge =
            verifier.challenge_bytes(crate::labels::CHALLENGE_TAU0, SQUEEZE_CHUNK_LEN);
        assert_eq!(prover_challenge, verifier_challenge);
        assert_ne!(live.as_slice(), prover_challenge);
    }

    #[test]
    fn grinding_preview_does_not_change_live_state() {
        let mut previewed = AkitaTranscript::<F>::prover(b"grinding-test", b"instance");
        let mut untouched = AkitaTranscript::<F>::prover(b"grinding-test", b"instance");
        previewed.absorb_bytes(crate::label!("context"), b"fixed history");
        untouched.absorb_bytes(crate::label!("context"), b"fixed history");

        assert!(search_grinding_nonce(&previewed, 4, 11).is_some());
        assert_eq!(
            previewed.squeeze_bytes(crate::label!("next"), SQUEEZE_CHUNK_LEN),
            untouched.squeeze_bytes(crate::label!("next"), SQUEEZE_CHUNK_LEN)
        );
    }

    #[test]
    fn zero_bit_grinding_is_a_transcript_no_op() {
        let mut no_op = AkitaTranscript::<F>::prover(b"grinding-test", b"instance");
        let mut untouched = AkitaTranscript::<F>::prover(b"grinding-test", b"instance");

        assert_eq!(
            Transcript::grinding_predicate(&mut no_op, crate::labels::CHALLENGE_TAU0, 0, 0, 0,),
            None
        );
        assert_eq!(
            no_op.squeeze_bytes(crate::label!("next"), SQUEEZE_CHUNK_LEN),
            untouched.squeeze_bytes(crate::label!("next"), SQUEEZE_CHUNK_LEN)
        );
    }
}
