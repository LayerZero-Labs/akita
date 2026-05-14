//! Jolt-backed Fiat-Shamir transcripts for Akita's label-aware protocol API.

use crate::Transcript;
use akita_field::{CanonicalBytes, CanonicalField, FieldCore, TranscriptChallenge};
use akita_serialization::AkitaSerialize;
use jolt_transcript::KeccakTranscript as JoltKeccakTranscript;
use jolt_transcript::{Blake2bTranscript as JoltBlake2bTranscript, Transcript as JoltTranscript};

#[derive(Clone)]
struct LabeledJoltTranscript<T> {
    inner: T,
}

impl<T> LabeledJoltTranscript<T>
where
    T: JoltTranscript,
{
    fn new_with_domain(domain_label: &[u8]) -> Self {
        let mut inner = T::default();
        inner.append_bytes(domain_label);
        Self { inner }
    }

    #[inline]
    fn append_labeled_bytes(&mut self, label: &[u8], bytes: &[u8]) {
        self.inner.append_bytes(&labeled_payload(label, bytes));
    }

    #[inline]
    fn challenge_labeled(&mut self, label: &[u8]) -> T::Challenge {
        self.inner.append_bytes(&labeled_payload(label, &[]));
        self.inner.challenge()
    }

    #[inline]
    fn append_challenge_label(&mut self, label: &[u8]) {
        self.inner.append_bytes(&labeled_payload(label, &[]));
    }
}

#[inline]
fn labeled_payload(label: &[u8], bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(9 + label.len() + bytes.len());
    if let Ok(label_len) = u8::try_from(label.len()) {
        out.push(label_len);
    } else {
        out.push(u8::MAX);
        out.extend_from_slice(&(label.len() as u64).to_le_bytes());
    }
    out.extend_from_slice(label);
    out.extend_from_slice(bytes);
    out
}

#[inline]
fn serialized_payload<S: AkitaSerialize>(s: &S) -> Vec<u8> {
    let mut bytes = Vec::new();
    if s.serialize_compressed(&mut bytes).is_err() {
        bytes.extend_from_slice(b"AKITA_SERIALIZE_ERROR");
    }
    bytes
}

#[inline]
fn field_transcript_bytes<F: CanonicalBytes>(x: &F) -> Vec<u8> {
    let mut bytes = vec![0u8; F::NUM_BYTES];
    x.to_bytes_le(&mut bytes);
    bytes.reverse();
    bytes
}

/// Blake2b-256 transcript backed by Jolt's digest transcript engine.
#[derive(Clone)]
pub struct Blake2bTranscript<F>
where
    F: TranscriptChallenge,
{
    transcript: LabeledJoltTranscript<JoltBlake2bTranscript<F>>,
}

impl<F> Blake2bTranscript<F>
where
    F: FieldCore + CanonicalField + CanonicalBytes + TranscriptChallenge + 'static,
{
    /// Construct a transcript under a domain label.
    pub fn new(domain_label: &[u8]) -> Self {
        <Self as Transcript<F>>::new(domain_label)
    }

    /// Reset transcript state under a new domain label.
    ///
    /// This is an inherent method (not part of the `Transcript` trait) to
    /// discourage use in production protocol code where resetting the
    /// Fiat-Shamir chain would be unsound.
    pub fn reset(&mut self, domain_label: &[u8]) {
        *self = Self::new(domain_label);
    }
}

impl<F> Transcript<F> for Blake2bTranscript<F>
where
    F: FieldCore + CanonicalField + CanonicalBytes + TranscriptChallenge + 'static,
{
    fn new(domain_label: &[u8]) -> Self {
        Self {
            transcript: LabeledJoltTranscript::new_with_domain(domain_label),
        }
    }

    fn append_bytes(&mut self, label: &[u8], bytes: &[u8]) {
        self.transcript.append_labeled_bytes(label, bytes);
    }

    fn append_field(&mut self, label: &[u8], x: &F) {
        self.transcript
            .append_labeled_bytes(label, &field_transcript_bytes(x));
    }

    fn append_serde<S: AkitaSerialize>(&mut self, label: &[u8], s: &S) {
        self.transcript
            .append_labeled_bytes(label, &serialized_payload(s));
    }

    fn challenge_scalar(&mut self, label: &[u8]) -> F {
        self.transcript.challenge_labeled(label)
    }

    fn challenge_bytes(&mut self, label: &[u8], len: usize) -> Vec<u8> {
        self.transcript.append_challenge_label(label);
        let mut out = Vec::with_capacity(len);
        while out.len() < len {
            let challenge: F = self.transcript.inner.challenge();
            let mut bytes = vec![0u8; F::NUM_BYTES];
            challenge.to_bytes_le(&mut bytes);
            let take = (len - out.len()).min(bytes.len());
            out.extend_from_slice(&bytes[..take]);
        }
        out
    }
}

/// Keccak-256 transcript backed by Jolt's digest transcript engine.
#[derive(Clone)]
pub struct KeccakTranscript<F>
where
    F: TranscriptChallenge,
{
    transcript: LabeledJoltTranscript<JoltKeccakTranscript<F>>,
}

impl<F> KeccakTranscript<F>
where
    F: FieldCore + CanonicalField + CanonicalBytes + TranscriptChallenge + 'static,
{
    /// Construct a transcript under a domain label.
    pub fn new(domain_label: &[u8]) -> Self {
        <Self as Transcript<F>>::new(domain_label)
    }

    /// Reset transcript state under a new domain label.
    ///
    /// This is an inherent method (not part of the `Transcript` trait) to
    /// discourage use in production protocol code where resetting the
    /// Fiat-Shamir chain would be unsound.
    pub fn reset(&mut self, domain_label: &[u8]) {
        *self = Self::new(domain_label);
    }
}

impl<F> Transcript<F> for KeccakTranscript<F>
where
    F: FieldCore + CanonicalField + CanonicalBytes + TranscriptChallenge + 'static,
{
    fn new(domain_label: &[u8]) -> Self {
        Self {
            transcript: LabeledJoltTranscript::new_with_domain(domain_label),
        }
    }

    fn append_bytes(&mut self, label: &[u8], bytes: &[u8]) {
        self.transcript.append_labeled_bytes(label, bytes);
    }

    fn append_field(&mut self, label: &[u8], x: &F) {
        self.transcript
            .append_labeled_bytes(label, &field_transcript_bytes(x));
    }

    fn append_serde<S: AkitaSerialize>(&mut self, label: &[u8], s: &S) {
        self.transcript
            .append_labeled_bytes(label, &serialized_payload(s));
    }

    fn challenge_scalar(&mut self, label: &[u8]) -> F {
        self.transcript.challenge_labeled(label)
    }

    fn challenge_bytes(&mut self, label: &[u8], len: usize) -> Vec<u8> {
        self.transcript.append_challenge_label(label);
        let mut out = Vec::with_capacity(len);
        while out.len() < len {
            let challenge: F = self.transcript.inner.challenge();
            let mut bytes = vec![0u8; F::NUM_BYTES];
            challenge.to_bytes_le(&mut bytes);
            let take = (len - out.len()).min(bytes.len());
            out.extend_from_slice(&bytes[..take]);
        }
        out
    }
}

#[cfg(all(test, not(feature = "zk")))]
mod tests {
    use super::*;
    use akita_field::{FixedByteSize, Prime128Offset275};

    type F = Prime128Offset275;

    fn expected_blake2b_challenge_bytes(domain: &[u8], label: &[u8], len: usize) -> Vec<u8> {
        let mut inner = JoltBlake2bTranscript::<F>::default();
        inner.append_bytes(domain);
        inner.append_bytes(&labeled_payload(label, &[]));

        let mut out = Vec::with_capacity(len);
        while out.len() < len {
            let challenge: F = inner.challenge();
            let mut bytes = vec![0u8; F::NUM_BYTES];
            challenge.to_bytes_le(&mut bytes);
            let take = (len - out.len()).min(bytes.len());
            out.extend_from_slice(&bytes[..take]);
        }
        out
    }

    #[test]
    fn challenge_scalar_uses_framed_label() {
        let domain = b"transcript-test";
        let label = b"challenge";

        let mut transcript = Blake2bTranscript::<F>::new(domain);
        let got = transcript.challenge_scalar(label);

        let mut inner = JoltBlake2bTranscript::<F>::default();
        inner.append_bytes(domain);
        inner.append_bytes(&labeled_payload(label, &[]));
        let expected: F = inner.challenge();

        assert_eq!(got, expected);
    }

    #[test]
    fn challenge_bytes_uses_framed_label() {
        let domain = b"transcript-test";
        let label = b"bytes";
        let len = F::NUM_BYTES + 7;

        let mut transcript = Blake2bTranscript::<F>::new(domain);
        let got = transcript.challenge_bytes(label, len);
        let expected = expected_blake2b_challenge_bytes(domain, label, len);

        assert_eq!(got, expected);
    }
}
