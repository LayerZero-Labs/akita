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
        self.inner.append_bytes(label);
        self.inner.challenge()
    }
}

#[inline]
fn labeled_payload(label: &[u8], bytes: &[u8]) -> Vec<u8> {
    let label_len = u8::try_from(label.len()).expect("transcript labels must fit in one byte");
    let mut out = Vec::with_capacity(1 + label.len() + bytes.len());
    out.push(label_len);
    out.extend_from_slice(label);
    out.extend_from_slice(bytes);
    out
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
        let mut bytes = Vec::new();
        s.serialize_compressed(&mut bytes)
            .expect("AkitaSerialize should not fail");
        self.transcript.append_labeled_bytes(label, &bytes);
    }

    fn challenge_scalar(&mut self, label: &[u8]) -> F {
        self.transcript.challenge_labeled(label)
    }

    fn challenge_bytes(&mut self, label: &[u8], len: usize) -> Vec<u8> {
        self.transcript.inner.append_bytes(label);
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
        let mut bytes = Vec::new();
        s.serialize_compressed(&mut bytes)
            .expect("AkitaSerialize should not fail");
        self.transcript.append_labeled_bytes(label, &bytes);
    }

    fn challenge_scalar(&mut self, label: &[u8]) -> F {
        self.transcript.challenge_labeled(label)
    }

    fn challenge_bytes(&mut self, label: &[u8], len: usize) -> Vec<u8> {
        self.transcript.inner.append_bytes(label);
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
