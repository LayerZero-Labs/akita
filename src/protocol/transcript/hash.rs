//! Generic hash-based transcript for protocol-layer Fiat-Shamir.
//!
//! Parameterised over any `Digest + Clone` hasher, eliminating the
//! near-identical Blake2b and Keccak implementations.

use super::Transcript;
use crate::primitives::serialization::HachiSerialize;
use crate::{CanonicalField, FieldCore};
use blake2::{Blake2b512, Digest};
use sha3::Keccak256;
use std::marker::PhantomData;

/// Hash-based transcript with labeled framing.
///
/// Works with any cryptographic hash that implements `Digest + Clone`.
#[derive(Clone)]
pub struct HashTranscript<D: Digest + Clone, F>
where
    F: FieldCore + CanonicalField + 'static,
{
    hasher: D,
    _field: PhantomData<F>,
}

impl<D: Digest + Clone, F> HashTranscript<D, F>
where
    F: FieldCore + CanonicalField + 'static,
{
    #[inline]
    fn append_bytes_impl(&mut self, label: &[u8], bytes: &[u8]) {
        self.hasher.update(label);
        self.hasher.update((bytes.len() as u64).to_le_bytes());
        self.hasher.update(bytes);
    }

    #[inline]
    fn challenge_and_chain(&mut self, label: &[u8]) -> Vec<u8> {
        self.hasher.update(label);
        let digest = self.hasher.clone().finalize();
        let out = digest.to_vec();
        self.hasher.update(&out);
        out
    }
}

impl<D: Digest + Clone + Send + Sync + 'static, F> Transcript<F> for HashTranscript<D, F>
where
    F: FieldCore + CanonicalField + 'static,
{
    fn new(domain_label: &[u8]) -> Self {
        let mut hasher = D::new();
        hasher.update(domain_label);
        Self {
            hasher,
            _field: PhantomData,
        }
    }

    fn append_bytes(&mut self, label: &[u8], bytes: &[u8]) {
        self.append_bytes_impl(label, bytes);
    }

    fn append_field(&mut self, label: &[u8], x: &F) {
        self.append_bytes_impl(label, &x.to_canonical_u128().to_le_bytes());
    }

    fn append_serde<S: HachiSerialize>(&mut self, label: &[u8], s: &S) {
        let mut bytes = Vec::new();
        s.serialize_compressed(&mut bytes)
            .expect("HachiSerialize should not fail");
        self.append_bytes_impl(label, &bytes);
    }

    fn challenge_scalar(&mut self, label: &[u8]) -> F {
        let bytes = self.challenge_and_chain(label);
        let mut lo = [0u8; 16];
        lo.copy_from_slice(&bytes[..16]);
        let sampled = u128::from_le_bytes(lo);
        F::from_canonical_u128_reduced(sampled)
    }
}

impl<D: Digest + Clone, F> HashTranscript<D, F>
where
    F: FieldCore + CanonicalField + 'static,
{
    /// Reset transcript state under a new domain label.
    ///
    /// This is an inherent method (not part of the `Transcript` trait) to
    /// discourage use in production protocol code where resetting the
    /// Fiat-Shamir chain would be unsound.
    pub fn reset(&mut self, domain_label: &[u8]) {
        let mut hasher = D::new();
        hasher.update(domain_label);
        self.hasher = hasher;
    }
}

/// Blake2b512 transcript with labeled framing.
pub type Blake2bTranscript<F> = HashTranscript<Blake2b512, F>;

/// Keccak256 transcript with labeled framing.
pub type KeccakTranscript<F> = HashTranscript<Keccak256, F>;
