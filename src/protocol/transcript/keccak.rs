//! Keccak transcript implementation for protocol-layer Fiat-Shamir.

use super::Transcript;
use crate::primitives::serialization::HachiSerialize;
use crate::{CanonicalField, FieldCore};
use sha3::{Digest, Keccak256};
use std::marker::PhantomData;

/// Keccak256 transcript with labeled framing.
#[derive(Clone)]
pub struct KeccakTranscript<F>
where
    F: FieldCore + CanonicalField + 'static,
{
    hasher: Keccak256,
    _field: PhantomData<F>,
}

impl<F> KeccakTranscript<F>
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
    fn challenge_bytes(&mut self, label: &[u8]) -> [u8; 32] {
        self.hasher.update(label);
        let digest = self.hasher.clone().finalize();
        let mut out = [0u8; 32];
        out.copy_from_slice(digest.as_slice());
        self.hasher.update(out);
        out
    }
}

impl<F> Transcript<F> for KeccakTranscript<F>
where
    F: FieldCore + CanonicalField + 'static,
{
    fn new(domain_label: &[u8]) -> Self {
        let mut hasher = Keccak256::new();
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
        let bytes = self.challenge_bytes(label);
        let mut lo = [0u8; 16];
        lo.copy_from_slice(&bytes[..16]);
        let sampled = u128::from_le_bytes(lo);
        F::from_canonical_u128_reduced(sampled)
    }

    fn reset(&mut self, domain_label: &[u8]) {
        let mut hasher = Keccak256::new();
        hasher.update(domain_label);
        self.hasher = hasher;
    }
}
