//! Protocol transcript contracts and implementations.

mod hash;
pub mod labels;

use akita_field::{CanonicalField, FieldCore};
use akita_serialization::AkitaSerialize;

pub use hash::{Blake2bTranscript, KeccakTranscript};

/// Transcript interface for protocol Fiat-Shamir transforms.
///
/// The protocol layer is label-aware and uses deterministic byte encoding for
/// all absorbed values.
pub trait Transcript<F>: Clone + Send + Sync + 'static
where
    F: FieldCore + CanonicalField,
{
    /// Construct a new transcript under a domain label.
    fn new(domain_label: &[u8]) -> Self;

    /// Append labeled raw bytes.
    fn append_bytes(&mut self, label: &[u8], bytes: &[u8]);

    /// Append a field element with deterministic encoding.
    fn append_field(&mut self, label: &[u8], x: &F);

    /// Append a serializable protocol value.
    fn append_serde<S: AkitaSerialize>(&mut self, label: &[u8], s: &S);

    /// Derive a challenge scalar under the provided label.
    fn challenge_scalar(&mut self, label: &[u8]) -> F;

    /// Squeeze `len` challenge bytes under the provided label.
    fn challenge_bytes(&mut self, label: &[u8], len: usize) -> Vec<u8>;
}

/// Sample `n` scalar challenges under the same transcript label.
pub fn sample_challenge_scalars<F, T>(transcript: &mut T, label: &[u8], n: usize) -> Vec<F>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
{
    (0..n).map(|_| transcript.challenge_scalar(label)).collect()
}
