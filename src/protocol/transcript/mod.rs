//! Protocol transcript contracts and implementations.

mod blake2b;
mod keccak;
pub mod labels;

use crate::{CanonicalField, FieldCore, HachiSerialize};

pub use blake2b::Blake2bTranscript;
pub use keccak::KeccakTranscript;

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
    fn append_serde<S: HachiSerialize>(&mut self, label: &[u8], s: &S);

    /// Derive a challenge scalar under the provided label.
    fn challenge_scalar(&mut self, label: &[u8]) -> F;

    /// Reset transcript state under a new domain label.
    fn reset(&mut self, domain_label: &[u8]);
}
