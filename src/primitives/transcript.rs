//! Transcript trait for Fiat-Shamir transformations

#![allow(missing_docs)]

use akita_field::{CanonicalField, FieldCore, Module};
use akita_serialization::HachiSerialize;

/// Transcript for Fiat-Shamir transformations
pub trait Transcript {
    /// Field type for challenges
    type Field: FieldCore + CanonicalField;

    /// Append raw bytes to the transcript
    fn append_bytes(&mut self, label: &[u8], bytes: &[u8]);

    /// Append a field element to the transcript
    fn append_field(&mut self, label: &[u8], x: &Self::Field);

    /// Append a module element to the transcript
    fn append_module<M: Module>(&mut self, label: &[u8], m: &M);

    /// Append a serializable element to the transcript
    fn append_serde<S: HachiSerialize>(&mut self, label: &[u8], s: &S);

    /// Generate a challenge scalar from the transcript
    fn challenge_scalar(&mut self, label: &[u8]) -> Self::Field;

    /// Reset the transcript with a new domain label
    fn reset(&mut self, domain_label: &[u8]);
}
