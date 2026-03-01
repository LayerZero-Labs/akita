//! Protocol transcript contracts and implementations.

mod hash;
pub mod labels;

use crate::algebra::fields::lift::ExtField;
use crate::{CanonicalField, FieldCore, HachiSerialize};

pub use hash::{Blake2bTranscript, HashTranscript, KeccakTranscript};

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
}

/// Sample an extension field challenge by drawing `EXT_DEGREE` base-field
/// challenges and assembling them via `from_base_slice`.
///
/// When `E = F` (degree 1), this compiles to a single `challenge_scalar` call.
pub fn sample_ext_challenge<F, E, T>(tr: &mut T, label: &[u8]) -> E
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
    E: ExtField<F>,
{
    E::from_base_slice(
        &(0..E::EXT_DEGREE)
            .map(|_| tr.challenge_scalar(label))
            .collect::<Vec<_>>(),
    )
}
