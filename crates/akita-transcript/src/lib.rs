//! Protocol transcript contracts and implementations.

mod hash;
mod label;
pub mod labels;
mod sponge;

use akita_field::{CanonicalField, ExtField, FieldCore};
use akita_serialization::AkitaSerialize;

pub use hash::{Blake2bTranscript, KeccakTranscript};
pub use label::Label;
pub use sponge::{AkitaTranscript, TranscriptSponge, PROTOCOL_TAG};

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

/// Append an extension-field element by absorbing its base-field coordinates.
pub fn append_ext_field<F, E, T>(transcript: &mut T, label: &[u8], x: &E)
where
    F: FieldCore + CanonicalField,
    E: ExtField<F>,
    T: Transcript<F>,
{
    let coeffs = x.to_base_vec();
    if E::EXT_DEGREE == 1 {
        for coeff in coeffs.iter().take(1) {
            transcript.append_field(label, coeff);
        }
        return;
    }

    for (limb, coeff) in coeffs.iter().enumerate() {
        transcript.append_field(&ext_limb_label(label, limb), coeff);
    }
}

/// Sample an extension-field challenge from base-field transcript limbs.
///
/// This draws `E::EXT_DEGREE` base-field challenges under distinct limb labels
/// and assembles the extension element with [`ExtField::from_base_slice`].
pub fn sample_ext_challenge<F, E, T>(transcript: &mut T, label: &[u8]) -> E
where
    F: FieldCore + CanonicalField,
    E: ExtField<F>,
    T: Transcript<F>,
{
    if E::EXT_DEGREE == 1 {
        let coeff = transcript.challenge_scalar(label);
        return E::from_base_slice(&[coeff]);
    }

    let coeffs = (0..E::EXT_DEGREE)
        .map(|limb| transcript.challenge_scalar(&ext_limb_label(label, limb)))
        .collect::<Vec<_>>();
    E::from_base_slice(&coeffs)
}

fn ext_limb_label(label: &[u8], limb: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(label.len() + 17);
    out.extend_from_slice(label);
    out.push(0xff);
    out.extend_from_slice(&(limb as u64).to_le_bytes());
    out.extend_from_slice(b"ext");
    out
}
