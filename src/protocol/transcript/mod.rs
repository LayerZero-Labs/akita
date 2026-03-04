//! Protocol transcript contracts and implementations.

mod hash;
pub mod labels;

use crate::algebra::fields::lift::ExtField;
use crate::algebra::ring::CyclotomicRing;
use crate::error::HachiError;
use crate::protocol::labrador::challenge::sample_labrador_challenge_coeffs;
use crate::{CanonicalField, FieldCore, FromSmallInt, HachiSerialize};

pub use hash::{Blake2bTranscript, HashTranscript, KeccakTranscript};

/// Fixed nonce for single-polynomial rejection sampling. The seed is
/// already transcript-derived and unique per challenge invocation, so
/// the nonce is only needed for batching (`len > 1`). When sampling a
/// single polynomial the nonce carries no additional entropy.
const REJECTION_SAMPLER_SINGLE_NONCE: u64 = 0;

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

/// Sample a dense ring-element challenge by drawing `D` scalar challenges.
pub fn challenge_ring_element<F, T, const D: usize>(
    tr: &mut T,
    label: &[u8],
) -> CyclotomicRing<F, D>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
{
    CyclotomicRing::from_coefficients(std::array::from_fn(|_| tr.challenge_scalar(label)))
}

/// Sample a sparse ring-element challenge with operator-norm rejection sampling.
///
/// Squeezes a 16-byte seed from the transcript, then delegates to the Labrador
/// rejection sampler (`sample_labrador_challenge_coeffs`) which produces a
/// polynomial with exactly `TAU1` coefficients in {±1} and `TAU2` in {±2},
/// retrying until the operator norm is at most `LABRADOR_CHALLENGE_OPNORM_BOUND`.
///
/// # Errors
///
/// Returns an error if `D` is incompatible with the rejection sampler
/// (must be a power of two, at most 256, and >= TAU1 + TAU2).
pub fn challenge_ring_element_rejection_sampled<F, T, const D: usize>(
    tr: &mut T,
    label: &[u8],
) -> Result<CyclotomicRing<F, D>, HachiError>
where
    F: FieldCore + CanonicalField + FromSmallInt,
    T: Transcript<F>,
{
    let mut seed = [0u8; 16];
    for chunk in seed.chunks_mut(8) {
        let s = tr.challenge_scalar(label);
        let v = s.to_canonical_u128();
        let len = chunk.len();
        chunk.copy_from_slice(&v.to_le_bytes()[..len]);
    }
    let coeffs = sample_labrador_challenge_coeffs::<D>(1, &seed, REJECTION_SAMPLER_SINGLE_NONCE)?;
    let poly = coeffs
        .into_iter()
        .next()
        .ok_or_else(|| HachiError::InvalidInput("rejection sampler produced no output".into()))?;
    Ok(CyclotomicRing::from_coefficients(std::array::from_fn(
        |i| F::from_i64(poly[i] as i64),
    )))
}
