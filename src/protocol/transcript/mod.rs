//! Protocol transcript contracts and implementations.

mod hash;
pub mod labels;

use crate::algebra::fields::lift::ExtField;
use crate::algebra::ring::CyclotomicRing;
use crate::algebra::SparseChallenge;
use crate::error::HachiError;
use crate::protocol::labrador::challenge::{
    sample_labrador_challenges, sample_labrador_sparse_challenges,
};
use crate::{CanonicalField, FieldCore, FromSmallInt, HachiSerialize};

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

    /// Squeeze `len` challenge bytes under the provided label.
    fn challenge_bytes(&mut self, label: &[u8], len: usize) -> Vec<u8>;
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

/// Fixed nonce for single-polynomial rejection sampling.
const REJECTION_SAMPLER_SINGLE_NONCE: u64 = 0;

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
/// rejection sampler which produces a polynomial with exactly `TAU1` coefficients
/// in {+/-1} and `TAU2` in {+/-2}, retrying until the operator norm is bounded.
///
/// # Errors
///
/// Returns an error if `D` is incompatible with the rejection sampler.
pub fn challenge_ring_element_rejection_sampled<F, T, const D: usize>(
    tr: &mut T,
    label: &[u8],
) -> Result<CyclotomicRing<F, D>, HachiError>
where
    F: FieldCore + CanonicalField + FromSmallInt,
    T: Transcript<F>,
{
    let mut polys = challenge_ring_elements_rejection_sampled::<F, T, D>(tr, label, 1)?;
    polys
        .pop()
        .ok_or_else(|| HachiError::InvalidInput("rejection sampler produced no output".into()))
}

/// Sample multiple sparse ring-element challenges from one transcript-bound seed.
///
/// # Errors
///
/// Returns an error if `D` is incompatible with the rejection sampler.
pub fn challenge_ring_elements_rejection_sampled<F, T, const D: usize>(
    tr: &mut T,
    label: &[u8],
    len: usize,
) -> Result<Vec<CyclotomicRing<F, D>>, HachiError>
where
    F: FieldCore + CanonicalField + FromSmallInt,
    T: Transcript<F>,
{
    let seed_vec = tr.challenge_bytes(label, 16);
    let seed: [u8; 16] = seed_vec
        .try_into()
        .map_err(|_| HachiError::InvalidInput("rejection sampler seed length mismatch".into()))?;
    sample_labrador_challenges::<F, D>(len, &seed, REJECTION_SAMPLER_SINGLE_NONCE)
}

/// Sample multiple sparse ring-element challenges from one transcript-bound seed.
///
/// # Errors
///
/// Returns an error if `D` is incompatible with the rejection sampler.
pub fn challenge_sparse_ring_elements_rejection_sampled<F, T, const D: usize>(
    tr: &mut T,
    label: &[u8],
    len: usize,
) -> Result<Vec<SparseChallenge>, HachiError>
where
    F: FieldCore + CanonicalField + FromSmallInt,
    T: Transcript<F>,
{
    let seed_vec = tr.challenge_bytes(label, 16);
    let seed: [u8; 16] = seed_vec
        .try_into()
        .map_err(|_| HachiError::InvalidInput("rejection sampler seed length mismatch".into()))?;
    sample_labrador_sparse_challenges::<D>(len, &seed, REJECTION_SAMPLER_SINGLE_NONCE)
}
