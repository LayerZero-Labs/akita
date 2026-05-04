//! Protocol-level Fiat–Shamir challenge samplers.
//!
//! These utilities derive structured challenges (e.g. sparse ring elements) from
//! the transcript while keeping the low-level representations in the algebra layer.

pub mod rejection;
pub mod sparse;

use crate::rejection::{
    sample_challenges, sample_sparse_challenges as sample_rejection_sparse_challenges,
};
use akita_algebra::fields::lift::ExtField;
use akita_algebra::ring::CyclotomicRing;
use akita_algebra::SparseChallenge;
use akita_field::AkitaError;
use akita_field::{CanonicalField, FieldCore, FromSmallInt};
use akita_transcript::Transcript;

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
/// Squeezes a 16-byte seed from the transcript, then delegates to the
/// rejection sampler which produces a polynomial with exactly `TAU1` coefficients
/// in {+/-1} and `TAU2` in {+/-2}, retrying until the operator norm is bounded.
///
/// # Errors
///
/// Returns an error if `D` is incompatible with the rejection sampler.
pub fn challenge_ring_element_rejection_sampled<F, T, const D: usize>(
    tr: &mut T,
    label: &[u8],
) -> Result<CyclotomicRing<F, D>, AkitaError>
where
    F: FieldCore + CanonicalField + FromSmallInt,
    T: Transcript<F>,
{
    let mut polys = challenge_ring_elements_rejection_sampled::<F, T, D>(tr, label, 1)?;
    polys
        .pop()
        .ok_or_else(|| AkitaError::InvalidInput("rejection sampler produced no output".into()))
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
) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>
where
    F: FieldCore + CanonicalField + FromSmallInt,
    T: Transcript<F>,
{
    let seed_vec = tr.challenge_bytes(label, 16);
    let seed: [u8; 16] = seed_vec
        .try_into()
        .map_err(|_| AkitaError::InvalidInput("rejection sampler seed length mismatch".into()))?;
    sample_challenges::<F, D>(len, &seed, REJECTION_SAMPLER_SINGLE_NONCE)
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
) -> Result<Vec<SparseChallenge>, AkitaError>
where
    F: FieldCore + CanonicalField + FromSmallInt,
    T: Transcript<F>,
{
    let seed_vec = tr.challenge_bytes(label, 16);
    let seed: [u8; 16] = seed_vec
        .try_into()
        .map_err(|_| AkitaError::InvalidInput("rejection sampler seed length mismatch".into()))?;
    sample_rejection_sparse_challenges::<D>(len, &seed, REJECTION_SAMPLER_SINGLE_NONCE)
}

/// Evaluate a sparse ring challenge against precomputed scalar powers.
///
/// # Errors
///
/// Returns an error when `alpha_pows` does not have length `D`.
pub fn eval_sparse_challenge_at_pows<F: FieldCore + CanonicalField, const D: usize>(
    challenge: &SparseChallenge,
    alpha_pows: &[F],
) -> Result<F, AkitaError> {
    if alpha_pows.len() != D {
        return Err(AkitaError::InvalidSize {
            expected: D,
            actual: alpha_pows.len(),
        });
    }

    debug_assert_eq!(challenge.positions.len(), challenge.coeffs.len());

    let mut acc = F::zero();
    for (&pos, &coeff) in challenge.positions.iter().zip(challenge.coeffs.iter()) {
        let idx = pos as usize;
        debug_assert!(idx < D);
        debug_assert_ne!(coeff, 0);
        acc += F::from_i64(coeff as i64) * alpha_pows[idx];
    }
    Ok(acc)
}
