//! Exact canonical rejection sampling for field challenges.

use super::{ProverChannel, VerifierChannel};
use crate::TranscriptSponge;
use akita_error::AkitaError;
use jolt_field::CanonicalEncoding;
use spongefish::DuplexSpongeInterface;

/// Maximum candidate width supported by exact field sampling.
pub const FIELD_CHALLENGE_BYTES: u64 = 64;

/// Global cap used to account for declared field-challenge oracle queries.
///
/// This is a protocol query budget, not a bound on rejection-sampling retries.
pub const FIELD_SAMPLING_QUERY_LIMIT: u64 = u32::MAX as u64;

/// Certify that exact rejection sampling supports a canonical field encoding.
#[must_use]
pub const fn field_sampling_is_certified(
    canonical_bytes: usize,
    modulus_bits: u32,
    query_budget: u64,
) -> bool {
    let Some(canonical_bits) = canonical_bytes.checked_mul(u8::BITS as usize) else {
        return false;
    };
    canonical_bytes > 0
        && canonical_bytes <= FIELD_CHALLENGE_BYTES as usize
        && modulus_bits > 0
        && (modulus_bits as usize) <= canonical_bits
        && canonical_bits - (modulus_bits as usize) < u8::BITS as usize
        && query_budget <= FIELD_SAMPLING_QUERY_LIMIT
}

/// Candidate bytes consumed by one attempt of exact field sampling.
#[must_use]
pub const fn field_challenge_bytes<F: CanonicalEncoding>() -> u64 {
    F::NUM_BYTES as u64
}

fn sample_field<F: CanonicalEncoding>(sponge: &mut TranscriptSponge) -> Result<F, AkitaError> {
    if !field_sampling_is_certified(F::NUM_BYTES, F::MODULUS_BITS, FIELD_SAMPLING_QUERY_LIMIT) {
        return Err(AkitaError::InvalidProof);
    }
    let mut candidate = [0u8; FIELD_CHALLENGE_BYTES as usize];
    let width = F::NUM_BYTES;
    let modulus_bits = F::MODULUS_BITS as usize;
    let excess_bits = width * 8 - modulus_bits;
    #[cfg(feature = "transcript-keccak")]
    let mut attempts = 0u32;
    loop {
        #[cfg(feature = "transcript-keccak")]
        {
            attempts = attempts.checked_add(1).ok_or(AkitaError::InvalidProof)?;
        }
        sponge.squeeze(&mut candidate[..width]);
        if excess_bits != 0 {
            candidate[width - 1] &= u8::MAX >> excess_bits;
        }
        if let Some(value) = F::from_bytes_le_checked(&candidate[..width]) {
            // The pinned Keccak duplex forgets squeeze length after a later
            // absorb. Bind rejection-path length so distinct retry histories
            // cannot reconverge when the next protocol message is absorbed.
            #[cfg(feature = "transcript-keccak")]
            sponge.absorb(&attempts.to_le_bytes());
            return Ok(value);
        }
    }
}

/// Draw one exactly uniform base-field challenge by canonical rejection sampling.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidProof`] when `F` is not certified for exact
/// sampling.
pub fn prover_field_challenge<F: CanonicalEncoding>(
    state: &mut ProverChannel,
) -> Result<F, AkitaError> {
    sample_field(&mut state.duplex_sponge_state)
}

/// Draw one exactly uniform base-field challenge by canonical rejection sampling.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidProof`] when `F` is not certified for exact
/// sampling or the verifier state is already invalid.
pub fn verifier_field_challenge<F: CanonicalEncoding>(
    state: &mut VerifierChannel<'_>,
) -> Result<F, AkitaError> {
    if state.is_invalid() {
        return Err(AkitaError::InvalidProof);
    }
    let result = sample_field(state.sponge_mut());
    if result.is_err() {
        state.invalidate();
    }
    result
}
