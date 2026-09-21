//! Exact canonical rejection sampling for native field challenges.

use super::{NativeContextError, NativeProverState, NativeVerifierState};
use crate::TranscriptSponge;
use jolt_field::CanonicalEncoding;
use spongefish::DuplexSpongeInterface;

/// Maximum candidate width supported by native exact field sampling.
pub const NATIVE_FIELD_CHALLENGE_BYTES: u64 = 64;

/// Global cap used to account for declared field-challenge oracle queries.
///
/// This is a protocol query budget, not a bound on rejection-sampling retries.
pub const NATIVE_FIELD_SAMPLING_QUERY_LIMIT: u64 = u32::MAX as u64;

/// Certify that exact rejection sampling supports a canonical field encoding.
#[must_use]
pub const fn native_field_sampling_is_certified(
    canonical_bytes: usize,
    modulus_bits: u32,
    query_budget: u64,
) -> bool {
    let Some(canonical_bits) = canonical_bytes.checked_mul(u8::BITS as usize) else {
        return false;
    };
    canonical_bytes > 0
        && canonical_bytes <= NATIVE_FIELD_CHALLENGE_BYTES as usize
        && modulus_bits > 0
        && (modulus_bits as usize) <= canonical_bits
        && canonical_bits - (modulus_bits as usize) < u8::BITS as usize
        && query_budget <= NATIVE_FIELD_SAMPLING_QUERY_LIMIT
}

/// Candidate bytes consumed by one attempt of exact native field sampling.
#[must_use]
pub const fn native_field_challenge_bytes<F: CanonicalEncoding>() -> u64 {
    F::NUM_BYTES as u64
}

fn sample_native_field<F: CanonicalEncoding>(
    sponge: &mut TranscriptSponge,
) -> Result<F, NativeContextError> {
    if !native_field_sampling_is_certified(
        F::NUM_BYTES,
        F::MODULUS_BITS,
        NATIVE_FIELD_SAMPLING_QUERY_LIMIT,
    ) {
        return Err(NativeContextError);
    }
    let mut candidate = [0u8; NATIVE_FIELD_CHALLENGE_BYTES as usize];
    let width = F::NUM_BYTES;
    let modulus_bits = F::MODULUS_BITS as usize;
    let excess_bits = width * 8 - modulus_bits;
    #[cfg(feature = "transcript-keccak")]
    let mut attempts = 0u32;
    loop {
        #[cfg(feature = "transcript-keccak")]
        {
            attempts = attempts.checked_add(1).ok_or(NativeContextError)?;
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
pub fn native_prover_field_challenge<F: CanonicalEncoding>(
    state: &mut NativeProverState,
) -> Result<F, NativeContextError> {
    sample_native_field(&mut state.duplex_sponge_state)
}

/// Draw one exactly uniform base-field challenge by canonical rejection sampling.
pub fn native_verifier_field_challenge<F: CanonicalEncoding>(
    state: &mut NativeVerifierState<'_>,
) -> Result<F, NativeContextError> {
    sample_native_field(&mut state.duplex_sponge_state)
}
