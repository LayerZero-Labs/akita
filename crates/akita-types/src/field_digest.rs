//! Digests of field values that the verifier recomputes instead of receiving.

use akita_error::AkitaError;
pub use akita_params::FIELD_DIGEST_BYTES;
use jolt_algebra::CanonicalBytes;
use jolt_transcript::{Channel, SiteId, Sponge, VerifierTranscript};

/// Domain tag absorbed ahead of the values of every [`field_digest`].
const FIELD_DIGEST_DOMAIN: [u8; 32] = *b"akita-pcs/field-digest/v1\0\0\0\0\0\0\0";

/// Collision-resistant digest of the canonical encoding of `values`.
///
/// The digest runs on a fresh `H`, independent of any proof transcript, so the
/// prover can bind values the verifier only learns later. A proof sends it as
/// a fixed-length byte message in place of the values.
#[must_use]
pub fn field_digest<H: Sponge, F: CanonicalBytes>(values: &[F]) -> [u8; FIELD_DIGEST_BYTES] {
    let mut sponge = H::default();
    sponge.absorb(&FIELD_DIGEST_DOMAIN);
    let mut encoded = vec![0u8; F::NUM_BYTES];
    for value in values {
        value.to_bytes_le(&mut encoded);
        sponge.absorb(&encoded);
    }
    let mut digest = [0u8; FIELD_DIGEST_BYTES];
    sponge.squeeze(&mut digest);
    digest
}

/// Send the [`field_digest`] of `values` at `site`, in place of the values.
///
/// # Errors
///
/// The channel's: a masked channel sends no digest.
pub fn send_field_digest<C: crate::ProverGrindingChannel, F: CanonicalBytes>(
    state: &mut C,
    site: SiteId,
    values: &[F],
) -> Result<(), AkitaError> {
    state.site(site);
    state.send_public_bytes(&field_digest::<C::Sponge, F>(values))
}

/// Receive a digest sent by [`send_field_digest`] at `site`.
///
/// # Errors
///
/// [`AkitaError::InvalidProof`] if the proof ends before the digest.
pub fn receive_field_digest<H: Sponge>(
    state: &mut VerifierTranscript<'_, H>,
    site: SiteId,
) -> Result<[u8; FIELD_DIGEST_BYTES], AkitaError> {
    state.site(site);
    state
        .receive_bytes(FIELD_DIGEST_BYTES)?
        .try_into()
        .map_err(|_| AkitaError::InvalidProof)
}
