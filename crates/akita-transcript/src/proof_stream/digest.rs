//! Digests of field values that the verifier recomputes instead of receiving.

use akita_error::AkitaError;
use jolt_field::CanonicalEncoding;
use spongefish::{DuplexSpongeInterface, Encoding};

use super::{
    receive_byte_group, send_byte_group, FieldAtom, ProtocolSiteId, ProverChannel, VerifierChannel,
};
use crate::TranscriptSponge;

/// Byte length of a [`field_digest`].
pub const FIELD_DIGEST_BYTES: usize = 32;

/// Domain tag absorbed ahead of the values of every [`field_digest`].
const FIELD_DIGEST_DOMAIN: [u8; 32] = *b"akita-pcs/field-digest/v1\0\0\0\0\0\0\0";

/// Collision-resistant digest of the canonical encoding of `values`.
///
/// The digest is computed by a fresh transcript sponge, independent of any
/// proof channel, so the prover can bind values the verifier only learns
/// later.
#[must_use]
pub fn field_digest<F: CanonicalEncoding>(values: &[F]) -> [u8; FIELD_DIGEST_BYTES] {
    let mut sponge = TranscriptSponge::default();
    sponge.absorb(&FIELD_DIGEST_DOMAIN);
    for &value in values {
        sponge.absorb(FieldAtom::new(value).encode().as_ref());
    }
    let mut digest = [0u8; FIELD_DIGEST_BYTES];
    sponge.squeeze(&mut digest);
    digest
}

/// Emit the [`field_digest`] of `values` in place of the values themselves.
pub fn send_field_digest<F: CanonicalEncoding>(
    state: &mut ProverChannel,
    site: ProtocolSiteId,
    values: &[F],
) -> Result<(), AkitaError> {
    send_byte_group(state, site, &field_digest(values))
}

/// Receive a digest emitted by [`send_field_digest`].
pub fn receive_field_digest(
    state: &mut VerifierChannel<'_>,
    site: ProtocolSiteId,
) -> Result<[u8; FIELD_DIGEST_BYTES], AkitaError> {
    receive_byte_group(state, site, FIELD_DIGEST_BYTES)?
        .try_into()
        .map_err(|_| AkitaError::InvalidProof)
}
