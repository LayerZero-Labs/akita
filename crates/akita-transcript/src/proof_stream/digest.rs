//! Digests of field values that the verifier recomputes instead of receiving.

use akita_error::AkitaError;
use jolt_algebra::CanonicalEncoding;
use jolt_transcript::{protocol_tag, DigestHasher, TAG_BYTES};
use sha3::Digest;
use spongefish::Encoding;

use super::{
    receive_byte_group, send_byte_group, FieldAtom, ProtocolSiteId, ProverChannel, VerifierChannel,
};

/// Byte length of a [`field_digest`].
pub const FIELD_DIGEST_BYTES: usize = 32;

/// Domain tag hashed ahead of the values of every [`field_digest`].
const FIELD_DIGEST_TAG: [u8; TAG_BYTES] = protocol_tag(b"akita-pcs/field-digest/v8");

/// Collision-resistant digest of the canonical encoding of `values`.
///
/// The digest is [`jolt_transcript::DIGEST_HASH`] of the protocol tag
/// `akita-pcs/field-digest/v8` followed by each value's canonical
/// little-endian bytes, the hash that
/// commitment digests use and that the protocol tag names. It is computed
/// outside any proof channel, so the prover can bind values the verifier only
/// learns later.
#[must_use]
pub fn field_digest<F: CanonicalEncoding>(values: &[F]) -> [u8; FIELD_DIGEST_BYTES] {
    let mut hasher = DigestHasher::new().chain_update(FIELD_DIGEST_TAG);
    for &value in values {
        hasher.update(FieldAtom::new(value).encode().as_ref());
    }
    hasher.finalize().into()
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
