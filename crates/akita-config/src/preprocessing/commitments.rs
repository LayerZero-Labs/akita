//! Setup-prefix commitments a loader uses: read from registry files or
//! derived from the seed, and in both cases checked against the manifest.

use akita_error::AkitaError;
use akita_params::{GroupCommitPhaseParams, SetupPrefixSlotId};
use akita_serialization::{AkitaDeserialize, AkitaSerialize, Compress, Valid, Validate};
use akita_types::preprocessing::{
    prefix_commitment_digest, profile_key, read_release_file, registry_file_name,
    setup_prefix_payload, AnchoredManifest, ProfileKey, ReleaseFileSource, SetRecord, StreamEntry,
};
use akita_types::{AkitaSetupSeed, SetupPrefixPublicCommitment};
use jolt_field::Field;
use std::collections::BTreeMap;

/// Largest registry payload a loader reads.
const MAX_REGISTRY_PAYLOAD_BYTES: usize = 1 << 24;

/// Computes setup-prefix commitments from the seed for a host loader's
/// seed-derived mode.
///
/// An implementation needs no trust: the loader compares every derived
/// payload's prefix commitment digest with the anchored manifest.
pub trait PrefixCommitmentDeriver<F: Field> {
    /// Derive the public commitment of every slot in `ids` over the stream
    /// of `setup_seed`, in the order of `ids`.
    ///
    /// # Errors
    ///
    /// Returns an error when a slot cannot be committed.
    fn derive_prefix_commitments(
        &self,
        setup_seed: &AkitaSetupSeed,
        ids: &[SetupPrefixSlotId],
    ) -> Result<Vec<SetupPrefixPublicCommitment<F>>, AkitaError>;
}

/// Where a loader takes the setup-prefix commitments its rows use from.
#[derive(Clone, Copy)]
pub enum PrefixCommitmentSource<'a, F: Field> {
    /// Read `registry/…` files. Every loader, including a guest, may use it.
    Registry,
    /// Derive each commitment from the seed (hosts only). The loader reads
    /// no registry or hint files and costs one derivation per profile.
    SeedDerived(&'a dyn PrefixCommitmentDeriver<F>),
}

/// Load the commitment of every slot in `ids`, in the order of `ids`, from a
/// set's stream.
///
/// Each profile's commitment is read from its registry file or derived from
/// the seed once, and checked with [`check_prefix_commitment`]. Slot ids that
/// differ only in `natural_len` share it.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidSetup`] when the release has no commitment
/// for a profile, a payload is malformed, or a digest differs.
pub fn load_prefix_commitments<F>(
    release: &AnchoredManifest,
    record: &SetRecord,
    stream: &StreamEntry,
    files: &dyn ReleaseFileSource,
    ids: &[SetupPrefixSlotId],
    source: PrefixCommitmentSource<'_, F>,
) -> Result<Vec<SetupPrefixPublicCommitment<F>>, AkitaError>
where
    F: Field + Valid + AkitaSerialize + AkitaDeserialize<Context = ()>,
{
    let mut by_profile = BTreeMap::<ProfileKey, &SetupPrefixSlotId>::new();
    for id in ids {
        by_profile
            .entry(profile_key(&id.commitment_profile))
            .or_insert(id);
    }
    let representatives = by_profile
        .values()
        .map(|id| (*id).clone())
        .collect::<Vec<_>>();
    let commitments = match source {
        PrefixCommitmentSource::Registry => by_profile
            .keys()
            .map(|key| {
                let payload = read_release_file(
                    files,
                    release.manifest(),
                    &registry_file_name(&record.stream_id, key),
                    MAX_REGISTRY_PAYLOAD_BYTES,
                )?;
                decode_payload(&payload, key)
            })
            .collect::<Result<Vec<_>, AkitaError>>()?,
        PrefixCommitmentSource::SeedDerived(deriver) => {
            deriver.derive_prefix_commitments(stream.identity.setup_seed(), &representatives)?
        }
    };
    if commitments.len() != representatives.len() {
        return Err(AkitaError::InvalidSetup(
            "seed-derived commitments do not cover the requested slots".into(),
        ));
    }
    let mut by_key = BTreeMap::new();
    for ((key, id), commitment) in by_profile.into_iter().zip(commitments) {
        check_prefix_commitment(release, record, stream, &id.commitment_profile, &commitment)?;
        by_key.insert(key, commitment);
    }
    ids.iter()
        .map(|id| {
            by_key
                .get(&profile_key(&id.commitment_profile))
                .cloned()
                .ok_or_else(|| AkitaError::Internal("a requested profile was not loaded".into()))
        })
        .collect()
}

/// Require `commitment` to have the prefix commitment digest the manifest
/// names for `(stream, profile)`.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidSetup`] when the release has no commitment
/// for the profile or the digest differs.
pub fn check_prefix_commitment<F: Field + AkitaSerialize>(
    release: &AnchoredManifest,
    record: &SetRecord,
    stream: &StreamEntry,
    profile: &GroupCommitPhaseParams,
    commitment: &SetupPrefixPublicCommitment<F>,
) -> Result<(), AkitaError> {
    let key = profile_key(profile);
    let entry = release
        .manifest()
        .registry_entry(&record.stream_id, &key)
        .ok_or_else(|| {
            AkitaError::InvalidSetup(format!(
                "release publishes no commitment for profile {key} of set {:?}",
                record.name
            ))
        })?;
    if prefix_commitment_digest(&stream.identity, profile, commitment)?
        != entry.prefix_commitment_digest
    {
        return Err(AkitaError::InvalidSetup(format!(
            "commitment for profile {key} does not have its manifest prefix commitment digest"
        )));
    }
    Ok(())
}

/// Decode one registry payload, requiring its canonical encoding.
fn decode_payload<F>(
    payload: &[u8],
    key: &ProfileKey,
) -> Result<SetupPrefixPublicCommitment<F>, AkitaError>
where
    F: Field + Valid + AkitaSerialize + AkitaDeserialize<Context = ()>,
{
    let malformed = || AkitaError::InvalidSetup(format!("malformed registry payload {key}"));
    let mut reader = payload;
    let commitment = SetupPrefixPublicCommitment::<F>::deserialize_with_mode(
        &mut reader,
        Compress::No,
        Validate::Yes,
        &(),
    )
    .map_err(|_| malformed())?;
    if !reader.is_empty() || setup_prefix_payload(&commitment)? != payload {
        return Err(malformed());
    }
    Ok(commitment)
}
