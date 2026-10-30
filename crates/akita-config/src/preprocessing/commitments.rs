//! Setup-prefix commitments a loader uses: read from registry files or
//! derived from the seed, and in both cases checked against the manifest.

use super::set_stream;
use crate::CommitmentConfig;
use akita_error::AkitaError;
use akita_params::SetupPrefixSlotId;
use akita_serialization::{AkitaDeserialize, AkitaSerialize, Compress, Valid, Validate};
use akita_types::preprocessing::{
    prefix_commitment_digest, profile_descriptor, profile_key, read_release_file,
    registry_file_name, setup_prefix_payload, AnchoredManifest, ProfileKey, ReleaseFileSource,
    SetRecord,
};
use akita_types::{AkitaSetupSeed, SetupPrefixPublicCommitment};
use std::collections::BTreeMap;

/// Largest registry payload a loader reads.
const MAX_REGISTRY_PAYLOAD_BYTES: usize = 1 << 24;

/// Computes setup-prefix commitments from the seed for a host loader's
/// seed-derived mode.
///
/// An implementation needs no trust: the loader compares every derived
/// payload's prefix commitment digest with the anchored manifest.
pub trait PrefixCommitmentDeriver<F: jolt_field::Field> {
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
pub enum CommitmentSource<'a, F: jolt_field::Field> {
    /// Read `registry/…` files. Every loader, including a guest, may use it.
    Registry,
    /// Derive each commitment from the seed (hosts only). The loader reads
    /// no registry or hint files and costs one derivation per profile.
    SeedDerived(&'a dyn PrefixCommitmentDeriver<F>),
}

impl<F: jolt_field::Field> std::fmt::Debug for CommitmentSource<'_, F> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::Registry => "Registry",
            Self::SeedDerived(_) => "SeedDerived",
        })
    }
}

/// Load the commitment of every profile `ids` use, for a set's stream.
///
/// Each payload, read from its registry file or derived from the seed, is
/// required to have the prefix commitment digest the manifest names for its
/// `(stream, profile)` under the set's algorithm. Slot ids that differ only
/// in `natural_len` share one commitment.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidSetup`] when the release has no commitment
/// for a profile, a payload is malformed, or a digest differs.
pub fn load_prefix_commitments<Cfg>(
    release: &AnchoredManifest,
    record: &SetRecord,
    files: &dyn ReleaseFileSource,
    ids: &[SetupPrefixSlotId],
    source: CommitmentSource<'_, Cfg::Field>,
) -> Result<BTreeMap<ProfileKey, SetupPrefixPublicCommitment<Cfg::Field>>, AkitaError>
where
    Cfg: CommitmentConfig,
    Cfg::Field: Valid + AkitaSerialize + AkitaDeserialize<Context = ()>,
{
    let stream = set_stream::<Cfg>(release, record)?;
    let mut by_profile = BTreeMap::<ProfileKey, &SetupPrefixSlotId>::new();
    for id in ids {
        by_profile
            .entry(profile_key(&id.commitment_profile))
            .or_insert(id);
    }
    let payloads: Vec<(ProfileKey, Vec<u8>, SetupPrefixPublicCommitment<Cfg::Field>)> = match source
    {
        CommitmentSource::Registry => by_profile
            .keys()
            .map(|key| {
                let payload = read_release_file(
                    files,
                    release.manifest(),
                    &registry_file_name(&record.stream_id, key),
                    MAX_REGISTRY_PAYLOAD_BYTES,
                )?;
                let commitment = decode_payload::<Cfg::Field>(&payload, key)?;
                Ok((*key, payload, commitment))
            })
            .collect::<Result<_, AkitaError>>()?,
        CommitmentSource::SeedDerived(deriver) => {
            let representatives = by_profile
                .values()
                .map(|id| (*id).clone())
                .collect::<Vec<_>>();
            let derived = deriver
                .derive_prefix_commitments(stream.identity.setup_seed(), &representatives)?;
            if derived.len() != representatives.len() {
                return Err(AkitaError::InvalidSetup(
                    "seed-derived commitments do not cover the requested slots".into(),
                ));
            }
            by_profile
                .keys()
                .zip(derived)
                .map(|(key, commitment)| Ok((*key, setup_prefix_payload(&commitment)?, commitment)))
                .collect::<Result<_, AkitaError>>()?
        }
    };
    let mut out = BTreeMap::new();
    for (key, payload, commitment) in payloads {
        let id = by_profile[&key];
        let entry = release
            .manifest()
            .registry_entry(&record.stream_id, &key)
            .ok_or_else(|| {
                AkitaError::InvalidSetup(format!(
                    "release publishes no commitment for profile {key} of set {:?}",
                    record.name
                ))
            })?;
        let digest = prefix_commitment_digest(
            &stream.identity,
            &record.algorithm_identity,
            &profile_descriptor(&id.commitment_profile),
            &payload,
        );
        if digest != entry.prefix_commitment_digest {
            return Err(AkitaError::InvalidSetup(format!(
                "commitment for profile {key} does not have its manifest prefix commitment digest"
            )));
        }
        out.insert(key, commitment);
    }
    Ok(out)
}

/// Decode one registry payload, requiring its canonical encoding.
fn decode_payload<F>(
    payload: &[u8],
    key: &ProfileKey,
) -> Result<SetupPrefixPublicCommitment<F>, AkitaError>
where
    F: jolt_field::Field + Valid + AkitaSerialize + AkitaDeserialize<Context = ()>,
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
