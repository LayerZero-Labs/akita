//! Prover loader: builds a [`PinnedProverSetup`] from an anchored release.

use super::AkitaProverSetup;
use crate::commitment::{PortableCommitmentHandle, SetupPrefixProverRegistry, SetupPrefixSlot};
use crate::CpuBackend;
use akita_config::preprocessing::{
    check_prefix_commitment, load_prefix_commitments, load_set_catalog, open_set, Admission,
    PrefixCommitmentDeriver, PrefixCommitmentSource, ProgramRows,
};
use akita_config::{CommitmentConfig, TrustedScheduleCatalog};
use akita_error::AkitaError;
use akita_params::SetupPrefixSlotId;
use akita_serialization::{AkitaDeserialize, AkitaSerialize, Compress, Valid, Validate};
use akita_types::preprocessing::{
    hints_file_name, profile_key, read_release_file, AnchoredManifest, BundleId, ProfileKey,
    ReleaseFileSource,
};
use akita_types::{AkitaSetupDescriptor, AkitaSetupSeed, SetupPrefixPublicCommitment};
use jolt_field::{CanonicalEncoding, Field, Unreduced, WithCommitAccumulator};
use std::collections::btree_map::Entry;
use std::collections::{BTreeMap, BTreeSet};

/// Largest prover hint file a loader reads.
const MAX_HINT_FILE_BYTES: usize = 1 << 32;

/// A prover setup issued by [`load_prover_setup`] from an anchored release.
///
/// Its fields are private and it has no public constructor and no
/// deserializer. Its matrix is derived from the release's seed, its catalog
/// is the catalog identity of the admitted rows, and every setup-prefix slot
/// carries the anchored commitment: loaded with its hints after both matched
/// the manifest, or exported from the matrix in this process and compared
/// with the manifest.
///
/// ```compile_fail
/// use akita_cpu_backend::PinnedProverSetup;
/// use akita_config::proof_optimized::fp128::OneHot;
/// fn forge(setup: akita_cpu_backend::AkitaProverSetup<akita_config::proof_optimized::fp128::Field>,
///          catalog: akita_config::TrustedScheduleCatalog<OneHot>,
///          bundle_id: akita_types::preprocessing::BundleId) -> PinnedProverSetup<OneHot> {
///     PinnedProverSetup { setup, catalog, bundle_id, set_name: String::new() }
/// }
/// ```
#[derive(Debug, Clone)]
pub struct PinnedProverSetup<Cfg: CommitmentConfig> {
    setup: AkitaProverSetup<Cfg::Field>,
    catalog: TrustedScheduleCatalog<Cfg>,
    bundle_id: BundleId,
    set_name: String,
}

impl<Cfg: CommitmentConfig> PinnedProverSetup<Cfg> {
    /// The prover setup.
    pub fn setup(&self) -> &AkitaProverSetup<Cfg::Field> {
        &self.setup
    }

    /// The catalog identity of the admitted rows.
    pub fn catalog(&self) -> &TrustedScheduleCatalog<Cfg> {
        &self.catalog
    }

    /// The release this setup was loaded from.
    pub fn bundle_id(&self) -> BundleId {
        self.bundle_id
    }

    /// The anchored parameter set this setup was loaded for.
    pub fn set_name(&self) -> &str {
        &self.set_name
    }
}

/// Load the prover setup of one anchored parameter set.
///
/// Rows are admitted exactly as a verifier admits them (A1); the matrix is
/// sized from the rows the prover may prove (A2) and derived from the
/// release's seed (A3). With hint files in the release, every slot is
/// loaded from its registry payload and hints after both match the manifest,
/// before the matrix is derived; otherwise the slots are exported from the
/// derived matrix and each payload is compared with the manifest.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidSetup`] when the set record does not match
/// `Cfg`, a file does not match the manifest, no row is admitted, or a
/// commitment differs from the manifest.
#[tracing::instrument(skip_all, name = "load_prover_setup", fields(set = set_name))]
pub fn load_prover_setup<Cfg>(
    release: &AnchoredManifest,
    files: &dyn ReleaseFileSource,
    set_name: &str,
    program: Option<&ProgramRows>,
) -> Result<PinnedProverSetup<Cfg>, AkitaError>
where
    Cfg: CommitmentConfig,
    Cfg::Field: CanonicalEncoding
        + Unreduced
        + WithCommitAccumulator
        + Valid
        + AkitaSerialize
        + AkitaDeserialize<Context = ()>
        + 'static,
{
    let (record, stream) = open_set::<Cfg>(release, set_name)?;
    let admission = Admission::admit(
        record,
        load_set_catalog::<Cfg>(release, record, files)?,
        program,
    )?;
    let requirements = admission.prover_requirements()?;
    let ids = requirements.prefix_slot_ids();
    let setup_seed = stream.identity.setup_seed();
    let hint_name = |id: &SetupPrefixSlotId| {
        hints_file_name(&record.stream_id, &profile_key(&id.commitment_profile))
    };
    let loaded = if ids
        .iter()
        .all(|id| release.manifest().file(&hint_name(id)).is_some())
    {
        let commitments = load_prefix_commitments(
            release,
            record,
            stream,
            files,
            ids,
            PrefixCommitmentSource::Registry,
        )?;
        let mut hints = BTreeMap::<ProfileKey, PortableCommitmentHandle<Cfg::Field>>::new();
        let mut registry = SetupPrefixProverRegistry::new(setup_seed.clone());
        for (id, commitment) in ids.iter().zip(commitments) {
            let hint = match hints.entry(profile_key(&id.commitment_profile)) {
                Entry::Occupied(entry) => entry.get().clone(),
                Entry::Vacant(entry) => entry
                    .insert(decode_hint(&read_release_file(
                        files,
                        release.manifest(),
                        &hint_name(id),
                        MAX_HINT_FILE_BYTES,
                    )?)?)
                    .clone(),
            };
            registry.insert(SetupPrefixSlot {
                id: id.clone(),
                commitment,
                hint,
            })?;
        }
        // Payload and hints matched the anchored manifest (Invariant 7).
        registry.mark_backend_validated();
        Some(registry)
    } else {
        None
    };
    let mut setup = AkitaProverSetup::derive(
        admission.setup_descriptor(setup_seed, requirements.matrix_capacity()),
    )?;
    match loaded {
        Some(registry) => setup.prefix_slots = registry,
        None => {
            setup.export_setup_prefixes(ids)?;
            let mut checked = BTreeSet::new();
            for (id, slot) in setup.prefix_slots.iter() {
                if checked.insert(profile_key(&id.commitment_profile)) {
                    check_prefix_commitment(
                        release,
                        record,
                        stream,
                        &id.commitment_profile,
                        &slot.commitment,
                    )?;
                }
            }
        }
    }
    Ok(PinnedProverSetup {
        setup,
        catalog: admission.catalog().clone(),
        bundle_id: release.bundle_id(),
        set_name: set_name.to_owned(),
    })
}

fn decode_hint<F>(bytes: &[u8]) -> Result<PortableCommitmentHandle<F>, AkitaError>
where
    F: Field + Valid + AkitaSerialize + AkitaDeserialize<Context = ()>,
{
    let malformed = || AkitaError::InvalidSetup("malformed prover hint file".into());
    let mut reader = bytes;
    let hint = PortableCommitmentHandle::<F>::deserialize_with_mode(
        &mut reader,
        Compress::No,
        Validate::Yes,
        &(),
    )
    .map_err(|_| malformed())?;
    let mut encoded = Vec::with_capacity(bytes.len());
    hint.serialize_with_mode(&mut encoded, Compress::No)
        .map_err(|_| malformed())?;
    if !reader.is_empty() || encoded != bytes {
        return Err(malformed());
    }
    Ok(hint)
}

/// Seed-derived mode for host loaders: commits each requested prefix from
/// the seed with the CPU reference path, streaming its source.
#[derive(Debug, Clone, Copy, Default)]
pub struct CpuPrefixDeriver;

impl<F> PrefixCommitmentDeriver<F> for CpuPrefixDeriver
where
    F: Field + CanonicalEncoding + Unreduced + WithCommitAccumulator + Valid + 'static,
{
    fn derive_prefix_commitments(
        &self,
        setup_seed: &AkitaSetupSeed,
        ids: &[SetupPrefixSlotId],
    ) -> Result<Vec<SetupPrefixPublicCommitment<F>>, AkitaError> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let mut num_field_elements = 1;
        for id in ids {
            num_field_elements =
                num_field_elements.max(akita_params::setup_prefix_slot_field_elements(id)?);
        }
        let registry = CpuBackend::<F, F>::export_setup_prefixes_from_seed(
            &AkitaSetupDescriptor {
                max_num_vars: 1,
                max_num_batched_polys: 1,
                num_field_elements,
                setup_seed: setup_seed.clone(),
            },
            ids,
        )?;
        ids.iter()
            .map(|id| {
                registry
                    .get(id)
                    .map(|slot| slot.commitment.clone())
                    .ok_or_else(|| {
                        AkitaError::Internal("an exported setup prefix is missing".into())
                    })
            })
            .collect()
    }
}
