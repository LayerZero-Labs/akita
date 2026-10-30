//! Prover loader: builds a [`PinnedProverSetup`] from an anchored release.

use super::AkitaProverSetup;
use crate::commitment::{PortableCommitmentHandle, SetupPrefixProverRegistry, SetupPrefixSlot};
use crate::CpuBackend;
use akita_config::preprocessing::{
    check_set_record_code, load_prefix_commitments, load_set_catalog, set_stream, Admission,
    CommitmentSource, PrefixCommitmentDeriver, ProgramRows,
};
use akita_config::{CommitmentConfig, TrustedScheduleCatalog};
use akita_error::AkitaError;
use akita_params::SetupPrefixSlotId;
use akita_serialization::{AkitaDeserialize, AkitaSerialize, Compress, Valid, Validate};
use akita_types::preprocessing::{
    hints_file_name, profile_key, read_release_file, AnchoredManifest, BundleId, ReleaseFileSource,
};
use akita_types::{AkitaSetupDescriptor, AkitaSetupSeed, SetupPrefixPublicCommitment};
use jolt_field::{CanonicalEncoding, Field, Unreduced, WithCommitAccumulator};

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
/// loaded from its registry payload and hints after both match the manifest;
/// otherwise the slots are exported from the derived matrix and each payload
/// is compared with the manifest.
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
    let record = release.set(set_name)?;
    check_set_record_code::<Cfg>(record)?;
    let stream = set_stream::<Cfg>(release, record)?;
    let admission = Admission::admit(
        record,
        load_set_catalog::<Cfg>(release, record, files)?,
        program,
    )?;
    let requirements = admission.prover_requirements()?;
    let ids = requirements.prefix_slot_ids();
    let mut setup = AkitaProverSetup::derive(
        admission.setup_descriptor(stream.identity.setup_seed(), requirements.matrix_capacity()),
    )?;
    let hint_names = ids
        .iter()
        .map(|id| hints_file_name(&record.stream_id, &profile_key(&id.commitment_profile)))
        .collect::<Vec<_>>();
    if !ids.is_empty()
        && hint_names
            .iter()
            .all(|name| release.manifest().file(name).is_some())
    {
        let commitments = load_prefix_commitments::<Cfg>(
            release,
            record,
            files,
            ids,
            CommitmentSource::Registry,
        )?;
        let mut registry = SetupPrefixProverRegistry::new(stream.identity.setup_seed().clone());
        for (id, name) in ids.iter().zip(&hint_names) {
            let key = profile_key(&id.commitment_profile);
            registry.insert(SetupPrefixSlot {
                id: id.clone(),
                commitment: commitments[&key].clone(),
                hint: decode_hint(&read_release_file(
                    files,
                    release.manifest(),
                    name,
                    MAX_HINT_FILE_BYTES,
                )?)?,
            })?;
        }
        // Payload and hints matched the anchored manifest (Invariant 7).
        registry.mark_backend_validated();
        setup.prefix_slots = registry;
    } else {
        setup.export_setup_prefixes(ids)?;
        load_prefix_commitments::<Cfg>(
            release,
            record,
            files,
            ids,
            CommitmentSource::SeedDerived(&ExportedPrefixes(&setup.prefix_slots)),
        )?;
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

/// Slots this process just exported, presented to the manifest comparison.
struct ExportedPrefixes<'a, F: Field>(&'a SetupPrefixProverRegistry<F>);

impl<F: Field> PrefixCommitmentDeriver<F> for ExportedPrefixes<'_, F> {
    fn derive_prefix_commitments(
        &self,
        setup_seed: &AkitaSetupSeed,
        ids: &[SetupPrefixSlotId],
    ) -> Result<Vec<SetupPrefixPublicCommitment<F>>, AkitaError> {
        if setup_seed != self.0.setup_seed() {
            return Err(AkitaError::InvalidSetup(
                "exported setup prefixes belong to another stream".into(),
            ));
        }
        ids.iter()
            .map(|id| {
                self.0
                    .get(id)
                    .map(|slot| slot.commitment.clone())
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup("an exported setup prefix is missing".into())
                    })
            })
            .collect()
    }
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
        ExportedPrefixes(&registry).derive_prefix_commitments(setup_seed, ids)
    }
}
