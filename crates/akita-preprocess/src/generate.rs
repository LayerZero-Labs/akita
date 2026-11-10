//! Release generation: one deterministic pass from a description to every
//! file and the manifest.

use crate::description::{ReleaseDescription, MANIFEST_FILE};
use crate::plan::{insert_profile, SetPlan};
use crate::ConfigRegistry;
use akita_config::preprocessing::MAX_REGISTRY_PAYLOAD_BYTES;
use akita_config::MAX_TRUSTED_SCHEDULE_ARTIFACT_BYTES;
use akita_cpu_backend::{CpuBackend, MAX_HINT_FILE_BYTES};
use akita_error::AkitaError;
use akita_params::sis::SisModulusProfileId;
use akita_params::{FlatMatrix, SetupPrefixSlotId};
use akita_serialization::{AkitaSerialize, Valid};
use akita_types::preprocessing::{
    bundle_id, catalog_file_name, catalog_view_file_name, file_hash, hints_file_name,
    part_file_name, prefix_commitment_digest, registry_file_name, setup_prefix_payload,
    terminal_cache_file_name, BundleId, FileEntry, MemoryReleaseFiles, ProfileKey, RegistryEntry,
    ReleaseManifest, StreamEntry, StreamId, StreamIdentity, DIGEST_HASH,
};
use akita_types::{
    AkitaExpandedSetup, AkitaSetupDescriptor, AkitaVerifierSetup, SetupPrefixVerifierRegistry,
    PREPARED_VERIFIER_NTT_CACHE_MAX_BYTES,
};
use jolt_field::{CanonicalEncoding, Field, Unreduced, WithCommitAccumulator};
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

/// Which files [`write_release`] writes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Role {
    /// Every file.
    All,
    /// Registry, hints, and catalogs.
    Prover,
    /// Registry and catalogs.
    Verifier,
    /// The files one guest target embeds: registry, catalog views, and the
    /// target's terminal caches.
    Target(String),
}

impl Role {
    /// Parse `all`, `prover`, `verifier`, or a target name.
    #[must_use]
    pub fn parse(role: &str) -> Self {
        match role {
            "all" => Self::All,
            "prover" => Self::Prover,
            "verifier" => Self::Verifier,
            target => Self::Target(target.to_owned()),
        }
    }

    fn writes(&self, name: &str) -> bool {
        let top = name.split('/').next().unwrap_or_default();
        match self {
            Self::All => true,
            Self::Prover => matches!(top, "registry" | "hints" | "catalog"),
            Self::Verifier => matches!(top, "registry" | "catalog"),
            Self::Target(target) => match top {
                "registry" | "catalog-view" => true,
                "terminal-cache" => name.split('/').nth(3) == Some(target.as_str()),
                _ => false,
            },
        }
    }
}

/// A generated release.
#[derive(Debug, Clone)]
pub struct GeneratedRelease {
    /// Encoded manifest.
    pub manifest: Vec<u8>,
    /// `bundle_id` of the manifest.
    pub bundle_id: BundleId,
    /// Every file, at its release name.
    pub files: MemoryReleaseFiles,
    /// The slot id each published commitment was computed for, one per
    /// stream and profile.
    pub profiles: Vec<SetupPrefixSlotId>,
    /// Largest stored part of one file when written, from the description.
    /// Transport only: it is not part of the release identity.
    pub max_file_part: u64,
}

/// Generate a release from scratch.
///
/// The output depends only on the description, the code, and the digest-hash
/// selection: no cache, previous bundle, or candidate is read. `configs`
/// names the compiled configuration of each set. Catalog paths in the
/// description are relative to `repo_root`.
///
/// # Errors
///
/// Returns an error for an invalid description, an unknown config, or a
/// failed commitment.
pub fn generate(
    configs: &dyn ConfigRegistry,
    repo_root: &Path,
    description: &ReleaseDescription,
) -> Result<GeneratedRelease, AkitaError> {
    description.validate()?;
    let setup_seed = description.setup_seed()?;
    let plans = description
        .sets
        .iter()
        .map(|set| configs.plan_set(repo_root, set, &setup_seed))
        .collect::<Result<Vec<_>, _>>()?;
    let mut by_stream = BTreeMap::<StreamId, Vec<&SetPlan>>::new();
    for plan in &plans {
        by_stream
            .entry(plan.record.stream_id)
            .or_default()
            .push(plan);
    }
    let mut files = BTreeMap::<String, Vec<u8>>::new();
    let mut registry = Vec::new();
    let mut streams = Vec::new();
    let mut published = Vec::new();
    for stream_plans in by_stream.into_values() {
        let mut profiles = BTreeMap::new();
        for plan in &stream_plans {
            for id in plan.profiles.values() {
                insert_profile(&mut profiles, id.clone());
            }
        }
        let first = stream_plans[0];
        let element_count = crate::dispatch_field!(first.field, |F| generate_stream::<F>(
            description,
            &first.stream,
            first.field,
            &stream_plans,
            &profiles,
            &mut files,
            &mut registry,
        ))?;
        streams.push(StreamEntry {
            identity: first.stream.clone(),
            element_count,
        });
        published.extend(profiles.into_values());
    }
    for plan in &plans {
        files.insert(
            catalog_file_name(&plan.record.name),
            plan.catalog_bytes.clone(),
        );
        if !description.targets.is_empty() {
            for row in &plan.rows {
                files.insert(
                    catalog_view_file_name(&plan.record.name, &row.digest),
                    row.view.clone(),
                );
            }
        }
    }
    check_loader_caps(&files)?;
    let mut targets = description.targets.clone();
    targets.sort();
    let mut sets = plans
        .iter()
        .map(|plan| plan.record.clone())
        .collect::<Vec<_>>();
    sets.sort_by(|left, right| left.name.cmp(&right.name));
    registry.sort_by_key(|entry: &RegistryEntry| (entry.stream_id, entry.profile_key));
    let manifest = ReleaseManifest {
        digest_hash: DIGEST_HASH,
        release_name: description.name.clone(),
        kind: description.kind()?,
        generator_version: description.generator_version.clone(),
        targets,
        streams,
        registry,
        sets,
        files: files
            .iter()
            .map(|(name, bytes)| FileEntry {
                name: name.clone(),
                length: bytes.len() as u64,
                file_hash: file_hash(name, bytes),
            })
            .collect(),
    }
    .to_bytes()?;
    let mut out = MemoryReleaseFiles::new();
    for (name, bytes) in files {
        out.insert(name, bytes);
    }
    Ok(GeneratedRelease {
        bundle_id: bundle_id(&manifest),
        manifest,
        files: out,
        profiles: published,
        max_file_part: description.max_file_part,
    })
}

/// Require every file to fit the cap of the loader that reads it, so the
/// generator never publishes a release no loader can read.
fn check_loader_caps(files: &BTreeMap<String, Vec<u8>>) -> Result<(), AkitaError> {
    for (name, bytes) in files {
        let cap = match name.split('/').next().unwrap_or_default() {
            "registry" => MAX_REGISTRY_PAYLOAD_BYTES,
            "hints" => MAX_HINT_FILE_BYTES,
            "catalog" | "catalog-view" => MAX_TRUSTED_SCHEDULE_ARTIFACT_BYTES,
            "terminal-cache" => PREPARED_VERIFIER_NTT_CACHE_MAX_BYTES,
            _ => {
                return Err(AkitaError::Internal(format!(
                    "release file {name:?} has no loader"
                )))
            }
        };
        if bytes.len() > cap {
            return Err(AkitaError::InvalidSetup(format!(
                "release file {name:?} of {} bytes exceeds its loader's {cap}-byte cap",
                bytes.len()
            )));
        }
    }
    Ok(())
}

/// Derive one stream once, commit every profile in `profiles`, and write
/// its registry, hints, and terminal caches. Returns the stream's element
/// count.
fn generate_stream<F>(
    description: &ReleaseDescription,
    stream: &StreamIdentity,
    field: SisModulusProfileId,
    plans: &[&SetPlan],
    profiles: &BTreeMap<ProfileKey, SetupPrefixSlotId>,
    files: &mut BTreeMap<String, Vec<u8>>,
    registry: &mut Vec<RegistryEntry>,
) -> Result<u64, AkitaError>
where
    F: Field
        + CanonicalEncoding
        + Unreduced
        + WithCommitAccumulator
        + Valid
        + AkitaSerialize
        + 'static,
{
    let stream_id = stream.stream_id();
    let mut element_count = 1usize;
    for plan in plans {
        element_count = element_count.max(plan.prover_capacity);
        for row in &plan.rows {
            element_count = element_count.max(row.verifier_capacity);
        }
    }
    for id in profiles.values() {
        element_count = element_count.max(akita_params::setup_prefix_slot_field_elements(id)?);
    }
    let page = stream.page_field_elements();
    // The final page is derived in full even if it extends past the count.
    let expanded = Arc::new(AkitaExpandedSetup::<F>::derive(AkitaSetupDescriptor {
        max_num_vars: 1,
        max_num_batched_polys: 1,
        num_field_elements: element_count.div_ceil(page) * page,
        setup_seed: stream.setup_seed().clone(),
    })?);

    let ids = profiles.values().cloned().collect::<Vec<_>>();
    if !ids.is_empty() {
        let mut keys = profiles
            .iter()
            .map(|(key, id)| (id.clone(), *key))
            .collect::<BTreeMap<_, _>>();
        let exported =
            CpuBackend::<F, F>::new(Arc::clone(&expanded))?.export_setup_prefixes(&ids)?;
        // Consume each slot as it is written, so its hint is not held twice.
        for slot in exported.into_slots() {
            let key = keys.remove(&slot.id).ok_or_else(|| {
                AkitaError::Internal("setup-prefix export returned an unrequested slot".into())
            })?;
            registry.push(RegistryEntry {
                stream_id,
                profile_key: key,
                prefix_commitment_digest: prefix_commitment_digest(
                    stream,
                    &slot.id.commitment_profile,
                    &slot.commitment,
                )?,
            });
            files.insert(
                registry_file_name(&stream_id, &key),
                setup_prefix_payload(&slot.commitment)?,
            );
            if description.hints {
                files.insert(hints_file_name(&stream_id, &key), slot.hint_bytes()?);
            }
        }
        if !keys.is_empty() {
            return Err(AkitaError::Internal(
                "exported setup prefixes miss a requested slot".into(),
            ));
        }
    }

    // The RISC-V guest's scalar terminal caches exist for Q128 only. A row
    // shared between sets is built once per target, and rows of one verifier
    // capacity share one setup, built in ascending capacity order and
    // dropped before the next.
    if field == SisModulusProfileId::Q128Offset6A23801F && !description.targets.is_empty() {
        let mut rows = BTreeMap::new();
        for plan in plans {
            for row in &plan.rows {
                rows.entry((row.verifier_capacity, row.digest))
                    .or_insert(row);
            }
        }
        let matrix = expanded.shared_matrix().as_field_slice();
        let mut setup = None;
        for ((capacity, digest), row) in rows {
            if setup.as_ref().is_none_or(|(built, _)| *built != capacity) {
                // Free the previous prefix before copying the next.
                drop(setup.take());
                setup = Some((
                    capacity,
                    terminal_cache_setup::<F>(stream, matrix, capacity)?,
                ));
            }
            let Some((_, verifier_setup)) = setup.as_ref() else {
                return Err(AkitaError::Internal(
                    "terminal cache setup is missing".into(),
                ));
            };
            for target in &description.targets {
                files.insert(
                    terminal_cache_file_name(&stream_id, &digest, target, capacity),
                    akita_verifier::build_riscv64_terminal_ntt_cache(
                        verifier_setup,
                        &row.schedule,
                        digest,
                    )?,
                );
            }
        }
    }
    Ok(element_count as u64)
}

/// The verifier setup a terminal cache of verifier capacity `capacity` is
/// built from: that prefix of the derived stream, with no commitments.
fn terminal_cache_setup<F>(
    stream: &StreamIdentity,
    matrix: &[F],
    capacity: usize,
) -> Result<AkitaVerifierSetup<F>, AkitaError>
where
    F: Field + CanonicalEncoding + Valid + AkitaSerialize,
{
    let prefix = matrix
        .get(..capacity)
        .ok_or_else(|| AkitaError::Internal("terminal cache exceeds the derived stream".into()))?;
    // SAFETY: a prefix of the matrix this generator derived from the seed;
    // the cache reads no setup-prefix commitment.
    unsafe {
        AkitaVerifierSetup::from_parts(
            Arc::new(
                AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
                    AkitaSetupDescriptor {
                        max_num_vars: 1,
                        max_num_batched_polys: 1,
                        num_field_elements: prefix.len(),
                        setup_seed: stream.setup_seed().clone(),
                    },
                    FlatMatrix::from_flat_data(prefix.to_vec()),
                ),
            ),
            SetupPrefixVerifierRegistry::new(stream.setup_seed().clone()),
        )
    }
}

/// Write the files of `role` and the manifest, splitting a file larger than
/// the release's `max_file_part` into parts. Any stored copy or part of a
/// written file left by an earlier run is removed first, so stale parts never
/// make a correct release unreadable.
///
/// # Errors
///
/// Returns an error when `role` names a target the release does not have or
/// a file cannot be written.
pub fn write_release(
    dir: &Path,
    release: &GeneratedRelease,
    role: &Role,
) -> Result<(), AkitaError> {
    let manifest = ReleaseManifest::decode(&release.manifest)?;
    if let Role::Target(target) = role {
        if !manifest.targets.contains(target) {
            return Err(AkitaError::InvalidInput(format!(
                "role {target:?} is not a target of the release"
            )));
        }
    }
    let io = |path: &Path, error: std::io::Error| {
        AkitaError::InvalidInput(format!("cannot write {}: {error}", path.display()))
    };
    std::fs::create_dir_all(dir).map_err(|error| io(dir, error))?;
    let manifest_path = dir.join(MANIFEST_FILE);
    std::fs::write(&manifest_path, &release.manifest).map_err(|error| io(&manifest_path, error))?;
    let part = usize::try_from(release.max_file_part)
        .ok()
        .filter(|part| *part > 0)
        .ok_or_else(|| AkitaError::InvalidInput("max_file_part is zero".into()))?;
    for (name, bytes) in release.files.iter().filter(|(name, _)| role.writes(name)) {
        let path = dir.join(name);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|error| io(parent, error))?;
        }
        remove_stored(dir, name).map_err(|error| io(&path, error))?;
        if bytes.len() <= part {
            std::fs::write(&path, bytes).map_err(|error| io(&path, error))?;
        } else {
            for (index, chunk) in bytes.chunks(part).enumerate() {
                let part_path = dir.join(part_file_name(name, index));
                std::fs::write(&part_path, chunk).map_err(|error| io(&part_path, error))?;
            }
        }
    }
    Ok(())
}

/// Remove `name` and its consecutive parts under `dir`.
fn remove_stored(dir: &Path, name: &str) -> std::io::Result<()> {
    let remove = |path: &Path| match std::fs::remove_file(path) {
        Ok(()) => Ok(true),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error),
    };
    remove(&dir.join(name))?;
    let mut index = 0;
    while remove(&dir.join(part_file_name(name, index)))? {
        index += 1;
    }
    Ok(())
}
