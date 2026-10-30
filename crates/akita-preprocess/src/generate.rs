//! Release generation: one deterministic pass from a description to every
//! file and the manifest.

use crate::description::{ReleaseDescription, MANIFEST_FILE};
use crate::plan::{plan_set, RowPlan, SetPlan};
use akita_cpu_backend::CpuBackend;
use akita_error::AkitaError;
use akita_params::sis::SisModulusProfileId;
use akita_params::{FlatMatrix, SetupPrefixSlotId};
use akita_serialization::{AkitaSerialize, Compress, Valid};
use akita_types::preprocessing::{
    bundle_id, catalog_file_name, catalog_view_file_name, file_hash, hints_file_name,
    matrix_file_name, page_hash, page_hashes_file_name, prefix_commitment_digest,
    profile_descriptor, registry_file_name, setup_prefix_payload, terminal_cache_file_name,
    AlgorithmIdentity, BundleId, FileEntry, MemoryReleaseFiles, ProfileKey, RegistryEntry,
    ReleaseManifest, StreamEntry, StreamId, StreamIdentity, DIGEST_HASH,
};
use akita_types::{
    AkitaExpandedSetup, AkitaSetupDescriptor, AkitaVerifierSetup, SetupPrefixVerifierRegistry,
};
use jolt_field::{CanonicalEncoding, Field, Unreduced, WithCommitAccumulator};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::Arc;

/// The guest target whose terminal caches this generator builds.
pub const RISCV64_TARGET: &str = "riscv64";

/// Which files a run writes. Every role computes the same manifest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Role {
    /// Every file.
    All,
    /// Registry, hints, and catalogs.
    Prover,
    /// Registry and catalogs.
    Verifier,
    /// The files one guest target embeds.
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
                "registry" | "catalog" | "catalog-view" | "matrix" | "page-hashes" => true,
                "terminal-cache" => name.split('/').nth(3) == Some(target.as_str()),
                _ => false,
            },
        }
    }
}

/// A generated release: its manifest and the files of the requested role.
#[derive(Debug, Clone)]
pub struct GeneratedRelease {
    /// Encoded manifest.
    pub manifest: Vec<u8>,
    /// `bundle_id` of the manifest.
    pub bundle_id: BundleId,
    /// Files of the requested role, at their release names.
    pub files: MemoryReleaseFiles,
}

/// Generate a release from scratch.
///
/// The output depends only on the description, the code, and the digest-hash
/// selection: no cache, previous bundle, or candidate is read. Catalog paths
/// in the description are relative to `repo_root`.
///
/// # Errors
///
/// Returns an error for an invalid description, an unknown config or target,
/// or a failed commitment.
pub fn generate(
    repo_root: &Path,
    description: &ReleaseDescription,
    role: &Role,
) -> Result<GeneratedRelease, AkitaError> {
    if let Role::Target(target) = role {
        if !description.targets.contains(target) {
            return Err(AkitaError::InvalidInput(format!(
                "role {target:?} is not a target of the release"
            )));
        }
    }
    let setup_seed = description.setup_seed()?;
    let plans = description
        .sets
        .iter()
        .map(|set| {
            crate::dispatch_config!(set.config.as_str(), |Cfg| plan_set::<Cfg>(
                repo_root,
                set,
                &setup_seed
            ))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut files = BTreeMap::<String, Vec<u8>>::new();
    let mut registry = Vec::new();
    let mut streams = BTreeMap::<StreamId, StreamEntry>::new();
    for (stream_id, stream_plans) in group_by_stream(&plans)? {
        let first = stream_plans[0];
        let element_count = crate::dispatch_field!(first.field, |F| generate_stream::<F>(
            description,
            &first.stream,
            first.field,
            &stream_plans,
            &mut files,
            &mut registry,
        ))?;
        streams.insert(
            stream_id,
            StreamEntry {
                identity: first.stream.clone(),
                element_count,
            },
        );
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
        max_file_part: description.max_file_part,
        streams: streams.into_values().collect(),
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
        if role.writes(&name) {
            out.insert(name, bytes);
        }
    }
    Ok(GeneratedRelease {
        bundle_id: bundle_id(&manifest),
        manifest,
        files: out,
    })
}

fn group_by_stream(plans: &[SetPlan]) -> Result<BTreeMap<StreamId, Vec<&SetPlan>>, AkitaError> {
    let mut names = BTreeSet::new();
    let mut grouped = BTreeMap::<StreamId, Vec<&SetPlan>>::new();
    for plan in plans {
        if !names.insert(plan.record.name.as_str()) {
            return Err(AkitaError::InvalidInput(format!(
                "duplicate set name {:?}",
                plan.record.name
            )));
        }
        grouped.entry(plan.record.stream_id).or_default().push(plan);
    }
    Ok(grouped)
}

/// Derive one stream once, commit every profile its sets need, and write
/// its registry, hints, page hashes, matrix ranges, and terminal caches.
/// Returns the stream's element count.
fn generate_stream<F>(
    description: &ReleaseDescription,
    stream: &StreamIdentity,
    field: SisModulusProfileId,
    plans: &[&SetPlan],
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
    let mut profiles = BTreeMap::<ProfileKey, SetupPrefixSlotId>::new();
    let mut element_count = 1usize;
    for plan in plans {
        element_count = element_count.max(plan.prover_capacity);
        for row in &plan.rows {
            element_count = element_count.max(row.verifier_capacity);
        }
        for (key, id) in &plan.profiles {
            element_count = element_count.max(akita_params::setup_prefix_slot_field_elements(id)?);
            profiles
                .entry(*key)
                .and_modify(|kept| {
                    if id < kept {
                        *kept = id.clone();
                    }
                })
                .or_insert_with(|| id.clone());
        }
    }
    let page = stream.page_field_elements();
    let num_pages = element_count.div_ceil(page);
    // The final page is derived in full even if it extends past the count.
    let expanded = Arc::new(AkitaExpandedSetup::<F>::derive(AkitaSetupDescriptor {
        max_num_vars: 1,
        max_num_batched_polys: 1,
        num_field_elements: num_pages * page,
        setup_seed: stream.setup_seed().clone(),
    })?);

    let ids = profiles.values().cloned().collect::<Vec<_>>();
    if !ids.is_empty() {
        let exported =
            CpuBackend::<F, F>::new(Arc::clone(&expanded))?.export_setup_prefixes(&ids)?;
        let algorithm = AlgorithmIdentity::compiled(field);
        for (key, id) in &profiles {
            let slot = exported.get(id).ok_or_else(|| {
                AkitaError::Internal("exported setup prefixes miss a requested slot".into())
            })?;
            let payload = setup_prefix_payload(&slot.commitment)?;
            registry.push(RegistryEntry {
                stream_id,
                profile_key: *key,
                prefix_commitment_digest: prefix_commitment_digest(
                    stream,
                    &algorithm,
                    &profile_descriptor(&id.commitment_profile),
                    &payload,
                ),
            });
            files.insert(registry_file_name(&stream_id, key), payload);
            if description.hints {
                files.insert(hints_file_name(&stream_id, key), slot.hint_bytes()?);
            }
        }
    }

    let matrix = expanded.shared_matrix().as_field_slice();
    let page_bytes = |index: usize| -> Result<Vec<u8>, AkitaError> {
        let mut out = Vec::with_capacity(page * F::NUM_BYTES);
        for value in &matrix[index * page..(index + 1) * page] {
            value
                .serialize_with_mode(&mut out, Compress::No)
                .map_err(|error| AkitaError::Internal(format!("page encoding: {error}")))?;
        }
        Ok(out)
    };
    if description.page_hashes {
        let mut hashes = Vec::with_capacity(num_pages * 32);
        for index in 0..num_pages {
            hashes.extend_from_slice(&page_hash(&stream_id, index as u64, &page_bytes(index)?));
        }
        files.insert(page_hashes_file_name(&stream_id), hashes);
    }
    for &[first, last] in &description.matrix_pages {
        let (first_index, last_index) = (first as usize, last as usize);
        if first > last || last_index >= num_pages {
            return Err(AkitaError::InvalidInput(format!(
                "matrix page range {first}-{last} is outside the stream's {num_pages} pages"
            )));
        }
        let mut content = Vec::new();
        for index in first_index..=last_index {
            content.extend_from_slice(&page_bytes(index)?);
        }
        files.insert(matrix_file_name(&stream_id, first, last), content);
    }

    for target in &description.targets {
        if target != RISCV64_TARGET {
            return Err(AkitaError::InvalidInput(format!(
                "no terminal caches are defined for target {target:?}"
            )));
        }
        // The RISC-V guest's scalar terminal caches exist for Q128 only.
        if field != SisModulusProfileId::Q128OffsetA7F7 {
            continue;
        }
        for plan in plans {
            for row in &plan.rows {
                files.insert(
                    terminal_cache_file_name(
                        &stream_id,
                        &row.digest,
                        target,
                        row.verifier_capacity,
                    ),
                    riscv64_terminal_cache::<F>(stream, matrix, row)?,
                );
            }
        }
    }
    Ok(element_count as u64)
}

fn riscv64_terminal_cache<F>(
    stream: &StreamIdentity,
    matrix: &[F],
    row: &RowPlan,
) -> Result<Vec<u8>, AkitaError>
where
    F: Field + CanonicalEncoding + Valid,
{
    let prefix = matrix
        .get(..row.verifier_capacity)
        .ok_or_else(|| AkitaError::Internal("terminal cache exceeds the derived stream".into()))?;
    // SAFETY: a prefix of the matrix this generator derived from the seed;
    // the cache reads no setup-prefix commitment.
    let setup = unsafe {
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
        )?
    };
    akita_verifier::build_riscv64_terminal_ntt_cache(&setup, &row.schedule, row.digest)
}

/// Write a generated release: `manifest` and every file, split into parts of
/// at most `max_file_part` bytes when larger.
///
/// # Errors
///
/// Returns an error when a file cannot be written.
pub fn write_release(
    dir: &Path,
    release: &GeneratedRelease,
    max_file_part: u64,
) -> Result<(), AkitaError> {
    let io = |path: &Path, error: std::io::Error| {
        AkitaError::InvalidInput(format!("cannot write {}: {error}", path.display()))
    };
    std::fs::create_dir_all(dir).map_err(|error| io(dir, error))?;
    let manifest_path = dir.join(MANIFEST_FILE);
    std::fs::write(&manifest_path, &release.manifest).map_err(|error| io(&manifest_path, error))?;
    let part = usize::try_from(max_file_part).unwrap_or(usize::MAX).max(1);
    for (name, bytes) in release.files.iter() {
        let path = dir.join(name);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|error| io(parent, error))?;
        }
        if bytes.len() <= part {
            std::fs::write(&path, bytes).map_err(|error| io(&path, error))?;
        } else {
            for (index, chunk) in bytes.chunks(part).enumerate() {
                let part_path = path.with_file_name(format!(
                    "{}.part-{index}",
                    path.file_name()
                        .and_then(|name| name.to_str())
                        .unwrap_or_default()
                ));
                std::fs::write(&part_path, chunk).map_err(|error| io(&part_path, error))?;
            }
        }
    }
    Ok(())
}
