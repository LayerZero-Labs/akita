//! Generator and auditor of pinned Akita preprocessing releases.
//!
//! A release is generated deterministically from a description checked in
//! under `params/<release>/`: the generator derives each stream's public
//! matrix once, commits every setup-prefix profile any set needs with the CPU
//! reference path, and writes registry payloads, optional prover hints,
//! catalogs, optional guest files, and the manifest that names every file by
//! hash. Its output depends only on the description, the code, and the
//! digest-hash selection.
//!
//! This crate is the only component that publishes setup-prefix commitments.
//! It never reads `akita-setup`'s cache, a previous bundle, or the candidate
//! it audits as input, and it depends on no anchor crate, so adding a release
//! never changes a `bundle_id`.

mod description;
mod generate;
pub mod known_answer;
mod plan;

pub use description::{
    ProgramRowsDescription, ReleaseDescription, SetDescription, DESCRIPTION_FILE,
    DESCRIPTION_FORMAT_VERSION, MANIFEST_FILE,
};
pub use generate::{generate, write_release, GeneratedRelease, Role, RISCV64_TARGET};

use akita_config::{CommitmentConfig, TrustedScheduleCatalog};
use akita_cpu_backend::CpuPrefixDeriver;
use akita_error::AkitaError;
use akita_types::preprocessing::{BundleId, ReleaseManifest};
use akita_types::{
    AkitaExpandedSetup, AkitaSetupDescriptor, AkitaSetupSeed, AkitaVerifierSetup,
    SetupPrefixVerifierRegistry, SetupPrefixVerifierSlot,
};
use akita_verifier::LocallyDerivedVerifierSetup;
use std::path::Path;
use std::sync::Arc;

/// Run `$body` with `$cfg` bound to the compiled config of family `$family`.
#[macro_export]
#[doc(hidden)]
macro_rules! dispatch_config {
    ($family:expr, |$cfg:ident| $body:expr) => {{
        use akita_config::proof_optimized::{fp128, fp32, fp64};
        use akita_config::RecursiveCommitmentConfig as Recursive;
        match $family {
            "fp128_dense" => {
                type $cfg = fp128::Dense;
                $body
            }
            "fp128_dense_bounded" => {
                type $cfg = fp128::DenseBounded;
                $body
            }
            "fp128_dense_multi_chunk" => {
                type $cfg = fp128::DenseMultiChunk;
                $body
            }
            "fp128_dense_multi_chunk_w8r3" => {
                type $cfg = fp128::DenseMultiChunkW8R3;
                $body
            }
            "fp128_dense_recursive" => {
                type $cfg = Recursive<fp128::Dense>;
                $body
            }
            "fp128_onehot" => {
                type $cfg = fp128::OneHot;
                $body
            }
            "fp128_onehot_multi_chunk" => {
                type $cfg = fp128::OneHotMultiChunk;
                $body
            }
            "fp128_onehot_multi_chunk_w2r2" => {
                type $cfg = fp128::OneHotMultiChunkW2R2;
                $body
            }
            "fp128_onehot_multi_chunk_w2r3" => {
                type $cfg = fp128::OneHotMultiChunkW2R3;
                $body
            }
            "fp128_onehot_multi_chunk_w4r2" => {
                type $cfg = fp128::OneHotMultiChunkW4R2;
                $body
            }
            "fp128_onehot_multi_chunk_w4r3" => {
                type $cfg = fp128::OneHotMultiChunkW4R3;
                $body
            }
            "fp128_onehot_multi_chunk_w8r3" => {
                type $cfg = fp128::OneHotMultiChunkW8R3;
                $body
            }
            "fp128_onehot_recursive" => {
                type $cfg = Recursive<fp128::OneHot>;
                $body
            }
            "fp128_onehot_recursive_multi_chunk_w2r3" => {
                type $cfg = Recursive<fp128::OneHotMultiChunkW2R3>;
                $body
            }
            "fp128_onehot_recursive_multi_chunk_w4r3" => {
                type $cfg = Recursive<fp128::OneHotMultiChunkW4R3>;
                $body
            }
            "fp128_onehot_recursive_multi_chunk_w8r2" => {
                type $cfg = Recursive<fp128::OneHotMultiChunk>;
                $body
            }
            "fp128_onehot_recursive_multi_chunk_w8r3" => {
                type $cfg = Recursive<fp128::OneHotMultiChunkW8R3>;
                $body
            }
            "fp32_dense" => {
                type $cfg = fp32::Dense;
                $body
            }
            "fp32_dense_recursive" => {
                type $cfg = Recursive<fp32::Dense>;
                $body
            }
            "fp32_onehot" => {
                type $cfg = fp32::OneHot;
                $body
            }
            "fp64_dense" => {
                type $cfg = fp64::Dense;
                $body
            }
            "fp64_dense_recursive" => {
                type $cfg = Recursive<fp64::Dense>;
                $body
            }
            "fp64_onehot" => {
                type $cfg = fp64::OneHot;
                $body
            }
            other => Err(akita_error::AkitaError::InvalidInput(format!(
                "no compiled config has schedule family {other:?}"
            ))),
        }
    }};
}

/// Run `$body` with `$field` bound to the base field of SIS modulus profile
/// `$profile`.
#[macro_export]
#[doc(hidden)]
macro_rules! dispatch_field {
    ($profile:expr, |$field:ident| $body:expr) => {{
        match $profile {
            akita_params::sis::SisModulusProfileId::Q32Offset99 => {
                type $field = akita_config::proof_optimized::fp32::Field;
                $body
            }
            akita_params::sis::SisModulusProfileId::Q64Offset59 => {
                type $field = akita_config::proof_optimized::fp64::Field;
                $body
            }
            akita_params::sis::SisModulusProfileId::Q128OffsetA7F7 => {
                type $field = akita_config::proof_optimized::fp128::Field;
                $body
            }
        }
    }};
}

/// Regenerate the release under `release_dir` from scratch and require its
/// manifest to equal the checked-in one, byte for byte. This is the merge
/// check and audit mode.
///
/// # Errors
///
/// Returns an error naming the first differing part of the manifest.
pub fn check_release(repo_root: &Path, release_dir: &Path) -> Result<BundleId, AkitaError> {
    let description = ReleaseDescription::read(&release_dir.join(DESCRIPTION_FILE))?;
    if release_dir.file_name().and_then(|name| name.to_str()) != Some(description.name.as_str()) {
        return Err(AkitaError::InvalidInput(format!(
            "release {:?} is not stored under params/{0}",
            description.name
        )));
    }
    let manifest_path = release_dir.join(MANIFEST_FILE);
    let checked_in = std::fs::read(&manifest_path).map_err(|error| {
        AkitaError::InvalidInput(format!("cannot read {}: {error}", manifest_path.display()))
    })?;
    let generated = generate(repo_root, &description, &Role::Verifier)?;
    if generated.manifest != checked_in {
        return Err(AkitaError::InvalidSetup(format!(
            "release {:?} does not regenerate: {}",
            description.name,
            manifest_difference(&checked_in, &generated.manifest)
        )));
    }
    Ok(generated.bundle_id)
}

/// Audit the files of a release stored under `files_dir` (for example a
/// downloaded copy) against a from-scratch regeneration.
///
/// # Errors
///
/// Returns an error when the release does not regenerate or a stored file
/// differs from the regenerated one.
pub fn audit_release_files(
    repo_root: &Path,
    release_dir: &Path,
    files_dir: &Path,
) -> Result<BundleId, AkitaError> {
    let bundle = check_release(repo_root, release_dir)?;
    let description = ReleaseDescription::read(&release_dir.join(DESCRIPTION_FILE))?;
    let generated = generate(repo_root, &description, &Role::All)?;
    let manifest = ReleaseManifest::decode(&generated.manifest)?;
    let stored = akita_types::preprocessing::DirectoryReleaseFiles::new(files_dir);
    for (name, bytes) in generated.files.iter() {
        if files_dir.join(name).exists() || files_dir.join(format!("{name}.part-0")).exists() {
            let read = akita_types::preprocessing::read_release_file(
                &stored,
                &manifest,
                name,
                bytes.len(),
            )?;
            if read != bytes {
                return Err(AkitaError::InvalidSetup(format!(
                    "stored file {name:?} differs from the regenerated one"
                )));
            }
        }
    }
    Ok(bundle)
}

fn manifest_difference(expected: &[u8], actual: &[u8]) -> String {
    let (Ok(expected), Ok(actual)) = (
        ReleaseManifest::decode(expected),
        ReleaseManifest::decode(actual),
    ) else {
        return "the checked-in manifest does not decode".into();
    };
    if expected.digest_hash != actual.digest_hash {
        return "digest hash differs".into();
    }
    if expected.streams != actual.streams {
        return "streams differ".into();
    }
    if expected.sets != actual.sets {
        return "set records differ".into();
    }
    if expected.registry != actual.registry {
        return "commitments differ".into();
    }
    if expected.files != actual.files {
        return "files differ".into();
    }
    "packaging header differs".into()
}

/// Release directories under `params_dir`: every subdirectory holding a
/// description, in name order.
///
/// # Errors
///
/// Returns an error when the directory cannot be listed.
pub fn release_dirs(params_dir: &Path) -> Result<Vec<std::path::PathBuf>, AkitaError> {
    let entries = std::fs::read_dir(params_dir).map_err(|error| {
        AkitaError::InvalidInput(format!("cannot list {}: {error}", params_dir.display()))
    })?;
    let mut dirs = entries
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.join(DESCRIPTION_FILE).is_file())
        .collect::<Vec<_>>();
    dirs.sort();
    Ok(dirs)
}

/// A representative slot id of every profile any release under
/// `params_dir` publishes.
///
/// # Errors
///
/// Returns an error when a description or catalog is invalid.
pub fn release_profiles(
    repo_root: &Path,
    params_dir: &Path,
) -> Result<Vec<akita_params::SetupPrefixSlotId>, AkitaError> {
    let mut out = Vec::new();
    for dir in release_dirs(params_dir)? {
        let description = ReleaseDescription::read(&dir.join(DESCRIPTION_FILE))?;
        let seed = description.setup_seed()?;
        for set in &description.sets {
            let plan = dispatch_config!(set.config.as_str(), |Cfg| plan::plan_set::<Cfg>(
                repo_root, set, &seed
            ))?;
            out.extend(plan.profiles.into_values());
        }
    }
    Ok(out)
}

/// Local profile derivation (opt-in, host only): a verifier setup whose
/// matrix and commitments this host derives itself from `setup_seed`, for
/// the rows of `catalog` within the bounds.
///
/// There is no manifest to compare with: the result is accepted only by the
/// `unsafe` `AkitaVerifier::from_locally_derived`.
///
/// # Errors
///
/// Returns an error when no row fits or a commitment cannot be derived.
pub fn derive_local_verifier_setup<Cfg>(
    catalog: &TrustedScheduleCatalog<Cfg>,
    setup_seed: &AkitaSetupSeed,
    max_num_vars: usize,
    max_num_batched_polys: usize,
) -> Result<LocallyDerivedVerifierSetup<Cfg::Field>, AkitaError>
where
    Cfg: CommitmentConfig,
    Cfg::Field: jolt_field::Unreduced
        + jolt_field::WithCommitAccumulator
        + akita_serialization::Valid
        + 'static,
{
    use akita_config::preprocessing::PrefixCommitmentDeriver;

    let rows = catalog.rows_within_setup_capacity(max_num_vars, max_num_batched_polys)?;
    let mut num_field_elements = 1;
    let mut ids = std::collections::BTreeSet::new();
    for row in &rows {
        let layout = row.profiles().opening_layout()?;
        num_field_elements = num_field_elements.max(
            akita_params::verifier_setup_matrix_capacity_for_schedule(row.schedule(), &layout)?
                .num_field_elements,
        );
        ids.extend(akita_config::required_setup_prefix_slot_ids_for_schedule(
            row.schedule(),
            &layout,
        )?);
    }
    let ids = ids.into_iter().collect::<Vec<_>>();
    let expanded = AkitaExpandedSetup::derive(AkitaSetupDescriptor {
        max_num_vars,
        max_num_batched_polys,
        num_field_elements,
        setup_seed: setup_seed.clone(),
    })?;
    let commitments = CpuPrefixDeriver.derive_prefix_commitments(setup_seed, &ids)?;
    let mut registry = SetupPrefixVerifierRegistry::new(setup_seed.clone());
    for (id, commitment) in ids.into_iter().zip(commitments) {
        registry.insert(SetupPrefixVerifierSlot { id, commitment })?;
    }
    // SAFETY: the matrix and every commitment were derived here from the seed.
    unsafe {
        Ok(LocallyDerivedVerifierSetup::assume_locally_derived(
            AkitaVerifierSetup::from_parts(Arc::new(expanded), registry)?,
        ))
    }
}

#[cfg(test)]
mod tests;
