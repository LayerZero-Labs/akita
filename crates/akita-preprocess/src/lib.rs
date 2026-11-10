//! Generator and checker of pinned Akita preprocessing releases.
//!
//! [`generate`] builds a release from a description checked in under
//! `params/<release>/`: it derives each stream's public matrix once, commits
//! every setup-prefix profile any set needs with the CPU reference path, and
//! produces registry payloads, optional prover hints, catalogs, optional
//! guest files, and the manifest that names every file by hash.
//! [`check_release`] regenerates a release and compares the manifest with the
//! checked-in one.
//!
//! The Akita Book's usage page *Preprocessing releases*
//! (`book/src/usage/preprocessing-releases.md`) walks through writing a
//! description, generating, anchoring, publishing, and loading a release.
//!
//! This crate is the only component that publishes setup-prefix commitments.
//! It depends on neither `akita-setup`, whose cache it must not read, nor an
//! anchor crate, so adding a release never changes a `bundle_id`.

mod cli;
mod description;
mod generate;
pub mod known_answer;
mod local;
mod plan;

pub use cli::cli_main;
pub use description::{
    ProgramRowsDescription, ReleaseDescription, SetDescription, DESCRIPTION_FILE,
    DESCRIPTION_FORMAT_VERSION, MANIFEST_FILE, RISCV64_TARGET,
};
pub use generate::{generate, write_release, GeneratedRelease, Role};
pub use local::derive_local_verifier_setup;
pub use plan::{plan_set, SetPlan};

use akita_error::AkitaError;
use akita_params::SetupPrefixSlotId;
use akita_serialization::{AkitaDeserialize, Valid};
use akita_types::preprocessing::{BundleId, ReleaseManifest};
use akita_types::AkitaSetupSeed;
use std::path::{Path, PathBuf};

/// Maps a description's `config` name to a compiled
/// [`CommitmentConfig`](akita_config::CommitmentConfig).
///
/// A generator plans each set under the configuration its `config` names by
/// calling [`plan_set`] with that type. [`AkitaConfigs`] covers Akita's own
/// families; an integration with its own configurations, such as Jolt,
/// implements this for its generator binary.
pub trait ConfigRegistry {
    /// Plan `set` with [`plan_set`] under the configuration named
    /// `set.config`.
    ///
    /// # Errors
    ///
    /// Returns an error when no configuration has that name, or planning
    /// fails.
    fn plan_set(
        &self,
        repo_root: &Path,
        set: &SetDescription,
        setup_seed: &AkitaSetupSeed,
    ) -> Result<SetPlan, AkitaError>;
}

/// Akita's compiled configurations, by `schedule_family_name()`.
#[derive(Debug, Clone, Copy, Default)]
pub struct AkitaConfigs;

impl ConfigRegistry for AkitaConfigs {
    fn plan_set(
        &self,
        repo_root: &Path,
        set: &SetDescription,
        setup_seed: &AkitaSetupSeed,
    ) -> Result<SetPlan, AkitaError> {
        use akita_config::proof_optimized::{fp128, fp32, fp64};
        use akita_config::RecursiveCommitmentConfig as Recursive;
        macro_rules! plan {
            ($cfg:ty) => {
                plan_set::<$cfg>(repo_root, set, setup_seed)
            };
        }
        match set.config.as_str() {
            "fp128_dense" => plan!(fp128::Dense),
            "fp128_dense_bounded" => plan!(fp128::DenseBounded),
            "fp128_dense_multi_chunk" => plan!(fp128::DenseMultiChunk),
            "fp128_dense_multi_chunk_w8r3" => plan!(fp128::DenseMultiChunkW8R3),
            "fp128_dense_recursive" => plan!(Recursive<fp128::Dense>),
            "fp128_onehot" => plan!(fp128::OneHot),
            "fp128_onehot_multi_chunk" => plan!(fp128::OneHotMultiChunk),
            "fp128_onehot_multi_chunk_w2r2" => plan!(fp128::OneHotMultiChunkW2R2),
            "fp128_onehot_multi_chunk_w2r3" => plan!(fp128::OneHotMultiChunkW2R3),
            "fp128_onehot_multi_chunk_w4r2" => plan!(fp128::OneHotMultiChunkW4R2),
            "fp128_onehot_multi_chunk_w4r3" => plan!(fp128::OneHotMultiChunkW4R3),
            "fp128_onehot_multi_chunk_w8r3" => plan!(fp128::OneHotMultiChunkW8R3),
            "fp128_onehot_recursive" => plan!(Recursive<fp128::OneHot>),
            "fp128_onehot_recursive_multi_chunk_w2r3" => {
                plan!(Recursive<fp128::OneHotMultiChunkW2R3>)
            }
            "fp128_onehot_recursive_multi_chunk_w4r3" => {
                plan!(Recursive<fp128::OneHotMultiChunkW4R3>)
            }
            "fp128_onehot_recursive_multi_chunk_w8r2" => plan!(Recursive<fp128::OneHotMultiChunk>),
            "fp128_onehot_recursive_multi_chunk_w8r3" => {
                plan!(Recursive<fp128::OneHotMultiChunkW8R3>)
            }
            "fp32_dense" => plan!(fp32::Dense),
            "fp32_dense_recursive" => plan!(Recursive<fp32::Dense>),
            "fp32_onehot" => plan!(fp32::OneHot),
            "fp64_dense" => plan!(fp64::Dense),
            "fp64_dense_recursive" => plan!(Recursive<fp64::Dense>),
            "fp64_onehot" => plan!(fp64::OneHot),
            other => Err(AkitaError::InvalidInput(format!(
                "no compiled config has schedule family {other:?}"
            ))),
        }
    }
}

/// Run `$body` with `$field` bound to the base field of SIS modulus profile
/// `$profile`.
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
pub(crate) use dispatch_field;

/// Regenerate the release under `release_dir` from scratch, require its
/// manifest to equal the checked-in one byte for byte, and require every
/// profile it publishes to fall in a known-answer corpus class. This is the
/// merge check and audit mode. `configs` names each set's compiled
/// configuration.
///
/// # Errors
///
/// Returns an error for a description that names a path that is not a
/// normalized repository path, naming the first differing part of the
/// manifest, or the corpus lines that would cover the missing classes.
pub fn check_release(
    configs: &dyn ConfigRegistry,
    repo_root: &Path,
    release_dir: &Path,
) -> Result<BundleId, AkitaError> {
    let description = ReleaseDescription::read(&release_dir.join(DESCRIPTION_FILE))?;
    description.validate_repository_paths()?;
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
    let generated = generate(configs, repo_root, &description)?;
    if generated.manifest != checked_in {
        return Err(AkitaError::InvalidSetup(format!(
            "release {:?} does not regenerate: {}",
            description.name,
            manifest_difference(&checked_in, &generated.manifest)
        )));
    }
    known_answer::check_corpus_coverage(&generated.profiles)?;
    Ok(generated.bundle_id)
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

/// Release directories under `params_dir`: every subdirectory, in name
/// order. Each must hold both a description and a manifest, so no manifest
/// can be anchored without the description the merge check regenerates it
/// from.
///
/// # Errors
///
/// Returns an error when the directory cannot be listed or a subdirectory
/// lacks its description or manifest.
pub fn release_dirs(params_dir: &Path) -> Result<Vec<PathBuf>, AkitaError> {
    let entries = std::fs::read_dir(params_dir).map_err(|error| {
        AkitaError::InvalidInput(format!("cannot list {}: {error}", params_dir.display()))
    })?;
    let mut dirs = Vec::new();
    for entry in entries {
        let path = entry
            .map_err(|error| {
                AkitaError::InvalidInput(format!("cannot list {}: {error}", params_dir.display()))
            })?
            .path();
        if !path.is_dir() {
            continue;
        }
        for file in [DESCRIPTION_FILE, MANIFEST_FILE] {
            if !path.join(file).is_file() {
                return Err(AkitaError::InvalidInput(format!(
                    "release directory {} has no {file}",
                    path.display()
                )));
            }
        }
        dirs.push(path);
    }
    dirs.sort();
    Ok(dirs)
}

/// Parse lowercase hex digits.
pub(crate) fn parse_hex(hex: &str) -> Result<Vec<u8>, AkitaError> {
    let digit = |byte: u8| match byte {
        b'0'..=b'9' => Ok(byte - b'0'),
        b'a'..=b'f' => Ok(byte - b'a' + 10),
        _ => Err(AkitaError::InvalidInput(format!(
            "{hex:?} is not lowercase hex"
        ))),
    };
    if !hex.len().is_multiple_of(2) {
        return Err(AkitaError::InvalidInput(format!(
            "{hex:?} has an odd number of hex digits"
        )));
    }
    hex.as_bytes()
        .chunks_exact(2)
        .map(|pair| Ok((digit(pair[0])? << 4) | digit(pair[1])?))
        .collect()
}

/// Parse slot-id lines: one hex-encoded uncompressed `SetupPrefixSlotId` per
/// line, skipping blank lines and `#` comments. `origin` names the text in
/// errors. Universe files and the known-answer corpus use this format.
///
/// # Errors
///
/// Returns an error when a line is not a valid, checked slot id.
pub fn parse_slot_ids(text: &str, origin: &str) -> Result<Vec<SetupPrefixSlotId>, AkitaError> {
    text.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(|line| {
            let invalid = |error: String| {
                AkitaError::InvalidInput(format!("invalid slot id in {origin}: {error}"))
            };
            let id = SetupPrefixSlotId::deserialize_uncompressed_exact(&parse_hex(line)?, &())
                .map_err(|error| invalid(error.to_string()))?;
            id.check().map_err(|error| invalid(error.to_string()))?;
            Ok(id)
        })
        .collect()
}

#[cfg(test)]
mod tests;
