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
//! This crate is the only component that publishes setup-prefix commitments.
//! It depends on neither `akita-setup`, whose cache it must not read, nor an
//! anchor crate, so adding a release never changes a `bundle_id`.

mod description;
mod generate;
pub mod known_answer;
mod plan;

pub use description::{
    ProgramRowsDescription, ReleaseDescription, SetDescription, DESCRIPTION_FILE,
    DESCRIPTION_FORMAT_VERSION, MANIFEST_FILE, RISCV64_TARGET,
};
pub use generate::{generate, write_release, GeneratedRelease, Role};

use akita_error::AkitaError;
use akita_params::SetupPrefixSlotId;
use akita_serialization::{AkitaDeserialize, Valid};
use akita_types::preprocessing::{BundleId, ReleaseManifest};
use std::path::{Path, PathBuf};

/// Run `$body` with `$cfg` bound to the compiled config of family `$family`.
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
pub(crate) use dispatch_config;

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
/// merge check and audit mode.
///
/// # Errors
///
/// Returns an error naming the first differing part of the manifest, or the
/// corpus lines that would cover the missing classes.
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
    let generated = generate(repo_root, &description)?;
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

/// Release directories under `params_dir`: every subdirectory holding a
/// description, in name order.
///
/// # Errors
///
/// Returns an error when the directory cannot be listed.
pub fn release_dirs(params_dir: &Path) -> Result<Vec<PathBuf>, AkitaError> {
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
/// errors.
pub(crate) fn parse_slot_ids(
    text: &str,
    origin: &str,
) -> Result<Vec<SetupPrefixSlotId>, AkitaError> {
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
