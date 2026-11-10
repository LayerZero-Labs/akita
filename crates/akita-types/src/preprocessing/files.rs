//! Release file names and content hashes.

use super::encoding::{push_str, push_u64};
use super::identity::{FileHash, ProfileKey, StreamId};
use akita_error::AkitaError;
use akita_params::{DescriptorHasher, ScheduleRowDigest};

const FILE_TAG: &str = "akita/preprocessing/file/v1";

/// Longest release file name.
pub const MAX_FILE_NAME_BYTES: usize = 255;
/// Longest set or target name.
pub(crate) const MAX_NAME_BYTES: usize = 64;

/// `H(tag("akita/preprocessing/file/v1") || str(name) || bytes(content))`.
#[must_use]
pub fn file_hash(name: &str, content: &[u8]) -> FileHash {
    let mut head = Vec::with_capacity(FILE_TAG.len() + name.len() + 32);
    push_str(&mut head, FILE_TAG);
    push_str(&mut head, name);
    push_u64(&mut head, content.len() as u64);
    let mut hasher = DescriptorHasher::new();
    hasher.update(&head);
    hasher.update(content);
    FileHash::from_bytes(hasher.finalize())
}

/// `registry/<hex(stream_id)>/<hex(profile_key)>`: the commitment payload.
#[must_use]
pub fn registry_file_name(stream_id: &StreamId, profile_key: &ProfileKey) -> String {
    format!("registry/{stream_id}/{profile_key}")
}

/// `hints/<hex(stream_id)>/<hex(profile_key)>`: prover opening material.
#[must_use]
pub fn hints_file_name(stream_id: &StreamId, profile_key: &ProfileKey) -> String {
    format!("hints/{stream_id}/{profile_key}")
}

/// `catalog/<set>`: the set's catalog artifact.
#[must_use]
pub fn catalog_file_name(set_name: &str) -> String {
    format!("catalog/{set_name}")
}

/// `catalog-view/<set>/<hex(row digest)>`: a selected-row verifier view.
#[must_use]
pub fn catalog_view_file_name(set_name: &str, row: &ScheduleRowDigest) -> String {
    format!(
        "catalog-view/{set_name}/{}",
        super::encoding::hex(row.as_bytes())
    )
}

/// `terminal-cache/<hex(stream_id)>/<hex(row digest)>/<target>/<capacity>`.
#[must_use]
pub fn terminal_cache_file_name(
    stream_id: &StreamId,
    row: &ScheduleRowDigest,
    target: &str,
    capacity: usize,
) -> String {
    format!(
        "terminal-cache/{stream_id}/{}/{target}/{capacity}",
        super::encoding::hex(row.as_bytes())
    )
}

/// `<name>.part-<index>`: one stored part of a file split for transport.
#[must_use]
pub fn part_file_name(name: &str, index: usize) -> String {
    format!("{name}.part-{index}")
}

/// Check a set or target name: 1 to 64 bytes of `[a-z0-9._-]`, starting with
/// a letter or digit.
pub fn validate_name(name: &str) -> Result<(), AkitaError> {
    let bytes = name.as_bytes();
    let valid = !bytes.is_empty()
        && bytes.len() <= MAX_NAME_BYTES
        && bytes[0].is_ascii_alphanumeric()
        && bytes.iter().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'.' | b'_' | b'-')
        });
    if valid {
        Ok(())
    } else {
        Err(AkitaError::InvalidSetup(format!(
            "invalid preprocessing set or target name {name:?}"
        )))
    }
}

/// Check a release file name: `/`-separated components, each a valid name,
/// and at most 255 bytes in all. No component may be `.` or `..`.
pub fn validate_file_name(name: &str) -> Result<(), AkitaError> {
    if name.is_empty() || name.len() > MAX_FILE_NAME_BYTES {
        return Err(AkitaError::InvalidSetup(format!(
            "invalid preprocessing file name {name:?}"
        )));
    }
    for component in name.split('/') {
        validate_name(component).map_err(|_| {
            AkitaError::InvalidSetup(format!("invalid preprocessing file name {name:?}"))
        })?;
    }
    Ok(())
}
