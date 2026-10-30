//! Untrusted transport of release files and their authenticated reads.

use super::files::{file_hash, part_file_name, validate_file_name};
use super::manifest::ReleaseManifest;
use akita_error::AkitaError;
use std::collections::BTreeMap;
use std::io::Read;
use std::path::PathBuf;

/// Where release files are read from: a directory, a cache, or memory. Every source is untrusted; [`read_release_file`] checks what it
/// returns against the manifest.
pub trait ReleaseFileSource {
    /// Read the stored bytes of file `name` into a new buffer of exactly
    /// `length` bytes, the manifest length. Implementations must not allocate
    /// more than `length + 1` bytes and must fail when the stored file has
    /// another length.
    ///
    /// # Errors
    ///
    /// Returns an error when the file is missing, unreadable, or has another
    /// length.
    fn read_file(&self, name: &str, length: usize) -> Result<Vec<u8>, AkitaError>;
}

/// Read one release file into a private buffer and check it against the
/// manifest entry of the same name.
///
/// The manifest length is checked against `max_len` before anything is
/// allocated, and the returned buffer holds exactly the bytes that were
/// hashed: decode it and never read the file again.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidSetup`] when the manifest has no such file,
/// the length exceeds `max_len`, or the bytes do not match.
pub fn read_release_file(
    source: &dyn ReleaseFileSource,
    manifest: &ReleaseManifest,
    name: &str,
    max_len: usize,
) -> Result<Vec<u8>, AkitaError> {
    let entry = manifest.file(name).ok_or_else(|| {
        AkitaError::InvalidSetup(format!("release manifest lists no file {name:?}"))
    })?;
    let length = usize::try_from(entry.length)
        .ok()
        .filter(|length| *length <= max_len)
        .ok_or_else(|| {
            AkitaError::InvalidSetup(format!(
                "release file {name:?} of {} bytes exceeds its {max_len}-byte cap",
                entry.length
            ))
        })?;
    let bytes = source.read_file(name, length)?;
    if bytes.len() != length || file_hash(name, &bytes) != entry.file_hash {
        return Err(AkitaError::InvalidSetup(format!(
            "release file {name:?} does not match its manifest entry"
        )));
    }
    Ok(bytes)
}

/// Release files held in memory, keyed by file name.
#[derive(Debug, Clone, Default)]
pub struct MemoryReleaseFiles {
    files: BTreeMap<String, Vec<u8>>,
}

impl MemoryReleaseFiles {
    /// An empty set of files.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add or replace one file.
    pub fn insert(&mut self, name: impl Into<String>, bytes: Vec<u8>) {
        self.files.insert(name.into(), bytes);
    }

    /// Borrow one stored file.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&[u8]> {
        self.files.get(name).map(Vec::as_slice)
    }

    /// Every stored file, in name order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &[u8])> {
        self.files
            .iter()
            .map(|(name, bytes)| (name.as_str(), bytes.as_slice()))
    }
}

impl ReleaseFileSource for MemoryReleaseFiles {
    fn read_file(&self, name: &str, length: usize) -> Result<Vec<u8>, AkitaError> {
        copy_exact(self.get(name), name, length)
    }
}

fn copy_exact(stored: Option<&[u8]>, name: &str, length: usize) -> Result<Vec<u8>, AkitaError> {
    match stored {
        Some(bytes) if bytes.len() == length => Ok(bytes.to_vec()),
        Some(_) => Err(AkitaError::InvalidSetup(format!(
            "release file {name:?} has the wrong length"
        ))),
        None => Err(AkitaError::InvalidSetup(format!(
            "release file {name:?} is missing"
        ))),
    }
}

/// Release files stored under a directory, at their release names.
///
/// A file larger than the release's largest part may instead be stored as
/// consecutive parts `<name>.part-0`, `<name>.part-1`, ….
#[derive(Debug, Clone)]
pub struct DirectoryReleaseFiles {
    root: PathBuf,
}

impl DirectoryReleaseFiles {
    /// Files stored under `root`.
    #[must_use]
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }
}

impl ReleaseFileSource for DirectoryReleaseFiles {
    fn read_file(&self, name: &str, length: usize) -> Result<Vec<u8>, AkitaError> {
        validate_file_name(name)?;
        let io_error = |error: std::io::Error| {
            AkitaError::InvalidSetup(format!("failed to read release file {name:?}: {error}"))
        };
        let limit = (length as u64).saturating_add(1);
        let mut bytes = Vec::with_capacity(length);
        match std::fs::File::open(self.root.join(name)) {
            Ok(file) => {
                file.take(limit).read_to_end(&mut bytes).map_err(io_error)?;
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                let mut part = 0usize;
                loop {
                    let file = match std::fs::File::open(self.root.join(part_file_name(name, part)))
                    {
                        Ok(file) => file,
                        Err(error) if error.kind() == std::io::ErrorKind::NotFound && part > 0 => {
                            break
                        }
                        Err(error) => return Err(io_error(error)),
                    };
                    let remaining = limit - bytes.len() as u64;
                    file.take(remaining)
                        .read_to_end(&mut bytes)
                        .map_err(io_error)?;
                    if bytes.len() > length {
                        break;
                    }
                    part += 1;
                }
            }
            Err(error) => return Err(io_error(error)),
        }
        if bytes.len() != length {
            return Err(AkitaError::InvalidSetup(format!(
                "release file {name:?} has the wrong length"
            )));
        }
        Ok(bytes)
    }
}
