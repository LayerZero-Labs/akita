//! Disk cache of release files keyed by `bundle_id`.

use akita_error::AkitaError;
use akita_types::preprocessing::{
    bundle_id, validate_file_name, BundleId, DirectoryReleaseFiles, MAX_MANIFEST_BYTES,
};
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static CACHE_TEMP_ID: AtomicU64 = AtomicU64::new(0);

/// Name of a release's manifest inside its cache directory. The manifest is
/// not a release file; it is stored under the `bundle_id` it hashes to.
const MANIFEST_FILE: &str = "manifest";

/// Release files cached on disk, one directory per `bundle_id`.
///
/// Nothing read from the cache is trusted: [`Self::manifest`] returns bytes
/// only when they hash to the requested `bundle_id`, and [`Self::files`]
/// hands a loader a source whose every file it checks against that manifest.
#[derive(Debug, Clone)]
pub struct ReleaseCache {
    root: PathBuf,
}

impl ReleaseCache {
    /// A cache rooted at `root`.
    #[must_use]
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// The per-user cache, `<cache dir>/akita/releases`.
    #[must_use]
    pub fn user_default() -> Option<Self> {
        let mut path = if let Some(local_app_data) = std::env::var_os("LOCALAPPDATA") {
            PathBuf::from(local_app_data)
        } else {
            let home = PathBuf::from(std::env::var_os("HOME")?);
            let macos_cache = home.join("Library").join("Caches");
            if macos_cache.exists() {
                macos_cache
            } else {
                home.join(".cache")
            }
        };
        path.push("akita");
        path.push("releases");
        Some(Self::new(path))
    }

    fn release_dir(&self, bundle_id: &BundleId) -> PathBuf {
        self.root.join(bundle_id.to_hex())
    }

    /// Untrusted source of the cached files of release `bundle_id`.
    #[must_use]
    pub fn files(&self, bundle_id: &BundleId) -> DirectoryReleaseFiles {
        DirectoryReleaseFiles::new(self.release_dir(bundle_id))
    }

    /// The cached manifest of release `bundle_id`, if its bytes hash to it.
    ///
    /// # Errors
    ///
    /// Returns an error when the manifest is missing, oversized, unreadable,
    /// or does not hash to `bundle_id`.
    pub fn manifest(&self, id: &BundleId) -> Result<Vec<u8>, AkitaError> {
        let path = self.release_dir(id).join(MANIFEST_FILE);
        let file = fs::File::open(&path).map_err(|error| io_error(&path, &error))?;
        let mut bytes = Vec::new();
        file.take(MAX_MANIFEST_BYTES as u64 + 1)
            .read_to_end(&mut bytes)
            .map_err(|error| io_error(&path, &error))?;
        if bytes.len() > MAX_MANIFEST_BYTES || bundle_id(&bytes) != *id {
            return Err(AkitaError::InvalidSetup(format!(
                "cached manifest {} does not hash to its bundle id",
                path.display()
            )));
        }
        Ok(bytes)
    }

    /// Store a manifest under the `bundle_id` it hashes to.
    ///
    /// # Errors
    ///
    /// Returns an error when the manifest is oversized or cannot be written.
    pub fn store_manifest(&self, manifest: &[u8]) -> Result<BundleId, AkitaError> {
        if manifest.len() > MAX_MANIFEST_BYTES {
            return Err(AkitaError::InvalidSetup(
                "preprocessing manifest exceeds the length cap".into(),
            ));
        }
        let id = bundle_id(manifest);
        atomic_write(&self.release_dir(&id).join(MANIFEST_FILE), manifest)?;
        Ok(id)
    }

    /// Store one release file of release `bundle_id`.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid file name or a failed write.
    pub fn store_file(
        &self,
        bundle_id: &BundleId,
        name: &str,
        bytes: &[u8],
    ) -> Result<(), AkitaError> {
        validate_file_name(name)?;
        atomic_write(&self.release_dir(bundle_id).join(name), bytes)
    }
}

fn io_error(path: &Path, error: &std::io::Error) -> AkitaError {
    AkitaError::InvalidSetup(format!("release cache {}: {error}", path.display()))
}

/// Removes an incomplete temporary file unless disarmed.
struct TempGuard<'a> {
    path: &'a Path,
    armed: bool,
}

impl Drop for TempGuard<'_> {
    fn drop(&mut self) {
        if self.armed {
            let _ = fs::remove_file(self.path);
        }
    }
}

fn atomic_write(path: &Path, bytes: &[u8]) -> Result<(), AkitaError> {
    let parent = path
        .parent()
        .ok_or_else(|| AkitaError::InvalidSetup("release cache path has no parent".into()))?;
    fs::create_dir_all(parent).map_err(|error| io_error(parent, &error))?;
    let temp_id = CACHE_TEMP_ID.fetch_add(1, Ordering::Relaxed);
    let temp = path.with_extension(format!("tmp-{}-{temp_id}", std::process::id()));
    let mut guard = TempGuard {
        path: &temp,
        armed: true,
    };
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temp)
        .map_err(|error| io_error(&temp, &error))?;
    file.write_all(bytes)
        .and_then(|()| file.flush())
        .map_err(|error| io_error(&temp, &error))?;
    drop(file);
    fs::rename(&temp, path).map_err(|error| io_error(path, &error))?;
    guard.armed = false;
    Ok(())
}
