//! Content-addressed storage for deterministic offline work results.

use sha2::{Digest, Sha256};
use std::{
    fs,
    io::Write,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

use crate::error::{EstimatorError, Result};

const RESULT_HEADER: &str = "akita-work-result-v1";
static TEMP_NONCE: AtomicU64 = AtomicU64::new(0);

/// Content address of one canonical offline work specification.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WorkId([u8; 32]);

impl WorkId {
    /// Hash one domain-separated canonical work specification.
    #[must_use]
    pub fn new(domain: &[u8], canonical_spec: &[u8]) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(b"akita-offline-work-v1\0");
        hasher.update((domain.len() as u64).to_le_bytes());
        hasher.update(domain);
        hasher.update((canonical_spec.len() as u64).to_le_bytes());
        hasher.update(canonical_spec);
        Self(hasher.finalize().into())
    }

    /// Lowercase hexadecimal identifier used for cache paths and plans.
    #[must_use]
    pub fn hex(self) -> String {
        self.0.iter().map(|byte| format!("{byte:02x}")).collect()
    }

    /// Stable zero-based shard assignment.
    ///
    /// # Errors
    ///
    /// Returns an error when `shard_count` is zero.
    pub fn shard(self, shard_count: u32) -> Result<u32> {
        if shard_count == 0 {
            return Err(EstimatorError::InvalidConfig {
                field: "shard_count",
                reason: "shard count must be positive".to_string(),
            });
        }
        let prefix = u64::from_le_bytes(
            self.0[..8]
                .try_into()
                .expect("work identifiers always contain eight prefix bytes"),
        );
        Ok((prefix % u64::from(shard_count)) as u32)
    }
}

/// Filesystem-backed immutable work-result cache.
#[derive(Clone, Debug)]
pub struct WorkCache {
    root: PathBuf,
}

impl WorkCache {
    /// Open a cache rooted at `root`.
    #[must_use]
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// Read and authenticate a cached payload.
    ///
    /// # Errors
    ///
    /// Returns an error for malformed, mismatched, or unreadable cache files.
    pub fn load(&self, id: WorkId) -> Result<Option<Vec<u8>>> {
        let path = self.path(id);
        let bytes = match fs::read(&path) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return cache_error(format!("read {}: {error}", path.display())),
        };
        decode_envelope(id, &bytes).map(Some)
    }

    /// Atomically store one immutable payload.
    ///
    /// Repeating an identical write is allowed. A different payload for an
    /// existing work identifier is rejected.
    ///
    /// # Errors
    ///
    /// Returns an error when storage fails or an existing result conflicts.
    pub fn store(&self, id: WorkId, payload: &[u8]) -> Result<()> {
        if let Some(existing) = self.load(id)? {
            return if existing == payload {
                Ok(())
            } else {
                cache_error(format!("conflicting result for work item {}", id.hex()))
            };
        }

        let destination = self.path(id);
        let parent = destination.parent().ok_or(EstimatorError::InvalidConfig {
            field: "work_cache",
            reason: "work-result path has no parent directory".to_string(),
        })?;
        fs::create_dir_all(parent)
            .map_err(|error| cache_error_value(format!("create {}: {error}", parent.display())))?;

        let nonce = TEMP_NONCE.fetch_add(1, Ordering::Relaxed);
        let temporary = parent.join(format!(".{}.{}.{nonce}.tmp", id.hex(), std::process::id()));
        let encoded = encode_envelope(id, payload);
        let mut file = fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temporary)
            .map_err(|error| {
                cache_error_value(format!("create {}: {error}", temporary.display()))
            })?;
        if let Err(error) = file.write_all(&encoded).and_then(|()| file.sync_all()) {
            let _ = fs::remove_file(&temporary);
            return cache_error(format!("write {}: {error}", temporary.display()));
        }

        match fs::hard_link(&temporary, &destination) {
            Ok(()) => {
                let _ = fs::remove_file(&temporary);
                Ok(())
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                let _ = fs::remove_file(&temporary);
                match self.load(id)? {
                    Some(existing) if existing == payload => Ok(()),
                    Some(_) => cache_error(format!(
                        "conflicting concurrent result for work item {}",
                        id.hex()
                    )),
                    None => cache_error(format!(
                        "work item {} disappeared during concurrent storage",
                        id.hex()
                    )),
                }
            }
            Err(error) => {
                let _ = fs::remove_file(&temporary);
                cache_error(format!("publish {}: {error}", destination.display()))
            }
        }
    }

    fn path(&self, id: WorkId) -> PathBuf {
        let hex = id.hex();
        self.root
            .join("objects")
            .join(&hex[..2])
            .join(format!("{}.result", &hex[2..]))
    }
}

fn encode_envelope(id: WorkId, payload: &[u8]) -> Vec<u8> {
    let mut encoded = format!("{RESULT_HEADER}\nwork_id={}\n\n", id.hex()).into_bytes();
    encoded.extend_from_slice(payload);
    encoded
}

fn decode_envelope(id: WorkId, encoded: &[u8]) -> Result<Vec<u8>> {
    let separator = b"\n\n";
    let split = encoded
        .windows(separator.len())
        .position(|window| window == separator)
        .ok_or_else(|| cache_error_value("work-result envelope is missing its payload"))?;
    let header = std::str::from_utf8(&encoded[..split])
        .map_err(|error| cache_error_value(format!("work-result header is not UTF-8: {error}")))?;
    let expected = format!("{RESULT_HEADER}\nwork_id={}", id.hex());
    if header != expected {
        return cache_error(format!(
            "work-result envelope does not match requested work item {}",
            id.hex()
        ));
    }
    Ok(encoded[split + separator.len()..].to_vec())
}

fn cache_error<T>(reason: impl Into<String>) -> Result<T> {
    Err(cache_error_value(reason))
}

fn cache_error_value(reason: impl Into<String>) -> EstimatorError {
    EstimatorError::InvalidConfig {
        field: "work_cache",
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_root(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "akita-work-cache-{label}-{}-{}",
            std::process::id(),
            TEMP_NONCE.fetch_add(1, Ordering::Relaxed)
        ))
    }

    #[test]
    fn identifiers_are_deterministic_and_domain_separated() {
        let first = WorkId::new(b"estimator-a", b"input");
        assert_eq!(first, WorkId::new(b"estimator-a", b"input"));
        assert_ne!(first, WorkId::new(b"estimator-b", b"input"));
        assert_ne!(first, WorkId::new(b"estimator-a", b"other"));
        assert_eq!(first.hex().len(), 64);
    }

    #[test]
    fn cache_is_atomic_idempotent_and_conflict_detecting() {
        let root = test_root("roundtrip");
        let cache = WorkCache::new(&root);
        let id = WorkId::new(b"test", b"one");
        assert_eq!(cache.load(id).unwrap(), None);
        cache.store(id, b"result").unwrap();
        cache.store(id, b"result").unwrap();
        assert_eq!(cache.load(id).unwrap(), Some(b"result".to_vec()));
        assert!(cache.store(id, b"different").is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn malformed_or_mismatched_entries_fail_closed() {
        let root = test_root("malformed");
        let cache = WorkCache::new(&root);
        let id = WorkId::new(b"test", b"one");
        let path = cache.path(id);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, b"not an envelope").unwrap();
        assert!(cache.load(id).is_err());
        fs::remove_dir_all(root).unwrap();
    }
}
