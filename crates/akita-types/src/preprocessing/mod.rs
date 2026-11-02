//! Pinned preprocessing releases: identities, the manifest, trust anchors,
//! and authenticated file reads.
//!
//! A release is the generator's deterministic output for one description:
//! setup-prefix commitments, prover hints, catalogs, and optional guest
//! files, all named by hash in one manifest whose hash is the release's
//! `bundle_id`. A binary trusts only the `(bundle_id, set name)` pairs of its
//! compiled [`TrustAnchor`]. Loaders in `akita-verifier` and
//! `akita-cpu-backend` open a manifest with [`open_anchored_manifest`],
//! which checks the anchor before decoding anything, and then read every file
//! through [`read_release_file`].

mod anchor;
mod encoding;
mod files;
mod identity;
mod manifest;
mod source;

#[cfg(test)]
mod tests;

pub use akita_params::descriptor_bytes::{DigestHash, DIGEST_HASH};
pub use anchor::{AnchorEntry, TrustAnchor};
pub use encoding::hex;
pub use files::{
    catalog_file_name, catalog_view_file_name, file_hash, hints_file_name, part_file_name,
    registry_file_name, terminal_cache_file_name, validate_file_name, validate_name,
    MAX_FILE_NAME_BYTES,
};
pub use identity::{
    known_answer_corpus_digest, known_answer_digest, prefix_commitment_digest, profile_descriptor,
    profile_key, profile_route, setup_prefix_payload, AlgorithmIdentity, BundleId, FileHash,
    PrefixCommitmentDigest, ProfileKey, StreamId, StreamIdentity, KNOWN_ANSWER_CORPUS_VERSION,
    SETUP_PREFIX_PAYLOAD_VERSION,
};
pub use manifest::{
    bundle_id, FileEntry, RegistryEntry, ReleaseKind, ReleaseManifest, SetRecord, StreamEntry,
    MANIFEST_FORMAT_VERSION, MAX_MANIFEST_BYTES,
};
pub use source::{read_release_file, DirectoryReleaseFiles, MemoryReleaseFiles, ReleaseFileSource};

use akita_error::AkitaError;

/// A manifest whose `bundle_id` the anchor accepted for every requested set.
///
/// Every file of one construction comes from this one manifest.
#[derive(Debug, Clone)]
pub struct AnchoredManifest {
    bundle_id: BundleId,
    manifest: ReleaseManifest,
    anchored_sets: Vec<String>,
}

impl AnchoredManifest {
    /// The release identity.
    #[must_use]
    pub const fn bundle_id(&self) -> BundleId {
        self.bundle_id
    }

    /// The decoded manifest.
    #[must_use]
    pub const fn manifest(&self) -> &ReleaseManifest {
        &self.manifest
    }

    /// The record of an anchored set.
    ///
    /// # Errors
    ///
    /// Returns an error when the set was not requested when the manifest was
    /// opened, so its anchor pair was never checked.
    pub fn set(&self, name: &str) -> Result<&SetRecord, AkitaError> {
        if !self.anchored_sets.iter().any(|anchored| anchored == name) {
            return Err(AkitaError::InvalidSetup(format!(
                "parameter set {name:?} was not anchored when the release was opened"
            )));
        }
        self.manifest.set(name).ok_or_else(|| {
            AkitaError::InvalidSetup(format!("release has no parameter set {name:?}"))
        })
    }
}

/// Open a manifest: check its length, hash it, and require every
/// `(bundle_id, set name)` pair to be anchored before decoding it; then
/// require the compiled digest hash, the release kind, and every requested
/// set.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidSetup`] for an oversized manifest, an
/// unanchored pair, a malformed manifest, another digest hash or release
/// kind, or a missing set.
pub fn open_anchored_manifest(
    anchor: &TrustAnchor,
    kind: ReleaseKind,
    manifest_bytes: &[u8],
    set_names: &[&str],
) -> Result<AnchoredManifest, AkitaError> {
    if manifest_bytes.len() > MAX_MANIFEST_BYTES {
        return Err(AkitaError::InvalidSetup(
            "preprocessing manifest exceeds the length cap".into(),
        ));
    }
    if set_names.is_empty() {
        return Err(AkitaError::InvalidSetup(
            "a release load names at least one parameter set".into(),
        ));
    }
    let bundle_id = bundle_id(manifest_bytes);
    for set_name in set_names {
        anchor.require(&bundle_id, set_name)?;
    }
    let manifest = ReleaseManifest::decode(manifest_bytes)?;
    if manifest.digest_hash != DIGEST_HASH {
        return Err(AkitaError::InvalidSetup(format!(
            "release digest hash {:?} does not match the compiled {DIGEST_HASH:?}",
            manifest.digest_hash
        )));
    }
    if manifest.kind != kind {
        return Err(AkitaError::InvalidSetup(format!(
            "release kind {:?} does not match the integration's {kind:?}",
            manifest.kind
        )));
    }
    let opened = AnchoredManifest {
        bundle_id,
        manifest,
        anchored_sets: set_names.iter().map(|name| (*name).to_owned()).collect(),
    };
    for set_name in set_names {
        opened.set(set_name)?;
    }
    Ok(opened)
}
