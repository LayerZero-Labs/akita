//! Akita's compiled trust anchor.
//!
//! [`AKITA_ANCHOR`] lists the `(bundle_id, set name)` pairs of every release
//! committed under `params/`, and nothing else. Binaries pass it to the
//! release loaders; the generator never depends on this crate, so adding a
//! release never changes a `bundle_id`.
//!
//! To anchor a release, generate its manifest with
//! `akita-preprocess generate params/<release> <dir> --write-manifest` and
//! paste the lines `akita-preprocess anchor params/<release>` prints.

use akita_types::preprocessing::{AnchorEntry, TrustAnchor};

/// The pairs Akita trusts.
#[allow(clippy::disallowed_methods)] // the one place Akita declares its anchor
pub const AKITA_ANCHOR: TrustAnchor = TrustAnchor::new(&[
    AnchorEntry::new(AKITA_V1, "fp128-dense"),
    AnchorEntry::new(AKITA_V1, "fp128-dense-recursive"),
    AnchorEntry::new(AKITA_V1, "fp32-dense-recursive"),
    AnchorEntry::new(AKITA_V1, "fp64-dense-recursive"),
]);

/// `bundle_id` of `params/akita-v1`.
const AKITA_V1: &str = "da55142157ea66490419edb4cc16f732cd35a6c9b007add3f2e9677697da7ceb";

/// The manifest of `params/akita-v1`. Loaders still check it against the
/// anchor before decoding it.
pub const AKITA_V1_MANIFEST: &[u8] = include_bytes!("../../../params/akita-v1/manifest");

#[cfg(test)]
mod tests {
    use super::*;
    use akita_types::preprocessing::{bundle_id, ReleaseManifest};
    use std::collections::BTreeSet;
    use std::path::Path;

    /// The anchor lists exactly the pairs of the releases under `params/`.
    #[test]
    fn anchor_lists_exactly_the_committed_releases() {
        let params = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../params");
        let mut committed = BTreeSet::new();
        for entry in std::fs::read_dir(&params).expect("params directory") {
            let release = entry.expect("params entry").path();
            if !release.is_dir() {
                continue;
            }
            // `akita-preprocess check` regenerates every release from its
            // description; a manifest without one would be anchored but
            // never regenerated.
            assert!(
                release.join("description.json").is_file(),
                "{} has no description.json",
                release.display()
            );
            let bytes = std::fs::read(release.join("manifest")).expect("manifest");
            let id = bundle_id(&bytes);
            for set in ReleaseManifest::decode(&bytes)
                .expect("decode manifest")
                .sets
            {
                committed.insert((id, set.name));
            }
        }
        let anchored = AKITA_ANCHOR
            .entries()
            .iter()
            .map(|entry| (entry.bundle_id(), entry.set_name().to_owned()))
            .collect::<BTreeSet<_>>();
        assert_eq!(anchored, committed);
        assert_eq!(
            anchored.len(),
            AKITA_ANCHOR.entries().len(),
            "duplicate pair"
        );
    }
}
