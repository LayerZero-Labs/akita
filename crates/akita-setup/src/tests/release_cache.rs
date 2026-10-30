use crate::ReleaseCache;
use akita_types::preprocessing::{bundle_id, read_release_file, ReleaseManifest};
use std::fs;

fn scratch(name: &str) -> std::path::PathBuf {
    let root =
        std::env::temp_dir().join(format!("akita-release-cache-{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    root
}

#[test]
fn manifest_is_returned_only_under_its_own_bundle_id() {
    let root = scratch("manifest");
    let cache = ReleaseCache::new(&root);
    let id = cache.store_manifest(b"not a manifest").unwrap();
    assert_eq!(id, bundle_id(b"not a manifest"));
    assert_eq!(cache.manifest(&id).unwrap(), b"not a manifest");

    let other = bundle_id(b"another manifest");
    fs::create_dir_all(root.join(other.to_hex())).unwrap();
    fs::write(
        root.join(other.to_hex()).join("manifest"),
        b"not a manifest",
    )
    .unwrap();
    assert!(cache.manifest(&other).is_err());
    fs::remove_dir_all(&root).unwrap();
}

#[test]
fn cached_files_are_checked_against_the_manifest() {
    use akita_types::preprocessing::{file_hash, FileEntry, ReleaseKind, DIGEST_HASH};

    let root = scratch("files");
    let cache = ReleaseCache::new(&root);
    let manifest = ReleaseManifest {
        digest_hash: DIGEST_HASH,
        release_name: "cache-test".into(),
        kind: ReleaseKind::Akita,
        generator_version: "test".into(),
        targets: Vec::new(),
        max_file_part: 1 << 20,
        streams: Vec::new(),
        registry: Vec::new(),
        sets: Vec::new(),
        files: vec![FileEntry {
            name: "catalog/x".into(),
            length: 3,
            file_hash: file_hash("catalog/x", b"abc"),
        }],
    };
    let id = cache.store_manifest(&manifest.to_bytes().unwrap()).unwrap();
    cache.store_file(&id, "catalog/x", b"abz").unwrap();
    assert!(read_release_file(&cache.files(&id), &manifest, "catalog/x", 3).is_err());
    cache.store_file(&id, "catalog/x", b"abc").unwrap();
    assert_eq!(
        read_release_file(&cache.files(&id), &manifest, "catalog/x", 3).unwrap(),
        b"abc"
    );
    assert!(cache.store_file(&id, "../escape", b"x").is_err());
    fs::remove_dir_all(&root).unwrap();
}
