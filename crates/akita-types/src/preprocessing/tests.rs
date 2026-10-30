use super::*;
use crate::AkitaSetupSeed;
use jolt_field::Prime128OffsetA7F7;

fn stream() -> StreamIdentity {
    StreamIdentity::for_field::<Prime128OffsetA7F7>(&AkitaSetupSeed::shake256_paged_v1([7; 32]))
        .unwrap()
}

fn record(name: &str, stream_id: StreamId) -> SetRecord {
    SetRecord {
        name: name.to_owned(),
        config_family: "fp128_onehot".to_owned(),
        decomposition: vec![3, 0, 0, 0],
        sis_modulus_profile_tag: 2,
        sis_table_digest: [1; 32],
        sis_l2_table_digest: [2; 32],
        algorithm_identity: AlgorithmIdentity::compiled(
            akita_params::sis::SisModulusProfileId::Q128OffsetA7F7,
        ),
        stream_id,
        num_chunks: 1,
        num_activated_levels: 0,
        max_num_vars: 20,
        max_num_batched_polys: 1,
        catalog_digest: [3; 32],
        catalog_profiles: vec![
            ProfileKey::from_bytes([4; 32]),
            ProfileKey::from_bytes([5; 32]),
        ],
        admits_program_rows: true,
        universe: vec![
            ProfileKey::from_bytes([4; 32]),
            ProfileKey::from_bytes([5; 32]),
            ProfileKey::from_bytes([6; 32]),
        ],
    }
}

fn manifest() -> ReleaseManifest {
    let identity = stream();
    let stream_id = identity.stream_id();
    let mut files = vec![
        FileEntry {
            name: "catalog/b-set".to_owned(),
            length: 3,
            file_hash: file_hash("catalog/b-set", b"abc"),
        },
        FileEntry {
            name: registry_file_name(&stream_id, &ProfileKey::from_bytes([4; 32])),
            length: 1,
            file_hash: file_hash("x", b"x"),
        },
    ];
    files.sort_by(|left, right| left.name.cmp(&right.name));
    ReleaseManifest {
        digest_hash: DIGEST_HASH,
        release_name: "test-release".to_owned(),
        kind: ReleaseKind::Akita,
        generator_version: "akita-preprocess 0.1.0".to_owned(),
        targets: vec!["riscv64".to_owned()],
        max_file_part: 1 << 30,
        streams: vec![StreamEntry {
            identity,
            element_count: 4096,
        }],
        registry: vec![
            RegistryEntry {
                stream_id,
                profile_key: ProfileKey::from_bytes([4; 32]),
                prefix_commitment_digest: PrefixCommitmentDigest::from_bytes([9; 32]),
            },
            RegistryEntry {
                stream_id,
                profile_key: ProfileKey::from_bytes([5; 32]),
                prefix_commitment_digest: PrefixCommitmentDigest::from_bytes([8; 32]),
            },
        ],
        sets: vec![record("a-set", stream_id), record("b-set", stream_id)],
        files,
    }
}

#[test]
fn manifest_round_trips_canonically() {
    let manifest = manifest();
    let bytes = manifest.to_bytes().unwrap();
    assert_eq!(ReleaseManifest::decode(&bytes).unwrap(), manifest);
    assert_eq!(StreamIdentity::ENCODED_LEN, stream().to_bytes().len());
    assert_eq!(
        AlgorithmIdentity::ENCODED_LEN,
        manifest.sets[0].algorithm_identity.to_bytes().len()
    );
}

#[test]
fn manifest_decoding_rejects_every_truncation_and_extension() {
    let bytes = manifest().to_bytes().unwrap();
    for len in 0..bytes.len() {
        assert!(
            ReleaseManifest::decode(&bytes[..len]).is_err(),
            "prefix {len}"
        );
    }
    let mut extended = bytes.clone();
    extended.push(0);
    assert!(ReleaseManifest::decode(&extended).is_err());
}

#[test]
fn manifest_decoding_rejects_single_byte_changes_or_reports_another_manifest() {
    let original = manifest();
    let bytes = original.to_bytes().unwrap();
    for index in 0..bytes.len() {
        let mut changed = bytes.clone();
        changed[index] ^= 0x01;
        if let Ok(decoded) = ReleaseManifest::decode(&changed) {
            assert_ne!(decoded, original, "byte {index} changed nothing");
        }
    }
}

#[test]
fn unsorted_or_duplicate_entries_are_rejected() {
    let mut reordered = manifest();
    reordered.sets.swap(0, 1);
    assert!(reordered.to_bytes().is_err());

    let mut duplicated = manifest();
    duplicated.files.push(duplicated.files[1].clone());
    assert!(duplicated.to_bytes().is_err());

    let mut registry = manifest();
    registry.registry.swap(0, 1);
    assert!(registry.to_bytes().is_err());

    let mut universe = manifest();
    universe.sets[0].universe.swap(0, 1);
    assert!(universe.to_bytes().is_err());

    let mut stray = manifest();
    stray.sets[0].admits_program_rows = false;
    assert!(stray.to_bytes().is_err());
}

#[test]
fn file_names_reject_path_traversal() {
    for name in ["", "..", "a/../b", "/a", "a//b", "a/", "A", "a b", "./a"] {
        assert!(validate_file_name(name).is_err(), "{name:?}");
    }
    validate_file_name("terminal-cache/0a/1b/riscv64/4096").unwrap();
}

const ANCHOR_ENTRIES: &[AnchorEntry] = &[AnchorEntry::new(
    "0000000000000000000000000000000000000000000000000000000000000001",
    "a-set",
)];
const ANCHOR: TrustAnchor = TrustAnchor::new(ANCHOR_ENTRIES);

#[test]
fn opening_checks_the_anchor_before_decoding() {
    let error = open_anchored_manifest(&ANCHOR, ReleaseKind::Akita, b"not a manifest", &["a-set"])
        .unwrap_err()
        .to_string();
    assert!(error.contains("trust anchor"), "{error}");

    let bytes = manifest().to_bytes().unwrap();
    let id = bundle_id(&bytes);
    let leaked: &'static str = Box::leak(id.to_hex().into_boxed_str());
    let entries: &'static [AnchorEntry] = Box::leak(Box::new([
        AnchorEntry::new(leaked, "a-set"),
        AnchorEntry::new(leaked, "b-set"),
    ]));
    let anchor = TrustAnchor::new(entries);
    let opened = open_anchored_manifest(&anchor, ReleaseKind::Akita, &bytes, &["a-set"]).unwrap();
    assert_eq!(opened.bundle_id(), id);
    assert!(opened.set("a-set").is_ok());
    assert!(opened.set("b-set").is_err(), "unrequested set");
    assert!(open_anchored_manifest(&anchor, ReleaseKind::Jolt, &bytes, &["a-set"]).is_err());
    assert!(open_anchored_manifest(&anchor, ReleaseKind::Akita, &bytes, &["c-set"]).is_err());
}

#[test]
fn opening_rejects_another_digest_hash() {
    let mut manifest = manifest();
    manifest.digest_hash = match DIGEST_HASH {
        DigestHash::Sha3_256 => DigestHash::Keccak256,
        DigestHash::Keccak256 => DigestHash::Sha3_256,
    };
    let bytes = manifest.to_bytes().unwrap();
    let leaked: &'static str = Box::leak(bundle_id(&bytes).to_hex().into_boxed_str());
    let entries: &'static [AnchorEntry] = Box::leak(Box::new([AnchorEntry::new(leaked, "a-set")]));
    let error = open_anchored_manifest(
        &TrustAnchor::new(entries),
        ReleaseKind::Akita,
        &bytes,
        &["a-set"],
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("digest hash"), "{error}");
}

#[test]
fn authenticated_reads_reject_changed_missing_and_oversized_files() {
    let manifest = manifest();
    let mut files = MemoryReleaseFiles::new();
    files.insert("catalog/b-set", b"abc".to_vec());
    assert_eq!(
        read_release_file(&files, &manifest, "catalog/b-set", 3).unwrap(),
        b"abc"
    );
    assert!(read_release_file(&files, &manifest, "catalog/b-set", 2).is_err());
    assert!(read_release_file(&files, &manifest, "catalog/a-set", 3).is_err());
    files.insert("catalog/b-set", b"abz".to_vec());
    assert!(read_release_file(&files, &manifest, "catalog/b-set", 3).is_err());
    files.insert("catalog/b-set", b"abcd".to_vec());
    assert!(read_release_file(&files, &manifest, "catalog/b-set", 4).is_err());
}

#[test]
fn directory_reads_join_parts() {
    let root = std::env::temp_dir().join(format!(
        "akita-release-parts-{}-{}",
        std::process::id(),
        line!()
    ));
    std::fs::create_dir_all(root.join("catalog")).unwrap();
    std::fs::write(root.join("catalog/b-set.part-0"), b"ab").unwrap();
    std::fs::write(root.join("catalog/b-set.part-1"), b"c").unwrap();
    let files = DirectoryReleaseFiles::new(&root);
    assert_eq!(
        read_release_file(&files, &manifest(), "catalog/b-set", 3).unwrap(),
        b"abc"
    );
    std::fs::write(root.join("catalog/b-set.part-2"), b"d").unwrap();
    assert!(read_release_file(&files, &manifest(), "catalog/b-set", 3).is_err());
    std::fs::remove_dir_all(&root).unwrap();
}
