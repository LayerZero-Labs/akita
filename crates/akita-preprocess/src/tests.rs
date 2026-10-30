use super::*;
use akita_types::preprocessing::{
    bundle_id, read_release_file, ReleaseManifest, MAX_FILE_NAME_BYTES,
};
use std::path::PathBuf;

pub(crate) fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

const SEED: &str = "bebafecaefbeadde000000000000000000000000000000000000000000000000";

fn recursive_set(catalog: serde_json::Value) -> serde_json::Value {
    serde_json::json!({
        "name": "fp128-dense-recursive",
        "config": "fp128_dense_recursive",
        "catalog": catalog,
        "num_chunks": 1,
        "num_activated_levels": 0,
        "max_num_vars": 24,
        "max_num_batched_polys": 1
    })
}

/// A small release: one recursive set whose largest row offloads setup, and
/// every optional file.
fn small_description(name: &str) -> ReleaseDescription {
    serde_json::from_value(serde_json::json!({
        "format_version": 1,
        "name": name,
        "kind": "akita",
        "generator_version": "akita-preprocess test",
        "seed": SEED,
        "targets": [RISCV64_TARGET],
        "max_file_part": 4096,
        "hints": true,
        "page_hashes": true,
        "matrix_pages": [[0, 1]],
        "sets": [recursive_set("artifacts/schedules/fp128_dense_recursive.aks".into())]
    }))
    .expect("test description")
}

fn on_large_stack<T: Send + 'static>(test: impl FnOnce() -> T + Send + 'static) -> T {
    std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(test)
        .expect("spawn")
        .join()
        .expect("test thread")
}

#[test]
fn every_role_computes_the_same_manifest_and_writes_only_its_files() {
    on_large_stack(|| {
        let description = small_description("role-test");
        let all = generate(&repo_root(), &description, &Role::All).unwrap();
        for role in [
            Role::Prover,
            Role::Verifier,
            Role::Target(RISCV64_TARGET.into()),
        ] {
            let generated = generate(&repo_root(), &description, &role).unwrap();
            assert_eq!(generated.manifest, all.manifest, "{role:?}");
            assert_eq!(generated.bundle_id, bundle_id(&all.manifest));
            for (name, _) in generated.files.iter() {
                assert!(all.files.get(name).is_some(), "{name}");
            }
        }
        let prover = generate(&repo_root(), &description, &Role::Prover).unwrap();
        assert!(prover
            .files
            .iter()
            .any(|(name, _)| name.starts_with("hints/")));
        assert!(!prover
            .files
            .iter()
            .any(|(name, _)| name.starts_with("terminal-cache/")));
        let manifest = ReleaseManifest::decode(&all.manifest).unwrap();
        assert!(!manifest.registry.is_empty());
        for entry in &manifest.files {
            assert!(entry.name.len() <= MAX_FILE_NAME_BYTES);
            assert_eq!(
                read_release_file(&all.files, &manifest, &entry.name, usize::MAX).unwrap(),
                all.files.get(&entry.name).unwrap()
            );
        }
        assert!(generate(&repo_root(), &description, &Role::Target("other".into())).is_err());
    });
}

#[cfg(feature = "parallel")]
#[test]
fn bundle_id_does_not_depend_on_thread_count() {
    on_large_stack(|| {
        let description = small_description("thread-test");
        let run = |threads: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .stack_size(64 * 1024 * 1024)
                .build()
                .unwrap()
                .install(|| generate(&repo_root(), &description, &Role::All).unwrap())
        };
        let one = run(1);
        let many = run(4);
        assert_eq!(one.bundle_id, many.bundle_id);
        assert_eq!(one.manifest, many.manifest);
    });
}

#[test]
fn generation_ignores_a_poisoned_ambient_cache() {
    on_large_stack(|| {
        let description = small_description("poison-test");
        let honest = generate(&repo_root(), &description, &Role::All).unwrap();
        let cache =
            std::env::temp_dir().join(format!("akita-poisoned-cache-{}", std::process::id()));
        let release = cache.join("akita/releases").join(honest.bundle_id.to_hex());
        for (name, bytes) in honest.files.iter() {
            let path = release.join(name);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            let mut poisoned = bytes.to_vec();
            poisoned[0] ^= 0xff;
            std::fs::write(path, poisoned).unwrap();
        }
        std::fs::write(release.join(MANIFEST_FILE), b"poisoned").unwrap();
        std::env::set_var("LOCALAPPDATA", &cache);
        let regenerated = generate(&repo_root(), &description, &Role::All).unwrap();
        std::env::remove_var("LOCALAPPDATA");
        std::fs::remove_dir_all(&cache).unwrap();
        assert_eq!(regenerated.manifest, honest.manifest);
    });
}

#[test]
fn written_releases_split_large_files_and_audit_rejects_a_changed_file() {
    on_large_stack(|| {
        let root = std::env::temp_dir().join(format!("akita-audit-{}", std::process::id()));
        let params = root.join("params").join("audit-test");
        let files = root.join("files");
        std::fs::create_dir_all(&params).unwrap();
        let catalog = repo_root().join("artifacts/schedules/fp128_dense_recursive.aks");
        std::fs::write(
            params.join(DESCRIPTION_FILE),
            serde_json::to_vec(&serde_json::json!({
                "format_version": 1,
                "name": "audit-test",
                "kind": "akita",
                "generator_version": "akita-preprocess test",
                "seed": SEED,
                "max_file_part": 4096,
                "hints": true,
                "sets": [recursive_set(catalog.to_string_lossy().into())]
            }))
            .unwrap(),
        )
        .unwrap();
        let stored = ReleaseDescription::read(&params.join(DESCRIPTION_FILE)).unwrap();
        let generated = generate(&repo_root(), &stored, &Role::All).unwrap();
        std::fs::write(params.join(MANIFEST_FILE), &generated.manifest).unwrap();
        write_release(&files, &generated, stored.max_file_part).unwrap();
        let split = generated
            .files
            .iter()
            .find(|(_, bytes)| bytes.len() > 4096)
            .map(|(name, _)| name.to_owned())
            .expect("a file larger than one part");
        assert!(files.join(format!("{split}.part-1")).is_file());
        assert_eq!(
            audit_release_files(&repo_root(), &params, &files).unwrap(),
            generated.bundle_id
        );

        let (registry, bytes) = generated
            .files
            .iter()
            .find(|(name, _)| name.starts_with("registry/"))
            .map(|(name, bytes)| (name.to_owned(), bytes.to_vec()))
            .unwrap();
        let mut changed = bytes;
        changed[0] ^= 1;
        std::fs::write(files.join(&registry), changed).unwrap();
        assert!(audit_release_files(&repo_root(), &params, &files).is_err());

        std::fs::write(params.join(MANIFEST_FILE), b"stale").unwrap();
        assert!(check_release(&repo_root(), &params).is_err());
        std::fs::remove_dir_all(&root).unwrap();
    });
}

#[test]
fn checked_in_releases_regenerate() {
    on_large_stack(|| {
        let releases = release_dirs(&repo_root().join("params")).unwrap();
        assert!(
            !releases.is_empty(),
            "the repository holds at least one release"
        );
        for release in releases {
            check_release(&repo_root(), &release).unwrap();
        }
    });
}
