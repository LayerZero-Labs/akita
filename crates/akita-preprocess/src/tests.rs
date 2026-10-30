use super::*;
use akita_types::preprocessing::{
    part_file_name, read_release_file, DirectoryReleaseFiles, MAX_FILE_NAME_BYTES,
};
use std::collections::BTreeSet;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// A small release: one recursive set whose largest row offloads setup,
/// hints, a guest target, and parts small enough to split files.
fn small_description(name: &str) -> ReleaseDescription {
    serde_json::from_value(serde_json::json!({
        "format_version": 1,
        "name": name,
        "kind": "akita",
        "generator_version": "akita-preprocess test",
        "seed": "bebafecaefbeadde000000000000000000000000000000000000000000000000",
        "targets": [RISCV64_TARGET],
        "max_file_part": 4096,
        "hints": true,
        "sets": [{
            "name": "fp128-dense-recursive",
            "config": "fp128_dense_recursive",
            "catalog": "artifacts/schedules/fp128_dense_recursive.aks",
            "num_chunks": 1,
            "num_activated_levels": 0,
            "max_num_vars": 24,
            "max_num_batched_polys": 1
        }]
    }))
    .expect("test description")
}

pub(crate) fn on_large_stack<T: Send + 'static>(test: impl FnOnce() -> T + Send + 'static) -> T {
    std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(test)
        .expect("spawn")
        .join()
        .expect("test thread")
}

#[test]
fn each_role_writes_only_its_files() {
    on_large_stack(|| {
        let generated = generate(&repo_root(), &small_description("role-test")).unwrap();
        let manifest = ReleaseManifest::decode(&generated.manifest).unwrap();
        assert!(!manifest.registry.is_empty());
        let root = std::env::temp_dir().join(format!("akita-roles-{}", std::process::id()));
        // The manifest files a role's directory holds, each read back and
        // checked against the manifest.
        let written = |role: &Role, dir: &str| {
            let dir = root.join(dir);
            write_release(&dir, &generated, role).unwrap();
            let files = DirectoryReleaseFiles::new(&dir);
            manifest
                .files
                .iter()
                .filter(|entry| {
                    read_release_file(&files, &manifest, &entry.name, usize::MAX).is_ok()
                })
                .map(|entry| entry.name.clone())
                .collect::<BTreeSet<_>>()
        };
        let has =
            |names: &BTreeSet<String>, top: &str| names.iter().any(|name| name.starts_with(top));

        assert_eq!(written(&Role::All, "all").len(), manifest.files.len());
        for entry in &manifest.files {
            assert!(entry.name.len() <= MAX_FILE_NAME_BYTES, "{}", entry.name);
        }
        let (split, _) = generated
            .files
            .iter()
            .find(|(_, bytes)| bytes.len() > 4096)
            .expect("a file larger than one part");
        assert!(root.join("all").join(part_file_name(split, 1)).is_file());

        let prover = written(&Role::Prover, "prover");
        assert!(has(&prover, "hints/"));
        assert!(!has(&prover, "terminal-cache/"));
        let target = written(&Role::Target(RISCV64_TARGET.into()), "target");
        assert!(has(&target, "terminal-cache/") && has(&target, "catalog-view/"));
        assert!(!has(&target, "hints/") && !has(&target, "catalog/"));
        assert!(write_release(
            &root.join("other"),
            &generated,
            &Role::Target("other".into())
        )
        .is_err());
        std::fs::remove_dir_all(&root).unwrap();
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
                .install(|| generate(&repo_root(), &description).unwrap())
        };
        assert_eq!(run(1).manifest, run(4).manifest);
    });
}

#[test]
fn generation_ignores_a_poisoned_ambient_cache() {
    on_large_stack(|| {
        let description = small_description("poison-test");
        let honest = generate(&repo_root(), &description).unwrap();
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
        let regenerated = generate(&repo_root(), &description).unwrap();
        std::env::remove_var("LOCALAPPDATA");
        std::fs::remove_dir_all(&cache).unwrap();
        assert_eq!(regenerated.manifest, honest.manifest);
    });
}
