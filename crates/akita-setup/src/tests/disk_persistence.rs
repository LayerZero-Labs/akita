use super::*;
use std::fs;
use std::sync::{LazyLock, Mutex};

static DISK_TEST_ENV_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

fn cleanup_setup_file_shape(max_num_vars: usize, max_num_batched_polys: usize) {
    if let Some(path) = get_prefix_registry_storage_path::<TestF>(&requirements_at(
        max_num_vars,
        max_num_batched_polys,
    )) {
        let _ = fs::remove_file(path);
    }
    if let Ok(path) = get_public_matrix_storage_path::<TestF>(&sample_akita_setup_seed()) {
        let _ = fs::remove_file(path);
    }
}

fn with_test_cache_dir<T>(test_name: &str, f: impl FnOnce() -> T) -> T {
    let _guard = DISK_TEST_ENV_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let cache_root = std::env::temp_dir().join(format!("akita-disk-tests-{test_name}"));
    fs::create_dir_all(&cache_root).unwrap();

    let old_local_app_data = std::env::var_os("LOCALAPPDATA");
    std::env::set_var("LOCALAPPDATA", &cache_root);
    let out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
    match old_local_app_data {
        Some(path) => std::env::set_var("LOCALAPPDATA", path),
        None => std::env::remove_var("LOCALAPPDATA"),
    }
    match out {
        Ok(value) => value,
        Err(payload) => std::panic::resume_unwind(payload),
    }
}

#[test]
fn save_and_load_roundtrips() {
    with_test_cache_dir("roundtrip", || {
        const MAX_VARS: usize = 14;

        cleanup_setup_file_shape(MAX_VARS, 1);

        let prover_setup = new_prover_setup::<TestF>(&requirements_at(MAX_VARS, 1)).unwrap();

        let loaded = load_prover_setup::<TestF>(&requirements_at(MAX_VARS, 1)).unwrap();
        assert_eq!(loaded.expanded, prover_setup.expanded);

        cleanup_setup_file_shape(MAX_VARS, 1);
    });
}

#[test]
fn cache_file_name_stays_below_common_component_limits() {
    let name = prefix_registry_cache_file_name::<TestF>(&requirements_at(16, 4))
        .expect("registry cache name");
    assert!(
        name.len() < 200,
        "setup cache file name should stay comfortably below 255 bytes, got {}: {name}",
        name.len()
    );
}

#[test]
fn cache_file_names_use_current_namespaces() {
    let registry = prefix_registry_cache_file_name::<TestF>(&requirements_at(16, 4))
        .expect("registry cache name");
    assert!(registry.contains("prefix_v4_"), "cache name: {registry}");
    let matrix = public_matrix_cache_file_name::<TestF>(&sample_akita_setup_seed())
        .expect("matrix cache name");
    assert!(matrix.contains("flat_v3_"), "cache name: {matrix}");
}

#[test]
fn config_backed_cache_does_not_apply_generic_setup_decode_limit() {
    let setup_seed = sample_akita_setup_seed();
    let claimed_fields = akita_types::MAX_GENERIC_SETUP_DECODE_FIELD_ELEMENTS + 1;
    let mut bytes = Vec::new();
    setup_seed.serialize_compressed(&mut bytes).unwrap();
    claimed_fields.serialize_compressed(&mut bytes).unwrap();

    let error = deserialize_cached_public_matrix::<TestF>(
        &mut bytes.as_slice(),
        claimed_fields,
        &setup_seed,
    )
    .unwrap_err();
    assert!(
        !matches!(
            error,
            SerializationError::LengthLimitExceeded { max, .. }
                if max == akita_types::MAX_GENERIC_SETUP_DECODE_FIELD_ELEMENTS
        ),
        "config-backed cache decoder reused the generic setup limit"
    );
}

#[test]
fn combined_recursive_prefix_registry_persists_and_serves_each_family() {
    type OneHotRec = akita_config::RecursiveCommitmentConfig<fp128::OneHot>;
    type MultiChunkRec = akita_config::RecursiveCommitmentConfig<fp128::OneHotMultiChunk>;
    // The four-polynomial grouped roots are the rows at which both
    // recursive families carry a setup-prefix slot.
    const MAX_VARS: usize = 32;
    const MAX_POLYS: usize = 4;

    with_test_cache_dir("combined-recursive-prefixes", || {
        // Debug ring-dispatch arithmetic in the prefix commitments needs
        // the same enlarged stack as the backend's commitment fixtures.
        std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn(|| {
                let onehot = SetupRequirements::from_catalog::<OneHotRec>(
                    &akita_config::test_support::workspace_schedule_catalog::<OneHotRec>()
                        .expect("recursive one-hot catalog"),
                    MAX_VARS,
                    MAX_POLYS,
                )
                .expect("recursive one-hot requirements");
                let multichunk = SetupRequirements::from_catalog::<MultiChunkRec>(
                    &akita_config::test_support::workspace_schedule_catalog::<MultiChunkRec>()
                        .expect("recursive multi-chunk catalog"),
                    MAX_VARS,
                    MAX_POLYS,
                )
                .expect("recursive multi-chunk requirements");
                assert!(!onehot.prefix_slot_ids().is_empty());
                assert!(!multichunk.prefix_slot_ids().is_empty());
                assert_ne!(onehot.prefix_slot_ids(), multichunk.prefix_slot_ids());

                // The cache identity is the slot set: independent of union
                // order and of repeated slots, and distinct per set.
                let combined = onehot.clone().union(multichunk.clone()).unwrap();
                let reversed = multichunk.clone().union(onehot.clone()).unwrap();
                let overlapping = combined.clone().union(onehot.clone()).unwrap();
                assert_eq!(combined, reversed);
                assert_eq!(combined, overlapping);
                let key = |requirements: &SetupRequirements<TestF>| {
                    prefix_registry_cache_file_name::<TestF>(requirements).unwrap()
                };
                assert_eq!(key(&combined), key(&reversed));
                assert_eq!(key(&combined), key(&overlapping));
                assert_ne!(key(&combined), key(&onehot));
                assert_ne!(key(&onehot), key(&multichunk));

                let registry_path =
                    get_prefix_registry_storage_path::<TestF>(&combined).expect("registry path");
                let matrix_path =
                    get_public_matrix_storage_path::<TestF>(&sample_akita_setup_seed())
                        .expect("matrix path");
                let remove_cached = || {
                    let _ = fs::remove_file(&registry_path);
                    let _ = fs::remove_file(&matrix_path);
                };
                // Slot ids of the registry on disk, decoded as the loader does.
                let persisted_slot_ids = || {
                    let mut reader =
                        std::io::BufReader::new(fs::File::open(&registry_path).unwrap());
                    akita_cpu_backend::SetupPrefixProverRegistry::<TestF>::deserialize_with_mode(
                        &mut reader,
                        Compress::Yes,
                        Validate::Yes,
                        &(),
                    )
                    .unwrap()
                    .iter()
                    .map(|(id, _)| id.clone())
                    .collect::<Vec<_>>()
                };
                remove_cached();

                let generated = new_prover_setup::<TestF>(&combined).expect("cold setup");
                assert!(matrix_path.exists());
                assert_eq!(persisted_slot_ids(), combined.prefix_slot_ids());

                // Load directly: `new_prover_setup` would hide a failed load by
                // regenerating the same deterministic material.
                let loaded = load_prover_setup::<TestF>(&combined)
                    .expect("complete combined cache must load without regeneration");
                assert_eq!(loaded.expanded, generated.expanded);
                assert_eq!(loaded.prefix_slots, generated.prefix_slots);

                // One backend imports each family's own slots from the decoded
                // registry. Decoded artifacts are not backend-validated, so
                // import recomputes each one and compares it.
                let backend =
                    akita_cpu_backend::CpuBackend::<TestF, TestF>::new(loaded.expanded.clone())
                        .unwrap();
                for family in [&onehot, &multichunk] {
                    let imported = backend
                        .import_setup_prefixes(&loaded.prefix_slots, family.prefix_slot_ids())
                        .expect("family prefixes import from the combined registry");
                    for id in family.prefix_slot_ids() {
                        assert!(imported.get(id).is_some());
                    }
                }

                // A cached registry missing a required slot is rebuilt on load.
                let partial = AkitaProverSetup {
                    expanded: loaded.expanded.clone(),
                    prefix_slots: backend
                        .export_setup_prefixes(onehot.prefix_slot_ids())
                        .unwrap(),
                };
                save_prover_setup::<TestF>(&partial, &combined).unwrap();
                assert_eq!(persisted_slot_ids(), onehot.prefix_slot_ids());
                let repaired = load_prover_setup::<TestF>(&combined)
                    .expect("loader must repair the incomplete combined registry");
                assert_eq!(repaired.prefix_slots, generated.prefix_slots);
                assert_eq!(persisted_slot_ids(), combined.prefix_slot_ids());

                remove_cached();
            })
            .unwrap()
            .join()
            .unwrap();
    });
}

#[test]
fn setup_uses_cache_on_second_call() {
    with_test_cache_dir("second-call", || {
        const MAX_VARS: usize = 14;

        cleanup_setup_file_shape(MAX_VARS, 1);

        let first = new_prover_setup::<TestF>(&requirements_at(MAX_VARS, 1)).unwrap();

        let second = new_prover_setup::<TestF>(&requirements_at(MAX_VARS, 1)).unwrap();

        assert_eq!(first.expanded, second.expanded);

        cleanup_setup_file_shape(MAX_VARS, 1);
    });
}

#[test]
fn larger_public_prefix_covers_smaller_provisioning_request() {
    with_test_cache_dir("covering-prefix", || {
        const LARGE_VARS: usize = 15;
        const SMALL_VARS: usize = 14;

        cleanup_setup_file_shape(LARGE_VARS, 1);
        if let Some(path) =
            get_prefix_registry_storage_path::<TestF>(&requirements_at(SMALL_VARS, 1))
        {
            let _ = fs::remove_file(path);
        }

        let large = new_prover_setup::<TestF>(&requirements_at(LARGE_VARS, 1)).unwrap();
        let large_fields = large.expanded.shared_matrix().num_field_elements();
        let small_required = requirements_at(SMALL_VARS, 1)
            .matrix_capacity()
            .num_field_elements;
        assert!(large_fields >= small_required);

        let covered = new_prover_setup::<TestF>(&requirements_at(SMALL_VARS, 1)).unwrap();
        assert_eq!(
            covered.expanded.shared_matrix().num_field_elements(),
            large_fields
        );
        assert_eq!(
            covered.expanded.descriptor().setup_seed,
            large.expanded.descriptor().setup_seed
        );
        assert_eq!(covered.expanded.descriptor().max_num_vars, SMALL_VARS);
        assert_eq!(covered.expanded.descriptor().max_num_batched_polys, 1);

        cleanup_setup_file_shape(LARGE_VARS, 1);
        if let Some(path) =
            get_prefix_registry_storage_path::<TestF>(&requirements_at(SMALL_VARS, 1))
        {
            let _ = fs::remove_file(path);
        }
    });
}

#[test]
fn concurrent_public_matrix_writers_join_at_largest_prefix() {
    with_test_cache_dir("concurrent-prefix-writers", || {
        const SMALL_VARS: usize = 14;
        const LARGE_VARS: usize = 15;

        cleanup_setup_file_shape(LARGE_VARS, 1);
        if let Some(path) =
            get_prefix_registry_storage_path::<TestF>(&requirements_at(SMALL_VARS, 1))
        {
            let _ = fs::remove_file(path);
        }
        let small = AkitaProverSetup::generate_with_capacity(
            SMALL_VARS,
            1,
            requirements_at(SMALL_VARS, 1).matrix_capacity(),
        )
        .unwrap();
        let large = AkitaProverSetup::generate_with_capacity(
            LARGE_VARS,
            1,
            requirements_at(LARGE_VARS, 1).matrix_capacity(),
        )
        .unwrap();
        let large_fields = large.expanded.shared_matrix().num_field_elements();
        let barrier = Arc::new(std::sync::Barrier::new(3));
        std::thread::scope(|scope| {
            let first_barrier = Arc::clone(&barrier);
            scope.spawn(move || {
                first_barrier.wait();
                save_prover_setup::<TestF>(&small, &requirements_at(SMALL_VARS, 1)).unwrap();
            });
            let second_barrier = Arc::clone(&barrier);
            scope.spawn(move || {
                second_barrier.wait();
                save_prover_setup::<TestF>(&large, &requirements_at(LARGE_VARS, 1)).unwrap();
            });
            barrier.wait();
        });

        let loaded = load_prover_setup::<TestF>(&requirements_at(LARGE_VARS, 1)).unwrap();
        assert_eq!(
            loaded.expanded.shared_matrix().num_field_elements(),
            large_fields
        );

        cleanup_setup_file_shape(LARGE_VARS, 1);
        if let Some(path) =
            get_prefix_registry_storage_path::<TestF>(&requirements_at(SMALL_VARS, 1))
        {
            let _ = fs::remove_file(path);
        }
    });
}

#[test]
fn load_rejects_cached_matrix_that_does_not_match_seed() {
    with_test_cache_dir("corrupt-matrix", || {
        use akita_types::FlatMatrix;

        const MAX_VARS: usize = 14;

        cleanup_setup_file_shape(MAX_VARS, 1);

        let prover_setup = new_prover_setup::<TestF>(&requirements_at(MAX_VARS, 1)).unwrap();
        let total = prover_setup.expanded.shared_matrix().num_field_elements();
        let corrupt = AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
            prover_setup.expanded.descriptor().clone(),
            FlatMatrix::from_flat_data(vec![TestF::zero(); total]),
        );
        let path = get_public_matrix_storage_path::<TestF>(&sample_akita_setup_seed()).unwrap();
        atomic_write_cache(&path, |writer| {
            serialize_public_matrix_cache(&corrupt, writer)
        })
        .unwrap();

        let err = load_prover_setup::<TestF>(&requirements_at(MAX_VARS, 1))
            .expect_err("corrupt cached matrix must be rejected");
        assert!(err
            .to_string()
            .contains("setup shared_matrix does not match public matrix seed"));

        cleanup_setup_file_shape(MAX_VARS, 1);
    });
}

#[test]
fn load_rejects_cached_setup_with_trailing_bytes() {
    with_test_cache_dir("trailing-bytes", || {
        use std::io::Write;

        const MAX_VARS: usize = 14;

        cleanup_setup_file_shape(MAX_VARS, 1);

        new_prover_setup::<TestF>(&requirements_at(MAX_VARS, 1)).unwrap();
        let path = get_public_matrix_storage_path::<TestF>(&sample_akita_setup_seed()).unwrap();
        let mut file = fs::OpenOptions::new().append(true).open(path).unwrap();
        file.write_all(&[0]).unwrap();

        let err = load_prover_setup::<TestF>(&requirements_at(MAX_VARS, 1))
            .expect_err("cache with trailing bytes must be rejected");
        assert!(err.to_string().contains("trailing bytes"));

        cleanup_setup_file_shape(MAX_VARS, 1);
    });
}

#[test]
fn ntt_caches_rebuilt_correctly_from_disk() {
    std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(|| {
            with_test_cache_dir("ntt-rebuild", || {
                use akita_cpu_backend::{CpuBackend, DensePoly, GroupContext};

                const MAX_VARS: usize = 14;

                cleanup_setup_file_shape(MAX_VARS, 1);

                let fresh_setup = new_prover_setup::<TestF>(&requirements_at(MAX_VARS, 1)).unwrap();

                let disk_setup = load_prover_setup::<TestF>(&requirements_at(MAX_VARS, 1)).unwrap();

                let catalog = schedules();
                let poly = DensePoly::<TestF>::from_field_evals(
                    MAX_VARS,
                    vec![TestF::zero(); 1usize << MAX_VARS],
                )
                .unwrap();
                let commit_payload = |setup: &AkitaProverSetup<TestF>| {
                    let backend = CpuBackend::new(setup.expanded.clone()).unwrap();
                    let source = backend.import_source(vec![poly.clone()]).unwrap();
                    backend
                        .commit(
                            &catalog,
                            &source,
                            GroupContext::scheduler_without_precommitted_groups(),
                        )
                        .unwrap()
                        .committed_group
                };

                let fresh_payload = commit_payload(&fresh_setup);
                let disk_payload = commit_payload(&disk_setup);

                assert_eq!(fresh_payload, disk_payload);

                cleanup_setup_file_shape(MAX_VARS, 1);
            });
        })
        .expect("spawn NTT rebuild test")
        .join()
        .expect("NTT rebuild test panicked");
}
