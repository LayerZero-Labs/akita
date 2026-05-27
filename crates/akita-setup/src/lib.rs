//! Config-backed prover setup construction.

use akita_config::CommitmentConfig;
use akita_field::fields::wide::HasWide;
use akita_field::{AkitaError, CanonicalField, FieldCore, RandomSampling};
use akita_prover::AkitaProverSetup;
use akita_serialization::Valid;
#[cfg(feature = "disk-persistence")]
use akita_serialization::{
    AkitaDeserialize, AkitaSerialize, Compress, SerializationError, Validate,
};
#[cfg(any(feature = "disk-persistence", test))]
use akita_types::AkitaExpandedSetup;
#[cfg(test)]
use akita_types::AkitaVerifierSetup;
#[cfg(feature = "disk-persistence")]
use akita_types::{
    detect_field_modulus, planned_schedule_key_from_schedule, validate_public_matrix_matches_seed,
    AkitaScheduleLookupKey, AkitaSetupSeed, FlatMatrix,
};
#[cfg(feature = "disk-persistence")]
use std::fs;
#[cfg(feature = "disk-persistence")]
use std::io::Read;
#[cfg(feature = "disk-persistence")]
use std::path::PathBuf;
/// Construct prover setup from a root commitment config.
///
/// `akita-config` owns setup sizing policy; this crate owns optional disk
/// persistence; `akita-prover` owns the concrete setup artifact and
/// matrix expansion.
///
/// # Errors
///
/// Returns an error if `Cfg::D != D`, the requested setup capacity is invalid,
/// or setup expansion fails.
#[tracing::instrument(skip_all, name = "new_prover_setup")]
pub fn new_prover_setup<F, const D: usize, Cfg>(
    max_num_vars: usize,
    max_num_batched_polys: usize,
    max_num_points: usize,
) -> Result<AkitaProverSetup<F, D>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling + HasWide + Valid,
    Cfg: CommitmentConfig<Field = F>,
{
    if D != Cfg::D {
        return Err(AkitaError::InvalidSetup(format!(
            "const D={D} mismatches config D={}",
            Cfg::D
        )));
    }
    if max_num_batched_polys == 0 {
        return Err(AkitaError::InvalidSetup(
            "max_num_batched_polys must be at least 1".to_string(),
        ));
    }
    if max_num_points == 0 {
        return Err(AkitaError::InvalidSetup(
            "max_num_points must be at least 1".to_string(),
        ));
    }
    let (max_rows, max_stride) =
        Cfg::max_setup_matrix_size(max_num_vars, max_num_batched_polys, max_num_points)?;

    #[cfg(feature = "disk-persistence")]
    {
        let max_total = max_rows
            .checked_mul(max_stride)
            .ok_or_else(|| AkitaError::InvalidSetup("conservative total overflow".to_string()))?;
        match load_expanded_setup::<F, D, Cfg>(max_num_vars, max_num_batched_polys, max_num_points)
        {
            Ok(expanded) => {
                // A cached setup is acceptable only if its physical
                // backing is large enough *and* its recorded
                // `max_stride` matches (or exceeds) what the current
                // request needs. For configs where `max_rows` can vary
                // inversely with `max_stride`, a smaller cached stride
                // would cause `ring_view` to interpret rows/columns with
                // the wrong stride — the total-elements check alone is
                // insufficient.
                let cached_total = expanded.shared_matrix().total_ring_elements_at::<D>()?;
                let cached_stride = expanded.seed().max_stride;
                let cached_points = expanded.seed().max_num_points;
                if cached_total >= max_total
                    && cached_stride >= max_stride
                    && cached_points >= max_num_points
                {
                    tracing::info!("Loaded setup from disk; backend preparation is explicit");
                    return AkitaProverSetup::from_seed_validated_expanded(expanded);
                }
                if let Some(storage_path) =
                    get_storage_path::<Cfg>(max_num_vars, max_num_batched_polys, max_num_points)
                {
                    let _ = fs::remove_file(&storage_path);
                    tracing::warn!(
                            "Rejected cached setup from {}: have (total={cached_total}, stride={cached_stride}, points={cached_points}), need (total>={max_total}, stride>={max_stride}, points>={max_num_points}); regenerating",
                            storage_path.display()
                        );
                } else {
                    tracing::warn!(
                            "Rejected cached setup: have (total={cached_total}, stride={cached_stride}, points={cached_points}), need (total>={max_total}, stride>={max_stride}, points>={max_num_points}); regenerating"
                        );
                }
            }
            Err(e) => {
                if let Some(storage_path) =
                    get_storage_path::<Cfg>(max_num_vars, max_num_batched_polys, max_num_points)
                {
                    let _ = fs::remove_file(&storage_path);
                    tracing::warn!(
                        "Failed to load cached setup from {}: {e}; regenerating",
                        storage_path.display()
                    );
                } else {
                    tracing::warn!("Failed to load cached setup: {e}; regenerating");
                }
            }
        }
    }

    let setup = AkitaProverSetup::generate_with_capacity(
        max_num_vars,
        max_num_batched_polys,
        max_num_points,
        max_rows,
        max_stride,
    )?;

    #[cfg(feature = "disk-persistence")]
    save_expanded_setup::<F, Cfg>(
        &setup.expanded,
        max_num_vars,
        max_num_batched_polys,
        max_num_points,
    );

    Ok(setup)
}

// ---------------------------------------------------------------------------
// Disk persistence
// ---------------------------------------------------------------------------

#[cfg(feature = "disk-persistence")]
fn cache_file_name<Cfg: CommitmentConfig>(
    max_num_vars: usize,
    max_num_batched_polys: usize,
    max_num_points: usize,
) -> String {
    let envelope = Cfg::envelope(max_num_vars);
    let family = std::any::type_name::<Cfg>()
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '_' })
        .collect::<String>();
    let schedule_lookup_key = AkitaScheduleLookupKey::new(
        max_num_vars,
        max_num_batched_polys,
        max_num_batched_polys,
        max_num_points,
    );
    // Fingerprint the resolved schedule shape so cached setup files get
    // invalidated when the planner's per-level (log_basis, level_count)
    // outputs change for the same lookup key.
    let raw_schedule = match Cfg::schedule_plan(schedule_lookup_key) {
        Ok(Some(plan)) => planned_schedule_key_from_schedule(schedule_lookup_key, &plan),
        _ => format!(
            "miss_nv{}_g{}_t{}_w{}_z{}",
            schedule_lookup_key.num_vars,
            schedule_lookup_key.num_points,
            schedule_lookup_key.num_t_vectors,
            schedule_lookup_key.num_w_vectors,
            schedule_lookup_key.num_z_vectors,
        ),
    };
    let schedule = raw_schedule
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '_' })
        .collect::<String>();
    let modulus = detect_field_modulus::<Cfg::Field>();
    format!(
        "akita_q{modulus:032x}_{family}_sched_{schedule}_d{}_na{}_nb{}_nd{}_nv{max_num_vars}_batch{max_num_batched_polys}_pts{max_num_points}.setup",
        Cfg::D,
        envelope.max_n_a,
        envelope.max_n_b,
        envelope.max_n_d,
    )
}

#[cfg(feature = "disk-persistence")]
pub(crate) fn get_storage_path<Cfg: CommitmentConfig>(
    max_num_vars: usize,
    max_num_batched_polys: usize,
    max_num_points: usize,
) -> Option<PathBuf> {
    let cache_directory = if let Ok(local_app_data) = std::env::var("LOCALAPPDATA") {
        Some(PathBuf::from(local_app_data))
    } else if let Ok(home) = std::env::var("HOME") {
        let mut path = PathBuf::from(&home);
        let macos_cache = {
            let mut test_path = PathBuf::from(&home);
            test_path.push("Library");
            test_path.push("Caches");
            test_path.exists()
        };
        if macos_cache {
            path.push("Library");
            path.push("Caches");
        } else {
            path.push(".cache");
        }
        Some(path)
    } else {
        None
    };

    cache_directory.map(|mut path| {
        path.push("akita");
        path.push(cache_file_name::<Cfg>(
            max_num_vars,
            max_num_batched_polys,
            max_num_points,
        ));
        path
    })
}

#[cfg(feature = "disk-persistence")]
fn save_expanded_setup<F: FieldCore + CanonicalField, Cfg: CommitmentConfig<Field = F>>(
    setup: &AkitaExpandedSetup<F>,
    max_num_vars: usize,
    max_num_batched_polys: usize,
    max_num_points: usize,
) {
    let Some(storage_path) =
        get_storage_path::<Cfg>(max_num_vars, max_num_batched_polys, max_num_points)
    else {
        tracing::warn!("Could not determine storage directory; skipping setup save");
        return;
    };

    if let Some(parent) = storage_path.parent() {
        if let Err(e) = fs::create_dir_all(parent) {
            tracing::warn!(
                "Failed to create setup cache directory {}: {e}",
                parent.display()
            );
            return;
        }
    }

    tracing::info!("Saving setup to {}", storage_path.display());

    let file = match fs::File::create(&storage_path) {
        Ok(file) => file,
        Err(e) => {
            tracing::warn!(
                "Failed to create setup cache file {}: {e}",
                storage_path.display()
            );
            return;
        }
    };
    let mut writer = std::io::BufWriter::new(file);

    if let Err(e) = setup.serialize_compressed(&mut writer) {
        tracing::warn!(
            "Failed to serialize setup cache {}: {e}",
            storage_path.display()
        );
        let _ = fs::remove_file(&storage_path);
        return;
    }

    tracing::info!("Successfully saved setup to disk");
}

#[cfg(feature = "disk-persistence")]
pub(crate) fn load_expanded_setup<
    F: FieldCore + Valid + CanonicalField + RandomSampling,
    const D: usize,
    Cfg: CommitmentConfig<Field = F>,
>(
    max_num_vars: usize,
    max_num_batched_polys: usize,
    max_num_points: usize,
) -> Result<AkitaExpandedSetup<F>, AkitaError> {
    let storage_path = get_storage_path::<Cfg>(max_num_vars, max_num_batched_polys, max_num_points)
        .ok_or_else(|| {
            AkitaError::InvalidSetup("Failed to determine storage directory".to_string())
        })?;

    if !storage_path.exists() {
        return Err(AkitaError::InvalidSetup(format!(
            "Setup file not found at {}",
            storage_path.display()
        )));
    }

    tracing::info!("Loading setup from {}", storage_path.display());

    let file = fs::File::open(&storage_path)
        .map_err(|e| AkitaError::InvalidSetup(format!("Failed to open setup file: {e}")))?;
    let mut reader = std::io::BufReader::new(file);

    // Disk cache load first validates the byte structure and field elements,
    // then `validate_cached_matrix` verifies the seed-derived matrix content.
    let setup = deserialize_cached_setup::<F, D, Cfg>(
        &mut reader,
        max_num_vars,
        max_num_batched_polys,
        max_num_points,
    )
    .map_err(|e| AkitaError::InvalidSetup(format!("Failed to deserialize setup: {e}")))?;
    let mut trailing = [0u8; 1];
    if reader
        .read(&mut trailing)
        .map_err(|e| AkitaError::InvalidSetup(format!("Failed to check setup EOF: {e}")))?
        != 0
    {
        return Err(AkitaError::InvalidSetup(format!(
            "cached setup has trailing bytes starting with 0x{:02x}",
            trailing[0]
        )));
    }
    validate_cached_matrix::<F, D>(&setup)?;

    tracing::info!(
        "Loaded setup for max_num_vars={max_num_vars}, max_num_batched_polys={max_num_batched_polys}, max_num_points={max_num_points}"
    );
    Ok(setup)
}

#[cfg(feature = "disk-persistence")]
fn deserialize_cached_setup<
    F: FieldCore + Valid + AkitaDeserialize<Context = ()>,
    const D: usize,
    Cfg: CommitmentConfig<Field = F>,
>(
    reader: &mut impl Read,
    expected_max_num_vars: usize,
    expected_max_num_batched_polys: usize,
    expected_max_num_points: usize,
) -> Result<AkitaExpandedSetup<F>, SerializationError> {
    let seed =
        AkitaSetupSeed::deserialize_with_mode(&mut *reader, Compress::Yes, Validate::Yes, &())?;
    if seed.gen_ring_dim != D {
        return Err(SerializationError::InvalidData(format!(
            "cached setup ring dimension {} does not match config D={D}",
            seed.gen_ring_dim
        )));
    }
    if seed.max_num_vars != expected_max_num_vars
        || seed.max_num_batched_polys != expected_max_num_batched_polys
        || seed.max_num_points != expected_max_num_points
    {
        return Err(SerializationError::InvalidData(
            "cached setup seed capacity does not match cache key".to_string(),
        ));
    }
    let (expected_rows, expected_stride) = Cfg::max_setup_matrix_size(
        expected_max_num_vars,
        expected_max_num_batched_polys,
        expected_max_num_points,
    )
    .map_err(|err| {
        SerializationError::InvalidData(format!("cached setup expected shape failed: {err}"))
    })?;
    let expected_total = expected_rows.checked_mul(expected_stride).ok_or_else(|| {
        SerializationError::InvalidData("cached setup expected matrix size overflow".to_string())
    })?;
    if seed.max_stride != expected_stride || seed.total_ring_elements != expected_total {
        return Err(SerializationError::InvalidData(
            "cached setup seed matrix shape does not match cache key".to_string(),
        ));
    }
    let shared_matrix = FlatMatrix::<F>::deserialize_with_expected_shape(
        &mut *reader,
        Compress::Yes,
        Validate::Yes,
        seed.total_ring_elements,
        seed.gen_ring_dim,
        seed.matrix_field_elements()?,
    )?;
    Ok(AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(seed, shared_matrix))
}

#[cfg(feature = "disk-persistence")]
fn validate_cached_matrix<
    F: FieldCore + CanonicalField + RandomSampling + Valid,
    const D: usize,
>(
    setup: &AkitaExpandedSetup<F>,
) -> Result<(), AkitaError> {
    if setup.shared_matrix().gen_ring_dim() != D {
        return Err(AkitaError::InvalidSetup(format!(
            "cached setup ring dimension {} does not match config D={D}",
            setup.shared_matrix().gen_ring_dim()
        )));
    }
    validate_public_matrix_matches_seed(setup.shared_matrix(), setup.seed())
        .map_err(|e| AkitaError::InvalidSetup(format!("cached setup matrix validation: {e}")))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_config::proof_optimized::fp128;
    use akita_serialization::{AkitaDeserialize, AkitaSerialize};
    use std::sync::Arc;

    type Cfg = fp128::D64Full;
    type TestF = fp128::Field;
    const TEST_D: usize = 64;

    #[test]
    fn expanded_setup_roundtrips_and_derives_same_verifier() {
        let prover_setup = new_prover_setup::<TestF, TEST_D, Cfg>(10, 3, 1).unwrap();
        let verifier_setup = AkitaVerifierSetup {
            expanded: Arc::clone(&prover_setup.expanded),
        };

        let mut bytes = Vec::new();
        prover_setup
            .expanded
            .serialize_compressed(&mut bytes)
            .unwrap();
        let decoded = AkitaExpandedSetup::<TestF>::deserialize_compressed(&bytes[..], &()).unwrap();

        assert_eq!(decoded, prover_setup.expanded.as_ref().clone());
        assert_eq!(decoded.seed().max_num_batched_polys, 3);

        let derived_verifier = AkitaVerifierSetup {
            expanded: Arc::new(decoded.clone()),
        };
        assert_eq!(derived_verifier, verifier_setup);
    }

    #[test]
    fn setup_accepts_field_coupled_presets() {
        // D128Full has no schedule table at all; wrap in `PlannerCfg` so
        // setup-matrix sizing falls through to DP. D32Full has a singleton
        // table but the (max_num_vars=12, polys=1, points=1) iteration is a
        // table hit so the inner Cfg suffices without DP.
        new_prover_setup::<fp128::Field, 128, akita_planner::test_utils::PlannerCfg<fp128::D128Full>>(12, 1, 1)
            .expect("default fp128 D=128 preset should accept the fp128 field");
        new_prover_setup::<fp128::Field, 32, fp128::D32Full>(12, 1, 1)
            .expect("small-D fp128 preset should accept the default field");
    }

    #[cfg(feature = "disk-persistence")]
    mod disk_persistence {
        use super::*;
        use std::fs;
        use std::sync::{LazyLock, Mutex};

        static DISK_TEST_ENV_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

        fn cleanup_setup_file_shape(
            max_num_vars: usize,
            max_num_batched_polys: usize,
            max_num_points: usize,
        ) {
            if let Some(path) =
                get_storage_path::<Cfg>(max_num_vars, max_num_batched_polys, max_num_points)
            {
                let _ = fs::remove_file(path);
            }
        }

        fn cleanup_setup_file(max_num_vars: usize) {
            cleanup_setup_file_shape(max_num_vars, 1, 1);
        }

        fn with_test_cache_dir<T>(test_name: &str, f: impl FnOnce() -> T) -> T {
            let _guard = DISK_TEST_ENV_LOCK.lock().unwrap();
            let cache_root = std::env::temp_dir().join(format!("akita-disk-tests-{test_name}"));
            fs::create_dir_all(&cache_root).unwrap();

            let old_local_app_data = std::env::var_os("LOCALAPPDATA");
            std::env::set_var("LOCALAPPDATA", &cache_root);
            let out = f();
            match old_local_app_data {
                Some(path) => std::env::set_var("LOCALAPPDATA", path),
                None => std::env::remove_var("LOCALAPPDATA"),
            }
            out
        }

        #[test]
        fn save_and_load_roundtrips() {
            with_test_cache_dir("roundtrip", || {
                const MAX_VARS: usize = 12;

                cleanup_setup_file(MAX_VARS);

                let prover_setup = new_prover_setup::<TestF, TEST_D, Cfg>(MAX_VARS, 1, 1).unwrap();

                let loaded = load_expanded_setup::<TestF, TEST_D, Cfg>(MAX_VARS, 1, 1).unwrap();
                assert_eq!(loaded, prover_setup.expanded.as_ref().clone());

                cleanup_setup_file(MAX_VARS);
            });
        }

        #[test]
        fn setup_uses_cache_on_second_call() {
            with_test_cache_dir("second-call", || {
                const MAX_VARS: usize = 13;

                cleanup_setup_file(MAX_VARS);

                let first = new_prover_setup::<TestF, TEST_D, Cfg>(MAX_VARS, 1, 1).unwrap();

                let second = new_prover_setup::<TestF, TEST_D, Cfg>(MAX_VARS, 1, 1).unwrap();

                assert_eq!(first.expanded, second.expanded);

                cleanup_setup_file(MAX_VARS);
            });
        }

        #[test]
        fn load_rejects_cached_matrix_that_does_not_match_seed() {
            with_test_cache_dir("corrupt-matrix", || {
                use akita_types::FlatMatrix;

                const MAX_VARS: usize = 13;

                cleanup_setup_file(MAX_VARS);

                let prover_setup = new_prover_setup::<TestF, TEST_D, Cfg>(MAX_VARS, 1, 1).unwrap();
                let total = prover_setup.expanded.shared_matrix().total_ring_elements();
                let corrupt = AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
                    prover_setup.expanded.seed().clone(),
                    FlatMatrix::from_flat_data(vec![TestF::zero(); total * TEST_D], TEST_D),
                );
                save_expanded_setup::<TestF, Cfg>(&corrupt, MAX_VARS, 1, 1);

                let err = load_expanded_setup::<TestF, TEST_D, Cfg>(MAX_VARS, 1, 1)
                    .expect_err("corrupt cached matrix must be rejected");
                assert!(err
                    .to_string()
                    .contains("setup shared_matrix does not match public matrix seed"));

                cleanup_setup_file(MAX_VARS);
            });
        }

        #[test]
        fn load_rejects_cached_setup_with_trailing_bytes() {
            with_test_cache_dir("trailing-bytes", || {
                use std::io::Write;

                const MAX_VARS: usize = 13;

                cleanup_setup_file(MAX_VARS);

                new_prover_setup::<TestF, TEST_D, Cfg>(MAX_VARS, 1, 1).unwrap();
                let path = get_storage_path::<Cfg>(MAX_VARS, 1, 1).unwrap();
                let mut file = fs::OpenOptions::new().append(true).open(path).unwrap();
                file.write_all(&[0]).unwrap();

                let err = load_expanded_setup::<TestF, TEST_D, Cfg>(MAX_VARS, 1, 1)
                    .expect_err("cache with trailing bytes must be rejected");
                assert!(err.to_string().contains("trailing bytes"));

                cleanup_setup_file(MAX_VARS);
            });
        }

        #[test]
        fn cache_rejects_seed_capacity_that_is_too_small() {
            with_test_cache_dir("undersized-seed", || {
                const MAX_VARS: usize = 13;
                const MAX_BATCH: usize = 2;

                cleanup_setup_file_shape(MAX_VARS, MAX_BATCH, 1);

                let prover_setup =
                    new_prover_setup::<TestF, TEST_D, Cfg>(MAX_VARS, MAX_BATCH, 1).unwrap();
                let mut stale_seed = prover_setup.expanded.seed().clone();
                stale_seed.max_num_vars = MAX_VARS - 1;
                stale_seed.max_num_batched_polys = MAX_BATCH - 1;
                let stale = AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
                    stale_seed,
                    prover_setup.expanded.shared_matrix().clone(),
                );
                save_expanded_setup::<TestF, Cfg>(&stale, MAX_VARS, MAX_BATCH, 1);

                let regenerated =
                    new_prover_setup::<TestF, TEST_D, Cfg>(MAX_VARS, MAX_BATCH, 1).unwrap();
                assert_eq!(regenerated.expanded.seed().max_num_vars, MAX_VARS);
                assert_eq!(regenerated.expanded.seed().max_num_batched_polys, MAX_BATCH);

                cleanup_setup_file_shape(MAX_VARS, MAX_BATCH, 1);
            });
        }

        #[test]
        fn ntt_caches_rebuilt_correctly_from_disk() {
            with_test_cache_dir("ntt-rebuild", || {
                use akita_algebra::CyclotomicRing;
                use akita_config::CommitmentConfig;
                use akita_prover::AkitaPolyOps;
                use akita_prover::DensePoly;
                use akita_prover::{ComputeBackendSetup, CpuBackend, DigitRowsComputeBackend};

                const MAX_VARS: usize = 14;

                cleanup_setup_file(MAX_VARS);

                let fresh_setup = new_prover_setup::<TestF, TEST_D, Cfg>(MAX_VARS, 1, 1).unwrap();

                let loaded_expanded =
                    load_expanded_setup::<TestF, TEST_D, Cfg>(MAX_VARS, 1, 1).unwrap();
                let disk_setup =
                    AkitaProverSetup::<TestF, TEST_D>::from_validated_expanded(loaded_expanded)
                        .unwrap();

                let lp = Cfg::get_params_for_batched_commitment(
                    &akita_types::ClaimIncidenceSummary::same_point(MAX_VARS, 1)
                        .expect("singleton incidence"),
                )
                .unwrap();
                let num_coeffs = lp.num_blocks * lp.block_len;
                let coeffs = vec![CyclotomicRing::<TestF, TEST_D>::zero(); num_coeffs];
                let poly = DensePoly::<TestF, TEST_D>::from_ring_coeffs(coeffs);

                let commit_u = |setup: &AkitaProverSetup<TestF, TEST_D>| {
                    let prepared = CpuBackend.prepare_setup(setup).unwrap();
                    let inner = poly
                        .commit_inner_witness(
                            &CpuBackend,
                            &prepared,
                            lp.a_key.row_len(),
                            lp.block_len,
                            lp.num_digits_commit,
                            lp.num_digits_open,
                            lp.log_basis,
                        )
                        .unwrap();
                    CpuBackend
                        .digit_rows::<TEST_D>(
                            &prepared,
                            lp.b_key.row_len(),
                            inner.decomposed_inner_rows.flat_digits(),
                        )
                        .unwrap()
                };

                let fresh_u = commit_u(&fresh_setup);
                let disk_u = commit_u(&disk_setup);

                assert_eq!(fresh_u, disk_u);

                cleanup_setup_file(MAX_VARS);
            });
        }
    }
}
