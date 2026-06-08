//! Config-backed prover setup construction.

mod recursion;

pub use recursion::new_prover_setup_recursion;

use akita_config::CommitmentConfig;
use akita_field::unreduced::HasWide;
use akita_field::{AkitaError, CanonicalField, FieldCore, RandomSampling};
use akita_prover::AkitaProverSetup;
#[cfg(feature = "disk-persistence")]
use akita_prover::{select_prover_setup_prefix_slot, CommitmentComputeBackend};
use akita_serialization::Valid;
#[cfg(feature = "disk-persistence")]
use akita_serialization::{
    AkitaDeserialize, AkitaSerialize, Compress, SerializationError, Validate,
};
#[cfg(any(feature = "disk-persistence", test))]
use akita_types::AkitaExpandedSetup;
#[cfg(feature = "disk-persistence")]
use akita_types::{
    detect_field_modulus, digest_effective_schedule, AkitaScheduleLookupKey, AkitaSetupSeed,
    ClaimIncidenceSummary, FlatMatrix, LevelParams, MissingSetupPrefixSlotPolicy,
    SetupPrefixProverRegistry, SetupPrefixSelectionOutcome,
};
#[cfg(test)]
use akita_types::{AkitaVerifierSetup, SetupPrefixVerifierRegistry};
#[cfg(feature = "disk-persistence")]
use std::fmt::Write as _;
#[cfg(feature = "disk-persistence")]
use std::fs;
#[cfg(feature = "disk-persistence")]
use std::io::Read;
#[cfg(feature = "disk-persistence")]
use std::path::PathBuf;
#[cfg(feature = "disk-persistence")]
use std::sync::Arc;
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
    let setup_envelope =
        Cfg::max_setup_matrix_size(max_num_vars, max_num_batched_polys, max_num_points)?;
    #[cfg(feature = "disk-persistence")]
    let max_setup_len = setup_envelope.max_setup_len;

    #[cfg(feature = "disk-persistence")]
    {
        match load_prover_setup::<F, D, Cfg>(max_num_vars, max_num_batched_polys, max_num_points) {
            Ok(setup) => {
                // A cached setup is acceptable only if its physical backing
                // covers the packed setup envelope for the current request.
                let cached_total = setup
                    .expanded
                    .shared_matrix()
                    .total_ring_elements_at::<D>()?;
                let cached_points = setup.expanded.seed().max_num_points;
                #[cfg(feature = "zk")]
                let cached_zk_b_total =
                    setup.expanded.zk_b_matrix().total_ring_elements_at::<D>()?;
                #[cfg(feature = "zk")]
                let cached_zk_d_total =
                    setup.expanded.zk_d_matrix().total_ring_elements_at::<D>()?;
                let cached_shape_covers_request =
                    cached_total >= max_setup_len && cached_points >= max_num_points;
                #[cfg(feature = "zk")]
                let cached_shape_covers_request = cached_shape_covers_request
                    && cached_zk_b_total >= setup_envelope.max_zk_b_len
                    && cached_zk_d_total >= setup_envelope.max_zk_d_len;
                if cached_shape_covers_request {
                    tracing::info!("Loaded setup from disk; backend preparation is explicit");
                    return Ok(setup);
                }
                if let Some(storage_path) =
                    get_storage_path::<Cfg>(max_num_vars, max_num_batched_polys, max_num_points)
                {
                    let _ = fs::remove_file(&storage_path);
                    tracing::warn!(
                            "Rejected cached setup from {}: have (total={cached_total}, points={cached_points}), need (total>={max_setup_len}, points>={max_num_points}); regenerating",
                            storage_path.display()
                        );
                } else {
                    tracing::warn!(
                            "Rejected cached setup: have (total={cached_total}, points={cached_points}), need (total>={max_setup_len}, points>={max_num_points}); regenerating"
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
        setup_envelope,
    )?;

    #[cfg(feature = "disk-persistence")]
    if let Err(err) = persist_prover_setup::<F, D, Cfg>(
        &setup,
        max_num_vars,
        max_num_batched_polys,
        max_num_points,
    ) {
        tracing::warn!("Failed to persist setup cache: {err}");
    }

    Ok(setup)
}

/// Persist a prover setup, including any prover-ready setup-prefix slots, to
/// the config-derived setup cache path.
///
/// Call this after `GenerateAndPersist` materializes prefix slots so later
/// `new_prover_setup` calls can reload the preprocessed artifacts.
#[cfg(feature = "disk-persistence")]
pub fn persist_prover_setup<
    F: FieldCore + CanonicalField + akita_serialization::AkitaSerialize,
    const D: usize,
    Cfg: CommitmentConfig<Field = F>,
>(
    setup: &AkitaProverSetup<F, D>,
    max_num_vars: usize,
    max_num_batched_polys: usize,
    max_num_points: usize,
) -> Result<(), AkitaError> {
    save_prover_setup::<F, D, Cfg>(setup, max_num_vars, max_num_batched_polys, max_num_points)
}

/// Inputs for selecting a setup-prefix slot and writing the setup cache when
/// `GenerateAndPersist` is active.
#[cfg(feature = "disk-persistence")]
#[derive(Debug, Clone, Copy)]
pub struct SetupPrefixPersistRequest<'a> {
    pub level_params: &'a LevelParams,
    pub incidence: &'a ClaimIncidenceSummary,
    pub n_min: usize,
    pub missing_slot_policy: MissingSetupPrefixSlotPolicy,
    pub max_num_vars: usize,
    pub max_num_batched_polys: usize,
    pub max_num_points: usize,
}

/// Select a setup-prefix slot and persist the setup cache if the
/// `GenerateAndPersist` policy materializes a missing slot.
#[cfg(feature = "disk-persistence")]
pub fn select_or_persist_setup_prefix_slot<F, const D: usize, Cfg, B>(
    setup: &mut AkitaProverSetup<F, D>,
    backend: &B,
    prepared: &B::PreparedSetup<D>,
    request: SetupPrefixPersistRequest<'_>,
) -> Result<SetupPrefixSelectionOutcome<F, D>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling + akita_serialization::AkitaSerialize,
    Cfg: CommitmentConfig<Field = F>,
    B: CommitmentComputeBackend<F>,
{
    let slot_count_before = setup.prefix_slots.len();
    let outcome = select_prover_setup_prefix_slot(
        setup,
        backend,
        prepared,
        request.level_params,
        request.incidence,
        request.n_min,
        request.missing_slot_policy,
    )?;
    if matches!(
        (&outcome, request.missing_slot_policy),
        (
            SetupPrefixSelectionOutcome::Selected(_),
            MissingSetupPrefixSlotPolicy::GenerateAndPersist
        )
    ) && setup.prefix_slots.len() > slot_count_before
    {
        persist_prover_setup::<F, D, Cfg>(
            setup,
            request.max_num_vars,
            request.max_num_batched_polys,
            request.max_num_points,
        )?;
    }
    Ok(outcome)
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
    // invalidated when the planner's per-level layout (including the
    // SIS-derived `n_a`/`n_b`/`n_d` ranks) changes for the same lookup
    // key — the full per-level params are hashed by
    // `digest_effective_schedule`.
    let raw_schedule = match Cfg::runtime_schedule(schedule_lookup_key) {
        Ok(schedule) => {
            let digest = digest_effective_schedule(&schedule);
            let mut hex = String::with_capacity(digest.len() * 2);
            for byte in digest {
                let _ = write!(hex, "{byte:02x}");
            }
            format!(
                "planner_v6_nv{}_g{}_t{}_w{}_z{}_{hex}",
                schedule_lookup_key.num_vars,
                schedule_lookup_key.num_points,
                schedule_lookup_key.num_t_vectors,
                schedule_lookup_key.num_w_vectors,
                schedule_lookup_key.num_z_vectors,
            )
        }
        Err(_) => format!(
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
        "akita_q{modulus:032x}_{family}_sched_{schedule}_d{}_nv{max_num_vars}_batch{max_num_batched_polys}_pts{max_num_points}.setup",
        Cfg::D,
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
fn save_prover_setup<
    F: FieldCore + CanonicalField + akita_serialization::AkitaSerialize,
    const D: usize,
    Cfg: CommitmentConfig<Field = F>,
>(
    setup: &AkitaProverSetup<F, D>,
    max_num_vars: usize,
    max_num_batched_polys: usize,
    max_num_points: usize,
) -> Result<(), AkitaError> {
    let Some(storage_path) =
        get_storage_path::<Cfg>(max_num_vars, max_num_batched_polys, max_num_points)
    else {
        return Err(AkitaError::InvalidSetup(
            "could not determine storage directory".to_string(),
        ));
    };

    if let Some(parent) = storage_path.parent() {
        if let Err(e) = fs::create_dir_all(parent) {
            return Err(AkitaError::InvalidSetup(format!(
                "failed to create setup cache directory {}: {e}",
                parent.display()
            )));
        }
    }

    tracing::info!("Saving setup to {}", storage_path.display());

    let file = match fs::File::create(&storage_path) {
        Ok(file) => file,
        Err(e) => {
            return Err(AkitaError::InvalidSetup(format!(
                "failed to create setup cache file {}: {e}",
                storage_path.display()
            )));
        }
    };
    let mut writer = std::io::BufWriter::new(file);

    if let Err(e) = setup.expanded.serialize_compressed(&mut writer) {
        let _ = fs::remove_file(&storage_path);
        return Err(AkitaError::InvalidSetup(format!(
            "failed to serialize setup cache {}: {e}",
            storage_path.display()
        )));
    }
    if let Err(e) = setup.prefix_slots.serialize_compressed(&mut writer) {
        let _ = fs::remove_file(&storage_path);
        return Err(AkitaError::InvalidSetup(format!(
            "failed to serialize setup-prefix cache {}: {e}",
            storage_path.display()
        )));
    }

    tracing::info!("Successfully saved setup to disk");
    Ok(())
}

#[cfg(feature = "disk-persistence")]
pub(crate) fn load_prover_setup<
    F: FieldCore + Valid + CanonicalField + RandomSampling,
    const D: usize,
    Cfg: CommitmentConfig<Field = F>,
>(
    max_num_vars: usize,
    max_num_batched_polys: usize,
    max_num_points: usize,
) -> Result<AkitaProverSetup<F, D>, AkitaError> {
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
    let prefix_slots = SetupPrefixProverRegistry::<F, D>::deserialize_with_mode(
        &mut reader,
        Compress::Yes,
        Validate::Yes,
        &(),
    )
    .map_err(|e| {
        AkitaError::InvalidSetup(format!("Failed to deserialize setup-prefix slots: {e}"))
    })?;
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
    Ok(AkitaProverSetup {
        expanded: Arc::new(setup),
        prefix_slots,
    })
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
    let expected_envelope = Cfg::max_setup_matrix_size(
        expected_max_num_vars,
        expected_max_num_batched_polys,
        expected_max_num_points,
    )
    .map_err(|err| {
        SerializationError::InvalidData(format!("cached setup expected shape failed: {err}"))
    })?;
    if seed.max_setup_len != expected_envelope.max_setup_len {
        return Err(SerializationError::InvalidData(
            "cached setup seed matrix shape does not match cache key".to_string(),
        ));
    }
    #[cfg(feature = "zk")]
    if seed.max_zk_b_len != expected_envelope.max_zk_b_len
        || seed.max_zk_d_len != expected_envelope.max_zk_d_len
    {
        return Err(SerializationError::InvalidData(
            "cached setup seed ZK matrix shape does not match cache key".to_string(),
        ));
    }
    let shared_matrix = FlatMatrix::<F>::deserialize_with_expected_shape(
        &mut *reader,
        Compress::Yes,
        Validate::Yes,
        seed.max_setup_len,
        seed.gen_ring_dim,
        seed.matrix_field_elements()?,
    )?;
    #[cfg(feature = "zk")]
    let zk_b_matrix = FlatMatrix::<F>::deserialize_with_expected_shape(
        &mut *reader,
        Compress::Yes,
        Validate::Yes,
        seed.max_zk_b_len,
        seed.gen_ring_dim,
        seed.zk_b_matrix_field_elements()?,
    )?;
    #[cfg(feature = "zk")]
    let zk_d_matrix = FlatMatrix::<F>::deserialize_with_expected_shape(
        &mut *reader,
        Compress::Yes,
        Validate::Yes,
        seed.max_zk_d_len,
        seed.gen_ring_dim,
        seed.zk_d_matrix_field_elements()?,
    )?;
    Ok(
        AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
            seed,
            shared_matrix,
            #[cfg(feature = "zk")]
            zk_b_matrix,
            #[cfg(feature = "zk")]
            zk_d_matrix,
        ),
    )
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
    setup
        .check()
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
        let verifier_setup = prover_setup.verifier_setup();

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
            prefix_slots: SetupPrefixVerifierRegistry::new(),
        };
        assert_eq!(derived_verifier, verifier_setup);
    }

    #[test]
    fn setup_accepts_field_coupled_presets() {
        // D128Full has no schedule table at all, so setup-matrix sizing
        // falls through to the planner DP via the default `runtime_schedule`
        // fallback. D32Full has a singleton table but the
        // (max_num_vars=12, polys=1, points=1) iteration is a table hit.
        new_prover_setup::<fp128::Field, 128, fp128::D128Full>(12, 1, 1)
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

                let loaded = load_prover_setup::<TestF, TEST_D, Cfg>(MAX_VARS, 1, 1).unwrap();
                assert_eq!(loaded.expanded, prover_setup.expanded);

                cleanup_setup_file(MAX_VARS);
            });
        }

        #[test]
        fn prefix_slots_roundtrip_through_setup_cache() {
            with_test_cache_dir("prefix-slots", || {
                use akita_algebra::CyclotomicRing;
                use akita_types::{
                    setup_seed_digest, AkitaCommitmentHint, FlatDigitBlocks, RingCommitment,
                    SetupPrefixSlot, SetupPrefixSlotId,
                };

                const MAX_VARS: usize = 13;

                cleanup_setup_file(MAX_VARS);

                let mut setup = new_prover_setup::<TestF, TEST_D, Cfg>(MAX_VARS, 1, 1).unwrap();
                let id = SetupPrefixSlotId {
                    setup_seed_digest: setup_seed_digest(setup.expanded.seed()).unwrap(),
                    d_setup: TEST_D,
                    n_prefix: TEST_D,
                    level_params_digest: [9u8; 32],
                };
                let decomposed = FlatDigitBlocks::<TEST_D>::from_blocks(vec![Vec::new()]);
                let recomposed = vec![Vec::new()];
                #[cfg(feature = "zk")]
                let hint = AkitaCommitmentHint::singleton_with_recomposed_inner_rows(
                    decomposed,
                    recomposed,
                    FlatDigitBlocks::empty(),
                );
                #[cfg(not(feature = "zk"))]
                let hint = AkitaCommitmentHint::singleton_with_recomposed_inner_rows(
                    decomposed, recomposed,
                );
                setup
                    .prefix_slots
                    .insert(SetupPrefixSlot {
                        id,
                        natural_len: 1,
                        padded_len: TEST_D,
                        commitment: RingCommitment {
                            u: vec![CyclotomicRing::zero()],
                        },
                        hint,
                    })
                    .unwrap();
                persist_prover_setup::<TestF, TEST_D, Cfg>(&setup, MAX_VARS, 1, 1).unwrap();

                let loaded = load_prover_setup::<TestF, TEST_D, Cfg>(MAX_VARS, 1, 1).unwrap();
                assert_eq!(loaded.prefix_slots, setup.prefix_slots);

                cleanup_setup_file(MAX_VARS);
            });
        }

        #[test]
        fn generate_and_persist_skips_preexisting_prefix_slot() {
            with_test_cache_dir("generate-and-persist", || {
                use akita_algebra::CyclotomicRing;
                use akita_prover::{ComputeBackendSetup, CpuBackend};
                use akita_types::{
                    active_setup_field_len, digest_level_params, padded_setup_prefix_len,
                    setup_prefix_slot_id, setup_seed_digest, AkitaCommitmentHint, FlatDigitBlocks,
                    RingCommitment, SetupPrefixSelectionOutcome, SetupPrefixSlot,
                };

                type PersistCfg = fp128::D32Full;
                const PERSIST_D: usize = 32;
                const MAX_VARS: usize = 12;
                const MAX_BATCH: usize = 2;
                const MAX_POINTS: usize = 2;

                if let Some(path) = get_storage_path::<PersistCfg>(MAX_VARS, MAX_BATCH, MAX_POINTS)
                {
                    let _ = fs::remove_file(path);
                }

                let mut setup = new_prover_setup::<TestF, PERSIST_D, PersistCfg>(
                    MAX_VARS, MAX_BATCH, MAX_POINTS,
                )
                .unwrap();
                let incidence =
                    ClaimIncidenceSummary::from_counts(MAX_VARS, MAX_POINTS, MAX_POINTS)
                        .expect("incidence");
                let level_params =
                    PersistCfg::get_params_for_batched_commitment(&incidence).unwrap();
                let natural_len =
                    active_setup_field_len(&level_params, &incidence, PERSIST_D).unwrap();
                let n_prefix = padded_setup_prefix_len(natural_len);
                let slot_id = setup_prefix_slot_id(
                    setup_seed_digest(setup.expanded.seed()).unwrap(),
                    PERSIST_D,
                    n_prefix,
                    digest_level_params(std::slice::from_ref(&level_params)),
                );
                let decomposed = FlatDigitBlocks::<PERSIST_D>::from_blocks(vec![Vec::new()]);
                let recomposed = vec![Vec::new()];
                #[cfg(feature = "zk")]
                let hint = AkitaCommitmentHint::singleton_with_recomposed_inner_rows(
                    decomposed,
                    recomposed,
                    FlatDigitBlocks::empty(),
                );
                #[cfg(not(feature = "zk"))]
                let hint = AkitaCommitmentHint::singleton_with_recomposed_inner_rows(
                    decomposed, recomposed,
                );
                setup
                    .prefix_slots
                    .insert(SetupPrefixSlot {
                        id: slot_id,
                        natural_len,
                        padded_len: n_prefix,
                        commitment: RingCommitment {
                            u: vec![CyclotomicRing::zero()],
                        },
                        hint,
                    })
                    .unwrap();
                let backend = CpuBackend;
                let prepared = backend.prepare_setup::<PERSIST_D>(&setup).unwrap();

                let outcome =
                    select_or_persist_setup_prefix_slot::<TestF, PERSIST_D, PersistCfg, _>(
                        &mut setup,
                        &backend,
                        &prepared,
                        SetupPrefixPersistRequest {
                            level_params: &level_params,
                            incidence: &incidence,
                            n_min: 1,
                            missing_slot_policy: MissingSetupPrefixSlotPolicy::GenerateAndPersist,
                            max_num_vars: MAX_VARS,
                            max_num_batched_polys: MAX_BATCH,
                            max_num_points: MAX_POINTS,
                        },
                    )
                    .unwrap();

                assert!(matches!(outcome, SetupPrefixSelectionOutcome::Selected(_)));
                let loaded = load_prover_setup::<TestF, PERSIST_D, PersistCfg>(
                    MAX_VARS, MAX_BATCH, MAX_POINTS,
                )
                .unwrap();
                assert!(loaded.prefix_slots.is_empty());

                if let Some(path) = get_storage_path::<PersistCfg>(MAX_VARS, MAX_BATCH, MAX_POINTS)
                {
                    let _ = fs::remove_file(path);
                }
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
                    #[cfg(feature = "zk")]
                    prover_setup.expanded.zk_b_matrix().clone(),
                    #[cfg(feature = "zk")]
                    prover_setup.expanded.zk_d_matrix().clone(),
                );
                let corrupt = AkitaProverSetup {
                    expanded: Arc::new(corrupt),
                    prefix_slots: SetupPrefixProverRegistry::new(),
                };
                save_prover_setup::<TestF, TEST_D, Cfg>(&corrupt, MAX_VARS, 1, 1).unwrap();

                let err = load_prover_setup::<TestF, TEST_D, Cfg>(MAX_VARS, 1, 1)
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

                let err = load_prover_setup::<TestF, TEST_D, Cfg>(MAX_VARS, 1, 1)
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
                    #[cfg(feature = "zk")]
                    prover_setup.expanded.zk_b_matrix().clone(),
                    #[cfg(feature = "zk")]
                    prover_setup.expanded.zk_d_matrix().clone(),
                );
                let stale = AkitaProverSetup {
                    expanded: Arc::new(stale),
                    prefix_slots: SetupPrefixProverRegistry::new(),
                };
                save_prover_setup::<TestF, TEST_D, Cfg>(&stale, MAX_VARS, MAX_BATCH, 1).unwrap();

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

                let disk_setup = load_prover_setup::<TestF, TEST_D, Cfg>(MAX_VARS, 1, 1).unwrap();

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
                            lp.log_basis,
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
