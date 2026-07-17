//! Config-backed prover setup construction.
//!
//! With `disk-persistence`, setup cache files store the expanded setup followed
//! by setup-prefix slots. Caches written before setup-prefix persistence will
//! fail to deserialize and should be regenerated.

mod recursive_prefixes;

use akita_config::CommitmentConfig;
use akita_field::unreduced::HasWide;
use akita_field::{AkitaError, CanonicalField, FieldCore, RandomSampling};
use akita_prover::AkitaProverSetup;
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
    FlatMatrix, PolynomialGroupLayout, SetupPrefixProverRegistry,
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

/// The setup-time generation ring dimension for config `Cfg`.
///
/// Per the cutover decision, `gen_ring_dim` is the **max `ring_dimension` across
/// the config's schedule policy/catalog**. Setup is generated once at capacity
/// time and reused across instances, so it cannot depend on one runtime
/// schedule. For the current uniform-D presets the policy carries a single
/// `ring_dimension == Cfg::D`, so this equals `Cfg::D` (and the verifier binds
/// the same `gen_ring_dim`, preserving transcript byte-parity). A future mixed-D
/// catalog would set this to the maximum dimension its levels use.
fn setup_gen_ring_dim<Cfg: CommitmentConfig>() -> usize {
    akita_config::policy_of::<Cfg>().ring_dimension
}

#[cfg(feature = "disk-persistence")]
fn validate_loaded_prefix_registry<F, Cfg>(
    setup: &AkitaProverSetup<F>,
    max_num_vars: usize,
    max_num_batched_polys: usize,
) -> Result<(), AkitaError>
where
    F: FieldCore,
    Cfg: CommitmentConfig<Field = F>,
{
    if !Cfg::recursive_setup_planning() {
        return Ok(());
    }
    let required_ids = akita_config::setup_prefix_slot_ids_for_capacity::<Cfg>(
        max_num_vars,
        max_num_batched_polys,
    )?;
    recursive_prefixes::validate_prefix_registry_complete(&setup.prefix_slots, &required_ids)
}

/// Construct prover setup from a root commitment config.
///
/// `akita-config` owns setup sizing policy; this crate owns optional disk
/// persistence; `akita-prover` owns the concrete setup artifact and
/// matrix expansion.
///
/// The prover setup artifact is D-free; the setup-time generation ring
/// dimension `gen_ring_dim` is derived from `Cfg`'s schedule policy.
///
/// # Errors
///
/// Returns an error if the requested setup capacity is invalid or setup
/// expansion fails.
#[tracing::instrument(skip_all, name = "new_prover_setup")]
pub fn new_prover_setup<F, Cfg>(
    max_num_vars: usize,
    max_num_batched_polys: usize,
) -> Result<AkitaProverSetup<F>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling + HasWide + Valid,
    Cfg: CommitmentConfig<Field = F>,
{
    if max_num_batched_polys == 0 {
        return Err(AkitaError::InvalidSetup(
            "max_num_batched_polys must be at least 1".to_string(),
        ));
    }
    let gen_ring_dim = setup_gen_ring_dim::<Cfg>();

    #[cfg(feature = "disk-persistence")]
    let max_setup_len =
        Cfg::max_setup_matrix_size(max_num_vars, max_num_batched_polys)?.max_setup_len;

    #[cfg(feature = "disk-persistence")]
    {
        match load_prover_setup::<F, Cfg>(max_num_vars, max_num_batched_polys) {
            Ok(setup) => {
                let cached_total = setup.expanded.shared_matrix().total_ring_elements();
                let cached_shape_covers_request = cached_total >= max_setup_len;
                if cached_shape_covers_request {
                    validate_loaded_prefix_registry::<F, Cfg>(
                        &setup,
                        max_num_vars,
                        max_num_batched_polys,
                    )?;
                    tracing::info!("Loaded setup from disk; backend preparation is explicit");
                    return Ok(setup);
                }
                if let Some(storage_path) =
                    get_storage_path::<Cfg>(max_num_vars, max_num_batched_polys)
                {
                    let _ = fs::remove_file(&storage_path);
                    tracing::warn!(
                        "Rejected cached setup from {}: have total={cached_total}, need total>={max_setup_len}; regenerating",
                        storage_path.display()
                    );
                } else {
                    tracing::warn!(
                        "Rejected cached setup: have total={cached_total}, need total>={max_setup_len}; regenerating"
                    );
                }
            }
            Err(e) => {
                if let Some(storage_path) =
                    get_storage_path::<Cfg>(max_num_vars, max_num_batched_polys)
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

    let setup_envelope = Cfg::max_setup_matrix_size(max_num_vars, max_num_batched_polys)?;

    let mut setup = AkitaProverSetup::generate_with_capacity(
        max_num_vars,
        max_num_batched_polys,
        gen_ring_dim,
        setup_envelope,
    )?;

    recursive_prefixes::populate_required_setup_prefix_slots::<F, Cfg>(
        &mut setup,
        max_num_vars,
        max_num_batched_polys,
    )?;

    #[cfg(feature = "disk-persistence")]
    if let Err(err) = save_prover_setup::<F, Cfg>(&setup, max_num_vars, max_num_batched_polys) {
        tracing::warn!("Failed to persist setup cache: {err}");
    }

    Ok(setup)
}

// ---------------------------------------------------------------------------
// Disk persistence
// ---------------------------------------------------------------------------

#[cfg(feature = "disk-persistence")]
fn stable_type_hash(type_name: &str) -> u64 {
    // FNV-1a keeps cache names short while remaining stable across processes.
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;
    type_name.as_bytes().iter().fold(FNV_OFFSET, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(FNV_PRIME)
    })
}

#[cfg(feature = "disk-persistence")]
fn cache_file_name<Cfg: CommitmentConfig>(
    max_num_vars: usize,
    max_num_batched_polys: usize,
) -> String {
    let type_name = std::any::type_name::<Cfg>();
    let family_hash = stable_type_hash(type_name);
    let schedule_lookup_key = PolynomialGroupLayout::new(max_num_vars, max_num_batched_polys);
    // Fingerprint the resolved schedule shape so cached setup files get
    // invalidated when the planner's per-level layout (including the
    // SIS-derived `n_a`/`n_b`/`n_d` ranks) changes for the same lookup
    // key — the full per-level params are hashed by
    // `digest_effective_schedule`. The `planner_v7_` prefix marks the
    // two-field lookup key cutover; old `planner_v6_*` files are not reused.
    let raw_schedule =
        match Cfg::runtime_schedule(AkitaScheduleLookupKey::single(schedule_lookup_key)) {
            Ok(schedule) => {
                let digest = digest_effective_schedule(&schedule);
                let mut hex = String::with_capacity(digest.len() * 2);
                for byte in digest {
                    let _ = write!(hex, "{byte:02x}");
                }
                format!(
                    "planner_v7_nv{}_batch{}_{hex}",
                    schedule_lookup_key.num_vars(),
                    schedule_lookup_key.num_polynomials(),
                )
            }
            Err(_) => format!(
                "miss_nv{}_batch{}",
                schedule_lookup_key.num_vars(),
                schedule_lookup_key.num_polynomials(),
            ),
        };
    let schedule = raw_schedule
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '_' })
        .collect::<String>();
    let modulus = detect_field_modulus::<Cfg::Field>();
    format!(
        "akita_q{modulus:032x}_cfg{family_hash:016x}_sched_{schedule}_d{}_nv{max_num_vars}_batch{max_num_batched_polys}.setup",
        Cfg::D,
    )
}

#[cfg(feature = "disk-persistence")]
pub(crate) fn get_storage_path<Cfg: CommitmentConfig>(
    max_num_vars: usize,
    max_num_batched_polys: usize,
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
        path.push(cache_file_name::<Cfg>(max_num_vars, max_num_batched_polys));
        path
    })
}

#[cfg(feature = "disk-persistence")]
pub(crate) fn save_prover_setup<
    F: FieldCore + CanonicalField + akita_serialization::AkitaSerialize,
    Cfg: CommitmentConfig<Field = F>,
>(
    setup: &AkitaProverSetup<F>,
    max_num_vars: usize,
    max_num_batched_polys: usize,
) -> Result<(), AkitaError> {
    let Some(storage_path) = get_storage_path::<Cfg>(max_num_vars, max_num_batched_polys) else {
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
    Cfg: CommitmentConfig<Field = F>,
>(
    max_num_vars: usize,
    max_num_batched_polys: usize,
) -> Result<AkitaProverSetup<F>, AkitaError> {
    let storage_path =
        get_storage_path::<Cfg>(max_num_vars, max_num_batched_polys).ok_or_else(|| {
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
    let setup =
        deserialize_cached_setup::<F, Cfg>(&mut reader, max_num_vars, max_num_batched_polys)
            .map_err(|e| AkitaError::InvalidSetup(format!("Failed to deserialize setup: {e}")))?;
    let prefix_slots = SetupPrefixProverRegistry::<F>::deserialize_with_mode(
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
    validate_cached_matrix::<F, Cfg>(&setup)?;

    tracing::info!(
        "Loaded setup for max_num_vars={max_num_vars}, max_num_batched_polys={max_num_batched_polys}"
    );
    Ok(AkitaProverSetup {
        expanded: Arc::new(setup),
        prefix_slots,
    })
}

#[cfg(feature = "disk-persistence")]
fn deserialize_cached_setup<
    F: FieldCore + Valid + AkitaDeserialize<Context = ()>,
    Cfg: CommitmentConfig<Field = F>,
>(
    reader: &mut impl Read,
    expected_max_num_vars: usize,
    expected_max_num_batched_polys: usize,
) -> Result<AkitaExpandedSetup<F>, SerializationError> {
    let seed =
        AkitaSetupSeed::deserialize_with_mode(&mut *reader, Compress::Yes, Validate::Yes, &())?;
    let expected_gen_ring_dim = setup_gen_ring_dim::<Cfg>();
    if seed.gen_ring_dim != expected_gen_ring_dim {
        return Err(SerializationError::InvalidData(format!(
            "cached setup ring dimension {} does not match config gen_ring_dim={expected_gen_ring_dim}",
            seed.gen_ring_dim
        )));
    }
    if seed.max_num_vars != expected_max_num_vars
        || seed.max_num_batched_polys != expected_max_num_batched_polys
    {
        return Err(SerializationError::InvalidData(
            "cached setup seed capacity does not match cache key".to_string(),
        ));
    }
    let expected_envelope = Cfg::max_setup_matrix_size(
        expected_max_num_vars,
        expected_max_num_batched_polys,
    )
    .map_err(|err| {
        SerializationError::InvalidData(format!("cached setup expected shape failed: {err}"))
    })?;
    if seed.max_setup_len != expected_envelope.max_setup_len {
        return Err(SerializationError::InvalidData(
            "cached setup seed matrix shape does not match cache key".to_string(),
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
    Ok(AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(seed, shared_matrix))
}

#[cfg(feature = "disk-persistence")]
fn validate_cached_matrix<
    F: FieldCore + CanonicalField + RandomSampling + Valid,
    Cfg: CommitmentConfig<Field = F>,
>(
    setup: &AkitaExpandedSetup<F>,
) -> Result<(), AkitaError> {
    let expected_gen_ring_dim = setup_gen_ring_dim::<Cfg>();
    if setup.shared_matrix().gen_ring_dim() != expected_gen_ring_dim {
        return Err(AkitaError::InvalidSetup(format!(
            "cached setup ring dimension {} does not match config gen_ring_dim={expected_gen_ring_dim}",
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

    #[test]
    fn expanded_setup_roundtrips_and_derives_same_verifier() {
        let prover_setup = new_prover_setup::<TestF, Cfg>(10, 3).unwrap();
        let verifier_setup = prover_setup.verifier_setup().unwrap();

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
        // fallback. D64Full has a singleton table but the
        // (max_num_vars=12, polys=1, points=1) iteration is a table hit.
        new_prover_setup::<fp128::Field, fp128::D128Full>(12, 1)
            .expect("default fp128 D=128 preset should accept the fp128 field");
        new_prover_setup::<fp128::Field, fp128::D64Full>(12, 1)
            .expect("small-D fp128 preset should accept the default field");
    }

    #[cfg(feature = "disk-persistence")]
    mod disk_persistence {
        const TEST_D: usize = 64;
        use super::*;
        use std::fs;
        use std::sync::{LazyLock, Mutex};

        static DISK_TEST_ENV_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

        fn cleanup_setup_file_shape(max_num_vars: usize, max_num_batched_polys: usize) {
            if let Some(path) = get_storage_path::<Cfg>(max_num_vars, max_num_batched_polys) {
                let _ = fs::remove_file(path);
            }
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

                cleanup_setup_file_shape(MAX_VARS, 1);

                let prover_setup = new_prover_setup::<TestF, Cfg>(MAX_VARS, 1).unwrap();

                let loaded = load_prover_setup::<TestF, Cfg>(MAX_VARS, 1).unwrap();
                assert_eq!(loaded.expanded, prover_setup.expanded);

                cleanup_setup_file_shape(MAX_VARS, 1);
            });
        }

        #[test]
        fn cache_file_name_stays_below_common_component_limits() {
            let name = cache_file_name::<Cfg>(16, 4);
            assert!(
                name.len() < 200,
                "setup cache file name should stay comfortably below 255 bytes, got {}: {name}",
                name.len()
            );
        }

        #[test]
        fn prefix_slots_roundtrip_through_setup_cache() {
            with_test_cache_dir("prefix-slots", || {
                use akita_types::{
                    setup_prefix_slot_id, AjtaiKeyParams, AkitaCommitmentHint, DigitBlocks,
                    PolynomialGroupLayout, PrecommittedGroupParams, PrecommittedLevelParams,
                    RingVec, SetupPrefixPublicCommitment, SetupPrefixSlot, SisMatrixRole,
                    SisModulusProfileId, SisTableDigest, DEFAULT_SIS_SECURITY_POLICY,
                };

                const MAX_VARS: usize = 13;

                cleanup_setup_file_shape(MAX_VARS, 1);

                let mut setup = new_prover_setup::<TestF, Cfg>(MAX_VARS, 1).unwrap();
                let commitment_params = PrecommittedLevelParams {
                    layout: PrecommittedGroupParams {
                        group: PolynomialGroupLayout::singleton(TEST_D.trailing_zeros() as usize),
                        num_live_ring_elements_per_claim: 1,
                        num_positions_per_block: 1,
                        num_live_blocks: 1,
                        fold_challenge_shape: akita_types::TensorChallengeShape::Flat,
                        log_basis_inner: 1,
                        log_basis_outer: 1,
                        n_a: 1,
                        a_coeff_linf_bound: 1,
                        n_b: 1,
                        b_coeff_linf_bound: 1,
                    },
                    a_key: AjtaiKeyParams::new_unchecked(
                        DEFAULT_SIS_SECURITY_POLICY,
                        SisTableDigest::CURRENT,
                        SisModulusProfileId::Q128OffsetA7F7,
                        SisMatrixRole::A,
                        1,
                        1,
                        1,
                        TEST_D,
                    ),
                    b_key: AjtaiKeyParams::new_unchecked(
                        DEFAULT_SIS_SECURITY_POLICY,
                        SisTableDigest::CURRENT,
                        SisModulusProfileId::Q128OffsetA7F7,
                        SisMatrixRole::B,
                        1,
                        1,
                        1,
                        TEST_D,
                    ),
                    log_basis_open: 1,
                    num_digits_inner: 1,
                    num_digits_outer: 1,
                    num_digits_open: 1,
                    num_digits_fold_one: 1,
                };
                let id = setup_prefix_slot_id(TEST_D, 1, commitment_params);
                // One block of zero planes at the setup ring dimension.
                let decomposed = DigitBlocks::empty(TEST_D);
                let hint = AkitaCommitmentHint::singleton(decomposed);
                setup
                    .prefix_slots
                    .insert(SetupPrefixSlot {
                        id,
                        natural_len: 1,
                        padded_len: TEST_D,
                        commitment: SetupPrefixPublicCommitment {
                            rows: vec![RingVec::from_coeffs(vec![TestF::zero(); TEST_D])],
                        },
                        hint,
                    })
                    .unwrap();
                save_prover_setup::<TestF, Cfg>(&setup, MAX_VARS, 1).unwrap();

                let loaded = load_prover_setup::<TestF, Cfg>(MAX_VARS, 1).unwrap();
                assert_eq!(loaded.prefix_slots, setup.prefix_slots);

                cleanup_setup_file_shape(MAX_VARS, 1);
            });
        }

        #[test]
        fn setup_uses_cache_on_second_call() {
            with_test_cache_dir("second-call", || {
                const MAX_VARS: usize = 13;

                cleanup_setup_file_shape(MAX_VARS, 1);

                let first = new_prover_setup::<TestF, Cfg>(MAX_VARS, 1).unwrap();

                let second = new_prover_setup::<TestF, Cfg>(MAX_VARS, 1).unwrap();

                assert_eq!(first.expanded, second.expanded);

                cleanup_setup_file_shape(MAX_VARS, 1);
            });
        }

        #[test]
        fn load_rejects_cached_matrix_that_does_not_match_seed() {
            with_test_cache_dir("corrupt-matrix", || {
                use akita_types::FlatMatrix;

                const MAX_VARS: usize = 13;

                cleanup_setup_file_shape(MAX_VARS, 1);

                let prover_setup = new_prover_setup::<TestF, Cfg>(MAX_VARS, 1).unwrap();
                let total = prover_setup.expanded.shared_matrix().total_ring_elements();
                let corrupt = AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
                    prover_setup.expanded.seed().clone(),
                    FlatMatrix::from_flat_data(vec![TestF::zero(); total * TEST_D], TEST_D),
                );
                let corrupt = AkitaProverSetup {
                    expanded: Arc::new(corrupt),
                    prefix_slots: SetupPrefixProverRegistry::new(),
                };
                save_prover_setup::<TestF, Cfg>(&corrupt, MAX_VARS, 1).unwrap();

                let err = load_prover_setup::<TestF, Cfg>(MAX_VARS, 1)
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

                const MAX_VARS: usize = 13;

                cleanup_setup_file_shape(MAX_VARS, 1);

                new_prover_setup::<TestF, Cfg>(MAX_VARS, 1).unwrap();
                let path = get_storage_path::<Cfg>(MAX_VARS, 1).unwrap();
                let mut file = fs::OpenOptions::new().append(true).open(path).unwrap();
                file.write_all(&[0]).unwrap();

                let err = load_prover_setup::<TestF, Cfg>(MAX_VARS, 1)
                    .expect_err("cache with trailing bytes must be rejected");
                assert!(err.to_string().contains("trailing bytes"));

                cleanup_setup_file_shape(MAX_VARS, 1);
            });
        }

        #[test]
        fn cache_rejects_seed_capacity_that_is_too_small() {
            with_test_cache_dir("undersized-seed", || {
                const MAX_VARS: usize = 13;
                const MAX_BATCH: usize = 2;

                cleanup_setup_file_shape(MAX_VARS, MAX_BATCH);

                let prover_setup = new_prover_setup::<TestF, Cfg>(MAX_VARS, MAX_BATCH).unwrap();
                let mut stale_seed = prover_setup.expanded.seed().clone();
                stale_seed.max_num_vars = MAX_VARS - 1;
                stale_seed.max_num_batched_polys = MAX_BATCH - 1;
                let stale = AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
                    stale_seed,
                    prover_setup.expanded.shared_matrix().clone(),
                );
                let stale = AkitaProverSetup {
                    expanded: Arc::new(stale),
                    prefix_slots: SetupPrefixProverRegistry::new(),
                };
                save_prover_setup::<TestF, Cfg>(&stale, MAX_VARS, MAX_BATCH).unwrap();

                let regenerated = new_prover_setup::<TestF, Cfg>(MAX_VARS, MAX_BATCH).unwrap();
                assert_eq!(regenerated.expanded.seed().max_num_vars, MAX_VARS);
                assert_eq!(regenerated.expanded.seed().max_num_batched_polys, MAX_BATCH);

                cleanup_setup_file_shape(MAX_VARS, MAX_BATCH);
            });
        }

        #[test]
        fn ntt_caches_rebuilt_correctly_from_disk() {
            with_test_cache_dir("ntt-rebuild", || {
                use akita_algebra::CyclotomicRing;
                use akita_config::CommitmentConfig;
                use akita_prover::compute::{CommitInnerPlan, RootCommitKernel, RootCommitSource};
                use akita_prover::DensePoly;
                use akita_prover::{ComputeBackendSetup, CpuBackend, DigitRowsComputeBackend};

                const MAX_VARS: usize = 14;

                cleanup_setup_file_shape(MAX_VARS, 1);

                let fresh_setup = new_prover_setup::<TestF, Cfg>(MAX_VARS, 1).unwrap();

                let disk_setup = load_prover_setup::<TestF, Cfg>(MAX_VARS, 1).unwrap();

                let lp = Cfg::get_params_for_batched_commitment(
                    &akita_types::OpeningClaimsLayout::new(MAX_VARS, 1)
                        .expect("singleton opening batch"),
                )
                .unwrap();
                let num_coeffs = lp.num_live_blocks * lp.num_positions_per_block;
                let coeffs = vec![CyclotomicRing::<TestF, TEST_D>::zero(); num_coeffs];
                let poly = DensePoly::<TestF>::from_ring_coeffs(coeffs);

                let commit_u = |setup: &AkitaProverSetup<TestF>| {
                    let prepared = CpuBackend.prepare_setup(setup).unwrap();
                    let plan = CommitInnerPlan::from_level(&lp);
                    let inner = RootCommitKernel::commit_inner(
                        &CpuBackend,
                        &prepared,
                        RootCommitSource::<TestF, TEST_D>::commit_view(&poly).unwrap(),
                        plan,
                    )
                    .unwrap();
                    let typed_digits = inner.decomposed_inner_rows_trusted::<TEST_D>().unwrap();
                    CpuBackend
                        .digit_rows::<TEST_D>(
                            &prepared,
                            lp.b_key.row_len(),
                            typed_digits.typed_planes::<TEST_D>().unwrap(),
                            lp.log_basis_outer,
                        )
                        .unwrap()
                };

                let fresh_u = commit_u(&fresh_setup);
                let disk_u = commit_u(&disk_setup);

                assert_eq!(fresh_u, disk_u);

                cleanup_setup_file_shape(MAX_VARS, 1);
            });
        }
    }
}
