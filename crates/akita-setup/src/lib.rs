//! Requirement-backed prover setup construction.
//!
//! With `disk-persistence`, the public field prefix is stored by field and
//! [`akita_types::AkitaSetupSeed`], separately from requirement-bound
//! setup-prefix registries. Backend NTT caches are never persisted.

mod recursive_prefixes;

use akita_config::SetupRequirements;
use akita_cpu_backend::AkitaProverSetup;
#[cfg(feature = "disk-persistence")]
use akita_cpu_backend::SetupPrefixProverRegistry;
use akita_error::AkitaError;
use akita_serialization::{AkitaDeserialize, AkitaSerialize, Valid};
#[cfg(feature = "disk-persistence")]
use akita_serialization::{Compress, SerializationError, Validate};
#[cfg(any(feature = "disk-persistence", test))]
use akita_types::AkitaExpandedSetup;
#[cfg(feature = "disk-persistence")]
use akita_types::{
    detect_field_modulus, digest_descriptor_bytes, sample_akita_setup_seed, setup_seed_digest,
    AkitaSetupDescriptor, AkitaSetupSeed, FlatMatrix,
};
use jolt_field::{CanonicalEncoding, Field};
use jolt_field::{Unreduced, WithCommitAccumulator};
#[cfg(feature = "disk-persistence")]
use std::fmt::Write as _;
#[cfg(feature = "disk-persistence")]
use std::fs;
#[cfg(feature = "disk-persistence")]
use std::io::{Read, Write};
#[cfg(feature = "disk-persistence")]
use std::path::PathBuf;
#[cfg(feature = "disk-persistence")]
use std::sync::atomic::{AtomicU64, Ordering};
#[cfg(feature = "disk-persistence")]
use std::sync::{Arc, LazyLock, Mutex};

#[cfg(feature = "disk-persistence")]
static CACHE_TEMP_ID: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "disk-persistence")]
static PUBLIC_MATRIX_CACHE_WRITE_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

/// Construct prover setup that meets `requirements`.
///
/// `akita-config` owns setup sizing policy and derives the requirements from
/// one or more trusted catalogs; this crate owns optional disk persistence;
/// `akita-cpu-backend` owns the concrete setup artifact and matrix expansion.
///
/// # Errors
///
/// Returns an error if the requested setup capacity is invalid or setup
/// expansion fails.
#[tracing::instrument(skip_all, name = "new_prover_setup")]
pub fn new_prover_setup<F>(
    requirements: &SetupRequirements,
) -> Result<AkitaProverSetup<F>, AkitaError>
where
    F: Field
        + CanonicalEncoding
        + Unreduced
        + Valid
        + AkitaSerialize
        + AkitaDeserialize<Context = ()>
        + WithCommitAccumulator
        + 'static,
{
    #[cfg(feature = "disk-persistence")]
    {
        match load_prover_setup::<F>(requirements) {
            Ok(setup) => {
                tracing::info!("Loaded setup from disk; backend preparation is explicit");
                return Ok(setup);
            }
            Err(err) => {
                tracing::warn!("Failed to load cached setup: {err}; regenerating");
            }
        }
    }

    let mut setup = AkitaProverSetup::generate_with_capacity(
        requirements.max_num_vars,
        requirements.max_num_batched_polys,
        requirements.matrix_capacity,
    )?;

    recursive_prefixes::populate_required_setup_prefix_slots(
        &mut setup,
        &requirements.prefix_slot_ids,
    )?;

    #[cfg(feature = "disk-persistence")]
    if let Err(err) = save_prover_setup::<F>(&setup, requirements) {
        tracing::warn!("Failed to persist setup cache: {err}");
    }

    Ok(setup)
}

// ---------------------------------------------------------------------------
// Disk persistence
// ---------------------------------------------------------------------------

/// Name the setup-prefix registry cache by field modulus and requirements.
///
/// The key digests the capacity bound and the sorted prefix slot ids, so one
/// cached registry serves every catalog combination with those requirements.
#[cfg(feature = "disk-persistence")]
fn prefix_registry_cache_file_name<F: Field + CanonicalEncoding>(
    requirements: &SetupRequirements,
) -> Result<String, AkitaError> {
    let mut key = Vec::new();
    key.extend_from_slice(b"AKITA-SETUP-PREFIX-REQUIREMENTS-V1");
    key.extend_from_slice(&(requirements.max_num_vars as u64).to_le_bytes());
    key.extend_from_slice(&(requirements.max_num_batched_polys as u64).to_le_bytes());
    requirements
        .prefix_slot_ids
        .serialize_uncompressed(&mut key)
        .map_err(|err| AkitaError::InvalidSetup(format!("setup-prefix registry key: {err}")))?;
    let mut requirements_hex = String::with_capacity(64);
    for byte in digest_descriptor_bytes(&key) {
        let _ = write!(requirements_hex, "{byte:02x}");
    }
    let modulus = detect_field_modulus::<F>()?;
    Ok(format!(
        "akita_prefix_v4_q{modulus:032x}_req_{requirements_hex}.registry",
    ))
}

#[cfg(feature = "disk-persistence")]
fn public_matrix_cache_file_name<F: Field + CanonicalEncoding>(
    setup_seed: &AkitaSetupSeed,
) -> Result<String, AkitaError> {
    let digest = setup_seed_digest(setup_seed)
        .map_err(|err| AkitaError::InvalidSetup(format!("public matrix identity: {err}")))?;
    let mut hex = String::with_capacity(digest.len() * 2);
    for byte in digest {
        let _ = write!(hex, "{byte:02x}");
    }
    let modulus = detect_field_modulus::<F>()?;
    Ok(format!("akita_flat_v3_q{modulus:032x}_id{hex}.matrix"))
}

#[cfg(feature = "disk-persistence")]
fn cache_directory() -> Option<PathBuf> {
    let mut path = if let Ok(local_app_data) = std::env::var("LOCALAPPDATA") {
        PathBuf::from(local_app_data)
    } else if let Ok(home) = std::env::var("HOME") {
        let mut path = PathBuf::from(&home);
        let mut macos_cache = PathBuf::from(&home);
        macos_cache.push("Library");
        macos_cache.push("Caches");
        if macos_cache.exists() {
            path.push("Library");
            path.push("Caches");
        } else {
            path.push(".cache");
        }
        path
    } else {
        return None;
    };
    path.push("akita");
    Some(path)
}

#[cfg(feature = "disk-persistence")]
pub(crate) fn get_prefix_registry_storage_path<F: Field + CanonicalEncoding>(
    requirements: &SetupRequirements,
) -> Option<PathBuf> {
    let mut path = cache_directory()?;
    path.push(prefix_registry_cache_file_name::<F>(requirements).ok()?);
    Some(path)
}

#[cfg(feature = "disk-persistence")]
fn get_public_matrix_storage_path<F: Field + CanonicalEncoding>(
    setup_seed: &AkitaSetupSeed,
) -> Result<PathBuf, AkitaError> {
    let mut path = cache_directory().ok_or_else(|| {
        AkitaError::InvalidSetup("could not determine storage directory".to_string())
    })?;
    path.push(public_matrix_cache_file_name::<F>(setup_seed)?);
    Ok(path)
}

#[cfg(feature = "disk-persistence")]
fn atomic_write_cache(
    storage_path: &std::path::Path,
    write_cache: impl FnOnce(&mut std::io::BufWriter<fs::File>) -> Result<(), SerializationError>,
) -> Result<(), AkitaError> {
    let parent = storage_path.parent().ok_or_else(|| {
        AkitaError::InvalidSetup("setup cache path has no parent directory".to_string())
    })?;
    fs::create_dir_all(parent).map_err(|err| {
        AkitaError::InvalidSetup(format!(
            "failed to create setup cache directory {}: {err}",
            parent.display()
        ))
    })?;
    let temp_id = CACHE_TEMP_ID.fetch_add(1, Ordering::Relaxed);
    let temp_path = storage_path.with_extension(format!("tmp-{}-{temp_id}", std::process::id()));
    let result = (|| {
        let file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp_path)
            .map_err(|err| {
                AkitaError::InvalidSetup(format!(
                    "failed to create temporary setup cache {}: {err}",
                    temp_path.display()
                ))
            })?;
        let mut writer = std::io::BufWriter::new(file);
        write_cache(&mut writer).map_err(|err| {
            AkitaError::InvalidSetup(format!(
                "failed to serialize setup cache {}: {err}",
                storage_path.display()
            ))
        })?;
        writer.flush().map_err(|err| {
            AkitaError::InvalidSetup(format!(
                "failed to flush setup cache {}: {err}",
                temp_path.display()
            ))
        })?;
        // These files are recoverable performance caches: a failed or partial
        // write is rejected and regenerated on the next load. Flushing before
        // the atomic rename gives readers a complete file without forcing a
        // device flush on the setup hot path.
        drop(writer);
        fs::rename(&temp_path, storage_path).map_err(|err| {
            AkitaError::InvalidSetup(format!(
                "failed to atomically replace setup cache {}: {err}",
                storage_path.display()
            ))
        })
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temp_path);
    }
    result
}

#[cfg(feature = "disk-persistence")]
fn serialize_public_matrix_cache<F: Field + AkitaSerialize>(
    expanded: &AkitaExpandedSetup<F>,
    writer: &mut std::io::BufWriter<fs::File>,
) -> Result<(), SerializationError> {
    expanded
        .descriptor()
        .setup_seed
        .serialize_compressed(&mut *writer)?;
    expanded
        .shared_matrix()
        .num_field_elements()
        .serialize_compressed(&mut *writer)?;
    expanded.shared_matrix().serialize_compressed(writer)
}

#[cfg(feature = "disk-persistence")]
pub(crate) fn save_prover_setup<
    F: Field + CanonicalEncoding + Valid + AkitaSerialize + AkitaDeserialize<Context = ()>,
>(
    setup: &AkitaProverSetup<F>,
    requirements: &SetupRequirements,
) -> Result<(), AkitaError> {
    // `setup` was just derived inside this crate. Re-deriving and comparing
    // every field element here would repeat the full setup-generation pass;
    // public-matrix cache bytes are deterministically validated on load.
    // Prefix-registry provenance is a separate setup-validation boundary.
    let public_matrix_path =
        get_public_matrix_storage_path::<F>(&setup.expanded.descriptor().setup_seed)?;
    let Some(prefix_registry_path) = get_prefix_registry_storage_path::<F>(requirements) else {
        return Err(AkitaError::InvalidSetup(
            "could not determine storage directory".to_string(),
        ));
    };

    let _matrix_write_guard = PUBLIC_MATRIX_CACHE_WRITE_LOCK
        .lock()
        .map_err(|_| AkitaError::InvalidSetup("public matrix cache lock poisoned".to_string()))?;
    let matrix_parent = public_matrix_path.parent().ok_or_else(|| {
        AkitaError::InvalidSetup("public matrix cache path has no parent directory".to_string())
    })?;
    fs::create_dir_all(matrix_parent).map_err(|err| {
        AkitaError::InvalidSetup(format!(
            "failed to create public matrix cache directory: {err}"
        ))
    })?;
    let matrix_lock_path = public_matrix_path.with_extension("matrix.lock");
    let matrix_lock = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&matrix_lock_path)
        .map_err(|err| {
            AkitaError::InvalidSetup(format!("failed to open public matrix cache lock: {err}"))
        })?;
    matrix_lock.lock().map_err(|err| {
        AkitaError::InvalidSetup(format!("failed to lock public matrix cache: {err}"))
    })?;
    let replace_public_matrix = match fs::File::open(&public_matrix_path) {
        Ok(file) => {
            let mut reader = std::io::BufReader::new(file);
            let existing = deserialize_cached_public_matrix::<F>(
                &mut reader,
                0,
                &setup.expanded.descriptor().setup_seed,
            );
            let mut trailing = [0u8; 1];
            match existing {
                Ok(existing)
                    if reader.read(&mut trailing).is_ok_and(|read| read == 0)
                        && validate_cached_matrix::<F>(&existing).is_ok() =>
                {
                    existing.shared_matrix().num_field_elements()
                        < setup.expanded.shared_matrix().num_field_elements()
                }
                _ => true,
            }
        }
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => true,
        Err(err) => {
            return Err(AkitaError::InvalidSetup(format!(
                "failed to inspect public matrix cache: {err}"
            )))
        }
    };
    if replace_public_matrix {
        atomic_write_cache(&public_matrix_path, |writer| {
            serialize_public_matrix_cache(&setup.expanded, writer)
        })?;
    }
    drop(matrix_lock);
    drop(_matrix_write_guard);
    atomic_write_cache(&prefix_registry_path, |writer| {
        setup.prefix_slots.serialize_compressed(writer)
    })?;

    tracing::info!(
        "Saved public matrix to {} and setup-prefix registry to {}",
        public_matrix_path.display(),
        prefix_registry_path.display()
    );
    Ok(())
}

#[cfg(feature = "disk-persistence")]
pub(crate) fn load_prover_setup<
    F: Field
        + Valid
        + CanonicalEncoding
        + Unreduced
        + WithCommitAccumulator
        + AkitaSerialize
        + AkitaDeserialize<Context = ()>
        + 'static,
>(
    requirements: &SetupRequirements,
) -> Result<AkitaProverSetup<F>, AkitaError> {
    let SetupRequirements {
        max_num_vars,
        max_num_batched_polys,
        ..
    } = *requirements;
    let setup_seed = sample_akita_setup_seed();
    let public_matrix_path = get_public_matrix_storage_path::<F>(&setup_seed)?;
    if !public_matrix_path.exists() {
        return Err(AkitaError::InvalidSetup(format!(
            "public matrix cache not found at {}",
            public_matrix_path.display()
        )));
    }
    let required_num_field_elements = requirements.matrix_capacity.num_field_elements;
    let file = fs::File::open(&public_matrix_path).map_err(|err| {
        AkitaError::InvalidSetup(format!("failed to open public matrix cache: {err}"))
    })?;
    let mut reader = std::io::BufReader::new(file);
    let mut expanded = deserialize_cached_public_matrix::<F>(
        &mut reader,
        required_num_field_elements,
        &setup_seed,
    )
    .map_err(|err| {
        AkitaError::InvalidSetup(format!("failed to deserialize public matrix: {err}"))
    })?;
    let mut trailing = [0u8; 1];
    if reader
        .read(&mut trailing)
        .map_err(|err| AkitaError::InvalidSetup(format!("failed to check matrix EOF: {err}")))?
        != 0
    {
        return Err(AkitaError::InvalidSetup(format!(
            "cached public matrix has trailing bytes starting with 0x{:02x}",
            trailing[0]
        )));
    }
    expanded.descriptor = AkitaSetupDescriptor {
        max_num_vars,
        max_num_batched_polys,
        num_field_elements: expanded.shared_matrix().num_field_elements(),
        setup_seed: setup_seed.clone(),
    };
    validate_cached_matrix::<F>(&expanded)?;

    let prefix_registry_path = get_prefix_registry_storage_path::<F>(requirements)
        .ok_or_else(|| AkitaError::InvalidSetup("failed to determine registry path".to_string()))?;
    let prefix_slots = if prefix_registry_path.exists() {
        let file = fs::File::open(&prefix_registry_path).map_err(|err| {
            AkitaError::InvalidSetup(format!("failed to open setup-prefix registry: {err}"))
        })?;
        let mut reader = std::io::BufReader::new(file);
        let slots = SetupPrefixProverRegistry::<F>::deserialize_with_mode(
            &mut reader,
            Compress::Yes,
            Validate::Yes,
            &(),
        )
        .map_err(|err| {
            AkitaError::InvalidSetup(format!(
                "failed to deserialize setup-prefix registry: {err}"
            ))
        })?;
        if reader.read(&mut trailing).map_err(|err| {
            AkitaError::InvalidSetup(format!("failed to check registry EOF: {err}"))
        })? != 0
        {
            return Err(AkitaError::InvalidSetup(format!(
                "cached setup-prefix registry has trailing bytes starting with 0x{:02x}",
                trailing[0]
            )));
        }
        slots
    } else {
        SetupPrefixProverRegistry::new(setup_seed)
    };
    if prefix_slots.setup_seed() != &expanded.descriptor().setup_seed {
        return Err(AkitaError::InvalidSetup(
            "cached setup-prefix registry belongs to a different public matrix".to_string(),
        ));
    }

    let mut setup = AkitaProverSetup {
        expanded: Arc::new(expanded),
        prefix_slots,
    };
    if recursive_prefixes::validate_prefix_registry_complete(
        &setup.prefix_slots,
        &requirements.prefix_slot_ids,
    )
    .is_err()
    {
        setup.prefix_slots =
            SetupPrefixProverRegistry::new(setup.expanded.descriptor().setup_seed.clone());
        recursive_prefixes::populate_required_setup_prefix_slots(
            &mut setup,
            &requirements.prefix_slot_ids,
        )?;
        save_prover_setup::<F>(&setup, requirements)?;
    }

    tracing::info!(
        "Loaded covering public matrix for max_num_vars={max_num_vars}, max_num_batched_polys={max_num_batched_polys}"
    );
    Ok(setup)
}

#[cfg(feature = "disk-persistence")]
fn deserialize_cached_public_matrix<F: Field + Valid + AkitaDeserialize<Context = ()>>(
    reader: &mut impl Read,
    minimum_num_field_elements: usize,
    expected_setup_seed: &AkitaSetupSeed,
) -> Result<AkitaExpandedSetup<F>, SerializationError> {
    let setup_seed =
        AkitaSetupSeed::deserialize_with_mode(&mut *reader, Compress::Yes, Validate::Yes, &())?;
    if &setup_seed != expected_setup_seed {
        return Err(SerializationError::InvalidData(
            "cached public matrix identity does not match its lineage key".to_string(),
        ));
    }
    let num_field_elements =
        usize::deserialize_with_mode(&mut *reader, Compress::Yes, Validate::Yes, &())?;
    if num_field_elements < minimum_num_field_elements {
        return Err(SerializationError::InvalidData(
            "cached public matrix prefix does not cover the requested field capacity".to_string(),
        ));
    }
    let shared_matrix = FlatMatrix::<F>::deserialize_with_expected_shape(
        &mut *reader,
        Compress::Yes,
        Validate::Yes,
        num_field_elements,
        num_field_elements,
    )?;
    Ok(
        AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
            AkitaSetupDescriptor {
                max_num_vars: 0,
                max_num_batched_polys: 1,
                num_field_elements,
                setup_seed,
            },
            shared_matrix,
        ),
    )
}

#[cfg(feature = "disk-persistence")]
fn validate_cached_matrix<F: Field + CanonicalEncoding + Valid>(
    setup: &AkitaExpandedSetup<F>,
) -> Result<(), AkitaError> {
    setup
        .check()
        .map_err(|e| AkitaError::InvalidSetup(format!("cached setup matrix validation: {e}")))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_config::proof_optimized::fp128;
    use akita_config::{CommitmentConfig, TrustedScheduleCatalog};
    use akita_serialization::{AkitaDeserialize, AkitaSerialize};
    use akita_types::SetupMatrixCapacity;
    #[cfg(feature = "disk-persistence")]
    use jolt_field::Zero;

    type Cfg = fp128::Dense;
    type TestF = fp128::Field;

    fn schedules() -> TrustedScheduleCatalog<Cfg> {
        akita_config::test_support::workspace_schedule_catalog::<Cfg>()
            .expect("workspace schedule catalog")
    }

    fn requirements_at(max_num_vars: usize, max_num_batched_polys: usize) -> SetupRequirements {
        SetupRequirements::from_catalog::<Cfg>(&schedules(), max_num_vars, max_num_batched_polys)
            .expect("workspace setup requirements")
    }

    #[derive(Clone)]
    struct WrongModulusProfileConfig;

    impl CommitmentConfig for WrongModulusProfileConfig {
        type Field = TestF;
        type ExtField = <Cfg as CommitmentConfig>::ExtField;

        fn schedule_family_name() -> &'static str {
            "test_wrong_modulus_profile"
        }

        const RING_DIMENSION_SCHEDULE_MODE: akita_config::RingDimensionScheduleMode =
            Cfg::RING_DIMENSION_SCHEDULE_MODE;

        fn decomposition() -> akita_types::DecompositionParams {
            Cfg::decomposition()
        }

        fn ring_challenge_config(
            d: usize,
        ) -> Result<akita_challenges::SparseChallengeConfig, AkitaError> {
            Cfg::ring_challenge_config(d)
        }

        fn sis_modulus_profile() -> akita_types::SisModulusProfileId {
            akita_types::SisModulusProfileId::Q64Offset59
        }

        fn opening_basis_range() -> (u32, u32) {
            Cfg::opening_basis_range()
        }

        fn inner_basis_range() -> (u32, u32) {
            Cfg::inner_basis_range()
        }

        fn committed_source_class() -> akita_types::sis::CommittedSourceClass {
            Cfg::committed_source_class()
        }
    }

    #[test]
    fn expanded_setup_roundtrips_and_derives_same_verifier() {
        let prover_setup = new_prover_setup::<TestF>(&requirements_at(14, 3)).unwrap();
        let capacity = SetupMatrixCapacity {
            num_field_elements: prover_setup.expanded.shared_matrix().num_field_elements() / 2,
        };
        let verifier_setup = prover_setup.to_verifier_setup(capacity).unwrap();

        let mut bytes = Vec::new();
        verifier_setup
            .expanded()
            .serialize_compressed(&mut bytes)
            .unwrap();
        let decoded = AkitaExpandedSetup::<TestF>::deserialize_compressed(&bytes[..], &()).unwrap();

        assert_eq!(decoded, verifier_setup.expanded().as_ref().clone());
        assert_eq!(decoded.descriptor().max_num_batched_polys, 3);

        decoded.check().unwrap();
        let decoded_prover = AkitaProverSetup {
            prefix_slots: akita_cpu_backend::SetupPrefixProverRegistry::new(
                decoded.descriptor().setup_seed.clone(),
            ),
            expanded: std::sync::Arc::new(decoded.clone()),
        };
        let derived_verifier = decoded_prover.to_verifier_setup(capacity).unwrap();
        assert_eq!(derived_verifier, verifier_setup);
        assert_eq!(
            verifier_setup
                .expanded()
                .shared_matrix()
                .num_field_elements(),
            capacity.num_field_elements
        );
    }

    #[test]
    fn setup_accepts_field_coupled_presets() {
        // The D64 catalog begins at nv=14, the first singleton shape with the
        // required root and suffix folds.
        new_prover_setup::<fp128::Field>(&requirements_at(14, 1))
            .expect("fp128 dense preset should accept the default field");
    }

    #[test]
    fn setup_rejects_a_mismatched_field_profile_before_materialization() {
        let error =
            TrustedScheduleCatalog::<WrongModulusProfileConfig>::new(schedules().catalog().clone())
                .expect_err(
                    "field modulus and SIS profile must agree before setup materialization",
                );
        assert!(error.to_string().contains("does not match field modulus"));
    }

    #[cfg(feature = "disk-persistence")]
    mod disk_persistence {
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

                let prover_setup =
                    new_prover_setup::<TestF>(&requirements_at(MAX_VARS, 1)).unwrap();

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
        fn prefix_slots_roundtrip_through_setup_cache() {
            with_test_cache_dir("prefix-slots", || {
                const MAX_VARS: usize = 14;
                cleanup_setup_file_shape(MAX_VARS, 1);
                let catalog = schedules();
                let mut setup = new_prover_setup::<TestF>(&requirements_at(MAX_VARS, 1)).unwrap();
                let row = catalog
                    .resolve_key(&akita_types::AkitaScheduleLookupKey::single(
                        akita_types::PolynomialGroupLayout::new(MAX_VARS, 1),
                    ))
                    .unwrap();
                let params = &row.schedule().root.params;
                let n_prefix =
                    (params.d_a() * params.outer_slice_count().get()).next_power_of_two();
                let prefix =
                    akita_types::setup_prefix_precommitted_params(params, n_prefix).unwrap();
                let id = akita_types::scheduled_setup_prefix(n_prefix, prefix)
                    .slot_id()
                    .unwrap();
                let mut requirements = requirements_at(MAX_VARS, 1);
                requirements.prefix_slot_ids = vec![id.clone()];
                let backend = akita_cpu_backend::CpuBackend::<
                    TestF,
                    <Cfg as CommitmentConfig>::ExtField,
                >::new(setup.expanded.clone())
                .unwrap();
                setup.prefix_slots = backend.export_setup_prefixes(&[id]).unwrap();
                save_prover_setup::<TestF>(&setup, &requirements).unwrap();

                let loaded = load_prover_setup::<TestF>(&requirements).unwrap();
                assert_eq!(loaded.prefix_slots, setup.prefix_slots);

                cleanup_setup_file_shape(MAX_VARS, 1);
                if let Some(path) = get_prefix_registry_storage_path::<TestF>(&requirements) {
                    let _ = fs::remove_file(path);
                }
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
                    .matrix_capacity
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
                    requirements_at(SMALL_VARS, 1).matrix_capacity,
                )
                .unwrap();
                let large = AkitaProverSetup::generate_with_capacity(
                    LARGE_VARS,
                    1,
                    requirements_at(LARGE_VARS, 1).matrix_capacity,
                )
                .unwrap();
                let large_fields = large.expanded.shared_matrix().num_field_elements();
                let barrier = Arc::new(std::sync::Barrier::new(3));
                std::thread::scope(|scope| {
                    let first_barrier = Arc::clone(&barrier);
                    scope.spawn(move || {
                        first_barrier.wait();
                        save_prover_setup::<TestF>(&small, &requirements_at(SMALL_VARS, 1))
                            .unwrap();
                    });
                    let second_barrier = Arc::clone(&barrier);
                    scope.spawn(move || {
                        second_barrier.wait();
                        save_prover_setup::<TestF>(&large, &requirements_at(LARGE_VARS, 1))
                            .unwrap();
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

                let prover_setup =
                    new_prover_setup::<TestF>(&requirements_at(MAX_VARS, 1)).unwrap();
                let total = prover_setup.expanded.shared_matrix().num_field_elements();
                let corrupt = AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
                    prover_setup.expanded.descriptor().clone(),
                    FlatMatrix::from_flat_data(vec![TestF::zero(); total]),
                );
                let path =
                    get_public_matrix_storage_path::<TestF>(&sample_akita_setup_seed()).unwrap();
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
                let path =
                    get_public_matrix_storage_path::<TestF>(&sample_akita_setup_seed()).unwrap();
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

                        let fresh_setup =
                            new_prover_setup::<TestF>(&requirements_at(MAX_VARS, 1)).unwrap();

                        let disk_setup =
                            load_prover_setup::<TestF>(&requirements_at(MAX_VARS, 1)).unwrap();

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
    }
}
