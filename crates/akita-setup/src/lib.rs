//! Config-backed prover setup construction.
//!
//! With `disk-persistence`, the public field prefix is stored by field and
//! [`akita_types::AkitaSetupSeed`], separately from schedule-bound setup-prefix
//! registries. Backend NTT caches are never persisted.

mod recursive_prefixes;

use akita_config::CommitmentConfig;
use akita_error::AkitaError;
use akita_prover::AkitaProverSetup;
use akita_serialization::{AkitaDeserialize, AkitaSerialize, Valid};
#[cfg(feature = "disk-persistence")]
use akita_serialization::{Compress, SerializationError, Validate};
#[cfg(any(feature = "disk-persistence", test))]
use akita_types::AkitaExpandedSetup;
use akita_types::AkitaSetupSeed;
#[cfg(feature = "disk-persistence")]
use akita_types::{
    detect_field_modulus, digest_effective_schedule, setup_seed_digest, AkitaScheduleLookupKey,
    AkitaSetupDescriptor, FlatMatrix, PolynomialGroupLayout, SetupPrefixProverRegistry,
};
use jolt_field::Unreduced;
use jolt_field::{CanonicalEncoding, Field};
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

#[cfg(feature = "disk-persistence")]
fn validate_loaded_prefix_registry_coverage<F, Cfg>(
    setup: &AkitaProverSetup<F>,
    max_num_vars: usize,
    max_num_batched_polys: usize,
) -> Result<(), AkitaError>
where
    F: Field,
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

/// Lowercase hex, for cache identities and setup-identity logs.
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// Construct prover setup from a root commitment config and public setup seed.
///
/// `akita-config` owns setup sizing policy; this crate owns optional disk
/// persistence; `akita-prover` owns the concrete setup artifact and
/// matrix expansion.
///
/// `setup_seed` is the public identity of the deterministic matrix stream.
/// Prover and verifier must supply the same seed to agree on a setup, and the
/// seed is trusted deployment configuration on both sides: it must never be
/// taken from a proof. Use [`akita_types::AkitaSetupSeed::DEFAULT`] for the
/// stable protocol default, or `AkitaSetupSeed::from_rng` once at deployment
/// time to mint a deployment-specific identity.
///
/// # Errors
///
/// Returns an error if the requested setup capacity is invalid or setup
/// expansion fails.
#[tracing::instrument(skip_all, name = "new_prover_setup")]
pub fn new_prover_setup<F, Cfg>(
    max_num_vars: usize,
    max_num_batched_polys: usize,
    setup_seed: AkitaSetupSeed,
) -> Result<AkitaProverSetup<F>, AkitaError>
where
    F: Field
        + CanonicalEncoding
        + Unreduced
        + Valid
        + AkitaSerialize
        + AkitaDeserialize<Context = ()>
        + 'static,
    Cfg: CommitmentConfig<Field = F>,
{
    akita_config::validate_config_policy::<Cfg>()?;
    if max_num_batched_polys == 0 {
        return Err(AkitaError::InvalidSetup(
            "max_num_batched_polys must be at least 1".to_string(),
        ));
    }
    // Nothing in a failed verification names the seed, so this is the only
    // record that makes a prover/verifier seed mismatch attributable.
    tracing::info!(
        setup_seed = %hex(&setup_seed.seed),
        derivation = ?setup_seed.derivation,
        max_num_vars,
        max_num_batched_polys,
        "building Akita setup for public seed identity"
    );
    #[cfg(feature = "disk-persistence")]
    {
        match load_prover_setup::<F, Cfg>(max_num_vars, max_num_batched_polys, &setup_seed) {
            Ok(setup) => {
                validate_loaded_prefix_registry_coverage::<F, Cfg>(
                    &setup,
                    max_num_vars,
                    max_num_batched_polys,
                )?;
                tracing::info!("Loaded setup from disk; backend preparation is explicit");
                return Ok(setup);
            }
            Err(err) => {
                tracing::warn!("Failed to load cached setup: {err}; regenerating");
            }
        }
    }

    let setup_capacity = Cfg::setup_matrix_capacity(max_num_vars, max_num_batched_polys)?;

    let mut setup = AkitaProverSetup::generate_with_capacity(
        max_num_vars,
        max_num_batched_polys,
        setup_capacity,
        setup_seed,
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

/// Name the prefix-registry cache entry for one config, shape, and setup seed.
///
/// Every input that must invalidate the entry is folded into one digest rather
/// than concatenated into the name, which bounds the name length by
/// construction so no component needs truncating or sanitizing. `nv` and
/// `batch` stay plaintext to keep a cache directory readable.
#[cfg(feature = "disk-persistence")]
fn prefix_registry_cache_file_name<Cfg: CommitmentConfig>(
    max_num_vars: usize,
    max_num_batched_polys: usize,
    setup_seed: &AkitaSetupSeed,
) -> Result<String, AkitaError> {
    let mut identity = Vec::new();
    identity.extend_from_slice(&detect_field_modulus::<Cfg::Field>()?.to_le_bytes());

    // Length-prefixed: the config family is the one variable-width component,
    // so without it the boundary with the following fields is ambiguous.
    let family = std::any::type_name::<Cfg>();
    identity.extend_from_slice(&(family.len() as u64).to_le_bytes());
    identity.extend_from_slice(family.as_bytes());

    // Fingerprint the resolved layout so a replanned schedule at the same
    // lookup key lands on a different entry. `digest_effective_schedule` covers
    // the full per-level params, including the SIS-derived `n_a`/`n_b`/`n_d`
    // ranks. A catalog miss is its own distinct case, not a missing field.
    let schedule_lookup_key = PolynomialGroupLayout::new(max_num_vars, max_num_batched_polys);
    match Cfg::resolve_catalog_row_for_key(&AkitaScheduleLookupKey::single(schedule_lookup_key)) {
        Ok(schedule) => {
            identity.push(1);
            identity.extend_from_slice(&digest_effective_schedule(schedule.schedule()));
        }
        Err(_) => identity.push(0),
    }

    // Without the seed, two seeds at the same shape would target the same path
    // and each run would invalidate the other's registry.
    identity.extend_from_slice(
        &setup_seed_digest(setup_seed)
            .map_err(|err| AkitaError::InvalidSetup(format!("public matrix identity: {err}")))?,
    );
    identity.extend_from_slice(&(max_num_vars as u64).to_le_bytes());
    identity.extend_from_slice(&(max_num_batched_polys as u64).to_le_bytes());

    let digest = akita_types::instance_descriptor::digest_descriptor_bytes(&identity);
    Ok(format!(
        "akita_prefix_v4_nv{max_num_vars}_batch{max_num_batched_polys}_id{}.registry",
        hex(&digest)
    ))
}

#[cfg(feature = "disk-persistence")]
fn public_matrix_cache_file_name<F: Field + CanonicalEncoding>(
    setup_seed: &AkitaSetupSeed,
) -> Result<String, AkitaError> {
    let digest = setup_seed_digest(setup_seed)
        .map_err(|err| AkitaError::InvalidSetup(format!("public matrix identity: {err}")))?;
    let modulus = detect_field_modulus::<F>()?;
    Ok(format!(
        "akita_flat_v3_q{modulus:032x}_id{}.matrix",
        hex(&digest)
    ))
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
pub(crate) fn get_prefix_registry_storage_path<Cfg: CommitmentConfig>(
    max_num_vars: usize,
    max_num_batched_polys: usize,
    setup_seed: &AkitaSetupSeed,
) -> Result<PathBuf, AkitaError> {
    let mut path = cache_directory().ok_or_else(|| {
        AkitaError::InvalidSetup("could not determine storage directory".to_string())
    })?;
    path.push(prefix_registry_cache_file_name::<Cfg>(
        max_num_vars,
        max_num_batched_polys,
        setup_seed,
    )?);
    Ok(path)
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
    expanded.setup_seed().serialize_compressed(&mut *writer)?;
    expanded
        .shared_matrix()
        .num_field_elements()
        .serialize_compressed(&mut *writer)?;
    expanded.shared_matrix().serialize_compressed(writer)
}

#[cfg(feature = "disk-persistence")]
pub(crate) fn save_prover_setup<
    F: Field + CanonicalEncoding + Valid + AkitaSerialize + AkitaDeserialize<Context = ()>,
    Cfg: CommitmentConfig<Field = F>,
>(
    setup: &AkitaProverSetup<F>,
    max_num_vars: usize,
    max_num_batched_polys: usize,
) -> Result<(), AkitaError> {
    // `setup` was just derived inside this crate. Re-deriving and comparing
    // every field element here would repeat the full setup-generation pass;
    // public-matrix cache bytes are deterministically validated on load.
    // Prefix-registry provenance is a separate setup-validation boundary.
    let public_matrix_path = get_public_matrix_storage_path::<F>(setup.setup_seed())?;
    let prefix_registry_path = get_prefix_registry_storage_path::<Cfg>(
        max_num_vars,
        max_num_batched_polys,
        setup.setup_seed(),
    )?;

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
            let existing =
                deserialize_cached_public_matrix::<F>(&mut reader, 0, setup.setup_seed());
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
    F: Field + Valid + CanonicalEncoding + AkitaSerialize + AkitaDeserialize<Context = ()> + 'static,
    Cfg: CommitmentConfig<Field = F>,
>(
    max_num_vars: usize,
    max_num_batched_polys: usize,
    setup_seed: &AkitaSetupSeed,
) -> Result<AkitaProverSetup<F>, AkitaError> {
    let public_matrix_path = get_public_matrix_storage_path::<F>(setup_seed)?;
    if !public_matrix_path.exists() {
        return Err(AkitaError::InvalidSetup(format!(
            "public matrix cache not found at {}",
            public_matrix_path.display()
        )));
    }
    let required_num_field_elements =
        Cfg::setup_matrix_capacity(max_num_vars, max_num_batched_polys)?.num_field_elements;
    let file = fs::File::open(&public_matrix_path).map_err(|err| {
        AkitaError::InvalidSetup(format!("failed to open public matrix cache: {err}"))
    })?;
    let mut reader = std::io::BufReader::new(file);
    let mut expanded =
        deserialize_cached_public_matrix::<F>(&mut reader, required_num_field_elements, setup_seed)
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

    let prefix_registry_path =
        get_prefix_registry_storage_path::<Cfg>(max_num_vars, max_num_batched_polys, setup_seed)?;
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
        SetupPrefixProverRegistry::new(setup_seed.clone())
    };
    if prefix_slots.setup_seed() != expanded.setup_seed() {
        return Err(AkitaError::InvalidSetup(
            "cached setup-prefix registry belongs to a different public matrix".to_string(),
        ));
    }

    let mut setup = AkitaProverSetup {
        expanded: Arc::new(expanded),
        prefix_slots,
    };
    if validate_loaded_prefix_registry_coverage::<F, Cfg>(
        &setup,
        max_num_vars,
        max_num_batched_polys,
    )
    .is_err()
    {
        setup.prefix_slots = SetupPrefixProverRegistry::new(setup.setup_seed().clone());
        recursive_prefixes::populate_required_setup_prefix_slots::<F, Cfg>(
            &mut setup,
            max_num_vars,
            max_num_batched_polys,
        )?;
        save_prover_setup::<F, Cfg>(&setup, max_num_vars, max_num_batched_polys)?;
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
    use akita_serialization::{AkitaDeserialize, AkitaSerialize};
    use akita_types::SetupMatrixCapacity;
    #[cfg(feature = "disk-persistence")]
    use jolt_field::Zero;

    type Cfg = fp128::Dense;
    type TestF = fp128::Field;

    #[derive(Clone)]
    struct WrongModulusProfileConfig;

    impl CommitmentConfig for WrongModulusProfileConfig {
        type Field = TestF;
        type ExtField = <Cfg as CommitmentConfig>::ExtField;

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

        fn setup_matrix_capacity(
            _max_num_vars: usize,
            _max_num_batched_polys: usize,
        ) -> Result<akita_types::SetupMatrixCapacity, AkitaError> {
            panic!("invalid config reached setup capacity materialization")
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
        let prover_setup = new_prover_setup::<TestF, Cfg>(14, 3, AkitaSetupSeed::DEFAULT).unwrap();
        let capacity = SetupMatrixCapacity {
            num_field_elements: prover_setup.expanded.shared_matrix().num_field_elements() / 2,
        };
        let verifier_setup = prover_setup.to_verifier_setup(capacity).unwrap();

        let mut bytes = Vec::new();
        prover_setup
            .expanded
            .serialize_compressed(&mut bytes)
            .unwrap();
        let decoded = AkitaExpandedSetup::<TestF>::deserialize_compressed(&bytes[..], &()).unwrap();

        assert_eq!(decoded, prover_setup.expanded.as_ref().clone());
        assert_eq!(decoded.descriptor().max_num_batched_polys, 3);

        let decoded_prover = AkitaProverSetup::from_validated_expanded(decoded.clone()).unwrap();
        let derived_verifier = decoded_prover.to_verifier_setup(capacity).unwrap();
        assert_eq!(derived_verifier, verifier_setup);
        assert_eq!(
            verifier_setup.expanded.shared_matrix().num_field_elements(),
            capacity.num_field_elements
        );
    }

    #[test]
    fn setup_accepts_field_coupled_presets() {
        // The D64 catalog begins at nv=14, the first singleton shape with the
        // required root and suffix folds.
        new_prover_setup::<fp128::Field, fp128::Dense>(14, 1, AkitaSetupSeed::DEFAULT)
            .expect("fp128 dense preset should accept the default field");
    }

    #[test]
    fn setup_rejects_a_mismatched_field_profile_before_materialization() {
        let error =
            new_prover_setup::<TestF, WrongModulusProfileConfig>(14, 1, AkitaSetupSeed::DEFAULT)
                .expect_err("field modulus and SIS profile must agree");
        assert!(error.to_string().contains("does not match field modulus"));
    }

    #[cfg(feature = "disk-persistence")]
    mod disk_persistence {
        const TEST_D: usize = 64;
        use super::*;
        use std::fs;
        use std::sync::{LazyLock, Mutex};

        static DISK_TEST_ENV_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

        fn cleanup_setup_file_shape(max_num_vars: usize, max_num_batched_polys: usize) {
            if let Ok(path) = get_prefix_registry_storage_path::<Cfg>(
                max_num_vars,
                max_num_batched_polys,
                &AkitaSetupSeed::DEFAULT,
            ) {
                let _ = fs::remove_file(path);
            }
            if let Ok(path) = get_public_matrix_storage_path::<TestF>(&AkitaSetupSeed::DEFAULT) {
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
                    new_prover_setup::<TestF, Cfg>(MAX_VARS, 1, AkitaSetupSeed::DEFAULT).unwrap();

                let loaded =
                    load_prover_setup::<TestF, Cfg>(MAX_VARS, 1, &AkitaSetupSeed::DEFAULT).unwrap();
                assert_eq!(loaded.expanded, prover_setup.expanded);

                cleanup_setup_file_shape(MAX_VARS, 1);
            });
        }

        #[test]
        fn cache_file_name_stays_below_common_component_limits() {
            let name = prefix_registry_cache_file_name::<Cfg>(16, 4, &AkitaSetupSeed::DEFAULT)
                .expect("registry cache name");
            // Hashing the identity bounds the name by construction. Keep the
            // bound tight so re-adding an unbounded component fails here rather
            // than at the 255-byte filesystem limit.
            assert!(
                name.len() < 128,
                "setup cache file name must stay far below the 255-byte component limit, got {}: {name}",
                name.len()
            );
        }

        #[test]
        fn cache_file_names_use_current_namespaces() {
            let registry = prefix_registry_cache_file_name::<Cfg>(16, 4, &AkitaSetupSeed::DEFAULT)
                .expect("registry cache name");
            assert!(registry.contains("prefix_v4_"), "cache name: {registry}");
            let matrix = public_matrix_cache_file_name::<TestF>(&AkitaSetupSeed::DEFAULT)
                .expect("matrix cache name");
            assert!(matrix.contains("flat_v3_"), "cache name: {matrix}");
        }

        #[test]
        fn prefix_registry_cache_names_separate_setup_seeds() {
            let other_seed = AkitaSetupSeed::shake256_paged_v1([7u8; 32]);
            let default_name =
                prefix_registry_cache_file_name::<Cfg>(16, 4, &AkitaSetupSeed::DEFAULT)
                    .expect("default registry cache name");
            let other_name = prefix_registry_cache_file_name::<Cfg>(16, 4, &other_seed)
                .expect("other registry cache name");

            assert_ne!(
                default_name, other_name,
                "two seeds at one shape must not share a registry cache path"
            );
        }

        #[test]
        fn config_backed_cache_does_not_apply_generic_setup_decode_limit() {
            let setup_seed = AkitaSetupSeed::DEFAULT;
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
                use akita_types::{
                    scheduled_setup_prefix, AkitaCommitmentHint, CompressionChainPlan,
                    CompressionChainWitness, GroupCommitPhaseParams, GroupOpenPhaseParams,
                    InnerCommitMatrixParams, OuterCommitMatrixParams, PackedNegativeBinary,
                    PolynomialGroupLayout, RingVec, SetupPrefixPublicCommitment, SetupPrefixSlot,
                    SisModulusProfileId, SisTableDigest, SisTableKey, DEFAULT_SIS_SECURITY_POLICY,
                };

                const MAX_VARS: usize = 14;

                cleanup_setup_file_shape(MAX_VARS, 1);

                let mut setup =
                    new_prover_setup::<TestF, Cfg>(MAX_VARS, 1, AkitaSetupSeed::DEFAULT).unwrap();
                let inner_commit_matrix = InnerCommitMatrixParams::try_new_with_min_rank(
                    SisTableKey {
                        policy: DEFAULT_SIS_SECURITY_POLICY,
                        table_digest: SisTableDigest::CURRENT,
                        modulus_profile: SisModulusProfileId::Q128OffsetA7F7,
                        role: akita_types::SisMatrixRole::Inner,
                        ring_dimension: u32::try_from(TEST_D).expect("test ring dimension"),
                        coeff_linf_bound: 32_767,
                    },
                    1,
                )
                .expect("audited prefix A matrix");
                let outer_commit_matrix = OuterCommitMatrixParams::try_new_with_min_rank(
                    SisTableKey {
                        policy: DEFAULT_SIS_SECURITY_POLICY,
                        table_digest: SisTableDigest::CURRENT,
                        modulus_profile: SisModulusProfileId::Q128OffsetA7F7,
                        role: akita_types::SisMatrixRole::Outer,
                        ring_dimension: u32::try_from(TEST_D).expect("test ring dimension"),
                        coeff_linf_bound: 3,
                    },
                    inner_commit_matrix.output_rank(),
                )
                .expect("audited prefix B matrix");
                let commitment_rows = outer_commit_matrix.output_rank();
                let commitment_params = GroupOpenPhaseParams {
                    setup_natural_len: None,
                    profile: GroupCommitPhaseParams {
                        version: GroupCommitPhaseParams::VERSION,
                        group: PolynomialGroupLayout::singleton(TEST_D.trailing_zeros() as usize),
                        blocks: akita_types::BlockGeometry::new(1, 1, 1),
                        outer_slice_count: akita_types::CommitmentSliceCount::ONE,
                        inner: akita_types::RoleParams::new(
                            akita_types::GadgetDigits::new(1, 1),
                            inner_commit_matrix,
                        ),
                        outer: akita_types::RoleParams::new(
                            akita_types::GadgetDigits::new(1, 1),
                            outer_commit_matrix,
                        ),
                    },
                    opening: akita_types::GroupOpeningPlan::evaluation_trace(
                        akita_challenges::SparseChallengeConfig::pm1_only(0),
                        1,
                        1,
                        1,
                    ),
                };
                let id = scheduled_setup_prefix(TEST_D, commitment_params)
                    .slot_id()
                    .expect("setup prefix group");
                let compression_plan = CompressionChainPlan::for_complete_source(
                    commitment_params.profile.outer.matrix.sis_modulus_profile(),
                    commitment_params.profile.outer.matrix.output_rank() * TEST_D,
                )
                .expect("compression plan");
                let compression_stages = compression_plan
                    .maps()
                    .iter()
                    .map(|map| {
                        PackedNegativeBinary::from_bytes(*map, vec![0; map.packed_digit_bytes()])
                    })
                    .collect::<Result<Vec<_>, _>>()
                    .expect("zero compression stages");
                let compression_witness =
                    CompressionChainWitness::new(compression_plan, compression_stages)
                        .expect("zero compression witness");
                let compression_quotients = compression_witness
                    .plan()
                    .maps()
                    .iter()
                    .map(|map| {
                        RingVec::from_coeffs_with_ring_dim(
                            vec![TestF::zero(); map.output_coefficients()],
                            map.ring_dimension(),
                        )
                        .expect("zero compression quotient")
                    })
                    .collect::<Vec<_>>();
                let terminal_map = compression_witness
                    .plan()
                    .maps()
                    .last()
                    .expect("terminal compression map");
                let commitment_row =
                    RingVec::from_coeffs(vec![TestF::zero(); terminal_map.output_coefficients()]);
                let hint = AkitaCommitmentHint::singleton_with_outer_compression(
                    RingVec::from_coeffs_with_ring_dim(vec![TestF::zero(); TEST_D], TEST_D)
                        .expect("inner rows"),
                    &compression_witness,
                    &compression_quotients,
                )
                .expect("hint");
                setup
                    .prefix_slots
                    .insert(SetupPrefixSlot {
                        id,
                        commitment: SetupPrefixPublicCommitment {
                            rows: vec![commitment_row; commitment_rows],
                        },
                        hint,
                    })
                    .unwrap();
                save_prover_setup::<TestF, Cfg>(&setup, MAX_VARS, 1).unwrap();

                let loaded =
                    load_prover_setup::<TestF, Cfg>(MAX_VARS, 1, &AkitaSetupSeed::DEFAULT).unwrap();
                assert_eq!(loaded.prefix_slots, setup.prefix_slots);

                cleanup_setup_file_shape(MAX_VARS, 1);
            });
        }

        #[test]
        fn setup_uses_cache_on_second_call() {
            with_test_cache_dir("second-call", || {
                const MAX_VARS: usize = 14;

                cleanup_setup_file_shape(MAX_VARS, 1);

                let first =
                    new_prover_setup::<TestF, Cfg>(MAX_VARS, 1, AkitaSetupSeed::DEFAULT).unwrap();

                let second =
                    new_prover_setup::<TestF, Cfg>(MAX_VARS, 1, AkitaSetupSeed::DEFAULT).unwrap();

                assert_eq!(first.expanded, second.expanded);

                cleanup_setup_file_shape(MAX_VARS, 1);
            });
        }

        #[test]
        fn two_seeds_share_a_cache_directory_without_invalidating_each_other() {
            with_test_cache_dir("seed-separation", || {
                const MAX_VARS: usize = 14;
                let other_seed = AkitaSetupSeed::shake256_paged_v1([0x5a; 32]);

                let cleanup = || {
                    for seed in [AkitaSetupSeed::DEFAULT, other_seed.clone()] {
                        if let Ok(path) =
                            get_prefix_registry_storage_path::<Cfg>(MAX_VARS, 1, &seed)
                        {
                            let _ = fs::remove_file(path);
                        }
                        if let Ok(path) = get_public_matrix_storage_path::<TestF>(&seed) {
                            let _ = fs::remove_file(path);
                        }
                    }
                };
                cleanup();

                let default_setup =
                    new_prover_setup::<TestF, Cfg>(MAX_VARS, 1, AkitaSetupSeed::DEFAULT).unwrap();
                let other_setup =
                    new_prover_setup::<TestF, Cfg>(MAX_VARS, 1, other_seed.clone()).unwrap();
                assert_ne!(
                    default_setup.expanded.shared_matrix().as_field_slice(),
                    other_setup.expanded.shared_matrix().as_field_slice()
                );

                // Each seed keeps its own entries, so reloading either returns
                // the matrix derived from that seed rather than from whichever
                // generation ran last.
                let default_reloaded =
                    load_prover_setup::<TestF, Cfg>(MAX_VARS, 1, &AkitaSetupSeed::DEFAULT).unwrap();
                let other_reloaded =
                    load_prover_setup::<TestF, Cfg>(MAX_VARS, 1, &other_seed).unwrap();
                assert_eq!(default_reloaded.expanded, default_setup.expanded);
                assert_eq!(other_reloaded.expanded, other_setup.expanded);

                cleanup();
            });
        }

        /// The seed embedded in a registry file is authoritative, not the path
        /// it was found at. Plant a foreign-seed registry where the requested
        /// seed resolves to prove the embedded check is what rejects it.
        #[test]
        fn load_rejects_a_registry_belonging_to_another_seed() {
            with_test_cache_dir("foreign-registry", || {
                const MAX_VARS: usize = 14;
                let foreign_seed = AkitaSetupSeed::shake256_paged_v1([0x3c; 32]);

                cleanup_setup_file_shape(MAX_VARS, 1);

                new_prover_setup::<TestF, Cfg>(MAX_VARS, 1, AkitaSetupSeed::DEFAULT).unwrap();
                let registry_path =
                    get_prefix_registry_storage_path::<Cfg>(MAX_VARS, 1, &AkitaSetupSeed::DEFAULT)
                        .unwrap();
                atomic_write_cache(&registry_path, |writer| {
                    SetupPrefixProverRegistry::<TestF>::new(foreign_seed.clone())
                        .serialize_compressed(writer)
                })
                .unwrap();

                let err = load_prover_setup::<TestF, Cfg>(MAX_VARS, 1, &AkitaSetupSeed::DEFAULT)
                    .expect_err("a registry from another seed must be rejected");
                assert!(
                    err.to_string()
                        .contains("belongs to a different public matrix"),
                    "unexpected rejection reason: {err}"
                );

                cleanup_setup_file_shape(MAX_VARS, 1);
            });
        }

        #[test]
        fn larger_public_prefix_covers_smaller_provisioning_request() {
            with_test_cache_dir("covering-prefix", || {
                const LARGE_VARS: usize = 15;
                const SMALL_VARS: usize = 14;

                cleanup_setup_file_shape(LARGE_VARS, 1);
                if let Ok(path) =
                    get_prefix_registry_storage_path::<Cfg>(SMALL_VARS, 1, &AkitaSetupSeed::DEFAULT)
                {
                    let _ = fs::remove_file(path);
                }

                let large =
                    new_prover_setup::<TestF, Cfg>(LARGE_VARS, 1, AkitaSetupSeed::DEFAULT).unwrap();
                let large_fields = large.expanded.shared_matrix().num_field_elements();
                let small_required = Cfg::setup_matrix_capacity(SMALL_VARS, 1)
                    .unwrap()
                    .num_field_elements;
                assert!(large_fields >= small_required);

                let covered =
                    new_prover_setup::<TestF, Cfg>(SMALL_VARS, 1, AkitaSetupSeed::DEFAULT).unwrap();
                assert_eq!(
                    covered.expanded.shared_matrix().num_field_elements(),
                    large_fields
                );
                assert_eq!(covered.setup_seed(), large.setup_seed());
                assert_eq!(covered.expanded.descriptor().max_num_vars, SMALL_VARS);
                assert_eq!(covered.expanded.descriptor().max_num_batched_polys, 1);

                cleanup_setup_file_shape(LARGE_VARS, 1);
                if let Ok(path) =
                    get_prefix_registry_storage_path::<Cfg>(SMALL_VARS, 1, &AkitaSetupSeed::DEFAULT)
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
                if let Ok(path) =
                    get_prefix_registry_storage_path::<Cfg>(SMALL_VARS, 1, &AkitaSetupSeed::DEFAULT)
                {
                    let _ = fs::remove_file(path);
                }
                let small = AkitaProverSetup::generate_with_capacity(
                    SMALL_VARS,
                    1,
                    Cfg::setup_matrix_capacity(SMALL_VARS, 1).unwrap(),
                    AkitaSetupSeed::DEFAULT,
                )
                .unwrap();
                let large = AkitaProverSetup::generate_with_capacity(
                    LARGE_VARS,
                    1,
                    Cfg::setup_matrix_capacity(LARGE_VARS, 1).unwrap(),
                    AkitaSetupSeed::DEFAULT,
                )
                .unwrap();
                let large_fields = large.expanded.shared_matrix().num_field_elements();
                let barrier = Arc::new(std::sync::Barrier::new(3));
                std::thread::scope(|scope| {
                    let first_barrier = Arc::clone(&barrier);
                    scope.spawn(move || {
                        first_barrier.wait();
                        save_prover_setup::<TestF, Cfg>(&small, SMALL_VARS, 1).unwrap();
                    });
                    let second_barrier = Arc::clone(&barrier);
                    scope.spawn(move || {
                        second_barrier.wait();
                        save_prover_setup::<TestF, Cfg>(&large, LARGE_VARS, 1).unwrap();
                    });
                    barrier.wait();
                });

                let loaded =
                    load_prover_setup::<TestF, Cfg>(LARGE_VARS, 1, &AkitaSetupSeed::DEFAULT)
                        .unwrap();
                assert_eq!(
                    loaded.expanded.shared_matrix().num_field_elements(),
                    large_fields
                );

                cleanup_setup_file_shape(LARGE_VARS, 1);
                if let Ok(path) =
                    get_prefix_registry_storage_path::<Cfg>(SMALL_VARS, 1, &AkitaSetupSeed::DEFAULT)
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
                    new_prover_setup::<TestF, Cfg>(MAX_VARS, 1, AkitaSetupSeed::DEFAULT).unwrap();
                let total = prover_setup.expanded.shared_matrix().num_field_elements();
                let corrupt = AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
                    prover_setup.expanded.descriptor().clone(),
                    FlatMatrix::from_flat_data(vec![TestF::zero(); total]),
                );
                let path =
                    get_public_matrix_storage_path::<TestF>(&AkitaSetupSeed::DEFAULT).unwrap();
                atomic_write_cache(&path, |writer| {
                    serialize_public_matrix_cache(&corrupt, writer)
                })
                .unwrap();

                let err = load_prover_setup::<TestF, Cfg>(MAX_VARS, 1, &AkitaSetupSeed::DEFAULT)
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

                new_prover_setup::<TestF, Cfg>(MAX_VARS, 1, AkitaSetupSeed::DEFAULT).unwrap();
                let path =
                    get_public_matrix_storage_path::<TestF>(&AkitaSetupSeed::DEFAULT).unwrap();
                let mut file = fs::OpenOptions::new().append(true).open(path).unwrap();
                file.write_all(&[0]).unwrap();

                let err = load_prover_setup::<TestF, Cfg>(MAX_VARS, 1, &AkitaSetupSeed::DEFAULT)
                    .expect_err("cache with trailing bytes must be rejected");
                assert!(err.to_string().contains("trailing bytes"));

                cleanup_setup_file_shape(MAX_VARS, 1);
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

                let fresh_setup =
                    new_prover_setup::<TestF, Cfg>(MAX_VARS, 1, AkitaSetupSeed::DEFAULT).unwrap();

                let disk_setup =
                    load_prover_setup::<TestF, Cfg>(MAX_VARS, 1, &AkitaSetupSeed::DEFAULT).unwrap();

                let lp = Cfg::resolve_catalog_row_for_opening(
                    &akita_types::OpeningClaimsLayout::new(MAX_VARS, 1)
                        .expect("singleton opening batch"),
                )
                .unwrap()
                .schedule()
                .root
                .params
                .clone();
                let num_coeffs = lp.blocks().live_blocks * lp.blocks().positions_per_block;
                let coeffs = vec![CyclotomicRing::<TestF, TEST_D>::zero(); num_coeffs];
                let poly = DensePoly::<TestF>::from_ring_coeffs(coeffs);

                let commit_u = |setup: &AkitaProverSetup<TestF>| {
                    let prepared = CpuBackend::DEFAULT.prepare_setup(setup).unwrap();
                    let plan = CommitInnerPlan::from_level(&lp);
                    let mut inner_group = CpuBackend::DEFAULT
                        .commit_inner_group(
                            &prepared,
                            vec![RootCommitSource::<TestF, TEST_D>::commit_view(&poly).unwrap()],
                            plan,
                        )
                        .unwrap();
                    let inner = inner_group.pop().expect("singleton commit result");
                    let n_a = lp.inner().matrix.output_rank();
                    let blocks = (0..lp.blocks().live_blocks)
                        .map(|block| inner.block_rows::<TEST_D>(block, n_a).unwrap())
                        .collect::<Vec<_>>();
                    let digits = akita_prover::kernels::linear::decompose_commit_blocks_into::<
                        TestF,
                        TEST_D,
                        TEST_D,
                    >(
                        &blocks,
                        lp.outer().digits.num_digits,
                        lp.outer().digits.log_basis,
                    )
                    .unwrap();
                    let slice_geometry = akita_types::CommitmentSliceGeometry::try_new(
                        lp.outer_slice_count(),
                        lp.blocks().live_blocks,
                        1,
                        n_a,
                        lp.outer().digits.num_digits,
                        TEST_D,
                        TEST_D,
                    )
                    .unwrap();
                    let block_width = slice_geometry.ring_elements_per_block_per_polynomial();
                    let range = slice_geometry
                        .block_ranges()
                        .iter()
                        .max_by_key(|range| range.len())
                        .unwrap();
                    let plane_start = range.start * block_width;
                    let plane_end = range.end * block_width;
                    let mut slice_digits =
                        digits.typed_planes::<TEST_D>().unwrap()[plane_start..plane_end].to_vec();
                    slice_digits.resize(slice_geometry.physical_input_width(), [0i8; TEST_D]);
                    let mut batches = CpuBackend::DEFAULT
                        .digit_rows::<TEST_D>(
                            &prepared,
                            lp.outer().matrix.output_rank(),
                            &[slice_digits.as_slice()],
                            lp.outer().digits.log_basis,
                        )
                        .unwrap();
                    assert_eq!(batches.len(), 1);
                    batches.pop().unwrap()
                };

                let fresh_u = commit_u(&fresh_setup);
                let disk_u = commit_u(&disk_setup);

                assert_eq!(fresh_u, disk_u);

                cleanup_setup_file_shape(MAX_VARS, 1);
            });
        }
    }
}
