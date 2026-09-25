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
    requirements: &SetupRequirements<F>,
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
        requirements.max_num_vars(),
        requirements.max_num_batched_polys(),
        requirements.matrix_capacity(),
    )?;

    recursive_prefixes::populate_required_setup_prefix_slots(
        &mut setup,
        requirements.prefix_slot_ids(),
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

/// Name the setup-prefix registry cache by field modulus, setup seed, and
/// required prefix slots.
///
/// A prefix commitment is a pure function of the seed-derived matrix and its
/// slot id, so the capacity bound is deliberately absent from the key: one
/// cached registry serves every bound and catalog combination that requires
/// the same slots.
#[cfg(feature = "disk-persistence")]
fn prefix_registry_cache_file_name<F: Field + CanonicalEncoding>(
    requirements: &SetupRequirements<F>,
) -> Result<String, AkitaError> {
    let mut key = Vec::new();
    key.extend_from_slice(b"AKITA-SETUP-PREFIX-SLOTS-V1");
    key.extend_from_slice(
        &setup_seed_digest(&sample_akita_setup_seed())
            .map_err(|err| AkitaError::InvalidSetup(format!("setup-prefix registry key: {err}")))?,
    );
    requirements
        .prefix_slot_ids()
        .to_vec()
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
    requirements: &SetupRequirements<F>,
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
    requirements: &SetupRequirements<F>,
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
    requirements: &SetupRequirements<F>,
) -> Result<AkitaProverSetup<F>, AkitaError> {
    let max_num_vars = requirements.max_num_vars();
    let max_num_batched_polys = requirements.max_num_batched_polys();
    let setup_seed = sample_akita_setup_seed();
    let public_matrix_path = get_public_matrix_storage_path::<F>(&setup_seed)?;
    if !public_matrix_path.exists() {
        return Err(AkitaError::InvalidSetup(format!(
            "public matrix cache not found at {}",
            public_matrix_path.display()
        )));
    }
    let required_num_field_elements = requirements.matrix_capacity().num_field_elements;
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
        requirements.prefix_slot_ids(),
    )
    .is_err()
    {
        setup.prefix_slots =
            SetupPrefixProverRegistry::new(setup.expanded.descriptor().setup_seed.clone());
        recursive_prefixes::populate_required_setup_prefix_slots(
            &mut setup,
            requirements.prefix_slot_ids(),
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
mod tests;
