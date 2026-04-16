//! Commitment scheme setup types and construction.

use crate::algebra::fields::wide::HasWide;
use crate::error::HachiError;
use crate::primitives::serialization::{
    Compress, HachiDeserialize, HachiSerialize, SerializationError, Valid, Validate,
};
use crate::protocol::commitment::utils::crt_ntt::{build_ntt_slot, NttSlotCache};
use crate::protocol::commitment::utils::flat_matrix::FlatMatrix;
use crate::protocol::commitment::utils::matrix::{
    derive_public_matrix_flat, sample_public_matrix_seed, PublicMatrixSeed,
};
use crate::protocol::commitment::CommitmentConfig;
#[cfg(feature = "disk-persistence")]
use crate::protocol::commitment::{HachiRootBatchSummary, HachiScheduleLookupKey};
use crate::{CanonicalField, FieldCore, FieldSampling};
#[cfg(feature = "disk-persistence")]
use std::fs;
use std::io::{Read, Write};
#[cfg(feature = "disk-persistence")]
use std::path::PathBuf;
use std::sync::Arc;

/// Seed-only stage for deterministic setup expansion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HachiSetupSeed {
    /// Maximum supported variable count.
    pub max_num_vars: usize,
    /// Maximum number of batched polynomials supported by setup.
    pub max_num_batched_polys: usize,
    /// Global row stride for the flat NTT cache (max column width).
    pub max_stride: usize,
    /// Public seed used to derive commitment matrices.
    pub public_matrix_seed: PublicMatrixSeed,
}

/// Expanded setup stage containing a single shared coefficient-form matrix
/// stored as a D-agnostic flat field-element array.
///
/// All role matrices (A, B, D) are row/column prefixes of this shared vector.
/// The same setup can be viewed at different ring dimensions by calling
/// [`FlatMatrix::ring_view`] with the desired const-generic `D` and
/// role-specific `(num_rows, num_cols)` dimensions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HachiExpandedSetup<F: FieldCore> {
    /// Setup seed and runtime layout metadata.
    pub seed: HachiSetupSeed,
    /// Shared 1D flat backing vector. Each role matrix (A, B, D) views a
    /// prefix of this vector reshaped with role-specific dimensions.
    pub shared_matrix: FlatMatrix<F>,
}

/// Prover setup artifact (expanded setup + single shared NTT cache).
///
/// The NTT cache is tied to a specific ring dimension D and covers the
/// full shared backing matrix. Role-specific mat-vec operations use row
/// slicing and input-vector-length column clamping.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HachiProverSetup<F: FieldCore, const D: usize> {
    /// Expanded matrix stage used by both prover and verifier.
    pub expanded: Arc<HachiExpandedSetup<F>>,
    /// Shared NTT cache for the backing matrix at ring dimension D.
    pub ntt_shared: NttSlotCache<D>,
}

/// Verifier setup artifact derived from prover setup.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HachiVerifierSetup<F: FieldCore> {
    /// Expanded matrix stage used for verification.
    pub expanded: Arc<HachiExpandedSetup<F>>,
}

impl<F: FieldCore, const D: usize> HachiProverSetup<F, D> {
    /// Construct prover setup for at most `max_num_vars` variables and
    /// `max_num_batched_polys` batched polynomials.
    ///
    /// # Errors
    ///
    /// Returns an error if `Cfg::D != D` or on arithmetic overflow.
    #[tracing::instrument(skip_all, name = "HachiProverSetup::new")]
    pub fn new<Cfg>(max_num_vars: usize, max_num_batched_polys: usize) -> Result<Self, HachiError>
    where
        F: CanonicalField + FieldSampling + HasWide + Valid,
        Cfg: CommitmentConfig<Field = F>,
    {
        if D != Cfg::D {
            return Err(HachiError::InvalidSetup(format!(
                "const D={D} mismatches config D={}",
                Cfg::D
            )));
        }
        let (max_rows, max_stride) =
            Cfg::max_setup_matrix_size(max_num_vars, max_num_batched_polys)?;
        let max_total = max_rows
            .checked_mul(max_stride)
            .ok_or_else(|| HachiError::InvalidSetup("conservative total overflow".to_string()))?;

        #[cfg(feature = "disk-persistence")]
        {
            match load_expanded_setup::<F, Cfg>(max_num_vars, max_num_batched_polys) {
                Ok(expanded) => {
                    if expanded.shared_matrix.total_ring_elements_at::<D>() >= max_total {
                        tracing::info!("Loaded setup from disk, rebuilding NTT caches");
                        return Self::from_expanded(expanded);
                    }
                    if let Some(storage_path) =
                        get_storage_path::<Cfg>(max_num_vars, max_num_batched_polys)
                    {
                        let _ = fs::remove_file(&storage_path);
                        tracing::warn!(
                            "Rejected cached setup from {}: matrix too small; regenerating",
                            storage_path.display()
                        );
                    } else {
                        tracing::warn!("Rejected cached setup: matrix too small; regenerating");
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

        let public_matrix_seed = sample_public_matrix_seed();
        let shared_flat = derive_public_matrix_flat::<F, D>(max_total, &public_matrix_seed);
        let ntt_shared = build_ntt_slot(shared_flat.ring_view::<D>(1, max_total))?;

        let expanded = Arc::new(HachiExpandedSetup {
            seed: HachiSetupSeed {
                max_num_vars,
                max_num_batched_polys,
                max_stride,
                public_matrix_seed,
            },
            shared_matrix: shared_flat,
        });

        #[cfg(feature = "disk-persistence")]
        save_expanded_setup::<F, Cfg>(&expanded, max_num_vars, max_num_batched_polys);

        Ok(Self {
            expanded,
            ntt_shared,
        })
    }

    /// Derive a verifier setup from this prover setup.
    pub fn verifier_setup(&self) -> HachiVerifierSetup<F> {
        HachiVerifierSetup {
            expanded: self.expanded.clone(),
        }
    }

    /// Wrap a pre-built [`HachiExpandedSetup`] in a prover setup by
    /// reconstructing the shared NTT cache at ring dimension `D`.
    ///
    /// # Errors
    ///
    /// Returns an error if the NTT cache cannot be built for the current
    /// field/ring-dimension pair.
    #[cfg(feature = "disk-persistence")]
    pub(crate) fn from_expanded(expanded: HachiExpandedSetup<F>) -> Result<Self, HachiError>
    where
        F: CanonicalField,
    {
        let expanded = Arc::new(expanded);
        let total = expanded.shared_matrix.total_ring_elements_at::<D>();
        let ntt_shared = build_ntt_slot(expanded.shared_matrix.ring_view::<D>(1, total))?;
        Ok(Self {
            expanded,
            ntt_shared,
        })
    }
}

// ---------------------------------------------------------------------------
// Serialization
// ---------------------------------------------------------------------------

impl Valid for HachiSetupSeed {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl HachiSerialize for HachiSetupSeed {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.max_num_vars
            .serialize_with_mode(&mut writer, compress)?;
        self.max_num_batched_polys
            .serialize_with_mode(&mut writer, compress)?;
        self.max_stride.serialize_with_mode(&mut writer, compress)?;
        writer.write_all(&self.public_matrix_seed)?;
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.max_num_vars.serialized_size(compress)
            + self.max_num_batched_polys.serialized_size(compress)
            + self.max_stride.serialized_size(compress)
            + 32
    }
}

impl HachiDeserialize for HachiSetupSeed {
    type Context = ();
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &(),
    ) -> Result<Self, SerializationError> {
        let max_num_vars = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let max_num_batched_polys =
            usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let max_stride = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let mut public_matrix_seed = [0u8; 32];
        reader.read_exact(&mut public_matrix_seed)?;
        let out = Self {
            max_num_vars,
            max_num_batched_polys,
            max_stride,
            public_matrix_seed,
        };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

impl<F: FieldCore + Valid> Valid for HachiExpandedSetup<F> {
    fn check(&self) -> Result<(), SerializationError> {
        self.seed.check()?;
        self.shared_matrix.check()?;
        Ok(())
    }
}

impl<F: FieldCore> HachiSerialize for HachiExpandedSetup<F> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.seed.serialize_with_mode(&mut writer, compress)?;
        self.shared_matrix
            .serialize_with_mode(&mut writer, compress)?;
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.seed.serialized_size(compress) + self.shared_matrix.serialized_size(compress)
    }
}

impl<F: FieldCore + Valid> HachiDeserialize for HachiExpandedSetup<F> {
    type Context = ();
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &(),
    ) -> Result<Self, SerializationError> {
        let out = Self {
            seed: HachiSetupSeed::deserialize_with_mode(&mut reader, compress, validate, &())?,
            shared_matrix: FlatMatrix::deserialize_with_mode(&mut reader, compress, validate, &())?,
        };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

impl<F: FieldCore + Valid, const D: usize> Valid for HachiProverSetup<F, D> {
    fn check(&self) -> Result<(), SerializationError> {
        self.expanded.check()
    }
}

impl<F: FieldCore + Valid> Valid for HachiVerifierSetup<F> {
    fn check(&self) -> Result<(), SerializationError> {
        self.expanded.check()
    }
}

impl<F: FieldCore> HachiSerialize for HachiVerifierSetup<F> {
    fn serialize_with_mode<W: Write>(
        &self,
        writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.expanded.serialize_with_mode(writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.expanded.serialized_size(compress)
    }
}

impl<F: FieldCore + Valid> HachiDeserialize for HachiVerifierSetup<F> {
    type Context = ();
    fn deserialize_with_mode<R: Read>(
        reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &(),
    ) -> Result<Self, SerializationError> {
        Ok(Self {
            expanded: Arc::new(HachiExpandedSetup::deserialize_with_mode(
                reader,
                compress,
                validate,
                &(),
            )?),
        })
    }
}

// ---------------------------------------------------------------------------
// Disk persistence
// ---------------------------------------------------------------------------

#[cfg(feature = "disk-persistence")]
fn cache_file_name<Cfg: CommitmentConfig>(
    max_num_vars: usize,
    max_num_batched_polys: usize,
) -> String {
    let envelope = Cfg::envelope(max_num_vars);
    let family = Cfg::family_key()
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '_' })
        .collect::<String>();
    let schedule_lookup_key = HachiScheduleLookupKey::with_batch(
        max_num_vars,
        max_num_vars,
        max_num_batched_polys,
        HachiRootBatchSummary::new(
            max_num_batched_polys,
            max_num_batched_polys,
            max_num_batched_polys,
        )
        .expect("setup cache key requires positive batch counts"),
    );
    let schedule = Cfg::schedule_key(schedule_lookup_key)
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '_' })
        .collect::<String>();
    let modulus = Cfg::field_modulus();
    format!(
        "hachi_q{modulus:032x}_{family}_sched_{schedule}_d{}_na{}_nb{}_nd{}_nv{max_num_vars}_batch{max_num_batched_polys}.setup",
        Cfg::D,
        envelope.max_n_a,
        envelope.max_n_b,
        envelope.max_n_d,
    )
}

#[cfg(feature = "disk-persistence")]
fn get_storage_path<Cfg: CommitmentConfig>(
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
        path.push("hachi");
        path.push(cache_file_name::<Cfg>(max_num_vars, max_num_batched_polys));
        path
    })
}

#[cfg(feature = "disk-persistence")]
fn save_expanded_setup<F: FieldCore + CanonicalField, Cfg: CommitmentConfig<Field = F>>(
    setup: &HachiExpandedSetup<F>,
    max_num_vars: usize,
    max_num_batched_polys: usize,
) {
    let Some(storage_path) = get_storage_path::<Cfg>(max_num_vars, max_num_batched_polys) else {
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
fn load_expanded_setup<F: FieldCore + Valid + CanonicalField, Cfg: CommitmentConfig<Field = F>>(
    max_num_vars: usize,
    max_num_batched_polys: usize,
) -> Result<HachiExpandedSetup<F>, HachiError> {
    let storage_path =
        get_storage_path::<Cfg>(max_num_vars, max_num_batched_polys).ok_or_else(|| {
            HachiError::InvalidSetup("Failed to determine storage directory".to_string())
        })?;

    if !storage_path.exists() {
        return Err(HachiError::InvalidSetup(format!(
            "Setup file not found at {}",
            storage_path.display()
        )));
    }

    tracing::info!("Loading setup from {}", storage_path.display());

    let file = fs::File::open(&storage_path)
        .map_err(|e| HachiError::InvalidSetup(format!("Failed to open setup file: {e}")))?;
    let mut reader = std::io::BufReader::new(file);

    let setup = HachiExpandedSetup::deserialize_compressed(&mut reader, &())
        .map_err(|e| HachiError::InvalidSetup(format!("Failed to deserialize setup: {e}")))?;

    tracing::info!(
        "Loaded setup for max_num_vars={max_num_vars}, max_num_batched_polys={max_num_batched_polys}"
    );
    Ok(setup)
}

/// Build prover and verifier setup from a pre-existing expanded setup by
/// reconstructing the NTT cache.
#[cfg(feature = "disk-persistence")]
pub(crate) fn setup_from_expanded<F: FieldCore + CanonicalField, const D: usize>(
    expanded: HachiExpandedSetup<F>,
) -> Result<(HachiProverSetup<F, D>, HachiVerifierSetup<F>), HachiError> {
    let prover_setup = HachiProverSetup::<F, D>::from_expanded(expanded)?;
    let verifier_setup = prover_setup.verifier_setup();
    Ok((prover_setup, verifier_setup))
}
