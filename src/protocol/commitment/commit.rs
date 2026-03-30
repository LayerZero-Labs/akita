//! Ring-native §4.1 commitment core implementation.

use super::config::{
    ensure_block_layout, ensure_supported_num_vars, validate_and_derive_layout,
    HachiCommitmentLayout,
};
use super::onehot::{inner_ajtai_onehot_wide, map_onehot_to_sparse_blocks};
use super::schedule::HachiScheduleInputs;
use super::scheme::{CommitWitness, RingCommitmentScheme};
use super::types::RingCommitment;
use super::utils::crt_ntt::{build_ntt_slot, NttSlotCache};
use super::utils::flat_matrix::FlatMatrix;
use super::utils::linear::{
    decompose_rows_i8, flatten_i8_blocks, mat_vec_mul_ntt_i8, mat_vec_mul_ntt_single_i8,
};
use super::utils::matrix::{
    derive_public_matrix_flat, sample_public_matrix_seed, PublicMatrixSeed,
};
use super::CommitmentConfig;
use crate::algebra::fields::wide::HasWide;
use crate::algebra::CyclotomicRing;
use crate::error::HachiError;
#[cfg(feature = "parallel")]
use crate::parallel::*;
use crate::primitives::serialization::{
    Compress, HachiDeserialize, HachiSerialize, SerializationError, Valid, Validate,
};
use crate::protocol::commitment_scheme::should_stop_folding;
use crate::protocol::hachi_poly_ops::OneHotIndex;
#[cfg(test)]
use crate::protocol::ring_switch::w_commitment_layout;
use crate::protocol::ring_switch::w_ring_element_count;
use crate::{cfg_into_iter, cfg_iter, CanonicalField, FieldCore, FieldSampling};
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
    /// Runtime commitment layout.
    pub layout: HachiCommitmentLayout,
    /// Public seed used to derive commitment matrices.
    pub public_matrix_seed: PublicMatrixSeed,
}

/// Expanded setup stage containing a single shared coefficient-form matrix
/// stored as a D-agnostic flat field-element array.
///
/// All role matrices (A, B, D) are row/column prefixes of this shared matrix.
/// See `SHARED_PREFIX_BINDING.md` for the security argument. The same setup
/// can be viewed at different ring dimensions by calling [`FlatMatrix::view`]
/// with the desired const-generic `D`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HachiExpandedSetup<F: FieldCore> {
    /// Setup seed and runtime layout metadata.
    pub seed: HachiSetupSeed,
    /// Shared backing matrix (max_rows × max_cols). Each role matrix
    /// (A, B, D) is a row/column prefix of this matrix.
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

impl<F: FieldCore> HachiExpandedSetup<F> {
    /// Runtime layout carried by this setup (the max-dimension layout).
    pub fn layout(&self) -> HachiCommitmentLayout {
        self.seed.layout
    }
}

impl<F: FieldCore, const D: usize> HachiProverSetup<F, D> {
    /// Runtime layout carried by this setup (the max-dimension layout).
    pub fn layout(&self) -> HachiCommitmentLayout {
        self.expanded.layout()
    }

    /// Panic if `layout`'s matrix dimensions exceed this setup's maximums.
    ///
    /// # Panics
    ///
    /// Panics if any of `layout`'s matrix widths (inner, outer, D) exceed
    /// those of this setup.
    pub fn assert_layout_fits(&self, layout: &HachiCommitmentLayout) {
        let max = &self.expanded.seed.layout;
        assert!(
            layout.inner_width <= max.inner_width,
            "A matrix too narrow: need {} but setup has {}",
            layout.inner_width,
            max.inner_width
        );
        assert!(
            layout.outer_width <= max.outer_width,
            "B matrix too narrow: need {} but setup has {}",
            layout.outer_width,
            max.outer_width
        );
        assert!(
            layout.d_matrix_width <= max.d_matrix_width,
            "D matrix too narrow: need {} but setup has {}",
            layout.d_matrix_width,
            max.d_matrix_width
        );
    }
}

impl Valid for HachiSetupSeed {
    fn check(&self) -> Result<(), SerializationError> {
        self.layout.check()
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
        self.layout.serialize_with_mode(&mut writer, compress)?;
        writer.write_all(&self.public_matrix_seed)?;
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.max_num_vars.serialized_size(compress) + self.layout.serialized_size(compress) + 32
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
        let layout =
            HachiCommitmentLayout::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let mut public_matrix_seed = [0u8; 32];
        reader.read_exact(&mut public_matrix_seed)?;
        let out = Self {
            max_num_vars,
            layout,
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

impl<F: FieldCore, const D: usize> HachiSerialize for HachiProverSetup<F, D> {
    fn serialize_with_mode<W: Write>(
        &self,
        _writer: W,
        _compress: Compress,
    ) -> Result<(), SerializationError> {
        Err(SerializationError::InvalidData(
            "HachiProverSetup contains runtime NTT caches and is not serializable".into(),
        ))
    }

    fn serialized_size(&self, _compress: Compress) -> usize {
        0
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

pub(crate) fn root_current_w_len<const D: usize>(layout: HachiCommitmentLayout) -> usize {
    layout
        .num_blocks
        .checked_mul(layout.block_len)
        .and_then(|len| len.checked_mul(D))
        .unwrap_or(0)
}

#[derive(Debug, Clone, Copy, Default)]
struct LayoutChainStats {
    max_inner_width: usize,
    max_outer_width: usize,
    max_d_matrix_width: usize,
    max_r_vars: usize,
    max_num_digits_open: usize,
    max_num_digits_fold: usize,
    max_log_basis: u32,
}

impl LayoutChainStats {
    fn include(&mut self, layout: HachiCommitmentLayout) {
        self.max_inner_width = self.max_inner_width.max(layout.inner_width);
        self.max_outer_width = self.max_outer_width.max(layout.outer_width);
        self.max_d_matrix_width = self.max_d_matrix_width.max(layout.d_matrix_width);
        self.max_r_vars = self.max_r_vars.max(layout.r_vars);
        self.max_num_digits_open = self.max_num_digits_open.max(layout.num_digits_open);
        self.max_num_digits_fold = self.max_num_digits_fold.max(layout.num_digits_fold);
        self.max_log_basis = self.max_log_basis.max(layout.log_basis);
    }
}

fn scan_layout_chain<F, const D: usize, Cfg>(
    max_num_vars: usize,
    root_layout: HachiCommitmentLayout,
) -> Result<LayoutChainStats, HachiError>
where
    F: FieldCore + CanonicalField,
    Cfg: CommitmentConfig,
{
    let mut stats = LayoutChainStats::default();
    stats.include(root_layout);

    let can_use_planned_root =
        Cfg::commitment_layout(max_num_vars).is_ok_and(|planned_root| planned_root == root_layout);
    if can_use_planned_root {
        if let Some(plan) = Cfg::schedule_plan(max_num_vars)? {
            for level in plan.levels.iter().skip(1) {
                stats.include(level.layout);
            }
            return Ok(stats);
        }
    }

    let root_inputs = HachiScheduleInputs {
        max_num_vars,
        level: 0,
        current_w_len: root_current_w_len::<D>(root_layout),
    };
    let root_params = Cfg::level_params_with_log_basis(root_inputs, root_layout.log_basis);
    let mut prev_w_len = root_inputs.current_w_len;
    let mut level = 1usize;
    let mut current_w_len = w_ring_element_count::<F>(&root_params, root_layout) * root_params.d;
    let mut current_params = Cfg::level_params(HachiScheduleInputs {
        max_num_vars,
        level,
        current_w_len,
    });
    let mut current_layout =
        super::hachi_recursive_level_layout_from_params::<Cfg>(&current_params, current_w_len)?;
    stats.include(current_layout);

    loop {
        if should_stop_folding(current_w_len, prev_w_len) {
            break;
        }

        let next_w_len =
            w_ring_element_count::<F>(&current_params, current_layout) * current_params.d;
        let next_level = level + 1;
        let next_params = Cfg::level_params(HachiScheduleInputs {
            max_num_vars,
            level: next_level,
            current_w_len: next_w_len,
        });
        let next_layout =
            super::hachi_recursive_level_layout_from_params::<Cfg>(&next_params, next_w_len)?;
        stats.include(next_layout);

        prev_w_len = current_w_len;
        current_w_len = next_w_len;
        current_params = next_params;
        current_layout = next_layout;
        level = next_level;
    }

    Ok(stats)
}

#[cfg(feature = "disk-persistence")]
fn cache_file_name<Cfg: CommitmentConfig>(max_num_vars: usize) -> String {
    let envelope = Cfg::envelope(max_num_vars);
    let family = Cfg::family_key()
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '_' })
        .collect::<String>();
    let schedule = Cfg::schedule_key(max_num_vars)
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '_' })
        .collect::<String>();
    format!(
        "hachi_{family}_sched_{schedule}_d{}_na{}_nb{}_nd{}_nv{max_num_vars}.setup",
        Cfg::D,
        envelope.max_n_a,
        envelope.max_n_b,
        envelope.max_n_d,
    )
}

#[cfg(feature = "disk-persistence")]
fn get_storage_path<Cfg: CommitmentConfig>(max_num_vars: usize) -> Option<PathBuf> {
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
        path.push(cache_file_name::<Cfg>(max_num_vars));
        path
    })
}

#[cfg(feature = "disk-persistence")]
fn save_expanded_setup<F: FieldCore, Cfg: CommitmentConfig>(
    setup: &HachiExpandedSetup<F>,
    max_num_vars: usize,
) {
    let Some(storage_path) = get_storage_path::<Cfg>(max_num_vars) else {
        tracing::warn!("Could not determine storage directory; skipping setup save");
        return;
    };

    if let Some(parent) = storage_path.parent() {
        fs::create_dir_all(parent)
            .unwrap_or_else(|e| panic!("Failed to create storage directory: {e}"));
    }

    tracing::info!("Saving setup to {}", storage_path.display());

    let file = fs::File::create(&storage_path)
        .unwrap_or_else(|e| panic!("Failed to create setup file: {e}"));
    let mut writer = std::io::BufWriter::new(file);

    setup
        .serialize_compressed(&mut writer)
        .unwrap_or_else(|e| panic!("Failed to serialize setup: {e}"));

    tracing::info!("Successfully saved setup to disk");
}

#[cfg(feature = "disk-persistence")]
fn load_expanded_setup<F: FieldCore + Valid, Cfg: CommitmentConfig>(
    max_num_vars: usize,
) -> Result<HachiExpandedSetup<F>, HachiError> {
    let storage_path = get_storage_path::<Cfg>(max_num_vars).ok_or_else(|| {
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

    let setup = HachiExpandedSetup::deserialize_compressed(&mut reader)
        .map_err(|e| HachiError::InvalidSetup(format!("Failed to deserialize setup: {e}")))?;

    tracing::info!("Loaded setup for max_num_vars={max_num_vars}");
    Ok(setup)
}

/// Build prover and verifier setup from a pre-existing expanded setup by
/// reconstructing the NTT cache.
#[cfg(feature = "disk-persistence")]
pub(crate) fn setup_from_expanded<F: FieldCore + CanonicalField, const D: usize>(
    expanded: HachiExpandedSetup<F>,
) -> Result<(HachiProverSetup<F, D>, HachiVerifierSetup<F>), HachiError> {
    let expanded = Arc::new(expanded);
    let ntt_shared = build_ntt_slot(expanded.shared_matrix.view::<D>())?;
    let prover_setup = HachiProverSetup {
        expanded: Arc::clone(&expanded),
        ntt_shared,
    };
    let verifier_setup = HachiVerifierSetup { expanded };
    Ok((prover_setup, verifier_setup))
}

/// Concrete §4.1 commitment core.
#[derive(Clone, Copy, Default)]
pub struct HachiCommitmentCore;

impl<F, const D: usize, Cfg> RingCommitmentScheme<F, D, Cfg> for HachiCommitmentCore
where
    F: FieldCore + CanonicalField + FieldSampling + HasWide + Valid,
    Cfg: CommitmentConfig,
{
    type ProverSetup = HachiProverSetup<F, D>;
    type VerifierSetup = HachiVerifierSetup<F>;
    type Commitment = RingCommitment<F, D>;

    #[tracing::instrument(skip_all, name = "RingCommitmentScheme::setup")]
    fn setup(max_num_vars: usize) -> Result<(Self::ProverSetup, Self::VerifierSetup), HachiError> {
        let layout = validate_and_derive_layout::<Cfg, D>(max_num_vars)?;
        let envelope = Cfg::envelope(max_num_vars);
        ensure_supported_num_vars(max_num_vars, layout.required_num_vars::<D>()?)?;

        #[cfg(feature = "disk-persistence")]
        {
            match load_expanded_setup::<F, Cfg>(max_num_vars) {
                Ok(expanded) => {
                    tracing::info!("Loaded setup from disk, rebuilding NTT caches");
                    return setup_from_expanded(expanded);
                }
                Err(e) => {
                    if let Some(storage_path) = get_storage_path::<Cfg>(max_num_vars) {
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

        let chain_stats = scan_layout_chain::<F, D, Cfg>(max_num_vars, layout)?;
        let a_cols = chain_stats.max_inner_width;
        let b_cols = chain_stats.max_outer_width;
        let d_cols = chain_stats.max_d_matrix_width;

        let max_rows = [envelope.max_n_a, envelope.max_n_b, envelope.max_n_d]
            .into_iter()
            .max()
            .unwrap();
        let max_cols = [a_cols, b_cols, d_cols].into_iter().max().unwrap();

        let public_matrix_seed = sample_public_matrix_seed();
        let shared_flat =
            derive_public_matrix_flat::<F, D>(max_rows, max_cols, &public_matrix_seed);
        let ntt_shared = build_ntt_slot(shared_flat.view::<D>())?;

        let expanded = Arc::new(HachiExpandedSetup {
            seed: HachiSetupSeed {
                max_num_vars,
                layout,
                public_matrix_seed,
            },
            shared_matrix: shared_flat,
        });

        #[cfg(feature = "disk-persistence")]
        save_expanded_setup::<F, Cfg>(&expanded, max_num_vars);

        let prover_setup = HachiProverSetup {
            expanded: Arc::clone(&expanded),
            ntt_shared,
        };
        let verifier_setup = HachiVerifierSetup { expanded };
        Ok((prover_setup, verifier_setup))
    }

    fn layout(setup: &Self::ProverSetup) -> Result<HachiCommitmentLayout, HachiError> {
        Ok(setup.layout())
    }

    #[tracing::instrument(skip_all, name = "RingCommitmentScheme::commit_ring_blocks")]
    fn commit_ring_blocks(
        f_blocks: &[Vec<CyclotomicRing<F, D>>],
        setup: &Self::ProverSetup,
    ) -> Result<CommitWitness<Self::Commitment, F, D>, HachiError> {
        let layout = setup.layout();
        let root_params = Cfg::level_params(HachiScheduleInputs {
            max_num_vars: setup.expanded.seed.max_num_vars,
            level: 0,
            current_w_len: root_current_w_len::<D>(layout),
        });
        ensure_supported_num_vars(
            setup.expanded.seed.max_num_vars,
            layout.required_num_vars::<D>()?,
        )?;
        ensure_block_layout(f_blocks, layout)?;

        let depth_commit = layout.num_digits_commit;
        let depth_open = layout.num_digits_open;
        let log_basis = layout.log_basis;
        let block_slices: Vec<&[CyclotomicRing<F, D>]> =
            f_blocks.iter().map(|b| b.as_slice()).collect();
        let t_all = mat_vec_mul_ntt_i8(
            &setup.ntt_shared,
            root_params.n_a,
            &block_slices,
            depth_commit,
            log_basis,
        );
        let t_hat_all: Vec<Vec<[i8; D]>> = cfg_into_iter!(t_all)
            .map(|t_i| decompose_rows_i8(&t_i, depth_open, log_basis))
            .collect();

        let t_hat_flat = flatten_i8_blocks(&t_hat_all);
        let u: Vec<CyclotomicRing<F, D>> =
            mat_vec_mul_ntt_single_i8(&setup.ntt_shared, root_params.n_b, &t_hat_flat);
        Ok(CommitWitness::new(RingCommitment { u }, t_hat_all))
    }

    #[tracing::instrument(skip_all, name = "RingCommitmentScheme::commit_coeffs")]
    fn commit_coeffs(
        f_coeffs: &[CyclotomicRing<F, D>],
        setup: &Self::ProverSetup,
    ) -> Result<CommitWitness<Self::Commitment, F, D>, HachiError> {
        let layout = setup.layout();
        let root_params = Cfg::level_params(HachiScheduleInputs {
            max_num_vars: setup.expanded.seed.max_num_vars,
            level: 0,
            current_w_len: root_current_w_len::<D>(layout),
        });
        let num_blocks = layout.num_blocks;
        let block_len = layout.block_len;
        let max_len = num_blocks
            .checked_mul(block_len)
            .ok_or_else(|| HachiError::InvalidSetup("coefficient length overflow".to_string()))?;
        if f_coeffs.len() > max_len {
            return Err(HachiError::InvalidSize {
                expected: max_len,
                actual: f_coeffs.len(),
            });
        }

        let depth_commit = layout.num_digits_commit;
        let depth_open = layout.num_digits_open;
        let log_basis = layout.log_basis;
        let coeff_len = f_coeffs.len();

        let block_slices: Vec<&[CyclotomicRing<F, D>]> = (0..num_blocks)
            .map(|i| {
                let start = i * block_len;
                if start >= coeff_len {
                    &[] as &[CyclotomicRing<F, D>]
                } else {
                    &f_coeffs[start..(start + block_len).min(coeff_len)]
                }
            })
            .collect();

        let t_all = mat_vec_mul_ntt_i8(
            &setup.ntt_shared,
            root_params.n_a,
            &block_slices,
            depth_commit,
            log_basis,
        );
        let t_hat_all: Vec<Vec<[i8; D]>> = cfg_into_iter!(t_all)
            .map(|t_i| decompose_rows_i8(&t_i, depth_open, log_basis))
            .collect();

        let t_hat_flat = flatten_i8_blocks(&t_hat_all);
        let u: Vec<CyclotomicRing<F, D>> =
            mat_vec_mul_ntt_single_i8(&setup.ntt_shared, root_params.n_b, &t_hat_flat);
        Ok(CommitWitness::new(RingCommitment { u }, t_hat_all))
    }

    #[tracing::instrument(skip_all, name = "RingCommitmentScheme::commit_onehot")]
    fn commit_onehot<I: OneHotIndex>(
        onehot_k: usize,
        indices: &[Option<I>],
        setup: &Self::ProverSetup,
    ) -> Result<CommitWitness<Self::Commitment, F, D>, HachiError> {
        let layout = setup.layout();
        let root_params = Cfg::level_params(HachiScheduleInputs {
            max_num_vars: setup.expanded.seed.max_num_vars,
            level: 0,
            current_w_len: root_current_w_len::<D>(layout),
        });
        ensure_supported_num_vars(
            setup.expanded.seed.max_num_vars,
            layout.required_num_vars::<D>()?,
        )?;

        let sparse_blocks =
            map_onehot_to_sparse_blocks(onehot_k, indices, layout.r_vars, layout.m_vars, D)?;

        let depth_commit = layout.num_digits_commit;
        let depth_open = layout.num_digits_open;
        let log_basis = layout.log_basis;
        let zero_block_len = root_params.n_a.checked_mul(depth_open).unwrap();
        let a_view = setup.expanded.shared_matrix.view::<D>();
        let block_len = layout.block_len;

        let t_hat_all: Vec<Vec<[i8; D]>> = cfg_iter!(sparse_blocks)
            .map(|block_entries| {
                if block_entries.is_empty() {
                    vec![[0i8; D]; zero_block_len]
                } else {
                    let mut t_i =
                        inner_ajtai_onehot_wide(&a_view, block_entries, block_len, depth_commit);
                    t_i.truncate(root_params.n_a);
                    decompose_rows_i8(&t_i, depth_open, log_basis)
                }
            })
            .collect();

        let t_hat_flat = flatten_i8_blocks(&t_hat_all);
        let u: Vec<CyclotomicRing<F, D>> =
            mat_vec_mul_ntt_single_i8(&setup.ntt_shared, root_params.n_b, &t_hat_flat);
        Ok(CommitWitness::new(RingCommitment { u }, t_hat_all))
    }
}

impl HachiCommitmentCore {
    #[allow(clippy::too_many_arguments)]
    fn layout_envelope<const D: usize>(
        max_num_vars: usize,
        inner_width: usize,
        outer_width: usize,
        d_matrix_width: usize,
        preferred_r_vars: usize,
        num_digits_open: usize,
        num_digits_fold: usize,
        log_basis: u32,
    ) -> Result<HachiCommitmentLayout, HachiError> {
        let alpha = D.trailing_zeros() as usize;
        let outer_vars = max_num_vars.checked_sub(alpha).ok_or_else(|| {
            HachiError::InvalidSetup("max_num_vars is smaller than alpha".to_string())
        })?;
        let r_vars = preferred_r_vars.min(outer_vars);
        let m_vars = outer_vars - r_vars;
        let num_blocks = 1usize
            .checked_shl(r_vars as u32)
            .ok_or_else(|| HachiError::InvalidSetup("num_blocks overflow".to_string()))?;
        let block_len = 1usize
            .checked_shl(m_vars as u32)
            .ok_or_else(|| HachiError::InvalidSetup("block_len overflow".to_string()))?;

        Ok(HachiCommitmentLayout {
            m_vars,
            r_vars,
            num_blocks,
            block_len,
            inner_width,
            outer_width,
            d_matrix_width,
            // Setup metadata only tracks width envelopes; runtime commits/proofs
            // carry their own exact decomposition parameters.
            num_digits_commit: 1,
            num_digits_open,
            num_digits_fold,
            log_basis,
        })
    }

    /// Create a setup with a caller-specified layout, bypassing
    /// `CommitmentConfig::commitment_layout`.
    ///
    /// Use this when the desired `(m_vars, r_vars)` split differs from what
    /// the config's heuristic would choose (e.g. mega-polynomial commitments
    /// where each sub-polynomial occupies one block).
    ///
    /// # Errors
    ///
    /// Returns `HachiError` on invalid layout or matrix generation failures.
    pub fn setup_with_layout<F, const D: usize, Cfg>(
        layout: HachiCommitmentLayout,
    ) -> Result<(HachiProverSetup<F, D>, HachiVerifierSetup<F>), HachiError>
    where
        F: FieldCore + CanonicalField + FieldSampling,
        Cfg: CommitmentConfig,
    {
        let max_num_vars = layout.required_num_vars::<D>()?;
        let public_matrix_seed = sample_public_matrix_seed();
        Self::setup_with_layout_and_seed::<F, D, Cfg>(layout, max_num_vars, public_matrix_seed)
    }

    /// Create a setup that supports any of the provided runtime layouts.
    ///
    /// This sizes the public matrices from the exact per-layout maxima
    /// (including recursive `w` commitments) instead of inflating through a
    /// synthetic max layout.
    ///
    /// # Errors
    ///
    /// Returns `HachiError` if `layouts` is empty, uses inconsistent
    /// decomposition parameters, or matrix generation fails.
    pub fn setup_with_layouts<F, const D: usize, Cfg>(
        layouts: &[HachiCommitmentLayout],
    ) -> Result<(HachiProverSetup<F, D>, HachiVerifierSetup<F>), HachiError>
    where
        F: FieldCore + CanonicalField + FieldSampling,
        Cfg: CommitmentConfig,
    {
        let Some((&first_layout, _)) = layouts.split_first() else {
            return Err(HachiError::InvalidSetup(
                "setup_with_layouts requires at least one layout".to_string(),
            ));
        };

        let mut max_num_vars = 0usize;
        let mut max_inner_width = 0usize;
        let mut max_outer_width = 0usize;
        let mut max_d_matrix_width = 0usize;
        let mut max_r_vars = 0usize;
        let mut max_num_digits_open = 0usize;
        let mut max_num_digits_fold = 0usize;
        let mut max_log_basis = first_layout.log_basis;

        for &layout in layouts {
            let layout_num_vars = layout.required_num_vars::<D>()?;
            let chain_stats = scan_layout_chain::<F, D, Cfg>(layout_num_vars, layout)?;
            tracing::debug!(?layout, ?chain_stats, "setup layout chain");
            max_num_vars = max_num_vars.max(layout_num_vars);
            max_inner_width = max_inner_width.max(chain_stats.max_inner_width);
            max_outer_width = max_outer_width.max(chain_stats.max_outer_width);
            max_d_matrix_width = max_d_matrix_width.max(chain_stats.max_d_matrix_width);
            max_r_vars = max_r_vars.max(chain_stats.max_r_vars);
            max_num_digits_open = max_num_digits_open.max(chain_stats.max_num_digits_open);
            max_num_digits_fold = max_num_digits_fold.max(chain_stats.max_num_digits_fold);
            max_log_basis = max_log_basis.max(chain_stats.max_log_basis);
        }

        let envelope_layout = Self::layout_envelope::<D>(
            max_num_vars,
            max_inner_width,
            max_outer_width,
            max_d_matrix_width,
            max_r_vars,
            max_num_digits_open,
            max_num_digits_fold,
            max_log_basis,
        )?;
        tracing::debug!(?envelope_layout, max_num_vars, "setup envelope");
        let public_matrix_seed = sample_public_matrix_seed();
        Self::setup_with_matrix_widths_and_seed::<F, D, Cfg>(
            envelope_layout,
            max_num_vars,
            public_matrix_seed,
            max_inner_width,
            max_outer_width,
            max_d_matrix_width,
        )
    }

    /// Like `setup_with_layout` but reuses an existing setup's random seed and
    /// shared matrix (which is prefix-stable). Only extends the shared matrix
    /// if the new layout requires wider columns or more rows.
    ///
    /// Use this when creating a mega-polynomial setup that shares `m_vars` with
    /// an individual polynomial setup.
    ///
    /// # Panics
    ///
    /// Panics if the envelope contains zero rows or the layout chain has zero columns.
    ///
    /// # Errors
    ///
    /// Returns `HachiError` if the new layout is incompatible with the existing
    /// setup or matrix shapes are inconsistent.
    pub fn setup_from_existing<F, const D: usize, Cfg>(
        existing: &HachiExpandedSetup<F>,
        new_r_vars: usize,
    ) -> Result<(HachiProverSetup<F, D>, HachiVerifierSetup<F>), HachiError>
    where
        F: FieldCore + CanonicalField + FieldSampling,
        Cfg: CommitmentConfig,
    {
        let old_layout = existing.seed.layout;
        let new_layout = HachiCommitmentLayout::new::<Cfg>(
            old_layout.m_vars,
            new_r_vars,
            &Cfg::decomposition(),
        )?;

        if new_layout.inner_width != old_layout.inner_width {
            return Err(HachiError::InvalidSetup(
                "setup_from_existing requires matching m_vars/inner_width".to_string(),
            ));
        }

        let max_num_vars = new_layout.required_num_vars::<D>()?;
        let envelope = Cfg::envelope(max_num_vars);
        let seed = existing.seed.public_matrix_seed;

        let layout_num_vars = new_layout.required_num_vars::<D>()?;
        let chain_stats = scan_layout_chain::<F, D, Cfg>(layout_num_vars, new_layout)?;
        let a_cols = chain_stats.max_inner_width;
        let b_cols = chain_stats.max_outer_width;
        let d_cols = chain_stats.max_d_matrix_width;

        let max_rows = [envelope.max_n_a, envelope.max_n_b, envelope.max_n_d]
            .into_iter()
            .max()
            .unwrap();
        let max_cols = [a_cols, b_cols, d_cols].into_iter().max().unwrap();

        let existing_rows = existing.shared_matrix.num_rows();
        let existing_cols = existing.shared_matrix.first_row_len::<D>();

        let (shared_flat, ntt_shared) = if existing_rows >= max_rows && existing_cols >= max_cols {
            let ntt_shared = build_ntt_slot(existing.shared_matrix.view::<D>())?;
            (existing.shared_matrix.clone(), ntt_shared)
        } else {
            let actual_rows = max_rows.max(existing_rows);
            let actual_cols = max_cols.max(existing_cols);
            let shared_flat = derive_public_matrix_flat::<F, D>(actual_rows, actual_cols, &seed);
            let ntt_shared = build_ntt_slot(shared_flat.view::<D>())?;
            (shared_flat, ntt_shared)
        };

        let expanded = Arc::new(HachiExpandedSetup {
            seed: HachiSetupSeed {
                max_num_vars,
                layout: new_layout,
                public_matrix_seed: seed,
            },
            shared_matrix: shared_flat,
        });
        let prover_setup = HachiProverSetup {
            expanded: Arc::clone(&expanded),
            ntt_shared,
        };
        let verifier_setup = HachiVerifierSetup { expanded };
        Ok((prover_setup, verifier_setup))
    }

    fn setup_with_layout_and_seed<F, const D: usize, Cfg>(
        layout: HachiCommitmentLayout,
        max_num_vars: usize,
        public_matrix_seed: PublicMatrixSeed,
    ) -> Result<(HachiProverSetup<F, D>, HachiVerifierSetup<F>), HachiError>
    where
        F: FieldCore + CanonicalField + FieldSampling,
        Cfg: CommitmentConfig,
    {
        let layout_num_vars = layout.required_num_vars::<D>()?;
        let chain_stats = scan_layout_chain::<F, D, Cfg>(layout_num_vars, layout)?;
        let a_cols = chain_stats.max_inner_width;
        let b_cols = chain_stats.max_outer_width;
        let d_cols = chain_stats.max_d_matrix_width;

        Self::setup_with_matrix_widths_and_seed::<F, D, Cfg>(
            layout,
            max_num_vars,
            public_matrix_seed,
            a_cols,
            b_cols,
            d_cols,
        )
    }

    fn setup_with_matrix_widths_and_seed<F, const D: usize, Cfg>(
        layout: HachiCommitmentLayout,
        max_num_vars: usize,
        public_matrix_seed: PublicMatrixSeed,
        a_cols: usize,
        b_cols: usize,
        d_cols: usize,
    ) -> Result<(HachiProverSetup<F, D>, HachiVerifierSetup<F>), HachiError>
    where
        F: FieldCore + CanonicalField + FieldSampling,
        Cfg: CommitmentConfig,
    {
        let envelope = Cfg::envelope(max_num_vars);
        let max_rows = [envelope.max_n_a, envelope.max_n_b, envelope.max_n_d]
            .into_iter()
            .max()
            .unwrap();
        let max_cols = [a_cols, b_cols, d_cols].into_iter().max().unwrap();
        {
            let ring_bytes = std::mem::size_of::<CyclotomicRing<F, D>>();
            let shared_mb = (max_rows * max_cols * ring_bytes) as f64 / (1024.0_f64 * 1024.0_f64);
            tracing::debug!(
                a_cols,
                b_cols,
                d_cols,
                max_rows,
                max_cols,
                ring_bytes,
                shared_mb,
                "setup shared matrix size"
            );
        }
        let shared_flat =
            derive_public_matrix_flat::<F, D>(max_rows, max_cols, &public_matrix_seed);
        let ntt_shared = build_ntt_slot(shared_flat.view::<D>())?;

        let expanded = Arc::new(HachiExpandedSetup {
            seed: HachiSetupSeed {
                max_num_vars,
                layout,
                public_matrix_seed,
            },
            shared_matrix: shared_flat,
        });
        let prover_setup = HachiProverSetup {
            expanded: Arc::clone(&expanded),
            ntt_shared,
        };
        let verifier_setup = HachiVerifierSetup { expanded };
        Ok((prover_setup, verifier_setup))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::{HachiDeserialize, HachiSerialize};
    use crate::test_utils::{TinyConfig, F as TestF};

    #[test]
    fn expanded_setup_roundtrips_and_derives_same_verifier() {
        const TEST_D: usize = 64;
        let (prover_setup, verifier_setup) =
            <HachiCommitmentCore as RingCommitmentScheme<TestF, TEST_D, TinyConfig>>::setup(16)
                .unwrap();

        let mut bytes = Vec::new();
        prover_setup
            .expanded
            .serialize_compressed(&mut bytes)
            .unwrap();
        let decoded = HachiExpandedSetup::<TestF>::deserialize_compressed(&bytes[..], &()).unwrap();

        assert_eq!(decoded, prover_setup.expanded.as_ref().clone());

        let derived_verifier = HachiVerifierSetup {
            expanded: Arc::new(decoded.clone()),
        };
        assert_eq!(derived_verifier, verifier_setup);
    }

    #[test]
    fn setup_with_layouts_uses_exact_width_envelope() {
        const TEST_D: usize = 64;

        let layout_a =
            HachiCommitmentLayout::new::<TinyConfig>(4, 2, &TinyConfig::decomposition()).unwrap();
        let layout_b =
            HachiCommitmentLayout::new::<TinyConfig>(1, 6, &TinyConfig::decomposition()).unwrap();
        let params_a = TinyConfig::level_params(HachiScheduleInputs {
            max_num_vars: layout_a.required_num_vars::<TEST_D>().unwrap(),
            level: 0,
            current_w_len: 1usize << layout_a.required_num_vars::<TEST_D>().unwrap(),
        });
        let params_b = TinyConfig::level_params(HachiScheduleInputs {
            max_num_vars: layout_b.required_num_vars::<TEST_D>().unwrap(),
            level: 0,
            current_w_len: 1usize << layout_b.required_num_vars::<TEST_D>().unwrap(),
        });
        let w_layout_a =
            w_commitment_layout::<TestF, TEST_D, TinyConfig>(&params_a, layout_a).unwrap();
        let w_layout_b =
            w_commitment_layout::<TestF, TEST_D, TinyConfig>(&params_b, layout_b).unwrap();

        let expected_inner = [
            layout_a.inner_width,
            layout_b.inner_width,
            w_layout_a.inner_width,
            w_layout_b.inner_width,
        ]
        .into_iter()
        .max()
        .unwrap();
        let expected_outer = [
            layout_a.outer_width,
            layout_b.outer_width,
            w_layout_a.outer_width,
            w_layout_b.outer_width,
        ]
        .into_iter()
        .max()
        .unwrap();
        let expected_d = [
            layout_a.d_matrix_width,
            layout_b.d_matrix_width,
            w_layout_a.d_matrix_width,
            w_layout_b.d_matrix_width,
        ]
        .into_iter()
        .max()
        .unwrap();
        let expected_max_num_vars = [
            layout_a.required_num_vars::<TEST_D>().unwrap(),
            layout_b.required_num_vars::<TEST_D>().unwrap(),
        ]
        .into_iter()
        .max()
        .unwrap();

        let (setup, _) = HachiCommitmentCore::setup_with_layouts::<TestF, TEST_D, TinyConfig>(&[
            layout_a, layout_b,
        ])
        .unwrap();
        let envelope = setup.layout();

        assert_eq!(setup.expanded.seed.max_num_vars, expected_max_num_vars);
        assert_eq!(envelope.inner_width, expected_inner);
        assert_eq!(envelope.outer_width, expected_outer);
        assert_eq!(envelope.d_matrix_width, expected_d);
        let shared_cols = setup.expanded.shared_matrix.first_row_len::<TEST_D>();
        assert!(shared_cols >= expected_inner);
        assert!(shared_cols >= expected_outer);
        assert!(shared_cols >= expected_d);
    }

    #[cfg(feature = "disk-persistence")]
    mod disk_persistence {
        use super::*;
        use std::fs;

        fn cleanup_setup_file(max_num_vars: usize) {
            if let Some(path) = get_storage_path::<TinyConfig>(max_num_vars) {
                let _ = fs::remove_file(path);
            }
        }

        #[test]
        fn save_and_load_roundtrips() {
            const TEST_D: usize = 64;
            const MAX_VARS: usize = 100;

            cleanup_setup_file(MAX_VARS);

            let (prover_setup, _) =
                <HachiCommitmentCore as RingCommitmentScheme<TestF, TEST_D, TinyConfig>>::setup(
                    MAX_VARS,
                )
                .unwrap();

            let loaded = load_expanded_setup::<TestF, TinyConfig>(MAX_VARS).unwrap();
            assert_eq!(loaded, prover_setup.expanded.as_ref().clone());

            cleanup_setup_file(MAX_VARS);
        }

        #[test]
        fn setup_uses_cache_on_second_call() {
            const TEST_D: usize = 64;
            const MAX_VARS: usize = 101;

            cleanup_setup_file(MAX_VARS);

            let (first, _) =
                <HachiCommitmentCore as RingCommitmentScheme<TestF, TEST_D, TinyConfig>>::setup(
                    MAX_VARS,
                )
                .unwrap();

            let (second, _) =
                <HachiCommitmentCore as RingCommitmentScheme<TestF, TEST_D, TinyConfig>>::setup(
                    MAX_VARS,
                )
                .unwrap();

            assert_eq!(first.expanded, second.expanded);

            cleanup_setup_file(MAX_VARS);
        }

        #[test]
        fn ntt_caches_rebuilt_correctly_from_disk() {
            use crate::algebra::CyclotomicRing;

            const TEST_D: usize = 64;
            const MAX_VARS: usize = 102;

            cleanup_setup_file(MAX_VARS);

            let (fresh_setup, _) =
                <HachiCommitmentCore as RingCommitmentScheme<TestF, TEST_D, TinyConfig>>::setup(
                    MAX_VARS,
                )
                .unwrap();

            let loaded_expanded = load_expanded_setup::<TestF, TinyConfig>(MAX_VARS).unwrap();
            let (disk_setup, _) = setup_from_expanded::<TestF, TEST_D>(loaded_expanded).unwrap();

            let layout = fresh_setup.layout();
            let num_coeffs = layout.num_blocks * layout.block_len;
            let coeffs = vec![CyclotomicRing::<TestF, TEST_D>::zero(); num_coeffs];

            let fresh_commit = <HachiCommitmentCore as RingCommitmentScheme<
                TestF,
                TEST_D,
                TinyConfig,
            >>::commit_coeffs(&coeffs, &fresh_setup)
            .unwrap();
            let disk_commit = <HachiCommitmentCore as RingCommitmentScheme<
                TestF,
                TEST_D,
                TinyConfig,
            >>::commit_coeffs(&coeffs, &disk_setup)
            .unwrap();

            assert_eq!(fresh_commit.commitment, disk_commit.commitment);

            cleanup_setup_file(MAX_VARS);
        }
    }
}
