//! Ring-native §4.1 commitment core implementation.

use super::config::{
    ensure_block_layout, ensure_matrix_shape_ge, ensure_supported_num_vars,
    validate_and_derive_layout, HachiCommitmentLayout,
};
use super::onehot::{inner_ajtai_onehot_wide, map_onehot_to_sparse_blocks};
use super::scheme::{CommitWitness, RingCommitmentScheme};
use super::types::RingCommitment;
#[cfg(feature = "disk-persistence")]
use super::utils::crt_ntt::build_ntt_slots;
use super::utils::crt_ntt::{build_ntt_slot, NttSlotCache};
use super::utils::flat_matrix::FlatMatrix;
use super::utils::linear::{
    decompose_rows_i8, flatten_i8_blocks, mat_vec_mul_ntt_i8, mat_vec_mul_ntt_single_i8,
};
use super::utils::matrix::{derive_public_matrix, sample_public_matrix_seed, PublicMatrixSeed};
use super::CommitmentConfig;
use crate::algebra::fields::wide::HasWide;
use crate::algebra::CyclotomicRing;
use crate::error::HachiError;
#[cfg(feature = "parallel")]
use crate::parallel::*;
use crate::primitives::serialization::{
    Compress, HachiDeserialize, HachiSerialize, SerializationError, Valid, Validate,
};
use crate::protocol::hachi_poly_ops::OneHotIndex;
use crate::protocol::ring_switch::w_commitment_layout;
use crate::{cfg_into_iter, cfg_iter, CanonicalField, FieldCore, FieldSampling};
#[cfg(feature = "disk-persistence")]
use std::fs;
use std::io::{Read, Write};
#[cfg(feature = "disk-persistence")]
use std::path::PathBuf;

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

/// Expanded setup stage containing coefficient-form matrices stored as
/// D-agnostic flat field-element arrays.
///
/// The same `HachiExpandedSetup` can be viewed at different ring dimensions by
/// calling [`FlatMatrix::view`] or [`FlatMatrix::row`] with the desired
/// const-generic `D`.
#[allow(non_snake_case)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HachiExpandedSetup<F: FieldCore> {
    /// Setup seed and runtime layout metadata.
    pub seed: HachiSetupSeed,
    /// Inner matrix `A`.
    pub A: FlatMatrix<F>,
    /// Outer matrix `B`.
    pub B: FlatMatrix<F>,
    /// Prover matrix `D ∈ R_q^{n_D × δ·2^R}` (§4.2).
    pub D_mat: FlatMatrix<F>,
}

/// Prover setup artifact (expanded setup + per-matrix NTT caches).
///
/// The NTT caches are tied to a specific ring dimension D.
#[allow(non_snake_case)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HachiProverSetup<F: FieldCore, const D: usize> {
    /// Expanded matrix stage used by both prover and verifier.
    pub expanded: HachiExpandedSetup<F>,
    /// NTT cache for the A matrix.
    pub ntt_A: NttSlotCache<D>,
    /// NTT cache for the B matrix.
    pub ntt_B: NttSlotCache<D>,
    /// NTT cache for the D matrix.
    pub ntt_D: NttSlotCache<D>,
}

/// Verifier setup artifact derived from prover setup.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HachiVerifierSetup<F: FieldCore> {
    /// Expanded matrix stage used for verification.
    pub expanded: HachiExpandedSetup<F>,
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
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let max_num_vars = usize::deserialize_with_mode(&mut reader, compress, validate)?;
        let layout = HachiCommitmentLayout::deserialize_with_mode(&mut reader, compress, validate)?;
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
        self.A.check()?;
        self.B.check()?;
        self.D_mat.check()?;
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
        self.A.serialize_with_mode(&mut writer, compress)?;
        self.B.serialize_with_mode(&mut writer, compress)?;
        self.D_mat.serialize_with_mode(&mut writer, compress)?;
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.seed.serialized_size(compress)
            + self.A.serialized_size(compress)
            + self.B.serialized_size(compress)
            + self.D_mat.serialized_size(compress)
    }
}

impl<F: FieldCore + Valid> HachiDeserialize for HachiExpandedSetup<F> {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let out = Self {
            seed: HachiSetupSeed::deserialize_with_mode(&mut reader, compress, validate)?,
            A: FlatMatrix::deserialize_with_mode(&mut reader, compress, validate)?,
            B: FlatMatrix::deserialize_with_mode(&mut reader, compress, validate)?,
            D_mat: FlatMatrix::deserialize_with_mode(&mut reader, compress, validate)?,
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
    fn deserialize_with_mode<R: Read>(
        reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        Ok(Self {
            expanded: HachiExpandedSetup::deserialize_with_mode(reader, compress, validate)?,
        })
    }
}

#[cfg(feature = "disk-persistence")]
fn get_storage_path(max_num_vars: usize) -> Option<PathBuf> {
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
        path.push(format!("hachi_{max_num_vars}.setup"));
        path
    })
}

#[cfg(feature = "disk-persistence")]
fn save_expanded_setup<F: FieldCore>(setup: &HachiExpandedSetup<F>, max_num_vars: usize) {
    let Some(storage_path) = get_storage_path(max_num_vars) else {
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
fn load_expanded_setup<F: FieldCore + Valid>(
    max_num_vars: usize,
) -> Result<HachiExpandedSetup<F>, HachiError> {
    let storage_path = get_storage_path(max_num_vars).ok_or_else(|| {
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
/// reconstructing the NTT caches.
#[cfg(feature = "disk-persistence")]
pub(crate) fn setup_from_expanded<F: FieldCore + CanonicalField, const D: usize>(
    expanded: HachiExpandedSetup<F>,
) -> Result<(HachiProverSetup<F, D>, HachiVerifierSetup<F>), HachiError> {
    let (ntt_a, ntt_b, ntt_d) = build_ntt_slots(
        expanded.A.view::<D>(),
        expanded.B.view::<D>(),
        expanded.D_mat.view::<D>(),
    )?;
    let prover_setup = HachiProverSetup {
        expanded: expanded.clone(),
        ntt_A: ntt_a,
        ntt_B: ntt_b,
        ntt_D: ntt_d,
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
        ensure_supported_num_vars(max_num_vars, layout.required_num_vars::<D>()?)?;

        #[cfg(feature = "disk-persistence")]
        {
            match load_expanded_setup::<F>(max_num_vars) {
                Ok(expanded) => {
                    tracing::info!("Loaded setup from disk, rebuilding NTT caches");
                    return setup_from_expanded(expanded);
                }
                Err(HachiError::InvalidSetup(msg)) if msg.contains("not found") => {
                    tracing::debug!("Setup file not found, will generate new one");
                }
                Err(e) => {
                    panic!("Failed to load setup from disk: {e}");
                }
            }
        }

        let w_layout = w_commitment_layout::<F, D, Cfg>(layout)?;
        let a_cols = layout.inner_width.max(w_layout.inner_width);
        let b_cols = layout.outer_width.max(w_layout.outer_width);
        let d_cols = layout.d_matrix_width.max(w_layout.d_matrix_width);

        let public_matrix_seed = sample_public_matrix_seed();
        let a_matrix = derive_public_matrix::<F, D>(Cfg::N_A, a_cols, &public_matrix_seed, b"A");
        let b_matrix = derive_public_matrix::<F, D>(Cfg::N_B, b_cols, &public_matrix_seed, b"B");
        let d_matrix = derive_public_matrix::<F, D>(Cfg::N_D, d_cols, &public_matrix_seed, b"D");

        let a_flat = FlatMatrix::from_ring_matrix(&a_matrix);
        let b_flat = FlatMatrix::from_ring_matrix(&b_matrix);
        let d_flat = FlatMatrix::from_ring_matrix(&d_matrix);

        let ntt_a = build_ntt_slot(a_flat.view::<D>())?;
        let ntt_b = build_ntt_slot(b_flat.view::<D>())?;
        let ntt_d = build_ntt_slot(d_flat.view::<D>())?;
        let expanded = HachiExpandedSetup {
            seed: HachiSetupSeed {
                max_num_vars,
                layout,
                public_matrix_seed,
            },
            A: a_flat,
            B: b_flat,
            D_mat: d_flat,
        };

        #[cfg(feature = "disk-persistence")]
        save_expanded_setup(&expanded, max_num_vars);

        let prover_setup = HachiProverSetup {
            expanded: expanded.clone(),
            ntt_A: ntt_a,
            ntt_B: ntt_b,
            ntt_D: ntt_d,
        };
        let verifier_setup = HachiVerifierSetup { expanded };
        ensure_matrix_shape_ge::<F, D>(
            &prover_setup.expanded.A,
            Cfg::N_A,
            layout.inner_width,
            "A",
        )?;
        ensure_matrix_shape_ge::<F, D>(
            &prover_setup.expanded.B,
            Cfg::N_B,
            layout.outer_width,
            "B",
        )?;
        ensure_matrix_shape_ge::<F, D>(
            &prover_setup.expanded.D_mat,
            Cfg::N_D,
            layout.d_matrix_width,
            "D",
        )?;
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
        ensure_supported_num_vars(
            setup.expanded.seed.max_num_vars,
            layout.required_num_vars::<D>()?,
        )?;
        ensure_block_layout(f_blocks, layout)?;
        ensure_matrix_shape_ge::<F, D>(&setup.expanded.A, Cfg::N_A, layout.inner_width, "A")?;
        ensure_matrix_shape_ge::<F, D>(&setup.expanded.B, Cfg::N_B, layout.outer_width, "B")?;

        let depth_commit = layout.num_digits_commit;
        let depth_open = layout.num_digits_open;
        let log_basis = layout.log_basis;
        let block_slices: Vec<&[CyclotomicRing<F, D>]> =
            f_blocks.iter().map(|b| b.as_slice()).collect();
        let t_all = mat_vec_mul_ntt_i8(&setup.ntt_A, &block_slices, depth_commit, log_basis);
        let t_hat_all: Vec<Vec<[i8; D]>> = cfg_into_iter!(t_all)
            .map(|t_i| decompose_rows_i8(&t_i, depth_open, log_basis))
            .collect();

        let t_hat_flat = flatten_i8_blocks(&t_hat_all);

        let u: Vec<CyclotomicRing<F, D>> = mat_vec_mul_ntt_single_i8(&setup.ntt_B, &t_hat_flat);
        Ok(CommitWitness::new(RingCommitment { u }, t_hat_all))
    }

    #[tracing::instrument(skip_all, name = "RingCommitmentScheme::commit_coeffs")]
    fn commit_coeffs(
        f_coeffs: &[CyclotomicRing<F, D>],
        setup: &Self::ProverSetup,
    ) -> Result<CommitWitness<Self::Commitment, F, D>, HachiError> {
        let layout = setup.layout();
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

        let t_all = mat_vec_mul_ntt_i8(&setup.ntt_A, &block_slices, depth_commit, log_basis);
        let t_hat_all: Vec<Vec<[i8; D]>> = cfg_into_iter!(t_all)
            .map(|t_i| decompose_rows_i8(&t_i, depth_open, log_basis))
            .collect();

        let t_hat_flat = flatten_i8_blocks(&t_hat_all);

        let u: Vec<CyclotomicRing<F, D>> = mat_vec_mul_ntt_single_i8(&setup.ntt_B, &t_hat_flat);
        Ok(CommitWitness::new(RingCommitment { u }, t_hat_all))
    }

    #[tracing::instrument(skip_all, name = "RingCommitmentScheme::commit_onehot")]
    fn commit_onehot<I: OneHotIndex>(
        onehot_k: usize,
        indices: &[Option<I>],
        setup: &Self::ProverSetup,
    ) -> Result<CommitWitness<Self::Commitment, F, D>, HachiError> {
        let layout = setup.layout();
        ensure_supported_num_vars(
            setup.expanded.seed.max_num_vars,
            layout.required_num_vars::<D>()?,
        )?;
        ensure_matrix_shape_ge::<F, D>(&setup.expanded.A, Cfg::N_A, layout.inner_width, "A")?;
        ensure_matrix_shape_ge::<F, D>(&setup.expanded.B, Cfg::N_B, layout.outer_width, "B")?;

        let sparse_blocks =
            map_onehot_to_sparse_blocks(onehot_k, indices, layout.r_vars, layout.m_vars, D)?;

        let depth_commit = layout.num_digits_commit;
        let depth_open = layout.num_digits_open;
        let log_basis = layout.log_basis;
        let zero_block_len = Cfg::N_A.checked_mul(depth_open).unwrap();
        let a_view = setup.expanded.A.view::<D>();
        let block_len = layout.block_len;

        let t_hat_all: Vec<Vec<[i8; D]>> = cfg_iter!(sparse_blocks)
            .map(|block_entries| {
                if block_entries.is_empty() {
                    vec![[0i8; D]; zero_block_len]
                } else {
                    let t_i =
                        inner_ajtai_onehot_wide(&a_view, block_entries, block_len, depth_commit);
                    decompose_rows_i8(&t_i, depth_open, log_basis)
                }
            })
            .collect();

        let t_hat_flat = flatten_i8_blocks(&t_hat_all);

        let u: Vec<CyclotomicRing<F, D>> = mat_vec_mul_ntt_single_i8(&setup.ntt_B, &t_hat_flat);
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

        for &layout in layouts {
            if layout.log_basis != first_layout.log_basis {
                return Err(HachiError::InvalidSetup(format!(
                    "setup_with_layouts requires a shared log_basis (expected {}, got {})",
                    first_layout.log_basis, layout.log_basis
                )));
            }

            max_num_vars = max_num_vars.max(layout.required_num_vars::<D>()?);
            max_inner_width = max_inner_width.max(layout.inner_width);
            max_outer_width = max_outer_width.max(layout.outer_width);
            max_d_matrix_width = max_d_matrix_width.max(layout.d_matrix_width);
            max_r_vars = max_r_vars.max(layout.r_vars);
            max_num_digits_open = max_num_digits_open.max(layout.num_digits_open);
            max_num_digits_fold = max_num_digits_fold.max(layout.num_digits_fold);

            let w_layout = w_commitment_layout::<F, D, Cfg>(layout)?;
            if std::env::var_os("HACHI_SETUP_DIAGNOSTICS").is_some() {
                eprintln!("[hachi setup] layout={layout:?}");
                eprintln!("[hachi setup] w_layout={w_layout:?}");
            }
            max_inner_width = max_inner_width.max(w_layout.inner_width);
            max_outer_width = max_outer_width.max(w_layout.outer_width);
            max_d_matrix_width = max_d_matrix_width.max(w_layout.d_matrix_width);
            max_r_vars = max_r_vars.max(w_layout.r_vars);
            max_num_digits_open = max_num_digits_open.max(w_layout.num_digits_open);
            max_num_digits_fold = max_num_digits_fold.max(w_layout.num_digits_fold);
        }

        let envelope_layout = Self::layout_envelope::<D>(
            max_num_vars,
            max_inner_width,
            max_outer_width,
            max_d_matrix_width,
            max_r_vars,
            max_num_digits_open,
            max_num_digits_fold,
            first_layout.log_basis,
        )?;
        if std::env::var_os("HACHI_SETUP_DIAGNOSTICS").is_some() {
            eprintln!("[hachi setup] envelope_layout={envelope_layout:?}");
            eprintln!("[hachi setup] max_num_vars={max_num_vars}");
        }
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
    /// A matrix (which depends only on `m_vars`). Only regenerates B and D
    /// matrices for the new `r_vars`.
    ///
    /// Use this when creating a mega-polynomial setup that shares `m_vars` with
    /// an individual polynomial setup — avoids re-deriving and NTT-transforming
    /// the A matrix.
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

        let w_layout = w_commitment_layout::<F, D, Cfg>(new_layout)?;
        let a_width = existing.A.first_row_len::<D>();
        if a_width < w_layout.inner_width {
            return Err(HachiError::InvalidSetup(format!(
                "existing A width {a_width} < w inner_width {}",
                w_layout.inner_width
            )));
        }
        let b_cols = new_layout.outer_width.max(w_layout.outer_width);

        let max_num_vars = new_layout.required_num_vars::<D>()?;
        let seed = existing.seed.public_matrix_seed;

        let d_cols = new_layout.d_matrix_width.max(w_layout.d_matrix_width);
        let b_matrix = derive_public_matrix::<F, D>(Cfg::N_B, b_cols, &seed, b"B");
        let d_matrix = derive_public_matrix::<F, D>(Cfg::N_D, d_cols, &seed, b"D");

        let b_flat = FlatMatrix::from_ring_matrix(&b_matrix);
        let d_flat = FlatMatrix::from_ring_matrix(&d_matrix);

        let ntt_a = build_ntt_slot(existing.A.view::<D>())?;
        let ntt_b = build_ntt_slot(b_flat.view::<D>())?;
        let ntt_d = build_ntt_slot(d_flat.view::<D>())?;
        let expanded = HachiExpandedSetup {
            seed: HachiSetupSeed {
                max_num_vars,
                layout: new_layout,
                public_matrix_seed: seed,
            },
            A: existing.A.clone(),
            B: b_flat,
            D_mat: d_flat,
        };
        let prover_setup = HachiProverSetup {
            expanded: expanded.clone(),
            ntt_A: ntt_a,
            ntt_B: ntt_b,
            ntt_D: ntt_d,
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
        let w_layout = w_commitment_layout::<F, D, Cfg>(layout)?;
        let a_cols = layout.inner_width.max(w_layout.inner_width);
        let b_cols = layout.outer_width.max(w_layout.outer_width);
        let d_cols = layout.d_matrix_width.max(w_layout.d_matrix_width);

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
        if std::env::var_os("HACHI_SETUP_DIAGNOSTICS").is_some() {
            let ring_bytes = std::mem::size_of::<CyclotomicRing<F, D>>();
            let a_raw_mb = (Cfg::N_A * a_cols * ring_bytes) as f64 / (1024.0_f64 * 1024.0_f64);
            let b_raw_mb = (Cfg::N_B * b_cols * ring_bytes) as f64 / (1024.0_f64 * 1024.0_f64);
            let d_raw_mb = (Cfg::N_D * d_cols * ring_bytes) as f64 / (1024.0_f64 * 1024.0_f64);
            eprintln!(
                "[hachi setup] a_cols={a_cols}, b_cols={b_cols}, d_cols={d_cols}, ring_bytes={ring_bytes}"
            );
            eprintln!(
                "[hachi setup] raw_matrix_mb: A={a_raw_mb:.1}, B={b_raw_mb:.1}, D={d_raw_mb:.1}, total={:.1}",
                a_raw_mb + b_raw_mb + d_raw_mb
            );
        }
        let a_matrix = derive_public_matrix::<F, D>(Cfg::N_A, a_cols, &public_matrix_seed, b"A");
        let b_matrix = derive_public_matrix::<F, D>(Cfg::N_B, b_cols, &public_matrix_seed, b"B");
        let d_matrix = derive_public_matrix::<F, D>(Cfg::N_D, d_cols, &public_matrix_seed, b"D");

        let a_flat = FlatMatrix::from_ring_matrix(&a_matrix);
        let b_flat = FlatMatrix::from_ring_matrix(&b_matrix);
        let d_flat = FlatMatrix::from_ring_matrix(&d_matrix);

        let ntt_a = build_ntt_slot(a_flat.view::<D>())?;
        let ntt_b = build_ntt_slot(b_flat.view::<D>())?;
        let ntt_d = build_ntt_slot(d_flat.view::<D>())?;
        let expanded = HachiExpandedSetup {
            seed: HachiSetupSeed {
                max_num_vars,
                layout,
                public_matrix_seed,
            },
            A: a_flat,
            B: b_flat,
            D_mat: d_flat,
        };
        let prover_setup = HachiProverSetup {
            expanded: expanded.clone(),
            ntt_A: ntt_a,
            ntt_B: ntt_b,
            ntt_D: ntt_d,
        };
        let verifier_setup = HachiVerifierSetup { expanded };
        ensure_matrix_shape_ge::<F, D>(
            &prover_setup.expanded.A,
            Cfg::N_A,
            layout.inner_width,
            "A",
        )?;
        ensure_matrix_shape_ge::<F, D>(
            &prover_setup.expanded.B,
            Cfg::N_B,
            layout.outer_width,
            "B",
        )?;
        ensure_matrix_shape_ge::<F, D>(
            &prover_setup.expanded.D_mat,
            Cfg::N_D,
            layout.d_matrix_width,
            "D",
        )?;
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
        let decoded = HachiExpandedSetup::<TestF>::deserialize_compressed(&bytes[..]).unwrap();

        assert_eq!(decoded, prover_setup.expanded);

        let derived_verifier = HachiVerifierSetup {
            expanded: decoded.clone(),
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
        let w_layout_a = w_commitment_layout::<TestF, TEST_D, TinyConfig>(layout_a).unwrap();
        let w_layout_b = w_commitment_layout::<TestF, TEST_D, TinyConfig>(layout_b).unwrap();

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
        assert_eq!(setup.expanded.A.first_row_len::<TEST_D>(), expected_inner);
        assert_eq!(setup.expanded.B.first_row_len::<TEST_D>(), expected_outer);
        assert_eq!(setup.expanded.D_mat.first_row_len::<TEST_D>(), expected_d);
    }

    #[cfg(feature = "disk-persistence")]
    mod disk_persistence {
        use super::*;
        use std::fs;

        fn cleanup_setup_file(max_num_vars: usize) {
            if let Some(path) = get_storage_path(max_num_vars) {
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

            let loaded = load_expanded_setup::<TestF>(MAX_VARS).unwrap();
            assert_eq!(loaded, prover_setup.expanded);

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

            let loaded_expanded = load_expanded_setup::<TestF>(MAX_VARS).unwrap();
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
