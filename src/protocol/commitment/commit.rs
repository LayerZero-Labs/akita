//! Ring-native §4.1 commitment core implementation.

use super::config::{
    compute_num_digits, ensure_block_layout, ensure_layout_supported_num_vars,
    validate_and_derive_layout,
};
use super::onehot::{inner_ajtai_onehot_wide, map_onehot_to_sparse_blocks};
use super::schedule::HachiScheduleInputs;
use super::schedule::{
    hachi_root_runtime_plan_from_root_layout, HachiRootBatchSummary, HachiScheduleLookupKey,
};
use super::scheme::{CommitWitness, RingCommitmentScheme};
use super::types::RingCommitment;
use super::utils::crt_ntt::{build_ntt_slot, NttSlotCache};
use super::utils::flat_matrix::FlatMatrix;
use super::utils::linear::{
    decompose_rows_i8_into, mat_vec_mul_ntt_i8_dense, mat_vec_mul_ntt_i8_dense_single_row,
    mat_vec_mul_ntt_single_i8,
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
use crate::protocol::params::LevelParams;
use crate::protocol::proof::FlatDigitBlocks;
use crate::protocol::ring_switch::w_ring_element_count;
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
    /// Maximum inner (A-matrix) width across all recursion levels.
    pub max_inner_width: usize,
    /// Maximum outer (B-matrix) width across all recursion levels.
    pub max_outer_width: usize,
    /// Maximum D-matrix width across all recursion levels.
    pub max_d_matrix_width: usize,
    /// Total ring-element count for the 1D flat backing vector.
    pub max_total_ring_elements: usize,
    /// Public seed used to derive commitment matrices.
    pub public_matrix_seed: PublicMatrixSeed,
}

impl HachiSetupSeed {
    /// Global row stride for the flat NTT cache.
    ///
    /// All levels and all matrix roles (A, B, D) share the same flat cache.
    /// The stride is the maximum column width across all levels and roles,
    /// ensuring that `(row, col)` maps to the same physical element regardless
    /// of which level or role is accessing it.
    pub fn max_stride(&self) -> usize {
        self.max_inner_width
            .max(self.max_outer_width)
            .max(self.max_d_matrix_width)
    }
}

/// Expanded setup stage containing a single shared coefficient-form matrix
/// stored as a D-agnostic flat field-element array.
///
/// All role matrices (A, B, D) are row/column prefixes of this shared vector.
/// See `SHARED_PREFIX_BINDING.md` for the security argument. The same setup
/// can be viewed at different ring dimensions by calling
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

impl<F: FieldCore> HachiExpandedSetup<F> {
    /// Maximum batched root-polynomial capacity carried by this setup.
    pub fn max_num_batched_polys(&self) -> usize {
        self.seed.max_num_batched_polys
    }

    /// Return an error if `lp` exceeds the setup's matrix-width envelope.
    ///
    /// # Errors
    ///
    /// Returns [`HachiError::InvalidSetup`] if any matrix width in `lp`
    /// exceeds the setup envelope.
    pub fn ensure_layout_fits(&self, lp: &LevelParams) -> Result<(), HachiError> {
        let seed = &self.seed;
        if lp.inner_width() > seed.max_inner_width {
            return Err(HachiError::InvalidSetup(format!(
                "A matrix too narrow: need {} but setup has {}",
                lp.inner_width(),
                seed.max_inner_width
            )));
        }
        if lp.outer_width() > seed.max_outer_width {
            return Err(HachiError::InvalidSetup(format!(
                "B matrix too narrow: need {} but setup has {}",
                lp.outer_width(),
                seed.max_outer_width
            )));
        }
        if lp.d_matrix_width() > seed.max_d_matrix_width {
            return Err(HachiError::InvalidSetup(format!(
                "D matrix too narrow: need {} but setup has {}",
                lp.d_matrix_width(),
                seed.max_d_matrix_width
            )));
        }
        Ok(())
    }

    /// Return an error if the batched root split exceeds the setup's
    /// matrix-width envelope.
    ///
    /// # Errors
    ///
    /// Returns [`HachiError::InvalidSetup`] if the widths implied by
    /// `split` and `num_claims` exceed the setup envelope.
    pub(crate) fn ensure_batched_root_split_fits(
        &self,
        split: &BatchedRootSplit,
        num_claims: usize,
    ) -> Result<(), HachiError> {
        let p = &split.params;
        let inner_width = p
            .block_len
            .checked_mul(p.num_digits_commit)
            .ok_or_else(|| HachiError::InvalidSetup("inner width overflow".to_string()))?;
        let outer_width = p
            .a_key
            .row_len
            .checked_mul(p.num_digits_open)
            .and_then(|x| x.checked_mul(p.num_blocks))
            .and_then(|x| x.checked_mul(num_claims))
            .ok_or_else(|| HachiError::InvalidSetup("batched outer width overflow".to_string()))?;
        let d_matrix_width = p
            .num_digits_open
            .checked_mul(p.num_blocks)
            .and_then(|x| x.checked_mul(num_claims))
            .ok_or_else(|| HachiError::InvalidSetup("batched D width overflow".to_string()))?;

        let seed = &self.seed;
        if inner_width > seed.max_inner_width {
            return Err(HachiError::InvalidSetup(format!(
                "A matrix too narrow: need {} but setup has {}",
                inner_width, seed.max_inner_width
            )));
        }
        if outer_width > seed.max_outer_width {
            return Err(HachiError::InvalidSetup(format!(
                "B matrix too narrow: need {} but setup has {}",
                outer_width, seed.max_outer_width
            )));
        }
        if d_matrix_width > seed.max_d_matrix_width {
            return Err(HachiError::InvalidSetup(format!(
                "D matrix too narrow: need {} but setup has {}",
                d_matrix_width, seed.max_d_matrix_width
            )));
        }
        Ok(())
    }

    /// Panic if `lp` exceeds the matrix-width envelope carried by this setup.
    ///
    /// # Panics
    ///
    /// Panics if any of `lp`'s matrix widths exceed the setup envelope.
    pub fn assert_layout_fits(&self, lp: &LevelParams) {
        self.ensure_layout_fits(lp)
            .unwrap_or_else(|err| panic!("{err}"));
    }
}

impl<F: FieldCore, const D: usize> HachiProverSetup<F, D> {
    /// Maximum batched root-polynomial capacity carried by this setup.
    pub fn max_num_batched_polys(&self) -> usize {
        self.expanded.max_num_batched_polys()
    }

    /// Return an error if `lp` exceeds this setup's matrix-width envelope.
    ///
    /// # Errors
    ///
    /// Returns [`HachiError::InvalidSetup`] if any matrix width in `lp`
    /// exceeds the setup envelope.
    pub fn ensure_layout_fits(&self, lp: &LevelParams) -> Result<(), HachiError> {
        self.expanded.ensure_layout_fits(lp)
    }

    /// Return an error if the batched root split exceeds this setup's
    /// matrix-width envelope.
    ///
    /// # Errors
    ///
    /// Returns [`HachiError::InvalidSetup`] if the widths implied by
    /// `split` and `num_claims` exceed the setup envelope.
    pub(crate) fn ensure_batched_root_split_fits(
        &self,
        split: &BatchedRootSplit,
        num_claims: usize,
    ) -> Result<(), HachiError> {
        self.expanded
            .ensure_batched_root_split_fits(split, num_claims)
    }

    /// Panic if `lp`'s matrix dimensions exceed this setup's maximums.
    ///
    /// # Panics
    ///
    /// Panics if any of `lp`'s matrix widths (inner, outer, D) exceed
    /// those of this setup.
    pub fn assert_layout_fits(&self, lp: &LevelParams) {
        self.expanded.assert_layout_fits(lp);
    }
}

impl Valid for HachiSetupSeed {
    fn check(&self) -> Result<(), SerializationError> {
        if self.max_inner_width == 0 || self.max_outer_width == 0 || self.max_d_matrix_width == 0 {
            return Err(SerializationError::InvalidData(
                "setup seed matrix widths must be non-zero".to_string(),
            ));
        }
        if self.max_total_ring_elements == 0 {
            return Err(SerializationError::InvalidData(
                "setup seed max_total_ring_elements must be non-zero".to_string(),
            ));
        }
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
        self.max_inner_width
            .serialize_with_mode(&mut writer, compress)?;
        self.max_outer_width
            .serialize_with_mode(&mut writer, compress)?;
        self.max_d_matrix_width
            .serialize_with_mode(&mut writer, compress)?;
        self.max_total_ring_elements
            .serialize_with_mode(&mut writer, compress)?;
        writer.write_all(&self.public_matrix_seed)?;
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.max_num_vars.serialized_size(compress)
            + self.max_num_batched_polys.serialized_size(compress)
            + self.max_inner_width.serialized_size(compress)
            + self.max_outer_width.serialized_size(compress)
            + self.max_d_matrix_width.serialized_size(compress)
            + self.max_total_ring_elements.serialized_size(compress)
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
        let max_inner_width = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let max_outer_width = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let max_d_matrix_width =
            usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let max_total_ring_elements =
            usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let mut public_matrix_seed = [0u8; 32];
        reader.read_exact(&mut public_matrix_seed)?;
        let out = Self {
            max_num_vars,
            max_num_batched_polys,
            max_inner_width,
            max_outer_width,
            max_d_matrix_width,
            max_total_ring_elements,
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

pub(crate) fn root_current_w_len<const D: usize>(lp: &LevelParams) -> usize {
    lp.num_blocks
        .checked_mul(lp.block_len)
        .and_then(|len| len.checked_mul(D))
        .unwrap_or(0)
}

#[derive(Debug, Clone, Copy, Default)]
struct LayoutChainStats {
    max_inner_width: usize,
    max_outer_width: usize,
    max_d_matrix_width: usize,
    max_n_a: usize,
    max_n_b: usize,
    max_n_d: usize,
    max_r_vars: usize,
    max_num_digits_open: usize,
    max_num_digits_fold: usize,
    max_log_basis: u32,
}

impl LayoutChainStats {
    fn include(&mut self, lp: &LevelParams) {
        self.max_inner_width = self.max_inner_width.max(lp.inner_width());
        self.max_outer_width = self.max_outer_width.max(lp.outer_width());
        self.max_d_matrix_width = self.max_d_matrix_width.max(lp.d_matrix_width());
        self.max_r_vars = self.max_r_vars.max(lp.r_vars);
        self.max_num_digits_open = self.max_num_digits_open.max(lp.num_digits_open);
        self.max_num_digits_fold = self.max_num_digits_fold.max(lp.num_digits_fold);
        self.max_log_basis = self.max_log_basis.max(lp.log_basis);
        self.max_n_a = self.max_n_a.max(lp.a_key.row_len);
        self.max_n_b = self.max_n_b.max(lp.b_key.row_len);
        self.max_n_d = self.max_n_d.max(lp.d_key.row_len);
    }
}

fn compute_num_digits_fold_batched(
    r_vars: usize,
    challenge_l1_mass: usize,
    log_basis: u32,
    num_claims: usize,
) -> usize {
    let shift = r_vars + (log_basis as usize) - 1;
    if shift >= 127 || challenge_l1_mass == 0 {
        return compute_num_digits(128, log_basis);
    }
    let beta = (challenge_l1_mass as u128)
        .saturating_mul(num_claims as u128)
        .saturating_mul(1u128 << shift);
    if beta == 0 {
        return 1;
    }
    let log_beta = 128 - beta.leading_zeros();
    compute_num_digits(log_beta, log_basis)
}

pub(crate) fn scale_batched_root_layout<Cfg, const D: usize>(
    max_num_vars: usize,
    root_lp: &LevelParams,
    num_claims: usize,
) -> Result<LevelParams, HachiError>
where
    Cfg: CommitmentConfig,
{
    if num_claims == 0 {
        return Err(HachiError::InvalidSetup(
            "max_num_batched_polys must be at least 1".to_string(),
        ));
    }

    let root_inputs = HachiScheduleInputs {
        max_num_vars,
        level: 0,
        current_w_len: root_current_w_len::<D>(root_lp),
    };
    let root_stage1_config =
        Cfg::stage1_challenge_config(Cfg::d_at_level(0, root_inputs.current_w_len));
    let mut scaled = root_lp.clone();
    scaled.b_key.col_len = root_lp
        .b_key
        .col_len
        .checked_mul(num_claims)
        .ok_or_else(|| HachiError::InvalidSetup("batched outer width overflow".to_string()))?;
    scaled.d_key.col_len = root_lp
        .d_key
        .col_len
        .checked_mul(num_claims)
        .ok_or_else(|| HachiError::InvalidSetup("batched D width overflow".to_string()))?;
    scaled.num_digits_fold = root_lp.num_digits_fold.max(compute_num_digits_fold_batched(
        root_lp.r_vars,
        root_stage1_config.l1_mass(),
        root_lp.log_basis,
        num_claims,
    ));
    Ok(scaled)
}

/// Planner-derived batched root split parameters.
pub(crate) struct BatchedRootSplit {
    /// Unified per-level parameters for the root split.
    pub params: LevelParams,
    /// Batched fold digits (from the planner's candidate layout).
    /// May differ from `params.num_digits_fold` when batching amplifies
    /// the fold bound.
    pub num_digits_fold_batched: usize,
}

/// Extract `BatchedRootSplit` from a pre-computed `HachiSchedulePlan`'s
/// first fold level, if one exists.
fn split_from_schedule_plan(plan: &super::schedule::HachiSchedulePlan) -> Option<BatchedRootSplit> {
    use super::config::compute_num_digits_fold;

    let root_level = plan.fold_levels().next()?;
    let per_poly_fold = compute_num_digits_fold(
        root_level.lp.r_vars,
        root_level.lp.challenge_l1_mass(),
        root_level.lp.log_basis,
    );
    let mut lp = root_level.lp.clone();
    let batched_fold = lp.num_digits_fold;
    lp.num_digits_fold = per_poly_fold;
    Some(BatchedRootSplit {
        params: lp,
        num_digits_fold_batched: batched_fold,
    })
}

fn fallback_batched_root_split<Cfg, const D: usize>(
    max_num_vars: usize,
    num_claims: usize,
) -> Result<BatchedRootSplit, HachiError>
where
    Cfg: CommitmentConfig,
{
    let root_lp = Cfg::commitment_layout(max_num_vars)?;
    let scaled_lp = scale_batched_root_layout::<Cfg, D>(max_num_vars, &root_lp, num_claims)?;
    let inputs = HachiScheduleInputs {
        max_num_vars,
        level: 0,
        current_w_len: root_current_w_len::<D>(&root_lp),
    };
    let lp = Cfg::root_level_params_for_layout_with_log_basis(inputs, &scaled_lp)?;
    Ok(BatchedRootSplit {
        num_digits_fold_batched: scaled_lp.num_digits_fold,
        params: lp,
    })
}

/// Find the optimal `(log_basis, m, r)` triple for a batched root opening.
///
/// First checks the pre-computed generated tables.  Falls back to the DP
/// planner only when no table entry exists.
pub(crate) fn optimal_root_batch_split<Cfg, const D: usize>(
    max_num_vars: usize,
    num_claims: usize,
) -> Result<BatchedRootSplit, HachiError>
where
    Cfg: CommitmentConfig,
{
    let lookup_key = HachiScheduleLookupKey::with_batch(
        max_num_vars,
        max_num_vars,
        num_claims,
        HachiRootBatchSummary::new(num_claims, 1, 1)?,
    );
    if let Some(plan) = Cfg::schedule_plan(lookup_key)? {
        if let Some(split) = split_from_schedule_plan(&plan) {
            tracing::info!(
                max_num_vars,
                num_claims,
                total_bytes = plan.exact_proof_bytes,
                root_m = split.params.log_block_len(),
                root_r = split.params.log_num_blocks(),
                root_lb = split.params.log_basis,
                "batched root split: read from pre-computed table"
            );
            return Ok(split);
        }
        let split = fallback_batched_root_split::<Cfg, D>(max_num_vars, num_claims)?;
        tracing::info!(
            max_num_vars,
            num_claims,
            "batched root split: schedule is direct-only, falling back to config root layout"
        );
        return Ok(split);
    }

    use crate::planner::schedule_params::{find_optimal_batched_schedule, BatchConfig, Step};

    let batch = BatchConfig {
        num_claims,
        num_commitment_groups: 1,
        num_points: 1,
    };
    let schedule = find_optimal_batched_schedule::<Cfg, D>(max_num_vars, batch)?;

    let root_step = match schedule.steps.first() {
        Some(Step::Fold(step)) => step,
        _ => return fallback_batched_root_split::<Cfg, D>(max_num_vars, num_claims),
    };

    let mut lp = root_step.params.clone();
    let batched_fold = lp.num_digits_fold;
    lp.num_digits_fold = root_step.delta_fold_per_poly;

    tracing::info!(
        max_num_vars,
        num_claims,
        total_bytes = schedule.total_bytes,
        root_m = lp.log_block_len(),
        root_r = lp.log_num_blocks(),
        root_lb = lp.log_basis,
        "batched root split: computed from scratch by DP planner (no pre-computed table)"
    );

    Ok(BatchedRootSplit {
        params: lp,
        num_digits_fold_batched: batched_fold,
    })
}

pub(crate) fn root_batched_layout<Cfg, const D: usize>(
    max_num_vars: usize,
    root_lp: &LevelParams,
    max_num_batched_polys: usize,
) -> Result<LevelParams, HachiError>
where
    Cfg: CommitmentConfig,
{
    let optimized_root_lp = if max_num_batched_polys > 1 {
        let split = optimal_root_batch_split::<Cfg, D>(max_num_vars, max_num_batched_polys)?;
        let mut lp = split.params.clone();
        lp.num_digits_fold = split.num_digits_fold_batched;
        lp
    } else {
        root_lp.clone()
    };
    scale_batched_root_layout::<Cfg, D>(max_num_vars, &optimized_root_lp, max_num_batched_polys)
}

/// Derive the per-polynomial commitment layout optimized for a batch of
/// `num_claims` polynomials with `max_num_vars` variables.
///
/// When `num_claims <= 1` this returns the singleton layout from
/// [`CommitmentConfig::commitment_layout`]. For larger batches the
/// `m_vars`/`r_vars` split is optimized to minimize proof size.
///
/// # Errors
///
/// Returns an error if the layout parameters overflow or are invalid.
pub fn hachi_batched_root_layout<Cfg, const D: usize>(
    max_num_vars: usize,
    num_claims: usize,
) -> Result<LevelParams, HachiError>
where
    Cfg: CommitmentConfig,
{
    if num_claims <= 1 {
        return Cfg::commitment_layout(max_num_vars);
    }

    let split = optimal_root_batch_split::<Cfg, D>(max_num_vars, num_claims)?;
    Ok(split.params)
}

fn scan_layout_chain<F, const D: usize, Cfg>(
    max_num_vars: usize,
    root_lp: &LevelParams,
    max_num_batched_polys: usize,
) -> Result<LayoutChainStats, HachiError>
where
    F: FieldCore + CanonicalField,
    Cfg: CommitmentConfig<Field = F>,
{
    let mut stats = LayoutChainStats::default();
    let batched_root_lp =
        root_batched_layout::<Cfg, D>(max_num_vars, root_lp, max_num_batched_polys)?;
    stats.include(&batched_root_lp);
    if max_num_batched_polys <= 1 {
        let root_inputs = HachiScheduleInputs {
            max_num_vars,
            level: 0,
            current_w_len: root_current_w_len::<D>(root_lp),
        };
        let singleton_lp = Cfg::root_level_params_for_layout_with_log_basis(root_inputs, root_lp)?;
        stats.include(&singleton_lp);
    }

    let can_use_planned_root =
        Cfg::commitment_layout(max_num_vars).is_ok_and(|planned_root| planned_root == *root_lp);
    if can_use_planned_root && max_num_batched_polys == 1 {
        let schedule_key = HachiScheduleLookupKey::singleton(max_num_vars, max_num_vars, 1);
        if let Some(plan) = Cfg::schedule_plan(schedule_key)? {
            for level in plan.fold_levels().skip(1) {
                let level_lp = level.lp.clone();
                stats.include(&level_lp);
            }
            return Ok(stats);
        }
    }

    let root_plan = hachi_root_runtime_plan_from_root_layout::<Cfg, D>(
        HachiScheduleLookupKey::with_batch(
            max_num_vars,
            max_num_vars,
            max_num_batched_polys,
            HachiRootBatchSummary::new(
                max_num_batched_polys,
                max_num_batched_polys,
                max_num_batched_polys,
            )?,
        ),
        root_lp,
    )?;
    let mut prev_w_len = root_plan
        .inputs
        .current_w_len
        .saturating_mul(root_plan.batch.num_claims);
    let mut level = 1usize;
    let mut current_w_len = root_plan.next_w_len();
    let mut current_lp = root_plan.next_level_params.clone();
    stats.include(&current_lp);

    loop {
        if should_stop_folding(current_w_len, prev_w_len) {
            break;
        }

        let next_w_len = w_ring_element_count::<F>(&current_lp) * current_lp.ring_dimension;
        let next_level = level + 1;
        let next_lp_partial = Cfg::level_params(HachiScheduleInputs {
            max_num_vars,
            level: next_level,
            current_w_len: next_w_len,
        });
        let next_lp =
            super::hachi_recursive_level_layout_from_params::<Cfg>(&next_lp_partial, next_w_len)?;
        stats.include(&next_lp);

        prev_w_len = current_w_len;
        current_w_len = next_w_len;
        current_lp = next_lp;
        level = next_level;
    }

    Ok(stats)
}

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
fn validate_cached_setup_dimensions<F, const D: usize, Cfg: CommitmentConfig<Field = F>>(
    expanded: &HachiExpandedSetup<F>,
    max_num_vars: usize,
    max_num_batched_polys: usize,
    root_lp: &LevelParams,
) -> Result<(), HachiError>
where
    F: FieldCore + CanonicalField,
{
    let envelope = Cfg::envelope(max_num_vars);
    let chain_stats = scan_layout_chain::<F, D, Cfg>(max_num_vars, root_lp, max_num_batched_polys)?;
    let a_cols = chain_stats.max_inner_width;
    let b_cols = chain_stats.max_outer_width;
    let d_cols = chain_stats.max_d_matrix_width;

    let max_stride = a_cols.max(b_cols).max(d_cols);
    let max_rows = chain_stats
        .max_n_a
        .max(chain_stats.max_n_b)
        .max(chain_stats.max_n_d)
        .max(envelope.max_n_a)
        .max(envelope.max_n_b)
        .max(envelope.max_n_d);
    let required_total = max_rows * max_stride;

    let actual_total = expanded.shared_matrix.total_ring_elements_at::<D>();
    if actual_total < required_total {
        return Err(HachiError::InvalidSetup(format!(
            "cached setup matrix too small: have {actual_total} ring elements, need at least {required_total}"
        )));
    }

    Ok(())
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
    let expanded = Arc::new(expanded);
    let total = expanded.shared_matrix.total_ring_elements_at::<D>();
    let ntt_shared = build_ntt_slot(expanded.shared_matrix.ring_view::<D>(1, total))?;
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
    Cfg: CommitmentConfig<Field = F>,
{
    type ProverSetup = HachiProverSetup<F, D>;
    type VerifierSetup = HachiVerifierSetup<F>;
    type Commitment = RingCommitment<F, D>;

    #[tracing::instrument(skip_all, name = "RingCommitmentScheme::setup")]
    fn setup(
        max_num_vars: usize,
        max_num_batched_polys: usize,
    ) -> Result<(Self::ProverSetup, Self::VerifierSetup), HachiError> {
        let root_lp = validate_and_derive_layout::<F, Cfg, D>(max_num_vars)?;
        let envelope = Cfg::envelope(max_num_vars);
        ensure_layout_supported_num_vars::<D>(max_num_vars, &root_lp)?;

        #[cfg(feature = "disk-persistence")]
        {
            match load_expanded_setup::<F, Cfg>(max_num_vars, max_num_batched_polys) {
                Ok(expanded) => {
                    if let Err(e) = validate_cached_setup_dimensions::<F, D, Cfg>(
                        &expanded,
                        max_num_vars,
                        max_num_batched_polys,
                        &root_lp,
                    ) {
                        if let Some(storage_path) =
                            get_storage_path::<Cfg>(max_num_vars, max_num_batched_polys)
                        {
                            let _ = fs::remove_file(&storage_path);
                            tracing::warn!(
                                "Rejected cached setup from {}: {e}; regenerating",
                                storage_path.display()
                            );
                        } else {
                            tracing::warn!("Rejected cached setup: {e}; regenerating");
                        }
                    } else {
                        tracing::info!("Loaded setup from disk, rebuilding NTT caches");
                        return setup_from_expanded(expanded);
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

        let chain_stats =
            scan_layout_chain::<F, D, Cfg>(max_num_vars, &root_lp, max_num_batched_polys)?;
        let a_cols = chain_stats.max_inner_width;
        let b_cols = chain_stats.max_outer_width;
        let d_cols = chain_stats.max_d_matrix_width;

        let max_stride = a_cols.max(b_cols).max(d_cols);
        let max_rows = chain_stats
            .max_n_a
            .max(chain_stats.max_n_b)
            .max(chain_stats.max_n_d)
            .max(envelope.max_n_a)
            .max(envelope.max_n_b)
            .max(envelope.max_n_d);
        let max_total = max_rows * max_stride;

        let public_matrix_seed = sample_public_matrix_seed();
        let shared_flat = derive_public_matrix_flat::<F, D>(max_total, &public_matrix_seed);
        let ntt_shared = build_ntt_slot(shared_flat.ring_view::<D>(1, max_total))?;

        let expanded = Arc::new(HachiExpandedSetup {
            seed: HachiSetupSeed {
                max_num_vars,
                max_num_batched_polys,
                max_inner_width: chain_stats.max_inner_width,
                max_outer_width: chain_stats.max_outer_width,
                max_d_matrix_width: chain_stats.max_d_matrix_width,
                max_total_ring_elements: max_total,
                public_matrix_seed,
            },
            shared_matrix: shared_flat,
        });

        #[cfg(feature = "disk-persistence")]
        save_expanded_setup::<F, Cfg>(&expanded, max_num_vars, max_num_batched_polys);

        let prover_setup = HachiProverSetup {
            expanded: Arc::clone(&expanded),
            ntt_shared,
        };
        let verifier_setup = HachiVerifierSetup { expanded };
        Ok((prover_setup, verifier_setup))
    }

    fn layout(setup: &Self::ProverSetup) -> Result<LevelParams, HachiError> {
        hachi_batched_root_layout::<Cfg, D>(
            setup.expanded.seed.max_num_vars,
            setup.expanded.seed.max_num_batched_polys,
        )
    }

    #[tracing::instrument(skip_all, name = "RingCommitmentScheme::commit_ring_blocks")]
    fn commit_ring_blocks(
        f_blocks: &[Vec<CyclotomicRing<F, D>>],
        setup: &Self::ProverSetup,
    ) -> Result<CommitWitness<Self::Commitment, F, D>, HachiError> {
        let root_lp = <Self as RingCommitmentScheme<F, D, Cfg>>::layout(setup)?;
        ensure_layout_supported_num_vars::<D>(setup.expanded.seed.max_num_vars, &root_lp)?;
        ensure_block_layout(f_blocks, &root_lp)?;

        let depth_commit = root_lp.num_digits_commit;
        let depth_open = root_lp.num_digits_open;
        let log_basis = root_lp.log_basis;
        let block_slices: Vec<&[CyclotomicRing<F, D>]> =
            f_blocks.iter().map(|b| b.as_slice()).collect();
        let t_hat = if root_lp.a_key.row_len == 1 {
            let t_single = mat_vec_mul_ntt_i8_dense_single_row(
                &setup.ntt_shared,
                setup.expanded.seed.max_stride(),
                &block_slices,
                depth_commit,
                log_basis,
            );
            let mut t_hat = FlatDigitBlocks::zeroed(vec![depth_open; t_single.len()])?;
            let dst_blocks = t_hat.split_blocks_mut();
            #[cfg(feature = "parallel")]
            cfg_into_iter!(dst_blocks)
                .zip(cfg_iter!(t_single))
                .for_each(|(dst, t_i)| {
                    decompose_rows_i8_into(std::slice::from_ref(t_i), dst, depth_open, log_basis)
                });
            #[cfg(not(feature = "parallel"))]
            dst_blocks
                .into_iter()
                .zip(t_single.iter())
                .for_each(|(dst, t_i)| {
                    decompose_rows_i8_into(std::slice::from_ref(t_i), dst, depth_open, log_basis)
                });
            t_hat
        } else {
            let t_all = mat_vec_mul_ntt_i8_dense(
                &setup.ntt_shared,
                root_lp.a_key.row_len,
                setup.expanded.seed.max_stride(),
                &block_slices,
                depth_commit,
                log_basis,
            );
            let block_sizes: Vec<usize> = t_all.iter().map(|t_i| t_i.len() * depth_open).collect();
            let mut t_hat = FlatDigitBlocks::zeroed(block_sizes)?;
            let dst_blocks = t_hat.split_blocks_mut();
            #[cfg(feature = "parallel")]
            cfg_into_iter!(dst_blocks)
                .zip(cfg_iter!(t_all))
                .for_each(|(dst, t_i)| decompose_rows_i8_into(t_i, dst, depth_open, log_basis));
            #[cfg(not(feature = "parallel"))]
            dst_blocks
                .into_iter()
                .zip(t_all.iter())
                .for_each(|(dst, t_i)| decompose_rows_i8_into(t_i, dst, depth_open, log_basis));
            t_hat
        };

        let u: Vec<CyclotomicRing<F, D>> = mat_vec_mul_ntt_single_i8(
            &setup.ntt_shared,
            root_lp.b_key.row_len,
            setup.expanded.seed.max_stride(),
            t_hat.flat_digits(),
        );
        Ok(CommitWitness::new(RingCommitment { u }, t_hat))
    }

    #[tracing::instrument(skip_all, name = "RingCommitmentScheme::commit_onehot")]
    fn commit_onehot<I: OneHotIndex>(
        onehot_k: usize,
        indices: &[Option<I>],
        setup: &Self::ProverSetup,
    ) -> Result<CommitWitness<Self::Commitment, F, D>, HachiError> {
        let root_lp = <Self as RingCommitmentScheme<F, D, Cfg>>::layout(setup)?;
        ensure_layout_supported_num_vars::<D>(setup.expanded.seed.max_num_vars, &root_lp)?;

        let sparse_blocks =
            map_onehot_to_sparse_blocks(onehot_k, indices, root_lp.r_vars, root_lp.m_vars, D)?;

        let depth_commit = root_lp.num_digits_commit;
        let depth_open = root_lp.num_digits_open;
        let log_basis = root_lp.log_basis;
        let zero_block_len = root_lp.a_key.row_len.checked_mul(depth_open).unwrap();
        let a_view = setup
            .expanded
            .shared_matrix
            .ring_view::<D>(root_lp.a_key.row_len, setup.expanded.seed.max_stride());
        let block_len = root_lp.block_len;

        let block_sizes = vec![zero_block_len; sparse_blocks.len()];
        let mut t_hat = FlatDigitBlocks::zeroed(block_sizes)?;
        let dst_blocks = t_hat.split_blocks_mut();
        #[cfg(feature = "parallel")]
        cfg_into_iter!(dst_blocks)
            .zip(cfg_iter!(sparse_blocks))
            .for_each(|(dst, block_entries)| {
                if !block_entries.is_empty() {
                    let mut t_i =
                        inner_ajtai_onehot_wide(&a_view, block_entries, block_len, depth_commit);
                    t_i.truncate(root_lp.a_key.row_len);
                    decompose_rows_i8_into(&t_i, dst, depth_open, log_basis);
                }
            });
        #[cfg(not(feature = "parallel"))]
        dst_blocks
            .into_iter()
            .zip(sparse_blocks.iter())
            .for_each(|(dst, block_entries)| {
                if !block_entries.is_empty() {
                    let mut t_i =
                        inner_ajtai_onehot_wide(&a_view, block_entries, block_len, depth_commit);
                    t_i.truncate(root_lp.a_key.row_len);
                    decompose_rows_i8_into(&t_i, dst, depth_open, log_basis);
                }
            });

        let u: Vec<CyclotomicRing<F, D>> = mat_vec_mul_ntt_single_i8(
            &setup.ntt_shared,
            root_lp.b_key.row_len,
            setup.expanded.seed.max_stride(),
            t_hat.flat_digits(),
        );
        Ok(CommitWitness::new(RingCommitment { u }, t_hat))
    }
}

impl HachiCommitmentCore {
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
        lp: &LevelParams,
    ) -> Result<(HachiProverSetup<F, D>, HachiVerifierSetup<F>), HachiError>
    where
        F: FieldCore + CanonicalField + FieldSampling,
        Cfg: CommitmentConfig<Field = F>,
    {
        let alpha = D.trailing_zeros() as usize;
        let max_num_vars = lp.m_vars + lp.r_vars + alpha;
        let public_matrix_seed = sample_public_matrix_seed();
        Self::setup_with_lp_and_seed::<F, D, Cfg>(lp, max_num_vars, public_matrix_seed)
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
        layouts: &[LevelParams],
    ) -> Result<(HachiProverSetup<F, D>, HachiVerifierSetup<F>), HachiError>
    where
        F: FieldCore + CanonicalField + FieldSampling,
        Cfg: CommitmentConfig<Field = F>,
    {
        let Some((first_lp, _)) = layouts.split_first() else {
            return Err(HachiError::InvalidSetup(
                "setup_with_layouts requires at least one layout".to_string(),
            ));
        };

        let alpha = D.trailing_zeros() as usize;
        let mut max_num_vars = 0usize;
        let mut max_inner_width = 0usize;
        let mut max_outer_width = 0usize;
        let mut max_d_matrix_width = 0usize;
        let mut max_r_vars = 0usize;
        let mut max_num_digits_open = 0usize;
        let mut max_num_digits_fold = 0usize;
        let mut max_log_basis = first_lp.log_basis;
        let mut max_n_a = 0usize;
        let mut max_n_b = 0usize;
        let mut max_n_d = 0usize;

        for lp in layouts {
            let layout_num_vars = lp.m_vars + lp.r_vars + alpha;
            let chain_stats = scan_layout_chain::<F, D, Cfg>(layout_num_vars, lp, 1)?;
            tracing::debug!(?lp, ?chain_stats, "setup layout chain");
            max_num_vars = max_num_vars.max(layout_num_vars);
            max_inner_width = max_inner_width.max(chain_stats.max_inner_width);
            max_outer_width = max_outer_width.max(chain_stats.max_outer_width);
            max_d_matrix_width = max_d_matrix_width.max(chain_stats.max_d_matrix_width);
            max_r_vars = max_r_vars.max(chain_stats.max_r_vars);
            max_num_digits_open = max_num_digits_open.max(chain_stats.max_num_digits_open);
            max_num_digits_fold = max_num_digits_fold.max(chain_stats.max_num_digits_fold);
            max_log_basis = max_log_basis.max(chain_stats.max_log_basis);
            max_n_a = max_n_a.max(chain_stats.max_n_a);
            max_n_b = max_n_b.max(chain_stats.max_n_b);
            max_n_d = max_n_d.max(chain_stats.max_n_d);
        }

        tracing::debug!(
            max_inner_width,
            max_outer_width,
            max_d_matrix_width,
            max_r_vars,
            max_num_vars,
            "setup envelope"
        );
        let public_matrix_seed = sample_public_matrix_seed();
        Self::setup_with_matrix_widths_and_seed::<F, D, Cfg>(
            max_num_vars,
            1,
            public_matrix_seed,
            max_inner_width,
            max_outer_width,
            max_d_matrix_width,
            max_n_a,
            max_n_b,
            max_n_d,
        )
    }

    fn setup_with_lp_and_seed<F, const D: usize, Cfg>(
        lp: &LevelParams,
        max_num_vars: usize,
        public_matrix_seed: PublicMatrixSeed,
    ) -> Result<(HachiProverSetup<F, D>, HachiVerifierSetup<F>), HachiError>
    where
        F: FieldCore + CanonicalField + FieldSampling,
        Cfg: CommitmentConfig<Field = F>,
    {
        let alpha = D.trailing_zeros() as usize;
        let layout_num_vars = lp.m_vars + lp.r_vars + alpha;
        let chain_stats = scan_layout_chain::<F, D, Cfg>(layout_num_vars, lp, 1)?;
        let a_cols = chain_stats.max_inner_width;
        let b_cols = chain_stats.max_outer_width;
        let d_cols = chain_stats.max_d_matrix_width;

        Self::setup_with_matrix_widths_and_seed::<F, D, Cfg>(
            max_num_vars,
            1,
            public_matrix_seed,
            a_cols,
            b_cols,
            d_cols,
            chain_stats.max_n_a,
            chain_stats.max_n_b,
            chain_stats.max_n_d,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn setup_with_matrix_widths_and_seed<F, const D: usize, Cfg>(
        max_num_vars: usize,
        max_num_batched_polys: usize,
        public_matrix_seed: PublicMatrixSeed,
        a_cols: usize,
        b_cols: usize,
        d_cols: usize,
        max_n_a: usize,
        max_n_b: usize,
        max_n_d: usize,
    ) -> Result<(HachiProverSetup<F, D>, HachiVerifierSetup<F>), HachiError>
    where
        F: FieldCore + CanonicalField + FieldSampling,
        Cfg: CommitmentConfig<Field = F>,
    {
        let envelope = Cfg::envelope(max_num_vars);
        let max_stride = a_cols.max(b_cols).max(d_cols);
        let max_rows = max_n_a
            .max(max_n_b)
            .max(max_n_d)
            .max(envelope.max_n_a)
            .max(envelope.max_n_b)
            .max(envelope.max_n_d);
        let max_total = max_rows * max_stride;
        {
            let ring_bytes = std::mem::size_of::<CyclotomicRing<F, D>>();
            let shared_mb = (max_total * ring_bytes) as f64 / (1024.0_f64 * 1024.0_f64);
            tracing::debug!(
                a_cols,
                b_cols,
                d_cols,
                max_stride,
                max_total,
                ring_bytes,
                shared_mb,
                "setup shared matrix size"
            );
        }
        let shared_flat = derive_public_matrix_flat::<F, D>(max_total, &public_matrix_seed);
        let ntt_shared = build_ntt_slot(shared_flat.ring_view::<D>(1, max_total))?;

        let expanded = Arc::new(HachiExpandedSetup {
            seed: HachiSetupSeed {
                max_num_vars,
                max_num_batched_polys,
                max_inner_width: a_cols,
                max_outer_width: b_cols,
                max_d_matrix_width: d_cols,
                max_total_ring_elements: max_total,
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
    use crate::protocol::commitment::{hachi_recursive_level_layout_from_params, presets::fp128};
    use crate::protocol::ring_switch::w_ring_element_count_with_num_claims_and_points;
    use crate::test_utils::{TinyConfig, F as TestF};

    #[test]
    fn expanded_setup_roundtrips_and_derives_same_verifier() {
        const TEST_D: usize = 64;
        let (prover_setup, verifier_setup) =
            <HachiCommitmentCore as RingCommitmentScheme<TestF, TEST_D, TinyConfig>>::setup(16, 3)
                .unwrap();

        let mut bytes = Vec::new();
        prover_setup
            .expanded
            .serialize_compressed(&mut bytes)
            .unwrap();
        let decoded = HachiExpandedSetup::<TestF>::deserialize_compressed(&bytes[..], &()).unwrap();

        assert_eq!(decoded, prover_setup.expanded.as_ref().clone());
        assert_eq!(decoded.seed.max_num_batched_polys, 3);

        let derived_verifier = HachiVerifierSetup {
            expanded: Arc::new(decoded.clone()),
        };
        assert_eq!(derived_verifier, verifier_setup);
    }

    #[test]
    fn setup_scales_root_batch_capacity() {
        const TEST_D: usize = 64;
        const MAX_NUM_VARS: usize = 16;
        const MAX_BATCH: usize = 3;

        let root_lp =
            validate_and_derive_layout::<TestF, TinyConfig, TEST_D>(MAX_NUM_VARS).unwrap();
        let single_stats =
            scan_layout_chain::<TestF, TEST_D, TinyConfig>(MAX_NUM_VARS, &root_lp, 1).unwrap();
        let batched_stats =
            scan_layout_chain::<TestF, TEST_D, TinyConfig>(MAX_NUM_VARS, &root_lp, MAX_BATCH)
                .unwrap();
        let scaled_root =
            root_batched_layout::<TinyConfig, TEST_D>(MAX_NUM_VARS, &root_lp, MAX_BATCH).unwrap();
        let worst_case_multipoint_w_len = w_ring_element_count_with_num_claims_and_points::<TestF>(
            &scaled_root,
            MAX_BATCH,
            MAX_BATCH,
        ) * TEST_D;
        let multipoint_level1_params = TinyConfig::level_params(HachiScheduleInputs {
            max_num_vars: MAX_NUM_VARS,
            level: 1,
            current_w_len: worst_case_multipoint_w_len,
        });
        let multipoint_level1_lp = hachi_recursive_level_layout_from_params::<TinyConfig>(
            &multipoint_level1_params,
            worst_case_multipoint_w_len,
        )
        .unwrap();

        let (setup, _) =
            <HachiCommitmentCore as RingCommitmentScheme<TestF, TEST_D, TinyConfig>>::setup(
                MAX_NUM_VARS,
                MAX_BATCH,
            )
            .unwrap();
        let seed = &setup.expanded.seed;

        assert_eq!(setup.max_num_batched_polys(), MAX_BATCH);
        assert!(batched_stats.max_outer_width >= single_stats.max_outer_width);
        assert!(batched_stats.max_d_matrix_width >= single_stats.max_d_matrix_width);
        assert!(batched_stats.max_outer_width >= scaled_root.outer_width());
        assert!(batched_stats.max_d_matrix_width >= scaled_root.d_matrix_width());
        assert!(seed.max_inner_width >= scaled_root.inner_width());
        assert!(seed.max_outer_width >= scaled_root.outer_width());
        assert!(seed.max_d_matrix_width >= scaled_root.d_matrix_width());
        assert!(batched_stats.max_inner_width >= multipoint_level1_lp.inner_width());
        assert!(batched_stats.max_outer_width >= multipoint_level1_lp.outer_width());
        assert!(batched_stats.max_d_matrix_width >= multipoint_level1_lp.d_matrix_width());
        assert!(seed.max_inner_width >= multipoint_level1_lp.inner_width());
        assert!(seed.max_outer_width >= multipoint_level1_lp.outer_width());
        assert!(seed.max_d_matrix_width >= multipoint_level1_lp.d_matrix_width());
        let envelope = TinyConfig::envelope(MAX_NUM_VARS);
        let total_elements = setup
            .expanded
            .shared_matrix
            .total_ring_elements_at::<TEST_D>();
        assert!(total_elements >= envelope.max_n_a * batched_stats.max_inner_width);
        assert!(total_elements >= envelope.max_n_b * batched_stats.max_outer_width);
        assert!(total_elements >= envelope.max_n_d * batched_stats.max_d_matrix_width);
    }

    #[test]
    fn onehot_batched_helper_matches_setup_root_layout() {
        type Cfg = fp128::D64OneHot;
        const TEST_D: usize = Cfg::D;
        const NV: usize = 15;
        const BATCH: usize = 2;

        let root_lp = Cfg::commitment_layout(NV).unwrap();
        let setup_root = root_batched_layout::<Cfg, TEST_D>(NV, &root_lp, BATCH).unwrap();
        let helper_root = hachi_batched_root_layout::<Cfg, TEST_D>(NV, BATCH).unwrap();
        let (setup, _) =
            <HachiCommitmentCore as RingCommitmentScheme<fp128::Field, TEST_D, Cfg>>::setup(
                NV, BATCH,
            )
            .unwrap();
        let runtime_lp =
            <HachiCommitmentCore as RingCommitmentScheme<fp128::Field, TEST_D, Cfg>>::layout(
                &setup,
            )
            .unwrap();

        assert_eq!(helper_root.m_vars, setup_root.m_vars);
        assert_eq!(helper_root.r_vars, setup_root.r_vars);
        assert_eq!(runtime_lp, helper_root);
        assert_eq!(helper_root.outer_width() * BATCH, setup_root.outer_width());
        assert_eq!(
            helper_root.d_matrix_width() * BATCH,
            setup_root.d_matrix_width()
        );
        assert!(
            helper_root.num_digits_fold <= setup_root.num_digits_fold,
            "per-poly num_digits_fold ({}) must not exceed batched value ({})",
            helper_root.num_digits_fold,
            setup_root.num_digits_fold,
        );
        assert!(setup.expanded.seed.max_outer_width >= setup_root.outer_width());
        assert!(setup.expanded.seed.max_d_matrix_width >= setup_root.d_matrix_width());
    }

    #[test]
    fn setup_with_layouts_uses_exact_width_envelope() {
        use crate::protocol::commitment::{compute_num_digits_fold, num_digits_for_bound};
        const TEST_D: usize = 64;
        const ALPHA: usize = 6; // TEST_D.trailing_zeros()

        let decomp = TinyConfig::decomposition();
        let depth_commit = num_digits_for_bound(decomp.log_commit_bound, decomp.log_basis);
        let open_bound = decomp.log_open_bound.unwrap_or(decomp.log_commit_bound);
        let depth_open = num_digits_for_bound(open_bound, decomp.log_basis);

        let nv_a = 4 + 2 + ALPHA;
        let nv_b = 1 + 6 + ALPHA;
        let params_a = TinyConfig::level_params(HachiScheduleInputs {
            max_num_vars: nv_a,
            level: 0,
            current_w_len: 1usize << nv_a,
        });
        let params_b = TinyConfig::level_params(HachiScheduleInputs {
            max_num_vars: nv_b,
            level: 0,
            current_w_len: 1usize << nv_b,
        });
        let depth_fold_a =
            compute_num_digits_fold(2, params_a.challenge_l1_mass(), decomp.log_basis);
        let depth_fold_b =
            compute_num_digits_fold(6, params_b.challenge_l1_mass(), decomp.log_basis);
        let lp_a = params_a
            .with_decomp(4, 2, depth_commit, depth_open, depth_fold_a, 0)
            .unwrap();
        let lp_b = params_b
            .with_decomp(1, 6, depth_commit, depth_open, depth_fold_b, 0)
            .unwrap();
        let w_len_a = w_ring_element_count::<TestF>(&lp_a) * TEST_D;
        let w_len_b = w_ring_element_count::<TestF>(&lp_b) * TEST_D;
        let w_lp_a =
            hachi_recursive_level_layout_from_params::<TinyConfig>(&params_a, w_len_a).unwrap();
        let w_lp_b =
            hachi_recursive_level_layout_from_params::<TinyConfig>(&params_b, w_len_b).unwrap();

        let expected_inner = [
            lp_a.inner_width(),
            lp_b.inner_width(),
            w_lp_a.inner_width(),
            w_lp_b.inner_width(),
        ]
        .into_iter()
        .max()
        .unwrap();
        let expected_outer = [
            lp_a.outer_width(),
            lp_b.outer_width(),
            w_lp_a.outer_width(),
            w_lp_b.outer_width(),
        ]
        .into_iter()
        .max()
        .unwrap();
        let expected_d = [
            lp_a.d_matrix_width(),
            lp_b.d_matrix_width(),
            w_lp_a.d_matrix_width(),
            w_lp_b.d_matrix_width(),
        ]
        .into_iter()
        .max()
        .unwrap();
        let expected_max_num_vars = nv_a.max(nv_b);

        let (setup, _) =
            HachiCommitmentCore::setup_with_layouts::<TestF, TEST_D, TinyConfig>(&[lp_a, lp_b])
                .unwrap();
        let seed = &setup.expanded.seed;

        assert_eq!(seed.max_num_vars, expected_max_num_vars);
        assert!(seed.max_inner_width >= expected_inner);
        assert!(seed.max_outer_width >= expected_outer);
        assert!(seed.max_d_matrix_width >= expected_d);
        let total_elements = setup
            .expanded
            .shared_matrix
            .total_ring_elements_at::<TEST_D>();
        let envelope = TinyConfig::envelope(expected_max_num_vars);
        assert!(total_elements >= envelope.max_n_a * expected_inner);
        assert!(total_elements >= envelope.max_n_b * expected_outer);
        assert!(total_elements >= envelope.max_n_d * expected_d);
    }

    #[test]
    fn setup_accepts_field_coupled_presets() {
        <HachiCommitmentCore as RingCommitmentScheme<fp128::Field, 128, fp128::D128Full>>::setup(
            12, 1,
        )
        .expect("legacy fp128 preset should accept the legacy field");

        <HachiCommitmentCore as RingCommitmentScheme<fp128::Field, 128, fp128::D128Full>>::setup(
            12, 1,
        )
        .expect("default fp128 fixed-D preset should accept the default field");

        <HachiCommitmentCore as RingCommitmentScheme<fp128::Field, 32, fp128::D32Full>>::setup(
            12, 1,
        )
        .expect("small-D fp128 preset should accept the default field");
    }

    #[cfg(feature = "disk-persistence")]
    mod disk_persistence {
        use super::*;
        use std::fs;
        use std::sync::{LazyLock, Mutex};

        static DISK_TEST_ENV_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

        fn cleanup_setup_file(max_num_vars: usize) {
            if let Some(path) = get_storage_path::<TinyConfig>(max_num_vars, 1) {
                let _ = fs::remove_file(path);
            }
        }

        fn with_test_cache_dir<T>(test_name: &str, f: impl FnOnce() -> T) -> T {
            let _guard = DISK_TEST_ENV_LOCK.lock().unwrap();
            let cache_root = std::env::temp_dir().join(format!("hachi-disk-tests-{test_name}"));
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
                const TEST_D: usize = 64;
                const MAX_VARS: usize = 100;

                cleanup_setup_file(MAX_VARS);

                let (prover_setup, _) = <HachiCommitmentCore as RingCommitmentScheme<
                    TestF,
                    TEST_D,
                    TinyConfig,
                >>::setup(MAX_VARS, 1)
                .unwrap();

                let loaded = load_expanded_setup::<TestF, TinyConfig>(MAX_VARS, 1).unwrap();
                assert_eq!(loaded, prover_setup.expanded.as_ref().clone());

                cleanup_setup_file(MAX_VARS);
            });
        }

        #[test]
        fn setup_uses_cache_on_second_call() {
            with_test_cache_dir("second-call", || {
                const TEST_D: usize = 64;
                const MAX_VARS: usize = 101;

                cleanup_setup_file(MAX_VARS);

                let (first, _) = <HachiCommitmentCore as RingCommitmentScheme<
                    TestF,
                    TEST_D,
                    TinyConfig,
                >>::setup(MAX_VARS, 1)
                .unwrap();

                let (second, _) = <HachiCommitmentCore as RingCommitmentScheme<
                    TestF,
                    TEST_D,
                    TinyConfig,
                >>::setup(MAX_VARS, 1)
                .unwrap();

                assert_eq!(first.expanded, second.expanded);

                cleanup_setup_file(MAX_VARS);
            });
        }

        #[test]
        fn ntt_caches_rebuilt_correctly_from_disk() {
            with_test_cache_dir("ntt-rebuild", || {
                use crate::algebra::CyclotomicRing;

                const TEST_D: usize = 64;
                const MAX_VARS: usize = 102;

                cleanup_setup_file(MAX_VARS);

                let (fresh_setup, _) = <HachiCommitmentCore as RingCommitmentScheme<
                    TestF,
                    TEST_D,
                    TinyConfig,
                >>::setup(MAX_VARS, 1)
                .unwrap();

                let loaded_expanded =
                    load_expanded_setup::<TestF, TinyConfig>(MAX_VARS, 1).unwrap();
                let (disk_setup, _) =
                    setup_from_expanded::<TestF, TEST_D>(loaded_expanded).unwrap();

                let lp = TinyConfig::commitment_layout(MAX_VARS).unwrap();
                let num_coeffs = lp.num_blocks * lp.block_len;
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
            });
        }
    }
}
