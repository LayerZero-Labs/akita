//! Setup-time preparation of the shared matrix for the setup-claim carry.
//!
//! The shared matrix is the flat matrix (A/B/D backing data) viewed as a
//! padded `row × col × k` tensor. This module defines the canonical tensor
//! layout and materializes the shared matrix as a flat field-evals vector so
//! the L=0 setup-claim sumcheck producer and the L=1 carry-closure consumer
//! share the same oracle data.

use crate::algebra::fields::wide::HasWide;
use crate::algebra::fields::HasUnreducedOps;
use crate::error::HachiError;
use crate::primitives::serialization::{
    Compress, HachiDeserialize, HachiSerialize, SerializationError, Valid, Validate,
};
use crate::protocol::commitment::utils::crt_ntt::build_ntt_slot;
use crate::protocol::commitment::utils::linear::mat_vec_mul_ntt_single_i8;
use crate::protocol::commitment::{
    exact_schedule_plan_for_lookup_key, hachi_recursive_level_layout_from_params,
    hachi_root_runtime_plan, profile::CommitmentFieldProfileSchedule, CommitmentConfig,
    CommitmentFieldProfile, CommitmentPreset, GeneratedAdaptivePolicy, StaticBoundedPolicy,
};
use crate::protocol::hachi_poly_ops::{DensePoly, HachiPolyOps};
use crate::protocol::params::LevelParams;
use crate::protocol::proof::{FlatRingVec, HachiCommitmentHint};
use crate::protocol::recursive_runtime::RecursiveCommitmentHintCache;
use crate::protocol::setup::{HachiExpandedSetup, HachiProverSetup};
use crate::{CanonicalField, FieldCore, FieldSampling};
use std::sync::Arc;

/// Canonical tensor layout for the shared matrix commitment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SharedMatrixTensorLayout {
    /// Actual row count covered by the setup envelope.
    pub max_rows: usize,
    /// Actual shared row stride from the main setup.
    pub stride: usize,
    /// Row count padded to a power of two for the MLE.
    pub padded_rows: usize,
    /// Stride padded to a power of two for the MLE.
    pub padded_stride: usize,
    /// Number of row variables in the padded tensor MLE.
    pub row_vars: usize,
    /// Number of column variables in the padded tensor MLE.
    pub col_vars: usize,
    /// Number of ring-coordinate variables (`log2(D)`).
    pub ring_vars: usize,
    /// Total number of MLE variables in the shared matrix tensor.
    pub num_vars: usize,
}

impl Valid for SharedMatrixTensorLayout {
    fn check(&self) -> Result<(), SerializationError> {
        if self.stride == 0 || self.padded_rows == 0 || self.padded_stride == 0 {
            return Err(SerializationError::InvalidData(
                "shared matrix tensor layout dimensions must be positive".to_string(),
            ));
        }
        if self.field_len() != (1usize << self.num_vars) {
            return Err(SerializationError::InvalidData(
                "shared matrix tensor layout must describe a power-of-two field table".to_string(),
            ));
        }
        Ok(())
    }
}

impl HachiSerialize for SharedMatrixTensorLayout {
    fn serialize_with_mode<W: std::io::Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.max_rows.serialize_with_mode(&mut writer, compress)?;
        self.stride.serialize_with_mode(&mut writer, compress)?;
        self.padded_rows
            .serialize_with_mode(&mut writer, compress)?;
        self.padded_stride
            .serialize_with_mode(&mut writer, compress)?;
        self.row_vars.serialize_with_mode(&mut writer, compress)?;
        self.col_vars.serialize_with_mode(&mut writer, compress)?;
        self.ring_vars.serialize_with_mode(&mut writer, compress)?;
        self.num_vars.serialize_with_mode(&mut writer, compress)?;
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.max_rows.serialized_size(compress)
            + self.stride.serialized_size(compress)
            + self.padded_rows.serialized_size(compress)
            + self.padded_stride.serialized_size(compress)
            + self.row_vars.serialized_size(compress)
            + self.col_vars.serialized_size(compress)
            + self.ring_vars.serialized_size(compress)
            + self.num_vars.serialized_size(compress)
    }
}

impl HachiDeserialize for SharedMatrixTensorLayout {
    type Context = ();

    fn deserialize_with_mode<R: std::io::Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &(),
    ) -> Result<Self, SerializationError> {
        let out = Self {
            max_rows: usize::deserialize_with_mode(&mut reader, compress, validate, &())?,
            stride: usize::deserialize_with_mode(&mut reader, compress, validate, &())?,
            padded_rows: usize::deserialize_with_mode(&mut reader, compress, validate, &())?,
            padded_stride: usize::deserialize_with_mode(&mut reader, compress, validate, &())?,
            row_vars: usize::deserialize_with_mode(&mut reader, compress, validate, &())?,
            col_vars: usize::deserialize_with_mode(&mut reader, compress, validate, &())?,
            ring_vars: usize::deserialize_with_mode(&mut reader, compress, validate, &())?,
            num_vars: usize::deserialize_with_mode(&mut reader, compress, validate, &())?,
        };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

impl SharedMatrixTensorLayout {
    /// Build the canonical tensor layout from the public expanded setup.
    ///
    /// `max_rows` is derived from the backing `FlatMatrix` rather than from
    /// `Cfg::envelope`. `CommitmentConfig::envelope` may overshoot the actual
    /// row count allocated by `Cfg::max_setup_matrix_size` (which is what
    /// `HachiProverSetup::new` uses to size the backing), so consulting the
    /// envelope here would cause `FlatMatrix::ring_view` to request more rows
    /// than the shared matrix actually holds.
    pub(crate) fn from_expanded<F: FieldCore, const D: usize>(
        expanded: &HachiExpandedSetup<F>,
    ) -> Self {
        let stride = expanded.seed.max_stride;
        let total_ring_elements_at_d = expanded.shared_matrix.total_ring_elements_at::<D>();
        assert!(
            stride > 0,
            "HachiSetupSeed::max_stride must be positive for tensor layout",
        );
        assert!(
            total_ring_elements_at_d.is_multiple_of(stride),
            "backing shared matrix size ({total_ring_elements_at_d}) must be a multiple of stride ({stride})",
        );
        let max_rows = total_ring_elements_at_d / stride;
        let padded_rows = max_rows.next_power_of_two();
        let padded_stride = stride.next_power_of_two();
        let row_vars = padded_rows.trailing_zeros() as usize;
        let col_vars = padded_stride.trailing_zeros() as usize;
        let ring_vars = D.trailing_zeros() as usize;
        Self {
            max_rows,
            stride,
            padded_rows,
            padded_stride,
            row_vars,
            col_vars,
            ring_vars,
            num_vars: row_vars + col_vars + ring_vars,
        }
    }

    #[inline]
    pub(crate) fn field_len(&self) -> usize {
        self.padded_rows * self.padded_stride * (1usize << self.ring_vars)
    }

    #[inline]
    pub(crate) fn flat_index(&self, row: usize, col: usize, k: usize) -> usize {
        (row * self.padded_stride + col) * (1usize << self.ring_vars) + k
    }

    /// Split a setup-sumcheck challenge point into `(row, col, k)` slices.
    #[allow(clippy::type_complexity)]
    pub(crate) fn split_point<'a, F: FieldCore>(
        &self,
        point: &'a [F],
    ) -> Result<(&'a [F], &'a [F], &'a [F]), HachiError> {
        if point.len() != self.num_vars {
            return Err(HachiError::InvalidPointDimension {
                expected: self.num_vars,
                actual: point.len(),
            });
        }
        let (r_k, rest) = point.split_at(self.ring_vars);
        let (r_col, r_row) = rest.split_at(self.col_vars);
        Ok((r_row, r_col, r_k))
    }
}

/// Precomputed verifier-side data for the setup-claim carry.
///
/// Built once during `setup_verifier` so that the L=1 carry-closure consumer
/// has the same flat field-evals view of the outer-D shared matrix as the
/// L=0 producer, without re-walking the `FlatMatrix` at verify time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SharedMatrixVerifierCache<F: FieldCore> {
    pub(crate) tensor_layout: SharedMatrixTensorLayout,
    /// Flat field-evals vector of the outer-D shared-matrix tensor, shaped as
    /// `padded_rows × padded_stride × D` (little-endian-in-k ordering per
    /// [`SharedMatrixTensorLayout::flat_index`]). Shared with the setup-claim
    /// carry consumer at downstream recursion levels.
    pub(crate) sm_evals_flat: Arc<Vec<F>>,
}

/// Public setup artifact for the fixed shared-matrix commitment used by the
/// carry-opening path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SharedMatrixCommitmentArtifact<F: FieldCore> {
    /// Canonical outer-D tensor layout for the carried setup claim.
    pub tensor_layout: SharedMatrixTensorLayout,
    /// Blessed recursive level layout that must consume the pending setup
    /// opening. Proof generation and verification fail fast if runtime drifts
    /// from this layout.
    pub opening_lp: LevelParams,
    /// Commitment to the zero-extended shared-matrix tensor polynomial, packed
    /// under `opening_lp.ring_dimension`.
    pub commitment: FlatRingVec<F>,
}

impl<F: FieldCore + Valid> Valid for SharedMatrixCommitmentArtifact<F> {
    fn check(&self) -> Result<(), SerializationError> {
        self.tensor_layout.check()?;
        self.opening_lp.check()?;
        self.commitment.check()?;
        if self.opening_lp.ring_dimension == 0
            || !self
                .commitment
                .coeff_len()
                .is_multiple_of(self.opening_lp.ring_dimension)
        {
            return Err(SerializationError::InvalidData(
                "shared matrix commitment coeff length must be a multiple of opening ring dimension"
                    .to_string(),
            ));
        }
        Ok(())
    }
}

impl<F: FieldCore> HachiSerialize for SharedMatrixCommitmentArtifact<F> {
    fn serialize_with_mode<W: std::io::Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.tensor_layout
            .serialize_with_mode(&mut writer, compress)?;
        self.opening_lp.serialize_with_mode(&mut writer, compress)?;
        self.commitment
            .coeff_len()
            .serialize_with_mode(&mut writer, compress)?;
        self.commitment.serialize_with_mode(&mut writer, compress)?;
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.tensor_layout.serialized_size(compress)
            + self.opening_lp.serialized_size(compress)
            + self.commitment.coeff_len().serialized_size(compress)
            + self.commitment.serialized_size(compress)
    }
}

impl<F: FieldCore + Valid> HachiDeserialize for SharedMatrixCommitmentArtifact<F> {
    type Context = ();

    fn deserialize_with_mode<R: std::io::Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &(),
    ) -> Result<Self, SerializationError> {
        let out = Self {
            tensor_layout: SharedMatrixTensorLayout::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                &(),
            )?,
            opening_lp: LevelParams::deserialize_with_mode(&mut reader, compress, validate, &())?,
            commitment: {
                let coeff_len = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
                FlatRingVec::deserialize_with_mode(&mut reader, compress, validate, &coeff_len)?
            },
        };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

/// Prover-only runtime cache derived from the persisted setup commitment
/// artifact.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SharedMatrixOpeningProverCache<F: FieldCore> {
    pub(crate) artifact: SharedMatrixCommitmentArtifact<F>,
    pub(crate) lifted_evals: Arc<Vec<F>>,
    pub(crate) hint: RecursiveCommitmentHintCache<F>,
}

/// All data needed at L=0 to emit the setup-claim carry.
pub(crate) struct SharedMatrixSetup<F: FieldCore> {
    /// Canonical tensor layout used by the shared matrix polynomial.
    pub tensor_layout: SharedMatrixTensorLayout,
    /// Flat field-evals vector of the outer-D shared-matrix tensor (see
    /// [`SharedMatrixVerifierCache::sm_evals_flat`]). Shared with the
    /// setup-claim carry consumer at downstream recursion levels.
    pub sm_evals_flat: Arc<Vec<F>>,
}

/// Config-level choice of commitment preset that supports the setup-claim
/// carry.
///
/// The setup-claim carry currently requires the outer matrix weight / tensor
/// layout routines that live in the same subsystem as the preset implementors
/// below; marker trait so only the supported presets can activate the carry.
pub trait SharedMatrixOpeningConfig: CommitmentConfig {}

impl<
        F: CanonicalField + FieldCore + Send + Sync + 'static,
        Profile: CommitmentFieldProfile + CommitmentFieldProfileSchedule,
        const D: usize,
    > SharedMatrixOpeningConfig for CommitmentPreset<F, GeneratedAdaptivePolicy<Profile, D, 1>>
{
}

impl<
        F: CanonicalField + FieldCore + Send + Sync + 'static,
        Profile: CommitmentFieldProfile + CommitmentFieldProfileSchedule,
        const D: usize,
    > SharedMatrixOpeningConfig for CommitmentPreset<F, GeneratedAdaptivePolicy<Profile, D, 128>>
{
}

impl<
        F: CanonicalField + FieldCore + Send + Sync + 'static,
        Profile: CommitmentFieldProfile + CommitmentFieldProfileSchedule,
        const D: usize,
        const LOG_BASIS: u32,
        const W_LOG_BASIS: u32,
        const N_A: usize,
        const N_B: usize,
        const N_D: usize,
    > SharedMatrixOpeningConfig
    for CommitmentPreset<
        F,
        StaticBoundedPolicy<Profile, D, 1, LOG_BASIS, W_LOG_BASIS, N_A, N_B, N_D>,
    >
{
}

impl<
        F: CanonicalField + FieldCore + Send + Sync + 'static,
        Profile: CommitmentFieldProfile + CommitmentFieldProfileSchedule,
        const D: usize,
        const LOG_BASIS: u32,
        const W_LOG_BASIS: u32,
        const N_A: usize,
        const N_B: usize,
        const N_D: usize,
    > SharedMatrixOpeningConfig
    for CommitmentPreset<
        F,
        StaticBoundedPolicy<Profile, D, 128, LOG_BASIS, W_LOG_BASIS, N_A, N_B, N_D>,
    >
{
}

/// Build the verifier cache from the main prover setup at setup time.
pub(crate) fn build_shared_matrix_verifier_cache<
    F: FieldCore + CanonicalField + FieldSampling + HasWide + HasUnreducedOps + Valid,
    const D: usize,
>(
    main_setup: &HachiProverSetup<F, D>,
) -> Result<SharedMatrixVerifierCache<F>, HachiError> {
    let sm = SharedMatrixSetup::<F>::from_main_prover_setup::<D>(main_setup)?;
    Ok(SharedMatrixVerifierCache {
        tensor_layout: sm.tensor_layout,
        sm_evals_flat: sm.sm_evals_flat,
    })
}

impl<F> SharedMatrixSetup<F>
where
    F: FieldCore + CanonicalField + FieldSampling + HasWide + HasUnreducedOps + Valid,
{
    /// Build the shared matrix setup from the expanded setup only.
    pub(crate) fn from_expanded<const D: usize>(
        expanded: &HachiExpandedSetup<F>,
    ) -> Result<Self, HachiError> {
        let tensor_layout = SharedMatrixTensorLayout::from_expanded::<F, D>(expanded);
        let field_evals =
            flat_matrix_to_field_evals::<F, D>(&expanded.shared_matrix, tensor_layout);
        Ok(Self {
            tensor_layout,
            sm_evals_flat: Arc::new(field_evals),
        })
    }

    /// Build the shared matrix setup from the main prover setup.
    pub(crate) fn from_main_prover_setup<const D: usize>(
        main_setup: &HachiProverSetup<F, D>,
    ) -> Result<Self, HachiError> {
        Self::from_expanded::<D>(&main_setup.expanded)
    }
}

pub(crate) fn shared_matrix_opening_num_vars(lp: &LevelParams) -> Result<usize, HachiError> {
    lp.m_vars
        .checked_add(lp.r_vars)
        .and_then(|n| n.checked_add(lp.ring_dimension.trailing_zeros() as usize))
        .ok_or_else(|| {
            HachiError::InvalidSetup("shared matrix opening point length overflow".to_string())
        })
}

fn embed_shared_matrix_field_evals_for_opening_lp<F: FieldCore>(
    sm_setup: &SharedMatrixSetup<F>,
    opening_lp: &LevelParams,
) -> Result<Vec<F>, HachiError> {
    let alpha_bits = opening_lp.ring_dimension.trailing_zeros() as usize;
    if alpha_bits != sm_setup.tensor_layout.ring_vars {
        return Err(HachiError::InvalidSetup(format!(
            "shared matrix opening ring vars mismatch: tensor has {}, opening lp has {}",
            sm_setup.tensor_layout.ring_vars, alpha_bits
        )));
    }
    let src_outer_vars = sm_setup
        .tensor_layout
        .col_vars
        .checked_add(sm_setup.tensor_layout.row_vars)
        .ok_or_else(|| {
            HachiError::InvalidSetup("shared matrix source outer vars overflow".to_string())
        })?;
    let dst_outer_vars = opening_lp
        .m_vars
        .checked_add(opening_lp.r_vars)
        .ok_or_else(|| {
            HachiError::InvalidSetup("shared matrix opening outer vars overflow".to_string())
        })?;
    if dst_outer_vars < src_outer_vars {
        return Err(HachiError::InvalidSetup(format!(
            "shared matrix opening layout too small: need at least {src_outer_vars} outer vars, got {dst_outer_vars}",
        )));
    }
    let opening_num_vars = shared_matrix_opening_num_vars(opening_lp)?;
    let total_len = 1usize.checked_shl(opening_num_vars as u32).ok_or_else(|| {
        HachiError::InvalidSetup("shared matrix embedded opening length overflow".to_string())
    })?;
    let mut out = vec![F::zero(); total_len];
    let ring_stride = 1usize << alpha_bits;
    let src_position_stride = 1usize << sm_setup.tensor_layout.col_vars;
    for row in 0..sm_setup.tensor_layout.padded_rows {
        for col in 0..sm_setup.tensor_layout.padded_stride {
            let outer_idx = row
                .checked_mul(src_position_stride)
                .and_then(|idx| idx.checked_add(col))
                .ok_or_else(|| {
                    HachiError::InvalidSetup(
                        "shared matrix embedded opening outer index overflow".to_string(),
                    )
                })?;
            let base = outer_idx.checked_mul(ring_stride).ok_or_else(|| {
                HachiError::InvalidSetup(
                    "shared matrix embedded opening field index overflow".to_string(),
                )
            })?;
            for k in 0..ring_stride {
                out[base + k] =
                    sm_setup.sm_evals_flat[sm_setup.tensor_layout.flat_index(row, col, k)];
            }
        }
    }
    Ok(out)
}

fn bless_opening_lp_for_setup_tensor<Cfg: CommitmentConfig>(
    opening_lp: &LevelParams,
    current_w_len: usize,
    required_num_vars: usize,
) -> Result<LevelParams, HachiError> {
    let base = hachi_recursive_level_layout_from_params::<Cfg>(opening_lp, current_w_len)?;
    let alpha_bits = base.ring_dimension.trailing_zeros() as usize;
    let base_num_vars = base
        .m_vars
        .checked_add(base.r_vars)
        .and_then(|n| n.checked_add(alpha_bits))
        .ok_or_else(|| {
            HachiError::InvalidSetup("carry-opening layout arity overflow".to_string())
        })?;
    if base_num_vars >= required_num_vars {
        return Ok(base);
    }
    let extra_m_vars = required_num_vars - base_num_vars;
    let num_ring = current_w_len / base.ring_dimension;
    base.with_decomp(
        base.m_vars + extra_m_vars,
        base.r_vars,
        base.num_digits_commit,
        base.num_digits_open,
        base.num_digits_fold,
        num_ring,
    )
}

fn carry_opening_schedule_level_params<F, const D: usize, Cfg>(
    expanded: &HachiExpandedSetup<F>,
    sm_setup: &SharedMatrixSetup<F>,
) -> Result<Option<LevelParams>, HachiError>
where
    F: FieldCore,
    Cfg: CommitmentConfig<Field = F>,
{
    let root_plan = hachi_root_runtime_plan::<Cfg, D>(
        expanded.seed.max_num_vars,
        expanded.seed.max_num_vars,
        expanded.seed.max_num_batched_polys,
    )?;
    let root_key = root_plan.lookup_key();
    let schedule = if let Some(plan) = Cfg::schedule_plan(root_key)? {
        plan
    } else {
        exact_schedule_plan_for_lookup_key::<Cfg, D>(root_key)?
    };
    let opening_lp = schedule
        .fold_levels()
        .nth(2)
        .map(|level| {
            bless_opening_lp_for_setup_tensor::<Cfg>(
                &level.lp,
                level.inputs.current_w_len,
                sm_setup.tensor_layout.num_vars,
            )
        })
        .transpose()?;
    Ok(opening_lp)
}

fn build_shared_matrix_opening_cache_at_d<F, const D_OUTER: usize, const D_OPEN: usize>(
    expanded: &HachiExpandedSetup<F>,
    sm_setup: &SharedMatrixSetup<F>,
    opening_lp: &LevelParams,
) -> Result<SharedMatrixOpeningProverCache<F>, HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling + HasWide + HasUnreducedOps + Valid,
{
    let opening_num_vars = shared_matrix_opening_num_vars(opening_lp)?;
    let lifted_evals = embed_shared_matrix_field_evals_for_opening_lp(sm_setup, opening_lp)?;
    let poly = DensePoly::<F, D_OPEN>::from_field_evals(opening_num_vars, &lifted_evals)?;
    let total = expanded.shared_matrix.total_ring_elements_at::<D_OPEN>();
    let ntt_shared = build_ntt_slot(expanded.shared_matrix.ring_view::<D_OPEN>(1, total))?;
    let mut inner = poly.commit_inner_witness(
        &expanded.shared_matrix,
        &ntt_shared,
        opening_lp.a_key.row_len(),
        opening_lp.block_len,
        opening_lp.num_digits_commit,
        opening_lp.num_digits_open,
        opening_lp.log_basis,
        expanded.seed.max_stride,
    )?;
    for t_i in &mut inner.t {
        t_i.truncate(opening_lp.a_key.row_len());
    }
    inner
        .t_hat
        .truncate_each_block(opening_lp.a_key.row_len() * opening_lp.num_digits_open);
    let commitment_rows = mat_vec_mul_ntt_single_i8(
        &ntt_shared,
        opening_lp.b_key.row_len(),
        expanded.seed.max_stride,
        inner.t_hat.flat_digits(),
    );
    let hint = RecursiveCommitmentHintCache::from_typed(HachiCommitmentHint::with_t(
        inner.t_hat,
        inner.t,
    ))?;
    Ok(SharedMatrixOpeningProverCache {
        artifact: SharedMatrixCommitmentArtifact {
            tensor_layout: sm_setup.tensor_layout,
            opening_lp: opening_lp.clone(),
            commitment: FlatRingVec::from_ring_elems(&commitment_rows).into_compact(),
        },
        lifted_evals: Arc::new(lifted_evals),
        hint,
    })
}

pub(crate) fn embed_setup_sumcheck_point_for_opening_lp<F: FieldCore>(
    raw_point: &[F],
    artifact: &SharedMatrixCommitmentArtifact<F>,
) -> Result<Vec<F>, HachiError> {
    if raw_point.len() != artifact.tensor_layout.num_vars {
        return Err(HachiError::InvalidPointDimension {
            expected: artifact.tensor_layout.num_vars,
            actual: raw_point.len(),
        });
    }
    let alpha_bits = artifact.opening_lp.ring_dimension.trailing_zeros() as usize;
    if alpha_bits != artifact.tensor_layout.ring_vars {
        return Err(HachiError::InvalidSetup(format!(
            "shared matrix opening ring vars mismatch: tensor has {}, opening lp has {}",
            artifact.tensor_layout.ring_vars, alpha_bits
        )));
    }
    let src_outer_vars = artifact
        .tensor_layout
        .col_vars
        .checked_add(artifact.tensor_layout.row_vars)
        .ok_or_else(|| {
            HachiError::InvalidSetup("shared matrix source outer vars overflow".to_string())
        })?;
    let dst_outer_vars = artifact
        .opening_lp
        .m_vars
        .checked_add(artifact.opening_lp.r_vars)
        .ok_or_else(|| {
            HachiError::InvalidSetup("shared matrix opening outer vars overflow".to_string())
        })?;
    if dst_outer_vars < src_outer_vars {
        return Err(HachiError::InvalidSetup(format!(
            "shared matrix opening layout too small: need at least {src_outer_vars} outer vars, got {dst_outer_vars}",
        )));
    }
    let (r_row, r_col, r_k) = artifact.tensor_layout.split_point(raw_point)?;
    let mut outer_bits = Vec::with_capacity(src_outer_vars);
    outer_bits.extend_from_slice(r_col);
    outer_bits.extend_from_slice(r_row);
    let mut out = Vec::with_capacity(shared_matrix_opening_num_vars(&artifact.opening_lp)?);
    out.extend_from_slice(r_k);
    let position_take = artifact.opening_lp.m_vars.min(outer_bits.len());
    out.extend_from_slice(&outer_bits[..position_take]);
    out.extend(std::iter::repeat_n(
        F::zero(),
        artifact.opening_lp.m_vars - position_take,
    ));
    out.extend_from_slice(&outer_bits[position_take..]);
    out.extend(std::iter::repeat_n(
        F::zero(),
        artifact.opening_lp.r_vars - outer_bits.len().saturating_sub(position_take),
    ));
    Ok(out)
}

pub(crate) fn build_shared_matrix_opening_cache<
    F: FieldCore + CanonicalField + FieldSampling + HasWide + HasUnreducedOps + Valid,
    const D: usize,
    Cfg: CommitmentConfig<Field = F>,
>(
    expanded: &HachiExpandedSetup<F>,
) -> Result<Option<SharedMatrixOpeningProverCache<F>>, HachiError> {
    let sm_setup = SharedMatrixSetup::<F>::from_expanded::<D>(expanded)?;
    let Some(opening_lp) = carry_opening_schedule_level_params::<F, D, Cfg>(expanded, &sm_setup)?
    else {
        return Ok(None);
    };
    crate::dispatch_ring_dim!(opening_lp.ring_dimension, |D_OPEN| {
        build_shared_matrix_opening_cache_at_d::<F, D, { D_OPEN }>(expanded, &sm_setup, &opening_lp)
            .map(Some)
    })
}

#[cfg_attr(not(feature = "disk-persistence"), allow(dead_code))]
pub(crate) fn rebuild_shared_matrix_opening_cache_from_artifact<
    F: FieldCore + CanonicalField + FieldSampling + HasWide + HasUnreducedOps + Valid,
    const D: usize,
>(
    expanded: &HachiExpandedSetup<F>,
) -> Result<Option<SharedMatrixOpeningProverCache<F>>, HachiError> {
    let Some(artifact) = expanded.shared_matrix_commitment.as_ref() else {
        return Ok(None);
    };
    let sm_setup = SharedMatrixSetup::<F>::from_expanded::<D>(expanded)?;
    crate::dispatch_ring_dim!(artifact.opening_lp.ring_dimension, |D_OPEN| {
        build_shared_matrix_opening_cache_at_d::<F, D, { D_OPEN }>(
            expanded,
            &sm_setup,
            &artifact.opening_lp,
        )
        .map(|mut cache| {
            cache.artifact = artifact.clone();
            Some(cache)
        })
    })
}

/// Extract the shared matrix as a padded field-element tensor evaluation vector.
fn flat_matrix_to_field_evals<F: FieldCore, const D: usize>(
    matrix: &crate::protocol::commitment::utils::flat_matrix::FlatMatrix<F>,
    tensor_layout: SharedMatrixTensorLayout,
) -> Vec<F> {
    let view = matrix.ring_view::<D>(tensor_layout.max_rows, tensor_layout.stride);
    let mut evals = vec![F::zero(); tensor_layout.field_len()];
    for row in 0..tensor_layout.max_rows {
        for col in 0..tensor_layout.stride {
            let ring_elem = &view.row(row)[col];
            for (k, coeff) in ring_elem.coefficients().iter().enumerate() {
                evals[tensor_layout.flat_index(row, col, k)] = *coeff;
            }
        }
    }
    evals
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::commitment::presets::fp128;
    use crate::protocol::commitment::CommitmentScheme;
    use crate::protocol::commitment_scheme::HachiCommitmentScheme;

    type F = fp128::Field;
    type Cfg = fp128::D128Full;
    const D: usize = Cfg::D;

    #[test]
    fn shared_matrix_setup_exposes_nonempty_flat_evals() {
        const NV: usize = 12;
        let main_setup =
            <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::setup_prover(NV, 1, 1);

        let sm_setup = SharedMatrixSetup::<F>::from_main_prover_setup::<D>(&main_setup)
            .expect("SharedMatrixSetup creation");

        assert_eq!(
            sm_setup.sm_evals_flat.len(),
            sm_setup.tensor_layout.field_len(),
            "flat evals length must match tensor layout field length"
        );
        assert!(
            sm_setup.tensor_layout.num_vars > 0,
            "shared matrix num_vars must be positive"
        );
    }

    #[test]
    fn shared_matrix_tensor_layout_is_power_of_two_tensor() {
        const NV: usize = 12;
        let main_setup =
            <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::setup_prover(NV, 1, 1);
        let tensor_layout = SharedMatrixTensorLayout::from_expanded::<F, D>(&main_setup.expanded);
        assert_eq!(tensor_layout.field_len(), 1usize << tensor_layout.num_vars);
        assert_eq!(tensor_layout.ring_vars, D.trailing_zeros() as usize);
        assert!(tensor_layout.padded_rows >= tensor_layout.max_rows);
        assert!(tensor_layout.padded_stride >= tensor_layout.stride);
    }
}
