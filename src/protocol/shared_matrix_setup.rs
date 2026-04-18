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
use crate::primitives::serialization::Valid;
use crate::protocol::commitment::{
    profile::CommitmentFieldProfileSchedule, CommitmentConfig, CommitmentFieldProfile,
    CommitmentPreset, GeneratedAdaptivePolicy, StaticBoundedPolicy,
};
use crate::protocol::setup::{HachiExpandedSetup, HachiProverSetup};
use crate::{CanonicalField, FieldCore, FieldSampling};
use std::sync::Arc;

/// Canonical tensor layout for the shared matrix commitment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SharedMatrixTensorLayout {
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
    /// Build the shared matrix setup from the main prover setup.
    pub(crate) fn from_main_prover_setup<const D: usize>(
        main_setup: &HachiProverSetup<F, D>,
    ) -> Result<Self, HachiError> {
        let tensor_layout = SharedMatrixTensorLayout::from_expanded::<F, D>(&main_setup.expanded);
        let field_evals =
            flat_matrix_to_field_evals::<F, D>(&main_setup.expanded.shared_matrix, tensor_layout);
        Ok(Self {
            tensor_layout,
            sm_evals_flat: Arc::new(field_evals),
        })
    }
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
