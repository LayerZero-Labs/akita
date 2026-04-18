//! Setup-time commitment to the shared matrix.
//!
//! The shared matrix is the flat matrix (A/B/D backing data) viewed as a
//! padded `row × col × k` tensor. This module defines the canonical tensor
//! layout and packages the deterministic commitment/opening data needed for the
//! delegated setup-claim proof.

use crate::algebra::fields::wide::HasWide;
use crate::algebra::fields::HasUnreducedOps;
use crate::error::HachiError;
use crate::primitives::serialization::Valid;
use crate::protocol::commitment::utils::crt_ntt::build_ntt_slot;
use crate::protocol::commitment::utils::matrix::derive_public_matrix_flat;
use crate::protocol::commitment::{
    profile::CommitmentFieldProfileSchedule, CommitmentConfig, CommitmentFieldProfile,
    CommitmentPreset, CommitmentScheme, GeneratedAdaptivePolicy, RingCommitment,
    StaticBoundedPolicy,
};
use crate::protocol::commitment_scheme::HachiCommitmentScheme;
use crate::protocol::hachi_poly_ops::DensePoly;
use crate::protocol::proof::{FlatRingVec, HachiBatchedCommitmentHint};
use crate::protocol::setup::{
    HachiExpandedSetup, HachiProverSetup, HachiSetupSeed, HachiVerifierSetup,
};
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

/// Precomputed verifier-side data for setup delegation.
///
/// Built once during `setup_verifier` so that the verifier never re-derives
/// the inner PCS setup or re-commits the shared matrix at verification time.
/// Holds a pre-materialized flat field-evals view of the shared matrix so
/// later recursion levels can close the delegated setup claim directly
/// without re-walking the `FlatMatrix` or running a nested PCS opening.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SharedMatrixVerifierCache<F: FieldCore> {
    pub(crate) tensor_layout: SharedMatrixTensorLayout,
    pub(crate) inner_verifier_setup: Box<HachiVerifierSetup<F>>,
    pub(crate) commitment: FlatRingVec<F>,
    /// Flat field-evals vector of the outer-D shared-matrix tensor, shaped as
    /// `padded_rows × padded_stride × D` (little-endian-in-k ordering per
    /// [`SharedMatrixTensorLayout::flat_index`]). Shared with the setup-claim
    /// carry consumer at downstream recursion levels.
    pub(crate) sm_evals_flat: Arc<Vec<F>>,
}

impl<F: FieldCore> SharedMatrixVerifierCache<F> {
    pub(crate) fn typed_commitment<const D: usize>(
        &self,
    ) -> Result<RingCommitment<F, D>, HachiError> {
        let rings = self.commitment.as_ring_slice::<D>()?;
        Ok(RingCommitment { u: rings.to_vec() })
    }
}

/// All data needed to open the shared matrix polynomial at a point.
pub(crate) struct SharedMatrixSetup<F: FieldCore, const D: usize> {
    /// Canonical tensor layout used by the committed shared matrix polynomial.
    pub tensor_layout: SharedMatrixTensorLayout,
    /// Reused PCS prover setup from the main proof setup.
    pub prover_setup: HachiProverSetup<F, D>,
    /// Reused PCS verifier setup from the main proof setup.
    pub verifier_setup: HachiVerifierSetup<F>,
    /// Commitment to the shared matrix polynomial.
    pub commitment: RingCommitment<F, D>,
    /// Prover hint from the initial commitment (needed for opening proof).
    pub commit_hint: HachiBatchedCommitmentHint<F, D>,
    /// Shared matrix as a DensePoly for the delegated opening proof.
    pub shared_matrix_poly: DensePoly<F, D>,
    /// Flat field-evals vector of the outer-D shared-matrix tensor (see
    /// [`SharedMatrixVerifierCache::sm_evals_flat`]). Shared with the
    /// setup-claim carry consumer at downstream recursion levels.
    pub sm_evals_flat: Arc<Vec<F>>,
}

/// Choose the inner PCS config used to open the shared matrix polynomial.
///
/// The delegated shared matrix always has dense, arbitrary field coefficients,
/// so onehot outer configs must switch to a full-field inner PCS.
/// Config-level choice of inner PCS used to open the shared matrix at a
/// setup-delegation level.
///
/// The delegated shared matrix always has dense, arbitrary field coefficients,
/// so onehot outer configs must switch to a full-field inner PCS. Preset
/// implementations in this module pick the right inner config automatically.
pub trait SharedMatrixOpeningConfig: CommitmentConfig {
    /// Inner PCS config used for the recursive shared-matrix opening.
    type InnerCfg: SharedMatrixOpeningConfig<Field = Self::Field>;
}

impl<
        F: CanonicalField + FieldCore + Send + Sync + 'static,
        Profile: CommitmentFieldProfile + CommitmentFieldProfileSchedule,
        const D: usize,
    > SharedMatrixOpeningConfig for CommitmentPreset<F, GeneratedAdaptivePolicy<Profile, D, 1>>
{
    type InnerCfg = CommitmentPreset<F, GeneratedAdaptivePolicy<Profile, D, 128>>;
}

impl<
        F: CanonicalField + FieldCore + Send + Sync + 'static,
        Profile: CommitmentFieldProfile + CommitmentFieldProfileSchedule,
        const D: usize,
    > SharedMatrixOpeningConfig for CommitmentPreset<F, GeneratedAdaptivePolicy<Profile, D, 128>>
{
    type InnerCfg = Self;
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
    type InnerCfg = CommitmentPreset<
        F,
        StaticBoundedPolicy<Profile, D, 128, LOG_BASIS, W_LOG_BASIS, N_A, N_B, N_D>,
    >;
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
    type InnerCfg = Self;
}

/// Build the verifier cache from the main prover setup at setup time.
pub(crate) fn build_shared_matrix_verifier_cache<
    F: FieldCore + CanonicalField + FieldSampling + HasWide + HasUnreducedOps + Valid,
    const D: usize,
    Cfg: SharedMatrixOpeningConfig<Field = F>,
>(
    main_setup: &HachiProverSetup<F, D>,
) -> Result<SharedMatrixVerifierCache<F>, HachiError> {
    let sm = SharedMatrixSetup::<F, D>::from_main_prover_setup::<Cfg>(main_setup)?;
    Ok(SharedMatrixVerifierCache {
        tensor_layout: sm.tensor_layout,
        inner_verifier_setup: Box::new(sm.verifier_setup),
        commitment: FlatRingVec::from_commitment(&sm.commitment),
        sm_evals_flat: sm.sm_evals_flat,
    })
}

impl<F, const D: usize> SharedMatrixSetup<F, D>
where
    F: FieldCore + CanonicalField + FieldSampling + HasWide + HasUnreducedOps + Valid,
{
    /// Build the shared matrix setup from the main prover setup.
    ///
    /// This reuses the main public setup rather than sampling a fresh PCS setup.
    pub(crate) fn from_main_prover_setup<Cfg: SharedMatrixOpeningConfig<Field = F>>(
        main_setup: &HachiProverSetup<F, D>,
    ) -> Result<Self, HachiError> {
        let tensor_layout = SharedMatrixTensorLayout::from_expanded::<F, D>(&main_setup.expanded);
        if <Cfg::InnerCfg as CommitmentConfig>::D != D {
            return Err(HachiError::InvalidSetup(
                "shared matrix inner config must preserve ring dimension".to_string(),
            ));
        }
        let prover_setup = derive_inner_prover_setup::<F, D, Cfg::InnerCfg>(
            &main_setup.expanded.seed,
            tensor_layout.num_vars,
        )?;
        let verifier_setup = inner_verifier_setup(&prover_setup);
        Self::from_shared_setup::<Cfg::InnerCfg>(
            &main_setup.expanded.shared_matrix,
            tensor_layout,
            prover_setup,
            verifier_setup,
        )
    }

    fn from_shared_setup<Cfg: SharedMatrixOpeningConfig<Field = F>>(
        source_matrix: &crate::protocol::commitment::utils::flat_matrix::FlatMatrix<F>,
        tensor_layout: SharedMatrixTensorLayout,
        prover_setup: HachiProverSetup<F, D>,
        verifier_setup: HachiVerifierSetup<F>,
    ) -> Result<Self, HachiError> {
        let field_evals = flat_matrix_to_field_evals::<F, D>(source_matrix, tensor_layout);
        let shared_matrix_poly =
            DensePoly::<F, D>::from_field_evals(tensor_layout.num_vars, &field_evals)?;
        let (commitment, commit_hint) =
            <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::commit(
                &[shared_matrix_poly.clone()],
                &prover_setup,
            )?;

        Ok(Self {
            tensor_layout,
            prover_setup,
            verifier_setup,
            commitment,
            commit_hint,
            shared_matrix_poly,
            sm_evals_flat: Arc::new(field_evals),
        })
    }
}

fn derive_inner_prover_setup<F, const D: usize, Cfg: SharedMatrixOpeningConfig<Field = F>>(
    main_seed: &HachiSetupSeed,
    shared_matrix_num_vars: usize,
) -> Result<HachiProverSetup<F, D>, HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling + HasWide + HasUnreducedOps + Valid,
{
    let sampled_setup = <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::setup_prover(
        shared_matrix_num_vars,
        1,
        1,
    );
    let mut inner_seed = sampled_setup.expanded.seed.clone();
    inner_seed.public_matrix_seed = main_seed.public_matrix_seed;
    let total_ring_elements = sampled_setup
        .expanded
        .shared_matrix
        .total_ring_elements_at::<D>();
    let shared_matrix =
        derive_public_matrix_flat::<F, D>(total_ring_elements, &inner_seed.public_matrix_seed);
    let ntt_shared = build_ntt_slot(shared_matrix.ring_view::<D>(1, total_ring_elements))?;
    let expanded = Arc::new(HachiExpandedSetup {
        seed: inner_seed,
        shared_matrix,
    });
    Ok(HachiProverSetup {
        expanded,
        ntt_shared,
        mode: crate::protocol::protocol_mode::HachiProtocolMode::default(),
        delegation: crate::protocol::commitment_scheme::SetupDelegationMode::default(),
    })
}

fn inner_verifier_setup<F: FieldCore, const D: usize>(
    prover_setup: &HachiProverSetup<F, D>,
) -> HachiVerifierSetup<F> {
    HachiVerifierSetup {
        expanded: prover_setup.expanded.clone(),
        shared_matrix_cache: None,
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
    use crate::protocol::hachi_poly_ops::HachiPolyOps;

    type F = fp128::Field;
    type Cfg = fp128::D128Full;
    const D: usize = Cfg::D;

    #[test]
    fn shared_matrix_setup_creates_valid_commitment() {
        const NV: usize = 12;
        let main_setup =
            <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::setup_prover(NV, 1, 1);

        let sm_setup = SharedMatrixSetup::<F, D>::from_main_prover_setup::<Cfg>(&main_setup)
            .expect("SharedMatrixSetup creation");

        assert!(
            !sm_setup.commitment.u.is_empty(),
            "shared matrix commitment must be non-empty"
        );
        assert!(
            sm_setup.tensor_layout.num_vars > 0,
            "shared matrix num_vars must be positive"
        );
        assert_eq!(
            sm_setup.shared_matrix_poly.num_vars(),
            sm_setup.tensor_layout.num_vars,
            "poly num_vars must match tensor layout num_vars"
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
