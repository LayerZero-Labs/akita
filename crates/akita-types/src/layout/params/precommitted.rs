use akita_challenges::TensorChallengeShape;
use akita_field::AkitaError;

use crate::descriptor_bytes::push_usize;
use crate::schedule::PrecommittedGroupDescriptor;
use crate::sis::{InnerCommitMatrixParams, OuterCommitMatrixParams};

use super::CommittedGroupParams;

/// Group-local root parameters for a precommitted commitment group.
///
/// These fields mirror the group-local pieces of [`CommittedGroupParams`]. Widths are
/// derived from the Ajtai keys and block geometry rather than stored twice.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrecommittedLevelParams {
    /// Frozen standalone group layout bound into the multi-group root key.
    pub layout: PrecommittedGroupDescriptor,
    /// Inner Ajtai matrix (A) used by this group.
    pub inner_commit_matrix: InnerCommitMatrixParams,
    /// Outer commitment matrix (B) used by this group.
    pub outer_commit_matrix: OuterCommitMatrixParams,
    /// Opening basis used by the shared D matrix for fresh `e_hat` digits.
    pub log_basis_open: u32,
    /// Gadget decomposition depth for A/source coefficients.
    pub num_digits_inner: usize,
    /// Gadget decomposition depth for B/`t_hat` values.
    pub num_digits_outer: usize,
    /// Gadget decomposition depth for fresh `e_hat` values.
    pub num_digits_open: usize,
    /// Cached folded-witness digit count for a singleton group relation.
    pub num_digits_fold_one: usize,
}

impl PrecommittedLevelParams {
    /// Validate role ownership and exact A/B widths for serialized group params.
    pub fn validate(&self) -> Result<(), AkitaError> {
        self.layout.validate()?;
        if self.log_basis_open == 0
            || self.num_digits_inner == 0
            || self.num_digits_outer == 0
            || self.num_digits_open == 0
            || self.num_digits_fold_one == 0
        {
            return Err(AkitaError::InvalidSetup(
                "precommitted semantic bases and digit depths must be nonzero".to_string(),
            ));
        }
        if self.log_basis_open < self.layout.log_basis_inner
            || self.log_basis_open < self.layout.log_basis_outer
        {
            return Err(AkitaError::InvalidSetup(
                "certified opening basis must dominate precommitted inner/outer bases".to_string(),
            ));
        }
        let expected_a_width = self
            .layout
            .num_positions_per_block
            .checked_mul(self.num_digits_inner)
            .ok_or_else(|| AkitaError::InvalidSetup("precommitted A width overflow".to_string()))?;
        let expected_b_width = self
            .inner_commit_matrix
            .output_rank()
            .checked_mul(self.num_digits_outer)
            .and_then(|width| width.checked_mul(self.layout.num_live_blocks))
            .and_then(|width| width.checked_mul(self.layout.group.num_polynomials()))
            .ok_or_else(|| AkitaError::InvalidSetup("precommitted B width overflow".to_string()))?;
        if self.layout.n_a != self.inner_commit_matrix.output_rank()
            || self.layout.a_coeff_linf_bound != self.inner_commit_matrix.coeff_linf_bound()
            || self.layout.n_b != self.outer_commit_matrix.output_rank()
            || self.layout.b_coeff_linf_bound != self.outer_commit_matrix.coeff_linf_bound()
            || self.inner_commit_matrix.sis_table_key().role != crate::sis::SisMatrixRole::Inner
            || self.inner_commit_matrix.input_width() != expected_a_width
            || self.outer_commit_matrix.sis_table_key().role != crate::sis::SisMatrixRole::Outer
            || self.outer_commit_matrix.input_width() != expected_b_width
        {
            return Err(AkitaError::InvalidSetup(
                "precommitted A/B keys do not match frozen ranks, bounds, or digit depths"
                    .to_string(),
            ));
        }
        Ok(())
    }

    /// Width of this group's A matrix.
    #[inline]
    pub fn inner_width(&self) -> usize {
        self.inner_commit_matrix.input_width()
    }

    /// Width of this group's B matrix.
    #[inline]
    pub fn outer_width(&self) -> usize {
        self.outer_commit_matrix.input_width()
    }

    /// Width contribution to the shared D matrix (`w_hat_g` segment).
    pub fn d_segment_width(&self) -> Result<usize, AkitaError> {
        self.num_digits_open
            .checked_mul(self.layout.num_live_blocks)
            .and_then(|width| width.checked_mul(self.layout.group.num_polynomials()))
            .ok_or_else(|| AkitaError::InvalidSetup("group D segment width overflow".to_string()))
    }

    /// Width contribution of this group's decomposed folded response.
    pub fn z_segment_width(&self, num_digits_fold: usize) -> Result<usize, AkitaError> {
        self.inner_width()
            .checked_mul(num_digits_fold)
            .ok_or_else(|| AkitaError::InvalidSetup("group z segment width overflow".to_string()))
    }

    pub(crate) fn append_descriptor_bytes(&self, bytes: &mut Vec<u8>) {
        self.layout.append_descriptor_bytes(bytes);
        self.inner_commit_matrix.append_descriptor_bytes(bytes);
        self.outer_commit_matrix.append_descriptor_bytes(bytes);
        crate::descriptor_bytes::push_u32(bytes, self.log_basis_open);
        push_usize(bytes, self.num_digits_inner);
        push_usize(bytes, self.num_digits_outer);
        push_usize(bytes, self.num_digits_open);
        push_usize(bytes, self.num_digits_fold_one);
    }
}

/// Common view over full and precommitted level parameters.
///
/// Use this trait when code only needs the shared commitment geometry carried
/// by both [`CommittedGroupParams`] and [`PrecommittedLevelParams`].
pub trait LevelParamsLike {
    fn inner_commit_matrix_params(&self) -> &InnerCommitMatrixParams;
    fn a_rows_len(&self) -> usize;
    fn a_col_len(&self) -> usize;
    fn b_rows_len(&self) -> usize;
    fn b_col_len(&self) -> usize;
    fn num_live_ring_elements_per_claim(&self) -> usize;
    fn num_positions_per_block(&self) -> usize;
    fn num_live_blocks(&self) -> usize;
    fn fold_challenge_shape(&self) -> TensorChallengeShape;
    fn position_index_bits(&self) -> usize;
    fn block_index_bits(&self) -> usize;
    fn num_digits_inner(&self) -> usize;
    fn num_digits_outer(&self) -> usize;
    fn num_digits_open(&self) -> usize;
    fn num_digits_fold_one(&self) -> usize;
    fn log_basis_inner(&self) -> u32;
    fn log_basis_outer(&self) -> u32;
    fn log_basis_open(&self) -> u32;
}

impl LevelParamsLike for CommittedGroupParams {
    fn inner_commit_matrix_params(&self) -> &InnerCommitMatrixParams {
        &self.inner_commit_matrix
    }

    fn a_rows_len(&self) -> usize {
        self.inner_commit_matrix.output_rank()
    }

    fn a_col_len(&self) -> usize {
        self.inner_commit_matrix.input_width()
    }

    fn b_rows_len(&self) -> usize {
        self.outer_commit_matrix.output_rank()
    }

    fn b_col_len(&self) -> usize {
        self.outer_commit_matrix.input_width()
    }

    fn num_live_ring_elements_per_claim(&self) -> usize {
        self.num_live_ring_elements_per_claim
    }

    fn num_positions_per_block(&self) -> usize {
        self.num_positions_per_block
    }

    fn num_live_blocks(&self) -> usize {
        self.num_live_blocks
    }

    fn fold_challenge_shape(&self) -> TensorChallengeShape {
        self.fold_challenge_shape
    }

    fn position_index_bits(&self) -> usize {
        self.position_index_bits()
    }

    fn block_index_bits(&self) -> usize {
        self.block_index_bits()
    }

    fn num_digits_inner(&self) -> usize {
        self.num_digits_inner
    }

    fn num_digits_outer(&self) -> usize {
        self.num_digits_outer
    }

    fn num_digits_open(&self) -> usize {
        self.num_digits_open
    }

    fn num_digits_fold_one(&self) -> usize {
        self.num_digits_fold_one
    }

    fn log_basis_outer(&self) -> u32 {
        self.log_basis_outer
    }

    fn log_basis_inner(&self) -> u32 {
        self.log_basis_inner
    }

    fn log_basis_open(&self) -> u32 {
        self.log_basis_open
    }
}

impl LevelParamsLike for PrecommittedLevelParams {
    fn inner_commit_matrix_params(&self) -> &InnerCommitMatrixParams {
        &self.inner_commit_matrix
    }

    fn a_rows_len(&self) -> usize {
        self.inner_commit_matrix.output_rank()
    }

    fn a_col_len(&self) -> usize {
        self.inner_commit_matrix.input_width()
    }

    fn b_rows_len(&self) -> usize {
        self.outer_commit_matrix.output_rank()
    }

    fn b_col_len(&self) -> usize {
        self.outer_commit_matrix.input_width()
    }

    fn num_live_ring_elements_per_claim(&self) -> usize {
        self.layout.num_live_ring_elements_per_claim
    }

    fn num_positions_per_block(&self) -> usize {
        self.layout.num_positions_per_block
    }

    fn num_live_blocks(&self) -> usize {
        self.layout.num_live_blocks
    }

    fn fold_challenge_shape(&self) -> TensorChallengeShape {
        TensorChallengeShape::Flat
    }

    fn position_index_bits(&self) -> usize {
        self.layout.num_positions_per_block.trailing_zeros() as usize
    }

    fn block_index_bits(&self) -> usize {
        self.layout
            .num_live_blocks
            .checked_next_power_of_two()
            .map_or(0, |capacity| capacity.trailing_zeros() as usize)
    }

    fn num_digits_inner(&self) -> usize {
        self.num_digits_inner
    }

    fn num_digits_outer(&self) -> usize {
        self.num_digits_outer
    }

    fn num_digits_open(&self) -> usize {
        self.num_digits_open
    }

    fn num_digits_fold_one(&self) -> usize {
        self.num_digits_fold_one
    }

    fn log_basis_outer(&self) -> u32 {
        self.layout.log_basis_outer
    }

    fn log_basis_inner(&self) -> u32 {
        self.layout.log_basis_inner
    }

    fn log_basis_open(&self) -> u32 {
        self.log_basis_open
    }
}
