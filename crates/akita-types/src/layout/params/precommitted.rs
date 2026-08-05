use akita_challenges::{SparseChallengeConfig, TensorChallengeShape};
use akita_field::AkitaError;

use crate::descriptor_bytes::push_usize;
use crate::schedule::CommittedGroupProfile;
use crate::sis::InnerCommitMatrixParams;
use crate::CommitmentRingDims;

use super::CommittedGroupParams;

/// Group-local root parameters for a precommitted commitment group.
///
/// These fields mirror the group-local pieces of [`CommittedGroupParams`]. Widths are
/// derived from the Ajtai keys and block geometry rather than stored twice.
#[derive(Debug, Clone)]
pub struct PrecommittedLevelParams {
    /// Frozen standalone group layout bound into the multi-group root key.
    pub layout: CommittedGroupProfile,
    /// Opening basis used by the shared D matrix for fresh `e_hat` digits.
    pub log_basis_open: u32,
    /// Sparse fold-challenge family certified for this group's native A ring.
    pub fold_challenge_config: SparseChallengeConfig,
    /// Gadget decomposition depth for fresh `e_hat` values.
    pub num_digits_open: usize,
    /// Exact folded-witness digit depth selected by this schedule row.
    pub num_digits_fold: usize,
}

impl PartialEq for PrecommittedLevelParams {
    fn eq(&self, other: &Self) -> bool {
        self.layout == other.layout
            && self.log_basis_open == other.log_basis_open
            && self.fold_challenge_config == other.fold_challenge_config
            && self.num_digits_open == other.num_digits_open
            && self.num_digits_fold == other.num_digits_fold
    }
}

impl Eq for PrecommittedLevelParams {}

impl PrecommittedLevelParams {
    /// This group's A/B dimensions completed with the consuming level's shared
    /// D dimension.
    #[must_use]
    pub fn role_dims(&self, shared_opening_ring_dimension: usize) -> CommitmentRingDims {
        CommitmentRingDims {
            inner: self.layout.inner_commit_matrix.ring_dimension(),
            outer: self.layout.outer_commit_matrix.ring_dimension(),
            opening: shared_opening_ring_dimension,
        }
    }

    /// Validate role ownership and exact A/B widths for serialized group params.
    pub fn validate(&self) -> Result<(), AkitaError> {
        let field_bits = self
            .layout
            .inner_commit_matrix
            .sis_modulus_profile()
            .field_bits();
        self.layout.validate(field_bits)?;
        if self.fold_challenge_config.weight() != 0 {
            self.fold_challenge_config
                .validate_for_ring_dim(self.layout.inner_commit_matrix.ring_dimension())
                .map_err(|msg| AkitaError::InvalidSetup(msg.to_string()))?;
        }
        if self.log_basis_open == 0 || self.num_digits_open == 0 || self.num_digits_fold == 0 {
            return Err(AkitaError::InvalidSetup(
                "precommitted exact fold plan is missing or inconsistent".to_string(),
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
            .checked_mul(self.layout.num_digits_inner)
            .ok_or_else(|| AkitaError::InvalidSetup("precommitted A width overflow".to_string()))?;
        let inner_ring_dimension = self.layout.inner_commit_matrix.ring_dimension();
        let outer_ring_dimension = self.layout.outer_commit_matrix.ring_dimension();
        if outer_ring_dimension == 0 || !inner_ring_dimension.is_multiple_of(outer_ring_dimension) {
            return Err(AkitaError::InvalidSetup(
                "precommitted A-native source rings do not decompose into B-native subcolumns"
                    .to_string(),
            ));
        }
        let outer_projection_ratio = inner_ring_dimension / outer_ring_dimension;
        let expected_b_width = self
            .layout
            .inner_commit_matrix
            .output_rank()
            .checked_mul(self.layout.num_digits_outer)
            .and_then(|width| width.checked_mul(self.layout.num_live_blocks))
            .and_then(|width| width.checked_mul(self.layout.group.num_polynomials()))
            .and_then(|width| width.checked_mul(outer_projection_ratio))
            .ok_or_else(|| AkitaError::InvalidSetup("precommitted B width overflow".to_string()))?;
        if self.layout.inner_commit_matrix.input_width() != expected_a_width
            || self.layout.outer_commit_matrix.input_width() != expected_b_width
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
        self.layout.inner_commit_matrix.input_width()
    }

    /// Width of this group's B matrix.
    #[inline]
    pub fn outer_width(&self) -> usize {
        self.layout.outer_commit_matrix.input_width()
    }

    /// Width contribution to the consuming batch's shared D matrix
    /// (`w_hat_g` segment).
    ///
    /// Group metadata owns its A/B dimensions. The D role is batch-shared, so
    /// the caller supplies the consuming level's opening dimension.
    pub fn d_segment_width(&self, opening_ring_dimension: usize) -> Result<usize, AkitaError> {
        let role_dims = self.role_dims(opening_ring_dimension);
        role_dims.validate_role_projection()?;
        let inner_ring_dimension = role_dims.d_a();
        let projection_ratio = inner_ring_dimension / opening_ring_dimension;
        self.num_digits_open
            .checked_mul(self.layout.num_live_blocks)
            .and_then(|width| width.checked_mul(self.layout.group.num_polynomials()))
            .and_then(|width| width.checked_mul(projection_ratio))
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
        crate::descriptor_bytes::push_u32(bytes, self.log_basis_open);
        super::append_schedule_sparse_challenge_descriptor_bytes(
            bytes,
            &self.fold_challenge_config,
        );
        push_usize(bytes, self.num_digits_open);
        push_usize(bytes, self.num_digits_fold);
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
    fn fold_challenge_config(&self) -> SparseChallengeConfig;
    fn position_index_bits(&self) -> usize;
    fn block_index_bits(&self) -> usize;
    fn num_digits_inner(&self) -> usize;
    fn num_digits_outer(&self) -> usize;
    fn num_digits_open(&self) -> usize;
    fn num_digits_fold(&self) -> usize;
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

    fn fold_challenge_config(&self) -> SparseChallengeConfig {
        self.fold_challenge_config
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

    fn num_digits_fold(&self) -> usize {
        self.num_digits_fold
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
        &self.layout.inner_commit_matrix
    }

    fn a_rows_len(&self) -> usize {
        self.layout.inner_commit_matrix.output_rank()
    }

    fn a_col_len(&self) -> usize {
        self.layout.inner_commit_matrix.input_width()
    }

    fn b_rows_len(&self) -> usize {
        self.layout.outer_commit_matrix.output_rank()
    }

    fn b_col_len(&self) -> usize {
        self.layout.outer_commit_matrix.input_width()
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

    fn fold_challenge_config(&self) -> SparseChallengeConfig {
        self.fold_challenge_config
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
        self.layout.num_digits_inner
    }

    fn num_digits_outer(&self) -> usize {
        self.layout.num_digits_outer
    }

    fn num_digits_open(&self) -> usize {
        self.num_digits_open
    }

    fn num_digits_fold(&self) -> usize {
        self.num_digits_fold
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
