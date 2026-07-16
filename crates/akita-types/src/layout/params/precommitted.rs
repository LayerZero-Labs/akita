use akita_error::AkitaError;

use crate::descriptor_bytes::push_usize;
use crate::schedule::PrecommittedGroupParams;
use crate::sis::AjtaiKeyParams;

use super::LevelParams;

/// Group-local root parameters for a precommitted commitment group.
///
/// These fields mirror the group-local pieces of [`LevelParams`]. Widths are
/// derived from the Ajtai keys and block geometry rather than stored twice.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrecommittedLevelParams {
    /// Frozen standalone group layout bound into the multi-group root key.
    pub layout: PrecommittedGroupParams,
    /// Inner Ajtai matrix (A) used by this group.
    pub a_key: AjtaiKeyParams,
    /// Outer commitment matrix (B) used by this group.
    pub b_key: AjtaiKeyParams,
    /// Number of committed blocks (`2^r_vars`) for this group.
    pub num_blocks: usize,
    /// Number of ring elements per block (`2^m_vars`) for this group.
    pub block_len: usize,
    /// Gadget decomposition depth for committed coefficients.
    pub num_digits_commit: usize,
    /// Gadget decomposition depth for opening-side values.
    pub num_digits_open: usize,
    /// Cached folded-witness digit count for a singleton group relation.
    pub num_digits_fold_one: usize,
}

impl PrecommittedLevelParams {
    /// Width of this group's A matrix.
    #[inline]
    pub fn inner_width(&self) -> usize {
        self.a_key.col_len()
    }

    /// Width of this group's B matrix.
    #[inline]
    pub fn outer_width(&self) -> usize {
        self.b_key.col_len()
    }

    /// Width contribution to the shared D matrix (`w_hat_g` segment).
    pub fn d_segment_width(&self) -> Result<usize, AkitaError> {
        self.num_digits_open
            .checked_mul(self.num_blocks)
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
        self.a_key.append_descriptor_bytes(bytes);
        self.b_key.append_descriptor_bytes(bytes);
        push_usize(bytes, self.num_blocks);
        push_usize(bytes, self.block_len);
        push_usize(bytes, self.num_digits_commit);
        push_usize(bytes, self.num_digits_open);
        push_usize(bytes, self.num_digits_fold_one);
    }
}

/// Common view over full and precommitted level parameters.
///
/// Use this trait when code only needs the shared commitment geometry carried
/// by both [`LevelParams`] and [`PrecommittedLevelParams`].
pub trait LevelParamsLike {
    fn a_rows_len(&self) -> usize;
    fn a_col_len(&self) -> usize;
    fn b_rows_len(&self) -> usize;
    fn num_blocks(&self) -> usize;
    fn block_len(&self) -> usize;
    fn m_vars(&self) -> usize;
    fn r_vars(&self) -> usize;
    fn num_digits_commit(&self) -> usize;
    fn num_digits_open(&self) -> usize;
    fn num_digits_fold_one(&self) -> usize;
    fn log_basis(&self) -> u32;
}

impl LevelParamsLike for LevelParams {
    fn a_rows_len(&self) -> usize {
        self.a_key.row_len()
    }

    fn a_col_len(&self) -> usize {
        self.a_key.col_len()
    }

    fn b_rows_len(&self) -> usize {
        self.b_key.row_len()
    }

    fn num_blocks(&self) -> usize {
        self.num_blocks
    }

    fn block_len(&self) -> usize {
        self.block_len
    }

    fn m_vars(&self) -> usize {
        self.m_vars
    }

    fn r_vars(&self) -> usize {
        self.r_vars
    }

    fn num_digits_commit(&self) -> usize {
        self.num_digits_commit
    }

    fn num_digits_open(&self) -> usize {
        self.num_digits_open
    }

    fn num_digits_fold_one(&self) -> usize {
        self.num_digits_fold_one
    }

    fn log_basis(&self) -> u32 {
        self.log_basis
    }
}

impl LevelParamsLike for PrecommittedLevelParams {
    fn a_rows_len(&self) -> usize {
        self.a_key.row_len()
    }

    fn a_col_len(&self) -> usize {
        self.a_key.col_len()
    }

    fn b_rows_len(&self) -> usize {
        self.b_key.row_len()
    }

    fn num_blocks(&self) -> usize {
        self.num_blocks
    }

    fn block_len(&self) -> usize {
        self.block_len
    }

    fn m_vars(&self) -> usize {
        self.layout.m_vars
    }

    fn r_vars(&self) -> usize {
        self.layout.r_vars
    }

    fn num_digits_commit(&self) -> usize {
        self.num_digits_commit
    }

    fn num_digits_open(&self) -> usize {
        self.num_digits_open
    }

    fn num_digits_fold_one(&self) -> usize {
        self.num_digits_fold_one
    }

    fn log_basis(&self) -> u32 {
        self.layout.log_basis
    }
}
