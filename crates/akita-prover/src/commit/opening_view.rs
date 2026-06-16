//! The witness → opening seam.
//!
//! Each polynomial representation implements [`AjtaiOpeningView`] to present
//! itself as an [`AjtaiOpeningType`] for the inner `A` commit. This replaces
//! the per-representation `AkitaPolyOps::commit_inner` method: a representation
//! now only says how to present itself as an opening; the shared
//! `commit_inner_one` helper owns the commit + decompose + validation.

use akita_field::{AkitaError, FieldCore};

use crate::commit::ajtai::opening::AjtaiOpeningType;

/// Borrow this witness as the `A`-side opening for the given block shape.
pub trait AjtaiOpeningView<F: FieldCore, const D: usize> {
    /// Present this witness as an [`AjtaiOpeningType`] for an `A` commit of
    /// `num_blocks` blocks, each `block_len` ring elements wide, at
    /// `num_digits_commit` commit digits.
    ///
    /// # Errors
    ///
    /// Returns an error if the representation cannot build a block view for the
    /// requested shape (e.g. a one-hot block cache mismatch).
    fn to_ajtai_opening(
        &self,
        block_len: usize,
        num_blocks: usize,
        num_digits_commit: usize,
        log_basis: u32,
    ) -> Result<AjtaiOpeningType<'_, F, D>, AkitaError>;
}

impl<F, const D: usize, P> AjtaiOpeningView<F, D> for &P
where
    F: FieldCore,
    P: AjtaiOpeningView<F, D>,
{
    fn to_ajtai_opening(
        &self,
        block_len: usize,
        num_blocks: usize,
        num_digits_commit: usize,
        log_basis: u32,
    ) -> Result<AjtaiOpeningType<'_, F, D>, AkitaError> {
        <P as AjtaiOpeningView<F, D>>::to_ajtai_opening(
            *self,
            block_len,
            num_blocks,
            num_digits_commit,
            log_basis,
        )
    }
}
