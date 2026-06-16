//! Shared row-decomposition unit for the commit pipeline.
//!
//! Sits between the `A`-side and `B`-side matrix multiplies: `t̂ = decompose(t)`.
//! All-zero blocks are skipped against a pre-zeroed destination, which is
//! byte-identical to decomposing the zero ring, so this one helper reproduces
//! both the previous dense (no-skip) and one-hot/sparse/recursive (skip-zero)
//! decompositions.

use akita_algebra::CyclotomicRing;
#[cfg(feature = "parallel")]
use akita_field::parallel::*;
use akita_field::{AkitaError, CanonicalField, FieldCore};
use akita_types::FlatDigitBlocks;

use crate::kernels::linear::decompose_rows_i8_into;

/// Decompose `t = A·s` rows into opening digits, one block per `A`-image block.
///
/// `out[b]` has `rows[b].len() * num_digits_open` digit planes.
///
/// # Errors
///
/// Returns an error if the flat digit allocation overflows.
pub(crate) fn decompose_rows<F, const D: usize>(
    rows: &[Vec<CyclotomicRing<F, D>>],
    num_digits_open: usize,
    log_basis: u32,
) -> Result<FlatDigitBlocks<D>, AkitaError>
where
    F: FieldCore + CanonicalField,
{
    let block_sizes: Vec<usize> = rows.iter().map(|t_i| t_i.len() * num_digits_open).collect();
    let mut out = FlatDigitBlocks::zeroed(block_sizes)?;
    decompose_rows_into(rows, &mut out, num_digits_open, log_basis);
    Ok(out)
}

/// Decompose `rows` into a pre-zeroed flat digit destination, skipping all-zero
/// blocks (whose pre-zeroed slot already holds the correct zero digits).
pub(crate) fn decompose_rows_into<F, const D: usize>(
    rows: &[Vec<CyclotomicRing<F, D>>],
    dst: &mut FlatDigitBlocks<D>,
    num_digits_open: usize,
    log_basis: u32,
) where
    F: FieldCore + CanonicalField,
{
    let dst_blocks = dst.split_blocks_mut();
    #[cfg(feature = "parallel")]
    cfg_into_iter!(dst_blocks)
        .zip(cfg_iter!(rows))
        .for_each(|(block_dst, t_i)| {
            if !t_i.iter().all(|r| *r == CyclotomicRing::zero()) {
                decompose_rows_i8_into(t_i, block_dst, num_digits_open, log_basis);
            }
        });
    #[cfg(not(feature = "parallel"))]
    dst_blocks
        .into_iter()
        .zip(rows.iter())
        .for_each(|(block_dst, t_i)| {
            if !t_i.iter().all(|r| *r == CyclotomicRing::zero()) {
                decompose_rows_i8_into(t_i, block_dst, num_digits_open, log_basis);
            }
        });
}
