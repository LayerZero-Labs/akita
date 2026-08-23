use super::CpuPreparedSetup;
use crate::backend::packed_digits::PackedSignedDigitView;
use crate::compute::PackedDenseCommitInput;
use akita_algebra::CyclotomicRing;
use akita_error::AkitaError;
#[allow(unused_imports)]
use akita_field::parallel::*;
use akita_field::{CanonicalField, FieldCore};
use akita_types::{balanced_signed_digit_abs_bound, NttCacheKey, NttTransformDomain};
use std::array::from_fn;

#[tracing::instrument(skip_all, name = "dense_commit_rows_exact_i16")]
pub(super) fn dense_commit_rows<F: FieldCore + CanonicalField, const D: usize>(
    prepared: &CpuPreparedSetup<F>,
    n_a: usize,
    row_width: usize,
    block_slices: &[&[CyclotomicRing<F, D>]],
    num_digits_inner: usize,
    log_basis_inner: u32,
) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError> {
    let rhs_abs_bound = balanced_signed_digit_abs_bound(log_basis_inner)
        .ok_or_else(|| AkitaError::InvalidSetup("invalid signed digit basis".into()))?;
    prepared.with_shared_ntt::<D, _>(
        NttCacheKey::from_matrix_shape(
            D,
            n_a,
            row_width,
            NttTransformDomain::ExactNegacyclicI16 {
                width: row_width,
                rhs_abs_bound,
            },
        )?,
        |ntt| {
            cfg_iter!(block_slices)
                .map(|block| {
                    let mut rhs = vec![[0i16; D]; row_width];
                    for (ring_idx, ring) in block.iter().enumerate() {
                        let start = ring_idx * num_digits_inner;
                        ring.balanced_decompose_pow2_i16_into(
                            &mut rhs[start..start + num_digits_inner],
                            log_basis_inner,
                        );
                    }
                    ntt.mat_vec_i16::<F>(log_basis_inner, n_a, &rhs)
                })
                .collect()
        },
    )
}

#[tracing::instrument(skip_all, name = "dense_commit_packed_digit_rows_exact_i16")]
pub(super) fn dense_commit_packed_digit_rows<F: FieldCore + CanonicalField, const D: usize>(
    prepared: &CpuPreparedSetup<F>,
    n_a: usize,
    source: PackedDenseCommitInput<'_>,
    log_basis_inner: u32,
) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError> {
    let row_width = source.row_width()?;
    let rhs_abs_bound = balanced_signed_digit_abs_bound(log_basis_inner)
        .ok_or_else(|| AkitaError::InvalidSetup("invalid signed digit basis".into()))?;
    prepared.with_shared_ntt::<D, _>(
        NttCacheKey::from_matrix_shape(
            D,
            n_a,
            row_width,
            NttTransformDomain::ExactNegacyclicI16 {
                width: row_width,
                rhs_abs_bound,
            },
        )?,
        |ntt| {
            cfg_into_iter!(0..source.num_live_blocks())
                .map(|block_index| {
                    let block = source.decode_block::<D>(block_index)?;
                    let mut rhs = Vec::with_capacity(row_width);
                    rhs.extend(block.iter().map(|digits| from_fn(|k| i16::from(digits[k]))));
                    rhs.resize(row_width, [0i16; D]);
                    ntt.mat_vec_i16::<F>(log_basis_inner, n_a, &rhs)
                })
                .collect()
        },
    )
}

pub(super) fn recursive_packed_witness_commit_rows<
    F: FieldCore + CanonicalField,
    const D: usize,
>(
    prepared: &CpuPreparedSetup<F>,
    digits: PackedSignedDigitView<'_>,
    n_rows: usize,
    num_positions_per_block: usize,
    num_live_blocks: usize,
    num_digits_inner: usize,
    log_basis_inner: u32,
) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError> {
    let row_width = num_positions_per_block
        .checked_mul(num_digits_inner)
        .ok_or_else(|| AkitaError::InvalidSetup("recursive A width overflow".into()))?;
    let rhs_abs_bound = balanced_signed_digit_abs_bound(log_basis_inner)
        .ok_or_else(|| AkitaError::InvalidSetup("invalid signed digit basis".into()))?;
    let ring_elems = digits.len() / D;
    prepared.with_shared_ntt::<D, _>(
        NttCacheKey::from_matrix_shape(
            D,
            n_rows,
            row_width,
            NttTransformDomain::ExactNegacyclicI16 {
                width: row_width,
                rhs_abs_bound,
            },
        )?,
        |ntt| {
            cfg_into_iter!(0..num_live_blocks)
                .map(|block_index| {
                    let start_ring = block_index * num_positions_per_block;
                    let live = (ring_elems - start_ring).min(num_positions_per_block);
                    let block = digits.decode_rings::<D>(start_ring, live)?;
                    let mut rhs = vec![[0i16; D]; row_width];
                    if num_digits_inner == 1 {
                        for (dst, src) in rhs.iter_mut().zip(&block) {
                            *dst = from_fn(|k| i16::from(src[k]));
                        }
                    } else {
                        for (ring_idx, digit) in block.iter().enumerate() {
                            let ring = CyclotomicRing::from_coefficients(from_fn(|k| {
                                F::from_i8(digit[k])
                            }));
                            let start = ring_idx * num_digits_inner;
                            ring.balanced_decompose_pow2_i16_into(
                                &mut rhs[start..start + num_digits_inner],
                                log_basis_inner,
                            );
                        }
                    }
                    ntt.mat_vec_i16::<F>(log_basis_inner, n_rows, &rhs)
                })
                .collect()
        },
    )
}
