use super::exact_i16::{
    dense_commit_rows as dense_commit_rows_i16,
    recursive_witness_commit_rows as recursive_witness_commit_rows_i16,
};
use super::{CpuBackend, CpuPreparedSetup};
use crate::compute::plans::DenseCommitInput;
use crate::kernels::linear::{
    digit_blocks_are_balanced, mat_vec_mul_ntt_dense_digits_i8, mat_vec_mul_ntt_digits_i8,
    mat_vec_mul_ntt_i8, mat_vec_mul_ntt_i8_dense, mat_vec_mul_ntt_i8_dense_single_row,
    mat_vec_mul_ntt_raw_digits_i8,
};
use crate::validation::signed_digit_kernel_for_setup;
use akita_algebra::CyclotomicRing;
use akita_error::AkitaError;
use akita_field::{CanonicalField, FieldCore};
use akita_types::{NttCacheKey, NttTransformDomain, SignedDigitKernel};
use std::array::from_fn;

impl CpuBackend {
    pub(crate) fn dense_commit_rows<F, const D: usize>(
        &self,
        prepared: &CpuPreparedSetup<F>,
        n_a: usize,
        input: DenseCommitInput<'_, F, D>,
    ) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError>
    where
        F: FieldCore + CanonicalField,
    {
        match input {
            DenseCommitInput::CachedDigits {
                digit_block_slices,
                log_basis_inner,
            } => {
                let row_width = digit_block_slices.first().map_or(0, |digits| digits.len());
                prepared.with_shared_ntt::<D, _>(
                    NttCacheKey::from_matrix_shape(
                        D,
                        n_a,
                        row_width,
                        NttTransformDomain::Negacyclic,
                    )?,
                    |ntt| {
                        mat_vec_mul_ntt_dense_digits_i8(
                            ntt,
                            n_a,
                            row_width,
                            &digit_block_slices,
                            log_basis_inner,
                        )
                    },
                )
            }
            DenseCommitInput::CoeffBlocks {
                block_slices,
                num_digits_inner,
                log_basis_inner,
            } => {
                let row_width = block_slices.first().map_or(Ok(0usize), |block| {
                    block.len().checked_mul(num_digits_inner).ok_or_else(|| {
                        AkitaError::InvalidSetup("dense coefficient row width overflow".into())
                    })
                })?;
                if signed_digit_kernel_for_setup(log_basis_inner, "for dense commitment")?
                    == SignedDigitKernel::I16
                {
                    dense_commit_rows_i16(
                        prepared,
                        n_a,
                        row_width,
                        &block_slices,
                        num_digits_inner,
                        log_basis_inner,
                    )
                } else if n_a == 1 {
                    prepared.with_shared_ntt::<D, _>(
                        NttCacheKey::from_matrix_shape(
                            D,
                            1,
                            row_width,
                            NttTransformDomain::Negacyclic,
                        )?,
                        |ntt| {
                            Ok(mat_vec_mul_ntt_i8_dense_single_row(
                                ntt,
                                row_width,
                                &block_slices,
                                num_digits_inner,
                                log_basis_inner,
                            )?
                            .into_iter()
                            .map(|ring| vec![ring])
                            .collect())
                        },
                    )
                } else {
                    prepared.with_shared_ntt::<D, _>(
                        NttCacheKey::from_matrix_shape(
                            D,
                            n_a,
                            row_width,
                            NttTransformDomain::Negacyclic,
                        )?,
                        |ntt| {
                            mat_vec_mul_ntt_i8_dense(
                                ntt,
                                n_a,
                                row_width,
                                &block_slices,
                                num_digits_inner,
                                log_basis_inner,
                            )
                        },
                    )
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn recursive_witness_commit_rows<F, const D: usize>(
        &self,
        prepared: &CpuPreparedSetup<F>,
        coeffs: &[[i8; D]],
        n_rows: usize,
        num_positions_per_block: usize,
        num_live_blocks: usize,
        num_digits_inner: usize,
        log_basis_inner: u32,
        known_balanced_log_basis: Option<u32>,
    ) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError>
    where
        F: FieldCore + CanonicalField,
    {
        let row_width = num_positions_per_block
            .checked_mul(num_digits_inner)
            .ok_or_else(|| AkitaError::InvalidSetup("recursive A width overflow".into()))?;
        let minimum_ring_elems = num_live_blocks
            .saturating_sub(1)
            .checked_mul(num_positions_per_block)
            .and_then(|prefix| prefix.checked_add(1))
            .ok_or_else(|| {
                AkitaError::InvalidSetup("recursive witness block extent overflow".into())
            })?;
        if num_live_blocks == 0 || coeffs.len() < minimum_ring_elems {
            return Err(AkitaError::InvalidSetup(
                "recursive witness does not cover its live blocks".into(),
            ));
        }
        if signed_digit_kernel_for_setup(log_basis_inner, "for recursive witness commitment")?
            == SignedDigitKernel::I16
        {
            return recursive_witness_commit_rows_i16(
                prepared,
                coeffs,
                n_rows,
                num_positions_per_block,
                num_live_blocks,
                num_digits_inner,
                log_basis_inner,
            );
        }
        if num_digits_inner == 1 {
            let blocks = coeffs
                .chunks(num_positions_per_block)
                .take(num_live_blocks)
                .collect::<Vec<_>>();
            let known_balanced = known_balanced_log_basis
                .is_some_and(|source_log_basis| log_basis_inner >= source_log_basis);
            if known_balanced || digit_blocks_are_balanced(&blocks, row_width, log_basis_inner) {
                prepared.with_shared_ntt::<D, _>(
                    NttCacheKey::from_matrix_shape(
                        D,
                        n_rows,
                        row_width,
                        NttTransformDomain::Negacyclic,
                    )?,
                    |ntt| {
                        mat_vec_mul_ntt_digits_i8(ntt, n_rows, row_width, &blocks, log_basis_inner)
                    },
                )
            } else {
                prepared.with_shared_ntt::<D, _>(
                    NttCacheKey::from_matrix_shape(
                        D,
                        n_rows,
                        row_width,
                        NttTransformDomain::Negacyclic,
                    )?,
                    |ntt| mat_vec_mul_ntt_raw_digits_i8(ntt, n_rows, row_width, &blocks),
                )
            }
        } else {
            let ring_elems = coeffs
                .iter()
                .map(|digit| CyclotomicRing::from_coefficients(from_fn(|k| F::from_i8(digit[k]))))
                .collect::<Vec<_>>();
            let blocks = ring_elems
                .chunks(num_positions_per_block)
                .take(num_live_blocks)
                .collect::<Vec<_>>();
            prepared.with_shared_ntt::<D, _>(
                NttCacheKey::from_matrix_shape(
                    D,
                    n_rows,
                    row_width,
                    NttTransformDomain::Negacyclic,
                )?,
                |ntt| {
                    mat_vec_mul_ntt_i8(
                        ntt,
                        n_rows,
                        row_width,
                        &blocks,
                        num_digits_inner,
                        log_basis_inner,
                    )
                },
            )
        }
    }
}
