use super::exact_i16::{
    dense_commit_byte_digit_rows as dense_commit_byte_digit_rows_i16,
    dense_commit_packed_digit_rows as dense_commit_packed_digit_rows_i16,
    dense_commit_rows as dense_commit_rows_i16,
    recursive_packed_witness_commit_rows as recursive_packed_witness_commit_rows_i16,
};
use super::{CpuBackend, CpuPreparedSetup};
use crate::backend::packed_digits::PackedSignedDigitView;
use crate::compute::plans::DenseCommitInput;
use crate::kernels::linear::{
    mat_vec_mul_ntt_dense_digits_i8, mat_vec_mul_ntt_i8, mat_vec_mul_ntt_i8_dense,
    mat_vec_mul_ntt_i8_dense_single_row, mat_vec_mul_ntt_packed_dense_digits_i8,
    mat_vec_mul_ntt_packed_digits_i8, mat_vec_mul_ntt_packed_raw_i8,
};
use crate::validation::signed_digit_kernel_for_setup;
use akita_algebra::CyclotomicRing;
use akita_error::AkitaError;
use akita_field::parallel::*;
use akita_field::{CanonicalField, FieldCore};
use akita_types::{
    balanced_signed_digit_abs_bound, dense_i8_commit_prefers_exact_ifma52, field_modulus,
    NttCacheKey, NttTransformDomain, SignedDigitKernel,
};
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
            DenseCommitInput::PackedDigits {
                source,
                log_basis_inner,
            } => {
                let row_width = source.row_width()?;
                let num_live_blocks = source.num_live_blocks();
                if let Some(blocks) = source.borrowed_blocks::<D>()? {
                    if dense_i8_exact_ifma52_preferred::<F, D>(row_width, log_basis_inner)? {
                        return dense_commit_byte_digit_rows_i16(
                            prepared,
                            n_a,
                            row_width,
                            &blocks,
                            log_basis_inner,
                        );
                    }
                    return prepared.with_shared_ntt::<D, _>(
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
                                &blocks,
                                log_basis_inner,
                            )
                        },
                    );
                }
                if dense_i8_exact_ifma52_preferred::<F, D>(row_width, log_basis_inner)? {
                    return dense_commit_packed_digit_rows_i16(
                        prepared,
                        n_a,
                        source,
                        log_basis_inner,
                    );
                }
                let decode_block = |block_index: usize| source.decode_block::<D>(block_index);
                prepared.with_shared_ntt::<D, _>(
                    NttCacheKey::from_matrix_shape(
                        D,
                        n_a,
                        row_width,
                        NttTransformDomain::Negacyclic,
                    )?,
                    |ntt| {
                        mat_vec_mul_ntt_packed_dense_digits_i8(
                            ntt,
                            n_a,
                            row_width,
                            num_live_blocks,
                            &decode_block,
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
                let signed_digit_kernel =
                    signed_digit_kernel_for_setup(log_basis_inner, "for dense commitment")?;
                let use_exact_ifma52 = signed_digit_kernel == SignedDigitKernel::I8
                    && dense_i8_exact_ifma52_preferred::<F, D>(row_width, log_basis_inner)?;
                if signed_digit_kernel == SignedDigitKernel::I16 || use_exact_ifma52 {
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
    pub(crate) fn recursive_packed_witness_commit_rows<F, const D: usize>(
        &self,
        prepared: &CpuPreparedSetup<F>,
        digits: PackedSignedDigitView<'_>,
        n_rows: usize,
        num_positions_per_block: usize,
        num_live_blocks: usize,
        num_digits_inner: usize,
        log_basis_inner: u32,
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
        let ring_elems = digits.len() / D;
        if num_live_blocks == 0 || ring_elems < minimum_ring_elems {
            return Err(AkitaError::InvalidSetup(
                "recursive witness does not cover its live blocks".into(),
            ));
        }
        if signed_digit_kernel_for_setup(log_basis_inner, "for recursive witness commitment")?
            == SignedDigitKernel::I16
        {
            return recursive_packed_witness_commit_rows_i16(
                prepared,
                digits,
                n_rows,
                num_positions_per_block,
                num_live_blocks,
                num_digits_inner,
                log_basis_inner,
            );
        }
        if num_digits_inner == 1 {
            let decode_block = |block_index: usize| {
                let start_ring = block_index * num_positions_per_block;
                let live = (ring_elems - start_ring).min(num_positions_per_block);
                digits.decode_rings::<D>(start_ring, live)
            };
            let bounds = digits.bounds();
            let stored_is_balanced = bounds.fits_balanced_log_basis(log_basis_inner);
            if stored_is_balanced {
                prepared.with_shared_ntt::<D, _>(
                    NttCacheKey::from_matrix_shape(
                        D,
                        n_rows,
                        row_width,
                        NttTransformDomain::Negacyclic,
                    )?,
                    |ntt| {
                        mat_vec_mul_ntt_packed_digits_i8(
                            ntt,
                            n_rows,
                            row_width,
                            num_live_blocks,
                            &decode_block,
                            log_basis_inner,
                        )
                    },
                )
            } else {
                let rhs_bound = u64::from(bounds.negative_abs_max().max(bounds.positive_max()));
                prepared.with_shared_ntt::<D, _>(
                    NttCacheKey::from_matrix_shape(
                        D,
                        n_rows,
                        row_width,
                        NttTransformDomain::Negacyclic,
                    )?,
                    |ntt| {
                        mat_vec_mul_ntt_packed_raw_i8(
                            ntt,
                            n_rows,
                            row_width,
                            num_live_blocks,
                            rhs_bound,
                            &decode_block,
                        )
                    },
                )
            }
        } else {
            prepared.with_shared_ntt::<D, _>(
                NttCacheKey::from_matrix_shape(
                    D,
                    n_rows,
                    row_width,
                    NttTransformDomain::Negacyclic,
                )?,
                |ntt| {
                    cfg_into_iter!(0..num_live_blocks)
                        .map(|block_index| {
                            let start_ring = block_index * num_positions_per_block;
                            let live = (ring_elems - start_ring).min(num_positions_per_block);
                            let decoded = digits.decode_rings::<D>(start_ring, live)?;
                            let block = decoded
                                .iter()
                                .map(|digit| {
                                    CyclotomicRing::from_coefficients(from_fn(|k| {
                                        F::from_i8(digit[k])
                                    }))
                                })
                                .collect::<Vec<_>>();
                            let mut rows = mat_vec_mul_ntt_i8(
                                ntt,
                                n_rows,
                                row_width,
                                &[block.as_slice()],
                                num_digits_inner,
                                log_basis_inner,
                            )?;
                            rows.pop().ok_or(AkitaError::InvalidProof)
                        })
                        .collect()
                },
            )
        }
    }
}

fn dense_i8_exact_ifma52_preferred<F: CanonicalField, const D: usize>(
    row_width: usize,
    log_basis: u32,
) -> Result<bool, AkitaError> {
    let rhs_abs_bound = balanced_signed_digit_abs_bound(log_basis)
        .ok_or_else(|| AkitaError::InvalidSetup("invalid signed digit basis".into()))?;
    Ok(dense_i8_commit_prefers_exact_ifma52(
        field_modulus::<F>(),
        D,
        row_width,
        rhs_abs_bound,
    ))
}
