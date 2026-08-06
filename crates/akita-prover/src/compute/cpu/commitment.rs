use super::exact_i16::{
    dense_commit_rows as dense_commit_rows_i16,
    recursive_witness_commit_rows as recursive_witness_commit_rows_i16,
};
use super::CpuBackend;
use crate::backend::onehot::{column_sweep_ajtai_onehot, MultiChunkEntry, SingleChunkEntry};
use crate::backend::sparse_ring::column_sweep_sparse;
use crate::compute::backend::CommitmentComputeBackend;
use crate::compute::plans::{
    DenseCommitInput, DenseCommitRowsPlan, OneHotCommitBlocks, OneHotCommitRowsPlan,
    RecursiveWitnessCommitRowsPlan, SparseRingCommitRowsPlan,
};
use crate::kernels::linear::{
    digit_blocks_are_balanced, mat_vec_mul_ntt_dense_digits_i8, mat_vec_mul_ntt_digits_i8,
    mat_vec_mul_ntt_i8, mat_vec_mul_ntt_i8_dense, mat_vec_mul_ntt_i8_dense_single_row,
    mat_vec_mul_ntt_raw_digits_i8,
};
use akita_algebra::CyclotomicRing;
use akita_field::unreduced::{HasWide, ReduceTo};
use akita_field::{AdditiveGroup, AkitaError, CanonicalField, FieldCore};
use akita_types::{NttCacheKey, NttTransformDomain};
use std::array::from_fn;

impl<F> CommitmentComputeBackend<F> for CpuBackend
where
    F: FieldCore + CanonicalField,
{
    fn dense_commit_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup,
        plan: DenseCommitRowsPlan<'_, F, D>,
    ) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError> {
        match plan.input {
            DenseCommitInput::CachedDigits {
                digit_block_slices,
                log_basis_inner,
            } => {
                let row_width = digit_block_slices.first().map_or(0, |digits| digits.len());
                prepared.with_shared_ntt::<D, _>(
                    NttCacheKey::from_matrix_shape(
                        D,
                        plan.n_a,
                        row_width,
                        NttTransformDomain::Negacyclic,
                    )?,
                    |ntt| {
                        mat_vec_mul_ntt_dense_digits_i8(
                            ntt,
                            plan.n_a,
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
                        AkitaError::InvalidSetup("dense coefficient row width overflow".to_string())
                    })
                })?;
                if log_basis_inner > 8 {
                    dense_commit_rows_i16(
                        prepared,
                        plan.n_a,
                        row_width,
                        &block_slices,
                        num_digits_inner,
                        log_basis_inner,
                    )
                } else if plan.n_a == 1 {
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
                            plan.n_a,
                            row_width,
                            NttTransformDomain::Negacyclic,
                        )?,
                        |ntt| {
                            mat_vec_mul_ntt_i8_dense(
                                ntt,
                                plan.n_a,
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

    fn onehot_commit_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup,
        plan: OneHotCommitRowsPlan<'_>,
    ) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError>
    where
        F: HasWide,
        F::Wide: AdditiveGroup + From<F> + ReduceTo<F>,
    {
        let active_a_cols = plan
            .num_positions_per_block
            .checked_mul(plan.num_digits_inner)
            .ok_or_else(|| AkitaError::InvalidSetup("active A width overflow".to_string()))?;
        let a_view = prepared
            .expanded
            .shared_matrix
            .ring_view::<D>(plan.n_a, active_a_cols)?;
        Ok(match plan.blocks {
            OneHotCommitBlocks::SingleChunk(blocks) => {
                column_sweep_ajtai_onehot::<SingleChunkEntry, F, D>(
                    &a_view,
                    &blocks.block_slices()?,
                    plan.n_a,
                    active_a_cols,
                    plan.num_digits_inner,
                )
            }
            OneHotCommitBlocks::MultiChunk(blocks) => {
                column_sweep_ajtai_onehot::<MultiChunkEntry, F, D>(
                    &a_view,
                    &blocks.block_slices()?,
                    plan.n_a,
                    active_a_cols,
                    plan.num_digits_inner,
                )
            }
        })
    }

    fn sparse_ring_commit_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup,
        plan: SparseRingCommitRowsPlan<'_>,
    ) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError>
    where
        F: HasWide,
        F::Wide: AdditiveGroup + From<F> + ReduceTo<F>,
    {
        let active_a_cols = plan
            .num_positions_per_block
            .checked_mul(plan.num_digits_inner)
            .ok_or_else(|| AkitaError::InvalidSetup("active A width overflow".to_string()))?;
        let a_view = prepared
            .expanded
            .shared_matrix
            .ring_view::<D>(plan.n_a, active_a_cols)?;
        Ok(column_sweep_sparse(
            &a_view,
            &plan.blocks.block_slices()?,
            plan.n_a,
            plan.num_positions_per_block,
            plan.num_digits_inner,
        ))
    }

    fn recursive_witness_commit_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup,
        plan: RecursiveWitnessCommitRowsPlan<'_, D>,
    ) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError> {
        let row_width = plan
            .num_positions_per_block
            .checked_mul(plan.num_digits_inner)
            .ok_or_else(|| AkitaError::InvalidSetup("recursive A width overflow".to_string()))?;
        let minimum_ring_elems = plan
            .num_live_blocks
            .saturating_sub(1)
            .checked_mul(plan.num_positions_per_block)
            .and_then(|prefix| prefix.checked_add(1))
            .ok_or_else(|| {
                AkitaError::InvalidSetup("recursive witness block extent overflow".to_string())
            })?;
        if plan.num_live_blocks == 0 || plan.coeffs.len() < minimum_ring_elems {
            return Err(AkitaError::InvalidSetup(
                "recursive witness does not cover its live blocks".to_string(),
            ));
        }
        if plan.log_basis_inner > 8 {
            return recursive_witness_commit_rows_i16(prepared, &plan, row_width);
        }
        if plan.num_digits_inner == 1 {
            let blocks = plan
                .coeffs
                .chunks(plan.num_positions_per_block)
                .take(plan.num_live_blocks)
                .collect::<Vec<_>>();
            // The `num_digits_inner == 1` recursive witness is a raw signed-i8
            // coefficient stream. Degree-one fields yield balanced gadget digits
            // (fast predecomposed-digit kernel), but extension-field tensor
            // base-lift packing sums gadget digits and can push coefficients
            // past the balanced range; those must commit through the general
            // raw ring mat-vec instead of the balanced-digit LUT kernel.
            let known_balanced = plan
                .known_balanced_log_basis
                .is_some_and(|source_log_basis| plan.log_basis_inner >= source_log_basis);
            if known_balanced || digit_blocks_are_balanced(&blocks, row_width, plan.log_basis_inner)
            {
                prepared.with_shared_ntt::<D, _>(
                    NttCacheKey::from_matrix_shape(
                        D,
                        plan.n_rows,
                        row_width,
                        NttTransformDomain::Negacyclic,
                    )?,
                    |ntt| {
                        mat_vec_mul_ntt_digits_i8(
                            ntt,
                            plan.n_rows,
                            row_width,
                            &blocks,
                            plan.log_basis_inner,
                        )
                    },
                )
            } else {
                prepared.with_shared_ntt::<D, _>(
                    NttCacheKey::from_matrix_shape(
                        D,
                        plan.n_rows,
                        row_width,
                        NttTransformDomain::Negacyclic,
                    )?,
                    |ntt| mat_vec_mul_ntt_raw_digits_i8(ntt, plan.n_rows, row_width, &blocks),
                )
            }
        } else {
            let ring_elems: Vec<CyclotomicRing<F, D>> = plan
                .coeffs
                .iter()
                .map(|digit| {
                    let coeffs = from_fn(|k| F::from_i8(digit[k]));
                    CyclotomicRing::from_coefficients(coeffs)
                })
                .collect();
            let blocks = ring_elems
                .chunks(plan.num_positions_per_block)
                .take(plan.num_live_blocks)
                .collect::<Vec<_>>();
            prepared.with_shared_ntt::<D, _>(
                NttCacheKey::from_matrix_shape(
                    D,
                    plan.n_rows,
                    row_width,
                    NttTransformDomain::Negacyclic,
                )?,
                |ntt| {
                    mat_vec_mul_ntt_i8(
                        ntt,
                        plan.n_rows,
                        row_width,
                        &blocks,
                        plan.num_digits_inner,
                        plan.log_basis_inner,
                    )
                },
            )
        }
    }
}
