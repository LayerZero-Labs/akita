//! CPU implementation of the Ajtai commit primitive.
//!
//! This impl **is** the CPU backend, so it is the privileged holder of raw
//! `CpuPreparedSetup` access (the NTT slot + the shared matrix). Representation
//! (`backend/`) and protocol code never reach through `CpuPreparedSetup`; they
//! only call `ajtai_commit` / `AjtaiOpeningView`.

use std::array::from_fn;

use akita_algebra::CyclotomicRing;
use akita_field::unreduced::{HasWide, ReduceTo};
use akita_field::{AdditiveGroup, AkitaError, CanonicalField, FieldCore};

use crate::backend::onehot::{MultiChunkEntry, SingleChunkEntry};
use crate::commit::ajtai::backend::CommitBackend;
use crate::commit::ajtai::column_sweep::{column_sweep_ajtai_onehot, column_sweep_sparse};
use crate::commit::ajtai::opening::{AjtaiOpeningType, OneHotCommitBlocks, ZeroScan};
use crate::commit::ajtai::spec::{MatrixSpec, RingDomain};
use crate::compute::{CpuBackend, CpuPreparedSetup};
use crate::kernels::linear::{
    mat_vec_mul_ntt_dense_digits_i8_trusted, mat_vec_mul_ntt_digits_i8, mat_vec_mul_ntt_i8,
    mat_vec_mul_ntt_i8_dense, mat_vec_mul_ntt_i8_dense_single_row, mat_vec_mul_ntt_i8_strided,
    mat_vec_mul_ntt_raw_i8_strided, mat_vec_mul_ntt_single_i8, mat_vec_mul_ntt_single_i8_cyclic,
};

/// Validate that the requested window fits the commitment key.
fn validate_matrix<F, const D: usize>(
    commitment_key: &CpuPreparedSetup<F, D>,
    spec: &MatrixSpec,
) -> Result<(), AkitaError>
where
    F: FieldCore,
{
    if spec.rows == 0 || spec.cols == 0 {
        return Err(AkitaError::InvalidSetup(
            "ajtai commit requires nonzero matrix rows and cols".to_string(),
        ));
    }
    let total = commitment_key
        .expanded
        .shared_matrix
        .total_ring_elements_at::<D>()?;
    let required = spec.rows.checked_mul(spec.cols).ok_or_else(|| {
        AkitaError::InvalidSetup(format!(
            "ajtai matrix footprint overflows: rows={} cols={}",
            spec.rows, spec.cols
        ))
    })?;
    if required > total {
        return Err(AkitaError::InvalidSetup(format!(
            "ajtai matrix needs {required} setup ring elements but commitment key has {total}"
        )));
    }
    Ok(())
}

/// Require that a single-block opening matches the matrix window width.
fn require_block_width(cols: usize, actual: usize) -> Result<(), AkitaError> {
    if cols != actual {
        return Err(AkitaError::InvalidSetup(format!(
            "ajtai opening width {actual} does not match matrix window cols {cols}"
        )));
    }
    Ok(())
}

impl<F> CommitBackend<F> for CpuBackend
where
    F: FieldCore + CanonicalField,
{
    fn ajtai_commit<const D: usize>(
        &self,
        commitment_key: &Self::PreparedSetup<D>,
        spec: MatrixSpec,
        opening: AjtaiOpeningType<'_, F, D>,
    ) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError>
    where
        F: HasWide,
        F::Wide: AdditiveGroup + From<F> + ReduceTo<F>,
    {
        validate_matrix::<F, D>(commitment_key, &spec)?;
        match (spec.domain, opening) {
            (RingDomain::Negacyclic, AjtaiOpeningType::DigitVector { digits, log_basis }) => {
                require_block_width(spec.cols, digits.len())?;
                Ok(vec![mat_vec_mul_ntt_single_i8(
                    &commitment_key.ntt_shared,
                    spec.rows,
                    spec.cols,
                    digits,
                    log_basis,
                )?])
            }
            (RingDomain::Cyclic, AjtaiOpeningType::DigitVector { digits, log_basis }) => {
                require_block_width(spec.cols, digits.len())?;
                Ok(vec![mat_vec_mul_ntt_single_i8_cyclic(
                    &commitment_key.ntt_shared,
                    spec.rows,
                    spec.cols,
                    digits,
                    log_basis,
                )?])
            }
            (
                RingDomain::Negacyclic,
                AjtaiOpeningType::DigitBlocks {
                    blocks,
                    log_basis,
                    zero_scan,
                },
            ) => match zero_scan {
                ZeroScan::Dense => mat_vec_mul_ntt_dense_digits_i8_trusted(
                    &commitment_key.ntt_shared,
                    spec.rows,
                    spec.cols,
                    &blocks,
                    log_basis,
                ),
                ZeroScan::SkipZeros => mat_vec_mul_ntt_digits_i8(
                    &commitment_key.ntt_shared,
                    spec.rows,
                    spec.cols,
                    &blocks,
                    log_basis,
                ),
            },
            (
                RingDomain::Negacyclic,
                AjtaiOpeningType::CoeffBlocks {
                    blocks,
                    num_digits,
                    log_basis,
                    zero_scan,
                },
            ) => match zero_scan {
                ZeroScan::Dense => {
                    if spec.rows == 1 {
                        Ok(mat_vec_mul_ntt_i8_dense_single_row(
                            &commitment_key.ntt_shared,
                            spec.cols,
                            &blocks,
                            num_digits,
                            log_basis,
                        )?
                        .into_iter()
                        .map(|ring| vec![ring])
                        .collect())
                    } else {
                        mat_vec_mul_ntt_i8_dense(
                            &commitment_key.ntt_shared,
                            spec.rows,
                            spec.cols,
                            &blocks,
                            num_digits,
                            log_basis,
                        )
                    }
                }
                ZeroScan::SkipZeros => mat_vec_mul_ntt_i8(
                    &commitment_key.ntt_shared,
                    spec.rows,
                    spec.cols,
                    &blocks,
                    num_digits,
                    log_basis,
                ),
            },
            (
                RingDomain::Negacyclic,
                AjtaiOpeningType::StridedDigits {
                    coeffs,
                    num_blocks,
                    block_len,
                    num_digits,
                    log_basis,
                    raw,
                },
            ) => {
                if raw {
                    mat_vec_mul_ntt_raw_i8_strided(
                        &commitment_key.ntt_shared,
                        spec.rows,
                        spec.cols,
                        coeffs,
                        num_blocks,
                        block_len,
                    )
                } else {
                    let ring_elems: Vec<CyclotomicRing<F, D>> = coeffs
                        .iter()
                        .map(|digit| {
                            let coeffs = from_fn(|k| F::from_i8(digit[k]));
                            CyclotomicRing::from_coefficients(coeffs)
                        })
                        .collect();
                    mat_vec_mul_ntt_i8_strided(
                        &commitment_key.ntt_shared,
                        spec.rows,
                        spec.cols,
                        &ring_elems,
                        num_blocks,
                        block_len,
                        num_digits,
                        log_basis,
                    )
                }
            }
            (
                RingDomain::Negacyclic,
                AjtaiOpeningType::OneHot {
                    blocks,
                    num_digits_commit,
                },
            ) => {
                let a_view = commitment_key
                    .expanded
                    .shared_matrix
                    .ring_view::<D>(spec.rows, spec.cols)?;
                Ok(match blocks {
                    OneHotCommitBlocks::SingleChunk(table) => {
                        column_sweep_ajtai_onehot::<SingleChunkEntry, F, D>(
                            &a_view,
                            &table.block_slices()?,
                            spec.rows,
                            spec.cols,
                            num_digits_commit,
                        )
                    }
                    OneHotCommitBlocks::MultiChunk(table) => {
                        column_sweep_ajtai_onehot::<MultiChunkEntry, F, D>(
                            &a_view,
                            &table.block_slices()?,
                            spec.rows,
                            spec.cols,
                            num_digits_commit,
                        )
                    }
                })
            }
            (
                RingDomain::Negacyclic,
                AjtaiOpeningType::SparseRing {
                    blocks,
                    num_digits_commit,
                },
            ) => {
                let block_len = spec.cols.checked_div(num_digits_commit).ok_or_else(|| {
                    AkitaError::InvalidSetup(
                        "sparse-ring commit requires nonzero num_digits_commit".to_string(),
                    )
                })?;
                let a_view = commitment_key
                    .expanded
                    .shared_matrix
                    .ring_view::<D>(spec.rows, spec.cols)?;
                let a_rows = (0..spec.rows)
                    .map(|idx| a_view.row(idx))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(column_sweep_sparse(
                    &a_rows,
                    &blocks.block_slices()?,
                    spec.rows,
                    block_len,
                    num_digits_commit,
                ))
            }
            (RingDomain::Cyclic, _) => Err(AkitaError::InvalidSetup(
                "cyclic ajtai commit only supports the DigitVector opening".to_string(),
            )),
        }
    }
}
