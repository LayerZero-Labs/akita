use super::exact_i16::{
    dense_commit_cached_digit_rows as dense_commit_cached_digit_rows_i16,
    dense_commit_rows as dense_commit_rows_i16,
    recursive_packed_witness_commit_rows as recursive_packed_witness_commit_rows_i16,
};
use super::{CpuBackend, CpuPreparedSetup};
use crate::backend::packed_digits::PackedSignedDigitView;
use crate::backend::{commit_onehot_sources, OneHotSource};
use crate::commitment::{
    CommitSourceDescriptor, DenseCoefficientSource, DenseRepresentation, DenseType,
    OneHotIndexWidth, PolynomialRepresentation, PolynomialType, PredecomposedDigitPlanes,
    ResolvedCommitSource, UnitPositionSlice,
};
use crate::compute::plans::DenseCommitInput;
use crate::compute::CommitInnerPlan;
use crate::kernels::linear::{
    mat_vec_mul_ntt_dense_digits_i8, mat_vec_mul_ntt_i8, mat_vec_mul_ntt_i8_dense,
    mat_vec_mul_ntt_i8_dense_single_row, mat_vec_mul_ntt_packed_digits_i8,
    mat_vec_mul_ntt_packed_raw_i8,
};
use crate::validation::signed_digit_kernel_for_setup;
use crate::CommitInnerWitness;
use akita_algebra::CyclotomicRing;
use akita_error::AkitaError;
use akita_types::{
    balanced_signed_digit_abs_bound, dense_i8_commit_prefers_exact_ifma52, field_modulus,
    NttCacheKey, NttTransformDomain, SignedDigitKernel,
};
use jolt_field::solinas::parallel::*;
use jolt_field::{CanonicalEncoding, Field, Unreduced, WithCommitAccumulator};
use std::array::from_fn;

impl CpuBackend {
    /// Execute the standard CPU inner stage over request-compiled sources.
    ///
    /// Every source has the same request-compiled representation. Results
    /// remain in source order.
    pub fn commit_resolved_inner_host<F, const D: usize>(
        &self,
        prepared: &CpuPreparedSetup<F>,
        sources: &[ResolvedCommitSource<'_, F>],
        plan: CommitInnerPlan,
    ) -> Result<Vec<CommitInnerWitness<F>>, AkitaError>
    where
        F: Field + CanonicalEncoding + Unreduced + WithCommitAccumulator,
    {
        if plan.ring_dimension != D || sources.is_empty() {
            return Err(AkitaError::InvalidInput(
                "resolved CPU inner stage has invalid ring dispatch or an empty source group"
                    .into(),
            ));
        }

        for source in sources {
            source.validate_plan(&plan)?;
            if source.selected_type() != sources[0].selected_type() {
                return Err(AkitaError::InvalidInput(
                    "CPU inner source group is not representation-homogeneous".into(),
                ));
            }
        }

        macro_rules! commit_onehot_group {
            ($variant:ident) => {{
                let group = sources
                    .iter()
                    .map(|source| {
                        let Some(PolynomialRepresentation::OneHot(onehot)) =
                            source.representation()
                        else {
                            return Err(AkitaError::InvalidInput(
                                "compiled one-hot group contains another representation".into(),
                            ));
                        };
                        let UnitPositionSlice::$variant(positions) = onehot.positions else {
                            return Err(AkitaError::InvalidInput(
                                "compiled one-hot group contains another index width".into(),
                            ));
                        };
                        Ok(OneHotSource {
                            indices: positions,
                            chunk_size: onehot.chunk_size,
                            num_vars: onehot.num_vars,
                        })
                    })
                    .collect::<Result<Vec<_>, AkitaError>>()?;
                commit_onehot_sources::<F, D, _>(self, prepared, &group, plan)
            }};
        }

        match sources[0].selected_type() {
            Some(PolynomialType::Dense(DenseType::Coefficients)) => sources
                .iter()
                .map(|source| {
                    let Some(PolynomialRepresentation::Dense(DenseRepresentation::Coefficients(
                        dense,
                    ))) = source.representation()
                    else {
                        return Err(AkitaError::InvalidInput(
                            "compiled dense group contains another representation".into(),
                        ));
                    };
                    self.dense_coefficient_commit_rows::<F, D>(
                        prepared,
                        *dense,
                        source.descriptor(),
                        plan,
                    )
                    .map(CommitInnerWitness::from_rows)
                })
                .collect(),
            Some(PolynomialType::Dense(DenseType::PredecomposedDigits)) => sources
                .iter()
                .map(|source| {
                    let Some(PolynomialRepresentation::Dense(
                        DenseRepresentation::PredecomposedDigits(planes),
                    )) = source.representation()
                    else {
                        return Err(AkitaError::InvalidInput(
                            "compiled dense-digit group contains another representation".into(),
                        ));
                    };
                    self.predecomposed_dense_commit_rows::<F, D>(prepared, planes, plan)
                        .map(CommitInnerWitness::from_rows)
                })
                .collect(),
            Some(PolynomialType::ShortNorm(_)) => sources
                .iter()
                .map(|source| {
                    let Some(PolynomialRepresentation::ShortNorm(short)) = source.representation()
                    else {
                        return Err(AkitaError::InvalidInput(
                            "compiled short-norm group contains another representation".into(),
                        ));
                    };
                    let digits = short.packed_view.ok_or_else(|| {
                        AkitaError::InvalidInput(
                            "CPU packed inner stage requires a validated zero-copy packed view"
                                .into(),
                        )
                    })?;
                    self.recursive_packed_witness_commit_rows::<F, D>(
                        prepared,
                        digits,
                        plan.n_a,
                        plan.num_positions_per_block,
                        plan.num_live_blocks,
                        plan.num_digits_inner,
                        plan.log_basis_inner,
                    )
                    .map(CommitInnerWitness::from_rows)
                })
                .collect(),
            Some(PolynomialType::OneHot(kind)) => match kind.index_width() {
                OneHotIndexWidth::U8 => commit_onehot_group!(U8),
                OneHotIndexWidth::U16 => commit_onehot_group!(U16),
                OneHotIndexWidth::U32 => commit_onehot_group!(U32),
                OneHotIndexWidth::Usize => commit_onehot_group!(Usize),
            },
            None => Err(AkitaError::InvalidInput(
                "CPU standard inner stage cannot execute an external source path".into(),
            )),
        }
    }

    fn predecomposed_dense_commit_rows<F, const D: usize>(
        &self,
        prepared: &CpuPreparedSetup<F>,
        planes: &PredecomposedDigitPlanes<'_>,
        plan: CommitInnerPlan,
    ) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError>
    where
        F: Field + CanonicalEncoding,
    {
        if planes.ring_dimension != D
            || planes.num_digits != plan.num_digits_inner
            || planes.log_basis != plan.log_basis_inner
        {
            return Err(AkitaError::InvalidInput(
                "predecomposed dense planes disagree with the inner stage plan".into(),
            ));
        }
        let (digit_planes, remainder) = planes.bytes.as_chunks::<D>();
        if !remainder.is_empty() {
            return Err(AkitaError::InvalidSize {
                expected: D,
                actual: remainder.len(),
            });
        }
        let blocks = dense_digit_block_slices(
            digit_planes,
            planes.logical_ring_count,
            plan.num_positions_per_block,
            plan.num_digits_inner,
        );
        if blocks.len() != plan.num_live_blocks {
            return Err(AkitaError::InvalidSetup(
                "predecomposed dense block count disagrees with the inner stage plan".into(),
            ));
        }
        self.dense_commit_rows(
            prepared,
            plan.n_a,
            DenseCommitInput::CachedDigits {
                digit_block_slices: blocks,
                log_basis_inner: plan.log_basis_inner,
            },
        )
    }

    pub(crate) fn dense_coefficient_commit_rows<F, const D: usize>(
        &self,
        prepared: &CpuPreparedSetup<F>,
        source: &dyn DenseCoefficientSource<F>,
        descriptor: &CommitSourceDescriptor,
        plan: CommitInnerPlan,
    ) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError>
    where
        F: Field + CanonicalEncoding,
    {
        if plan.ring_dimension != D {
            return Err(AkitaError::InvalidSetup(
                "dense commitment plan disagrees with ring dispatch".into(),
            ));
        }
        let logical_len = akita_error::checked::pow2(descriptor.num_vars())
            .ok_or_else(|| AkitaError::InvalidInput("dense logical extent overflow".into()))?;
        let ring_count = akita_error::checked::div_ceil(logical_len, D)
            .ok_or_else(|| AkitaError::InvalidInput("dense ring extent is invalid".into()))?;
        let physical_len = akita_error::checked::product([ring_count, D])
            .ok_or_else(|| AkitaError::InvalidInput("dense physical extent overflow".into()))?;
        let coefficients =
            source
                .coefficients()
                .get(..physical_len)
                .ok_or_else(|| AkitaError::InvalidSize {
                    expected: physical_len,
                    actual: source.coefficients().len(),
                })?;
        // SAFETY: `CyclotomicRing<F, D>` is transparent over `[F; D]`, and the
        // checked source prefix contains an integral number of D-sized rings.
        let rings = unsafe {
            std::slice::from_raw_parts(
                coefficients.as_ptr() as *const CyclotomicRing<F, D>,
                ring_count,
            )
        };
        let block_slices = dense_coefficient_block_slices(rings, plan.num_positions_per_block);
        if block_slices.len() != plan.num_live_blocks {
            return Err(AkitaError::InvalidSetup(format!(
                "dense source produced {} live blocks, expected {}",
                block_slices.len(),
                plan.num_live_blocks
            )));
        }
        self.dense_commit_rows(
            prepared,
            plan.n_a,
            DenseCommitInput::CoeffBlocks {
                block_slices,
                num_digits_inner: plan.num_digits_inner,
                log_basis_inner: plan.log_basis_inner,
            },
        )
    }

    pub(crate) fn dense_commit_rows<F, const D: usize>(
        &self,
        prepared: &CpuPreparedSetup<F>,
        n_a: usize,
        input: DenseCommitInput<'_, F, D>,
    ) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError>
    where
        F: Field + CanonicalEncoding,
    {
        match input {
            DenseCommitInput::CachedDigits {
                digit_block_slices,
                log_basis_inner,
            } => {
                let row_width = digit_block_slices.first().map_or(0, |digits| digits.len());
                if dense_i8_exact_ifma52_preferred::<F, D>(row_width, log_basis_inner)? {
                    return dense_commit_cached_digit_rows_i16(
                        prepared,
                        n_a,
                        row_width,
                        &digit_block_slices,
                        log_basis_inner,
                    );
                }
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
        F: Field + CanonicalEncoding,
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

pub(crate) fn dense_coefficient_block_slices<const D: usize, F: Field>(
    rings: &[CyclotomicRing<F, D>],
    num_positions_per_block: usize,
) -> Vec<&[CyclotomicRing<F, D>]> {
    rings.chunks(num_positions_per_block).collect()
}

pub(crate) fn dense_digit_block_slices<const D: usize>(
    digit_planes: &[[i8; D]],
    num_rings: usize,
    num_positions_per_block: usize,
    num_digits: usize,
) -> Vec<&[[i8; D]]> {
    let num_live_blocks = num_rings.div_ceil(num_positions_per_block);
    (0..num_live_blocks)
        .map(|block_idx| {
            let ring_start = block_idx * num_positions_per_block;
            let ring_end = (ring_start + num_positions_per_block).min(num_rings);
            let digit_start = ring_start * num_digits;
            let digit_end = ring_end * num_digits;
            &digit_planes[digit_start..digit_end]
        })
        .collect()
}

fn dense_i8_exact_ifma52_preferred<F: Field + CanonicalEncoding, const D: usize>(
    row_width: usize,
    log_basis: u32,
) -> Result<bool, AkitaError> {
    let rhs_abs_bound = balanced_signed_digit_abs_bound(log_basis)
        .ok_or_else(|| AkitaError::InvalidSetup("invalid signed digit basis".into()))?;
    Ok(dense_i8_commit_prefers_exact_ifma52(
        field_modulus::<F>()?,
        D,
        row_width,
        rhs_abs_bound,
    ))
}
