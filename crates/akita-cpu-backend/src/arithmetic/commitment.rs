use super::exact_i16::commit_rows_i16;
use super::{CpuBackend, CpuPreparedSetup};
use crate::commitment::{
    AcceptedReach, CommitSourceDescriptor, DenseCoefficientSource, DenseRepresentation, DenseType,
    OneHotIndexWidth, PolynomialRepresentation, PolynomialType, ResolvedCommitSource, SignedBlocks,
    SignedValues, UnitPositionSlice,
};
use crate::kernels::linear::{mat_vec_mul_ntt_blocks_i8, I8CommitPolicy};
use crate::opaque::CommitInnerPlan;
use crate::opaque::{commit_onehot_sources, OneHotSource};
use crate::sources::block_digits::{
    DecomposedPackedDigits, DigitShape, I16Rows, I8Rows, LiveRings, PackedDigits,
};
use crate::sources::packed_digits::PackedSignedDigitView;
use crate::sources::poly_helpers::SignedDigitBasis;
use crate::sources::signed_blocks::{Decoded, I16Slice, I8Slice, SignedBlockDigits};
use crate::validation::signed_digit_kernel_for_setup;
use akita_algebra::ring::cyclotomic::BalancedDecomposePow2Params;
use akita_algebra::CyclotomicRing;
use akita_error::AkitaError;
use akita_params::{balanced_signed_digit_abs_bound, field_modulus, SignedDigitKernel};
use akita_types::RingVec;
use akita_types::{dense_i8_commit_prefers_exact_ifma52, NttCacheKey, NttTransformDomain};
use jolt_field::{CanonicalEncoding, Field, Unreduced, WithCommitAccumulator};
use std::ops::Range;

impl<F: Field, E> CpuBackend<F, E> {
    /// Execute the standard CPU inner stage over request-compiled sources.
    ///
    /// Every source has the same request-compiled representation. Results
    /// remain in source order.
    pub(crate) fn commit_resolved_inner_host<const D: usize>(
        &self,
        prepared: &CpuPreparedSetup<F>,
        sources: &[ResolvedCommitSource<'_, F>],
        plan: CommitInnerPlan,
    ) -> Result<Vec<RingVec<F>>, AkitaError>
    where
        F: Field + CanonicalEncoding + Unreduced + WithCommitAccumulator,
    {
        if plan.ring_dimension != D || sources.is_empty() {
            return Err(AkitaError::Internal(
                "resolved CPU inner stage has invalid ring dispatch or an empty source group"
                    .into(),
            ));
        }

        for source in sources {
            source.validate_plan(&plan)?;
            if source.selected_type() != sources[0].selected_type() {
                return Err(AkitaError::Internal(
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
                            return Err(AkitaError::Internal(
                                "compiled one-hot group contains another representation".into(),
                            ));
                        };
                        let UnitPositionSlice::$variant(positions) = onehot.positions else {
                            return Err(AkitaError::Internal(
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
                commit_onehot_sources::<F, D, _, E>(self, prepared, &group, plan)
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
                        return Err(AkitaError::Internal(
                            "compiled dense group contains another representation".into(),
                        ));
                    };
                    crate::typed_inner_rows(self.dense_coefficient_commit_rows::<D>(
                        prepared,
                        *dense,
                        source.descriptor(),
                        plan,
                    )?)
                })
                .collect(),
            Some(PolynomialType::Dense(DenseType::SignedBlocks)) => sources
                .iter()
                .map(|source| {
                    let Some(PolynomialRepresentation::Dense(DenseRepresentation::SignedBlocks(
                        blocks,
                    ))) = source.representation()
                    else {
                        return Err(AkitaError::Internal(
                            "compiled signed-block group contains another representation".into(),
                        ));
                    };
                    let accepted = source.accepted.ok_or_else(|| {
                        AkitaError::InvalidInput(
                            "signed blocks require accepted coefficient bounds".into(),
                        )
                    })?;
                    crate::typed_inner_rows(
                        self.signed_block_commit_rows::<D>(prepared, *blocks, accepted, plan)?,
                    )
                })
                .collect(),
            Some(PolynomialType::ShortNorm(_)) => sources
                .iter()
                .map(|source| {
                    let Some(PolynomialRepresentation::ShortNorm(short)) = source.representation()
                    else {
                        return Err(AkitaError::Internal(
                            "compiled short-norm group contains another representation".into(),
                        ));
                    };
                    let digits = short.packed_view();
                    crate::typed_inner_rows(self.recursive_packed_witness_commit_rows::<D>(
                        prepared,
                        digits,
                        plan.n_a,
                        plan.num_positions_per_block,
                        plan.num_live_blocks,
                        plan.num_digits_inner,
                        plan.log_basis_inner,
                    )?)
                })
                .collect(),
            Some(PolynomialType::OneHot(kind)) => match kind.index_width() {
                OneHotIndexWidth::U8 => commit_onehot_group!(U8),
                OneHotIndexWidth::U16 => commit_onehot_group!(U16),
                OneHotIndexWidth::U32 => commit_onehot_group!(U32),
                OneHotIndexWidth::Usize => commit_onehot_group!(Usize),
            },
            None => Err(AkitaError::Internal(
                "CPU standard inner stage cannot execute an external source path".into(),
            )),
        }
    }

    pub(crate) fn dense_coefficient_commit_rows<const D: usize>(
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
            return Err(AkitaError::Internal(
                "dense commitment plan disagrees with ring dispatch".into(),
            ));
        }
        let logical_len = akita_error::checked::pow2(descriptor.num_vars())
            .ok_or_else(|| AkitaError::InvalidInput("dense logical extent overflow".into()))?;
        let ring_count = akita_error::checked::div_ceil(logical_len, D)
            .ok_or_else(|| AkitaError::InvalidInput("dense ring extent is invalid".into()))?;
        let nonzero_rings = source.nonzero_coefficient_len().div_ceil(D).min(ring_count);
        let physical_len = akita_error::checked::product([nonzero_rings, D])
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
                nonzero_rings,
            )
        };
        let block_ranges = (0..ring_count)
            .step_by(plan.num_positions_per_block)
            .map(|start| {
                start.min(nonzero_rings)..(start + plan.num_positions_per_block).min(nonzero_rings)
            })
            .collect::<Vec<_>>();
        if block_ranges.len() != plan.num_live_blocks {
            return Err(AkitaError::Internal(format!(
                "dense source produced {} live blocks, expected {}",
                block_ranges.len(),
                plan.num_live_blocks
            )));
        }
        let shape = DigitShape {
            num_digits: plan.num_digits_inner,
            log_basis: plan.log_basis_inner,
        };
        let row_width = ring_count
            .min(plan.num_positions_per_block)
            .checked_mul(shape.num_digits)
            .ok_or_else(|| {
                AkitaError::InvalidSetup("dense coefficient row width overflow".into())
            })?;
        let signed_digit_kernel =
            signed_digit_kernel_for_setup(shape.log_basis, "for dense commitment")?;
        let params = BalancedDecomposePow2Params::new(shape.num_digits, shape.log_basis);
        let source = LiveRings {
            coeffs: &rings[..nonzero_rings],
            params: &params,
            basis: SignedDigitBasis::new(shape.log_basis)?,
        };
        self.commit_rows(
            prepared,
            plan.n_a,
            row_width,
            &source,
            &block_ranges,
            shape,
            signed_digit_kernel,
            false,
            I8CommitPolicy::Dense,
        )
    }

    pub(crate) fn signed_block_commit_rows<const D: usize>(
        &self,
        prepared: &CpuPreparedSetup<F>,
        blocks: SignedBlocks<'_>,
        accepted: AcceptedReach,
        plan: CommitInnerPlan,
    ) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError>
    where
        F: Field + CanonicalEncoding,
    {
        if plan.ring_dimension != D || plan.num_positions_per_block == 0 {
            return Err(AkitaError::InvalidInput(
                "signed-block commitment geometry is invalid".into(),
            ));
        }
        let ring_count = akita_error::checked::pow2(blocks.num_vars())
            .and_then(|len| akita_error::checked::div_ceil(len, D))
            .ok_or_else(|| {
                AkitaError::InvalidInput("signed-block ring extent is invalid".into())
            })?;
        let block_ranges = (0..ring_count)
            .step_by(plan.num_positions_per_block)
            .map(|start| start..(start + plan.num_positions_per_block).min(ring_count))
            .collect::<Vec<_>>();
        if block_ranges.len() != plan.num_live_blocks {
            return Err(AkitaError::InvalidInput(
                "signed-block live block count disagrees with plan".into(),
            ));
        }
        let shape = DigitShape {
            num_digits: plan.num_digits_inner,
            log_basis: plan.log_basis_inner,
        };
        let row_width = akita_error::checked::product([
            ring_count.min(plan.num_positions_per_block),
            shape.num_digits,
        ])
        .ok_or_else(|| AkitaError::InvalidSetup("signed-block row width overflow".into()))?;
        let kernel = signed_digit_kernel_for_setup(shape.log_basis, "for dense commitment")?;
        macro_rules! commit {
            ($storage:expr) => {{
                let source = SignedBlockDigits::<F, _, D>::new(
                    $storage,
                    blocks.num_vars(),
                    blocks.len(),
                    Some(accepted),
                    Some(blocks.extrema()),
                )?;
                self.commit_rows(
                    prepared,
                    plan.n_a,
                    row_width,
                    &source,
                    &block_ranges,
                    shape,
                    kernel,
                    false,
                    I8CommitPolicy::Dense,
                )
            }};
        }
        match blocks.values {
            SignedValues::I8(values) => commit!(I8Slice(values)),
            SignedValues::I16(values) => commit!(I16Slice(values)),
            SignedValues::DecodedI8(decoder) => commit!(Decoded(decoder)),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn commit_rows<S, const D: usize>(
        &self,
        prepared: &CpuPreparedSetup<F>,
        n_a: usize,
        row_width: usize,
        source: &S,
        block_ranges: &[Range<usize>],
        shape: DigitShape,
        signed_digit_kernel: SignedDigitKernel,
        check_zero: bool,
        policy: I8CommitPolicy,
    ) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError>
    where
        F: Field + CanonicalEncoding,
        S: I8Rows<D> + I16Rows<D>,
    {
        let use_exact_ifma52 = matches!(policy, I8CommitPolicy::Dense)
            && signed_digit_kernel == SignedDigitKernel::I8
            && dense_i8_exact_ifma52_preferred::<F, D>(row_width, shape.log_basis)?;
        if signed_digit_kernel == SignedDigitKernel::I16 || use_exact_ifma52 {
            commit_rows_i16(prepared, n_a, row_width, source, block_ranges, shape)
        } else {
            prepared.with_shared_ntt::<D, _>(
                NttCacheKey::from_matrix_shape(D, n_a, row_width, NttTransformDomain::Negacyclic)?,
                |ntt| {
                    mat_vec_mul_ntt_blocks_i8::<F, S, D>(
                        ntt,
                        n_a,
                        row_width,
                        source,
                        block_ranges,
                        shape,
                        check_zero,
                        policy,
                    )
                },
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn recursive_packed_witness_commit_rows<const D: usize>(
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
        let signed_digit_kernel =
            signed_digit_kernel_for_setup(log_basis_inner, "for recursive witness commitment")?;
        let block_ranges = (0..num_live_blocks)
            .map(|block| {
                let start = block * num_positions_per_block;
                let live = (ring_elems - start).min(num_positions_per_block);
                start..start + live
            })
            .collect::<Vec<_>>();
        let source = PackedDigits {
            digits,
            num_rings: ring_elems,
        };
        let shape = DigitShape {
            num_digits: num_digits_inner,
            log_basis: log_basis_inner,
        };
        if num_digits_inner == 1 {
            let policy = I8CommitPolicy::Recursive {
                raw: !digits.bounds().fits_balanced_log_basis(log_basis_inner),
            };
            self.commit_rows(
                prepared,
                n_rows,
                row_width,
                &source,
                &block_ranges,
                shape,
                signed_digit_kernel,
                true,
                policy,
            )
        } else {
            let source = DecomposedPackedDigits::<F> {
                packed: source,
                params: BalancedDecomposePow2Params::new(num_digits_inner, log_basis_inner),
            };
            self.commit_rows(
                prepared,
                n_rows,
                row_width,
                &source,
                &block_ranges,
                shape,
                signed_digit_kernel,
                true,
                I8CommitPolicy::Recursive { raw: false },
            )
        }
    }
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
