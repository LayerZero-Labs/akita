//! Dense polynomial: all ring coefficients materialized in memory.
//!
//! [`DensePoly`] uses standard dense algorithms — balanced-digit decomposition,
//! NTT-based matrix-vector multiply, and parallel block folds.

use crate::backend::RootTensorProjectionPoly;
use crate::protocol::extension_opening_reduction::SparseExtensionOpeningWitness;
use akita_algebra::ring::cyclotomic::{
    decompose_centering_threshold, BalancedDecomposePow2I8Params,
};
use akita_algebra::{CyclotomicRing, SplitEqEvals};
use akita_challenges::{SparseChallenge, TensorChallenges as TensorChallengeSet};
use akita_field::parallel::*;
use akita_field::{
    AkitaError, CanonicalField, ExtField, FieldCore, FromPrimitiveInt, MulBaseUnreduced,
};
use akita_types::RingSubfieldEncoding;
use akita_types::{
    embed_ring_subfield_vector, tensor_column_partials_split_fold, tensor_opening_split,
    TensorColumnSource,
};

use crate::backend::poly_helpers::{
    balanced_ring_decompose_fold_partitioned, build_decompose_fold_witness,
    decompose_ring_single_digit, sparse_mul_acc, try_small_i8_cache_from_ring_coeffs,
    DecomposeParams,
};
use crate::compute::{
    CommitInnerPlan, CommitmentComputeBackend, CpuBackend, DecomposeFoldBatchPlan,
    DecomposeFoldPlan, DenseCommitInput, DenseCommitRowsPlan, DirectRootWitnessSource,
    OpeningBatchKernel, OpeningFoldKernel, OpeningFoldOutput, OpeningFoldPlan, RootCommitKernel,
    RootCommitSource, RootOpeningSource, RootPolyShape, RootTensorSource, TensorPackedWitness,
    TensorProjectionBatchKernel, TensorProjectionKernel,
};
use crate::kernels::linear::{decompose_rows_i8_into, try_centered_i8};
use akita_types::{CleartextWitnessProof, FlatDigitBlocks, FlatRingVec};
use std::sync::OnceLock;

use crate::{CommitInnerWitness, DecomposeFoldWitness};

mod tensor_fold;

#[derive(Debug, Clone, PartialEq, Eq)]
struct DenseDigitCache<const D: usize> {
    num_digits: usize,
    log_basis: u32,
    planes: Vec<[i8; D]>,
}

/// Dense polynomial: all ring coefficients materialized in memory.
#[derive(Debug)]
pub struct DensePoly<F: FieldCore, const D: usize> {
    /// Actual multilinear variable count of the source witness.
    num_vars: usize,
    /// Ring coefficients in sequential block order.
    pub coeffs: Vec<CyclotomicRing<F, D>>,
    small_i8_coeffs: Option<Vec<[i8; D]>>,
    digit_cache: OnceLock<DenseDigitCache<D>>,
}

impl<F: FieldCore + Clone, const D: usize> Clone for DensePoly<F, D> {
    fn clone(&self) -> Self {
        Self {
            num_vars: self.num_vars,
            coeffs: self.coeffs.clone(),
            small_i8_coeffs: self.small_i8_coeffs.clone(),
            digit_cache: OnceLock::new(),
        }
    }
}

impl<F: FieldCore + PartialEq, const D: usize> PartialEq for DensePoly<F, D> {
    fn eq(&self, other: &Self) -> bool {
        self.num_vars == other.num_vars
            && self.coeffs == other.coeffs
            && self.small_i8_coeffs == other.small_i8_coeffs
    }
}

impl<F: FieldCore + Eq, const D: usize> Eq for DensePoly<F, D> {}

impl<F: FieldCore + CanonicalField, const D: usize> DensePoly<F, D> {
    /// Pack field-element evaluations into ring elements.
    ///
    /// The first `α = log₂(D)` variables become coefficient slots within each
    /// ring element; the remaining variables index ring elements.
    ///
    /// # Errors
    ///
    /// Returns an error if `D` is not a power of two or if
    /// `evals.len() != 2^num_vars`.
    pub fn from_field_evals(num_vars: usize, evals: &[F]) -> Result<Self, AkitaError> {
        if D == 0 || !D.is_power_of_two() {
            return Err(AkitaError::InvalidInput(format!(
                "ring degree D={D} is not a power of two"
            )));
        }
        let expected_len = 1usize
            .checked_shl(num_vars as u32)
            .ok_or_else(|| AkitaError::InvalidInput(format!("2^{num_vars} does not fit usize")))?;
        if evals.len() != expected_len {
            return Err(AkitaError::InvalidSize {
                expected: expected_len,
                actual: evals.len(),
            });
        }

        let outer_len = expected_len.div_ceil(D);
        let q = (-F::one()).to_canonical_u128() + 1;
        let half_q = q / 2;
        let mut coeffs = Vec::with_capacity(outer_len);
        let mut small_i8_coeffs = Vec::with_capacity(outer_len);
        let mut all_small_i8 = true;

        for i in 0..outer_len {
            let start = i * D;
            let end = ((i + 1) * D).min(expected_len);
            let slice = &evals[start..end];
            let mut ring = CyclotomicRing::<F, D>::zero();
            for (coeff_idx, coeff) in slice.iter().enumerate() {
                ring.coeffs[coeff_idx] = *coeff;
            }
            coeffs.push(ring);

            if all_small_i8 {
                let mut digits = [0i8; D];
                for (coeff_idx, coeff) in slice.iter().enumerate() {
                    if let Some(centered) = try_centered_i8(*coeff, q, half_q) {
                        digits[coeff_idx] = centered;
                    } else {
                        all_small_i8 = false;
                        break;
                    }
                }
                if all_small_i8 {
                    small_i8_coeffs.push(digits);
                }
            }
        }

        Ok(Self {
            num_vars,
            coeffs,
            small_i8_coeffs: all_small_i8.then_some(small_i8_coeffs),
            digit_cache: OnceLock::new(),
        })
    }

    /// Wrap an existing vector of ring elements.
    ///
    /// # Panics
    ///
    /// Panics if `coeffs.len() * D` overflows `usize`.
    pub fn from_ring_coeffs(coeffs: Vec<CyclotomicRing<F, D>>) -> Self {
        let small_i8_coeffs = try_small_i8_cache_from_ring_coeffs(&coeffs);
        let total = coeffs
            .len()
            .checked_mul(D)
            .expect("ring elems * D overflow");
        Self {
            num_vars: total.trailing_zeros() as usize,
            coeffs,
            small_i8_coeffs,
            digit_cache: OnceLock::new(),
        }
    }

    fn digit_planes_for(&self, num_digits: usize, log_basis: u32) -> Option<&[[i8; D]]> {
        if let Some(cache) = self.digit_cache.get() {
            return (cache.num_digits == num_digits && cache.log_basis == log_basis)
                .then_some(cache.planes.as_slice());
        }

        let q = (-F::one()).to_canonical_u128() + 1;
        let params = BalancedDecomposePow2I8Params::new(num_digits, log_basis, q);
        let mut planes = vec![[0i8; D]; self.coeffs.len() * num_digits];
        cfg_chunks_mut!(planes, num_digits)
            .zip(cfg_iter!(self.coeffs))
            .for_each(|(dst, ring)| {
                ring.balanced_decompose_pow2_i8_into_with_params(dst, &params);
            });
        let _ = self.digit_cache.set(DenseDigitCache {
            num_digits,
            log_basis,
            planes,
        });
        let cache = self.digit_cache.get()?;
        (cache.num_digits == num_digits && cache.log_basis == log_basis)
            .then_some(cache.planes.as_slice())
    }

    fn live_coeff_len(&self) -> Result<usize, AkitaError> {
        1usize.checked_shl(self.num_vars as u32).ok_or_else(|| {
            AkitaError::InvalidInput(format!("2^{} does not fit usize", self.num_vars))
        })
    }

    fn tensor_shape<E>(&self, logical_point: Option<&[E]>) -> Result<(usize, usize), AkitaError>
    where
        E: ExtField<F>,
    {
        let (split_bits, width) = tensor_opening_split::<F, E>()?;
        if split_bits > self.num_vars {
            return Err(AkitaError::InvalidInput(
                "extension-opening tensor split exceeds polynomial arity".to_string(),
            ));
        }
        if width > D || !D.is_multiple_of(width) {
            return Err(AkitaError::InvalidInput(format!(
                "extension degree {width} does not evenly pack into dense ring degree {D}"
            )));
        }
        if let Some(point) = logical_point {
            if point.len() != self.num_vars {
                return Err(AkitaError::InvalidPointDimension {
                    expected: self.num_vars,
                    actual: point.len(),
                });
            }
        }
        Ok((split_bits, width))
    }
}

/// Column source over dense ring storage: `row(tail)` is the `width`-length
/// base-field run at flat index `tail*width`. `width` divides `D` and runs are
/// `width`-aligned within a ring, so a run never crosses a ring boundary.
struct DenseColumnSource<'a, F: FieldCore, const D: usize> {
    coeffs: &'a [CyclotomicRing<F, D>],
    width: usize,
}

impl<F: FieldCore, const D: usize> TensorColumnSource<F> for DenseColumnSource<'_, F, D> {
    #[inline]
    fn row(&self, tail: usize) -> &[F] {
        let flat = tail * self.width;
        let ring_idx = flat / D;
        let coeff_idx = flat % D;
        &self.coeffs[ring_idx].coefficients()[coeff_idx..coeff_idx + self.width]
    }
}

/// Borrowed commit view over dense ring storage.
#[derive(Debug, Clone, Copy)]
pub struct DenseCommitView<'a, F: FieldCore, const D: usize> {
    poly: &'a DensePoly<F, D>,
}

/// Borrowed opening view for fold and decompose-fold kernels.
#[derive(Debug, Clone, Copy)]
pub struct DenseOpeningView<'a, F: FieldCore, const D: usize> {
    poly: &'a DensePoly<F, D>,
}

/// Same-point batch opening view over several dense polynomials.
#[derive(Debug, Clone, Copy)]
pub struct DenseOpeningBatchView<'a, F: FieldCore, const D: usize> {
    polys: &'a [&'a DensePoly<F, D>],
}

/// Borrowed tensor projection view over dense ring storage.
#[derive(Debug, Clone, Copy)]
pub struct DenseTensorView<'a, F: FieldCore, const D: usize> {
    poly: &'a DensePoly<F, D>,
}

/// Same-point batch tensor view over several dense polynomials.
#[derive(Debug, Clone, Copy)]
pub struct DenseTensorBatchView<'a, F: FieldCore, const D: usize> {
    polys: &'a [&'a DensePoly<F, D>],
}

impl<F, const D: usize> RootPolyShape<F, D> for DensePoly<F, D>
where
    F: FieldCore,
{
    fn num_ring_elems(&self) -> usize {
        self.coeffs.len()
    }

    fn num_vars(&self) -> usize {
        self.num_vars
    }
}

impl<F, const D: usize> RootCommitSource<F, D> for DensePoly<F, D>
where
    F: FieldCore,
{
    type CommitView<'a>
        = DenseCommitView<'a, F, D>
    where
        Self: 'a;

    fn commit_view(&self) -> Result<Self::CommitView<'_>, AkitaError> {
        Ok(DenseCommitView { poly: self })
    }
}

impl<F, const D: usize> RootOpeningSource<F, D> for DensePoly<F, D>
where
    F: FieldCore,
{
    type OpeningView<'a>
        = DenseOpeningView<'a, F, D>
    where
        Self: 'a;

    type OpeningBatchView<'a>
        = DenseOpeningBatchView<'a, F, D>
    where
        Self: 'a;

    fn opening_view(&self) -> Result<Self::OpeningView<'_>, AkitaError> {
        Ok(DenseOpeningView { poly: self })
    }

    fn opening_batch<'a>(polys: &'a [&'a Self]) -> Result<Self::OpeningBatchView<'a>, AkitaError> {
        Ok(DenseOpeningBatchView { polys })
    }
}

impl<F, const D: usize> RootTensorSource<F, D> for DensePoly<F, D>
where
    F: FieldCore,
{
    type TensorView<'a>
        = DenseTensorView<'a, F, D>
    where
        Self: 'a;

    type TensorBatchView<'a>
        = DenseTensorBatchView<'a, F, D>
    where
        Self: 'a;

    fn tensor_view(&self) -> Result<Self::TensorView<'_>, AkitaError> {
        Ok(DenseTensorView { poly: self })
    }

    fn tensor_batch<'a>(polys: &'a [&'a Self]) -> Result<Self::TensorBatchView<'a>, AkitaError> {
        Ok(DenseTensorBatchView { polys })
    }
}

impl<F, const D: usize> DirectRootWitnessSource<F, D> for DensePoly<F, D>
where
    F: FieldCore,
{
    fn direct_root_witness(&self) -> Result<CleartextWitnessProof<F>, AkitaError> {
        let live_len = 1usize.checked_shl(self.num_vars as u32).ok_or_else(|| {
            AkitaError::InvalidInput(format!("2^{} does not fit usize", self.num_vars))
        })?;
        let mut coeffs = Vec::with_capacity(live_len);
        let mut remaining = live_len;
        for ring in &self.coeffs {
            let take = remaining.min(D);
            coeffs.extend_from_slice(&ring.coefficients()[..take]);
            remaining -= take;
            if remaining == 0 {
                break;
            }
        }
        Ok(CleartextWitnessProof::FieldElements(
            FlatRingVec::from_coeffs(coeffs),
        ))
    }
}

impl<F, const D: usize> RootCommitKernel<DenseCommitView<'_, F, D>, F, D> for CpuBackend
where
    F: FieldCore + CanonicalField,
{
    fn commit_inner(
        &self,
        prepared: &Self::PreparedSetup<D>,
        source: DenseCommitView<'_, F, D>,
        plan: CommitInnerPlan,
    ) -> Result<FlatDigitBlocks<D>, AkitaError> {
        source.poly.commit_inner(
            self,
            prepared,
            plan.n_a,
            plan.block_len,
            plan.num_digits_commit,
            plan.num_digits_open,
            plan.log_basis,
        )
    }

    fn commit_inner_witness(
        &self,
        prepared: &Self::PreparedSetup<D>,
        source: DenseCommitView<'_, F, D>,
        plan: CommitInnerPlan,
    ) -> Result<CommitInnerWitness<F, D>, AkitaError> {
        source.poly.commit_inner_witness(
            self,
            prepared,
            plan.n_a,
            plan.block_len,
            plan.num_digits_commit,
            plan.num_digits_open,
            plan.log_basis,
        )
    }
}

impl<F, const D: usize> OpeningFoldKernel<DenseOpeningView<'_, F, D>, F, D> for CpuBackend
where
    F: FieldCore + CanonicalField,
{
    fn evaluate_and_fold(
        &self,
        _prepared: Option<&Self::PreparedSetup<D>>,
        source: DenseOpeningView<'_, F, D>,
        plan: OpeningFoldPlan<'_, F, D>,
    ) -> Result<OpeningFoldOutput<F, D>, AkitaError> {
        let (eval, folded) = match plan {
            OpeningFoldPlan::Base {
                eval_outer_scalars,
                fold_scalars,
                block_len,
            } => source
                .poly
                .evaluate_and_fold(eval_outer_scalars, fold_scalars, block_len),
            OpeningFoldPlan::Ring {
                eval_outer_scalars,
                fold_scalars,
                block_len,
            } => source
                .poly
                .evaluate_and_fold_ring(eval_outer_scalars, fold_scalars, block_len),
        };
        Ok(OpeningFoldOutput { eval, folded })
    }

    fn decompose_fold(
        &self,
        _prepared: Option<&Self::PreparedSetup<D>>,
        source: DenseOpeningView<'_, F, D>,
        plan: DecomposeFoldPlan<'_>,
    ) -> Result<DecomposeFoldWitness<F, D>, AkitaError> {
        Ok(source.poly.decompose_fold(
            plan.challenges,
            plan.block_len,
            plan.num_digits,
            plan.log_basis,
        ))
    }
}

impl<F, const D: usize> OpeningBatchKernel<DenseOpeningBatchView<'_, F, D>, F, D> for CpuBackend
where
    F: FieldCore + CanonicalField,
{
    fn decompose_fold_batch(
        &self,
        _prepared: Option<&Self::PreparedSetup<D>>,
        source: DenseOpeningBatchView<'_, F, D>,
        plan: DecomposeFoldBatchPlan<'_>,
    ) -> Result<Option<DecomposeFoldWitness<F, D>>, AkitaError> {
        match plan {
            DecomposeFoldBatchPlan::Sparse {
                challenges,
                block_len,
                num_digits,
                log_basis,
            } => Ok(DensePoly::decompose_fold_batched(
                source.polys,
                challenges,
                block_len,
                num_digits,
                log_basis,
            )),
            DecomposeFoldBatchPlan::Tensor {
                tensor,
                block_len,
                num_digits,
                log_basis,
            } => DensePoly::decompose_fold_tensor_batched(
                source.polys,
                tensor,
                block_len,
                num_digits,
                log_basis,
            ),
        }
    }
}

impl<F, E, const D: usize> TensorProjectionKernel<DenseTensorView<'_, F, D>, F, E, D> for CpuBackend
where
    F: FieldCore + CanonicalField + FromPrimitiveInt,
    E: ExtField<F>,
{
    fn column_partials(
        &self,
        _prepared: Option<&Self::PreparedSetup<D>>,
        source: DenseTensorView<'_, F, D>,
        logical_point: &[E],
    ) -> Result<Vec<E>, AkitaError>
    where
        E: MulBaseUnreduced<F>,
    {
        source.poly.tensor_extension_column_partials(logical_point)
    }

    fn packed_witness(
        &self,
        _prepared: Option<&Self::PreparedSetup<D>>,
        source: DenseTensorView<'_, F, D>,
    ) -> Result<TensorPackedWitness<E>, AkitaError> {
        Ok(TensorPackedWitness::Dense(
            source.poly.tensor_packed_extension_evals()?,
        ))
    }

    fn root_projection(
        &self,
        _prepared: Option<&Self::PreparedSetup<D>>,
        source: DenseTensorView<'_, F, D>,
    ) -> Result<RootTensorProjectionPoly<F, D>, AkitaError>
    where
        E: RingSubfieldEncoding<F>,
    {
        source.poly.tensor_packed_extension_root_poly::<E>()
    }
}

impl<F, E, const D: usize> TensorProjectionBatchKernel<DenseTensorBatchView<'_, F, D>, F, E, D>
    for CpuBackend
where
    F: FieldCore + CanonicalField,
    E: ExtField<F>,
{
    fn column_partials_batch(
        &self,
        _prepared: Option<&Self::PreparedSetup<D>>,
        source: DenseTensorBatchView<'_, F, D>,
        logical_point: &[E],
    ) -> Result<Vec<Vec<E>>, AkitaError>
    where
        E: MulBaseUnreduced<F>,
    {
        DensePoly::tensor_extension_column_partials_batch(source.polys, logical_point)
    }

    fn sparse_linear_combination(
        &self,
        _prepared: Option<&Self::PreparedSetup<D>>,
        source: DenseTensorBatchView<'_, F, D>,
        coeffs: &[E],
    ) -> Result<Option<SparseExtensionOpeningWitness<E>>, AkitaError> {
        DensePoly::tensor_packed_extension_sparse_linear_combination(source.polys, coeffs)
    }
}

impl<F, const D: usize> DensePoly<F, D>
where
    F: FieldCore + CanonicalField,
{
    pub(crate) fn fold_blocks(&self, scalars: &[F], block_len: usize) -> Vec<CyclotomicRing<F, D>> {
        let n = self.coeffs.len();
        let num_blocks = n.div_ceil(block_len);
        cfg_into_iter!(0..num_blocks)
            .map(|i| {
                let start = i * block_len;
                let end = (start + block_len).min(n);
                let block = &self.coeffs[start..end];
                let mut acc = CyclotomicRing::<F, D>::zero();
                for (b_j, &a_j) in block.iter().zip(scalars.iter()) {
                    acc += b_j.scale(&a_j);
                }
                acc
            })
            .collect()
    }

    pub(crate) fn fold_blocks_ring(
        &self,
        scalars: &[CyclotomicRing<F, D>],
        block_len: usize,
    ) -> Vec<CyclotomicRing<F, D>> {
        let n = self.coeffs.len();
        let num_blocks = n.div_ceil(block_len);
        cfg_into_iter!(0..num_blocks)
            .map(|i| {
                let start = i * block_len;
                let end = (start + block_len).min(n);
                let block = &self.coeffs[start..end];
                let mut acc = CyclotomicRing::<F, D>::zero();
                for (b_j, &a_j) in block.iter().zip(scalars.iter()) {
                    b_j.mul_accumulate_sparse_rhs_into(&a_j, &mut acc);
                }
                acc
            })
            .collect()
    }

    pub(crate) fn evaluate_and_fold(
        &self,
        eval_outer_scalars: &[F],
        fold_scalars: &[F],
        block_len: usize,
    ) -> (CyclotomicRing<F, D>, Vec<CyclotomicRing<F, D>>) {
        crate::backend::poly_helpers::fused_evaluate_and_fold_base(
            self.fold_blocks(fold_scalars, block_len),
            eval_outer_scalars,
        )
    }

    pub(crate) fn evaluate_and_fold_ring(
        &self,
        eval_outer_scalars: &[CyclotomicRing<F, D>],
        fold_scalars: &[CyclotomicRing<F, D>],
        block_len: usize,
    ) -> (CyclotomicRing<F, D>, Vec<CyclotomicRing<F, D>>) {
        crate::backend::poly_helpers::fused_evaluate_and_fold_ring(
            self.fold_blocks_ring(fold_scalars, block_len),
            eval_outer_scalars,
        )
    }

    pub(crate) fn tensor_extension_column_partials<E>(&self, logical_point: &[E]) -> Result<Vec<E>, AkitaError>
    where
        E: MulBaseUnreduced<F>,
    {
        let (split_bits, width) = self.tensor_shape::<E>(Some(logical_point))?;
        let split = SplitEqEvals::new(&logical_point[split_bits..])?;
        let source = DenseColumnSource {
            coeffs: &self.coeffs,
            width,
        };
        Ok(tensor_column_partials_split_fold::<F, E, _>(
            &split, width, &source,
        ))
    }

    pub(crate) fn tensor_extension_column_partials_batch<E>(
        polys: &[&Self],
        logical_point: &[E],
    ) -> Result<Vec<Vec<E>>, AkitaError>
    where
        E: MulBaseUnreduced<F>,
    {
        let Some(first) = polys.first() else {
            return Ok(Vec::new());
        };
        let (split_bits, width) = first.tensor_shape::<E>(Some(logical_point))?;
        // The Dao-Thaler / Gruen split of the tail equality table is
        // point-dependent only, so it is built once and shared across the batch.
        let split = SplitEqEvals::new(&logical_point[split_bits..])?;
        polys
            .iter()
            .map(|poly| {
                poly.tensor_shape::<E>(Some(logical_point))?;
                let source = DenseColumnSource {
                    coeffs: &poly.coeffs,
                    width,
                };
                Ok(tensor_column_partials_split_fold::<F, E, _>(
                    &split, width, &source,
                ))
            })
            .collect()
    }

    pub(crate) fn tensor_packed_extension_evals<E>(&self) -> Result<Vec<E>, AkitaError>
    where
        E: ExtField<F>,
    {
        let (_split_bits, width) = self.tensor_shape::<E>(None)?;
        let live_len = self.live_coeff_len()?;
        let mut evals = Vec::with_capacity(live_len / width);
        let mut remaining = live_len;
        for ring in &self.coeffs {
            let take = remaining.min(D);
            for coeffs in ring.coefficients()[..take].chunks_exact(width) {
                evals.push(E::from_base_slice(coeffs));
            }
            remaining -= take;
            if remaining == 0 {
                break;
            }
        }
        Ok(evals)
    }

    pub(crate) fn tensor_packed_extension_sparse_evals<E>(
        &self,
    ) -> Result<Option<SparseExtensionOpeningWitness<E>>, AkitaError>
    where
        E: ExtField<F>,
    {
        Ok(None)
    }

    pub(crate) fn tensor_packed_extension_sparse_linear_combination<E>(
        polys: &[&Self],
        coeffs: &[E],
    ) -> Result<Option<SparseExtensionOpeningWitness<E>>, AkitaError>
    where
        E: ExtField<F>,
    {
        if polys.len() != coeffs.len() {
            return Err(AkitaError::InvalidSize {
                expected: polys.len(),
                actual: coeffs.len(),
            });
        }
        let mut witnesses = Vec::with_capacity(polys.len());
        for poly in polys {
            let Some(witness) = poly.tensor_packed_extension_sparse_evals::<E>()? else {
                return Ok(None);
            };
            witnesses.push(witness);
        }
        Ok(Some(SparseExtensionOpeningWitness::linear_combination(
            coeffs.iter().copied().zip(witnesses.iter()),
        )?))
    }

    pub(crate) fn tensor_packed_extension_poly<E>(&self) -> Result<DensePoly<F, D>, AkitaError>
    where
        F: CanonicalField + FromPrimitiveInt,
        E: RingSubfieldEncoding<F>,
    {
        let evals = self.tensor_packed_extension_evals::<E>()?;
        let packed_len = D / E::EXT_DEGREE;
        if packed_len == 0 {
            return Err(AkitaError::InvalidInput(
                "extension degree exceeds root ring dimension".to_string(),
            ));
        }
        let mut rings = Vec::with_capacity(evals.len().div_ceil(packed_len));
        for chunk in evals.chunks(packed_len) {
            let mut values = chunk.to_vec();
            values.resize(packed_len, E::zero());
            rings.push(embed_ring_subfield_vector::<F, E, D>(
                &values,
                AkitaError::InvalidInput(
                    "root transformed witness does not encode in the ring-subfield basis"
                        .to_string(),
                ),
            )?);
        }
        Ok(DensePoly::<F, D>::from_ring_coeffs(rings))
    }

    pub(crate) fn tensor_packed_extension_root_poly<E>(
        &self,
    ) -> Result<RootTensorProjectionPoly<F, D>, AkitaError>
    where
        F: CanonicalField + FromPrimitiveInt,
        E: RingSubfieldEncoding<F>,
    {
        Ok(self.tensor_packed_extension_poly::<E>()?.into())
    }

    pub(crate) fn decompose_fold_batched(
        _polys: &[&Self],
        _challenges: &[SparseChallenge],
        _block_len: usize,
        _num_digits: usize,
        _log_basis: u32,
    ) -> Option<DecomposeFoldWitness<F, D>> {
        None
    }

    #[tracing::instrument(skip_all, name = "DensePoly::decompose_fold")]
    pub(crate) fn decompose_fold(
        &self,
        challenges: &[SparseChallenge],
        block_len: usize,
        num_digits: usize,
        log_basis: u32,
    ) -> DecomposeFoldWitness<F, D> {
        let n = self.coeffs.len();
        let coeffs = &self.coeffs;

        if let Some(digit_planes) = self.digit_planes_for(num_digits, log_basis) {
            let coeff_accum = {
                let _span = tracing::info_span!("dense_cached_digit_accumulate").entered();
                accumulate_cached_digit_planes::<D>(digit_planes, challenges, block_len, num_digits)
            };
            let modulus = (-F::one()).to_canonical_u128() + 1;
            return build_decompose_fold_witness::<F, D>(coeff_accum, modulus);
        }

        let q = (-F::one()).to_canonical_u128() + 1;
        let threshold = decompose_centering_threshold(num_digits, log_basis, q);
        let params = DecomposeParams {
            threshold,
            q,
            mask: (1i128 << log_basis) - 1,
            half_b: 1i128 << (log_basis - 1),
            b_val: 1i128 << log_basis,
            log_basis,
            overflow_possible: q.saturating_sub(threshold) > i128::MAX as u128,
        };

        if num_digits == 1 {
            if let Some(small_coeffs) = &self.small_i8_coeffs {
                let coeff_accum: Vec<[i32; D]> = {
                    let _span =
                        tracing::info_span!("dense_single_digit_cached_accumulate").entered();
                    cfg_into_iter!(0..block_len)
                        .map(|elem_idx| {
                            let mut z_local = [0i32; D];

                            for (block_idx, c_i) in challenges.iter().enumerate() {
                                let global_idx = block_idx * block_len + elem_idx;
                                if global_idx >= small_coeffs.len() {
                                    continue;
                                }
                                sparse_mul_acc::<D>(&small_coeffs[global_idx], c_i, &mut z_local);
                            }

                            z_local
                        })
                        .collect()
                };

                let _span = tracing::info_span!("dense_single_digit_convert").entered();
                return build_decompose_fold_witness::<F, D>(coeff_accum, params.q);
            }

            let coeff_accum: Vec<[i32; D]> = {
                let _span = tracing::info_span!("dense_single_digit_accumulate").entered();
                cfg_into_iter!(0..block_len)
                    .map(|elem_idx| {
                        let mut z_local = [0i32; D];
                        let mut digit_plane = [0i8; D];

                        for (block_idx, c_i) in challenges.iter().enumerate() {
                            let global_idx = block_idx * block_len + elem_idx;
                            if global_idx >= n {
                                continue;
                            }
                            let ring = &coeffs[global_idx];
                            decompose_ring_single_digit::<F, D>(ring, &mut digit_plane, &params);
                            sparse_mul_acc::<D>(&digit_plane, c_i, &mut z_local);
                        }

                        z_local
                    })
                    .collect()
            };

            let _span = tracing::info_span!("dense_single_digit_convert").entered();
            return build_decompose_fold_witness::<F, D>(coeff_accum, params.q);
        }

        let centered_coeffs = {
            let _span = tracing::info_span!("dense_multi_digit_accumulate").entered();
            balanced_ring_decompose_fold_partitioned::<F, D>(
                coeffs, challenges, block_len, num_digits, &params,
            )
        };

        let _span = tracing::info_span!("dense_multi_digit_convert").entered();
        build_decompose_fold_witness::<F, D>(centered_coeffs, params.q)
    }

    #[tracing::instrument(skip_all, name = "DensePoly::decompose_fold_tensor_batched")]
    pub(crate) fn decompose_fold_tensor_batched(
        polys: &[&Self],
        tensor: &TensorChallengeSet,
        block_len: usize,
        num_digits: usize,
        log_basis: u32,
    ) -> Result<Option<DecomposeFoldWitness<F, D>>, AkitaError> {
        tensor_fold::decompose_fold_batched_tensor_dense(
            polys, tensor, block_len, num_digits, log_basis,
        )
    }

    #[allow(clippy::too_many_arguments)]
    #[tracing::instrument(skip_all, name = "DensePoly::commit_inner")]
    pub(crate) fn commit_inner<B>(
        &self,
        backend: &B,
        prepared: &B::PreparedSetup<D>,
        n_a: usize,
        block_len: usize,
        num_digits_commit: usize,
        num_digits_open: usize,
        log_basis: u32,
    ) -> Result<FlatDigitBlocks<D>, AkitaError>
    where
        B: CommitmentComputeBackend<F>,
    {
        let t = self.commit_rows(
            backend,
            prepared,
            n_a,
            block_len,
            num_digits_commit,
            log_basis,
        )?;
        decompose_commit_rows::<F, D>(&t, num_digits_open, log_basis)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn commit_inner_witness<B>(
        &self,
        backend: &B,
        prepared: &B::PreparedSetup<D>,
        n_a: usize,
        block_len: usize,
        num_digits_commit: usize,
        num_digits_open: usize,
        log_basis: u32,
    ) -> Result<CommitInnerWitness<F, D>, AkitaError>
    where
        B: CommitmentComputeBackend<F>,
    {
        let t = self.commit_rows(
            backend,
            prepared,
            n_a,
            block_len,
            num_digits_commit,
            log_basis,
        )?;
        let decomposed_inner_rows = decompose_commit_rows::<F, D>(&t, num_digits_open, log_basis)?;
        Ok(CommitInnerWitness {
            recomposed_inner_rows: t,
            decomposed_inner_rows,
        })
    }
}

impl<F, const D: usize> DensePoly<F, D>
where
    F: FieldCore + CanonicalField,
{
    fn commit_rows<B>(
        &self,
        backend: &B,
        prepared: &B::PreparedSetup<D>,
        n_a: usize,
        block_len: usize,
        num_digits_commit: usize,
        log_basis: u32,
    ) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError>
    where
        B: CommitmentComputeBackend<F>,
    {
        let n = self.coeffs.len();
        let num_blocks = n.div_ceil(block_len);

        if let Some(digit_planes) = self.digit_planes_for(num_digits_commit, log_basis) {
            let digit_block_slices =
                digit_block_slices(digit_planes, n, block_len, num_digits_commit);
            return backend.dense_commit_rows(
                prepared,
                DenseCommitRowsPlan {
                    n_a,
                    input: DenseCommitInput::CachedDigits {
                        digit_block_slices,
                        log_basis,
                    },
                },
            );
        }

        let block_slices: Vec<&[CyclotomicRing<F, D>]> = (0..num_blocks)
            .map(|i| {
                let start = i * block_len;
                if start >= n {
                    &[] as &[CyclotomicRing<F, D>]
                } else {
                    &self.coeffs[start..(start + block_len).min(n)]
                }
            })
            .collect();

        backend.dense_commit_rows(
            prepared,
            DenseCommitRowsPlan {
                n_a,
                input: DenseCommitInput::CoeffBlocks {
                    block_slices,
                    num_digits_commit,
                    log_basis,
                },
            },
        )
    }
}

fn decompose_commit_rows<F, const D: usize>(
    rows: &[Vec<CyclotomicRing<F, D>>],
    num_digits_open: usize,
    log_basis: u32,
) -> Result<FlatDigitBlocks<D>, AkitaError>
where
    F: FieldCore + CanonicalField,
{
    let block_sizes: Vec<usize> = rows.iter().map(|t_i| t_i.len() * num_digits_open).collect();
    let mut t_hat = FlatDigitBlocks::zeroed(block_sizes)?;
    let dst_blocks = t_hat.split_blocks_mut();
    #[cfg(feature = "parallel")]
    cfg_into_iter!(dst_blocks)
        .zip(cfg_iter!(rows))
        .for_each(|(dst, t_i)| decompose_rows_i8_into(t_i, dst, num_digits_open, log_basis));
    #[cfg(not(feature = "parallel"))]
    dst_blocks
        .into_iter()
        .zip(rows.iter())
        .for_each(|(dst, t_i)| decompose_rows_i8_into(t_i, dst, num_digits_open, log_basis));

    Ok(t_hat)
}

fn digit_block_slices<const D: usize>(
    digit_planes: &[[i8; D]],
    num_rings: usize,
    block_len: usize,
    num_digits: usize,
) -> Vec<&[[i8; D]]> {
    let num_blocks = num_rings.div_ceil(block_len);
    (0..num_blocks)
        .map(|block_idx| {
            let ring_start = block_idx * block_len;
            let ring_end = (ring_start + block_len).min(num_rings);
            let digit_start = ring_start * num_digits;
            let digit_end = ring_end * num_digits;
            &digit_planes[digit_start..digit_end]
        })
        .collect()
}

fn accumulate_cached_digit_planes<const D: usize>(
    digit_planes: &[[i8; D]],
    challenges: &[SparseChallenge],
    block_len: usize,
    num_digits: usize,
) -> Vec<[i32; D]> {
    let inner_width = block_len * num_digits;
    cfg_into_iter!(0..inner_width)
        .map(|inner_idx| {
            let elem_idx = inner_idx / num_digits;
            let digit_idx = inner_idx % num_digits;
            let mut acc = [0i32; D];
            for (block_idx, challenge) in challenges.iter().enumerate() {
                let ring_idx = block_idx * block_len + elem_idx;
                let plane_idx = ring_idx * num_digits + digit_idx;
                let Some(digit_plane) = digit_planes.get(plane_idx) else {
                    continue;
                };
                sparse_mul_acc::<D>(digit_plane, challenge, &mut acc);
            }
            acc
        })
        .collect()
}

/// Test-only helpers for [`DensePoly`].
///
/// These live outside the production polynomial API because they are
/// only used by cross-check tests (e.g. verifying that fused prover paths
/// match a straight-line reference implementation).
#[cfg(test)]
pub(crate) mod test_helpers {
    use super::DensePoly;
    use akita_algebra::CyclotomicRing;
    use akita_field::FieldCore;
    #[cfg(feature = "parallel")]
    use rayon::prelude::*;

    /// Reference ring-space evaluation for [`DensePoly`].
    ///
    /// Computes the global weighted sum `y = Σᵢ scalars[i] · self.coeffs[i]`.
    #[allow(dead_code)]
    pub(crate) fn evaluate_ring_dense<F, const D: usize>(
        poly: &DensePoly<F, D>,
        scalars: &[F],
    ) -> CyclotomicRing<F, D>
    where
        F: FieldCore,
    {
        #[cfg(feature = "parallel")]
        {
            poly.coeffs
                .par_iter()
                .zip(scalars.par_iter())
                .fold(
                    || CyclotomicRing::<F, D>::zero(),
                    |acc, (f_i, w_i)| acc + f_i.scale(w_i),
                )
                .reduce(|| CyclotomicRing::<F, D>::zero(), |a, b| a + b)
        }
        #[cfg(not(feature = "parallel"))]
        {
            poly.coeffs
                .iter()
                .zip(scalars.iter())
                .fold(CyclotomicRing::<F, D>::zero(), |acc, (f_i, w_i)| {
                    acc + f_i.scale(w_i)
                })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_field::Prime128OffsetA7F7 as F;
    use akita_field::{TowerBasisFpExt4, TwoNr, UnitNr};
    use akita_types::{tensor_column_partials_from_base_evals, tensor_packed_witness_evals};

    fn ring<const D: usize>(offset: u64) -> CyclotomicRing<F, D> {
        CyclotomicRing::from_coefficients(std::array::from_fn(|idx| {
            F::from_u64(offset + idx as u64 + 1)
        }))
    }

    #[test]
    fn ring_fold_matches_dense_multiplication_reference() {
        const D: usize = 8;
        let coeffs = (0..4).map(|idx| ring::<D>(10 * idx)).collect::<Vec<_>>();
        let poly = DensePoly::<F, D>::from_ring_coeffs(coeffs.clone());
        let scalars = vec![ring::<D>(100), ring::<D>(200)];
        let got = poly.fold_blocks_ring(&scalars, 2);
        let expected = coeffs
            .chunks(2)
            .map(|block| {
                block
                    .iter()
                    .zip(scalars.iter())
                    .fold(CyclotomicRing::<F, D>::zero(), |acc, (coeff, scalar)| {
                        acc + (*coeff * *scalar)
                    })
            })
            .collect::<Vec<_>>();

        assert_eq!(got, expected);
    }

    #[test]
    fn dense_kernel_tensor_paths_match_akitapolyops() {
        const D: usize = 8;
        type E = TowerBasisFpExt4<F, TwoNr, UnitNr>;

        let num_vars = 5;
        let evals = (0..(1usize << num_vars))
            .map(|idx| F::from_u64(17 * idx as u64 + 9))
            .collect::<Vec<_>>();
        let point = (0..num_vars)
            .map(|idx| {
                E::from_base_slice(&[
                    F::from_u64(idx as u64 + 2),
                    F::from_u64(3 * idx as u64 + 4),
                    F::from_u64(5 * idx as u64 + 6),
                    F::from_u64(7 * idx as u64 + 8),
                ])
            })
            .collect::<Vec<_>>();
        let poly = DensePoly::<F, D>::from_field_evals(num_vars, &evals).unwrap();
        let backend = CpuBackend;
        let tensor_view = poly.tensor_view().unwrap();

        let ops_partials = poly.tensor_extension_column_partials::<E>(&point).unwrap();
        let kernel_partials =
            TensorProjectionKernel::<DenseTensorView<'_, F, D>, F, E, D>::column_partials(
                &backend,
                None,
                tensor_view,
                &point,
            )
            .unwrap();
        assert_eq!(kernel_partials, ops_partials);

        let ops_packed = poly.tensor_packed_extension_evals::<E>().unwrap();
        let kernel_packed =
            match TensorProjectionKernel::<DenseTensorView<'_, F, D>, F, E, D>::packed_witness(
                &backend,
                None,
                tensor_view,
            )
            .unwrap()
            {
                TensorPackedWitness::Dense(v) => v,
                TensorPackedWitness::Sparse(_) => {
                    panic!("dense kernel must return dense packed witness")
                }
            };
        assert_eq!(kernel_packed, ops_packed);

        let polys = [&poly, &poly];
        let batch_view = DensePoly::<F, D>::tensor_batch(&polys).unwrap();
        let ops_batch =
            DensePoly::<F, D>::tensor_extension_column_partials_batch::<E>(&polys, &point).unwrap();
        let kernel_batch = TensorProjectionBatchKernel::<DenseTensorBatchView<'_, F, D>, F, E, D>::column_partials_batch(
            &backend,
            None,
            batch_view,
            &point,
        )
        .unwrap();
        assert_eq!(kernel_batch, ops_batch);
    }

    #[test]
    fn dense_kernel_root_projection_matches_akitapolyops() {
        use akita_field::{Prime32Offset99, RingSubfieldFpExt4};

        const D: usize = 8;
        type F = Prime32Offset99;
        type E = RingSubfieldFpExt4<F>;

        let num_vars = 4;
        let evals = (0..(1usize << num_vars))
            .map(|idx| F::from_u64(idx as u64 + 1))
            .collect::<Vec<_>>();
        let poly = DensePoly::<F, D>::from_field_evals(num_vars, &evals).unwrap();
        let backend = CpuBackend;
        let tensor_view = poly.tensor_view().unwrap();

        let ops_root =
            DensePoly::tensor_packed_extension_root_poly::<E>(&poly)
                .unwrap();
        let kernel_root =
            TensorProjectionKernel::<DenseTensorView<'_, F, D>, F, E, D>::root_projection(
                &backend,
                None,
                tensor_view,
            )
            .unwrap();
        match (kernel_root, ops_root) {
            (RootTensorProjectionPoly::Dense(got), RootTensorProjectionPoly::Dense(expected)) => {
                assert_eq!(got, expected);
            }
            _ => panic!("dense root projection must stay dense"),
        }
    }

    #[test]
    fn dense_kernel_direct_witness_matches_akitapolyops() {
        const D: usize = 8;
        let num_vars = 4;
        let evals = (0..(1usize << num_vars))
            .map(|idx| F::from_u64(idx as u64 + 1))
            .collect::<Vec<_>>();
        let poly = DensePoly::<F, D>::from_field_evals(num_vars, &evals).unwrap();
        let witness =
            <DensePoly<F, D> as DirectRootWitnessSource<F, D>>::direct_root_witness(&poly)
                .unwrap();
        assert!(matches!(witness, CleartextWitnessProof::FieldElements(_)));
    }

    #[test]
    fn dense_tensor_opening_methods_match_flat_reference() {
        const D: usize = 8;
        type E = TowerBasisFpExt4<F, TwoNr, UnitNr>;

        let num_vars = 5;
        let evals = (0..(1usize << num_vars))
            .map(|idx| F::from_u64(17 * idx as u64 + 9))
            .collect::<Vec<_>>();
        let point = (0..num_vars)
            .map(|idx| {
                E::from_base_slice(&[
                    F::from_u64(idx as u64 + 2),
                    F::from_u64(3 * idx as u64 + 4),
                    F::from_u64(5 * idx as u64 + 6),
                    F::from_u64(7 * idx as u64 + 8),
                ])
            })
            .collect::<Vec<_>>();
        let poly = DensePoly::<F, D>::from_field_evals(num_vars, &evals).unwrap();

        let expected_partials =
            tensor_column_partials_from_base_evals::<F, E>(num_vars, &evals, &point).unwrap();
        let got_partials = poly.tensor_extension_column_partials::<E>(&point).unwrap();
        assert_eq!(got_partials, expected_partials);

        let expected_packed = tensor_packed_witness_evals::<F, E>(num_vars, &evals).unwrap();
        let got_packed = poly.tensor_packed_extension_evals::<E>().unwrap();
        assert_eq!(got_packed, expected_packed);
    }
}
