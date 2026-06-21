//! Source-typed views and `CpuBackend` kernels for [`super::SparseRingPoly`].

use akita_field::unreduced::{HasWide, ReduceTo};
use akita_field::{
    AdditiveGroup, AkitaError, CanonicalField, ExtField, FieldCore, FromPrimitiveInt,
    MulBaseUnreduced,
};
use akita_types::{CleartextWitnessProof, FlatRingVec, FpExtEncoding};

use super::SparseRingPoly;
use crate::backend::RootTensorProjectionPoly;
use crate::compute::{
    CommitInnerPlan, CpuBackend, DecomposeFoldBatchPlan, DecomposeFoldPlan,
    DirectRootWitnessSource, OpeningBatchKernel, OpeningFoldKernel, OpeningFoldOutput,
    OpeningFoldPlan, RootCommitKernel, RootCommitSource, RootOpeningSource, RootPolyShape,
    RootTensorSource, TensorPackedWitness, TensorProjectionBatchKernel, TensorProjectionKernel,
};
use crate::protocol::extension_opening_reduction::SparseExtensionOpeningWitness;
use crate::{CommitInnerWitness, DecomposeFoldWitness};

/// Borrowed commit view over sparse signed ring coefficients.
#[derive(Debug, Clone, Copy)]
pub struct SparseRingCommitView<'a, F: FieldCore, const D: usize> {
    pub(super) poly: &'a SparseRingPoly<F, D>,
}

/// Borrowed opening view for sparse-ring fold and decompose-fold kernels.
#[derive(Debug, Clone, Copy)]
pub struct SparseRingOpeningView<'a, F: FieldCore, const D: usize> {
    pub(super) poly: &'a SparseRingPoly<F, D>,
}

/// Same-point batch opening view over several sparse-ring polynomials.
#[derive(Debug, Clone, Copy)]
pub struct SparseRingOpeningBatchView<'a, F: FieldCore, const D: usize> {
    pub(super) polys: &'a [&'a SparseRingPoly<F, D>],
}

/// Borrowed tensor projection view over sparse-ring coefficients.
#[derive(Debug, Clone, Copy)]
pub struct SparseRingTensorView<'a, F: FieldCore, const D: usize> {
    pub(super) poly: &'a SparseRingPoly<F, D>,
}

/// Same-point batch tensor view over several sparse-ring polynomials.
#[derive(Debug, Clone, Copy)]
pub struct SparseRingTensorBatchView<'a, F: FieldCore, const D: usize> {
    pub(super) polys: &'a [&'a SparseRingPoly<F, D>],
}

impl<F, const D: usize> RootPolyShape<F, D> for SparseRingPoly<F, D>
where
    F: FieldCore,
{
    fn num_ring_elems(&self) -> usize {
        self.total_ring_elems
    }

    fn num_vars(&self) -> usize {
        self.num_vars
    }
}

impl<F, const D: usize> RootCommitSource<F, D> for SparseRingPoly<F, D>
where
    F: FieldCore,
{
    type CommitView<'a>
        = SparseRingCommitView<'a, F, D>
    where
        Self: 'a;

    fn commit_view(&self) -> Result<Self::CommitView<'_>, AkitaError> {
        Ok(SparseRingCommitView { poly: self })
    }
}

impl<F, const D: usize> RootOpeningSource<F, D> for SparseRingPoly<F, D>
where
    F: FieldCore,
{
    type OpeningView<'a>
        = SparseRingOpeningView<'a, F, D>
    where
        Self: 'a;

    type OpeningBatchView<'a>
        = SparseRingOpeningBatchView<'a, F, D>
    where
        Self: 'a;

    fn opening_view(&self) -> Result<Self::OpeningView<'_>, AkitaError> {
        Ok(SparseRingOpeningView { poly: self })
    }

    fn opening_batch<'a>(polys: &'a [&'a Self]) -> Result<Self::OpeningBatchView<'a>, AkitaError> {
        Ok(SparseRingOpeningBatchView { polys })
    }
}

impl<F, const D: usize> RootTensorSource<F, D> for SparseRingPoly<F, D>
where
    F: FieldCore,
{
    type TensorView<'a>
        = SparseRingTensorView<'a, F, D>
    where
        Self: 'a;

    type TensorBatchView<'a>
        = SparseRingTensorBatchView<'a, F, D>
    where
        Self: 'a;

    fn tensor_view(&self) -> Result<Self::TensorView<'_>, AkitaError> {
        Ok(SparseRingTensorView { poly: self })
    }

    fn tensor_batch<'a>(polys: &'a [&'a Self]) -> Result<Self::TensorBatchView<'a>, AkitaError> {
        Ok(SparseRingTensorBatchView { polys })
    }
}

impl<F, const D: usize> DirectRootWitnessSource<F, D> for SparseRingPoly<F, D>
where
    F: FieldCore + FromPrimitiveInt,
{
    fn direct_root_witness(&self) -> Result<CleartextWitnessProof<F>, AkitaError> {
        let total_coeffs = self.total_ring_elems.checked_mul(D).ok_or_else(|| {
            AkitaError::InvalidInput("sparse direct witness length overflow".to_string())
        })?;
        let mut coeffs = vec![F::zero(); total_coeffs];
        for entry in &self.coeffs {
            let idx = (entry.ring_idx as usize)
                .checked_mul(D)
                .and_then(|base| base.checked_add(entry.coeff_idx as usize))
                .ok_or_else(|| {
                    AkitaError::InvalidInput("sparse direct witness index overflow".to_string())
                })?;
            coeffs[idx] += F::from_i8(entry.value);
        }
        Ok(CleartextWitnessProof::FieldElements(
            FlatRingVec::from_coeffs(coeffs),
        ))
    }
}

impl<F, const D: usize> RootCommitKernel<SparseRingCommitView<'_, F, D>, F, D> for CpuBackend
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide,
    F::Wide: AdditiveGroup + From<F> + ReduceTo<F>,
{
    fn commit_inner(
        &self,
        prepared: &Self::PreparedSetup<D>,
        source: SparseRingCommitView<'_, F, D>,
        plan: CommitInnerPlan,
    ) -> Result<CommitInnerWitness<F, D>, AkitaError> {
        source.poly.commit_inner(self, prepared, plan)
    }
}

impl<F, const D: usize> OpeningFoldKernel<SparseRingOpeningView<'_, F, D>, F, D> for CpuBackend
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide,
    F::Wide: AdditiveGroup + From<F> + ReduceTo<F>,
{
    fn evaluate_and_fold(
        &self,
        _prepared: Option<&Self::PreparedSetup<D>>,
        source: SparseRingOpeningView<'_, F, D>,
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
        source: SparseRingOpeningView<'_, F, D>,
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

impl<F, const D: usize> OpeningBatchKernel<SparseRingOpeningBatchView<'_, F, D>, F, D>
    for CpuBackend
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide,
    F::Wide: AdditiveGroup + From<F> + ReduceTo<F>,
{
    fn decompose_fold_batch(
        &self,
        _prepared: Option<&Self::PreparedSetup<D>>,
        source: SparseRingOpeningBatchView<'_, F, D>,
        plan: DecomposeFoldBatchPlan<'_>,
    ) -> Result<Option<DecomposeFoldWitness<F, D>>, AkitaError> {
        match plan {
            DecomposeFoldBatchPlan::Sparse { .. } => Ok(None),
            DecomposeFoldBatchPlan::Tensor {
                tensor,
                block_len,
                num_digits,
                log_basis,
            } => SparseRingPoly::decompose_fold_tensor_batched(
                source.polys,
                tensor,
                block_len,
                num_digits,
                log_basis,
            ),
        }
    }
}

impl<F, E, const D: usize> TensorProjectionKernel<SparseRingTensorView<'_, F, D>, F, E, D>
    for CpuBackend
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide,
    F::Wide: AdditiveGroup + From<F> + ReduceTo<F>,
    E: ExtField<F>,
{
    fn column_partials(
        &self,
        _prepared: Option<&Self::PreparedSetup<D>>,
        source: SparseRingTensorView<'_, F, D>,
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
        source: SparseRingTensorView<'_, F, D>,
    ) -> Result<TensorPackedWitness<E>, AkitaError> {
        Ok(match source.poly.tensor_packed_extension_sparse_evals()? {
            Some(witness) => TensorPackedWitness::Sparse(witness),
            None => TensorPackedWitness::Dense(source.poly.tensor_packed_extension_evals()?),
        })
    }

    fn root_projection(
        &self,
        _prepared: Option<&Self::PreparedSetup<D>>,
        source: SparseRingTensorView<'_, F, D>,
    ) -> Result<RootTensorProjectionPoly<F, D>, AkitaError>
    where
        E: FpExtEncoding<F>,
    {
        Ok(RootTensorProjectionPoly::Dense(
            source.poly.tensor_packed_extension_poly::<E>()?,
        ))
    }
}

impl<F, E, const D: usize> TensorProjectionBatchKernel<SparseRingTensorBatchView<'_, F, D>, F, E, D>
    for CpuBackend
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide,
    F::Wide: AdditiveGroup + From<F> + ReduceTo<F>,
    E: ExtField<F>,
{
    fn column_partials_batch(
        &self,
        _prepared: Option<&Self::PreparedSetup<D>>,
        source: SparseRingTensorBatchView<'_, F, D>,
        logical_point: &[E],
    ) -> Result<Vec<Vec<E>>, AkitaError>
    where
        E: MulBaseUnreduced<F>,
    {
        source
            .polys
            .iter()
            .map(|poly| poly.tensor_extension_column_partials(logical_point))
            .collect()
    }

    fn sparse_linear_combination(
        &self,
        _prepared: Option<&Self::PreparedSetup<D>>,
        source: SparseRingTensorBatchView<'_, F, D>,
        coeffs: &[E],
    ) -> Result<Option<SparseExtensionOpeningWitness<E>>, AkitaError> {
        if source.polys.len() != coeffs.len() {
            return Err(AkitaError::InvalidSize {
                expected: source.polys.len(),
                actual: coeffs.len(),
            });
        }
        let mut witnesses = Vec::with_capacity(source.polys.len());
        for poly in source.polys {
            let Some(witness) = poly.tensor_packed_extension_sparse_evals()? else {
                return Ok(None);
            };
            witnesses.push(witness);
        }
        Ok(Some(SparseExtensionOpeningWitness::linear_combination(
            coeffs.iter().copied().zip(witnesses.iter()),
        )?))
    }
}
