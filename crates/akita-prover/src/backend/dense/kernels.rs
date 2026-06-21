//! CpuBackend kernels over dense polynomial views.

use super::poly::DensePoly;
use super::views::{
    DenseCommitView, DenseOpeningBatchView, DenseOpeningView, DenseTensorBatchView, DenseTensorView,
};
use crate::backend::RootTensorProjectionPoly;
use crate::protocol::extension_opening_reduction::SparseExtensionOpeningWitness;
use crate::compute::{
    CommitInnerPlan, CpuBackend, DecomposeFoldBatchPlan,
    DecomposeFoldPlan, OpeningBatchKernel, OpeningFoldKernel, OpeningFoldOutput, OpeningFoldPlan,
    RootCommitKernel, TensorPackedWitness, TensorProjectionBatchKernel, TensorProjectionKernel,
};
use akita_field::{
    AkitaError, CanonicalField, ExtField, FieldCore, FromPrimitiveInt, MulBaseUnreduced,
};
use akita_types::FpExtEncoding;
use crate::{CommitInnerWitness, DecomposeFoldWitness};

impl<F, const D: usize> RootCommitKernel<DenseCommitView<'_, F, D>, F, D> for CpuBackend
where
    F: FieldCore + CanonicalField,
{
    fn commit_inner(
        &self,
        prepared: &Self::PreparedSetup<D>,
        source: DenseCommitView<'_, F, D>,
        plan: CommitInnerPlan,
    ) -> Result<CommitInnerWitness<F, D>, AkitaError> {
        source.poly.commit_inner(self, prepared, plan)
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
        E: FpExtEncoding<F>,
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
