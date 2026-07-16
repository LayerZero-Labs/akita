//! CpuBackend kernels over dense polynomial views.

use super::poly::DensePoly;
use super::views::{DenseBatchView, DenseView};
use crate::backend::RootTensorProjectionPoly;
use crate::compute::{
    BatchDecomposeFoldOutcome, CommitInnerPlan, CpuBackend, DecomposeFoldBatchPlan,
    DecomposeFoldPlan, OpeningBatchKernel, OpeningFoldKernel, OpeningFoldOutput, OpeningFoldPlan,
    RootCommitKernel, TensorPackedWitness, TensorProjectionBatchKernel, TensorProjectionKernel,
};
use crate::protocol::extension_opening_reduction::SparseExtensionOpeningWitness;
use crate::{CommitInnerWitness, DecomposeFoldWitness};
use akita_field::{
    AkitaError, CanonicalField, ExtField, FieldCore, FromPrimitiveInt, MulBaseUnreduced,
};
use akita_types::FpExtEncoding;

impl<F, const D: usize> RootCommitKernel<DenseView<'_, F, D>, F, D> for CpuBackend
where
    F: FieldCore + CanonicalField,
{
    fn commit_inner(
        &self,
        prepared: &Self::PreparedSetup,
        source: DenseView<'_, F, D>,
        plan: CommitInnerPlan,
    ) -> Result<CommitInnerWitness<F>, AkitaError> {
        source.poly.commit_inner::<_, D>(self, prepared, plan)
    }
}

impl<F, const D: usize> OpeningFoldKernel<DenseView<'_, F, D>, F, D> for CpuBackend
where
    F: FieldCore + CanonicalField,
{
    fn evaluate_and_fold(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        source: DenseView<'_, F, D>,
        plan: OpeningFoldPlan<'_, F, D>,
    ) -> Result<OpeningFoldOutput<F, D>, AkitaError> {
        let block_len = plan.block_len();
        if block_len == 0 {
            return Err(AkitaError::InvalidInput(
                "block_len must be positive".to_string(),
            ));
        }
        let num_blocks = source.poly.ring_coeffs::<D>()?.len().div_ceil(block_len);
        plan.validate(num_blocks)?;
        let (eval, folded) = match plan {
            OpeningFoldPlan::Base {
                block_weights,
                position_weights,
                block_len,
            } => source
                .poly
                .evaluate_and_fold::<D>(block_weights, position_weights, block_len),
            OpeningFoldPlan::Ring {
                block_weights,
                position_weights,
                block_len,
            } => source
                .poly
                .evaluate_and_fold_ring(block_weights, position_weights, block_len),
        };
        Ok(OpeningFoldOutput { eval, folded })
    }

    fn decompose_fold(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        source: DenseView<'_, F, D>,
        plan: DecomposeFoldPlan<'_>,
    ) -> Result<DecomposeFoldWitness<F>, AkitaError> {
        Ok(source.poly.decompose_fold::<D>(
            plan.challenges,
            plan.block_len,
            plan.num_digits,
            plan.log_basis,
        ))
    }
}

impl<F, const D: usize> OpeningBatchKernel<DenseBatchView<'_, F, D>, F, D> for CpuBackend
where
    F: FieldCore + CanonicalField,
{
    fn decompose_fold_batch(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        source: DenseBatchView<'_, F, D>,
        plan: DecomposeFoldBatchPlan<'_>,
    ) -> Result<BatchDecomposeFoldOutcome<F, D>, AkitaError> {
        match plan {
            DecomposeFoldBatchPlan::Sparse { .. } => Ok(BatchDecomposeFoldOutcome::FallbackPerPoly),
            DecomposeFoldBatchPlan::Tensor {
                tensor,
                block_len,
                num_digits,
                log_basis,
            } => match DensePoly::decompose_fold_tensor_batched::<D>(
                source.polys,
                tensor,
                block_len,
                num_digits,
                log_basis,
            )? {
                Some(witness) => Ok(BatchDecomposeFoldOutcome::Fused(witness)),
                None => Ok(BatchDecomposeFoldOutcome::Unsupported),
            },
        }
    }
}

impl<F, E, const D: usize> TensorProjectionKernel<DenseView<'_, F, D>, F, E, D> for CpuBackend
where
    F: FieldCore + CanonicalField + FromPrimitiveInt,
    E: ExtField<F>,
{
    fn column_partials(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        source: DenseView<'_, F, D>,
        logical_point: &[E],
    ) -> Result<Vec<E>, AkitaError>
    where
        E: MulBaseUnreduced<F>,
    {
        source
            .poly
            .tensor_extension_column_partials::<E, D>(logical_point)
    }

    fn packed_witness(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        source: DenseView<'_, F, D>,
    ) -> Result<TensorPackedWitness<E>, AkitaError> {
        Ok(TensorPackedWitness::Dense(
            source.poly.tensor_packed_extension_evals::<E, D>()?,
        ))
    }

    fn root_projection(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        source: DenseView<'_, F, D>,
    ) -> Result<RootTensorProjectionPoly<F>, AkitaError>
    where
        E: FpExtEncoding<F>,
    {
        source.poly.tensor_packed_extension_root_poly::<E, D>()
    }
}

impl<F, E, const D: usize> TensorProjectionBatchKernel<DenseBatchView<'_, F, D>, F, E, D>
    for CpuBackend
where
    F: FieldCore + CanonicalField,
    E: ExtField<F>,
{
    fn column_partials_batch(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        source: DenseBatchView<'_, F, D>,
        logical_point: &[E],
    ) -> Result<Vec<Vec<E>>, AkitaError>
    where
        E: MulBaseUnreduced<F>,
    {
        DensePoly::tensor_extension_column_partials_batch::<E, D>(source.polys, logical_point)
    }

    fn sparse_linear_combination(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        source: DenseBatchView<'_, F, D>,
        coeffs: &[E],
    ) -> Result<Option<SparseExtensionOpeningWitness<E>>, AkitaError> {
        DensePoly::tensor_packed_extension_sparse_linear_combination(source.polys, coeffs)
    }
}
