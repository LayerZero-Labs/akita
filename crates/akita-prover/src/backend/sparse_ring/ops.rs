//! Source-typed views and `CpuBackend` kernels for [`super::SparseRingPoly`].

use akita_field::unreduced::{HasWide, ReduceTo};
use akita_field::{
    AdditiveGroup, AkitaError, CanonicalField, ExtField, FieldCore, FromPrimitiveInt,
    MulBaseUnreduced,
};
use akita_types::{CleartextWitnessProof, FpExtEncoding, RingVec};

use super::SparseRingPoly;
use crate::backend::RootTensorProjectionPoly;
use crate::compute::{
    BatchDecomposeFoldOutcome, CommitInnerPlan, CpuBackend, DecomposeFoldBatchPlan,
    DecomposeFoldPlan, DirectRootWitnessSource, OpeningBatchKernel, OpeningFoldKernel,
    OpeningFoldOutput, OpeningFoldPlan, RootCommitKernel, RootCommitSource, RootOpeningSource,
    RootPolyMeta, RootPolyShape, RootTensorSource, TensorPackedWitness,
    TensorProjectionBatchKernel, TensorProjectionKernel,
};
use crate::protocol::extension_opening_reduction::SparseExtensionOpeningWitness;
use crate::{CommitInnerWitness, DecomposeFoldWitness};

/// Borrowed single-polynomial view over sparse signed ring coefficients.
///
/// One view type backs the commit, opening-fold, and tensor-projection kernels;
/// the kernel trait it is passed to selects the operation.
#[derive(Debug, Clone, Copy)]
pub struct SparseRingView<'a, F: FieldCore, const D: usize> {
    pub(super) poly: &'a SparseRingPoly<F, D>,
}

/// Same-point batch view over several sparse-ring polynomials.
#[derive(Debug, Clone, Copy)]
pub struct SparseRingBatchView<'a, F: FieldCore, const D: usize> {
    pub(super) polys: &'a [&'a SparseRingPoly<F, D>],
}

impl<F, const D: usize> RootPolyMeta<F> for SparseRingPoly<F, D>
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
        = SparseRingView<'a, F, D>
    where
        Self: 'a;

    fn commit_view(&self) -> Result<Self::CommitView<'_>, AkitaError> {
        Ok(SparseRingView { poly: self })
    }
}

impl<F, const D: usize> RootOpeningSource<F, D> for SparseRingPoly<F, D>
where
    F: FieldCore,
{
    type OpeningView<'a>
        = SparseRingView<'a, F, D>
    where
        Self: 'a;

    type OpeningBatchView<'a>
        = SparseRingBatchView<'a, F, D>
    where
        Self: 'a;

    fn opening_view(&self) -> Result<Self::OpeningView<'_>, AkitaError> {
        Ok(SparseRingView { poly: self })
    }

    fn opening_batch<'a>(polys: &'a [&'a Self]) -> Result<Self::OpeningBatchView<'a>, AkitaError> {
        Ok(SparseRingBatchView { polys })
    }
}

impl<F, const D: usize> RootTensorSource<F, D> for SparseRingPoly<F, D>
where
    F: FieldCore,
{
    type TensorView<'a>
        = SparseRingView<'a, F, D>
    where
        Self: 'a;

    type TensorBatchView<'a>
        = SparseRingBatchView<'a, F, D>
    where
        Self: 'a;

    fn tensor_view(&self) -> Result<Self::TensorView<'_>, AkitaError> {
        Ok(SparseRingView { poly: self })
    }

    fn tensor_batch<'a>(polys: &'a [&'a Self]) -> Result<Self::TensorBatchView<'a>, AkitaError> {
        Ok(SparseRingBatchView { polys })
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
        Ok(CleartextWitnessProof::FieldElements(RingVec::from_coeffs(
            coeffs,
        )))
    }
}

impl<F, const D: usize> RootCommitKernel<SparseRingView<'_, F, D>, F, D> for CpuBackend
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide,
    F::Wide: AdditiveGroup + From<F> + ReduceTo<F>,
{
    fn commit_inner(
        &self,
        prepared: &Self::PreparedSetup,
        source: SparseRingView<'_, F, D>,
        plan: CommitInnerPlan,
    ) -> Result<CommitInnerWitness<F>, AkitaError> {
        source.poly.commit_inner(self, prepared, plan)
    }
}

impl<F, const D: usize> OpeningFoldKernel<SparseRingView<'_, F, D>, F, D> for CpuBackend
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide,
    F::Wide: AdditiveGroup + From<F> + ReduceTo<F>,
{
    fn evaluate_and_fold(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        source: SparseRingView<'_, F, D>,
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
        _prepared: Option<&Self::PreparedSetup>,
        source: SparseRingView<'_, F, D>,
        plan: DecomposeFoldPlan<'_>,
    ) -> Result<DecomposeFoldWitness<F>, AkitaError> {
        Ok(source.poly.decompose_fold(
            plan.challenges,
            plan.block_len,
            plan.num_digits,
            plan.log_basis,
        ))
    }
}

impl<F, const D: usize> OpeningBatchKernel<SparseRingBatchView<'_, F, D>, F, D> for CpuBackend
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide,
    F::Wide: AdditiveGroup + From<F> + ReduceTo<F>,
{
    fn decompose_fold_batch(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        source: SparseRingBatchView<'_, F, D>,
        plan: DecomposeFoldBatchPlan<'_>,
    ) -> Result<BatchDecomposeFoldOutcome<F, D>, AkitaError> {
        match plan {
            DecomposeFoldBatchPlan::Sparse { .. } => Ok(BatchDecomposeFoldOutcome::FallbackPerPoly),
            DecomposeFoldBatchPlan::Tensor {
                tensor,
                block_len,
                num_digits,
                log_basis,
            } => match SparseRingPoly::decompose_fold_tensor_batched(
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

impl<F, E, const D: usize> TensorProjectionKernel<SparseRingView<'_, F, D>, F, E, D> for CpuBackend
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide,
    F::Wide: AdditiveGroup + From<F> + ReduceTo<F>,
    E: ExtField<F>,
{
    fn column_partials(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        source: SparseRingView<'_, F, D>,
        logical_point: &[E],
    ) -> Result<Vec<E>, AkitaError>
    where
        E: MulBaseUnreduced<F>,
    {
        source.poly.tensor_extension_column_partials(logical_point)
    }

    fn packed_witness(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        source: SparseRingView<'_, F, D>,
    ) -> Result<TensorPackedWitness<E>, AkitaError> {
        Ok(match source.poly.tensor_packed_extension_sparse_evals()? {
            Some(witness) => TensorPackedWitness::Sparse(witness),
            None => TensorPackedWitness::Dense(source.poly.tensor_packed_extension_evals()?),
        })
    }

    fn root_projection(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        source: SparseRingView<'_, F, D>,
    ) -> Result<RootTensorProjectionPoly<F, D>, AkitaError>
    where
        E: FpExtEncoding<F>,
    {
        Ok(RootTensorProjectionPoly::Dense(
            source.poly.tensor_packed_extension_poly::<E>()?,
        ))
    }
}

impl<F, E, const D: usize> TensorProjectionBatchKernel<SparseRingBatchView<'_, F, D>, F, E, D>
    for CpuBackend
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide,
    F::Wide: AdditiveGroup + From<F> + ReduceTo<F>,
    E: ExtField<F>,
{
    fn column_partials_batch(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        source: SparseRingBatchView<'_, F, D>,
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
        _prepared: Option<&Self::PreparedSetup>,
        source: SparseRingBatchView<'_, F, D>,
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
