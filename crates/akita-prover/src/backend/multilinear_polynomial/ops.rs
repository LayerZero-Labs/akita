//! `CpuBackend` kernel impls for the multilinear-polynomial wrapper.
//!
//! Each kernel dispatches a source-typed view to the dense or one-hot backend,
//! falling back to a per-polynomial path for truly mixed batches.

use akita_error::AkitaError;
use akita_types::FpExtEncoding;
use jolt_field::unreduced::HasWide;
use jolt_field::{CanonicalField, ExtField, FieldCore, FromPrimitiveInt, MulBaseUnreduced};

use crate::backend::{DenseBatchView, DenseView, OneHotBatchView, OneHotView};
use crate::compute::{
    BatchDecomposeFoldOutcome, CommitInnerPlan, CpuBackend, DecomposeFoldBatchPlan,
    DecomposeFoldPlan, OpeningBatchKernel, OpeningFoldKernel, OpeningFoldOutput, OpeningFoldPlan,
    RootCommitKernel, RootCommitSource, RootOpeningSource, RootTensorSource, TensorPackedWitness,
    TensorProjectionBatchKernel, TensorProjectionKernel,
};
use crate::protocol::extension_opening_reduction::SparseExtensionOpeningWitness;
use crate::{
    CommitInnerWitness, DecomposeFoldWitness, DensePoly, OneHotIndex, OneHotPoly,
    RootTensorProjectionPoly,
};

use super::poly::{
    MultilinearPolynomial, MultilinearPolynomialBatchView, MultilinearPolynomialView,
};

impl<F, const D: usize, I> RootCommitKernel<MultilinearPolynomialView<'_, F, D, I>, F, D>
    for CpuBackend
where
    F: FieldCore + CanonicalField + HasWide,
    I: OneHotIndex,
{
    fn commit_inner(
        &self,
        prepared: &Self::PreparedSetup,
        source: MultilinearPolynomialView<'_, F, D, I>,
        plan: CommitInnerPlan,
    ) -> Result<CommitInnerWitness<F>, AkitaError> {
        source.dispatch(
            |poly| {
                RootCommitKernel::<DenseView<'_, F, D>, F, D>::commit_inner(
                    self,
                    prepared,
                    poly.commit_view()?,
                    plan,
                )
            },
            |poly| {
                RootCommitKernel::<OneHotView<'_, F, D, I>, F, D>::commit_inner(
                    self,
                    prepared,
                    poly.commit_view()?,
                    plan,
                )
            },
        )
    }
}

impl<F, const D: usize, I> OpeningFoldKernel<MultilinearPolynomialView<'_, F, D, I>, F, D>
    for CpuBackend
where
    F: FieldCore + CanonicalField + HasWide,
    I: OneHotIndex,
{
    fn evaluate_and_fold(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        source: MultilinearPolynomialView<'_, F, D, I>,
        plan: OpeningFoldPlan<'_, F, D>,
    ) -> Result<OpeningFoldOutput<F, D>, AkitaError> {
        source.dispatch(
            |poly| {
                OpeningFoldKernel::<DenseView<'_, F, D>, F, D>::evaluate_and_fold(
                    self,
                    prepared,
                    poly.opening_view()?,
                    plan,
                )
            },
            |poly| {
                OpeningFoldKernel::<OneHotView<'_, F, D, I>, F, D>::evaluate_and_fold(
                    self,
                    prepared,
                    poly.opening_view()?,
                    plan,
                )
            },
        )
    }

    fn decompose_fold(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        source: MultilinearPolynomialView<'_, F, D, I>,
        plan: DecomposeFoldPlan<'_>,
    ) -> Result<DecomposeFoldWitness<F>, AkitaError> {
        source.dispatch(
            |poly| {
                OpeningFoldKernel::<DenseView<'_, F, D>, F, D>::decompose_fold(
                    self,
                    prepared,
                    poly.opening_view()?,
                    plan,
                )
            },
            |poly| {
                OpeningFoldKernel::<OneHotView<'_, F, D, I>, F, D>::decompose_fold(
                    self,
                    prepared,
                    poly.opening_view()?,
                    plan,
                )
            },
        )
    }
}

impl<F, const D: usize, I> OpeningBatchKernel<MultilinearPolynomialBatchView<'_, F, D, I>, F, D>
    for CpuBackend
where
    F: FieldCore + CanonicalField + HasWide,
    I: OneHotIndex,
{
    fn decompose_fold_batch(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        source: MultilinearPolynomialBatchView<'_, F, D, I>,
        plan: DecomposeFoldBatchPlan<'_>,
    ) -> Result<BatchDecomposeFoldOutcome<F, D>, AkitaError> {
        let Some(first) = source.polys().first() else {
            return Ok(match plan {
                DecomposeFoldBatchPlan::Sparse { .. } => BatchDecomposeFoldOutcome::FallbackPerPoly,
                DecomposeFoldBatchPlan::Tensor { .. } => BatchDecomposeFoldOutcome::Unsupported,
            });
        };
        match first {
            MultilinearPolynomial::Dense(_) => {
                let Some(dense_polys) = source.homogeneous_dense_polys() else {
                    return Ok(match plan {
                        DecomposeFoldBatchPlan::Sparse { .. } => {
                            BatchDecomposeFoldOutcome::FallbackPerPoly
                        }
                        DecomposeFoldBatchPlan::Tensor { .. } => {
                            BatchDecomposeFoldOutcome::Unsupported
                        }
                    });
                };
                let dense_view =
                    <DensePoly<F> as RootOpeningSource<F, D>>::opening_batch(&dense_polys)?;
                OpeningBatchKernel::<DenseBatchView<'_, F, D>, F, D>::decompose_fold_batch(
                    self, prepared, dense_view, plan,
                )
            }
            MultilinearPolynomial::OneHot(_) => {
                let Some(onehot_polys) = source.homogeneous_onehot_polys() else {
                    return Ok(match plan {
                        DecomposeFoldBatchPlan::Sparse { .. } => {
                            BatchDecomposeFoldOutcome::FallbackPerPoly
                        }
                        DecomposeFoldBatchPlan::Tensor { .. } => {
                            BatchDecomposeFoldOutcome::Unsupported
                        }
                    });
                };
                let onehot_view =
                    <OneHotPoly<F, I> as RootOpeningSource<F, D>>::opening_batch(&onehot_polys)?;
                OpeningBatchKernel::<OneHotBatchView<'_, F, D, I>, F, D>::decompose_fold_batch(
                    self,
                    prepared,
                    onehot_view,
                    plan,
                )
            }
        }
    }
}

impl<F, E, const D: usize, I>
    TensorProjectionKernel<MultilinearPolynomialView<'_, F, D, I>, F, E, D> for CpuBackend
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide,
    E: ExtField<F>,
    I: OneHotIndex,
{
    fn column_partials(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        source: MultilinearPolynomialView<'_, F, D, I>,
        logical_point: &[E],
    ) -> Result<Vec<E>, AkitaError>
    where
        E: MulBaseUnreduced<F>,
    {
        source.dispatch(
            |poly| {
                TensorProjectionKernel::<DenseView<'_, F, D>, F, E, D>::column_partials(
                    self,
                    prepared,
                    poly.tensor_view()?,
                    logical_point,
                )
            },
            |poly| {
                TensorProjectionKernel::<OneHotView<'_, F, D, I>, F, E, D>::column_partials(
                    self,
                    prepared,
                    poly.tensor_view()?,
                    logical_point,
                )
            },
        )
    }

    fn packed_witness(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        source: MultilinearPolynomialView<'_, F, D, I>,
    ) -> Result<TensorPackedWitness<E>, AkitaError> {
        source.dispatch(
            |poly| {
                TensorProjectionKernel::<DenseView<'_, F, D>, F, E, D>::packed_witness(
                    self,
                    prepared,
                    poly.tensor_view()?,
                )
            },
            |poly| {
                TensorProjectionKernel::<OneHotView<'_, F, D, I>, F, E, D>::packed_witness(
                    self,
                    prepared,
                    poly.tensor_view()?,
                )
            },
        )
    }

    fn root_projection(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        source: MultilinearPolynomialView<'_, F, D, I>,
    ) -> Result<RootTensorProjectionPoly<F>, AkitaError>
    where
        E: FpExtEncoding<F>,
    {
        source.dispatch(
            |poly| {
                TensorProjectionKernel::<DenseView<'_, F, D>, F, E, D>::root_projection(
                    self,
                    prepared,
                    poly.tensor_view()?,
                )
            },
            |poly| {
                TensorProjectionKernel::<OneHotView<'_, F, D, I>, F, E, D>::root_projection(
                    self,
                    prepared,
                    poly.tensor_view()?,
                )
            },
        )
    }
}

impl<F, E, const D: usize, I>
    TensorProjectionBatchKernel<MultilinearPolynomialBatchView<'_, F, D, I>, F, E, D> for CpuBackend
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide,
    E: ExtField<F>,
    I: OneHotIndex,
{
    fn column_partials_batch(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        source: MultilinearPolynomialBatchView<'_, F, D, I>,
        logical_point: &[E],
    ) -> Result<Vec<Vec<E>>, AkitaError>
    where
        E: MulBaseUnreduced<F>,
    {
        let Some(first) = source.polys().first() else {
            return Ok(Vec::new());
        };
        match first {
            MultilinearPolynomial::Dense(_) => {
                let Some(dense_polys) = source.homogeneous_dense_polys() else {
                    return source.column_partials_per_poly(self, prepared, logical_point);
                };
                let dense_view =
                    <DensePoly<F> as RootTensorSource<F, D>>::tensor_batch(&dense_polys)?;
                TensorProjectionBatchKernel::<DenseBatchView<'_, F, D>, F, E, D>::column_partials_batch(
                    self,
                    prepared,
                    dense_view,
                    logical_point,
                )
            }
            MultilinearPolynomial::OneHot(_) => {
                let Some(onehot_polys) = source.homogeneous_onehot_polys() else {
                    return source.column_partials_per_poly(self, prepared, logical_point);
                };
                let onehot_view =
                    <OneHotPoly<F, I> as RootTensorSource<F, D>>::tensor_batch(&onehot_polys)?;
                TensorProjectionBatchKernel::<OneHotBatchView<'_, F, D, I>, F, E, D>::column_partials_batch(
                    self,
                    prepared,
                    onehot_view,
                    logical_point,
                )
            }
        }
    }

    fn sparse_linear_combination(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        source: MultilinearPolynomialBatchView<'_, F, D, I>,
        coeffs: &[E],
    ) -> Result<Option<SparseExtensionOpeningWitness<E>>, AkitaError> {
        let Some(first) = source.polys().first() else {
            return Ok(None);
        };
        match first {
            MultilinearPolynomial::Dense(_) => {
                let Some(dense_polys) = source.homogeneous_dense_polys() else {
                    return Ok(None);
                };
                let dense_view =
                    <DensePoly<F> as RootTensorSource<F, D>>::tensor_batch(&dense_polys)?;
                TensorProjectionBatchKernel::<DenseBatchView<'_, F, D>, F, E, D>::sparse_linear_combination(
                    self,
                    prepared,
                    dense_view,
                    coeffs,
                )
            }
            MultilinearPolynomial::OneHot(_) => {
                let Some(onehot_polys) = source.homogeneous_onehot_polys() else {
                    return Ok(None);
                };
                let onehot_view =
                    <OneHotPoly<F, I> as RootTensorSource<F, D>>::tensor_batch(&onehot_polys)?;
                TensorProjectionBatchKernel::<OneHotBatchView<'_, F, D, I>, F, E, D>::sparse_linear_combination(
                    self,
                    prepared,
                    onehot_view,
                    coeffs,
                )
            }
        }
    }
}
