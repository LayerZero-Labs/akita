//! `CpuBackend` kernel impls for the multilinear-polynomial wrapper.
//!
//! Each kernel dispatches a source-typed view to the dense or one-hot backend,
//! falling back to a per-polynomial path for truly mixed batches.

use akita_field::unreduced::{HasCommitAccum, HasWide};
use akita_field::{
    AkitaError, CanonicalField, ExtField, FieldCore, FromPrimitiveInt, MulBaseUnreduced,
};
use akita_types::FpExtEncoding;

use crate::backend::{DenseBatchView, DenseView, OneHotBatchView, OneHotView};
use crate::compute::{
    BatchDecomposeFoldOutcome, CommitInnerPlan, CpuBackend, DecomposeFoldBatchPlan,
    DecomposeFoldPlan, OpeningBatchKernel, OpeningFoldKernel, OpeningFoldOutput, OpeningFoldPlan,
    RootCommitKernel, RootCommitSource, RootOpeningSource, RootTensorSource,
    SubringCoefficientPackingBatchKernel, SubringCoefficientPackingPartials,
    SubringCoefficientPackingPlan, TensorPackedWitness, TensorProjectionBatchKernel,
    TensorProjectionKernel,
};
use crate::protocol::extension_opening_reduction::SparseExtensionOpeningWitness;
use crate::{
    CommitInnerWitness, DecomposeFoldWitness, DensePoly, OneHotIndex, OneHotPoly,
    RootTensorProjectionPoly,
};

use super::poly::{
    MultilinearPolynomial, MultilinearPolynomialBatchView, MultilinearPolynomialView,
};

impl<F, E, const D: usize, I>
    SubringCoefficientPackingBatchKernel<MultilinearPolynomialBatchView<'_, F, D, I>, F, E, D>
    for CpuBackend
where
    F: FieldCore + CanonicalField + FromPrimitiveInt,
    E: ExtField<F> + FpExtEncoding<F>,
    I: OneHotIndex,
{
    fn coefficient_packing_partials_batch(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        source: MultilinearPolynomialBatchView<'_, F, D, I>,
        plan: SubringCoefficientPackingPlan<'_, E>,
    ) -> Result<Vec<SubringCoefficientPackingPartials<F>>, AkitaError> {
        if let Some(dense_polys) = source.homogeneous_dense_polys() {
            let view = <DensePoly<F> as RootOpeningSource<F, D>>::opening_batch(&dense_polys)?;
            return SubringCoefficientPackingBatchKernel::<
                DenseBatchView<'_, F, D>,
                F,
                E,
                D,
            >::coefficient_packing_partials_batch(self, prepared, view, plan);
        }
        if let Some(onehot_polys) = source.homogeneous_onehot_polys() {
            let view = <OneHotPoly<F, I> as RootOpeningSource<F, D>>::opening_batch(&onehot_polys)?;
            return SubringCoefficientPackingBatchKernel::<
                OneHotBatchView<'_, F, D, I>,
                F,
                E,
                D,
            >::coefficient_packing_partials_batch(self, prepared, view, plan);
        }
        let mut outputs = Vec::with_capacity(source.polys().len());
        for poly in source.polys() {
            match poly {
                MultilinearPolynomial::Dense(poly) => {
                    let polys = [poly];
                    let view = <DensePoly<F> as RootOpeningSource<F, D>>::opening_batch(&polys)?;
                    outputs.extend(SubringCoefficientPackingBatchKernel::<
                        DenseBatchView<'_, F, D>,
                        F,
                        E,
                        D,
                    >::coefficient_packing_partials_batch(
                        self, prepared, view, plan
                    )?);
                }
                MultilinearPolynomial::OneHot(poly) => {
                    let polys = [poly];
                    let view =
                        <OneHotPoly<F, I> as RootOpeningSource<F, D>>::opening_batch(&polys)?;
                    outputs.extend(SubringCoefficientPackingBatchKernel::<
                        OneHotBatchView<'_, F, D, I>,
                        F,
                        E,
                        D,
                    >::coefficient_packing_partials_batch(
                        self, prepared, view, plan
                    )?);
                }
            }
        }
        Ok(outputs)
    }
}

impl<F, const D: usize, I> RootCommitKernel<MultilinearPolynomialView<'_, F, D, I>, F, D>
    for CpuBackend
where
    F: FieldCore + CanonicalField + HasWide + HasCommitAccum,
    I: OneHotIndex,
{
    fn commit_inner_group(
        &self,
        prepared: &Self::PreparedSetup,
        sources: Vec<MultilinearPolynomialView<'_, F, D, I>>,
        plan: CommitInnerPlan,
    ) -> Result<Vec<CommitInnerWitness<F>>, AkitaError> {
        if sources
            .iter()
            .all(|source| matches!(source.poly(), MultilinearPolynomial::Dense(_)))
        {
            let views = sources
                .into_iter()
                .map(|source| match source.poly() {
                    MultilinearPolynomial::Dense(poly) => poly.commit_view(),
                    MultilinearPolynomial::OneHot(_) => unreachable!("checked dense group"),
                })
                .collect::<Result<Vec<_>, _>>()?;
            return RootCommitKernel::<DenseView<'_, F, D>, F, D>::commit_inner_group(
                self, prepared, views, plan,
            );
        }
        if sources
            .iter()
            .all(|source| matches!(source.poly(), MultilinearPolynomial::OneHot(_)))
        {
            let views = sources
                .into_iter()
                .map(|source| match source.poly() {
                    MultilinearPolynomial::OneHot(poly) => poly.commit_view(),
                    MultilinearPolynomial::Dense(_) => unreachable!("checked one-hot group"),
                })
                .collect::<Result<Vec<_>, _>>()?;
            return RootCommitKernel::<OneHotView<'_, F, D, I>, F, D>::commit_inner_group(
                self, prepared, views, plan,
            );
        }
        let mut witnesses = Vec::with_capacity(sources.len());
        for source in sources {
            let committed = match source.poly() {
                MultilinearPolynomial::Dense(poly) => {
                    RootCommitKernel::<DenseView<'_, F, D>, F, D>::commit_inner_group(
                        self,
                        prepared,
                        vec![poly.commit_view()?],
                        plan,
                    )?
                }
                MultilinearPolynomial::OneHot(poly) => {
                    RootCommitKernel::<OneHotView<'_, F, D, I>, F, D>::commit_inner_group(
                        self,
                        prepared,
                        vec![poly.commit_view()?],
                        plan,
                    )?
                }
            };
            let [witness] = committed.try_into().map_err(|committed: Vec<_>| {
                AkitaError::InvalidSetup(format!(
                    "child kernel returned {} mixed-group sources, expected one",
                    committed.len()
                ))
            })?;
            witnesses.push(witness);
        }
        Ok(witnesses)
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
        plan: OpeningFoldPlan<'_, F>,
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
            return Ok(BatchDecomposeFoldOutcome::FallbackPerPoly);
        };
        match first {
            MultilinearPolynomial::Dense(_) => {
                let Some(dense_polys) = source.homogeneous_dense_polys() else {
                    return Ok(BatchDecomposeFoldOutcome::FallbackPerPoly);
                };
                let dense_view =
                    <DensePoly<F> as RootOpeningSource<F, D>>::opening_batch(&dense_polys)?;
                OpeningBatchKernel::<DenseBatchView<'_, F, D>, F, D>::decompose_fold_batch(
                    self, prepared, dense_view, plan,
                )
            }
            MultilinearPolynomial::OneHot(_) => {
                let Some(onehot_polys) = source.homogeneous_onehot_polys() else {
                    return Ok(BatchDecomposeFoldOutcome::FallbackPerPoly);
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
