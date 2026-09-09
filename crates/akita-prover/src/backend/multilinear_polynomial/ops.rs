//! `CpuBackend` kernel impls for the multilinear-polynomial wrapper.
//!
//! Each kernel dispatches a source-typed view to the dense or one-hot backend,
//! falling back to a per-polynomial path for truly mixed batches.

use akita_error::AkitaError;
use akita_types::FpExtEncoding;
use jolt_field::Unreduced;
use jolt_field::{CanonicalEncoding, ExtField, Field, MulBaseUnreduced, Ring};

use crate::backend::coefficient_packing::FusedPackingWeights;
use crate::backend::dense::dense_coefficient_packing_partials;
use crate::backend::onehot::onehot_coefficient_packing_partials;
use crate::backend::{DenseBatchView, DenseView, OneHotBatchView, OneHotView};
use crate::compute::{
    BatchDecomposeFoldOutcome, CpuBackend, DecomposeFoldBatchPlan, DecomposeFoldPlan,
    OpeningBatchKernel, OpeningFoldKernel, OpeningFoldOutput, OpeningFoldPlan, RootOpeningSource,
    SubringCoefficientPackingBatchKernel, SubringCoefficientPackingPartials,
    SubringCoefficientPackingPlan,
};
use crate::{DecomposeFoldWitness, DensePoly, OneHotIndex, OneHotPoly};

use super::poly::{
    MultilinearPolynomial, MultilinearPolynomialBatchView, MultilinearPolynomialView,
};

impl<F, E, const D: usize, I>
    SubringCoefficientPackingBatchKernel<MultilinearPolynomialBatchView<'_, F, D, I>, F, E, D>
    for CpuBackend
where
    F: Field + CanonicalEncoding + Ring,
    E: ExtField<F> + FpExtEncoding<F> + MulBaseUnreduced<F>,
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
            let fused_weights = FusedPackingWeights::new(plan.point)?;
            return dense_coefficient_packing_partials(view, &fused_weights);
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
        let fused_weights = FusedPackingWeights::new(plan.point)?;
        let mut outputs = Vec::with_capacity(source.polys().len());
        for poly in source.polys() {
            match poly {
                MultilinearPolynomial::Dense(poly) => {
                    let polys = [poly];
                    let view = <DensePoly<F> as RootOpeningSource<F, D>>::opening_batch(&polys)?;
                    outputs.extend(dense_coefficient_packing_partials(view, &fused_weights)?);
                }
                MultilinearPolynomial::OneHot(poly) => {
                    let polys = [poly];
                    let view =
                        <OneHotPoly<F, I> as RootOpeningSource<F, D>>::opening_batch(&polys)?;
                    outputs.extend(onehot_coefficient_packing_partials(view, &fused_weights)?);
                }
            }
        }
        Ok(outputs)
    }
}

impl<F, const D: usize, I> OpeningFoldKernel<MultilinearPolynomialView<'_, F, D, I>, F, D>
    for CpuBackend
where
    F: Field + CanonicalEncoding + Unreduced,
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
    F: Field + CanonicalEncoding + Unreduced,
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
