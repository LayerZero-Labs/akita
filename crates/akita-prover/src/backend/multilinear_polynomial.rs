//! Canonical multilinear-polynomial wrapper for dense and one-hot representations.
//!
//! This is the intended public wrapper for heterogeneous root batches. All
//! wrapped polynomials must still share the same commitment config and root
//! layout chosen by the caller, but one batch can contain both dense and
//! one-hot roots.
//!
//! Homogeneous batches still reuse the existing backend-specific batched fast
//! paths; truly mixed batches fall back to the caller's per-polynomial
//! aggregation path.

use akita_algebra::CyclotomicRing;
use akita_challenges::{SparseChallenge, TensorChallenges};
use akita_field::unreduced::HasWide;
use akita_field::{
    AkitaError, CanonicalField, ExtField, FieldCore, FromPrimitiveInt, MulBaseUnreduced,
};
use akita_types::{CleartextWitnessProof, FlatDigitBlocks, FpExtEncoding};

use crate::backend::{
    DenseCommitView, DenseOpeningBatchView, DenseOpeningView, DenseTensorBatchView,
    DenseTensorView, OneHotCommitView, OneHotOpeningBatchView, OneHotOpeningView,
    OneHotTensorBatchView, OneHotTensorView,
};
use crate::compute::{
    CommitInnerPlan, CommitmentComputeBackend, CpuBackend, DecomposeFoldBatchPlan,
    DecomposeFoldPlan, DirectRootWitnessSource, OpeningBatchKernel, OpeningFoldKernel,
    OpeningFoldOutput, OpeningFoldPlan, RootBaseEvalsSource, RootCommitKernel, RootCommitSource,
    RootExtensionEvalSource, RootOpeningSource, RootPolyShape, RootTensorSource,
    TensorPackedWitness, TensorProjectionBatchKernel, TensorProjectionKernel,
};
use crate::protocol::extension_opening_reduction::SparseExtensionOpeningWitness;
use crate::{
    AkitaPolyOps, CommitInnerWitness, DecomposeFoldWitness, DensePoly, OneHotIndex, OneHotPoly,
    RootTensorProjectionPoly,
};

/// Owned multilinear-polynomial wrapper for dense and one-hot batches.
///
/// This is an Akita-owned private sum type (allowed by the polyops cutover
/// spec): it erases `DensePoly` vs `OneHotPoly` for heterogeneous batches while
/// exposing the source-typed view/kernel boundary (`RootCommitSource`,
/// `RootOpeningSource`, `RootTensorSource`, and matching `CpuBackend` kernels).
/// Wrappers take ownership of the inner polynomial by move so `P` has no lifetime
/// parameter and participates in generic `commit<P, B>` like `DensePoly`.
#[derive(Debug, Clone)]
pub enum MultilinearPolynomial<F: FieldCore, const D: usize, I: OneHotIndex = usize> {
    /// Dense multilinear polynomial.
    Dense(DensePoly<F, D>),
    /// One-hot multilinear polynomial.
    OneHot(OneHotPoly<F, D, I>),
}

impl<F: FieldCore, const D: usize, I: OneHotIndex> MultilinearPolynomial<F, D, I> {
    /// Wrap a dense polynomial.
    pub fn dense(poly: DensePoly<F, D>) -> Self {
        Self::Dense(poly)
    }

    /// Wrap a one-hot polynomial.
    pub fn onehot(poly: OneHotPoly<F, D, I>) -> Self {
        Self::OneHot(poly)
    }
}

/// Borrowed dispatch view for an Akita-owned multilinear root wrapper.
#[derive(Debug, Clone, Copy)]
pub struct MultilinearPolynomialView<'a, F: FieldCore, const D: usize, I: OneHotIndex = usize> {
    poly: &'a MultilinearPolynomial<F, D, I>,
}

/// Same-point batch dispatch view over multilinear root wrappers.
#[derive(Debug, Clone, Copy)]
pub struct MultilinearPolynomialBatchView<'a, F: FieldCore, const D: usize, I: OneHotIndex = usize>
{
    polys: &'a [&'a MultilinearPolynomial<F, D, I>],
}

impl<F, const D: usize, I> RootPolyShape<F, D> for MultilinearPolynomial<F, D, I>
where
    F: FieldCore,
    I: OneHotIndex,
{
    fn num_ring_elems(&self) -> usize {
        match self {
            Self::Dense(poly) => RootPolyShape::num_ring_elems(poly),
            Self::OneHot(poly) => RootPolyShape::num_ring_elems(poly),
        }
    }

    fn num_vars(&self) -> usize {
        match self {
            Self::Dense(poly) => RootPolyShape::num_vars(poly),
            Self::OneHot(poly) => RootPolyShape::num_vars(poly),
        }
    }
}

impl<F, const D: usize, I> RootCommitSource<F, D> for MultilinearPolynomial<F, D, I>
where
    F: FieldCore,
    I: OneHotIndex,
{
    type CommitView<'view>
        = MultilinearPolynomialView<'view, F, D, I>
    where
        Self: 'view;

    fn commit_view(&self) -> Result<Self::CommitView<'_>, AkitaError> {
        Ok(MultilinearPolynomialView { poly: self })
    }
}

impl<F, const D: usize, I> RootOpeningSource<F, D> for MultilinearPolynomial<F, D, I>
where
    F: FieldCore,
    I: OneHotIndex,
{
    type OpeningView<'view>
        = MultilinearPolynomialView<'view, F, D, I>
    where
        Self: 'view;

    type OpeningBatchView<'view>
        = MultilinearPolynomialBatchView<'view, F, D, I>
    where
        Self: 'view;

    fn opening_view(&self) -> Result<Self::OpeningView<'_>, AkitaError> {
        Ok(MultilinearPolynomialView { poly: self })
    }

    fn opening_batch<'view>(
        polys: &'view [&'view Self],
    ) -> Result<Self::OpeningBatchView<'view>, AkitaError> {
        Ok(MultilinearPolynomialBatchView { polys })
    }
}

impl<F, const D: usize, I> RootTensorSource<F, D> for MultilinearPolynomial<F, D, I>
where
    F: FieldCore,
    I: OneHotIndex,
{
    type TensorView<'view>
        = MultilinearPolynomialView<'view, F, D, I>
    where
        Self: 'view;

    type TensorBatchView<'view>
        = MultilinearPolynomialBatchView<'view, F, D, I>
    where
        Self: 'view;

    fn tensor_view(&self) -> Result<Self::TensorView<'_>, AkitaError> {
        Ok(MultilinearPolynomialView { poly: self })
    }

    fn tensor_batch<'view>(
        polys: &'view [&'view Self],
    ) -> Result<Self::TensorBatchView<'view>, AkitaError> {
        Ok(MultilinearPolynomialBatchView { polys })
    }
}

impl<F, const D: usize, I> DirectRootWitnessSource<F, D> for MultilinearPolynomial<F, D, I>
where
    F: FieldCore + CanonicalField,
    I: OneHotIndex,
{
    fn direct_root_witness(&self) -> Result<CleartextWitnessProof<F>, AkitaError> {
        match self {
            Self::Dense(poly) => DirectRootWitnessSource::direct_root_witness(poly),
            Self::OneHot(poly) => DirectRootWitnessSource::direct_root_witness(poly),
        }
    }
}

impl<F, const D: usize, I> RootBaseEvalsSource<F, D> for MultilinearPolynomial<F, D, I>
where
    F: FieldCore + CanonicalField + HasWide,
    I: OneHotIndex,
{
    fn base_evals(&self) -> Result<Vec<F>, AkitaError> {
        match self {
            Self::Dense(poly) => RootBaseEvalsSource::base_evals(poly),
            Self::OneHot(poly) => RootBaseEvalsSource::base_evals(poly),
        }
    }
}

impl<F, const D: usize, I> RootExtensionEvalSource<F, D> for MultilinearPolynomial<F, D, I>
where
    F: FieldCore + CanonicalField + HasWide,
    I: OneHotIndex,
{
    fn evaluate_extension<E>(&self, point: &[E]) -> Result<E, AkitaError>
    where
        E: ExtField<F>,
    {
        match self {
            Self::Dense(poly) => RootExtensionEvalSource::evaluate_extension(poly, point),
            Self::OneHot(poly) => RootExtensionEvalSource::evaluate_extension(poly, point),
        }
    }
}

impl<F, const D: usize, I> RootCommitKernel<MultilinearPolynomialView<'_, F, D, I>, F, D>
    for CpuBackend
where
    F: FieldCore + CanonicalField + HasWide,
    I: OneHotIndex,
{
    fn commit_inner(
        &self,
        prepared: &Self::PreparedSetup<D>,
        source: MultilinearPolynomialView<'_, F, D, I>,
        plan: CommitInnerPlan,
    ) -> Result<CommitInnerWitness<F, D>, AkitaError> {
        match source.poly {
            MultilinearPolynomial::Dense(poly) => {
                RootCommitKernel::<DenseCommitView<'_, F, D>, F, D>::commit_inner(
                    self,
                    prepared,
                    poly.commit_view()?,
                    plan,
                )
            }
            MultilinearPolynomial::OneHot(poly) => RootCommitKernel::<
                OneHotCommitView<'_, F, D, I>,
                F,
                D,
            >::commit_inner(
                self, prepared, poly.commit_view()?, plan
            ),
        }
    }

    fn commit_inner_blocks(
        &self,
        prepared: &Self::PreparedSetup<D>,
        source: MultilinearPolynomialView<'_, F, D, I>,
        plan: CommitInnerPlan,
    ) -> Result<FlatDigitBlocks<D>, AkitaError> {
        match source.poly {
            MultilinearPolynomial::Dense(poly) => {
                RootCommitKernel::<DenseCommitView<'_, F, D>, F, D>::commit_inner_blocks(
                    self,
                    prepared,
                    poly.commit_view()?,
                    plan,
                )
            }
            MultilinearPolynomial::OneHot(poly) => RootCommitKernel::<
                OneHotCommitView<'_, F, D, I>,
                F,
                D,
            >::commit_inner_blocks(
                self, prepared, poly.commit_view()?, plan
            ),
        }
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
        prepared: Option<&Self::PreparedSetup<D>>,
        source: MultilinearPolynomialView<'_, F, D, I>,
        plan: OpeningFoldPlan<'_, F, D>,
    ) -> Result<OpeningFoldOutput<F, D>, AkitaError> {
        match source.poly {
            MultilinearPolynomial::Dense(poly) => OpeningFoldKernel::<
                DenseOpeningView<'_, F, D>,
                F,
                D,
            >::evaluate_and_fold(
                self, prepared, poly.opening_view()?, plan
            ),
            MultilinearPolynomial::OneHot(poly) => OpeningFoldKernel::<
                OneHotOpeningView<'_, F, D, I>,
                F,
                D,
            >::evaluate_and_fold(
                self, prepared, poly.opening_view()?, plan
            ),
        }
    }

    fn decompose_fold(
        &self,
        prepared: Option<&Self::PreparedSetup<D>>,
        source: MultilinearPolynomialView<'_, F, D, I>,
        plan: DecomposeFoldPlan<'_>,
    ) -> Result<DecomposeFoldWitness<F, D>, AkitaError> {
        match source.poly {
            MultilinearPolynomial::Dense(poly) => OpeningFoldKernel::<
                DenseOpeningView<'_, F, D>,
                F,
                D,
            >::decompose_fold(
                self, prepared, poly.opening_view()?, plan
            ),
            MultilinearPolynomial::OneHot(poly) => OpeningFoldKernel::<
                OneHotOpeningView<'_, F, D, I>,
                F,
                D,
            >::decompose_fold(
                self, prepared, poly.opening_view()?, plan
            ),
        }
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
        prepared: Option<&Self::PreparedSetup<D>>,
        source: MultilinearPolynomialBatchView<'_, F, D, I>,
        plan: DecomposeFoldBatchPlan<'_>,
    ) -> Result<Option<DecomposeFoldWitness<F, D>>, AkitaError> {
        let Some(first) = source.polys.first() else {
            return Ok(None);
        };
        match first {
            MultilinearPolynomial::Dense(_) => {
                let mut dense_polys = Vec::with_capacity(source.polys.len());
                for poly in source.polys {
                    match poly {
                        MultilinearPolynomial::Dense(inner) => dense_polys.push(inner),
                        MultilinearPolynomial::OneHot(_) => return Ok(None),
                    }
                }
                let dense_view = DensePoly::<F, D>::opening_batch(&dense_polys)?;
                OpeningBatchKernel::<DenseOpeningBatchView<'_, F, D>, F, D>::decompose_fold_batch(
                    self, prepared, dense_view, plan,
                )
            }
            MultilinearPolynomial::OneHot(_) => {
                let mut onehot_polys = Vec::with_capacity(source.polys.len());
                for poly in source.polys {
                    match poly {
                        MultilinearPolynomial::OneHot(inner) => onehot_polys.push(inner),
                        MultilinearPolynomial::Dense(_) => return Ok(None),
                    }
                }
                let onehot_view = OneHotPoly::<F, D, I>::opening_batch(&onehot_polys)?;
                OpeningBatchKernel::<OneHotOpeningBatchView<'_, F, D, I>, F, D>::decompose_fold_batch(
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
        prepared: Option<&Self::PreparedSetup<D>>,
        source: MultilinearPolynomialView<'_, F, D, I>,
        logical_point: &[E],
    ) -> Result<Vec<E>, AkitaError>
    where
        E: MulBaseUnreduced<F>,
    {
        match source.poly {
            MultilinearPolynomial::Dense(poly) => {
                TensorProjectionKernel::<DenseTensorView<'_, F, D>, F, E, D>::column_partials(
                    self,
                    prepared,
                    poly.tensor_view()?,
                    logical_point,
                )
            }
            MultilinearPolynomial::OneHot(poly) => {
                TensorProjectionKernel::<OneHotTensorView<'_, F, D, I>, F, E, D>::column_partials(
                    self,
                    prepared,
                    poly.tensor_view()?,
                    logical_point,
                )
            }
        }
    }

    fn packed_witness(
        &self,
        prepared: Option<&Self::PreparedSetup<D>>,
        source: MultilinearPolynomialView<'_, F, D, I>,
    ) -> Result<TensorPackedWitness<E>, AkitaError> {
        match source.poly {
            MultilinearPolynomial::Dense(poly) => {
                TensorProjectionKernel::<DenseTensorView<'_, F, D>, F, E, D>::packed_witness(
                    self,
                    prepared,
                    poly.tensor_view()?,
                )
            }
            MultilinearPolynomial::OneHot(poly) => {
                TensorProjectionKernel::<OneHotTensorView<'_, F, D, I>, F, E, D>::packed_witness(
                    self,
                    prepared,
                    poly.tensor_view()?,
                )
            }
        }
    }

    fn root_projection(
        &self,
        prepared: Option<&Self::PreparedSetup<D>>,
        source: MultilinearPolynomialView<'_, F, D, I>,
    ) -> Result<RootTensorProjectionPoly<F, D>, AkitaError>
    where
        E: FpExtEncoding<F>,
    {
        match source.poly {
            MultilinearPolynomial::Dense(poly) => {
                TensorProjectionKernel::<DenseTensorView<'_, F, D>, F, E, D>::root_projection(
                    self,
                    prepared,
                    poly.tensor_view()?,
                )
            }
            MultilinearPolynomial::OneHot(poly) => {
                TensorProjectionKernel::<OneHotTensorView<'_, F, D, I>, F, E, D>::root_projection(
                    self,
                    prepared,
                    poly.tensor_view()?,
                )
            }
        }
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
        prepared: Option<&Self::PreparedSetup<D>>,
        source: MultilinearPolynomialBatchView<'_, F, D, I>,
        logical_point: &[E],
    ) -> Result<Vec<Vec<E>>, AkitaError>
    where
        E: MulBaseUnreduced<F>,
    {
        let Some(first) = source.polys.first() else {
            return Ok(Vec::new());
        };
        match first {
            MultilinearPolynomial::Dense(_) => {
                let mut dense_polys = Vec::with_capacity(source.polys.len());
                for poly in source.polys {
                    match poly {
                        MultilinearPolynomial::Dense(inner) => dense_polys.push(inner),
                        MultilinearPolynomial::OneHot(_) => {
                            return source
                                .polys
                                .iter()
                                .map(|poly| {
                                    TensorProjectionKernel::<
                                        MultilinearPolynomialView<'_, F, D, I>,
                                        F,
                                        E,
                                        D,
                                    >::column_partials(
                                        self,
                                        prepared,
                                        poly.tensor_view()?,
                                        logical_point,
                                    )
                                })
                                .collect();
                        }
                    }
                }
                let dense_view = DensePoly::<F, D>::tensor_batch(&dense_polys)?;
                TensorProjectionBatchKernel::<DenseTensorBatchView<'_, F, D>, F, E, D>::column_partials_batch(
                    self,
                    prepared,
                    dense_view,
                    logical_point,
                )
            }
            MultilinearPolynomial::OneHot(_) => {
                let mut onehot_polys = Vec::with_capacity(source.polys.len());
                for poly in source.polys {
                    match poly {
                        MultilinearPolynomial::OneHot(inner) => onehot_polys.push(inner),
                        MultilinearPolynomial::Dense(_) => {
                            return source
                                .polys
                                .iter()
                                .map(|poly| {
                                    TensorProjectionKernel::<
                                        MultilinearPolynomialView<'_, F, D, I>,
                                        F,
                                        E,
                                        D,
                                    >::column_partials(
                                        self,
                                        prepared,
                                        poly.tensor_view()?,
                                        logical_point,
                                    )
                                })
                                .collect();
                        }
                    }
                }
                let onehot_view = OneHotPoly::<F, D, I>::tensor_batch(&onehot_polys)?;
                TensorProjectionBatchKernel::<OneHotTensorBatchView<'_, F, D, I>, F, E, D>::column_partials_batch(
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
        prepared: Option<&Self::PreparedSetup<D>>,
        source: MultilinearPolynomialBatchView<'_, F, D, I>,
        coeffs: &[E],
    ) -> Result<Option<SparseExtensionOpeningWitness<E>>, AkitaError> {
        let Some(first) = source.polys.first() else {
            return Ok(None);
        };
        match first {
            MultilinearPolynomial::Dense(_) => {
                let mut dense_polys = Vec::with_capacity(source.polys.len());
                for poly in source.polys {
                    match poly {
                        MultilinearPolynomial::Dense(inner) => dense_polys.push(inner),
                        MultilinearPolynomial::OneHot(_) => return Ok(None),
                    }
                }
                let dense_view = DensePoly::<F, D>::tensor_batch(&dense_polys)?;
                TensorProjectionBatchKernel::<DenseTensorBatchView<'_, F, D>, F, E, D>::sparse_linear_combination(
                    self,
                    prepared,
                    dense_view,
                    coeffs,
                )
            }
            MultilinearPolynomial::OneHot(_) => {
                let mut onehot_polys = Vec::with_capacity(source.polys.len());
                for poly in source.polys {
                    match poly {
                        MultilinearPolynomial::OneHot(inner) => onehot_polys.push(inner),
                        MultilinearPolynomial::Dense(_) => return Ok(None),
                    }
                }
                let onehot_view = OneHotPoly::<F, D, I>::tensor_batch(&onehot_polys)?;
                TensorProjectionBatchKernel::<OneHotTensorBatchView<'_, F, D, I>, F, E, D>::sparse_linear_combination(
                    self,
                    prepared,
                    onehot_view,
                    coeffs,
                )
            }
        }
    }
}

impl<F, const D: usize, I> AkitaPolyOps<F, D> for MultilinearPolynomial<F, D, I>
where
    F: FieldCore + CanonicalField + HasWide,
    I: OneHotIndex,
{
    fn num_ring_elems(&self) -> usize {
        match self {
            Self::Dense(poly) => <DensePoly<F, D> as AkitaPolyOps<F, D>>::num_ring_elems(poly),
            Self::OneHot(poly) => <OneHotPoly<F, D, I> as AkitaPolyOps<F, D>>::num_ring_elems(poly),
        }
    }

    fn num_vars(&self) -> usize {
        match self {
            Self::Dense(poly) => <DensePoly<F, D> as AkitaPolyOps<F, D>>::num_vars(poly),
            Self::OneHot(poly) => <OneHotPoly<F, D, I> as AkitaPolyOps<F, D>>::num_vars(poly),
        }
    }

    fn onehot_chunk_size(&self) -> Option<usize> {
        match self {
            Self::Dense(poly) => <DensePoly<F, D> as AkitaPolyOps<F, D>>::onehot_chunk_size(poly),
            Self::OneHot(poly) => {
                <OneHotPoly<F, D, I> as AkitaPolyOps<F, D>>::onehot_chunk_size(poly)
            }
        }
    }

    fn base_evals(&self) -> Result<Vec<F>, AkitaError> {
        match self {
            Self::Dense(poly) => <DensePoly<F, D> as AkitaPolyOps<F, D>>::base_evals(poly),
            Self::OneHot(poly) => <OneHotPoly<F, D, I> as AkitaPolyOps<F, D>>::base_evals(poly),
        }
    }

    fn fold_blocks(&self, scalars: &[F], block_len: usize) -> Vec<CyclotomicRing<F, D>> {
        match self {
            Self::Dense(poly) => {
                <DensePoly<F, D> as AkitaPolyOps<F, D>>::fold_blocks(poly, scalars, block_len)
            }
            Self::OneHot(poly) => {
                <OneHotPoly<F, D, I> as AkitaPolyOps<F, D>>::fold_blocks(poly, scalars, block_len)
            }
        }
    }

    fn fold_blocks_ring(
        &self,
        scalars: &[CyclotomicRing<F, D>],
        block_len: usize,
    ) -> Vec<CyclotomicRing<F, D>> {
        match self {
            Self::Dense(poly) => {
                <DensePoly<F, D> as AkitaPolyOps<F, D>>::fold_blocks_ring(poly, scalars, block_len)
            }
            Self::OneHot(poly) => <OneHotPoly<F, D, I> as AkitaPolyOps<F, D>>::fold_blocks_ring(
                poly, scalars, block_len,
            ),
        }
    }

    fn evaluate_extension<E>(&self, point: &[E]) -> Result<E, AkitaError>
    where
        E: ExtField<F>,
    {
        match self {
            Self::Dense(poly) => {
                <DensePoly<F, D> as AkitaPolyOps<F, D>>::evaluate_extension(poly, point)
            }
            Self::OneHot(poly) => {
                <OneHotPoly<F, D, I> as AkitaPolyOps<F, D>>::evaluate_extension(poly, point)
            }
        }
    }

    fn tensor_extension_column_partials<E>(&self, logical_point: &[E]) -> Result<Vec<E>, AkitaError>
    where
        E: MulBaseUnreduced<F>,
    {
        match self {
            Self::Dense(poly) => {
                <DensePoly<F, D> as AkitaPolyOps<F, D>>::tensor_extension_column_partials(
                    poly,
                    logical_point,
                )
            }
            Self::OneHot(poly) => {
                <OneHotPoly<F, D, I> as AkitaPolyOps<F, D>>::tensor_extension_column_partials(
                    poly,
                    logical_point,
                )
            }
        }
    }

    fn tensor_extension_column_partials_batch<E>(
        polys: &[&Self],
        logical_point: &[E],
    ) -> Result<Vec<Vec<E>>, AkitaError>
    where
        E: MulBaseUnreduced<F>,
    {
        let Some(first) = polys.first() else {
            return Ok(Vec::new());
        };
        match first {
            Self::Dense(_) => {
                let mut dense = Vec::with_capacity(polys.len());
                for poly in polys {
                    match poly {
                        Self::Dense(inner) => dense.push(inner),
                        Self::OneHot(_) => {
                            return polys
                                .iter()
                                .map(|poly| poly.tensor_extension_column_partials(logical_point))
                                .collect();
                        }
                    }
                }
                <DensePoly<F, D> as AkitaPolyOps<F, D>>::tensor_extension_column_partials_batch(
                    &dense,
                    logical_point,
                )
            }
            Self::OneHot(_) => {
                let mut onehot = Vec::with_capacity(polys.len());
                for poly in polys {
                    match poly {
                        Self::OneHot(inner) => onehot.push(inner),
                        Self::Dense(_) => {
                            return polys
                                .iter()
                                .map(|poly| poly.tensor_extension_column_partials(logical_point))
                                .collect();
                        }
                    }
                }
                <OneHotPoly<F, D, I> as AkitaPolyOps<F, D>>::tensor_extension_column_partials_batch(
                    &onehot,
                    logical_point,
                )
            }
        }
    }

    fn tensor_packed_extension_sparse_evals<E>(
        &self,
    ) -> Result<Option<SparseExtensionOpeningWitness<E>>, AkitaError>
    where
        E: ExtField<F>,
    {
        match self {
            Self::Dense(poly) => {
                <DensePoly<F, D> as AkitaPolyOps<F, D>>::tensor_packed_extension_sparse_evals::<E>(
                    poly,
                )
            }
            Self::OneHot(poly) => {
                <OneHotPoly<F, D, I> as AkitaPolyOps<F, D>>::tensor_packed_extension_sparse_evals::<E>(
                    poly,
                )
            }
        }
    }

    fn tensor_packed_extension_sparse_linear_combination<E>(
        polys: &[&Self],
        coeffs: &[E],
    ) -> Result<Option<SparseExtensionOpeningWitness<E>>, AkitaError>
    where
        E: ExtField<F>,
    {
        let Some(first) = polys.first() else {
            return Ok(None);
        };
        match first {
            Self::Dense(_) => {
                let mut dense = Vec::with_capacity(polys.len());
                for poly in polys {
                    match poly {
                        Self::Dense(inner) => dense.push(inner),
                        Self::OneHot(_) => return Ok(None),
                    }
                }
                <DensePoly<F, D> as AkitaPolyOps<F, D>>::tensor_packed_extension_sparse_linear_combination(
                    &dense,
                    coeffs,
                )
            }
            Self::OneHot(_) => {
                let mut onehot = Vec::with_capacity(polys.len());
                for poly in polys {
                    match poly {
                        Self::OneHot(inner) => onehot.push(inner),
                        Self::Dense(_) => return Ok(None),
                    }
                }
                <OneHotPoly<F, D, I> as AkitaPolyOps<F, D>>::tensor_packed_extension_sparse_linear_combination(
                    &onehot,
                    coeffs,
                )
            }
        }
    }

    fn decompose_fold(
        &self,
        challenges: &[SparseChallenge],
        block_len: usize,
        num_digits: usize,
        log_basis: u32,
    ) -> DecomposeFoldWitness<F, D> {
        match self {
            Self::Dense(poly) => <DensePoly<F, D> as AkitaPolyOps<F, D>>::decompose_fold(
                poly, challenges, block_len, num_digits, log_basis,
            ),
            Self::OneHot(poly) => <OneHotPoly<F, D, I> as AkitaPolyOps<F, D>>::decompose_fold(
                poly, challenges, block_len, num_digits, log_basis,
            ),
        }
    }

    fn decompose_fold_batched(
        polys: &[&Self],
        challenges: &[SparseChallenge],
        block_len: usize,
        num_digits: usize,
        log_basis: u32,
    ) -> Option<DecomposeFoldWitness<F, D>> {
        let first = polys.first()?;
        match first {
            Self::Dense(_) => {
                let mut dense = Vec::with_capacity(polys.len());
                for poly in polys {
                    match poly {
                        Self::Dense(inner) => dense.push(inner),
                        Self::OneHot(_) => return None,
                    }
                }
                <DensePoly<F, D> as AkitaPolyOps<F, D>>::decompose_fold_batched(
                    &dense, challenges, block_len, num_digits, log_basis,
                )
            }
            Self::OneHot(_) => {
                let mut onehot = Vec::with_capacity(polys.len());
                for poly in polys {
                    match poly {
                        Self::OneHot(inner) => onehot.push(inner),
                        Self::Dense(_) => return None,
                    }
                }
                <OneHotPoly<F, D, I> as AkitaPolyOps<F, D>>::decompose_fold_batched(
                    &onehot, challenges, block_len, num_digits, log_basis,
                )
            }
        }
    }

    fn decompose_fold_tensor_batched(
        polys: &[&Self],
        tensor: &TensorChallenges,
        block_len: usize,
        num_digits: usize,
        log_basis: u32,
    ) -> Result<Option<DecomposeFoldWitness<F, D>>, AkitaError> {
        let Some(first) = polys.first() else {
            return Ok(None);
        };
        match first {
            Self::Dense(_) => {
                let mut dense = Vec::with_capacity(polys.len());
                for poly in polys {
                    match poly {
                        Self::Dense(inner) => dense.push(inner),
                        Self::OneHot(_) => return Ok(None),
                    }
                }
                <DensePoly<F, D> as AkitaPolyOps<F, D>>::decompose_fold_tensor_batched(
                    &dense, tensor, block_len, num_digits, log_basis,
                )
            }
            Self::OneHot(_) => {
                let mut onehot = Vec::with_capacity(polys.len());
                for poly in polys {
                    match poly {
                        Self::OneHot(inner) => onehot.push(inner),
                        Self::Dense(_) => return Ok(None),
                    }
                }
                <OneHotPoly<F, D, I> as AkitaPolyOps<F, D>>::decompose_fold_tensor_batched(
                    &onehot, tensor, block_len, num_digits, log_basis,
                )
            }
        }
    }

    fn commit_inner<B>(
        &self,
        backend: &B,
        prepared: &B::PreparedSetup<D>,
        n_a: usize,
        block_len: usize,
        num_blocks: usize,
        num_digits_commit: usize,
        num_digits_open: usize,
        log_basis: u32,
    ) -> Result<CommitInnerWitness<F, D>, AkitaError>
    where
        F: CanonicalField,
        B: CommitmentComputeBackend<F>,
    {
        match self {
            Self::Dense(poly) => <DensePoly<F, D> as AkitaPolyOps<F, D>>::commit_inner(
                poly,
                backend,
                prepared,
                n_a,
                block_len,
                num_blocks,
                num_digits_commit,
                num_digits_open,
                log_basis,
            ),
            Self::OneHot(poly) => <OneHotPoly<F, D, I> as AkitaPolyOps<F, D>>::commit_inner(
                poly,
                backend,
                prepared,
                n_a,
                block_len,
                num_blocks,
                num_digits_commit,
                num_digits_open,
                log_basis,
            ),
        }
    }

    fn direct_root_witness(&self) -> Result<CleartextWitnessProof<F>, AkitaError> {
        match self {
            Self::Dense(poly) => <DensePoly<F, D> as AkitaPolyOps<F, D>>::direct_root_witness(poly),
            Self::OneHot(poly) => {
                <OneHotPoly<F, D, I> as AkitaPolyOps<F, D>>::direct_root_witness(poly)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_field::{FpExt4, Prime24Offset3};

    fn sample_dense<const D: usize>() -> DensePoly<Prime24Offset3, D> {
        let num_vars = 5;
        let evals = (0..(1usize << num_vars))
            .map(|idx| Prime24Offset3::from_canonical_u128_reduced(17 * idx as u128 + 9))
            .collect::<Vec<_>>();
        DensePoly::from_field_evals(num_vars, &evals).unwrap()
    }

    fn sample_onehot<const D: usize>() -> OneHotPoly<Prime24Offset3, D> {
        OneHotPoly::<Prime24Offset3, D>::new(
            8,
            vec![
                Some(0usize),
                Some(7),
                None,
                Some(3),
                Some(5),
                Some(1),
                None,
                Some(6),
            ],
        )
        .unwrap()
    }

    fn sample_point<E: ExtField<Prime24Offset3>>(num_vars: usize) -> Vec<E> {
        (0..num_vars)
            .map(|idx| {
                E::from_base_slice(&[
                    Prime24Offset3::from_canonical_u128_reduced(5 * idx as u128 + 2),
                    Prime24Offset3::from_canonical_u128_reduced(5 * idx as u128 + 3),
                    Prime24Offset3::from_canonical_u128_reduced(5 * idx as u128 + 5),
                    Prime24Offset3::from_canonical_u128_reduced(5 * idx as u128 + 7),
                ])
            })
            .collect()
    }

    #[test]
    fn multilinear_kernel_homogeneous_dense_tensor_batch_matches_inner() {
        type F = Prime24Offset3;
        type E = FpExt4<F>;
        const D: usize = 16;

        let dense0 = sample_dense::<D>();
        let dense1 = sample_dense::<D>();
        let num_vars = RootPolyShape::num_vars(&dense0);
        let wrapped = [
            MultilinearPolynomial::dense(dense0),
            MultilinearPolynomial::dense(dense1),
        ];
        let wrapped_refs = [&wrapped[0], &wrapped[1]];
        let point = sample_point::<E>(num_vars);
        let backend = CpuBackend;

        let inner_refs: Vec<&DensePoly<F, D>> = wrapped
            .iter()
            .map(|poly| match poly {
                MultilinearPolynomial::Dense(dense) => dense,
                MultilinearPolynomial::OneHot(_) => unreachable!(),
            })
            .collect();
        let dense_view = DensePoly::<F, D>::tensor_batch(&inner_refs).unwrap();
        let expected = TensorProjectionBatchKernel::<DenseTensorBatchView<'_, F, D>, F, E, D>::column_partials_batch(
            &backend,
            None,
            dense_view,
            &point,
        )
        .unwrap();
        let batch_view = MultilinearPolynomial::<F, D>::tensor_batch(&wrapped_refs).unwrap();
        let got = TensorProjectionBatchKernel::<
            MultilinearPolynomialBatchView<'_, F, D>,
            F,
            E,
            D,
        >::column_partials_batch(&backend, None, batch_view, &point)
        .unwrap();
        assert_eq!(got, expected);
    }

    #[test]
    fn multilinear_kernel_homogeneous_onehot_tensor_batch_matches_inner() {
        type F = Prime24Offset3;
        type E = FpExt4<F>;
        const D: usize = 16;

        let onehot0 = sample_onehot::<D>();
        let onehot1 = sample_onehot::<D>();
        let num_vars = RootPolyShape::num_vars(&onehot0);
        let wrapped = [
            MultilinearPolynomial::onehot(onehot0),
            MultilinearPolynomial::onehot(onehot1),
        ];
        let wrapped_refs = [&wrapped[0], &wrapped[1]];
        let point = sample_point::<E>(num_vars);
        let backend = CpuBackend;

        let inner_refs: Vec<&OneHotPoly<F, D>> = wrapped
            .iter()
            .map(|poly| match poly {
                MultilinearPolynomial::OneHot(onehot) => onehot,
                MultilinearPolynomial::Dense(_) => unreachable!(),
            })
            .collect();
        let onehot_view = OneHotPoly::<F, D>::tensor_batch(&inner_refs).unwrap();
        let expected = TensorProjectionBatchKernel::<
            OneHotTensorBatchView<'_, F, D>,
            F,
            E,
            D,
        >::column_partials_batch(&backend, None, onehot_view, &point)
        .unwrap();
        let batch_view = MultilinearPolynomial::<F, D>::tensor_batch(&wrapped_refs).unwrap();
        let got = TensorProjectionBatchKernel::<
            MultilinearPolynomialBatchView<'_, F, D>,
            F,
            E,
            D,
        >::column_partials_batch(&backend, None, batch_view, &point)
        .unwrap();
        assert_eq!(got, expected);
    }

    #[test]
    fn multilinear_kernel_mixed_batch_column_partials_falls_back_per_poly() {
        type F = Prime24Offset3;
        type E = FpExt4<F>;
        const D: usize = 16;

        let onehot = sample_onehot::<D>();
        let num_vars = RootPolyShape::num_vars(&onehot);
        let evals = (0..(1usize << num_vars))
            .map(|idx| Prime24Offset3::from_canonical_u128_reduced(17 * idx as u128 + 9))
            .collect::<Vec<_>>();
        let dense = DensePoly::from_field_evals(num_vars, &evals).unwrap();
        let wrapped = [
            MultilinearPolynomial::dense(dense),
            MultilinearPolynomial::onehot(onehot),
        ];
        let wrapped_refs = [&wrapped[0], &wrapped[1]];
        let point = sample_point::<E>(num_vars);
        let backend = CpuBackend;

        let expected = wrapped_refs
            .iter()
            .map(|poly| {
                let view = poly.tensor_view().unwrap();
                TensorProjectionKernel::<MultilinearPolynomialView<'_, F, D>, F, E, D>::column_partials(
                    &backend,
                    None,
                    view,
                    &point,
                )
            })
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let batch_view = MultilinearPolynomial::<F, D>::tensor_batch(&wrapped_refs).unwrap();
        let got = TensorProjectionBatchKernel::<
            MultilinearPolynomialBatchView<'_, F, D>,
            F,
            E,
            D,
        >::column_partials_batch(&backend, None, batch_view, &point)
        .unwrap();
        assert_eq!(got, expected);
    }

    #[test]
    fn multilinear_kernel_mixed_batch_sparse_linear_combination_returns_none() {
        type F = Prime24Offset3;
        type E = FpExt4<F>;
        const D: usize = 16;

        let onehot = sample_onehot::<D>();
        let num_vars = RootPolyShape::num_vars(&onehot);
        let evals = (0..(1usize << num_vars))
            .map(|idx| Prime24Offset3::from_canonical_u128_reduced(17 * idx as u128 + 9))
            .collect::<Vec<_>>();
        let dense = DensePoly::from_field_evals(num_vars, &evals).unwrap();
        let wrapped = [
            MultilinearPolynomial::dense(dense),
            MultilinearPolynomial::onehot(onehot),
        ];
        let wrapped_refs = [&wrapped[0], &wrapped[1]];
        let coeffs = vec![E::one(), E::one()];
        let backend = CpuBackend;

        let batch_view = MultilinearPolynomial::<F, D>::tensor_batch(&wrapped_refs).unwrap();
        let got = TensorProjectionBatchKernel::<
            MultilinearPolynomialBatchView<'_, F, D>,
            F,
            E,
            D,
        >::sparse_linear_combination(&backend, None, batch_view, &coeffs)
        .unwrap();
        assert!(got.is_none());
    }
}
