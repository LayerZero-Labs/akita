//! The multilinear-polynomial wrapper enum, its borrowed dispatch views, and
//! the source-trait impls. The `CpuBackend` kernel impls live in [`super::ops`].

use akita_field::unreduced::HasWide;
use akita_field::{
    AkitaError, CanonicalField, ExtField, FieldCore, FromPrimitiveInt, MulBaseUnreduced,
};
use akita_types::CleartextWitnessProof;

use crate::compute::{
    CpuBackend, CpuPreparedSetup, DirectRootWitnessSource, RootCommitSource, RootOpeningSource,
    RootPolyMeta, RootPolyShape, RootTensorSource, TensorProjectionKernel,
};
use crate::{DensePoly, OneHotIndex, OneHotPoly};

/// Owned multilinear-polynomial wrapper for dense and one-hot batches.
///
/// This is an Akita-owned private sum type (allowed by the polyops cutover
/// spec): it erases `DensePoly` vs `OneHotPoly` for heterogeneous batches while
/// exposing the source-typed view/kernel boundary (`RootCommitSource`,
/// `RootOpeningSource`, `RootTensorSource`, and matching `CpuBackend` kernels).
/// Wrappers take ownership of the inner polynomial by move so `P` has no lifetime
/// parameter and participates in generic `commit<P, B>` like `DensePoly`.
#[derive(Debug, Clone)]
pub enum MultilinearPolynomial<F: FieldCore, I: OneHotIndex = usize> {
    /// Dense multilinear polynomial.
    Dense(DensePoly<F>),
    /// One-hot multilinear polynomial.
    OneHot(OneHotPoly<F, I>),
}

impl<F: FieldCore, I: OneHotIndex> MultilinearPolynomial<F, I> {
    /// Wrap a dense polynomial.
    pub fn dense(poly: DensePoly<F>) -> Self {
        Self::Dense(poly)
    }

    /// Wrap a one-hot polynomial.
    pub fn onehot(poly: OneHotPoly<F, I>) -> Self {
        Self::OneHot(poly)
    }
}

/// Borrowed dispatch view for an Akita-owned multilinear root wrapper at
/// dimension `D`.
#[derive(Debug, Clone, Copy)]
pub struct MultilinearPolynomialView<'a, F: FieldCore, const D: usize, I: OneHotIndex = usize> {
    poly: &'a MultilinearPolynomial<F, I>,
}

/// Same-point batch dispatch view over multilinear root wrappers at
/// dimension `D`.
#[derive(Debug, Clone, Copy)]
pub struct MultilinearPolynomialBatchView<'a, F: FieldCore, const D: usize, I: OneHotIndex = usize>
{
    polys: &'a [&'a MultilinearPolynomial<F, I>],
}

impl<'a, F, const D: usize, I> MultilinearPolynomialView<'a, F, D, I>
where
    F: FieldCore,
    I: OneHotIndex,
{
    pub(super) fn dispatch<T>(
        self,
        dense: impl FnOnce(&DensePoly<F>) -> Result<T, AkitaError>,
        onehot: impl FnOnce(&OneHotPoly<F, I>) -> Result<T, AkitaError>,
    ) -> Result<T, AkitaError> {
        match self.poly {
            MultilinearPolynomial::Dense(poly) => dense(poly),
            MultilinearPolynomial::OneHot(poly) => onehot(poly),
        }
    }
}

impl<'a, F, const D: usize, I> MultilinearPolynomialBatchView<'a, F, D, I>
where
    F: FieldCore,
    I: OneHotIndex,
{
    pub(super) fn polys(self) -> &'a [&'a MultilinearPolynomial<F, I>] {
        self.polys
    }

    pub(super) fn homogeneous_dense_polys(self) -> Option<Vec<&'a DensePoly<F>>> {
        let mut dense = Vec::with_capacity(self.polys.len());
        for poly in self.polys {
            match poly {
                MultilinearPolynomial::Dense(inner) => dense.push(inner),
                MultilinearPolynomial::OneHot(_) => return None,
            }
        }
        Some(dense)
    }

    pub(super) fn homogeneous_onehot_polys(self) -> Option<Vec<&'a OneHotPoly<F, I>>> {
        let mut onehot = Vec::with_capacity(self.polys.len());
        for poly in self.polys {
            match poly {
                MultilinearPolynomial::OneHot(inner) => onehot.push(inner),
                MultilinearPolynomial::Dense(_) => return None,
            }
        }
        Some(onehot)
    }

    pub(super) fn column_partials_per_poly<E>(
        self,
        backend: &CpuBackend,
        prepared: Option<&CpuPreparedSetup<F>>,
        logical_point: &[E],
    ) -> Result<Vec<Vec<E>>, AkitaError>
    where
        F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide,
        E: ExtField<F> + MulBaseUnreduced<F>,
    {
        self.polys
            .iter()
            .map(|poly| {
                TensorProjectionKernel::<MultilinearPolynomialView<'_, F, D, I>, F, E, D>::column_partials(
                    backend,
                    prepared,
                    poly.tensor_view()?,
                    logical_point,
                )
            })
            .collect()
    }
}

impl<F, I> RootPolyMeta<F> for MultilinearPolynomial<F, I>
where
    F: FieldCore,
    I: OneHotIndex,
{
    fn num_ring_elems(&self) -> usize {
        match self {
            Self::Dense(poly) => RootPolyMeta::num_ring_elems(poly),
            Self::OneHot(poly) => RootPolyMeta::num_ring_elems(poly),
        }
    }

    fn num_vars(&self) -> usize {
        match self {
            Self::Dense(poly) => RootPolyMeta::num_vars(poly),
            Self::OneHot(poly) => RootPolyMeta::num_vars(poly),
        }
    }

    fn onehot_chunk_size(&self) -> Option<usize> {
        match self {
            Self::Dense(_) => None,
            Self::OneHot(poly) => RootPolyMeta::onehot_chunk_size(poly),
        }
    }
}

impl<F, const D: usize, I> RootPolyShape<F, D> for MultilinearPolynomial<F, I>
where
    F: FieldCore,
    I: OneHotIndex,
{
    fn num_ring_elems(&self) -> usize {
        match self {
            Self::Dense(poly) => RootPolyShape::<F, D>::num_ring_elems(poly),
            Self::OneHot(poly) => RootPolyShape::<F, D>::num_ring_elems(poly),
        }
    }

    fn num_vars(&self) -> usize {
        match self {
            Self::Dense(poly) => RootPolyShape::<F, D>::num_vars(poly),
            Self::OneHot(poly) => RootPolyShape::<F, D>::num_vars(poly),
        }
    }

    fn onehot_chunk_size(&self) -> Option<usize> {
        match self {
            Self::Dense(_) => None,
            Self::OneHot(poly) => RootPolyShape::<F, D>::onehot_chunk_size(poly),
        }
    }
}

impl<F, const D: usize, I> RootCommitSource<F, D> for MultilinearPolynomial<F, I>
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

impl<F, const D: usize, I> RootOpeningSource<F, D> for MultilinearPolynomial<F, I>
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

impl<F, const D: usize, I> RootTensorSource<F, D> for MultilinearPolynomial<F, I>
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

impl<F, const D: usize, I> DirectRootWitnessSource<F, D> for MultilinearPolynomial<F, I>
where
    F: FieldCore + CanonicalField,
    I: OneHotIndex,
{
    fn direct_root_witness(&self) -> Result<CleartextWitnessProof<F>, AkitaError> {
        match self {
            Self::Dense(poly) => DirectRootWitnessSource::<F, D>::direct_root_witness(poly),
            Self::OneHot(poly) => DirectRootWitnessSource::<F, D>::direct_root_witness(poly),
        }
    }
}
