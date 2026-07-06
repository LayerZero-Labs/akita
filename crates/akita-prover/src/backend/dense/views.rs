//! Dense polynomial source views and capability traits.
//!
//! [`DensePoly`] storage is D-free; the views here are the const-D
//! kernel-entry types. View construction validates the requested ring
//! dimension against the flat storage (via [`DensePoly::ring_coeffs`]) so
//! kernels can trust the D-view afterwards.

use super::poly::DensePoly;
use crate::compute::{
    DirectRootWitnessSource, RootCommitSource, RootOpeningSource, RootPolyMeta, RootPolyShape,
    RootTensorSource,
};
use akita_field::{AkitaError, FieldCore};
use akita_types::{CleartextWitnessProof, RingVec};

/// Borrowed single-polynomial view over dense ring storage at dimension `D`.
///
/// One view type backs the commit, opening-fold, and tensor-projection kernels;
/// the kernel trait it is passed to selects the operation.
#[derive(Debug, Clone, Copy)]
pub struct DenseView<'a, F: FieldCore, const D: usize> {
    pub(super) poly: &'a DensePoly<F>,
}

/// Same-point batch view over several dense polynomials at dimension `D`.
#[derive(Debug, Clone, Copy)]
pub struct DenseBatchView<'a, F: FieldCore, const D: usize> {
    pub(super) polys: &'a [&'a DensePoly<F>],
}

impl<F> RootPolyMeta<F> for DensePoly<F>
where
    F: FieldCore,
{
    fn num_ring_elems(&self) -> usize {
        self.meta_ring_elems()
    }

    fn num_vars(&self) -> usize {
        self.num_vars
    }
}

impl<F, const D: usize> RootPolyShape<F, D> for DensePoly<F>
where
    F: FieldCore,
{
    fn num_ring_elems(&self) -> usize {
        self.num_ring_elems_at(D)
    }

    fn num_vars(&self) -> usize {
        self.num_vars
    }
}

impl<F, const D: usize> RootCommitSource<F, D> for DensePoly<F>
where
    F: FieldCore,
{
    type CommitView<'a>
        = DenseView<'a, F, D>
    where
        Self: 'a;

    fn commit_view(&self) -> Result<Self::CommitView<'_>, AkitaError> {
        self.ring_coeffs::<D>()?;
        Ok(DenseView { poly: self })
    }
}

impl<F, const D: usize> RootOpeningSource<F, D> for DensePoly<F>
where
    F: FieldCore,
{
    type OpeningView<'a>
        = DenseView<'a, F, D>
    where
        Self: 'a;

    type OpeningBatchView<'a>
        = DenseBatchView<'a, F, D>
    where
        Self: 'a;

    fn opening_view(&self) -> Result<Self::OpeningView<'_>, AkitaError> {
        self.ring_coeffs::<D>()?;
        Ok(DenseView { poly: self })
    }

    fn opening_batch<'a>(polys: &'a [&'a Self]) -> Result<Self::OpeningBatchView<'a>, AkitaError> {
        for poly in polys {
            poly.ring_coeffs::<D>()?;
        }
        Ok(DenseBatchView { polys })
    }
}

impl<F, const D: usize> RootTensorSource<F, D> for DensePoly<F>
where
    F: FieldCore,
{
    type TensorView<'a>
        = DenseView<'a, F, D>
    where
        Self: 'a;

    type TensorBatchView<'a>
        = DenseBatchView<'a, F, D>
    where
        Self: 'a;

    fn tensor_view(&self) -> Result<Self::TensorView<'_>, AkitaError> {
        self.ring_coeffs::<D>()?;
        Ok(DenseView { poly: self })
    }

    fn tensor_batch<'a>(polys: &'a [&'a Self]) -> Result<Self::TensorBatchView<'a>, AkitaError> {
        for poly in polys {
            poly.ring_coeffs::<D>()?;
        }
        Ok(DenseBatchView { polys })
    }
}

impl<F, const D: usize> DirectRootWitnessSource<F, D> for DensePoly<F>
where
    F: FieldCore,
{
    fn direct_root_witness(&self) -> Result<CleartextWitnessProof<F>, AkitaError> {
        let live_len = self.live_coeff_len()?;
        Ok(CleartextWitnessProof::FieldElements(RingVec::from_coeffs(
            self.field_coeffs()[..live_len].to_vec(),
        )))
    }
}
