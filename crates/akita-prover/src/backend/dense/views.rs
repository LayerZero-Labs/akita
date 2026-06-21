//! Dense polynomial source views and capability traits.

use super::poly::DensePoly;
use crate::compute::{
    DirectRootWitnessSource, RootCommitSource, RootOpeningSource, RootPolyShape, RootTensorSource,
};
use akita_field::{AkitaError, FieldCore};
use akita_types::{CleartextWitnessProof, FlatRingVec};

/// Borrowed single-polynomial view over dense ring storage.
///
/// One view type backs the commit, opening-fold, and tensor-projection kernels;
/// the kernel trait it is passed to selects the operation.
#[derive(Debug, Clone, Copy)]
pub struct DenseView<'a, F: FieldCore, const D: usize> {
    pub(super) poly: &'a DensePoly<F, D>,
}

/// Same-point batch view over several dense polynomials.
#[derive(Debug, Clone, Copy)]
pub struct DenseBatchView<'a, F: FieldCore, const D: usize> {
    pub(super) polys: &'a [&'a DensePoly<F, D>],
}

impl<F, const D: usize> RootPolyShape<F, D> for DensePoly<F, D>
where
    F: FieldCore,
{
    fn num_ring_elems(&self) -> usize {
        self.coeffs.len()
    }

    fn num_vars(&self) -> usize {
        self.num_vars
    }
}

impl<F, const D: usize> RootCommitSource<F, D> for DensePoly<F, D>
where
    F: FieldCore,
{
    type CommitView<'a>
        = DenseView<'a, F, D>
    where
        Self: 'a;

    fn commit_view(&self) -> Result<Self::CommitView<'_>, AkitaError> {
        Ok(DenseView { poly: self })
    }
}

impl<F, const D: usize> RootOpeningSource<F, D> for DensePoly<F, D>
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
        Ok(DenseView { poly: self })
    }

    fn opening_batch<'a>(polys: &'a [&'a Self]) -> Result<Self::OpeningBatchView<'a>, AkitaError> {
        Ok(DenseBatchView { polys })
    }
}

impl<F, const D: usize> RootTensorSource<F, D> for DensePoly<F, D>
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
        Ok(DenseView { poly: self })
    }

    fn tensor_batch<'a>(polys: &'a [&'a Self]) -> Result<Self::TensorBatchView<'a>, AkitaError> {
        Ok(DenseBatchView { polys })
    }
}

impl<F, const D: usize> DirectRootWitnessSource<F, D> for DensePoly<F, D>
where
    F: FieldCore,
{
    fn direct_root_witness(&self) -> Result<CleartextWitnessProof<F>, AkitaError> {
        let live_len = 1usize.checked_shl(self.num_vars as u32).ok_or_else(|| {
            AkitaError::InvalidInput(format!("2^{} does not fit usize", self.num_vars))
        })?;
        let mut coeffs = Vec::with_capacity(live_len);
        let mut remaining = live_len;
        for ring in &self.coeffs {
            let take = remaining.min(D);
            coeffs.extend_from_slice(&ring.coefficients()[..take]);
            remaining -= take;
            if remaining == 0 {
                break;
            }
        }
        Ok(CleartextWitnessProof::FieldElements(
            FlatRingVec::from_coeffs(coeffs),
        ))
    }
}
