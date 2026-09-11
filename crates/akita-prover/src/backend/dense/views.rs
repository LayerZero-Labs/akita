//! Dense polynomial source views and capability traits.
//!
//! [`DensePoly`] storage is D-free; the views here are the const-D
//! kernel-entry types. View construction validates the requested ring
//! dimension against the flat storage (via [`DensePoly::ring_coeffs`]) so
//! kernels can trust the D-view afterwards.

use super::poly::DensePoly;
use crate::compute::{RootOpeningSource, RootPolyMeta, RootPolyShape};
use akita_error::AkitaError;
use jolt_field::Field;

/// Borrowed single-polynomial view over dense ring storage at dimension `D`.
///
/// Opening-fold view over a dense polynomial.
#[derive(Debug, Clone, Copy)]
pub struct DenseView<'a, F: Field, const D: usize> {
    pub(super) poly: &'a DensePoly<F>,
}

/// Same-point batch view over several dense polynomials at dimension `D`.
#[derive(Debug, Clone, Copy)]
pub struct DenseBatchView<'a, F: Field, const D: usize> {
    pub(super) polys: &'a [&'a DensePoly<F>],
}

impl<F> RootPolyMeta<F> for DensePoly<F>
where
    F: Field,
{
    fn num_vars(&self) -> usize {
        self.num_vars
    }
}

impl<F, const D: usize> RootPolyShape<F, D> for DensePoly<F>
where
    F: Field,
{
    fn num_ring_elems(&self) -> usize {
        self.num_ring_elems_at(D)
    }

    fn num_vars(&self) -> usize {
        self.num_vars
    }
}

impl<F, const D: usize> RootOpeningSource<F, D> for DensePoly<F>
where
    F: Field,
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
