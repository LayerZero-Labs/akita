//! Dense polynomial source views and capability traits.
//!
//! [`DensePoly`] storage is D-free; the views here are the const-D
//! kernel-entry types. View construction validates the requested ring
//! dimension against the flat storage (via [`DensePoly::ring_coeffs`]) so
//! kernels can trust the D-view afterwards.

use super::poly::DensePoly;
use crate::compute::{RootCommitSource, RootOpeningSource, RootPolyMeta, RootPolyShape};
use akita_error::AkitaError;
use jolt_field::Field;

/// Borrowed single-polynomial view over dense ring storage at dimension `D`.
///
/// One view type backs the commit and opening-fold kernels; the kernel trait it
/// is passed to selects the operation.
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

impl<F, const D: usize> RootCommitSource<F, D> for DensePoly<F>
where
    F: Field,
{
    type CommitView<'a>
        = DenseView<'a, F, D>
    where
        Self: 'a;

    fn commit_view(&self) -> Result<Self::CommitView<'_>, AkitaError> {
        self.ring_coeffs::<D>()?;
        Ok(DenseView { poly: self })
    }

    /// Exact scan of the committed ring view.
    ///
    /// A dense source carries arbitrary field elements, so this is the one root
    /// representation that can exceed a bounded schedule's digit envelope. The
    /// scan covers the same coefficients the commit view decomposes, physical
    /// zero padding included (padding is centered zero and cannot raise either
    /// reach).
    fn committed_centered_reach(
        &self,
        modulus: u128,
        centering_threshold: u128,
    ) -> Result<(u128, u128), AkitaError>
    where
        F: jolt_field::CanonicalEncoding,
    {
        // `ring_coeffs` both validates `D` and pins the live prefix the commit
        // kernel reads, so scanning its exact flat span keeps the check and the
        // decomposition over the same coefficients.
        let live_coeffs = self.ring_coeffs::<D>()?.len() * D;
        Ok(crate::compute::centered_reach_of_field_coeffs(
            &self.field_coeffs()[..live_coeffs],
            modulus,
            centering_threshold,
        ))
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
