//! Shared field and base-profile NTT backings for one logical public matrix.

use core::ops::Range;
use std::sync::Arc;

use akita_algebra::{CanonicalEncoding, CyclotomicCrtNtt, CyclotomicRing, Field};
use akita_error::{checked, AkitaError};
use akita_params::{MatrixWindow, RingMatrixView};
use jolt_field::WithAccumulator;

use super::{PreparedNttBaseView, PreparedNttCache};

impl<const D: usize> PreparedNttCache<D> {
    /// Apply the public transposed product to a rectangle in this prefix.
    /// Rejects unsupported representations, missing negacyclic rows, and windows
    /// outside the prefix. The matrix and output field must be the same field.
    pub fn mat_vec_transposed<F: Field + CanonicalEncoding>(
        &self,
        window: &MatrixWindow,
        weights: &[CyclotomicRing<F, D>],
    ) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError> {
        macro_rules! dispatch {
            ($view:expr) => {
                if let Some(view) = $view {
                    let rows = window_rows(view, window)?;
                    if weights.len() != rows.len() {
                        return Err(AkitaError::InvalidSize {
                            expected: rows.len(),
                            actual: weights.len(),
                        });
                    }
                    if rows.is_empty() {
                        return Ok(vec![CyclotomicRing::zero(); window.columns.len()]);
                    }
                    return CyclotomicCrtNtt::mat_vec_transposed(&rows, weights, view.params());
                }
            };
        }
        dispatch!(self.q32_base());
        dispatch!(self.q64_base());
        dispatch!(self.q128_base());
        Err(AkitaError::InvalidSetup(
            "public windows require base-profile negacyclic rows".into(),
        ))
    }
}

/// Borrow only the selected entries after checking the entire backing extent.
fn window_rows<'a, const K: usize, const D: usize>(
    view: PreparedNttBaseView<'a, i32, K, D>,
    window: &MatrixWindow,
) -> Result<Vec<&'a [CyclotomicCrtNtt<i32, K, D>]>, AkitaError> {
    let requirement = window.prefix_requirement()?;
    if window.ring_d != D {
        return Err(AkitaError::InvalidSetup(
            "matrix window ring dimension mismatch".into(),
        ));
    }
    let prefix = view
        .negacyclic()
        .and_then(|rows| rows.get(..requirement.num_ring_elements))
        .ok_or_else(|| {
            AkitaError::InvalidSetup("matrix window exceeds negacyclic prefix".into())
        })?;
    window
        .rows
        .clone()
        .map(|row| {
            let start = checked::mul_add(row, window.stride, window.columns.start)
                .ok_or_else(|| AkitaError::InvalidSetup("matrix window offset overflow".into()))?;
            let range = checked::range(start, window.columns.len())
                .ok_or_else(|| AkitaError::InvalidSetup("matrix window extent overflow".into()))?;
            prefix
                .get(range)
                .ok_or_else(|| AkitaError::InvalidSetup("matrix window row is unavailable".into()))
        })
        .collect()
}

/// One logical public matrix over shared field and base-profile NTT prefixes.
///
/// The caller must supply two forms of the same stream in the same field.
/// Construction checks their extents, without copying or inverse-transforming
/// either backing. Sharing only the NTT allocation does not identify matrices.
pub struct PreparedWindow<F: Field, const D: usize> {
    field: Arc<dyn AsRef<[F]> + Send + Sync>,
    rows: Arc<PreparedNttCache<D>>,
    window: MatrixWindow,
}

impl<F: Field + CanonicalEncoding, const D: usize> PreparedWindow<F, D> {
    /// Check both backings and retain their handles without copying the stream.
    pub fn new(
        field: Arc<dyn AsRef<[F]> + Send + Sync>,
        rows: Arc<PreparedNttCache<D>>,
        window: MatrixWindow,
    ) -> Result<Arc<Self>, AkitaError> {
        let requirement = window.prefix_requirement()?;
        if window.ring_d != D || field.as_ref().as_ref().len() < requirement.num_field_elements()? {
            return Err(AkitaError::InvalidSetup(
                "matrix window exceeds field prefix or has wrong dimension".into(),
            ));
        }
        // Validate the NTT backing without performing a product.
        let valid = if let Some(view) = rows.q32_base() {
            window_rows(view, &window).map(|_| ())
        } else if let Some(view) = rows.q64_base() {
            window_rows(view, &window).map(|_| ())
        } else if let Some(view) = rows.q128_base() {
            window_rows(view, &window).map(|_| ())
        } else {
            Err(AkitaError::InvalidSetup(
                "public windows require base-profile negacyclic rows".into(),
            ))
        };
        valid?;
        Ok(Arc::new(Self {
            field,
            rows,
            window,
        }))
    }

    /// Rectangle selected from the shared prefix.
    pub fn window(&self) -> &MatrixWindow {
        &self.window
    }

    /// Shared NTT form, also available to prover kernels.
    pub fn rows(&self) -> &PreparedNttCache<D> {
        &self.rows
    }

    /// Borrow rows through `window.rows.end` at full stride in field form.
    pub fn field_view(&self) -> Result<RingMatrixView<'_, F, D>, AkitaError> {
        RingMatrixView::from_field_slice(
            self.field.as_ref().as_ref(),
            self.window.rows.end,
            self.window.stride,
        )
    }

    /// Public ring-weighted transposed product of the selected rectangle.
    pub fn transposed(
        &self,
        weights: &[CyclotomicRing<F, D>],
    ) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError> {
        self.rows.mat_vec_transposed(&self.window, weights)
    }

    /// Public field-scaled sum of selected columns, one ring per selected row.
    pub fn combine_columns(&self, scales: &[F]) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>
    where
        F: WithAccumulator,
    {
        self.field_view()?.combine_columns(&self.window, scales)
    }

    /// Combine a range relative to the selected columns, rather than the stride.
    pub fn combine_column_range(
        &self,
        columns: Range<usize>,
        scales: &[F],
    ) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>
    where
        F: WithAccumulator,
    {
        if columns.start > columns.end || columns.end > self.window.columns.len() {
            return Err(AkitaError::InvalidInput(
                "column range exceeds matrix window".into(),
            ));
        }
        let mut window = self.window.clone();
        window.columns =
            (self.window.columns.start + columns.start)..(self.window.columns.start + columns.end);
        self.field_view()?.combine_columns(&window, scales)
    }

    /// Same logical matrix: identical handles, or the same field allocation and
    /// window. A shared NTT allocation alone never establishes identity.
    pub fn same_matrix(a: &Arc<Self>, b: &Arc<Self>) -> bool {
        Arc::ptr_eq(a, b) || (Arc::ptr_eq(&a.field, &b.field) && a.window == b.window)
    }
}

#[cfg(test)]
mod tests;
