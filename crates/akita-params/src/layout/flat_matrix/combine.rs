use akita_algebra::{cfg_fold_reduce, CyclotomicRing};
use akita_error::AkitaError;
#[allow(unused_imports)]
use jolt_field::solinas::parallel::*;
use jolt_field::{Accumulator, Field, WithAccumulator};

use super::RingMatrixView;
use crate::MatrixWindow;

impl<F: Field + WithAccumulator, const D: usize> RingMatrixView<'_, F, D> {
    /// Sum `scales[col] * M[row][col]` over the window's columns for each
    /// selected row. Scales are field scalars, rather than ring multipliers.
    /// Public inputs only. Both the window and scale count are checked first.
    pub fn combine_columns(
        &self,
        window: &MatrixWindow,
        scales: &[F],
    ) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError> {
        window.prefix_requirement()?;
        if window.ring_d != D || window.stride != self.num_cols || window.rows.end > self.num_rows {
            return Err(AkitaError::InvalidSetup(
                "matrix window does not fit field view".into(),
            ));
        }
        if scales.len() != window.columns.len() {
            return Err(AkitaError::InvalidSize {
                expected: window.columns.len(),
                actual: scales.len(),
            });
        }
        const COLUMN_CHUNK: usize = 256;
        window
            .rows
            .clone()
            .map(|row| {
                let entries = self.row(row)?.get(window.columns.clone()).ok_or_else(|| {
                    AkitaError::InvalidSetup("matrix window columns are unavailable".into())
                })?;
                let sums = cfg_fold_reduce!(
                    0..scales.len().div_ceil(COLUMN_CHUNK),
                    || [F::Accumulator::default(); D],
                    |mut sums, chunk| {
                        let start = chunk * COLUMN_CHUNK;
                        let end = (start + COLUMN_CHUNK).min(scales.len());
                        for (entry, &scale) in entries[start..end].iter().zip(&scales[start..end]) {
                            if scale.is_zero() {
                                continue;
                            }
                            for (sum, &coefficient) in sums.iter_mut().zip(entry.coefficients()) {
                                sum.fmadd(coefficient, scale);
                            }
                        }
                        sums
                    },
                    |mut sums, other| {
                        for (sum, other) in sums.iter_mut().zip(other) {
                            sum.merge(other);
                        }
                        sums
                    }
                );
                Ok(CyclotomicRing::from_coefficients(
                    sums.map(Accumulator::reduce),
                ))
            })
            .collect()
    }
}

#[cfg(test)]
#[path = "combine/tests.rs"]
mod tests;
