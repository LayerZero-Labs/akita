//! CPU fold output retained for standalone kernels and backend implementations.

use akita_error::AkitaError;
use jolt_field::solinas::parallel::*;

/// Prover-side output of the decompose + challenge-fold step.
///
/// This is a CPU kernel value, not a protocol-facing witness. Protocol code
/// transports the corresponding opaque accepted-fold handle instead.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecomposeFoldWitness {
    centered_coeffs_flat: Vec<i32>,
    centered_min: i32,
    centered_max: i32,
    ring_dim: usize,
}

impl DecomposeFoldWitness {
    pub fn from_centered_rows<const D: usize>(centered_coeffs: Vec<[i32; D]>) -> Self {
        let (centered_min, centered_max) = centered_coefficient_bounds(&centered_coeffs);
        Self {
            centered_coeffs_flat: centered_coeffs.into_flattened(),
            centered_min,
            centered_max,
            ring_dim: D,
        }
    }

    pub fn from_centered_flat<const D: usize>(
        centered_coeffs_flat: Vec<i32>,
    ) -> Result<Self, AkitaError> {
        if D == 0 {
            return Err(AkitaError::InvalidInput(
                "decompose fold witness ring dimension must be non-zero".into(),
            ));
        }
        let (centered_rows, remainder) = centered_coeffs_flat.as_chunks::<D>();
        if remainder.is_empty() {
            let (centered_min, centered_max) = centered_coefficient_bounds(centered_rows);
            return Ok(Self {
                centered_coeffs_flat,
                centered_min,
                centered_max,
                ring_dim: D,
            });
        }
        Err(AkitaError::InvalidSize {
            expected: D,
            actual: centered_coeffs_flat.len(),
        })
    }

    pub fn into_centered_coeffs_flat(self) -> Vec<i32> {
        self.centered_coeffs_flat
    }

    /// Number of folded witness rows.
    #[must_use]
    pub fn row_count(&self) -> usize {
        self.centered_coeffs_flat
            .len()
            .checked_div(self.ring_dim)
            .unwrap_or(0)
    }

    /// Check that the stored runtime ring dimension matches `D`.
    ///
    /// # Errors
    ///
    /// Returns an error when the ring dimensions or row counts disagree.
    pub fn ensure_ring_dim<const D: usize>(&self) -> Result<(), AkitaError> {
        if self.ring_dim != D {
            return Err(AkitaError::InvalidInput(format!(
                "decompose fold witness ring_d={} does not match requested D={D}",
                self.ring_dim
            )));
        }
        if D == 0 || !self.centered_coeffs_flat.len().is_multiple_of(D) {
            return Err(AkitaError::InvalidSize {
                expected: D,
                actual: self.centered_coeffs_flat.len(),
            });
        }
        Ok(())
    }

    pub fn centered_coeffs_flat(&self) -> &[i32] {
        &self.centered_coeffs_flat
    }

    #[cfg(test)]
    pub(crate) fn centered_coeffs_trusted<const D: usize>(&self) -> &[[i32; D]] {
        debug_assert_eq!(self.ring_dim, D);
        let (chunks, remainder) = self.centered_coeffs_flat.as_chunks::<D>();
        debug_assert!(remainder.is_empty());
        chunks
    }

    #[cfg(test)]
    pub(crate) fn centered_coeffs_owned<const D: usize>(&self) -> Vec<[i32; D]> {
        self.centered_coeffs_trusted::<D>().to_vec()
    }

    /// Infinity norm derived from the centered coefficient buffer.
    #[must_use]
    pub fn centered_inf_norm(&self) -> u32 {
        self.centered_min
            .unsigned_abs()
            .max(self.centered_max.unsigned_abs())
    }

    /// Signed extrema derived from the centered coefficient buffer.
    #[must_use]
    pub fn centered_signed_extrema(&self) -> (i32, i32) {
        (self.centered_min, self.centered_max)
    }
}

fn centered_coefficient_bounds<const D: usize>(rows: &[[i32; D]]) -> (i32, i32) {
    if rows.is_empty() || D == 0 {
        return (0, 0);
    }
    cfg_fold_reduce!(
        rows,
        || (i32::MAX, i32::MIN),
        |(mut min, mut max), row| {
            for &coefficient in row {
                min = min.min(coefficient);
                max = max.max(coefficient);
            }
            (min, max)
        },
        |(left_min, left_max), (right_min, right_max)| {
            (left_min.min(right_min), left_max.max(right_max))
        }
    )
}
