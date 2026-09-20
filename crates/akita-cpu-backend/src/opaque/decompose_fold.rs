//! CPU fold output retained for standalone kernels and backend implementations.

#[cfg(test)]
use akita_algebra::CyclotomicRing;
use akita_error::AkitaError;
use akita_types::RingVec;
use jolt_field::solinas::parallel::*;
use jolt_field::Field;

/// Prover-side output of the decompose + challenge-fold step.
///
/// This is a CPU kernel value, not a protocol-facing witness. Protocol code
/// transports the corresponding opaque accepted-fold handle instead.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DecomposeFoldWitness<F: Field> {
    pub(crate) z_folded_rings: RingVec<F>,
    centered_coeffs_flat: Vec<i32>,
    centered_min: i32,
    centered_max: i32,
    ring_dim: usize,
}

impl<F: Field> DecomposeFoldWitness<F> {
    pub(crate) fn from_coefficient_parts<const D: usize>(
        z_folded_coeffs: Vec<[F; D]>,
        centered_coeffs: Vec<[i32; D]>,
    ) -> Self {
        debug_assert_eq!(z_folded_coeffs.len(), centered_coeffs.len());
        let (centered_min, centered_max) = centered_coefficient_bounds(&centered_coeffs);
        Self {
            z_folded_rings: RingVec::from_coefficient_rows(z_folded_coeffs),
            centered_coeffs_flat: centered_coeffs.into_flattened(),
            centered_min,
            centered_max,
            ring_dim: D,
        }
    }

    #[cfg(test)]
    pub(crate) fn from_parts<const D: usize>(
        z_folded_rings: Vec<CyclotomicRing<F, D>>,
        centered_coeffs: Vec<[i32; D]>,
    ) -> Self {
        debug_assert_eq!(z_folded_rings.len(), centered_coeffs.len());
        let (centered_min, centered_max) = centered_coefficient_bounds(&centered_coeffs);
        Self {
            z_folded_rings: RingVec::from_ring_elems(&z_folded_rings),
            centered_coeffs_flat: centered_coeffs.into_flattened(),
            centered_min,
            centered_max,
            ring_dim: D,
        }
    }

    pub(crate) fn from_owned_flat_parts<const D: usize>(
        z_folded_rings: RingVec<F>,
        centered_coeffs_flat: Vec<i32>,
    ) -> Result<Self, AkitaError> {
        let (centered_rows, remainder) = centered_coeffs_flat.as_chunks::<D>();
        if remainder.is_empty()
            && z_folded_rings.ring_dim() == D
            && z_folded_rings.count() == centered_rows.len()
        {
            let (centered_min, centered_max) = centered_coefficient_bounds(centered_rows);
            return Ok(Self {
                z_folded_rings,
                centered_coeffs_flat,
                centered_min,
                centered_max,
                ring_dim: D,
            });
        }
        Err(AkitaError::InvalidInput(
            "owned decompose fold buffers have inconsistent ring geometry".into(),
        ))
    }

    pub(crate) fn into_owned_flat_parts(self) -> (RingVec<F>, Vec<i32>) {
        (self.z_folded_rings, self.centered_coeffs_flat)
    }

    /// Number of folded witness rows.
    #[must_use]
    pub(crate) fn row_count(&self) -> usize {
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
    pub(crate) fn ensure_ring_dim<const D: usize>(&self) -> Result<(), AkitaError> {
        if self.ring_dim != D {
            return Err(AkitaError::InvalidInput(format!(
                "decompose fold witness ring_d={} does not match requested D={D}",
                self.ring_dim
            )));
        }
        if !self.centered_coeffs_flat.len().is_multiple_of(D) {
            return Err(AkitaError::InvalidSize {
                expected: D,
                actual: self.centered_coeffs_flat.len(),
            });
        }
        if !self.z_folded_rings.can_decode_vec(D) {
            return Err(AkitaError::InvalidSize {
                expected: D,
                actual: self.z_folded_rings.coeff_len(),
            });
        }
        let ring_count = self.z_folded_rings.count();
        let row_count = self.centered_coeffs_flat.len() / D;
        if ring_count != row_count {
            return Err(AkitaError::InvalidInput(
                "decompose fold witness ring row count mismatch".to_string(),
            ));
        }
        Ok(())
    }

    pub(crate) fn centered_coeffs_flat(&self) -> &[i32] {
        &self.centered_coeffs_flat
    }

    #[cfg(test)]
    pub(crate) fn z_folded_rings_trusted<const D: usize>(
        &self,
    ) -> Result<&[CyclotomicRing<F, D>], AkitaError> {
        self.ensure_ring_dim::<D>()?;
        self.z_folded_rings.as_ring_slice::<D>()
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
    pub(crate) fn centered_inf_norm(&self) -> u32 {
        self.centered_min
            .unsigned_abs()
            .max(self.centered_max.unsigned_abs())
    }

    /// Signed extrema derived from the centered coefficient buffer.
    #[must_use]
    pub(crate) fn centered_signed_extrema(&self) -> (i32, i32) {
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
