//! Test-only helpers for [`DensePoly`].

use super::poly::DensePoly;
use akita_algebra::CyclotomicRing;
use akita_field::FieldCore;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Reference ring-space evaluation for [`DensePoly`].
///
/// Computes the global weighted sum `y = Σᵢ scalars[i] · self.coeffs[i]`.
pub(crate) fn evaluate_ring_dense<F, const D: usize>(
    poly: &DensePoly<F, D>,
    scalars: &[F],
) -> CyclotomicRing<F, D>
where
    F: FieldCore,
{
    #[cfg(feature = "parallel")]
    {
        poly.coeffs
            .par_iter()
            .zip(scalars.par_iter())
            .fold(
                || CyclotomicRing::<F, D>::zero(),
                |acc, (f_i, w_i)| acc + f_i.scale(w_i),
            )
            .reduce(|| CyclotomicRing::<F, D>::zero(), |a, b| a + b)
    }
    #[cfg(not(feature = "parallel"))]
    {
        poly.coeffs
            .iter()
            .zip(scalars.iter())
            .fold(CyclotomicRing::<F, D>::zero(), |acc, (f_i, w_i)| {
                acc + f_i.scale(w_i)
            })
    }
}
