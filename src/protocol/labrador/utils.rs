//! Shared utility helpers for the Labrador sub-protocol.

use crate::algebra::ring::CyclotomicRing;
use crate::FieldCore;

pub(crate) fn mat_vec_mul<F: FieldCore, const D: usize>(
    mat: &[Vec<CyclotomicRing<F, D>>],
    vec: &[CyclotomicRing<F, D>],
) -> Vec<CyclotomicRing<F, D>> {
    mat.iter()
        .map(|row| {
            debug_assert_eq!(row.len(), vec.len());
            let mut acc = CyclotomicRing::<F, D>::zero();
            for (a, x) in row.iter().zip(vec.iter()) {
                acc += *a * *x;
            }
            acc
        })
        .collect()
}
