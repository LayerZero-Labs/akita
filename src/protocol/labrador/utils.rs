//! Shared utility helpers for the Labrador sub-protocol.

use crate::algebra::ring::CyclotomicRing;
#[cfg(feature = "parallel")]
use crate::parallel::*;
use crate::protocol::commitment::utils::linear::try_centered_i8_cache_from_ring_coeffs;
use crate::{CanonicalField, FieldCore};

pub(crate) fn mat_vec_mul<F: FieldCore, const D: usize>(
    mat: &[Vec<CyclotomicRing<F, D>>],
    vec: &[CyclotomicRing<F, D>],
) -> Vec<CyclotomicRing<F, D>> {
    cfg_iter!(mat)
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

pub(crate) fn try_centered_i8_rows<F: CanonicalField, const D: usize>(
    rows: &[Vec<CyclotomicRing<F, D>>],
) -> Option<Vec<Vec<[i8; D]>>> {
    rows.iter()
        .map(|row| try_centered_i8_cache_from_ring_coeffs(row))
        .collect()
}

pub(crate) fn pow2_field<F: FieldCore>(exp: usize) -> F {
    let two = F::one() + F::one();
    let mut acc = F::one();
    for _ in 0..exp {
        acc = acc * two;
    }
    acc
}
