//! Linear algebra helpers for ring commitment.

use crate::algebra::ring::CyclotomicRing;
use crate::{CanonicalField, FieldCore};

pub(crate) fn mat_vec_mul_unchecked<F: FieldCore + CanonicalField, const D: usize>(
    mat: &[Vec<CyclotomicRing<F, D>>],
    vec: &[CyclotomicRing<F, D>],
) -> Vec<CyclotomicRing<F, D>> {
    let mut out = Vec::with_capacity(mat.len());
    for row in mat {
        debug_assert_eq!(row.len(), vec.len());
        let mut acc = CyclotomicRing::<F, D>::zero();
        for (a, x) in row.iter().zip(vec.iter()) {
            acc += *a * *x;
        }
        out.push(acc);
    }
    out
}

pub(crate) fn decompose_block<F: FieldCore + CanonicalField, const D: usize>(
    block: &[CyclotomicRing<F, D>],
    delta: usize,
    log_basis: u32,
) -> Vec<CyclotomicRing<F, D>> {
    let mut out = Vec::with_capacity(block.len() * delta);
    for coeff_vec in block {
        out.extend(coeff_vec.balanced_decompose_pow2(delta, log_basis));
    }
    out
}

pub(crate) fn decompose_rows<F: FieldCore + CanonicalField, const D: usize>(
    rows: &[CyclotomicRing<F, D>],
    delta: usize,
    log_basis: u32,
) -> Vec<CyclotomicRing<F, D>> {
    let mut out = Vec::with_capacity(rows.len() * delta);
    for row in rows {
        out.extend(row.balanced_decompose_pow2(delta, log_basis));
    }
    out
}
