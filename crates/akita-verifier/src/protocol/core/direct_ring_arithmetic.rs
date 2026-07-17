//! Checked plain ring arithmetic shared by direct verifier paths.
//!
//! These kernels deliberately operate on validated setup matrix views. They
//! are verifier soundness code: callers own protocol layout and shape checks;
//! this module owns only the canonical arithmetic over those checked slices.

use akita_algebra::CyclotomicRing;
use akita_field::{CanonicalField, FieldCore};
use std::array::from_fn;

pub(super) fn mat_vec_mul_i8<F, const D: usize>(
    matrix_rows: &[&[CyclotomicRing<F, D>]],
    digits: &[[i8; D]],
) -> Vec<CyclotomicRing<F, D>>
where
    F: FieldCore + CanonicalField,
{
    matrix_rows
        .iter()
        .map(|row| {
            row.iter().zip(digits.iter()).fold(
                CyclotomicRing::<F, D>::zero(),
                |acc, (entry, digit)| {
                    let digit_ring = CyclotomicRing::from_coefficients(from_fn(|idx| {
                        F::from_i64(digit[idx] as i64)
                    }));
                    acc + (*entry * digit_ring)
                },
            )
        })
        .collect()
}

pub(super) fn decompose_rows_i8<F, const D: usize>(
    rows: &[CyclotomicRing<F, D>],
    num_digits: usize,
    log_basis: u32,
) -> Vec<[i8; D]>
where
    F: FieldCore + CanonicalField,
{
    let mut out = vec![[0i8; D]; rows.len() * num_digits];
    for (dst_chunk, row) in out.chunks_mut(num_digits).zip(rows.iter()) {
        row.balanced_decompose_pow2_i8_into(dst_chunk, log_basis);
    }
    out
}
