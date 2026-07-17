//! Checked plain ring arithmetic shared by direct verifier paths.
//!
//! These kernels deliberately operate on validated setup matrix views. They
//! are verifier soundness code: callers own protocol layout and shape checks;
//! this module owns only the canonical arithmetic over those checked slices.

use akita_algebra::CyclotomicRing;
use akita_field::{AkitaError, CanonicalField, FieldCore};
use std::array::from_fn;

pub(super) fn mat_vec_mul_i8<F, const D: usize>(
    matrix_rows: &[&[CyclotomicRing<F, D>]],
    digits: &[[i8; D]],
) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>
where
    F: FieldCore + CanonicalField,
{
    if matrix_rows.iter().any(|row| row.len() != digits.len()) {
        return Err(AkitaError::InvalidProof);
    }
    let digit_rings = digits
        .iter()
        .map(|digit| {
            CyclotomicRing::from_coefficients(from_fn(|idx| F::from_i64(digit[idx] as i64)))
        })
        .collect::<Vec<_>>();
    Ok(matrix_rows
        .iter()
        .map(|row| {
            row.iter()
                .zip(&digit_rings)
                .fold(CyclotomicRing::<F, D>::zero(), |acc, (entry, digit)| {
                    acc + (*entry * *digit)
                })
        })
        .collect())
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

#[cfg(test)]
mod tests {
    use super::*;
    use akita_field::Prime32Offset99 as F;

    #[test]
    fn mat_vec_rejects_width_mismatch() {
        let entry = CyclotomicRing::<F, 2>::one();
        assert!(mat_vec_mul_i8::<F, 2>(&[&[entry, entry]], &[[1, 0]]).is_err());
    }

    #[test]
    fn mat_vec_reuses_checked_digit_rings() {
        let one = CyclotomicRing::<F, 2>::one();
        let zero = CyclotomicRing::<F, 2>::zero();
        let first = [one, zero];
        let second = [zero, one];
        let actual = mat_vec_mul_i8::<F, 2>(&[&first, &second], &[[2, -1], [-3, 1]])
            .expect("matched matrix width");

        assert_eq!(
            actual,
            vec![
                CyclotomicRing::from_coefficients([F::from_i64(2), F::from_i64(-1)]),
                CyclotomicRing::from_coefficients([F::from_i64(-3), F::from_i64(1)]),
            ]
        );
    }
}
