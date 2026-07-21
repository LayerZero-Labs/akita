use akita_algebra::{CanonicalField, CyclotomicRing, FieldCore, MixedCrtNtt, MixedCrtNttParamSet};

/// Multiply a prepared mixed-CRT matrix by signed i16 ring vectors.
///
/// The caller selects `params` with the exact accumulation bound and prepares
/// `ntt_mat` with the same profile. No i16 transform or tail residue is built
/// on the ordinary i8 path.
#[must_use]
pub fn mat_vec_mul_mixed_ntt_digits_i16<
    F: FieldCore + CanonicalField,
    const K: usize,
    const D: usize,
>(
    ntt_mat: &[&[MixedCrtNtt<K, D>]],
    blocks: &[&[[i16; D]]],
    params: &MixedCrtNttParamSet<K, D>,
) -> Vec<Vec<CyclotomicRing<F, D>>> {
    let num_rows = ntt_mat.len();
    let matrix_width = ntt_mat.first().map_or(0, |row| row.len());
    blocks
        .iter()
        .map(|block| {
            let width = matrix_width.min(block.len());
            let mut accumulators: Vec<_> = (0..num_rows).map(|_| MixedCrtNtt::zero()).collect();
            for (column, digits) in block.iter().take(width).enumerate() {
                if digits.iter().all(|digit| *digit == 0) {
                    continue;
                }
                let rhs = MixedCrtNtt::from_i16(digits, params);
                for (accumulator, matrix_row) in accumulators.iter_mut().zip(ntt_mat.iter()) {
                    accumulator.add_assign_pointwise_mul(&matrix_row[column], &rhs, params);
                }
            }
            accumulators
                .iter()
                .map(|accumulator| accumulator.to_ring(params))
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_algebra::ntt::tables::{
        q128_primes, I16_TAIL_PRIME, Q128_NUM_PRIMES, Q32_NUM_PRIMES, Q32_PRIMES, Q64_NUM_PRIMES,
        Q64_PRIMES,
    };
    use akita_algebra::CrtNttParamSet;
    use akita_field::{Prime128Offset275, Prime32Offset99, Prime64Offset59};
    use std::array::from_fn;

    fn differential<F: FieldCore + CanonicalField, const K: usize, const D: usize>(
        wide: CrtNttParamSet<i32, K, D>,
    ) {
        let params = MixedCrtNttParamSet::new(wide, CrtNttParamSet::new([I16_TAIL_PRIME]));
        let matrix: Vec<Vec<CyclotomicRing<F, D>>> = (0..2)
            .map(|row| {
                (0..3)
                    .map(|column| {
                        CyclotomicRing::from_coefficients(from_fn(|coefficient| {
                            F::from_i64(((row * 19 + column * 7 + coefficient) % 23) as i64 - 11)
                        }))
                    })
                    .collect()
            })
            .collect();
        let prepared: Vec<Vec<_>> = matrix
            .iter()
            .map(|row| {
                row.iter()
                    .map(|ring| MixedCrtNtt::from_ring(ring, &params))
                    .collect()
            })
            .collect();
        let prepared_rows: Vec<&[_]> = prepared.iter().map(Vec::as_slice).collect();
        let digits = vec![
            from_fn(|i| match i % 6 {
                0 => i16::MIN,
                1 => -1024,
                2 => -1,
                3 => 0,
                4 => 1023,
                _ => i16::MAX,
            }),
            from_fn(|i| (i as i16 % 31) - 15),
            [0i16; D],
        ];
        let blocks = [digits.as_slice()];
        let actual = mat_vec_mul_mixed_ntt_digits_i16::<F, K, D>(&prepared_rows, &blocks, &params);

        for row in 0..2 {
            let expected = (0..3).fold(CyclotomicRing::<F, D>::zero(), |sum, column| {
                let rhs = CyclotomicRing::from_coefficients(
                    digits[column].map(|digit| F::from_i64(i64::from(digit))),
                );
                sum + matrix[row][column] * rhs
            });
            assert_eq!(actual[0][row], expected);
        }
    }

    #[test]
    fn mixed_i16_matvec_matches_reference_for_every_field_and_multiple_degrees() {
        differential::<Prime32Offset99, Q32_NUM_PRIMES, 64>(CrtNttParamSet::new(Q32_PRIMES));
        differential::<Prime64Offset59, Q64_NUM_PRIMES, 128>(CrtNttParamSet::new(Q64_PRIMES));
        differential::<Prime128Offset275, Q128_NUM_PRIMES, 256>(CrtNttParamSet::new(q128_primes()));
    }
}
