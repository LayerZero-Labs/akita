use super::*;
use crate::tables::{
    q128_primes, Q128_NUM_PRIMES, Q32_NUM_PRIMES, Q32_PRIMES, Q64_NUM_PRIMES, Q64_PRIMES,
};
use jolt_field::{Prime128Offset275, Prime128OffsetA7F7, Prime32Offset99, Prime64Offset59};
use rand::{rngs::StdRng, SeedableRng};

fn oracle<F: Field + CanonicalEncoding, const K: usize, const D: usize>(
    primes: [NttPrime<i32>; K],
) {
    let params = CrtNttParamSet::new(primes);
    let mut rng = StdRng::seed_from_u64(0xc03);
    for (num_rows, width) in [(1, 1), (3, 5), (5, 2)] {
        let mut matrix: Vec<Vec<CyclotomicRing<F, D>>> = (0..num_rows)
            .map(|_| {
                (0..width)
                    .map(|_| CyclotomicRing::random(&mut rng))
                    .collect()
            })
            .collect();
        // Include maximal centered coefficients of both signs in every case.
        let half = F::from_u128_reduced((-F::one()).to_u128_checked().unwrap().div_ceil(2));
        matrix[0][0].coeffs[0] = half;
        matrix[0][0].coeffs[D - 1] = -half;
        let ntt: Vec<Vec<_>> = matrix
            .iter()
            .map(|row| {
                row.iter()
                    .map(|entry| CyclotomicCrtNtt::from_ring(entry, &params))
                    .collect()
            })
            .collect();
        let rows: Vec<&[_]> = ntt.iter().map(Vec::as_slice).collect();
        let mut weights: Vec<CyclotomicRing<F, D>> = (0..num_rows)
            .map(|_| CyclotomicRing::random(&mut rng))
            .collect();
        for case in 0..4 {
            match case {
                1 => {
                    weights.fill(CyclotomicRing::from_coefficients([-F::one(); D]));
                }
                2 => {
                    weights.fill(CyclotomicRing::zero());
                }
                3 => {
                    weights.fill(CyclotomicRing::one());
                }
                _ => {}
            }
            let expected: Vec<_> = (0..width)
                .map(|col| {
                    let mut out = [F::zero(); D];
                    // Independent schoolbook negacyclic convolution, in F.
                    for (row, weight) in matrix.iter().zip(&weights) {
                        for (i, a) in row[col].coefficients().iter().enumerate() {
                            for (j, b) in weight.coefficients().iter().enumerate() {
                                if i + j < D {
                                    out[i + j] += *a * *b;
                                } else {
                                    out[i + j - D] -= *a * *b;
                                }
                            }
                        }
                    }
                    CyclotomicRing::from_coefficients(out)
                })
                .collect();
            assert_eq!(
                CyclotomicCrtNtt::mat_vec_transposed(&rows, &weights, &params).unwrap(),
                expected
            );
        }
    }
}

#[test]
fn transposed_matches_schoolbook_for_all_field_profiles() {
    oracle::<Prime32Offset99, Q32_NUM_PRIMES, 64>(Q32_PRIMES);
    oracle::<Prime64Offset59, Q64_NUM_PRIMES, 128>(Q64_PRIMES);
    oracle::<Prime128Offset275, Q128_NUM_PRIMES, 64>(q128_primes());
    oracle::<Prime128OffsetA7F7, Q128_NUM_PRIMES, 256>(q128_primes());
}

#[test]
fn transposed_checks_shapes_and_crt_margin() {
    type F = Prime128Offset275;
    type Ntt = CyclotomicCrtNtt<i32, Q32_NUM_PRIMES, 64>;
    let params = CrtNttParamSet::new(Q32_PRIMES);
    let entry = Ntt::zero();
    let row = [entry];
    let weights = [CyclotomicRing::<F, 64>::one()];
    assert!(Ntt::mat_vec_transposed::<F>(&[&row], &[], &params).is_err());
    assert!(Ntt::mat_vec_transposed(&[&row, &[]], &[weights[0]; 2], &params).is_err());
    // Q32 primes cannot hold a product with Q128 matrix coefficients.
    assert!(Ntt::mat_vec_transposed(&[&row], &weights, &params).is_err());
    assert!(Ntt::mat_vec_transposed::<F>(&[], &[], &params)
        .unwrap()
        .is_empty());
    assert!(Ntt::mat_vec_transposed(&[&[]], &weights, &params)
        .unwrap()
        .is_empty());
}
