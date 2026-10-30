use super::*;
use crate::binary_source::{BinaryCommitmentProfiles, BinaryIndexMap, BinarySourceDescriptor};
use jolt_field::{One, Prime128Offset275 as F, Ring, Zero};
use rand::{rngs::StdRng, Rng, SeedableRng};

fn plan(rows: usize, rank: usize) -> BinaryRootWitnessPlan {
    let index = BinaryIndexMap::new(128, 1, 4 * rows, 1).unwrap();
    let descriptor = BinarySourceDescriptor::new(
        index,
        vec![[1; 32]],
        BinaryCommitmentProfiles {
            a: [2; 32],
            b: [3; 32],
        },
    )
    .unwrap();
    BinaryRootWitnessPlan::new(&descriptor, rank, 64).unwrap()
}

fn random_scalar(rng: &mut StdRng) -> F {
    F::from_i128(rng.gen_range(-31..=31))
}

// Polynomial long division, deliberately independent of the adjoint kernel.
// Psi = Y^648 - Y^324 + 1, so Y^n = Y^(n-324) - Y^(n-648).
fn signed_remainder(mut polynomial: Vec<i128>) -> [i128; DEGREE] {
    for degree in (DEGREE..polynomial.len()).rev() {
        let value = polynomial[degree];
        polynomial[degree - DEGREE / 2] += value;
        polynomial[degree - DEGREE] -= value;
        polynomial[degree] = 0;
    }
    core::array::from_fn(|index| polynomial[index])
}

fn monomial_row(rho: &[F; DEGREE]) -> Vec<F> {
    (0..2 * DEGREE - 1)
        .map(|degree| {
            let mut monomial = vec![0; 2 * DEGREE - 1];
            monomial[degree] = 1;
            signed_remainder(monomial)
                .into_iter()
                .zip(rho)
                .filter(|(coefficient, _)| *coefficient != 0)
                .fold(F::zero(), |sum, (coefficient, &weight)| {
                    sum + F::from_i128(coefficient) * weight
                })
        })
        .collect()
}

fn direct_response_weights(
    matrix: &[Vec<[i128; DEGREE]>],
    rho: &[[F; DEGREE]],
) -> Vec<[F; DEGREE]> {
    let rows = matrix[0].len();
    let mut weights = vec![[F::zero(); DEGREE]; rows];
    for (matrix_row, row) in matrix.iter().zip(rho) {
        let reduced_monomials = monomial_row(row);
        for (target, entry) in weights.iter_mut().zip(matrix_row) {
            for (response_coefficient, weight) in target.iter_mut().enumerate() {
                *weight +=
                    entry
                        .iter()
                        .enumerate()
                        .fold(F::zero(), |sum, (setup_coefficient, &value)| {
                            sum + F::from_i128(value)
                                * reduced_monomials[setup_coefficient + response_coefficient]
                        });
            }
        }
    }
    weights
}

fn matrix_claim(matrix: &[Vec<[i128; DEGREE]>], index: &[F], coefficient: &[F]) -> F {
    matrix.iter().enumerate().fold(F::zero(), |sum, (i, row)| {
        sum + row.iter().enumerate().fold(F::zero(), |sum, (j, entry)| {
            sum + entry
                .iter()
                .enumerate()
                .fold(F::zero(), |sum, (s, &value)| {
                    sum + F::from_i128(value) * index[j] * coefficient[i * STRIDE + s]
                })
        })
    })
}

#[test]
fn stage3_rho_boundary_matches_coefficient_functional() {
    for alpha in [F::zero(), F::one(), F::from_u64(7)] {
        let batches = [F::zero(), F::from_u64(3), -F::one()];
        let rho = binary_a_rho(alpha, &batches);
        for (row, batch) in rho.iter().zip(batches) {
            for (exponent, &value) in row.iter().enumerate() {
                let mut power = F::one();
                let mut square = alpha;
                let mut exponent = exponent;
                while exponent != 0 {
                    if exponent & 1 != 0 {
                        power *= square;
                    }
                    square *= square;
                    exponent >>= 1;
                }
                assert_eq!(value, batch * power);
            }
        }
    }
    assert!(binary_a_rho(F::one(), &[]).is_empty());
}

#[test]
fn stage3_a_product_matches_independent_signed_remainder_and_response_digits() {
    let mut rng = StdRng::seed_from_u64(0xa753_0239);
    for rows in [1usize, 2, 3, 5] {
        for rank in [1usize, 3] {
            // Public descriptors currently require power-of-two source rows.
            // An internal live-row cutoff also exercises odd-row arithmetic.
            let plan = plan(rows.next_power_of_two(), rank);
            let row_stride = rank.next_power_of_two() * STRIDE;
            let prefix_len = (rows * row_stride).next_power_of_two() * 2;
            let point = (0..plan.repacking().domain_len().trailing_zeros())
                .map(|_| random_scalar(&mut rng))
                .collect::<Vec<_>>();
            let rho = (0..rank)
                .map(|_| core::array::from_fn(|_| random_scalar(&mut rng)))
                .collect::<Vec<_>>();
            let matrix = (0..rank)
                .map(|_| {
                    (0..rows)
                        .map(|_| core::array::from_fn(|_| rng.gen_range(-31..=31)))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let mut product = BinaryAProduct::new(&plan, &rho, &point, prefix_len).unwrap();
            product.rows = rows;
            let (index, coefficient) = product.factors().unwrap();
            let direct_weights = direct_response_weights(&matrix, &rho);
            let mut direct_at_point = F::zero();
            for (j, weights) in direct_weights.iter().enumerate() {
                let mut digit_power = F::one();
                for digit in 0..4 {
                    for (k, &weight) in weights.iter().enumerate() {
                        let address = plan
                            .repacking()
                            .coefficient_address(0, j, digit, k)
                            .unwrap();
                        direct_at_point += eq_eval_at_index(&point, address) * digit_power * weight;
                    }
                    digit_power *= F::from_u64(8);
                }
            }
            assert_eq!(
                matrix_claim(&matrix, &index, &coefficient),
                direct_at_point,
                "rows={rows}, rank={rank}"
            );

            // Contract arbitrary response digits at their actual repacking addresses.
            let mut table = vec![0i128; plan.repacking().domain_len()];
            let mut response = vec![[0i128; DEGREE]; rows];
            let mut weight_claim = F::zero();
            for (j, response) in response.iter_mut().enumerate() {
                let mut digit_power = 1i128;
                for digit in 0..4 {
                    for (k, response) in response.iter_mut().enumerate() {
                        let address = plan
                            .repacking()
                            .coefficient_address(0, j, digit, k)
                            .unwrap();
                        table[address] = rng.gen_range(-4..=3);
                        *response += digit_power * table[address];
                        weight_claim +=
                            direct_weights[j][k] * F::from_i128(digit_power * table[address]);
                    }
                    digit_power *= 8;
                }
            }
            let mut rem_claim = F::zero();
            for (matrix_row, rho) in matrix.iter().zip(&rho) {
                let mut convolution = vec![0i128; 2 * DEGREE - 1];
                for (entry, response) in matrix_row.iter().zip(&response) {
                    for (s, &a) in entry.iter().enumerate() {
                        for (k, &z) in response.iter().enumerate() {
                            convolution[s + k] += a * z;
                        }
                    }
                }
                rem_claim += signed_remainder(convolution)
                    .into_iter()
                    .zip(rho)
                    .fold(F::zero(), |sum, (value, &rho)| {
                        sum + F::from_i128(value) * rho
                    });
            }
            assert_eq!(
                weight_claim, rem_claim,
                "response contraction rows={rows}, rank={rank}"
            );
        }
    }
}

#[test]
fn stage3_a_product_terminal_matches_dense_factors_with_zero_coordinates_and_odd_cutoff() {
    let mut rng = StdRng::seed_from_u64(0x7621_aa31);
    for rows in [1usize, 2, 3, 5] {
        let plan = plan(rows.next_power_of_two(), 3);
        let prefix_len = (rows * 4 * STRIDE).next_power_of_two() * 2;
        for zero_coordinates in [false, true] {
            let stage2_point = (0..plan.repacking().domain_len().trailing_zeros())
                .map(|i| {
                    if zero_coordinates && i % 3 == 0 {
                        F::zero()
                    } else {
                        random_scalar(&mut rng)
                    }
                })
                .collect::<Vec<_>>();
            let rho = (0..3)
                .map(|_| core::array::from_fn(|_| random_scalar(&mut rng)))
                .collect::<Vec<_>>();
            let mut product = BinaryAProduct::new(&plan, &rho, &stage2_point, prefix_len).unwrap();
            product.rows = rows;
            let (index, coefficient) = product.factors().unwrap();
            assert!(index[rows..].iter().all(|&value| value == F::zero()));
            for all_zero in [false, true] {
                let point = (0..prefix_len.trailing_zeros())
                    .map(|i| {
                        if all_zero || i % 4 == 0 {
                            F::zero()
                        } else {
                            random_scalar(&mut rng)
                        }
                    })
                    .collect::<Vec<_>>();
                let coefficient_bits = coefficient.len().trailing_zeros() as usize;
                let coefficient_value =
                    coefficient
                        .iter()
                        .enumerate()
                        .fold(F::zero(), |sum, (address, &value)| {
                            sum + value * eq_eval_at_index(&point[..coefficient_bits], address)
                        });
                let index_value =
                    index
                        .iter()
                        .enumerate()
                        .fold(F::zero(), |sum, (address, &value)| {
                            sum + value * eq_eval_at_index(&point[coefficient_bits..], address)
                        });
                assert_eq!(
                    product.evaluate(&point).unwrap(),
                    coefficient_value * index_value,
                    "rows={rows}, zero={all_zero}"
                );
            }
        }
    }
}

#[test]
fn stage3_a_product_rejects_malformed_rho_point_and_prefix_without_panic() {
    let plan = plan(2, 3);
    let point = vec![F::zero(); plan.repacking().domain_len().trailing_zeros() as usize];
    let rho = vec![[F::one(); DEGREE]; 3];
    let prefix_len = 2 * 4 * STRIDE;
    assert!(BinaryAProduct::new(&plan, &rho[..2], &point, prefix_len).is_err());
    assert!(BinaryAProduct::new(&plan, &rho, &point[..point.len() - 1], prefix_len).is_err());
    let mut longer = point.clone();
    longer.push(F::one());
    assert!(BinaryAProduct::new(&plan, &rho, &longer, prefix_len).is_err());
    for bad_prefix in [0, 1, STRIDE, prefix_len - 1] {
        assert!(BinaryAProduct::new(&plan, &rho, &point, bad_prefix).is_err());
    }
    let product = BinaryAProduct::new(&plan, &rho, &point, prefix_len).unwrap();
    for bad_point in [
        vec![],
        vec![F::zero(); prefix_len.trailing_zeros() as usize - 1],
        vec![F::zero(); prefix_len.trailing_zeros() as usize + 1],
    ] {
        assert!(product.evaluate(&bad_point).is_err());
    }
}
