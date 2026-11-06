use super::*;
use crate::binary_source::{BinaryCommitmentProfiles, BinaryIndexMap, BinarySourceDescriptor};
use jolt_field::{One, Prime128Offset275 as F, Ring, Zero};
use rand::{rngs::StdRng, Rng, SeedableRng};

fn plan(rows: usize, rank: usize) -> BinaryRootWitnessPlan {
    group_plan(rows, rank, 1)
}

fn group_plan(rows: usize, rank: usize, owners: usize) -> BinaryRootWitnessPlan {
    let index = BinaryIndexMap::new(128, 1, 4 * rows, 1).unwrap();
    let descriptor = BinarySourceDescriptor::new(
        index,
        (0..owners).map(|owner| [owner as u8 + 1; 32]).collect(),
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

fn monomial_row<F: Field>(rho: &[F; DEGREE]) -> Vec<F> {
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

fn direct_response_weights<F: Field>(
    matrix: &[Vec<[i128; DEGREE]>],
    rho: &[[F; DEGREE]],
) -> Vec<[F; DEGREE]> {
    let rows = matrix[0].len();
    let mut weights = Vec::new();
    for owner_rho in rho.chunks_exact(matrix.len()) {
        let mut owner_weights = vec![[F::zero(); DEGREE]; rows];
        for (matrix_row, row) in matrix.iter().zip(owner_rho) {
            let reduced_monomials = monomial_row(row);
            for (target, entry) in owner_weights.iter_mut().zip(matrix_row) {
                for (response_coefficient, weight) in target.iter_mut().enumerate() {
                    *weight += entry.iter().enumerate().fold(
                        F::zero(),
                        |sum, (setup_coefficient, &value)| {
                            sum + F::from_i128(value)
                                * reduced_monomials[setup_coefficient + response_coefficient]
                        },
                    );
                }
            }
        }
        weights.extend(owner_weights);
    }
    weights
}

fn matrix_claim<F: Field>(matrix: &[Vec<[i128; DEGREE]>], index: &[F], coefficient: &[F]) -> F {
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
    let plan = group_plan(2, 3, 2);
    let point = vec![F::from_u64(7); plan.repacking().domain_len().trailing_zeros() as usize];
    let mu = [
        F::zero(),
        F::one(),
        F::from_u64(3),
        -F::one(),
        F::from_u64(9),
        F::zero(),
        F::from_u64(11),
    ];
    let projections = (0..7)
        .map(|l| {
            (0..6)
                .map(|row| {
                    core::array::from_fn(|t| F::from_u64(((l * 13 + row * 17 + t) % 101) as u64))
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let rho = (0..6)
        .map(|row| {
            core::array::from_fn(|t| {
                projections
                    .iter()
                    .zip(mu)
                    .fold(F::zero(), |sum, (j, mu)| sum + mu * j[row][t])
            })
        })
        .collect::<Vec<_>>();
    let prefix = 2 * 4 * STRIDE;
    let (_, joined) = BinaryAProduct::new(&plan, &rho, &point, prefix)
        .unwrap()
        .factors()
        .unwrap();
    let mut expected = vec![F::zero(); joined.len()];
    for (j, mu) in projections.iter().zip(mu) {
        let (_, factor) = BinaryAProduct::new(&plan, j, &point, prefix)
            .unwrap()
            .factors()
            .unwrap();
        for (target, coefficient) in expected.iter_mut().zip(factor) {
            *target += mu * coefficient;
        }
    }
    assert_eq!(joined, expected);
}

#[test]
fn stage3_a_product_matches_independent_signed_remainder_and_response_digits() {
    let mut rng = StdRng::seed_from_u64(0xa753_0239);
    for owners in [1, 2] {
        for rows in [1usize, 2, 3, 5] {
            for rank in [1usize, 3] {
                // Public descriptors currently require power-of-two source rows.
                // An internal live-row cutoff also exercises odd-row arithmetic.
                let plan = group_plan(rows.next_power_of_two(), rank, owners);
                let row_stride = rank.next_power_of_two() * STRIDE;
                let prefix_len = (rows * row_stride).next_power_of_two() * 2;
                let point = (0..plan.repacking().domain_len().trailing_zeros())
                    .map(|_| random_scalar(&mut rng))
                    .collect::<Vec<_>>();
                let rho = (0..owners * rank)
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
                    let element = (j / rows) * rows.next_power_of_two() + j % rows;
                    let mut digit_power = F::one();
                    for digit in 0..4 {
                        for (k, &weight) in weights.iter().enumerate() {
                            let address = plan
                                .repacking()
                                .coefficient_address(0, element, digit, k)
                                .unwrap();
                            direct_at_point +=
                                eq_eval_at_index(&point, address) * digit_power * weight;
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
                let mut response = vec![[0i128; DEGREE]; owners * rows];
                let mut weight_claim = F::zero();
                for (j, response) in response.iter_mut().enumerate() {
                    let element = (j / rows) * rows.next_power_of_two() + j % rows;
                    let mut digit_power = 1i128;
                    for digit in 0..4 {
                        for (k, response) in response.iter_mut().enumerate() {
                            let address = plan
                                .repacking()
                                .coefficient_address(0, element, digit, k)
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
                for (owner, owner_rho) in rho.chunks_exact(rank).enumerate() {
                    for (matrix_row, rho) in matrix.iter().zip(owner_rho) {
                        let mut convolution = vec![0i128; 2 * DEGREE - 1];
                        for (entry, response) in matrix_row
                            .iter()
                            .zip(&response[owner * rows..(owner + 1) * rows])
                        {
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
                }
                assert_eq!(
                    weight_claim, rem_claim,
                    "response contraction owners={owners}, rows={rows}, rank={rank}"
                );
            }
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

#[test]
fn full_size_typed_prefix_requires_padded_rank_and_coefficient_axes() {
    // (carriers, columns, rank, rows, padded rank * 1024, domain)
    for (carriers, columns, rank, rows, row_stride, domain) in [
        (1usize << 22, 256, 3, 4096, 4096, 1usize << 24),
        (1 << 23, 256, 3, 8192, 4096, 1 << 25),
        (1 << 10, 2, 3, 128, 4096, 1 << 19),
        (1 << 10, 2, 1, 128, 1024, 1 << 17),
        (1 << 10, 2, 5, 128, 8192, 1 << 20),
    ] {
        let descriptor = BinarySourceDescriptor::new(
            BinaryIndexMap::new(128, carriers, carriers, columns).unwrap(),
            vec![[1; 32]],
            BinaryCommitmentProfiles {
                a: [2; 32],
                b: [3; 32],
            },
        )
        .unwrap();
        let size = BinaryAPrefixLen::new(descriptor.index_map(), rank).unwrap();
        assert_eq!(
            size,
            BinaryAPrefixLen {
                rows,
                row_stride,
                natural_len: rows * row_stride,
                domain_len: domain,
            }
        );
        let plan = BinaryRootWitnessPlan::new(&descriptor, rank, 64)
            .unwrap()
            .with_prime_planes(true, 1)
            .unwrap();
        let point = vec![F::zero(); plan.repacking().domain_len().trailing_zeros() as usize];
        let rho = vec![[F::zero(); 648]; rank];
        // Rounding rank*m*648 as a flat array loses the padding on both
        // independently addressed axes of the canonical typed A layout.
        assert!(BinaryAProduct::new(&plan, &rho, &point, domain / 2).is_err());
        assert!(size.validate_prefix(domain / 2).is_err());
        assert!(size.validate_prefix(domain + 1).is_err());
        size.validate_prefix(domain).unwrap();
        size.validate_prefix(2 * domain).unwrap();
        let product = BinaryAProduct::new(&plan, &rho, &point, domain).unwrap();
        let terminal = vec![F::zero(); domain.trailing_zeros() as usize];
        assert_eq!(product.evaluate(&terminal).unwrap(), F::zero());
    }
    let index = BinaryIndexMap::new(128, 1 << 10, 1 << 10, 2).unwrap();
    assert!(BinaryAPrefixLen::new(&index, 0).is_err());
    assert!(BinaryAPrefixLen::new(&index, usize::MAX).is_err());
}

#[test]
fn p64_coordinate_integer_adjoint_and_typed_a_match_independent_long_division() {
    use crate::binary_source::{binary_direct_a_weights, BinaryAOperand, CANONICAL_IMAGE_WINDOW};
    use akita_challenges::{Challenges, SparseChallenge};
    use akita_params::binary_setup::BinaryFieldProfile;
    use jolt_field::{CanonicalEncoding, Ext2, ExtField, Prime64Offset59 as Base};
    type E = Ext2<Base>;
    let descriptor = group_plan(2, 3, 2).descriptor().clone();
    let plan =
        BinaryRootWitnessPlan::for_profile(&descriptor, 3, 64, BinaryFieldProfile::P64, true, 0)
            .unwrap();
    let mut rng = StdRng::seed_from_u64(59);
    let matrix = (0..3)
        .map(|_| {
            (0..2)
                .map(|_| {
                    std::array::from_fn(|_| {
                        i128::from(
                            rng.gen_range(CANONICAL_IMAGE_WINDOW.0..=CANONICAL_IMAGE_WINDOW.1),
                        )
                    })
                })
                .collect()
        })
        .collect::<Vec<_>>();
    let mut rho = (0..6)
        .map(|_| std::array::from_fn(|_| E::from_base_fn(|_| Base::from_u64(rng.gen()))))
        .collect::<Vec<_>>();
    rho[0][0] = E::from_base_fn(|i| {
        if i == 0 {
            -Base::one()
        } else {
            Base::from_u128((1u128 << 43) - 1)
        }
    });
    rho[0][1] = E::from_base_fn(|i| if i == 0 { Base::zero() } else { -Base::one() });
    assert!(rho
        .iter()
        .flatten()
        .all(|x| x.base_coefficient(1) != Base::zero()));
    let expected = direct_response_weights(&matrix, &rho);
    let challenges = Challenges::from_sparse(
        vec![
            SparseChallenge {
                positions: Default::default(),
                coeffs: Default::default()
            };
            2
        ],
        1,
        2,
    )
    .unwrap();
    let start = std::time::Instant::now();
    let compiled = binary_direct_a_weights(
        &plan,
        Some(BinaryAOperand::new::<Base>(&matrix)),
        &challenges,
        &rho,
    )
    .unwrap();
    eprintln!(
        "p64 coordinate A adjoints rows=2 owners=2 elapsed_us={}",
        start.elapsed().as_micros()
    );
    let dense = compiled.materialize().unwrap();
    for (element, row) in expected.iter().enumerate() {
        for digit in 0..4 {
            for (k, &value) in row.iter().enumerate() {
                let address = plan
                    .repacking()
                    .coefficient_address(0, element, digit, k)
                    .unwrap();
                assert_eq!(dense[address], value * E::from_u64(8u64.pow(digit as u32)));
            }
        }
    }
    let point = (0..plan.repacking().domain_len().trailing_zeros())
        .map(|_| E::from_base_fn(|_| Base::from_u64(rng.gen())))
        .collect::<Vec<_>>();
    let prefix = 2 * 4 * STRIDE;
    let product = BinaryAProduct::new(&plan, &rho, &point, prefix).unwrap();
    let (index, coefficient) = product.factors().unwrap();
    let direct = expected
        .iter()
        .enumerate()
        .fold(E::zero(), |sum, (element, row)| {
            sum + row.iter().enumerate().fold(E::zero(), |sum, (k, &value)| {
                sum + (0..4).fold(E::zero(), |sum, digit| {
                    sum + value
                        * E::from_u64(8u64.pow(digit as u32))
                        * eq_eval_at_index(
                            &point,
                            plan.repacking()
                                .coefficient_address(0, element, digit, k)
                                .unwrap(),
                        )
                })
            })
        });
    assert_eq!(matrix_claim(&matrix, &index, &coefficient), direct);
    for coordinate in 0..2 {
        // Each independent long-division check also matches the corresponding
        // coordinate of the extension result, including canonical p-1 values.
        let base_rho = rho
            .iter()
            .map(|row| row.map(|x| x.base_coefficient(coordinate)))
            .collect::<Vec<_>>();
        let base_expected = direct_response_weights(&matrix, &base_rho);
        for (extension, base) in expected
            .iter()
            .flatten()
            .zip(base_expected.iter().flatten())
        {
            assert_eq!(extension.base_coefficient(coordinate), *base);
            assert!(base.to_u128_checked().unwrap() < (1u128 << 86));
        }
    }
    let twice_bound =
        2 * 3 * 3 * 648 * u128::from(CANONICAL_IMAGE_WINDOW.1.unsigned_abs()) * (1u128 << 43);
    assert_eq!(twice_bound, 731_182_933_063_452_850_126_848);
    assert!(twice_bound < 67_091_329u128 * 134_217_649 * 268_433_353);
}
