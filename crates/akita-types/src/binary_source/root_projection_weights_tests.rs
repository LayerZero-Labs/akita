use super::*;
use crate::binary_source::{BinaryCommitmentProfiles, BinaryIndexMap, BinarySourceDescriptor};
use akita_algebra::trinomial::module_map::pack_signed_components;
use akita_algebra::trinomial::signed::signed_trinomial_product;
use akita_challenges::SparseChallenge;
use jolt_algebra::{One, Prime128Offset275 as F, Ring, Zero};

#[test]
fn projection_direct_weights_match_integer_residual_for_arbitrary_coefficient_rows() {
    let descriptor = BinarySourceDescriptor::new(
        BinaryIndexMap::new(128, 8, 8, 2).unwrap(),
        vec![[1; 32], [9; 32]],
        BinaryCommitmentProfiles {
            a: [2; 32],
            b: [3; 32],
        },
    )
    .unwrap();
    let plan = BinaryRootWitnessPlan::new(&descriptor, 3, 256).unwrap();
    let challenges = Challenges::from_sparse(
        (0..4)
            .map(|column| SparseChallenge {
                positions: vec![0, 81, 161].into(),
                coeffs: vec![1, -1, if column == 0 { 1 } else { -1 }].into(),
            })
            .collect(),
        2,
        2,
    )
    .unwrap();
    let matrix: Vec<Vec<[i128; 648]>> = (0..3)
        .map(|i| {
            vec![std::array::from_fn(|t| {
                ((t * 7 + i * 13) % 101) as i128 - 50
            })]
        })
        .collect();
    let rho: Vec<[F; 648]> = (0..6)
        .map(|i| std::array::from_fn(|t| F::from_u64((t * 17 + i * 29 + 1) as u64)))
        .collect();
    let mut natural = plan
        .repacking()
        .segments()
        .iter()
        .map(|segment| {
            vec![
                0i8;
                segment.elements * segment.digits * segment.geometry.polynomial_modulus_dimension()
            ]
        })
        .collect::<Vec<_>>();
    for (segment, values) in natural.iter_mut().take(2).enumerate() {
        for (index, digit) in values.iter_mut().enumerate() {
            *digit = ((index * 3 + segment * 5) % 8) as i8 - 4;
        }
    }
    let recompose = |segment: usize, element: usize, digits: usize| {
        std::array::from_fn::<_, 648, _>(|t| {
            (0..digits)
                .map(|d| {
                    i128::from(natural[segment][(element * digits + d) * 648 + t])
                        * 8i128.pow(d as u32)
                })
                .sum::<i128>()
        })
    };
    let scalar = binary_scalar_challenges(&descriptor, &challenges).unwrap();
    let mut expected = F::zero();
    for owner in 0..2 {
        let response = recompose(0, owner, 4);
        for (i, row) in matrix.iter().enumerate() {
            let mut residual =
                signed_trinomial_product::<648, MinusTrinomial>(&row[0], &response).unwrap();
            for (column, scalar) in scalar[owner * 2..(owner + 1) * 2].iter().enumerate() {
                let challenge =
                    pack_signed_components(&[*scalar, [0; 162], [0; 162], [0; 162]]).unwrap();
                let image = recompose(1, (owner * 2 + column) * 3 + i, 8);
                let product =
                    signed_trinomial_product::<648, MinusTrinomial>(&challenge, &image).unwrap();
                for (value, term) in residual.iter_mut().zip(product) {
                    *value -= term;
                }
            }
            for (&weight, value) in rho[owner * 3 + i].iter().zip(residual) {
                expected += weight * F::from_i128(value);
            }
        }
    }
    let weights = binary_direct_a_weights(
        &plan,
        Some(BinaryAOperand::from_canonical(&matrix)),
        &challenges,
        &rho,
    )
    .unwrap();
    let table = plan
        .repacking()
        .repack(&natural.iter().map(Vec::as_slice).collect::<Vec<_>>())
        .unwrap()
        .into_iter()
        .map(|x| F::from_i64(i64::from(x)))
        .collect::<Vec<_>>();
    let terms = weights
        .materialize()
        .unwrap()
        .into_iter()
        .enumerate()
        .filter(|(_, weight)| !weight.is_zero())
        .collect();
    let relation = NativeRootWeights::new(table.len(), terms, expected).unwrap();
    relation.check_table(&table).unwrap();
    let mut changed = table.clone();
    let address = plan.repacking().coefficient_address(1, 0, 0, 647).unwrap();
    changed[address] += F::one();
    assert!(relation.check_table(&changed).is_err());
    assert!(binary_direct_a_weights(
        &plan,
        Some(BinaryAOperand::from_canonical(&matrix)),
        &challenges,
        &rho[..2]
    )
    .is_err());
}

use jolt_algebra::{Ext2, Prime64Offset59};
use rand::{rngs::StdRng, Rng, SeedableRng};
type E64 = Ext2<Prime64Offset59>;

fn extension(rng: &mut StdRng) -> E64 {
    E64::new(
        Prime64Offset59::from_u64(rng.gen()),
        Prime64Offset59::from_u64(rng.gen::<u64>() | 1),
    )
}

fn quotient_plan(owners: usize, rows: usize, columns: usize) -> BinaryRootWitnessPlan {
    use akita_params::binary_setup::BinaryFieldProfile;
    let descriptor = BinarySourceDescriptor::new(
        BinaryIndexMap::new(128, 4 * rows * columns, 4 * rows * columns, columns).unwrap(),
        (0..owners).map(|owner| [owner as u8; 32]).collect(),
        BinaryCommitmentProfiles {
            a: [2; 32],
            b: [3; 32],
        },
    )
    .unwrap();
    BinaryRootWitnessPlan::for_profile(&descriptor, 3, 256, BinaryFieldProfile::P64, true, 0)
        .unwrap()
}

#[test]
fn p64_generic_adjoint_and_quotient_weights_match_dense_reference_at_extension_rho() {
    let u = E64::new(Prime64Offset59::zero(), Prime64Offset59::one());
    assert_eq!(u * u, E64::from_u64(2));
    let mut rng = StdRng::seed_from_u64(0x0006_4864);
    for (owners, rows, columns) in [(1, 1, 2), (2, 1, 2), (1, 2, 1)] {
        let plan = quotient_plan(owners, rows, columns);
        let challenges = Challenges::from_sparse(
            (0..owners * columns)
                .map(|_| SparseChallenge {
                    positions: vec![0, 81, 161].into(),
                    coeffs: vec![1, -1, 1].into(),
                })
                .collect(),
            columns,
            owners,
        )
        .unwrap();
        let matrix = (0..3)
            .map(|_| {
                (0..rows)
                    .map(|_| std::array::from_fn(|_| rng.gen_range(-100..=100)))
                    .collect()
            })
            .collect::<Vec<Vec<[i128; 648]>>>();
        let rho = (0..3 * owners)
            .map(|_| std::array::from_fn(|_| extension(&mut rng)))
            .collect::<Vec<_>>();
        for with_a in [false, true] {
            let operand = with_a.then_some(BinaryAOperand::new::<Prime64Offset59>(&matrix));
            let fast = binary_direct_a_weights(&plan, operand, &challenges, &rho).unwrap();
            let reference = crate::binary_source::root_public::reference::direct_a_weights(
                &plan,
                with_a.then_some(matrix.as_slice()),
                &challenges,
                &rho,
            )
            .unwrap();
            let dense = fast.materialize().unwrap();
            assert_eq!(dense, reference.materialize().unwrap());
            if with_a {
                let natural = plan
                    .repacking()
                    .segments()
                    .iter()
                    .map(|shape| {
                        (0..shape.elements
                            * shape.digits
                            * shape.geometry.polynomial_modulus_dimension())
                            .map(|_| rng.gen_range(-4..=3))
                            .collect::<Vec<i8>>()
                    })
                    .collect::<Vec<_>>();
                let integer = plan
                    .repacking()
                    .repack(&natural.iter().map(Vec::as_slice).collect::<Vec<_>>())
                    .unwrap();
                let table = integer
                    .into_iter()
                    .map(|digit| E64::from_i64(i64::from(digit)))
                    .collect::<Vec<_>>();
                let recompose = |segment: usize, element: usize| {
                    let shape = plan.repacking().segments()[segment];
                    std::array::from_fn::<_, 648, _>(|coefficient| {
                        (0..shape.digits)
                            .map(|digit| {
                                i128::from(
                                    natural[segment]
                                        [(element * shape.digits + digit) * 648 + coefficient],
                                ) * 8i128.pow(digit as u32)
                            })
                            .sum::<i128>()
                    })
                };
                let scalars = binary_scalar_challenges(plan.descriptor(), &challenges).unwrap();
                let mut residuals = vec![[0i128; 648]; 3 * owners];
                for owner in 0..owners {
                    for (row_index, matrix_row) in matrix.iter().enumerate() {
                        let residual = &mut residuals[owner * 3 + row_index];
                        for (column, a) in matrix_row.iter().enumerate() {
                            let z = recompose(0, owner * rows + column);
                            let product =
                                signed_trinomial_product::<648, MinusTrinomial>(a, &z).unwrap();
                            for (slot, term) in residual.iter_mut().zip(product) {
                                *slot += term;
                            }
                        }
                        for column in 0..columns {
                            let scalar = scalars[owner * columns + column];
                            let challenge =
                                pack_signed_components(&[scalar, [0; 162], [0; 162], [0; 162]])
                                    .unwrap();
                            let image = recompose(1, (owner * columns + column) * 3 + row_index);
                            let product =
                                signed_trinomial_product::<648, MinusTrinomial>(&challenge, &image)
                                    .unwrap();
                            for (slot, term) in residual.iter_mut().zip(product) {
                                *slot -= term;
                            }
                        }
                    }
                }
                let expected = crate::binary_source::root_public::reference::coefficient_defect(
                    &plan, &residuals, &table, &rho,
                )
                .unwrap();
                let actual = dense
                    .iter()
                    .zip(&table)
                    .fold(E64::zero(), |sum, (&weight, &value)| sum + weight * value);
                assert_eq!(actual, expected);
            }
            let point = (0..plan.repacking().domain_len().trailing_zeros())
                .map(|_| extension(&mut rng))
                .collect::<Vec<_>>();
            assert_eq!(
                fast.evaluate(&point).unwrap(),
                reference.evaluate(&point).unwrap()
            );
            let segment = plan.quotient_segment().unwrap();
            let depth = plan.projection_bounds().quotient().unwrap().digits;
            for (element, row) in rho.iter().enumerate() {
                let mut power = E64::one();
                for digit in 0..depth {
                    for (coefficient, &weight) in row.iter().enumerate() {
                        let address = plan
                            .repacking()
                            .coefficient_address(segment, element, digit, coefficient)
                            .unwrap();
                        assert_eq!(dense[address], -E64::from_u64(14_253_409) * power * weight);
                    }
                    power *= E64::from_u64(8);
                }
            }
            for (address, &weight) in dense.iter().enumerate() {
                if plan.repacking().is_padding_address(address).unwrap() {
                    assert_eq!(weight, E64::zero());
                }
            }
        }
    }
}

fn write_balanced(natural: &mut [i8], digits: usize, coefficient: usize, mut value: i128) {
    for digit in 0..digits {
        let residue = value.rem_euclid(8);
        let next = if residue >= 4 { residue - 8 } else { residue };
        natural[digit * 648 + coefficient] = next as i8;
        value = (value - next) / 8;
    }
    assert_eq!(value, 0);
}

#[test]
fn p64_coefficient_relation_vanishes_exactly_for_planted_quotients() {
    let plan = quotient_plan(1, 1, 2);
    let challenges = Challenges::from_sparse(
        vec![
            SparseChallenge {
                positions: vec![0].into(),
                coeffs: vec![1].into()
            };
            2
        ],
        2,
        1,
    )
    .unwrap();
    let mut matrix = vec![vec![[0i128; 648]]; 3];
    matrix[0][0][0] = i128::from(CANONICAL_IMAGE_WINDOW.1);
    let mut rng = StdRng::seed_from_u64(59);
    let rho = (0..3)
        .map(|_| std::array::from_fn(|_| extension(&mut rng)))
        .collect::<Vec<_>>();
    let weights = binary_direct_a_weights(
        &plan,
        Some(BinaryAOperand::new::<Prime64Offset59>(&matrix)),
        &challenges,
        &rho,
    )
    .unwrap();
    let segment = plan.quotient_segment().unwrap();
    let depth = plan.projection_bounds().quotient().unwrap().digits;
    for quotient in [-4, -1, 1, 3] {
        let mut natural = plan
            .repacking()
            .segments()
            .iter()
            .map(|shape| {
                vec![
                    0i8;
                    shape.elements * shape.digits * shape.geometry.polynomial_modulus_dimension()
                ]
            })
            .collect::<Vec<_>>();
        // A*(2*k) - (-k) = (q-1)*k + k = q*k.
        write_balanced(&mut natural[0], 4, 0, 2 * quotient);
        write_balanced(&mut natural[1], 8, 0, -quotient);
        write_balanced(&mut natural[segment], depth, 0, quotient);
        let table = plan
            .repacking()
            .repack(&natural.iter().map(Vec::as_slice).collect::<Vec<_>>())
            .unwrap()
            .into_iter()
            .map(|digit| E64::from_i64(i64::from(digit)))
            .collect::<Vec<_>>();
        let mut residuals = vec![[0i128; 648]; 3];
        residuals[0][0] = 14_253_409 * quotient;
        assert_eq!(
            crate::binary_source::root_public::reference::coefficient_defect(
                &plan, &residuals, &table, &rho
            )
            .unwrap(),
            E64::zero()
        );
        weights.check_table(&table).unwrap();
        let reference = crate::binary_source::root_public::reference::direct_a_weights(
            &plan,
            Some(&matrix),
            &challenges,
            &rho,
        )
        .unwrap();
        reference.check_table(&table).unwrap();
        for digit in [0, depth - 1] {
            let mut altered = table.clone();
            let address = plan
                .repacking()
                .coefficient_address(segment, 0, digit, 0)
                .unwrap();
            // Move either endpoint inward, keeping the digit accepted.
            altered[address] += if quotient == 3 && digit == 0 {
                -E64::one()
            } else {
                E64::one()
            };
            let defect = crate::binary_source::root_public::reference::coefficient_defect(
                &plan, &residuals, &altered, &rho,
            )
            .unwrap();
            assert_ne!(defect, E64::zero());
            let evaluated = weights
                .materialize()
                .unwrap()
                .iter()
                .zip(&altered)
                .fold(E64::zero(), |sum, (&weight, &value)| sum + weight * value);
            assert_eq!(evaluated, defect);
            assert!(weights.check_table(&altered).is_err());
            assert!(reference.check_table(&altered).is_err());
        }
    }
    assert!(binary_direct_a_weights(&plan, None, &challenges, &rho[..2]).is_err());
}
