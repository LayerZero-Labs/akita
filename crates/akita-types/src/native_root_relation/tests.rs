use super::*;
use crate::binary_source::{BinaryCommitmentProfiles, BinaryIndexMap};
use crate::root_repacking::RootRepackingSegment;
use akita_params::RelationRowGeometry;
use jolt_field::{One, Prime128OffsetA7F7, Ring, Zero};

type F = Prime128OffsetA7F7;

fn descriptor() -> BinarySourceDescriptor {
    BinarySourceDescriptor::new(
        BinaryIndexMap::new(8, 19, 8, 2).unwrap(),
        vec![[1; 32]],
        BinaryCommitmentProfiles {
            a: [2; 32],
            b: [3; 32],
        },
    )
    .unwrap()
}

/// Independent integer convolution followed by monic long division.
/// No quotient-ring or shifted-remainder implementation is used here.
fn reference_product(polynomial: RelationPolynomial, left: &[i128], right: &[i128]) -> Vec<i128> {
    let degree = polynomial.degree();
    assert_eq!(left.len(), degree);
    assert_eq!(right.len(), degree);
    let mut product = vec![0i128; 2 * degree - 1];
    for (i, &a) in left.iter().enumerate() {
        for (j, &b) in right.iter().enumerate() {
            product[i + j] += a * b;
        }
    }
    for exponent in (degree..product.len()).rev() {
        let quotient = product[exponent];
        product[exponent] = 0;
        product[exponent - degree] -= quotient;
        match polynomial.kind() {
            RelationPolynomialKind::Negacyclic => {}
            RelationPolynomialKind::Trinomial(sign) => {
                let middle = exponent - degree + degree / 2;
                match sign {
                    TrinomialSign::Plus => product[middle] -= quotient,
                    TrinomialSign::Minus => product[middle] += quotient,
                }
            }
        }
    }
    product.truncate(degree);
    product
}

fn balanced_digits(values: &[i128], depth: usize) -> Vec<i8> {
    let mut digits = vec![0; values.len() * depth];
    for (coefficient, &value) in values.iter().enumerate() {
        let mut remaining = value;
        for digit in 0..depth {
            let residue = remaining.rem_euclid(8);
            let balanced = if residue >= 4 { residue - 8 } else { residue };
            digits[digit * values.len() + coefficient] = i8::try_from(balanced).unwrap();
            remaining = (remaining - balanced) / 8;
        }
        assert_eq!(remaining, 0, "reference integer fits its digit depth");
    }
    digits
}

fn fields(coefficients: &[i128]) -> Vec<F> {
    coefficients.iter().copied().map(F::from_i128).collect()
}

fn term(segment: usize, multiplier: &[i128]) -> NativeRootTerm<F> {
    NativeRootTerm {
        segment,
        element: 0,
        multiplier: fields(multiplier),
    }
}

fn evaluate_integer_polynomial(coefficients: &[i128], alpha: F) -> F {
    coefficients
        .iter()
        .rev()
        .fold(F::zero(), |value, &coefficient| {
            value * alpha + F::from_i128(coefficient)
        })
}

fn fixture() -> (NativeRootRelation<F>, Vec<F>, Vec<Vec<i128>>) {
    let binary_polynomial = RelationPolynomial::minus_trinomial(648).unwrap();
    let ordinary_polynomial = RelationPolynomial::negacyclic(8).unwrap();
    let binary_geometry = RelationRowGeometry::for_polynomial(binary_polynomial, 1, 1024).unwrap();
    let ordinary_geometry = RelationRowGeometry::native(8).unwrap();
    let repacking = RootCoefficientRepacking::new(
        vec![
            RootRepackingSegment {
                geometry: binary_geometry,
                elements: 1,
                digits: 1,
            },
            RootRepackingSegment {
                geometry: binary_geometry,
                elements: 1,
                digits: 4,
            },
            RootRepackingSegment {
                geometry: binary_geometry,
                elements: 1,
                digits: 4,
            },
            RootRepackingSegment {
                geometry: ordinary_geometry,
                elements: 1,
                digits: 2,
            },
            RootRepackingSegment {
                geometry: ordinary_geometry,
                elements: 1,
                digits: 4,
            },
        ],
        128,
    )
    .unwrap();

    let mut source = vec![0i128; 648];
    for coefficient in [0, 2, 324, 647] {
        source[coefficient] = 1;
    }
    let mut matrix = vec![0i128; 648];
    for (coefficient, value) in [(0, 2), (1, 1), (323, -1), (647, 1)] {
        matrix[coefficient] = value;
    }
    let mut challenge = vec![0i128; 648];
    for (coefficient, value) in [(0, 1), (1, -1), (324, 1), (647, 1)] {
        challenge[coefficient] = value;
    }
    let image = reference_product(binary_polynomial, &matrix, &source);
    let response = reference_product(binary_polynomial, &challenge, &source);
    let ordinary_source = vec![1, -2, 3, -4, 5, -6, 7, -8];
    let ordinary_matrix = vec![2, 1, 0, -1, 0, 0, 1, 1];
    let ordinary_response =
        reference_product(ordinary_polynomial, &ordinary_matrix, &ordinary_source);
    let values = [source, image, response, ordinary_source, ordinary_response];
    let natural = values
        .iter()
        .zip(repacking.segments())
        .map(|(values, segment)| balanced_digits(values, segment.digits))
        .collect::<Vec<_>>();
    let references = natural.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let table = repacking
        .repack(&references)
        .unwrap()
        .into_iter()
        .map(|digit| F::from_i64(i64::from(digit)))
        .collect::<Vec<_>>();
    repacking.validate_digit_table(&table).unwrap();

    let mut minus_one = vec![0; 648];
    minus_one[0] = -1;
    let nonzero_target = values[0]
        .iter()
        .zip(&values[1])
        .zip(&values[2])
        .map(|((&source, &image), &response)| source + 2 * image - 3 * response)
        .collect::<Vec<_>>();
    let targets = vec![
        vec![0; 648],
        vec![0; 648],
        nonzero_target,
        vec![0; 8],
        values[4].clone(),
    ];
    let mut one = vec![0; 648];
    one[0] = 1;
    let mut two = vec![0; 648];
    two[0] = 2;
    let mut minus_three = vec![0; 648];
    minus_three[0] = -3;
    let mut ordinary_minus_one = vec![0; 8];
    ordinary_minus_one[0] = -1;
    let mut ordinary_one = vec![0; 8];
    ordinary_one[0] = 1;
    let groups = vec![
        NativeRootGroup {
            kind: NativeRootGroupKind::Binary(descriptor()),
            rows: vec![
                NativeRootRow {
                    polynomial: binary_polynomial,
                    terms: vec![term(0, &matrix), term(1, &minus_one)],
                    target: fields(&targets[0]),
                },
                NativeRootRow {
                    polynomial: binary_polynomial,
                    terms: vec![term(0, &challenge), term(2, &minus_one)],
                    target: fields(&targets[1]),
                },
                NativeRootRow {
                    polynomial: binary_polynomial,
                    terms: vec![term(0, &one), term(1, &two), term(2, &minus_three)],
                    target: fields(&targets[2]),
                },
            ],
        },
        NativeRootGroup {
            kind: NativeRootGroupKind::Ordinary(FoldChallengeFamily::pm1_only(4)),
            rows: vec![
                NativeRootRow {
                    polynomial: ordinary_polynomial,
                    terms: vec![term(3, &ordinary_matrix), term(4, &ordinary_minus_one)],
                    target: fields(&targets[3]),
                },
                NativeRootRow {
                    polynomial: ordinary_polynomial,
                    terms: vec![term(4, &ordinary_one)],
                    target: fields(&targets[4]),
                },
            ],
        },
    ];
    (
        NativeRootRelation::new(repacking, groups).unwrap(),
        table,
        targets,
    )
}

fn compile(relation: &NativeRootRelation<F>) -> NativeRootWeights<F> {
    let row_weights = [3, 5, 7, 11, 13].map(F::from_u64);
    let padding = (0..relation.repacking().domain_len())
        .filter(|&index| relation.repacking().is_padding_address(index).unwrap())
        .map(|index| F::from_u64(17 + index as u64))
        .collect::<Vec<_>>();
    relation
        .ring_switch(F::from_u64(7), &row_weights, &padding)
        .unwrap()
}

#[test]
fn mixed_groups_ring_switch_matches_independent_integer_relation() {
    let (relation, table, targets) = fixture();
    assert_eq!(relation.groups().len(), 2);
    assert_eq!(relation.row_count(), 5);
    assert_eq!(
        relation.groups()[0].kind.challenge_family(),
        FoldChallengeFamily::BinaryRoot
    );
    assert_eq!(
        relation.groups()[1].kind.challenge_family(),
        FoldChallengeFamily::pm1_only(4)
    );
    assert_eq!(
        relation.groups()[0].kind.commitment_modulus(),
        Some(BINARY_SOURCE_Q)
    );
    assert_eq!(relation.groups()[1].kind.commitment_modulus(), None);
    let weights = compile(&relation);
    let expected = targets
        .iter()
        .zip([3, 5, 7, 11, 13])
        .map(|(target, batch)| {
            F::from_u64(batch) * evaluate_integer_polynomial(target, F::from_u64(7))
        })
        .sum::<F>();
    assert_eq!(weights.claim(), expected);
    let actual = weights
        .terms()
        .iter()
        .map(|&(address, weight)| table[address] * weight)
        .sum::<F>();
    assert_eq!(actual, expected);
    weights.check_table(&table).unwrap();

    // Compare the public sparse MLE with its direct Boolean-basis definition.
    let point = (0..weights.table_len().trailing_zeros())
        .map(|coordinate| F::from_u64(19 + u64::from(coordinate)))
        .collect::<Vec<_>>();
    let direct_mle = weights
        .terms()
        .iter()
        .map(|&(address, weight)| {
            point.iter().enumerate().fold(weight, |product, (bit, &r)| {
                product
                    * if address & (1 << bit) == 0 {
                        F::one() - r
                    } else {
                        r
                    }
            })
        })
        .sum::<F>();
    assert_eq!(weights.evaluate(&point).unwrap(), direct_mle);
}

#[test]
fn each_source_image_response_and_repacked_padding_mutation_rejects() {
    let (relation, table, _) = fixture();
    let weights = compile(&relation);
    for segment in 0..relation.repacking().segments().len() {
        // Every selected natural digit changes within [-4,3], so the linear
        // relation must catch the mutation independently of range checking.
        let address = relation
            .repacking()
            .coefficient_address(segment, 0, 0, 0)
            .unwrap();
        let mut changed = table.clone();
        changed[address] -= F::one();
        relation.repacking().validate_digit_table(&changed).unwrap();
        assert!(weights.check_table(&changed).is_err(), "segment {segment}");
    }
    for address in [648, 1023, relation.repacking().domain_len() - 1] {
        assert!(relation.repacking().is_padding_address(address).unwrap());
        let mut changed = table.clone();
        changed[address] = F::one();
        assert!(relation.repacking().validate_digit_table(&changed).is_err());
        assert!(weights.check_table(&changed).is_err());
    }
}

#[test]
fn malformed_group_families_terms_and_geometry_reject() {
    let (relation, _, _) = fixture();
    let repacking = relation.repacking().clone();
    let groups = relation.groups().to_vec();
    assert!(NativeRootRelation::<F>::new(repacking.clone(), vec![]).is_err());
    let mut changed = groups.clone();
    changed[0].rows.clear();
    assert!(NativeRootRelation::new(repacking.clone(), changed).is_err());
    let mut changed = groups.clone();
    changed[1].kind = NativeRootGroupKind::Ordinary(FoldChallengeFamily::BinaryRoot);
    assert!(NativeRootRelation::new(repacking.clone(), changed).is_err());
    let mut changed = groups.clone();
    changed[0].kind = NativeRootGroupKind::Ordinary(FoldChallengeFamily::pm1_only(4));
    assert!(NativeRootRelation::new(repacking.clone(), changed).is_err());
    for mutation in 0..6 {
        let mut changed = groups.clone();
        let row = &mut changed[0].rows[0];
        match mutation {
            0 => {
                row.target.pop();
            }
            1 => row.terms.clear(),
            2 => row.terms[0].segment = usize::MAX,
            3 => row.terms[0].element = 1,
            4 => {
                row.terms[0].multiplier.pop();
            }
            5 => row.polynomial = RelationPolynomial::plus_trinomial(648).unwrap(),
            _ => unreachable!(),
        }
        assert!(NativeRootRelation::new(repacking.clone(), changed).is_err());
    }
    let mut shapes = repacking.segments().to_vec();
    shapes[0].geometry = RelationRowGeometry::for_polynomial(
        RelationPolynomial::minus_trinomial(648).unwrap(),
        2,
        1024,
    )
    .unwrap();
    let wrong_planes = RootCoefficientRepacking::new(shapes, 128).unwrap();
    assert!(NativeRootRelation::new(wrong_planes, groups.clone()).is_err());
    let mut shapes = repacking.segments().to_vec();
    shapes[0].geometry = RelationRowGeometry::native(1024).unwrap();
    let wrong_polynomial = RootCoefficientRepacking::new(shapes, 128).unwrap();
    assert!(NativeRootRelation::new(wrong_polynomial, groups).is_err());

    assert!(relation.ring_switch(F::from_u64(7), &[], &[]).is_err());
    assert!(relation
        .ring_switch(F::from_u64(7), &[F::one(); 5], &[])
        .is_err());
    let weights = compile(&relation);
    assert!(weights.check_table(&[]).is_err());
    assert!(weights.evaluate(&[]).is_err());
    assert!(NativeRootWeights::<F>::new(3, vec![], F::zero()).is_err());
    assert!(NativeRootWeights::<F>::new(8, vec![(8, F::one())], F::zero()).is_err());
}

#[test]
fn both_trinomial_signs_match_independent_long_division() {
    for polynomial in [
        RelationPolynomial::plus_trinomial(162).unwrap(),
        RelationPolynomial::minus_trinomial(648).unwrap(),
    ] {
        let degree = polynomial.degree();
        let source = (0..degree)
            .map(|index| index as i128 % 3 - 1)
            .collect::<Vec<_>>();
        let multiplier = (0..degree)
            .map(|index| (index * 7 + 2) as i128 % 5 - 2)
            .collect::<Vec<_>>();
        let target = reference_product(polynomial, &multiplier, &source);
        let repacking = RootCoefficientRepacking::new(
            vec![RootRepackingSegment {
                geometry: RelationRowGeometry::for_polynomial(
                    polynomial,
                    1,
                    degree.next_power_of_two(),
                )
                .unwrap(),
                elements: 1,
                digits: 1,
            }],
            128,
        )
        .unwrap();
        let natural = balanced_digits(&source, 1);
        let table = repacking
            .repack(&[&natural])
            .unwrap()
            .into_iter()
            .map(|digit| F::from_i64(i64::from(digit)))
            .collect::<Vec<_>>();
        let relation = NativeRootRelation::new(
            repacking,
            vec![NativeRootGroup {
                kind: NativeRootGroupKind::Binary(descriptor()),
                rows: vec![NativeRootRow {
                    polynomial,
                    terms: vec![term(0, &multiplier)],
                    target: fields(&target),
                }],
            }],
        )
        .unwrap();
        let padding = (0..relation.repacking().domain_len())
            .filter(|&index| relation.repacking().is_padding_address(index).unwrap())
            .map(|_| F::from_u64(13))
            .collect::<Vec<_>>();
        let weights = relation
            .ring_switch(F::from_u64(11), &[F::from_u64(17)], &padding)
            .unwrap();
        assert_eq!(
            weights.claim(),
            F::from_u64(17) * evaluate_integer_polynomial(&target, F::from_u64(11))
        );
        weights.check_table(&table).unwrap();
    }
}
