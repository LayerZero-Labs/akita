use super::*;
use akita_challenges::{
    BinaryChallengeProfile, BinaryScalarRing, FoldChallengeDrawDomain, FoldChallengeFamily,
    FoldDraw,
};
use akita_types::binary::{BinaryCommitmentProfiles, BinaryIndexMap};
use akita_types::native_root_relation::NativeRootRelation;
use jolt_field::{Prime128Offset275, Ring};

struct SeedDraw;
impl FoldDraw for SeedDraw {
    fn draw_message<T>(
        &mut self,
        len: usize,
        decode: impl FnOnce(&[u8]) -> Result<Option<T>, AkitaError>,
    ) -> Result<Option<T>, AkitaError> {
        decode(&vec![17; len])
    }
}

struct Fixture {
    descriptor: BinarySourceDescriptor,
    matrix: Vec<Vec<[i128; D]>>,
    sources: Vec<[i128; D]>,
    images: Vec<[Q24Image; D]>,
    challenges: Challenges,
}

// Independent polynomial convolution and descending monic division over Z.
fn reference(lhs: &[i128; D], rhs: &[i128; D]) -> Vec<i128> {
    let mut product = vec![0; 2 * D - 1];
    for (i, &left) in lhs.iter().enumerate() {
        for (j, &right) in rhs.iter().enumerate() {
            product[i + j] += left * right;
        }
    }
    product
}

fn divide(mut product: Vec<i128>) -> ([i128; D], Vec<i128>) {
    let mut quotient = vec![0; D - 1];
    for k in (D..product.len()).rev() {
        let value = product[k];
        quotient[k - D] = value;
        product[k] = 0;
        product[k - D] -= value;
        product[k - D / 2] += value;
    }
    (product[..D].try_into().unwrap(), quotient)
}

fn fixture() -> Fixture {
    let descriptor = BinarySourceDescriptor::new(
        BinaryIndexMap::new(128, 8, 8, 2).unwrap(),
        vec![[1; 32], [2; 32]],
        BinaryCommitmentProfiles {
            a: [3; 32],
            b: [4; 32],
        },
    )
    .unwrap();
    let matrix = (0..2)
        .map(|row| {
            vec![std::array::from_fn(|i| {
                i128::from(Q24_CENTERED_BOUND) - ((i * 13 + row * 19) % 200) as i128
            })]
        })
        .collect::<Vec<_>>();
    let sources = (0..4)
        .map(|element| {
            std::array::from_fn(|i| {
                if i < 4 * 128 && (i + element * 7) % 31 == 0 {
                    if (i / 4) % 2 == 0 {
                        1
                    } else {
                        -1
                    }
                } else {
                    0
                }
            })
        })
        .collect::<Vec<_>>();
    let mut images = Vec::new();
    for source in &sources {
        for row in &matrix {
            let remainder = divide(reference(&row[0], source)).0;
            images
                .push(remainder.map(|x| {
                    Q24Image::from_residue(x.rem_euclid(i128::from(Q24)) as u32).unwrap()
                }));
        }
    }
    let challenges = SeedDraw
        .draw_fold_challenges(
            &FoldChallengeFamily::new(
                FoldChallengeDrawDomain::EvaluationTrace,
                162,
                FoldChallengeDistribution::BinaryRoot,
                None,
            )
            .unwrap(),
            2,
            2,
        )
        .unwrap()
        .unwrap();
    Fixture {
        descriptor,
        matrix,
        sources,
        images,
        challenges,
    }
}

fn build(input: &Fixture) -> Result<BinaryRootRelation, AkitaError> {
    BinaryRootRelation::build(
        &input.descriptor,
        &input.matrix,
        &input.sources,
        &input.images,
        &input.challenges,
        FoldChallengeDistribution::BinaryRoot,
    )
}

#[test]
fn q24_root_matches_integer_fold_division_and_coefficient_repacking() {
    let input = fixture();
    let relation = build(&input).unwrap();
    for (polynomial, actual) in relation.response.iter().enumerate() {
        let mut expected = [0; D];
        for column in 0..2 {
            let source = input.sources[polynomial * 2 + column];
            let challenge = relation.embedded_challenges[polynomial * 2 + column];
            for (acc, value) in expected
                .iter_mut()
                .zip(divide(reference(&source, &challenge)).0)
            {
                *acc += value;
            }
        }
        assert_eq!(*actual, expected);
        for row in 0..2 {
            let mut expected = reference(&input.matrix[row][0], actual);
            for column in 0..2 {
                let image = input.images[(polynomial * 2 + column) * 2 + row]
                    .map(|x| i128::from(x.centered()));
                let challenge = relation.embedded_challenges[polynomial * 2 + column];
                for (acc, value) in expected.iter_mut().zip(reference(&image, &challenge)) {
                    *acc -= value;
                }
            }
            let (remainder, quotient) = divide(expected);
            let actual = &relation.rows[polynomial * 2 + row];
            assert_eq!(actual.remainder, remainder);
            assert_eq!(actual.quotient, quotient);
            assert!(
                remainder.iter().any(|&x| x != 0),
                "q congruence is not equality over P"
            );
            assert!(remainder.iter().all(|&x| x % i128::from(Q24) == 0));
            assert_eq!(
                relation
                    .residual_in_proof_field::<Prime128Offset275>(polynomial * 2 + row)
                    .unwrap(),
                remainder.map(Prime128Offset275::from_i128)
            );
        }
    }
    let (plan, table) =
        BinaryRootRelation::successor_table(&relation.response, &relation.images, 256).unwrap();
    let table = table
        .into_iter()
        .map(|x| Prime128Offset275::from_i64(i64::from(x)))
        .collect::<Vec<_>>();
    plan.validate_digit_table(&table).unwrap();
    for (element, response) in relation.response.iter().enumerate() {
        for (coefficient, &value) in response.iter().enumerate() {
            assert_eq!(
                plan.recompose_coefficient(&table, 0, element, coefficient)
                    .unwrap(),
                Prime128Offset275::from_i128(value)
            );
        }
    }
    for (element, image) in relation.images.iter().enumerate() {
        for (coefficient, &value) in image.iter().enumerate() {
            assert_eq!(
                plan.recompose_coefficient(&table, 1, element, coefficient)
                    .unwrap(),
                Prime128Offset275::from_i64(i64::from(value.centered()))
            );
        }
    }
    let address = plan.coefficient_address(0, 0, 0, 0).unwrap();
    let expected = plan.recompose_coefficient(&table, 0, 0, 0).unwrap();
    let mut corrupted = table;
    corrupted[address] += Prime128Offset275::one();
    assert_ne!(
        plan.recompose_coefficient(&corrupted, 0, 0, 0).unwrap(),
        expected
    );
    assert!(plan.destination_ring_dimension().is_power_of_two());
}

#[test]
fn q24_root_rejects_image_source_and_response_mutation() {
    let mut input = fixture();
    let relation = build(&input).unwrap();
    input.sources[0][0] += 1;
    assert!(build(&input).is_err());
    input.sources[0][0] -= 1;
    let value = input.images[0][0].centered();
    input.images[0][0] = Q24Image::from_centered(value + 1).unwrap();
    assert!(build(&input).is_err());
    input.images[0][0] = Q24Image::from_centered(value).unwrap();
    let mut response = relation.response;
    response[0][0] += 1;
    assert!(BinaryRootRelation::from_response(
        &input.descriptor,
        &input.matrix,
        response,
        &input.images,
        &input.challenges,
        FoldChallengeDistribution::BinaryRoot
    )
    .is_err());
}

#[test]
fn q24_root_rejects_wrong_family_counts_bounds_and_matrix_representatives() {
    let input = fixture();
    assert!(build(&input).is_ok());
    assert!(BinaryRootRelation::build(
        &input.descriptor,
        &input.matrix,
        &input.sources,
        &input.images,
        &input.challenges,
        FoldChallengeDistribution::pm1_only(46)
    )
    .is_err());
    let input = fixture();
    let mut response = build(&input).unwrap().response;
    response[0][0] = i128::from(BINARY_RESPONSE_INTERVAL.1) + 1;
    assert!(BinaryRootRelation::from_response(
        &input.descriptor,
        &input.matrix,
        response,
        &input.images,
        &input.challenges,
        FoldChallengeDistribution::BinaryRoot
    )
    .is_err());
    let mut input = fixture();
    assert!(build(&input).is_ok());
    let original = input.matrix[0][0][0];
    input.matrix[0][0][0] = i128::from(Q24_CENTERED_BOUND) + 1;
    assert!(build(&input).is_err());
    input.matrix[0][0][0] = original;
    assert!(build(&input).is_ok());
    let mut input = fixture();
    assert!(build(&input).is_ok());
    input.images.pop();
    assert!(build(&input).is_err());
    let mut input = fixture();
    assert!(build(&input).is_ok());
    input.sources.pop();
    assert!(build(&input).is_err());
}

/// One image coefficient lies below the centered q24 window, at the
/// non-canonical representative -7,126,705 of the enforced image window.
fn full_window_fixture() -> Fixture {
    let descriptor = BinarySourceDescriptor::new(
        BinaryIndexMap::new(128, 4, 4, 1).unwrap(),
        vec![[1; 32]],
        BinaryCommitmentProfiles {
            a: [3; 32],
            b: [4; 32],
        },
    )
    .unwrap();
    let matrix = vec![vec![[i128::from(Q24_CENTERED_BOUND); D]]];
    let mut source = [0; D];
    source[0] = 1;
    let mut image = [Q24Image::from_centered(Q24_CENTERED_BOUND).unwrap(); D];
    image[0] = Q24Image::from_centered(Q24_CENTERED_BOUND - Q24 as i32).unwrap();
    assert!(image[0].centered() < -Q24_CENTERED_BOUND);
    let challenges = SeedDraw
        .draw_fold_challenges(
            &FoldChallengeFamily::new(
                FoldChallengeDrawDomain::EvaluationTrace,
                162,
                FoldChallengeDistribution::BinaryRoot,
                None,
            )
            .unwrap(),
            1,
            1,
        )
        .unwrap()
        .unwrap();
    Fixture {
        descriptor,
        matrix,
        sources: vec![source],
        images: vec![image],
        challenges,
    }
}

#[test]
fn q24_root_accepts_the_full_enforced_image_window() {
    let relation = build(&full_window_fixture()).unwrap();
    assert!(relation.rows[0].remainder.iter().any(|&x| x != 0));
}

#[test]
fn q24_root_rows_authenticate_the_repacked_response_image_and_padding() {
    type E = Prime128Offset275;
    let input = fixture();
    let fixed = build(&input).unwrap();
    let (plan, digits) =
        BinaryRootRelation::successor_table(&fixed.response, &fixed.images, 256).unwrap();
    let table = digits
        .into_iter()
        .map(|x| E::from_i64(i64::from(x)))
        .collect::<Vec<_>>();
    let group = fixed
        .commitment_group::<E>(&input.descriptor, &input.matrix)
        .unwrap();
    let relation = NativeRootRelation::new(plan.clone(), vec![group]).unwrap();
    let padding = (0..plan.domain_len())
        .filter(|&index| plan.is_padding_address(index).unwrap())
        .collect::<Vec<_>>();
    let weights = relation
        .ring_switch(
            E::from_u64(17),
            &vec![E::one(); relation.row_count()],
            &vec![E::one(); padding.len()],
        )
        .unwrap();
    weights.check_table(&table).unwrap();
    for address in [
        plan.coefficient_address(0, 0, 0, 0).unwrap(),
        plan.coefficient_address(1, 0, 0, 0).unwrap(),
        padding[0],
    ] {
        let mut mutated = table.clone();
        mutated[address] += E::one();
        assert!(weights.check_table(&mutated).is_err());
    }
}

#[path = "binary_root_mixed_tests.rs"]
mod mixed;

#[path = "binary_root/projection_tests.rs"]
mod projection;

#[path = "binary_root/evaluation_tests.rs"]
mod evaluation_tests;

#[test]
fn binary_root_p64_quotient_digits_are_exact_bounded_and_complete() {
    use akita_types::binary::BinaryQuotientBounds;
    use jolt_field::{Ext2, Prime64Offset59};
    type E = Ext2<Prime64Offset59>;
    let bounds = BinaryQuotientBounds::new(1, 2).unwrap();
    let mut input = fixture();
    let relation = build(&input).unwrap();
    let residuals = BinaryRootRelation::remainders(
        &input.descriptor,
        &input.matrix,
        &relation.response,
        &input.images,
        &input.challenges,
        FoldChallengeDistribution::BinaryRoot,
    )
    .unwrap();
    assert_eq!(
        residuals,
        relation
            .rows
            .iter()
            .map(|row| row.remainder)
            .collect::<Vec<_>>()
    );
    let digits = BinaryRootRelation::quotient_digits(&residuals, &bounds).unwrap();
    for (row, (residual, digits)) in residuals
        .iter()
        .zip(digits.chunks_exact(bounds.digits * D))
        .enumerate()
    {
        assert_eq!(
            relation.residual_in_proof_field::<E>(row).unwrap(),
            residual.map(E::from_i128)
        );
        for (coefficient, &value) in residual.iter().enumerate() {
            let quotient: i128 = digits
                .chunks_exact(D)
                .rev()
                .fold(0, |value, lane| 8 * value + i128::from(lane[coefficient]));
            assert_eq!(quotient * i128::from(Q24), value);
        }
    }
    let mut residuals = [[0i128; D]; 3];
    for (slot, quotient) in residuals[0].iter_mut().zip([
        -4,
        3,
        -1,
        1,
        -(bounds.quotient_floor as i128),
        bounds.quotient_floor as i128,
    ]) {
        *slot = quotient * i128::from(Q24);
    }
    let digits = BinaryRootRelation::quotient_digits(&residuals, &bounds).unwrap();
    assert_eq!(&digits[..4], &[-4, 3, -1, 1]);
    assert!(digits.iter().all(|digit| (-4..=3).contains(digit)));
    let shallow = BinaryQuotientBounds {
        digits: bounds.digits - 1,
        ..bounds
    };
    assert!(BinaryRootRelation::quotient_digits(&residuals, &shallow).is_err());
    residuals[0][0] += 1;
    assert!(matches!(
        BinaryRootRelation::quotient_digits(&residuals, &bounds),
        Err(AkitaError::InvalidInput(_))
    ));
    residuals[0][0] = bounds.residual_bound as i128 + 1;
    assert!(BinaryRootRelation::quotient_digits(&residuals, &bounds).is_err());
    let value = input.images[0][0].centered();
    input.images[0][0] = Q24Image::from_centered(value + 1).unwrap();
    assert!(BinaryRootRelation::remainders(
        &input.descriptor,
        &input.matrix,
        &relation.response,
        &input.images,
        &input.challenges,
        FoldChallengeDistribution::BinaryRoot,
    )
    .is_err());
}

// Independent descending polynomial long division, rather than the
// production reducer's closed monomial folding.
fn schoolbook_remainder(mut coefficients: Vec<i128>, degree: usize) -> Vec<i128> {
    for exponent in (degree..coefficients.len()).rev() {
        let leading = coefficients[exponent];
        coefficients[exponent] = 0;
        coefficients[exponent - degree] -= leading;
        coefficients[exponent - degree / 2] -= leading;
    }
    coefficients.resize(degree, 0);
    coefficients
}

fn check_multiplication_bound<const D: usize>(ring: BinaryScalarRing) {
    use akita_algebra::ring::trinomial::{
        signed::{signed_trinomial_product, signed_trinomial_remainder},
        PlusTrinomial,
    };

    assert_eq!(ring.degree(), D);
    let kernels: Vec<_> = (0..2 * D - 1)
        .map(|exponent| {
            let mut monomial = vec![0; exponent + 1];
            monomial[exponent] = 1;
            let expected = schoolbook_remainder(monomial.clone(), D);
            assert_eq!(
                signed_trinomial_remainder::<D, PlusTrinomial>(&monomial)
                    .unwrap()
                    .as_slice(),
                expected
            );
            expected
        })
        .collect();
    // The infinity operator norm is the maximum absolute row sum.
    // Derive it for every monomial multiplier from the independent oracle.
    let factor = (0..D)
        .map(|shift| {
            let mut row_norms = [0u64; D];
            for kernel in &kernels[shift..shift + D] {
                for (norm, coefficient) in row_norms.iter_mut().zip(kernel) {
                    *norm += u64::try_from(coefficient.abs()).unwrap();
                }
            }
            row_norms.into_iter().max().unwrap()
        })
        .max()
        .unwrap();
    // Triangle inequality extends the monomial norm to any signed support.
    for cap in 0..=D {
        let profile = BinaryChallengeProfile::bounded_weight(ring, cap).unwrap();
        assert_eq!(
            profile.multiplication_linf_operator_bound(),
            factor * cap as u64
        );
    }

    // A real signed product witnesses that the derived factor is needed.
    let mut challenge = [0i8; D];
    challenge[D / 2] = 1;
    let mut source = [1i8; D];
    source[D / 2] = -1;
    let mut convolution = vec![0i128; 2 * D - 1];
    for (left, &a) in challenge.iter().enumerate() {
        for (right, &b) in source.iter().enumerate() {
            convolution[left + right] += i128::from(a) * i128::from(b);
        }
    }
    let expected = schoolbook_remainder(convolution, D);
    let actual = signed_trinomial_product::<D, PlusTrinomial>(&challenge, &source).unwrap();
    assert_eq!(actual.as_slice(), expected);
    assert_eq!(
        actual.into_iter().map(i128::unsigned_abs).max().unwrap(),
        u128::from(factor)
    );
}

#[test]
fn trinomial_multiplication_matches_schoolbook_and_certified_bound() {
    check_multiplication_bound::<162>(BinaryScalarRing::Cyclotomic243);
    check_multiplication_bound::<486>(BinaryScalarRing::Cyclotomic729);
}
