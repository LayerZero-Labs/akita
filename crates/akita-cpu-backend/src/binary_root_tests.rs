use super::*;
use akita_challenges::{FoldChallengeDrawDomain, FoldDraw};
use akita_types::binary_source::{BinaryCommitmentProfiles, BinaryIndexMap};
use akita_types::native_root_relation::NativeRootRelation;
use jolt_field::Ring;

struct SeedDraw;
impl FoldDraw for SeedDraw {
    fn absorb_and_squeeze(&mut self, _: &[u8]) -> Result<[u8; 32], AkitaError> {
        Ok([17; 32])
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
                if (i + element * 7) % 31 == 0 {
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
        .draw_folding_challenges_with_rejection(
            FoldChallengeDrawDomain::EvaluationTrace,
            162,
            0,
            2,
            2,
            &FoldChallengeFamily::BinaryRoot,
            None,
        )
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
        FoldChallengeFamily::BinaryRoot,
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
                    .residual_in_proof_field(polynomial * 2 + row)
                    .unwrap(),
                remainder.map(signed_field::<Prime128Offset275>)
            );
        }
    }
    let (plan, table) = relation.successor_table(256).unwrap();
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
                signed_field::<Prime128Offset275>(value)
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
        FoldChallengeFamily::BinaryRoot
    )
    .is_err());
}

#[test]
fn q24_root_rejects_wrong_family_counts_bounds_and_matrix_representatives() {
    let mut input = fixture();
    assert!(BinaryRootRelation::build(
        &input.descriptor,
        &input.matrix,
        &input.sources,
        &input.images,
        &input.challenges,
        FoldChallengeFamily::pm1_only(46)
    )
    .is_err());
    let mut response = build(&input).unwrap().response;
    response[0][0] = i128::from(BINARY_RESPONSE_INTERVAL.1) + 1;
    assert!(BinaryRootRelation::from_response(
        &input.descriptor,
        &input.matrix,
        response,
        &input.images,
        &input.challenges,
        FoldChallengeFamily::BinaryRoot
    )
    .is_err());
    input.matrix[0][0][0] = i128::from(Q24_CENTERED_BOUND) + 1;
    assert!(build(&input).is_err());
    input.matrix[0][0][0] -= 1;
    input.images.pop();
    assert!(build(&input).is_err());
    input.sources.pop();
    assert!(build(&input).is_err());
}

#[test]
fn q24_root_accepts_the_full_enforced_image_window() {
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
        .draw_folding_challenges_with_rejection(
            FoldChallengeDrawDomain::EvaluationTrace,
            162,
            0,
            1,
            1,
            &FoldChallengeFamily::BinaryRoot,
            None,
        )
        .unwrap();
    let relation = BinaryRootRelation::build(
        &descriptor,
        &matrix,
        &[source],
        &[image],
        &challenges,
        FoldChallengeFamily::BinaryRoot,
    )
    .unwrap();
    assert!(relation.rows[0].remainder.iter().any(|&x| x != 0));
}

#[test]
fn q24_root_rows_authenticate_the_repacked_response_image_and_padding() {
    type E = Prime128Offset275;
    let input = fixture();
    let fixed = build(&input).unwrap();
    let (plan, digits) = fixed.successor_table(256).unwrap();
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
