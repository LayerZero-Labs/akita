use super::*;

#[test]
fn projection_successor_contains_only_response_and_image_segments() {
    let input = fixture();
    let relation = build(&input).unwrap();
    let (plan, digits) =
        BinaryRootRelation::successor_table(&relation.response, &relation.images, 256).unwrap();
    assert_eq!(plan.segments().len(), 2);
    assert_eq!(plan.segments()[0].elements, relation.response.len());
    assert_eq!(plan.segments()[0].digits, 4);
    assert_eq!(plan.segments()[1].elements, relation.images.len());
    assert_eq!(plan.segments()[1].digits, 8);
    let table = digits
        .iter()
        .map(|&digit| Prime128Offset275::from_i64(i64::from(digit)))
        .collect::<Vec<_>>();
    plan.validate_digit_table(&table).unwrap();
    assert!(
        BinaryRootRelation::successor_table(&relation.response, &relation.images, 648).is_err()
    );
    // Every residual remains an integer construction witness. Its coefficients
    // are neither another successor segment nor an unauthenticated P-zero row.
    assert!(relation
        .rows
        .iter()
        .any(|row| row.remainder.iter().any(|&x| x != 0)));
}

#[test]
fn projection_residuals_match_independent_integer_dot_products() {
    let input = fixture();
    let relation = build(&input).unwrap();
    for projection in 0..7 {
        let mut expected = 0i128;
        let mut actual = 0i128;
        for polynomial in 0..input.descriptor.polynomials().len() {
            for (matrix_row, multipliers) in input.matrix.iter().enumerate() {
                let mut unreduced = vec![0i128; 2 * D - 1];
                for (response, multiplier) in relation.response[polynomial..=polynomial]
                    .iter()
                    .zip(multipliers)
                {
                    for (sum, value) in unreduced.iter_mut().zip(reference(multiplier, response)) {
                        *sum += value;
                    }
                }
                for column in 0..input.descriptor.index_map().columns() {
                    let image = input.images[(polynomial * 2 + column) * 2 + matrix_row]
                        .map(|coefficient| i128::from(coefficient.centered()));
                    let challenge = relation.embedded_challenges[polynomial * 2 + column];
                    for (sum, value) in unreduced.iter_mut().zip(reference(&challenge, &image)) {
                        *sum -= value;
                    }
                }
                let residual = divide(unreduced).0;
                let row_index = polynomial * input.matrix.len() + matrix_row;
                for (coefficient, (&reference, &computed)) in residual
                    .iter()
                    .zip(&relation.rows[row_index].remainder)
                    .enumerate()
                {
                    let weight = ((projection * 97 + row_index * 71 + coefficient * 13) as i128)
                        .rem_euclid(i128::from(Q24));
                    expected += weight * reference;
                    actual += weight * computed;
                }
            }
        }
        assert_eq!(actual, expected);
        assert_eq!(actual % i128::from(Q24), 0);
    }
}

#[test]
fn projection_replay_rejects_malformed_challenges_and_nondivisible_images() {
    let mut input = fixture();
    let original = build(&input).unwrap();
    let replay = |input: &Fixture, challenges: &Challenges| {
        BinaryRootRelation::from_response(
            &input.descriptor,
            &input.matrix,
            original.response.clone(),
            &input.images,
            challenges,
            FoldChallengeDistribution::BinaryRoot,
        )
    };
    for coefficient in [0, D / 2, D - 1] {
        let original = input.images[0][coefficient];
        input.images[0][coefficient] = Q24Image::from_centered(original.centered() + 1).unwrap();
        assert!(replay(&input, &input.challenges).is_err());
        input.images[0][coefficient] = original;
    }
    let mut sparse = input.challenges.as_slice().to_vec();
    sparse[0].coeffs[0] = 2;
    let malformed = Challenges::from_sparse(sparse, 2, 2).unwrap();
    assert!(replay(&input, &malformed).is_err());
    let mut sparse = input.challenges.as_slice().to_vec();
    sparse[0].positions[0] = 162;
    let malformed = Challenges::from_sparse(sparse, 2, 2).unwrap();
    assert!(replay(&input, &malformed).is_err());
    let wrong_geometry =
        Challenges::from_sparse(input.challenges.as_slice().to_vec(), 4, 1).unwrap();
    assert!(replay(&input, &wrong_geometry).is_err());
    let replayed = replay(&input, &input.challenges).unwrap();
    for (left, right) in original.rows.iter().zip(replayed.rows) {
        assert_eq!(left.remainder, right.remainder);
        assert_eq!(left.quotient, right.quotient);
    }
}

#[test]
fn limb_remainders_equal_transform_remainders_and_reject_like_them() {
    let mut input = fixture();
    let relation = build(&input).unwrap();
    let remainders = |input: &Fixture, challenges: &Challenges| {
        BinaryRootRelation::remainders(
            &input.descriptor,
            &input.matrix,
            &relation.response,
            &input.images,
            challenges,
            FoldChallengeDistribution::BinaryRoot,
        )
    };
    let actual = remainders(&input, &input.challenges).unwrap();
    assert!(actual
        .iter()
        .eq(relation.rows.iter().map(|row| &row.remainder)));
    for coefficient in [0, D / 2, D - 1] {
        let original = input.images[0][coefficient];
        input.images[0][coefficient] = Q24Image::from_centered(original.centered() + 1).unwrap();
        assert!(remainders(&input, &input.challenges).is_err());
        input.images[0][coefficient] = original;
    }
    let mut sparse = input.challenges.as_slice().to_vec();
    sparse[0].coeffs[0] = 2;
    let malformed = Challenges::from_sparse(sparse, 2, 2).unwrap();
    assert!(remainders(&input, &malformed).is_err());
    let mut response = relation.response.clone();
    response[0][0] = i128::from(BINARY_RESPONSE_INTERVAL.1) + 1;
    assert!(BinaryRootRelation::remainders(
        &input.descriptor,
        &input.matrix,
        &response,
        &input.images,
        &input.challenges,
        FoldChallengeDistribution::BinaryRoot,
    )
    .is_err());
}
