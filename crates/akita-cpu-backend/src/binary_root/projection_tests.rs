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
    // The full-window fixture carries a non-canonical q24 image coefficient.
    for input in [fixture(), full_window_fixture()] {
        let relation = build(&input).unwrap();
        let actual = BinaryRootRelation::remainders(
            &input.descriptor,
            &input.matrix,
            &relation.response,
            &input.images,
            &input.challenges,
            FoldChallengeDistribution::BinaryRoot,
        )
        .unwrap();
        assert!(actual
            .iter()
            .eq(relation.rows.iter().map(|row| &row.remainder)));
    }
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
