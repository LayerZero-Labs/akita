use super::*;

#[test]
fn matches_dense_affine_chunk_axis() {
    let mut rng = StdRng::seed_from_u64(0xC4A1_5EED);
    let left = random_vec(&mut rng, 10);
    let right = random_vec(&mut rng, 12);
    let row_weights = random_vec(&mut rng, 2);
    let family = EqPairTensorFamily::new(
        3,
        11,
        F::from_u64(13),
        vec![
            EqPairTensorAxis::unit(16, 1, 1),
            EqPairTensorAxis::dense(64, 0, row_weights.clone()),
            EqPairTensorAxis::unit(8, 16, 53),
        ],
    )
    .unwrap();

    let expected = row_weights
        .iter()
        .enumerate()
        .flat_map(|(row, &row_weight)| {
            let left = &left;
            let right = &right;
            (0..8).flat_map(move |chunk| {
                (0..16).map(move |inner| {
                    F::from_u64(13)
                        * row_weight
                        * eq_eval_at_index(left, 3 + 64 * row + 16 * chunk + inner)
                        * eq_eval_at_index(right, 11 + 53 * chunk + inner)
                })
            })
        })
        .sum::<F>();
    assert_eq!(
        eval_boolean_pair_tensor_families::<_, false, false>(&left, &right, &[family]).unwrap(),
        expected
    );
}

#[test]
fn single_stream_batch_matches_truncated_dense_oracle() {
    let mut rng = StdRng::seed_from_u64(0x51A6_1E57);
    let left = random_vec(&mut rng, 5);
    let right = random_vec(&mut rng, 6);
    let dense_weights = random_vec(&mut rng, 3);
    let scalar = F::from_u64(19);
    let family = EqPairTensorFamily::new(
        20,
        7,
        scalar,
        vec![
            EqPairTensorAxis::dense(3, 1, dense_weights.clone()),
            EqPairTensorAxis::unit(16, 1, 2),
        ],
    )
    .unwrap();
    assert!(20 + 15 >= 1usize << left.len());

    let monomial_at = |challenges: &[F], index: usize| {
        if index >= 1usize << challenges.len() {
            return F::zero();
        }
        challenges
            .iter()
            .enumerate()
            .filter(|(bit, _)| index & (1usize << bit) != 0)
            .fold(F::one(), |weight, (_, &challenge)| weight * challenge)
    };
    let mut lagrange_expected = F::zero();
    let mut monomial_expected = F::zero();
    for (dense, &dense_weight) in dense_weights.iter().enumerate() {
        for stream in 0..16 {
            let left_index = 20 + 3 * dense + stream;
            let right_index = 7 + dense + 2 * stream;
            let common = scalar * dense_weight * eq_eval_at_index(&right, right_index);
            lagrange_expected += common * eq_eval_at_index(&left, left_index);
            monomial_expected += common * monomial_at(&left, left_index);
        }
    }

    assert_eq!(
        eval_boolean_pair_tensor_families::<_, false, false>(
            &left,
            &right,
            std::slice::from_ref(&family),
        )
        .unwrap(),
        lagrange_expected
    );
    assert_eq!(
        eval_boolean_pair_tensor_families::<_, true, false>(
            &left,
            &right,
            std::slice::from_ref(&family),
        )
        .unwrap(),
        monomial_expected
    );
}

#[test]
fn batches_matching_multi_axis_recurrences() {
    let mut rng = StdRng::seed_from_u64(0x0BA7_C4E5);
    let left = random_vec(&mut rng, 10);
    let right = random_vec(&mut rng, 12);
    let families = [(3, 11, 13), (101, 211, 17)]
        .into_iter()
        .map(|(left_offset, right_offset, scalar)| {
            EqPairTensorFamily::new(
                left_offset,
                right_offset,
                F::from_u64(scalar),
                vec![
                    EqPairTensorAxis::unit(16, 1, 1),
                    EqPairTensorAxis::dense(64, 0, random_vec(&mut rng, 2)),
                    EqPairTensorAxis::unit(8, 16, 53),
                ],
            )
            .unwrap()
        })
        .collect::<Vec<_>>();

    let lagrange_batch =
        eval_boolean_pair_tensor_families::<_, false, false>(&left, &right, &families).unwrap();
    let lagrange_separate = families
        .iter()
        .map(|family| {
            eval_boolean_pair_tensor_families::<_, false, false>(
                &left,
                &right,
                std::slice::from_ref(family),
            )
            .unwrap()
        })
        .sum();
    assert_eq!(lagrange_batch, lagrange_separate);

    let monomial_batch =
        eval_boolean_pair_tensor_families::<_, true, false>(&left, &right, &families).unwrap();
    let monomial_separate = families
        .iter()
        .map(|family| {
            eval_boolean_pair_tensor_families::<_, true, false>(
                &left,
                &right,
                std::slice::from_ref(family),
            )
            .unwrap()
        })
        .sum();
    assert_eq!(monomial_batch, monomial_separate);
}
