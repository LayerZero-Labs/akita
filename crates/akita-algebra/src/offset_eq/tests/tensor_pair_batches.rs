use super::*;
use crate::offset_eq::tensor_pair::{DenseLeftOverlap, ResidualTerms, MAX_RESIDUAL_TABLE_BYTES};

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

#[test]
fn bit_product_axes_match_dense_coordinate_oracle() {
    let mut rng = StdRng::seed_from_u64(0xB17F_AC70);
    let right = random_vec(&mut rng, 13);
    let dense_weights = random_vec(&mut rng, 3);
    let first_bits = (0..3)
        .map(|_| [F::random(&mut rng), F::random(&mut rng)])
        .collect::<Vec<_>>();
    let second_bits = (0..2)
        .map(|_| [F::random(&mut rng), F::random(&mut rng)])
        .collect::<Vec<_>>();
    let scalar = F::from_u64(23);
    let family = EqPairTensorFamily::new(
        0,
        7,
        scalar,
        vec![
            EqPairTensorAxis::bit_product(0, 5, first_bits.clone()).unwrap(),
            EqPairTensorAxis::bit_product(0, 131, second_bits.clone()).unwrap(),
            EqPairTensorAxis::dense(0, 1000, dense_weights.clone()),
        ],
    )
    .unwrap();

    let bit_weight = |factors: &[[F; 2]], coordinate: usize| {
        factors
            .iter()
            .enumerate()
            .fold(F::one(), |weight, (bit, factor)| {
                weight * factor[(coordinate >> bit) & 1]
            })
    };
    let mut expected = F::zero();
    for (dense, &dense_weight) in dense_weights.iter().enumerate() {
        for second in 0..4 {
            for first in 0..8 {
                expected += scalar
                    * dense_weight
                    * bit_weight(&first_bits, first)
                    * bit_weight(&second_bits, second)
                    * eq_eval_at_index(&right, 7 + 1000 * dense + 131 * second + 5 * first);
            }
        }
    }
    assert_eq!(
        eval_boolean_pair_tensor_families::<_, false, false>(&[], &right, &[family]).unwrap(),
        expected
    );
}

#[test]
fn out_of_domain_bit_product_axis_keeps_zero_branch_factor() {
    let right = vec![F::from_u64(7)];
    let scalar = F::from_u64(11);
    let zero_factor = F::from_u64(13);
    let family = EqPairTensorFamily::new(
        0,
        0,
        scalar,
        vec![EqPairTensorAxis::bit_product(0, 2, vec![[zero_factor, F::from_u64(17)]]).unwrap()],
    )
    .unwrap();

    assert_eq!(
        eval_boolean_pair_tensor_families::<_, false, false>(&[], &right, &[family]).unwrap(),
        scalar * zero_factor * eq_eval_at_index(&right, 0)
    );
}

#[test]
fn mutable_dense_axis_rejects_missing_coordinate_weights() {
    let mut malformed = EqPairTensorAxis::dense(0, 1, vec![F::one()]);
    malformed.len = 3;
    let equality = OffsetEqWindow::new(&[F::one(); 4]).unwrap();
    for axes in [
        vec![malformed.clone()],
        vec![EqPairTensorAxis::unit(2, 1, 3), malformed.clone()],
        vec![
            EqPairTensorAxis::unit(2, 1, 3),
            EqPairTensorAxis::unit(2, 4, 5),
            malformed,
        ],
    ] {
        let family = EqPairTensorFamily {
            left_offset: 0,
            right_offset: 0,
            scalar: F::one(),
            axes,
        };
        assert!(matches!(
            eval_boolean_pair_tensor_families::<_, false, false>(
                &[F::one(); 4],
                &[F::one(); 4],
                std::slice::from_ref(&family),
            ),
            Err(AkitaError::InvalidInput(_))
        ));
        assert!(matches!(
            materialize_eq_tensor_left(&equality, &[family], 16),
            Err(AkitaError::InvalidInput(_))
        ));
    }
}

#[test]
fn mutable_bit_product_axis_rejects_missing_factors() {
    for stride in [1, 3] {
        let mut axis = EqPairTensorAxis::bit_product(stride, stride, vec![[F::one(); 2]]).unwrap();
        axis.len = 4;
        let family = EqPairTensorFamily {
            left_offset: 0,
            right_offset: 0,
            scalar: F::one(),
            axes: vec![axis],
        };
        assert!(matches!(
            eval_boolean_pair_tensor_families::<_, false, false>(
                &[F::one(); 4],
                &[F::one(); 4],
                &[family],
            ),
            Err(AkitaError::InvalidInput(_))
        ));
    }
}

#[test]
fn materialization_of_zero_weight_residuals_is_zero() {
    // The residual product is 2^27 coordinates, but every one carries a zero
    // weight, so the residual table must stay empty.
    let challenges = [F::from_u64(3); 8];
    let equality = OffsetEqWindow::new(&challenges).unwrap();
    let families = [EqPairTensorFamily::new(
        0,
        0,
        F::one(),
        vec![
            EqPairTensorAxis::unit(2, 1, 1),
            EqPairTensorAxis::dense(0, 0, vec![F::zero(); 2]),
            EqPairTensorAxis::unit(1 << 26, 0, 1),
        ],
    )
    .unwrap()];
    let got = materialize_eq_tensor_left(&equality, &families, 2).unwrap();
    assert_eq!(got, vec![F::zero(); 2]);
}

#[test]
fn materialization_rejects_weights_beyond_axis_len() {
    // Sixty-four unit factor pairs describe 2^64 coordinates on a two-coordinate
    // axis; counting live residuals from the factors would overflow.
    let oversized_bit_product = EqPairTensorAxis {
        len: 2,
        left_stride: 0,
        right_stride: 1,
        weights: EqPairTensorWeights::BitProduct(vec![[F::one(); 2]; 64].into()),
    };
    let oversized_dense = EqPairTensorAxis {
        len: 2,
        left_stride: 0,
        right_stride: 1,
        weights: EqPairTensorWeights::Dense(vec![F::one(); 4].into()),
    };
    let equality = OffsetEqWindow::new(&[F::from_u64(3); 4]).unwrap();
    for residual in [oversized_bit_product, oversized_dense] {
        let family = EqPairTensorFamily {
            left_offset: 0,
            right_offset: 0,
            scalar: F::one(),
            axes: vec![EqPairTensorAxis::unit(2, 1, 3), residual],
        };
        assert!(matches!(
            materialize_eq_tensor_left(&equality, std::slice::from_ref(&family), 2),
            Err(AkitaError::InvalidInput(_))
        ));
        assert!(matches!(
            eval_boolean_pair_tensor_families::<_, false, false>(
                &[F::one(); 4],
                &[F::one(); 4],
                &[family],
            ),
            Err(AkitaError::InvalidInput(_))
        ));
    }
}

#[test]
fn dense_left_residuals_above_table_budget_are_streamed() {
    // Two outputs against a 2^27-term residual: within the work bound, but a
    // residual table would take gigabytes. Planning must not build it.
    let large = [EqPairTensorFamily::new(
        0,
        0,
        F::one(),
        vec![
            EqPairTensorAxis::unit(2, 1, 3),
            EqPairTensorAxis::unit(1 << 27, 0, 1),
        ],
    )
    .unwrap()];
    let plan = DenseLeftOverlap::plan(&large, 2, MAX_RESIDUAL_TABLE_BYTES)
        .unwrap()
        .unwrap();
    assert!(plan
        .views
        .iter()
        .all(|view| matches!(view.residual, ResidualTerms::Streamed)));

    let mut rng = StdRng::seed_from_u64(0x5717_EA3D);
    let challenges = random_vec(&mut rng, 10);
    let equality = OffsetEqWindow::new(&challenges).unwrap();
    let folds = random_vec(&mut rng, 3);
    let bits = [
        [F::from_u64(2), F::from_u64(3)],
        [F::from_u64(5), F::zero()],
    ];
    let families = [
        EqPairTensorFamily::new(
            1,
            9,
            F::from_u64(7),
            vec![
                EqPairTensorAxis::unit(6, 1, 5),
                EqPairTensorAxis::dense(0, 2, folds.clone()),
                EqPairTensorAxis::bit_product(0, 37, bits.to_vec()).unwrap(),
            ],
        )
        .unwrap(),
        EqPairTensorFamily::new(
            1,
            200,
            F::from_u64(11),
            vec![
                EqPairTensorAxis::unit(6, 1, 1),
                EqPairTensorAxis::unit(3, 0, 64),
            ],
        )
        .unwrap(),
    ];
    let expected = (0..8)
        .map(|left| {
            if !(1..7).contains(&left) {
                return F::zero();
            }
            let coordinate = left - 1;
            let mut value = F::zero();
            for (fold, &fold_weight) in folds.iter().enumerate() {
                for bit_coordinate in 0..4 {
                    let bit_weight =
                        bits[0][bit_coordinate & 1] * bits[1][(bit_coordinate >> 1) & 1];
                    value += F::from_u64(7)
                        * fold_weight
                        * bit_weight
                        * eq_eval_at_index(
                            &challenges,
                            9 + 5 * coordinate + 2 * fold + 37 * bit_coordinate,
                        );
                }
            }
            for row in 0..3 {
                value +=
                    F::from_u64(11) * eq_eval_at_index(&challenges, 200 + coordinate + 64 * row);
            }
            value
        })
        .collect::<Vec<_>>();
    for (budget, tabulated) in [(0, false), (MAX_RESIDUAL_TABLE_BYTES, true)] {
        let plan = DenseLeftOverlap::plan(&families, 8, budget)
            .unwrap()
            .unwrap();
        assert!(plan
            .views
            .iter()
            .all(|view| matches!(view.residual, ResidualTerms::Table(_)) == tabulated));
        assert_eq!(plan.materialize(&equality, 8).unwrap(), expected);
    }
    assert_eq!(
        materialize_eq_tensor_left(&equality, &families, 8).unwrap(),
        expected
    );
}
