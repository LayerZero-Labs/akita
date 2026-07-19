use super::*;
use akita_field::Prime128Offset275;
use akita_sumcheck::{advance_eq_factored_claim, multilinear_eval};
use akita_types::DigitRangeEqualityPoint;

type F = Prime128Offset275;

fn ordered_equality_point(
    challenges: &[F],
    column_variables: usize,
    ring_variables: usize,
) -> Vec<F> {
    DigitRangeEqualityPoint::from_column_then_ring_challenges(
        challenges,
        column_variables,
        ring_variables,
    )
    .expect("valid test point")
    .into_coordinates()
}

#[test]
fn stage1_new_rejects_malformed_shapes_without_panicking() {
    let tau = vec![F::zero(); usize::BITS as usize];
    assert!(LowBasisRangeCheckProver::<F>::new(
        std::sync::Arc::from([]),
        &tau,
        DigitRangePlan::new(4).unwrap(),
        1,
        0,
        usize::BITS as usize
    )
    .is_err());

    let tau = vec![F::zero(); usize::BITS as usize + 1];
    assert!(LowBasisRangeCheckProver::<F>::new(
        std::sync::Arc::from([]),
        &tau,
        DigitRangePlan::new(4).unwrap(),
        3,
        2,
        usize::BITS as usize - 1
    )
    .is_err());

    assert!(LowBasisRangeCheckProver::<F>::new(
        std::sync::Arc::from([]),
        &[],
        DigitRangePlan::new(16).unwrap(),
        1,
        0,
        0
    )
    .is_err());
}

fn fold_compact_range_image_prefix_x_reference(
    compact_range_image: &[i16],
    live_x_cols: usize,
    y_len: usize,
    r: F,
) -> Vec<F> {
    let next_live_x_cols = live_x_cols.div_ceil(2);
    let mut out = vec![F::zero(); y_len * next_live_x_cols];
    for (y, row_out) in out.chunks_mut(next_live_x_cols).enumerate() {
        let row_start = y * live_x_cols;
        let row = &compact_range_image[row_start..row_start + live_x_cols];
        for (pair_x, dst) in row_out.iter_mut().enumerate() {
            let left = 2 * pair_x;
            let s_0 = F::from_i64(i64::from(row[left]));
            let s_1 = if left + 1 < live_x_cols {
                F::from_i64(i64::from(row[left + 1]))
            } else {
                F::zero()
            };
            *dst = s_0 + r * (s_1 - s_0);
        }
    }
    out
}

fn fold_compact_range_image_to_full_reference(compact_range_image: &[i16], r: F) -> Vec<F> {
    (0..compact_range_image.len() / 2)
        .map(|j| {
            let s_0 = F::from_i64(i64::from(compact_range_image[2 * j]));
            let s_1 = F::from_i64(i64::from(compact_range_image[2 * j + 1]));
            s_0 + r * (s_1 - s_0)
        })
        .collect()
}

#[test]
fn stage1_compact_fold_lookup_matches_direct_formula() {
    let basis = 8usize;
    let r = F::from_u64(41);

    let range_image_prefix = vec![2, 6, 12, 2, 6, 12, 2, 6, 12, 2];
    let fold_lut = LowBasisRangeCheckProver::<F>::build_range_image_fold_lut(basis, r);
    assert_eq!(
        LowBasisRangeCheckProver::<F>::fold_compact_range_image_prefix_x(
            &range_image_prefix,
            5,
            2,
            &fold_lut
        ),
        fold_compact_range_image_prefix_x_reference(&range_image_prefix, 5, 2, r)
    );

    let dense_range_image = vec![2, 6, 12, 2, 6, 12];
    let dense_lut = LowBasisRangeCheckProver::<F>::build_range_image_fold_lut(basis, r);
    assert_eq!(
        LowBasisRangeCheckProver::<F>::fold_compact_range_image_to_full(
            &dense_range_image,
            &dense_lut
        ),
        fold_compact_range_image_to_full_reference(&dense_range_image, r)
    );
}

#[test]
fn stage1_round0_matches_dense_reference() {
    let col_bits = 3usize;
    let ring_bits = 2usize;
    let n = 1usize << (col_bits + ring_bits);
    let tau0: Vec<F> = (0..(col_bits + ring_bits))
        .map(|i| F::from_u64((i as u64) + 2))
        .collect();
    let tau0 = ordered_equality_point(&tau0, col_bits, ring_bits);

    for basis in [4usize, 8] {
        let half = (basis / 2) as i8;
        let compact_digit_witness: Vec<i8> =
            (0..n).map(|i| ((i * 5 + 3) % basis) as i8 - half).collect();

        let mut prover = LowBasisRangeCheckProver::new(
            std::sync::Arc::from(compact_digit_witness.as_slice()),
            &tau0,
            DigitRangePlan::new(basis).unwrap(),
            1usize << col_bits,
            col_bits,
            ring_bits,
        )
        .unwrap();
        let stage1_poly = prover.compute_round_eq_factored(0);
        let compact_range_image = build_compact_range_image(&compact_digit_witness);
        let reference = compute_range_round_polynomial_from_compact_image(
            &prover.split_eq,
            &compact_range_image,
            &prover.polynomial_precomputation,
        );

        assert_eq!(
            stage1_poly, reference,
            "stage1 round0 mismatch for basis={basis}"
        );
    }
}

#[test]
fn stage1_prefix_aware_rounds_match_explicit_zero_padding() {
    let ring_bits = 2usize;
    for basis in [4usize, 8] {
        let half = (basis / 2) as i8;
        for live_x_cols in [5usize, 6usize] {
            let col_bits = live_x_cols.next_power_of_two().trailing_zeros() as usize;
            let y_len = 1usize << ring_bits;
            let digit_witness_prefix: Vec<i8> = (0..(live_x_cols * y_len))
                .map(|i| ((i * 7 + 5) % basis) as i8 - half)
                .collect();
            let padded_digit_witness =
                pad_compact_witness(&digit_witness_prefix, live_x_cols, col_bits, ring_bits);
            let tau0: Vec<F> = (0..(col_bits + ring_bits))
                .map(|i| F::from_u64((i as u64) + 19))
                .collect();
            let tau0 = ordered_equality_point(&tau0, col_bits, ring_bits);
            let mut prefix_prover = LowBasisRangeCheckProver::new(
                std::sync::Arc::from(digit_witness_prefix.as_slice()),
                &tau0,
                DigitRangePlan::new(basis).unwrap(),
                live_x_cols,
                col_bits,
                ring_bits,
            )
            .unwrap();
            let mut padded_prover = LowBasisRangeCheckProver::new(
                std::sync::Arc::from(padded_digit_witness.as_slice()),
                &tau0,
                DigitRangePlan::new(basis).unwrap(),
                1usize << col_bits,
                col_bits,
                ring_bits,
            )
            .unwrap();
            let mut challenges = Vec::new();
            let mut prefix_claim = F::zero();
            let mut prefix_scale = F::one();
            let mut padded_claim = F::zero();
            let mut padded_scale = F::one();

            for round in 0..(col_bits + ring_bits) {
                let prefix_poly = prefix_prover.compute_round_eq_factored(round);
                let padded_poly = padded_prover.compute_round_eq_factored(round);
                assert_eq!(
                    prefix_poly, padded_poly,
                    "round {round} polynomial mismatch live_x_cols={live_x_cols} basis={basis}"
                );

                let challenge = F::from_u64((round as u64) + 29);
                challenges.push(challenge);
                let (prefix_linear_at_zero, prefix_linear_at_one) =
                    prefix_prover.current_linear_factor_evals();
                (prefix_claim, prefix_scale) = advance_eq_factored_claim(
                    prefix_claim,
                    prefix_scale,
                    prefix_linear_at_zero,
                    prefix_linear_at_one,
                    &prefix_poly,
                    challenge,
                );
                let (padded_linear_at_zero, padded_linear_at_one) =
                    padded_prover.current_linear_factor_evals();
                (padded_claim, padded_scale) = advance_eq_factored_claim(
                    padded_claim,
                    padded_scale,
                    padded_linear_at_zero,
                    padded_linear_at_one,
                    &padded_poly,
                    challenge,
                );
                prefix_prover.ingest_challenge(round, challenge);
                padded_prover.ingest_challenge(round, challenge);
            }

            assert_eq!(
                prefix_prover.final_range_image_eval(),
                padded_prover.final_range_image_eval()
            );
            assert_eq!(prefix_claim, padded_claim);
            assert_eq!(prefix_scale, padded_scale);
            let padded_range_image: Vec<F> = build_compact_range_image(&padded_digit_witness)
                .into_iter()
                .map(|s| F::from_i64(i64::from(s)))
                .collect();
            assert_eq!(
                prefix_prover.final_range_image_eval(),
                multilinear_eval(&padded_range_image, &challenges).unwrap(),
                "final s-claim mismatch live_x_cols={live_x_cols} basis={basis}"
            );
        }
    }
}

#[test]
fn stage1_fused_round2_transition_matches_two_pass_reference() {
    let col_bits = 3usize;
    let ring_bits = 2usize;
    let live_x_cols = 6usize;
    let y_len = 1usize << ring_bits;
    for basis in [4usize, 8] {
        let half = (basis / 2) as i8;
        let digit_witness_prefix: Vec<i8> = (0..(live_x_cols * y_len))
            .map(|i| ((i * 9 + 5) % basis) as i8 - half)
            .collect();
        let compact_range_image = build_compact_range_image(&digit_witness_prefix);
        let tau0: Vec<F> = (0..(col_bits + ring_bits))
            .map(|i| F::from_u64((i as u64) + 53))
            .collect();
        let tau0 = ordered_equality_point(&tau0, col_bits, ring_bits);

        let mut prover = LowBasisRangeCheckProver::new(
            std::sync::Arc::from(digit_witness_prefix.as_slice()),
            &tau0,
            DigitRangePlan::new(basis).unwrap(),
            live_x_cols,
            col_bits,
            ring_bits,
        )
        .unwrap();
        let round0 = prover.compute_round_eq_factored(0);
        let r0 = F::from_u64(61);
        let (linear_at_zero, linear_at_one) = prover.current_linear_factor_evals();
        let (claim1, scale1) = advance_eq_factored_claim(
            F::zero(),
            F::one(),
            linear_at_zero,
            linear_at_one,
            &round0,
            r0,
        );
        prover.ingest_challenge(0, r0);
        let round1 = prover.compute_round_eq_factored(1);
        let r1 = F::from_u64(67);
        let (linear_at_zero, linear_at_one) = prover.current_linear_factor_evals();
        let (_claim2, _scale2) =
            advance_eq_factored_claim(claim1, scale1, linear_at_zero, linear_at_one, &round1, r1);

        let expected_range_image =
            LowBasisRangeCheckProver::<F>::fold_compact_range_image_to_round2(
                &compact_range_image,
                live_x_cols,
                y_len,
                r0,
                r1,
            );
        let mut expected = LowBasisRangeCheckProver::new(
            std::sync::Arc::from(digit_witness_prefix.as_slice()),
            &tau0,
            DigitRangePlan::new(basis).unwrap(),
            live_x_cols,
            col_bits,
            ring_bits,
        )
        .unwrap();
        expected.split_eq.bind(r0);
        expected.split_eq.bind(r1);
        expected.rounds_completed = 2;
        let expected_round2 = expected.compute_round_full_prefix_x(&expected_range_image);

        prover.ingest_challenge(1, r1);

        match &prover.range_image {
            LowBasisRangeImageStorage::Materialized(range_image) => {
                assert_eq!(range_image, &expected_range_image)
            }
            LowBasisRangeImageStorage::Compact(_) => {
                panic!("expected fused stage1 transition to materialize full table")
            }
        }
        assert_eq!(prover.cached_round_poly.as_ref(), Some(&expected_round2));
    }
}

#[test]
fn stage1_later_full_prefix_fusion_matches_two_pass_reference() {
    let col_bits = 5usize;
    let ring_bits = 2usize;
    let live_x_cols = 12usize;
    let y_len = 1usize << ring_bits;
    for basis in [4usize, 8] {
        let half = (basis / 2) as i8;
        let digit_witness_prefix: Vec<i8> = (0..(live_x_cols * y_len))
            .map(|i| ((i * 5 + 11) % basis) as i8 - half)
            .collect();
        let tau0: Vec<F> = (0..(col_bits + ring_bits))
            .map(|i| F::from_u64((i as u64) + 101))
            .collect();
        let tau0 = ordered_equality_point(&tau0, col_bits, ring_bits);

        let mut prover = LowBasisRangeCheckProver::new(
            std::sync::Arc::from(digit_witness_prefix.as_slice()),
            &tau0,
            DigitRangePlan::new(basis).unwrap(),
            live_x_cols,
            col_bits,
            ring_bits,
        )
        .unwrap();
        let round0 = prover.compute_round_eq_factored(0);
        let r0 = F::from_u64(107);
        let (linear_at_zero, linear_at_one) = prover.current_linear_factor_evals();
        let (claim1, scale1) = advance_eq_factored_claim(
            F::zero(),
            F::one(),
            linear_at_zero,
            linear_at_one,
            &round0,
            r0,
        );
        prover.ingest_challenge(0, r0);

        let round1 = prover.compute_round_eq_factored(1);
        let r1 = F::from_u64(109);
        let (linear_at_zero, linear_at_one) = prover.current_linear_factor_evals();
        let (claim2, scale2) =
            advance_eq_factored_claim(claim1, scale1, linear_at_zero, linear_at_one, &round1, r1);
        prover.ingest_challenge(1, r1);

        let round2 = prover.compute_round_eq_factored(2);
        let r2 = F::from_u64(113);
        let (linear_at_zero, linear_at_one) = prover.current_linear_factor_evals();
        let (claim3, _scale3) =
            advance_eq_factored_claim(claim2, scale2, linear_at_zero, linear_at_one, &round2, r2);

        let mut expected = LowBasisRangeCheckProver::new(
            std::sync::Arc::from(digit_witness_prefix.as_slice()),
            &tau0,
            DigitRangePlan::new(basis).unwrap(),
            live_x_cols,
            col_bits,
            ring_bits,
        )
        .unwrap();
        let expected_round0 = expected.compute_round_eq_factored(0);
        assert_eq!(expected_round0, round0);
        expected.ingest_challenge(0, r0);
        let expected_round1 = expected.compute_round_eq_factored(1);
        assert_eq!(expected_round1, round1);
        expected.ingest_challenge(1, r1);
        let expected_round2 = expected.compute_round_eq_factored(2);
        assert_eq!(expected_round2, round2);

        let current_range_image = match &expected.range_image {
            LowBasisRangeImageStorage::Materialized(range_image) => range_image.clone(),
            LowBasisRangeImageStorage::Compact(_) => {
                panic!("expected later prefix state to be full")
            }
        };
        let current_y_len = current_range_image.len() / expected.live_x_cols;
        let expected_next_range_image = LowBasisRangeCheckProver::<F>::fold_range_image_prefix_x(
            &current_range_image,
            expected.live_x_cols,
            current_y_len,
            r2,
        );
        expected.split_eq.bind(r2);
        expected.live_x_cols = expected.live_x_cols.div_ceil(2);
        expected.rounds_completed += 1;
        let _ = claim3;
        let expected_round3 = expected.compute_round_full_prefix_x(&expected_next_range_image);

        prover.ingest_challenge(2, r2);

        match &prover.range_image {
            LowBasisRangeImageStorage::Materialized(range_image) => {
                assert_eq!(range_image, &expected_next_range_image)
            }
            LowBasisRangeImageStorage::Compact(_) => {
                panic!("expected fused later prefix stage to stay full")
            }
        }
        assert_eq!(prover.cached_round_poly.as_ref(), Some(&expected_round3));
    }
}

#[test]
fn stage1_sparse_x_y_fusion_matches_two_pass_reference() {
    let col_bits = 3usize;
    let ring_bits = 4usize;
    let live_x_cols = 6usize;
    let y_len = 1usize << ring_bits;
    for basis in [4usize, 8] {
        let half = (basis / 2) as i8;
        let digit_witness_prefix: Vec<i8> = (0..(live_x_cols * y_len))
            .map(|i| ((i * 7 + 9) % basis) as i8 - half)
            .collect();
        let tau0: Vec<F> = (0..(col_bits + ring_bits))
            .map(|i| F::from_u64((i as u64) + 131))
            .collect();
        let tau0 = ordered_equality_point(&tau0, col_bits, ring_bits);

        let mut prover = LowBasisRangeCheckProver::new(
            std::sync::Arc::from(digit_witness_prefix.as_slice()),
            &tau0,
            DigitRangePlan::new(basis).unwrap(),
            live_x_cols,
            col_bits,
            ring_bits,
        )
        .unwrap();
        let round0 = prover.compute_round_eq_factored(0);
        let r0 = F::from_u64(137);
        let (linear_at_zero, linear_at_one) = prover.current_linear_factor_evals();
        let (claim1, scale1) = advance_eq_factored_claim(
            F::zero(),
            F::one(),
            linear_at_zero,
            linear_at_one,
            &round0,
            r0,
        );
        prover.ingest_challenge(0, r0);

        let round1 = prover.compute_round_eq_factored(1);
        let r1 = F::from_u64(139);
        let (linear_at_zero, linear_at_one) = prover.current_linear_factor_evals();
        let (claim2, scale2) =
            advance_eq_factored_claim(claim1, scale1, linear_at_zero, linear_at_one, &round1, r1);
        prover.ingest_challenge(1, r1);

        let round2 = prover.compute_round_eq_factored(2);
        let r2 = F::from_u64(149);
        let (linear_at_zero, linear_at_one) = prover.current_linear_factor_evals();
        let (_claim3, _scale3) =
            advance_eq_factored_claim(claim2, scale2, linear_at_zero, linear_at_one, &round2, r2);

        let mut expected = LowBasisRangeCheckProver::new(
            std::sync::Arc::from(digit_witness_prefix.as_slice()),
            &tau0,
            DigitRangePlan::new(basis).unwrap(),
            live_x_cols,
            col_bits,
            ring_bits,
        )
        .unwrap();
        let expected_round0 = expected.compute_round_eq_factored(0);
        assert_eq!(expected_round0, round0);
        expected.ingest_challenge(0, r0);
        let expected_round1 = expected.compute_round_eq_factored(1);
        assert_eq!(expected_round1, round1);
        expected.ingest_challenge(1, r1);
        let expected_round2 = expected.compute_round_eq_factored(2);
        assert_eq!(expected_round2, round2);

        let current_range_image = match &expected.range_image {
            LowBasisRangeImageStorage::Materialized(range_image) => range_image.clone(),
            LowBasisRangeImageStorage::Compact(_) => panic!("expected sparse-x/y state to be full"),
        };
        let current_y_len = current_range_image.len() / expected.live_x_cols;
        let expected_next_range_image = LowBasisRangeCheckProver::<F>::fold_range_image_sparse_x_y(
            &current_range_image,
            expected.live_x_cols,
            current_y_len,
            r2,
        );
        expected.split_eq.bind(r2);
        expected.rounds_completed += 1;
        let expected_round3 = expected.compute_round_full_sparse_x_y(&expected_next_range_image);

        prover.ingest_challenge(2, r2);

        match &prover.range_image {
            LowBasisRangeImageStorage::Materialized(range_image) => {
                assert_eq!(range_image, &expected_next_range_image)
            }
            LowBasisRangeImageStorage::Compact(_) => {
                panic!("expected sparse-x/y fusion to stay full")
            }
        }
        assert_eq!(prover.cached_round_poly.as_ref(), Some(&expected_round3));
    }
}
