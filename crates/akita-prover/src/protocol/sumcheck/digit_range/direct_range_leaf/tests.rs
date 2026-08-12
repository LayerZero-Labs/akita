use super::*;
use akita_field::Prime128Offset275;
use akita_serialization::AkitaSerialize;
use akita_sumcheck::{
    advance_eq_factored_claim, multilinear_eval, EqFactoredSumcheckInstanceProverExt,
};
use akita_transcript::{labels as transcript_labels, AkitaTranscript, Transcript};
use akita_types::DigitRangeEqualityPoint;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

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

fn fold_compact_range_image_to_materialized_reference(compact_range_image: &[i16], r: F) -> Vec<F> {
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
        LowBasisRangeCheckProver::<F>::fold_compact_range_image_to_materialized(
            &dense_range_image,
            &dense_lut
        ),
        fold_compact_range_image_to_materialized_reference(&dense_range_image, r)
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
        let expected_round2 = expected.compute_round_materialized_prefix_x(&expected_range_image);

        prover.ingest_challenge(1, r1);

        match &prover.range_image {
            LowBasisRangeImageStorage::Materialized(range_image) => {
                assert_eq!(range_image, &expected_range_image)
            }
            LowBasisRangeImageStorage::Compact(_) => {
                panic!("expected fused stage1 transition to materialize full table")
            }
            LowBasisRangeImageStorage::FoldedOctets(_) => {
                panic!("two-round transition must not produce folded octets")
            }
        }
        assert_eq!(prover.cached_round_poly.as_ref(), Some(&expected_round2));
    }
}

#[test]
fn stage1_low_basis_range_image_third_round_deferral_matches_materialized_reference() {
    let col_bits = 3usize;
    let ring_bits = 4usize;
    let live_x_cols = 6usize;
    let y_len = 1usize << ring_bits;
    let tau0 = ordered_equality_point(
        &(0..col_bits + ring_bits)
            .map(|index| F::from_u64(index as u64 + 211))
            .collect::<Vec<_>>(),
        col_bits,
        ring_bits,
    );
    let r0 = F::from_u64(223);
    let r1 = F::from_u64(227);
    let r2 = F::from_u64(229);
    let r3 = F::from_u64(233);
    for basis in [4usize, 8] {
        let half = (basis / 2) as i8;
        let digit_witness: Vec<i8> = (0..live_x_cols * y_len)
            .map(|index| ((index * 7 + 3) % basis) as i8 - half)
            .collect();
        let compact_range_image = build_compact_range_image(&digit_witness);
        let mut deferred = LowBasisRangeCheckProver::new(
            std::sync::Arc::from(digit_witness.as_slice()),
            &tau0,
            DigitRangePlan::new(basis).unwrap(),
            live_x_cols,
            col_bits,
            ring_bits,
        )
        .unwrap();
        deferred.compute_round_eq_factored(0);
        deferred.ingest_challenge(0, r0);
        deferred.compute_round_eq_factored(1);
        deferred.ingest_challenge(1, r1);

        assert!(matches!(
            deferred.range_image,
            LowBasisRangeImageStorage::Compact(_)
        ));
        let deferred_round2 = deferred.compute_round_eq_factored(2);

        let round2_range_image = LowBasisRangeCheckProver::<F>::fold_compact_range_image_to_round2(
            &compact_range_image,
            live_x_cols,
            y_len,
            r0,
            r1,
        );
        let mut reference = LowBasisRangeCheckProver::new(
            std::sync::Arc::from(digit_witness.as_slice()),
            &tau0,
            DigitRangePlan::new(basis).unwrap(),
            live_x_cols,
            col_bits,
            ring_bits,
        )
        .unwrap();
        reference.split_eq.bind(r0);
        reference.split_eq.bind(r1);
        reference.rounds_completed = 2;
        let reference_round2 =
            reference.compute_round_sparse_x_y(round2_range_image.len(), |out, index| {
                let left = round2_range_image[index];
                compute_entry_coefficients(
                    out,
                    &reference.polynomial_precomputation,
                    left,
                    round2_range_image[index + 1] - left,
                );
            });
        assert_eq!(deferred_round2, reference_round2);

        let expected_round3_range_image =
            LowBasisRangeCheckProver::<F>::fold_range_image_sparse_x_y(
                round2_range_image.len(),
                |index| round2_range_image[index],
                live_x_cols,
                y_len / 4,
                r2,
            );
        deferred.ingest_challenge(2, r2);
        match &deferred.range_image {
            LowBasisRangeImageStorage::FoldedOctets(actual) => assert_eq!(
                (0..actual.len())
                    .map(|index| actual.value(index))
                    .collect::<Vec<_>>(),
                expected_round3_range_image
            ),
            LowBasisRangeImageStorage::Compact(_) => {
                panic!("low-basis range image must fold after round three")
            }
            LowBasisRangeImageStorage::Materialized(_) => {
                panic!("low-basis range image materialized before it was necessary")
            }
        }

        let expected_round4_range_image =
            LowBasisRangeCheckProver::<F>::fold_range_image_sparse_x_y(
                expected_round3_range_image.len(),
                |index| expected_round3_range_image[index],
                live_x_cols,
                y_len / 8,
                r3,
            );
        deferred.compute_round_eq_factored(3);
        deferred.ingest_challenge(3, r3);
        match &deferred.range_image {
            LowBasisRangeImageStorage::Materialized(actual) => {
                assert_eq!(actual, &expected_round4_range_image)
            }
            LowBasisRangeImageStorage::Compact(_) | LowBasisRangeImageStorage::FoldedOctets(_) => {
                panic!("the last sparse fold must materialize its output")
            }
        }
    }
}

#[test]
fn stage1_later_materialized_prefix_fusion_matches_two_pass_reference() {
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
            LowBasisRangeImageStorage::FoldedOctets(_) => {
                panic!("x-prefix state cannot contain folded octets")
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
        let expected_round3 =
            expected.compute_round_materialized_prefix_x(&expected_next_range_image);

        prover.ingest_challenge(2, r2);

        match &prover.range_image {
            LowBasisRangeImageStorage::Materialized(range_image) => {
                assert_eq!(range_image, &expected_next_range_image)
            }
            LowBasisRangeImageStorage::Compact(_) => {
                panic!("expected fused later prefix stage to stay full")
            }
            LowBasisRangeImageStorage::FoldedOctets(_) => {
                panic!("x-prefix state cannot contain folded octets")
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
        let compact_range_image = build_compact_range_image(&digit_witness_prefix);
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
            LowBasisRangeImageStorage::Compact(_) => {
                LowBasisRangeCheckProver::<F>::fold_compact_range_image_to_round2(
                    &compact_range_image,
                    live_x_cols,
                    y_len,
                    r0,
                    r1,
                )
            }
            LowBasisRangeImageStorage::FoldedOctets(_) => {
                panic!("reference has not ingested the third challenge")
            }
        };
        let current_y_len = current_range_image.len() / expected.live_x_cols;
        let expected_next_range_image = LowBasisRangeCheckProver::<F>::fold_range_image_sparse_x_y(
            current_range_image.len(),
            |index| current_range_image[index],
            expected.live_x_cols,
            current_y_len,
            r2,
        );
        expected.split_eq.bind(r2);
        expected.rounds_completed += 1;
        let expected_round3 =
            expected.compute_round_sparse_x_y(expected_next_range_image.len(), |out, index| {
                let left = expected_next_range_image[index];
                compute_entry_coefficients(
                    out,
                    &expected.polynomial_precomputation,
                    left,
                    expected_next_range_image[index + 1] - left,
                );
            });

        prover.ingest_challenge(2, r2);

        match &prover.range_image {
            LowBasisRangeImageStorage::FoldedOctets(range_image) => assert_eq!(
                (0..range_image.len())
                    .map(|index| range_image.value(index))
                    .collect::<Vec<_>>(),
                expected_next_range_image
            ),
            LowBasisRangeImageStorage::Compact(_) => {
                panic!("expected sparse-x/y transition to fold octets")
            }
            LowBasisRangeImageStorage::Materialized(_) => {
                panic!("sparse-x/y transition materialized before it was necessary")
            }
        }
        assert_eq!(prover.cached_round_poly.as_ref(), Some(&expected_round3));
    }
}

fn scalar_range_entry_coefficients(basis: usize, left: F, right: F) -> Vec<F> {
    let delta = right - left;
    match basis {
        4 => {
            let twice_left = left + left;
            vec![
                left * (left - F::from_u64(2)),
                delta * (twice_left - F::from_u64(2)),
                delta * delta,
            ]
        }
        8 => {
            let twice_left = left + left;
            let sixteen_times_left = twice_left + twice_left;
            let sixteen_times_left = sixteen_times_left + sixteen_times_left;
            let sixteen_times_left = sixteen_times_left + sixteen_times_left;
            let left_squared = left * left;
            let first_quadratic = left_squared - twice_left;
            let second_quadratic =
                left_squared - (sixteen_times_left + twice_left) + F::from_u64(72);
            let delta_squared = delta * delta;
            let first_linear = delta * (twice_left - F::from_u64(2));
            let second_linear = delta * (twice_left - F::from_u64(18));
            vec![
                first_quadratic * second_quadratic,
                first_quadratic * second_linear + first_linear * second_quadratic,
                first_quadratic * delta_squared
                    + first_linear * second_linear
                    + delta_squared * second_quadratic,
                delta_squared * (first_linear + second_linear),
                delta_squared * delta_squared,
            ]
        }
        _ => unreachable!("scalar Stage 1 reference only supports basis 4 or 8"),
    }
}

struct ScalarStage1Reference {
    range_image: Vec<F>,
    split_eq: GruenSplitEq<F>,
    basis: usize,
    num_vars: usize,
    rounds_completed: usize,
}

impl ScalarStage1Reference {
    fn new(
        digit_witness: &[i8],
        tau: &[F],
        basis: usize,
        live_x_cols: usize,
        col_bits: usize,
        ring_bits: usize,
    ) -> Self {
        let padded = pad_compact_witness(digit_witness, live_x_cols, col_bits, ring_bits);
        Self {
            range_image: build_compact_range_image(&padded)
                .into_iter()
                .map(|value| F::from_i64(i64::from(value)))
                .collect(),
            split_eq: GruenSplitEq::new(tau).unwrap(),
            basis,
            num_vars: col_bits + ring_bits,
            rounds_completed: 0,
        }
    }

    fn round_polynomial(&self) -> EqFactoredUniPoly<F> {
        let (e_first, e_second) = self.split_eq.remaining_eq_tables();
        let first_bits = e_first.len().trailing_zeros();
        let mut coefficients = vec![F::zero(); self.basis / 2 + 1];
        for (pair_index, pair) in self.range_image.chunks_exact(2).enumerate() {
            let inner_index = pair_index & (e_first.len() - 1);
            let outer_index = pair_index >> first_bits;
            let weight = e_first[inner_index] * e_second[outer_index];
            let entry = scalar_range_entry_coefficients(self.basis, pair[0], pair[1]);
            for (coefficient, entry_coefficient) in coefficients.iter_mut().zip(entry) {
                *coefficient += weight * entry_coefficient;
            }
        }
        EqFactoredUniPoly::from_q_coeffs(coefficients)
    }

    fn fold(&mut self, challenge: F) {
        let mut next = Vec::with_capacity(self.range_image.len() / 2);
        for pair in self.range_image.chunks_exact(2) {
            next.push(pair[0] + challenge * (pair[1] - pair[0]));
        }
        self.range_image = next;
        self.split_eq.bind(challenge);
        self.rounds_completed += 1;
    }

    fn final_range_image_eval(&self) -> F {
        assert_eq!(self.range_image.len(), 1);
        self.range_image[0]
    }
}

impl EqFactoredSumcheckInstanceProver<F> for ScalarStage1Reference {
    fn num_rounds(&self) -> usize {
        self.num_vars
    }

    fn degree_bound(&self) -> usize {
        self.basis / 2
    }

    fn input_claim(&self) -> F {
        F::zero()
    }

    fn current_linear_factor_evals(&self) -> (F, F) {
        self.split_eq.linear_factor_evals()
    }

    fn compute_round_eq_factored(&mut self, round: usize) -> EqFactoredUniPoly<F> {
        assert_eq!(round, self.rounds_completed);
        self.round_polynomial()
    }

    fn ingest_challenge(&mut self, round: usize, challenge: F) {
        assert_eq!(round, self.rounds_completed);
        self.fold(challenge);
    }
}

fn scalar_fold_octet_class(basis: usize, class: usize, challenges: [F; 3]) -> F {
    let mut values = match basis {
        4 => (0..8)
            .map(|index| F::from_u64(((class >> index) & 1) as u64 * 2))
            .collect::<Vec<_>>(),
        8 => {
            let range_values = [0u64, 2, 6, 12];
            let left = class >> 8;
            let right = class & 0xff;
            (0..4)
                .map(|index| F::from_u64(range_values[(left >> (2 * index)) & 3]))
                .chain((0..4).map(|index| F::from_u64(range_values[(right >> (2 * index)) & 3])))
                .collect()
        }
        _ => unreachable!("folded octets only support basis 4 or 8"),
    };
    for challenge in challenges {
        values = values
            .chunks_exact(2)
            .map(|pair| pair[0] + challenge * (pair[1] - pair[0]))
            .collect();
    }
    values[0]
}

fn expected_octet_class_code(basis: usize, octet: &[i16]) -> u16 {
    match basis {
        4 => octet.iter().enumerate().fold(0u16, |code, (index, value)| {
            code | (u16::from(*value == 2) << index)
        }),
        8 => {
            let digit_code = |values: &[i16]| {
                values
                    .iter()
                    .enumerate()
                    .fold(0u16, |code, (index, value)| {
                        let digit = match value {
                            0 => 0,
                            2 => 1,
                            6 => 2,
                            12 => 3,
                            _ => unreachable!("basis-8 range image"),
                        };
                        code | (digit << (2 * index))
                    })
            };
            (digit_code(&octet[..4]) << 8) | digit_code(&octet[4..])
        }
        _ => unreachable!("folded octets only support basis 4 or 8"),
    }
}

fn assert_folded_octets_match_scalar(
    folded: &FoldedOctetRangeImage<F>,
    scalar_range_image: &[F],
    compact_range_image: &[i16],
    basis: usize,
    live_x_cols: usize,
    y_len: usize,
    challenges: [F; 3],
) {
    let next_y_len = y_len / 8;
    let expected_codes = compact_range_image
        .chunks_exact(y_len)
        .flat_map(|column| {
            column
                .chunks_exact(8)
                .map(|octet| expected_octet_class_code(basis, octet))
        })
        .collect::<Vec<_>>();
    assert_eq!(expected_codes.len(), live_x_cols * next_y_len);
    assert_eq!(folded.class_codes, expected_codes);

    for (class, (&value, taylor)) in folded
        .class_values
        .iter()
        .zip(&folded.class_taylor_coefficients)
        .enumerate()
    {
        let expected_value = scalar_fold_octet_class(basis, class, challenges);
        assert_eq!(
            value, expected_value,
            "class value mismatch for class={class}"
        );
        let expected_taylor =
            scalar_range_entry_coefficients(basis, expected_value, expected_value + F::one());
        let expected_taylor = match basis {
            4 => [expected_taylor[0], expected_taylor[1], F::zero(), F::zero()],
            8 => [
                expected_taylor[0],
                expected_taylor[1],
                expected_taylor[2],
                expected_taylor[3],
            ],
            _ => unreachable!("folded octets only support basis 4 or 8"),
        };
        assert_eq!(
            *taylor, expected_taylor,
            "Taylor row mismatch for class={class}"
        );
    }

    assert_eq!(
        (0..folded.len())
            .map(|index| folded.value(index))
            .collect::<Vec<_>>(),
        scalar_range_image
    );
}

fn new_stage1_transcript() -> AkitaTranscript<F> {
    AkitaTranscript::new(transcript_labels::DOMAIN_AKITA_PROTOCOL)
}

fn sample_stage1_challenge(transcript: &mut AkitaTranscript<F>) -> F {
    transcript.challenge_scalar(transcript_labels::CHALLENGE_SUMCHECK_ROUND)
}

#[test]
fn stage1_compact_pipeline_randomized_scalar_transcript_and_bytes_differential() {
    let shapes = [
        (6usize, 3usize, 4usize),
        (1, 3, 4),
        (7, 3, 4),
        (8, 3, 4),
        (5, 3, 5),
    ];
    for basis in [4usize, 8] {
        for (case, &(live_x_cols, col_bits, ring_bits)) in shapes.iter().enumerate() {
            let seed = 0x5a17_0000_u64 | ((basis as u64) << 8) | case as u64;
            let mut rng = StdRng::seed_from_u64(seed);
            let y_len = 1usize << ring_bits;
            let half = (basis / 2) as i8;
            let digit_witness = (0..live_x_cols * y_len)
                .map(|_| rng.gen_range(0..basis) as i8 - half)
                .collect::<Vec<_>>();
            let compact_range_image = build_compact_range_image(&digit_witness);
            let column_then_ring_tau = (0..col_bits + ring_bits)
                .map(|_| F::from_u64(rng.gen_range(2..u64::MAX)))
                .collect::<Vec<_>>();
            let tau = ordered_equality_point(&column_then_ring_tau, col_bits, ring_bits);
            let manual_challenges = (0..col_bits + ring_bits)
                .map(|_| F::from_u64(rng.gen_range(2..u64::MAX)))
                .collect::<Vec<_>>();

            let mut optimized = LowBasisRangeCheckProver::new(
                std::sync::Arc::from(digit_witness.as_slice()),
                &tau,
                DigitRangePlan::new(basis).unwrap(),
                live_x_cols,
                col_bits,
                ring_bits,
            )
            .unwrap();
            let mut scalar = ScalarStage1Reference::new(
                &digit_witness,
                &tau,
                basis,
                live_x_cols,
                col_bits,
                ring_bits,
            );

            for (round, &challenge) in manual_challenges.iter().enumerate() {
                let optimized_poly = optimized.compute_round_eq_factored(round);
                let scalar_poly = scalar.compute_round_eq_factored(round);
                assert_eq!(
                    optimized_poly, scalar_poly,
                    "round polynomial mismatch basis={basis} case={case} round={round}"
                );
                assert_eq!(
                    optimized.current_linear_factor_evals(),
                    scalar.current_linear_factor_evals(),
                    "linear factor mismatch basis={basis} case={case} round={round}"
                );

                optimized.ingest_challenge(round, challenge);
                scalar.ingest_challenge(round, challenge);
                if round == 2 {
                    match &optimized.range_image {
                        LowBasisRangeImageStorage::FoldedOctets(folded)
                            if live_x_cols < 1usize << col_bits =>
                        {
                            assert_folded_octets_match_scalar(
                                folded,
                                &scalar.range_image[..live_x_cols * (y_len / 8)],
                                &compact_range_image,
                                basis,
                                live_x_cols,
                                y_len,
                                [
                                    manual_challenges[0],
                                    manual_challenges[1],
                                    manual_challenges[2],
                                ],
                            );
                        }
                        LowBasisRangeImageStorage::Compact(_) => {
                            panic!("round three must leave compact digit storage")
                        }
                        LowBasisRangeImageStorage::FoldedOctets(_) => {
                            panic!("full-width round must not use the sparse folded state")
                        }
                        LowBasisRangeImageStorage::Materialized(actual)
                            if live_x_cols == 1usize << col_bits =>
                        {
                            assert_eq!(actual, &scalar.range_image);
                        }
                        LowBasisRangeImageStorage::Materialized(_) => {
                            panic!("round three must retain folded octet classes")
                        }
                    }
                } else if round >= 3 {
                    match &optimized.range_image {
                        LowBasisRangeImageStorage::Materialized(actual) => {
                            let live_len = actual.len();
                            assert_eq!(actual, &scalar.range_image[..live_len]);
                        }
                        LowBasisRangeImageStorage::Compact(_)
                        | LowBasisRangeImageStorage::FoldedOctets(_) => {
                            panic!("folded octets must materialize after the next fold")
                        }
                    }
                }
            }
            assert_eq!(
                optimized.final_range_image_eval(),
                scalar.final_range_image_eval(),
                "final evaluation mismatch basis={basis} case={case}"
            );

            let mut optimized_for_proof = LowBasisRangeCheckProver::new(
                std::sync::Arc::from(digit_witness.as_slice()),
                &tau,
                DigitRangePlan::new(basis).unwrap(),
                live_x_cols,
                col_bits,
                ring_bits,
            )
            .unwrap();
            let mut scalar_for_proof = ScalarStage1Reference::new(
                &digit_witness,
                &tau,
                basis,
                live_x_cols,
                col_bits,
                ring_bits,
            );
            let mut optimized_transcript = new_stage1_transcript();
            let mut scalar_transcript = new_stage1_transcript();
            let (optimized_proof, optimized_challenges, optimized_claim) = optimized_for_proof
                .prove::<F, _, _>(&mut optimized_transcript, sample_stage1_challenge)
                .unwrap();
            let (scalar_proof, scalar_challenges, scalar_claim) = scalar_for_proof
                .prove::<F, _, _>(&mut scalar_transcript, sample_stage1_challenge)
                .unwrap();
            assert_eq!(optimized_proof, scalar_proof);
            assert_eq!(optimized_challenges, scalar_challenges);
            assert_eq!(optimized_claim, scalar_claim);
            assert_eq!(
                optimized_for_proof.final_range_image_eval(),
                scalar_for_proof.final_range_image_eval()
            );

            let mut optimized_bytes = Vec::new();
            optimized_proof
                .serialize_compressed(&mut optimized_bytes)
                .unwrap();
            let mut scalar_bytes = Vec::new();
            scalar_proof
                .serialize_compressed(&mut scalar_bytes)
                .unwrap();
            assert_eq!(optimized_bytes, scalar_bytes);
            assert_eq!(
                optimized_transcript.challenge_scalar(b"stage1-differential-after-proof"),
                scalar_transcript.challenge_scalar(b"stage1-differential-after-proof")
            );
        }
    }
}
