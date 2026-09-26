use super::*;
use akita_sumcheck::multilinear_eval;
use akita_types::DigitRangeEqualityPoint;
use jolt_field::Prime128Offset275;
use jolt_poly::OmittedConstantPoly;

type F = Prime128Offset275;

fn advance_eq_factored_claim(claim: F, tau: F, poly: &OmittedConstantPoly<F>, challenge: F) -> F {
    let constant = claim - tau * poly.nonconstant_term_sum_at_one();
    constant + poly.evaluate_nonconstant_terms(challenge)
}

fn packed(witness: &[i8]) -> PackedSignedDigits {
    PackedSignedDigits::from_i8_digits_auto(witness.to_vec())
}

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
        packed(&[]),
        &tau,
        DigitRangePlan::new(4).unwrap(),
        1,
        0,
        usize::BITS as usize
    )
    .is_err());

    let tau = vec![F::zero(); usize::BITS as usize + 1];
    assert!(LowBasisRangeCheckProver::<F>::new(
        packed(&[]),
        &tau,
        DigitRangePlan::new(4).unwrap(),
        3,
        2,
        usize::BITS as usize - 1
    )
    .is_err());

    assert!(LowBasisRangeCheckProver::<F>::new(
        packed(&[]),
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
fn stage1_materialized_fold_matches_direct_formula() {
    let r = F::from_u64(41);

    let range_image_prefix = vec![2, 6, 12, 2, 6, 12, 2, 6, 12, 2];
    let materialized: Vec<F> = range_image_prefix
        .iter()
        .map(|&value| F::from_i64(i64::from(value)))
        .collect();
    let folded: Vec<F> = materialized
        .chunks(5)
        .flat_map(|row| LowBasisRangeCheckProver::<F>::fold_live_prefix(row, r))
        .collect();
    assert_eq!(
        folded,
        fold_compact_range_image_prefix_x_reference(&range_image_prefix, 5, 2, r)
    );

    let dense_range_image = vec![2, 6, 12, 2, 6, 12];
    let mut materialized: Vec<F> = dense_range_image
        .iter()
        .map(|&value| F::from_i64(i64::from(value)))
        .collect();
    fold_evals_in_place(&mut materialized, r);
    assert_eq!(
        materialized,
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
            packed(&compact_digit_witness),
            &tau0,
            DigitRangePlan::new(basis).unwrap(),
            1usize << col_bits,
            col_bits,
            ring_bits,
        )
        .unwrap();
        let stage1_poly = prover.compute_round_eq_factored(0);
        let compact_range_image = build_compact_range_image(&compact_digit_witness);
        let reference = compute_range_round_polynomial_from_range_image(
            &prover.split_eq,
            &prover.range_poly,
            |j| {
                (
                    F::from_i64(i64::from(compact_range_image[2 * j])),
                    F::from_i64(i64::from(compact_range_image[2 * j + 1])),
                )
            },
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
                packed(&digit_witness_prefix),
                &tau0,
                DigitRangePlan::new(basis).unwrap(),
                live_x_cols,
                col_bits,
                ring_bits,
            )
            .unwrap();
            let mut padded_prover = LowBasisRangeCheckProver::new(
                packed(&padded_digit_witness),
                &tau0,
                DigitRangePlan::new(basis).unwrap(),
                1usize << col_bits,
                col_bits,
                ring_bits,
            )
            .unwrap();
            let mut challenges = Vec::new();
            let mut prefix_claim = F::zero();
            let mut padded_claim = F::zero();

            for round in 0..(col_bits + ring_bits) {
                let prefix_poly = prefix_prover.compute_round_eq_factored(round);
                let padded_poly = padded_prover.compute_round_eq_factored(round);
                assert_eq!(
                    prefix_poly, padded_poly,
                    "round {round} polynomial mismatch live_x_cols={live_x_cols} basis={basis}"
                );

                let challenge = F::from_u64((round as u64) + 29);
                challenges.push(challenge);
                prefix_claim = advance_eq_factored_claim(
                    prefix_claim,
                    prefix_prover.current_tau(),
                    &prefix_poly,
                    challenge,
                );
                padded_claim = advance_eq_factored_claim(
                    padded_claim,
                    padded_prover.current_tau(),
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
fn stage1_prefix_x_rounds_allow_ring_bits_at_least_col_bits() {
    // Regression test: the live-x-prefix round kernels used to assert
    // rounds_completed < col_bits, but they run in the x phase where
    // rounds_completed >= ring_bits, so the assert fired spuriously on the
    // first x-phase round whenever ring_bits >= col_bits (the true invariant
    // is rounds_completed < num_vars). Ring dimension 128 (ring_bits = 7)
    // with 100 live columns (col_bits = 7) hits exactly that round.
    let ring_bits = 7usize;
    let live_x_cols = 100usize;
    let col_bits = live_x_cols.next_power_of_two().trailing_zeros() as usize;
    assert!(ring_bits >= col_bits);
    assert!(live_x_cols < 1usize << col_bits);
    let y_len = 1usize << ring_bits;
    for basis in [4usize, 8] {
        let half = (basis / 2) as i8;
        let digit_witness_prefix: Vec<i8> = (0..(live_x_cols * y_len))
            .map(|i| ((i * 11 + 3) % basis) as i8 - half)
            .collect();
        let padded_digit_witness =
            pad_compact_witness(&digit_witness_prefix, live_x_cols, col_bits, ring_bits);
        let tau0: Vec<F> = (0..(col_bits + ring_bits))
            .map(|i| F::from_u64((i as u64) + 157))
            .collect();
        let tau0 = ordered_equality_point(&tau0, col_bits, ring_bits);
        let mut prefix_prover = LowBasisRangeCheckProver::new(
            packed(&digit_witness_prefix),
            &tau0,
            DigitRangePlan::new(basis).unwrap(),
            live_x_cols,
            col_bits,
            ring_bits,
        )
        .unwrap();
        let mut padded_prover = LowBasisRangeCheckProver::new(
            packed(&padded_digit_witness),
            &tau0,
            DigitRangePlan::new(basis).unwrap(),
            1usize << col_bits,
            col_bits,
            ring_bits,
        )
        .unwrap();
        let mut challenges = Vec::new();
        let mut prefix_claim = F::zero();
        let mut padded_claim = F::zero();

        for round in 0..(col_bits + ring_bits) {
            let prefix_poly = prefix_prover.compute_round_eq_factored(round);
            let padded_poly = padded_prover.compute_round_eq_factored(round);
            assert_eq!(
                prefix_poly, padded_poly,
                "round {round} polynomial mismatch live_x_cols={live_x_cols} basis={basis}"
            );

            let challenge = F::from_u64((round as u64) + 163);
            challenges.push(challenge);
            prefix_claim = advance_eq_factored_claim(
                prefix_claim,
                prefix_prover.current_tau(),
                &prefix_poly,
                challenge,
            );
            padded_claim = advance_eq_factored_claim(
                padded_claim,
                padded_prover.current_tau(),
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

/// Run every round against the dense kernel on the zero-padded field table,
/// with the witness packed at `bit_width` bits per digit.
fn assert_rounds_match_dense_reference(
    basis: usize,
    bit_width: u8,
    col_bits: usize,
    ring_bits: usize,
    live_x_cols: usize,
) {
    let half = (basis / 2) as i8;
    let num_vars = col_bits + ring_bits;
    let witness: Vec<i8> = (0..live_x_cols << ring_bits)
        .map(|i| {
            let mixed = (i as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 58;
            (mixed as usize % basis) as i8 - half
        })
        .collect();
    let tau0: Vec<F> = (0..num_vars)
        .map(|i| F::from_u64(3 * i as u64 + 43))
        .collect();
    let tau0 = ordered_equality_point(&tau0, col_bits, ring_bits);
    let mut prover = LowBasisRangeCheckProver::new(
        PackedSignedDigits::from_i8_digits(witness.clone(), bit_width).unwrap(),
        &tau0,
        DigitRangePlan::new(basis).unwrap(),
        live_x_cols,
        col_bits,
        ring_bits,
    )
    .unwrap();
    let mut reference: Vec<F> = build_compact_range_image(&witness)
        .into_iter()
        .map(|s| F::from_i64(i64::from(s)))
        .collect();
    reference.resize(1usize << num_vars, F::zero());
    let mut reference_eq = GruenSplitEq::new(&tau0).unwrap();
    let precomputation = RangePoly::new(basis);
    let shape = format!(
        "basis={basis} width={bit_width} col_bits={col_bits} ring_bits={ring_bits} live={live_x_cols}"
    );

    assert_eq!(
        matches!(prover.range_image, LowBasisRangeImageStorage::Compact(_)),
        num_vars >= octet_prefix::OCTET_PREFIX_ROUNDS,
        "{shape} initial storage"
    );
    for round in 0..num_vars {
        let poly = prover.compute_round_eq_factored(round);
        let expected =
            compute_range_round_polynomial_from_range_image(&reference_eq, &precomputation, |j| {
                (reference[2 * j], reference[2 * j + 1])
            });
        assert_eq!(poly, expected, "{shape} round={round}");

        let r = F::from_u64(5 * round as u64 + 71);
        prover.ingest_challenge(round, r);
        reference_eq.bind(r);
        fold_evals_in_place(&mut reference, r);
        assert_eq!(
            matches!(prover.range_image, LowBasisRangeImageStorage::Compact(_)),
            num_vars >= octet_prefix::OCTET_PREFIX_ROUNDS
                && round + 1 < octet_prefix::OCTET_PREFIX_ROUNDS,
            "{shape} storage after round={round}"
        );
    }
    assert_eq!(reference.len(), 1);
    assert_eq!(prover.final_range_image_eval(), reference[0], "{shape}");
}

#[test]
fn stage1_rounds_match_dense_reference() {
    const SHAPES: [(usize, usize, usize); 23] = [
        (1, 0, 1),
        (3, 0, 5),
        (2, 0, 3),
        (1, 2, 1),
        (3, 2, 5),
        (3, 2, 6),
        (3, 2, 8),
        (1, 2, 2),
        (2, 2, 3),
        (3, 3, 5),
        (0, 4, 1),
        (3, 4, 6),
        (2, 6, 3),
        (5, 2, 12),
        (4, 6, 16),
        (2, 3, 1),
        (7, 7, 100),
        (5, 0, 20),
        (4, 1, 9),
        (1, 1, 2),
        (0, 2, 1),
        (2, 1, 3),
        (0, 3, 1),
    ];
    // The octet prefix reads classes from the packed bytes: table lookups up
    // to three bits per digit, one digit at a time above that.
    for basis in [4usize, 8] {
        let log_basis = basis.trailing_zeros() as u8;
        for bit_width in [log_basis, log_basis + 1, 8] {
            for (col_bits, ring_bits, live_x_cols) in SHAPES {
                assert_rounds_match_dense_reference(
                    basis,
                    bit_width,
                    col_bits,
                    ring_bits,
                    live_x_cols,
                );
            }
        }
    }
}

fn build_compact_range_image(digit_witness: &[i8]) -> Vec<i16> {
    digit_witness
        .iter()
        .copied()
        .map(range_image_from_digit)
        .collect()
}

pub(crate) fn pad_compact_witness(
    digit_witness_prefix: &[i8],
    live_x_cols: usize,
    col_bits: usize,
    ring_bits: usize,
) -> Vec<i8> {
    let x_len = 1usize << col_bits;
    let y_len = 1usize << ring_bits;
    let mut padded = vec![0i8; x_len * y_len];
    for x in 0..live_x_cols {
        let offset = x * y_len;
        padded[offset..offset + y_len]
            .copy_from_slice(&digit_witness_prefix[offset..offset + y_len]);
    }
    padded
}
