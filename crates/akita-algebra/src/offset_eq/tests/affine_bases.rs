use super::*;

#[test]
fn weighted_bases_match_independent_oracle() {
    let mut rng = StdRng::seed_from_u64(0xBA5E_5CA1);
    let challenges = random_vec(&mut rng, 17);
    let base_offsets = [3usize, 211, 419];
    let base_scales = random_vec(&mut rng, base_offsets.len());
    let digit_weights = random_vec(&mut rng, 3);
    let high = random_vec(&mut rng, 26);
    let outer_stride = digit_weights.len();

    let got = eval_affine_digit_intervals(
        &challenges,
        &base_offsets,
        0,
        high.len(),
        outer_stride,
        1,
        &digit_weights,
        &high,
        &[],
        &base_scales,
    )
    .unwrap();
    let expected = base_offsets
        .iter()
        .zip(&base_scales)
        .map(|(&base, &base_scale)| {
            let mut family = F::zero();
            for (outer, &high_weight) in high.iter().enumerate() {
                for (digit, &digit_weight) in digit_weights.iter().enumerate() {
                    family += high_weight
                        * digit_weight
                        * eq_eval_at_index(&challenges, base + outer_stride * outer + digit);
                }
            }
            base_scale * family
        })
        .sum();
    assert_eq!(got, expected);

    let err = eval_affine_digit_intervals(
        &challenges,
        &base_offsets,
        0,
        high.len(),
        outer_stride,
        1,
        &digit_weights,
        &high,
        &[],
        &base_scales[..2],
    )
    .unwrap_err();
    assert!(matches!(err, AkitaError::InvalidSize { .. }));
}

#[test]
fn weighted_bases_match_oracle_across_residues_and_parallel_buckets() {
    let mut rng = StdRng::seed_from_u64(0xBA5E_B0C7);
    let challenges = random_vec(&mut rng, 16);
    let base_offsets = [0usize, 64, 128, 192, 256, 320, 384, 448, 512, 576, 1, 65];
    let base_scales = random_vec(&mut rng, base_offsets.len());
    let digit_weights = random_vec(&mut rng, 3);
    let high = random_vec(&mut rng, 512);
    let low = random_vec(&mut rng, 4);
    let outer_stride = 5;
    let live_len = high.len() * low.len();

    let same_residue_rows = 10 * high.len();
    assert!(same_residue_rows >= PARALLEL_HIGH_ROWS_MIN);
    assert!(bucketed_high_rows_plan(
        same_residue_rows,
        outer_stride + 1,
        challenges.len() - low.len().trailing_zeros() as usize,
    )
    .unwrap()
    .is_some());

    let got = eval_affine_digit_intervals(
        &challenges,
        &base_offsets,
        0,
        live_len,
        outer_stride,
        1,
        &digit_weights,
        &high,
        &low,
        &base_scales,
    )
    .unwrap();
    let expected = base_offsets
        .iter()
        .zip(&base_scales)
        .map(|(&base, &base_scale)| {
            base_scale
                * reference_affine_digit_interval(
                    &challenges,
                    base,
                    0,
                    live_len,
                    outer_stride,
                    &digit_weights,
                    &high,
                    &low,
                )
        })
        .sum();
    assert_eq!(got, expected);
}

#[test]
fn many_short_families_avoid_quadratic_bucketing() {
    let mut rng = StdRng::seed_from_u64(0xA11F_1EE7);
    let challenges = random_vec(&mut rng, 13);
    let base_offsets = (0..8usize).collect::<Vec<_>>();
    let outer_stride = 4095usize;
    assert_eq!(
        bucketed_high_rows_plan(base_offsets.len(), outer_stride + 1, challenges.len()).unwrap(),
        None
    );

    let got = eval_affine_digit_intervals(
        &challenges,
        &base_offsets,
        0,
        1,
        outer_stride,
        1,
        &[F::one()][..],
        &[F::one()][..],
        &[F::one()],
        &[],
    )
    .unwrap();
    let expected = base_offsets
        .iter()
        .map(|&base| eq_eval_at_index(&challenges, base))
        .sum();
    assert_eq!(got, expected);
}

#[test]
fn bit_aligned_digit_intervals_factor_without_carry_buckets() {
    let mut rng = StdRng::seed_from_u64(0xB17A_11ED);
    let challenge_cases = [
        random_vec(&mut rng, 24),
        (0..24)
            .map(|bit| if bit % 2 == 0 { F::zero() } else { F::one() })
            .collect::<Vec<_>>(),
    ];
    let digit_weights = random_vec(&mut rng, 32);
    let high = random_vec(&mut rng, 269);
    let base_offsets = [0usize, 112, 512, 1008];
    let base_scales = random_vec(&mut rng, base_offsets.len());
    let outer_stride = 128usize;
    let outer_start = 13usize;

    for challenges in &challenge_cases {
        for live_len in [145usize, 256] {
            let factored = try_eval_bit_aligned_digit_intervals(
                challenges,
                &base_offsets,
                outer_start,
                live_len,
                outer_stride,
                1,
                &digit_weights,
                &high,
                &[],
                &base_scales,
            )
            .unwrap()
            .expect("production packing geometry has at most one carry");
            let expected = base_offsets
                .iter()
                .zip(&base_scales)
                .map(|(&base, &base_scale)| {
                    let mut family = F::zero();
                    for (outer, &high_weight) in
                        high.iter().enumerate().skip(outer_start).take(live_len)
                    {
                        for (digit, &digit_weight) in digit_weights.iter().enumerate() {
                            family += high_weight
                                * digit_weight
                                * eq_eval_at_index(
                                    challenges,
                                    base + outer_stride * (outer - outer_start) + digit,
                                );
                        }
                    }
                    base_scale * family
                })
                .sum();
            assert_eq!(factored, expected);
            assert_eq!(
                eval_affine_digit_intervals(
                    challenges,
                    &base_offsets,
                    outer_start,
                    live_len,
                    outer_stride,
                    1,
                    &digit_weights,
                    &high,
                    &[],
                    &base_scales,
                )
                .unwrap(),
                expected
            );
        }
    }

    assert!(try_eval_bit_aligned_digit_intervals(
        &challenge_cases[0],
        &[0],
        0,
        high.len(),
        outer_stride - 1,
        1,
        &digit_weights,
        &high,
        &[],
        &[],
    )
    .unwrap()
    .is_none());
}
