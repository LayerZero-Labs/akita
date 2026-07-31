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
        &[F::one()],
        &[F::one()],
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
