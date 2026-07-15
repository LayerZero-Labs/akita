use super::*;
use crate::eq_poly::EqPolynomial;
use crate::RandomSampling;
use akita_field::Fp64;
use rand::rngs::StdRng;
use rand::SeedableRng;

type F = Fp64<4294967197>;

#[test]
fn offset_eq_window_matches_scalar_eq_across_low_bits() {
    let mut rng = StdRng::seed_from_u64(0x0FF5E7);
    for n in 0..14usize {
        let challenges: Vec<F> = (0..n).map(|_| F::random(&mut rng)).collect();
        let domain = 1usize << n;
        for low_cap in [0usize, 1, 4, 8, 16] {
            let window = OffsetEqWindow::with_low_bits(&challenges, low_cap).unwrap();
            // Every in-domain index, plus a couple of out-of-domain probes.
            for index in (0..domain).chain([domain, domain + 1, domain + 7]) {
                assert_eq!(
                    window.eval(index),
                    eq_eval_at_index(&challenges, index),
                    "n={n} low_cap={low_cap} index={index}"
                );
            }
        }
    }
}

#[test]
fn offset_eq_window_caps_low_table_and_crosses_boundary() {
    let mut rng = StdRng::seed_from_u64(0xB0117);
    // 20 coordinates exceed the 16-bit cap; the low table must stay capped
    // while evaluation still matches the scalar oracle across the low/high
    // boundary and at the exact domain end.
    let n = 20usize;
    let challenges: Vec<F> = (0..n).map(|_| F::random(&mut rng)).collect();
    let window = OffsetEqWindow::new(&challenges).unwrap();
    assert_eq!(window.eq_low.len(), 1usize << OFFSET_EQ_LOW_BITS_CAP);
    let domain = 1usize << n;
    for index in [
        0usize,
        (1 << OFFSET_EQ_LOW_BITS_CAP) - 1,
        1 << OFFSET_EQ_LOW_BITS_CAP,
        (1 << OFFSET_EQ_LOW_BITS_CAP) + 5,
        domain - 1,
        domain,
        domain + 9,
    ] {
        assert_eq!(
            window.eval(index),
            eq_eval_at_index(&challenges, index),
            "index={index}"
        );
    }
}

fn reference_offset_eq_tensor(x_challenges: &[F], offset: usize, scale: F, factors: &[&[F]]) -> F {
    let dims: Vec<usize> = factors
        .iter()
        .map(|f| {
            if f.is_empty() {
                1
            } else {
                f.len().next_power_of_two()
            }
        })
        .collect();
    let total: usize = dims.iter().product();
    let eq_table = EqPolynomial::evals(x_challenges).unwrap();
    let mut acc = F::zero();
    for z in 0..total {
        let mut idx = z;
        let mut prod = scale;
        for (j, &f) in factors.iter().enumerate() {
            let local = idx % dims[j];
            idx /= dims[j];
            prod *= if f.is_empty() {
                if local == 0 {
                    F::one()
                } else {
                    F::zero()
                }
            } else if local < f.len() {
                f[local]
            } else {
                F::zero()
            };
        }
        let global = offset + z;
        if global < eq_table.len() {
            acc += eq_table[global] * prod;
        }
    }
    acc
}

fn random_vec(rng: &mut StdRng, len: usize) -> Vec<F> {
    (0..len).map(|_| F::random(rng)).collect()
}

fn reference_pow2_peeled_blocks(x_challenges: &[F], offset: usize, blocks: &[Vec<F>]) -> F {
    let inner_len = blocks.first().map_or(1, Vec::len);
    let eq_table = EqPolynomial::evals(x_challenges).unwrap();
    let mut acc = F::zero();

    for (q, block) in blocks.iter().enumerate() {
        assert_eq!(block.len(), inner_len);
        for (u, &value) in block.iter().enumerate() {
            let idx = offset + u + inner_len * q;
            if idx < eq_table.len() {
                acc += value * eq_table[idx];
            }
        }
    }

    acc
}

#[test]
fn interval_matches_reference_offset_zero() {
    let mut rng = StdRng::seed_from_u64(0xB1);
    let factor = random_vec(&mut rng, 21);
    let r = random_vec(&mut rng, 5);
    let scale = F::random(&mut rng);

    let got = eval_offset_eq_interval(&r, 0, scale, &factor).unwrap();
    let expected = reference_offset_eq_tensor(&r, 0, scale, &[&factor]);
    assert_eq!(got, expected);
}

#[test]
fn interval_matches_reference_carry_offset() {
    let mut rng = StdRng::seed_from_u64(0xB2);
    let factor = random_vec(&mut rng, 21);
    let r = random_vec(&mut rng, 5);
    let scale = F::random(&mut rng);

    // Interval [11, 31] inside domain 2^5 = 32, carry-heavy offset.
    let got = eval_offset_eq_interval(&r, 11, scale, &factor).unwrap();
    let expected = reference_offset_eq_tensor(&r, 11, scale, &[&factor]);
    assert_eq!(got, expected);
}

#[test]
fn interval_matches_reference_sweep() {
    let mut rng = StdRng::seed_from_u64(0xB3);
    for n in 3..12usize {
        let domain = 1usize << n;
        for &len in &[1usize, 3, 8, 21, 100, 300] {
            let factor = random_vec(&mut rng, len);
            let r = random_vec(&mut rng, n);
            let scale = F::random(&mut rng);
            // Offsets: zero, carry-heavy flush-to-top, a mid value, plus an
            // offset that pushes the interval tail past the domain (clamp).
            let max_offset = domain.saturating_sub(len);
            let mut offsets = vec![0usize];
            if max_offset > 0 {
                offsets.push(max_offset);
                offsets.push(max_offset / 2);
            }
            offsets.push(domain); // fully outside the domain -> zero
            for &offset in &offsets {
                let got = eval_offset_eq_interval(&r, offset, scale, &factor).unwrap();
                let expected = reference_offset_eq_tensor(&r, offset, scale, &[&factor]);
                assert_eq!(got, expected, "n={n} len={len} offset={offset}");
            }
        }
    }
}

#[test]
fn interval_matches_reference_with_partial_clamp() {
    let mut rng = StdRng::seed_from_u64(0xB4);
    // len 300 padded to 512 = 2^9 fits in n = 9 bits; offset pushes the tail
    // of the interval past 2^9 so the high indices are clamped/dropped.
    let n = 9usize;
    let factor = random_vec(&mut rng, 300);
    let r = random_vec(&mut rng, n);
    let scale = F::random(&mut rng);
    let offset = 300; // 300 + 300 = 600 > 512, so indices >= 512 drop out
    let got = eval_offset_eq_interval(&r, offset, scale, &factor).unwrap();
    let expected = reference_offset_eq_tensor(&r, offset, scale, &[&factor]);
    assert_eq!(got, expected);
}

#[test]
fn interval_offset_outside_domain_is_zero() {
    let mut rng = StdRng::seed_from_u64(0xB5);
    let factor = random_vec(&mut rng, 4);
    let r = random_vec(&mut rng, 3);
    let got = eval_offset_eq_interval(&r, 1usize << r.len(), F::one(), &factor).unwrap();
    assert_eq!(got, F::zero());
}

/// Combine per-block carry buckets `[A0, A1]` with the high `eq` factor,
/// the way `compute_r_contribution` does: `A0` lands on `offset_high + q`
/// and the carried `A1` on `offset_high + q + 1`.
fn combine_pow2_carry_terms(
    x_challenges: &[F],
    offset: usize,
    peeled_bits: usize,
    carry_terms: &[[F; 2]],
) -> F {
    let offset_high = offset >> peeled_bits;
    let high = &x_challenges[peeled_bits..];
    let mut out = F::zero();
    for (q, terms) in carry_terms.iter().enumerate() {
        out += terms[0] * eq_eval_at_index(high, offset_high + q);
        out += terms[1] * eq_eval_at_index(high, offset_high + q + 1);
    }
    out
}

#[test]
fn summarize_pow2_block_carries_matches_reference_ragged() {
    let mut rng = StdRng::seed_from_u64(0xAC);
    let peeled_bits = 3usize;
    let inner_len = 1usize << peeled_bits;
    let outer_len = 5usize;
    let r = random_vec(&mut rng, 7);
    let offset = 0b101101usize;
    let eq_low = EqPolynomial::evals(&r[..peeled_bits]).unwrap();
    let offset_low = offset & (inner_len - 1);

    let blocks: Vec<Vec<F>> = (0..outer_len)
        .map(|_| random_vec(&mut rng, inner_len))
        .collect();
    let carry_terms: Vec<[F; 2]> = blocks
        .iter()
        .map(|block| summarize_pow2_block_carries(&eq_low, offset_low, block))
        .collect::<Result<_, _>>()
        .unwrap();

    let got = combine_pow2_carry_terms(&r, offset, peeled_bits, &carry_terms);
    let expected = reference_pow2_peeled_blocks(&r, offset, &blocks);
    assert_eq!(got, expected);
}

#[test]
fn summarize_pow2_block_carries_matches_reference_high_overflow() {
    let mut rng = StdRng::seed_from_u64(0xAD);
    let peeled_bits = 2usize;
    let inner_len = 1usize << peeled_bits;
    let outer_len = 6usize;
    let r = random_vec(&mut rng, 5);
    let offset = 27usize;
    let eq_low = EqPolynomial::evals(&r[..peeled_bits]).unwrap();
    let offset_low = offset & (inner_len - 1);

    let blocks: Vec<Vec<F>> = (0..outer_len)
        .map(|_| random_vec(&mut rng, inner_len))
        .collect();
    let carry_terms: Vec<[F; 2]> = blocks
        .iter()
        .map(|block| summarize_pow2_block_carries(&eq_low, offset_low, block))
        .collect::<Result<_, _>>()
        .unwrap();

    let got = combine_pow2_carry_terms(&r, offset, peeled_bits, &carry_terms);
    let expected = reference_pow2_peeled_blocks(&r, offset, &blocks);
    assert_eq!(got, expected);
}

fn reference_compact_stride(
    challenges: &[F],
    offset: usize,
    stride: usize,
    lanes: &[F],
    outer: &[F],
) -> F {
    let mut acc = F::zero();
    for (q, &outer_weight) in outer.iter().enumerate() {
        for (lane, &lane_weight) in lanes.iter().enumerate() {
            let address = offset + stride * q + lane;
            acc += outer_weight * lane_weight * eq_eval_at_index(challenges, address);
        }
    }
    acc
}

#[test]
fn compact_stride_matches_direct_small_sweep() {
    let mut rng = StdRng::seed_from_u64(0x00C0_11A7);
    for bits in 1..8usize {
        let challenges = random_vec(&mut rng, bits);
        for stride in [1usize, 3, 5, 8] {
            for outer_len in [1usize, 2, 7] {
                let lanes = random_vec(&mut rng, stride);
                let outer = random_vec(&mut rng, outer_len);
                for offset in [0usize, 1, 7, (1usize << bits).saturating_sub(2)] {
                    let got = eval_compact_stride_eq(
                        &challenges,
                        offset,
                        stride,
                        &lanes,
                        CompactOuterWeights::Dense(&outer),
                    )
                    .unwrap();
                    let expected =
                        reference_compact_stride(&challenges, offset, stride, &lanes, &outer);
                    assert_eq!(
                        got, expected,
                        "bits={bits} stride={stride} outer_len={outer_len} offset={offset}"
                    );
                }
            }
        }
    }
}

#[test]
fn compact_stride_supports_gaps_and_partial_clipping() {
    let mut rng = StdRng::seed_from_u64(0x00C0_11A8);
    let challenges = random_vec(&mut rng, 6);
    let lanes = random_vec(&mut rng, 3);
    let outer = random_vec(&mut rng, 9);
    let got = eval_compact_stride_eq(
        &challenges,
        41,
        5,
        &lanes,
        CompactOuterWeights::Dense(&outer),
    )
    .unwrap();
    let expected = reference_compact_stride(&challenges, 41, 5, &lanes, &outer);
    assert_eq!(got, expected);
}

#[test]
fn compact_stride_boolean_challenges_need_no_inverses() {
    let challenges = [F::zero(), F::one(), F::one(), F::zero()];
    let lanes = [F::from_u64(2), F::from_u64(3), F::from_u64(5)];
    let outer = [F::from_u64(7), F::from_u64(11)];
    let got = eval_compact_stride_eq(
        &challenges,
        3,
        lanes.len(),
        &lanes,
        CompactOuterWeights::Dense(&outer),
    )
    .unwrap();
    let expected = reference_compact_stride(&challenges, 3, lanes.len(), &lanes, &outer);
    assert_eq!(got, expected);
}

#[test]
fn compact_stride_tensor_matches_dense_outer() {
    let mut rng = StdRng::seed_from_u64(0x00C0_11A9);
    let challenges = random_vec(&mut rng, 8);
    let lanes = random_vec(&mut rng, 5);
    let factor0 = random_vec(&mut rng, 3);
    let factor1 = random_vec(&mut rng, 4);
    let dense = (0..factor0.len() * factor1.len())
        .map(|index| factor0[index % factor0.len()] * factor1[index / factor0.len()])
        .collect::<Vec<_>>();
    let factors = [&factor0[..], &factor1[..]];
    let tensor = eval_compact_stride_eq(
        &challenges,
        9,
        lanes.len(),
        &lanes,
        CompactOuterWeights::Tensor(&factors),
    )
    .unwrap();
    let materialized = eval_compact_stride_eq(
        &challenges,
        9,
        lanes.len(),
        &lanes,
        CompactOuterWeights::Dense(&dense),
    )
    .unwrap();
    assert_eq!(tensor, materialized);
}

#[test]
fn compact_stride_rejects_overflow_before_clipping() {
    let lanes = [F::one(), F::one()];
    let outer = [F::one(), F::one()];
    let err = eval_compact_stride_eq(
        &[F::one()],
        usize::MAX,
        lanes.len(),
        &lanes,
        CompactOuterWeights::Dense(&outer),
    )
    .expect_err("overflow must be rejected");
    assert!(matches!(err, AkitaError::InvalidInput(_)));
}

#[test]
fn compact_pair_matches_direct_non_power_of_two_sweep() {
    let mut rng = StdRng::seed_from_u64(0x00C0_11AA);
    for left_bits in 2..8usize {
        for right_bits in 2..8usize {
            let left = random_vec(&mut rng, left_bits);
            let right = random_vec(&mut rng, right_bits);
            for left_stride in [1usize, 2, 3, 5] {
                for right_stride in [1usize, 2, 3, 7] {
                    for len in [1usize, 2, 3, 5, 9] {
                        let left_offset = left_bits;
                        let right_offset = right_bits + 1;
                        let got = eval_compact_pair_eq(
                            &left,
                            left_offset,
                            left_stride,
                            &right,
                            right_offset,
                            right_stride,
                            len,
                        )
                        .unwrap();
                        let expected = (0..len)
                            .map(|index| {
                                eq_eval_at_index(&left, left_offset + left_stride * index)
                                    * eq_eval_at_index(&right, right_offset + right_stride * index)
                            })
                            .sum();
                        assert_eq!(
                            got, expected,
                            "left_bits={left_bits} right_bits={right_bits} left_stride={left_stride} right_stride={right_stride} len={len}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn compact_pair_boolean_challenges_need_no_inverses() {
    let left = [F::zero(), F::one(), F::one(), F::zero(), F::one()];
    let right = [F::one(), F::zero(), F::one(), F::one(), F::zero()];
    let got = eval_compact_pair_eq(&left, 3, 3, &right, 1, 5, 7).unwrap();
    let expected = (0..7)
        .map(|index| {
            eq_eval_at_index(&left, 3 + 3 * index) * eq_eval_at_index(&right, 1 + 5 * index)
        })
        .sum();
    assert_eq!(got, expected);
}

#[allow(clippy::too_many_arguments)]
fn reference_affine_digit_interval(
    challenges: &[F],
    base_offset: usize,
    outer_start: usize,
    live_len: usize,
    outer_stride: usize,
    digits: &[F],
    high: &[F],
    low: &[F],
) -> F {
    (outer_start..outer_start + live_len)
        .flat_map(|outer| {
            digits
                .iter()
                .enumerate()
                .map(move |(digit, &digit_weight)| {
                    high[outer / low.len()]
                        * low[outer % low.len()]
                        * digit_weight
                        * eq_eval_at_index(
                            challenges,
                            base_offset + outer_stride * (outer - outer_start) + digit,
                        )
                })
        })
        .sum()
}

#[test]
fn affine_digit_interval_matches_dense_subwindows_and_partial_rows() {
    let mut rng = StdRng::seed_from_u64(0x00af_f16e);
    for &(low_len, high_len, outer_start, live_len, digits, stride, base) in &[
        (1, 7, 0, 7, 1, 1, 3),
        (4, 4, 1, 11, 3, 5, 9),
        (8, 3, 5, 13, 5, 17, 6),
        (16, 2, 15, 17, 4, 8, 31),
    ] {
        let challenges = random_vec(&mut rng, 12);
        let digit_weights = random_vec(&mut rng, digits);
        let high = random_vec(&mut rng, high_len);
        let low = random_vec(&mut rng, low_len);
        let got = eval_affine_digit_interval(
            &challenges,
            base,
            outer_start,
            live_len,
            stride,
            &digit_weights,
            &high,
            &low,
        )
        .unwrap();
        let expected = reference_affine_digit_interval(
            &challenges,
            base,
            outer_start,
            live_len,
            stride,
            &digit_weights,
            &high,
            &low,
        );
        assert_eq!(got, expected);
    }
}

#[test]
fn affine_digit_interval_handles_boolean_challenges_without_inversion() {
    let challenges = [
        F::zero(),
        F::one(),
        F::one(),
        F::zero(),
        F::one(),
        F::zero(),
        F::one(),
        F::one(),
    ];
    let digits = [F::from_u64(2), F::from_u64(3), F::from_u64(5)];
    let high = [F::from_u64(7), F::from_u64(11), F::from_u64(13)];
    let low = [
        F::from_u64(17),
        F::from_u64(19),
        F::from_u64(23),
        F::from_u64(29),
    ];
    let got = eval_affine_digit_interval(&challenges, 5, 3, 7, 6, &digits, &high, &low).unwrap();
    assert_eq!(
        got,
        reference_affine_digit_interval(&challenges, 5, 3, 7, 6, &digits, &high, &low)
    );
}

#[test]
fn affine_digit_interval_rejects_work_above_cap() {
    let challenges = vec![F::from_u64(2); 20];
    let digits = vec![F::one(); 1 << 14];
    let low = vec![F::one(); 1 << 14];
    let err = eval_affine_digit_interval(
        &challenges,
        0,
        0,
        1 << 14,
        1 << 14,
        &digits,
        &[F::one()],
        &low,
    )
    .unwrap_err();
    assert!(matches!(err, AkitaError::InvalidSize { .. }));
}

#[test]
fn affine_digit_interval_rejects_addresses_outside_eq_domain() {
    let err = eval_affine_digit_interval(
        &[F::from_u64(2); 3],
        7,
        0,
        2,
        2,
        &[F::one()],
        &[F::one()],
        &[F::one(), F::one()],
    )
    .unwrap_err();
    assert!(matches!(err, AkitaError::InvalidSize { .. }));
}
