use super::*;
use crate::eq_poly::EqPolynomial;
use crate::RandomSampling;
use akita_field::Fp64;
use rand::rngs::StdRng;
use rand::RngCore;
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

#[test]
fn offset_eq_window_precomputed_high_table_matches_scalar_eq() {
    let mut rng = StdRng::seed_from_u64(0x11165);
    // Widths in (16, 32] exercise the bounded high table: low is capped at 16
    // bits and the high remainder (1..=16 bits) is materialized, so `eval`
    // becomes two lookups. Sample rather than sweep the full domain.
    for n in [17usize, 20, 24, 28, 32] {
        let challenges: Vec<F> = (0..n).map(|_| F::random(&mut rng)).collect();
        let window = OffsetEqWindow::new(&challenges).unwrap();
        // The high remainder is bounded, so the high table is materialized.
        assert!(
            window.eq_high.is_some(),
            "n={n} should precompute high table"
        );
        let domain = 1u128 << n;
        let mut probes: Vec<usize> = vec![
            0,
            1,
            (1 << OFFSET_EQ_LOW_BITS_CAP) - 1,
            1 << OFFSET_EQ_LOW_BITS_CAP,
            (1 << OFFSET_EQ_LOW_BITS_CAP) + 1,
            (domain - 1) as usize,
        ];
        for _ in 0..64 {
            probes.push((rng.next_u64() as u128 % domain) as usize);
        }
        // Out-of-domain probes must return zero like the scalar oracle.
        if domain < u128::from(u64::MAX) {
            probes.push(domain as usize);
            probes.push((domain + 5) as usize);
        }
        for index in probes {
            assert_eq!(
                window.eval(index),
                eq_eval_at_index(&challenges, index),
                "n={n} index={index}"
            );
        }
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

// ---- Reference / oracle evaluators (test-only, relocated from offset_eq.rs) ----

/// Dense or factored outer weights for a compact-stride contraction.
enum CompactOuterWeights<'a, F> {
    /// Materialized outer weights in compact-index order.
    Dense(&'a [F]),
    /// Mixed-radix tensor factors, with the first factor as the fastest axis.
    Tensor(&'a [&'a [F]]),
}

impl<F: FieldCore> CompactOuterWeights<'_, F> {
    fn len(&self) -> Result<usize, AkitaError> {
        match self {
            Self::Dense(weights) => Ok(weights.len()),
            Self::Tensor(factors) => {
                if factors.is_empty() || factors.iter().any(|factor| factor.is_empty()) {
                    return Err(AkitaError::InvalidInput(
                        "compact-stride tensor factors must be non-empty".into(),
                    ));
                }
                factors.iter().try_fold(1usize, |len, factor| {
                    len.checked_mul(factor.len()).ok_or_else(|| {
                        AkitaError::InvalidInput(
                            "compact-stride tensor outer length overflow".into(),
                        )
                    })
                })
            }
        }
    }

    fn value(&self, mut index: usize) -> Result<F, AkitaError> {
        match self {
            Self::Dense(weights) => weights
                .get(index)
                .copied()
                .ok_or_else(|| AkitaError::InvalidInput("outer weight index out of range".into())),
            Self::Tensor(factors) => {
                let mut value = F::one();
                for factor in *factors {
                    let digit = index % factor.len();
                    index /= factor.len();
                    value *= factor[digit];
                }
                if index != 0 {
                    return Err(AkitaError::InvalidInput(
                        "tensor outer weight index out of range".into(),
                    ));
                }
                Ok(value)
            }
        }
    }
}

/// Evaluate one exact compact-stride equality contraction.
///
/// This computes
///
/// ```text
/// sum_{q < Q} outer[q] * sum_{lane < L}
///     lanes[lane] * eq(r, offset + stride*q + lane).
/// ```
///
/// The equality trie is visited only over live address intervals.
/// The traversal uses multiplication and addition only, so Boolean challenges
/// need no special case and no inversions.
/// Tensor outer weights are read from their factors and are never materialized.
///
/// # Errors
///
/// Returns an error for malformed geometry, arithmetic overflow, a domain that
/// cannot be represented by `usize`, or work above [`MAX_COMPACT_STRIDE_TERMS`].
fn eval_compact_stride_eq<F: FieldCore>(
    challenges: &[F],
    offset: usize,
    stride: usize,
    lanes: &[F],
    outer: CompactOuterWeights<'_, F>,
) -> Result<F, AkitaError> {
    if stride == 0 || lanes.is_empty() || lanes.len() > stride {
        return Err(AkitaError::InvalidInput(
            "compact-stride geometry requires 0 < lanes <= stride".into(),
        ));
    }
    if challenges.len() >= usize::BITS as usize {
        return Err(AkitaError::InvalidSize {
            expected: usize::BITS as usize - 1,
            actual: challenges.len(),
        });
    }
    let outer_len = outer.len()?;
    if outer_len == 0 {
        return Ok(F::zero());
    }
    let terms = outer_len
        .checked_mul(lanes.len())
        .ok_or_else(|| AkitaError::InvalidInput("compact-stride term count overflow".into()))?;
    if terms > MAX_COMPACT_STRIDE_TERMS {
        return Err(AkitaError::InvalidSize {
            expected: MAX_COMPACT_STRIDE_TERMS,
            actual: terms,
        });
    }
    let last_outer = outer_len - 1;
    let last = stride
        .checked_mul(last_outer)
        .and_then(|base| base.checked_add(lanes.len() - 1))
        .and_then(|local| offset.checked_add(local))
        .ok_or_else(|| AkitaError::InvalidInput("compact-stride address overflow".into()))?;
    let domain_len = 1usize << challenges.len();
    if offset >= domain_len {
        return Ok(F::zero());
    }
    let clipped_last = last.min(domain_len - 1);
    let mut acc = F::zero();
    if lanes.len() == stride {
        visit_eq_interval(
            challenges,
            0,
            domain_len,
            F::one(),
            offset,
            clipped_last + 1,
            &mut |address, eq_weight| {
                let local = address - offset;
                let q = local / stride;
                let lane = local % stride;
                let outer_weight = outer.value(q)?;
                acc += outer_weight * lanes[lane] * eq_weight;
                Ok(())
            },
        )?;
    } else {
        for q in 0..outer_len {
            let start = offset
                .checked_add(stride.checked_mul(q).ok_or_else(|| {
                    AkitaError::InvalidInput("compact-stride address overflow".into())
                })?)
                .ok_or_else(|| {
                    AkitaError::InvalidInput("compact-stride address overflow".into())
                })?;
            if start >= domain_len {
                break;
            }
            let end = start
                .checked_add(lanes.len())
                .ok_or_else(|| AkitaError::InvalidInput("compact-stride address overflow".into()))?
                .min(domain_len);
            let outer_weight = outer.value(q)?;
            visit_eq_interval(
                challenges,
                0,
                domain_len,
                F::one(),
                start,
                end,
                &mut |address, eq_weight| {
                    acc += outer_weight * lanes[address - start] * eq_weight;
                    Ok(())
                },
            )?;
        }
    }
    Ok(acc)
}
#[allow(clippy::too_many_arguments)]
fn visit_eq_interval<F, Visit>(
    challenges: &[F],
    node_start: usize,
    node_len: usize,
    node_weight: F,
    live_start: usize,
    live_end: usize,
    visit: &mut Visit,
) -> Result<(), AkitaError>
where
    F: FieldCore,
    Visit: FnMut(usize, F) -> Result<(), AkitaError>,
{
    let node_end = node_start
        .checked_add(node_len)
        .ok_or_else(|| AkitaError::InvalidInput("equality trie range overflow".into()))?;
    if node_end <= live_start || node_start >= live_end {
        return Ok(());
    }
    if node_len == 1 {
        return visit(node_start, node_weight);
    }
    let half = node_len / 2;
    let bit = half.trailing_zeros() as usize;
    let challenge = *challenges
        .get(bit)
        .ok_or_else(|| AkitaError::InvalidInput("equality trie bit out of range".into()))?;
    visit_eq_interval(
        challenges,
        node_start,
        half,
        node_weight * (F::one() - challenge),
        live_start,
        live_end,
        visit,
    )?;
    visit_eq_interval(
        challenges,
        node_start + half,
        half,
        node_weight * challenge,
        live_start,
        live_end,
        visit,
    )
}

/// Sparse/pruned partial multilinear evaluation of a single materialized
/// factor over the contiguous global interval `[offset, offset + factor.len())`.
///
/// Computes:
///
/// ```text
/// scale · Σ_{z=0}^{factor.len()-1}  eq(x_challenges, offset + z) · factor[z]
/// ```
///
/// where indices `offset + z ≥ 2^n` (with `n = x_challenges.len()`) fall
/// outside the equality domain and contribute zero.
///
/// This places the values in **global** index coordinates and runs the
/// standard little-endian multilinear binding fold, pruning every parent node
/// whose whole subtree is outside the live interval. Each live parent costs
/// exactly one field multiplication, so the
/// total is `Σ_k (⌊hi/2^{k+1}⌋ − ⌊lo/2^{k+1}⌋ + 1)` multiplications plus one
/// final `scale` product.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidInput`] if `offset + factor.len()` overflows
/// `usize`.
fn eval_offset_eq_interval<F: FieldCore>(
    x_challenges: &[F],
    offset: usize,
    scale: F,
    factor: &[F],
) -> Result<F, AkitaError> {
    let n = x_challenges.len();
    if factor.is_empty() {
        return Ok(F::zero());
    }

    // Indices at or beyond `2^n` are outside the equality domain (weight 0).
    let in_domain = n < usize::BITS as usize;
    if in_domain && offset >= (1usize << n) {
        return Ok(F::zero());
    }

    let last = offset
        .checked_add(factor.len() - 1)
        .ok_or_else(|| AkitaError::InvalidInput("offset-eq interval overflow".to_string()))?;

    let mut lo = offset;
    let mut hi = if in_domain {
        core::cmp::min(last, (1usize << n) - 1)
    } else {
        last
    };

    // Active values in global coordinates: `a[i - lo] = factor[i - offset]`.
    let mut a: Vec<F> = factor[..=(hi - lo)].to_vec();

    for &r in x_challenges.iter() {
        let new_lo = lo >> 1;
        let new_hi = hi >> 1;
        let mut next = Vec::with_capacity(new_hi - new_lo + 1);
        for p in new_lo..=new_hi {
            let left = 2 * p;
            let right = left + 1;
            let has_left = left >= lo && left <= hi;
            let has_right = right >= lo && right <= hi;
            let val = if has_left && has_right {
                let x0 = a[left - lo];
                let x1 = a[right - lo];
                x0 + r * (x1 - x0)
            } else if has_left {
                let x0 = a[left - lo];
                x0 - r * x0
            } else {
                let x1 = a[right - lo];
                r * x1
            };
            next.push(val);
        }
        a = next;
        lo = new_lo;
        hi = new_hi;
    }

    debug_assert_eq!(a.len(), 1);
    Ok(scale * a[0])
}

/// Summarize one power-of-two inner block `values[u]` into the two carry cases
/// induced by adding `offset_low + u`, where `offset_low < values.len()`.
///
/// `eq_low` must be the equality table on the low `log2(values.len())` bits.
///
/// # Errors
///
/// Returns an error if `values` is not power-of-two sized, if `eq_low` has the
/// wrong length, or if `offset_low` does not lie inside the peeled block.
fn summarize_pow2_block_carries<F: FieldCore>(
    eq_low: &[F],
    offset_low: usize,
    values: &[F],
) -> Result<[F; 2], AkitaError> {
    if !values.len().is_power_of_two() {
        return Err(AkitaError::InvalidInput(
            "peeled inner block length must be a power of two".to_string(),
        ));
    }
    if eq_low.len() != values.len() {
        return Err(AkitaError::InvalidSize {
            expected: values.len(),
            actual: eq_low.len(),
        });
    }
    if offset_low >= values.len() {
        return Err(AkitaError::InvalidInput(
            "low offset must lie inside the peeled block".to_string(),
        ));
    }

    let inner_bits = values.len().trailing_zeros() as usize;
    let inner_mask = values.len() - 1;
    let mut out = [F::zero(), F::zero()];

    for (u, &value) in values.iter().enumerate() {
        let sum = offset_low + u;
        let carry = sum >> inner_bits;
        debug_assert!(
            carry < 2,
            "sum of two peeled indices must carry at most one bit"
        );
        let low_idx = sum & inner_mask;
        out[carry] += value * eq_low[low_idx];
    }

    Ok(out)
}

#[test]
fn affine_digit_interval_matches_reference() {
    let mut rng = StdRng::seed_from_u64(0xF00D_1234);
    // (low_len Q, high_len, outer_start, live_len, digits, stride, base, challenge_bits)
    // Small cases hit the fallback; large cases (rows >= 8) hit the bucketed path.
    let cases: &[AffineDigitCase] = &[
        (1, 7, 0, 7, 1, 1, 3, 12),
        (4, 4, 1, 11, 3, 5, 9, 12),
        (8, 3, 5, 13, 5, 17, 6, 12),
        (16, 2, 15, 17, 4, 8, 31, 12),
        (4, 64, 0, 256, 3, 3, 0, 20),
        (4, 64, 1, 200, 3, 3, 7, 20),
        (8, 32, 3, 250, 5, 5, 6, 20),
        (8, 20, 0, 160, 16, 16, 0, 22),
        (16, 16, 5, 240, 8, 8, 11, 22),
        (2, 100, 0, 200, 4, 4, 3, 20),
        (32, 16, 10, 480, 7, 7, 40, 22),
        (4, 64, 2, 254, 32, 32, 9, 24),
    ];
    for &(low_len, high_len, outer_start, live_len, digits, stride, base, bits) in cases {
        let challenges = random_vec(&mut rng, bits);
        let digit_weights = random_vec(&mut rng, digits);
        let high = random_vec(&mut rng, high_len);
        let low = random_vec(&mut rng, low_len);
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
        let tag = (
            low_len,
            high_len,
            outer_start,
            live_len,
            digits,
            stride,
            base,
        );
        assert_eq!(got, expected, "canonical evaluator mismatch {tag:?}");
    }
}

#[test]
fn affine_digit_interval_matches_boolean_challenges() {
    // Boolean challenges, enough rows to trigger the bucketed path; verifies the
    // no-inversion split-eq path agrees with the dense reference bit-for-bit.
    let mut rng = StdRng::seed_from_u64(0xB001_F00D);
    let challenges: Vec<F> = (0..20)
        .map(|_| {
            if rng.next_u32() & 1 == 1 {
                F::one()
            } else {
                F::zero()
            }
        })
        .collect();
    let digit_weights = random_vec(&mut rng, 5);
    let high = random_vec(&mut rng, 64);
    let low = random_vec(&mut rng, 4);
    for &(outer_start, live_len, base) in &[(0usize, 256usize, 0usize), (3, 250, 6), (1, 200, 9)] {
        let expected = reference_affine_digit_interval(
            &challenges,
            base,
            outer_start,
            live_len,
            5,
            &digit_weights,
            &high,
            &low,
        );
        let got = eval_affine_digit_interval(
            &challenges,
            base,
            outer_start,
            live_len,
            5,
            &digit_weights,
            &high,
            &low,
        )
        .unwrap();
        assert_eq!(
            got, expected,
            "boolean canonical mismatch {outer_start} {live_len} {base}"
        );
    }
}

// Micro-benchmark the canonical kernel across digit counts at balanced folds.
// Run with:  cargo test -p akita-algebra --release affine_digit_interval_bench -- --ignored --nocapture
#[test]
#[ignore]
fn affine_digit_interval_bench() {
    use std::time::Instant;
    let mut rng = StdRng::seed_from_u64(0xBEEF_CAFE);
    let q = 512usize; // fold_low
    let h = 512usize; // fold_high
    let live_len = q * h; // B = Q*H
    let bits = 26usize;
    let iters = 20;
    eprintln!("\n Q={q} H={h} B={live_len} challenge_bits={bits}  (median of {iters} runs)");
    eprintln!(" delta | canonical (ms)");
    for &delta in &[4usize, 8, 16, 32, 64] {
        let stride = delta;
        let challenges = random_vec(&mut rng, bits);
        let digit_weights = random_vec(&mut rng, delta);
        let high = random_vec(&mut rng, h);
        let low = random_vec(&mut rng, q);

        let mut best = f64::INFINITY;
        for _ in 0..iters {
            let start = Instant::now();
            std::hint::black_box(
                eval_affine_digit_interval(
                    &challenges,
                    0,
                    0,
                    live_len,
                    stride,
                    &digit_weights,
                    &high,
                    &low,
                )
                .unwrap(),
            );
            best = best.min(start.elapsed().as_secs_f64() * 1e3);
        }
        eprintln!(" {delta:>5} | {best:>14.3}");
    }
}

/// (low_len, high_len, outer_start, live_len, digits, stride, base, challenge_bits)
type AffineDigitCase = (usize, usize, usize, usize, usize, usize, usize, usize);

fn geometric_digit_vec(base: F, ratio: F, len: usize) -> Vec<F> {
    let mut out = Vec::with_capacity(len);
    let mut cur = base;
    for _ in 0..len {
        out.push(cur);
        cur *= ratio;
    }
    out
}

#[test]
fn affine_digit_interval_matches_geometric_digits() {
    // Geometric digit weights exercise the prefix-scan low summary (step 2).
    // Cases with digits <= low_len activate it; digits > low_len fall back —
    // both must match the dense test-only reference.
    let mut rng = StdRng::seed_from_u64(0x6E03_E17A);
    let ratio = F::from_u64(7);
    let cases: &[AffineDigitCase] = &[
        (4, 64, 0, 256, 3, 3, 0, 20),    // digits<=Q -> prefix path
        (4, 64, 1, 200, 3, 3, 7, 20),    // partial first row + prefix path
        (8, 32, 3, 250, 5, 5, 6, 20),    // digits<=Q, partial rows
        (16, 16, 5, 240, 8, 8, 11, 22),  // digits<=Q
        (8, 20, 0, 160, 16, 16, 0, 22),  // digits>Q -> low falls back
        (2, 100, 0, 200, 4, 4, 3, 20),   // digits>Q -> low falls back
        (32, 16, 0, 512, 32, 32, 0, 24), // digits==Q, full geometric window edge
        (4, 64, 2, 254, 3, 3, 9, 24),
    ];
    for &(low_len, high_len, outer_start, live_len, digits, stride, base, bits) in cases {
        let challenges = random_vec(&mut rng, bits);
        let mut base_w = F::random(&mut rng);
        if base_w == F::zero() {
            base_w = F::one();
        }
        let digit_weights = geometric_digit_vec(base_w, ratio, digits);
        let high = random_vec(&mut rng, high_len);
        let low = random_vec(&mut rng, low_len);
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
        let tag = (
            low_len,
            high_len,
            outer_start,
            live_len,
            digits,
            stride,
            base,
        );
        assert_eq!(got, expected, "canonical evaluator mismatch (geom) {tag:?}");
    }
}

// Geometric-digit microbenchmark: captures step (1) (high bucketing) and step
// (2) (prefix low summary). The random-digit benchmark above exercises step (1)
// without the geometric prefix path.
// Run: cargo test -p akita-algebra --release affine_digit_interval_bench_geometric -- --ignored --nocapture
#[test]
#[ignore]
fn affine_digit_interval_bench_geometric() {
    use std::time::Instant;
    let mut rng = StdRng::seed_from_u64(0x9E0_BEEF1);
    let q = 512usize;
    let h = 512usize;
    let live_len = q * h;
    let bits = 26usize;
    let iters = 20;
    let ratio = F::from_u64(7);
    eprintln!("\n GEOMETRIC digits  Q={q} H={h} B={live_len}  (median of {iters} runs)");
    eprintln!(" delta | canonical (ms)");
    for &delta in &[4usize, 8, 16, 32, 64, 128, 256] {
        let stride = delta;
        let challenges = random_vec(&mut rng, bits);
        let digit_weights = geometric_digit_vec(F::from_u64(3), ratio, delta);
        let high = random_vec(&mut rng, h);
        let low = random_vec(&mut rng, q);
        let mut best = f64::INFINITY;
        for _ in 0..iters {
            let start = Instant::now();
            std::hint::black_box(
                eval_affine_digit_interval(
                    &challenges,
                    0,
                    0,
                    live_len,
                    stride,
                    &digit_weights,
                    &high,
                    &low,
                )
                .unwrap(),
            );
            best = best.min(start.elapsed().as_secs_f64() * 1e3);
        }
        eprintln!(" {delta:>5} | {best:>14.3}");
    }
}
