use super::*;

const REDUCTION_TERMS: usize = TrinomialLimbAccumulator::REDUCTION_TERMS as usize;

fn next(seed: &mut u64) -> u64 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    *seed
}

fn centered(value: i128, prime: u32) -> i32 {
    let p = i128::from(prime);
    let residue = value.rem_euclid(p);
    (if residue > p / 2 {
        residue - p
    } else {
        residue
    }) as i32
}

fn random_coefficients(prime: u32, seed: &mut u64) -> [i32; DEGREE] {
    std::array::from_fn(|_| centered(i128::from(next(seed)), prime))
}

fn random_bits(seed: &mut u64) -> [i32; DEGREE] {
    std::array::from_fn(|_| (next(seed) & 1) as i32)
}

fn packed_bits(coefficients: &[i32; DEGREE]) -> [u64; 11] {
    let mut bits = [0; 11];
    for (index, &coefficient) in coefficients.iter().enumerate() {
        bits[index / 64] |= (coefficient as u64) << (index % 64);
    }
    bits
}

// Independent signed convolution; descending reduction applies
// X^648 = X^324 - 1 before taking any residue modulo the prime.
fn schoolbook(lhs: &[i32; DEGREE], rhs: &[i32; DEGREE], prime: u32) -> [i32; DEGREE] {
    let mut convolution = [0i128; 2 * DEGREE - 1];
    for (i, &a) in lhs.iter().enumerate() {
        for (j, &b) in rhs.iter().enumerate() {
            convolution[i + j] += i128::from(a) * i128::from(b);
        }
    }
    for degree in (DEGREE..convolution.len()).rev() {
        let value = convolution[degree];
        convolution[degree - 324] += value;
        convolution[degree - DEGREE] -= value;
    }
    std::array::from_fn(|i| centered(convolution[i], prime))
}

fn transform(domain: &TrinomialLimbDomain, input: &[i32; DEGREE]) -> TrinomialLimbSlots {
    let mut slots = domain.zero_slots();
    domain.forward_centered(input, &mut slots).unwrap();
    slots
}

fn inverse(domain: &TrinomialLimbDomain, slots: &TrinomialLimbSlots) -> [i32; DEGREE] {
    let mut coefficients = [0; DEGREE];
    domain.inverse_centered(slots, &mut coefficients).unwrap();
    let half = (domain.prime() / 2) as i32;
    assert!(coefficients
        .iter()
        .all(|&value| (-half..=half).contains(&value)));
    coefficients
}

fn assert_product(domain: &TrinomialLimbDomain, lhs: &[i32; DEGREE], rhs: &[i32; DEGREE]) {
    let mut accumulator = TrinomialLimbAccumulator::new(domain);
    accumulator
        .add_product(&transform(domain, lhs), &transform(domain, rhs))
        .unwrap();
    let mut product = domain.zero_slots();
    accumulator.finish(&mut product).unwrap();
    assert_eq!(
        inverse(domain, &product),
        schoolbook(lhs, rhs, domain.prime())
    );
}

#[test]
fn limb_admitted_primes_equal_largest_split_primes_enumeration() {
    let enumerated: [u32; 3] = [26, 27, 28].map(|bits| {
        // Enumerate admissible integers downwards; primality uses independent
        // trial division, so neither the supplied values nor domain setup is
        // used to decide which candidate wins.
        let largest = ((1u32 << bits) - 2) / 1944;
        (1..=largest)
            .rev()
            .map(|k| 1944 * k + 1)
            .find(|&candidate| {
                (2..)
                    .take_while(|d| d * d <= candidate)
                    .all(|d| candidate % d != 0)
            })
            .unwrap()
    });
    assert_eq!(enumerated, TrinomialLimbDomain::ADMITTED_PRIMES);
    for prime in enumerated {
        assert_eq!(TrinomialLimbDomain::new(prime).unwrap().prime(), prime);
    }
}

#[test]
fn limb_centered_random_roundtrip() {
    let mut seed = 0x4a89_01d2_69cb_30ef;
    for prime in TrinomialLimbDomain::ADMITTED_PRIMES {
        let domain = TrinomialLimbDomain::new(prime).unwrap();
        for _ in 0..8 {
            let input = random_coefficients(prime, &mut seed);
            assert_eq!(inverse(&domain, &transform(&domain, &input)), input);
        }
    }
}

#[test]
fn limb_random_centered_products_match_schoolbook() {
    let mut seed = 0x80e1_276b_dcd3_9051;
    for prime in TrinomialLimbDomain::ADMITTED_PRIMES {
        let domain = TrinomialLimbDomain::new(prime).unwrap();
        for _ in 0..3 {
            assert_product(
                &domain,
                &random_coefficients(prime, &mut seed),
                &random_coefficients(prime, &mut seed),
            );
        }
    }
}

#[test]
fn limb_random_bit_products_match_schoolbook() {
    let mut seed = 0x19af_d42e_bb81_5329;
    for prime in TrinomialLimbDomain::ADMITTED_PRIMES {
        let domain = TrinomialLimbDomain::new(prime).unwrap();
        for _ in 0..3 {
            let lhs = random_bits(&mut seed);
            let rhs = random_bits(&mut seed);
            let mut left = domain.zero_slots();
            let mut right = domain.zero_slots();
            domain
                .forward_interleaved_bits(&packed_bits(&lhs), &mut left)
                .unwrap();
            domain
                .forward_interleaved_bits(&packed_bits(&rhs), &mut right)
                .unwrap();
            let mut accumulator = TrinomialLimbAccumulator::new(&domain);
            accumulator.add_product(&left, &right).unwrap();
            accumulator.finish(&mut left).unwrap();
            let signed = |input: [i32; DEGREE]| {
                std::array::from_fn(|i| {
                    if (i / 4) % 2 == 0 {
                        input[i]
                    } else {
                        -input[i]
                    }
                })
            };
            assert_eq!(
                inverse(&domain, &left),
                schoolbook(&signed(lhs), &signed(rhs), prime)
            );
        }
    }
}

#[test]
fn limb_extremal_coefficient_products_and_roundtrip() {
    for prime in TrinomialLimbDomain::ADMITTED_PRIMES {
        let domain = TrinomialLimbDomain::new(prime).unwrap();
        let half = (prime / 2) as i32;
        for input in [[0; DEGREE], [1; DEGREE], [half; DEGREE], [-half; DEGREE]] {
            assert_eq!(inverse(&domain, &transform(&domain, &input)), input);
            assert_product(&domain, &input, &input);
        }
        assert_product(&domain, &[half; DEGREE], &[-half; DEGREE]);
    }
}

#[test]
fn limb_single_ones_match_schoolbook() {
    for prime in TrinomialLimbDomain::ADMITTED_PRIMES {
        let domain = TrinomialLimbDomain::new(prime).unwrap();
        for left in [0, 323, 324, 647] {
            let mut lhs = [0; DEGREE];
            lhs[left] = 1;
            for right in [0, 323, 324, 647] {
                let mut rhs = [0; DEGREE];
                rhs[right] = 1;
                assert_product(&domain, &lhs, &rhs);
            }
        }
    }
}

fn checked_shadow_butterfly(
    domain: &TrinomialLimbDomain,
    values: &mut [i64; DEGREE],
    split: &arithmetic::Split,
    inverse: bool,
    input_bound: i64,
    output_bound: i64,
) {
    let p = domain.prime();
    let mut actual = values.map(|v| i32::try_from(v).unwrap());
    for t in 0..split.stride {
        let offset = (split.start + t) * LANES;
        let step = split.stride * LANES;
        for lane in 0..LANES {
            let a = values[offset + lane];
            let mut b = values[offset + step + lane];
            let mut c = values[offset + 2 * step + lane];
            assert!([a, b, c].iter().all(|v| v.abs() <= input_bound));
            if !inverse {
                b = constant_product(p, i32::try_from(b).unwrap(), split.r[lane]);
                c = constant_product(p, i32::try_from(c).unwrap(), split.r2[lane]);
                assert!(b.abs() <= arithmetic::multiply_bound(p, input_bound));
                assert!(c.abs() <= arithmetic::multiply_bound(p, input_bound));
            }
            let difference_input = b.checked_sub(c).unwrap();
            i32::try_from(difference_input).unwrap();
            let difference = constant_product(
                p,
                difference_input as i32,
                if inverse {
                    domain.inverse_omega
                } else {
                    domain.omega
                },
            );
            assert!(difference.abs() <= arithmetic::multiply_bound(p, difference_input.abs()));
            // Check both intermediate additions, not just the final sums.
            let ab = a.checked_add(b).unwrap();
            let ac = a.checked_sub(c).unwrap();
            let am_b = a.checked_sub(b).unwrap();
            let o0 = ab.checked_add(c).unwrap();
            let o1 = ac.checked_add(difference).unwrap();
            let o2 = am_b.checked_sub(difference).unwrap();
            for value in [ab, ac, am_b, difference_input, difference, o0, o1, o2] {
                i32::try_from(value).unwrap();
                assert!(
                    value.abs() <= output_bound,
                    "prime={p}, stride={}, value={value}, bound={output_bound}",
                    split.stride
                );
            }
            if inverse {
                values[offset + lane] = i64::from(centered(i128::from(o0), p));
                values[offset + step + lane] = i64::from(centered(
                    i128::from(constant_product(p, o1 as i32, split.ri[lane])),
                    p,
                ));
                values[offset + 2 * step + lane] = i64::from(centered(
                    i128::from(constant_product(p, o2 as i32, split.ri2[lane])),
                    p,
                ));
            } else {
                values[offset + lane] = o0;
                values[offset + step + lane] = o1;
                values[offset + 2 * step + lane] = o2;
            }
        }
    }
    arithmetic::butterfly(
        domain.arithmetic,
        domain.omega,
        domain.inverse_omega,
        &mut actual,
        split,
        inverse,
    );
    assert_eq!(actual.map(i64::from), *values);
}

#[test]
fn limb_lazy_forward_levels_match_checked_shadow() {
    let mut seed = 0xbfc9_0512_487e_9013;
    for prime in TrinomialLimbDomain::ADMITTED_PRIMES {
        let domain = TrinomialLimbDomain::new(prime).unwrap();
        let half = (prime / 2) as i32;
        let bounds = arithmetic::forward_bounds(prime);
        let inputs = [
            [1; DEGREE],
            [half; DEGREE],
            [-half; DEGREE],
            std::array::from_fn(|i| if i % 2 == 0 { half } else { -half }),
            std::array::from_fn(|i| {
                if domain.evaluation[0][i / PIECE] >= 0 {
                    half
                } else {
                    -half
                }
            }),
            random_coefficients(prime, &mut seed),
        ];
        for input in inputs {
            let mut shadow = [0i64; DEGREE];
            for t in 0..PIECE {
                for lane in 0..LANES {
                    let dot: i128 = (0..LANES)
                        .map(|j| {
                            i128::from(input[t + PIECE * j])
                                * i128::from(domain.evaluation[lane][j])
                        })
                        .sum();
                    assert!(dot.abs() <= 8 * i128::from(half).pow(2));
                    shadow[t * LANES + lane] = i64::from(centered(dot, prime));
                }
            }
            assert!(shadow.iter().all(|v| v.abs() <= bounds[0]));
            for (level, stride) in [27, 9, 3, 1].into_iter().enumerate() {
                for split in domain.splits.iter().filter(|s| s.stride == stride) {
                    checked_shadow_butterfly(
                        &domain,
                        &mut shadow,
                        split,
                        false,
                        bounds[level],
                        bounds[level + 1],
                    );
                }
                assert!(shadow.iter().all(|v| v.abs() <= bounds[level + 1]));
            }
            let normalized = shadow
                .map(|v| i32::try_from(constant_product(prime, v as i32, domain.one)).unwrap());
            assert!(normalized
                .iter()
                .all(|&v| i64::from(v).abs() <= arithmetic::multiply_bound(prime, bounds[4])));
            assert!(normalized
                .iter()
                .all(|&v| i64::from(v).abs() < i64::from(prime)));
            let mut actual = domain.zero_slots();
            domain
                .forward_centered_backend(&input, &mut actual, false)
                .unwrap();
            assert_eq!(actual.values, normalized);
        }
        // Each level is also fed its own extremal admitted input bound. This
        // catches an overflow even if structured polynomial inputs cancel.
        for (level, stride) in [27, 9, 3, 1].into_iter().enumerate() {
            for sign in [-1i64, 1] {
                let mut shadow = [0; DEGREE];
                for split in domain.splits.iter().filter(|s| s.stride == stride) {
                    for t in 0..split.stride {
                        let offset = (split.start + t) * LANES;
                        let step = split.stride * LANES;
                        for lane in 0..LANES {
                            shadow[offset + lane] = sign * bounds[level];
                            // Choose extremal signs against the actual
                            // rounded twiddle products to make all three
                            // terms in the first output reinforce.
                            let choose = |w| {
                                let product = constant_product(prime, bounds[level] as i32, w);
                                sign * if product < 0 {
                                    -bounds[level]
                                } else {
                                    bounds[level]
                                }
                            };
                            shadow[offset + step + lane] = choose(split.r[lane]);
                            shadow[offset + 2 * step + lane] = choose(split.r2[lane]);
                        }
                    }
                    checked_shadow_butterfly(
                        &domain,
                        &mut shadow,
                        split,
                        false,
                        bounds[level],
                        bounds[level + 1],
                    );
                }
            }
        }
    }
}

#[test]
fn limb_lazy_inverse_intermediates_match_checked_shadow() {
    let mut seed = 0x315a_621d_45a0_ae8c;
    for prime in TrinomialLimbDomain::ADMITTED_PRIMES {
        let domain = TrinomialLimbDomain::new(prime).unwrap();
        let half = i64::from(prime / 2);
        let p = i64::from(prime);
        let intermediate_bound =
            (3 * half).max(2 * half + arithmetic::multiply_bound(prime, 2 * half));
        for initial in [
            [half; DEGREE],
            [-half; DEGREE],
            std::array::from_fn(|i| if i % 2 == 0 { half } else { -half }),
            random_coefficients(prime, &mut seed).map(i64::from),
        ] {
            let mut shadow = initial;
            for split in domain.splits.iter().rev() {
                checked_shadow_butterfly(
                    &domain,
                    &mut shadow,
                    split,
                    true,
                    half,
                    intermediate_bound,
                );
                assert!(shadow.iter().all(|v| v.abs() <= half));
            }
            assert!(intermediate_bound < i64::from(i32::MAX));
            assert!(intermediate_bound <= 2 * p);
            let mut actual = initial.map(|v| v as i32);
            domain.transform(&mut actual, true, false, false);
            assert_eq!(actual.map(i64::from), shadow);
        }
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[test]
fn limb_neon_matches_scalar_directly() {
    let mut seed = 0x238a_903d_b861_072f;
    for prime in TrinomialLimbDomain::ADMITTED_PRIMES {
        let domain = TrinomialLimbDomain::new(prime).unwrap();
        let half = (prime / 2) as i32;
        for input in [
            random_coefficients(prime, &mut seed),
            [half; DEGREE],
            [-half; DEGREE],
            std::array::from_fn(|i| if i % 2 == 0 { half } else { -half }),
        ] {
            let mut scalar = domain.zero_slots();
            let mut neon = domain.zero_slots();
            domain
                .forward_centered_backend(&input, &mut scalar, false)
                .unwrap();
            domain
                .forward_centered_backend(&input, &mut neon, true)
                .unwrap();
            assert_eq!(scalar.values, neon.values);
            let mut scalar_output = [0; DEGREE];
            let mut neon_output = [0; DEGREE];
            domain
                .inverse_centered_backend(&scalar, &mut scalar_output, false)
                .unwrap();
            domain
                .inverse_centered_backend(&neon, &mut neon_output, true)
                .unwrap();
            assert_eq!(scalar_output, input);
            assert_eq!(neon_output, input);
            domain
                .inverse_centered_backend(&scalar, &mut neon_output, true)
                .unwrap();
            assert_eq!(neon_output, input);
            domain
                .inverse_centered_backend(&neon, &mut scalar_output, false)
                .unwrap();
            assert_eq!(scalar_output, input);
        }
        for input in [random_bits(&mut seed), [0; DEGREE], [1; DEGREE]] {
            let bits = packed_bits(&input);
            let mut scalar = domain.zero_slots();
            let mut neon = domain.zero_slots();
            domain
                .forward_interleaved_bits_backend(&bits, &mut scalar, false)
                .unwrap();
            domain
                .forward_interleaved_bits_backend(&bits, &mut neon, true)
                .unwrap();
            assert_eq!(scalar.values, neon.values);
            let signed = std::array::from_fn(|i| {
                if (i / 4) % 2 == 0 {
                    input[i]
                } else {
                    -input[i]
                }
            });
            assert_eq!(inverse(&domain, &scalar), signed);
            assert_eq!(inverse(&domain, &neon), signed);
        }
    }
}

#[test]
fn limb_equivalent_representatives_invert_identically() {
    let mut seed = 0x9893_4974_d974_0d26;
    for prime in TrinomialLimbDomain::ADMITTED_PRIMES {
        let domain = TrinomialLimbDomain::new(prime).unwrap();
        let slots = transform(&domain, &random_coefficients(prime, &mut seed));
        let mut equivalent = slots.clone();
        for (index, value) in equivalent.values.iter_mut().enumerate() {
            *value = centered(i128::from(*value), prime)
                + if index % 2 == 0 {
                    prime as i32
                } else {
                    -(prime as i32)
                };
        }
        assert_eq!(inverse(&domain, &slots), inverse(&domain, &equivalent));
    }
}

#[test]
fn limb_wrong_lengths_leave_destinations_unchanged() {
    for prime in TrinomialLimbDomain::ADMITTED_PRIMES {
        let domain = TrinomialLimbDomain::new(prime).unwrap();
        let mut slots = transform(&domain, &[1; DEGREE]);
        let original = slots.clone();
        for len in [0, 1, 10, 12, 100] {
            assert_eq!(
                domain.forward_interleaved_bits(&vec![0; len], &mut slots),
                Err(TrinomialError::CoefficientLength {
                    expected: 11,
                    actual: len
                })
            );
        }
        for len in [0, 1, 647, 649, 1000] {
            assert_eq!(
                domain.forward_centered(&vec![0; len], &mut slots),
                Err(TrinomialError::CoefficientLength {
                    expected: DEGREE,
                    actual: len
                })
            );
            let mut output = vec![17; len];
            assert_eq!(
                domain.inverse_centered(&slots, &mut output),
                Err(TrinomialError::CoefficientLength {
                    expected: DEGREE,
                    actual: len
                })
            );
            assert_eq!(output, vec![17; len]);
        }
        assert_eq!(canonical_slots(&slots), canonical_slots(&original));
        assert_eq!(slots.values, original.values);
        assert_eq!(slots.prime, original.prime);
    }
}

#[test]
fn limb_invalid_primes_tail_bits_and_coefficients_are_rejected() {
    for prime in [
        0,
        1,
        2,
        1944,
        1945,
        67_091_328,
        268_433_354,
        1 << 28,
        u32::MAX,
    ] {
        assert!(matches!(
            TrinomialLimbDomain::new(prime),
            Err(TrinomialError::LimbInput { .. })
        ));
    }
    for prime in TrinomialLimbDomain::ADMITTED_PRIMES {
        let domain = TrinomialLimbDomain::new(prime).unwrap();
        let mut slots = transform(&domain, &[1; DEGREE]);
        let original = slots.clone();
        for bit in 8..64 {
            let mut bits = [0; 11];
            bits[10] = 1 << bit;
            assert!(matches!(
                domain.forward_interleaved_bits(&bits, &mut slots),
                Err(TrinomialError::LimbInput { .. })
            ));
        }
        let half = (prime / 2) as i32;
        for position in [0, 323, 324, 647] {
            for value in [half + 1, -half - 1, i32::MAX, i32::MIN] {
                let mut input = [0; DEGREE];
                input[position] = value;
                assert!(matches!(
                    domain.forward_centered(&input, &mut slots),
                    Err(TrinomialError::LimbInput { .. })
                ));
            }
        }
        assert_eq!(canonical_slots(&slots), canonical_slots(&original));
        assert_eq!(slots.values, original.values);
        assert_eq!(slots.prime, original.prime);
    }
}

#[test]
fn limb_cross_prime_slots_are_rejected_without_mutation() {
    for prime in TrinomialLimbDomain::ADMITTED_PRIMES {
        let domain = TrinomialLimbDomain::new(prime).unwrap();
        let slots = transform(&domain, &[1; DEGREE]);
        for other in TrinomialLimbDomain::ADMITTED_PRIMES
            .into_iter()
            .filter(|&p| p != prime)
        {
            let mut wrong = TrinomialLimbDomain::new(other).unwrap().zero_slots();
            let original = wrong.clone();
            assert!(matches!(
                domain.forward_interleaved_bits(&[0; 11], &mut wrong),
                Err(TrinomialError::LimbInput { .. })
            ));
            assert!(matches!(
                domain.forward_centered(&[0; DEGREE], &mut wrong),
                Err(TrinomialError::LimbInput { .. })
            ));
            let mut output = [17; DEGREE];
            assert!(matches!(
                domain.inverse_centered(&wrong, &mut output),
                Err(TrinomialError::LimbInput { .. })
            ));
            assert_eq!(output, [17; DEGREE]);
            let mut accumulator = TrinomialLimbAccumulator::new(&domain);
            assert!(matches!(
                accumulator.add_product(&wrong, &slots),
                Err(TrinomialError::LimbInput { .. })
            ));
            assert!(matches!(
                accumulator.add_product(&slots, &wrong),
                Err(TrinomialError::LimbInput { .. })
            ));
            accumulator.add_product(&slots, &slots).unwrap();
            assert!(matches!(
                accumulator.finish(&mut wrong),
                Err(TrinomialError::LimbInput { .. })
            ));
            assert_eq!(canonical_slots(&wrong), canonical_slots(&original));
            assert_eq!(wrong.values, original.values);
            assert_eq!(wrong.prime, original.prime);
            let mut result = domain.zero_slots();
            accumulator.finish(&mut result).unwrap();
            assert_eq!(
                inverse(&domain, &result),
                schoolbook(&[1; DEGREE], &[1; DEGREE], prime)
            );
        }
    }
}

#[test]
fn limb_accumulator_finish_clears_sum() {
    for prime in TrinomialLimbDomain::ADMITTED_PRIMES {
        let domain = TrinomialLimbDomain::new(prime).unwrap();
        let mut output = transform(&domain, &[1; DEGREE]);
        let mut accumulator = TrinomialLimbAccumulator::new(&domain);
        accumulator.add_product(&output, &output).unwrap();
        accumulator.finish(&mut output).unwrap();
        assert_eq!(
            inverse(&domain, &output),
            schoolbook(&[1; DEGREE], &[1; DEGREE], prime)
        );
        accumulator.finish(&mut output).unwrap();
        assert_eq!(inverse(&domain, &output), [0; DEGREE]);
    }
}

#[test]
fn limb_gather_matches_independent_one_bit_loop() {
    let mut seed = 0x759d_5472_d0eb_6cb1;
    for bits in std::iter::once([u64::MAX; 11])
        .chain((0..128).map(|_| std::array::from_fn(|_| next(&mut seed))))
    {
        let mut bits = bits;
        bits[10] &= 255;
        let expected: [u8; PIECE] = std::array::from_fn(|t| {
            (0..LANES).fold(0, |byte, j| {
                let position = t + PIECE * j;
                byte | (((bits[position / 64] >> (position % 64)) & 1) as u8) << j
            })
        });
        let mut actual = [0; PIECE];
        gather::indices(&bits, &mut actual);
        assert_eq!(actual, expected);
    }
    for position in 0..DEGREE {
        let mut bits = [0; 11];
        bits[position / 64] = 1 << (position % 64);
        let mut actual = [0; PIECE];
        gather::indices(&bits, &mut actual);
        let mut expected = [0; PIECE];
        expected[position % PIECE] = 1 << (position / PIECE);
        assert_eq!(actual, expected);
    }
}

fn rounded_high(a: i32, b: i32) -> i32 {
    ((i128::from(a) * i128::from(b) + (1i128 << 30)) >> 31)
        .clamp(i128::from(i32::MIN), i128::from(i32::MAX)) as i32
}

fn constant_product(prime: u32, a: i32, w: arithmetic::Twiddle) -> i64 {
    let quotient = rounded_high(a, w.quotient);
    (i128::from(a) * i128::from(w.value) - i128::from(quotient) * i128::from(prime)) as i64
}

#[test]
fn limb_sqrdmulh_extremes_and_saturating_corner_are_exact() {
    for a in [i32::MIN, i32::MIN + 1, -1, 0, 1, i32::MAX - 1, i32::MAX] {
        for b in [
            i32::MIN,
            i32::MIN + 1,
            -(1 << 30),
            -1,
            0,
            1,
            1 << 30,
            i32::MAX,
        ] {
            assert_eq!(
                arithmetic::sqrdmulh(a, b),
                rounded_high(a, b),
                "a={a}, b={b}"
            );
        }
    }
    assert_eq!(arithmetic::sqrdmulh(i32::MIN, i32::MIN), i32::MAX);
}

#[test]
fn limb_constant_multiply_extremes_obey_exact_bound() {
    let mut seed = 0x4af5_1f78_4361_5ec3;
    for prime in TrinomialLimbDomain::ADMITTED_PRIMES {
        let arithmetic = arithmetic::Arithmetic { prime };
        let half = (prime / 2) as i32;
        let bounds = arithmetic::forward_bounds(prime);
        let mut operands = vec![i32::MIN, i32::MIN + 1, i32::MAX, 0, 1, -1, half, -half];
        for bound in bounds {
            operands.extend([bound as i32, -(bound as i32)]);
        }
        let mut weights = vec![-half, -half + 1, -1, 0, 1, half - 1, half];
        weights.extend((0..128).map(|_| centered(i128::from(next(&mut seed)), prime)));
        for weight in weights {
            let twiddle = arithmetic::Twiddle::new(arithmetic, weight);
            let numerator = i128::from(weight) * (1i128 << 31);
            assert_eq!(
                i128::from(twiddle.quotient),
                (numerator + i128::from(prime / 2)).div_euclid(i128::from(prime))
            );
            // Centered twiddles cannot reach the SQRDMULH saturation input.
            assert!(i64::from(twiddle.quotient).abs() <= 1 << 30);
            for &operand in &operands {
                let exact = constant_product(prime, operand, twiddle);
                assert_eq!(i64::from(arithmetic.mul(operand, twiddle)), exact);
                assert!(exact.abs() <= arithmetic::multiply_bound(prime, i64::from(operand).abs()));
                assert_eq!(
                    centered(i128::from(exact), prime),
                    centered(i128::from(operand) * i128::from(weight), prime)
                );
            }
        }
        // Identity multiplication at half the modulus approaches p/2 to
        // within one, exercising rather than merely accepting its bound.
        let one = arithmetic::Twiddle::new(arithmetic, 1);
        assert!(i64::from(arithmetic.mul(half, one)).abs() >= i64::from(half) - 1);
    }
}

#[test]
fn limb_accumulator_extremal_slots_cross_reduction_boundaries() {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    let backends = [false, true];
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    let backends = [false];
    for prime in TrinomialLimbDomain::ADMITTED_PRIMES {
        let domain = TrinomialLimbDomain::new(prime).unwrap();
        let largest = (prime - 1) as i32;
        let mut lhs = domain.zero_slots();
        let mut rhs = domain.zero_slots();
        lhs.values = std::array::from_fn(|i| if i % 2 == 0 { largest } else { -largest });
        rhs.values = std::array::from_fn(|i| if i % 4 < 2 { largest } else { -largest });
        for neon in backends {
            let mut accumulator = TrinomialLimbAccumulator::new(&domain);
            for terms in 1..=8192 {
                accumulator.add_product_backend(&lhs, &rhs, neon).unwrap();
                if [1, 2, 127, 128, 129, 255, 256, 257, 4096, 8192].contains(&terms) {
                    assert_eq!(usize::from(accumulator.terms), terms % REDUCTION_TERMS);
                    let bound = REDUCTION_TERMS as i128 * i128::from(largest).pow(2)
                        + i128::from(prime / 2);
                    assert!(bound < i128::from(i64::MAX));
                    assert!(accumulator
                        .sums
                        .iter()
                        .all(|&sum| i128::from(sum).abs() <= bound));
                    let expected: [i32; DEGREE] = std::array::from_fn(|i| {
                        centered(
                            i128::from(lhs.values[i]) * i128::from(rhs.values[i]) * terms as i128,
                            prime,
                        )
                    });
                    let mut snapshot = accumulator.clone();
                    let mut actual = domain.zero_slots();
                    snapshot.finish(&mut actual).unwrap();
                    assert_eq!(
                        actual.values, expected,
                        "prime={prime}, terms={terms}, neon={neon}"
                    );
                }
            }
        }
    }
}

#[test]
fn limb_accumulator_worst_carry_has_room_for_full_batch() {
    for prime in TrinomialLimbDomain::ADMITTED_PRIMES {
        let domain = TrinomialLimbDomain::new(prime).unwrap();
        let largest = (prime - 1) as i32;
        let half = i64::from(prime / 2);
        let mut lhs = domain.zero_slots();
        let mut rhs = domain.zero_slots();
        lhs.values.fill(largest);
        for sign in [-1i64, 1] {
            rhs.values.fill((sign * i64::from(largest)) as i32);
            let mut accumulator = TrinomialLimbAccumulator::new(&domain);
            accumulator.sums.fill(sign * half);
            for _ in 0..REDUCTION_TERMS - 1 {
                accumulator.add_product_backend(&lhs, &rhs, false).unwrap();
            }
            let before = sign * (half + (REDUCTION_TERMS - 1) as i64 * i64::from(largest).pow(2));
            assert!(accumulator.sums.iter().all(|&sum| sum == before));
            let worst = i128::from(sign)
                * (i128::from(half) + REDUCTION_TERMS as i128 * i128::from(largest).pow(2));
            assert!(worst.abs() <= i128::from(i64::MAX));
            accumulator.add_product_backend(&lhs, &rhs, false).unwrap();
            assert_eq!(accumulator.terms, 0);
            let mut result = domain.zero_slots();
            accumulator.finish(&mut result).unwrap();
            assert!(result
                .values
                .iter()
                .all(|&value| value == centered(worst, prime)));
        }
    }
}

#[test]
fn limb_column_accumulation_matches_scalar_and_i128_oracle() {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    let backends = [false, true];
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    let backends = [false];
    let mut seed = 0x9a50_37a2_a6b1_8e02;
    for prime in TrinomialLimbDomain::ADMITTED_PRIMES {
        let domain = TrinomialLimbDomain::new(prime).unwrap();
        let half = (prime / 2) as i32;
        let rows = [
            transform(&domain, &random_coefficients(prime, &mut seed)),
            transform(&domain, &[half; DEGREE]),
            transform(&domain, &[-half; DEGREE]),
            transform(&domain, &[1; DEGREE]),
        ];
        let source = transform(&domain, &random_bits(&mut seed));
        for rank in 1..=4 {
            let refs: Vec<_> = rows[..rank].iter().collect();
            for neon in backends {
                let mut column = vec![TrinomialLimbAccumulator::new(&domain); rank];
                let mut scalar = column.clone();
                for _ in 0..REDUCTION_TERMS + 1 {
                    for (accumulator, row) in column.iter_mut().zip(&refs) {
                        accumulator.add_product_backend(row, &source, neon).unwrap();
                    }
                    for (accumulator, row) in scalar.iter_mut().zip(&refs) {
                        accumulator
                            .add_product_backend(row, &source, false)
                            .unwrap();
                    }
                }
                for (index, (a, b)) in column.iter_mut().zip(&mut scalar).enumerate() {
                    assert_eq!(a.sums, b.sums);
                    assert_eq!(a.terms, b.terms);
                    let mut actual = domain.zero_slots();
                    let mut expected = domain.zero_slots();
                    a.finish(&mut actual).unwrap();
                    b.finish(&mut expected).unwrap();
                    assert_eq!(canonical_slots(&actual), canonical_slots(&expected));
                    for lane in 0..DEGREE {
                        let oracle = centered(
                            i128::from(rows[index].values[lane])
                                * i128::from(source.values[lane])
                                * (REDUCTION_TERMS + 1) as i128,
                            prime,
                        );
                        assert_eq!(actual.values[lane], oracle);
                    }
                }
            }
        }
    }
}

#[test]
fn limb_column_rows_reject_wrong_tags_without_mutation() {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    let backends = [false, true];
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    let backends = [false];
    for prime in TrinomialLimbDomain::ADMITTED_PRIMES {
        let domain = TrinomialLimbDomain::new(prime).unwrap();
        let slots = transform(&domain, &[1; DEGREE]);
        let other = TrinomialLimbDomain::ADMITTED_PRIMES
            .into_iter()
            .find(|&p| p != prime)
            .unwrap();
        let wrong = TrinomialLimbDomain::new(other).unwrap().zero_slots();
        for rank in 1..=4 {
            for neon in backends {
                let mut accumulators = vec![TrinomialLimbAccumulator::new(&domain); rank];
                for accumulator in &mut accumulators {
                    accumulator
                        .add_product_backend(&slots, &slots, neon)
                        .unwrap();
                }
                let original = accumulators.clone();
                for accumulator in &mut accumulators {
                    for (lhs, rhs) in [(&wrong, &slots), (&slots, &wrong), (&wrong, &wrong)] {
                        assert!(matches!(
                            accumulator.add_product_backend(lhs, rhs, neon),
                            Err(TrinomialError::LimbInput { .. })
                        ));
                    }
                }
                for (actual, expected) in accumulators.iter().zip(&original) {
                    assert_eq!(actual.sums, expected.sums);
                    assert_eq!(actual.terms, expected.terms);
                }
                let other_domain = TrinomialLimbDomain::new(other).unwrap();
                let mut mixed = TrinomialLimbAccumulator::new(&other_domain);
                mixed.add_product_backend(&wrong, &wrong, neon).unwrap();
                let before = mixed.clone();
                assert!(matches!(
                    mixed.add_product_backend(&slots, &slots, neon),
                    Err(TrinomialError::LimbInput { .. })
                ));
                assert_eq!(mixed.sums, before.sums);
                assert_eq!(mixed.terms, before.terms);
            }
        }
    }
}

#[test]
fn limb_every_level_extremal_signs_match_checked_shadow() {
    for prime in TrinomialLimbDomain::ADMITTED_PRIMES {
        let domain = TrinomialLimbDomain::new(prime).unwrap();
        let bounds = arithmetic::forward_bounds(prime);
        let half = i64::from(prime / 2);
        let inverse_bound = (3 * half).max(2 * half + arithmetic::multiply_bound(prime, 2 * half));
        // Exercise all eight endpoint sign patterns independently at every
        // split of every retained level. Unlike index-parity alternation,
        // opposite b/c signs attain the 2h inverse difference bound. Both
        // reinforcing and subtractive forward outputs are checked.
        for inverse in [false, true] {
            for (level, stride) in [27, 9, 3, 1].into_iter().enumerate() {
                let bound = if inverse { half } else { bounds[level] };
                let output_bound = if inverse {
                    inverse_bound
                } else {
                    bounds[level + 1]
                };
                for signs in 0..8 {
                    for split in domain.splits.iter().filter(|s| s.stride == stride) {
                        let mut shadow = [0; DEGREE];
                        for t in 0..split.stride {
                            for branch in 0..3 {
                                let value = if signs & (1 << branch) == 0 {
                                    bound
                                } else {
                                    -bound
                                };
                                let offset = (split.start + t + branch * split.stride) * LANES;
                                shadow[offset..offset + LANES].fill(value);
                            }
                        }
                        checked_shadow_butterfly(
                            &domain,
                            &mut shadow,
                            split,
                            inverse,
                            bound,
                            output_bound,
                        );
                        if inverse {
                            assert!(shadow.iter().all(|v| v.abs() <= half));
                        }
                    }
                }
            }
        }
        // The eight-term final interpolation has the same dot bound as
        // initial evaluation. Choose signs aligned and anti-aligned to each
        // actual weight to exercise its largest possible magnitude.
        for weights in domain.evaluation.iter().chain(&domain.interpolation) {
            for sign in [-1i32, 1] {
                let input =
                    weights.map(|w| sign * if w < 0 { -(half as i32) } else { half as i32 });
                let dot: i128 = input
                    .iter()
                    .zip(weights)
                    .map(|(&a, &w)| i128::from(a) * i128::from(w))
                    .sum();
                assert!(dot.abs() <= 8 * i128::from(half).pow(2));
                i64::try_from(dot).unwrap();
                assert_eq!(domain.arithmetic.dot(&input, weights), centered(dot, prime));
            }
        }
    }
}
