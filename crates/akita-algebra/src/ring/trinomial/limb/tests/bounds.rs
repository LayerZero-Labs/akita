use super::*;

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

fn rounded_high(a: i32, b: i32) -> i32 {
    ((i128::from(a) * i128::from(b) + (1i128 << 30)) >> 31)
        .clamp(i128::from(i32::MIN), i128::from(i32::MAX)) as i32
}

fn constant_product(prime: u32, a: i32, w: arithmetic::Twiddle) -> i64 {
    let quotient = rounded_high(a, w.quotient);
    (i128::from(a) * i128::from(w.value) - i128::from(quotient) * i128::from(prime)) as i64
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
