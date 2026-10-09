use super::*;

const DEGREE: usize = 648;

fn random_coefficients(prime: u16, seed: &mut u64) -> [u16; DEGREE] {
    std::array::from_fn(|_| {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        (*seed % u64::from(prime)) as u16
    })
}

fn packed_bits(coefficients: &[u16; DEGREE]) -> [u64; 11] {
    let mut bits = [0u64; 11];
    for (index, &coefficient) in coefficients.iter().enumerate() {
        bits[index / 64] |= u64::from(coefficient) << (index % 64);
    }
    bits
}

// Independent polynomial convolution, followed by descending reduction using
// X^648 = X^324 - 1. Even the largest convolution coefficient fits in u64.
fn schoolbook(lhs: &[u16; DEGREE], rhs: &[u16; DEGREE], prime: u16) -> [u16; DEGREE] {
    let modulus = u64::from(prime);
    let mut convolution = [0u64; 2 * DEGREE - 1];
    for (i, &a) in lhs.iter().enumerate() {
        for (j, &b) in rhs.iter().enumerate() {
            convolution[i + j] += u64::from(a) * u64::from(b);
        }
    }
    for coefficient in &mut convolution {
        *coefficient %= modulus;
    }
    for degree in (DEGREE..convolution.len()).rev() {
        let value = convolution[degree];
        convolution[degree - 324] = (convolution[degree - 324] + value) % modulus;
        convolution[degree - DEGREE] = (convolution[degree - DEGREE] + modulus - value) % modulus;
    }
    std::array::from_fn(|index| convolution[index] as u16)
}

// Sum before reducing, with wide integer coefficients. Repeating the same
// extremal pair makes thousands of accumulator terms inexpensive to check.
fn wide_sum_oracle(
    lhs: &[u16; DEGREE],
    rhs: &[u16; DEGREE],
    terms: usize,
    prime: u16,
) -> [u16; DEGREE] {
    let modulus = u128::from(prime);
    let mut convolution = [0u128; 2 * DEGREE - 1];
    for (i, &a) in lhs.iter().enumerate() {
        for (j, &b) in rhs.iter().enumerate() {
            convolution[i + j] += u128::from(a) * u128::from(b) * terms as u128;
        }
    }
    for coefficient in &mut convolution {
        *coefficient %= modulus;
    }
    for degree in (DEGREE..convolution.len()).rev() {
        let value = convolution[degree];
        convolution[degree - 324] = (convolution[degree - 324] + value) % modulus;
        convolution[degree - DEGREE] = (convolution[degree - DEGREE] + modulus - value) % modulus;
    }
    std::array::from_fn(|index| convolution[index] as u16)
}

fn transform(domain: &TrinomialLimbDomain, coefficients: &[u16; DEGREE]) -> TrinomialLimbSlots {
    let mut slots = domain.zero_slots();
    domain.forward_canonical(coefficients, &mut slots).unwrap();
    slots
}

fn inverse(domain: &TrinomialLimbDomain, slots: &TrinomialLimbSlots) -> [u16; DEGREE] {
    let mut coefficients = [0; DEGREE];
    domain.inverse(slots, &mut coefficients).unwrap();
    assert!(coefficients.iter().all(|&value| value < domain.prime()));
    coefficients
}

fn assert_product(domain: &TrinomialLimbDomain, lhs: &[u16; DEGREE], rhs: &[u16; DEGREE]) {
    let lhs_slots = transform(domain, lhs);
    let rhs_slots = transform(domain, rhs);
    let mut accumulator = TrinomialLimbAccumulator::new(domain);
    accumulator.add_product(&lhs_slots, &rhs_slots).unwrap();
    let mut product = domain.zero_slots();
    accumulator.finish(&mut product).unwrap();
    assert_eq!(
        inverse(domain, &product),
        schoolbook(lhs, rhs, domain.prime()),
        "prime {}",
        domain.prime()
    );
}

#[test]
fn limb_admitted_primes_equal_complete_enumeration() {
    let enumerated: Vec<u16> = (2u32..1 << 15)
        .filter(|candidate| (candidate - 1) % 1944 == 0)
        .filter(|&candidate| {
            (2..)
                .take_while(|divisor| divisor * divisor <= candidate)
                .all(|divisor| candidate % divisor != 0)
        })
        .map(|prime| prime as u16)
        .collect();
    assert_eq!(enumerated, TrinomialLimbDomain::ADMITTED_PRIMES);
    for prime in enumerated {
        assert_eq!(TrinomialLimbDomain::new(prime).unwrap().prime(), prime);
    }
}

#[test]
fn limb_canonical_roundtrip_is_exact() {
    let mut seed = 0x4a89_01d2_69cb_30ef;
    for prime in TrinomialLimbDomain::ADMITTED_PRIMES {
        let domain = TrinomialLimbDomain::new(prime).unwrap();
        for _ in 0..8 {
            let coefficients = random_coefficients(prime, &mut seed);
            assert_eq!(
                inverse(&domain, &transform(&domain, &coefficients)),
                coefficients
            );
        }
        for coefficients in [[0; DEGREE], [1; DEGREE], [prime - 1; DEGREE]] {
            assert_eq!(
                inverse(&domain, &transform(&domain, &coefficients)),
                coefficients
            );
        }
    }
}

#[test]
fn limb_random_canonical_products_match_schoolbook() {
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
            let lhs = random_coefficients(2, &mut seed);
            let rhs = random_coefficients(2, &mut seed);
            let mut lhs_slots = domain.zero_slots();
            let mut rhs_slots = domain.zero_slots();
            domain
                .forward_bits(&packed_bits(&lhs), &mut lhs_slots)
                .unwrap();
            domain
                .forward_bits(&packed_bits(&rhs), &mut rhs_slots)
                .unwrap();
            let mut accumulator = TrinomialLimbAccumulator::new(&domain);
            accumulator.add_product(&lhs_slots, &rhs_slots).unwrap();
            let mut product = domain.zero_slots();
            accumulator.finish(&mut product).unwrap();
            assert_eq!(inverse(&domain, &product), schoolbook(&lhs, &rhs, prime));
        }
    }
}

#[test]
fn limb_bit_transform_agrees_with_canonical_transform() {
    let mut seed = 0xd2b8_5c16_4e09_7fab;
    for prime in TrinomialLimbDomain::ADMITTED_PRIMES {
        let domain = TrinomialLimbDomain::new(prime).unwrap();
        for _ in 0..12 {
            let coefficients = random_coefficients(2, &mut seed);
            let canonical_slots = transform(&domain, &coefficients);
            let mut bit_slots = domain.zero_slots();
            domain
                .forward_bits(&packed_bits(&coefficients), &mut bit_slots)
                .unwrap();
            assert_eq!(
                inverse(&domain, &bit_slots),
                inverse(&domain, &canonical_slots)
            );
            assert_eq!(inverse(&domain, &bit_slots), coefficients);
        }
    }
}

#[test]
fn limb_structured_products_match_schoolbook() {
    for prime in TrinomialLimbDomain::ADMITTED_PRIMES {
        let domain = TrinomialLimbDomain::new(prime).unwrap();
        let cases = [[0; DEGREE], [1; DEGREE], [prime - 1; DEGREE]];
        for lhs in &cases {
            for rhs in &cases {
                assert_product(&domain, lhs, rhs);
            }
        }
        for position in [0, 1, 80, 81, 323, 324, 567, 647] {
            let mut monomial = [0; DEGREE];
            monomial[position] = 1;
            assert_product(&domain, &monomial, &monomial);
            assert_product(&domain, &monomial, &[prime - 1; DEGREE]);
            let mut slots = domain.zero_slots();
            domain
                .forward_bits(&packed_bits(&monomial), &mut slots)
                .unwrap();
            assert_eq!(inverse(&domain, &slots), monomial);
        }
        for coefficients in [[0; DEGREE], [1; DEGREE]] {
            let mut slots = domain.zero_slots();
            domain
                .forward_bits(&packed_bits(&coefficients), &mut slots)
                .unwrap();
            assert_eq!(inverse(&domain, &slots), coefficients);
        }
    }
}

#[test]
fn limb_accumulator_extremal_sums_match_wide_oracle() {
    for prime in TrinomialLimbDomain::ADMITTED_PRIMES {
        let domain = TrinomialLimbDomain::new(prime).unwrap();
        let lhs = [prime - 1; DEGREE];
        let lhs_slots = transform(&domain, &lhs);
        for rhs in [[prime - 1; DEGREE], [1; DEGREE]] {
            let rhs_slots = transform(&domain, &rhs);
            for terms in [1, 2, 4096, 8192] {
                let mut accumulator = TrinomialLimbAccumulator::new(&domain);
                for _ in 0..terms {
                    accumulator.add_product(&lhs_slots, &rhs_slots).unwrap();
                }
                let mut sum = domain.zero_slots();
                accumulator.finish(&mut sum).unwrap();
                assert_eq!(
                    inverse(&domain, &sum),
                    wide_sum_oracle(&lhs, &rhs, terms, prime),
                    "prime {prime}, terms {terms}"
                );
            }
        }
    }
}

#[test]
fn limb_accumulator_finish_resets_and_empty_sum_is_zero() {
    for prime in TrinomialLimbDomain::ADMITTED_PRIMES {
        let domain = TrinomialLimbDomain::new(prime).unwrap();
        let slots = transform(&domain, &[prime - 1; DEGREE]);
        let mut accumulator = TrinomialLimbAccumulator::new(&domain);
        let mut sum = domain.zero_slots();
        accumulator.finish(&mut sum).unwrap();
        assert_eq!(inverse(&domain, &sum), [0; DEGREE]);
        accumulator.add_product(&slots, &slots).unwrap();
        accumulator.finish(&mut sum).unwrap();
        assert_eq!(
            inverse(&domain, &sum),
            schoolbook(&[prime - 1; DEGREE], &[prime - 1; DEGREE], prime)
        );
        accumulator.finish(&mut sum).unwrap();
        assert_eq!(inverse(&domain, &sum), [0; DEGREE]);
    }
}

#[test]
fn limb_malformed_inputs_are_rejected() {
    for prime in [0, 1, 2, 3, 3888, 3890, 32749, 32768, u16::MAX] {
        assert!(matches!(
            TrinomialLimbDomain::new(prime),
            Err(TrinomialError::LimbInput { .. })
        ));
    }
    for prime in TrinomialLimbDomain::ADMITTED_PRIMES {
        let domain = TrinomialLimbDomain::new(prime).unwrap();
        let mut slots = transform(&domain, &[prime - 1; DEGREE]);
        let original_slots = slots.clone();
        for len in [0, 1, 10, 12, 100] {
            assert_eq!(
                domain.forward_bits(&vec![0; len], &mut slots),
                Err(TrinomialError::CoefficientLength {
                    expected: 11,
                    actual: len
                })
            );
        }
        for bit in 8..64 {
            let mut bits = [0u64; 11];
            bits[10] = 1u64 << bit;
            assert!(matches!(
                domain.forward_bits(&bits, &mut slots),
                Err(TrinomialError::LimbInput { .. })
            ));
        }
        for len in [0, 1, 647, 649, 1000] {
            assert_eq!(
                domain.forward_canonical(&vec![0; len], &mut slots),
                Err(TrinomialError::CoefficientLength {
                    expected: DEGREE,
                    actual: len
                })
            );
            let mut output = vec![17; len];
            assert_eq!(
                domain.inverse(&slots, &mut output),
                Err(TrinomialError::CoefficientLength {
                    expected: DEGREE,
                    actual: len
                })
            );
            assert_eq!(output, vec![17; len]);
        }
        for position in [0, 323, 324, 647] {
            for value in [prime, prime + 1, u16::MAX] {
                let mut coefficients = [0; DEGREE];
                coefficients[position] = value;
                assert!(matches!(
                    domain.forward_canonical(&coefficients, &mut slots),
                    Err(TrinomialError::LimbInput { .. })
                ));
            }
        }
        assert_eq!(slots, original_slots);
    }
}

#[test]
fn limb_cross_prime_slots_are_rejected() {
    for prime in TrinomialLimbDomain::ADMITTED_PRIMES {
        let domain = TrinomialLimbDomain::new(prime).unwrap();
        let slots = transform(&domain, &[1; DEGREE]);
        for other_prime in TrinomialLimbDomain::ADMITTED_PRIMES {
            if other_prime == prime {
                continue;
            }
            let other = TrinomialLimbDomain::new(other_prime).unwrap();
            let mut wrong_slots = other.zero_slots();
            assert!(matches!(
                domain.forward_bits(&[0; 11], &mut wrong_slots),
                Err(TrinomialError::LimbInput { .. })
            ));
            assert!(matches!(
                domain.forward_canonical(&[0; DEGREE], &mut wrong_slots),
                Err(TrinomialError::LimbInput { .. })
            ));
            assert!(matches!(
                other.inverse(&slots, &mut [0; DEGREE]),
                Err(TrinomialError::LimbInput { .. })
            ));
            let mut accumulator = TrinomialLimbAccumulator::new(&domain);
            assert!(matches!(
                accumulator.add_product(&wrong_slots, &slots),
                Err(TrinomialError::LimbInput { .. })
            ));
            assert!(matches!(
                accumulator.add_product(&slots, &wrong_slots),
                Err(TrinomialError::LimbInput { .. })
            ));
            accumulator.add_product(&slots, &slots).unwrap();
            assert!(matches!(
                accumulator.finish(&mut wrong_slots),
                Err(TrinomialError::LimbInput { .. })
            ));
            let mut output = domain.zero_slots();
            accumulator.finish(&mut output).unwrap();
            assert_eq!(
                inverse(&domain, &output),
                schoolbook(&[1; DEGREE], &[1; DEGREE], prime)
            );
        }
    }
}

// Compute R^-1 independently from Montgomery arithmetic, using Fermat's
// theorem and ordinary wide modular multiplication in this test oracle.
fn inverse_radix(prime: u16) -> i64 {
    let modulus = u64::from(prime);
    let mut power = 65536 % modulus;
    let mut exponent = modulus - 2;
    let mut result = 1;
    while exponent != 0 {
        if exponent & 1 != 0 {
            result = result * power % modulus;
        }
        power = power * power % modulus;
        exponent >>= 1;
    }
    result as i64
}

#[test]
fn limb_accumulator_extremal_slots_cross_every_reduction_boundary() {
    #[cfg(target_arch = "aarch64")]
    let backends = [false, true];
    #[cfg(not(target_arch = "aarch64"))]
    let backends = [false];
    for prime in TrinomialLimbDomain::ADMITTED_PRIMES {
        let domain = TrinomialLimbDomain::new(prime).unwrap();
        let half = (prime / 2) as i16;
        let mut lhs = domain.zero_slots();
        let mut rhs = domain.zero_slots();
        lhs.values = std::array::from_fn(|index| if index % 2 == 0 { half } else { -half });
        rhs.values = std::array::from_fn(|index| if index % 4 < 2 { half } else { -half });
        let modulus = i64::from(prime);
        let radix_inverse = inverse_radix(prime);
        for neon in backends {
            let mut accumulator = TrinomialLimbAccumulator::new(&domain);
            for terms in 1..=8192 {
                accumulator.add_product_backend(&lhs, &rhs, neon).unwrap();
                if [1, 2, 14, 15, 16, 29, 30, 31, 4096, 8192].contains(&terms) {
                    assert_eq!(usize::from(accumulator.terms), terms % 15);
                    let bound = 15 * i64::from(half).pow(2) + i64::from(half) * 65536;
                    assert!(accumulator
                        .sums
                        .iter()
                        .all(|&sum| i64::from(sum).abs() <= bound));
                    let expected: [i16; DEGREE] = std::array::from_fn(|index| {
                        let product = i64::from(lhs.values[index]) * i64::from(rhs.values[index]);
                        let residue = (product * terms as i64 * radix_inverse).rem_euclid(modulus);
                        if residue > i64::from(half) {
                            (residue - modulus) as i16
                        } else {
                            residue as i16
                        }
                    });
                    let mut snapshot = accumulator.clone();
                    let mut actual = domain.zero_slots();
                    snapshot.finish(&mut actual).unwrap();
                    assert_eq!(
                        actual.values, expected,
                        "prime {prime}, terms {terms}, neon {neon}"
                    );
                }
            }
        }
    }
}

#[test]
fn limb_accumulator_accepts_worst_carried_residue() {
    for prime in TrinomialLimbDomain::ADMITTED_PRIMES {
        let domain = TrinomialLimbDomain::new(prime).unwrap();
        let half = (prime / 2) as i16;
        let mut lhs = domain.zero_slots();
        let mut rhs = domain.zero_slots();
        lhs.values.fill(half);
        let modulus = i64::from(prime);
        for sign in [-1i16, 1] {
            rhs.values.fill(sign * half);
            let mut accumulator = TrinomialLimbAccumulator::new(&domain);
            // This is exactly the valid carry invariant after internal
            // reduction, at its largest positive or negative magnitude.
            accumulator.sums.fill(i32::from(sign * half) * 65536);
            for _ in 0..14 {
                accumulator.add_product_backend(&lhs, &rhs, false).unwrap();
            }
            let before_last =
                i64::from(sign) * (i64::from(half) * 65536 + 14 * i64::from(half).pow(2));
            assert!(accumulator
                .sums
                .iter()
                .all(|&sum| i64::from(sum) == before_last));
            let worst = i64::from(sign) * (i64::from(half) * 65536 + 15 * i64::from(half).pow(2));
            assert!(worst.abs() <= i64::from(i32::MAX));
            accumulator.add_product_backend(&lhs, &rhs, false).unwrap();
            assert_eq!(accumulator.terms, 0);
            let mut output = domain.zero_slots();
            accumulator.finish(&mut output).unwrap();
            let residue = (worst * inverse_radix(prime)).rem_euclid(modulus);
            let expected = if residue > i64::from(half) {
                (residue - modulus) as i16
            } else {
                residue as i16
            };
            assert!(output.values.iter().all(|&value| value == expected));
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[test]
fn limb_neon_backends_match_scalar_directly() {
    let mut seed = 0x238a_903d_b861_072f;
    for prime in TrinomialLimbDomain::ADMITTED_PRIMES {
        let domain = TrinomialLimbDomain::new(prime).unwrap();
        for coefficients in [random_coefficients(prime, &mut seed), [prime - 1; DEGREE]] {
            let mut scalar = domain.zero_slots();
            let mut neon = domain.zero_slots();
            domain
                .forward_canonical_backend(&coefficients, &mut scalar, false)
                .unwrap();
            domain
                .forward_canonical_backend(&coefficients, &mut neon, true)
                .unwrap();
            let mut scalar_output = [0; DEGREE];
            let mut neon_output = [0; DEGREE];
            domain
                .inverse_backend(&scalar, &mut scalar_output, false)
                .unwrap();
            domain
                .inverse_backend(&neon, &mut neon_output, true)
                .unwrap();
            assert_eq!(scalar_output, coefficients);
            assert_eq!(neon_output, scalar_output);
            domain
                .inverse_backend(&neon, &mut neon_output, false)
                .unwrap();
            assert_eq!(neon_output, scalar_output);
            domain
                .inverse_backend(&scalar, &mut neon_output, true)
                .unwrap();
            assert_eq!(neon_output, scalar_output);
            let mut scalar_sum = TrinomialLimbAccumulator::new(&domain);
            let mut neon_sum = TrinomialLimbAccumulator::new(&domain);
            for _ in 0..8192 {
                scalar_sum
                    .add_product_backend(&scalar, &scalar, false)
                    .unwrap();
                neon_sum.add_product_backend(&neon, &neon, true).unwrap();
            }
            scalar_sum.finish(&mut scalar).unwrap();
            neon_sum.finish(&mut neon).unwrap();
            assert_eq!(inverse(&domain, &scalar), inverse(&domain, &neon));
        }
        for coefficients in [random_coefficients(2, &mut seed), [0; DEGREE], [1; DEGREE]] {
            let bits = packed_bits(&coefficients);
            let mut scalar = domain.zero_slots();
            let mut neon = domain.zero_slots();
            domain
                .forward_bits_backend(&bits, &mut scalar, false)
                .unwrap();
            domain.forward_bits_backend(&bits, &mut neon, true).unwrap();
            assert_eq!(inverse(&domain, &scalar), coefficients);
            assert_eq!(inverse(&domain, &neon), coefficients);
        }
    }
}
