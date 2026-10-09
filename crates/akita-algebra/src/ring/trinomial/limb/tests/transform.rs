use super::*;

fn packed_bits(coefficients: &[i32; DEGREE]) -> [u64; 11] {
    let mut bits = [0; 11];
    for (index, &coefficient) in coefficients.iter().enumerate() {
        bits[index / 64] |= (coefficient as u64) << (index % 64);
    }
    bits
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
