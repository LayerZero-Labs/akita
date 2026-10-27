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
fn limb_admitted_primes_equal_split_primes_enumeration() {
    // Primality uses independent trial division, so neither the supplied
    // values nor domain setup is used to decide which candidate wins.
    let is_prime = |candidate: u32| {
        (2..)
            .take_while(|d| d * d <= candidate)
            .all(|d| !candidate.is_multiple_of(d))
    };
    // Smallest prime above 2^25 that splits the conductors 1944, 3888, 5832
    // and 7776, whose least common multiple is 23328.
    let commitment = ((1u32 << 25) / 23_328 + 1..)
        .map(|k| 23_328 * k + 1)
        .find(|&candidate| is_prime(candidate))
        .unwrap();
    // Largest prime below each power of two that splits the conductor 1944.
    let [p26, p27, p28] = [26, 27, 28].map(|bits| {
        (1..=((1u32 << bits) - 2) / 1944)
            .rev()
            .map(|k| 1944 * k + 1)
            .find(|&candidate| is_prime(candidate))
            .unwrap()
    });
    // Largest splitting prime whose centered window fits eight balanced
    // base-8 digits: h <= 3*(8^8-1)/7 = 7_190_235.
    let q24 = (1..=14_380_470 / 23_328)
        .rev()
        .map(|k| 23_328 * k + 1)
        .find(|&candidate| is_prime(candidate))
        .unwrap();
    assert_eq!(q24, 14_253_409);
    assert_eq!(q24, 1 + 23_328 * 611);
    let enumerated = [q24, commitment, p26, p27, p28];
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

#[test]
fn limb_all_available_backends_match_scalar_directly() {
    let backends = available_backends();
    let mut seed = 0x238a_903d_b861_072f;
    for prime in TrinomialLimbDomain::ADMITTED_PRIMES {
        let domain = TrinomialLimbDomain::new(prime).unwrap();
        let half = (prime / 2) as i32;
        for input in [
            random_coefficients(prime, &mut seed),
            [0; DEGREE],
            [1; DEGREE],
            [half; DEGREE],
            [-half; DEGREE],
            std::array::from_fn(|i| if i % 2 == 0 { half } else { -half }),
        ] {
            let mut scalar = domain.zero_slots();
            domain
                .forward_centered_backend(&input, &mut scalar, Backend::Scalar)
                .unwrap();
            let mut scalar_output = [0; DEGREE];
            domain
                .inverse_centered_backend(&scalar, &mut scalar_output, Backend::Scalar)
                .unwrap();
            assert_eq!(scalar_output, input);
            for backend in backends.iter().copied() {
                let mut actual = domain.zero_slots();
                domain
                    .forward_centered_backend(&input, &mut actual, backend)
                    .unwrap();
                assert_eq!(
                    scalar.values, actual.values,
                    "prime={prime}, backend={backend:?}"
                );
                let mut output = [0; DEGREE];
                domain
                    .inverse_centered_backend(&actual, &mut output, backend)
                    .unwrap();
                assert_eq!(output, input);
                domain
                    .inverse_centered_backend(&scalar, &mut output, backend)
                    .unwrap();
                assert_eq!(output, input);
                domain
                    .inverse_centered_backend(&actual, &mut output, Backend::Scalar)
                    .unwrap();
                assert_eq!(output, input);
            }
        }
        for input in [random_bits(&mut seed), [0; DEGREE], [1; DEGREE]] {
            let bits = packed_bits(&input);
            let mut scalar = domain.zero_slots();
            domain
                .forward_interleaved_bits_backend(&bits, &mut scalar, Backend::Scalar)
                .unwrap();
            let signed = std::array::from_fn(|i| {
                if (i / 4) % 2 == 0 {
                    input[i]
                } else {
                    -input[i]
                }
            });
            assert_eq!(inverse(&domain, &scalar), signed);
            for backend in backends.iter().copied() {
                let mut actual = domain.zero_slots();
                domain
                    .forward_interleaved_bits_backend(&bits, &mut actual, backend)
                    .unwrap();
                assert_eq!(
                    scalar.values, actual.values,
                    "prime={prime}, backend={backend:?}"
                );
                let mut output = [0; DEGREE];
                domain
                    .inverse_centered_backend(&actual, &mut output, backend)
                    .unwrap();
                assert_eq!(output, signed);
            }
        }
        // Arbitrary strict-lazy slots exercise inverse centering independently
        // of the representatives produced by either forward path.
        let largest = (prime - 1) as i32;
        for values in [
            [largest; DEGREE],
            [-largest; DEGREE],
            std::array::from_fn(|i| if i % 2 == 0 { largest } else { -largest }),
            std::array::from_fn(|_| {
                (next(&mut seed) % (2 * u64::from(prime) - 1)) as i32 - largest
            }),
        ] {
            let mut slots = domain.zero_slots();
            slots.values = values;
            let mut expected = [0; DEGREE];
            domain
                .inverse_centered_backend(&slots, &mut expected, Backend::Scalar)
                .unwrap();
            let centered = values.map(|v| domain.arithmetic.center(i64::from(v)));
            let mut expected_transform = centered;
            domain.transform(&mut expected_transform, true, Backend::Scalar, false);
            for backend in backends.iter().copied() {
                let mut actual = [0; DEGREE];
                domain
                    .inverse_centered_backend(&slots, &mut actual, backend)
                    .unwrap();
                assert_eq!(actual, expected, "prime={prime}, backend={backend:?}");
                let mut actual_transform = centered;
                domain.transform(&mut actual_transform, true, backend, false);
                assert_eq!(
                    actual_transform, expected_transform,
                    "prime={prime}, backend={backend:?}"
                );
            }
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
fn q24_primitive_root_has_exact_transform_order() {
    let arithmetic = arithmetic::Arithmetic { prime: 14_253_409 };
    // 1944 = 2^3 * 3^5; excluding order divisors 1944/2 and 1944/3
    // while checking r^1944=1 proves exact order.
    let root = arithmetic.pow(19, (arithmetic.prime - 1) / 1944);
    assert_eq!(root, 13_979_858);
    assert_eq!(arithmetic.pow(root, 1944), 1);
    assert_eq!(arithmetic.pow(root, 972), arithmetic.prime - 1);
    assert_eq!(arithmetic.pow(root, 648), 6_877_823);
    assert_ne!(arithmetic.pow(root, 648), 1);
    // The common conductor multiple is 23328 = 2^5 * 3^6.
    let common_root = arithmetic.pow(19, (arithmetic.prime - 1) / 23_328);
    assert_eq!(common_root, 13_834_210);
    assert_eq!(arithmetic.pow(common_root, 23_328), 1);
    assert_ne!(arithmetic.pow(common_root, 11_664), 1);
    assert_ne!(arithmetic.pow(common_root, 7_776), 1);
}

#[test]
fn limb_balanced_digits_and_signed_sources_match_integer_products() {
    use crate::ring::trinomial::{signed::signed_trinomial_product, MinusTrinomial};

    let mut seed = 0x21af_8acd_0236_4219;
    for prime in TrinomialLimbDomain::ADMITTED_PRIMES {
        let domain = TrinomialLimbDomain::new(prime).unwrap();
        let half = (prime / 2) as i32;
        for matrix in [
            [half; DEGREE],
            [-half; DEGREE],
            random_coefficients(prime, &mut seed),
        ] {
            for source in [
                [-4i8; DEGREE],
                [3i8; DEGREE],
                std::array::from_fn(|i| if (i / 4) % 2 == 0 { 1 } else { -1 }),
                std::array::from_fn(|_| (next(&mut seed) % 8) as i8 - 4),
            ] {
                let reference =
                    signed_trinomial_product::<DEGREE, MinusTrinomial>(&matrix, &source).unwrap();
                let expected = reference.map(|value| centered(value, prime));
                let mut accumulator = TrinomialLimbAccumulator::new(&domain);
                let left = transform(&domain, &matrix);
                let right = transform(&domain, &source.map(i32::from));
                // Cross a checkpoint to compare accumulated kernels too.
                for _ in 0..129 {
                    accumulator.add_product(&left, &right).unwrap();
                }
                let mut sum = domain.zero_slots();
                accumulator.finish(&mut sum).unwrap();
                assert_eq!(
                    inverse(&domain, &sum),
                    reference.map(|v| centered(129 * v, prime))
                );
                assert_product(&domain, &matrix, &source.map(i32::from));
                assert_eq!(schoolbook(&matrix, &source.map(i32::from), prime), expected);
            }
        }
    }
}
