use super::*;

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
