use super::*;

const REDUCTION_TERMS: usize = TrinomialLimbAccumulator::REDUCTION_TERMS as usize;

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
