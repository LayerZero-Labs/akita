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
    let backends = available_backends();
    for prime in TrinomialLimbDomain::ADMITTED_PRIMES {
        let domain = TrinomialLimbDomain::new(prime).unwrap();
        let largest = (prime - 1) as i32;
        let mut lhs = domain.zero_slots();
        let mut rhs = domain.zero_slots();
        lhs.values = std::array::from_fn(|i| if i % 2 == 0 { largest } else { -largest });
        rhs.values = std::array::from_fn(|i| if i % 4 < 2 { largest } else { -largest });
        for backend in backends.iter().copied() {
            let mut accumulator = TrinomialLimbAccumulator::new(&domain);
            for terms in 1..=8192 {
                accumulator
                    .add_product_backend(&lhs, &rhs, backend)
                    .unwrap();
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
                        "prime={prime}, terms={terms}, backend={backend:?}"
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
        for backend in available_backends() {
            for sign in [-1i64, 1] {
                rhs.values.fill((sign * i64::from(largest)) as i32);
                let mut accumulator = TrinomialLimbAccumulator::new(&domain);
                accumulator.sums.fill(sign * half);
                for _ in 0..REDUCTION_TERMS - 1 {
                    accumulator
                        .add_product_backend(&lhs, &rhs, backend)
                        .unwrap();
                }
                let before =
                    sign * (half + (REDUCTION_TERMS - 1) as i64 * i64::from(largest).pow(2));
                assert!(accumulator.sums.iter().all(|&sum| sum == before));
                let worst = i128::from(sign)
                    * (i128::from(half) + REDUCTION_TERMS as i128 * i128::from(largest).pow(2));
                assert!(worst.abs() <= i128::from(i64::MAX));
                accumulator
                    .add_product_backend(&lhs, &rhs, backend)
                    .unwrap();
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
}

#[test]
fn limb_column_accumulation_matches_scalar_and_i128_oracle() {
    let backends = available_backends();
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
            for backend in backends.iter().copied() {
                let mut column = vec![TrinomialLimbAccumulator::new(&domain); rank];
                let mut scalar = column.clone();
                for _ in 0..REDUCTION_TERMS + 1 {
                    for (accumulator, row) in column.iter_mut().zip(&refs) {
                        accumulator
                            .add_product_backend(row, &source, backend)
                            .unwrap();
                    }
                    for (accumulator, row) in scalar.iter_mut().zip(&refs) {
                        accumulator
                            .add_product_backend(row, &source, Backend::Scalar)
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
fn limb_product_batches_match_sequential_internal_state() {
    for prime in TrinomialLimbDomain::ADMITTED_PRIMES {
        let domain = TrinomialLimbDomain::new(prime).unwrap();
        let largest = (prime - 1) as i32;
        // Distinct per-term sign patterns exercise positive, negative and
        // cancelling products, including the exact admitted extremal values.
        let lhs: Vec<_> = (0..257)
            .map(|term| TrinomialLimbSlots {
                values: std::array::from_fn(|lane| {
                    if (term + lane) % 3 == 0 {
                        -largest
                    } else {
                        largest
                    }
                }),
                prime,
            })
            .collect();
        let rhs: Vec<_> = (0..257)
            .map(|term| TrinomialLimbSlots {
                values: std::array::from_fn(|lane| {
                    if (term + lane) % 5 < 2 {
                        largest
                    } else {
                        -largest
                    }
                }),
                prime,
            })
            .collect();
        for backend in available_backends() {
            // Incoming phases cover E=4/8/16 landing on and straddling the
            // checkpoint, plus batches crossing multiple checkpoints.
            for initial in [0, 1, 112, 120, 124, 127] {
                let mut starting = TrinomialLimbAccumulator::new(&domain);
                // Begin at the largest centered carry in both signs.
                starting.sums = std::array::from_fn(|lane| {
                    if lane % 2 == 0 {
                        i64::from(prime / 2)
                    } else {
                        -i64::from(prime / 2)
                    }
                });
                for (a, b) in lhs.iter().zip(&rhs).take(initial) {
                    starting.add_product_backend(a, b, Backend::Scalar).unwrap();
                }
                for count in [0, 1, 4, 8, 16, 127, 128, 129, 255, 256, 257] {
                    let mut sequential = starting.clone();
                    let mut batched = starting.clone();
                    for (a, b) in lhs.iter().zip(&rhs).take(count) {
                        sequential
                            .add_product_backend(a, b, Backend::Scalar)
                            .unwrap();
                    }
                    batched
                        .add_product_batch_backend(&lhs[..count], &rhs[..count], backend)
                        .unwrap();
                    assert_eq!(
                        batched.terms, sequential.terms,
                        "prime={prime}, backend={backend:?}, initial={initial}, count={count}"
                    );
                    assert_eq!(
                        batched.sums, sequential.sums,
                        "prime={prime}, backend={backend:?}, initial={initial}, count={count}"
                    );
                }
            }
            // Several separate calls preserve the same state as one long
            // call, including partial batches at both sides of checkpoints.
            for batch in [4, 8, 16] {
                let mut sequential = TrinomialLimbAccumulator::new(&domain);
                let mut batched = sequential.clone();
                for (left, right) in lhs.chunks(batch).zip(rhs.chunks(batch)) {
                    for (a, b) in left.iter().zip(right) {
                        sequential
                            .add_product_backend(a, b, Backend::Scalar)
                            .unwrap();
                    }
                    batched
                        .add_product_batch_backend(left, right, backend)
                        .unwrap();
                    assert_eq!(batched.sums, sequential.sums);
                    assert_eq!(batched.terms, sequential.terms);
                }
            }
        }
        let mut public = TrinomialLimbAccumulator::new(&domain);
        let mut sequential = public.clone();
        public.add_product_batch(&lhs, &rhs).unwrap();
        for (a, b) in lhs.iter().zip(&rhs) {
            sequential.add_product(a, b).unwrap();
        }
        assert_eq!(public.sums, sequential.sums);
        assert_eq!(public.terms, sequential.terms);
    }
}

#[test]
fn limb_product_batch_rejects_invalid_input_before_mutation() {
    for prime in TrinomialLimbDomain::ADMITTED_PRIMES {
        let domain = TrinomialLimbDomain::new(prime).unwrap();
        let slots = domain.zero_slots();
        let other = TrinomialLimbDomain::ADMITTED_PRIMES
            .into_iter()
            .find(|&candidate| candidate != prime)
            .unwrap();
        let wrong = TrinomialLimbDomain::new(other).unwrap().zero_slots();
        let valid = vec![slots.clone(); 129];
        let mut invalid = valid.clone();
        invalid[128] = wrong;
        for backend in available_backends() {
            let mut accumulator = TrinomialLimbAccumulator::new(&domain);
            accumulator.sums.fill(17);
            accumulator.terms = 127;
            for (lhs, rhs) in [
                (&valid[..], &valid[..128]),
                (&invalid, &valid),
                (&valid, &invalid),
            ] {
                let before = accumulator.clone();
                assert!(accumulator
                    .add_product_batch_backend(lhs, rhs, backend)
                    .is_err());
                assert_eq!(accumulator.sums, before.sums);
                assert_eq!(accumulator.terms, before.terms);
            }
        }
    }
}

#[test]
fn limb_product_batch_extremal_carry_matches_sequential() {
    for prime in TrinomialLimbDomain::ADMITTED_PRIMES {
        let domain = TrinomialLimbDomain::new(prime).unwrap();
        let largest = (prime - 1) as i32;
        let mut slot = domain.zero_slots();
        slot.values.fill(largest);
        let lhs = vec![slot.clone(); 257];
        for sign in [-1i64, 1] {
            slot.values.fill((sign * i64::from(largest)) as i32);
            let rhs = vec![slot.clone(); 257];
            for backend in available_backends() {
                for initial in [0, 112, 120, 124, 127] {
                    let mut sequential = TrinomialLimbAccumulator::new(&domain);
                    sequential.sums.fill(sign * i64::from(prime / 2));
                    for (a, b) in lhs.iter().zip(&rhs).take(initial) {
                        sequential
                            .add_product_backend(a, b, Backend::Scalar)
                            .unwrap();
                    }
                    let mut batched = sequential.clone();
                    for (a, b) in lhs.iter().zip(&rhs) {
                        sequential
                            .add_product_backend(a, b, Backend::Scalar)
                            .unwrap();
                    }
                    batched
                        .add_product_batch_backend(&lhs, &rhs, backend)
                        .unwrap();
                    assert_eq!(batched.sums, sequential.sums);
                    assert_eq!(batched.terms, sequential.terms);
                }
            }
        }
    }
}
