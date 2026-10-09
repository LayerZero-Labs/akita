use super::*;

fn next(seed: &mut u64) -> u64 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    *seed
}

fn signed_coefficients(bits: &[u64; 11]) -> [i32; DEGREE] {
    std::array::from_fn(|n| {
        let bit = ((bits[n / 64] >> (n % 64)) & 1) as i32;
        if (n / 4) % 2 == 0 {
            bit
        } else {
            -bit
        }
    })
}

fn assert_centered_agreement(domain: &TrinomialWideLimbDomain, bits: &[u64; 11]) {
    let coefficients = signed_coefficients(bits);
    let mut expected = domain.zero_slots();
    domain
        .forward_centered(&coefficients, &mut expected)
        .unwrap();
    let mut expected_coefficients = [0; DEGREE];
    domain
        .inverse_centered(&expected, &mut expected_coefficients)
        .unwrap();
    assert_eq!(expected_coefficients, coefficients);
    #[cfg(target_arch = "aarch64")]
    let backends = [false, true];
    #[cfg(not(target_arch = "aarch64"))]
    let backends = [false];
    let mut previous = None;
    for neon in backends {
        let mut actual = domain.zero_slots();
        domain
            .forward_interleaved_bits_backend(bits, &mut actual, neon)
            .unwrap();
        let mut actual_coefficients = [0; DEGREE];
        domain
            .inverse_centered(&actual, &mut actual_coefficients)
            .unwrap();
        assert_eq!(actual_coefficients, expected_coefficients);
        if let Some(previous) = previous {
            assert_eq!(actual, previous);
        }
        previous = Some(actual);
    }
}

#[test]
fn wide_limb_interleaved_every_single_bit_matches_centered() {
    for prime in TrinomialWideLimbDomain::ADMITTED_PRIMES {
        let domain = TrinomialWideLimbDomain::new(prime).unwrap();
        for position in 0..DEGREE {
            let mut bits = [0; 11];
            bits[position / 64] = 1 << (position % 64);
            assert_centered_agreement(&domain, &bits);
        }
    }
}

#[test]
fn wide_limb_interleaved_random_zero_and_ones_match_centered() {
    let mut seed = 0x40f5_bca8_721f_291d;
    for prime in TrinomialWideLimbDomain::ADMITTED_PRIMES {
        let domain = TrinomialWideLimbDomain::new(prime).unwrap();
        assert_centered_agreement(&domain, &[0; 11]);
        let mut ones = [u64::MAX; 11];
        ones[10] = 255;
        assert_centered_agreement(&domain, &ones);
        for _ in 0..32 {
            let mut bits = std::array::from_fn(|_| next(&mut seed));
            bits[10] &= 255;
            assert_centered_agreement(&domain, &bits);
        }
        let half = i64::from(prime / 2);
        assert!(domain
            .interleaved_first_level
            .get()
            .unwrap()
            .iter()
            .flatten()
            .flatten()
            .all(|&entry| i64::from(entry).abs() <= half));
    }
}

#[test]
fn wide_limb_interleaved_rejects_bad_bits_and_tags_without_mutation() {
    for prime in TrinomialWideLimbDomain::ADMITTED_PRIMES {
        let domain = TrinomialWideLimbDomain::new(prime).unwrap();
        let mut out = domain.zero_slots();
        out.values.fill(17);
        let original = out.clone();
        for len in [0, 1, 10, 12, 100] {
            assert_eq!(
                domain.forward_interleaved_bits(&vec![0; len], &mut out),
                Err(TrinomialError::CoefficientLength {
                    expected: 11,
                    actual: len,
                })
            );
        }
        for bit in 8..64 {
            let mut bits = [0; 11];
            bits[10] = 1 << bit;
            assert_eq!(
                domain.forward_interleaved_bits(&bits, &mut out),
                Err(TrinomialError::LimbInput {
                    reason: "unused tail bits must be zero",
                })
            );
        }
        assert_eq!(out, original);
        for other in TrinomialWideLimbDomain::ADMITTED_PRIMES
            .into_iter()
            .filter(|&p| p != prime)
        {
            let mut wrong = TrinomialWideLimbDomain::new(other).unwrap().zero_slots();
            let original = wrong.clone();
            assert_eq!(
                domain.forward_interleaved_bits(&[0; 11], &mut wrong),
                Err(TrinomialError::LimbInput {
                    reason: "slot prime does not match the domain",
                })
            );
            assert_eq!(wrong, original);
        }
        assert!(domain.interleaved_first_level.get().is_none());
    }
}

#[test]
fn wide_limb_interleaved_tables_are_lazy_and_reported() {
    let domain = TrinomialWideLimbDomain::new(268_433_353).unwrap();
    assert_eq!(domain.table_storage_bytes(), 81_920);
    let mut out = domain.zero_slots();
    domain.forward_bits(&[0; 11], &mut out).unwrap();
    assert_eq!(domain.table_storage_bytes(), 81_920);
    domain.forward_interleaved_bits(&[0; 11], &mut out).unwrap();
    assert_eq!(domain.table_storage_bytes(), 81_920 + 294_912);
    let original = domain.interleaved_first_level.get().unwrap().as_ptr();
    domain.forward_interleaved_bits(&[0; 11], &mut out).unwrap();
    assert_eq!(
        domain.interleaved_first_level.get().unwrap().as_ptr(),
        original
    );
}

fn bits_from_indices(indices: &[u8; PIECE]) -> [u64; 11] {
    let mut bits = [0; 11];
    for (t, &byte) in indices.iter().enumerate() {
        for j in 0..LANES {
            let n = t + PIECE * j;
            bits[n / 64] |= u64::from((byte >> j) & 1) << (n % 64);
        }
    }
    bits
}

// An independent i128 rounded-product oracle, including a check of the exact
// reciprocal used by production arithmetic, rather than calling its mul.
fn shadow_product(prime: u32, value: i128, twiddle: super::super::arithmetic::Twiddle) -> i128 {
    i32::try_from(value).unwrap();
    let p = i128::from(prime);
    let reciprocal = (i128::from(twiddle.value) * (1i128 << 31) + p / 2).div_euclid(p);
    assert_eq!(reciprocal, i128::from(twiddle.quotient));
    let quotient = (value * reciprocal + (1i128 << 30)) >> 31;
    let result = value * i128::from(twiddle.value) - quotient * p;
    i32::try_from(result).unwrap();
    assert!(
        result.abs()
            <= i128::from(super::super::arithmetic::multiply_bound(
                prime,
                i64::try_from(value.abs()).unwrap(),
            ))
    );
    result
}

fn assert_signed_shadow(domain: &TrinomialWideLimbDomain, indices: &[u8; PIECE]) {
    use super::super::arithmetic;

    let prime = domain.prime();
    let half = i128::from(prime / 2);
    let tables = domain.interleaved_first_level.get().unwrap();
    let mut shadow = [0i128; DEGREE];
    // Reconstruct the fused signed first level, checking its intermediate
    // additions as well as its final three-centered-entry bound.
    for t in 0..27 {
        for i in 0..3 * LANES {
            let entries: [_; 3] = std::array::from_fn(|source| {
                let phase = (t + 27 * source) & 7;
                let entry = i128::from(
                    tables[(phase & 3) * 3 + source][usize::from(indices[t + 27 * source])][i],
                );
                assert!(entry.abs() <= half);
                if phase & 4 == 0 {
                    entry
                } else {
                    -entry
                }
            });
            let partial = entries[0] + entries[1];
            let sum = partial + entries[2];
            i32::try_from(partial).unwrap();
            i32::try_from(sum).unwrap();
            assert!(sum.abs() <= 3 * half);
            shadow[(t + (i / LANES) * 27) * LANES + i % LANES] = sum;
        }
    }
    let mut actual = [0; DEGREE];
    table_scalar(tables, indices, &mut actual);
    assert_eq!(actual.map(i128::from), shadow);
    let bounds = arithmetic::forward_bounds(prime).map(i128::from);
    assert!(3 * half <= bounds[1]);
    for (level, stride) in [9, 3, 1].into_iter().enumerate() {
        for split in domain.splits.iter().filter(|s| s.stride == stride) {
            let step = split.stride * LANES;
            for t in 0..split.stride {
                let offset = (split.start + t) * LANES;
                for lane in 0..LANES {
                    let a = shadow[offset + lane];
                    let input_b = shadow[offset + step + lane];
                    let input_c = shadow[offset + 2 * step + lane];
                    assert!([a, input_b, input_c]
                        .iter()
                        .all(|v| v.abs() <= bounds[level + 1]));
                    let b = shadow_product(prime, input_b, split.r[lane]);
                    let c = shadow_product(prime, input_c, split.r2[lane]);
                    let difference_input = b - c;
                    i32::try_from(difference_input).unwrap();
                    let difference = shadow_product(prime, difference_input, domain.omega);
                    let ab = a + b;
                    let ac = a - c;
                    let am_b = a - b;
                    let outputs = [ab + c, ac + difference, am_b - difference];
                    for value in [ab, ac, am_b, difference_input, difference]
                        .into_iter()
                        .chain(outputs)
                    {
                        i32::try_from(value).unwrap();
                        assert!(
                            value.abs() <= bounds[level + 2],
                            "prime={prime}, stride={stride}, value={value}"
                        );
                    }
                    shadow[offset + lane] = outputs[0];
                    shadow[offset + step + lane] = outputs[1];
                    shadow[offset + 2 * step + lane] = outputs[2];
                }
            }
            arithmetic::butterfly(
                domain.arithmetic,
                domain.omega,
                domain.inverse_omega,
                &mut actual,
                split,
                false,
            );
            assert_eq!(actual.map(i128::from), shadow);
        }
        assert!(shadow.iter().all(|v| v.abs() <= bounds[level + 2]));
    }
    let normalized = shadow.map(|v| {
        let reduced = shadow_product(prime, v, domain.one);
        assert!(reduced.abs() < i128::from(prime));
        i32::try_from(reduced).unwrap()
    });
    let bits = bits_from_indices(indices);
    #[cfg(target_arch = "aarch64")]
    let backends = [false, true];
    #[cfg(not(target_arch = "aarch64"))]
    let backends = [false];
    for neon in backends {
        let mut slots = domain.zero_slots();
        domain
            .forward_interleaved_bits_backend(&bits, &mut slots, neon)
            .unwrap();
        assert_eq!(slots.values, normalized);
        let mut coefficients = [0; DEGREE];
        domain.inverse_centered(&slots, &mut coefficients).unwrap();
        assert_eq!(coefficients, signed_coefficients(&bits));
    }
}

#[test]
fn wide_limb_interleaved_lazy_levels_match_checked_signed_shadow() {
    let mut seed = 0x7e40_c152_a4d9_116f;
    for prime in TrinomialWideLimbDomain::ADMITTED_PRIMES {
        let domain = TrinomialWideLimbDomain::new(prime).unwrap();
        domain
            .forward_interleaved_bits(&[0; 11], &mut domain.zero_slots())
            .unwrap();
        for indices in [[0; PIECE], [255; PIECE]]
            .into_iter()
            .chain((0..8).map(|_| std::array::from_fn(|_| next(&mut seed) as u8)))
        {
            assert_signed_shadow(&domain, &indices);
        }
        let tables = domain.interleaved_first_level.get().unwrap();
        // Every source byte is independently selectable by real coefficient
        // bits. For each first-level output, choose all three table extrema
        // after applying their phase signs; these inputs attain both bounds.
        for t in 0..27 {
            for i in 0..3 * LANES {
                for maximum in [false, true] {
                    let mut indices = [0; PIECE];
                    let mut extremum = 0i128;
                    for source in 0..3 {
                        let phase = (t + 27 * source) & 7;
                        let signed_entry = |byte: usize| {
                            let value = i128::from(tables[(phase & 3) * 3 + source][byte][i]);
                            if phase & 4 == 0 {
                                value
                            } else {
                                -value
                            }
                        };
                        let byte = if maximum {
                            (0..256).max_by_key(|&byte| signed_entry(byte))
                        } else {
                            (0..256).min_by_key(|&byte| signed_entry(byte))
                        }
                        .unwrap();
                        indices[t + 27 * source] = byte as u8;
                        extremum += signed_entry(byte);
                    }
                    assert!(extremum.abs() <= 3 * i128::from(prime / 2));
                    let mut values = [0; DEGREE];
                    table_scalar(tables, &indices, &mut values);
                    assert_eq!(
                        i128::from(values[(t + (i / LANES) * 27) * LANES + i % LANES]),
                        extremum
                    );
                    let bits = bits_from_indices(&indices);
                    let mut gathered = [0; PIECE];
                    TrinomialWideLimbDomain::gather_bits(&bits, &mut gathered).unwrap();
                    assert_eq!(gathered, indices);
                    assert_signed_shadow(&domain, &indices);
                }
            }
        }
    }
}
