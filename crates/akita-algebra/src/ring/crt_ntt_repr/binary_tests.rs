use super::*;
use crate::ntt::{tables::Q64_PRIMES, BinaryNttStrategy};

const STRATEGIES: [BinaryNttStrategy; 7] = [
    BinaryNttStrategy::Mask,
    BinaryNttStrategy::Compact4,
    BinaryNttStrategy::Positional4,
    BinaryNttStrategy::SelectedPairs,
    BinaryNttStrategy::Selected4,
    BinaryNttStrategy::Split4,
    BinaryNttStrategy::Split8,
];

fn check<const D: usize>() {
    let params = CrtNttParamSet::new(Q64_PRIMES);
    let lut = DigitMontLut::new_with_digit_bound(&params, 128);
    let mut cases: Vec<[i8; D]> = (0..16)
        .map(|nibble| std::array::from_fn(|i| ((nibble >> (i / (D / 4))) & 1) as i8))
        .collect();
    cases.push(std::array::from_fn(|i| {
        (((i % (D / 4)) * 7 + 3) >> (i / (D / 4)) & 1) as i8
    }));
    let mut state = 0x943ae31af7717u64;
    for _ in 0..8 {
        cases.push(std::array::from_fn(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state & 1) as i8
        }));
    }
    for position in [0, D / 4 - 1, D / 4, D / 2, D - 1] {
        let mut one = [0; D];
        one[position] = 1;
        cases.push(one);
    }
    for digits in cases {
        let expected = CyclotomicCrtNtt::from_i8_with_params(&digits, &params);
        for strategy in STRATEGIES {
            if D < 8 && strategy == BinaryNttStrategy::Split8 {
                continue;
            }
            let actual =
                CyclotomicCrtNtt::from_binary_with_lut(&digits, &params, &lut, strategy).unwrap();
            assert_eq!(actual, expected, "D={D}, {strategy:?}");
            let tables = lut.binary.get().unwrap();
            for k in 0..3 {
                let mut scalar = [MontCoeff::from_raw(0); D];
                crate::ntt::binary::forward_scalar(
                    &mut scalar,
                    &digits,
                    params.primes[k],
                    &params.twiddles[k],
                    &tables[k],
                    strategy,
                );
                assert_eq!(
                    scalar, expected.limbs[k],
                    "scalar D={D}, {strategy:?}, limb={k}"
                );
            }
        }
        let automatic = CyclotomicCrtNtt::from_i8_with_lut(&digits, &params, &lut);
        assert_eq!(automatic, expected);
    }
    for invalid in [i8::MIN, -1, 2, i8::MAX] {
        let mut digits = [0; D];
        digits[D - 1] = invalid;
        for strategy in STRATEGIES {
            assert!(
                CyclotomicCrtNtt::from_binary_with_lut(&digits, &params, &lut, strategy).is_none()
            );
        }
        assert_eq!(
            CyclotomicCrtNtt::from_i8_with_lut(&digits, &params, &lut),
            CyclotomicCrtNtt::from_i8_with_params(&digits, &params)
        );
    }
}

#[test]
fn binary_ntt_startups_match_all_limbs_and_layouts() {
    check::<4>();
    check::<8>();
    check::<16>();
    check::<32>();
    check::<128>();
    check::<256>();
    check::<512>();
    check::<1024>();
}

#[test]
fn binary_ntt_cache_dimension_mismatch_falls_back() {
    let small = CrtNttParamSet::<_, 3, 128>::new(Q64_PRIMES);
    let large = CrtNttParamSet::<_, 3, 1024>::new(Q64_PRIMES);
    let lut = DigitMontLut::new_with_digit_bound(&small, 2);
    let first = CyclotomicCrtNtt::from_binary_with_lut(
        &[1; 128],
        &small,
        &lut,
        BinaryNttStrategy::Compact4,
    );
    assert!(first.is_some());
    for strategy in STRATEGIES {
        assert!(
            CyclotomicCrtNtt::from_binary_with_lut(&[1; 1024], &large, &lut, strategy).is_none()
        );
    }
    assert_eq!(
        CyclotomicCrtNtt::from_i8_with_lut(&[1; 1024], &large, &lut),
        CyclotomicCrtNtt::from_i8_with_params(&[1; 1024], &large)
    );
}

#[test]
fn binary_ntt_cache_prime_mismatch_and_tiny_dimensions() {
    let first = CrtNttParamSet::<_, 3, 128>::new(Q64_PRIMES);
    let second = CrtNttParamSet::<_, 3, 128>::new([Q64_PRIMES[1], Q64_PRIMES[2], Q64_PRIMES[0]]);
    let lut = DigitMontLut::new_with_digit_bound(&first, 2);
    assert!(CyclotomicCrtNtt::from_binary_with_lut(
        &[1; 128],
        &first,
        &lut,
        BinaryNttStrategy::Mask
    )
    .is_some());
    for strategy in STRATEGIES {
        assert!(
            CyclotomicCrtNtt::from_binary_with_lut(&[1; 128], &second, &lut, strategy).is_none()
        );
    }

    let tiny = CrtNttParamSet::<_, 3, 2>::new(Q64_PRIMES);
    let tiny_lut = DigitMontLut::new_with_digit_bound(&tiny, 2);
    for strategy in STRATEGIES {
        assert!(
            CyclotomicCrtNtt::from_binary_with_lut(&[0, 1], &tiny, &tiny_lut, strategy).is_none()
        );
    }
    let one = CrtNttParamSet::<_, 3, 1>::new(Q64_PRIMES);
    let one_lut = DigitMontLut::new_with_digit_bound(&one, 2);
    for strategy in STRATEGIES {
        assert!(CyclotomicCrtNtt::from_binary_with_lut(&[1], &one, &one_lut, strategy).is_none());
    }
}

#[test]
fn binary_ntt_i16_scalar_reference() {
    let params = CrtNttParamSet::<_, 1, 128>::new([crate::ntt::tables::I16_TAIL_PRIME]);
    let lut = DigitMontLut::new_with_digit_bound(&params, 2);
    for nibble in 0..16 {
        let digits = std::array::from_fn(|i| ((nibble >> (i / 32)) & 1) as i8);
        for strategy in STRATEGIES {
            assert_eq!(
                CyclotomicCrtNtt::from_binary_with_lut(&digits, &params, &lut, strategy).unwrap(),
                CyclotomicCrtNtt::from_i8_with_params(&digits, &params)
            );
        }
    }
}

#[test]
fn binary_ntt_split8_exhaustive_octets() {
    fn check_octets<const D: usize>() {
        let params = CrtNttParamSet::<_, 3, D>::new(Q64_PRIMES);
        let lut = DigitMontLut::new_with_digit_bound(&params, 2);
        for octet in 0..256 {
            let digits = std::array::from_fn(|i| ((octet >> (i / (D / 8))) & 1) as i8);
            let actual = CyclotomicCrtNtt::from_binary_with_lut(
                &digits,
                &params,
                &lut,
                BinaryNttStrategy::Split8,
            )
            .unwrap();
            assert_eq!(
                actual,
                CyclotomicCrtNtt::from_i8_with_params(&digits, &params),
                "D={D}, octet={octet}"
            );
        }
    }
    check_octets::<8>();
    check_octets::<16>();
    check_octets::<32>();
    check_octets::<1024>();
}
