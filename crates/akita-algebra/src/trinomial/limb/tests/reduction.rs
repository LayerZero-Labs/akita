use super::*;

#[test]
fn center_matches_i128_over_checkpoint_interval_and_i64_extremes() {
    let mut seed = 0x6931_83fc_74a0_2d5b;
    for prime in TrinomialLimbDomain::ADMITTED_PRIMES {
        let arithmetic = arithmetic::Arithmetic { prime };
        let p = i128::from(prime);
        let h = p / 2;
        let product = (p - 1).pow(2);
        let bound = h + i128::from(TrinomialLimbAccumulator::REDUCTION_TERMS) * product;
        assert!(bound <= i128::from(i64::MAX));
        let check = |value: i128| {
            let input = i64::try_from(value).unwrap();
            assert_eq!(
                arithmetic.center(input),
                centered(value, prime),
                "prime={prime}, value={value}"
            );
        };

        // Check both ends and neighboring inputs for the full checkpoint
        // interval, plus the smaller eight-product evaluation/interpolation
        // interval and centered/lazy slot boundaries.
        for limit in [0, h, p - 1, 8 * h * h, bound] {
            for sign in [-1, 1] {
                for delta in [-1, 0, 1] {
                    check(sign * limit + delta);
                }
            }
        }

        // Every possible checkpoint batch size, with extremal product signs
        // and both centered carry extremes. i128 keeps the oracle independent
        // of the accumulator's i64 additions and its reduction implementation.
        for terms in 0..=TrinomialLimbAccumulator::REDUCTION_TERMS {
            for carry in [-h, 0, h] {
                for sign in [-1, 1] {
                    check(carry + sign * i128::from(terms) * product);
                }
            }
        }

        // Multiples of p exercise signed-remainder quotient changes. Half-p
        // offsets exercise the final centered-representative changes. Sample
        // both kinds around zero, the checkpoint extremes, and the complete
        // signed i64 limits, including i64::MIN.
        for anchor in [
            i128::from(i64::MIN),
            -bound,
            -bound / 2,
            0,
            bound / 2,
            bound,
            i128::from(i64::MAX),
        ] {
            for quotient in (anchor / p - 1)..=(anchor / p + 1) {
                for offset in [-h - 1, -h, -h + 1, -1, 0, 1, h - 1, h, h + 1, p - 1] {
                    let value = quotient * p + offset;
                    if i64::try_from(value).is_ok() {
                        check(value);
                    }
                }
            }
        }
        for value in [i64::MIN, i64::MIN + 1, i64::MAX - 1, i64::MAX] {
            check(i128::from(value));
        }

        // Sample the entire admitted checkpoint interval, without narrowing
        // either the random input or interval width to a signed 64-bit value.
        let width = 2 * bound + 1;
        for _ in 0..10_000 {
            check(i128::from(next(&mut seed)) % width - bound);
        }
        for _ in 0..1_000 {
            check(i128::from(next(&mut seed) as i64));
        }
    }
}
