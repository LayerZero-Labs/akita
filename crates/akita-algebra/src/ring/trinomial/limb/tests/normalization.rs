use super::*;

#[test]
fn normalization_matches_i128_over_every_quotient_interval() {
    for prime in TrinomialLimbDomain::ADMITTED_PRIMES {
        let domain = TrinomialLimbDomain::new(prime).unwrap();
        let q = domain.one.quotient;
        let bound = arithmetic::forward_bounds(prime)[4];
        for a in normalization_transition_operands(q, bound) {
            let oracle = (i128::from(a) * i128::from(q) + (1i128 << 30)) >> 31;
            if (q as u32).is_power_of_two() {
                let k = 31 - q.trailing_zeros();
                let bias = 1i64 << (k - 1);
                assert!(bound + bias < i64::from(i32::MAX));
                let t = (a + bias as i32) >> k;
                assert_eq!(i128::from(t), oracle);
            }
            let actual = domain.arithmetic.mul(a, domain.one);
            assert_eq!(
                i128::from(actual),
                i128::from(a) - oracle * i128::from(prime)
            );
            assert!(i64::from(actual).abs() < i64::from(prime));
        }
    }
}

#[test]
fn q24_limb_accumulator_and_normalization_bounds() {
    let prime = 14_253_409;
    let p = i128::from(prime);
    let h = p / 2;
    // Recompute the rational ceiling in i128, independently of the i64
    // production bound formulas and the retained machine lane values.
    let product_bound = |b: i128| ((p << 31) + b * p + (1i128 << 32) - 1) >> 32;
    let mut bounds = [h; 5];
    for level in 0..4 {
        let b = bounds[level];
        let c = product_bound(b);
        bounds[level + 1] = (b + 2 * c).max(b + c + product_bound(2 * c));
    }
    assert_eq!(
        bounds,
        [7_126_704, 21_451_224, 35_847_012, 50_338_348, 64_925_866]
    );
    assert_eq!(arithmetic::forward_bounds(prime).map(i128::from), bounds);
    assert!(bounds.into_iter().all(|b| b < i128::from(i32::MAX)));
    assert_eq!(product_bound(bounds[4]), 7_342_170);
    assert!(product_bound(bounds[4]) < p);
    let inverse = (3 * h).max(2 * h + product_bound(2 * h));
    assert_eq!(inverse, 21_427_415);
    assert_eq!(product_bound(inverse), 7_197_815);
    assert!(inverse < i128::from(i32::MAX));
    assert!(product_bound(inverse) < p);
    assert_eq!(8 * h * h, 406_319_279_228_928);
    let accumulator = h + i128::from(TrinomialLimbAccumulator::REDUCTION_TERMS) * (p - 1).pow(2);
    assert_eq!(accumulator, 26_004_433_877_778_096);
    assert!(accumulator < i128::from(i64::MAX));
    let domain = TrinomialLimbDomain::new(prime).unwrap();
    assert_eq!(domain.one.quotient, 151);
    assert!(!(domain.one.quotient as u32).is_power_of_two());
}
