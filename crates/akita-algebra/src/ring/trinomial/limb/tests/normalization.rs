use super::*;

#[test]
fn normalization_shift_matches_i128_over_every_quotient_interval() {
    for prime in TrinomialLimbDomain::ADMITTED_PRIMES {
        let domain = TrinomialLimbDomain::new(prime).unwrap();
        let q = domain.one.quotient;
        let k = 31 - q.trailing_zeros();
        assert_eq!(q, 1 << (31 - k));
        let bound = arithmetic::forward_bounds(prime)[4];
        let bias = 1i64 << (k - 1);
        assert!(bound + bias < i64::from(i32::MAX));
        // Both quotient expressions are monotone and constant between these
        // exact transition points. Test both ends of EVERY interval in
        // [-B4,B4], not just samples near zero, plus the admitted extremes.
        let mut operands = vec![-bound, bound];
        for t in ((-bound + bias) >> k)..=((bound + bias) >> k) + 1 {
            let transition = (t << k) - bias;
            for a in [transition - 1, transition, transition + 1] {
                if (-bound..=bound).contains(&a) {
                    operands.push(a);
                }
            }
        }
        for a in operands {
            let a = a as i32;
            let t = (a + bias as i32) >> k;
            let oracle = (i128::from(a) * i128::from(q) + (1i128 << 30)) >> 31;
            assert_eq!(i128::from(t), oracle);
            let actual = a.wrapping_sub(t.wrapping_mul(prime as i32));
            assert_eq!(
                i128::from(actual),
                i128::from(a) - oracle * i128::from(prime)
            );
            assert_eq!(actual, domain.arithmetic.mul(a, domain.one));
        }
    }
}
