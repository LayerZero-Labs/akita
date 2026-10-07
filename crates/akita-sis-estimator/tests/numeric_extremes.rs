use akita_sis_estimator::{
    akita_q32, akita_q64, cost_infinity, probability::log2_amplify, Bound, EstimateConfig,
    EstimatorError, SisNorm, SisParameters,
};
use num_bigint::BigUint;

#[test]
fn equivalent_exact_rational_bounds_have_equal_fixed_candidate_costs() {
    for value in [1u64, 3, 15, 1023] {
        let mut params = SisParameters::try_new(
            32,
            akita_q32(),
            Some(4096),
            Bound::from_u64(value),
            SisNorm::Infinity,
        )
        .unwrap();
        let expected = cost_infinity(483, &params, 0, &EstimateConfig::default()).unwrap();
        assert!(expected.rop.log2().unwrap().is_finite());
        for factor in [
            BigUint::from(1u8),
            BigUint::from(1u8) << 2048usize,
            (BigUint::from(1u8) << 4096usize) + BigUint::from(17u8),
        ] {
            params.length_bound = Bound::Rational {
                numerator: BigUint::from(value) * &factor,
                denominator: factor,
            };
            let actual = cost_infinity(483, &params, 0, &EstimateConfig::default()).unwrap();
            assert_eq!(actual, expected);
        }
    }
}

#[test]
fn non_binary_rational_bounds_are_invariant_under_common_factors() {
    let mut params = SisParameters::try_new(
        32,
        akita_q32(),
        Some(4096),
        Bound::Rational {
            numerator: BigUint::from(7u8),
            denominator: BigUint::from(3u8),
        },
        SisNorm::Infinity,
    )
    .unwrap();
    let expected = cost_infinity(483, &params, 0, &EstimateConfig::default()).unwrap();
    for factor in [
        BigUint::from(3u8),
        BigUint::from(5u8),
        BigUint::from(7u8),
        (BigUint::from(1u8) << 2048usize) + BigUint::from(1u8),
    ] {
        params.length_bound = Bound::Rational {
            numerator: BigUint::from(7u8) * &factor,
            denominator: BigUint::from(3u8) * factor,
        };
        assert_eq!(
            cost_infinity(483, &params, 0, &EstimateConfig::default()).unwrap(),
            expected
        );
    }
}

#[test]
fn subnormal_rational_bound_matches_its_exact_float_representation() {
    let mut params = SisParameters::try_new(
        32,
        akita_q32(),
        Some(4096),
        Bound::Float(f64::from_bits(1)),
        SisNorm::Infinity,
    )
    .unwrap();
    let expected = cost_infinity(483, &params, 0, &EstimateConfig::default()).unwrap();
    assert!(expected.rop.log2().unwrap().is_finite());
    params.length_bound = Bound::Rational {
        numerator: BigUint::from(1u8),
        denominator: BigUint::from(1u8) << 1074usize,
    };
    assert_eq!(
        cost_infinity(483, &params, 0, &EstimateConfig::default()).unwrap(),
        expected
    );
}

#[test]
fn unsupported_bound_magnitudes_return_errors_instead_of_infinite_costs() {
    let large = BigUint::from(1u8) << 2048usize;
    for (numerator, denominator) in [
        (large.clone(), BigUint::from(1u8)),
        (BigUint::from(1u8), large),
    ] {
        let params = SisParameters::try_new(
            32,
            akita_q32(),
            Some(4096),
            Bound::Rational {
                numerator,
                denominator,
            },
            SisNorm::Infinity,
        )
        .unwrap();
        assert!(matches!(
            cost_infinity(483, &params, 0, &EstimateConfig::default()),
            Err(EstimatorError::InvalidParameter {
                field: "length_bound",
                ..
            })
        ));
    }
}

#[test]
fn log_amplification_is_continuous_across_overflow_and_underflow() {
    // log2(-ln(0.01)), independently evaluated to 50 decimal digits.
    const OFFSET: f64 = 2.203_254_472_699_722;
    for center in [54.0, 1022.0, 1023.0, 1074.0, 1100.0] {
        let mut previous = 0.0;
        for step in -16..=16 {
            let exponent = center + f64::from(step) / 16.0;
            let actual = log2_amplify(0.99, -exponent);
            assert!(actual.is_finite());
            assert!((actual - (exponent + OFFSET)).abs() < 1e-11);
            assert!(actual > previous);
            previous = actual;
        }
    }
    assert_eq!(log2_amplify(0.99, 0.0), 0.0);
    assert_eq!(log2_amplify(0.99, f64::NEG_INFINITY), f64::INFINITY);
    assert_eq!(
        log2_amplify(2.5 * 2.0_f64.powi(-100), -100.0),
        3.0_f64.log2()
    );
}

#[test]
fn fixed_candidate_cost_stays_finite_as_the_bound_narrows() {
    let mut previous = 0.0;
    for bound in [1023, 511] {
        let params = SisParameters::try_new(
            32,
            akita_q64(),
            Some(85),
            Bound::from_u64(bound),
            SisNorm::Infinity,
        )
        .unwrap();
        let cost = cost_infinity(63, &params, 0, &EstimateConfig::default()).unwrap();
        let bits = cost.rop.log2().unwrap();
        assert!(bits.is_finite() && bits > 1000.0);
        assert!(bits > previous);
        previous = bits;
    }
}
