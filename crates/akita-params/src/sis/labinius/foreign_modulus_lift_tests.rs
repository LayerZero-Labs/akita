use super::*;
use crate::sis::labinius::{LabiniusCoefficientPrime, LabiniusDigitBase, LabiniusRootProfile};

const PROFILE: LabiniusRootProfile = LabiniusRootProfile::D648P128Q28BoundedW46Delta16;

fn invalid<T: std::fmt::Debug>(result: Result<T, AkitaError>, expected: &str) {
    match result {
        Err(AkitaError::InvalidSetup(message)) => assert!(message.contains(expected), "{message}"),
        other => panic!("expected InvalidSetup containing {expected:?}, got {other:?}"),
    }
}

#[test]
fn small_modulus_shape_and_digit_base_bounds_are_literal_fixtures() {
    let shape = LabiniusRootShape::derive(PROFILE, 22, 8, 128).unwrap();
    assert_eq!(shape.commitment_modulus(), PROFILE.commitment_modulus());
    assert_eq!(shape.ring_elements_per_column(), 4096);
    assert_eq!(shape.fold_width(), 256);
    assert_eq!(shape.rank_a(), 3);
    assert_eq!(shape.eta_a(), 24_116_880);
    assert_eq!(shape.image_len(), 497_664);
    assert_eq!(shape.a_quotient_len(), 1941);
    assert_eq!(shape.a_carry_len(), 1944);
    assert_eq!(shape.a_residual_bound(), Some(23_346_480_637_983_191_040));
    assert_eq!(shape.honest_a_carry_bound(), Some(260_919_297_587));
    assert_eq!(shape.derivation_bias_bits(), Some(79));
    let expected = [
        (
            LabiniusDigitBase::Bits1,
            39,
            39,
            287_649_523_880_552_564_791u128,
            105_855_024_788_043_343_843_088u128,
            (-274_877_906_944, 274_877_906_943),
        ),
        (
            LabiniusDigitBase::Bits2,
            40,
            20,
            435_222_320_333_752_371_255u128,
            160_161_813_882_820_872_621_840u128,
            (-549_755_813_888, 549_755_813_887),
        ),
        (
            LabiniusDigitBase::Bits4,
            40,
            10,
            435_222_320_333_752_371_255u128,
            160_161_813_882_820_872_621_840u128,
            (-549_755_813_888, 549_755_813_887),
        ),
    ];
    for (base, bits, digits, nu, cross, interval) in expected {
        let encoding = shape.derive_encoding(base).unwrap();
        let range = encoding.a_carry().unwrap();
        assert_eq!(range.bits(), bits);
        assert_eq!(range.digit_count(), digits);
        assert_eq!(range.interval(), interval);
        assert_eq!(encoding.a_carry_len(), 1944);
        assert_eq!(encoding.image_table_len(), 786_432);
        assert_eq!(encoding.image_table_log_len(), 20);
        assert_eq!(shape.a_carry_difference_bound(bits).unwrap(), Some(nu));
        assert_eq!(4 * 92 * nu, cross);
        shape.check_a_carry_no_wrap(bits).unwrap();
        invalid(
            shape.check_a_carry_no_wrap(bits - base.bits()),
            "honest endpoint",
        );
    }
}

#[test]
fn lift_admission_rejects_each_integer_inequality_and_overflow() {
    let challenge = PROFILE.challenge_profile().unwrap();
    let derive = |prime, modulus, log_n| {
        super::super::derive_shape(
            PROFILE,
            prime,
            Some(modulus),
            PROFILE.ring_degree(),
            &challenge,
            PROFILE.response_interval(),
            log_n,
            8,
            128,
        )
    };
    invalid(
        derive(PROFILE.coefficient_prime(), 48_233_759, 22),
        "2*eta_A < q0",
    );
    invalid(
        derive(LabiniusCoefficientPrime::P64Offset23703, 268_433_353, 22),
        "6*H_A < P",
    );
    invalid(
        derive(PROFILE.coefficient_prime(), 268_433_353, 38),
        "bias is below policy floor",
    );
    let mut shape = derive(PROFILE.coefficient_prime(), 268_433_353, 22).unwrap();
    // Isolate extraction over P64 from recovery and the independent bias floor.
    shape.coefficient_prime = LabiniusCoefficientPrime::P64Offset23703;
    for base in [
        LabiniusDigitBase::Bits1,
        LabiniusDigitBase::Bits2,
        LabiniusDigitBase::Bits4,
    ] {
        invalid(shape.derive_encoding(base), "4*Gamma_inf*B_nu < P");
    }
    let shape = derive(PROFILE.coefficient_prime(), 268_433_353, 22).unwrap();
    invalid(shape.check_a_carry_no_wrap(0), "envelope overflow");
    invalid(shape.check_a_carry_no_wrap(129), "envelope overflow");
    invalid(shape.check_a_carry_no_wrap(128), "envelope overflow");
    let mut shape = shape;
    shape.response_interval = (i128::MIN, i128::MAX);
    invalid(
        derive_foreign_modulus_lift(&shape, 268_433_353, &challenge),
        "bound overflow",
    );
    shape.response_interval = PROFILE.response_interval();
    shape.ring_elements_per_column = usize::MAX;
    invalid(
        derive_foreign_modulus_lift(&shape, 268_433_353, &challenge),
        "bound overflow",
    );
}

#[test]
fn achieved_derivation_bias_matches_an_independent_big_integer_oracle() {
    use num_bigint::BigUint;
    let p = LabiniusCoefficientPrime::P128OffsetA7F7.modulus();
    for denominator in [1, 2, 3, 4, 7_962_624 * 268_433_353, u128::MAX] {
        let threshold = BigUint::from(p) * 4u8;
        let expected = (0..=130)
            .filter(|&bits| (BigUint::from(denominator) << bits) <= threshold)
            .max();
        assert_eq!(derivation_bias_bits(p, denominator), expected);
    }
    assert_eq!(derivation_bias_bits(p, 0), None);
    assert_eq!(LABINIUS_MIN_DERIVATION_BIAS_BITS, 64);
}

#[test]
fn shared_prime_has_no_foreign_modulus_objects() {
    let shape =
        LabiniusRootShape::derive(LabiniusRootProfile::D648P128BoundedW46Delta16, 22, 8, 128)
            .unwrap();
    assert_eq!(shape.a_residual_bound(), None);
    assert_eq!(shape.honest_a_carry_bound(), None);
    assert_eq!(shape.derivation_bias_bits(), None);
    assert_eq!(shape.a_carry_len(), 0);
    assert_eq!(shape.a_carry_difference_bound(0).unwrap(), None);
    shape.check_a_carry_no_wrap(0).unwrap();
    for base in [
        LabiniusDigitBase::Bits1,
        LabiniusDigitBase::Bits2,
        LabiniusDigitBase::Bits4,
    ] {
        let encoding = shape.derive_encoding(base).unwrap();
        assert_eq!(encoding.a_carry(), None);
        assert_eq!(encoding.a_carry_len(), 0);
    }
}
