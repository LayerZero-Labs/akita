use super::*;
use crate::sis::labinius::{LabiniusCoefficientPrime, LabiniusRootProfile};

const PROFILE: LabiniusRootProfile = LabiniusRootProfile::D648P128BoundedW46Delta16;
const BASES: [LabiniusDigitBase; 3] = [
    LabiniusDigitBase::Bits1,
    LabiniusDigitBase::Bits2,
    LabiniusDigitBase::Bits4,
];

#[test]
fn golden_enforced_ranges_and_root_witness_lengths() {
    let shape = LabiniusRootShape::derive(PROFILE, 22, 8, 128).unwrap();
    let expected = [
        (
            39,
            38,
            1186484727296,
            [42_467_328, 6_279, 6_156],
            42_479_763,
            26,
        ),
        (
            40,
            38,
            2011118448128,
            [21_233_664, 3_220, 3_078],
            21_239_962,
            25,
        ),
        (
            40,
            40,
            2835752168960,
            [10_616_832, 1_610, 1_620],
            10_620_062,
            24,
        ),
    ];
    for (base, (eq, ek, total, lengths, natural, padded)) in BASES.into_iter().zip(expected) {
        let encoding = shape.derive_encoding(base).unwrap();
        assert_eq!(encoding.base(), base);
        assert_eq!(encoding.response().bits(), 16);
        assert_eq!(encoding.response().interval(), PROFILE.response_interval());
        assert_eq!(encoding.response().offset(), 32_768);
        assert_eq!(encoding.quotient().bits(), eq);
        assert_eq!(encoding.carry().bits(), ek);
        assert_eq!(encoding.no_wrap_total(), total);
        assert_eq!(encoding.segment_lengths(), lengths);
        assert_eq!(encoding.natural_len(), natural);
        assert_eq!(encoding.padded_log_len(), padded);
        assert_eq!(shape.image_len(), 165_888);
        assert_eq!(encoding.image_padded_log_len(), 18);
        for (range, honest) in [
            (
                encoding.response(),
                u128::try_from(shape.fold_width()).unwrap() * 92,
            ),
            (encoding.quotient(), shape.honest_quotient_bound()),
            (encoding.carry(), shape.honest_carry_bound()),
        ] {
            let honest = i128::try_from(honest).unwrap();
            let (lower, upper) = range.interval();
            assert!(lower <= -honest && honest <= upper);
            assert_eq!(
                range.digit_count(),
                usize::try_from(range.bits() / base.bits()).unwrap()
            );
        }
    }
}

// Pure scalar map; iterates only the digits of one integer, not a root vector.
fn map_uniform_digit(
    range: LabiniusSignedDigitRange,
    base: LabiniusDigitBase,
    digit: u128,
) -> i128 {
    let mut encoded = 0u128;
    for i in 0..range.digit_count() {
        encoded += digit << (u32::try_from(i).unwrap() * base.bits());
    }
    i128::try_from(encoded).unwrap() - i128::try_from(range.offset()).unwrap()
}

#[test]
fn every_integer_family_uses_the_same_exact_signed_digit_map() {
    let shape = LabiniusRootShape::derive(PROFILE, 22, 8, 128).unwrap();
    for base in BASES {
        let encoding = shape.derive_encoding(base).unwrap();
        for range in [encoding.response(), encoding.quotient(), encoding.carry()] {
            let (lower, upper) = range.interval();
            assert_eq!(map_uniform_digit(range, base, 0), lower);
            assert_eq!(
                map_uniform_digit(range, base, (1u128 << base.bits()) - 1),
                upper
            );
        }
    }
}

#[test]
fn digit_base_tags_and_envelope_minimality_are_checked() {
    for base in BASES {
        assert_eq!(LabiniusDigitBase::from_tag(base.tag()).unwrap(), base);
        for honest in [
            0,
            1,
            2,
            3,
            7,
            8,
            15,
            16,
            127,
            128,
            1_024,
            (1u128 << 126) - 1,
        ] {
            let range = envelope_for_bound(honest, base).unwrap();
            let required = 2 * honest + 1;
            assert!(
                (num_bigint::BigUint::from(1u8) << range.bits())
                    >= num_bigint::BigUint::from(required)
            );
            if range.bits() > base.bits() {
                assert!((1u128 << (range.bits() - base.bits())) < required);
            }
            assert!(range.offset() > honest);
        }
        assert!(envelope_for_bound(u128::MAX, base).is_err());
        assert!(signed_digit_range(0, base).is_err());
        assert!(signed_digit_range(132, base).is_err());
    }
    for tag in 3..=u8::MAX {
        assert!(matches!(
            LabiniusDigitBase::from_tag(tag),
            Err(AkitaError::InvalidSetup(_))
        ));
    }
    let widest = signed_digit_range(128, LabiniusDigitBase::Bits4).unwrap();
    assert_eq!(widest.interval(), (i128::MIN, i128::MAX));
}

fn custom_shape(
    prime: LabiniusCoefficientPrime,
    interval: (i128, i128),
    log_n: u32,
    log_c: u32,
) -> Result<LabiniusRootShape, AkitaError> {
    super::super::derive_shape(
        PROFILE,
        prime,
        PROFILE.ring_degree(),
        &PROFILE.challenge_profile()?,
        interval,
        log_n,
        log_c,
        0,
    )
}

#[test]
fn incompatible_response_intervals_and_digit_divisibility_reject() {
    for interval in [(-32_768, 32_768), (-32_767, 32_767)] {
        let shape = custom_shape(PROFILE.coefficient_prime(), interval, 2, 0).unwrap();
        for base in BASES {
            assert!(matches!(
                shape.derive_encoding(base),
                Err(AkitaError::InvalidSetup(_))
            ));
        }
    }
    let odd_bits = custom_shape(PROFILE.coefficient_prime(), (-16_384, 16_383), 2, 0).unwrap();
    odd_bits.derive_encoding(LabiniusDigitBase::Bits1).unwrap();
    for base in [LabiniusDigitBase::Bits2, LabiniusDigitBase::Bits4] {
        assert!(matches!(
            odd_bits.derive_encoding(base),
            Err(AkitaError::InvalidSetup(_))
        ));
    }
}

#[test]
fn rounded_no_wrap_rejects_after_an_internal_prime_perturbation() {
    // Derive an admitted P128 shape, then isolate the envelope gate at P64.
    // This perturbed shape is test-only: P64's SIS cells cannot admit its width.
    let interval = (-(1i128 << 31), (1i128 << 31) - 1);
    let mut shape = custom_shape(PROFILE.coefficient_prime(), interval, 22, 0).unwrap();
    shape.coefficient_prime = LabiniusCoefficientPrime::P64Offset23703;
    shape
        .check_parity_no_wrap(shape.honest_quotient_bound(), shape.honest_carry_bound())
        .unwrap();
    for base in BASES {
        match shape.derive_encoding(base) {
            Err(AkitaError::InvalidSetup(message)) => {
                assert!(message.contains("H + 3*B_Q + 2*B_K < P"))
            }
            other => panic!("expected rounded-envelope rejection, got {other:?}"),
        }
    }
    // The adjacent power-of-two geometry remains inside the rounded envelope.
    let mut smaller = custom_shape(PROFILE.coefficient_prime(), interval, 21, 0).unwrap();
    smaller.coefficient_prime = LabiniusCoefficientPrime::P64Offset23703;
    smaller.derive_encoding(LabiniusDigitBase::Bits1).unwrap();
    assert!(custom_shape(LabiniusCoefficientPrime::P64Offset23703, interval, 22, 0).is_err());
}

#[test]
fn nearest_certified_envelope_boundary_for_supported_primes() {
    let challenge = PROFILE.challenge_profile().unwrap();
    let mut largest_p64 = 0;
    // These cover every centered interval and dyadic geometry the D648 cells
    // can admit: e>=33 exceeds every certified norm; M>=2^33 exceeds every
    // certified width; C>=2^25 cannot fit any covered honest response interval.
    for prime in [
        LabiniusCoefficientPrime::P64Offset23703,
        PROFILE.coefficient_prime(),
    ] {
        for bits in 1..=32 {
            let offset = 1i128 << (bits - 1);
            for log_m in 2..=32 {
                for log_c in 0..=24 {
                    let Ok(shape) = super::super::derive_shape(
                        PROFILE,
                        prime,
                        PROFILE.ring_degree(),
                        &challenge,
                        (-offset, offset - 1),
                        log_m + log_c,
                        log_c,
                        0,
                    ) else {
                        continue;
                    };
                    shape
                        .check_parity_no_wrap(
                            shape.honest_quotient_bound(),
                            shape.honest_carry_bound(),
                        )
                        .unwrap();
                    for base in BASES {
                        if bits % base.bits() != 0 {
                            continue;
                        }
                        let encoding = shape.derive_encoding(base).unwrap();
                        if prime == LabiniusCoefficientPrime::P64Offset23703 {
                            largest_p64 = largest_p64.max(encoding.no_wrap_total());
                        }
                    }
                }
            }
        }
    }
    let nearest = custom_shape(
        LabiniusCoefficientPrime::P64Offset23703,
        (-(1i128 << 31), (1i128 << 31) - 1),
        40,
        24,
    )
    .unwrap();
    assert_eq!(nearest.scalars_per_column(), 1 << 16);
    let total = nearest
        .derive_encoding(LabiniusDigitBase::Bits4)
        .unwrap()
        .no_wrap_total();
    assert_eq!(total, 1_824_239_324_833_513_472);
    assert_eq!(total, largest_p64);
    assert!(total < LabiniusCoefficientPrime::P64Offset23703.modulus());
}

#[test]
fn range_and_witness_overflows_reject_without_allocation() {
    let mut shape = LabiniusRootShape::derive(PROFILE, 22, 8, 128).unwrap();
    shape.honest_quotient_bound = u128::MAX;
    assert!(matches!(
        shape.derive_encoding(LabiniusDigitBase::Bits1),
        Err(AkitaError::InvalidSetup(_))
    ));
    let mut shape = LabiniusRootShape::derive(PROFILE, 22, 8, 128).unwrap();
    shape.ring_elements_per_column = usize::MAX;
    assert!(matches!(
        shape.derive_encoding(LabiniusDigitBase::Bits1),
        Err(AkitaError::InvalidSetup(_))
    ));
    let mut shape = LabiniusRootShape::derive(PROFILE, 22, 8, 128).unwrap();
    shape.image_len = usize::MAX;
    assert!(matches!(
        shape.derive_encoding(LabiniusDigitBase::Bits1),
        Err(AkitaError::InvalidSetup(_))
    ));
}

#[test]
fn stable_digit_base_tags_and_bits_have_literal_fixtures() {
    assert_eq!(LabiniusDigitBase::Bits1.tag(), 0);
    assert_eq!(LabiniusDigitBase::Bits2.tag(), 1);
    assert_eq!(LabiniusDigitBase::Bits4.tag(), 2);
    assert_eq!(
        LabiniusDigitBase::from_tag(0).unwrap(),
        LabiniusDigitBase::Bits1
    );
    assert_eq!(
        LabiniusDigitBase::from_tag(1).unwrap(),
        LabiniusDigitBase::Bits2
    );
    assert_eq!(
        LabiniusDigitBase::from_tag(2).unwrap(),
        LabiniusDigitBase::Bits4
    );
    assert_eq!(LabiniusDigitBase::Bits1.bits(), 1);
    assert_eq!(LabiniusDigitBase::Bits2.bits(), 2);
    assert_eq!(LabiniusDigitBase::Bits4.bits(), 4);
}
