use super::*;
use crate::sis::labinius::LabiniusRootProfile;

const PROFILE: LabiniusRootProfile = LabiniusRootProfile::D648Q25BoundedW46;
const P64_OFFSET_59: u128 = (1 << 64) - 59;
const P128_OFFSET_275: u128 = u128::MAX - 274;

fn rejection(result: Result<LabiniusRootEncoding, AkitaError>, message: &str) {
    match result {
        Err(AkitaError::InvalidSetup(actual)) => assert!(actual.contains(message), "{actual}"),
        other => panic!("expected InvalidSetup containing {message:?}, got {other:?}"),
    }
}

/// Both proof primes of the design admit the sample geometry with the same
/// encoding; lengths are compared with their definitions.
#[test]
fn sample_encoding_has_three_digits_and_plain_bit_width_ranges() {
    let shape = LabiniusRootShape::derive(PROFILE, 22, 8, 128).unwrap();
    let encoding = shape.derive_encoding(P128_OFFSET_275).unwrap();
    assert_eq!(shape.derive_encoding(P64_OFFSET_59).unwrap(), encoding);
    assert_eq!(encoding.response_interval(), (-2_184, 1_911));
    assert_eq!(encoding.response_digit_count(), 3);
    assert_eq!(encoding.response_digit_slots(), 4);
    assert_eq!(encoding.padded_coefficient_len(), 1_024);
    assert_eq!(encoding.response_table_len(), 4_096 * 1_024 * 4);
    assert_eq!(encoding.response_table_log_len(), 24);
    // Seven digits cover a canonical residue; the eighth slot is padding.
    assert_eq!(encoding.image_digit_count(), 7);
    assert_eq!(encoding.image_digit_slots(), 8);
    assert_eq!(encoding.image_offset(), 8 * (16u128.pow(7) - 1) / 15);
    assert_eq!(1 << encoding.image_table_log_len(), 8 * 1_024 * 256 * 2);
    assert_eq!(1 << encoding.prime_table_log_len(), 1_024 * 256);
    assert_eq!(encoding.a_carry_len(), 2 * 648);
    assert_eq!(encoding.parity_quotient_len(), 161);
    assert_eq!(encoding.parity_carry_len(), 162);
    for (range, honest) in [
        (encoding.quotient(), shape.honest_quotient_bound()),
        (encoding.carry(), shape.honest_carry_bound()),
        (encoding.a_carry(), shape.honest_a_carry_bound()),
    ] {
        // The narrowest two's-complement width containing `[-honest, honest]`.
        let (lower, upper) = range.interval();
        assert_eq!(lower, -upper - 1);
        assert_eq!(range.magnitude(), lower.unsigned_abs());
        assert!(honest <= upper.unsigned_abs() && upper.unsigned_abs() / 2 < honest);
    }
    assert_eq!(
        (
            encoding.quotient().bits(),
            encoding.carry().bits(),
            encoding.a_carry().bits()
        ),
        (35, 35, 36)
    );
}

#[test]
fn signed_range_is_the_narrowest_cover_and_rejects_past_i128() {
    for bound in [
        0u128,
        1,
        2,
        3,
        4,
        255,
        256,
        257,
        (1 << 126) + 1,
        (1 << 127) - 1,
    ] {
        let range = LabiniusSignedRange::covering(bound).unwrap();
        let (lower, upper) = range.interval();
        assert_eq!(lower.unsigned_abs(), 1 << (range.bits() - 1));
        assert!(upper.unsigned_abs() >= bound);
        assert!(range.bits() == 1 || (upper.unsigned_abs() - 1) / 2 < bound);
    }
    assert_eq!(
        LabiniusSignedRange::covering(0).unwrap().interval(),
        (-1, 0)
    );
    assert!(LabiniusSignedRange::covering(1 << 127).is_err());
}

/// The unit condition rejects both retired proof primes before any no-wrap
/// arithmetic, and a prime that passes it can still be too small to lift the
/// commitment rows.
#[test]
fn proof_prime_admission_rejects_non_units_and_wrapping_primes() {
    let shape = LabiniusRootShape::derive(PROFILE, 22, 8, 128).unwrap();
    for retired in [(1 << 64) - 23_703, u128::MAX - (1 << 32) + 22_538] {
        rejection(
            shape.derive_encoding(retired),
            "does not make challenge differences units",
        );
    }
    rejection(shape.derive_encoding(3), "must be coprime to three");
    // 2^59 - 55 is prime with residue degree 81 modulo 243.
    let small = (1 << 59) - 55;
    rejection(
        shape.derive_encoding(small),
        "3*m*D*(q-1)*B_z + C*Gamma*B_T + q*B_K < P",
    );
    LabiniusRootShape::derive(PROFILE, 12, 4, 128)
        .unwrap()
        .derive_encoding(small)
        .unwrap();
    // The longest certified column still lifts to both proof primes' widths
    // only when the prime is large enough.
    let longest = LabiniusRootShape::derive(PROFILE, 44, 0, 128).unwrap();
    longest.derive_encoding(P128_OFFSET_275).unwrap();
    rejection(
        longest.derive_encoding(P64_OFFSET_59),
        "3*m*D*(q-1)*B_z + C*Gamma*B_T + q*B_K < P",
    );
}
