use super::*;

const P: u128 = u128::MAX - 274;

#[test]
fn projection_bounds_match_independent_integer_arithmetic() {
    let bounds = BinaryProjectionBounds::new(1, 2, 3, P).unwrap();
    assert_eq!(bounds.residual_bound(), 16_211_309_718_240);
    assert_eq!(bounds.projection_bound(), 449_193_104_205_686_897_172_480);
    let larger = BinaryProjectionBounds::new(4096, 256, 3, P).unwrap();
    assert_eq!(larger.residual_bound(), 66_394_525_036_769_280);
    assert_eq!(
        larger.projection_bound(),
        1_839_701_006_388_918_996_628_930_560
    );
    assert_eq!(larger.rank(), 3);
}

#[test]
fn projection_admission_checks_strict_lift_encoding_and_overflow_bounds() {
    let bounds = BinaryProjectionBounds::new(1, 2, 3, P).unwrap();
    let threshold = 2 * bounds.projection_bound();
    assert!(BinaryProjectionBounds::new(1, 2, 3, threshold).is_err());
    assert!(BinaryProjectionBounds::new(1, 2, 3, threshold + 1).is_ok());
    assert!(BinaryProjectionBounds::new(1_000_000_000_000, 2, 3, P).is_err());
    for (rows, columns, rank) in [
        (0, 1, 1),
        (1, 0, 1),
        (1, 1, 0),
        (usize::MAX, 1, 1),
        (1, usize::MAX, 1),
        (1, 1, usize::MAX),
    ] {
        assert!(BinaryProjectionBounds::new(rows, columns, rank, P).is_err());
    }
}

#[test]
fn projection_encoding_rejects_noncanonical_bounds_and_lengths() {
    let bounds = BinaryProjectionBounds::new(1, 2, 3, P).unwrap();
    let q = i128::from(BINARY_SOURCE_Q);
    let max = (bounds.projection_bound() / q as u128) as i128 * q;
    for value in [0, q, -q, max, -max] {
        let bytes = encode_binary_projection(value, &bounds).unwrap();
        assert_eq!(decode_binary_projection(&bytes, &bounds).unwrap(), value);
        assert!(decode_binary_projection(&bytes[..11], &bounds).is_err());
        let mut extra = bytes.to_vec();
        extra.push(0);
        assert!(decode_binary_projection(&extra, &bounds).is_err());
    }
    assert!(encode_binary_projection(max + q, &bounds).is_err());
    let outside = (max + q).to_le_bytes();
    assert!(decode_binary_projection(&outside[..12], &bounds).is_err());
    assert!(encode_binary_projection(i128::MIN, &bounds).is_err());
    let mut nonmultiple = encode_binary_projection(q, &bounds).unwrap();
    nonmultiple[0] += 1;
    assert!(decode_binary_projection(&nonmultiple, &bounds).is_err());
    let mut negative_zero = [0; 12];
    negative_zero[11] = 0x80;
    assert!(decode_binary_projection(&negative_zero, &bounds).is_err());
    // A bounded sign flip and addition of q are valid integer messages; the
    // authenticated Stage-2 equation must reject those changes in a proof.
    assert!(
        decode_binary_projection(&encode_binary_projection(-q, &bounds).unwrap(), &bounds).is_ok()
    );
    assert!(
        decode_binary_projection(&encode_binary_projection(2 * q, &bounds).unwrap(), &bounds)
            .is_ok()
    );
}

#[test]
fn projection_twelve_bytes_cover_every_current_carry_admitted_geometry() {
    let max = BinaryProjectionBounds::new(8192, 256, 3, P).unwrap();
    assert_eq!(max.residual_bound(), 132_788_824_280_985_600);
    assert_eq!(
        max.projection_bound(),
        3_679_395_756_376_625_911_509_811_200
    );
    assert!(2 * max.projection_bound() < P);
    assert!(max.projection_bound() < 1u128 << 95);
    for row_log in 0..=13 {
        for column_log in 0..=8 {
            let (rows, columns) = (1 << row_log, 1 << column_log);
            crate::binary_source::BinaryEvaluationBounds::new(4 * rows, columns, P).unwrap();
            let bounds = BinaryProjectionBounds::new(rows, columns, 3, P).unwrap();
            assert!(bounds.projection_bound() <= max.projection_bound());
        }
    }
    // The next powers of two already exceed a fixed positive carry reach.
    assert!(crate::binary_source::BinaryEvaluationBounds::new(4 * 16384, 1, P).is_err());
    assert!(crate::binary_source::BinaryEvaluationBounds::new(4, 512, P).is_err());
}
