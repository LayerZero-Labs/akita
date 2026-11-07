use super::*;

#[test]
fn encoding_constants_and_byte_order() {
    assert_eq!(IMAGE_DIGIT_WEIGHT_SUM, ((1 << 24) - 1) / 7);
    assert_eq!(IMAGE_MIN, -9_586_980);
    assert_eq!(IMAGE_MAX, 7_190_235);
    assert_eq!(IMAGE_OFFSET, 0x92_49_24);
    assert_eq!(Q24_CENTERED_BOUND, 7_126_704);
    assert_eq!(core::mem::size_of::<Q24Image>(), 3);
    assert_eq!(
        Q24Image::from_centered(0).unwrap().bytes(),
        [0x24, 0x49, 0x92]
    );
    assert_eq!(Q24Image::from_centered(IMAGE_MIN).unwrap().bytes(), [0; 3]);
    assert_eq!(
        Q24Image::from_centered(IMAGE_MAX).unwrap().bytes(),
        [255; 3]
    );
    assert_eq!(
        Q24Image::from_bytes([0x56, 0x34, 0x12]).offset_word(),
        0x12_34_56
    );
}

#[test]
fn all_offset_words_obey_digit_sum_and_parity_identities() {
    // Exhaust the entire enforced interval, including images outside Jq.
    for word in 0..1u32 << 24 {
        let image = Q24Image::from_offset_word(word).unwrap();
        let digits = image.digits();
        let sum = digits
            .into_iter()
            .enumerate()
            .map(|(i, digit)| i32::from(digit) * (1 << (3 * i)))
            .sum::<i32>();
        assert_eq!(sum, image.centered());
        assert_eq!(image.parity(), sum.rem_euclid(2) as u8);
        assert_eq!(image.parity(), (digits[0] & 1) as u8);
        assert_eq!(image.offset_word(), word);
    }
}

#[test]
fn residue_and_centered_conversions_preserve_representatives() {
    for residue in [
        0,
        1,
        Q24_CENTERED_BOUND as u32,
        Q24_CENTERED_BOUND as u32 + 1,
        Q24 - 1,
    ]
    .into_iter()
    .chain((0..Q24).step_by(997))
    {
        let image = Q24Image::from_residue(residue).unwrap();
        assert_eq!(image.residue(), residue);
        assert!((-Q24_CENTERED_BOUND..=Q24_CENTERED_BOUND).contains(&image.centered()));
        assert_eq!(image.canonical_centered(), image.centered());
        assert_eq!(Q24Image::from_centered(image.centered()).unwrap(), image);
        assert_eq!(Q24Image::from_digits(image.digits()).unwrap(), image);
        assert_eq!(Q24Image::from_bytes(image.bytes()), image);
    }
    for value in [
        IMAGE_MIN,
        -Q24_CENTERED_BOUND - 1,
        Q24_CENTERED_BOUND + 1,
        IMAGE_MAX,
    ] {
        let image = Q24Image::from_centered(value).unwrap();
        assert_eq!(image.centered(), value);
        assert_eq!(image.residue(), value.rem_euclid(Q24 as i32) as u32);
        assert_ne!(image.centered(), image.canonical_centered());
        assert_eq!(
            image.canonical_centered().rem_euclid(Q24 as i32),
            image.residue() as i32
        );
    }
}

#[test]
fn malformed_encodings_return_typed_errors() {
    for value in [i32::MIN, IMAGE_MIN - 1, IMAGE_MAX + 1, i32::MAX] {
        assert!(matches!(
            Q24Image::from_centered(value),
            Err(AkitaError::InvalidInput(_))
        ));
    }
    for residue in [Q24, u32::MAX] {
        assert!(matches!(
            Q24Image::from_residue(residue),
            Err(AkitaError::InvalidInput(_))
        ));
    }
    for word in [1 << 24, u32::MAX] {
        assert!(matches!(
            Q24Image::from_offset_word(word),
            Err(AkitaError::InvalidInput(_))
        ));
    }
    for index in 0..8 {
        for invalid in [i8::MIN, -5, 4, i8::MAX] {
            let mut digits = [0; 8];
            digits[index] = invalid;
            assert!(matches!(
                Q24Image::from_digits(digits),
                Err(AkitaError::InvalidInput(_))
            ));
        }
    }
}

#[test]
fn bulk_slices_keep_three_byte_storage_and_validate_lengths() {
    let values = [
        IMAGE_MIN,
        -Q24_CENTERED_BOUND,
        -1,
        0,
        1,
        Q24_CENTERED_BOUND,
        IMAGE_MAX,
    ];
    let mut bytes = [0u8; 21];
    encode_centered_slice(&values, &mut bytes).unwrap();
    let mut decoded = [0i32; 7];
    decode_centered_slice(&bytes, &mut decoded).unwrap();
    assert_eq!(decoded, values);
    let mut residues = [0u32; 7];
    decode_residue_slice(&bytes, &mut residues).unwrap();
    assert_eq!(
        residues,
        values.map(|value| value.rem_euclid(Q24 as i32) as u32)
    );
    encode_residue_slice(&residues, &mut bytes).unwrap();
    decode_centered_slice(&bytes, &mut decoded).unwrap();
    assert!(decoded
        .iter()
        .all(|value| (-Q24_CENTERED_BOUND..=Q24_CENTERED_BOUND).contains(value)));
    let mut decoded_residues = [0u32; 7];
    decode_residue_slice(&bytes, &mut decoded_residues).unwrap();
    assert_eq!(decoded_residues, residues);

    assert_eq!(encode_centered_slice(&[], &mut []), Ok(()));
    assert_eq!(encode_residue_slice(&[], &mut []), Ok(()));
    assert_eq!(decode_centered_slice(&[], &mut []), Ok(()));
    assert_eq!(decode_residue_slice(&[], &mut []), Ok(()));
    for bad_length in [0, 1, 2, 3, 20] {
        let mut bad_bytes = vec![0; bad_length];
        assert!(matches!(
            encode_centered_slice(&values, &mut bad_bytes),
            Err(AkitaError::InvalidSize { .. })
        ));
        assert!(matches!(
            encode_residue_slice(&residues, &mut bad_bytes),
            Err(AkitaError::InvalidSize { .. })
        ));
        assert!(matches!(
            decode_centered_slice(&bad_bytes, &mut decoded),
            Err(AkitaError::InvalidSize { .. })
        ));
        assert!(matches!(
            decode_residue_slice(&bad_bytes, &mut decoded_residues),
            Err(AkitaError::InvalidSize { .. })
        ));
    }
    assert!(encode_centered_slice(&[IMAGE_MIN - 1], &mut [0; 3]).is_err());
    assert!(encode_residue_slice(&[Q24], &mut [0; 3]).is_err());
    assert!(matches!(
        validate_packed_length(usize::MAX, 0),
        Err(AkitaError::InvalidInput(_))
    ));
}
