use super::*;

#[test]
fn randomness_columns_match_the_headline_sizes() {
    // Pad commitment: rank 3 over R_256 plus the 64-coefficient row L.
    // Residual commitment: the same published image without row L.
    let pad = HiddenImage {
        rows: 3,
        ring_dim: 256,
        modulus: SisModulusProfileId::Q128Offset6A23801F,
        hidden: HiddenRows::StrideZero,
    };
    let residual = HiddenImage {
        hidden: HiddenRows::None,
        ..pad
    };
    let alphabet = DigitAlphabet { log_basis: 4 };
    assert_eq!(hiding_randomness_columns(pad, alphabet).unwrap(), 105);
    assert_eq!(hiding_randomness_columns(residual, alphabet).unwrap(), 97);
}

#[test]
fn pad_sizing_includes_the_stride_zero_image() {
    let pad = HiddenImage {
        rows: 8,
        ring_dim: 128,
        modulus: SisModulusProfileId::Q128Offset6A23801F,
        hidden: HiddenRows::StrideZero,
    };
    let alphabet = DigitAlphabet { log_basis: 4 };
    assert_eq!(hiding_randomness_columns(pad, alphabet).unwrap(), 273);
    assert_eq!(
        hiding_randomness_columns(HiddenImage { rows: 9, ..pad }, alphabet).unwrap(),
        305
    );
    assert_eq!(
        hiding_randomness_columns(
            HiddenImage {
                hidden: HiddenRows::None,
                ..pad
            },
            alphabet,
        )
        .unwrap(),
        257
    );
}

#[test]
fn headline_slice_needs_forty_five_columns() {
    let image = HiddenImage {
        rows: 1,
        ring_dim: 64,
        modulus: SisModulusProfileId::Q128Offset6A23801F,
        hidden: HiddenRows::None,
    };
    assert_eq!(
        hiding_randomness_columns(image, DigitAlphabet { log_basis: 3 }).unwrap(),
        45
    );
}

#[test]
fn sizing_is_minimal_for_every_profile_and_alphabet() {
    for modulus in [
        SisModulusProfileId::Q32Offset99,
        SisModulusProfileId::Q64Offset59,
        SisModulusProfileId::Q128Offset6A23801F,
    ] {
        for (hidden, extra) in [(HiddenRows::None, 0), (HiddenRows::StrideZero, 64)] {
            for ring_dim in [64, 128, 256] {
                for rows in [0, 1, 3, 8] {
                    for log_basis in 1..=8 {
                        let columns = hiding_randomness_columns(
                            HiddenImage {
                                rows,
                                ring_dim,
                                modulus,
                                hidden,
                            },
                            DigitAlphabet { log_basis },
                        )
                        .unwrap();
                        // Compute the entropy inequality independently in u128.
                        let field_bits = u128::BITS - (modulus.modulus() - 1).leading_zeros();
                        let need = (rows * ring_dim + extra) as u128 * u128::from(field_bits)
                            + 2 * 128
                            + 8;
                        let per_column = ring_dim as u128 * u128::from(log_basis);
                        assert!(columns as u128 * per_column >= need);
                        assert!((columns - 1) as u128 * per_column < need);
                    }
                }
            }
        }
    }
}

#[test]
fn rejects_invalid_dimensions_alphabets_and_overflow() {
    let image = HiddenImage {
        rows: 1,
        ring_dim: 64,
        modulus: SisModulusProfileId::Q64Offset59,
        hidden: HiddenRows::None,
    };
    let alphabet = DigitAlphabet { log_basis: 4 };
    for log_basis in [0, 9, u32::MAX] {
        assert!(hiding_randomness_columns(image, DigitAlphabet { log_basis }).is_err());
    }
    for invalid in [
        HiddenImage {
            ring_dim: 0,
            ..image
        },
        HiddenImage {
            rows: usize::MAX,
            ..image
        },
        // Overflow while adding the hidden output coefficients.
        HiddenImage {
            rows: usize::MAX - 63,
            ring_dim: 1,
            hidden: HiddenRows::StrideZero,
            ..image
        },
        // Overflow while converting coefficients to bits.
        HiddenImage {
            rows: usize::MAX / 64 + 1,
            ring_dim: 1,
            ..image
        },
        // Overflow only when adding the security margin.
        HiddenImage {
            rows: usize::MAX / 64 - 1,
            ring_dim: 1,
            ..image
        },
        // Overflow in per-column entropy even though image rank is zero.
        HiddenImage {
            rows: 0,
            ring_dim: usize::MAX,
            ..image
        },
    ] {
        assert!(hiding_randomness_columns(invalid, alphabet).is_err());
    }
}
