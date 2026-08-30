use std::io::{Read, Write};

use jolt_field::{CanonicalEncoding, Ext2Config, Field, Fp128, Fp32, Fp64, FpExt2, FpExt4, FpExt8};

use crate::{AkitaDeserialize, AkitaSerialize, Compress, SerializationError, Valid, Validate};

macro_rules! impl_prime_serialization {
    ($field:ident, $modulus:ty, $bytes:expr, $to_canonical:ident) => {
        impl<const P: $modulus> Valid for $field<P> {
            fn check(&self) -> Result<(), SerializationError> {
                if (self.$to_canonical() as u128) < P as u128 {
                    Ok(())
                } else {
                    Err(SerializationError::InvalidData(
                        concat!(stringify!($field), " out of range").into(),
                    ))
                }
            }
        }

        impl<const P: $modulus> AkitaSerialize for $field<P> {
            fn serialize_with_mode<W: Write>(
                &self,
                mut writer: W,
                _compress: Compress,
            ) -> Result<(), SerializationError> {
                let value: $modulus = self.$to_canonical();
                value.serialize_with_mode(&mut writer, Compress::No)
            }

            fn serialized_size(&self, _compress: Compress) -> usize {
                $bytes
            }
        }

        impl<const P: $modulus> AkitaDeserialize for $field<P> {
            type Context = ();

            fn deserialize_with_mode<R: Read>(
                mut reader: R,
                _compress: Compress,
                validate: Validate,
                _ctx: &(),
            ) -> Result<Self, SerializationError> {
                let value =
                    <$modulus>::deserialize_with_mode(&mut reader, Compress::No, validate, &())?;
                if validate == Validate::Yes && value >= P {
                    return Err(SerializationError::InvalidData(
                        concat!(stringify!($field), " out of range").into(),
                    ));
                }
                if validate == Validate::Yes {
                    <$field<P> as CanonicalEncoding>::from_u128_checked(value as u128).ok_or_else(
                        || {
                            SerializationError::InvalidData(
                                concat!(stringify!($field), " out of range").into(),
                            )
                        },
                    )
                } else {
                    Ok(<$field<P> as CanonicalEncoding>::from_u128_reduced(
                        value as u128,
                    ))
                }
            }
        }
    };
}

impl_prime_serialization!(Fp32, u32, 4, to_canonical_u32);
impl_prime_serialization!(Fp64, u64, 8, to_canonical_u64);
impl_prime_serialization!(Fp128, u128, 16, to_canonical_u128);

impl<F, C> Valid for FpExt2<F, C>
where
    F: Field + Valid,
    C: Ext2Config<F>,
{
    fn check(&self) -> Result<(), SerializationError> {
        self.coeffs[0].check()?;
        self.coeffs[1].check()
    }
}

impl<F, C> AkitaSerialize for FpExt2<F, C>
where
    F: Field + AkitaSerialize,
    C: Ext2Config<F>,
{
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.coeffs[0].serialize_with_mode(&mut writer, compress)?;
        self.coeffs[1].serialize_with_mode(&mut writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.coeffs[0].serialized_size(compress) + self.coeffs[1].serialized_size(compress)
    }
}

impl<F, C> AkitaDeserialize for FpExt2<F, C>
where
    F: Field + Valid + AkitaDeserialize<Context = ()>,
    C: Ext2Config<F>,
{
    type Context = ();

    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &(),
    ) -> Result<Self, SerializationError> {
        let c0 = F::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let c1 = F::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let value = Self::new(c0, c1);
        if validate == Validate::Yes {
            value.check()?;
        }
        Ok(value)
    }
}

macro_rules! impl_extension_serialization {
    ($extension:ident, $degree:expr) => {
        impl<F: Field + Valid> Valid for $extension<F> {
            fn check(&self) -> Result<(), SerializationError> {
                for coefficient in &self.coeffs {
                    coefficient.check()?;
                }
                Ok(())
            }
        }

        impl<F: Field + AkitaSerialize> AkitaSerialize for $extension<F> {
            fn serialize_with_mode<W: Write>(
                &self,
                mut writer: W,
                compress: Compress,
            ) -> Result<(), SerializationError> {
                for coefficient in &self.coeffs {
                    coefficient.serialize_with_mode(&mut writer, compress)?;
                }
                Ok(())
            }

            fn serialized_size(&self, compress: Compress) -> usize {
                self.coeffs
                    .iter()
                    .map(|coefficient| coefficient.serialized_size(compress))
                    .sum()
            }
        }

        impl<F> AkitaDeserialize for $extension<F>
        where
            F: Field + Valid + AkitaDeserialize<Context = ()>,
        {
            type Context = ();

            fn deserialize_with_mode<R: Read>(
                mut reader: R,
                compress: Compress,
                validate: Validate,
                _ctx: &(),
            ) -> Result<Self, SerializationError> {
                let mut coefficients = [F::zero(); $degree];
                for coefficient in &mut coefficients {
                    *coefficient = F::deserialize_with_mode(&mut reader, compress, validate, &())?;
                }
                let value = Self::new(coefficients);
                if validate == Validate::Yes {
                    value.check()?;
                }
                Ok(value)
            }
        }
    };
}

impl_extension_serialization!(FpExt4, 4);
impl_extension_serialization!(FpExt8, 8);

#[cfg(test)]
mod tests {
    use jolt_field::{CanonicalEncoding, Ext2, Ring};
    use jolt_field::{
        Fp64, FpExt4, FpExt8, Prime128Offset275, Prime128OffsetA7F7, Prime32Offset99,
        Prime64Offset59,
    };

    use crate::{AkitaDeserialize, AkitaSerialize, Compress, Validate};

    type Base = Fp64<4294967197>;
    type Extension = FpExt8<Base>;

    fn serialized<T: AkitaSerialize>(value: &T) -> Vec<u8> {
        let mut bytes = Vec::new();
        value.serialize_with_mode(&mut bytes, Compress::No).unwrap();
        bytes
    }

    fn decode_fixture_hex(encoded: &str) -> Vec<u8> {
        encoded
            .as_bytes()
            .chunks_exact(2)
            .map(|pair| {
                u8::from_str_radix(std::str::from_utf8(pair).expect("fixture is ASCII"), 16)
                    .expect("fixture is hexadecimal")
            })
            .collect()
    }

    #[test]
    fn prime_field_wire_bytes_are_stable() {
        let fp32 = Prime32Offset99::from_canonical_u32(0x0102_0304);
        assert_eq!(serialized(&fp32), [0x04, 0x03, 0x02, 0x01]);

        let fp64 = Prime64Offset59::from_canonical_u64(0x0102_0304_0506_0708);
        assert_eq!(
            serialized(&fp64),
            [0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01]
        );

        let fp128 = Prime128Offset275::from_u128_checked(0x0102_0304_0506_0708_090a_0b0c_0d0e_0f10)
            .unwrap();
        assert_eq!(
            serialized(&fp128),
            [
                0x10, 0x0f, 0x0e, 0x0d, 0x0c, 0x0b, 0x0a, 0x09, 0x08, 0x07, 0x06, 0x05, 0x04, 0x03,
                0x02, 0x01,
            ]
        );
    }

    #[test]
    fn pre_cutover_scalar_and_extension_fixtures_are_stable() {
        let extension2 = Ext2::<Prime64Offset59>::new(
            Prime64Offset59::from_u64(1),
            Prime64Offset59::from_u64(2),
        );
        let extension4 = FpExt4::new([
            Prime32Offset99::from_u64(1),
            Prime32Offset99::from_u64(2),
            Prime32Offset99::from_u64(3),
            Prime32Offset99::from_u64(4),
        ]);
        let extension8 = FpExt8::new([
            Prime32Offset99::from_u64(1),
            Prime32Offset99::from_u64(2),
            Prime32Offset99::from_u64(3),
            Prime32Offset99::from_u64(4),
            Prime32Offset99::from_u64(5),
            Prime32Offset99::from_u64(6),
            Prime32Offset99::from_u64(7),
            Prime32Offset99::from_u64(8),
        ]);
        let fixtures = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/jolt-field-cutover/fields.txt"
        ));
        let actual = [
            ("prime32_zero", serialized(&Prime32Offset99::from_u64(0))),
            ("prime32_one", serialized(&Prime32Offset99::from_u64(1))),
            (
                "prime32_max",
                serialized(&Prime32Offset99::from_u64(0xffff_ff9c)),
            ),
            (
                "prime32_random",
                serialized(&Prime32Offset99::from_u64(0x5e95_dc11)),
            ),
            ("prime64_zero", serialized(&Prime64Offset59::from_u64(0))),
            ("prime64_one", serialized(&Prime64Offset59::from_u64(1))),
            (
                "prime64_max",
                serialized(&Prime64Offset59::from_u64(0xffff_ffff_ffff_ffc4)),
            ),
            (
                "prime64_random",
                serialized(&Prime64Offset59::from_u64(0x03db_45b0_675b_9f54)),
            ),
            (
                "prime128_a7f7_zero",
                serialized(&Prime128OffsetA7F7::from_u64(0)),
            ),
            (
                "prime128_a7f7_one",
                serialized(&Prime128OffsetA7F7::from_u64(1)),
            ),
            (
                "prime128_a7f7_max",
                serialized(
                    &Prime128OffsetA7F7::from_u128_checked(
                        0xffff_ffff_ffff_ffff_ffff_ffff_0000_5808,
                    )
                    .unwrap(),
                ),
            ),
            (
                "prime128_a7f7_random_a",
                serialized(
                    &Prime128OffsetA7F7::from_u128_checked(
                        0x59c9_f049_a454_d1c9_2917_8971_d900_881d,
                    )
                    .unwrap(),
                ),
            ),
            (
                "prime128_a7f7_random_b",
                serialized(
                    &Prime128OffsetA7F7::from_u128_checked(
                        0x97ca_5c79_1111_19c2_d65b_7558_6cc1_d87a,
                    )
                    .unwrap(),
                ),
            ),
            ("fp_ext2_prime64", serialized(&extension2)),
            ("fp_ext4_prime32", serialized(&extension4)),
            ("fp_ext8_prime32", serialized(&extension8)),
        ];
        let expected = fixtures
            .lines()
            .map(|line| line.split_once('=').expect("labeled field fixture"))
            .collect::<Vec<_>>();
        assert_eq!(actual.len(), expected.len());
        for ((label, bytes), (expected_label, expected_hex)) in actual.iter().zip(expected) {
            assert_eq!(*label, expected_label);
            let encoded = bytes
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>();
            assert_eq!(encoded, expected_hex, "fixture {label}");
        }
    }

    #[test]
    fn pre_cutover_malformed_decoding_behavior_is_stable() {
        let fixtures = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/jolt-field-cutover/malformed.txt"
        ));
        let fixture = |name: &str| {
            let (_, encoded) = fixtures
                .lines()
                .filter_map(|line| line.split_once('='))
                .find(|(label, _)| *label == name)
                .unwrap_or_else(|| panic!("missing malformed fixture {name}"));
            decode_fixture_hex(encoded)
        };

        let invalid_scalar = fixture("invalid_prime128_a7f7_modulus");
        assert!(Prime128OffsetA7F7::deserialize_with_mode(
            &invalid_scalar[..],
            Compress::No,
            Validate::Yes,
            &(),
        )
        .is_err());
        let scalar = Prime128OffsetA7F7::deserialize_with_mode(
            &invalid_scalar[..],
            Compress::No,
            Validate::No,
            &(),
        )
        .unwrap();
        assert_eq!(
            serialized(&scalar),
            fixture("unvalidated_prime128_a7f7_modulus")
        );

        let invalid_ext2 = fixture("invalid_fp_ext2_prime64_second_modulus");
        assert!(Ext2::<Prime64Offset59>::deserialize_with_mode(
            &invalid_ext2[..],
            Compress::No,
            Validate::Yes,
            &(),
        )
        .is_err());
        let extension2 = Ext2::<Prime64Offset59>::deserialize_with_mode(
            &invalid_ext2[..],
            Compress::No,
            Validate::No,
            &(),
        )
        .unwrap();
        assert_eq!(
            serialized(&extension2),
            fixture("unvalidated_fp_ext2_prime64_second_modulus")
        );

        let invalid_ext8 = fixture("invalid_fp_ext8_prime32_fifth_modulus");
        assert!(FpExt8::<Prime32Offset99>::deserialize_with_mode(
            &invalid_ext8[..],
            Compress::No,
            Validate::Yes,
            &(),
        )
        .is_err());
        let extension8 = FpExt8::<Prime32Offset99>::deserialize_with_mode(
            &invalid_ext8[..],
            Compress::No,
            Validate::No,
            &(),
        )
        .unwrap();
        assert_eq!(
            serialized(&extension8),
            fixture("unvalidated_fp_ext8_prime32_fifth_modulus")
        );
    }

    #[test]
    fn fp_ext8_serialization_is_coefficient_ordered() {
        let value = Extension::new(std::array::from_fn(|index| {
            Base::from_u64(index as u64 + 1)
        }));
        let mut bytes = Vec::new();
        value.serialize_with_mode(&mut bytes, Compress::No).unwrap();

        let expected = value
            .coeffs
            .iter()
            .flat_map(|coefficient| {
                let mut coefficient_bytes = Vec::new();
                coefficient
                    .serialize_with_mode(&mut coefficient_bytes, Compress::No)
                    .unwrap();
                coefficient_bytes
            })
            .collect::<Vec<_>>();

        assert_eq!(value.serialized_size(Compress::No), expected.len());
        assert_eq!(bytes, expected);
        assert_eq!(
            Extension::deserialize_with_mode(&bytes[..], Compress::No, Validate::Yes, &()).unwrap(),
            value
        );
    }
}
