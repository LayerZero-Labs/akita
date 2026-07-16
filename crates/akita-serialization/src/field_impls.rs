use std::io::{Read, Write};

use jolt_field::{
    CanonicalField, FieldCore, Fp128, Fp32, Fp64, FpExt2, FpExt2Config, FpExt4, FpExt8,
};

use crate::{AkitaDeserialize, AkitaSerialize, Compress, SerializationError, Valid, Validate};

macro_rules! impl_prime_serialization {
    ($field:ident, $modulus:ty, $bytes:expr, $canonical:ident) => {
        impl<const P: $modulus> Valid for $field<P> {
            fn check(&self) -> Result<(), SerializationError> {
                if self.to_canonical_u128() < P as u128 {
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
                let value: $modulus = self.to_canonical_u128() as $modulus;
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
                Ok(if validate == Validate::Yes {
                    $field::<P>::$canonical(value)
                } else {
                    <$field<P> as CanonicalField>::from_canonical_u128_reduced(value as u128)
                })
            }
        }
    };
}

impl_prime_serialization!(Fp32, u32, 4, from_canonical_u32);
impl_prime_serialization!(Fp64, u64, 8, from_canonical_u64);
impl_prime_serialization!(Fp128, u128, 16, from_canonical_u128);

impl<F, C> Valid for FpExt2<F, C>
where
    F: FieldCore + Valid,
    C: FpExt2Config<F>,
{
    fn check(&self) -> Result<(), SerializationError> {
        self.coeffs[0].check()?;
        self.coeffs[1].check()
    }
}

impl<F, C> AkitaSerialize for FpExt2<F, C>
where
    F: FieldCore + AkitaSerialize,
    C: FpExt2Config<F>,
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
    F: FieldCore + Valid + AkitaDeserialize<Context = ()>,
    C: FpExt2Config<F>,
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
        impl<F: FieldCore + Valid> Valid for $extension<F> {
            fn check(&self) -> Result<(), SerializationError> {
                for coefficient in &self.coeffs {
                    coefficient.check()?;
                }
                Ok(())
            }
        }

        impl<F: FieldCore + AkitaSerialize> AkitaSerialize for $extension<F> {
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
            F: FieldCore + Valid + AkitaDeserialize<Context = ()>,
        {
            type Context = ();

            fn deserialize_with_mode<R: Read>(
                mut reader: R,
                compress: Compress,
                validate: Validate,
                _ctx: &(),
            ) -> Result<Self, SerializationError> {
                let mut coefficients = Vec::with_capacity($degree);
                for _ in 0..$degree {
                    coefficients.push(F::deserialize_with_mode(
                        &mut reader,
                        compress,
                        validate,
                        &(),
                    )?);
                }
                let coefficients: [F; $degree] = coefficients.try_into().map_err(|_| {
                    SerializationError::InvalidData("invalid extension degree".into())
                })?;
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
    use jolt_field::{Fp64, FpExt8, Prime128Offset275, Prime32Offset99, Prime64Offset59};

    use crate::{AkitaDeserialize, AkitaSerialize, Compress, Validate};

    type Base = Fp64<4294967197>;
    type Extension = FpExt8<Base>;

    fn serialized<T: AkitaSerialize>(value: &T) -> Vec<u8> {
        let mut bytes = Vec::new();
        value.serialize_with_mode(&mut bytes, Compress::No).unwrap();
        bytes
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

        let fp128 =
            Prime128Offset275::from_canonical_u128(0x0102_0304_0506_0708_090a_0b0c_0d0e_0f10);
        assert_eq!(
            serialized(&fp128),
            [
                0x10, 0x0f, 0x0e, 0x0d, 0x0c, 0x0b, 0x0a, 0x09, 0x08, 0x07, 0x06, 0x05, 0x04, 0x03,
                0x02, 0x01,
            ]
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
