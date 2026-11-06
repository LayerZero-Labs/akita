//! Akita's wire format for Jolt's cyclotomic ring elements: the `D`
//! coefficients in ascending degree order, each in its field encoding.

use std::io::{Read, Write};

use jolt_algebra::{CyclotomicRing, Field};

use crate::{AkitaDeserialize, AkitaSerialize, Compress, SerializationError, Valid, Validate};

impl<F: Field + Valid, const D: usize> Valid for CyclotomicRing<F, D> {
    fn check(&self) -> Result<(), SerializationError> {
        self.coefficients().iter().try_for_each(Valid::check)
    }
}

impl<F: Field + AkitaSerialize, const D: usize> AkitaSerialize for CyclotomicRing<F, D> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        for coefficient in self.coefficients() {
            coefficient.serialize_with_mode(&mut writer, compress)?;
        }
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.coefficients()
            .iter()
            .map(|coefficient| coefficient.serialized_size(compress))
            .sum()
    }
}

impl<F: Field + Valid + AkitaDeserialize<Context = ()>, const D: usize> AkitaDeserialize
    for CyclotomicRing<F, D>
{
    type Context = ();

    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &(),
    ) -> Result<Self, SerializationError> {
        let mut coeffs = [F::zero(); D];
        for coefficient in &mut coeffs {
            *coefficient = F::deserialize_with_mode(&mut reader, compress, validate, &())?;
        }
        let out = Self::from_coefficients(coeffs);
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}
