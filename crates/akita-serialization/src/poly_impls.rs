//! Akita's wire format for Jolt's shared univariate polynomial types.

use std::io::{Read, Write};

use jolt_field::Field;
use jolt_poly::{CompressedPoly, OmittedConstantPoly, UnivariatePoly};

use crate::{AkitaDeserialize, AkitaSerialize, Compress, SerializationError, Valid, Validate};

impl<F: Field + Valid> Valid for UnivariatePoly<F> {
    fn check(&self) -> Result<(), SerializationError> {
        self.coefficients().iter().try_for_each(Valid::check)
    }
}

impl<F: Field + AkitaSerialize> AkitaSerialize for UnivariatePoly<F> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        (self.coefficients().len() as u64).serialize_with_mode(&mut writer, compress)?;
        for coefficient in self.coefficients() {
            coefficient.serialize_with_mode(&mut writer, compress)?;
        }
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        8 + self
            .coefficients()
            .iter()
            .map(|coefficient| coefficient.serialized_size(compress))
            .sum::<usize>()
    }
}

impl<F: Field + Valid + AkitaDeserialize<Context = ()>> AkitaDeserialize for UnivariatePoly<F> {
    type Context = ();

    fn deserialize_with_mode<R: Read>(
        reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &(),
    ) -> Result<Self, SerializationError> {
        let coefficients = Vec::<F>::deserialize_with_mode(reader, compress, validate, &())?;
        let poly = Self::new(coefficients);
        if validate == Validate::Yes {
            poly.check()?;
        }
        Ok(poly)
    }
}

impl<F: Field + Valid> Valid for CompressedPoly<F> {
    fn check(&self) -> Result<(), SerializationError> {
        if self.is_empty() {
            return Err(SerializationError::InvalidData(
                "compressed polynomial has no constant coefficient".to_string(),
            ));
        }
        self.coeffs_except_linear_term()
            .iter()
            .try_for_each(Valid::check)
    }
}

impl<F: Field + AkitaSerialize> AkitaSerialize for CompressedPoly<F> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        for coefficient in self.coeffs_except_linear_term() {
            coefficient.serialize_with_mode(&mut writer, compress)?;
        }
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.coeffs_except_linear_term()
            .iter()
            .map(|coefficient| coefficient.serialized_size(compress))
            .sum()
    }
}

impl<F: Field + Valid + AkitaDeserialize<Context = ()>> AkitaDeserialize for CompressedPoly<F> {
    /// Number of transmitted coefficients, including the constant coefficient.
    type Context = usize;

    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        degree: &usize,
    ) -> Result<Self, SerializationError> {
        // A compressed round always needs c0 to recover its omitted linear term.
        // Reject the empty shape in both validation modes, before any allocation.
        if *degree == 0 {
            return Err(SerializationError::InvalidData(
                "compressed polynomial has no constant coefficient".to_string(),
            ));
        }
        let mut coefficients = Vec::new();
        coefficients.try_reserve_exact(*degree).map_err(|_| {
            SerializationError::InvalidData("compressed polynomial allocation failed".to_string())
        })?;
        for _ in 0..*degree {
            coefficients.push(F::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                &(),
            )?);
        }
        let poly = Self::new(coefficients);
        if validate == Validate::Yes {
            poly.check()?;
        }
        Ok(poly)
    }
}

impl<F: Field + Valid> Valid for OmittedConstantPoly<F> {
    fn check(&self) -> Result<(), SerializationError> {
        self.coefficients().iter().try_for_each(Valid::check)
    }
}

impl<F: Field + AkitaSerialize> AkitaSerialize for OmittedConstantPoly<F> {
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

impl<F: Field + Valid + AkitaDeserialize<Context = ()>> AkitaDeserialize
    for OmittedConstantPoly<F>
{
    /// Number of nonconstant coefficients transmitted; zero is valid.
    type Context = usize;

    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        degree: &usize,
    ) -> Result<Self, SerializationError> {
        let mut coefficients = Vec::new();
        coefficients.try_reserve_exact(*degree).map_err(|_| {
            SerializationError::InvalidData("normalized polynomial allocation failed".to_string())
        })?;
        for _ in 0..*degree {
            coefficients.push(F::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                &(),
            )?);
        }
        let poly = Self::new(coefficients);
        if validate == Validate::Yes {
            poly.check()?;
        }
        Ok(poly)
    }
}

#[cfg(test)]
mod tests {
    use jolt_field::{Prime32Offset99, Ring};

    use super::*;

    type F = Prime32Offset99;

    #[test]
    fn univariate_wire_keeps_vector_length_prefix() {
        let poly = UnivariatePoly::new(vec![F::from_u64(1), F::from_u64(2)]);
        let mut bytes = Vec::new();
        poly.serialize_compressed(&mut bytes).unwrap();
        assert_eq!(
            bytes,
            [2_u64.to_le_bytes().as_slice(), &[1, 0, 0, 0, 2, 0, 0, 0]].concat()
        );
        assert_eq!(poly.compressed_size(), bytes.len());
        assert_eq!(
            UnivariatePoly::<F>::deserialize_compressed_exact(&bytes, &()).unwrap(),
            poly
        );
    }

    #[test]
    fn compressed_wire_is_headerless_and_rejects_empty_shape() {
        let poly = CompressedPoly::new(vec![F::from_u64(3), F::from_u64(4)]);
        let mut bytes = Vec::new();
        poly.serialize_compressed(&mut bytes).unwrap();
        assert_eq!(bytes, [3, 0, 0, 0, 4, 0, 0, 0]);
        assert_eq!(poly.compressed_size(), bytes.len());
        assert_eq!(
            CompressedPoly::<F>::deserialize_compressed_exact(&bytes, &2).unwrap(),
            poly
        );
        for validate in [Validate::Yes, Validate::No] {
            assert!(matches!(
                CompressedPoly::<F>::deserialize_with_mode(&[][..], Compress::Yes, validate, &0),
                Err(SerializationError::InvalidData(_))
            ));
        }
        assert!(CompressedPoly::<F>::deserialize_compressed_exact(&bytes[..4], &2).is_err());
    }

    #[test]
    fn normalized_wire_is_headerless_and_allows_empty_shape() {
        let poly = OmittedConstantPoly::new(vec![F::from_u64(5), F::from_u64(0)]);
        let mut bytes = Vec::new();
        poly.serialize_compressed(&mut bytes).unwrap();
        assert_eq!(bytes, [5, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(poly.compressed_size(), bytes.len());
        assert_eq!(
            OmittedConstantPoly::<F>::deserialize_compressed_exact(&bytes, &2).unwrap(),
            poly
        );
        assert_eq!(
            OmittedConstantPoly::<F>::deserialize_compressed_exact(&[], &0).unwrap(),
            OmittedConstantPoly::new(Vec::new())
        );
    }
}
