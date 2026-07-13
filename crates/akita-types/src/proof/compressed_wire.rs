//! Dormant exact-count wire candidates for compressed commitments.
//!
//! Payloads carry no tags or length prefixes. The validated schedule-derived
//! shape is the sole authority for source count, per-source coefficient count,
//! and whether the co-generated H payload is present.

use super::{checked_shape_len, checked_shape_sequence_len, reserve_shape_len};
use akita_field::FieldCore;
use akita_serialization::{
    AkitaDeserialize, AkitaSerialize, Compress, SerializationError, Valid, Validate,
};
use std::io::{Cursor, Read, Write};

/// Exact wire shape for one fold's terminal F payloads and optional H payload.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompressedFoldPayloadShape {
    pub terminal_f_coeffs: Vec<usize>,
    pub terminal_h_coeffs: Option<usize>,
}

impl Valid for CompressedFoldPayloadShape {
    fn check(&self) -> Result<(), SerializationError> {
        checked_shape_sequence_len(self.terminal_f_coeffs.len())?;
        if self.terminal_f_coeffs.is_empty() {
            return Err(SerializationError::InvalidData(
                "compressed fold payload requires at least one terminal F source".into(),
            ));
        }
        for &coeffs in &self.terminal_f_coeffs {
            checked_nonzero_coeff_count(coeffs)?;
        }
        if let Some(coeffs) = self.terminal_h_coeffs {
            checked_nonzero_coeff_count(coeffs)?;
        }
        Ok(())
    }
}

impl AkitaSerialize for CompressedFoldPayloadShape {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.check()?;
        self.terminal_f_coeffs
            .len()
            .serialize_with_mode(&mut writer, compress)?;
        for coeffs in &self.terminal_f_coeffs {
            coeffs.serialize_with_mode(&mut writer, compress)?;
        }
        self.terminal_h_coeffs
            .is_some()
            .serialize_with_mode(&mut writer, compress)?;
        if let Some(coeffs) = self.terminal_h_coeffs {
            coeffs.serialize_with_mode(&mut writer, compress)?;
        }
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.terminal_f_coeffs.len().serialized_size(compress)
            + self
                .terminal_f_coeffs
                .iter()
                .map(|coeffs| coeffs.serialized_size(compress))
                .sum::<usize>()
            + true.serialized_size(compress)
            + self
                .terminal_h_coeffs
                .map_or(0, |coeffs| coeffs.serialized_size(compress))
    }
}

impl AkitaDeserialize for CompressedFoldPayloadShape {
    type Context = ();

    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &(),
    ) -> Result<Self, SerializationError> {
        let f_count = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
        checked_shape_sequence_len(f_count)?;
        let mut terminal_f_coeffs = Vec::new();
        reserve_shape_len(&mut terminal_f_coeffs, f_count)?;
        for _ in 0..f_count {
            terminal_f_coeffs.push(usize::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                &(),
            )?);
        }
        let has_h = bool::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let terminal_h_coeffs = has_h
            .then(|| usize::deserialize_with_mode(&mut reader, compress, validate, &()))
            .transpose()?;
        let shape = Self {
            terminal_f_coeffs,
            terminal_h_coeffs,
        };
        shape.check()?;
        Ok(shape)
    }
}

/// Headerless terminal coefficient payload for one source.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExactFieldPayload<F: FieldCore> {
    coeffs: Vec<F>,
}

impl<F: FieldCore> ExactFieldPayload<F> {
    pub fn new(coeffs: Vec<F>) -> Result<Self, SerializationError> {
        checked_nonzero_coeff_count(coeffs.len())?;
        Ok(Self { coeffs })
    }

    #[must_use]
    pub fn coeffs(&self) -> &[F] {
        &self.coeffs
    }
}

impl<F: FieldCore + Valid> Valid for ExactFieldPayload<F> {
    fn check(&self) -> Result<(), SerializationError> {
        checked_nonzero_coeff_count(self.coeffs.len())?;
        self.coeffs.check()
    }
}

impl<F: FieldCore + AkitaSerialize> AkitaSerialize for ExactFieldPayload<F> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        for coeff in &self.coeffs {
            coeff.serialize_with_mode(&mut writer, compress)?;
        }
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.coeffs
            .iter()
            .map(|coeff| coeff.serialized_size(compress))
            .sum()
    }
}

impl<F> AkitaDeserialize for ExactFieldPayload<F>
where
    F: FieldCore + Valid + AkitaDeserialize<Context = ()>,
{
    type Context = usize;

    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        coeff_count: &usize,
    ) -> Result<Self, SerializationError> {
        checked_nonzero_coeff_count(*coeff_count)?;
        let mut coeffs = Vec::new();
        reserve_shape_len(&mut coeffs, *coeff_count)?;
        for _ in 0..*coeff_count {
            coeffs.push(F::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                &(),
            )?);
        }
        let payload = Self { coeffs };
        if validate == Validate::Yes {
            payload.check()?;
        }
        Ok(payload)
    }
}

/// Headerless compressed fold payload in schedule source order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompressedFoldPayload<F: FieldCore> {
    pub terminal_f: Vec<ExactFieldPayload<F>>,
    pub terminal_h: Option<ExactFieldPayload<F>>,
}

impl<F: FieldCore + Valid> Valid for CompressedFoldPayload<F> {
    fn check(&self) -> Result<(), SerializationError> {
        checked_shape_sequence_len(self.terminal_f.len())?;
        if self.terminal_f.is_empty() {
            return Err(SerializationError::InvalidData(
                "compressed fold payload requires at least one terminal F source".into(),
            ));
        }
        self.terminal_f.check()?;
        if let Some(payload) = &self.terminal_h {
            payload.check()?;
        }
        Ok(())
    }
}

impl<F: FieldCore + AkitaSerialize> AkitaSerialize for CompressedFoldPayload<F> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        for payload in &self.terminal_f {
            payload.serialize_with_mode(&mut writer, compress)?;
        }
        if let Some(payload) = &self.terminal_h {
            payload.serialize_with_mode(&mut writer, compress)?;
        }
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.terminal_f
            .iter()
            .map(|payload| payload.serialized_size(compress))
            .sum::<usize>()
            + self
                .terminal_h
                .as_ref()
                .map_or(0, |payload| payload.serialized_size(compress))
    }
}

impl<F> AkitaDeserialize for CompressedFoldPayload<F>
where
    F: FieldCore + Valid + AkitaDeserialize<Context = ()>,
{
    type Context = CompressedFoldPayloadShape;

    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        shape: &CompressedFoldPayloadShape,
    ) -> Result<Self, SerializationError> {
        shape.check()?;
        let mut terminal_f = Vec::new();
        reserve_shape_len(&mut terminal_f, shape.terminal_f_coeffs.len())?;
        for coeffs in &shape.terminal_f_coeffs {
            terminal_f.push(ExactFieldPayload::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                coeffs,
            )?);
        }
        let terminal_h = shape
            .terminal_h_coeffs
            .map(|coeffs| {
                ExactFieldPayload::deserialize_with_mode(&mut reader, compress, validate, &coeffs)
            })
            .transpose()?;
        let payload = Self {
            terminal_f,
            terminal_h,
        };
        if validate == Validate::Yes {
            payload.check()?;
        }
        Ok(payload)
    }
}

impl<F> CompressedFoldPayload<F>
where
    F: FieldCore + Valid + AkitaDeserialize<Context = ()>,
{
    /// Decode one exact payload and reject both truncation and trailing bytes.
    pub fn deserialize_exact(
        bytes: &[u8],
        compress: Compress,
        validate: Validate,
        shape: &CompressedFoldPayloadShape,
    ) -> Result<Self, SerializationError> {
        shape.check()?;
        let mut reader = Cursor::new(bytes);
        let payload = Self::deserialize_with_mode(&mut reader, compress, validate, shape)?;
        if reader.position() != bytes.len() as u64 {
            return Err(SerializationError::InvalidData(
                "compressed fold payload has trailing bytes".into(),
            ));
        }
        Ok(payload)
    }
}

fn checked_nonzero_coeff_count(coeffs: usize) -> Result<(), SerializationError> {
    if coeffs == 0 {
        return Err(SerializationError::InvalidData(
            "compressed source payload must be nonempty".into(),
        ));
    }
    checked_shape_len(coeffs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_field::Prime128Offset275;
    use akita_serialization::DEFAULT_MAX_SEQUENCE_LEN;

    type F = Prime128Offset275;

    fn payload() -> (CompressedFoldPayloadShape, CompressedFoldPayload<F>) {
        let shape = CompressedFoldPayloadShape {
            terminal_f_coeffs: vec![2, 1],
            terminal_h_coeffs: Some(2),
        };
        let payload = CompressedFoldPayload {
            terminal_f: vec![
                ExactFieldPayload::new(vec![F::zero(), F::one()]).unwrap(),
                ExactFieldPayload::new(vec![F::one()]).unwrap(),
            ],
            terminal_h: Some(ExactFieldPayload::new(vec![F::one(), F::zero()]).unwrap()),
        };
        (shape, payload)
    }

    #[test]
    fn compressed_payload_has_exact_headerless_size() {
        let (shape, payload) = payload();
        let mut bytes = Vec::new();
        payload.serialize_uncompressed(&mut bytes).unwrap();
        let field_bytes = F::zero().serialized_size(Compress::No);
        assert_eq!(bytes.len(), 5 * field_bytes);
        assert_eq!(payload.serialized_size(Compress::No), bytes.len());
        assert_eq!(
            CompressedFoldPayload::<F>::deserialize_exact(
                &bytes,
                Compress::No,
                Validate::Yes,
                &shape,
            )
            .unwrap(),
            payload
        );
    }

    #[test]
    fn compressed_payload_rejects_truncation_and_trailing_bytes() {
        let (shape, payload) = payload();
        let mut bytes = Vec::new();
        payload.serialize_uncompressed(&mut bytes).unwrap();

        assert!(CompressedFoldPayload::<F>::deserialize_exact(
            &bytes[..bytes.len() - 1],
            Compress::No,
            Validate::Yes,
            &shape,
        )
        .is_err());
        bytes.push(0);
        assert!(matches!(
            CompressedFoldPayload::<F>::deserialize_exact(
                &bytes,
                Compress::No,
                Validate::Yes,
                &shape,
            ),
            Err(SerializationError::InvalidData(_))
        ));
    }

    #[test]
    fn compressed_payload_validates_shape_before_allocation() {
        let oversized = CompressedFoldPayloadShape {
            terminal_f_coeffs: vec![DEFAULT_MAX_SEQUENCE_LEN + 1],
            terminal_h_coeffs: None,
        };
        assert!(matches!(
            CompressedFoldPayload::<F>::deserialize_exact(
                &[],
                Compress::No,
                Validate::Yes,
                &oversized,
            ),
            Err(SerializationError::LengthLimitExceeded { .. })
        ));
        let empty = CompressedFoldPayloadShape {
            terminal_f_coeffs: Vec::new(),
            terminal_h_coeffs: None,
        };
        assert!(matches!(
            empty.check(),
            Err(SerializationError::InvalidData(_))
        ));
    }

    #[test]
    fn compressed_payload_shape_round_trips_exactly() {
        let (shape, _) = payload();
        let mut bytes = Vec::new();
        shape.serialize_compressed(&mut bytes).unwrap();
        let decoded = CompressedFoldPayloadShape::deserialize_compressed(&bytes[..], &()).unwrap();
        assert_eq!(decoded, shape);
    }
}
