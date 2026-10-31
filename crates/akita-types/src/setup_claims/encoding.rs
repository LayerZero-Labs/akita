//! Context-bound accumulator transport. Decoding uses the caller's expected domain and bounded arity.
use super::{SetupAccumulator, SetupPolynomialId};
use akita_serialization::{
    AkitaDeserialize, AkitaSerialize, Compress, SerializationError, Valid, Validate,
};
use jolt_field::Field;
use std::io::{Read, Write};

impl<E: Field + Valid> Valid for SetupAccumulator<E> {
    fn check(&self) -> Result<(), SerializationError> {
        if self.point.len() != self.polynomial.num_variables() {
            return Err(SerializationError::InvalidData(
                "setup accumulator arity mismatch".into(),
            ));
        }
        for value in &self.point {
            value.check()?;
        }
        self.value.check()
    }
}
impl<E: Field + AkitaSerialize> AkitaSerialize for SetupAccumulator<E> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        let mut identity = Vec::new();
        self.polynomial
            .encode(&mut identity)
            .map_err(|error| SerializationError::InvalidData(error.to_string()))?;
        writer.write_all(&identity)?;
        for value in &self.point {
            value.serialize_with_mode(&mut writer, compress)?;
        }
        self.value.serialize_with_mode(writer, compress)
    }
    fn serialized_size(&self, compress: Compress) -> usize {
        b"akita/setup-claim-batch/v1/flat-lsb-first".len()
            + self.polynomial.seed.serialized_size(Compress::Yes)
            + 32
            + self
                .polynomial
                .extension_degree
                .serialized_size(Compress::Yes)
            + self.polynomial.field_len.serialized_size(Compress::Yes)
            + self
                .point
                .iter()
                .map(|v| v.serialized_size(compress))
                .sum::<usize>()
            + self.value.serialized_size(compress)
    }
}
impl<E> AkitaDeserialize for SetupAccumulator<E>
where
    E: Field + Valid + AkitaDeserialize<Context = ()>,
{
    type Context = SetupPolynomialId;
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        expected: &SetupPolynomialId,
    ) -> Result<Self, SerializationError> {
        let mut identity = Vec::new();
        expected
            .encode(&mut identity)
            .map_err(|error| SerializationError::InvalidData(error.to_string()))?;
        let mut received = vec![0u8; identity.len()];
        reader.read_exact(&mut received)?;
        if received != identity {
            return Err(SerializationError::InvalidData(
                "unexpected setup accumulator identity".into(),
            ));
        }
        let mut point = Vec::new();
        akita_params::wire_limits::reserve_shape_len(&mut point, expected.num_variables())?;
        for _ in 0..expected.num_variables() {
            point.push(E::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                &(),
            )?);
        }
        let value = E::deserialize_with_mode(reader, compress, validate, &())?;
        let result = Self::new(expected.clone(), point, value)
            .map_err(|error| SerializationError::InvalidData(error.to_string()))?;
        if validate == Validate::Yes {
            result.check()?;
        }
        Ok(result)
    }
}
