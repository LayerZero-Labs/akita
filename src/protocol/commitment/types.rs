//! Minimal protocol commitment/opening wrapper types.

use super::transcript_append::AppendToTranscript;
use crate::primitives::serialization::{
    Compress, HachiDeserialize, HachiSerialize, SerializationError, Valid, Validate,
};
use crate::protocol::transcript::Transcript;
use crate::{CanonicalField, FieldCore};
use std::io::{Read, Write};

/// A Hachi opening point represented as field coordinates.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HachiOpeningPoint<F: FieldCore> {
    /// Point coordinates used for multilinear opening.
    pub r: Vec<F>,
}

/// A Hachi opening claim `(point, value)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HachiOpeningClaim<F: FieldCore> {
    /// Opening point.
    pub point: HachiOpeningPoint<F>,
    /// Claimed value at `point`.
    pub value: F,
}

/// Minimal commitment wrapper used by protocol traits/tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct HachiCommitment(pub u128);

/// Minimal proof wrapper used by protocol traits/tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct HachiProof(pub u128);

impl Valid for HachiCommitment {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl HachiSerialize for HachiCommitment {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        _compress: Compress,
    ) -> Result<(), SerializationError> {
        self.0.serialize_with_mode(&mut writer, Compress::No)
    }

    fn serialized_size(&self, _compress: Compress) -> usize {
        16
    }
}

impl HachiDeserialize for HachiCommitment {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        _compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let value = u128::deserialize_with_mode(&mut reader, Compress::No, validate)?;
        Ok(Self(value))
    }
}

impl Valid for HachiProof {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl HachiSerialize for HachiProof {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        _compress: Compress,
    ) -> Result<(), SerializationError> {
        self.0.serialize_with_mode(&mut writer, Compress::No)
    }

    fn serialized_size(&self, _compress: Compress) -> usize {
        16
    }
}

impl HachiDeserialize for HachiProof {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        _compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let value = u128::deserialize_with_mode(&mut reader, Compress::No, validate)?;
        Ok(Self(value))
    }
}

impl<F> AppendToTranscript<F> for HachiCommitment
where
    F: FieldCore + CanonicalField,
{
    fn append_to_transcript<T: Transcript<F>>(&self, label: &[u8], transcript: &mut T) {
        transcript.append_serde(label, self);
    }
}
