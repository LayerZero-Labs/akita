//! Protocol commitment/opening wrapper types.

use crate::proof::RingSliceSerializer;
use crate::transcript::AppendToTranscript;
use akita_algebra::ring::CyclotomicRing;
use akita_field::{CanonicalField, FieldCore};
use akita_serialization::{
    AkitaDeserialize, AkitaSerialize, Compress, SerializationError, Valid, Validate,
};
use akita_transcript::Transcript;
use std::io::{Read, Write};

/// Minimal commitment wrapper used by protocol traits/tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct AkitaCommitment(pub u128);

/// Minimal proof wrapper used by protocol trait stubs and tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DummyProof(pub u128);

impl Valid for AkitaCommitment {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl AkitaSerialize for AkitaCommitment {
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

impl AkitaDeserialize for AkitaCommitment {
    type Context = ();
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        _compress: Compress,
        validate: Validate,
        _ctx: &(),
    ) -> Result<Self, SerializationError> {
        let value = u128::deserialize_with_mode(&mut reader, Compress::No, validate, &())?;
        Ok(Self(value))
    }
}

impl Valid for DummyProof {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl AkitaSerialize for DummyProof {
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

impl AkitaDeserialize for DummyProof {
    type Context = ();
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        _compress: Compress,
        validate: Validate,
        _ctx: &(),
    ) -> Result<Self, SerializationError> {
        let value = u128::deserialize_with_mode(&mut reader, Compress::No, validate, &())?;
        Ok(Self(value))
    }
}

impl<F> AppendToTranscript<F> for AkitaCommitment
where
    F: FieldCore + CanonicalField,
{
    fn append_to_transcript<T: Transcript<F>>(&self, label: &[u8], transcript: &mut T) {
        transcript.append_serde(label, self);
    }
}

/// Ring-native commitment object `u in R_q^{n_B}` used by §4.1.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RingCommitment<F: FieldCore, const D: usize> {
    /// Outer commitment vector.
    pub u: Vec<CyclotomicRing<F, D>>,
}

impl<F: FieldCore + Valid, const D: usize> Valid for RingCommitment<F, D> {
    fn check(&self) -> Result<(), SerializationError> {
        self.u.check()
    }
}

impl<F: FieldCore + AkitaSerialize, const D: usize> AkitaSerialize for RingCommitment<F, D> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.u.serialize_with_mode(&mut writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.u.serialized_size(compress)
    }
}

impl<F: FieldCore + Valid + AkitaDeserialize<Context = ()>, const D: usize> AkitaDeserialize
    for RingCommitment<F, D>
{
    type Context = ();
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &(),
    ) -> Result<Self, SerializationError> {
        let u = Vec::<CyclotomicRing<F, D>>::deserialize_with_mode(
            &mut reader,
            compress,
            validate,
            &(),
        )?;
        let out = Self { u };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

impl<F, const D: usize> AppendToTranscript<F> for RingCommitment<F, D>
where
    F: FieldCore + CanonicalField,
{
    fn append_to_transcript<T: Transcript<F>>(&self, label: &[u8], transcript: &mut T) {
        transcript.append_serde(label, &RingSliceSerializer(&self.u));
    }
}
