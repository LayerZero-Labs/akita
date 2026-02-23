//! Protocol commitment/opening wrapper types.

use super::transcript_append::AppendToTranscript;
use crate::algebra::ring::CyclotomicRing;
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

/// Minimal proof wrapper used by protocol trait stubs and tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DummyProof(pub u128);

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

impl Valid for DummyProof {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl HachiSerialize for DummyProof {
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

impl HachiDeserialize for DummyProof {
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

/// Ring-native commitment object `u in R_q^{n_B}` used by §4.1.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RingCommitment<F: FieldCore, const D: usize> {
    /// Outer commitment vector.
    pub u: Vec<CyclotomicRing<F, D>>,
}

/// Ring-native opening witness `(s_i, t_hat_i)_i`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RingOpening<F: FieldCore, const D: usize> {
    /// Decomposed vectors `s_i`.
    pub s: Vec<Vec<CyclotomicRing<F, D>>>,
    /// Decomposed inner commitments `t_hat_i`.
    pub t_hat: Vec<Vec<CyclotomicRing<F, D>>>,
}

/// Placeholder proof wrapper for open-check path wiring.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RingOpenProof<F: FieldCore, const D: usize> {
    /// Embedded opening witness.
    pub opening: RingOpening<F, D>,
}

impl<F: FieldCore + Valid, const D: usize> Valid for RingCommitment<F, D> {
    fn check(&self) -> Result<(), SerializationError> {
        self.u.check()
    }
}

impl<F: FieldCore + Valid, const D: usize> Valid for RingOpening<F, D> {
    fn check(&self) -> Result<(), SerializationError> {
        self.s.check()?;
        self.t_hat.check()
    }
}

impl<F: FieldCore + Valid, const D: usize> Valid for RingOpenProof<F, D> {
    fn check(&self) -> Result<(), SerializationError> {
        self.opening.check()
    }
}

impl<F: FieldCore, const D: usize> HachiSerialize for RingCommitment<F, D> {
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

impl<F: FieldCore + Valid, const D: usize> HachiDeserialize for RingCommitment<F, D> {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let u =
            Vec::<CyclotomicRing<F, D>>::deserialize_with_mode(&mut reader, compress, validate)?;
        let out = Self { u };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

impl<F: FieldCore, const D: usize> HachiSerialize for RingOpening<F, D> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.s.serialize_with_mode(&mut writer, compress)?;
        self.t_hat.serialize_with_mode(&mut writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.s.serialized_size(compress) + self.t_hat.serialized_size(compress)
    }
}

impl<F: FieldCore + Valid, const D: usize> HachiDeserialize for RingOpening<F, D> {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let s = Vec::<Vec<CyclotomicRing<F, D>>>::deserialize_with_mode(
            &mut reader,
            compress,
            validate,
        )?;
        let t_hat = Vec::<Vec<CyclotomicRing<F, D>>>::deserialize_with_mode(
            &mut reader,
            compress,
            validate,
        )?;
        let out = Self { s, t_hat };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

impl<F: FieldCore, const D: usize> HachiSerialize for RingOpenProof<F, D> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.opening.serialize_with_mode(&mut writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.opening.serialized_size(compress)
    }
}

impl<F: FieldCore + Valid, const D: usize> HachiDeserialize for RingOpenProof<F, D> {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let opening = RingOpening::<F, D>::deserialize_with_mode(&mut reader, compress, validate)?;
        let out = Self { opening };
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
        transcript.append_serde(label, self);
    }
}
