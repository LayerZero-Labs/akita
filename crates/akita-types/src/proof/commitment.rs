//! Protocol commitment/opening wrapper types.

use crate::proof::RingVec;
use crate::transcript::AppendToTranscript;
use akita_algebra::ring::CyclotomicRing;
use akita_error::AkitaError;
use akita_serialization::{
    AkitaDeserialize, AkitaSerialize, Compress, SerializationError, Valid, Validate,
};
use akita_transcript::Transcript;
use jolt_field::{CanonicalField, FieldCore};
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

/// D-free protocol commitment storage: a flat ring-coefficient buffer.
///
/// This is the protocol-facing replacement for the former
/// `RingCommitment<F, D>` storage. It carries the outer commitment vector
/// `u in R_q^{n_B}` as raw field coefficients (a [`RingVec`]), with the ring
/// dimension supplied at runtime from the schedule rather than a const generic.
/// Transcript absorption goes through the flat coefficient encoder; the bytes
/// are identical to the former typed path (proven by the S2 byte-identity test).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Commitment<F: FieldCore>(pub RingVec<F>);

impl<F: FieldCore> Commitment<F> {
    /// Wrap a flat ring-coefficient buffer.
    pub fn new(rows: RingVec<F>) -> Self {
        Self(rows)
    }

    /// Construct from typed ring elements.
    pub fn from_ring_elems<const D: usize>(elems: &[CyclotomicRing<F, D>]) -> Self {
        Self(RingVec::from_ring_elems(elems))
    }

    /// Borrow the underlying flat ring-coefficient buffer.
    pub fn rows(&self) -> &RingVec<F> {
        &self.0
    }

    /// Consume into the underlying flat ring-coefficient buffer.
    pub fn into_rows(self) -> RingVec<F> {
        self.0
    }

    /// Absorb this commitment into `transcript` using the canonical flat
    /// coefficient encoding under the schedule-derived `ring_dim`.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidProof`] if the stored buffer is not
    /// well-formed for `ring_dim` (see [`RingVec::append_flat_to_transcript`]).
    pub fn append_to_transcript<T: Transcript<F>>(
        &self,
        label: &[u8],
        ring_dim: usize,
        transcript: &mut T,
    ) -> Result<(), AkitaError>
    where
        F: CanonicalField + AkitaSerialize,
    {
        self.0
            .append_flat_to_transcript(label, ring_dim, transcript)
    }
}

impl<F: FieldCore + Valid> Valid for Commitment<F> {
    fn check(&self) -> Result<(), SerializationError> {
        self.0.check()
    }
}

impl<F: FieldCore + AkitaSerialize> AkitaSerialize for Commitment<F> {
    fn serialize_with_mode<W: Write>(
        &self,
        writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.0.serialize_with_mode(writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.0.serialized_size(compress)
    }
}

impl<F: FieldCore + Valid + AkitaDeserialize<Context = ()>> AkitaDeserialize for Commitment<F> {
    /// Number of field-element coefficients to read (same as [`RingVec`]).
    type Context = usize;
    fn deserialize_with_mode<R: Read>(
        reader: R,
        compress: Compress,
        validate: Validate,
        num_coeffs: &usize,
    ) -> Result<Self, SerializationError> {
        Ok(Self(RingVec::deserialize_with_mode(
            reader, compress, validate, num_coeffs,
        )?))
    }
}

/// Ring-native commitment object `u in R_q^{n_B}` used by §4.1.
///
/// **Arithmetic-only leaf helper.** As of S4 this type is no longer used for
/// protocol-facing storage, serialization, or transcript absorption — that role
/// belongs to the D-free [`Commitment`] / [`RingVec`]. It is kept solely as a
/// typed arithmetic carrier inside kernels.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RingCommitment<F: FieldCore, const D: usize> {
    /// Outer commitment vector.
    pub u: Vec<CyclotomicRing<F, D>>,
}

/// Borrow ring rows from commitment-like prover inputs.
pub trait ProverCommitmentRows<CommitF: FieldCore, const D: usize> {
    fn commitment_rows(&self) -> &[CyclotomicRing<CommitF, D>];
}

impl<CommitF: FieldCore, const D: usize> ProverCommitmentRows<CommitF, D>
    for RingCommitment<CommitF, D>
{
    fn commitment_rows(&self) -> &[CyclotomicRing<CommitF, D>] {
        &self.u
    }
}

impl<CommitF: FieldCore, const D: usize> ProverCommitmentRows<CommitF, D>
    for [CyclotomicRing<CommitF, D>]
{
    fn commitment_rows(&self) -> &[CyclotomicRing<CommitF, D>] {
        self
    }
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
