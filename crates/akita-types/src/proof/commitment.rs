//! Protocol commitment/opening wrapper types.

use crate::proof::{RingVec, MAX_SETUP_MATRIX_FIELD_ELEMENTS};
use crate::transcript::AppendToTranscript;
use crate::{
    detect_field_modulus, CommittedGroupDescriptor, GroupSource, GroupSourceEncoding,
    GroupSourceRegistration, PolynomialGroupLayout,
};
use akita_algebra::ring::CyclotomicRing;
use akita_field::{AkitaError, CanonicalField, FieldCore};
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

impl Valid for GroupSource {
    fn check(&self) -> Result<(), SerializationError> {
        self.validate(128)
            .map_err(|err| SerializationError::InvalidData(err.to_string()))
    }
}

impl AkitaSerialize for GroupSource {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        _compress: Compress,
    ) -> Result<(), SerializationError> {
        self.check()?;
        let registration = self.registration();
        writer.write_all(&registration.type_id())?;
        writer.write_all(&registration.parameters())?;
        let (tag, value) = match self.encoding() {
            GroupSourceEncoding::Bounded { coefficient_bits } => (0u8, u64::from(coefficient_bits)),
            GroupSourceEncoding::SparseBinary { chunk_size } => (
                1u8,
                u64::try_from(chunk_size).map_err(|_| {
                    SerializationError::InvalidData(
                        "sparse-binary chunk size exceeds u64".to_string(),
                    )
                })?,
            ),
        };
        tag.serialize_with_mode(&mut writer, Compress::No)?;
        value.serialize_with_mode(&mut writer, Compress::No)
    }

    fn serialized_size(&self, _compress: Compress) -> usize {
        Self::SERIALIZED_SIZE
    }
}

impl AkitaDeserialize for GroupSource {
    type Context = ();

    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        _compress: Compress,
        validate: Validate,
        _ctx: &(),
    ) -> Result<Self, SerializationError> {
        let mut type_id = [0u8; 16];
        let mut parameters = [0u8; 16];
        reader.read_exact(&mut type_id)?;
        reader.read_exact(&mut parameters)?;
        let tag = u8::deserialize_with_mode(&mut reader, Compress::No, Validate::Yes, &())?;
        let value = u64::deserialize_with_mode(&mut reader, Compress::No, Validate::Yes, &())?;
        let encoding = match tag {
            0 => GroupSourceEncoding::Bounded {
                coefficient_bits: u32::try_from(value).map_err(|_| {
                    SerializationError::InvalidData(
                        "bounded coefficient width exceeds u32".to_string(),
                    )
                })?,
            },
            1 => GroupSourceEncoding::SparseBinary {
                chunk_size: usize::try_from(value).map_err(|_| {
                    SerializationError::InvalidData(
                        "sparse-binary chunk size exceeds usize".to_string(),
                    )
                })?,
            },
            _ => {
                return Err(SerializationError::InvalidData(
                    "unknown group-source encoding tag".to_string(),
                ))
            }
        };
        let source = Self::registered(GroupSourceRegistration::new(type_id, parameters), encoding);
        if matches!(validate, Validate::Yes) {
            source.check()?;
        }
        Ok(source)
    }
}

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
        F: CanonicalField,
    {
        self.0
            .append_flat_to_transcript(label, ring_dim, transcript)
    }
}

/// Public commitment to one polynomial group together with its frozen contract.
///
/// The descriptor is the exact schedule identity selected when the group was
/// committed. Callers pass this object through claims so proving and
/// verification never reconstruct group metadata from a bare layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommittedGroup<F: FieldCore> {
    /// Exact group layout, source contract, and commitment geometry.
    pub descriptor: CommittedGroupDescriptor,
    /// Outer SIS commitment rows.
    pub commitment: Commitment<F>,
}

impl<F: FieldCore> CommittedGroup<F> {
    /// Build a self-describing group commitment.
    pub fn new(descriptor: CommittedGroupDescriptor, commitment: Commitment<F>) -> Self {
        Self {
            descriptor,
            commitment,
        }
    }

    /// Borrow the exact frozen descriptor.
    pub fn descriptor(&self) -> &CommittedGroupDescriptor {
        &self.descriptor
    }

    /// Borrow the underlying SIS commitment.
    pub fn commitment(&self) -> &Commitment<F> {
        &self.commitment
    }

    /// Borrow the underlying flat commitment rows.
    pub fn rows(&self) -> &RingVec<F> {
        self.commitment.rows()
    }
}

impl<F: FieldCore + CanonicalField + Valid> Valid for CommittedGroup<F> {
    fn check(&self) -> Result<(), SerializationError> {
        let field_bits = 128 - (detect_field_modulus::<F>() - 1).leading_zeros();
        self.descriptor
            .validate_frozen_precommit(field_bits)
            .map_err(|err| SerializationError::InvalidData(err.to_string()))?;
        self.commitment.check()?;
        let expected_coeffs = self
            .descriptor
            .n_b
            .checked_mul(self.descriptor.outer_ring_dimension)
            .ok_or_else(|| {
                SerializationError::InvalidData(
                    "committed-group coefficient count overflow".to_string(),
                )
            })?;
        if self.commitment.rows().coeff_len() != expected_coeffs {
            return Err(SerializationError::InvalidData(
                "committed-group rows do not match the frozen descriptor".to_string(),
            ));
        }
        Ok(())
    }
}

impl<F: FieldCore + CanonicalField + Valid + AkitaSerialize> AkitaSerialize for CommittedGroup<F> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        fn write_usize<W: Write>(writer: &mut W, value: usize) -> Result<(), SerializationError> {
            u64::try_from(value)
                .map_err(|_| {
                    SerializationError::InvalidData(
                        "committed-group integer exceeds u64".to_string(),
                    )
                })?
                .serialize_with_mode(writer, Compress::No)
        }

        self.check()?;
        let descriptor = &self.descriptor;
        write_usize(&mut writer, descriptor.group.num_vars())?;
        write_usize(&mut writer, descriptor.group.num_polynomials())?;
        descriptor
            .source
            .serialize_with_mode(&mut writer, Compress::No)?;
        for value in [
            descriptor.num_live_ring_elements_per_claim,
            descriptor.num_positions_per_block,
            descriptor.num_live_blocks,
        ] {
            write_usize(&mut writer, value)?;
        }
        descriptor
            .log_basis_inner
            .serialize_with_mode(&mut writer, Compress::No)?;
        descriptor
            .log_basis_outer
            .serialize_with_mode(&mut writer, Compress::No)?;
        for value in [
            descriptor.inner_ring_dimension,
            descriptor.outer_ring_dimension,
            descriptor.n_a,
        ] {
            write_usize(&mut writer, value)?;
        }
        descriptor
            .a_coeff_linf_bound
            .serialize_with_mode(&mut writer, Compress::No)?;
        write_usize(&mut writer, descriptor.n_b)?;
        descriptor
            .b_coeff_linf_bound
            .serialize_with_mode(&mut writer, Compress::No)?;
        self.commitment.serialize_with_mode(&mut writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        112 + GroupSource::SERIALIZED_SIZE + self.commitment.serialized_size(compress)
    }
}

impl<F> AkitaDeserialize for CommittedGroup<F>
where
    F: FieldCore + CanonicalField + Valid + AkitaSerialize + AkitaDeserialize<Context = ()>,
{
    type Context = ();

    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &(),
    ) -> Result<Self, SerializationError> {
        fn read_u64<R: Read>(reader: &mut R) -> Result<u64, SerializationError> {
            u64::deserialize_with_mode(reader, Compress::No, Validate::Yes, &())
        }
        fn read_usize<R: Read>(reader: &mut R) -> Result<usize, SerializationError> {
            usize::try_from(read_u64(reader)?).map_err(|_| {
                SerializationError::InvalidData("committed-group integer exceeds usize".to_string())
            })
        }

        let num_vars = read_usize(&mut reader)?;
        let num_polynomials = read_usize(&mut reader)?;
        let source = GroupSource::deserialize_with_mode(&mut reader, Compress::No, validate, &())?;
        let num_live_ring_elements_per_claim = read_usize(&mut reader)?;
        let num_positions_per_block = read_usize(&mut reader)?;
        let num_live_blocks = read_usize(&mut reader)?;
        let log_basis_inner =
            u32::deserialize_with_mode(&mut reader, Compress::No, Validate::Yes, &())?;
        let log_basis_outer =
            u32::deserialize_with_mode(&mut reader, Compress::No, Validate::Yes, &())?;
        let inner_ring_dimension = read_usize(&mut reader)?;
        let outer_ring_dimension = read_usize(&mut reader)?;
        let n_a = read_usize(&mut reader)?;
        let a_coeff_linf_bound =
            u128::deserialize_with_mode(&mut reader, Compress::No, Validate::Yes, &())?;
        let n_b = read_usize(&mut reader)?;
        let b_coeff_linf_bound =
            u128::deserialize_with_mode(&mut reader, Compress::No, Validate::Yes, &())?;

        let descriptor = CommittedGroupDescriptor {
            group: PolynomialGroupLayout::new(num_vars, num_polynomials),
            source,
            num_live_ring_elements_per_claim,
            num_positions_per_block,
            num_live_blocks,
            log_basis_inner,
            log_basis_outer,
            inner_ring_dimension,
            outer_ring_dimension,
            n_a,
            a_coeff_linf_bound,
            n_b,
            b_coeff_linf_bound,
        };
        let field_bits = 128 - (detect_field_modulus::<F>() - 1).leading_zeros();
        descriptor
            .validate_frozen_precommit(field_bits)
            .map_err(|err| SerializationError::InvalidData(err.to_string()))?;
        let num_coeffs = n_b.checked_mul(outer_ring_dimension).ok_or_else(|| {
            SerializationError::InvalidData(
                "committed-group coefficient count overflow".to_string(),
            )
        })?;
        if num_coeffs > MAX_SETUP_MATRIX_FIELD_ELEMENTS {
            return Err(SerializationError::InvalidData(format!(
                "committed-group coefficient count {num_coeffs} exceeds allocation cap \
                 {MAX_SETUP_MATRIX_FIELD_ELEMENTS}"
            )));
        }
        let commitment =
            Commitment::deserialize_with_mode(&mut reader, compress, validate, &num_coeffs)?;
        let group = Self::new(descriptor, commitment);
        if matches!(validate, Validate::Yes) {
            group.check()?;
        }
        Ok(group)
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

#[cfg(test)]
mod committed_group_tests {
    use super::*;
    use akita_field::Fp32;

    type F = Fp32<251>;

    fn group() -> CommittedGroup<F> {
        CommittedGroup::new(
            CommittedGroupDescriptor {
                group: PolynomialGroupLayout::new(1, 1),
                source: GroupSource::bounded(8),
                num_live_ring_elements_per_claim: 1,
                num_positions_per_block: 1,
                num_live_blocks: 1,
                log_basis_inner: 1,
                log_basis_outer: 1,
                inner_ring_dimension: 2,
                outer_ring_dimension: 2,
                n_a: 1,
                a_coeff_linf_bound: 1,
                n_b: 1,
                b_coeff_linf_bound: 1,
            },
            Commitment::new(RingVec::from_coeffs(vec![F::zero(), F::one()])),
        )
    }

    #[test]
    fn committed_group_serialization_binds_and_validates_source() {
        let group = group();
        let mut bytes = Vec::new();
        group
            .serialize_with_mode(&mut bytes, Compress::Yes)
            .expect("serialize committed group");
        assert_eq!(bytes.len(), group.serialized_size(Compress::Yes));
        let decoded = CommittedGroup::<F>::deserialize_with_mode(
            bytes.as_slice(),
            Compress::Yes,
            Validate::Yes,
            &(),
        )
        .expect("deserialize committed group");
        assert_eq!(decoded, group);

        let source_tag_offset = 2 * std::mem::size_of::<u64>() + 2 * 16;
        bytes[source_tag_offset] = 9;
        assert!(CommittedGroup::<F>::deserialize_with_mode(
            bytes.as_slice(),
            Compress::Yes,
            Validate::Yes,
            &(),
        )
        .is_err());
    }

    #[test]
    fn committed_group_rejects_dense_bound_above_field_width() {
        let mut group = group();
        group.descriptor.source = GroupSource::bounded(9);
        assert!(group.check().is_err());
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
