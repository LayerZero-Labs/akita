//! Protocol commitment/opening wrapper types.

use crate::proof::{RingVec, MAX_UNTRUSTED_COMMITMENT_COEFFICIENTS};
use crate::sis::{
    InnerCommitMatrixParams, OuterCommitMatrixParams, SisMatrixRole, SisModulusProfileId,
    SisSecurityPolicyId, SisTableDigest,
};
use crate::transcript::AppendToTranscript;
use crate::{
    CommitmentSliceCount, CompressionChainPlan, GroupCommitPhaseParams, PolynomialGroupLayout,
};

type MatrixFields = (
    SisSecurityPolicyId,
    SisTableDigest,
    SisModulusProfileId,
    usize,
    usize,
    u128,
    usize,
);
use akita_algebra::ring::CyclotomicRing;
use akita_error::AkitaError;
use akita_serialization::{
    AkitaDeserialize, AkitaSerialize, Compress, SerializationError, Valid, Validate,
};
use akita_transcript::Transcript;
use jolt_field::{CanonicalEncoding, Field};
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
    F: Field + CanonicalEncoding,
{
    fn append_to_transcript<T: Transcript<F>>(&self, label: &[u8], transcript: &mut T) {
        transcript.append_serde(label, self);
    }
}

/// D-free public commitment payload stored as flat field coefficients.
///
/// For a committed polynomial group this carries the terminal compressed
/// payload `p_F`. Its native compression dimension is derived from the frozen
/// modulus profile and supplied at the transcript boundary rather than encoded
/// as a const generic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Commitment<F: Field>(pub RingVec<F>);

impl<F: Field> Commitment<F> {
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

    /// Absorb this payload using its canonical flat coefficient encoding under
    /// the caller-derived terminal compression `ring_dim`.
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
        F: CanonicalEncoding + AkitaSerialize,
    {
        self.0
            .append_flat_to_transcript(label, ring_dim, transcript)
    }
}

/// Public commitment to one polynomial group together with its frozen contract.
///
/// The profile is the exact commitment identity selected when the group was
/// committed. Callers pass this object through claims so proving and
/// verification never reconstruct group metadata from a bare layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommittedGroup<F: Field> {
    /// Exact public algebraic profile and commitment geometry.
    pub profile: GroupCommitPhaseParams,
    /// Terminal compressed `p_F` payload.
    pub commitment: Commitment<F>,
}

impl<F: Field> CommittedGroup<F> {
    /// Build a self-describing group commitment.
    pub fn new(profile: GroupCommitPhaseParams, commitment: Commitment<F>) -> Self {
        Self {
            profile,
            commitment,
        }
    }

    /// Borrow the exact frozen commitment profile.
    pub fn profile(&self) -> &GroupCommitPhaseParams {
        &self.profile
    }

    /// Borrow the terminal compressed commitment payload.
    pub fn commitment(&self) -> &Commitment<F> {
        &self.commitment
    }

    /// Borrow the terminal payload coefficients.
    pub fn rows(&self) -> &RingVec<F> {
        self.commitment.rows()
    }
}

impl<F: Field + CanonicalEncoding + Valid> Valid for CommittedGroup<F> {
    fn check(&self) -> Result<(), SerializationError> {
        let field_bits = F::MODULUS_BITS;
        self.profile
            .validate_frozen_precommit(field_bits)
            .map_err(|err| SerializationError::InvalidData(err.to_string()))?;
        self.commitment.check()?;
        let source_coefficients = self
            .profile
            .outer_slice_count
            .complete_source_coefficients(
                self.profile.outer.matrix.output_rank(),
                self.profile.outer.matrix.ring_dimension(),
            )
            .map_err(|err| SerializationError::InvalidData(err.to_string()))?;
        let expected_coeffs = CompressionChainPlan::for_complete_source(
            self.profile.outer.matrix.sis_table_key().modulus_profile,
            source_coefficients,
        )
        .map_err(|error| SerializationError::InvalidData(error.to_string()))?
        .terminal_coefficients();
        if self.commitment.rows().coeff_len() != expected_coeffs {
            return Err(SerializationError::InvalidData(
                "committed-group rows do not match the frozen descriptor".to_string(),
            ));
        }
        Ok(())
    }
}

impl<F: Field + CanonicalEncoding + Valid + AkitaSerialize> AkitaSerialize for CommittedGroup<F> {
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
        let profile = &self.profile;
        profile
            .version
            .serialize_with_mode(&mut writer, Compress::No)?;
        write_usize(&mut writer, profile.group.num_vars())?;
        write_usize(&mut writer, profile.group.num_polynomials())?;
        for value in [
            profile.blocks.live_ring_elements_per_claim,
            profile.blocks.positions_per_block,
            profile.blocks.live_blocks,
        ] {
            write_usize(&mut writer, value)?;
        }
        write_usize(&mut writer, profile.outer_slice_count.get())?;
        profile
            .inner
            .digits
            .log_basis
            .serialize_with_mode(&mut writer, Compress::No)?;
        write_usize(&mut writer, profile.inner.digits.num_digits)?;
        let inner_table_key = profile.inner.matrix.sis_table_key().ok_or_else(|| {
            SerializationError::InvalidData(
                "precommitted group cannot use an L2 A security route".into(),
            )
        })?;
        for matrix in [inner_table_key, profile.outer.matrix.sis_table_key()] {
            matrix
                .modulus_profile
                .tag()
                .serialize_with_mode(&mut writer, Compress::No)?;
            matrix
                .policy
                .tag()
                .serialize_with_mode(&mut writer, Compress::No)?;
            matrix
                .role
                .tag()
                .serialize_with_mode(&mut writer, Compress::No)?;
            writer.write_all(&matrix.table_digest.0)?;
            matrix
                .ring_dimension
                .serialize_with_mode(&mut writer, Compress::No)?;
            let params = if matrix.role == SisMatrixRole::Inner {
                (
                    profile.inner.matrix.output_rank(),
                    profile.inner.matrix.input_width(),
                )
            } else {
                (
                    profile.outer.matrix.output_rank(),
                    profile.outer.matrix.input_width(),
                )
            };
            write_usize(&mut writer, params.0)?;
            write_usize(&mut writer, params.1)?;
            matrix
                .coeff_linf_bound
                .serialize_with_mode(&mut writer, Compress::No)?;
            if matrix.role == SisMatrixRole::Inner {
                profile
                    .outer
                    .digits
                    .log_basis
                    .serialize_with_mode(&mut writer, Compress::No)?;
                write_usize(&mut writer, profile.outer.digits.num_digits)?;
            }
        }
        self.commitment.serialize_with_mode(&mut writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        const MATRIX_SIZE: usize = 1 + 1 + 1 + 32 + 4 + 8 + 8 + 16;
        1 + 16
            + 32
            + 4
            + 8
            + MATRIX_SIZE
            + 4
            + 8
            + MATRIX_SIZE
            + self.commitment.serialized_size(compress)
    }
}

impl<F> AkitaDeserialize for CommittedGroup<F>
where
    F: Field + CanonicalEncoding + Valid + AkitaSerialize + AkitaDeserialize<Context = ()>,
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

        fn read_matrix_fields<R: Read>(
            reader: &mut R,
            expected_role: SisMatrixRole,
        ) -> Result<MatrixFields, SerializationError> {
            let modulus_tag =
                u8::deserialize_with_mode(&mut *reader, Compress::No, Validate::Yes, &())?;
            let policy_tag =
                u8::deserialize_with_mode(&mut *reader, Compress::No, Validate::Yes, &())?;
            let role_tag =
                u8::deserialize_with_mode(&mut *reader, Compress::No, Validate::Yes, &())?;
            let modulus_profile = SisModulusProfileId::from_tag(modulus_tag).ok_or_else(|| {
                SerializationError::InvalidData("unknown SIS modulus-profile tag".into())
            })?;
            let policy = SisSecurityPolicyId::from_tag(policy_tag).ok_or_else(|| {
                SerializationError::InvalidData("unknown SIS security-policy tag".into())
            })?;
            let role = SisMatrixRole::from_tag(role_tag).ok_or_else(|| {
                SerializationError::InvalidData("unknown SIS matrix-role tag".into())
            })?;
            if role != expected_role {
                return Err(SerializationError::InvalidData(
                    "committed-group matrix role mismatch".into(),
                ));
            }
            let mut digest = [0u8; 32];
            reader.read_exact(&mut digest)?;
            let ring_dimension =
                u32::deserialize_with_mode(&mut *reader, Compress::No, Validate::Yes, &())?
                    as usize;
            let output_rank = read_usize(reader)?;
            let input_width = read_usize(reader)?;
            let coeff_linf_bound =
                u128::deserialize_with_mode(&mut *reader, Compress::No, Validate::Yes, &())?;
            Ok((
                policy,
                SisTableDigest(digest),
                modulus_profile,
                output_rank,
                input_width,
                coeff_linf_bound,
                ring_dimension,
            ))
        }

        let version = u8::deserialize_with_mode(&mut reader, Compress::No, Validate::Yes, &())?;
        if version != GroupCommitPhaseParams::VERSION {
            return Err(SerializationError::InvalidData(format!(
                "unknown committed-group profile version {version}"
            )));
        }
        let num_vars = read_usize(&mut reader)?;
        let num_polynomials = read_usize(&mut reader)?;
        let group = PolynomialGroupLayout::new(num_vars, num_polynomials);
        let num_live_ring_elements_per_claim = read_usize(&mut reader)?;
        let num_positions_per_block = read_usize(&mut reader)?;
        let num_live_blocks = read_usize(&mut reader)?;
        let outer_slice_count = CommitmentSliceCount::try_new(read_usize(&mut reader)?)
            .map_err(|err| SerializationError::InvalidData(err.to_string()))?;
        let log_basis_inner =
            u32::deserialize_with_mode(&mut reader, Compress::No, Validate::Yes, &())?;
        let num_digits_inner = read_usize(&mut reader)?;
        let (a_policy, a_digest, a_modulus, n_a, a_width, a_coeff_linf_bound, inner_ring_dimension) =
            read_matrix_fields(&mut reader, SisMatrixRole::Inner)?;
        let inner_commit_matrix = InnerCommitMatrixParams::try_new(
            a_policy,
            a_digest,
            a_modulus,
            n_a,
            a_width,
            a_coeff_linf_bound,
            inner_ring_dimension,
        )
        .map_err(|err| SerializationError::InvalidData(err.to_string()))?;
        let log_basis_outer =
            u32::deserialize_with_mode(&mut reader, Compress::No, Validate::Yes, &())?;
        let num_digits_outer = read_usize(&mut reader)?;
        let (b_policy, b_digest, b_modulus, n_b, b_width, b_coeff_linf_bound, outer_ring_dimension) =
            read_matrix_fields(&mut reader, SisMatrixRole::Outer)?;
        let outer_commit_matrix = OuterCommitMatrixParams::try_new(
            b_policy,
            b_digest,
            b_modulus,
            n_b,
            b_width,
            b_coeff_linf_bound,
            outer_ring_dimension,
        )
        .map_err(|err| SerializationError::InvalidData(err.to_string()))?;

        let descriptor = GroupCommitPhaseParams {
            version,
            group,

            blocks: crate::BlockGeometry::new(
                num_live_ring_elements_per_claim,
                num_positions_per_block,
                num_live_blocks,
            ),

            outer_slice_count,
            inner: crate::RoleParams::new(
                crate::GadgetDigits::new(log_basis_inner, num_digits_inner),
                inner_commit_matrix,
            ),
            outer: crate::RoleParams::new(
                crate::GadgetDigits::new(log_basis_outer, num_digits_outer),
                outer_commit_matrix,
            ),
        };
        let field_bits = F::MODULUS_BITS;
        descriptor
            .validate_frozen_precommit(field_bits)
            .map_err(|err| SerializationError::InvalidData(err.to_string()))?;
        let source_coefficients = descriptor
            .outer_slice_count
            .complete_source_coefficients(
                descriptor.outer.matrix.output_rank(),
                descriptor.outer.matrix.ring_dimension(),
            )
            .map_err(|err| SerializationError::InvalidData(err.to_string()))?;
        let num_coeffs = CompressionChainPlan::for_complete_source(
            descriptor.outer.matrix.sis_table_key().modulus_profile,
            source_coefficients,
        )
        .map_err(|error| SerializationError::InvalidData(error.to_string()))?
        .terminal_coefficients();
        if num_coeffs > MAX_UNTRUSTED_COMMITMENT_COEFFICIENTS {
            return Err(SerializationError::InvalidData(format!(
                "committed-group coefficient count {num_coeffs} exceeds allocation cap \
                 {MAX_UNTRUSTED_COMMITMENT_COEFFICIENTS}"
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

impl<F: Field + Valid> Valid for Commitment<F> {
    fn check(&self) -> Result<(), SerializationError> {
        self.0.check()
    }
}

impl<F: Field + AkitaSerialize> AkitaSerialize for Commitment<F> {
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

impl<F: Field + Valid + AkitaDeserialize<Context = ()>> AkitaDeserialize for Commitment<F> {
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
    use jolt_field::{Fp32, Zero};

    type F = Fp32<4294967197>;

    fn group() -> CommittedGroup<F> {
        let a_bound = *crate::sis::inner_coeff_linf_bounds(SisModulusProfileId::Q32Offset99, 64)
            .first()
            .expect("D64 exact A bounds");
        let inner_commit_matrix = InnerCommitMatrixParams::try_new_with_min_rank(
            crate::SisTableKey {
                policy: crate::sis::DEFAULT_SIS_SECURITY_POLICY,
                table_digest: SisTableDigest::CURRENT,
                modulus_profile: SisModulusProfileId::Q32Offset99,
                role: SisMatrixRole::Inner,
                ring_dimension: 64,
                coeff_linf_bound: a_bound,
            },
            32,
        )
        .expect("audited A profile");
        let outer_width = inner_commit_matrix.output_rank();
        let outer_commit_matrix = OuterCommitMatrixParams::try_new_with_min_rank(
            crate::SisTableKey {
                policy: crate::sis::DEFAULT_SIS_SECURITY_POLICY,
                table_digest: SisTableDigest::CURRENT,
                modulus_profile: SisModulusProfileId::Q32Offset99,
                role: SisMatrixRole::Outer,
                ring_dimension: 64,
                coeff_linf_bound: 3,
            },
            outer_width,
        )
        .expect("audited B profile");
        let profile = GroupCommitPhaseParams {
            version: GroupCommitPhaseParams::VERSION,
            group: PolynomialGroupLayout::new(11, 1),
            blocks: crate::BlockGeometry::new(32, 32, 1),
            outer_slice_count: CommitmentSliceCount::ONE,
            inner: crate::RoleParams::new(crate::GadgetDigits::new(1, 1), inner_commit_matrix),
            outer: crate::RoleParams::new(crate::GadgetDigits::new(1, 1), outer_commit_matrix),
        };
        let source_coefficients = outer_commit_matrix.output_rank() * 64;
        let payload_coefficients = crate::CompressionChainPlan::for_complete_source(
            outer_commit_matrix.sis_modulus_profile(),
            source_coefficients,
        )
        .expect("compression plan")
        .terminal_coefficients();
        CommittedGroup::new(
            profile,
            Commitment::new(RingVec::from_coeffs(vec![F::zero(); payload_coefficients])),
        )
    }

    #[test]
    fn committed_group_serialization_binds_and_validates_profile() {
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

        let mut unknown_version = bytes.clone();
        unknown_version[0] = GroupCommitPhaseParams::VERSION + 1;
        assert!(CommittedGroup::<F>::deserialize_with_mode(
            unknown_version.as_slice(),
            Compress::Yes,
            Validate::Yes,
            &(),
        )
        .is_err());

        let mut previous_version = bytes.clone();
        previous_version[0] = GroupCommitPhaseParams::VERSION - 1;
        assert!(CommittedGroup::<F>::deserialize_with_mode(
            previous_version.as_slice(),
            Compress::Yes,
            Validate::Yes,
            &(),
        )
        .is_err());

        let mut invalid_slice_count = bytes.clone();
        let slice_count_offset = 1 + 2 * 8 + 3 * 8;
        invalid_slice_count[slice_count_offset..slice_count_offset + 8]
            .copy_from_slice(&3u64.to_le_bytes());
        assert!(CommittedGroup::<F>::deserialize_with_mode(
            invalid_slice_count.as_slice(),
            Compress::Yes,
            Validate::Yes,
            &(),
        )
        .is_err());

        let inner_matrix_role_offset = 1 + 2 * 8 + 1 + 3 * 8 + 4 + 8 + 2;
        let mut wrong_matrix_role = bytes;
        wrong_matrix_role[inner_matrix_role_offset] = SisMatrixRole::Outer.tag();
        assert!(CommittedGroup::<F>::deserialize_with_mode(
            wrong_matrix_role.as_slice(),
            Compress::Yes,
            Validate::Yes,
            &(),
        )
        .is_err());
    }

    #[test]
    fn committed_group_rejects_commitment_row_count_mismatch() {
        let mut group = group();
        let coeffs = group.commitment.rows().coeffs();
        group.commitment =
            Commitment::new(RingVec::from_coeffs(coeffs[..coeffs.len() - 1].to_vec()));
        assert!(group.check().is_err());
    }

    #[test]
    fn committed_group_rejects_slicing_geometry_mutations_without_panicking() {
        let baseline = group();
        let outer = baseline.profile.outer.matrix;
        let mut malformed = Vec::new();

        let mut wrong_count = baseline.clone();
        wrong_count.profile.outer_slice_count = CommitmentSliceCount::TWO;
        malformed.push(wrong_count);

        let mut wrong_polynomial_count = baseline.clone();
        wrong_polynomial_count.profile.group = PolynomialGroupLayout::new(11, 2);
        malformed.push(wrong_polynomial_count);

        let mut wrong_physical_width = baseline.clone();
        wrong_physical_width.profile.outer.matrix = OuterCommitMatrixParams::new_unchecked(
            outer.security_policy(),
            outer.sis_table_key().table_digest,
            outer.sis_modulus_profile(),
            outer.output_rank(),
            outer.input_width() + 1,
            outer.coeff_linf_bound(),
            outer.ring_dimension(),
        );
        malformed.push(wrong_physical_width);

        for candidate in malformed {
            let result = std::panic::catch_unwind(|| candidate.check());
            assert!(result.is_ok(), "malformed descriptor must not panic");
            assert!(result.unwrap().is_err(), "malformed descriptor must reject");
        }
    }

    #[test]
    fn committed_group_reaudits_unchecked_sis_descriptors() {
        let baseline = group();
        let inner = baseline.profile.inner.matrix;
        let malformed = [
            InnerCommitMatrixParams::new_unchecked(
                inner.security_policy(),
                SisTableDigest([0; 32]),
                inner.sis_modulus_profile(),
                inner.output_rank(),
                inner.input_width(),
                inner.coeff_linf_bound().expect("L infinity test matrix"),
                inner.ring_dimension(),
            ),
            InnerCommitMatrixParams::new_unchecked(
                inner.security_policy(),
                inner
                    .sis_table_key()
                    .expect("L infinity test matrix")
                    .table_digest,
                inner.sis_modulus_profile(),
                inner.output_rank().saturating_sub(1),
                inner.input_width(),
                inner.coeff_linf_bound().expect("L infinity test matrix"),
                inner.ring_dimension(),
            ),
            InnerCommitMatrixParams::new_unchecked(
                inner.security_policy(),
                inner
                    .sis_table_key()
                    .expect("L infinity test matrix")
                    .table_digest,
                inner.sis_modulus_profile(),
                inner.output_rank(),
                inner.input_width(),
                inner.coeff_linf_bound().expect("L infinity test matrix") - 1,
                inner.ring_dimension(),
            ),
        ];

        for matrix in malformed {
            let mut candidate = baseline.clone();
            candidate.profile.inner.matrix = matrix;
            assert!(candidate.check().is_err());
            assert!(candidate
                .serialize_with_mode(Vec::new(), Compress::Yes)
                .is_err());
        }
    }
}

/// Ring-native commitment object `u in R_q^{n_B}` used by §4.1.
///
/// **Arithmetic-only leaf helper.** As of S4 this type is no longer used for
/// protocol-facing storage, serialization, or transcript absorption — that role
/// belongs to the D-free [`Commitment`] / [`RingVec`]. It is kept solely as a
/// typed arithmetic carrier inside kernels.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RingCommitment<F: Field, const D: usize> {
    /// Outer commitment vector.
    pub u: Vec<CyclotomicRing<F, D>>,
}

/// Borrow ring rows from commitment-like prover inputs.
pub trait ProverCommitmentRows<CommitF: Field, const D: usize> {
    fn commitment_rows(&self) -> &[CyclotomicRing<CommitF, D>];
}

impl<CommitF: Field, const D: usize> ProverCommitmentRows<CommitF, D>
    for RingCommitment<CommitF, D>
{
    fn commitment_rows(&self) -> &[CyclotomicRing<CommitF, D>] {
        &self.u
    }
}

impl<CommitF: Field, const D: usize> ProverCommitmentRows<CommitF, D>
    for [CyclotomicRing<CommitF, D>]
{
    fn commitment_rows(&self) -> &[CyclotomicRing<CommitF, D>] {
        self
    }
}

impl<F: Field + Valid, const D: usize> Valid for RingCommitment<F, D> {
    fn check(&self) -> Result<(), SerializationError> {
        self.u.check()
    }
}

impl<F: Field + AkitaSerialize, const D: usize> AkitaSerialize for RingCommitment<F, D> {
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

impl<F: Field + Valid + AkitaDeserialize<Context = ()>, const D: usize> AkitaDeserialize
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
