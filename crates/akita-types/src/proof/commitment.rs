//! Protocol commitment/opening wrapper types.

use crate::proof::{RingVec, MAX_UNTRUSTED_COMMITMENT_COEFFICIENTS};
use crate::sis::{
    InnerCommitMatrixParams, OuterCommitMatrixParams, SisMatrixRole, SisModulusProfileId,
    SisSecurityPolicyId, SisTableDigest,
};
use crate::transcript::AppendToTranscript;
use crate::{
    detect_field_modulus, CommittedGroupProfile, CompressionChainPlan, PolynomialGroupLayout,
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

/// D-free public commitment payload stored as flat field coefficients.
///
/// For a committed polynomial group this carries the terminal compressed
/// payload `p_F`. Its native compression dimension is derived from the frozen
/// modulus profile and supplied at the transcript boundary rather than encoded
/// as a const generic.
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
        F: CanonicalField,
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
pub struct CommittedGroup<F: FieldCore> {
    /// Exact public algebraic profile and commitment geometry.
    pub profile: CommittedGroupProfile,
    /// Terminal compressed `p_F` payload.
    pub commitment: Commitment<F>,
}

impl<F: FieldCore> CommittedGroup<F> {
    /// Build a self-describing group commitment.
    pub fn new(profile: CommittedGroupProfile, commitment: Commitment<F>) -> Self {
        Self {
            profile,
            commitment,
        }
    }

    /// Borrow the exact frozen commitment profile.
    pub fn profile(&self) -> &CommittedGroupProfile {
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

impl<F: FieldCore + CanonicalField + Valid> Valid for CommittedGroup<F> {
    fn check(&self) -> Result<(), SerializationError> {
        let field_bits = 128 - (detect_field_modulus::<F>() - 1).leading_zeros();
        self.profile
            .validate_frozen_precommit(field_bits)
            .map_err(|err| SerializationError::InvalidData(err.to_string()))?;
        self.commitment.check()?;
        let source_coefficients = self
            .profile
            .outer_commit_matrix
            .output_rank()
            .checked_mul(self.profile.outer_commit_matrix.ring_dimension())
            .ok_or_else(|| {
                SerializationError::InvalidData(
                    "committed-group coefficient count overflow".to_string(),
                )
            })?;
        let expected_coeffs = CompressionChainPlan::for_complete_source(
            self.profile
                .outer_commit_matrix
                .sis_table_key()
                .modulus_profile,
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
        let profile = &self.profile;
        profile
            .version
            .serialize_with_mode(&mut writer, Compress::No)?;
        write_usize(&mut writer, profile.group.num_vars())?;
        write_usize(&mut writer, profile.group.num_polynomials())?;
        for value in [
            profile.num_live_ring_elements_per_claim,
            profile.num_positions_per_block,
            profile.num_live_blocks,
        ] {
            write_usize(&mut writer, value)?;
        }
        profile
            .log_basis_inner
            .serialize_with_mode(&mut writer, Compress::No)?;
        write_usize(&mut writer, profile.num_digits_inner)?;
        for matrix in [
            profile.inner_commit_matrix.sis_table_key(),
            profile.outer_commit_matrix.sis_table_key(),
        ] {
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
                    profile.inner_commit_matrix.output_rank(),
                    profile.inner_commit_matrix.input_width(),
                )
            } else {
                (
                    profile.outer_commit_matrix.output_rank(),
                    profile.outer_commit_matrix.input_width(),
                )
            };
            write_usize(&mut writer, params.0)?;
            write_usize(&mut writer, params.1)?;
            matrix
                .coeff_linf_bound
                .serialize_with_mode(&mut writer, Compress::No)?;
            if matrix.role == SisMatrixRole::Inner {
                profile
                    .log_basis_outer
                    .serialize_with_mode(&mut writer, Compress::No)?;
                write_usize(&mut writer, profile.num_digits_outer)?;
            }
        }
        self.commitment.serialize_with_mode(&mut writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        const MATRIX_SIZE: usize = 1 + 1 + 1 + 32 + 4 + 8 + 8 + 16;
        1 + 16
            + 24
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
        if version != CommittedGroupProfile::VERSION {
            return Err(SerializationError::InvalidData(format!(
                "unknown committed-group profile version {version}"
            )));
        }
        let num_vars = read_usize(&mut reader)?;
        let num_polynomials = read_usize(&mut reader)?;
        let num_live_ring_elements_per_claim = read_usize(&mut reader)?;
        let num_positions_per_block = read_usize(&mut reader)?;
        let num_live_blocks = read_usize(&mut reader)?;
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

        let descriptor = CommittedGroupProfile {
            version,
            group: PolynomialGroupLayout::new(num_vars, num_polynomials),
            num_live_ring_elements_per_claim,
            num_positions_per_block,
            num_live_blocks,
            log_basis_inner,
            num_digits_inner,
            inner_commit_matrix,
            log_basis_outer,
            num_digits_outer,
            outer_commit_matrix,
        };
        let field_bits = 128 - (detect_field_modulus::<F>() - 1).leading_zeros();
        descriptor
            .validate_frozen_precommit(field_bits)
            .map_err(|err| SerializationError::InvalidData(err.to_string()))?;
        let source_coefficients = descriptor
            .outer_commit_matrix
            .output_rank()
            .checked_mul(descriptor.outer_commit_matrix.ring_dimension())
            .ok_or_else(|| {
                SerializationError::InvalidData(
                    "committed-group coefficient count overflow".to_string(),
                )
            })?;
        let num_coeffs = CompressionChainPlan::for_complete_source(
            descriptor
                .outer_commit_matrix
                .sis_table_key()
                .modulus_profile,
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

    type F = Fp32<4294967197>;

    fn group() -> CommittedGroup<F> {
        let inner_commit_matrix = InnerCommitMatrixParams::try_new_with_min_rank(
            crate::SisTableKey {
                policy: crate::sis::DEFAULT_SIS_SECURITY_POLICY,
                table_digest: SisTableDigest::CURRENT,
                modulus_profile: SisModulusProfileId::Q32Offset99,
                role: SisMatrixRole::Inner,
                ring_dimension: 64,
                coeff_linf_bound: 131_071,
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
        let profile = CommittedGroupProfile {
            version: CommittedGroupProfile::VERSION,
            group: PolynomialGroupLayout::new(11, 1),
            num_live_ring_elements_per_claim: 32,
            num_positions_per_block: 32,
            num_live_blocks: 1,
            log_basis_inner: 1,
            num_digits_inner: 1,
            inner_commit_matrix,
            log_basis_outer: 1,
            num_digits_outer: 1,
            outer_commit_matrix,
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
        unknown_version[0] = CommittedGroupProfile::VERSION + 1;
        assert!(CommittedGroup::<F>::deserialize_with_mode(
            unknown_version.as_slice(),
            Compress::Yes,
            Validate::Yes,
            &(),
        )
        .is_err());

        let mut previous_version = bytes.clone();
        previous_version[0] = CommittedGroupProfile::VERSION - 1;
        assert!(CommittedGroup::<F>::deserialize_with_mode(
            previous_version.as_slice(),
            Compress::Yes,
            Validate::Yes,
            &(),
        )
        .is_err());

        let inner_matrix_role_offset = 1 + 2 * 8 + 3 * 8 + 4 + 8 + 2;
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
    fn committed_group_reaudits_unchecked_sis_descriptors() {
        let baseline = group();
        let inner = baseline.profile.inner_commit_matrix;
        let malformed = [
            InnerCommitMatrixParams::new_unchecked(
                inner.security_policy(),
                SisTableDigest([0; 32]),
                inner.sis_modulus_profile(),
                inner.output_rank(),
                inner.input_width(),
                inner.coeff_linf_bound(),
                inner.ring_dimension(),
            ),
            InnerCommitMatrixParams::new_unchecked(
                inner.security_policy(),
                inner.sis_table_key().table_digest,
                inner.sis_modulus_profile(),
                inner.output_rank().saturating_sub(1),
                inner.input_width(),
                inner.coeff_linf_bound(),
                inner.ring_dimension(),
            ),
            InnerCommitMatrixParams::new_unchecked(
                inner.security_policy(),
                inner.sis_table_key().table_digest,
                inner.sis_modulus_profile(),
                inner.output_rank(),
                inner.input_width(),
                inner.coeff_linf_bound() - 1,
                inner.ring_dimension(),
            ),
        ];

        for matrix in malformed {
            let mut candidate = baseline.clone();
            candidate.profile.inner_commit_matrix = matrix;
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
