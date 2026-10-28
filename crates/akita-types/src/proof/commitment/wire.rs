//! Extension seam: commitment-kind validation and wire dispatch.
//!
//! Tracking issue: <https://github.com/LayerZero-Labs/akita/issues/285>.
//! Ordinary commitments retain their existing version-led encoding.

use super::{Commitment, CommittedGroup, CommittedGroupKind};
use crate::proof::MAX_UNTRUSTED_COMMITMENT_COEFFICIENTS;
use akita_params::sis::{
    InnerCommitMatrixParams, OuterCommitMatrixParams, SisMatrixRole, SisModulusProfileId,
    SisSecurityPolicyId, SisTableDigest,
};
use akita_params::{CommitmentSliceCount, GroupCommitPhaseParams, PolynomialGroupLayout};
use akita_serialization::{
    AkitaDeserialize, AkitaSerialize, Compress, SerializationError, Valid, Validate,
};
use jolt_field::{CanonicalEncoding, Field};
use std::io::{Read, Write};

type MatrixFields = (
    SisSecurityPolicyId,
    SisTableDigest,
    SisModulusProfileId,
    usize,
    usize,
    u128,
    usize,
);

/// Reject a profile whose A or B matrix names a SIS modulus profile other than
/// the modulus of `F`.
///
/// `GroupCommitPhaseParams::validate_frozen_precommit` compares bit widths,
/// which two profiles can share.
fn check_exact_field_modulus<F: Field + CanonicalEncoding>(
    profile: &GroupCommitPhaseParams,
) -> Result<(), SerializationError> {
    let field_modulus = akita_params::field_modulus::<F>()
        .map_err(|err| SerializationError::InvalidData(err.to_string()))?;
    for matrix_profile in [
        profile.inner.matrix.sis_modulus_profile(),
        profile.outer.matrix.sis_modulus_profile(),
    ] {
        if !matrix_profile.matches_modulus(field_modulus) {
            return Err(SerializationError::InvalidData(
                "committed-group SIS modulus profile does not match the field".into(),
            ));
        }
    }
    Ok(())
}

pub(crate) fn serialize_profile<W: Write>(
    profile: &GroupCommitPhaseParams,
    mut writer: W,
) -> Result<(), SerializationError> {
    fn write_usize<W: Write>(writer: &mut W, value: usize) -> Result<(), SerializationError> {
        u64::try_from(value)
            .map_err(|_| {
                SerializationError::InvalidData("committed-group integer exceeds u64".to_string())
            })?
            .serialize_with_mode(writer, Compress::No)
    }

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
    Ok(())
}

fn deserialize_profile<F: Field + CanonicalEncoding, R: Read>(
    mut reader: R,
) -> Result<GroupCommitPhaseParams, SerializationError> {
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
        let policy_tag = u8::deserialize_with_mode(&mut *reader, Compress::No, Validate::Yes, &())?;
        let role_tag = u8::deserialize_with_mode(&mut *reader, Compress::No, Validate::Yes, &())?;
        let modulus_profile = SisModulusProfileId::from_tag(modulus_tag).ok_or_else(|| {
            SerializationError::InvalidData("unknown SIS modulus-profile tag".into())
        })?;
        let policy = SisSecurityPolicyId::from_tag(policy_tag).ok_or_else(|| {
            SerializationError::InvalidData("unknown SIS security-policy tag".into())
        })?;
        let role = SisMatrixRole::from_tag(role_tag)
            .ok_or_else(|| SerializationError::InvalidData("unknown SIS matrix-role tag".into()))?;
        if role != expected_role {
            return Err(SerializationError::InvalidData(
                "committed-group matrix role mismatch".into(),
            ));
        }
        let mut digest = [0u8; 32];
        reader.read_exact(&mut digest)?;
        let ring_dimension =
            u32::deserialize_with_mode(&mut *reader, Compress::No, Validate::Yes, &())? as usize;
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
    // `try_new` resolves a bound to its audited table row and stores that
    // row's bound, so only the row's own bound is a canonical encoding.
    if inner_commit_matrix.coeff_linf_bound() != Some(a_coeff_linf_bound) {
        return Err(SerializationError::InvalidData(
            "committed-group A coefficient bound is not its SIS table's bound".into(),
        ));
    }
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
    if outer_commit_matrix.coeff_linf_bound() != b_coeff_linf_bound {
        return Err(SerializationError::InvalidData(
            "committed-group B coefficient bound is not its SIS table's bound".into(),
        ));
    }

    let descriptor = GroupCommitPhaseParams {
        version,
        group,

        blocks: akita_params::BlockGeometry::new(
            num_live_ring_elements_per_claim,
            num_positions_per_block,
            num_live_blocks,
        ),

        outer_slice_count,
        inner: akita_params::RoleParams::new(
            akita_params::GadgetDigits::new(log_basis_inner, num_digits_inner),
            inner_commit_matrix,
        ),
        outer: akita_params::RoleParams::new(
            akita_params::GadgetDigits::new(log_basis_outer, num_digits_outer),
            outer_commit_matrix,
        ),
    };
    let field_bits = F::MODULUS_BITS;
    descriptor
        .validate_frozen_precommit(field_bits)
        .map_err(|err| SerializationError::InvalidData(err.to_string()))?;
    check_exact_field_modulus::<F>(&descriptor)?;
    Ok(descriptor)
}

impl<F: Field + CanonicalEncoding + Valid> Valid for CommittedGroup<F> {
    fn check(&self) -> Result<(), SerializationError> {
        let field_bits = F::MODULUS_BITS;
        self.profile
            .validate_frozen_precommit(field_bits)
            .map_err(|err| SerializationError::InvalidData(err.to_string()))?;
        check_exact_field_modulus::<F>(&self.profile)?;
        self.commitment.check()?;
        let expected_coeffs = match &self.kind {
            CommittedGroupKind::Ordinary => self
                .profile
                .outer_compression_plan()
                .map_err(|error| SerializationError::InvalidData(error.to_string()))?
                .terminal_coefficients(),
            #[cfg(feature = "labinius-trinomial")]
            CommittedGroupKind::Binary(manifest) => {
                manifest
                    .validate_carrier(&self.profile)
                    .map_err(|error| SerializationError::InvalidData(error.to_string()))?;
                crate::binary_source::BINARY_COMMITTED_PAYLOAD_COEFFICIENTS
            }
        };
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
        self.check()?;
        #[cfg(feature = "labinius-trinomial")]
        if let CommittedGroupKind::Binary(manifest) = &self.kind {
            let bytes = manifest.canonical_bytes();
            let len = u32::try_from(bytes.len()).map_err(|_| {
                SerializationError::InvalidData("binary commitment manifest exceeds u32".into())
            })?;
            255u8.serialize_with_mode(&mut writer, Compress::No)?;
            len.serialize_with_mode(&mut writer, Compress::No)?;
            writer.write_all(&bytes)?;
        }
        serialize_profile(&self.profile, &mut writer)?;
        self.commitment.serialize_with_mode(&mut writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        const MATRIX_SIZE: usize = 1 + 1 + 1 + 32 + 4 + 8 + 8 + 16;
        let ordinary_size = 1
            + 16
            + 32
            + 4
            + 8
            + MATRIX_SIZE
            + 4
            + 8
            + MATRIX_SIZE
            + self.commitment.serialized_size(compress);
        match &self.kind {
            CommittedGroupKind::Ordinary => ordinary_size,
            #[cfg(feature = "labinius-trinomial")]
            CommittedGroupKind::Binary(manifest) => {
                akita_error::checked::sum([ordinary_size, 5, manifest.canonical_bytes().len()])
                    .unwrap_or(usize::MAX)
            }
        }
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
        let (kind, descriptor) = deserialize_kind_and_profile::<F, _>(&mut reader)?;
        let num_coeffs = match &kind {
            CommittedGroupKind::Ordinary => descriptor
                .outer_compression_plan()
                .map_err(|error| SerializationError::InvalidData(error.to_string()))?
                .terminal_coefficients(),
            #[cfg(feature = "labinius-trinomial")]
            CommittedGroupKind::Binary(manifest) => {
                manifest
                    .validate_carrier(&descriptor)
                    .map_err(|error| SerializationError::InvalidData(error.to_string()))?;
                crate::binary_source::BINARY_COMMITTED_PAYLOAD_COEFFICIENTS
            }
        };
        if num_coeffs > MAX_UNTRUSTED_COMMITMENT_COEFFICIENTS {
            return Err(SerializationError::InvalidData(format!(
                "committed-group coefficient count {num_coeffs} exceeds allocation cap \
                 {MAX_UNTRUSTED_COMMITMENT_COEFFICIENTS}"
            )));
        }
        let commitment =
            Commitment::deserialize_with_mode(&mut reader, compress, validate, &num_coeffs)?;
        let group = Self {
            profile: descriptor,
            commitment,
            kind,
        };
        if matches!(validate, Validate::Yes) {
            group.check()?;
        }
        Ok(group)
    }
}

fn deserialize_kind_and_profile<F: Field + CanonicalEncoding, R: Read>(
    mut reader: R,
) -> Result<(CommittedGroupKind, GroupCommitPhaseParams), SerializationError> {
    let tag = u8::deserialize_with_mode(&mut reader, Compress::No, Validate::Yes, &())?;
    match tag {
        #[cfg(feature = "labinius-trinomial")]
        255 => {
            use crate::binary_source::{
                BinaryCommittedGroupManifest, MAX_BINARY_COMMITTED_MANIFEST_BYTES,
            };
            let len = u32::deserialize_with_mode(&mut reader, Compress::No, Validate::Yes, &())?;
            let len = usize::try_from(len).map_err(|_| {
                SerializationError::InvalidData(
                    "binary commitment manifest length exceeds usize".into(),
                )
            })?;
            if len > MAX_BINARY_COMMITTED_MANIFEST_BYTES {
                return Err(SerializationError::InvalidData(
                    "binary commitment manifest exceeds cap".into(),
                ));
            }
            let mut bytes = Vec::new();
            bytes.try_reserve_exact(len).map_err(|_| {
                SerializationError::InvalidData(
                    "binary commitment manifest allocation failed".into(),
                )
            })?;
            bytes.resize(len, 0);
            reader.read_exact(&mut bytes)?;
            let manifest = BinaryCommittedGroupManifest::from_bytes(&bytes)?;
            let profile = deserialize_profile::<F, _>(&mut reader)?;
            Ok((CommittedGroupKind::Binary(manifest), profile))
        }
        _ => {
            let prefixed = std::io::Cursor::new([tag]).chain(reader);
            Ok((
                CommittedGroupKind::Ordinary,
                deserialize_profile::<F, _>(prefixed)?,
            ))
        }
    }
}
