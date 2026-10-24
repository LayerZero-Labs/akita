//! Protocol commitment/opening wrapper types.

use crate::proof::{RingVec, MAX_UNTRUSTED_COMMITMENT_COEFFICIENTS};
use akita_params::sis::{
    InnerCommitMatrixParams, OuterCommitMatrixParams, SisMatrixRole, SisModulusProfileId,
    SisSecurityPolicyId, SisTableDigest,
};
use akita_params::{CommitmentSliceCount, GroupCommitPhaseParams, PolynomialGroupLayout};

type MatrixFields = (
    SisSecurityPolicyId,
    SisTableDigest,
    SisModulusProfileId,
    usize,
    usize,
    u128,
    usize,
);
use akita_serialization::{
    AkitaDeserialize, AkitaSerialize, Compress, SerializationError, Valid, Validate,
};
use akita_transcript::CommitmentDigest;
use jolt_field::{CanonicalEncoding, Field};
use std::io::{Read, Write};

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

    /// Borrow the underlying flat ring-coefficient buffer.
    pub fn rows(&self) -> &RingVec<F> {
        &self.0
    }
}

/// Public commitment to one polynomial group together with its frozen contract.
///
/// The profile is the exact commitment identity selected when the group was
/// committed. Callers pass this object through claims so proving and
/// verification never reconstruct group metadata from a bare layout.
///
/// The group carries the [`CommitmentDigest`] of its canonical serialization.
/// Every constructor, the deserializer included, computes the digest from the
/// profile and rows it holds; no path accepts a digest from outside.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommittedGroup<F: Field> {
    profile: GroupCommitPhaseParams,
    commitment: Commitment<F>,
    digest: CommitmentDigest,
}

impl<F: Field + CanonicalEncoding + Valid + AkitaSerialize> CommittedGroup<F> {
    /// Build a self-describing group commitment and compute its digest.
    ///
    /// # Errors
    ///
    /// Returns an error when the profile is not a valid frozen profile, a
    /// coefficient is not canonical, or the rows do not have the length the
    /// profile fixes.
    pub fn new(
        profile: GroupCommitPhaseParams,
        commitment: Commitment<F>,
    ) -> Result<Self, SerializationError> {
        check_group(&profile, &commitment)?;
        let mut bytes = Vec::new();
        write_group(&profile, &commitment, &mut bytes, Compress::No)?;
        Ok(Self {
            profile,
            commitment,
            digest: CommitmentDigest::of_canonical_bytes(&bytes),
        })
    }
}

impl<F: Field> CommittedGroup<F> {
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

    /// The digest that binds this group into a transcript.
    pub fn digest(&self) -> &CommitmentDigest {
        &self.digest
    }
}

fn check_group<F: Field + CanonicalEncoding + Valid>(
    profile: &GroupCommitPhaseParams,
    commitment: &Commitment<F>,
) -> Result<(), SerializationError> {
    profile
        .validate_frozen_precommit(F::MODULUS_BITS)
        .map_err(|err| SerializationError::InvalidData(err.to_string()))?;
    commitment.check()?;
    let expected_coeffs = profile
        .outer_compression_plan()
        .map_err(|error| SerializationError::InvalidData(error.to_string()))?
        .terminal_coefficients();
    if commitment.rows().coeff_len() != expected_coeffs {
        return Err(SerializationError::InvalidData(
            "committed-group rows do not match the frozen descriptor".to_string(),
        ));
    }
    Ok(())
}

impl<F: Field + CanonicalEncoding + Valid> Valid for CommittedGroup<F> {
    fn check(&self) -> Result<(), SerializationError> {
        check_group(&self.profile, &self.commitment)
    }
}

/// Write the profile header and then the rows: the canonical group encoding.
fn write_group<F: Field + AkitaSerialize, W: Write>(
    profile: &GroupCommitPhaseParams,
    commitment: &Commitment<F>,
    mut writer: W,
    compress: Compress,
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
    commitment.serialize_with_mode(&mut writer, compress)
}

impl<F: Field + CanonicalEncoding + Valid + AkitaSerialize> AkitaSerialize for CommittedGroup<F> {
    fn serialize_with_mode<W: Write>(
        &self,
        writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.check()?;
        write_group(&self.profile, &self.commitment, writer, compress)
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
        let num_coeffs = descriptor
            .outer_compression_plan()
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
        // The group is checked and its digest computed from the decoded
        // value, whatever `validate` asks for.
        Self::new(descriptor, commitment)
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
    use jolt_field::{Fp32, Ring};

    type F = Fp32<4294967197>;

    fn group() -> CommittedGroup<F> {
        let (profile, commitment) = group_parts(128);
        CommittedGroup::new(profile, commitment).expect("valid committed group")
    }

    /// One nv11 group whose A and B matrices share `ring_dimension`.
    fn group_parts(ring_dimension: usize) -> (GroupCommitPhaseParams, Commitment<F>) {
        let ring_dimension_u32 = u32::try_from(ring_dimension).unwrap();
        let positions = (1 << 11) / ring_dimension;
        let a_bound = *akita_params::sis::inner_coeff_linf_bounds(
            SisModulusProfileId::Q32Offset99,
            ring_dimension_u32,
        )
        .first()
        .expect("exact A bounds");
        let inner_commit_matrix = InnerCommitMatrixParams::try_new_with_min_rank(
            akita_params::SisTableKey {
                policy: akita_params::sis::DEFAULT_SIS_SECURITY_POLICY,
                table_digest: SisTableDigest::CURRENT,
                modulus_profile: SisModulusProfileId::Q32Offset99,
                role: SisMatrixRole::Inner,
                ring_dimension: ring_dimension_u32,
                coeff_linf_bound: a_bound,
            },
            positions,
        )
        .expect("audited A profile");
        let outer_width = inner_commit_matrix.output_rank();
        let outer_commit_matrix = OuterCommitMatrixParams::try_new_with_min_rank(
            akita_params::SisTableKey {
                policy: akita_params::sis::DEFAULT_SIS_SECURITY_POLICY,
                table_digest: SisTableDigest::CURRENT,
                modulus_profile: SisModulusProfileId::Q32Offset99,
                role: SisMatrixRole::Outer,
                ring_dimension: ring_dimension_u32,
                coeff_linf_bound: 3,
            },
            outer_width,
        )
        .expect("audited B profile");
        let profile = GroupCommitPhaseParams {
            version: GroupCommitPhaseParams::VERSION,
            group: PolynomialGroupLayout::new(11, 1),
            blocks: akita_params::BlockGeometry::new(positions, positions, 1),
            outer_slice_count: CommitmentSliceCount::ONE,
            inner: akita_params::RoleParams::new(
                akita_params::GadgetDigits::new(1, 1),
                inner_commit_matrix,
            ),
            outer: akita_params::RoleParams::new(
                akita_params::GadgetDigits::new(1, 1),
                outer_commit_matrix,
            ),
        };
        let source_coefficients = outer_commit_matrix.output_rank() * ring_dimension;
        let payload_coefficients = akita_params::CompressionChainPlan::for_complete_source(
            outer_commit_matrix.sis_modulus_profile(),
            source_coefficients,
        )
        .expect("compression plan")
        .terminal_coefficients();
        let coefficients = (0..payload_coefficients)
            .map(|index| F::from_u64(index as u64 * 7 + 1))
            .collect();
        (profile, Commitment::new(RingVec::from_coeffs(coefficients)))
    }

    fn canonical_bytes(group: &CommittedGroup<F>) -> Vec<u8> {
        let mut bytes = Vec::new();
        group
            .serialize_with_mode(&mut bytes, Compress::No)
            .expect("serialize committed group");
        bytes
    }

    fn decode(bytes: &[u8]) -> Result<CommittedGroup<F>, SerializationError> {
        CommittedGroup::<F>::deserialize_with_mode(bytes, Compress::No, Validate::Yes, &())
    }

    #[test]
    fn deserialized_and_constructed_digests_agree() {
        let group = group();
        let bytes = canonical_bytes(&group);
        assert_eq!(
            *group.digest(),
            CommitmentDigest::of_canonical_bytes(&bytes)
        );
        assert_eq!(decode(&bytes).unwrap().digest(), group.digest());
    }

    /// The decoder hashes the group it decoded, so a changed row gives the
    /// digest of the changed group, never the digest of the original.
    #[test]
    fn digest_is_recomputed_on_deserialization() {
        let group = group();
        let mut bytes = canonical_bytes(&group);
        let last = bytes.len() - 4;
        bytes[last..].copy_from_slice(&5u32.to_le_bytes());
        let decoded = decode(&bytes).unwrap();
        assert_ne!(decoded.digest(), group.digest());
        let mut coefficients = group.rows().coeffs().to_vec();
        *coefficients.last_mut().unwrap() = F::from_u64(5);
        let rebuilt = CommittedGroup::new(
            *group.profile(),
            Commitment::new(RingVec::from_coeffs(coefficients)),
        )
        .unwrap();
        assert_eq!(decoded.digest(), rebuilt.digest());
        assert_eq!(
            *decoded.digest(),
            CommitmentDigest::of_canonical_bytes(&bytes)
        );
    }

    /// Each commitment has one encoding, so it has one digest: the decoder
    /// rejects a noncanonical coefficient and a short row.
    #[test]
    fn noncanonical_commitment_bytes_are_rejected() {
        let group = group();
        let mut bytes = canonical_bytes(&group);
        let last = bytes.len() - 4;
        bytes[last..].copy_from_slice(&4_294_967_197u32.to_le_bytes());
        assert!(decode(&bytes).is_err());

        let mut short = canonical_bytes(&group);
        short.truncate(short.len() - 4);
        assert!(decode(&short).is_err());
    }

    #[test]
    fn commitment_digest_has_a_known_answer() {
        let rendered = format!("{:?}", group().digest().as_bytes());
        #[cfg(feature = "transcript-blake2b")]
        let expected = "[212, 16, 192, 226, 248, 59, 14, 111, 246, 78, 110, 93, 180, 110, 10, 206, 125, 199, 50, 78, 211, 56, 132, 24, 105, 108, 18, 219, 228, 100, 57, 17]";
        #[cfg(feature = "transcript-keccak")]
        let expected = "[114, 212, 182, 223, 91, 176, 13, 1, 90, 199, 228, 142, 136, 96, 40, 136, 165, 119, 91, 182, 89, 58, 10, 142, 79, 9, 118, 170, 200, 14, 39, 251]";
        assert_eq!(rendered, expected);
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
    fn committed_group_validation_rejects_an_outer_image_the_ladder_cannot_recompose() {
        // The q32 ladder starts at D=128, so a D=64 outer image cannot be compressed.
        let (profile, commitment) = group_parts(64);
        assert!(check_group(&profile, &commitment).is_err());
        assert!(CommittedGroup::new(profile, commitment).is_err());
    }

    #[test]
    fn committed_group_decoding_accepts_only_canonical_coefficient_bounds() {
        let group = group();
        let mut bytes = Vec::new();
        group
            .serialize_with_mode(&mut bytes, Compress::Yes)
            .expect("serialize committed group");
        let bounds = [
            group
                .profile
                .inner
                .matrix
                .coeff_linf_bound()
                .expect("L infinity test matrix"),
            group.profile.outer.matrix.coeff_linf_bound(),
        ];
        for bound in bounds {
            let encoded = bound.to_le_bytes();
            let offset = bytes
                .windows(encoded.len())
                .position(|window| window == encoded)
                .expect("encoded coefficient bound");
            for replacement in [bound - 1, bound / 2, 1, bound + 1] {
                if replacement == bound {
                    continue;
                }
                let mut candidate = bytes.clone();
                candidate[offset..offset + 16].copy_from_slice(&replacement.to_le_bytes());
                if let Ok(decoded) = CommittedGroup::<F>::deserialize_with_mode(
                    candidate.as_slice(),
                    Compress::Yes,
                    Validate::Yes,
                    &(),
                ) {
                    let mut reencoded = Vec::new();
                    decoded
                        .serialize_with_mode(&mut reencoded, Compress::Yes)
                        .expect("re-encode");
                    assert_eq!(
                        reencoded, candidate,
                        "bound {replacement} (table bound {bound}) decoded to a different encoding"
                    );
                }
            }
        }
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
