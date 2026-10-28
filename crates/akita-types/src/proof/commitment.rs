//! Protocol commitment/opening wrapper types.

#[cfg(feature = "labinius-trinomial")]
mod binary;
mod kind;
/// Extension seam: validate and encode each commitment kind at one wire boundary.
/// Tracking issue: <https://github.com/LayerZero-Labs/akita/issues/285>.
pub(crate) mod wire;
pub use kind::CommittedGroupKind;

use crate::proof::RingVec;
#[cfg(test)]
use akita_params::sis::{
    InnerCommitMatrixParams, OuterCommitMatrixParams, SisMatrixRole, SisModulusProfileId,
    SisTableDigest,
};
use akita_params::GroupCommitPhaseParams;
#[cfg(test)]
use akita_params::{CommitmentSliceCount, PolynomialGroupLayout};

use akita_serialization::{
    AkitaDeserialize, AkitaSerialize, Compress, SerializationError, Valid, Validate,
};
use jolt_field::Field;
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
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommittedGroup<F: Field> {
    /// Exact public algebraic profile and commitment geometry.
    pub profile: GroupCommitPhaseParams,
    /// Terminal compressed `p_F` payload.
    pub commitment: Commitment<F>,
    kind: CommittedGroupKind,
}

impl<F: Field> CommittedGroup<F> {
    /// Build a self-describing group commitment.
    pub fn new(profile: GroupCommitPhaseParams, commitment: Commitment<F>) -> Self {
        Self {
            profile,
            commitment,
            kind: CommittedGroupKind::Ordinary,
        }
    }

    /// Extension seam: identify the source relation bound by this commitment.
    ///
    /// Tracking issue: <https://github.com/LayerZero-Labs/akita/issues/285>.
    pub fn kind(&self) -> &CommittedGroupKind {
        &self.kind
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
        group_with_ring_dimension(128)
    }

    /// One nv11 group whose A and B matrices share `ring_dimension`.
    fn group_with_ring_dimension(ring_dimension: usize) -> CommittedGroup<F> {
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
    fn committed_group_validation_rejects_an_outer_image_the_ladder_cannot_recompose() {
        // The q32 ladder starts at D=128, so a D=64 outer image cannot be compressed.
        let group = group_with_ring_dimension(64);
        assert!(group.check().is_err());
        assert!(group
            .serialize_with_mode(&mut Vec::new(), Compress::Yes)
            .is_err());
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
