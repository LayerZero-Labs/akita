//! Explicit construction of the binary group kind in the canonical commitment.

use super::{Commitment, CommittedGroup, CommittedGroupKind};
use crate::binary_source::BinaryCommittedGroupManifest;
use akita_error::AkitaError;
use akita_params::GroupCommitPhaseParams;
use akita_serialization::{AkitaSerialize, Valid};
use jolt_field::{CanonicalEncoding, Field};

impl<F: Field + CanonicalEncoding + Valid + AkitaSerialize> CommittedGroup<F> {
    /// Select the experimental binary root while retaining an ordinary suffix carrier.
    ///
    /// The carrier profile never describes the binary A/B commitment. Their real
    /// contracts are frozen in the typed manifest. This admission has no security claim.
    pub fn new_binary(
        profile: GroupCommitPhaseParams,
        commitment: Commitment<F>,
        manifest: BinaryCommittedGroupManifest,
    ) -> Result<Self, AkitaError> {
        profile.validate_frozen_precommit(F::MODULUS_BITS)?;
        manifest.validate_carrier(&profile)?;
        if commitment.rows().coeff_len() != manifest.payload_coefficients()? {
            return Err(AkitaError::InvalidInput(
                "binary commitment payload shape mismatch".into(),
            ));
        }
        Self::with_kind(profile, commitment, CommittedGroupKind::Binary(manifest))
            .map_err(|error| AkitaError::InvalidInput(error.to_string()))
    }
}

#[cfg(test)]
mod binary_source_tests {
    use super::*;
    use crate::binary_source::{
        BinaryIndexMap, BinarySourceDescriptor, BINARY_COMMITTED_PAYLOAD_COEFFICIENTS,
    };
    use crate::{Commitment, CommittedGroup, RingVec};
    use akita_params::binary_setup::{
        BinaryFieldProfile, SetupStreamModulus, SetupStreamRole, BINARY_SETUP_P64,
    };
    use akita_params::sis::{
        sis_table_key_for_linf_bound, InnerCommitMatrixParams, OuterCommitMatrixParams,
        SisMatrixRole, SisModulusProfileId, SisTableDigest, DEFAULT_SIS_SECURITY_POLICY,
    };
    use akita_params::{
        BlockGeometry, CommitmentSliceCount, GadgetDigits, PolynomialGroupLayout, RoleParams,
    };
    use akita_serialization::{AkitaDeserialize, AkitaSerialize, Compress, Valid, Validate};
    use jolt_field::{Prime128Offset275, Prime64Offset59, Ring, Zero};

    fn carrier(field: BinaryFieldProfile) -> GroupCommitPhaseParams {
        let (modulus, bits) = match field {
            BinaryFieldProfile::P128 => (SisModulusProfileId::Q128Offset275, 128),
            BinaryFieldProfile::P64 => (SisModulusProfileId::Q64Offset59, 64),
        };
        let key = |role| {
            sis_table_key_for_linf_bound(
                DEFAULT_SIS_SECURITY_POLICY,
                SisTableDigest::CURRENT,
                modulus,
                role,
                64,
                3,
            )
            .unwrap()
        };
        let inner =
            InnerCommitMatrixParams::try_new_with_min_rank(key(SisMatrixRole::Inner), 1).unwrap();
        let outer = OuterCommitMatrixParams::try_new_with_min_rank(
            key(SisMatrixRole::Outer),
            inner.output_rank(),
        )
        .unwrap();
        let profile = GroupCommitPhaseParams {
            version: GroupCommitPhaseParams::VERSION,
            group: PolynomialGroupLayout::singleton(6),
            blocks: BlockGeometry::new(1, 1, 1),
            outer_slice_count: CommitmentSliceCount::ONE,
            inner: RoleParams::new(GadgetDigits::new(1, 1), inner),
            outer: RoleParams::new(GadgetDigits::new(1, 1), outer),
        };
        profile.validate_frozen_precommit(bits).unwrap();
        profile
    }

    fn manifest(
        field: BinaryFieldProfile,
        columns: usize,
        owners: usize,
    ) -> BinaryCommittedGroupManifest {
        let index = BinaryIndexMap::new(128, 4 * columns, 4 * columns, columns).unwrap();
        let profiles =
            BinaryCommittedGroupManifest::profiles_for_field(&index, owners, 7, field).unwrap();
        let ids = (0..owners)
            .map(|owner| {
                let mut id = [0; 32];
                id[..8].copy_from_slice(&(owner as u64).to_le_bytes());
                id
            })
            .collect();
        BinaryCommittedGroupManifest::with_field_profile(
            BinarySourceDescriptor::new(index, ids, profiles).unwrap(),
            7,
            &carrier(field),
            field,
        )
        .unwrap()
    }

    #[test]
    fn paired_manifests_separate_b_and_preserve_a_and_legacy_sizes() {
        let legacy = manifest(BinaryFieldProfile::P128, 2, 1);
        let p64 = manifest(BinaryFieldProfile::P64, 2, 1);
        assert_eq!(p64, manifest(BinaryFieldProfile::P64, 2, 1));
        assert_eq!(legacy.a_label(), p64.a_label());
        assert_eq!(
            legacy.descriptor().profiles().a,
            p64.descriptor().profiles().a
        );
        assert_ne!(legacy.b_label(), p64.b_label());
        assert_ne!(
            legacy.descriptor().profiles().b,
            p64.descriptor().profiles().b
        );
        assert_eq!(
            legacy.payload_coefficients().unwrap(),
            BINARY_COMMITTED_PAYLOAD_COEFFICIENTS
        );
        for manifest in [legacy, p64] {
            let bytes = manifest.canonical_bytes();
            assert_eq!(
                BinaryCommittedGroupManifest::from_bytes(&bytes).unwrap(),
                manifest
            );
            let mut swapped = bytes.clone();
            swapped[b"AkitaBinaryCommitment".len()] = match manifest.field_profile() {
                BinaryFieldProfile::P128 => 2,
                BinaryFieldProfile::P64 => 1,
            };
            assert!(BinaryCommittedGroupManifest::from_bytes(&swapped).is_err());
            for len in 0..bytes.len() {
                assert!(BinaryCommittedGroupManifest::from_bytes(&bytes[..len]).is_err());
            }
            let mut trailing = bytes;
            trailing.push(0);
            assert!(BinaryCommittedGroupManifest::from_bytes(&trailing).is_err());
        }
    }

    #[test]
    fn p64_manifest_admits_b_and_d_independently_for_every_spec_geometry() {
        for (columns, owners, b, d, width) in [
            (2, 1, 2, 1, 228),
            (2, 4096, 3, 2, 21_850),
            (256, 1, 2, 2, 29_160),
            (256, 10, 3, 3, 149_040),
            (256, 4096, 5, 3, 2_796_768),
        ] {
            let manifest = manifest(BinaryFieldProfile::P64, columns, owners);
            assert_eq!(manifest.b_rows(), b);
            // Up to five E64 claims, each with two coordinate planes, plus U.
            let planes = 2 * owners.min(5);
            let d_label = manifest.d_label(true, planes).unwrap();
            assert_eq!(((planes * 22 + owners) * 162 * columns).div_ceil(64), width);
            assert_eq!(d_label.rank, d);
            assert_eq!(d_label.role, SetupStreamRole::WitnessCommitment);
            assert_eq!(d_label.modulus, SetupStreamModulus::Prime64Offset59);
            assert!(manifest.d_label(false, 0).is_err());
            assert!(manifest.d_label(true, 2 * owners + 2).is_err());
            assert!(manifest.d_label(true, 2 * owners - 1).is_err());
        }
    }

    #[test]
    fn p64_binary_payloads_round_trip_all_five_ranks_and_reject_malformed_lengths() {
        type F = Prime64Offset59;
        let profile = carrier(BinaryFieldProfile::P64);
        for (columns, owners, rank) in [
            (1, 1, 1),
            (2, 1, 2),
            (256, 2, 3),
            (256, 128, 4),
            (256, 4096, 5),
        ] {
            let manifest = manifest(BinaryFieldProfile::P64, columns, owners);
            assert_eq!(manifest.b_rows(), rank);
            let len = manifest.payload_coefficients().unwrap();
            assert_eq!(len, 64 * rank);
            let coefficients = (0..len).map(|i| F::from_u64(i as u64)).collect::<Vec<_>>();
            let payload = Commitment::new(RingVec::from_coeffs(coefficients));
            let group = CommittedGroup::new_binary(profile, payload, manifest.clone()).unwrap();
            group.check().unwrap();
            for compress in [Compress::No, Compress::Yes] {
                let mut bytes = Vec::new();
                group.serialize_with_mode(&mut bytes, compress).unwrap();
                assert_eq!(bytes.len(), group.serialized_size(compress));
                // Base payloads are exactly 8 bytes each, without an E64 zero plane.
                let mut profile_bytes = Vec::new();
                crate::proof::commitment::wire::serialize_profile(&profile, &mut profile_bytes)
                    .unwrap();
                assert_eq!(
                    bytes.len() - len * 8,
                    5 + manifest.canonical_bytes().len() + profile_bytes.len()
                );
                {
                    let validate = Validate::Yes;
                    let mut overlong = bytes.clone();
                    overlong.extend_from_slice(&0u64.to_le_bytes());
                    assert!(CommittedGroup::<F>::deserialize_exact_with_mode(
                        &overlong,
                        compress,
                        validate,
                        &(),
                    )
                    .is_err());
                    let decoded = CommittedGroup::<F>::deserialize_with_mode(
                        bytes.as_slice(),
                        compress,
                        validate,
                        &(),
                    )
                    .unwrap();
                    assert_eq!(decoded, group);
                    assert_eq!(decoded.digest(), group.digest());
                    assert!(CommittedGroup::<F>::deserialize_with_mode(
                        &bytes[..bytes.len() - 1],
                        compress,
                        validate,
                        &()
                    )
                    .is_err());
                    let mut noncanonical = bytes.clone();
                    noncanonical[bytes.len() - 8..]
                        .copy_from_slice(&BINARY_SETUP_P64.to_le_bytes());
                    assert!(CommittedGroup::<F>::deserialize_with_mode(
                        noncanonical.as_slice(),
                        compress,
                        validate,
                        &()
                    )
                    .is_err());
                }
            }
            for bad_len in [0, len - 1, len + 1, 64 * (rank + 1)] {
                let payload = Commitment::new(RingVec::from_coeffs(vec![F::zero(); bad_len]));
                assert!(
                    CommittedGroup::new_binary(profile, payload.clone(), manifest.clone()).is_err()
                );
                assert!(crate::proof::commitment::wire::check_group(
                    &profile,
                    &payload,
                    group.kind()
                )
                .is_err());
            }
            let wrong_field =
                Commitment::new(RingVec::from_coeffs(vec![Prime128Offset275::zero(); len]));
            assert!(CommittedGroup::new_binary(
                carrier(BinaryFieldProfile::P128),
                wrong_field,
                manifest
            )
            .is_err());
        }
    }

    #[test]
    fn p64_binary_root_statement_rejects_on_both_channels() {
        use akita_transcript::{new_prover_channel, new_verifier_channel};
        type F = Prime64Offset59;
        type E = jolt_field::Ext2<F>;
        let manifest = manifest(BinaryFieldProfile::P64, 2, 1);
        let prime =
            crate::binary_source::PrimePlaneManifest::new(manifest.descriptor(), None).unwrap();
        let payload = vec![F::zero(); manifest.payload_coefficients().unwrap()];
        let mut prover = new_prover_channel(b"p64-admission");
        let mut verifier = new_verifier_channel(b"p64-admission", &[]);
        let prove = crate::binary_source::binary_root_statement::<F, E, _>(
            &mut prover,
            &manifest,
            None,
            &prime,
            &payload,
            None,
        );
        let verify = crate::binary_source::binary_root_statement::<F, E, _>(
            &mut verifier,
            &manifest,
            None,
            &prime,
            &payload,
            None,
        );
        for result in [prove, verify] {
            assert!(
                matches!(result, Err(AkitaError::UnsupportedSchedule(message))
                if message.contains("not yet supported"))
            );
        }
    }

    #[test]
    fn e64_uses_nonresidue_two_and_two_canonical_base_coordinates() {
        type F = Prime64Offset59;
        type E = jolt_field::Ext2<F>;
        let u = E::new(F::zero(), F::from_u64(1));
        assert_eq!(u * u, E::new(F::from_u64(2), F::zero()));
        let value = E::new(F::from_u64(17), F::from_u64(19));
        let mut bytes = Vec::new();
        value.serialize_with_mode(&mut bytes, Compress::No).unwrap();
        let expected = [17u64.to_le_bytes(), 19u64.to_le_bytes()].concat();
        assert_eq!(bytes, expected);
        assert_eq!(
            E::deserialize_uncompressed(bytes.as_slice(), &()).unwrap(),
            value
        );
    }
}
