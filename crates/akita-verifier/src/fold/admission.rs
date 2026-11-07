use akita_error::AkitaError;
use akita_types::opening_claims::GroupClaimKind;
use akita_types::proof::commitment::CommittedGroupKind;
use akita_types::CommittedGroup;
use jolt_algebra::Field;

/// Extension seam: validate the payload according to its committed group kind.
/// Tracking issue: <https://github.com/LayerZero-Labs/akita/issues/285>.
pub(super) fn validate_group_payload<F: Field>(
    committed: &CommittedGroup<F>,
    kind: &GroupClaimKind,
    field_bits: u32,
) -> Result<(), AkitaError> {
    let descriptor = committed.profile();
    descriptor.validate_frozen_precommit(field_bits)?;
    match (committed.kind(), kind) {
        (CommittedGroupKind::Ordinary, GroupClaimKind::Ordinary) => {}
        #[cfg(feature = "labinius-trinomial")]
        (
            CommittedGroupKind::Binary(manifest),
            GroupClaimKind::Binary {
                manifest: claim_manifest,
                claim,
                prime,
            },
        ) => {
            if manifest != claim_manifest.as_ref() {
                return Err(AkitaError::InvalidInput(
                    "binary claim manifest differs from its commitment".into(),
                ));
            }
            manifest.validate_carrier(descriptor)?;
            if let Some(claim) = claim {
                claim.validate(manifest.descriptor())?;
            }
            prime.validate(manifest.descriptor())?;
            if committed.rows().coeff_len() != manifest.payload_coefficients()? {
                return Err(AkitaError::InvalidInput(
                    "binary commitment length does not match its manifest".into(),
                ));
            }
            return Ok(());
        }
        // Dependency features can expose kinds whose verifier handler is disabled.
        #[allow(unreachable_patterns)]
        _ => {
            return Err(AkitaError::InvalidInput(
                "opening claim kind differs from its commitment".into(),
            ));
        }
    }
    let plan = descriptor.outer_compression_plan()?;
    if committed.commitment().rows().coeff_len() != plan.terminal_coefficients() {
        return Err(AkitaError::InvalidInput(
            "commitment length does not match its compression chain".to_string(),
        ));
    }
    Ok(())
}

#[cfg(all(test, feature = "labinius-trinomial"))]
mod tests {
    use super::*;
    use akita_params::binary_setup::BinaryFieldProfile;
    use akita_params::sis::{
        sis_table_key_for_linf_bound, InnerCommitMatrixParams, OuterCommitMatrixParams,
        SisMatrixRole, SisModulusProfileId, SisTableDigest, DEFAULT_SIS_SECURITY_POLICY,
    };
    use akita_params::{
        BlockGeometry, CommitmentSliceCount, GadgetDigits, GroupCommitPhaseParams,
        PolynomialGroupLayout, RoleParams,
    };
    use akita_types::binary_source::{
        BinaryCommittedGroupManifest, BinaryIndexMap, BinarySourceDescriptor, BinaryWordClaim,
        PrimePlaneManifest,
    };
    use akita_types::{Commitment, RingVec};
    use jolt_algebra::{Prime64Offset59 as F, Zero};

    #[test]
    fn binary_source_payload_admission_uses_each_manifest_rank() {
        let key = |role| {
            sis_table_key_for_linf_bound(
                DEFAULT_SIS_SECURITY_POLICY,
                SisTableDigest::CURRENT,
                SisModulusProfileId::Q64Offset59,
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
        let mut previous_kind = None;
        for (columns, owners, rank) in [
            (1, 1, 1),
            (2, 1, 2),
            (256, 2, 3),
            (256, 128, 4),
            (256, 4096, 5),
        ] {
            let index = BinaryIndexMap::new(128, 4 * columns, 4 * columns, columns).unwrap();
            let profiles = BinaryCommittedGroupManifest::profiles_for_field(
                &index,
                owners,
                7,
                BinaryFieldProfile::P64,
            )
            .unwrap();
            let ids = (0..owners)
                .map(|owner| {
                    let mut id = [0; 32];
                    id[..8].copy_from_slice(&(owner as u64).to_le_bytes());
                    id
                })
                .collect();
            let descriptor = BinarySourceDescriptor::new(index, ids, profiles).unwrap();
            let split = descriptor.index_map().claim_variable_split();
            let claim = BinaryWordClaim::new(
                &descriptor,
                [0; 32],
                vec![[0; 16]; split.intra_carrier + split.carrier_address],
                [0; 16],
            )
            .unwrap();
            let prime = PrimePlaneManifest::new(&descriptor, None).unwrap();
            let manifest = BinaryCommittedGroupManifest::with_field_profile(
                descriptor,
                7,
                &profile,
                BinaryFieldProfile::P64,
            )
            .unwrap();
            let payload = Commitment::new(RingVec::from_coeffs(vec![F::zero(); 64 * rank]));
            let committed = CommittedGroup::new_binary(profile, payload, manifest.clone()).unwrap();
            let kind = GroupClaimKind::Binary {
                manifest: Box::new(manifest),
                claim: Some(claim),
                prime,
            };
            validate_group_payload(&committed, &kind, 64).unwrap();
            if let Some(previous) = &previous_kind {
                assert!(validate_group_payload(&committed, previous, 64).is_err());
            }
            assert!(validate_group_payload(&committed, &GroupClaimKind::Ordinary, 64).is_err());
            assert!(validate_group_payload(&committed, &kind, 128).is_err());
            previous_kind = Some(kind);
        }
    }
}
