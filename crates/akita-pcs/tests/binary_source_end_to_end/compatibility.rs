//! Portable encoding of the typed Stage 3 prefix identity.

use super::*;

/// Stage 3 absorbs the typed-prefix identity. Every word in it is a
/// little-endian `u64`, so the bytes do not depend on the pointer width; the
/// tail region encodes `top = 64`, no fixed bit and the free-bit marker
/// `u64::MAX`.
#[test]
fn typed_prefix_identity_encodes_tail_as_little_endian_u64s() {
    let scheme = AkitaCommitmentScheme::<BinaryTestSuffix>::new(test_catalog());
    let row = scheme
        .schedules()
        .resolve_key(&ScheduleLookupKey::single(
            PolynomialGroupLayout::singleton(18),
        ))
        .unwrap();
    let next = &row.schedule().recursive_folds[0].params;
    let slot = next.setup_prefix().unwrap().slot_id().unwrap();
    let index = BinaryIndexMap::new(128, 8, 8, 2).unwrap();
    let profiles =
        BinaryCommittedGroupManifest::profiles_for_field(&index, 1, 0, BinaryFieldProfile::P128)
            .unwrap();
    let descriptor = BinarySourceDescriptor::new(index, vec![[29; 32]], profiles).unwrap();
    let manifest = BinaryCommittedGroupManifest::with_field_profile(
        descriptor,
        0,
        &row.profiles().final_group,
        BinaryFieldProfile::P128,
    )
    .unwrap();
    let typed = BinaryTypedPrefix::new(&manifest, true, 0, next.d_a()).unwrap();
    let identity = typed.identity(&manifest, &slot).unwrap();
    let tail = [64u64, 0, u64::MAX].map(u64::to_le_bytes).concat();
    assert!(identity.windows(tail.len()).any(|word| word == tail));
}

#[test]
fn p64_cannot_use_an_ordinary_catalog_row() {
    use akita_params::binary_setup::BinaryFieldProfile;
    use akita_types::binary_source::{
        binary_ordinary_profile_fixture, binary_word_manifest_fixture,
    };
    use jolt_field::{Ext2, Prime64Offset59 as Base};
    let profile = BinaryFieldProfile::P64;
    let manifest = binary_word_manifest_fixture(profile, 1);
    let claim = BinaryWordClaim::new(
        manifest.descriptor(),
        manifest.descriptor().polynomials()[0],
        vec![1u128.to_le_bytes(); 3],
        0u128.to_le_bytes(),
    )
    .unwrap();
    let commitment = CommittedGroup::new_binary(
        binary_ordinary_profile_fixture(profile, None),
        akita_types::Commitment::new(akita_types::RingVec::from_coeffs(vec![
            Base::from_u64(0);
            manifest
                .payload_coefficients()
                .unwrap()
        ])),
        manifest,
    )
    .unwrap();
    let claims =
        OpeningClaims::from_groups(vec![PolynomialGroupClaims::<Ext2<Base>, _>::new_binary(
            commitment, claim,
        )
        .unwrap()])
        .unwrap();
    let catalog = test_catalog();
    let row = catalog
        .resolve_key(&ScheduleLookupKey::single(
            PolynomialGroupLayout::singleton(18),
        ))
        .unwrap();
    assert!(!row.is_native_root());
    let ordinary =
        akita_params::GrindingPlan::new(vec![], profile.challenge_order().unwrap()).unwrap();
    assert_eq!(
        claims.root_grinding_plan(ordinary, row.schedule(), row.is_native_root()),
        Err(AkitaError::UnsupportedSchedule(
            "p64 root requires an explicitly selected native catalog row".into()
        ))
    );
}
