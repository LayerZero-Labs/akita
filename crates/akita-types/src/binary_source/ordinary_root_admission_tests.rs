use super::*;
use crate::binary_source::binary_ordinary_profile_fixture;
use akita_params::sis::{
    sis_table_key_for_linf_bound, OuterCommitMatrixParams, SisMatrixRole, SisTableDigest,
    DEFAULT_SIS_SECURITY_POLICY,
};
use akita_params::{CommitmentSliceGeometry, PolynomialGroupLayout};

#[test]
fn ordinary_root_accepts_outer_64_and_rejects_valid_outer_128() {
    for field in [BinaryFieldProfile::P64, BinaryFieldProfile::P128] {
        let mut profile = binary_ordinary_profile_fixture(field);
        assert_eq!(profile.outer.matrix.ring_dimension(), 64);
        OrdinaryNativeRootPlan::new(profile, 0).unwrap();

        // Rebuild the entire outer matrix from its own audited table cell and
        // slice geometry; changing only the degree would make an invalid profile.
        let geometry = CommitmentSliceGeometry::try_new(
            profile.outer_slice_count,
            profile.blocks.live_blocks,
            profile.group.num_polynomials(),
            profile.inner.matrix.output_rank(),
            profile.outer.digits.num_digits,
            profile.inner.matrix.ring_dimension(),
            128,
        )
        .unwrap();
        let key = sis_table_key_for_linf_bound(
            DEFAULT_SIS_SECURITY_POLICY,
            SisTableDigest::CURRENT,
            profile.outer.matrix.sis_modulus_profile(),
            SisMatrixRole::Outer,
            128,
            8,
        )
        .unwrap();
        profile.outer.matrix =
            OuterCommitMatrixParams::try_new_with_min_rank(key, geometry.physical_input_width())
                .unwrap();
        profile
            .validate_frozen_precommit(profile.inner.matrix.sis_modulus_profile().field_bits())
            .unwrap();
        // The arithmetic and compression remain valid: only driver admission
        // must refuse this profile before setup, commitment or proof execution.
        OrdinaryNativeRootPlan::compile(profile, 0, None).unwrap();
        assert_eq!(
            OrdinaryNativeRootPlan::new(profile, 0).unwrap_err(),
            AkitaError::InvalidSetup(
                "mixed root ordinary profile requires outer ring dimension 64; got 128".into()
            )
        );
    }
}

#[test]
fn ordinary_root_rejects_valid_two_slot_group() {
    for field in [BinaryFieldProfile::P64, BinaryFieldProfile::P128] {
        let mut profile = binary_ordinary_profile_fixture(field);
        profile.group = PolynomialGroupLayout::new(profile.group.num_vars(), 2);
        let geometry = profile.derive_slice_geometry().unwrap();
        profile.outer.matrix = OuterCommitMatrixParams::try_new_with_min_rank(
            profile.outer.matrix.sis_table_key(),
            geometry.physical_input_width(),
        )
        .unwrap();
        profile
            .validate_frozen_precommit(profile.inner.matrix.sis_modulus_profile().field_bits())
            .unwrap();
        // A valid profile is insufficient for native-root admission. The
        // one-point and two-point drivers both reach this constructor.
        assert_eq!(
            OrdinaryNativeRootPlan::new(profile, 0).unwrap_err(),
            AkitaError::InvalidSetup(
                "mixed root ordinary group requires exactly one polynomial".into()
            )
        );
    }
}
