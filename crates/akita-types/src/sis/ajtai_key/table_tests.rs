use super::*;
use crate::sis::inner_coeff_linf_bounds;

fn key(
    table_digest: SisTableDigest,
    modulus_profile: SisModulusProfileId,
    role: SisMatrixRole,
    ring_dimension: u32,
    coeff_linf_bound: u128,
) -> SisTableKey {
    SisTableKey {
        policy: DEFAULT_SIS_SECURITY_POLICY,
        table_digest,
        modulus_profile,
        role,
        ring_dimension,
        coeff_linf_bound,
    }
}

#[test]
fn unsupported_shape_rejects_exact_linf_bound() {
    assert_eq!(
        ceil_supported_linf_bound(
            DEFAULT_SIS_SECURITY_POLICY,
            SisTableDigest::CURRENT,
            SisModulusProfileId::Q32Offset99,
            SisMatrixRole::Inner,
            31,
            1_428,
        ),
        None
    );
}

#[test]
fn fixed_matrix_capacity_inverts_the_checked_sis_table() {
    let key = SisTableKey {
        policy: DEFAULT_SIS_SECURITY_POLICY,
        table_digest: SisTableDigest::CURRENT,
        modulus_profile: SisModulusProfileId::Q128OffsetA7F7,
        role: SisMatrixRole::Inner,
        ring_dimension: 64,
        coeff_linf_bound: 6_684_468,
    };
    let matrix = InnerCommitMatrixParams::try_new_with_min_rank(key, 64).expect("audited matrix");
    let capacity = matrix
        .max_secure_collision_linf()
        .expect("fixed matrix capacity");
    assert!(capacity >= key.coeff_linf_bound);
    for larger in inner_coeff_linf_bounds(key.modulus_profile, key.ring_dimension)
        .into_iter()
        .filter(|&bound| bound > capacity)
    {
        let larger_key = SisTableKey {
            coeff_linf_bound: larger,
            ..key
        };
        assert!(min_secure_rank(larger_key, matrix.input_width() as u64)
            .is_none_or(|rank| rank > matrix.output_rank()));
    }
}

#[test]
fn inner_linf_key_rounds_up_to_the_next_audited_target() {
    let linf = 130_023_300u128;
    let key = sis_table_key_for_linf_bound(
        DEFAULT_SIS_SECURITY_POLICY,
        SisTableDigest::CURRENT,
        SisModulusProfileId::Q32Offset99,
        SisMatrixRole::Inner,
        128,
        linf,
    )
    .expect("exact q32 D128 target");
    assert_eq!(key.coeff_linf_bound, linf);
    assert_eq!(key.policy, DEFAULT_SIS_SECURITY_POLICY);
    let rounded = sis_table_key_for_linf_bound(
        DEFAULT_SIS_SECURITY_POLICY,
        SisTableDigest::CURRENT,
        SisModulusProfileId::Q32Offset99,
        SisMatrixRole::Inner,
        128,
        linf - 1,
    )
    .expect("intermediate A target rounds to the next audited cell");
    assert_eq!(rounded.coeff_linf_bound, linf);

    let largest = inner_coeff_linf_bounds(SisModulusProfileId::Q32Offset99, 128)
        .into_iter()
        .max()
        .expect("q32 D128 A coverage");
    assert_eq!(
        sis_table_key_for_linf_bound(
            DEFAULT_SIS_SECURITY_POLICY,
            SisTableDigest::CURRENT,
            SisModulusProfileId::Q32Offset99,
            SisMatrixRole::Inner,
            128,
            largest + 1,
        ),
        None,
    );
}

#[test]
fn exact_inner_reach_is_profile_and_dimension_specific() {
    for dimension in [64, 128, 256, 512, 1024, 2048] {
        assert!(sis_role_cell(
            SisMatrixRole::Inner,
            SisModulusProfileId::Q32Offset99,
            dimension,
            1_821_066_133_292,
        )
        .is_none());
    }
    assert!(sis_role_cell(
        SisMatrixRole::Inner,
        SisModulusProfileId::Q64Offset59,
        64,
        1_821_066_133_292,
    )
    .is_some());
    assert!(sis_role_cell(
        SisMatrixRole::Inner,
        SisModulusProfileId::Q64Offset59,
        128,
        1_821_066_133_292,
    )
    .is_none());
    assert!(sis_role_cell(
        SisMatrixRole::Inner,
        SisModulusProfileId::Q128OffsetA7F7,
        64,
        1_821_066_133_292,
    )
    .is_some());
}

#[test]
fn tier_max_dimension_coverage_is_inner_only() {
    for (profile, dimension, bound) in [
        (SisModulusProfileId::Q32Offset99, 2048, 392),
        (SisModulusProfileId::Q64Offset59, 2048, 392),
        (SisModulusProfileId::Q128OffsetA7F7, 1024, 448),
    ] {
        assert!(sis_role_cell(SisMatrixRole::Inner, profile, dimension, bound).is_some());
        for role in [SisMatrixRole::Outer, SisMatrixRole::Open] {
            assert!(sis_role_cell(role, profile, dimension, 3).is_none());
        }
    }
}

#[test]
fn d512_uses_current_digest_and_rejects_unknown_digest() {
    let current = key(
        SisTableDigest::CURRENT,
        SisModulusProfileId::Q128OffsetA7F7,
        SisMatrixRole::Inner,
        512,
        532,
    );
    assert!(min_secure_rank(current, 1).is_some());
    let unknown = key(
        SisTableDigest([0xFF; 32]),
        SisModulusProfileId::Q128OffsetA7F7,
        SisMatrixRole::Inner,
        64,
        1_428,
    );
    assert_eq!(min_secure_rank(unknown, 1), None);
}

#[test]
fn min_secure_rank_matches_linear_first_match_for_every_generated_row() {
    let profiles = [
        SisModulusProfileId::Q32Offset99,
        SisModulusProfileId::Q64Offset59,
        SisModulusProfileId::Q128OffsetA7F7,
    ];
    let roles = [
        SisMatrixRole::Outer,
        SisMatrixRole::Inner,
        SisMatrixRole::Open,
    ];
    for profile in profiles {
        for role in roles {
            for cell in crate::sis::sis_role_cells()
                .into_iter()
                .filter(|cell| cell.modulus_profile == profile && cell.role == role)
            {
                let dimension = cell.ring_dimension;
                let bound = cell.coeff_linf_bound;
                let widths = sis_max_widths(
                    DEFAULT_SIS_SECURITY_POLICY,
                    SisTableDigest::CURRENT,
                    profile,
                    dimension,
                    bound,
                )
                .expect("reachable role cell has a generated SIS row");
                let widths = &widths[..usize::try_from(cell.max_module_rank)
                    .expect("module rank fits usize")
                    .min(widths.len())];
                for width in widths
                    .iter()
                    .copied()
                    .flat_map(|width| [width, width.saturating_add(1)])
                {
                    let expected = widths
                        .iter()
                        .position(|&max_width| width <= max_width)
                        .map(|index| index + 1);
                    assert_eq!(
                        min_secure_rank(
                            key(
                                SisTableDigest::CURRENT,
                                profile,
                                role,
                                dimension,
                                bound,
                            ),
                            width,
                        ),
                        expected,
                        "profile={profile:?}, role={role:?}, D={dimension}, bound={bound}, width={width}",
                    );
                }
            }
        }
    }
}
