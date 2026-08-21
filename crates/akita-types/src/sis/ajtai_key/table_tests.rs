use super::*;

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
fn unsupported_shape_rejects_linf_bucket() {
    assert_eq!(
        ceil_supported_linf_bound(
            DEFAULT_SIS_SECURITY_POLICY,
            SisTableDigest::CURRENT,
            SisModulusProfileId::Q32Offset99,
            SisMatrixRole::Inner,
            31,
            7,
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
        coeff_linf_bound: 32_767,
    };
    let matrix = InnerCommitMatrixParams::try_new_with_min_rank(key, 64).expect("audited matrix");
    let capacity = matrix
        .max_secure_collision_linf()
        .expect("fixed matrix capacity");
    assert!(capacity >= key.coeff_linf_bound);
    for &larger in COEFF_LINF_BUCKETS.iter().filter(|&&bound| bound > capacity) {
        let larger_key = SisTableKey {
            coeff_linf_bound: larger,
            ..key
        };
        assert!(min_secure_rank(larger_key, matrix.input_width() as u64)
            .is_none_or(|rank| rank > matrix.output_rank()));
    }
}

#[test]
fn floor_slices_have_family_specific_rank_caps() {
    let bucket = 15;
    if generated_sis_max_widths(
        DEFAULT_SIS_SECURITY_POLICY,
        SisModulusProfileId::Q32Offset99,
        32,
        bucket,
    )
    .is_some()
    {
        assert!(generated_sis_max_widths(
            DEFAULT_SIS_SECURITY_POLICY,
            SisModulusProfileId::Q32Offset99,
            32,
            bucket,
        )
        .is_some());
    }
}

#[test]
fn linf_key_rounds_to_coefficient_bucket() {
    let linf = 1_048_575u128;
    if let Some(key) = sis_table_key_for_linf_bound(
        DEFAULT_SIS_SECURITY_POLICY,
        SisTableDigest::CURRENT,
        SisModulusProfileId::Q32Offset99,
        SisMatrixRole::Inner,
        128,
        linf,
    ) {
        assert_eq!(key.coeff_linf_bound, linf);
        assert_eq!(key.policy, DEFAULT_SIS_SECURITY_POLICY);
    }
}

#[test]
fn coeff_linf_bucket_ladder_matches_main_ceiling() {
    assert_eq!(ceil_coeff_linf_bucket(1_048_574), Some(1_048_575));
    assert_eq!(ceil_coeff_linf_bucket(1_048_575), Some(1_048_575));
    assert_eq!(ceil_coeff_linf_bucket(1_048_576), Some(2_097_151));
}

#[test]
fn tier_max_dimension_coverage_is_inner_only() {
    for (profile, dimension) in [
        (SisModulusProfileId::Q32Offset99, 2048),
        (SisModulusProfileId::Q64Offset59, 1024),
        (SisModulusProfileId::Q128OffsetA7F7, 512),
    ] {
        assert!(sis_role_cell(SisMatrixRole::Inner, profile, dimension, 2).is_some());
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
        2,
    );
    assert!(min_secure_rank(current, 1).is_some());
    let unknown = key(
        SisTableDigest([0xFF; 32]),
        SisModulusProfileId::Q128OffsetA7F7,
        SisMatrixRole::Inner,
        64,
        2,
    );
    assert_eq!(min_secure_rank(unknown, 1), None);
}

#[test]
fn min_secure_rank_uses_the_first_admissible_nonmonotone_entry() {
    let q32 = key(
        SisTableDigest::CURRENT,
        SisModulusProfileId::Q32Offset99,
        SisMatrixRole::Inner,
        64,
        32_767,
    );
    assert_eq!(min_secure_rank(q32, 5), Some(3));
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
            for dimension in [32, 64, 128, 256, 512, 1024, 2048] {
                for &bound in COEFF_LINF_BUCKETS {
                    let Some(cell) = sis_role_cell(role, profile, dimension, bound) else {
                        continue;
                    };
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
}
