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
        assert!(
            min_secure_rank(larger_key, matrix.input_width() as u64)
                .is_none_or(|rank| rank > matrix.output_rank()),
            "capacity must be the largest bucket supported by the fixed matrix"
        );
    }
}

#[test]
fn l2_key_rounds_the_complete_collision_to_a_power_of_two() {
    assert_eq!(ceil_supported_l2_collision_sq(1), Some(2));
    assert_eq!(ceil_supported_l2_collision_sq(2), Some(2));
    assert_eq!(ceil_supported_l2_collision_sq(3), Some(4));
    assert_eq!(
        ceil_supported_l2_collision_sq(1u128 << 84),
        Some(1u128 << 84)
    );
    assert_eq!(ceil_supported_l2_collision_sq((1u128 << 84) + 1), None);

    let key = sis_l2_table_key_for_collision_sq(
        DEFAULT_SIS_SECURITY_POLICY,
        SisL2TableDigest::CURRENT,
        SisModulusProfileId::Q128OffsetA7F7,
        64,
        (1u128 << 49) + 1,
    )
    .expect("generated L2 key");
    assert_eq!(key.collision_l2_sq, 1u128 << 50);
}

#[test]
fn l2_table_has_the_expected_q128_d64_rank_boundary() {
    let key = sis_l2_table_key_for_collision_sq(
        DEFAULT_SIS_SECURITY_POLICY,
        SisL2TableDigest::CURRENT,
        SisModulusProfileId::Q128OffsetA7F7,
        64,
        1u128 << 50,
    )
    .expect("generated L2 key");
    assert_eq!(min_secure_l2_rank(key, 21), Some(3));
    assert_eq!(min_secure_l2_rank(key, 22), Some(4));
    assert_eq!(min_secure_l2_rank(key, 512), Some(4));
}

#[test]
fn l2_lookup_rejects_an_unknown_table_digest() {
    assert!(sis_l2_table_key_for_collision_sq(
        DEFAULT_SIS_SECURITY_POLICY,
        SisL2TableDigest([0u8; 32]),
        SisModulusProfileId::Q128OffsetA7F7,
        64,
        1u128 << 50,
    )
    .is_none());
}

#[test]
fn inner_l2_route_owns_its_cap_shape_and_table_identity() {
    let table_key = sis_l2_table_key_for_collision_sq(
        DEFAULT_SIS_SECURITY_POLICY,
        SisL2TableDigest::CURRENT,
        SisModulusProfileId::Q128OffsetA7F7,
        64,
        1u128 << 50,
    )
    .expect("generated L2 key");
    let shape = PhysicalL2NormProofShape::Direct {
        physical_response_len: 512,
    };
    let matrix =
        InnerCommitMatrixParams::try_new_l2_with_min_rank(table_key, 21, 1u128 << 30, shape)
            .expect("audited L2 matrix");

    assert_eq!(matrix.output_rank(), 3);
    assert_eq!(matrix.sis_table_key(), None);
    assert_eq!(matrix.coeff_linf_bound(), None);
    assert_eq!(
        matrix.security_route(),
        InnerCommitSecurityRoute::L2 {
            table_key,
            response_l2_sq_cap: 1u128 << 30,
            norm_proof_shape: shape,
        }
    );
    matrix.validate().expect("L2 route re-audits");

    let mut descriptor = Vec::new();
    matrix.append_descriptor_bytes(&mut descriptor);
    let different_cap =
        InnerCommitMatrixParams::try_new_l2_with_min_rank(table_key, 21, (1u128 << 30) - 1, shape)
            .expect("second audited L2 matrix");
    let mut different_descriptor = Vec::new();
    different_cap.append_descriptor_bytes(&mut different_descriptor);
    assert_ne!(descriptor, different_descriptor);
}

#[test]
fn limb_gram_shape_derives_complete_pair_and_block_count() {
    let shape = PhysicalL2NormProofShape::LimbGram {
        physical_response_len: 65,
        block_len: 32,
        limb_count: 3,
    };
    shape.validate().expect("checked limb-Gram shape");
    assert_eq!(shape.subclaim_count(), Some(18));
    assert!(PhysicalL2NormProofShape::LimbGram {
        physical_response_len: 65,
        block_len: 0,
        limb_count: 3,
    }
    .validate()
    .is_err());
}

#[test]
fn physical_norm_shape_derivation_selects_direct_or_limb_gram() {
    let direct = PhysicalL2NormProofShape::derive(SisModulusProfileId::Q128OffsetA7F7, 64, 4, 2)
        .expect("large-field direct shape");
    assert_eq!(
        direct,
        PhysicalL2NormProofShape::Direct {
            physical_response_len: 64
        }
    );
    direct
        .validate_integer_soundness(SisModulusProfileId::Q128OffsetA7F7, 4, 2)
        .expect("direct no-wrap validation");

    let gram = PhysicalL2NormProofShape::derive(SisModulusProfileId::Q32Offset99, 1 << 22, 64, 5)
        .expect("small-field multi-block limb-Gram shape");
    let PhysicalL2NormProofShape::LimbGram {
        physical_response_len,
        block_len,
        limb_count,
    } = gram
    else {
        panic!("expected limb-Gram shape");
    };
    assert_eq!(physical_response_len, 1 << 22);
    assert_eq!(limb_count, 5);
    assert!((1..physical_response_len).contains(&block_len));
    assert!(gram.subclaim_count().expect("subclaim count") > 15);
    gram.validate_integer_soundness(SisModulusProfileId::Q32Offset99, 64, 5)
        .expect("limb-Gram no-wrap validation");
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
    let key = sis_table_key_for_linf_bound(
        DEFAULT_SIS_SECURITY_POLICY,
        SisTableDigest::CURRENT,
        SisModulusProfileId::Q32Offset99,
        SisMatrixRole::Inner,
        128,
        linf,
    );
    if let Some(key) = key {
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
fn d512_coverage_is_q128_inner_only() {
    let inner = sis_role_cell(
        SisMatrixRole::Inner,
        SisModulusProfileId::Q128OffsetA7F7,
        512,
        2,
    )
    .expect("q128 Inner/512 cell");
    assert_eq!(inner.max_module_rank, 20);

    for role in [SisMatrixRole::Outer, SisMatrixRole::Open] {
        assert!(sis_role_cell(role, SisModulusProfileId::Q128OffsetA7F7, 512, 3).is_none());
    }
    for profile in [
        SisModulusProfileId::Q32Offset99,
        SisModulusProfileId::Q64Offset59,
    ] {
        assert!(sis_role_cell(SisMatrixRole::Inner, profile, 512, 2).is_none());
    }
}

#[test]
fn d512_digest_has_direct_full_rank_rows() {
    for &bound in COEFF_LINF_BUCKETS {
        let d512 = sis_max_widths(
            DEFAULT_SIS_SECURITY_POLICY,
            SisTableDigest::Q128_INNER_D512,
            SisModulusProfileId::Q128OffsetA7F7,
            512,
            bound,
        )
        .expect("direct D512 rows");
        assert_eq!(d512.len(), 20);
        assert!(d512.windows(2).all(|pair| pair[0] <= pair[1]));
    }
}

#[test]
fn d512_requires_expanded_digest_and_rejects_unknown_digest() {
    let old = key(
        SisTableDigest::CURRENT,
        SisModulusProfileId::Q128OffsetA7F7,
        SisMatrixRole::Inner,
        512,
        2,
    );
    assert_eq!(min_secure_rank(old, 1), None);

    let expanded = key(
        SisTableDigest::Q128_INNER_D512,
        SisModulusProfileId::Q128OffsetA7F7,
        SisMatrixRole::Inner,
        512,
        2,
    );
    assert!(min_secure_rank(expanded, 1).is_some());

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
fn expanded_digest_preserves_existing_rows() {
    for &bound in COEFF_LINF_BUCKETS {
        assert_eq!(
            sis_max_widths(
                DEFAULT_SIS_SECURITY_POLICY,
                SisTableDigest::Q128_INNER_D512,
                SisModulusProfileId::Q128OffsetA7F7,
                256,
                bound,
            ),
            sis_max_widths(
                DEFAULT_SIS_SECURITY_POLICY,
                SisTableDigest::CURRENT,
                SisModulusProfileId::Q128OffsetA7F7,
                256,
                bound,
            ),
        );
    }
}
