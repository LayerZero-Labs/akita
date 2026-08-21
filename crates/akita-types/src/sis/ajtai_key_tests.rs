use super::*;
use crate::sis::{
    ceil_supported_l2_collision_sq, sis_l2_table_key_for_collision_sq, SisL2TableDigest,
};

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
        physical_response_len: 21 * 64,
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
fn inner_l2_route_rejects_a_shape_inconsistent_with_matrix_width() {
    let table_key = sis_l2_table_key_for_collision_sq(
        DEFAULT_SIS_SECURITY_POLICY,
        SisL2TableDigest::CURRENT,
        SisModulusProfileId::Q128OffsetA7F7,
        64,
        1u128 << 50,
    )
    .expect("generated L2 key");
    let mismatched = PhysicalL2NormProofShape::Direct {
        physical_response_len: 512,
    };
    assert!(InnerCommitMatrixParams::try_new_l2_with_min_rank(
        table_key,
        21,
        1u128 << 30,
        mismatched,
    )
    .is_err());

    let matrix = InnerCommitMatrixParams::try_new_l2_with_min_rank(
        table_key,
        21,
        1u128 << 30,
        PhysicalL2NormProofShape::Direct {
            physical_response_len: 21 * 64,
        },
    )
    .expect("consistent L2 matrix");
    assert!(matrix.try_with_input_width(22).is_err());
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

    let gram = PhysicalL2NormProofShape::derive(SisModulusProfileId::Q32Offset99, 1 << 22, 64, 5)
        .expect("small-field limb-Gram shape");
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
}
