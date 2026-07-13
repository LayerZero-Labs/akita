use super::*;

use crate::sis::{sis_table_key_for_linf_bound, AjtaiKeyParams};
use crate::{
    validate_compression_catalog, CompressionAlphabet, CompressionCatalogContext,
    CompressionChainSpec, CompressionMapSpec, OpeningClaimsLayout, DEFAULT_SIS_SECURITY_BITS,
};

fn compression_key(d: usize, raw_bound: u128, col_len: usize) -> AjtaiKeyParams {
    let table_key = sis_table_key_for_linf_bound(
        DEFAULT_SIS_SECURITY_BITS,
        SisModulusFamily::Q128,
        d as u32,
        raw_bound,
    )
    .expect("test SIS row");
    AjtaiKeyParams::try_new_with_min_rank(table_key, col_len).expect("test secure key")
}

fn compression_chain(
    lp: &LevelParams,
    source: CompressionSourceId,
    source_key: &AjtaiKeyParams,
    map_d: usize,
    terminal_extra_rows: usize,
) -> CompressionChainSpec {
    let source_d = source_key.sis_table_key().ring_dimension as usize;
    let mut previous_output = source_key.row_len() * source_d;
    let maps = [
        CompressionAlphabet::OpeningBase {
            log_basis: lp.log_basis,
        },
        CompressionAlphabet::NegativeBinary,
    ]
    .into_iter()
    .enumerate()
    .map(|(map, alphabet)| {
        let depth = match alphabet {
            CompressionAlphabet::OpeningBase { log_basis } => {
                (128usize).div_ceil(log_basis as usize)
            }
            CompressionAlphabet::NegativeBinary => 128,
        };
        let raw_bound = match alphabet {
            CompressionAlphabet::OpeningBase { .. } => (1u128 << lp.log_basis) - 1,
            CompressionAlphabet::NegativeBinary => 1,
        };
        let input_coeffs = previous_output * depth;
        assert_eq!(input_coeffs % map_d, 0);
        let mut key = compression_key(map_d, raw_bound, input_coeffs / map_d);
        if map == 1 && terminal_extra_rows != 0 {
            key = AjtaiKeyParams::try_new(
                DEFAULT_SIS_SECURITY_BITS,
                SisModulusFamily::Q128,
                key.row_len() + terminal_extra_rows,
                key.col_len(),
                raw_bound,
                map_d,
            )
            .expect("test key above SIS floor");
        }
        previous_output = key.row_len() * map_d;
        CompressionMapSpec::new(key, alphabet)
    })
    .collect();
    CompressionChainSpec::new(source, maps)
}

fn compression_projection(
    lp: &mut LevelParams,
    map_d: usize,
    outer_d: usize,
    opening_d: usize,
    outer_terminal_extra_rows: usize,
    opening_terminal_extra_rows: usize,
) -> CompressionCatalogProjection {
    lp.b_key = compression_key(outer_d, 63, 1);
    lp.d_key = compression_key(opening_d, 63, 1);
    lp.stamp_role_dims_from_keys();
    let opening = OpeningClaimsLayout::new(4, 1).unwrap();
    validate_compression_catalog::<F>(
        lp,
        CompressionCatalogContext::CoGeneratedLevel { opening: &opening },
        64,
        vec![
            compression_chain(
                lp,
                CompressionSourceId::CurrentOuter,
                &lp.b_key,
                map_d,
                outer_terminal_extra_rows,
            ),
            compression_chain(
                lp,
                CompressionSourceId::Opening,
                &lp.d_key,
                map_d,
                opening_terminal_extra_rows,
            ),
        ],
    )
    .unwrap()
    .project_for_schedule()
    .unwrap()
}

#[test]
fn projected_compression_replaces_only_h_and_successor_f_wire_bytes() {
    let challenge = SparseChallengeConfig::pm1_only(3);
    let mut current = LevelParams::params_only(SisModulusFamily::Q128, 64, 4, 1, 1, 1, challenge)
        .with_decomp(1, 1, 1, 1, 0)
        .unwrap();
    let mut successor = current.clone();
    let current_projection = compression_projection(&mut current, 32, 32, 64, 0, 1);
    let successor_projection = compression_projection(&mut successor, 64, 64, 32, 2, 0);
    let current_f = current_projection
        .payload_coeffs(CompressionSourceId::CurrentOuter)
        .unwrap();
    let current_h = current_projection
        .payload_coeffs(CompressionSourceId::Opening)
        .unwrap();
    let successor_f = successor_projection
        .payload_coeffs(CompressionSourceId::CurrentOuter)
        .unwrap();
    assert_ne!(
        current_f, current_h,
        "fixture must distinguish current B/F from D/H"
    );
    assert_ne!(
        current_f, successor_f,
        "fixture must distinguish current F from successor F"
    );
    assert_ne!(
        current_h, successor_f,
        "fixture must distinguish current H from successor F"
    );
    let compressed =
        CompressedFoldWirePayload::from_catalogs(&current_projection, &successor_projection, 64)
            .unwrap();
    let next_w_len = 64 * 8;

    let native = level_proof_bytes(
        128,
        128,
        &current,
        Some(FoldWirePayload::Native {
            next_level: &successor,
            next_base_field_bits: 64,
        }),
        next_w_len,
        1,
        RelationMatrixRowLayout::WithDBlock,
    )
    .unwrap();
    let projected = level_proof_bytes(
        128,
        128,
        &current,
        Some(FoldWirePayload::Compressed(compressed)),
        next_w_len,
        1,
        RelationMatrixRowLayout::WithDBlock,
    )
    .unwrap();
    let native_wire = current.d_key.row_len() * current.ring_dimension * field_bytes(128)
        + successor.b_key.row_len() * successor.ring_dimension * field_bytes(64);
    let projected_wire = current_projection
        .payload_coeffs(CompressionSourceId::Opening)
        .unwrap()
        * field_bytes(128)
        + successor_projection
            .payload_coeffs(CompressionSourceId::CurrentOuter)
            .unwrap()
            * field_bytes(64);
    assert_eq!(projected, native - native_wire + projected_wire);
    assert_eq!(native - native_wire, projected - projected_wire);
}

#[test]
fn projected_compression_requires_current_opening_identity() {
    let challenge = SparseChallengeConfig::pm1_only(3);
    let mut current = LevelParams::params_only(SisModulusFamily::Q128, 64, 4, 1, 1, 1, challenge)
        .with_decomp(1, 1, 1, 1, 0)
        .unwrap();
    let successor = current.clone();
    current.b_key = compression_key(32, 63, 1);
    current.stamp_role_dims_from_keys();
    let standalone = validate_compression_catalog::<F>(
        &current,
        CompressionCatalogContext::StandaloneCommitment {
            max_opening_log_basis: current.log_basis,
        },
        64,
        vec![compression_chain(
            &current,
            CompressionSourceId::CurrentOuter,
            &current.b_key,
            32,
            0,
        )],
    )
    .unwrap()
    .project_for_schedule()
    .unwrap();
    let successor_projection = compression_projection(&mut successor.clone(), 32, 32, 32, 0, 0);
    assert!(
        CompressedFoldWirePayload::from_catalogs(&standalone, &successor_projection, 128).is_err()
    );
}

#[test]
fn proof_byte_accounting_rejects_layout_mismatch_and_overflow() {
    let challenge = SparseChallengeConfig::pm1_only(3);
    let lp = LevelParams::params_only(SisModulusFamily::Q128, 64, 6, 1, 1, 1, challenge);
    assert!(level_proof_bytes(
        128,
        128,
        &lp,
        None,
        64,
        1,
        RelationMatrixRowLayout::WithDBlock,
    )
    .is_err());

    assert!(level_proof_bytes(
        128,
        128,
        &lp,
        None,
        0,
        1,
        RelationMatrixRowLayout::WithoutDBlock,
    )
    .is_err());
    let mut zero_d = lp.clone();
    zero_d.ring_dimension = 0;
    assert!(level_proof_bytes(
        128,
        128,
        &zero_d,
        None,
        64,
        1,
        RelationMatrixRowLayout::WithoutDBlock,
    )
    .is_err());
    let mut non_power_of_two_d = lp.clone();
    non_power_of_two_d.ring_dimension = 48;
    assert!(level_proof_bytes(
        128,
        128,
        &non_power_of_two_d,
        None,
        96,
        1,
        RelationMatrixRowLayout::WithoutDBlock,
    )
    .is_err());
    assert!(level_proof_bytes(
        128,
        128,
        &lp,
        None,
        65,
        1,
        RelationMatrixRowLayout::WithoutDBlock,
    )
    .is_err());
    assert!(level_proof_bytes(
        128,
        128,
        &lp,
        Some(FoldWirePayload::Native {
            next_level: &lp,
            next_base_field_bits: 128,
        }),
        64,
        1,
        RelationMatrixRowLayout::WithoutDBlock,
    )
    .is_err());

    let mut overflowing = lp.clone();
    overflowing.ring_dimension = usize::MAX;
    assert!(level_proof_bytes(
        128,
        128,
        &overflowing,
        Some(FoldWirePayload::Native {
            next_level: &lp,
            next_base_field_bits: 128,
        }),
        64,
        1,
        RelationMatrixRowLayout::WithDBlock,
    )
    .is_err());

    let multiplication_overflow = CompressedFoldWirePayload {
        opening_payload_coeffs: usize::MAX,
        next_commitment_payload_coeffs: 1,
        next_base_field_bits: 128,
    };
    assert!(FoldWirePayload::Compressed(multiplication_overflow)
        .bytes(&lp, 128)
        .is_err());
    let addition_overflow = CompressedFoldWirePayload {
        opening_payload_coeffs: usize::MAX / field_bytes(128),
        next_commitment_payload_coeffs: 2,
        next_base_field_bits: 64,
    };
    assert!(FoldWirePayload::Compressed(addition_overflow)
        .bytes(&lp, 128)
        .is_err());

    let exact_mixed_width = CompressedFoldWirePayload {
        opening_payload_coeffs: 3,
        next_commitment_payload_coeffs: 5,
        next_base_field_bits: 64,
    };
    assert_eq!(
        FoldWirePayload::Compressed(exact_mixed_width)
            .bytes(&lp, 128)
            .unwrap(),
        3 * field_bytes(128) + 5 * field_bytes(64),
    );

    let mut unit_d = lp.clone();
    unit_d.ring_dimension = 1;
    assert!(level_proof_bytes(
        128,
        128,
        &unit_d,
        None,
        usize::MAX,
        1,
        RelationMatrixRowLayout::WithoutDBlock,
    )
    .is_err());
}
