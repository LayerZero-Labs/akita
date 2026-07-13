use super::*;

use std::collections::BTreeMap;

use akita_field::{AkitaError, Prime128OffsetA7F7};

use crate::{
    compression_capacity_infeasible, COMPRESSION_CAPACITY_INFEASIBLE_PREFIX,
    DEFAULT_SIS_SECURITY_BITS,
};

type F = Prime128OffsetA7F7;

fn projection_with_map_d(map_d: usize, terminal_rows: usize) -> CompressionCatalogProjection {
    let mut lp = level();
    lp.b_key = key(SisModulusFamily::Q128, 32, 63, 1);
    lp.stamp_role_dims_from_keys();
    let source_d = lp.b_key.sis_table_key().ring_dimension as usize;
    let mut previous_output = lp.b_key.row_len() * source_d;
    let maps = [CompressionAlphabet::NegativeBinary; 2]
        .into_iter()
        .enumerate()
        .map(|(map_index, alphabet)| {
            let input_coeffs = previous_output * 128;
            assert_eq!(input_coeffs % map_d, 0);
            let mut map_key = key(SisModulusFamily::Q128, map_d, 1, input_coeffs / map_d);
            if map_index == 1 && terminal_rows != 0 {
                map_key = AjtaiKeyParams::try_new(
                    DEFAULT_SIS_SECURITY_BITS,
                    SisModulusFamily::Q128,
                    map_key.row_len() + terminal_rows,
                    map_key.col_len(),
                    1,
                    map_d,
                )
                .expect("terminal key");
            }
            previous_output = map_key.row_len() * map_d;
            CompressionMapSpec {
                key: map_key,
                alphabet,
            }
        })
        .collect();
    validate_compression_catalog::<F>(
        &lp,
        CompressionCatalogContext::StandaloneCommitment,
        64,
        vec![CompressionChainSpec {
            source: CompressionSourceId::CurrentOuter,
            max_opening_log_basis: lp.log_basis,
            maps,
        }],
    )
    .expect("catalog")
    .project_for_schedule()
    .expect("projection")
}

#[test]
fn aggregate_concatenates_map_hints_in_order() {
    let left = projection_with_map_d(32, 0);
    let right = projection_with_map_d(64, 1);
    let aggregated = aggregate_catalog_projections(&[&left, &right]).expect("aggregate");
    let mut expected = left.map_hints().to_vec();
    expected.extend_from_slice(right.map_hints());
    assert_eq!(aggregated.map_hints(), expected.as_slice());
}

#[test]
fn aggregate_max_coalesces_ntt_requirements_and_recomputes_cache_total() {
    let left = projection_with_map_d(32, 0);
    let right = projection_with_map_d(64, 1);
    let aggregated = aggregate_catalog_projections(&[&left, &right]).expect("aggregate");
    assert_eq!(
        aggregated.max_flat_setup_prefix_coeffs(),
        left.max_flat_setup_prefix_coeffs()
            .max(right.max_flat_setup_prefix_coeffs())
    );

    let mut expected_by_d = BTreeMap::<usize, usize>::new();
    for projection in [&left, &right] {
        for &(ring_d, prefix) in projection.ntt_requirements() {
            expected_by_d
                .entry(ring_d)
                .and_modify(|current| *current = (*current).max(prefix))
                .or_insert(prefix);
        }
    }
    let expected_ntt = expected_by_d.into_iter().collect::<Vec<_>>();
    assert_eq!(aggregated.ntt_requirements(), expected_ntt.as_slice());

    let expected_cache = expected_ntt
        .iter()
        .map(|&(ring_d, prefix)| ring_d * prefix)
        .sum::<usize>();
    assert_eq!(aggregated.coalesced_cache_field_coeffs(), expected_cache);
}

#[test]
fn aggregate_empty_projection_list_is_zeroed() {
    let aggregated = aggregate_catalog_projections(&[]).expect("aggregate");
    assert!(aggregated.map_hints().is_empty());
    assert!(aggregated.ntt_requirements().is_empty());
    assert_eq!(aggregated.max_flat_setup_prefix_coeffs(), 0);
    assert_eq!(aggregated.coalesced_cache_field_coeffs(), 0);
    assert_eq!(aggregated.gen_ring_dim(), None);
}

#[test]
fn aggregate_rejects_mismatched_gen_ring_dim() {
    let left = projection_with_map_d(32, 0);
    assert_eq!(left.gen_ring_dim(), 64);
    let right = {
        let mut lp = level();
        lp.b_key = key(SisModulusFamily::Q128, 32, 63, 1);
        lp.stamp_role_dims_from_keys();
        let source_d = lp.b_key.sis_table_key().ring_dimension as usize;
        let map_d = 64;
        let mut previous_output = lp.b_key.row_len() * source_d;
        let maps = [CompressionAlphabet::NegativeBinary; 2]
            .into_iter()
            .map(|alphabet| {
                let input_coeffs = previous_output * 128;
                assert_eq!(input_coeffs % map_d, 0);
                let map_key = key(SisModulusFamily::Q128, map_d, 1, input_coeffs / map_d);
                previous_output = map_key.row_len() * map_d;
                CompressionMapSpec {
                    key: map_key,
                    alphabet,
                }
            })
            .collect();
        validate_compression_catalog::<F>(
            &lp,
            CompressionCatalogContext::StandaloneCommitment,
            128,
            vec![CompressionChainSpec {
                source: CompressionSourceId::CurrentOuter,
                max_opening_log_basis: lp.log_basis,
                maps,
            }],
        )
        .expect("catalog")
        .project_for_schedule()
        .expect("projection")
    };
    assert_eq!(right.gen_ring_dim(), 128);
    assert!(aggregate_catalog_projections(&[&left, &right]).is_err());
}

#[test]
fn capacity_infeasible_classifier_matches_tagged_setup_errors() {
    let tagged = AkitaError::InvalidSetup(format!(
        "{COMPRESSION_CAPACITY_INFEASIBLE_PREFIX} map setup footprint exceeds setup matrix field cap"
    ));
    assert!(compression_capacity_infeasible(&tagged));
    assert!(!compression_capacity_infeasible(&AkitaError::InvalidSetup(
        "compression map 0 setup footprint 99 exceeds 1".into()
    )));
    assert!(!compression_capacity_infeasible(&AkitaError::InvalidProof));
}
