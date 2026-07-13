use super::*;
use crate::layout::compression::{
    validate_compression_catalog, CompressionAlphabet, CompressionCatalogContext,
    CompressionChainSpec, CompressionMapSpec, CompressionSourceId,
};
use crate::sis::{sis_table_key_for_linf_bound, AjtaiKeyParams, DEFAULT_SIS_SECURITY_BITS};
use crate::{LevelParams, OpeningClaimsLayout, RelationRowId, SisModulusFamily};
use akita_challenges::SparseChallengeConfig;
use akita_field::Prime128OffsetA7F7 as F;

fn key(d: usize, bound: u128, columns: usize) -> AjtaiKeyParams {
    let table = sis_table_key_for_linf_bound(
        DEFAULT_SIS_SECURITY_BITS,
        SisModulusFamily::Q128,
        d as u32,
        bound,
    )
    .unwrap();
    AjtaiKeyParams::try_new_with_min_rank(table, columns).unwrap()
}

fn chain(
    source: CompressionSourceId,
    source_key: &AjtaiKeyParams,
    dimensions: &[usize],
) -> CompressionChainSpec {
    let mut output_coeffs =
        source_key.row_len() * source_key.sis_table_key().ring_dimension as usize;
    let maps = dimensions
        .iter()
        .copied()
        .map(|d| {
            let input_coeffs = output_coeffs * 128;
            let map_key = key(d, 1, input_coeffs / d);
            output_coeffs = map_key.row_len() * d;
            CompressionMapSpec::new(map_key, CompressionAlphabet::NegativeBinary)
        })
        .collect();
    CompressionChainSpec::new(source, maps)
}

fn alphabet_chain(
    source: CompressionSourceId,
    source_key: &AjtaiKeyParams,
    maps: &[(usize, CompressionAlphabet)],
) -> CompressionChainSpec {
    let mut output_coeffs =
        source_key.row_len() * source_key.sis_table_key().ring_dimension as usize;
    let maps = maps
        .iter()
        .copied()
        .map(|(d, alphabet)| {
            let depth = crate::compression_digit_depth(alphabet, 128, 6).unwrap();
            let input_coeffs = output_coeffs * depth;
            let bound = match alphabet {
                CompressionAlphabet::NegativeBinary => 1,
                CompressionAlphabet::OpeningBase { log_basis } => (1u128 << log_basis) - 1,
            };
            let map_key = key(d, bound, input_coeffs / d);
            output_coeffs = map_key.row_len() * d;
            CompressionMapSpec::new(map_key, alphabet)
        })
        .collect();
    CompressionChainSpec::new(source, maps)
}

fn mixed_layout() -> RelationLayout {
    let mut lp = LevelParams::params_only(
        SisModulusFamily::Q128,
        64,
        6,
        1,
        1,
        1,
        SparseChallengeConfig::pm1_only(64),
    )
    .with_decomp(1, 1, 1, 1, 0)
    .unwrap();
    lp.b_key = key(64, 63, 1);
    lp.d_key = key(64, 63, 1);
    let opening = OpeningClaimsLayout::new(2, 1).unwrap();
    validate_compression_catalog::<F>(
        &lp,
        CompressionCatalogContext::CoGeneratedLevel { opening: &opening },
        64,
        vec![
            chain(CompressionSourceId::CurrentOuter, &lp.b_key, &[64, 32]),
            alphabet_chain(
                CompressionSourceId::Opening,
                &lp.d_key,
                &[
                    (32, CompressionAlphabet::OpeningBase { log_basis: 6 }),
                    (64, CompressionAlphabet::NegativeBinary),
                ],
            ),
        ],
    )
    .unwrap()
    .co_generated_relation_layout()
    .unwrap()
    .clone()
}

#[test]
fn matches_independent_segment_and_family_oracles() {
    let layout = mixed_layout();
    let cost = layout.compression_structural_cost().unwrap();
    let sum_segments = |kind: fn(RelationSegmentId) -> bool| {
        layout
            .segments()
            .iter()
            .filter(|segment| kind(segment.id()))
            .map(|segment| segment.span().len())
            .sum::<usize>()
    };
    let input_coeffs = sum_segments(|id| matches!(id, RelationSegmentId::CompressionInput { .. }));
    let quotient_coeffs =
        sum_segments(|id| matches!(id, RelationSegmentId::CompressionQuotient { .. }));
    let compression_families = layout
        .row_plan()
        .families()
        .iter()
        .filter(|family| matches!(family.id(), RelationRowId::Compression { .. }))
        .collect::<Vec<_>>();
    let terminal_coeffs = compression_families
        .iter()
        .filter_map(|family| match family.rhs() {
            RelationRowRhs::TerminalPayload { coeffs } => Some(coeffs),
            _ => None,
        })
        .sum::<usize>();

    assert_eq!(cost.map_count(), compression_families.len());
    assert_eq!(cost.witness_input_coeffs(), input_coeffs);
    assert_eq!(cost.witness_quotient_coeffs(), quotient_coeffs);
    assert_eq!(cost.witness_coeffs(), input_coeffs + quotient_coeffs);
    assert_eq!(cost.terminal_payload_coeffs(), terminal_coeffs);
    assert_eq!(
        cost.negative_binary_support_runs(),
        layout.negative_binary_support().len()
    );
    let span_len = |source, map| {
        layout
            .segment(RelationSegmentId::CompressionInput { source, map })
            .unwrap()
            .span()
            .len()
    };
    let expected_binary = span_len(CompressionSourceId::CurrentOuter, 0)
        + span_len(CompressionSourceId::CurrentOuter, 1)
        + span_len(CompressionSourceId::Opening, 1);
    assert_eq!(cost.negative_binary_support_coeffs(), expected_binary);
    assert_eq!(cost.relation_quotient_scan_coeffs(), quotient_coeffs);
    let expected_gadget = span_len(CompressionSourceId::CurrentOuter, 0)
        + span_len(CompressionSourceId::CurrentOuter, 1)
        + span_len(CompressionSourceId::Opening, 0)
        + span_len(CompressionSourceId::Opening, 1);
    assert_eq!(cost.relation_gadget_scan_coeffs(), expected_gadget);
    assert_eq!(
        cost.native_rows(),
        compression_families
            .iter()
            .map(|family| family.rows().len())
            .sum::<usize>()
    );
    assert_eq!(
        cost.dimensions()
            .iter()
            .map(CompressionDimensionCost::native_ring_dim)
            .collect::<Vec<_>>(),
        vec![32, 64]
    );
    assert_eq!(
        cost.coalesced_setup_cache_coeffs(),
        cost.dimensions()
            .iter()
            .map(CompressionDimensionCost::max_setup_prefix_coeffs)
            .sum::<usize>()
    );
    assert_eq!(
        cost.logical_setup_coeffs(),
        cost.dimensions()
            .iter()
            .map(CompressionDimensionCost::logical_setup_coeffs)
            .sum::<usize>()
    );
    assert!(cost.relation_scan_coeffs() > cost.witness_coeffs());
    let terminal_payloads = cost
        .maps()
        .iter()
        .filter_map(CompressionMapStructuralCost::terminal_payload_coeffs)
        .collect::<Vec<_>>();
    assert_eq!(terminal_payloads.len(), 2);
    assert_ne!(terminal_payloads[0], terminal_payloads[1]);
}
