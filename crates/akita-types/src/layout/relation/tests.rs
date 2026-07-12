use super::compiler::{compile_relation_layout, normalize_support};
use super::rows::checked_padded_row_count;
use super::*;
use crate::sis::compute_num_digits_full_field;
use akita_challenges::SparseChallengeConfig;
use akita_field::{CanonicalField, Prime128OffsetA7F7 as F};
use akita_serialization::DEFAULT_MAX_SEQUENCE_LEN;

use crate::layout::compression::{
    validate_compression_catalog, CompressionAlphabet, CompressionCatalogContext,
    CompressionChainSpec, CompressionMapSpec,
};
use crate::sis::{sis_table_key_for_linf_bound, AjtaiKeyParams, DEFAULT_SIS_SECURITY_BITS};
fn level() -> (LevelParams, OpeningClaimsLayout) {
    let lp = LevelParams::params_only(
        crate::SisModulusFamily::Q128,
        64,
        6,
        1,
        1,
        1,
        SparseChallengeConfig::pm1_only(64),
    )
    .with_decomp(1, 1, 1, 1, 0)
    .unwrap();
    (lp, OpeningClaimsLayout::new(2, 1).unwrap())
}

fn certified_key(d: usize, raw_bound: u128, cols: usize) -> AjtaiKeyParams {
    let table = sis_table_key_for_linf_bound(
        DEFAULT_SIS_SECURITY_BITS,
        crate::SisModulusFamily::Q128,
        d as u32,
        raw_bound,
    )
    .unwrap();
    AjtaiKeyParams::try_new_with_min_rank(table, cols).unwrap()
}

fn chain(
    source: CompressionSourceId,
    source_key: &AjtaiKeyParams,
    alphabets: &[CompressionAlphabet],
) -> CompressionChainSpec {
    let mut previous = source_key.row_len() * source_key.sis_table_key().ring_dimension as usize;
    let maps = alphabets
        .iter()
        .copied()
        .enumerate()
        .map(|(index, alphabet)| {
            let d = if index == 0 { 64 } else { 32 };
            let depth = match alphabet {
                CompressionAlphabet::NegativeBinary => 128,
                CompressionAlphabet::OpeningBase { log_basis } => {
                    crate::sis::num_digits_for_bound(128, 128, log_basis)
                }
            };
            let input = previous * depth;
            let raw_bound = match alphabet {
                CompressionAlphabet::NegativeBinary => 1,
                CompressionAlphabet::OpeningBase { log_basis } => {
                    ((1u128 << log_basis) - 1).max(63)
                }
            };
            let key = certified_key(d, raw_bound, input / d);
            previous = key.row_len() * d;
            CompressionMapSpec::new(key, alphabet)
        })
        .collect();
    CompressionChainSpec::new(source, maps)
}

#[test]
fn empty_compression_is_byte_order_identical_to_the_single_group_oracle() {
    let (lp, opening) = level();
    let layout = compile_relation_layout(
        &lp,
        &opening,
        RelationMatrixRowLayout::WithDBlock,
        lp.field_bits_for_cache(),
        None,
    )
    .unwrap();
    let base_plan =
        RelationRowPlan::compile_base(&lp, &opening, RelationMatrixRowLayout::WithDBlock).unwrap();
    let lens = layout.witness_layout(None).unwrap().chunk_lengths[0];
    let d = lp.ring_dimension;
    assert_eq!(
        layout.segments[0].span,
        CoeffSpan {
            start: 0,
            len: lens.z_len * d
        }
    );
    assert_eq!(layout.segments[1].span.len, lens.e_len * d);
    assert_eq!(layout.segments[2].span.len, lens.t_len * d);
    assert_eq!(layout.row_plan, base_plan);
    assert_eq!(
        layout.row_plan.families[0].rows,
        RowSpan { start: 0, len: 1 }
    );
    assert_eq!(layout.row_plan.families[1].rows.start, 1);
    assert_eq!(
        layout.row_plan.families[2].rows.start,
        1 + layout.row_plan.families[1].rows.len
    );
    assert!(layout.negative_binary_support.is_empty());
    assert!(layout
        .physical_negative_binary_support()
        .unwrap()
        .is_empty());
    assert!(layout
        .physical_compression_segment_span(RelationSegmentId::Z {
            group: RelationGroupId::Current,
        })
        .is_err());
    assert!(layout
        .physical_compression_segment_span(RelationSegmentId::CompressionInput {
            source: CompressionSourceId::CurrentOuter,
            map: 0,
        })
        .is_err());
    let quotient_coeffs = layout.row_plan.trace_row
        * compute_num_digits_full_field(F::modulus_bits(), lp.log_basis)
        * d;
    assert_eq!(
        layout.total_coeffs,
        (lens.z_len + lens.e_len + lens.t_len) * d + quotient_coeffs
    );
    assert_eq!(
        layout.physical_witness_field_coeff_len().unwrap(),
        layout.witness_layout(None).unwrap().ring_len().unwrap() * d
    );
    assert_eq!(
        layout.row_plan.padded_row_count,
        (layout.row_plan.trace_row + 1).next_power_of_two()
    );
}

#[test]
fn base_plan_scalar_layouts_and_padding_are_exact() {
    let lp = LevelParams::params_only(
        crate::SisModulusFamily::Q128,
        64,
        6,
        2,
        2,
        3,
        SparseChallengeConfig::pm1_only(64),
    )
    .with_decomp(1, 1, 1, 1, 0)
    .unwrap();
    let opening = OpeningClaimsLayout::new(2, 1).unwrap();
    let with_d =
        RelationRowPlan::compile_base(&lp, &opening, RelationMatrixRowLayout::WithDBlock).unwrap();
    assert_eq!(with_d.trace_row(), 8);
    assert_eq!(with_d.padded_row_count(), 16);
    assert_eq!(
        with_d.families().iter().map(|f| f.id()).collect::<Vec<_>>(),
        vec![
            RelationRowId::Consistency,
            RelationRowId::A {
                group: RelationGroupId::Current
            },
            RelationRowId::B {
                group: RelationGroupId::Current
            },
            RelationRowId::D,
        ]
    );
    let without_d =
        RelationRowPlan::compile_base(&lp, &opening, RelationMatrixRowLayout::WithoutDBlock)
            .unwrap();
    assert_eq!(without_d.trace_row(), 5);
    assert_eq!(without_d.padded_row_count(), 8);
    assert!(without_d.family(RelationRowId::D).is_err());
}

#[test]
fn relation_compiler_rejects_stale_field_width_and_invalid_basis_without_panicking() {
    let (mut lp, opening) = level();
    assert!(RelationLayout::from_authenticated_statement(
        &lp,
        &opening,
        RelationMatrixRowLayout::WithDBlock,
        32,
    )
    .is_err());
    assert!(RelationLayout::from_authenticated_statement(
        &lp,
        &opening,
        RelationMatrixRowLayout::WithDBlock,
        lp.field_bits_for_cache(),
    )
    .is_ok());
    for invalid in [0, 128] {
        lp.log_basis = invalid;
        let result = std::panic::catch_unwind(|| {
            RelationLayout::from_authenticated_statement(
                &lp,
                &opening,
                RelationMatrixRowLayout::WithDBlock,
                lp.field_bits_for_cache(),
            )
        });
        assert!(result.is_ok());
        assert!(result.unwrap().is_err());
    }
}

#[test]
fn malformed_support_is_rejected_and_adjacent_runs_are_normalized() {
    assert!(normalize_support(
        vec![
            CoeffSpan { start: 3, len: 2 },
            CoeffSpan { start: 4, len: 2 }
        ],
        8
    )
    .is_err());
    assert_eq!(
        normalize_support(
            vec![
                CoeffSpan { start: 3, len: 2 },
                CoeffSpan { start: 5, len: 2 }
            ],
            8
        )
        .unwrap(),
        vec![CoeffSpan { start: 3, len: 4 }]
    );
    assert!(normalize_support(vec![CoeffSpan { start: 7, len: 2 }], 8).is_err());
    assert_eq!(
        checked_padded_row_count(DEFAULT_MAX_SEQUENCE_LEN - 1).unwrap(),
        DEFAULT_MAX_SEQUENCE_LEN
    );
    assert!(checked_padded_row_count(DEFAULT_MAX_SEQUENCE_LEN).is_err());
}

#[test]
fn checked_compression_extends_rows_and_directly_augments_existing_sources() {
    let (mut lp, opening) = level();
    lp.b_key = certified_key(64, 63, 1);
    lp.d_key = certified_key(64, 63, 1);
    let specs = vec![
        chain(
            CompressionSourceId::CurrentOuter,
            &lp.b_key,
            &[
                CompressionAlphabet::NegativeBinary,
                CompressionAlphabet::NegativeBinary,
                CompressionAlphabet::NegativeBinary,
            ],
        ),
        chain(
            CompressionSourceId::Opening,
            &lp.d_key,
            &[
                CompressionAlphabet::OpeningBase { log_basis: 6 },
                CompressionAlphabet::NegativeBinary,
            ],
        ),
    ];
    let catalog = validate_compression_catalog::<F>(
        &lp,
        CompressionCatalogContext::CoGeneratedLevel { opening: &opening },
        64,
        specs,
    )
    .unwrap();
    let base = compile_relation_layout(
        &lp,
        &opening,
        RelationMatrixRowLayout::WithDBlock,
        lp.field_bits_for_cache(),
        None,
    )
    .unwrap();
    let layout = catalog.co_generated_relation_layout().unwrap();

    assert!(layout.row_plan.trace_row > base.row_plan.trace_row);
    assert!(matches!(
        &layout.row_plan.families[2].inputs,
        RelationRowInputs::B {
            compression_input: Some(GadgetInput {
                segment: RelationSegmentId::CompressionInput {
                    source: CompressionSourceId::CurrentOuter,
                    map: 0
                },
                log_basis: 1,
            }),
            ..
        }
    ));
    assert!(matches!(
        &layout.row_plan.families[3].inputs,
        RelationRowInputs::D {
            compression_input: Some(GadgetInput {
                segment: RelationSegmentId::CompressionInput {
                    source: CompressionSourceId::Opening,
                    map: 0
                },
                log_basis: 6,
            }),
            ..
        }
    ));
    let first_compression = &layout.row_plan.families[base.row_plan.families.len()];
    assert!(layout.row_plan.validate_uniform_execution(64).is_err());
    assert_eq!(first_compression.rows.start, base.row_plan.trace_row);
    assert!(matches!(
        first_compression.inputs,
        RelationRowInputs::Compression {
            successor: Some(_),
            ..
        }
    ));
    assert!(!layout.negative_binary_support.is_empty());
    assert!(layout
        .negative_binary_support
        .windows(2)
        .all(|pair| pair[0].end().unwrap() < pair[1].start));

    let compression_coeffs = layout
        .segments()
        .iter()
        .filter(|segment| {
            matches!(
                segment.id(),
                RelationSegmentId::CompressionInput { .. }
                    | RelationSegmentId::CompressionQuotient { .. }
            )
        })
        .map(|segment| segment.span().len())
        .sum::<usize>();
    assert_eq!(layout.compression_witness_coeffs, compression_coeffs);
    assert_eq!(
        layout.total_coeffs(),
        base.total_coeffs() + compression_coeffs
    );
    let carrier_d = lp.ring_dimension;
    let base_coeffs = layout.witness_layout(None).unwrap().ring_len().unwrap() * carrier_d;
    let unpadded = base_coeffs + compression_coeffs;
    let expected = unpadded.div_ceil(carrier_d) * carrier_d;
    assert_eq!(layout.physical_witness_field_coeff_len().unwrap(), expected);
    let mut semantic_cursor = 0;
    for segment in layout.segments() {
        assert_eq!(segment.span().start(), semantic_cursor);
        semantic_cursor = segment.span().end().unwrap();
    }
    assert_eq!(semantic_cursor, layout.total_coeffs());
    let logical_compression_base = base.total_coeffs();
    let physical_compression_base =
        layout.witness_layout(None).unwrap().ring_len().unwrap() * carrier_d;
    for segment in layout.segments().iter().filter(|segment| {
        matches!(
            segment.id(),
            RelationSegmentId::CompressionInput { .. }
                | RelationSegmentId::CompressionQuotient { .. }
        )
    }) {
        let physical = layout
            .physical_compression_segment_span(segment.id())
            .unwrap();
        assert_eq!(physical.len(), segment.span().len());
        assert_eq!(
            physical.start(),
            physical_compression_base + segment.span().start() - logical_compression_base
        );
    }
    let negative_xi = layout
        .segments()
        .iter()
        .filter_map(|segment| match segment.id() {
            RelationSegmentId::CompressionInput {
                source: CompressionSourceId::CurrentOuter,
                ..
            }
            | RelationSegmentId::CompressionInput {
                source: CompressionSourceId::Opening,
                map: 1,
            } => Some(segment.span()),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        layout.negative_binary_support(),
        normalize_support(negative_xi, layout.total_coeffs())
            .unwrap()
            .as_slice()
    );
    let physical_support = layout.physical_negative_binary_support().unwrap();
    assert_eq!(
        physical_support.len(),
        layout.negative_binary_support().len()
    );
    for (physical, logical) in physical_support
        .iter()
        .zip(layout.negative_binary_support())
    {
        assert_eq!(physical.len(), logical.len());
        assert_eq!(
            physical.start(),
            physical_compression_base + logical.start() - logical_compression_base
        );
    }
    assert!(physical_support
        .windows(2)
        .all(|pair| pair[0].end().unwrap() < pair[1].start()));
    let unpadded_physical_end = physical_compression_base + compression_coeffs;
    assert!(physical_support
        .iter()
        .all(|span| span.end().unwrap() <= unpadded_physical_end));
    assert!(unpadded_physical_end <= layout.physical_witness_field_coeff_len().unwrap());

    let compression_id = layout
        .segments()
        .iter()
        .find(|segment| matches!(segment.id(), RelationSegmentId::CompressionInput { .. }))
        .unwrap()
        .id();
    let mut malformed = layout.clone();
    malformed.total_coeffs = malformed.compression_witness_coeffs - 1;
    assert!(malformed
        .physical_compression_segment_span(compression_id)
        .is_err());
    let mut malformed_support = layout.clone();
    malformed_support.negative_binary_support[0].start = 0;
    assert!(malformed_support
        .physical_negative_binary_support()
        .is_err());
    let mut zero_carrier = layout.clone();
    zero_carrier.carrier_ring_dim = 0;
    assert!(zero_carrier
        .physical_compression_segment_span(compression_id)
        .is_err());
    let mut overflowing = layout.clone();
    overflowing.carrier_ring_dim = usize::MAX;
    assert!(overflowing
        .physical_compression_segment_span(compression_id)
        .is_err());
}

#[test]
fn terminal_standalone_sizes_f_geometry_without_a_d_base_quotient() {
    let (mut lp, opening) = level();
    lp.b_key = certified_key(64, 63, 1);
    let catalog = validate_compression_catalog::<F>(
        &lp,
        CompressionCatalogContext::StandaloneCommitment {
            max_opening_log_basis: 6,
        },
        64,
        vec![chain(
            CompressionSourceId::CurrentOuter,
            &lp.b_key,
            &[
                CompressionAlphabet::OpeningBase { log_basis: 4 },
                CompressionAlphabet::NegativeBinary,
            ],
        )],
    )
    .unwrap();
    let layout = catalog
        .terminal_relation_layout::<F>(&lp, &opening)
        .unwrap();
    assert!(layout.compression_witness_coeffs > 0);
    assert!(layout.segments().iter().any(|segment| matches!(
        segment.id(),
        RelationSegmentId::CompressionInput {
            source: CompressionSourceId::CurrentOuter,
            ..
        }
    )));
    assert!(!layout.segments().iter().any(|segment| matches!(
        segment.id(),
        RelationSegmentId::BaseQuotient {
            row: RelationRowId::D
        }
    )));
    let base_coeffs = layout.witness_layout(None).unwrap().ring_len().unwrap() * 64;
    assert!(layout.physical_witness_field_coeff_len().unwrap() > base_coeffs);
    for segment in layout.segments().iter().filter(|segment| {
        matches!(
            segment.id(),
            RelationSegmentId::CompressionInput { .. }
                | RelationSegmentId::CompressionQuotient { .. }
        )
    }) {
        let physical = layout
            .physical_compression_segment_span(segment.id())
            .unwrap();
        assert_eq!(physical.len(), segment.span().len());
        assert!(physical.start() >= base_coeffs);
    }
    assert!(layout
        .physical_negative_binary_support()
        .unwrap()
        .iter()
        .all(|span| span.start() >= base_coeffs));
}

#[test]
fn witness_field_sizing_rejects_zero_and_overflow() {
    let (mut lp, opening) = level();
    lp.ring_dimension = 0;
    assert!(RelationLayout::from_authenticated_statement(
        &lp,
        &opening,
        RelationMatrixRowLayout::WithDBlock,
        lp.field_bits_for_cache(),
    )
    .is_err());
    let (lp, opening) = level();
    let mut layout = RelationLayout::from_authenticated_statement(
        &lp,
        &opening,
        RelationMatrixRowLayout::WithDBlock,
        lp.field_bits_for_cache(),
    )
    .unwrap();
    layout.carrier_ring_dim = usize::MAX;
    assert!(layout.physical_witness_field_coeff_len().is_err());
    layout.carrier_ring_dim = 64;
    layout.compression_witness_coeffs = usize::MAX;
    assert!(layout.physical_witness_field_coeff_len().is_err());
}

#[test]
fn multi_chunk_sizing_counts_replicated_base_storage() {
    let mut lp = LevelParams::params_only(
        crate::SisModulusFamily::Q128,
        64,
        6,
        1,
        1,
        1,
        SparseChallengeConfig::pm1_only(64),
    )
    .with_decomp(1, 3, 1, 1, 0)
    .unwrap();
    lp.witness_chunk = crate::ChunkedWitnessCfg {
        num_chunks: 4,
        num_activated_levels: 1,
    };
    lp.b_key = certified_key(64, 63, 1);
    lp.d_key = certified_key(64, 63, 1);
    let opening = OpeningClaimsLayout::new(2, 1).unwrap();
    let base = RelationLayout::from_authenticated_statement(
        &lp,
        &opening,
        RelationMatrixRowLayout::WithDBlock,
        lp.field_bits_for_cache(),
    )
    .unwrap();
    let catalog = validate_compression_catalog::<F>(
        &lp,
        CompressionCatalogContext::CoGeneratedLevel { opening: &opening },
        64,
        vec![
            chain(
                CompressionSourceId::CurrentOuter,
                &lp.b_key,
                &[
                    CompressionAlphabet::NegativeBinary,
                    CompressionAlphabet::NegativeBinary,
                    CompressionAlphabet::NegativeBinary,
                ],
            ),
            chain(
                CompressionSourceId::Opening,
                &lp.d_key,
                &[
                    CompressionAlphabet::OpeningBase { log_basis: 6 },
                    CompressionAlphabet::NegativeBinary,
                ],
            ),
        ],
    )
    .unwrap();
    let layout = catalog.co_generated_relation_layout().unwrap();
    let physical = layout.witness_layout(None).unwrap();
    let base_physical = base.witness_layout(None).unwrap();
    assert_eq!(physical, base_physical);
    let z_sum = physical
        .chunk_lengths
        .iter()
        .map(|lengths| lengths.z_len)
        .sum::<usize>();
    assert_eq!(z_sum, physical.chunk_lengths[0].z_len * 4);
    let base_physical_coeffs = base_physical.ring_len().unwrap() * lp.ring_dimension;
    assert_eq!(
        physical.ring_len().unwrap() * lp.ring_dimension,
        base_physical_coeffs
    );
    let unpadded = base_physical_coeffs + layout.compression_witness_coeffs;
    let expected = unpadded.div_ceil(lp.ring_dimension) * lp.ring_dimension;
    assert_eq!(layout.physical_witness_field_coeff_len().unwrap(), expected);
    let first_compression = layout
        .segments()
        .iter()
        .find(|segment| matches!(segment.id(), RelationSegmentId::CompressionInput { .. }))
        .unwrap();
    assert_eq!(
        layout
            .physical_compression_segment_span(first_compression.id())
            .unwrap()
            .start(),
        base_physical_coeffs
    );
}
