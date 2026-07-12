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
    alphabets: [CompressionAlphabet; 2],
) -> CompressionChainSpec {
    let mut previous = source_key.row_len() * source_key.sis_table_key().ring_dimension as usize;
    let maps = alphabets
        .into_iter()
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
            let key = certified_key(
                d,
                if alphabet == CompressionAlphabet::NegativeBinary {
                    1
                } else {
                    63
                },
                input / d,
            );
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
    let quotient_coeffs = layout.row_plan.trace_row
        * compute_num_digits_full_field(F::modulus_bits(), lp.log_basis)
        * d;
    assert_eq!(
        layout.total_coeffs,
        (lens.z_len + lens.e_len + lens.t_len) * d + quotient_coeffs
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
            [
                CompressionAlphabet::NegativeBinary,
                CompressionAlphabet::NegativeBinary,
            ],
        ),
        chain(
            CompressionSourceId::Opening,
            &lp.d_key,
            [
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
}
