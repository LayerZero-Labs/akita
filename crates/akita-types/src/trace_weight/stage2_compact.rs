use super::build::{
    build_trace_weight_table_field_block_weights, build_trace_weight_table_field_terms,
    build_trace_weight_table_ring_terms,
};
use super::stage2::{
    build_trace_table_scaled, trace_public_weights_field_terms, trace_public_weights_ring_terms,
    trace_weight_evals_for_witness, TraceClaim,
};
use super::TraceWeightLayout;
use super::{eval_trace_terms_closed, TraceFieldBlockOpening, TraceRingBlockOpening, TraceTerm};
use crate::{
    lagrange_weights, reduce_inner_opening_to_ring_element, BasisMode, OpeningBatchWitnessGroup,
    OpeningBatchWitnessLayout, OpeningBlockLayout, SemanticGroupId,
};
use akita_algebra::CyclotomicRing;
use akita_field::{Ext2, Prime128OffsetA7F7};

type F = Prime128OffsetA7F7;
const D: usize = 8;

fn layout() -> TraceWeightLayout {
    let group_id = SemanticGroupId(0);
    let witness_layout = OpeningBatchWitnessLayout::new(
        vec![OpeningBatchWitnessGroup {
            id: group_id,
            num_claims: 1,
            live_fold_count: 2,
            fold_position_count: 1,
            depth_open: 2,
            depth_commit: 1,
            depth_fold: 1,
            n_a: 1,
            e_setup_col_offset: 0,
        }],
        vec![group_id],
        vec![group_id],
        1,
        1,
        1,
    )
    .unwrap();
    let opening_layout = OpeningBlockLayout::new(1, witness_layout.total_len()).unwrap();
    TraceWeightLayout {
        ring_bits: 3,
        col_bits: 4,
        live_fold_count: 2,
        num_digits_open: 2,
        fold_bits: 1,
        log_basis: 3,
        witness_layout,
        opening_layout,
        group_id,
    }
}

#[test]
fn compact_trace_table_keeps_live_columns_in_witness_order() {
    let layout = layout();
    let table = (0..layout.table_len().unwrap())
        .map(|idx| F::from_u64(idx as u64))
        .collect::<Vec<_>>();
    let compact = trace_weight_evals_for_witness(&layout, &table, 3).unwrap();

    let mut expected = Vec::new();
    for col in 0..3 {
        for ring in 0..layout.ring_len() {
            expected.push(table[layout.witness_index(col, ring)]);
        }
    }
    assert_eq!(compact, expected);
}

fn test_ring(seed: u64) -> CyclotomicRing<F, D> {
    CyclotomicRing::from_coefficients(std::array::from_fn(|i| {
        F::from_u64(seed + 3 * i as u64 + 1)
    }))
}

#[test]
fn trace_table_field_sparse_matches_materialized_dense() {
    let layout = layout();
    let terms = vec![
        TraceFieldBlockOpening {
            block_offset: 0,
            block_weights: vec![F::from_u64(2), F::from_u64(5)],
            inner_opening_ring: test_ring(10),
        },
        TraceFieldBlockOpening {
            block_offset: 1,
            block_weights: vec![F::from_u64(7)],
            inner_opening_ring: test_ring(40),
        },
    ];
    let public_weights = trace_public_weights_field_terms::<F, F, D>(&terms).unwrap();
    let live_x_cols = 5;
    let dense = build_trace_table_scaled(&layout, &public_weights, live_x_cols, F::one())
        .unwrap()
        .materialize_dense(live_x_cols, layout.ring_len());
    let sparse = build_trace_table_scaled(&layout, &public_weights, live_x_cols, F::one())
        .unwrap()
        .materialize_dense(live_x_cols, layout.ring_len());
    assert_eq!(sparse, dense);
}

#[test]
fn stage2_compact_field_matches_dense_slice_for_partial_live_columns() {
    let layout = layout();
    let terms = vec![
        TraceFieldBlockOpening {
            block_offset: 0,
            block_weights: vec![F::from_u64(2), F::from_u64(5)],
            inner_opening_ring: test_ring(10),
        },
        TraceFieldBlockOpening {
            block_offset: 1,
            block_weights: vec![F::from_u64(7)],
            inner_opening_ring: test_ring(40),
        },
    ];
    let public_weights = trace_public_weights_field_terms::<F, F, D>(&terms).unwrap();
    let dense = build_trace_weight_table_field_terms::<F, F, D>(&layout, &terms).unwrap();
    let expected = trace_weight_evals_for_witness(&layout, &dense, 5).unwrap();
    let actual = build_trace_table_scaled(&layout, &public_weights, 5, F::one())
        .unwrap()
        .materialize_dense(5, layout.ring_len());

    assert_eq!(actual, expected);
}

#[test]
fn stage2_compact_ring_matches_dense_slice_for_partial_live_columns() {
    type E = Ext2<F>;

    let layout = layout();
    let terms = vec![
        TraceRingBlockOpening {
            block_offset: 0,
            block_rings: vec![test_ring(3), test_ring(17)],
            packed_inner_point: test_ring(31),
        },
        TraceRingBlockOpening {
            block_offset: 1,
            block_rings: vec![test_ring(53)],
            packed_inner_point: test_ring(71),
        },
    ];
    let public_weights = trace_public_weights_ring_terms::<F, E, D>(&terms).unwrap();
    let dense = build_trace_weight_table_ring_terms::<F, E, D>(&layout, &terms).unwrap();
    let expected = trace_weight_evals_for_witness(&layout, &dense, 5).unwrap();
    let actual = build_trace_table_scaled(&layout, &public_weights, 5, E::one())
        .unwrap()
        .materialize_dense(5, layout.ring_len());

    assert_eq!(actual, expected);
}

#[test]
fn stage2_compact_scaled_matches_scaled_dense_slice() {
    type E = Ext2<F>;

    let layout = layout();
    let terms = vec![
        TraceRingBlockOpening {
            block_offset: 0,
            block_rings: vec![test_ring(3), test_ring(17)],
            packed_inner_point: test_ring(31),
        },
        TraceRingBlockOpening {
            block_offset: 1,
            block_rings: vec![test_ring(53)],
            packed_inner_point: test_ring(71),
        },
    ];
    let public_weights = trace_public_weights_ring_terms::<F, E, D>(&terms).unwrap();
    let output_scale = E::new(F::from_u64(11), F::from_u64(19));
    let dense = build_trace_weight_table_ring_terms::<F, E, D>(&layout, &terms).unwrap();
    let expected = trace_weight_evals_for_witness(&layout, &dense, 5)
        .unwrap()
        .into_iter()
        .map(|value| output_scale * value)
        .collect::<Vec<_>>();
    let actual = build_trace_table_scaled(&layout, &public_weights, 5, output_scale)
        .unwrap()
        .materialize_dense(5, layout.ring_len());

    assert_eq!(actual, expected);
}

#[test]
fn trace_claim_eval_matches_dense_table_for_k1() {
    let layout = layout();
    let inner_open = vec![F::from_u64(3), F::from_u64(5), F::from_u64(7)];
    let b_open = vec![F::from_u64(11)];
    let block_weights = lagrange_weights(&b_open).unwrap();
    let inner_ring =
        reduce_inner_opening_to_ring_element::<F, D>(&inner_open, BasisMode::Lagrange).unwrap();
    let claim = TraceClaim {
        layout: layout.clone(),
        trace_terms: vec![TraceTerm {
            block_offset: 0,
            b_open: b_open.clone(),
            basis: BasisMode::Lagrange,
            packed_inner_point: inner_ring,
            coefficient: F::one(),
        }],
        trace_coeff: F::from_u64(13),
        trace_opening_claim: F::from_u64(17),
        trace_term_batches: Vec::new(),
        dense_evals: None,
    };

    let table = build_trace_weight_table_field_block_weights::<F, F, D>(
        &layout,
        &block_weights,
        &inner_ring,
    )
    .unwrap();
    let ring_point = vec![F::from_u64(19), F::from_u64(23), F::from_u64(29)];
    let col_point = vec![
        F::from_u64(31),
        F::from_u64(37),
        F::from_u64(41),
        F::from_u64(43),
    ];
    let dense =
        crate::trace_weight::trace_weight_mle_eval(&layout, &table, &col_point, &ring_point)
            .unwrap();
    let closed = eval_trace_terms_closed::<F, F, D>(
        &claim.layout,
        &ring_point,
        &col_point,
        &claim.trace_terms,
    )
    .unwrap();

    assert_eq!(closed, dense);
}
