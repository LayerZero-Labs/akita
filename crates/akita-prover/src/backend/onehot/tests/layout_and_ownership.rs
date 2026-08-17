use super::*;

// Flat-storage mapping, validated views, and operation-local ownership.

#[test]
fn map_onehot_k_gt_d() {
    type F = Prime24Offset3;
    // K=16, D=4, T=2 chunks => 32 field elements => 8 ring elements
    // num_positions_per_block=4 => 2 blocks of 4 ring elements each.
    let k = 16;
    let d = 4;
    let indices: Vec<Option<usize>> = vec![Some(3), Some(10)];
    let num_live_blocks = 2;
    let poly = OneHotPoly::<F>::new(k, indices).unwrap();
    let blocks = poly
        .materialize_block_range(d, 4, 0..num_live_blocks)
        .unwrap();

    assert_eq!(blocks.num_live_blocks(), 2);
    let total_entries: usize = (0..blocks.num_live_blocks())
        .map(|i| blocks.block(i).len())
        .sum();
    assert_eq!(total_entries, 2, "T=2 nonzero ring elements");

    let block0 = blocks.block(0);
    assert_eq!(block0.len(), 1);
    assert_eq!(block0[0].pos_in_block(), 0);
    assert_eq!(block0[0].coeff_idx(), 3);

    let block1 = blocks.block(1);
    assert_eq!(block1.len(), 1);
    assert_eq!(block1[0].pos_in_block(), 2);
    assert_eq!(block1[0].coeff_idx(), 2);
}

#[test]
fn map_onehot_k_eq_d() {
    type F = Prime24Offset3;
    // K=4, D=4, T=4 chunks => 16 field elements => 4 ring elements
    // num_positions_per_block=2 => 2 blocks of 2 ring elements each.
    let k = 4;
    let d = 4;
    let indices: Vec<Option<usize>> = vec![Some(0), Some(2), Some(3), Some(1)];
    let num_live_blocks = 2;
    let poly = OneHotPoly::<F>::new(k, indices).unwrap();
    let blocks = poly
        .materialize_block_range(d, 2, 0..num_live_blocks)
        .unwrap();

    assert_eq!(blocks.num_live_blocks(), 2);
    let total_entries: usize = (0..blocks.num_live_blocks())
        .map(|i| blocks.block(i).len())
        .sum();
    assert_eq!(total_entries, 4, "K=D => every ring element is nonzero");

    let block0 = blocks.block(0);
    assert_eq!(block0.len(), 2);
    assert_eq!(block0[0].pos_in_block(), 0);
    assert_eq!(block0[0].coeff_idx(), 0);
    assert_eq!(block0[1].pos_in_block(), 1);
    assert_eq!(block0[1].coeff_idx(), 2);

    let block1 = blocks.block(1);
    assert_eq!(block1.len(), 2);
    assert_eq!(block1[0].pos_in_block(), 0);
    assert_eq!(block1[0].coeff_idx(), 3);
    assert_eq!(block1[1].pos_in_block(), 1);
    assert_eq!(block1[1].coeff_idx(), 1);
}

#[test]
fn map_onehot_k_lt_d() {
    type F = Prime24Offset3;
    // K=4, D=8, T=8 chunks => 32 field elements => 4 ring elements
    // num_positions_per_block=2 => 2 blocks of 2 ring elements each.
    let k = 4;
    let d = 8;
    let indices: Vec<Option<usize>> = vec![
        Some(0),
        Some(2),
        Some(3),
        Some(1),
        Some(0),
        Some(0),
        Some(3),
        Some(3),
    ];
    let num_live_blocks = 2;
    let poly = OneHotPoly::<F>::new(k, indices).unwrap();
    let blocks = poly
        .materialize_block_range(d, 2, 0..num_live_blocks)
        .unwrap();

    assert_eq!(blocks.num_live_blocks(), 2);
    let total_entries: usize = (0..blocks.num_live_blocks())
        .map(|i| blocks.block(i).len())
        .sum();
    assert_eq!(total_entries, 8, "one entry is emitted for each hot chunk");

    let block0 = blocks.block(0);
    assert_eq!(block0.len(), 4);
    assert_eq!(block0[0].pos_in_block(), 0);
    assert_eq!(block0[0].coeff_idx(), 0);
    assert_eq!(block0[1].pos_in_block(), 0);
    assert_eq!(block0[1].coeff_idx(), 6);
    assert_eq!(block0[2].pos_in_block(), 1);
    assert_eq!(block0[2].coeff_idx(), 3);
    assert_eq!(block0[3].pos_in_block(), 1);
    assert_eq!(block0[3].coeff_idx(), 5);

    let block1 = blocks.block(1);
    assert_eq!(block1.len(), 4);
    assert_eq!(block1[0].pos_in_block(), 0);
    assert_eq!(block1[0].coeff_idx(), 0);
    assert_eq!(block1[1].pos_in_block(), 0);
    assert_eq!(block1[1].coeff_idx(), 4);
    assert_eq!(block1[2].pos_in_block(), 1);
    assert_eq!(block1[2].coeff_idx(), 3);
    assert_eq!(block1[3].pos_in_block(), 1);
    assert_eq!(block1[3].coeff_idx(), 7);
}

#[test]
fn ranged_mapping_matches_full_mapping_for_both_dimension_orders() {
    fn check(k: usize, d: usize, indices: Vec<Option<usize>>) {
        type F = Prime24Offset3;
        let num_positions_per_block = 2;
        let num_rings = indices.len() * k / d;
        let num_blocks = num_rings.div_ceil(num_positions_per_block);
        let poly = OneHotPoly::<F>::new(k, indices).unwrap();
        let full = poly
            .materialize_block_range(d, num_positions_per_block, 0..num_blocks)
            .unwrap();

        for block_idx in 0..num_blocks {
            let ranged = poly
                .materialize_block_range(d, num_positions_per_block, block_idx..block_idx + 1)
                .unwrap();
            assert_eq!(ranged.num_live_blocks(), 1);
            assert_eq!(ranged.block(0), full.block(block_idx));
            assert!(ranged.block(0).iter().all(|entry| entry.value() == 1));
        }
    }

    check(16, 4, vec![Some(3), None, Some(10), Some(15)]);
    check(
        4,
        8,
        vec![
            Some(0),
            Some(2),
            None,
            Some(1),
            Some(3),
            Some(0),
            None,
            Some(2),
        ],
    );
}

#[test]
fn empty_final_block_range_is_accepted() {
    type F = Prime24Offset3;
    let poly = OneHotPoly::<F>::new(4, vec![Some(0usize); 8]).unwrap();
    let num_live_blocks = poly.num_live_blocks_for(8, 8).unwrap();

    let blocks = poly
        .materialize_block_range(8, 8, num_live_blocks..num_live_blocks)
        .unwrap();

    assert_eq!(blocks.num_live_blocks(), 0);
}

#[test]
fn ordered_ring_range_beyond_storage_is_empty() {
    type F = Prime24Offset3;
    let poly = OneHotPoly::<F>::new(4, vec![Some(0usize), Some(1), None, None]).unwrap();

    let (_, coefficients) = poly.ring_range_coefficients(4, 10..10).unwrap();

    assert_eq!(coefficients.count(), 0);
}

#[test]
#[should_panic(expected = "FlatBlocks::block: block index 1 out of range for 1 blocks")]
fn flat_blocks_block_panics_on_out_of_range_index() {
    let blocks = super::test_helpers::from_buckets(vec![vec![1u16]]);
    let _ = blocks.block(1);
}

#[test]
fn onehot_view_validates_runtime_dimension_and_exposes_semantics() {
    type F = Prime24Offset3;
    const D: usize = 16;
    const BAD_D: usize = 12;
    let poly = OneHotPoly::<F>::new(
        8,
        vec![
            Some(0usize),
            Some(7),
            None,
            Some(3),
            Some(5),
            Some(1),
            None,
            Some(6),
        ],
    )
    .unwrap();

    let view = RootCommitSource::<F, D>::commit_view(&poly).unwrap();
    assert_eq!(view.indices(), poly.indices());
    assert_eq!(view.onehot_k(), poly.onehot_k());
    assert_eq!(view.num_vars(), poly.num_vars);
    let polys = [&poly];
    let batch = RootOpeningSource::<F, D>::opening_batch(&polys).unwrap();
    assert_eq!(
        batch
            .views()
            .map(|view| view.num_vars())
            .collect::<Vec<_>>(),
        vec![poly.num_vars]
    );

    assert!(RootCommitSource::<F, BAD_D>::commit_view(&poly).is_err());
    assert!(RootOpeningSource::<F, BAD_D>::opening_view(&poly).is_err());
    assert!(RootOpeningSource::<F, BAD_D>::opening_batch(&[&poly]).is_err());
}

#[test]
fn onehot_poly_materializes_multiple_runtime_layouts() {
    type F = Prime24Offset3;
    let poly = OneHotPoly::<F>::new(
        64,
        vec![
            Some(0usize),
            Some(7),
            None,
            Some(63),
            Some(3),
            None,
            Some(12),
            Some(1),
            None,
            Some(42),
            Some(9),
            None,
            Some(55),
            Some(18),
            None,
            Some(4),
        ],
    )
    .unwrap();

    for ring_d in [64, 128, 256, 512, 1024] {
        let count = poly.num_live_blocks_for(ring_d, 2).unwrap();
        let blocks = poly.materialize_block_range(ring_d, 2, 0..count).unwrap();
        assert_eq!(blocks.num_live_blocks(), count);
    }

    assert!(poly.num_live_blocks_for(96, 2).is_err());
    assert!(poly.materialize_block_range(96, 2, 0..1).is_err());
}

#[test]
fn onehot_clone_owns_semantic_indices_independently() {
    type F = Prime24Offset3;
    let poly = OneHotPoly::<F>::new(32, vec![Some(0usize), Some(7), None, Some(31)]).unwrap();
    let mut cloned = poly.clone();

    cloned.indices[0] = None;
    assert_eq!(poly.indices[0], Some(0));
    assert_eq!(cloned.indices[0], None);

    let original = poly.materialize_block_range(32, 2, 0..2).unwrap();
    let changed = cloned.materialize_block_range(32, 2, 0..2).unwrap();
    assert_ne!(original.block(0), changed.block(0));
}
