use super::*;

#[test]
fn merge_sweep_matches_bucketed_core_across_polys() {
    use super::super::column_sweep::{
        column_sweep_core, column_sweep_core_merge, L2_TILE_BUDGET, MERGE_COL_CHUNK,
    };
    use akita_field::unreduced::HasCommitAccum;

    type F = Prime128Offset275;
    const D: usize = 64;

    let mut rng = StdRng::seed_from_u64(0x05ee_d0a5);
    let n_a = 3;
    let num_positions_per_block = 96;
    let num_digits_inner = 1;
    let active_a_cols = num_positions_per_block * num_digits_inner;

    let a_rows: Vec<CyclotomicRing<F, D>> = (0..n_a * active_a_cols)
        .map(|_| CyclotomicRing::random(&mut rng))
        .collect();
    let a_flat = FlatMatrix::from_ring_slice(&a_rows);
    let a_view = a_flat.ring_view::<D>(n_a, active_a_cols).unwrap();

    // Three "polys" with varying block counts, sparse sorted entries, and
    // some empty blocks — the shapes the fused sweep must round-trip.
    let mut polys_buckets: Vec<Vec<Vec<SingleChunkEntry>>> = Vec::new();
    for poly in 0..3usize {
        let num_blocks = 40 + poly * 17;
        let buckets = (0..num_blocks)
            .map(|block| {
                if (block + poly) % 7 == 0 {
                    return Vec::new();
                }
                (0..num_positions_per_block)
                    .filter(|pos| (pos + block + poly) % 3 != 0)
                    .map(|pos| SingleChunkEntry::new(pos as u32, ((pos * 11 + block) % D) as u16))
                    .collect()
            })
            .collect::<Vec<_>>();
        polys_buckets.push(buckets);
    }
    let polys_blocks: Vec<FlatBlocks<SingleChunkEntry>> = polys_buckets
        .iter()
        .map(|buckets| super::super::test_helpers::from_buckets(buckets.clone()))
        .collect();
    let polys_views: Vec<Vec<&[SingleChunkEntry]>> = polys_blocks
        .iter()
        .map(|blocks| {
            (0..blocks.num_live_blocks())
                .map(|i| blocks.block(i))
                .collect()
        })
        .collect();

    // Direct core-vs-core equality over the concatenated batch.
    let flat: Vec<&[SingleChunkEntry]> = polys_views.iter().flatten().copied().collect();
    let merge = column_sweep_core_merge::<SingleChunkEntry, F, D>(
        &a_view,
        &flat,
        n_a,
        active_a_cols,
        num_digits_inner,
        L2_TILE_BUDGET,
        MERGE_COL_CHUNK,
    );
    let bucketed = column_sweep_core::<SingleChunkEntry, F, D>(
        &a_view,
        &flat,
        n_a,
        active_a_cols,
        num_digits_inner,
    );
    assert_eq!(merge, bucketed, "merge sweep must match the bucketed core");

    // Wrapper equality: fused multi output must equal per-poly sweeps.
    let multi = column_sweep_ajtai_onehot_multi::<SingleChunkEntry, F, D>(
        &a_view,
        &polys_views,
        n_a,
        active_a_cols,
        num_digits_inner,
    );
    let per_poly: Vec<Vec<Vec<CyclotomicRing<F, D>>>> = polys_views
        .iter()
        .map(|views| {
            column_sweep_ajtai_onehot::<SingleChunkEntry, F, D>(
                &a_view,
                views,
                n_a,
                active_a_cols,
                num_digits_inner,
            )
        })
        .collect();
    assert_eq!(multi, per_poly, "fused multi must match per-poly sweeps");

    // Tiny tiles force multi-tile merge paths and cursor resets.
    let merge_tiny_tiles = column_sweep_core_merge::<SingleChunkEntry, F, D>(
        &a_view,
        &flat,
        n_a,
        active_a_cols,
        num_digits_inner,
        3 * D * std::mem::size_of::<<F as HasCommitAccum>::CommitAccum>(),
        5,
    );
    assert_eq!(
        merge_tiny_tiles, bucketed,
        "merge sweep must be tile-size independent"
    );
}

#[test]
fn lazy_multi_sweep_matches_eager_multi_sweep() {
    use super::super::column_sweep::{
        column_sweep_ajtai_onehot_multi, column_sweep_ajtai_onehot_multi_lazy,
    };
    use crate::compute::OneHotCommitBlocks;
    use rand::Rng;

    type F = Prime128Offset275;
    const D: usize = 64;
    const K: usize = 256;

    let mut rng = StdRng::seed_from_u64(0x1a2b_3c4d);
    let n_a = 2;
    let num_positions_per_block = 8;
    let num_vars = 14; // 2^14 field slots -> 256 rings -> 32 blocks
    let num_chunks = (1usize << num_vars) / K;

    let polys: Vec<OneHotPoly<F, u8>> = (0..3)
        .map(|_| {
            let indices: Vec<Option<u8>> = (0..num_chunks)
                .map(|_| (rng.gen::<u8>() % 4 != 0).then(|| rng.gen::<u8>()))
                .collect();
            OneHotPoly::new(K, D, indices).unwrap()
        })
        .collect();

    let active_a_cols = num_positions_per_block; // num_digits_inner = 1
    let a_rows: Vec<CyclotomicRing<F, D>> = (0..n_a * active_a_cols)
        .map(|_| CyclotomicRing::random(&mut rng))
        .collect();
    let a_flat = FlatMatrix::from_ring_slice(&a_rows);
    let a_view = a_flat.ring_view::<D>(n_a, active_a_cols).unwrap();

    let eager_blocks: Vec<_> = polys
        .iter()
        .map(|poly| poly.blocks_for(D, num_positions_per_block).unwrap())
        .collect();
    let eager_slices: Vec<Vec<&[SingleChunkEntry]>> = eager_blocks
        .iter()
        .map(|blocks| match blocks.as_ref() {
            OneHotBlocks::SingleChunk(blocks) => (0..blocks.num_live_blocks())
                .map(|i| blocks.block(i))
                .collect(),
            OneHotBlocks::MultiChunk(_) => panic!("K=256 D=64 is single-chunk"),
        })
        .collect();
    let eager = column_sweep_ajtai_onehot_multi::<SingleChunkEntry, F, D>(
        &a_view,
        &eager_slices,
        n_a,
        active_a_cols,
        1,
    );

    let lazy_plans: Vec<_> = polys
        .iter()
        .map(|poly| {
            poly.commit_plan_blocks_lazy(D, num_positions_per_block)
                .unwrap()
        })
        .collect();
    let sources: Vec<&LazyOneHotBlocks<'_, SingleChunkEntry>> = lazy_plans
        .iter()
        .map(|blocks| match blocks {
            OneHotCommitBlocks::SingleChunkLazy(source) => source,
            _ => panic!("K=256 D=64 is single-chunk lazy"),
        })
        .collect();
    let lazy = column_sweep_ajtai_onehot_multi_lazy::<SingleChunkEntry, F, D>(
        &a_view,
        &sources,
        n_a,
        active_a_cols,
        1,
    )
    .unwrap();

    assert_eq!(eager, lazy);
}

#[test]
fn merge_sweep_self_reduces_oversized_blocks() {
    use super::super::column_sweep::{column_sweep_core_merge, L2_TILE_BUDGET, MERGE_COL_CHUNK};

    type F = Prime128Offset275;
    const D: usize = 64;

    let mut rng = StdRng::seed_from_u64(0x05e2_517e);
    let n_a = 2;
    let num_positions_per_block = MAX_WIDE_SHIFT_ACCUMULATIONS + 129;
    let active_a_cols = num_positions_per_block;

    let a_rows: Vec<CyclotomicRing<F, D>> = (0..n_a * active_a_cols)
        .map(|_| CyclotomicRing::random(&mut rng))
        .collect();
    let a_flat = FlatMatrix::from_ring_slice(&a_rows);
    let a_view = a_flat.ring_view::<D>(n_a, active_a_cols).unwrap();

    // One oversized dense block (exceeds the wide-accumulator cap) plus a
    // small one; the merge kernel must self-reduce mid-row instead of
    // relying on the block-splitting wrapper.
    let big: Vec<SingleChunkEntry> = (0..num_positions_per_block)
        .map(|pos| SingleChunkEntry::new(pos as u32, ((pos * 7) % D) as u16))
        .collect();
    let small: Vec<SingleChunkEntry> = (0..97)
        .map(|pos| SingleChunkEntry::new((pos * 11) as u32, (pos % D) as u16))
        .collect();
    let blocks = super::super::test_helpers::from_buckets(vec![big, small]);
    let views: Vec<&[SingleChunkEntry]> = (0..blocks.num_live_blocks())
        .map(|i| blocks.block(i))
        .collect();

    let merge = column_sweep_core_merge::<SingleChunkEntry, F, D>(
        &a_view,
        &views,
        n_a,
        active_a_cols,
        1,
        L2_TILE_BUDGET,
        MERGE_COL_CHUNK,
    );
    // Reference: the splitting wrapper (overflow-safe by segmentation).
    let wrapper =
        column_sweep_ajtai_onehot::<SingleChunkEntry, F, D>(&a_view, &views, n_a, active_a_cols, 1);
    assert_eq!(
        merge, wrapper,
        "self-reducing merge must match the splitting wrapper"
    );
}
