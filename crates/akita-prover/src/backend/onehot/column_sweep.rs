use super::inner_ajtai::{
    inner_ajtai_wide_multi_chunk, inner_ajtai_wide_single_chunk,
    inner_ajtai_wide_single_chunk_tiled,
};
use super::*;

/// L2 cache budget (in bytes) for the tile of wide accumulators in the
/// column-sweep commit.  Each tile's `accums` allocation is capped to this
/// size so the scatter loop stays L2-resident.
///
/// 2 MB is a conservative middle ground: fits in Apple M-series L2
/// (~4 MB/core) and exceeds most x86 per-core L2 (~256 KB–1 MB) only
/// modestly, relying on the shared L3 backstop.
const L2_TILE_BUDGET: usize = 1 << 21;

/// Minimum blocks-per-thread required before enabling the column-sweep kernel.
const SWEEP_THRESHOLD: usize = 32;

/// One tile-local hot entry: `(a-column, local-block-index, coefficient-index)`.
///
/// All entries from one L2 tile are bucketed into this flat vector so the
/// outer loop can load each A-column exactly once, then scatter the column's
/// contribution into every block whose entry lands in that column.
type ColEntry = (usize, u32, u16);

/// Inner two-level-tiled column-sweep, shared between the regular and sparse
/// wrappers.
///
/// Threads partition blocks evenly (outer, for parallelism); within each
/// thread, blocks are processed in L2-sized tiles (inner, for cache
/// locality). For each tile, `push_entries` writes one `(col, local_b,
/// coeff_idx)` tuple per hot contribution; sort-by-col then drives a single
/// sweep per A row.
#[inline]
fn column_sweep_core<E, F, const D: usize>(
    a_view: &akita_types::RingMatrixView<'_, F, D>,
    blocks: &[&[E]],
    n_a: usize,
    num_digits_commit: usize,
    push_entries: impl Fn(&[E], u32, usize, &mut Vec<ColEntry>) + Send + Sync + Copy,
) -> Vec<Vec<CyclotomicRing<F, D>>>
where
    E: Sync,
    F: FieldCore + CanonicalField + HasWide,
    F::Wide: AdditiveGroup + From<F> + akita_field::unreduced::ReduceTo<F>,
{
    let num_blocks = blocks.len();
    let accum_bytes = n_a * D * std::mem::size_of::<F::Wide>();
    let block_tile = L2_TILE_BUDGET
        .checked_div(accum_bytes)
        .map_or(num_blocks, |tile| tile.max(1));

    #[cfg(feature = "parallel")]
    let num_threads = rayon::current_num_threads().min(num_blocks).max(1);
    #[cfg(not(feature = "parallel"))]
    let num_threads = 1;

    let blocks_per_thread = num_blocks.div_ceil(num_threads);

    let thread_results: Vec<Vec<Vec<CyclotomicRing<F, D>>>> = cfg_into_iter!(0..num_threads)
        .map(|tid| {
            let block_start = tid * blocks_per_thread;
            let block_end = (block_start + blocks_per_thread).min(num_blocks);
            if block_start >= block_end {
                return Vec::new();
            }
            let my_count = block_end - block_start;

            let mut result: Vec<Vec<CyclotomicRing<F, D>>> = Vec::with_capacity(my_count);
            result.resize_with(my_count, Vec::new);

            // Reuse across tiles so earlier capacity carries over, but only
            // allocate buckets for columns that are actually touched.
            let mut col_entries: Vec<ColEntry> = Vec::new();

            for tile_start in (0..my_count).step_by(block_tile) {
                let tile_end = (tile_start + block_tile).min(my_count);
                let tile_len = tile_end - tile_start;

                col_entries.clear();
                for local_b in 0..tile_len {
                    let block_entries = blocks[block_start + tile_start + local_b];
                    push_entries(
                        block_entries,
                        local_b as u32,
                        num_digits_commit,
                        &mut col_entries,
                    );
                }
                col_entries.sort_unstable_by_key(|&(col, _, _)| col);

                let mut accums: Vec<Vec<WideCyclotomicRing<F::Wide, D>>> = (0..tile_len)
                    .map(|_| vec![WideCyclotomicRing::zero(); n_a])
                    .collect();

                for (a_idx, a_row) in a_view.rows().enumerate().take(n_a) {
                    let mut idx = 0usize;
                    while idx < col_entries.len() {
                        let col = col_entries[idx].0;
                        let a_wide = WideCyclotomicRing::from_ring(&a_row[col]);
                        while idx < col_entries.len() && col_entries[idx].0 == col {
                            let (_, lb, ci) = col_entries[idx];
                            a_wide.shift_accumulate_into(
                                &mut accums[lb as usize][a_idx],
                                ci as usize,
                            );
                            idx += 1;
                        }
                    }
                }

                for (local_b, row_accums) in accums.into_iter().enumerate() {
                    result[tile_start + local_b] =
                        row_accums.into_iter().map(|w| w.reduce()).collect();
                }
            }

            result
        })
        .collect();

    let mut out: Vec<Vec<CyclotomicRing<F, D>>> = Vec::with_capacity(num_blocks);
    for thread_blocks in thread_results {
        out.extend(thread_blocks);
    }
    out
}

/// Column-sweep Ajtai commitment for single-chunk one-hot blocks.
///
/// Uses [`column_sweep_core`] for the tiled sweep plus a safety fallback when
/// any block has more than `MAX_WIDE_SHIFT_ACCUMULATIONS` hot entries (the
/// wide accumulator would overflow) and a small-block fast path when
/// `blocks_per_thread` is already L2-friendly.
pub(crate) fn column_sweep_ajtai_single_chunk<F, const D: usize>(
    a_view: &akita_types::RingMatrixView<'_, F, D>,
    single_chunk_blocks: &[&[SingleChunkEntry]],
    n_a: usize,
    active_a_cols: usize,
    num_digits_commit: usize,
) -> Vec<Vec<CyclotomicRing<F, D>>>
where
    F: FieldCore + CanonicalField + HasWide,
    F::Wide: AdditiveGroup + From<F> + akita_field::unreduced::ReduceTo<F>,
{
    let num_blocks = single_chunk_blocks.len();
    debug_assert!(
        active_a_cols <= a_view.num_cols(),
        "active A width exceeds setup envelope"
    );
    if single_chunk_blocks
        .iter()
        .any(|entries| entries.len() > MAX_WIDE_SHIFT_ACCUMULATIONS)
    {
        return cfg_into_iter!(0..num_blocks)
            .map(|i| {
                inner_ajtai_wide_single_chunk_tiled(
                    a_view,
                    single_chunk_blocks[i],
                    num_digits_commit,
                )
            })
            .collect();
    }

    #[cfg(feature = "parallel")]
    let num_threads = rayon::current_num_threads().min(num_blocks).max(1);
    #[cfg(not(feature = "parallel"))]
    let num_threads = 1;
    let blocks_per_thread = num_blocks.div_ceil(num_threads);

    if blocks_per_thread <= SWEEP_THRESHOLD {
        return cfg_into_iter!(0..num_blocks)
            .map(|i| {
                inner_ajtai_wide_single_chunk(a_view, single_chunk_blocks[i], num_digits_commit)
            })
            .collect();
    }

    column_sweep_core::<SingleChunkEntry, F, D>(
        a_view,
        single_chunk_blocks,
        n_a,
        num_digits_commit,
        |block_entries, local_b, num_digits, sink| {
            for entry in block_entries {
                let col = entry.pos_in_block() * num_digits;
                sink.push((col, local_b, entry.coeff_idx() as u16));
            }
        },
    )
}

/// Column-sweep Ajtai commitment for multi-chunk one-hot blocks.
///
/// Same two-level tiling as [`column_sweep_ajtai_single_chunk`]; each hot
/// ring element may contribute multiple coefficients, so `push_entries`
/// fans out the `nonzero_coeffs` list into individual `ColEntry` tuples.
///
/// Like the single-chunk twin, this falls back to the per-block inner kernel
/// whenever any block's total shift-accumulate count would overflow the
/// column-sweep wide accumulator. For the multi-chunk layout each entry
/// contributes `nonzero_coeffs.len()` shift-accumulates (not `1` like the
/// single-chunk case), so the overflow threshold is reached at smaller block
/// sizes when `K << D`.
pub(crate) fn column_sweep_ajtai_multi_chunk<F, const D: usize>(
    a_view: &akita_types::RingMatrixView<'_, F, D>,
    multi_chunk_blocks: &[&[MultiChunkEntry]],
    n_a: usize,
    active_a_cols: usize,
    num_digits_commit: usize,
) -> Vec<Vec<CyclotomicRing<F, D>>>
where
    F: FieldCore + CanonicalField + HasWide,
    F::Wide: AdditiveGroup + From<F> + akita_field::unreduced::ReduceTo<F>,
{
    let num_blocks = multi_chunk_blocks.len();
    debug_assert!(
        active_a_cols <= a_view.num_cols(),
        "active A width exceeds setup envelope"
    );

    #[cfg(feature = "parallel")]
    let num_threads = rayon::current_num_threads().min(num_blocks).max(1);
    #[cfg(not(feature = "parallel"))]
    let num_threads = 1;
    let blocks_per_thread = num_blocks.div_ceil(num_threads);

    if blocks_per_thread <= SWEEP_THRESHOLD {
        return cfg_into_iter!(0..num_blocks)
            .map(|i| inner_ajtai_wide_multi_chunk(a_view, multi_chunk_blocks[i], num_digits_commit))
            .collect();
    }

    if multi_chunk_blocks.iter().any(|entries| {
        entries
            .iter()
            .map(|e| e.nonzero_coeffs().len())
            .sum::<usize>()
            > MAX_WIDE_SHIFT_ACCUMULATIONS
    }) {
        return cfg_into_iter!(0..num_blocks)
            .map(|i| inner_ajtai_wide_multi_chunk(a_view, multi_chunk_blocks[i], num_digits_commit))
            .collect();
    }

    column_sweep_core::<MultiChunkEntry, F, D>(
        a_view,
        multi_chunk_blocks,
        n_a,
        num_digits_commit,
        |block_entries, local_b, num_digits, sink| {
            for entry in block_entries {
                let col = entry.pos_in_block() * num_digits;
                for &ci in entry.nonzero_coeffs() {
                    sink.push((col, local_b, ci));
                }
            }
        },
    )
}
