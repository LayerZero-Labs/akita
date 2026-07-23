use super::inner_ajtai::{inner_ajtai_wide_onehot, inner_ajtai_wide_onehot_safe};
use super::*;

/// L2 cache budget (in bytes) for the tile of wide accumulators in the
/// column-sweep commit. Each tile's `accums` allocation is capped to this
/// size so the scatter loop stays L2-resident.
///
/// 2 MB is a conservative middle ground: fits in Apple M-series L2
/// (~4 MB/core) and exceeds most x86 per-core L2 (~256 KB–1 MB) only
/// modestly, relying on the shared L3 backstop.
const L2_TILE_BUDGET: usize = 1 << 21;

/// Minimum blocks-per-thread required before enabling the column-sweep kernel.
const SWEEP_THRESHOLD: usize = 32;

/// One tile-local hot entry packed as `(local-block-index, coefficient-index)`.
/// The A-column is represented by the counting-bucket range containing it.
type PackedColEntry = u32;

#[inline(always)]
fn pack_col_entry(local_block: usize, coefficient: u16) -> PackedColEntry {
    // `block_tile` is capped so this conversion is valid in release builds as
    // well as debug builds.
    debug_assert!(u16::try_from(local_block).is_ok());
    ((local_block as u32) << 16) | u32::from(coefficient)
}

#[inline(always)]
fn unpack_col_entry(entry: PackedColEntry) -> (usize, usize) {
    ((entry >> 16) as usize, (entry & 0xffff) as usize)
}

/// Inner two-level-tiled column-sweep, shared between the regular and sparse
/// wrappers.
///
/// Threads partition blocks evenly (outer, for parallelism); within each
/// thread, blocks are processed in L2-sized tiles (inner, for cache
/// locality). For each tile, a counting/scatter pass groups packed
/// `(local_block, coefficient)` entries by their bounded A-column key, then
/// drives one sweep per A row.
#[inline]
fn column_sweep_core<E, F, const D: usize>(
    a_view: &RingMatrixView<'_, F, D>,
    blocks: &[&[E]],
    n_a: usize,
    active_a_cols: usize,
    num_digits_inner: usize,
) -> Vec<Vec<CyclotomicRing<F, D>>>
where
    E: OneHotEntry,
    F: FieldCore + CanonicalField + HasWide,
    F::Wide: AdditiveGroup + From<F> + ReduceTo<F>,
{
    let num_live_blocks = blocks.len();
    let accum_bytes = n_a * D * std::mem::size_of::<F::Wide>();
    let block_tile = L2_TILE_BUDGET
        .checked_div(accum_bytes)
        .map_or(num_live_blocks, |tile| tile.max(1))
        .min(usize::from(u16::MAX) + 1);

    #[cfg(feature = "parallel")]
    let num_threads = rayon::current_num_threads().min(num_live_blocks).max(1);
    #[cfg(not(feature = "parallel"))]
    let num_threads = 1;

    let blocks_per_thread = num_live_blocks.div_ceil(num_threads);

    let thread_results: Vec<Vec<Vec<CyclotomicRing<F, D>>>> = cfg_into_iter!(0..num_threads)
        .map(|tid| {
            let block_start = tid * blocks_per_thread;
            let block_end = (block_start + blocks_per_thread).min(num_live_blocks);
            if block_start >= block_end {
                return Vec::new();
            }
            let my_count = block_end - block_start;

            let mut result: Vec<Vec<CyclotomicRing<F, D>>> = Vec::with_capacity(my_count);
            result.resize_with(my_count, Vec::new);

            // Reuse the bounded-column counting buckets and packed payload
            // across tiles. Comparison sorting one tuple per hot coefficient
            // is needlessly O(N log N): the column key is always in the small
            // setup range `0..active_a_cols`.
            let mut col_counts = vec![0usize; active_a_cols];
            let mut col_offsets = vec![0usize; active_a_cols + 1];
            let mut write_offsets = vec![0usize; active_a_cols];
            let mut packed_entries: Vec<PackedColEntry> = Vec::new();

            for tile_start in (0..my_count).step_by(block_tile) {
                let tile_end = (tile_start + block_tile).min(my_count);
                let tile_len = tile_end - tile_start;

                debug_assert!(tile_len <= usize::from(u16::MAX) + 1);
                col_counts.fill(0);
                let entry_count = {
                    let _span = tracing::info_span!("onehot_column_bucket_count").entered();
                    let mut entry_count = 0usize;
                    for local_b in 0..tile_len {
                        let block_entries = blocks[block_start + tile_start + local_b];
                        for entry in block_entries {
                            let col = entry.commit_col(num_digits_inner);
                            debug_assert!(col < active_a_cols);
                            let count = entry.coeffs().len();
                            col_counts[col] += count;
                            entry_count += count;
                        }
                    }
                    entry_count
                };
                col_offsets[0] = 0;
                for col in 0..active_a_cols {
                    col_offsets[col + 1] = col_offsets[col] + col_counts[col];
                }
                write_offsets.copy_from_slice(&col_offsets[..active_a_cols]);
                packed_entries.resize(entry_count, 0);
                {
                    let _span = tracing::info_span!("onehot_column_bucket_scatter").entered();
                    for local_b in 0..tile_len {
                        let block_entries = blocks[block_start + tile_start + local_b];
                        for entry in block_entries {
                            let col = entry.commit_col(num_digits_inner);
                            for &coefficient in entry.coeffs() {
                                let dst = write_offsets[col];
                                packed_entries[dst] = pack_col_entry(local_b, coefficient);
                                write_offsets[col] += 1;
                            }
                        }
                    }
                }

                // The sweep is A-row-major, so keep the corresponding block
                // accumulators contiguous. Besides matching the traversal,
                // this replaces one allocation per block with one per tile.
                let mut accums = vec![WideCyclotomicRing::<F::Wide, D>::zero(); n_a * tile_len];

                {
                    let _span = tracing::info_span!("onehot_column_bucket_sweep").entered();
                    for (a_idx, a_row) in a_view.rows().enumerate().take(n_a) {
                        for col in 0..active_a_cols {
                            let start = col_offsets[col];
                            let end = col_offsets[col + 1];
                            if start == end {
                                continue;
                            }
                            let a_wide = WideCyclotomicRing::from_ring(&a_row[col]);
                            for &entry in &packed_entries[start..end] {
                                let (local_block, coefficient) = unpack_col_entry(entry);
                                a_wide.shift_accumulate_into(
                                    &mut accums[a_idx * tile_len + local_block],
                                    coefficient,
                                );
                            }
                        }
                    }
                }

                for local_b in 0..tile_len {
                    result[tile_start + local_b] = (0..n_a)
                        .map(|a_idx| accums[a_idx * tile_len + local_b].reduce())
                        .collect();
                }
            }

            result
        })
        .collect();

    let mut out: Vec<Vec<CyclotomicRing<F, D>>> = Vec::with_capacity(num_live_blocks);
    for thread_blocks in thread_results {
        out.extend(thread_blocks);
    }
    out
}

/// Column-sweep Ajtai commitment for one-hot blocks.
///
/// Uses [`column_sweep_core`] for the tiled sweep plus a safety fallback when
/// any block would exceed [`MAX_WIDE_SHIFT_ACCUMULATIONS`] shift-adds (the
/// wide accumulator would overflow) and a small-block fast path when
/// `blocks_per_thread` is already L2-friendly.
pub(crate) fn column_sweep_ajtai_onehot<E, F, const D: usize>(
    a_view: &RingMatrixView<'_, F, D>,
    blocks: &[&[E]],
    n_a: usize,
    active_a_cols: usize,
    num_digits_inner: usize,
) -> Vec<Vec<CyclotomicRing<F, D>>>
where
    E: OneHotEntry,
    F: FieldCore + CanonicalField + HasWide,
    F::Wide: AdditiveGroup + From<F> + ReduceTo<F>,
{
    let num_live_blocks = blocks.len();
    debug_assert!(
        active_a_cols <= a_view.num_cols(),
        "active A width exceeds setup envelope"
    );
    if num_live_blocks == 0 {
        return Vec::new();
    }

    if blocks
        .iter()
        .any(|entries| shift_accumulation_count(entries) > MAX_WIDE_SHIFT_ACCUMULATIONS)
    {
        return cfg_into_iter!(0..num_live_blocks)
            .map(|i| inner_ajtai_wide_onehot_safe(a_view, blocks[i], num_digits_inner))
            .collect();
    }

    #[cfg(feature = "parallel")]
    let num_threads = rayon::current_num_threads().min(num_live_blocks).max(1);
    #[cfg(not(feature = "parallel"))]
    let num_threads = 1;
    let blocks_per_thread = num_live_blocks.div_ceil(num_threads);

    if blocks_per_thread <= SWEEP_THRESHOLD {
        return cfg_into_iter!(0..num_live_blocks)
            .map(|i| inner_ajtai_wide_onehot(a_view, blocks[i], num_digits_inner))
            .collect();
    }

    column_sweep_core::<E, F, D>(a_view, blocks, n_a, active_a_cols, num_digits_inner)
}
