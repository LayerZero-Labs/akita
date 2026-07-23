use super::inner_ajtai::inner_ajtai_wide_onehot;
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
    column_sweep_core_budgeted::<E, F, D>(
        a_view,
        blocks,
        n_a,
        active_a_cols,
        num_digits_inner,
        L2_TILE_BUDGET,
    )
}

/// [`column_sweep_core`] with an explicit accumulator-tile budget; split out
/// so the (test-only) sweep benchmarks can compare tile sizes.
fn column_sweep_core_budgeted<E, F, const D: usize>(
    a_view: &RingMatrixView<'_, F, D>,
    blocks: &[&[E]],
    n_a: usize,
    active_a_cols: usize,
    num_digits_inner: usize,
    tile_budget: usize,
) -> Vec<Vec<CyclotomicRing<F, D>>>
where
    E: OneHotEntry,
    F: FieldCore + CanonicalField + HasWide,
    F::Wide: AdditiveGroup + From<F> + ReduceTo<F>,
{
    let num_live_blocks = blocks.len();
    // Row-pass accumulation: one wide ring per block is live at a time, so
    // the tile is sized by a single row's accumulators. This lets a thread's
    // whole block range fit one tile at trace-scale shapes, which is what
    // bounds how often the A matrix is re-streamed.
    let accum_bytes = D * std::mem::size_of::<F::Wide>();
    let block_tile = tile_budget
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

                for slot in &mut result[tile_start..tile_end] {
                    *slot = Vec::with_capacity(n_a);
                }
                let mut row_accums: Vec<WideCyclotomicRing<F::Wide, D>> =
                    vec![WideCyclotomicRing::zero(); tile_len];

                {
                    let _span = tracing::info_span!("onehot_column_bucket_sweep").entered();
                    for a_row in a_view.rows().take(n_a) {
                        for accum in &mut row_accums {
                            *accum = WideCyclotomicRing::zero();
                        }
                        for col in 0..active_a_cols {
                            let start = col_offsets[col];
                            let end = col_offsets[col + 1];
                            if start == end {
                                continue;
                            }
                            let a_wide = WideCyclotomicRing::from_ring(&a_row[col]);
                            for &entry in &packed_entries[start..end] {
                                let (local_block, coefficient) = unpack_col_entry(entry);
                                a_wide
                                    .shift_accumulate_into(&mut row_accums[local_block], coefficient);
                            }
                        }
                        for (local_b, accum) in row_accums.iter().enumerate() {
                            result[tile_start + local_b].push(accum.reduce());
                        }
                    }
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

/// The pre-row-pass sweep structure (per-block accumulators for all `n_a`
/// rows live simultaneously; rows outer, tiles sized by `n_a` rings). Kept
/// test-only for the sweep benchmarks to compare structures.
#[cfg(test)]
pub(crate) fn column_sweep_core_row_outer_budgeted<E, F, const D: usize>(
    a_view: &RingMatrixView<'_, F, D>,
    blocks: &[&[E]],
    n_a: usize,
    active_a_cols: usize,
    num_digits_inner: usize,
    tile_budget: usize,
) -> Vec<Vec<CyclotomicRing<F, D>>>
where
    E: OneHotEntry,
    F: FieldCore + CanonicalField + HasWide,
    F::Wide: AdditiveGroup + From<F> + ReduceTo<F>,
{
    let num_live_blocks = blocks.len();
    let accum_bytes = n_a * D * std::mem::size_of::<F::Wide>();
    let block_tile = tile_budget
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
            let mut col_counts = vec![0usize; active_a_cols];
            let mut col_offsets = vec![0usize; active_a_cols + 1];
            let mut write_offsets = vec![0usize; active_a_cols];
            let mut packed_entries: Vec<PackedColEntry> = Vec::new();

            for tile_start in (0..my_count).step_by(block_tile) {
                let tile_end = (tile_start + block_tile).min(my_count);
                let tile_len = tile_end - tile_start;
                col_counts.fill(0);
                let mut entry_count = 0usize;
                for local_b in 0..tile_len {
                    for entry in blocks[block_start + tile_start + local_b] {
                        let col = entry.commit_col(num_digits_inner);
                        col_counts[col] += entry.coeffs().len();
                        entry_count += entry.coeffs().len();
                    }
                }
                col_offsets[0] = 0;
                for col in 0..active_a_cols {
                    col_offsets[col + 1] = col_offsets[col] + col_counts[col];
                }
                write_offsets.copy_from_slice(&col_offsets[..active_a_cols]);
                packed_entries.resize(entry_count, 0);
                for local_b in 0..tile_len {
                    for entry in blocks[block_start + tile_start + local_b] {
                        let col = entry.commit_col(num_digits_inner);
                        for &coefficient in entry.coeffs() {
                            packed_entries[write_offsets[col]] = pack_col_entry(local_b, coefficient);
                            write_offsets[col] += 1;
                        }
                    }
                }
                let mut accums: Vec<Vec<WideCyclotomicRing<F::Wide, D>>> = (0..tile_len)
                    .map(|_| vec![WideCyclotomicRing::zero(); n_a])
                    .collect();
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
                            a_wide.shift_accumulate_into(&mut accums[local_block][a_idx], coefficient);
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

    let mut out: Vec<Vec<CyclotomicRing<F, D>>> = Vec::with_capacity(num_live_blocks);
    for thread_blocks in thread_results {
        out.extend(thread_blocks);
    }
    out
}

#[cfg(test)]
pub(crate) fn sweep_bench_entry<E, F, const D: usize>(
    variant: &str,
    a_view: &RingMatrixView<'_, F, D>,
    blocks: &[&[E]],
    n_a: usize,
    active_a_cols: usize,
    num_digits_inner: usize,
    tile_budget: usize,
) -> Vec<Vec<CyclotomicRing<F, D>>>
where
    E: OneHotEntry,
    F: FieldCore + CanonicalField + HasWide,
    F::Wide: AdditiveGroup + From<F> + ReduceTo<F>,
{
    match variant {
        "row_pass" => column_sweep_core_budgeted::<E, F, D>(
            a_view, blocks, n_a, active_a_cols, num_digits_inner, tile_budget,
        ),
        "row_outer" => column_sweep_core_row_outer_budgeted::<E, F, D>(
            a_view, blocks, n_a, active_a_cols, num_digits_inner, tile_budget,
        ),
        other => panic!("unknown sweep variant {other}"),
    }
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

    if blocks
        .iter()
        .any(|entries| shift_accumulation_count(entries) > MAX_WIDE_SHIFT_ACCUMULATIONS)
    {
        // Oversized blocks are split into segments that each respect the wide
        // accumulators' headroom, swept through the tiled kernel as
        // independent sub-blocks, and re-merged by parent block. This keeps
        // the bucketed, A-sequential sweep at any block size; the previous
        // per-block fallback walked entries in position order and re-streamed
        // `n_a` A rings per hot coefficient, which dominated trace-scale
        // commits (~2^18 hot coefficients per block at 2^26 cycles).
        let mut sub_blocks: Vec<&[E]> = Vec::new();
        let mut parents: Vec<usize> = Vec::new();
        for (parent, entries) in blocks.iter().enumerate() {
            let mut rest: &[E] = entries;
            loop {
                let mut take = 0usize;
                let mut accumulations = 0usize;
                for entry in rest {
                    let count = entry.coeffs().len();
                    if take > 0 && accumulations + count > MAX_WIDE_SHIFT_ACCUMULATIONS {
                        break;
                    }
                    accumulations += count;
                    take += 1;
                }
                let (segment, tail) = rest.split_at(take.max(1).min(rest.len()));
                sub_blocks.push(segment);
                parents.push(parent);
                if tail.is_empty() {
                    break;
                }
                rest = tail;
            }
        }
        let sub_out =
            column_sweep_core::<E, F, D>(a_view, &sub_blocks, n_a, active_a_cols, num_digits_inner);
        let mut out: Vec<Vec<CyclotomicRing<F, D>>> = vec![Vec::new(); num_live_blocks];
        for (parent, rows) in parents.into_iter().zip(sub_out) {
            if out[parent].is_empty() {
                out[parent] = rows;
            } else {
                for (dst, src) in out[parent].iter_mut().zip(rows) {
                    *dst += src;
                }
            }
        }
        return out;
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
