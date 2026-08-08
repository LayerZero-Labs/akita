use super::inner_ajtai::inner_ajtai_wide_onehot;
use super::*;

/// L2 cache budget (in bytes) for the tile of wide accumulators in the
/// column-sweep commit.  Each tile's `accums` allocation is capped to this
/// size so the scatter loop stays L2-resident.
///
/// 2 MB is a conservative middle ground: fits in Apple M-series L2
/// (~4 MB/core) and exceeds most x86 per-core L2 (~256 KB–1 MB) only
/// modestly, relying on the shared L3 backstop.
pub(super) const L2_TILE_BUDGET: usize = 1 << 21;

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
pub(super) fn column_sweep_core<F, const D: usize>(
    a_view: &RingMatrixView<'_, F, D>,
    blocks: &[&[SparseRingBlockEntry]],
    n_a: usize,
    active_a_cols: usize,
    num_digits_inner: usize,
) -> Vec<Vec<CyclotomicRing<F, D>>>
where
    F: FieldCore + CanonicalField + HasCommitAccum,
    F::CommitAccum: AdditiveGroup + From<F> + ReduceTo<F>,
{
    column_sweep_core_budgeted::<F, D>(
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
fn column_sweep_core_budgeted<F, const D: usize>(
    a_view: &RingMatrixView<'_, F, D>,
    blocks: &[&[SparseRingBlockEntry]],
    n_a: usize,
    active_a_cols: usize,
    num_digits_inner: usize,
    tile_budget: usize,
) -> Vec<Vec<CyclotomicRing<F, D>>>
where
    F: FieldCore + CanonicalField + HasCommitAccum,
    F::CommitAccum: AdditiveGroup + From<F> + ReduceTo<F>,
{
    let num_live_blocks = blocks.len();
    // Row-pass accumulation: one wide ring per block is live at a time, so
    // the tile is sized by a single row's accumulators. This lets a thread's
    // whole block range fit one tile at trace-scale shapes, which is what
    // bounds how often the A matrix is re-streamed.
    let accum_bytes = D * std::mem::size_of::<F::CommitAccum>();
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
                            let col = entry.pos_in_block() * num_digits_inner;
                            debug_assert!(col < active_a_cols);
                            col_counts[col] += 1;
                            entry_count += 1;
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
                            let col = entry.pos_in_block() * num_digits_inner;
                            let dst = write_offsets[col];
                            packed_entries[dst] = pack_col_entry(local_b, entry.coeff_idx() as u16);
                            write_offsets[col] += 1;
                        }
                    }
                }

                for slot in &mut result[tile_start..tile_end] {
                    *slot = Vec::with_capacity(n_a);
                }
                let mut row_accums: Vec<WideCyclotomicRing<F::CommitAccum, D>> =
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
                                a_wide.shift_accumulate_into(
                                    &mut row_accums[local_block],
                                    coefficient,
                                );
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

/// Number of A columns widened together by the merge sweep. Bench-tuned:
/// the (tile, chunk) matrix is flat within ~5-30% and (64 blocks, 32 cols)
/// is its minimum at trace-like sparse shapes.
pub(super) const MERGE_COL_CHUNK: usize = 32;

/// Split blocks whose shift-accumulation count exceeds `cap` into segments
/// that each respect it, tracking each segment's parent block.
fn split_oversized_blocks<'a>(
    blocks: &[&'a [SparseRingBlockEntry]],
    cap: usize,
) -> (Vec<&'a [SparseRingBlockEntry]>, Vec<usize>) {
    let mut sub_blocks: Vec<&[SparseRingBlockEntry]> = Vec::new();
    let mut parents: Vec<usize> = Vec::new();
    for (parent, entries) in blocks.iter().enumerate() {
        for segment in entries.chunks(cap) {
            sub_blocks.push(segment);
            parents.push(parent);
        }
    }
    (sub_blocks, parents)
}

/// Merge-based fused sweep: one A pass shared by every block of every
/// polynomial in the batch.
///
/// Blocks from the whole batch share the same A matrix, and their entries are
/// sorted by position (hence by A column) by construction, so each block
/// carries a cursor and the kernel walks A columns in `MERGE_COL_CHUNK`-sized
/// chunks: widen the chunk once into an L1 scratch buffer, then advance every
/// block's cursor through its entries that fall inside the chunk. Compared to
/// [`column_sweep_core`] this replaces the counting/scatter pass (whose
/// packed-entry buffer scales with tile size) with cursor walks, and — called
/// over a multi-polynomial batch — re-streams A once per (thread, tile, row)
/// instead of once per polynomial.
#[cfg(test)]
pub(super) fn column_sweep_core_merge<F, const D: usize>(
    a_view: &RingMatrixView<'_, F, D>,
    blocks: &[&[SparseRingBlockEntry]],
    n_a: usize,
    active_a_cols: usize,
    num_digits_inner: usize,
    tile_budget: usize,
    col_chunk: usize,
) -> Vec<Vec<CyclotomicRing<F, D>>>
where
    F: FieldCore + CanonicalField + HasCommitAccum,
    F::CommitAccum: AdditiveGroup + From<F> + ReduceTo<F>,
{
    let num_live_blocks = blocks.len();
    let accum_bytes = D * std::mem::size_of::<F::CommitAccum>();
    let block_tile = tile_budget
        .checked_div(accum_bytes)
        .map_or(num_live_blocks, |tile| tile.max(1));

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
            result.resize_with(my_count, || Vec::with_capacity(n_a));

            let mut chunk_buf: Vec<WideCyclotomicRing<F::CommitAccum, D>> =
                vec![WideCyclotomicRing::zero(); col_chunk];

            for tile_start in (0..my_count).step_by(block_tile) {
                let tile_end = (tile_start + block_tile).min(my_count);
                let tile_blocks = &blocks[block_start + tile_start..block_start + tile_end];
                let tile_rows = merge_sweep_tile::<F, D>(
                    a_view,
                    tile_blocks,
                    n_a,
                    active_a_cols,
                    num_digits_inner,
                    &mut chunk_buf,
                );
                for (local_b, rows) in tile_rows.into_iter().enumerate() {
                    result[tile_start + local_b] = rows;
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

/// One L1-resident tile pass of the merge sweep: returns `n_a` commit rows
/// per tile block. Extracted so eager (pre-built slices) and lazy
/// (per-tile-materialized) drivers share the exact accumulation order.
fn merge_sweep_tile<F, const D: usize>(
    a_view: &RingMatrixView<'_, F, D>,
    tile_blocks: &[&[SparseRingBlockEntry]],
    n_a: usize,
    active_a_cols: usize,
    num_digits_inner: usize,
    chunk_buf: &mut [WideCyclotomicRing<F::CommitAccum, D>],
) -> Vec<Vec<CyclotomicRing<F, D>>>
where
    F: FieldCore + CanonicalField + HasCommitAccum,
    F::CommitAccum: AdditiveGroup + From<F> + ReduceTo<F>,
{
    let col_chunk = chunk_buf.len();
    let tile_len = tile_blocks.len();
    let mut result: Vec<Vec<CyclotomicRing<F, D>>> = Vec::with_capacity(tile_len);
    result.resize_with(tile_len, || Vec::with_capacity(n_a));
    {
        let mut row_accums: Vec<WideCyclotomicRing<F::CommitAccum, D>> =
            vec![WideCyclotomicRing::zero(); tile_len];
        // Overflow control without block splitting: fold each wide
        // accumulator into a canonical partial whenever it reaches
        // the accumulation cap (a handful of reduces per block-row).
        let mut partials: Vec<CyclotomicRing<F, D>> = vec![CyclotomicRing::zero(); tile_len];
        let mut accum_counts: Vec<usize> = vec![0usize; tile_len];
        let mut cursors: Vec<usize> = vec![0usize; tile_len];

        let _span = tracing::info_span!("onehot_merge_sweep").entered();
        for a_row in a_view.rows().take(n_a) {
            for accum in &mut row_accums {
                *accum = WideCyclotomicRing::zero();
            }
            for partial in &mut partials {
                *partial = CyclotomicRing::zero();
            }
            accum_counts.fill(0);
            cursors.fill(0);

            for chunk_start in (0..active_a_cols).step_by(col_chunk) {
                let chunk_end = (chunk_start + col_chunk).min(active_a_cols);

                // Skip widening chunks no block has entries in.
                let live = tile_blocks.iter().zip(&cursors).any(|(entries, &cur)| {
                    entries
                        .get(cur)
                        .is_some_and(|e| e.pos_in_block() * num_digits_inner < chunk_end)
                });
                if !live {
                    continue;
                }
                for (buf, col) in chunk_buf.iter_mut().zip(chunk_start..chunk_end) {
                    *buf = WideCyclotomicRing::from_ring(&a_row[col]);
                }

                for (local_b, entries) in tile_blocks.iter().enumerate() {
                    let cur = &mut cursors[local_b];
                    while let Some(entry) = entries.get(*cur) {
                        let col = entry.pos_in_block() * num_digits_inner;
                        if col >= chunk_end {
                            break;
                        }
                        debug_assert!(
                            col >= chunk_start,
                            "one-hot entries must be sorted by position within a block"
                        );
                        let a_wide = &chunk_buf[col - chunk_start];
                        if accum_counts[local_b] + 1 > F::MAX_COMMIT_ACCUMULATIONS {
                            partials[local_b] += row_accums[local_b].reduce();
                            row_accums[local_b] = WideCyclotomicRing::zero();
                            accum_counts[local_b] = 0;
                        }
                        accum_counts[local_b] += 1;
                        a_wide.shift_accumulate_into(&mut row_accums[local_b], entry.coeff_idx());
                        *cur += 1;
                    }
                }
            }

            for (local_b, accum) in row_accums.iter().enumerate() {
                let mut row = partials[local_b];
                row += accum.reduce();
                result[local_b].push(row);
            }
        }
    }
    result
}

/// Fused multi-polynomial column sweep over eager or lazy block sources.
/// Entry geometry is already fixed by `E`; each source decides whether a tile
/// is borrowed from existing storage or built for the duration of the sweep.
pub(crate) fn column_sweep_ajtai_onehot_multi<F, const D: usize>(
    a_view: &RingMatrixView<'_, F, D>,
    sources: &[&OneHotBlockSource<'_>],
    n_a: usize,
    active_a_cols: usize,
    num_digits_inner: usize,
) -> Result<Vec<Vec<Vec<CyclotomicRing<F, D>>>>, AkitaError>
where
    F: FieldCore + CanonicalField + HasCommitAccum,
    F::CommitAccum: AdditiveGroup + From<F> + ReduceTo<F>,
{
    let counts: Vec<usize> = sources.iter().map(|s| s.num_live_blocks()).collect();
    let mut starts = Vec::with_capacity(counts.len() + 1);
    let mut total = 0usize;
    for count in &counts {
        starts.push(total);
        total += count;
    }
    starts.push(total);
    if total == 0 {
        return Ok(vec![Vec::new(); sources.len()]);
    }

    #[cfg(feature = "parallel")]
    let num_threads = rayon::current_num_threads().min(total).max(1);
    #[cfg(not(feature = "parallel"))]
    let num_threads = 1;

    if total.div_ceil(num_threads) <= SWEEP_THRESHOLD {
        return sources
            .iter()
            .map(|source| {
                let materialized = source.materialize_range(0..source.num_live_blocks())?;
                let blocks = materialized.block_slices()?;
                Ok(column_sweep_ajtai_onehot::<F, D>(
                    a_view,
                    &blocks,
                    n_a,
                    active_a_cols,
                    num_digits_inner,
                ))
            })
            .collect();
    }

    let accum_bytes = D * std::mem::size_of::<F::CommitAccum>();
    let tile_budget = (accum_bytes * 64).min(L2_TILE_BUDGET);
    let block_tile = tile_budget
        .checked_div(accum_bytes)
        .map_or(total, |tile| tile.max(1));

    let blocks_per_thread = total.div_ceil(num_threads);

    let thread_results: Vec<Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError>> =
        cfg_into_iter!(0..num_threads)
            .map(|tid| {
                let block_start = tid * blocks_per_thread;
                let block_end = (block_start + blocks_per_thread).min(total);
                if block_start >= block_end {
                    return Ok(Vec::new());
                }
                let my_count = block_end - block_start;
                let mut result: Vec<Vec<CyclotomicRing<F, D>>> = Vec::with_capacity(my_count);
                result.resize_with(my_count, Vec::new);
                let mut chunk_buf: Vec<WideCyclotomicRing<F::CommitAccum, D>> =
                    vec![WideCyclotomicRing::zero(); MERGE_COL_CHUNK];

                for tile_start in (block_start..block_end).step_by(block_tile) {
                    let tile_end = (tile_start + block_tile).min(block_end);
                    // Borrow or build the overlapping range from each source.
                    // Lazy owners and their entries are dropped at tile end.
                    let mut owners = Vec::new();
                    for (src_idx, source) in sources.iter().enumerate() {
                        let src_start = starts[src_idx];
                        let src_end = starts[src_idx + 1];
                        let lo = tile_start.max(src_start);
                        let hi = tile_end.min(src_end);
                        if lo >= hi {
                            continue;
                        }
                        owners.push(source.materialize_range(lo - src_start..hi - src_start)?);
                    }
                    let block_groups = owners
                        .iter()
                        .map(|blocks| blocks.block_slices())
                        .collect::<Result<Vec<_>, _>>()?;
                    let tile_blocks: Vec<&[SparseRingBlockEntry]> =
                        block_groups.iter().flatten().copied().collect();
                    let tile_rows = merge_sweep_tile::<F, D>(
                        a_view,
                        &tile_blocks,
                        n_a,
                        active_a_cols,
                        num_digits_inner,
                        &mut chunk_buf,
                    );
                    for (local_b, rows) in tile_rows.into_iter().enumerate() {
                        result[tile_start - block_start + local_b] = rows;
                    }
                }
                Ok(result)
            })
            .collect();

    let mut flat: Vec<Vec<CyclotomicRing<F, D>>> = Vec::with_capacity(total);
    for thread_blocks in thread_results {
        flat.extend(thread_blocks?);
    }
    let mut flat = flat.into_iter();
    Ok(counts
        .iter()
        .map(|&count| flat.by_ref().take(count).collect())
        .collect())
}

/// Column-sweep Ajtai commitment for one-hot blocks.
///
/// Uses [`column_sweep_core`] for the tiled sweep plus sub-block chunking when
/// a block would exceed the commitment accumulator's addition cap.
pub(crate) fn column_sweep_ajtai_onehot<F, const D: usize>(
    a_view: &RingMatrixView<'_, F, D>,
    blocks: &[&[SparseRingBlockEntry]],
    n_a: usize,
    active_a_cols: usize,
    num_digits_inner: usize,
) -> Vec<Vec<CyclotomicRing<F, D>>>
where
    F: FieldCore + CanonicalField + HasCommitAccum,
    F::CommitAccum: AdditiveGroup + From<F> + ReduceTo<F>,
{
    let num_live_blocks = blocks.len();
    debug_assert!(
        active_a_cols <= a_view.num_cols(),
        "active A width exceeds setup capacity"
    );

    if blocks
        .iter()
        .any(|entries| entries.len() > F::MAX_COMMIT_ACCUMULATIONS)
    {
        // Oversized blocks are split into segments that each respect the wide
        // accumulators' headroom, swept through the tiled kernel as
        // independent sub-blocks, and re-merged by parent block. This keeps
        // the bucketed, A-sequential sweep at any block size; the previous
        // per-block fallback walked entries in position order and re-streamed
        // `n_a` A rings per hot coefficient, which dominated trace-scale
        // commits (~2^18 hot coefficients per block at 2^26 cycles).
        let (sub_blocks, parents) = split_oversized_blocks(blocks, F::MAX_COMMIT_ACCUMULATIONS);
        let sub_out =
            column_sweep_core::<F, D>(a_view, &sub_blocks, n_a, active_a_cols, num_digits_inner);
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

    column_sweep_core::<F, D>(a_view, blocks, n_a, active_a_cols, num_digits_inner)
}
