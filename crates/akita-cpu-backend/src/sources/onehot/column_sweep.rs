#[cfg(test)]
use super::inner_ajtai::inner_ajtai_wide_onehot;
use super::*;

const COMMIT_SCRATCH_TARGET_BYTES_PER_WORKER: usize = 8 << 20;

/// Bucketed and merge are arithmetic choices inside the same block range
/// driver. This enum is private policy state, not a source or plan type.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum OneHotSweep {
    Bucketed,
    Merge,
}

impl OneHotSweep {
    const fn label(self) -> &'static str {
        match self {
            Self::Bucketed => "bucketed",
            Self::Merge => "merge",
        }
    }
}

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

/// `accum += a · Σ_k X^k` over `shifts` for the `a` held by `windows`,
/// folding `accum` into `partial` before any batch would exceed the commit
/// accumulation budget.
#[inline(always)]
fn accumulate_shift_run<F, const D: usize>(
    windows: &NegacyclicShiftWindows<F, D>,
    shifts: &[usize],
    accum: &mut WideCyclotomicRing<F::Wide, D>,
    partial: &mut CyclotomicRing<F, D>,
    count: &mut usize,
) where
    F: Field + WithCommitAccumulator,
{
    for batch in shifts.chunks(F::MAX_COMMIT_ACCUMULATIONS) {
        if *count + batch.len() > F::MAX_COMMIT_ACCUMULATIONS {
            *partial += accum.reduce();
            *accum = WideCyclotomicRing::zero();
            *count = 0;
        }
        windows.accumulate_shifts_into(accum, batch);
        *count += batch.len();
    }
}

/// Bucket one materialized tile by A column, then sweep each A row once.
pub(super) fn bucketed_sweep_tile<F, const D: usize>(
    a_view: &RingMatrixView<'_, F, D>,
    blocks: &[&[SparseRingBlockEntry]],
    n_a: usize,
    active_a_cols: usize,
    num_digits_inner: usize,
) -> Vec<Vec<CyclotomicRing<F, D>>>
where
    F: Field + CanonicalEncoding + WithCommitAccumulator,
    F::Wide: AdditiveGroup + From<F>,
{
    debug_assert!(blocks.len() <= usize::from(u16::MAX) + 1);
    let mut col_counts = vec![0usize; active_a_cols];
    let mut entry_count = 0usize;
    for block_entries in blocks {
        for entry in *block_entries {
            let pos_in_block = entry.pos_in_block();
            let col = pos_in_block * num_digits_inner;
            debug_assert!(col < active_a_cols);
            col_counts[col] += 1;
            entry_count += 1;
        }
    }

    let mut col_offsets = Vec::with_capacity(active_a_cols + 1);
    col_offsets.push(0usize);
    for count in &col_counts {
        col_offsets.push(col_offsets.last().copied().unwrap_or(0) + count);
    }
    let mut write_offsets = col_offsets[..active_a_cols].to_vec();
    let mut packed_entries = vec![0; entry_count];
    for (local_block, block_entries) in blocks.iter().enumerate() {
        for entry in *block_entries {
            let pos_in_block = entry.pos_in_block();
            let coeff_idx = entry.coeff_idx();
            let col = pos_in_block * num_digits_inner;
            let dst = write_offsets[col];
            packed_entries[dst] = pack_col_entry(local_block, coeff_idx as u16);
            write_offsets[col] += 1;
        }
    }

    let mut result = vec![Vec::with_capacity(n_a); blocks.len()];
    let mut row_accums: Vec<WideCyclotomicRing<F::Wide, D>> =
        vec![WideCyclotomicRing::zero(); blocks.len()];
    let mut partials = vec![CyclotomicRing::zero(); blocks.len()];
    let mut accum_counts = vec![0usize; blocks.len()];
    let mut windows = NegacyclicShiftWindows::<F, D>::default();
    let mut shifts = Vec::new();
    for a_row in a_view.rows().take(n_a) {
        row_accums.fill(WideCyclotomicRing::zero());
        partials.fill(CyclotomicRing::zero());
        accum_counts.fill(0);
        for col in 0..active_a_cols {
            let entries = &packed_entries[col_offsets[col]..col_offsets[col + 1]];
            if entries.is_empty() {
                continue;
            }
            windows.load(&a_row[col]);
            // Buckets fill in block order, so each block's entries in this
            // column are one contiguous run.
            for run in entries.chunk_by(|&x, &y| unpack_col_entry(x).0 == unpack_col_entry(y).0) {
                let (local_block, _) = unpack_col_entry(run[0]);
                shifts.clear();
                shifts.extend(run.iter().map(|&entry| unpack_col_entry(entry).1));
                accumulate_shift_run(
                    &windows,
                    &shifts,
                    &mut row_accums[local_block],
                    &mut partials[local_block],
                    &mut accum_counts[local_block],
                );
            }
        }
        for (local_block, (rows, accum)) in result.iter_mut().zip(&row_accums).enumerate() {
            rows.push(partials[local_block] + accum.reduce());
        }
    }
    result
}

/// Number of A columns loaded together by the merge sweep. Bench-tuned:
/// the (tile, chunk) matrix is flat within ~5-30% and (64 blocks, 32 cols)
/// is its minimum at trace-like sparse shapes.
const MERGE_COL_CHUNK: usize = 32;

/// Walk sorted block cursors while each active A-column chunk is loaded once.
pub(super) fn merge_sweep_tile<F, const D: usize>(
    a_view: &RingMatrixView<'_, F, D>,
    tile_blocks: &[&[SparseRingBlockEntry]],
    n_a: usize,
    active_a_cols: usize,
    num_digits_inner: usize,
) -> Vec<Vec<CyclotomicRing<F, D>>>
where
    F: Field + CanonicalEncoding + WithCommitAccumulator,
    F::Wide: AdditiveGroup + From<F>,
{
    let mut chunk_windows: Vec<NegacyclicShiftWindows<F, D>> =
        vec![NegacyclicShiftWindows::default(); MERGE_COL_CHUNK];
    let mut shifts = Vec::new();
    let tile_len = tile_blocks.len();
    let mut result: Vec<Vec<CyclotomicRing<F, D>>> = Vec::with_capacity(tile_len);
    result.resize_with(tile_len, || Vec::with_capacity(n_a));
    {
        let mut row_accums: Vec<WideCyclotomicRing<F::Wide, D>> =
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

            for chunk_start in (0..active_a_cols).step_by(MERGE_COL_CHUNK) {
                let chunk_end = (chunk_start + MERGE_COL_CHUNK).min(active_a_cols);

                // Skip loading chunks no block has entries in.
                let live = tile_blocks.iter().zip(&cursors).any(|(entries, &cur)| {
                    entries
                        .get(cur)
                        .is_some_and(|e| e.pos_in_block() * num_digits_inner < chunk_end)
                });
                if !live {
                    continue;
                }
                for (windows, col) in chunk_windows.iter_mut().zip(chunk_start..chunk_end) {
                    windows.load(&a_row[col]);
                }

                for (local_b, entries) in tile_blocks.iter().enumerate() {
                    let cur = &mut cursors[local_b];
                    // Entries are sorted by position, so each column's
                    // entries in this block are one contiguous run.
                    while let Some(entry) = entries.get(*cur) {
                        let pos_in_block = entry.pos_in_block();
                        let col = pos_in_block * num_digits_inner;
                        if col >= chunk_end {
                            break;
                        }
                        debug_assert!(
                            col >= chunk_start,
                            "one-hot entries must be sorted by position within a block"
                        );
                        shifts.clear();
                        while let Some(entry) = entries
                            .get(*cur)
                            .filter(|entry| entry.pos_in_block() == pos_in_block)
                        {
                            shifts.push(entry.coeff_idx());
                            *cur += 1;
                        }
                        accumulate_shift_run(
                            &chunk_windows[col - chunk_start],
                            &shifts,
                            &mut row_accums[local_b],
                            &mut partials[local_b],
                            &mut accum_counts[local_b],
                        );
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

fn worker_count(total_blocks: usize) -> usize {
    #[cfg(feature = "parallel")]
    let workers = rayon::current_num_threads();
    #[cfg(not(feature = "parallel"))]
    let workers = 1;
    workers.min(total_blocks).max(1)
}

fn estimated_matrix_passes(total_blocks: usize, workers: usize, block_tile: usize) -> usize {
    let blocks_per_worker = total_blocks.div_ceil(workers);
    (0..workers)
        .map(|worker| {
            let start = worker * blocks_per_worker;
            let end = (start + blocks_per_worker).min(total_blocks);
            end.saturating_sub(start).div_ceil(block_tile)
        })
        .sum()
}

fn max_entries_per_block<const D: usize, I: OneHotIndex>(
    sources: &[OneHotSource<'_, I>],
    num_positions_per_block: usize,
) -> Result<usize, AkitaError> {
    let field_elems_per_block = num_positions_per_block
        .checked_mul(D)
        .ok_or_else(|| AkitaError::InvalidSetup("one hot block field width overflow".into()))?;
    sources.iter().try_fold(0usize, |current, source| {
        let crossing_bound = field_elems_per_block
            .div_ceil(source.chunk_size)
            .saturating_add(1)
            .min(source.indices.len());
        Ok(current.max(crossing_bound))
    })
}

fn block_tile_for_scratch<F, const D: usize>(
    total_blocks: usize,
    active_a_cols: usize,
    max_entries_per_block: usize,
) -> Result<usize, AkitaError>
where
    F: Field + WithCommitAccumulator,
{
    let wide_ring = D
        .checked_mul(std::mem::size_of::<F::Wide>())
        .ok_or_else(|| AkitaError::InvalidSetup("one hot wide ring size overflow".into()))?;
    let shift_windows = D
        .checked_mul(2 * std::mem::size_of::<F::CommitLanes>())
        .ok_or_else(|| AkitaError::InvalidSetup("one hot shift window size overflow".into()))?;
    let field_ring = D
        .checked_mul(std::mem::size_of::<F>())
        .ok_or_else(|| AkitaError::InvalidSetup("one hot field ring size overflow".into()))?;
    let entry_storage = max_entries_per_block
        .checked_mul(
            std::mem::size_of::<SparseRingBlockEntry>() + std::mem::size_of::<PackedColEntry>(),
        )
        .ok_or_else(|| AkitaError::InvalidSetup("one hot entry scratch overflow".into()))?;
    let per_block = entry_storage
        .checked_add(wide_ring)
        .and_then(|bytes| bytes.checked_add(field_ring))
        .and_then(|bytes| bytes.checked_add(2 * std::mem::size_of::<usize>()))
        .ok_or_else(|| AkitaError::InvalidSetup("one hot per-block scratch overflow".into()))?;

    let bucket_fixed = active_a_cols
        .checked_mul(3 * std::mem::size_of::<usize>())
        .and_then(|bytes| bytes.checked_add(std::mem::size_of::<usize>()))
        .and_then(|bytes| bytes.checked_add(shift_windows))
        .ok_or_else(|| AkitaError::InvalidSetup("one hot bucket scratch overflow".into()))?;
    let merge_fixed = MERGE_COL_CHUNK
        .checked_mul(shift_windows)
        .ok_or_else(|| AkitaError::InvalidSetup("one hot merge scratch overflow".into()))?;
    let fixed = bucket_fixed.max(merge_fixed);
    let minimum = fixed
        .checked_add(per_block)
        .ok_or_else(|| AkitaError::InvalidSetup("one hot minimum scratch overflow".into()))?;
    // Preserve batching for small blocks while allowing any checked one-block minimum.
    let scratch_bytes_per_worker = COMMIT_SCRATCH_TARGET_BYTES_PER_WORKER.max(minimum);
    let available = scratch_bytes_per_worker - fixed;
    let tile = available / per_block;
    Ok(tile.min(usize::from(u16::MAX) + 1).min(total_blocks.max(1)))
}

pub(super) fn select_sweep(
    total_blocks: usize,
    active_a_cols: usize,
    workers: usize,
) -> OneHotSweep {
    let blocks_per_worker = total_blocks.div_ceil(workers);

    if active_a_cols <= 256 && blocks_per_worker >= 32 {
        OneHotSweep::Merge
    } else {
        OneHotSweep::Bucketed
    }
}

#[cfg(test)]
pub(super) fn direct_sweep_tile<F, const D: usize>(
    a_view: &RingMatrixView<'_, F, D>,
    blocks: &[&[SparseRingBlockEntry]],
    num_digits_inner: usize,
) -> Vec<Vec<CyclotomicRing<F, D>>>
where
    F: Field + CanonicalEncoding + WithCommitAccumulator,
    F::Wide: AdditiveGroup + From<F>,
{
    blocks
        .iter()
        .map(|entries| {
            let mut rows = Vec::new();
            for segment in entries.chunks(F::MAX_COMMIT_ACCUMULATIONS.max(1)) {
                let partial = inner_ajtai_wide_onehot(a_view, segment, num_digits_inner);
                if rows.is_empty() {
                    rows = partial;
                } else {
                    for (row, value) in rows.iter_mut().zip(partial) {
                        *row += value;
                    }
                }
            }
            if rows.is_empty() {
                rows.resize(a_view.num_rows(), CyclotomicRing::zero());
            }
            rows
        })
        .collect()
}

fn run_sweep_tile<F, const D: usize>(
    sweep: OneHotSweep,
    a_view: &RingMatrixView<'_, F, D>,
    blocks: &[&[SparseRingBlockEntry]],
    n_a: usize,
    active_a_cols: usize,
    num_digits_inner: usize,
) -> Vec<Vec<CyclotomicRing<F, D>>>
where
    F: Field + CanonicalEncoding + WithCommitAccumulator,
    F::Wide: AdditiveGroup + From<F>,
{
    debug_assert!(
        blocks
            .iter()
            .flat_map(|entries| entries.iter())
            .all(|entry| entry.value() == 1),
        "one-hot block builder must emit only unit entries"
    );
    match sweep {
        OneHotSweep::Bucketed => {
            bucketed_sweep_tile(a_view, blocks, n_a, active_a_cols, num_digits_inner)
        }
        OneHotSweep::Merge => {
            merge_sweep_tile(a_view, blocks, n_a, active_a_cols, num_digits_inner)
        }
    }
}

fn column_sweep_ajtai_onehot_multi_with_sweep<F, const D: usize, I>(
    a_view: &RingMatrixView<'_, F, D>,
    sources: &[OneHotSource<'_, I>],
    n_a: usize,
    active_a_cols: usize,
    num_digits_inner: usize,
    forced_sweep: Option<OneHotSweep>,
) -> Result<Vec<Vec<Vec<CyclotomicRing<F, D>>>>, AkitaError>
where
    F: Field + CanonicalEncoding + WithCommitAccumulator,
    F::Wide: AdditiveGroup + From<F>,
    I: OneHotIndex,
{
    if num_digits_inner == 0 || !active_a_cols.is_multiple_of(num_digits_inner) {
        return Err(AkitaError::InvalidSetup(
            "one hot active A width must be divisible by a nonzero digit count".into(),
        ));
    }
    let num_positions_per_block = active_a_cols / num_digits_inner;
    let counts = sources
        .iter()
        .map(|source| source.num_live_blocks_for(D, num_positions_per_block))
        .collect::<Result<Vec<_>, _>>()?;
    let mut starts = Vec::with_capacity(counts.len() + 1);
    let mut total = 0usize;
    for count in &counts {
        starts.push(total);
        total = total
            .checked_add(*count)
            .ok_or_else(|| AkitaError::InvalidSetup("one hot block count overflow".into()))?;
    }
    starts.push(total);
    if total == 0 {
        return Ok(vec![Vec::new(); sources.len()]);
    }

    let workers = worker_count(total);
    let hot_terms = if tracing::enabled!(tracing::Level::INFO) {
        Some(sources.iter().try_fold(0usize, |total, source| {
            total
                .checked_add(
                    source
                        .indices
                        .iter()
                        .filter(|entry| entry.is_some())
                        .count(),
                )
                .ok_or_else(|| AkitaError::InvalidSetup("one hot term count overflow".into()))
        })?)
    } else {
        None
    };
    let max_entries = max_entries_per_block::<D, I>(sources, num_positions_per_block)?;
    let block_tile = block_tile_for_scratch::<F, D>(total, active_a_cols, max_entries)?;
    let sweep = forced_sweep.unwrap_or_else(|| select_sweep(total, active_a_cols, workers));
    let matrix_passes = estimated_matrix_passes(total, workers, block_tile);
    tracing::info!(
        sweep = sweep.label(),
        block_tile,
        hot_terms = hot_terms.unwrap_or_default(),
        source_count = sources.len(),
        total_blocks = total,
        workers,
        n_a,
        active_a_cols,
        ring_dimension = D,
        estimated_matrix_passes = matrix_passes,
        "one hot commit schedule"
    );

    let blocks_per_worker = total.div_ceil(workers);

    let thread_results: Vec<Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError>> =
        cfg_into_iter!(0..workers)
            .map(|worker| {
                let block_start = worker * blocks_per_worker;
                let block_end = (block_start + blocks_per_worker).min(total);
                if block_start >= block_end {
                    return Ok(Vec::new());
                }
                let my_count = block_end - block_start;
                let mut result: Vec<Vec<CyclotomicRing<F, D>>> = Vec::with_capacity(my_count);
                result.resize_with(my_count, Vec::new);
                for tile_start in (block_start..block_end).step_by(block_tile) {
                    let tile_end = (tile_start + block_tile).min(block_end);
                    // Build the overlapping range from each source. Owners and
                    // their entries are dropped at tile end.
                    let mut owners = Vec::new();
                    for (src_idx, source) in sources.iter().enumerate() {
                        let src_start = starts[src_idx];
                        let src_end = starts[src_idx + 1];
                        let lo = tile_start.max(src_start);
                        let hi = tile_end.min(src_end);
                        if lo >= hi {
                            continue;
                        }
                        owners.push(source.materialize_block_range(
                            D,
                            num_positions_per_block,
                            lo - src_start..hi - src_start,
                        )?);
                    }
                    let block_groups = owners
                        .iter()
                        .map(|blocks| {
                            (0..blocks.num_live_blocks())
                                .map(|block| blocks.block(block))
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>();
                    let tile_blocks: Vec<&[SparseRingBlockEntry]> =
                        block_groups.iter().flatten().copied().collect();
                    let tile_rows = run_sweep_tile::<F, D>(
                        sweep,
                        a_view,
                        &tile_blocks,
                        n_a,
                        active_a_cols,
                        num_digits_inner,
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

/// Fused multi-polynomial column sweep over one hot source views.
pub(crate) fn column_sweep_ajtai_onehot_multi<F, const D: usize, I>(
    a_view: &RingMatrixView<'_, F, D>,
    sources: &[OneHotSource<'_, I>],
    n_a: usize,
    active_a_cols: usize,
    num_digits_inner: usize,
) -> Result<Vec<Vec<Vec<CyclotomicRing<F, D>>>>, AkitaError>
where
    F: Field + CanonicalEncoding + WithCommitAccumulator,
    F::Wide: AdditiveGroup + From<F>,
    I: OneHotIndex,
{
    column_sweep_ajtai_onehot_multi_with_sweep(
        a_view,
        sources,
        n_a,
        active_a_cols,
        num_digits_inner,
        None,
    )
}

#[cfg(test)]
pub(super) fn column_sweep_ajtai_onehot_multi_forced<F, const D: usize, I>(
    a_view: &RingMatrixView<'_, F, D>,
    sources: &[OneHotSource<'_, I>],
    n_a: usize,
    active_a_cols: usize,
    num_digits_inner: usize,
    sweep: OneHotSweep,
) -> Result<Vec<Vec<Vec<CyclotomicRing<F, D>>>>, AkitaError>
where
    F: Field + CanonicalEncoding + WithCommitAccumulator,
    F::Wide: AdditiveGroup + From<F>,
    I: OneHotIndex,
{
    column_sweep_ajtai_onehot_multi_with_sweep(
        a_view,
        sources,
        n_a,
        active_a_cols,
        num_digits_inner,
        Some(sweep),
    )
}

/// Simple direct arithmetic reference for block-level differential tests.
#[cfg(test)]
pub(crate) fn column_sweep_ajtai_onehot<F, const D: usize>(
    a_view: &RingMatrixView<'_, F, D>,
    blocks: &[&[SparseRingBlockEntry]],
    _n_a: usize,
    _active_a_cols: usize,
    num_digits_inner: usize,
) -> Vec<Vec<CyclotomicRing<F, D>>>
where
    F: Field + CanonicalEncoding + WithCommitAccumulator,
    F::Wide: AdditiveGroup + From<F>,
{
    direct_sweep_tile(a_view, blocks, num_digits_inner)
}
