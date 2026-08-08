use super::*;

/// Flat vector storing only the non-zero rings.
///
/// `offsets` says which entries belong to each block: block `i` occupies
/// `entries[offsets[i] as usize..offsets[i + 1] as usize]`.
///
/// Within one block, each entry records the position of a non-zero ring
/// (`pos_in_block`) together with the hot coefficient data for that ring
/// (`coeff_idx` for [`SingleChunkEntry`], `nonzero_coeffs` for
/// [`MultiChunkEntry`]).
///
/// Entries are sorted by `(block_idx, pos_in_block)`, so each per-block slice
/// is ascending in `pos_in_block`, matching the invariant the accumulators
/// rely on (they do `partition_point` on `pos_in_block`).
#[derive(Debug, Clone)]
pub(crate) struct FlatBlocks<E> {
    pub(super) entries: Vec<E>,
    /// `len == num_live_blocks + 1`, `offsets[0] == 0`, `offsets[num_live_blocks] == entries.len()`.
    pub(super) offsets: Vec<u32>,
}

impl<E> FlatBlocks<E> {
    #[inline]
    fn with_capacity(num_live_blocks: usize, entry_capacity: usize) -> Self {
        let mut offsets = Vec::with_capacity(num_live_blocks + 1);
        offsets.push(0);
        Self {
            entries: Vec::with_capacity(entry_capacity),
            offsets,
        }
    }

    /// Number of blocks.
    #[inline]
    pub(crate) fn num_live_blocks(&self) -> usize {
        self.offsets.len() - 1
    }

    /// Slice of entries for block `i`.
    pub(crate) fn block(&self, i: usize) -> &[E] {
        let num_live_blocks = self.num_live_blocks();
        assert!(
            i < num_live_blocks,
            "FlatBlocks::block: block index {i} out of range for {num_live_blocks} blocks"
        );
        let lo = self.offsets[i] as usize;
        let hi = self.offsets[i + 1] as usize;
        assert!(
            lo <= hi,
            "FlatBlocks::block: malformed offsets for block {i}: lo={lo} > hi={hi}"
        );
        &self.entries[lo..hi]
    }

    #[inline]
    fn advance_to_block(
        &mut self,
        current_block: &mut usize,
        block_idx: usize,
        num_live_blocks: usize,
    ) {
        debug_assert!(
            block_idx <= num_live_blocks,
            "FlatBlocks: block index {block_idx} out of range for {num_live_blocks} blocks"
        );
        while *current_block < block_idx {
            self.offsets.push(self.entries.len() as u32);
            *current_block += 1;
        }
    }

    #[inline]
    fn push_entry(
        &mut self,
        current_block: &mut usize,
        block_idx: usize,
        num_live_blocks: usize,
        entry: E,
    ) {
        debug_assert!(
            block_idx < num_live_blocks,
            "FlatBlocks: block index {block_idx} out of range for {num_live_blocks} blocks"
        );
        self.advance_to_block(current_block, block_idx, num_live_blocks);
        self.entries.push(entry);
    }

    fn finish_build(mut self, current_block: usize, num_live_blocks: usize) -> Self {
        let mut current_block = current_block;
        self.advance_to_block(&mut current_block, num_live_blocks, num_live_blocks);
        debug_assert_eq!(self.offsets.len(), num_live_blocks + 1);
        debug_assert_eq!(self.offsets[num_live_blocks] as usize, self.entries.len());
        self
    }

    #[cfg(test)]
    #[inline]
    pub(crate) fn table(&self) -> FlatBlockTable<'_, E> {
        FlatBlockTable::new(&self.entries, &self.offsets)
    }
}

impl FlatBlocks<MultiChunkEntry> {
    /// Build a multi-chunk-layout one-hot `FlatBlocks` from an index witness.
    ///
    /// This applies exactly to the `K < D && K | D` case, where each
    /// ring element contains `D/K` whole consecutive chunks. Grouping
    /// the witness by those chunk ranges lets us materialize each
    /// nonzero ring in one pass.
    ///
    /// # Errors
    ///
    /// Returns an error only if the internal offsets vector (bounded by
    /// `num_live_blocks + 1`) overflows `u32::MAX`.
    pub(crate) fn from_indices<I: OneHotIndex>(
        onehot_k: usize,
        indices: &[Option<I>],
        num_positions_per_block: usize,
        d: usize,
        num_live_blocks: usize,
    ) -> Result<Self, AkitaError> {
        assert!(
            onehot_k < d && d.is_multiple_of(onehot_k),
            "FlatBlocks::<MultiChunkEntry>::from_indices: K={onehot_k} and D={d} must satisfy K < D with K | D"
        );
        assert!(
            u32::try_from(num_positions_per_block).is_ok(),
            "FlatBlocks::<MultiChunkEntry>::from_indices: num_positions_per_block={num_positions_per_block} must fit in u32"
        );
        assert!(
            d <= usize::from(u16::MAX) + 1,
            "FlatBlocks::<MultiChunkEntry>::from_indices: D={d} must be <= 65536 so coeff_idx fits in u16"
        );
        let chunks_per_ring = d / onehot_k;
        assert!(
            indices.len().is_multiple_of(chunks_per_ring),
            "FlatBlocks::<MultiChunkEntry>::from_indices: index witness length {} must be divisible by D/K={chunks_per_ring}",
            indices.len()
        );
        Self::from_indices_ring_range(
            onehot_k,
            indices,
            num_positions_per_block,
            d,
            0..num_live_blocks * num_positions_per_block,
            0,
        )
    }

    /// Range variant of [`Self::from_indices`]: builds only blocks
    /// `block_range` (offsets local to the range). Blocks partition the
    /// ring-element index space contiguously, so this reads exactly the
    /// range's slice of the index witness — per-tile builds across a sweep
    /// sum to the same single pass as one full build.
    pub(crate) fn from_indices_ring_range<I: OneHotIndex>(
        onehot_k: usize,
        indices: &[Option<I>],
        num_positions_per_block: usize,
        d: usize,
        ring_range: std::ops::Range<usize>,
        first_block: usize,
    ) -> Result<Self, AkitaError> {
        assert!(
            onehot_k < d && d.is_multiple_of(onehot_k),
            "FlatBlocks::<MultiChunkEntry>::from_indices_ring_range: K={onehot_k} and D={d} must satisfy K < D with K | D"
        );
        let chunks_per_ring = d / onehot_k;
        let num_range_blocks = ring_range
            .end
            .div_ceil(num_positions_per_block)
            .saturating_sub(first_block);
        let ring_start = ring_range.start;
        let chunk_start = ring_start * chunks_per_ring;
        let chunk_end = (ring_range.end * chunks_per_ring).min(indices.len());
        let sub = indices.get(chunk_start..chunk_end).unwrap_or(&[]);
        let total_entries = sub.iter().filter(|opt| opt.is_some()).count();
        let mut blocks =
            FlatBlocks::<MultiChunkEntry>::with_capacity(num_range_blocks, total_entries);
        let mut current_block = 0usize;

        for (local_ring_idx, ring_chunks) in sub.chunks(chunks_per_ring).enumerate() {
            let mut nonzero_coeffs = Vec::with_capacity(ring_chunks.len());
            for (chunk_offset, opt) in ring_chunks.iter().copied().enumerate() {
                let Some(raw) = opt else {
                    continue;
                };
                let idx = raw.as_usize();
                let coeff_idx = chunk_offset
                    .checked_mul(onehot_k)
                    .and_then(|base| base.checked_add(idx))
                    .ok_or_else(|| AkitaError::InvalidInput("coefficient index overflow".into()))?;
                debug_assert!(coeff_idx < d);
                nonzero_coeffs.push(coeff_idx as u16);
            }
            if nonzero_coeffs.is_empty() {
                continue;
            }
            let ring_elem_idx = ring_start + local_ring_idx;
            let block_idx = ring_elem_idx / num_positions_per_block - first_block;
            let pos_in_block = (ring_elem_idx % num_positions_per_block) as u32;
            blocks.push_entry(
                &mut current_block,
                block_idx,
                num_range_blocks,
                MultiChunkEntry::new(pos_in_block, nonzero_coeffs),
            );
        }
        Ok(blocks.finish_build(current_block, num_range_blocks))
    }
}

impl FlatBlocks<SingleChunkEntry> {
    /// Build a single-chunk-layout one-hot `FlatBlocks` from an index witness.
    ///
    /// This applies to the common `K >= D && D | K` case, where each
    /// chunk spans one or more ring elements but still contributes
    /// exactly one nonzero coefficient in exactly one ring element.
    ///
    /// Like [`FlatBlocks::<MultiChunkEntry>::from_indices`],
    /// this constructor assumes its caller has already validated the
    /// structural preconditions: `K >= D && D | K`, `num_positions_per_block` is a
    /// power of two, `num_positions_per_block <=
    /// u32::MAX` and `D <= 65536`, and every `Some(idx)` entry in
    /// `indices` is in `[0, onehot_k)`. In production the sole caller is
    /// [`OneHotPoly::build_blocks_inner`].
    ///
    /// # Errors
    ///
    /// Returns an error only if the internal offsets vector (bounded by
    /// `num_live_blocks + 1`) overflows `u32::MAX`.
    pub(crate) fn from_indices<I: OneHotIndex>(
        onehot_k: usize,
        indices: &[Option<I>],
        num_positions_per_block: usize,
        d: usize,
        num_live_blocks: usize,
    ) -> Result<Self, AkitaError> {
        assert!(
            onehot_k >= d && onehot_k.is_multiple_of(d),
            "FlatBlocks::<SingleChunkEntry>::from_indices: K={onehot_k} and D={d} must satisfy K >= D with D | K"
        );
        assert!(
            u32::try_from(num_positions_per_block).is_ok(),
            "FlatBlocks::<SingleChunkEntry>::from_indices: num_positions_per_block={num_positions_per_block} must fit in u32"
        );
        debug_assert!(
            d <= usize::from(u16::MAX) + 1,
            "FlatBlocks::<SingleChunkEntry>::from_indices: D={d} must be <= 65536 so coeff_idx fits in u16"
        );
        Self::from_indices_ring_range(
            onehot_k,
            indices,
            num_positions_per_block,
            d,
            0..num_live_blocks * num_positions_per_block,
            0,
        )
    }
}

impl FlatBlocks<SingleChunkEntry> {
    /// Range variant of the single-chunk `from_indices`: builds only blocks
    /// `block_range` (offsets local to the range) from exactly the range's
    /// slice of the index witness. In the single-chunk layout (`K >= D`,
    /// `D | K`) each chunk spans `K/D` whole rings, so ring ranges map to
    /// contiguous chunk ranges.
    pub(crate) fn from_indices_ring_range<I: OneHotIndex>(
        onehot_k: usize,
        indices: &[Option<I>],
        num_positions_per_block: usize,
        d: usize,
        ring_range: std::ops::Range<usize>,
        first_block: usize,
    ) -> Result<Self, AkitaError> {
        assert!(
            onehot_k >= d && onehot_k.is_multiple_of(d),
            "FlatBlocks::<SingleChunkEntry>::from_indices_ring_range: K={onehot_k} and D={d} must satisfy K >= D with D | K"
        );
        let rings_per_chunk = onehot_k / d;
        let num_range_blocks = ring_range
            .end
            .div_ceil(num_positions_per_block)
            .saturating_sub(first_block);
        let ring_start = ring_range.start;
        let ring_end = ring_range.end;
        let chunk_start = ring_start / rings_per_chunk;
        let chunk_end = ring_end.div_ceil(rings_per_chunk).min(indices.len());
        let sub = indices.get(chunk_start..chunk_end).unwrap_or(&[]);
        let total_entries = sub.iter().filter(|opt| opt.is_some()).count();
        let mut blocks =
            FlatBlocks::<SingleChunkEntry>::with_capacity(num_range_blocks, total_entries);
        let mut current_block = 0usize;

        for (local_chunk_idx, opt) in sub.iter().copied().enumerate() {
            let Some(raw) = opt else {
                continue;
            };
            let chunk_idx = chunk_start + local_chunk_idx;
            let idx = raw.as_usize();
            let field_pos = chunk_idx
                .checked_mul(onehot_k)
                .and_then(|base| base.checked_add(idx))
                .ok_or_else(|| AkitaError::InvalidInput("field position overflow".into()))?;
            let ring_elem_idx = field_pos / d;
            if ring_elem_idx < ring_start || ring_elem_idx >= ring_end {
                // A chunk on the range boundary can own rings on either side;
                // its out-of-range hits belong to the neighboring tile.
                continue;
            }
            let coeff_idx = (field_pos % d) as u16;
            let block_idx = ring_elem_idx / num_positions_per_block - first_block;
            let pos_in_block = (ring_elem_idx % num_positions_per_block) as u32;
            blocks.push_entry(
                &mut current_block,
                block_idx,
                num_range_blocks,
                SingleChunkEntry::new(pos_in_block, coeff_idx),
            );
        }
        Ok(blocks.finish_build(current_block, num_range_blocks))
    }
}

/// Lazily buildable one-hot blocks: a monomorphized builder over the
/// polynomial's retained index columns, letting the commit sweep materialize
/// one block tile at a time instead of holding every block's entries.
pub struct LazyOneHotBlocks<'a, E> {
    /// Ring-range builder `(ring_range, first_block)`: entry `pos_in_block`
    /// stays absolute within its block; offsets are local to `first_block`.
    #[expect(clippy::type_complexity, reason = "one boxed builder signature")]
    build: Box<
        dyn Fn(std::ops::Range<usize>, usize) -> Result<FlatBlocks<E>, AkitaError>
            + Send
            + Sync
            + 'a,
    >,
    num_live_blocks: usize,
    num_positions_per_block: usize,
}

impl<'a, E> LazyOneHotBlocks<'a, E> {
    pub(crate) fn new(
        num_live_blocks: usize,
        num_positions_per_block: usize,
        build: impl Fn(std::ops::Range<usize>, usize) -> Result<FlatBlocks<E>, AkitaError>
            + Send
            + Sync
            + 'a,
    ) -> Self {
        Self {
            build: Box::new(build),
            num_live_blocks,
            num_positions_per_block,
        }
    }

    /// Number of blocks this source can build.
    #[inline]
    pub fn num_live_blocks(&self) -> usize {
        self.num_live_blocks
    }

    /// Materialize whole blocks `range` (offsets local to the range).
    pub(crate) fn build_range(
        &self,
        range: std::ops::Range<usize>,
    ) -> Result<FlatBlocks<E>, AkitaError> {
        (self.build)(
            range.start * self.num_positions_per_block..range.end * self.num_positions_per_block,
            range.start,
        )
    }
}

impl<E> core::fmt::Debug for LazyOneHotBlocks<'_, E> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("LazyOneHotBlocks")
            .field("num_live_blocks", &self.num_live_blocks)
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Clone)]
pub(crate) enum OneHotBlocks {
    SingleChunk(FlatBlocks<SingleChunkEntry>),
    MultiChunk(FlatBlocks<MultiChunkEntry>),
}
