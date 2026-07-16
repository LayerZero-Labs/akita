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
    /// `len == live_block_count + 1`, `offsets[0] == 0`, `offsets[live_block_count] == entries.len()`.
    pub(super) offsets: Vec<u32>,
}

impl<E> FlatBlocks<E> {
    #[inline]
    fn with_capacity(live_block_count: usize, entry_capacity: usize) -> Self {
        let mut offsets = Vec::with_capacity(live_block_count + 1);
        offsets.push(0);
        Self {
            entries: Vec::with_capacity(entry_capacity),
            offsets,
        }
    }

    /// Number of blocks.
    #[inline]
    pub(crate) fn live_block_count(&self) -> usize {
        self.offsets.len() - 1
    }

    /// Slice of entries for block `i`.
    pub(crate) fn block(&self, i: usize) -> &[E] {
        let live_block_count = self.live_block_count();
        assert!(
            i < live_block_count,
            "FlatBlocks::block: block index {i} out of range for {live_block_count} blocks"
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
        live_block_count: usize,
    ) {
        debug_assert!(
            block_idx <= live_block_count,
            "FlatBlocks: block index {block_idx} out of range for {live_block_count} blocks"
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
        live_block_count: usize,
        entry: E,
    ) {
        debug_assert!(
            block_idx < live_block_count,
            "FlatBlocks: block index {block_idx} out of range for {live_block_count} blocks"
        );
        self.advance_to_block(current_block, block_idx, live_block_count);
        self.entries.push(entry);
    }

    fn finish_build(mut self, current_block: usize, live_block_count: usize) -> Self {
        let mut current_block = current_block;
        self.advance_to_block(&mut current_block, live_block_count, live_block_count);
        debug_assert_eq!(self.offsets.len(), live_block_count + 1);
        debug_assert_eq!(self.offsets[live_block_count] as usize, self.entries.len());
        self
    }

    #[inline]
    fn table(&self) -> FlatBlockTable<'_, E> {
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
    /// `live_block_count + 1`) overflows `u32::MAX`.
    pub(crate) fn from_indices<I: OneHotIndex>(
        onehot_k: usize,
        indices: &[Option<I>],
        positions_per_block: usize,
        d: usize,
        live_block_count: usize,
    ) -> Result<Self, AkitaError> {
        assert!(
            onehot_k < d && d.is_multiple_of(onehot_k),
            "FlatBlocks::<MultiChunkEntry>::from_indices: K={onehot_k} and D={d} must satisfy K < D with K | D"
        );
        assert!(
            u32::try_from(positions_per_block).is_ok(),
            "FlatBlocks::<MultiChunkEntry>::from_indices: positions_per_block={positions_per_block} must fit in u32"
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
        let total_entries = indices.iter().filter(|opt| opt.is_some()).count();
        let mut blocks =
            FlatBlocks::<MultiChunkEntry>::with_capacity(live_block_count, total_entries);
        let mut current_block = 0usize;

        for (ring_elem_idx, ring_chunks) in indices.chunks(chunks_per_ring).enumerate() {
            let mut nonzero_coeffs = Vec::with_capacity(ring_chunks.len());

            for (chunk_offset, opt) in ring_chunks.iter().copied().enumerate() {
                let Some(raw) = opt else {
                    continue;
                };
                let idx = raw.as_usize();
                assert!(
                    idx < onehot_k,
                    "FlatBlocks::<MultiChunkEntry>::from_indices: index {idx} out of range for K={onehot_k} in ring {ring_elem_idx}, chunk offset {chunk_offset}"
                );
                let coeff_idx = chunk_offset
                    .checked_mul(onehot_k)
                    .and_then(|base| base.checked_add(idx))
                    .ok_or_else(|| AkitaError::InvalidInput("coefficient index overflow".into()))?;
                debug_assert!(
                    coeff_idx < d,
                    "multi-chunk onehot: coefficient indices inside one ring must stay < D"
                );
                nonzero_coeffs.push(coeff_idx as u16);
            }

            if nonzero_coeffs.is_empty() {
                continue;
            }

            let block_idx = ring_elem_idx / positions_per_block;
            let pos_in_block = (ring_elem_idx % positions_per_block) as u32;
            assert!(
                block_idx >= current_block,
                "multi-chunk onehot: entries must be non-decreasing in block index"
            );
            blocks.push_entry(
                &mut current_block,
                block_idx,
                live_block_count,
                MultiChunkEntry::new(pos_in_block, nonzero_coeffs),
            );
        }

        Ok(blocks.finish_build(current_block, live_block_count))
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
    /// structural preconditions: `K >= D && D | K`, `positions_per_block` is a
    /// power of two, `positions_per_block <=
    /// u32::MAX` and `D <= 65536`, and every `Some(idx)` entry in
    /// `indices` is in `[0, onehot_k)`. In production the sole caller is
    /// [`OneHotPoly::build_blocks_inner`].
    ///
    /// # Errors
    ///
    /// Returns an error only if the internal offsets vector (bounded by
    /// `live_block_count + 1`) overflows `u32::MAX`.
    pub(crate) fn from_indices<I: OneHotIndex>(
        onehot_k: usize,
        indices: &[Option<I>],
        positions_per_block: usize,
        d: usize,
        live_block_count: usize,
    ) -> Result<Self, AkitaError> {
        debug_assert!(
            onehot_k >= d && onehot_k.is_multiple_of(d),
            "FlatBlocks::<SingleChunkEntry>::from_indices: K={onehot_k} and D={d} must satisfy K >= D with D | K"
        );
        debug_assert!(
            u32::try_from(positions_per_block).is_ok(),
            "FlatBlocks::<SingleChunkEntry>::from_indices: positions_per_block={positions_per_block} must fit in u32"
        );
        debug_assert!(
            d <= usize::from(u16::MAX) + 1,
            "FlatBlocks::<SingleChunkEntry>::from_indices: D={d} must be <= 65536 so coeff_idx fits in u16"
        );

        let total_entries = indices.iter().filter(|opt| opt.is_some()).count();
        let mut blocks =
            FlatBlocks::<SingleChunkEntry>::with_capacity(live_block_count, total_entries);
        let mut current_block = 0usize;

        for (chunk_idx, opt) in indices.iter().copied().enumerate() {
            let Some(raw) = opt else {
                continue;
            };
            let idx = raw.as_usize();
            debug_assert!(
                idx < onehot_k,
                "FlatBlocks::<SingleChunkEntry>::from_indices: index {idx} out of range for K={onehot_k} at position {chunk_idx}"
            );

            let field_pos = chunk_idx
                .checked_mul(onehot_k)
                .and_then(|base| base.checked_add(idx))
                .ok_or_else(|| AkitaError::InvalidInput("field position overflow".into()))?;
            let ring_elem_idx = field_pos / d;
            let coeff_idx = (field_pos % d) as u16;
            let block_idx = ring_elem_idx / positions_per_block;
            let pos_in_block = (ring_elem_idx % positions_per_block) as u32;
            debug_assert!(
                block_idx >= current_block,
                "single-chunk onehot: entries must be non-decreasing in block index"
            );
            blocks.push_entry(
                &mut current_block,
                block_idx,
                live_block_count,
                SingleChunkEntry::new(pos_in_block, coeff_idx),
            );
        }

        Ok(blocks.finish_build(current_block, live_block_count))
    }
}

#[derive(Debug, Clone)]
pub(crate) enum OneHotBlocks {
    SingleChunk(FlatBlocks<SingleChunkEntry>),
    MultiChunk(FlatBlocks<MultiChunkEntry>),
}

impl OneHotBlocks {
    #[inline]
    pub(crate) fn live_block_count(&self) -> usize {
        match self {
            OneHotBlocks::SingleChunk(blocks) => blocks.live_block_count(),
            OneHotBlocks::MultiChunk(blocks) => blocks.live_block_count(),
        }
    }

    pub(super) fn commit_plan_blocks(&self) -> OneHotCommitBlocks<'_> {
        match self {
            OneHotBlocks::SingleChunk(blocks) => OneHotCommitBlocks::SingleChunk(blocks.table()),
            OneHotBlocks::MultiChunk(blocks) => OneHotCommitBlocks::MultiChunk(blocks.table()),
        }
    }
}
