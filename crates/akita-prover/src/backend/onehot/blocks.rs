use super::*;

impl FlatBlocks<SparseRingBlockEntry> {
    /// Build the requested ring range from the canonical one hot indices.
    ///
    /// For chunk `j` with hot index `i`, the single coordinate rule is
    /// `field_pos = j * K + i`, `ring_idx = field_pos / D`, and
    /// `coeff_idx = field_pos % D`. This does not depend on whether `K` or `D`
    /// is larger.
    pub(crate) fn from_onehot_ring_range<I: OneHotIndex>(
        onehot_k: usize,
        indices: &[Option<I>],
        num_positions_per_block: usize,
        d: usize,
        ring_range: std::ops::Range<usize>,
        first_block: usize,
    ) -> Result<Self, AkitaError> {
        let num_range_blocks = ring_range
            .end
            .div_ceil(num_positions_per_block)
            .saturating_sub(first_block);
        let field_start = ring_range
            .start
            .checked_mul(d)
            .ok_or_else(|| AkitaError::InvalidInput("one hot range start overflow".to_string()))?;
        let field_end = ring_range
            .end
            .checked_mul(d)
            .ok_or_else(|| AkitaError::InvalidInput("one hot range end overflow".to_string()))?;
        let chunk_start = field_start / onehot_k;
        let chunk_end = field_end.div_ceil(onehot_k).min(indices.len());
        let sub = indices.get(chunk_start..chunk_end).unwrap_or(&[]);
        let entry_capacity = sub.iter().filter(|entry| entry.is_some()).count();
        let mut blocks = Self::with_capacity(num_range_blocks, entry_capacity);
        let mut current_block = 0usize;

        for (local_chunk_idx, index) in sub.iter().copied().enumerate() {
            let Some(index) = index else {
                continue;
            };
            let chunk_idx = chunk_start + local_chunk_idx;
            let field_pos = chunk_idx
                .checked_mul(onehot_k)
                .and_then(|base| base.checked_add(index.as_usize()))
                .ok_or_else(|| {
                    AkitaError::InvalidInput("one hot field position overflow".to_string())
                })?;
            let ring_idx = field_pos / d;
            if !ring_range.contains(&ring_idx) {
                continue;
            }
            let block_idx = ring_idx / num_positions_per_block - first_block;
            let pos_in_block = u32::try_from(ring_idx % num_positions_per_block).map_err(|_| {
                AkitaError::InvalidInput("one hot block position exceeds u32".to_string())
            })?;
            let coeff_idx = u16::try_from(field_pos % d).map_err(|_| {
                AkitaError::InvalidInput("one hot coefficient index exceeds u16".to_string())
            })?;
            blocks.push_entry(
                &mut current_block,
                block_idx,
                num_range_blocks,
                // One-hot blocks are the `+1` subset of the shared signed
                // sparse entry representation. This builder is their sole
                // production construction boundary.
                SparseRingBlockEntry::new(pos_in_block, coeff_idx, 1),
            )?;
        }
        blocks.finish_build(current_block, num_range_blocks)
    }
}
