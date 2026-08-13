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
        let entries = OneHotRingRange::new(onehot_k, indices, d, ring_range)?;
        let entry_capacity = entries.entry_capacity();
        let mut blocks = Self::with_capacity(num_range_blocks, entry_capacity);
        let mut current_block = 0usize;

        for entry in entries {
            let entry = entry?;
            let block_idx = entry.ring_index / num_positions_per_block - first_block;
            let pos_in_block =
                u32::try_from(entry.ring_index % num_positions_per_block).map_err(|_| {
                    AkitaError::InvalidInput("one hot block position exceeds u32".to_string())
                })?;
            let coeff_idx = u16::try_from(entry.coefficient_index).map_err(|_| {
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
