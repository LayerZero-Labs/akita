use crate::compute::FlatBlockTable;
use akita_field::AkitaError;

/// Owned flat storage for sparse entries grouped by logical block.
///
/// Block `i` occupies `entries[offsets[i]..offsets[i + 1]]`. Builders append
/// entries in block order, so entries within each block preserve their source
/// order.
#[derive(Debug, Clone)]
pub(crate) struct FlatBlocks<E> {
    entries: Vec<E>,
    offsets: Vec<u32>,
}

impl<E> FlatBlocks<E> {
    pub(crate) fn with_capacity(num_live_blocks: usize, entry_capacity: usize) -> Self {
        let mut offsets = Vec::with_capacity(num_live_blocks + 1);
        offsets.push(0);
        Self {
            entries: Vec::with_capacity(entry_capacity),
            offsets,
        }
    }

    #[inline]
    pub(crate) fn num_live_blocks(&self) -> usize {
        self.offsets.len() - 1
    }

    pub(crate) fn block(&self, i: usize) -> &[E] {
        let num_live_blocks = self.num_live_blocks();
        assert!(
            i < num_live_blocks,
            "FlatBlocks::block: block index {i} out of range for {num_live_blocks} blocks"
        );
        let lo = self.offsets[i] as usize;
        let hi = self.offsets[i + 1] as usize;
        assert!(
            lo <= hi && hi <= self.entries.len(),
            "FlatBlocks::block: malformed offsets for block {i}: {lo}..{hi} over {} entries",
            self.entries.len()
        );
        &self.entries[lo..hi]
    }

    fn entry_offset(&self) -> Result<u32, AkitaError> {
        checked_entry_offset(self.entries.len())
    }

    fn advance_to_block(
        &mut self,
        current_block: &mut usize,
        block_idx: usize,
        num_live_blocks: usize,
    ) -> Result<(), AkitaError> {
        debug_assert!(
            block_idx <= num_live_blocks,
            "FlatBlocks: block index {block_idx} out of range for {num_live_blocks} blocks"
        );
        while *current_block < block_idx {
            self.offsets.push(self.entry_offset()?);
            *current_block += 1;
        }
        Ok(())
    }

    pub(crate) fn push_entry(
        &mut self,
        current_block: &mut usize,
        block_idx: usize,
        num_live_blocks: usize,
        entry: E,
    ) -> Result<(), AkitaError> {
        debug_assert!(
            block_idx < num_live_blocks,
            "FlatBlocks: block index {block_idx} out of range for {num_live_blocks} blocks"
        );
        self.advance_to_block(current_block, block_idx, num_live_blocks)?;
        self.entries.push(entry);
        Ok(())
    }

    pub(crate) fn finish_build(
        mut self,
        mut current_block: usize,
        num_live_blocks: usize,
    ) -> Result<Self, AkitaError> {
        self.advance_to_block(&mut current_block, num_live_blocks, num_live_blocks)?;
        debug_assert_eq!(self.offsets.len(), num_live_blocks + 1);
        debug_assert_eq!(self.offsets[num_live_blocks] as usize, self.entries.len());
        Ok(self)
    }

    #[inline]
    pub(crate) fn table(&self) -> FlatBlockTable<'_, E> {
        FlatBlockTable::new(&self.entries, &self.offsets)
    }

    #[cfg(test)]
    pub(crate) fn from_buckets(buckets: Vec<Vec<E>>) -> Result<Self, AkitaError> {
        let num_live_blocks = buckets.len();
        let entry_capacity = buckets.iter().map(Vec::len).sum();
        let mut blocks = Self::with_capacity(num_live_blocks, entry_capacity);
        let mut current_block = 0;
        for (block_idx, bucket) in buckets.into_iter().enumerate() {
            for entry in bucket {
                blocks.push_entry(&mut current_block, block_idx, num_live_blocks, entry)?;
            }
        }
        blocks.finish_build(current_block, num_live_blocks)
    }
}

fn checked_entry_offset(len: usize) -> Result<u32, AkitaError> {
    u32::try_from(len).map_err(|_| {
        AkitaError::InvalidInput(format!("flat block entry count {len} exceeds u32::MAX"))
    })
}

#[cfg(test)]
mod tests {
    use super::checked_entry_offset;

    #[test]
    fn rejects_entry_offset_above_u32_max() {
        if let Some(too_large) = (u32::MAX as usize).checked_add(1) {
            assert!(checked_entry_offset(too_large).is_err());
        }
    }
}
