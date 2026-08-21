use super::*;

/// One-hot polynomial: sparse witness with at most one nonzero field element
/// per chunk of size `onehot_k`.
///
/// The polynomial is stored layout-agnostically as the flat list of hot
/// indices supplied at construction. Each operation takes its layout at call
/// time and derives only the range it consumes.
///
/// Storage is D-free: the per-chunk hot indices are flat logical data, and
/// the ring dimension is a view selected at kernel entry (each ring-shaped
/// method takes it as a const generic).
///
/// Generic over `I`: the index type accepted and stored per chunk. Use `u8`
/// when `onehot_k <= 256` to reduce index storage footprint.
#[derive(Debug, Clone)]
pub struct OneHotPoly<F: Field, I: OneHotIndex = usize> {
    pub(crate) num_vars: usize,
    pub(crate) onehot_k: usize,
    /// Per-chunk hot-position indices. `None` denotes an all-zero chunk.
    pub(crate) indices: Vec<Option<I>>,
    pub(crate) _marker: PhantomData<F>,
}

impl<F: Field, I: OneHotIndex> OneHotPoly<F, I> {
    /// Build a one-hot polynomial from chunk size and hot-position indices.
    ///
    /// `indices[c]` is the hot position in chunk `c` (`None` for all-zero chunks).
    ///
    /// The commit-layout split (how blocks are tiled within the polynomial)
    /// is no longer baked in at construction. Each op receives `num_positions_per_block`
    /// from the caller and the per-block representation is materialized on
    /// demand.
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions are inconsistent or any index is out of
    /// range.
    pub fn new(onehot_k: usize, indices: Vec<Option<I>>) -> Result<Self, AkitaError> {
        if onehot_k == 0 {
            return Err(AkitaError::InvalidInput(
                "onehot_k must be nonzero".to_string(),
            ));
        }
        let total_field_elems = indices.len().checked_mul(onehot_k).ok_or_else(|| {
            AkitaError::InvalidInput("onehot total field element count overflow".to_string())
        })?;
        if !total_field_elems.is_power_of_two() {
            return Err(AkitaError::InvalidInput(format!(
                "onehot total field elements {total_field_elems} is not a power of two"
            )));
        }
        for (chunk_idx, opt) in indices.iter().copied().enumerate() {
            if let Some(raw) = opt {
                let idx = raw.as_usize();
                if idx >= onehot_k {
                    return Err(AkitaError::InvalidInput(format!(
                        "index {idx} out of range for chunk size K={onehot_k} at position {chunk_idx}"
                    )));
                }
            }
        }
        Ok(Self {
            num_vars: total_field_elems.trailing_zeros() as usize,
            onehot_k,
            indices,
            _marker: PhantomData,
        })
    }

    /// Number of field-evaluation slots in each compact one-hot chunk.
    #[inline]
    pub fn onehot_k(&self) -> usize {
        self.onehot_k
    }

    /// Per-chunk hot-position indices. `None` denotes an all-zero chunk.
    #[inline]
    pub fn indices(&self) -> &[Option<I>] {
        &self.indices
    }

    /// Traverse the one hot coefficients in a requested ring range without
    /// materializing another owner.
    ///
    /// The source chunk range lets block materialization retain its exact
    /// capacity calculation without repeating coordinate mapping. Fold paths
    /// ignore it and consume only the coefficient iterator.
    pub(super) fn ring_range_coefficients(
        &self,
        ring_d: usize,
        ring_range: std::ops::Range<usize>,
    ) -> Result<
        (
            std::ops::Range<usize>,
            impl Iterator<Item = Result<SparseRingCoeff, AkitaError>> + '_,
        ),
        AkitaError,
    > {
        if ring_range.start > ring_range.end {
            return Err(AkitaError::InvalidInput(
                "one hot ring range must be ordered".into(),
            ));
        }
        let num_rings = self.validate_ring_dimension(ring_d)?;
        let ring_start = ring_range.start.min(num_rings);
        let ring_end = ring_range.end.min(num_rings);
        let field_start = ring_start
            .checked_mul(ring_d)
            .ok_or_else(|| AkitaError::InvalidInput("one hot range start overflow".into()))?;
        let field_end = ring_end
            .checked_mul(ring_d)
            .ok_or_else(|| AkitaError::InvalidInput("one hot range end overflow".into()))?;
        let chunk_start = field_start / self.onehot_k;
        let chunk_end = field_end.div_ceil(self.onehot_k).min(self.indices.len());
        let source_chunks = chunk_start..chunk_end;
        let ring_range = ring_start..ring_end;

        let coefficients = self.indices[source_chunks.clone()]
            .iter()
            .copied()
            .enumerate()
            .filter_map(move |(local_chunk, hot_index)| {
                let hot_index = hot_index?;
                let chunk_index = chunk_start + local_chunk;
                let coefficient = self
                    .hot_field_position(chunk_index, hot_index, "ring range")
                    .and_then(|field_position| {
                        if ring_range.contains(&(field_position / ring_d)) {
                            SparseRingCoeff::new(field_position, 1).map(Some)
                        } else {
                            Ok(None)
                        }
                    });
                match coefficient {
                    Ok(Some(coefficient)) => Some(Ok(coefficient)),
                    Ok(None) => None,
                    Err(err) => Some(Err(err)),
                }
            });
        Ok((source_chunks, coefficients))
    }

    pub(super) fn materialize_block_range(
        &self,
        ring_d: usize,
        num_positions_per_block: usize,
        block_range: std::ops::Range<usize>,
    ) -> Result<FlatBlocks<SparseRingBlockEntry>, AkitaError> {
        let (ring_elems_at_d, num_live_blocks) =
            self.view_layout(ring_d, num_positions_per_block)?;
        if block_range.start > block_range.end || block_range.end > num_live_blocks {
            return Err(AkitaError::InvalidInput(format!(
                "one hot block range {:?} exceeds {num_live_blocks} blocks",
                block_range
            )));
        }
        let ring_start = block_range
            .start
            .checked_mul(num_positions_per_block)
            .ok_or_else(|| AkitaError::InvalidInput("one hot block range overflow".into()))?
            .min(ring_elems_at_d);
        let ring_end = block_range
            .end
            .checked_mul(num_positions_per_block)
            .ok_or_else(|| AkitaError::InvalidInput("one hot block range overflow".into()))?
            .min(ring_elems_at_d);
        let num_range_blocks = ring_end
            .div_ceil(num_positions_per_block)
            .saturating_sub(block_range.start);
        let (source_chunks, coefficients) =
            self.ring_range_coefficients(ring_d, ring_start..ring_end)?;
        let entry_capacity = self.indices[source_chunks]
            .iter()
            .filter(|entry| entry.is_some())
            .count();
        let mut blocks = FlatBlocks::with_capacity(num_range_blocks, entry_capacity);
        let mut current_block = 0usize;
        for coefficient in coefficients {
            let coefficient = coefficient?;
            let ring_index = coefficient.ring_idx(ring_d);
            let block_index = ring_index / num_positions_per_block - block_range.start;
            let position = u32::try_from(ring_index % num_positions_per_block).map_err(|_| {
                AkitaError::InvalidInput("one hot block position exceeds u32".into())
            })?;
            let coefficient_index = u16::try_from(coefficient.coeff_idx(ring_d)).map_err(|_| {
                AkitaError::InvalidInput("one hot coefficient index exceeds u16".into())
            })?;
            blocks.push_entry(
                &mut current_block,
                block_index,
                num_range_blocks,
                SparseRingBlockEntry::new(position, coefficient_index, 1),
            )?;
        }
        blocks.finish_build(current_block, num_range_blocks)
    }

    /// Validate one runtime ring view and return its ring-element count.
    pub(super) fn validate_ring_dimension(&self, ring_d: usize) -> Result<usize, AkitaError> {
        if ring_d == 0 {
            return Err(AkitaError::InvalidInput(
                "ring_d must be nonzero".to_string(),
            ));
        }
        if ring_d > usize::from(u16::MAX) + 1 {
            return Err(AkitaError::InvalidInput(format!(
                "D={ring_d} exceeds 65536 and cannot be packed into entry coefficient fields"
            )));
        }
        if !(self.onehot_k.is_multiple_of(ring_d) || ring_d.is_multiple_of(self.onehot_k)) {
            return Err(AkitaError::InvalidInput(format!(
                "onehot_k={} and D={ring_d} must be nicely matched (one divides the other)",
                self.onehot_k
            )));
        }
        let field_len = 1usize
            .checked_shl(self.num_vars as u32)
            .ok_or_else(|| AkitaError::InvalidInput("onehot arity overflow".to_string()))?;
        if !field_len.is_multiple_of(ring_d) {
            return Err(AkitaError::InvalidInput(format!(
                "onehot field length {field_len} is not divisible by D={ring_d}"
            )));
        }
        Ok(field_len / ring_d)
    }

    /// Validate a `(ring_d, num_positions_per_block)` view and return
    /// `(ring_elems_at_d, num_live_blocks)`.
    pub(super) fn view_layout(
        &self,
        ring_d: usize,
        num_positions_per_block: usize,
    ) -> Result<(usize, usize), AkitaError> {
        if num_positions_per_block == 0 || !num_positions_per_block.is_power_of_two() {
            return Err(AkitaError::InvalidInput(format!(
                "num_positions_per_block={num_positions_per_block} must be a nonzero power of two"
            )));
        }
        if u32::try_from(num_positions_per_block).is_err() {
            return Err(AkitaError::InvalidInput(format!(
                "num_positions_per_block={num_positions_per_block} exceeds u32::MAX and cannot be packed into an entry"
            )));
        }
        let ring_elems_at_d = self.validate_ring_dimension(ring_d)?;
        Ok((
            ring_elems_at_d,
            ring_elems_at_d.div_ceil(num_positions_per_block),
        ))
    }

    /// Number of live blocks at a `(ring_d, num_positions_per_block)` view,
    /// computed from the layout without building anything.
    pub(crate) fn num_live_blocks_for(
        &self,
        ring_d: usize,
        num_positions_per_block: usize,
    ) -> Result<usize, AkitaError> {
        Ok(self.view_layout(ring_d, num_positions_per_block)?.1)
    }

    pub(super) fn hot_field_position(
        &self,
        chunk_idx: usize,
        raw: I,
        context: &'static str,
    ) -> Result<usize, AkitaError> {
        chunk_idx
            .checked_mul(self.onehot_k)
            .and_then(|base| base.checked_add(raw.as_usize()))
            .ok_or_else(|| AkitaError::InvalidInput(format!("onehot {context} index overflow")))
    }
}
