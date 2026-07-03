use akita_field::AkitaError;

/// Chunked-witness geometry for the opening-digit trace columns.
///
/// `num_chunks = 1` is the single-chunk (historical) layout: the opening digits
/// form one contiguous `ê` segment at `opening_digit_offset` and
/// [`TraceWeightLayout::opening_digit_col_index`] uses the legacy formula. For
/// `num_chunks = W`, `ê` is partitioned into `W` windows of `blocks_per_chunk`
/// global blocks, each at its chunk's `offset_e` (`chunk·chunk_stride +
/// opening_digit_offset`); the per-window column order matches the distributed
/// prover's `[zᵢ|eᵢ|t̂ᵢ]` emission `(digit, claim, block_local)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TraceChunkLayout {
    /// Number of witness chunks (`1` = single-chunk).
    pub num_chunks: usize,
    /// Global blocks owned by each chunk (`num_blocks_global / num_chunks`).
    pub blocks_per_chunk: usize,
    /// Number of opening claims sharing the block axis.
    pub num_claims: usize,
    /// Per-claim global block count.
    pub num_blocks_global: usize,
    /// Witness ring columns between consecutive chunks (`0` when `num_chunks == 1`).
    pub chunk_stride: usize,
}

impl TraceChunkLayout {
    /// Single-chunk descriptor; the multi-chunk fields are inert.
    pub fn single() -> Self {
        Self {
            num_chunks: 1,
            blocks_per_chunk: 1,
            num_claims: 1,
            num_blocks_global: 1,
            chunk_stride: 0,
        }
    }
}

/// Geometry of the opening-digit column segment inside the stage-2 witness.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TraceWeightLayout {
    pub ring_bits: usize,
    pub col_bits: usize,
    pub opening_digit_offset: usize,
    pub num_blocks: usize,
    pub num_digits_open: usize,
    pub r_vars: usize,
    pub log_basis: u32,
    /// Chunked-witness geometry (default single-chunk).
    pub chunk: TraceChunkLayout,
}

impl TraceWeightLayout {
    pub fn opening_digit_col_count(&self) -> usize {
        self.num_blocks * self.num_digits_open
    }

    pub fn ring_len(&self) -> usize {
        1usize << self.ring_bits
    }

    pub fn table_len(&self) -> Result<usize, AkitaError> {
        1usize
            .checked_shl((self.col_bits + self.ring_bits) as u32)
            .ok_or_else(|| {
                AkitaError::InvalidInput("trace-weight table length overflow".to_string())
            })
    }

    pub fn opening_digit_col_index(&self, block: usize, plane: usize) -> usize {
        if self.chunk.num_chunks <= 1 {
            return self.opening_digit_offset + plane * self.num_blocks + block;
        }
        // Chunked: route the global block to its chunk, then place it inside the
        // chunk's `ê` window (order: digit, claim, block_local).
        let c = &self.chunk;
        let claim = block / c.num_blocks_global;
        let global_block = block % c.num_blocks_global;
        let chunk = global_block / c.blocks_per_chunk;
        let block_local = global_block % c.blocks_per_chunk;
        let chunk_offset_e = chunk * c.chunk_stride + self.opening_digit_offset;
        chunk_offset_e
            + plane * (c.num_claims * c.blocks_per_chunk)
            + claim * c.blocks_per_chunk
            + block_local
    }

    pub fn witness_index(&self, col: usize, ring_coord: usize) -> usize {
        col * self.ring_len() + ring_coord
    }

    pub(crate) fn validate_ring_dimension<const D: usize>(&self) -> Result<(), AkitaError> {
        if self.ring_bits != D.trailing_zeros() as usize {
            return Err(AkitaError::InvalidInput(
                "ring_bits does not match ring dimension".to_string(),
            ));
        }
        Ok(())
    }

    pub(crate) fn validate_opening_digit_segment(&self) -> Result<(), AkitaError> {
        if self.chunk.num_chunks <= 1 {
            let end = self
                .opening_digit_offset
                .checked_add(self.opening_digit_col_count())
                .ok_or_else(|| {
                    AkitaError::InvalidInput("opening-digit segment overflow".to_string())
                })?;
            if end > 1usize << self.col_bits {
                return Err(AkitaError::InvalidInput(
                    "opening-digit segment exceeds column hypercube".to_string(),
                ));
            }
            return Ok(());
        }
        // Chunked: the largest column lives in the last chunk's last plane.
        if self.num_blocks == 0 || self.num_digits_open == 0 {
            return Ok(());
        }
        let max_col = self.opening_digit_col_index(self.num_blocks - 1, self.num_digits_open - 1);
        let end = max_col.checked_add(1).ok_or_else(|| {
            AkitaError::InvalidInput("opening-digit segment overflow".to_string())
        })?;
        if end > 1usize << self.col_bits {
            return Err(AkitaError::InvalidInput(
                "opening-digit segment exceeds column hypercube".to_string(),
            ));
        }
        Ok(())
    }

    pub(crate) fn validate_trace_term_block_range(
        &self,
        block_offset: usize,
        block_span: usize,
    ) -> Result<(), AkitaError> {
        let end = block_offset.checked_add(block_span).ok_or_else(|| {
            AkitaError::InvalidInput("trace term block range overflow".to_string())
        })?;
        if end > self.num_blocks {
            return Err(AkitaError::InvalidInput(
                "trace term exceeds layout block count".to_string(),
            ));
        }
        Ok(())
    }
}
