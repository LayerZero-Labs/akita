use akita_field::AkitaError;
use std::ops::Range;

/// Types usable as one-hot position indices.
///
/// Implemented for `u8`, `u16`, `u32`, and `usize`.
pub trait OneHotIndex: Copy + Send + Sync + std::fmt::Debug + 'static {
    /// Convert to `usize` for indexing.
    fn as_usize(self) -> usize;
}

impl OneHotIndex for u8 {
    #[inline]
    fn as_usize(self) -> usize {
        self as usize
    }
}

impl OneHotIndex for u16 {
    #[inline]
    fn as_usize(self) -> usize {
        self as usize
    }
}

impl OneHotIndex for u32 {
    #[inline]
    fn as_usize(self) -> usize {
        self as usize
    }
}

impl OneHotIndex for usize {
    #[inline]
    fn as_usize(self) -> usize {
        self
    }
}

/// One nonzero one-hot coordinate within a requested ring range.
#[derive(Clone, Copy)]
pub(super) struct OneHotRingEntry {
    pub(super) ring_index: usize,
    pub(super) coefficient_index: usize,
}

/// Allocation-free traversal of canonical one-hot coordinates in a ring range.
pub(super) struct OneHotRingRange<'a, I> {
    onehot_k: usize,
    indices: &'a [Option<I>],
    ring_d: usize,
    ring_range: Range<usize>,
    next_chunk: usize,
    chunk_end: usize,
}

impl<'a, I: OneHotIndex> OneHotRingRange<'a, I> {
    pub(super) fn new(
        onehot_k: usize,
        indices: &'a [Option<I>],
        ring_d: usize,
        ring_range: Range<usize>,
    ) -> Result<Self, AkitaError> {
        if onehot_k == 0 || ring_d == 0 || ring_range.start > ring_range.end {
            return Err(AkitaError::InvalidInput(
                "one hot ring range requires nonzero ordered geometry".into(),
            ));
        }
        let field_start = ring_range
            .start
            .checked_mul(ring_d)
            .ok_or_else(|| AkitaError::InvalidInput("one hot range start overflow".into()))?;
        let field_end = ring_range
            .end
            .checked_mul(ring_d)
            .ok_or_else(|| AkitaError::InvalidInput("one hot range end overflow".into()))?;
        let next_chunk = field_start / onehot_k;
        let chunk_end = field_end.div_ceil(onehot_k).min(indices.len());
        Ok(Self {
            onehot_k,
            indices,
            ring_d,
            ring_range,
            next_chunk,
            chunk_end,
        })
    }

    pub(super) fn entry_capacity(&self) -> usize {
        self.indices[self.next_chunk..self.chunk_end]
            .iter()
            .filter(|entry| entry.is_some())
            .count()
    }
}

impl<I: OneHotIndex> Iterator for OneHotRingRange<'_, I> {
    type Item = Result<OneHotRingEntry, AkitaError>;

    fn next(&mut self) -> Option<Self::Item> {
        while self.next_chunk < self.chunk_end {
            let chunk_index = self.next_chunk;
            self.next_chunk += 1;
            let Some(hot_index) = self.indices[chunk_index] else {
                continue;
            };
            let field_position = match chunk_index
                .checked_mul(self.onehot_k)
                .and_then(|base| base.checked_add(hot_index.as_usize()))
            {
                Some(position) => position,
                None => {
                    return Some(Err(AkitaError::InvalidInput(
                        "one hot field position overflow".into(),
                    )))
                }
            };
            let ring_index = field_position / self.ring_d;
            if self.ring_range.contains(&ring_index) {
                return Some(Ok(OneHotRingEntry {
                    ring_index,
                    coefficient_index: field_position % self.ring_d,
                }));
            }
        }
        None
    }
}
