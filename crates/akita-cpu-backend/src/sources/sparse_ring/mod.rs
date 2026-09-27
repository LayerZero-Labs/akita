use akita_error::AkitaError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SparseRingCoeff {
    flat_idx: u64,
}

impl SparseRingCoeff {
    pub(crate) fn new(flat_idx: usize, value: i8) -> Result<Self, AkitaError> {
        if value != 1 {
            return Err(AkitaError::InvalidInput(
                "one-hot sparse coefficients must be positive units".into(),
            ));
        }
        Ok(Self {
            flat_idx: u64::try_from(flat_idx).map_err(|_| {
                AkitaError::InvalidInput("sparse flat coefficient index exceeds u64".into())
            })?,
        })
    }

    #[inline]
    pub(crate) fn ring_idx(self, ring_d: usize) -> usize {
        (self.flat_idx as usize) / ring_d
    }

    #[inline]
    pub(crate) fn coeff_idx(self, ring_d: usize) -> usize {
        (self.flat_idx as usize) % ring_d
    }
}

/// One sparse signed coefficient inside a ring-position block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SparseRingBlockEntry {
    pos_in_block: u32,
    coeff_idx: u16,
    value: i8,
}

impl SparseRingBlockEntry {
    #[inline]
    pub(crate) fn new(pos_in_block: u32, coeff_idx: u16, value: i8) -> Self {
        Self {
            pos_in_block,
            coeff_idx,
            value,
        }
    }

    #[inline]
    pub(crate) fn pos_in_block(self) -> usize {
        self.pos_in_block as usize
    }

    #[inline]
    pub(crate) fn coeff_idx(self) -> usize {
        self.coeff_idx as usize
    }

    #[inline]
    pub(crate) fn value(self) -> i8 {
        self.value
    }
}
