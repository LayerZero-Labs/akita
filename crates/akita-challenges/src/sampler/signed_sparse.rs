//! Sampler for fixed-weight signed sparse ring fold challenges.
//!
//! Sampling chooses `count_pm1 + count_pm2` distinct positions from `0..D`,
//! assigns `count_pm1` random signs at magnitude 1 and the remaining
//! `count_pm2` at magnitude 2. When `count_pm2 == 0` every non-zero
//! coefficient is ±1.

use crate::sampler::position_sample::{sample_distinct_positions_into, DistinctPositionScratch};
use crate::sampler::xof::XofCursor;
use crate::{SparseChallenge, SparseChallengeCoefficients, SparseChallengePositions};

/// Stack chunk for random sign bytes.
const SIGN_BYTE_CHUNK: usize = 64;

/// Reusable buffers for signed-sparse draws (batch loops and rejection sampling).
pub(crate) struct SignedSparseScratch {
    positions: Vec<u32>,
    coeffs: Vec<i8>,
    position_scratch: DistinctPositionScratch,
    total: usize,
}

impl SignedSparseScratch {
    pub(crate) fn new(count_pm1: usize, count_pm2: usize) -> Self {
        let total = count_pm1 + count_pm2;
        Self {
            positions: vec![0u32; total],
            coeffs: Vec::with_capacity(total),
            position_scratch: DistinctPositionScratch::new(),
            total,
        }
    }

    /// Draw one signed-sparse candidate into the scratch buffers.
    #[inline]
    pub(crate) fn sample(
        &mut self,
        cursor: &mut XofCursor,
        d: usize,
        count_pm1: usize,
        count_pm2: usize,
    ) -> Result<(), akita_error::AkitaError> {
        debug_assert_eq!(self.total, count_pm1 + count_pm2);
        sample_distinct_positions_into(cursor, d, &mut self.positions, &mut self.position_scratch)?;
        self.coeffs.resize(self.total, 0);
        let mut sign_bytes = [0u8; SIGN_BYTE_CHUNK];
        let mut written = 0;
        while written < self.total {
            let take = (self.total - written).min(SIGN_BYTE_CHUNK);
            cursor.fill_bytes(&mut sign_bytes[..take]);
            for (offset, &b) in sign_bytes[..take].iter().enumerate() {
                let i = written + offset;
                let magnitude = if i < count_pm1 { 1 } else { 2 };
                self.coeffs[i] = if (b & 1) == 0 { magnitude } else { -magnitude };
            }
            written += take;
        }
        Ok(())
    }

    /// Copy the accepted draw into inline challenge storage. The scratch
    /// vectors retain their allocation for the next slot.
    pub(crate) fn take_challenge(&self) -> SparseChallenge {
        SparseChallenge {
            positions: SparseChallengePositions::from_slice(&self.positions),
            coeffs: SparseChallengeCoefficients::from_slice(&self.coeffs),
        }
    }

    pub(crate) fn positions(&self) -> &[u32] {
        &self.positions
    }

    pub(crate) fn coeffs(&self) -> &[i8] {
        &self.coeffs
    }
}

// The production D64 family has fixed stack storage and two rejection-mask
// ranges. This preserves every byte consumed by the general sampler.
#[cfg(any(target_arch = "riscv64", test))]
pub(super) fn sample_production64(cursor: &mut XofCursor) -> SparseChallenge {
    const WEIGHT: usize = crate::D64_PRODUCTION_PM1_COUNT + crate::D64_PRODUCTION_PM2_COUNT;
    let mut permutation: [usize; 64] = std::array::from_fn(|i| i);
    #[repr(align(8))]
    struct Positions([u32; WEIGHT]);
    let mut positions = Positions([0u32; WEIGHT]);
    for (i, dst) in positions.0.iter_mut().enumerate() {
        let j = i + cursor.next_usize_masked(64 - i, if i < 32 { 63 } else { 31 });
        permutation.swap(i, j);
        *dst = permutation[i] as u32;
    }

    #[repr(align(8))]
    struct Signs([u8; WEIGHT]);
    let mut signs = Signs([0u8; WEIGHT]);
    cursor.fill_bytes(&mut signs.0);
    let coeffs: [i8; WEIGHT] = std::array::from_fn(|i| {
        let magnitude = if i < crate::D64_PRODUCTION_PM1_COUNT {
            1
        } else {
            2
        };
        if signs.0[i] & 1 == 0 {
            magnitude
        } else {
            -magnitude
        }
    });

    SparseChallenge {
        positions: SparseChallengePositions::from_slice(&positions.0),
        coeffs: SparseChallengeCoefficients::from_slice(&coeffs),
    }
}
#[cfg(test)]
mod fixed_tests {
    use super::*;
    use crate::sampler::IndexedXofPrefix;
    #[test]
    fn production64_preserves_challenges_and_xof_consumption() {
        let prefix = IndexedXofPrefix::new(&[91; 32]).unwrap();
        let mut scratch = SignedSparseScratch::new(31, 10);
        for index in 0..128 {
            let mut general = XofCursor::from_indexed_prefix(&prefix, index);
            let mut fixed = XofCursor::from_indexed_prefix(&prefix, index);
            scratch.sample(&mut general, 64, 31, 10).unwrap();
            assert_eq!(scratch.take_challenge(), sample_production64(&mut fixed));
            let (mut a, mut b) = ([0; 300], [0; 300]);
            general.fill_bytes(&mut a);
            fixed.fill_bytes(&mut b);
            assert_eq!(a, b);
        }
    }
}
