//! Sampler for fixed-weight signed sparse ring fold challenges.
//!
//! Sampling chooses `count_pm1 + count_pm2` distinct positions from `0..D`,
//! assigns `count_pm1` random signs at magnitude 1 and the remaining
//! `count_pm2` at magnitude 2. When `count_pm2 == 0` every non-zero
//! coefficient is ±1.

use crate::sampler::position_sample::{
    sample_distinct_positions_into, shuffle_positions, DistinctPositionScratch, LargeSlot,
};
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
            for (offset, &byte) in sign_bytes[..take].iter().enumerate() {
                let i = written + offset;
                self.coeffs[i] = signed_coefficient(i, count_pm1, byte);
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

#[repr(align(8))]
struct Aligned<T>(T);

/// Canonical sign/magnitude decoding shared by fixed and dynamic buffers.
#[inline(always)]
fn signed_coefficient(index: usize, count_pm1: usize, sign_byte: u8) -> i8 {
    let magnitude = if index < count_pm1 { 1 } else { 2 };
    if sign_byte & 1 == 0 {
        magnitude
    } else {
        -magnitude
    }
}

/// Fixed stack output for any statically known family. The shared dense
/// shuffle retains the D64 mask shortcut. Dispatch keeps very sparse families
/// on reusable virtual scratch rather than initializing D dense slots.
#[inline]
pub(super) fn sample_fixed<const D: usize, const PM1: usize, const WEIGHT: usize>(
    cursor: &mut XofCursor,
) -> SparseChallenge {
    const {
        assert!(PM1 <= WEIGHT && WEIGHT <= D && D <= super::MAX_STACK_RING_DIM);
    }
    let mut positions = Aligned([0u32; WEIGHT]);
    // D is bounded at compile time, so every index fits even native u16 slots.
    let mut permutation: [LargeSlot; D] = std::array::from_fn(|i| i as LargeSlot);
    shuffle_positions(cursor, &mut positions.0, &mut permutation);
    let mut signs = Aligned([0u8; WEIGHT]);
    cursor.fill_bytes(&mut signs.0);
    let coeffs: [i8; WEIGHT] = std::array::from_fn(|i| signed_coefficient(i, PM1, signs.0[i]));
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
        let mut scratch = SignedSparseScratch::new(
            crate::D64_PRODUCTION_PM1_COUNT,
            crate::D64_PRODUCTION_PM2_COUNT,
        );
        for index in 0..128 {
            let mut general = XofCursor::from_indexed_prefix(&prefix, index);
            let mut fixed = XofCursor::from_indexed_prefix(&prefix, index);
            scratch
                .sample(
                    &mut general,
                    64,
                    crate::D64_PRODUCTION_PM1_COUNT,
                    crate::D64_PRODUCTION_PM2_COUNT,
                )
                .unwrap();
            assert_eq!(
                scratch.take_challenge(),
                sample_fixed::<
                    64,
                    { crate::D64_PRODUCTION_PM1_COUNT },
                    { crate::D64_PRODUCTION_PM1_COUNT + crate::D64_PRODUCTION_PM2_COUNT },
                >(&mut fixed)
            );
            let (mut a, mut b) = ([0; 300], [0; 300]);
            general.fill_bytes(&mut a);
            fixed.fill_bytes(&mut b);
            assert_eq!(a, b);
        }
    }
}
