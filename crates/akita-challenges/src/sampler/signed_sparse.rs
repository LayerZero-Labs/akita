//! Sampler for fixed-weight signed sparse ring fold challenges.
//!
//! Sampling chooses `count_pm1 + count_pm2` distinct positions from `0..D`,
//! assigns `count_pm1` random signs at magnitude 1 and the remaining
//! `count_pm2` at magnitude 2. When `count_pm2 == 0` every non-zero
//! coefficient is ±1.

use crate::sampler::position_sample::{sample_distinct_positions_into, DistinctPositionScratch};
#[cfg(any(target_arch = "riscv64", test))]
use crate::sampler::position_sample::{sample_distinct_positions_into_stack, NarrowSlot};
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
        fill_signed_magnitudes(cursor, count_pm1, &mut self.coeffs);
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

/// Draw one sign byte per coefficient, in chunks of [`SIGN_BYTE_CHUNK`]: the
/// first `count_pm1` coefficients get magnitude 1, the rest magnitude 2.
#[inline]
fn fill_signed_magnitudes(cursor: &mut XofCursor, count_pm1: usize, coeffs: &mut [i8]) {
    #[repr(align(8))]
    struct SignBytes([u8; SIGN_BYTE_CHUNK]);
    let mut sign_bytes = SignBytes([0u8; SIGN_BYTE_CHUNK]);
    for (chunk_index, chunk) in coeffs.chunks_mut(SIGN_BYTE_CHUNK).enumerate() {
        let bytes = &mut sign_bytes.0[..chunk.len()];
        cursor.fill_bytes(bytes);
        for (offset, (coeff, &b)) in chunk.iter_mut().zip(bytes.iter()).enumerate() {
            let magnitude = if chunk_index * SIGN_BYTE_CHUNK + offset < count_pm1 {
                1
            } else {
                2
            };
            *coeff = if b & 1 == 0 { magnitude } else { -magnitude };
        }
    }
}

/// The production D64 family drawn into fixed stack storage, through the same
/// shuffle and sign rule as the general sampler, so every byte consumed matches.
#[cfg(any(target_arch = "riscv64", test))]
pub(super) fn sample_production64(
    cursor: &mut XofCursor,
) -> Result<SparseChallenge, akita_error::AkitaError> {
    const WEIGHT: usize = crate::D64_PRODUCTION_PM1_COUNT + crate::D64_PRODUCTION_PM2_COUNT;
    const _: () = assert!(WEIGHT <= 64);
    #[repr(align(8))]
    struct Positions([u32; WEIGHT]);
    let mut positions = Positions([0u32; WEIGHT]);
    sample_distinct_positions_into_stack::<NarrowSlot, 64>(cursor, 64, &mut positions.0)?;
    let mut coeffs = [0i8; WEIGHT];
    fill_signed_magnitudes(cursor, crate::D64_PRODUCTION_PM1_COUNT, &mut coeffs);
    Ok(SparseChallenge {
        positions: SparseChallengePositions::from_slice(&positions.0),
        coeffs: SparseChallengeCoefficients::from_slice(&coeffs),
    })
}

#[cfg(test)]
mod fixed_tests {
    use super::*;
    use crate::sampler::IndexedXofPrefix;
    #[test]
    fn production64_preserves_challenges_and_xof_consumption() {
        use crate::{D64_PRODUCTION_PM1_COUNT as PM1, D64_PRODUCTION_PM2_COUNT as PM2};
        let prefix = IndexedXofPrefix::new(&[91; 32]).unwrap();
        let mut scratch = SignedSparseScratch::new(PM1, PM2);
        for index in 0..128 {
            let mut general = XofCursor::from_indexed_prefix(&prefix, index);
            let mut fixed = XofCursor::from_indexed_prefix(&prefix, index);
            scratch.sample(&mut general, 64, PM1, PM2).unwrap();
            assert_eq!(
                scratch.take_challenge(),
                sample_production64(&mut fixed).unwrap()
            );
            let (mut a, mut b) = ([0; 300], [0; 300]);
            general.fill_bytes(&mut a);
            fixed.fill_bytes(&mut b);
            assert_eq!(a, b);
        }
    }
}
