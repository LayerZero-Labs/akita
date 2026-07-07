//! Sampler for fixed-weight signed sparse ring fold challenges.
//!
//! Sampling chooses `count_pm1 + count_pm2` distinct positions from `0..D`,
//! assigns `count_pm1` random signs at magnitude 1 and the remaining
//! `count_pm2` at magnitude 2. When `count_pm2 == 0` every non-zero
//! coefficient is ±1.

use crate::sampler::position_sample::sample_distinct_positions_into;
use crate::sampler::xof::XofCursor;
use crate::{SparseChallenge, SparseChallengeConfig};

/// Stack chunk for random sign bytes.
const SIGN_BYTE_CHUNK: usize = 64;

/// Reusable buffers for signed-sparse draws (batch loops and rejection sampling).
pub(crate) struct SignedSparseScratch {
    positions: Vec<u32>,
    coeffs: Vec<i8>,
    total: usize,
}

impl SignedSparseScratch {
    pub(crate) fn new(count_pm1: usize, count_pm2: usize) -> Self {
        let total = count_pm1 + count_pm2;
        Self {
            positions: vec![0u32; total],
            coeffs: Vec::with_capacity(total),
            total,
        }
    }

    pub(crate) fn sample_challenges(
        cursor: &mut XofCursor,
        d: usize,
        n: usize,
        cfg: &SparseChallengeConfig,
    ) -> Vec<SparseChallenge> {
        let mut challenges = Vec::with_capacity(n);
        let mut scratch = Self::new(cfg.count_pm1, cfg.count_pm2);
        for _ in 0..n {
            scratch.sample(cursor, d, cfg.count_pm1, cfg.count_pm2);
            challenges.push(scratch.take_challenge());
        }
        challenges
    }

    /// Draw one signed-sparse candidate into the scratch buffers.
    #[inline]
    pub(crate) fn sample(
        &mut self,
        cursor: &mut XofCursor,
        d: usize,
        count_pm1: usize,
        count_pm2: usize,
    ) {
        debug_assert_eq!(self.total, count_pm1 + count_pm2);
        sample_distinct_positions_into(cursor, d, &mut self.positions);
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
    }

    /// Move the accepted draw into an owned [`SparseChallenge`] and reset scratch
    /// storage for the next slot.
    pub(crate) fn take_challenge(&mut self) -> SparseChallenge {
        SparseChallenge {
            positions: std::mem::replace(&mut self.positions, vec![0u32; self.total]),
            coeffs: std::mem::replace(&mut self.coeffs, Vec::with_capacity(self.total)),
        }
    }
}
