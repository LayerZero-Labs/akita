//! Sampler for [`crate::SparseChallengeConfig::ExactShell`].
//!
//! An exact-shell challenge has a fixed Hamming weight split between magnitude
//! 1 and magnitude 2 coefficients: sampling chooses `count_mag1 + count_mag2`
//! distinct positions, assigns the first `count_mag1` of them random `±1`, and
//! the remaining `count_mag2` random `±2`. The resulting `L1` mass is
//! deterministic at `count_mag1 + 2 * count_mag2`, which is what makes this
//! family attractive for protocol sizing.

use crate::sampler::uniform::sample_distinct_positions_into;
use crate::sampler::xof::XofCursor;
use crate::SparseChallenge;

/// Reusable buffers for exact-shell draws (batch loops and rejection sampling).
pub(crate) struct ExactShellScratch {
    positions: Vec<u32>,
    coeffs: Vec<i8>,
    sign_bytes: Vec<u8>,
    total: usize,
}

impl ExactShellScratch {
    pub(crate) fn new(count_mag1: usize, count_mag2: usize) -> Self {
        let total = count_mag1 + count_mag2;
        Self {
            positions: vec![0u32; total],
            coeffs: Vec::with_capacity(total),
            sign_bytes: Vec::with_capacity(total),
            total,
        }
    }

    #[inline]
    pub(crate) fn positions(&self) -> &[u32] {
        &self.positions
    }

    #[inline]
    pub(crate) fn coeffs(&self) -> &[i8] {
        &self.coeffs
    }

    /// Draw one exact-shell candidate into the scratch buffers.
    #[inline]
    pub(crate) fn sample(
        &mut self,
        cursor: &mut XofCursor,
        d: usize,
        count_mag1: usize,
        count_mag2: usize,
    ) {
        debug_assert_eq!(self.total, count_mag1 + count_mag2);
        sample_distinct_positions_into(cursor, d, &mut self.positions);
        self.coeffs.resize(self.total, 0);
        self.sign_bytes.resize(self.total, 0);
        cursor.fill_bytes(&mut self.sign_bytes);
        for (i, &b) in self.sign_bytes[..count_mag1].iter().enumerate() {
            self.coeffs[i] = if (b & 1) == 0 { 1 } else { -1 };
        }
        for (i, &b) in self.sign_bytes[count_mag1..self.total].iter().enumerate() {
            self.coeffs[count_mag1 + i] = if (b & 1) == 0 { 2 } else { -2 };
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

/// Sample one [`crate::SparseChallengeConfig::ExactShell`] challenge.
pub(crate) fn sample_exact_shell_challenge(
    cursor: &mut XofCursor,
    d: usize,
    count_mag1: usize,
    count_mag2: usize,
) -> SparseChallenge {
    let mut scratch = ExactShellScratch::new(count_mag1, count_mag2);
    scratch.sample(cursor, d, count_mag1, count_mag2);
    scratch.take_challenge()
}
