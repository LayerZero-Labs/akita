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

/// Stack chunk for exact-shell sign bytes.
///
/// Production D=64 shells fit in one chunk. Larger valid shells are read in
/// multiple chunks so the sampler remains allocation-free for sign decoding.
const SIGN_BYTE_CHUNK: usize = 64;

/// Reusable buffers for exact-shell draws (batch loops and rejection sampling).
pub(crate) struct ExactShellScratch {
    positions: Vec<u32>,
    coeffs: Vec<i8>,
    total: usize,
}

impl ExactShellScratch {
    pub(crate) fn new(count_mag1: usize, count_mag2: usize) -> Self {
        let total = count_mag1 + count_mag2;
        Self {
            positions: vec![0u32; total],
            coeffs: Vec::with_capacity(total),
            total,
        }
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
        let mut sign_bytes = [0u8; SIGN_BYTE_CHUNK];
        let mut written = 0;
        while written < self.total {
            let take = (self.total - written).min(SIGN_BYTE_CHUNK);
            cursor.fill_bytes(&mut sign_bytes[..take]);
            for (offset, &b) in sign_bytes[..take].iter().enumerate() {
                let i = written + offset;
                let magnitude = if i < count_mag1 { 1 } else { 2 };
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
