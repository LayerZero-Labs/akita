//! Sampler for [`crate::SparseChallengeConfig::ExactShell`].
//!
//! An exact-shell challenge has a fixed Hamming weight split between magnitude
//! 1 and magnitude 2 coefficients: sampling chooses `count_mag1 + count_mag2`
//! distinct positions, assigns the first `count_mag1` of them random `±1`, and
//! the remaining `count_mag2` random `±2`. The resulting `L1` mass is
//! deterministic at `count_mag1 + 2 * count_mag2`, which is what makes this
//! family attractive for protocol sizing.

use crate::sampler::uniform::{
    sample_distinct_positions_into, sample_distinct_positions_into_general, MAX_STACK_RING_DIM,
};
use crate::sampler::xof::XofCursor;
use crate::SparseChallenge;

/// Sample one [`crate::SparseChallengeConfig::ExactShell`] challenge.
pub(crate) fn sample_exact_shell_challenge(
    cursor: &mut XofCursor,
    d: usize,
    count_mag1: usize,
    count_mag2: usize,
) -> SparseChallenge {
    let total = count_mag1 + count_mag2;
    let mut positions = vec![0u32; total];
    if d <= MAX_STACK_RING_DIM {
        sample_distinct_positions_into(cursor, d, &mut positions);
    } else {
        sample_distinct_positions_into_general(cursor, d, &mut positions);
    }
    let mut coeffs = Vec::with_capacity(total);
    for _ in 0..count_mag1 {
        coeffs.push(cursor.next_sign());
    }
    for _ in 0..count_mag2 {
        coeffs.push(2 * cursor.next_sign());
    }
    SparseChallenge { positions, coeffs }
}
