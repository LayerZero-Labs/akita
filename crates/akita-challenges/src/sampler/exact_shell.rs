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

/// Largest exact-shell Hamming weight on any shipping ring (`31 + 11` at `D = 64`).
const MAX_STACK_SIGN_BYTES: usize = 64;

/// Sample one [`crate::SparseChallengeConfig::ExactShell`] challenge.
pub(crate) fn sample_exact_shell_challenge(
    cursor: &mut XofCursor,
    d: usize,
    count_mag1: usize,
    count_mag2: usize,
) -> SparseChallenge {
    let total = count_mag1 + count_mag2;
    let mut positions = vec![0u32; total];
    sample_distinct_positions_into(cursor, d, &mut positions);
    let mut coeffs = Vec::with_capacity(total);
    let mut sign_bytes = [0u8; MAX_STACK_SIGN_BYTES];
    cursor.fill_bytes(&mut sign_bytes[..total]);
    for &b in &sign_bytes[..count_mag1] {
        coeffs.push(if (b & 1) == 0 { 1 } else { -1 });
    }
    for &b in &sign_bytes[count_mag1..total] {
        coeffs.push(if (b & 1) == 0 { 2 } else { -2 });
    }
    SparseChallenge { positions, coeffs }
}
