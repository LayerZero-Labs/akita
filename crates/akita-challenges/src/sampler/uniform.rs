//! Sampler for [`crate::SparseChallengeConfig::Uniform`].
//!
//! A uniform sparse challenge has a fixed Hamming weight `w`: sampling chooses
//! `w` distinct positions from `0..D` and assigns each a coefficient drawn
//! uniformly from a small alphabet `nonzero_coeffs`.
//!
//! This module also hosts the position-shuffle helpers used by both the
//! uniform and exact-shell families. The bounded-`L1` family does not draw
//! distinct positions; it descends a DP table where positions emerge
//! implicitly.

use crate::sampler::xof::XofCursor;
use crate::SparseChallenge;

/// Max ring dimension supported by the stack-buffer sampling paths.
///
/// Public API functions reject `D` above this with an error before reaching
/// the sampling internals.
pub(crate) const MAX_STACK_RING_DIM: usize = 512;

/// Largest stack array used by one concrete sampler tier.
const MAX_STACK_TIER_RING_DIM: usize = 512;

/// Fisher-Yates partial shuffle: sample `out.len()` distinct values from
/// `0..universe` into `out`.
///
/// Uses one of the fixed stack-buffer tiers to avoid per-call heap allocation
/// for every supported ring dimension.
///
/// # Safety contract
///
/// Caller must ensure `universe <= MAX_STACK_RING_DIM`. The public API
/// enforces this via a fallible check that returns `Err` instead of
/// panicking.
#[inline]
pub(crate) fn sample_distinct_positions_into(
    cursor: &mut XofCursor,
    universe: usize,
    out: &mut [u32],
) {
    debug_assert!(out.len() <= universe);
    debug_assert!(universe <= MAX_STACK_RING_DIM);
    match universe {
        0..=128 => sample_distinct_positions_into_stack::<128>(cursor, universe, out),
        129..=256 => sample_distinct_positions_into_stack::<256>(cursor, universe, out),
        257..=MAX_STACK_TIER_RING_DIM => {
            sample_distinct_positions_into_stack::<MAX_STACK_TIER_RING_DIM>(cursor, universe, out)
        }
        _ => unreachable!("ring dimension must be <= MAX_STACK_RING_DIM"),
    }
}

#[inline]
fn sample_distinct_positions_into_stack<const N: usize>(
    cursor: &mut XofCursor,
    universe: usize,
    out: &mut [u32],
) {
    debug_assert!(out.len() <= universe);
    debug_assert!(universe <= N);
    let mut perm = [0usize; N];
    for (i, slot) in perm[..universe].iter_mut().enumerate() {
        *slot = i;
    }
    for (i, dst) in out.iter_mut().enumerate() {
        let j = i + cursor.next_usize_mod(universe - i);
        perm.swap(i, j);
        *dst = perm[i] as u32;
    }
}

/// Sample one [`crate::SparseChallengeConfig::Uniform`] challenge.
pub(crate) fn sample_uniform_challenge(
    cursor: &mut XofCursor,
    d: usize,
    weight: usize,
    nonzero_coeffs: &[i8],
) -> SparseChallenge {
    let mut positions = vec![0u32; weight];
    sample_distinct_positions_into(cursor, d, &mut positions);
    let coeffs = (0..weight)
        .map(|_| {
            let coeff_idx = cursor.next_usize_mod(nonzero_coeffs.len());
            nonzero_coeffs[coeff_idx]
        })
        .collect();
    SparseChallenge { positions, coeffs }
}
