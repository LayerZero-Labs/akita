//! Distinct-position sampling for sparse ring fold challenges.
//!
//! Fisher-Yates partial shuffle helpers shared by the signed-sparse sampler.

use crate::sampler::xof::XofCursor;

/// Max ring dimension supported by the stack-buffer sampling paths.
///
/// Public API functions reject `D` above this with an error before reaching
/// the sampling internals.
pub(crate) const MAX_STACK_RING_DIM: usize = 2048;

/// Largest stack array used by one concrete sampler tier.
const MAX_STACK_TIER_RING_DIM: usize = 2048;

/// Fisher-Yates partial shuffle: sample `out.len()` distinct values from
/// `0..universe` into `out`.
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
        257..=512 => sample_distinct_positions_into_stack::<512>(cursor, universe, out),
        513..=1024 => sample_distinct_positions_into_stack::<1024>(cursor, universe, out),
        1025..=MAX_STACK_TIER_RING_DIM => {
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
