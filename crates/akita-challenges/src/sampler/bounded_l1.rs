//! Sampler for [`crate::SparseChallengeConfig::BoundedL1Norm`] at the
//! production preset `(D=32, M=8, B=121)`. Draws a uniform challenge from
//! a `2^128`-element subset of `{ c in Z^32 : ||c||_inf <= 8 && ||c||_1
//! <= 121 }`. See `specs/bounded-l1-sparse-challenge.md` for the security
//! argument.
//!
//! Two main components:
//!
//! - [`sample_bounded_l1_challenge`] — the public sampler. Reads one
//!   128-bit Fiat-Shamir rank from the transcript XOF and unranks it into
//!   a sparse challenge by descending the suffix-count DP one coordinate
//!   at a time.
//! - The compile-time precomputed table [`BOUNDED_L1_SUFFIX_TABLE_32`] (built
//!   by `compute_bounded_l1_suffix_table`) holding the bucket sizes the
//!   descent needs at every step. Lives in `.rodata`; no runtime
//!   construction, allocation, or `(M, B)` plumbing.

use crate::sampler::xof::XofCursor;
use crate::SparseChallenge;

/// `min(a, b)` for `usize` in `const` context. `Ord::min` and `core::cmp::min`
/// are not yet const-stable on `usize`, so we provide our own.
const fn const_min(a: usize, b: usize) -> usize {
    if a < b {
        a
    } else {
        b
    }
}

/// Compute `Table[i][j]` = the number of polynomials of degree
/// `i - 1` whose coefficients lie in `[-COEFFS_BOUND, COEFFS_BOUND]` and
/// satisfy `sum |coeff_k| <= j`.
///
/// The recurrence is
///
/// ```text
///   Table[i][j] = Table[i-1][j] + sum_{a = 1..=min(COEFFS_BOUND, j)}
///                                     2 * Table[i-1][j - a]
/// ```
///
/// the first term is the "new coefficient is 0" case and the sum is the
/// "new coefficient is +/- a" case (the factor `2` is the two signs).
///
/// Base case: `i = 1`, where there are `2 * min(COEFFS_BOUND, j) + 1`
/// degree-0 polynomials with the right sum-of-absolute-values bound (one
/// for the zero polynomial, two per reachable nonzero magnitude).
///
/// We store rows `i in 1..=ROWS` as a 2D array of shape `[ROWS][COLS]`,
/// where storage row `r` holds math row `i = r + 1`. Row `i = 0` (the
/// all-ones base case at zero degree) is not stored.
const fn compute_bounded_l1_suffix_table<
    const COEFFS_BOUND: usize,
    const MAX_L1_NORM: usize,
    const ROWS: usize,
    const COLS: usize,
>() -> [[u128; COLS]; ROWS] {
    assert!(
        COLS == MAX_L1_NORM + 1,
        "compute_bounded_l1_suffix_table: COLS must equal MAX_L1_NORM + 1",
    );

    let mut table: [[u128; COLS]; ROWS] = [[0u128; COLS]; ROWS];

    // Recursion base: row i = 1.
    let mut norm = 0usize;
    while norm <= MAX_L1_NORM {
        let max_mag = const_min(COEFFS_BOUND, norm);
        table[0][norm] = 1 + 2 * max_mag as u128;
        norm += 1;
    }

    // Recursion step: rows i = 2..=ROWS.
    let mut row = 1usize;
    while row < ROWS {
        let mut col = 0usize;
        while col < COLS {
            let mut acc = table[row - 1][col];
            let max_mag = const_min(COEFFS_BOUND, col);
            let mut mag = 1usize;
            while mag <= max_mag {
                let neighbor = table[row - 1][col - mag];
                acc += 2 * neighbor;
                mag += 1;
            }
            table[row][col] = acc;
            col += 1;
        }
        row += 1;
    }
    table
}

// Params for generating the suffix-count table for D = 32, max coefficient
// magnitude 8, and max L1 norm 121. This config provides 128-bit entropy
// for sampling randomness.
pub(crate) const D_32: usize = 32;
pub(crate) const COEFFS_BOUND_32: usize = 8;
pub(crate) const MAX_L1_NORM_32: usize = 121;

const ROWS_32: usize = D_32 - 1;
const COLS_32: usize = MAX_L1_NORM_32 + 1;

static BOUNDED_L1_SUFFIX_TABLE_32: [[u128; COLS_32]; ROWS_32] =
    compute_bounded_l1_suffix_table::<COEFFS_BOUND_32, MAX_L1_NORM_32, ROWS_32, COLS_32>();

/// Sample one bounded-`L1` challenge against the preset table.
pub(crate) fn sample_bounded_l1_challenge(cursor: &mut XofCursor) -> SparseChallenge {
    decode_rank(cursor.next_u128_le())
}

/// Decode a rank `r in [0, 2^128)` into the corresponding sparse challenge.
#[inline]
fn decode_rank(mut r: u128) -> SparseChallenge {
    let mut positions: Vec<u32> = Vec::with_capacity(D_32);
    let mut coeffs: Vec<i8> = Vec::with_capacity(D_32);
    let mut budget = MAX_L1_NORM_32;

    for i in 0..D_32 {
        if budget == 0 {
            break;
        }
        let remaining_coords = D_32 - i - 1;

        let chosen_bucket = find_bucket(remaining_coords, budget, &mut r);
        if chosen_bucket != 0 {
            positions.push(i as u32);
            coeffs.push(chosen_bucket);
            budget -= chosen_bucket.unsigned_abs() as usize;
        }
    }
    SparseChallenge { positions, coeffs }
}

/// Pick the next coefficient `a` for the descent and advance the rank.
///
/// At a descent step we have `remaining_coords` coordinates left to fill
/// and `budget` of the L1 norm still spendable. The valid candidates for
/// the next coefficient are `a in {-M, ..., -1, 0, 1, ..., M}` clipped to
/// `|a| <= budget`. Each candidate `a` "owns" a bucket of ranks of size
/// `count(remaining_coords, budget - |a|)`, i.e. the number of valid
/// completions if we commit to `a`. The buckets are laid out by increasing
/// magnitude `0, -1, +1, -2, +2, ...`, so the rank space splits into one
/// contiguous range per candidate while trying likely small magnitudes first.
///
/// We walk candidates in that order, maintaining `acc` = the cumulative
/// size of buckets already passed. As soon as `*r < acc + bucket_size`
/// for the current `a`, the rank lies in this bucket: we return `a` and
/// update `*r -= acc` so it becomes the offset *within* the chosen bucket
/// (which is exactly the rank for the recursive sub-problem).
///
/// At the very first descent step the bucket totals sum to
/// `count(D, B) > 2^128`, so `acc + bucket_size` can overflow `u128`.
/// `checked_add` returning `None` is treated identically to `*r < next`:
/// an overflow proves `acc + bucket_size > 2^128 > *r`, so the rank is
/// inside this bucket. After the first selection the sub-problem is
/// bounded by `count(D - 1, B) < 2^128` and no further overflow occurs.
#[inline]
fn find_bucket(remaining_coords: usize, budget: usize, r: &mut u128) -> i8 {
    if remaining_coords == ROWS_32 {
        find_bucket_impl::<true>(remaining_coords, budget, r)
    } else {
        find_bucket_impl::<false>(remaining_coords, budget, r)
    }
}

#[inline]
fn find_bucket_impl<const CHECK_OVERFLOW: bool>(
    remaining_coords: usize,
    budget: usize,
    r: &mut u128,
) -> i8 {
    let mut acc: u128 = 0;
    let max_mag = COEFFS_BOUND_32.min(budget);

    let zero_bucket_size = suffix_count(remaining_coords, budget);
    if let Some(a) = try_take_bucket::<CHECK_OVERFLOW>(0, zero_bucket_size, &mut acc, r) {
        return a;
    }

    for mag in 1..=max_mag {
        let bucket_size = suffix_count(remaining_coords, budget - mag);
        let mag = mag as i8;
        if let Some(a) = try_take_bucket::<CHECK_OVERFLOW>(-mag, bucket_size, &mut acc, r) {
            return a;
        }
        if let Some(a) = try_take_bucket::<CHECK_OVERFLOW>(mag, bucket_size, &mut acc, r) {
            return a;
        }
    }
    // Unreachable: the checks above cover every valid candidate exactly once
    // and their bucket sizes sum to `count(remaining_coords + 1, budget)`.
    // Reaching this point means the table or caller invariants are wrong.
    unreachable!("find_bucket: no bucket chosen for rank");
}

#[inline]
fn suffix_count(remaining_coords: usize, budget: usize) -> u128 {
    if remaining_coords == 0 {
        1
    } else {
        BOUNDED_L1_SUFFIX_TABLE_32[remaining_coords - 1][budget]
    }
}

#[inline]
fn try_take_bucket<const CHECK_OVERFLOW: bool>(
    a: i8,
    bucket_size: u128,
    acc: &mut u128,
    r: &mut u128,
) -> Option<i8> {
    if CHECK_OVERFLOW {
        match acc.checked_add(bucket_size) {
            Some(next) if *r >= next => {
                *acc = next;
                None
            }
            _ => {
                *r -= *acc;
                Some(a)
            }
        }
    } else {
        let next = *acc + bucket_size;
        if *r >= next {
            *acc = next;
            None
        } else {
            *r -= *acc;
            Some(a)
        }
    }
}

#[cfg(all(test, not(feature = "zk")))]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn suffix_count_row_one_matches_closed_form() {
        // count(1, b) = 1 + 2 * min(M, b),
        for b in 0..COLS_32 {
            let expected = 1 + 2 * COEFFS_BOUND_32.min(b) as u128;
            assert_eq!(
                BOUNDED_L1_SUFFIX_TABLE_32[0][b], expected,
                "count(1, {b}) should be 1 + 2*min(M, b)",
            );
        }
    }

    #[test]
    fn recurrence_step_matches_construction() {
        // Storage row `r >= 1` should equal the DP step applied to row
        // `r - 1`:
        //   Table[r][c] = Table[r-1][c]
        //              + sum_{a=1..=min(M, c)} 2 * Table[r-1][c - a]
        // Combined with `suffix_count_row_one_matches_closed_form` (which
        // pins storage row 0), this exhaustively re-derives every cell
        // and proves the const builder followed the recurrence.
        for r in 1..ROWS_32 {
            for c in 0..COLS_32 {
                let max_mag = COEFFS_BOUND_32.min(c);
                let mut expected = BOUNDED_L1_SUFFIX_TABLE_32[r - 1][c];
                for a in 1..=max_mag {
                    expected += 2 * BOUNDED_L1_SUFFIX_TABLE_32[r - 1][c - a];
                }
                assert_eq!(
                    BOUNDED_L1_SUFFIX_TABLE_32[r][c], expected,
                    "Table[{r}][{c}] does not match the DP recurrence",
                );
            }
        }
    }

    #[test]
    fn top_cell_overflows_u128() {
        // The truncated-`2^128` sampler requires `count(D, B) >= 2^128` so
        // that every top-level draw `r in [0, 2^128)` lands on some valid
        // descent path. Row 32 is not stored, but we can reconstruct
        // `count(32, 121)` from row 31 via
        //
        //   count(32, 121) = count(31, 121)
        //                  + sum_{a=1..=8} 2 * count(31, 121 - a)
        //
        // and assert the sum overflows `u128`. We use `checked_mul` /
        // `checked_add` because we don't know in advance which step tips
        // it over.
        let mut acc: u128 = BOUNDED_L1_SUFFIX_TABLE_32[30][MAX_L1_NORM_32];
        let mut overflowed = false;
        let mut a = 1usize;
        while a <= COEFFS_BOUND_32 {
            let neighbor = BOUNDED_L1_SUFFIX_TABLE_32[30][MAX_L1_NORM_32 - a];
            let doubled = match neighbor.checked_mul(2) {
                Some(v) => v,
                None => {
                    overflowed = true;
                    break;
                }
            };
            match acc.checked_add(doubled) {
                Some(next) => acc = next,
                None => {
                    overflowed = true;
                    break;
                }
            }
            a += 1;
        }
        assert!(
            overflowed,
            "count(32, 121) must exceed 2^128; got {acc} without overflow",
        );
    }

    #[test]
    fn decode_rank_is_injective() {
        // Distinct ranks must decode to distinct challenges, otherwise the
        // realized distribution is not uniform on the truncated-2^128 set
        // of descent paths. We can't enumerate all 2^128 ranks, so we
        // densely probe both endpoints of the rank space.
        const N: u128 = 4096;
        let mut seen: HashSet<(Vec<u32>, Vec<i8>)> = HashSet::with_capacity(2 * N as usize);

        for i in 0..N {
            let c = decode_rank(i);
            assert!(
                seen.insert((c.positions, c.coeffs)),
                "low-rank rank {i} collided with an earlier challenge",
            );
        }
        for i in 0..N {
            let r = u128::MAX - i;
            let c = decode_rank(r);
            assert!(
                seen.insert((c.positions, c.coeffs)),
                "high-rank rank {r} (i = {i}) collided with an earlier challenge",
            );
        }
        assert_eq!(seen.len() as u128, 2 * N);
    }
}
