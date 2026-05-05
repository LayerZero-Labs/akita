//! Sampler for [`crate::SparseChallengeConfig::BoundedL1Ball`] at the
//! production preset `(D=32, M=8, B=121)`.
//!
//! For this fixed `(D, M, B)` triple the bounded-`L1` ball has size
//! `count(D, B) >= 2^128`, where `count(n, b) = #{ v in [-M, M]^n : ||v||_1
//! <= b }`. The sampler draws one 128-bit Fiat-Shamir index `r in [0, 2^128)`
//! from the transcript-derived XOF and descends the standard suffix-count
//! DP. The realized distribution is uniform over the lexicographically-first
//! `2^128` valid descent paths through the DP recurrence, which is a
//! `2^128`-element subset of the full bounded-`L1` ball
//! `{ c in Z^D : ||c||_inf <= M and ||c||_1 <= B }`. Every retained outcome
//! appears with probability exactly `1 / 2^128`; outcomes outside the
//! retained subset have probability `0`. See
//! `specs/bounded-l1-sparse-challenge.md` for the full security argument.
//!
//! Two consequences for the implementation:
//!
//! - The top-level draw is a single 16-byte little-endian read from the XOF,
//!   with **no** rejection loop and no modulo reduction.
//! - Every cell the descent ever reads fits in `u128`: only the unstored top
//!   cell `count(32, 121) ~= 2^128.133` exceeds `u128`, and the descent
//!   always indexes rows `n <= 31`. The bucket scan therefore runs entirely
//!   in `u128`. Inner-loop sums use [`u128::checked_add`]: if the running
//!   cumulative sum would overflow `u128` (only possible at the very first
//!   descent step, where the buckets sum to `count(D, B) > 2^128`), the
//!   comparison `r < acc + bucket` is automatically true (because
//!   `r < 2^128 <= acc + bucket`), so we select the current coefficient
//!   immediately.
//!
//! This file is preset-only: the only suffix-count table that ever exists is
//! the compile-time `(D=32, M=8, B=121)` table baked into `.rodata`. There
//! is no runtime DP, no allocation, and no `(M, B)` plumbing on the hot
//! path.

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
///
/// `COLS` must equal `MAX_L1_NORM + 1`; any other value triggers a
/// const-time panic. Stable Rust's const generics don't permit arithmetic
/// in the array-length type, so the caller computes the dimensions and
/// passes them as the last two parameters.
///
/// `u128` is exact for every stored cell provided the caller picks
/// `(COEFFS_BOUND, MAX_L1_NORM, ROWS)` so that `Table[ROWS][MAX_L1_NORM] <
/// 2^128`. The per-row accumulator is monotonically nondecreasing and
/// dominated by the final cell, so it stays inside the same bound.
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
const PRESET_D: usize = 32;
const PRESET_M: usize = 8;
const PRESET_B: usize = 121;

const D32_ROWS: usize = PRESET_D - 1;
const D32_COLS: usize = PRESET_B + 1;

static BOUNDED_L1_SUFFIX_TABLE: [[u128; D32_COLS]; D32_ROWS] =
    compute_bounded_l1_suffix_table::<PRESET_M, PRESET_B, D32_ROWS, D32_COLS>();

/// Sample one bounded-`L1` challenge against the preset table.
pub(crate) fn sample_bounded_l1_sparse(cursor: &mut XofCursor) -> SparseChallenge {
    let mut positions: Vec<u32> = Vec::with_capacity(PRESET_D);
    let mut coeffs: Vec<i8> = Vec::with_capacity(PRESET_D);
    let mut budget = PRESET_B;
    let mut r: u128 = cursor.next_u128_le();

    for i in 0..PRESET_D {
        if budget == 0 {
            break;
        }
        let remaining_coords = PRESET_D - i - 1;

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
/// completions if we commit to `a`. The buckets are laid out in canonical
/// signed order `-M, ..., -1, 0, 1, ..., M`, so the rank space splits
/// into one contiguous range per candidate.
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
    let mut acc: u128 = 0;
    let max_mag = PRESET_M.min(budget) as i8;

    for a in -max_mag..=max_mag {
        let mag = a.unsigned_abs() as usize;
        let bucket_size = if remaining_coords == 0 {
            1
        } else {
            BOUNDED_L1_SUFFIX_TABLE[remaining_coords - 1][budget - mag]
        };
        match acc.checked_add(bucket_size) {
            Some(next) if *r >= next => {
                acc = next;
            }
            _ => {
                *r -= acc;
                return a;
            }
        }
    }
    // Unreachable: the loop above iterates over every valid candidate
    // `a in {-max_mag, ..., +max_mag}` and each owns a non-empty bucket
    // covering `count(remaining_coords, budget - |a|) >= 1` ranks. Their
    // sizes sum to `count(remaining_coords + 1, budget) >= *r + 1` (true
    // by induction from the truncated-`2^128` precondition), so some
    // candidate's bucket must have contained `*r` and the loop must have
    // returned. Reaching this point means we walked the whole range
    // without picking any bucket, which is a bug in the table or the
    // caller's `(remaining_coords, budget, r)` invariants.
    unreachable!("find_bucket: no bucket chosen for rank");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn suffix_count_row_one_matches_closed_form() {
        // count(1, b) = 1 + 2 * min(M, b). Pins the const builder's
        // closed-form initialization of math row 1.
        for b in 0..D32_COLS {
            let expected = 1 + 2 * PRESET_M.min(b) as u128;
            assert_eq!(
                BOUNDED_L1_SUFFIX_TABLE[0][b], expected,
                "count(1, {b}) should be 1 + 2*min(M, b)",
            );
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
        let mut acc: u128 = BOUNDED_L1_SUFFIX_TABLE[30][PRESET_B];
        let mut overflowed = false;
        let mut a = 1usize;
        while a <= PRESET_M {
            let neighbor = BOUNDED_L1_SUFFIX_TABLE[30][PRESET_B - a];
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
}
