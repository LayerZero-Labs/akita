//! Sampler for [`crate::SparseChallengeConfig::BoundedL1Ball`] at the
//! production preset `(D=32, M=8, B=121)`.
//!
//! For this fixed `(D, M, B)` triple the bounded-`L1` ball has size
//! `WAYS[D][B] >= 2^128`, so the sampler draws one 128-bit Fiat-Shamir index
//! `r in [0, 2^128)` from the transcript-derived XOF and descends the
//! standard suffix-count DP. The realized distribution is uniform over the
//! lexicographically-first `2^128` valid descent paths through the DP
//! recurrence, which is a `2^128`-element subset of the full bounded-`L1`
//! ball `{ c in Z^D : ||c||_inf <= M and ||c||_1 <= B }`. Every retained
//! outcome appears with probability exactly `1 / 2^128`; outcomes outside
//! the retained subset have probability `0`. See
//! `specs/bounded-l1-sparse-challenge.md` for the full security argument.
//!
//! Two consequences for the implementation:
//!
//! - The top-level draw is a single 16-byte little-endian read from the XOF,
//!   with **no** rejection loop and no modulo reduction.
//! - Every cell the descent ever reads fits in `u128`: only the unstored top
//!   cell `WAYS[32][121] ~= 2^128.133` exceeds `u128`, and the descent always
//!   indexes rows `n <= 31`. The bucket scan therefore runs entirely in
//!   `u128`. Inner-loop sums use [`u128::checked_add`]: if the running
//!   cumulative sum would overflow `u128` (only possible at the very first
//!   descent step, where the buckets sum to `WAYS[D][B] > 2^128`), the
//!   comparison `r < acc + bucket` is automatically true (because
//!   `r < 2^128 <= acc + bucket`), so we select the current coefficient
//!   immediately.
//!
//! This file is preset-only: the only WAYS table that ever exists is the
//! compile-time `(D=32, M=8, B=121)` table baked into `.rodata`. There is no
//! runtime DP, no allocation, and no `(M, B)` plumbing on the hot path.

use crate::sampler::xof::XofCursor;

// --- Production preset constants: (D=32, M=8, B=121) ------------------------

pub(crate) const PRESET_D: usize = 32;
pub(crate) const PRESET_M: usize = 8;
pub(crate) const PRESET_B: usize = 121;

// Storage covers math rows `n = 1..=PRESET_D - 1` (i.e. `1..=31`). Two rows
// at the boundaries are excluded:
//
// - Row `n = 0` is the constant `WAYS[0][b] = 1` (the only completion of
//   zero coordinates is the empty completion), so we don't store it;
//   `ways(0, _)` returns `1` directly.
// - Row `n = D = 32` is never read by the descent, since the bucket scan
//   always indexes `rem_after = D - i - 1 <= 31`. The top cell
//   `WAYS[32][121] ~= 2^128.133` also doesn't fit in `u128`, so storing it
//   would force back to multi-precision integers; not storing it lets the
//   whole table stay `u128`.
//
// Storage row index `s` corresponds to math row `n = s + 1`.
const ROWS: usize = PRESET_D - 1;
const COLS: usize = PRESET_B + 1;
const LEN: usize = ROWS * COLS;

/// Build the suffix-count table `WAYS[n][b] = #{ v in [-M, M]^n : ||v||_1 <= b }`
/// for `n = 1..=PRESET_D - 1` (i.e. `1..=31`), evaluated at compile time as
/// plain `u128`s.
///
/// `u128` is exact on the entire stored range: every final cell satisfies
/// `WAYS[n][b] <= WAYS[31][121] < 2^128`, and the per-row inner-loop sum is
/// a monotonically nondecreasing prefix of nonneg terms whose total equals
/// the final cell, so the running accumulator also stays `< 2^128`. The
/// only cell that exceeds `u128` is `WAYS[32][121] ~= 2^128.133`, and we
/// neither store nor build it; the `top_cell_overflows_u128` test pins that
/// down using the same recurrence and `u128::checked_add`.
///
/// Math row `n = 0` is the all-ones base case and is not stored; it is
/// served by [`ways`] as the constant `1`. Math row `n = 1` has the closed
/// form `1 + 2 * min(M, b)`, which we initialize directly so the DP loop
/// can start at `n = 2`.
const fn const_build_ways_table() -> [u128; LEN] {
    let mut table: [u128; LEN] = [0u128; LEN];

    // Math row `n = 1`: closed form `1 + 2 * min(M, b)`. With every
    // `WAYS[0][.]` equal to 1, the recurrence collapses to a single
    // multiply-by-2 over the reachable magnitudes.
    let mut b = 0usize;
    while b < COLS {
        let max_a = if PRESET_M < b { PRESET_M } else { b };
        table[b] = 1 + 2 * max_a as u128;
        b += 1;
    }

    // Math rows `n = 2..=PRESET_D` via the standard DP recurrence
    //   WAYS[n][b] = WAYS[n-1][b] + sum_{a=1..=min(M, b)} 2 * WAYS[n-1][b - a].
    // Storage row index `s` holds math row `n = s + 1`, so the DP indexes
    // storage row `s - 1` for the previous math row.
    let mut s = 1usize;
    while s < ROWS {
        let prev_row_start = (s - 1) * COLS;
        let row_start = s * COLS;
        let mut b = 0usize;
        while b < COLS {
            let mut acc = table[prev_row_start + b];
            let max_a = if PRESET_M < b { PRESET_M } else { b };
            let mut a = 1usize;
            while a <= max_a {
                let neighbor = table[prev_row_start + (b - a)];
                acc += 2 * neighbor;
                a += 1;
            }
            table[row_start + b] = acc;
            b += 1;
        }
        s += 1;
    }
    table
}

/// Compile-time WAYS table for the production `(D=32, M=8, B=121)` preset,
/// covering math rows `n = 1..=PRESET_D - 1`. Lives in `.rodata` as plain
/// `u128`s; the sampler reads cells directly with no per-call construction
/// and no allocation.
static WAYS: [u128; LEN] = const_build_ways_table();

/// Return `WAYS[n][b]` for `n in 0..=PRESET_D - 1`. The base row `n = 0` is
/// the constant `1` and is not stored; rows `n = 1..=PRESET_D - 1` come
/// from the static `WAYS` table at storage index `(n - 1) * COLS + b`.
#[inline]
fn ways(n: usize, b: usize) -> u128 {
    if n == 0 {
        1
    } else {
        WAYS[(n - 1) * COLS + b]
    }
}

// --- Descent ---------------------------------------------------------------

/// Run the canonical truncated-`2^128` rank-unranking sampler against the
/// preset WAYS table.
///
/// `cursor` must already be seeded from the transcript-derived XOF; the
/// sampler consumes XOF bytes deterministically. The realized distribution
/// is uniform over the lex-first `2^128` valid descent paths of the DP, which
/// is a `2^128`-element subset of the bounded-`L1` ball; see the module docs
/// and `specs/bounded-l1-sparse-challenge.md` for the security argument.
///
/// `positions` and `coeffs` are caller-owned scratch buffers that this
/// function clears and refills, so a batch sampler can reuse one allocation
/// across challenges. On return they hold the sparse representation.
pub(crate) fn sample_bounded_l1_into(
    cursor: &mut XofCursor,
    positions: &mut Vec<u32>,
    coeffs: &mut Vec<i16>,
) {
    positions.clear();
    coeffs.clear();
    let mut budget = PRESET_B;

    // Top-level draw: one 16-byte little-endian `u128` from the XOF, no
    // rejection. The full ball size `WAYS[32][121]` is approximately
    // `2^128.133`; truncating the sampled support to `[0, 2^128)` is exactly
    // the canonical truncation specified in
    // `specs/bounded-l1-sparse-challenge.md` and gives us 128 bits of
    // Fiat-Shamir min-entropy per challenge.
    let mut r: u128 = cursor.next_u128_le();

    for i in 0..PRESET_D {
        if budget == 0 {
            break;
        }
        let rem_after = PRESET_D - i - 1;

        let chosen = scan_buckets_u128_unrank(rem_after, budget, &mut r);
        if chosen != 0 {
            positions.push(i as u32);
            coeffs.push(chosen);
            budget -= chosen.unsigned_abs() as usize;
        }
    }
}

/// Rank-unranking bucket scan in `u128`. Visits coefficient candidates in
/// canonical signed order `-M, ..., -1, 0, 1, ..., M`, finds the bucket
/// containing the current rank `*r`, returns the chosen coefficient, and
/// updates `*r` to the offset *within* that bucket so the next position can
/// continue descending.
///
/// At the very first call (where the per-position bucket totals sum to
/// `WAYS[32][121] > 2^128`) the cumulative `acc + bucket` may overflow
/// `u128`. We use `checked_add`: an overflow proves `acc + bucket > 2^128 >
/// r`, so `r < acc + bucket` is automatically true and we select the current
/// coefficient immediately. After the first selection `r' = r - acc < bucket`
/// fits in `u128`, and every subsequent sub-table bucket is bounded by
/// `WAYS[31][121] < 2^128`, so no further overflow is possible.
#[inline]
fn scan_buckets_u128_unrank(rem_after: usize, budget: usize, r: &mut u128) -> i16 {
    let mut acc: u128 = 0;
    let max_a = PRESET_M.min(budget);

    let mut a: i32 = -(max_a as i32);
    while a <= -1 {
        let mag = a.unsigned_abs() as usize;
        let bucket = ways(rem_after, budget - mag);
        match acc.checked_add(bucket) {
            Some(next) if *r >= next => {
                acc = next;
            }
            _ => {
                *r -= acc;
                return a as i16;
            }
        }
        a += 1;
    }
    let bucket = ways(rem_after, budget);
    match acc.checked_add(bucket) {
        Some(next) if *r >= next => {
            acc = next;
        }
        _ => {
            *r -= acc;
            return 0;
        }
    }
    let mut a: i32 = 1;
    while a <= max_a as i32 {
        let mag = a as usize;
        let bucket = ways(rem_after, budget - mag);
        match acc.checked_add(bucket) {
            Some(next) if *r >= next => {
                acc = next;
            }
            _ => {
                *r -= acc;
                return a as i16;
            }
        }
        a += 1;
    }
    debug_assert!(false, "BoundedL1Ball: unrank scan exhausted buckets");
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ways_base_row_is_all_ones() {
        // Math row n = 0 is not stored; `ways` serves it as a constant.
        // Verify the helper still hands back `1` for every `b`, since the
        // descent reads this row at the last step (`rem_after = 0`).
        for b in 0..COLS {
            assert_eq!(ways(0, b), 1);
        }
    }

    #[test]
    fn ways_row_one_matches_closed_form() {
        // With `WAYS[0][.] = 1` the recurrence collapses to
        //   WAYS[1][b] = 1 + 2 * min(M, b)
        // (one "v_0 = 0" completion plus two completions per reachable
        // magnitude). The const builder initializes math row 1 from this
        // closed form, so this test pins down the shortcut.
        for b in 0..COLS {
            let expected = 1 + 2 * PRESET_M.min(b) as u128;
            assert_eq!(
                ways(1, b),
                expected,
                "ways[1][{b}] should be 1 + 2*min(M, b)",
            );
        }
    }

    #[test]
    fn top_cell_overflows_u128() {
        // The truncated-`2^128` sampling scheme requires the full ball size
        // `WAYS[D][B]` to be at least `2^128`, so that every top-level draw
        // `r in [0, 2^128)` lands on some valid descent path. We don't
        // store row 32 (the descent never reads it), but we can still
        // compute the would-be top cell from row 31 via the recurrence
        //
        //   WAYS[32][121] = WAYS[31][121] + sum_{a=1..=8} 2 * WAYS[31][121 - a]
        //
        // and assert the sum overflows `u128`. This pins down the security-
        // critical "top cell >= 2^128" property without bringing back any
        // multi-precision integer machinery.
        let mut acc: u128 = ways(31, PRESET_B);
        let mut overflowed = false;
        let mut a = 1usize;
        while a <= PRESET_M {
            let neighbor = ways(31, PRESET_B - a);
            // Use `checked_mul` and `checked_add` rather than asserting
            // intermediate fit: the whole point of the test is that this
            // sum overflows `u128`, and we don't know in advance which step
            // tips it over.
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
            "WAYS[32][121] must exceed 2^128; got {acc} without overflow",
        );
    }
}
