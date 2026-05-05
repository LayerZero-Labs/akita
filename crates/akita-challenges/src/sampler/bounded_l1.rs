//! Sampler for [`crate::SparseChallengeConfig::BoundedL1Ball`] at the
//! production preset `(D=32, M=8, B=121)`.
//!
//! For this fixed `(D, M, B)` triple `WAYS[D][B] >= 2^128`, so the sampler
//! draws one 128-bit Fiat-Shamir index `r in [0, 2^128)` from the
//! transcript-derived XOF and descends the standard suffix-count DP. The
//! realized distribution is uniform over the lexicographically-first `2^128`
//! valid descent paths through the DP recurrence, which is a `2^128`-element
//! subset of the full bounded-`L1` ball
//! `{ c in Z^D : ||c||_inf <= M and ||c||_1 <= B }`. Every retained outcome
//! appears with probability exactly `1 / 2^128`; outcomes outside the retained
//! subset have probability `0`. See `specs/bounded-l1-sparse-challenge.md` for
//! the full security argument.
//!
//! Two consequences for the implementation:
//!
//! - The top-level draw is a single 16-byte little-endian read from the XOF,
//!   with **no** rejection loop and no modulo reduction.
//! - Each per-coefficient bucket scan runs entirely in `u128` arithmetic.
//!   Inner-loop sums use [`u128::checked_add`]: if the running cumulative sum
//!   would overflow `u128` (only possible at the very first descent step,
//!   where the buckets sum to `WAYS[D][B] > 2^128`), the comparison
//!   `r < acc + bucket` is automatically true (because `r < 2^128 <= acc +
//!   bucket`), so we select the current coefficient immediately.
//!
//! This file is preset-only: the only WAYS table that ever exists is the
//! compile-time `(D=32, M=8, B=121)` table baked into `.rodata`. There is no
//! runtime DP, no allocation, and no `(M, B)` plumbing on the hot path.

use crate::sampler::xof::XofCursor;

// --- 256-bit `Wide` integer -------------------------------------------------

/// 256-bit little-endian unsigned integer used only by the const WAYS-table
/// builder. The low half is `lo`, the high half is `hi`; the value is
/// `lo + (hi << 128)`. Only the top cell `WAYS[32][121]` exceeds `u128`; every
/// cell consumed by the descent fits in `u128`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct Wide {
    lo: u128,
    hi: u128,
}

impl Wide {
    const ZERO: Self = Self { lo: 0, hi: 0 };
    const ONE: Self = Self { lo: 1, hi: 0 };

    /// Checked add. Returns `None` on 256-bit overflow.
    #[inline]
    const fn checked_add(self, rhs: Self) -> Option<Self> {
        let (lo, c1) = self.lo.overflowing_add(rhs.lo);
        let (hi1, c2) = self.hi.overflowing_add(rhs.hi);
        let (hi, c3) = hi1.overflowing_add(c1 as u128);
        if c2 || c3 {
            None
        } else {
            Some(Self { lo, hi })
        }
    }

    /// Add `rhs` to `self`. Panics on 256-bit overflow. Const-friendly.
    #[inline]
    const fn add_or_panic(self, rhs: Self) -> Self {
        match self.checked_add(rhs) {
            Some(v) => v,
            None => panic!("Wide overflow"),
        }
    }
}

// --- Production preset constants: (D=32, M=8, B=121) ------------------------

pub(crate) const PRESET_D: usize = 32;
pub(crate) const PRESET_M: usize = 8;
pub(crate) const PRESET_B: usize = 121;

// Storage covers math rows `n = 1..=PRESET_D`. The base row `n = 0` is the
// constant `WAYS[0][b] = 1` (the only completion of zero coordinates is the
// empty completion), so we don't store it; `ways(0, _)` returns `Wide::ONE`
// directly. Storage row index `s` corresponds to math row `n = s + 1`.
const ROWS: usize = PRESET_D;
const COLS: usize = PRESET_B + 1;
const LEN: usize = ROWS * COLS;

/// Build the suffix-count table `WAYS[n][b] = #{ v in [-M, M]^n : ||v||_1 <= b }`
/// for `n = 1..=PRESET_D`, evaluated at compile time.
///
/// Math row `n = 0` is the all-ones base case and is not stored; it is served
/// by [`ways`] as a constant. Math row `n = 1` has the closed form
/// `1 + 2 * min(M, b)`, which we initialize directly so the DP loop can start
/// at `n = 2`.
const fn const_build_ways_table() -> [Wide; LEN] {
    let mut table: [Wide; LEN] = [Wide::ZERO; LEN];

    // Math row `n = 1`: closed form `1 + 2 * min(M, b)`. With every
    // `WAYS[0][.]` equal to 1, the recurrence collapses to a single
    // multiply-by-2 over the reachable magnitudes.
    let mut b = 0usize;
    while b < COLS {
        let max_a = if PRESET_M < b { PRESET_M } else { b };
        let count = 1 + 2 * max_a as u128;
        table[b] = Wide { lo: count, hi: 0 };
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
                let doubled = neighbor.add_or_panic(neighbor);
                acc = acc.add_or_panic(doubled);
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
/// covering math rows `n = 1..=PRESET_D`. Lives in `.rodata`; the sampler
/// reads cells directly with no per-call construction and no allocation.
static WAYS: [Wide; LEN] = const_build_ways_table();

/// Return `WAYS[n][b]`. The base row `n = 0` is the constant `Wide::ONE` and
/// is not stored; rows `n = 1..=PRESET_D` come from the static `WAYS` table
/// at storage index `(n - 1) * COLS + b`.
#[inline]
fn ways(n: usize, b: usize) -> Wide {
    if n == 0 {
        Wide::ONE
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
        let bucket = ways(rem_after, budget - mag).lo;
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
    let bucket = ways(rem_after, budget).lo;
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
        let bucket = ways(rem_after, budget - mag).lo;
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

    /// `floor(log2(w)) + 1` for a `Wide`; only used to spot-check that the
    /// preset top cell really is the ~`2^128.133` value claimed by the spec.
    fn bit_len(w: Wide) -> u32 {
        if w.hi != 0 {
            128 + (128 - w.hi.leading_zeros())
        } else if w.lo != 0 {
            128 - w.lo.leading_zeros()
        } else {
            0
        }
    }

    #[test]
    fn ways_d32_m8_b121_total_matches_spec() {
        // Spec lists log2(WAYS[32][121]) ~= 128.133.
        let total = ways(PRESET_D, PRESET_B);
        assert_eq!(bit_len(total), 129);
        // Roughly 2^128.133 ~ 3.708e38; sanity check magnitude.
        assert!(total.hi >= 1);
        assert!(total.hi < 4);
    }

    #[test]
    fn ways_base_row_is_all_ones() {
        // Math row n = 0 is not stored; `ways` serves it as a constant.
        // Verify the helper still hands back `Wide::ONE` for every `b`, since
        // the descent reads this row at the last step (`rem_after = 0`).
        for b in 0..COLS {
            assert_eq!(ways(0, b), Wide::ONE);
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
                Wide {
                    lo: expected,
                    hi: 0,
                },
                "ways[1][{b}] should be 1 + 2*min(M, b)",
            );
        }
    }

    #[test]
    fn ways_table_all_descent_cells_fit_u128() {
        // Every cell the descent reads has `n <= 31` (we always descend with
        // `rem_after = D - i - 1 <= 31`). Those cells must fit in u128 so the
        // `lo`-only fast path in `scan_buckets_u128_unrank` is sound.
        for n in 0..PRESET_D {
            for b in 0..COLS {
                assert_eq!(ways(n, b).hi, 0, "ways[{n}][{b}] unexpectedly > 2^128");
            }
        }
    }
}
