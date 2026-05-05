//! Bounded-`L1` sparse challenge sampler with `2^128` truncated support.
//!
//! For fixed `(D, M, B)` with `WAYS[D][B] >= 2^128`, this draws one 128-bit
//! Fiat-Shamir index `r in [0, 2^128)` from the transcript-derived XOF and
//! descends the standard suffix-count DP. The realized distribution is uniform
//! over the lexicographically-first `2^128` valid descent paths through the
//! DP recurrence, which is a `2^128`-element subset of the full bounded-`L1`
//! ball `{ c in Z^D : ||c||_inf <= M and ||c||_1 <= B }`. Every retained
//! outcome appears with probability exactly `1 / 2^128`; outcomes outside the
//! retained subset have probability `0`. See
//! `specs/bounded-l1-sparse-challenge.md` for the full security argument and
//! the rationale for relaxing the earlier "exactly uniform over the full ball"
//! requirement.
//!
//! Two consequences for the implementation:
//!
//! - The top-level draw is a single 16-byte little-endian read from the XOF,
//!   with **no** rejection loop and no modulo reduction. The earlier
//!   bias-free 256-bit wide-rejection draw is no longer needed by the
//!   production path.
//! - Each per-coefficient bucket scan runs entirely in `u128` arithmetic.
//!   Inner-loop sums use [`u128::checked_add`]: if the running cumulative sum
//!   would overflow `u128` (only possible at the very first descent step,
//!   where the buckets sum to `WAYS[D][B] > 2^128`), the comparison
//!   `r < acc + bucket` is automatically true (because `r < 2^128 <= acc +
//!   bucket`), so we select the current coefficient immediately.
//!
//! The production preset's WAYS table is built at *compile time* via
//! [`const_build_ways_table_d32_m8_b121`] and lives in `.rodata`; the sampler
//! pays no per-call construction cost. Other `(D, M, B)` triples can still go
//! through the runtime [`build_ways_table`] path. The 256-bit [`Wide`] type
//! and its `checked_add` are kept only to express the top-of-table count
//! `WAYS[32][121] ~= 2^128.133` exactly; the sampler hot path no longer
//! touches `Wide` arithmetic.

use akita_field::AkitaError;

use crate::sparse::XofCursor;

#[cfg(test)]
use akita_algebra::ring::SparseChallenge;

/// 256-bit little-endian unsigned integer. The low half is `lo`, the high
/// half is `hi`; the value is `lo + (hi << 128)`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Wide {
    pub(crate) lo: u128,
    pub(crate) hi: u128,
}

impl Wide {
    pub(crate) const ZERO: Self = Self { lo: 0, hi: 0 };
    pub(crate) const ONE: Self = Self { lo: 1, hi: 0 };

    /// Wrap a `u128` value as the low half of a `Wide`. Used by the table
    /// brute-force test harness; the production code path constructs `Wide`
    /// values via [`Wide::checked_add`] / [`Wide::sub`] directly.
    #[cfg(test)]
    #[inline]
    pub(crate) const fn from_u128(v: u128) -> Self {
        Self { lo: v, hi: 0 }
    }

    /// Checked add. Returns `None` on 256-bit overflow.
    #[inline]
    pub(crate) const fn checked_add(self, rhs: Self) -> Option<Self> {
        let (lo, c1) = self.lo.overflowing_add(rhs.lo);
        let (hi1, c2) = self.hi.overflowing_add(rhs.hi);
        let (hi, c3) = hi1.overflowing_add(c1 as u128);
        if c2 || c3 {
            None
        } else {
            Some(Self { lo, hi })
        }
    }

    /// Subtract `rhs` from `self`. Caller must ensure `self >= rhs`. Used
    /// only by the unit tests today.
    #[cfg(test)]
    #[inline]
    pub(crate) const fn sub(self, rhs: Self) -> Self {
        let (lo, borrow) = self.lo.overflowing_sub(rhs.lo);
        let hi = self.hi.wrapping_sub(rhs.hi).wrapping_sub(borrow as u128);
        Self { lo, hi }
    }

    /// Number of significant bits, i.e. `floor(log2(self)) + 1`. Returns 0
    /// for [`Wide::ZERO`]. Used only by the unit tests today; the production
    /// sampler uses `u128`-only arithmetic and never inspects bit lengths.
    #[cfg(test)]
    #[inline]
    pub(crate) const fn bit_len(self) -> u32 {
        if self.hi != 0 {
            128 + (128 - self.hi.leading_zeros())
        } else if self.lo != 0 {
            128 - self.lo.leading_zeros()
        } else {
            0
        }
    }

    /// Returns `self.lo`. Caller must ensure `self.hi == 0` (i.e. that the
    /// value fits in `u128`); production callers know this from the table
    /// invariants for `(D=32, M=8, B=121)` sub-cells.
    #[inline]
    pub(crate) const fn as_u128(self) -> u128 {
        debug_assert!(self.hi == 0, "Wide::as_u128 lossy on >=2^128 value");
        self.lo
    }

    /// Add `rhs` to `self`. Panics on 256-bit overflow. Const-friendly.
    #[inline]
    pub(crate) const fn add_or_panic(self, rhs: Self) -> Self {
        match self.checked_add(rhs) {
            Some(v) => v,
            None => panic!("Wide overflow"),
        }
    }
}

/// Maximum supported coefficient `L_inf` bound for `BoundedL1Ball`.
/// This bound is large enough for the `D=32, M=8, B=121` preset and any
/// near-future preset with single-digit-magnitude coefficients.
pub(crate) const MAX_M: usize = 16;
/// Maximum supported coefficient `L1` bound for `BoundedL1Ball`. Sized
/// generously above the `D=32, M=8, B=121` preset to keep room for future
/// presets without re-tuning the upper bound.
pub(crate) const MAX_B: usize = 256;

// --- Production preset constants: (D=32, M=8, B=121) ------------------------
//
// The preset matches `fp128_stage1_challenge_config(32)` in
// `crates/akita-config/src/proof_optimized.rs`. Keeping the dimensions as
// `pub(crate) const` lets us materialize the WAYS table at compile time and
// reference it as a borrowed `&'static [Wide]`; no `LazyLock`, no init race.

pub(crate) const PRESET_D32_M8_B121_D: usize = 32;
pub(crate) const PRESET_D32_M8_B121_M: usize = 8;
pub(crate) const PRESET_D32_M8_B121_B: usize = 121;

const PRESET_D32_M8_B121_ROWS: usize = PRESET_D32_M8_B121_D + 1;
const PRESET_D32_M8_B121_COLS: usize = PRESET_D32_M8_B121_B + 1;
const PRESET_D32_M8_B121_LEN: usize = PRESET_D32_M8_B121_ROWS * PRESET_D32_M8_B121_COLS;

// Stable Rust does not allow `[Wide; ROWS * COLS]` in a generic const fn
// signature, so the production preset's const builder is monomorphized here.
// Other `(D, M, B)` triples go through the runtime [`build_ways_table`] path
// below.
const fn const_build_ways_table_d32_m8_b121() -> [Wide; PRESET_D32_M8_B121_LEN] {
    let mut table: [Wide; PRESET_D32_M8_B121_LEN] = [Wide::ZERO; PRESET_D32_M8_B121_LEN];
    let cols = PRESET_D32_M8_B121_COLS;
    let rows = PRESET_D32_M8_B121_ROWS;
    let m = PRESET_D32_M8_B121_M;

    let mut b = 0usize;
    while b < cols {
        table[b] = Wide::ONE;
        b += 1;
    }
    let mut n = 1usize;
    while n < rows {
        let prev_row_start = (n - 1) * cols;
        let row_start = n * cols;
        let mut b = 0usize;
        while b < cols {
            let mut acc = table[prev_row_start + b];
            let max_a = if m < b { m } else { b };
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
        n += 1;
    }
    table
}

/// Compile-time WAYS table for the production `(D=32, M=8, B=121)` preset.
/// Lives in `.rodata`; resolving the singleton WAYS table to a borrowed
/// `&'static [Wide]` lets the sampler skip per-call construction.
static WAYS_D32_M8_B121: [Wide; PRESET_D32_M8_B121_LEN] = const_build_ways_table_d32_m8_b121();

/// Static reference to the production preset's WAYS table.
pub(crate) const PRESET_D32_M8_B121_TABLE: WaysTableRef<'static> = WaysTableRef {
    cells: &WAYS_D32_M8_B121,
    cols: PRESET_D32_M8_B121_COLS,
};

/// Borrowed view of a WAYS table.
///
/// The production sampler reads cells via [`WaysTableRef::at`] and immediately
/// consumes them as `u128` values. Only the very top of `WAYS[D][B]` exceeds
/// `u128` for the `(D=32, M=8, B=121)` preset; the cumulative sum at that one
/// step is handled by the sampler's `u128::checked_add`-on-overflow rule, so
/// the table view itself does not need a parallel `fits_u128` bitmap.
#[derive(Copy, Clone)]
pub(crate) struct WaysTableRef<'a> {
    pub(crate) cells: &'a [Wide],
    pub(crate) cols: usize,
}

impl WaysTableRef<'_> {
    #[inline]
    pub(crate) fn at(&self, n: usize, b: usize) -> Wide {
        self.cells[n * self.cols + b]
    }
}

/// Owned WAYS table for runtime-built presets (i.e., not the production
/// `(D=32, M=8, B=121)` preset). Provides a `view()` that yields the borrowed
/// [`WaysTableRef`] consumed by the sampler.
pub(crate) struct OwnedWaysTable {
    cells: Vec<Wide>,
    cols: usize,
}

impl OwnedWaysTable {
    pub(crate) fn view(&self) -> WaysTableRef<'_> {
        WaysTableRef {
            cells: &self.cells,
            cols: self.cols,
        }
    }
}

/// Build the suffix-count table `WAYS[n][b] = #{ v in [-M, M]^n : ||v||_1 <= b }`
/// for `n = 0..=d` and `b = 0..=l1_bound`.
///
/// The table is laid out row-major: index `WAYS[n * (l1_bound + 1) + b]`.
///
/// # Errors
///
/// Returns an error when any DP cell would overflow 256 bits, or when the
/// inputs exceed the supported [`MAX_M`] / [`MAX_B`] caps.
pub(crate) fn build_ways_table(
    d: usize,
    max_abs_coeff: usize,
    l1_bound: usize,
) -> Result<OwnedWaysTable, AkitaError> {
    if max_abs_coeff == 0 || max_abs_coeff > MAX_M {
        return Err(AkitaError::InvalidInput(format!(
            "BoundedL1Ball: max_abs_coeff {max_abs_coeff} out of supported range 1..={MAX_M}"
        )));
    }
    if l1_bound == 0 || l1_bound > MAX_B {
        return Err(AkitaError::InvalidInput(format!(
            "BoundedL1Ball: l1_bound {l1_bound} out of supported range 1..={MAX_B}"
        )));
    }

    let cols = l1_bound + 1;
    let rows = d + 1;
    let mut cells = vec![Wide::ZERO; rows * cols];
    for slot in &mut cells[..cols] {
        *slot = Wide::ONE;
    }
    for n in 1..rows {
        let prev_row_start = (n - 1) * cols;
        let row_start = n * cols;
        for b in 0..cols {
            let mut acc = cells[prev_row_start + b];
            let max_a = max_abs_coeff.min(b);
            for a in 1..=max_a {
                let neighbor = cells[prev_row_start + (b - a)];
                let doubled = neighbor.checked_add(neighbor).ok_or_else(|| {
                    AkitaError::InvalidInput(
                        "BoundedL1Ball: WAYS table overflow during 2 * neighbor".into(),
                    )
                })?;
                acc = acc.checked_add(doubled).ok_or_else(|| {
                    AkitaError::InvalidInput("BoundedL1Ball: WAYS table overflow during sum".into())
                })?;
            }
            cells[row_start + b] = acc;
        }
    }
    Ok(OwnedWaysTable { cells, cols })
}

/// Run the canonical truncated-`2^128` rank-unranking sampler against a
/// precomputed WAYS table.
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
pub(crate) fn sample_bounded_l1_into<const D: usize>(
    cursor: &mut XofCursor,
    table: WaysTableRef<'_>,
    max_abs_coeff: usize,
    l1_bound: usize,
    positions: &mut Vec<u32>,
    coeffs: &mut Vec<i16>,
) {
    positions.clear();
    coeffs.clear();
    let mut budget = l1_bound;

    // Top-level draw: one 16-byte little-endian `u128` from the XOF, no
    // rejection. The full ball size `WAYS[D][B]` for the production preset is
    // approximately `2^128.133`; truncating the sampled support to
    // `[0, 2^128)` is exactly the canonical truncation specified in
    // `specs/bounded-l1-sparse-challenge.md` and gives us 128 bits of
    // Fiat-Shamir min-entropy per challenge.
    let mut r: u128 = cursor.next_u128_le();

    for i in 0..D {
        if budget == 0 {
            break;
        }
        let rem_after = D - i - 1;

        let chosen = scan_buckets_u128_unrank(table, rem_after, budget, max_abs_coeff, &mut r);
        if chosen != 0 {
            positions.push(i as u32);
            coeffs.push(chosen);
            budget -= chosen.unsigned_abs() as usize;
        }
    }
}

/// Convenience wrapper around [`sample_bounded_l1_into`] that allocates the
/// output buffers fresh. Used by tests; production callers go through
/// [`sample_bounded_l1_into`] with reused scratch.
#[cfg(test)]
pub(crate) fn sample_bounded_l1<const D: usize>(
    cursor: &mut XofCursor,
    table: WaysTableRef<'_>,
    max_abs_coeff: usize,
    l1_bound: usize,
) -> SparseChallenge {
    let cap = D.min(l1_bound);
    let mut positions = Vec::with_capacity(cap);
    let mut coeffs = Vec::with_capacity(cap);
    sample_bounded_l1_into::<D>(
        cursor,
        table,
        max_abs_coeff,
        l1_bound,
        &mut positions,
        &mut coeffs,
    );
    SparseChallenge { positions, coeffs }
}

/// Rank-unranking bucket scan in `u128`. Visits coefficient candidates in
/// canonical signed order `-M, ..., -1, 0, 1, ..., M`, finds the bucket
/// containing the current rank `*r`, returns the chosen coefficient, and
/// updates `*r` to the offset *within* that bucket so the next position can
/// continue descending.
///
/// At the very first call (where the per-position bucket totals sum to
/// `WAYS[D][B] > 2^128`) the cumulative `acc + bucket` may overflow `u128`.
/// We use `checked_add`: an overflow proves `acc + bucket > 2^128 > r`, so
/// `r < acc + bucket` is automatically true and we select the current
/// coefficient immediately. After the first selection `r' = r - acc < bucket`
/// fits in `u128`, and every subsequent sub-table bucket is bounded by
/// `WAYS[D-1][B] < 2^128` for `(D=32, M=8, B=121)`, so no further overflow is
/// possible.
#[inline]
fn scan_buckets_u128_unrank(
    table: WaysTableRef<'_>,
    rem_after: usize,
    budget: usize,
    max_abs_coeff: usize,
    r: &mut u128,
) -> i16 {
    let mut acc: u128 = 0;
    let max_a = max_abs_coeff.min(budget);

    let mut a: i32 = -(max_a as i32);
    while a <= -1 {
        let mag = a.unsigned_abs() as usize;
        let bucket = table.at(rem_after, budget - mag).as_u128();
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
    let bucket = table.at(rem_after, budget).as_u128();
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
        let bucket = table.at(rem_after, budget - mag).as_u128();
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
    fn wide_basic_arithmetic() {
        let a = Wide::from_u128(u128::MAX);
        let b = Wide::ONE;
        let sum = a.checked_add(b).unwrap();
        assert_eq!(sum, Wide { lo: 0, hi: 1 });
        let back = sum.sub(b);
        assert_eq!(back, a);
        assert_eq!(Wide::ZERO.bit_len(), 0);
        assert_eq!(Wide::ONE.bit_len(), 1);
        assert_eq!(Wide::from_u128(7).bit_len(), 3);
        assert_eq!(Wide { lo: 0, hi: 1 }.bit_len(), 129);
    }

    #[test]
    fn ways_d32_m8_b121_total_matches_spec() {
        // Spec lists log2(WAYS[32][121]) ~= 128.133.
        let runtime = build_ways_table(32, 8, 121).unwrap();
        let total = runtime.view().at(32, 121);
        assert!(total.bit_len() == 129);
        // Roughly 2^128.133 ~ 3.708e38; sanity check magnitude.
        assert!(total.hi >= 1);
        assert!(total.hi < 4);

        // The compile-time preset table must agree with the runtime build.
        let preset_total = PRESET_D32_M8_B121_TABLE.at(32, 121);
        assert_eq!(preset_total, total);
    }

    #[test]
    fn const_preset_table_matches_runtime_build() {
        let runtime = build_ways_table(
            PRESET_D32_M8_B121_D,
            PRESET_D32_M8_B121_M,
            PRESET_D32_M8_B121_B,
        )
        .unwrap();
        let runtime_view = runtime.view();
        for n in 0..=PRESET_D32_M8_B121_D {
            for b in 0..=PRESET_D32_M8_B121_B {
                assert_eq!(
                    PRESET_D32_M8_B121_TABLE.at(n, b),
                    runtime_view.at(n, b),
                    "ways cell ({n}, {b}) drifted"
                );
            }
        }
    }

    #[test]
    fn ways_table_brute_force_d3_m2_b3() {
        let m = 2usize;
        let b = 3usize;
        let d = 3usize;
        let table = build_ways_table(d, m, b).unwrap();
        let view = table.view();

        // Brute-force count of vectors in [-2,2]^3 with L1 <= 3.
        let mut expected = 0u128;
        for c0 in -(m as i32)..=(m as i32) {
            for c1 in -(m as i32)..=(m as i32) {
                for c2 in -(m as i32)..=(m as i32) {
                    let l1 = (c0.unsigned_abs() + c1.unsigned_abs() + c2.unsigned_abs()) as usize;
                    if l1 <= b {
                        expected += 1;
                    }
                }
            }
        }
        let dp_total = view.at(d, b);
        assert_eq!(dp_total.lo, expected);
        assert_eq!(dp_total.hi, 0);
    }
}
