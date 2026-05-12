//! Generic slice-MLE evaluation.
//!
//! # Why this abstraction exists
//!
//! The verifier needs the multilinear-extension evaluation of a virtual
//! table `M` at a random point `r`. The naive approach is to materialize
//! the full equality table `eq(r, ·)`: that costs `O(|M|)` field operations
//! and `O(|M|)` memory, where `|M|` is linear in the witness size. Both are
//! too expensive.
//!
//! # The structure we exploit
//!
//! `M` is mostly zero. Only a handful of contiguous **slices** of `M` are
//! non-trivial. The MLE evaluation decomposes additively over those slices,
//! so we can evaluate each slice in isolation against the same `r` and sum
//! the results — each slice is orders of magnitude smaller than `M`.
//!
//! # The shape of one slice
//!
//! Pick one slice `v`, starting at position `offset` inside `M`. Suppose
//! `v` has length `B · Q` where `B = 2^offset_low_bits`. View `v` as a 2-D
//! array `v[q][b]` with `q ∈ [0, Q)` (outer index) and `b ∈ [0, B)` (inner
//! index). The slice's contribution to the full MLE is
//!
//! ```text
//! Σ_{q, b}  v[q][b] · eq_full(r, offset + b + q · B)
//! ```
//!
//! where `eq_full(r, ·)` is the full equality polynomial we are trying to
//! avoid materializing.
//!
//! # Splitting `eq_full` into `eq_lo` and `eq_hi`
//!
//! The multilinear equality polynomial factors over disjoint bit ranges.
//! Split the bits of the global index into the low `offset_low_bits` bits
//! and everything above:
//!
//! ```text
//! eq_full(r, index) = eq_lo(r_lo, index_lo) · eq_hi(r_hi, index_hi)
//! ```
//!
//! where `r_lo = r[..offset_low_bits]` and `r_hi = r[offset_low_bits..]`.
//! `eq_lo` is a small table over `2^offset_low_bits` entries — we
//! materialize it once and reuse. `eq_hi` we never materialize; we evaluate
//! it pointwise only at the few `index_hi` values we actually need.
//!
//! With this split, the slice's contribution becomes
//!
//! ```text
//! Σ_q  eq_hi(index_hi(q))  ·  Σ_b  v[q][b] · eq_lo(index_lo(b))
//!     └── outer sum ──┘     ─────── inner sum at q ──────────────┘
//! ```
//!
//! # The carry: why each `q` produces *two* inner sums
//!
//! There is one wrinkle. The global index is `offset + b + q · B`, **not**
//! just `b + q · B`. When the low bits of `offset` are non-zero, adding
//! `b` to them can overflow past `B` and carry one bit into the high half.
//!
//! Let `offset_low = offset mod B` and `offset_high = offset div B`. Then
//!
//! ```text
//! index = offset + b + q · B
//!       = (offset_low + b) + B · (offset_high + q)
//! ```
//!
//! and `(offset_low + b)` may exceed `B - 1`. When it does, it wraps to
//! `(offset_low + b) - B` in the low part and adds **one** to the high
//! part. Because both `offset_low` and `b` are strictly less than `B`,
//! the carry is always either `0` or `1` — never `2` or more.
//!
//! Concretely, define
//!
//! ```text
//! low_idx = (offset_low + b) mod B
//! carry   = (offset_low + b) div B   ∈ {0, 1}
//! ```
//!
//! Then
//!
//! ```text
//! eq_full(r, index)
//!     = eq_lo(low_idx) · eq_hi(offset_high + q + carry)
//! ```
//!
//! For the same `q`, blocks `b` split into two groups: those with
//! `carry = 0` weight `eq_hi(offset_high + q)`, and those with
//! `carry = 1` weight `eq_hi(offset_high + q + 1)` — **a different**
//! high-bit equality value. So the inner sum at `q` must produce two
//! values, one per carry case:
//!
//! ```text
//! [low0, low1][q] = ( Σ_{b: carry=0} v[q][b] · eq_lo(low_idx),
//!                     Σ_{b: carry=1} v[q][b] · eq_lo(low_idx) )
//! ```
//!
//! and the outer sum becomes
//!
//! ```text
//! Σ_q ( low0[q] · eq_hi(offset_high + q)
//!     + low1[q] · eq_hi(offset_high + q + 1) )
//! ```
//!
//! # API summary
//!
//! - [`SliceMleEvaluator`] is the trait each slice implements; one
//!   evaluator struct per slice, fully self-contained. Required surface:
//!   - [`SliceMleEvaluator::num_outer_indices`] (= `Q`),
//!   - [`SliceMleEvaluator::get_high_challenges`] returns
//!     `r[offset_low_bits..]`,
//!   - [`SliceMleEvaluator::get_offset_high`] returns
//!     `offset >> offset_low_bits`,
//!   - [`SliceMleEvaluator::compute_inner_sum`] returns
//!     `[F; POSSIBLE_CARRIES]` for one outer index. Implementations may
//!     freely use any low-bit data they store internally
//!     (e.g. `eq_low`, `offset_low`, matrix views).
//! - The trait's default [`SliceMleEvaluator::compute_outer_sum`] handles
//!   the high-bit equality pass and reads `get_high_challenges` /
//!   `get_offset_high` off `&self`.
//! - [`SliceMleEvaluator::evaluate`] iterates `compute_inner_sum` over the
//!   outer dimension (in parallel iff
//!   [`SliceMleEvaluator::parallelize_outer`]) and feeds the resulting
//!   carry-term vector into `compute_outer_sum`. Takes no arguments.
//! - The number of carry buckets is fixed at [`POSSIBLE_CARRIES`] = 2
//!   (the only value the algebra above supports).

use akita_algebra::eq_poly::EqPolynomial;
use akita_algebra::offset_eq::{eval_offset_eq_tensor, summarize_pow2_block_carries};
use akita_algebra::ring::{eval_ring_at_pows, scalar_powers};
use akita_field::parallel::*;
use akita_field::{AkitaError, CanonicalField, ExtField, FieldCore};
use akita_types::{
    gadget_row_scalars, r_decomp_levels, AkitaExpandedSetup, RingMatrixView, RingOpeningPoint,
};

use crate::protocol::ring_switch::{summarize_pow2_block_carries_base, RingSwitchDeferredRowEval};

// ---------------------------------------------------------------------------
// 0. Carry-bucket constants
// ---------------------------------------------------------------------------

/// Number of carry buckets per outer index produced by
/// [`SliceMleEvaluator::compute_inner_sum`].
///
/// Adding an inner index `b ∈ [0, 2^offset_low_bits)` to
/// `offset_low ∈ [0, 2^offset_low_bits)` can carry at most `1` into the high
/// bits — never `2` or more — so the inner sum produces exactly two values,
/// one per carry case ([`CARRY0`], [`CARRY1`]).
///
/// **Note:** This module is only tested and intended for the
/// `POSSIBLE_CARRIES = 2` case. Anything other than `2` would require the
/// outer-sum algebra to be reworked; do not change this constant.
pub const POSSIBLE_CARRIES: usize = 2;

/// Inner-sum slot for the no-carry bucket (`carry = 0`).
pub const CARRY0: usize = 0;

/// Inner-sum slot for the one-carry bucket (`carry = 1`).
pub const CARRY1: usize = 1;

// ---------------------------------------------------------------------------
// 1. Trait
// ---------------------------------------------------------------------------

/// Strategy describing one slice's MLE contribution at a fixed offset inside
/// the full vector.
///
/// Each evaluator is **self-contained**: it owns the slice's high-bit
/// randomness and the slice's high-bit offset (exposed via
/// [`Self::get_high_challenges`] and [`Self::get_offset_high`]) plus
/// whatever low-bit data its [`Self::compute_inner_sum`] needs (e.g.
/// `eq_low` and `offset_low` for evaluators that scan a strided block).
///
/// Each evaluator factors into two pieces:
///
/// 1. **Inner sum** ([`Self::compute_inner_sum`]) — for one outer index,
///    returns this evaluator's `[CARRY0, CARRY1]` carry summary. Concrete
///    evaluators own the inner loop shape and any short-circuit on zero
///    weights.
/// 2. **Outer sum** ([`Self::compute_outer_sum`]) — combines the
///    per-outer-index carry summaries with the high-bit equality polynomial
///    to produce the final scalar. Default impl is the standard high-bit
///    equality pass; an evaluator may override (e.g., to skip the
///    [`CARRY1`] term when the slice's `offset_low == 0`).
///
/// [`Self::evaluate`] composes these two pieces.
pub trait SliceMleEvaluator<F: FieldCore>: Sync {
    /// Number of outer-loop indices.
    fn num_outer_indices(&self) -> usize;

    /// High-bit segment of the slice's randomness:
    /// `full_vec_randomness[offset_low_bits..]`.
    ///
    /// Used only by the default [`Self::compute_outer_sum`].
    fn get_high_challenges(&self) -> &[F];

    /// High-bit part of the slice offset: `offset >> offset_low_bits`.
    ///
    /// Used only by the default [`Self::compute_outer_sum`].
    fn get_offset_high(&self) -> usize;

    /// Compute the inner sum at `outer_index`: this evaluator's contribution
    /// to each carry bucket ([`CARRY0`], [`CARRY1`]) for that outer index.
    fn compute_inner_sum(&self, outer_index: usize) -> [F; POSSIBLE_CARRIES];

    /// Whether [`Self::evaluate`] should iterate the outer dimension in
    /// parallel when collecting carry terms.
    ///
    /// Default `false` (sequential). Override to `true` for evaluators with
    /// non-trivial per-outer-index work.
    #[inline]
    fn parallelize_outer(&self) -> bool {
        false
    }

    /// Compute the outer sum: combine the per-outer-index carry terms with
    /// the high-bit equality polynomial.
    ///
    /// Default implementation is the standard high-bit equality pass:
    ///
    /// ```text
    /// Σ_q  carry_terms[q][CARRY0] · eq_high(offset_high + q)
    ///    + carry_terms[q][CARRY1] · eq_high(offset_high + q + 1)
    /// ```
    ///
    /// where `offset_high = self.get_offset_high()` and `eq_high` is the
    /// multilinear equality polynomial on `self.get_high_challenges()`.
    ///
    /// This is a self-contained copy of
    /// `akita_algebra::offset_eq::eval_offset_eq_peeled_carry_terms` so
    /// evaluators may override it. The most useful override is the aligned
    /// fast path: when the slice's `offset_low == 0`, every
    /// `carry_terms[q][CARRY1]` is provably zero and the second term can be
    /// skipped, halving the number of high-bit `eq` evaluations.
    ///
    /// **Note:** Both this default impl and the algebra it implements are
    /// only tested and intended for [`POSSIBLE_CARRIES`] = 2. The two carry
    /// buckets [`CARRY0`] and [`CARRY1`] are the only ones that arise from
    /// the peeled-block split.
    #[inline]
    fn compute_outer_sum(&self, carry_terms: &[[F; POSSIBLE_CARRIES]]) -> F {
        let offset_high = self.get_offset_high();
        let high_challenges = self.get_high_challenges();

        carry_terms
            .iter()
            .enumerate()
            .fold(F::zero(), |acc, (q, terms)| {
                let acc = if terms[CARRY0].is_zero() {
                    acc
                } else {
                    acc + terms[CARRY0] * eq_eval_at_index(high_challenges, offset_high + q)
                };
                if terms[CARRY1].is_zero() {
                    acc
                } else {
                    acc + terms[CARRY1] * eq_eval_at_index(high_challenges, offset_high + q + 1)
                }
            })
    }

    /// Evaluate this slice's multilinear extension at the slice's
    /// randomness.
    ///
    /// Composition: collect [`Self::compute_inner_sum`] for every outer
    /// index into a carry-term vector (sequentially or in parallel
    /// depending on [`Self::parallelize_outer`]), then collapse via
    /// [`Self::compute_outer_sum`].
    #[inline]
    fn evaluate(&self) -> F {
        let n = self.num_outer_indices();
        let carry_terms: Vec<[F; POSSIBLE_CARRIES]> = if self.parallelize_outer() {
            cfg_into_iter!(0..n)
                .map(|outer_index| self.compute_inner_sum(outer_index))
                .collect()
        } else {
            (0..n)
                .map(|outer_index| self.compute_inner_sum(outer_index))
                .collect()
        };
        self.compute_outer_sum(&carry_terms)
    }
}

/// Evaluate `eq(challenges, index)` for a single hypercube index in
/// little-endian order. Self-contained copy of `akita_algebra`'s private
/// helper, kept here to avoid widening the algebra crate's API surface.
#[inline]
fn eq_eval_at_index<F: FieldCore>(challenges: &[F], index: usize) -> F {
    if challenges.len() < usize::BITS as usize && index >= (1usize << challenges.len()) {
        return F::zero();
    }

    challenges
        .iter()
        .enumerate()
        .fold(F::one(), |acc, (bit_idx, &r_t)| {
            let bit = (index >> bit_idx) & 1;
            acc * if bit == 1 { r_t } else { F::one() - r_t }
        })
}

// ---------------------------------------------------------------------------
// 2. Concrete evaluators
// ---------------------------------------------------------------------------

/// Evaluator for the **structured rows** of the `M`-table that contribute to
/// the *Eval* part of the witness — the `w` segment in the Hachi paper.
///
/// "Structured" here means these rows admit a separable decomposition into:
///
/// - the **input rows** (one per opening point), carrying `opening_point.b`,
///   weighted by `gamma · input_row_weights`, and
/// - the **consistency-challenge row**, carrying the per-claim challenge
///   vector `c_alpha`, weighted by `challenge_weight`.
pub struct WStructuredRowsEvaluator<'a, F, E> {
    /// `full_vec_randomness[offset_low_bits..]` — slice's high-bit randomness.
    pub high_challenges: &'a [E],
    /// `offset >> offset_low_bits` — slice's high-bit offset.
    pub offset_high: usize,
    /// Gadget vector for the digit decomposition of the witness `w`.
    /// Length = `num_digits` (the number of digits in the decomposition);
    pub gadget_vector: &'a [F],
    /// For each opening point `p`, the precomputed carry summary
    /// `[Σ_{b: carry=0} eq_low(low_idx(b)) · opening_point[p].b[b],
    ///   Σ_{b: carry=1} eq_low(low_idx(b)) · opening_point[p].b[b]]`
    /// over all block indices `b`.
    /// Length = number of opening points.
    pub opening_point_block_summaries: &'a [[E; 2]],
    /// Same carry summary as [`Self::opening_point_block_summaries`], but
    /// computed against the per-claim challenge vector `c_alpha` instead of
    /// `opening_point.b`: for each claim `c`,
    /// `[Σ_{b: carry=0} eq_low(low_idx(b)) · c_alpha[c, b],
    ///   Σ_{b: carry=1} eq_low(low_idx(b)) · c_alpha[c, b]]`.
    /// Length = `num_claims`.
    pub challenge_block_summaries: &'a [[E; 2]],
    /// Random-linear-combination weights used to batch the opening claims
    /// into a single `M`-evaluation. Length = `num_claims`; one weight per
    /// claim.
    pub gamma: &'a [E],
    /// `claim_to_point[claim_idx] = point_idx` (or all-zero in single-point).
    pub claim_to_point: &'a [usize],
    /// `tau1` equality weight for each input row of `M` (one per opening
    /// point). Length = number of opening points.
    pub input_row_weights: &'a [E],
    /// `tau1` equality weight for the consistency-challenge row of `M`.
    pub challenge_weight: E,
    /// Number of evaluation claims.
    pub num_claims: usize,
    /// Number of digits in the gadget decomposition of `w`
    /// (= `gadget_vector.len()`).
    pub num_digits: usize,
    /// Whether the protocol uses multiple opening points.
    pub is_multi_point: bool,
}

impl<F, E> SliceMleEvaluator<E> for WStructuredRowsEvaluator<'_, F, E>
where
    F: FieldCore,
    E: ExtField<F>,
{
    #[inline]
    fn num_outer_indices(&self) -> usize {
        self.num_claims * self.num_digits
    }

    #[inline]
    fn get_high_challenges(&self) -> &[E] {
        self.high_challenges
    }

    #[inline]
    fn get_offset_high(&self) -> usize {
        self.offset_high
    }

    #[inline]
    fn compute_inner_sum(&self, outer_index: usize) -> [E; POSSIBLE_CARRIES] {
        let digit = outer_index / self.num_claims;
        let claim_idx = outer_index % self.num_claims;

        let point_idx = if self.is_multi_point {
            self.claim_to_point[claim_idx]
        } else {
            0
        };
        let [aggregated_opening_carry0, aggregated_opening_carry1] =
            self.opening_point_block_summaries[point_idx];
        let [aggregated_challenge_carry0, aggregated_challenge_carry1] =
            self.challenge_block_summaries[claim_idx];

        [
            (self.input_row_weights[point_idx] * self.gamma[claim_idx] * aggregated_opening_carry0
                + self.challenge_weight * aggregated_challenge_carry0)
                .mul_base(self.gadget_vector[digit]),
            (self.input_row_weights[point_idx] * self.gamma[claim_idx] * aggregated_opening_carry1
                + self.challenge_weight * aggregated_challenge_carry1)
                .mul_base(self.gadget_vector[digit]),
        ]
    }
}

/// Evaluator for the **structured rows** of the `M`-table that contribute to
/// the *encoding* part of the witness — the `t` segment in the Hachi paper.
///
/// "Structured" here means these rows admit a separable decomposition over
/// the consistency-challenge vector `c_alpha`: for each `(a_row, digit)`
/// pair, the contribution is `a_row_weight · gadget · c_alpha[claim, ·]`,
/// which reduces to a small precomputed `[CARRY0, CARRY1]` block summary —
/// no matrix scan needed. The non-structured `B · \hat t` contribution to
/// the same segment is handled directly inside
/// `compute_matrix_rows_via_patterns` alongside the `D · \hat w` and
/// `A · \hat z` halves, since all three share the same per-row `r_eval`
/// cache.
///
/// `outer_index = num_claims · (num_digits · a_row_idx + digit) + claim_idx`.
/// One source per outer index (the consistency-challenge contribution),
/// looked up from a precomputed `[CARRY0, CARRY1]` summary.
pub struct TStructuredRowsEvaluator<'a, F, E> {
    /// `full_vec_randomness[offset_low_bits..]` — slice's high-bit randomness.
    pub high_challenges: &'a [E],
    /// `offset >> offset_low_bits` — slice's high-bit offset.
    pub offset_high: usize,
    /// Gadget vector for the digit decomposition of the witness `w`.
    /// Length = `num_digits`; entry `gadget_vector[digit]` is the base-field
    /// weight applied to digit `digit`.
    pub gadget_vector: &'a [F],
    /// Same per-claim challenge carry summary as
    /// [`WStructuredRowsEvaluator::challenge_block_summaries`]: for each
    /// claim `c`,
    /// `[Σ_{b: carry=0} eq_low(low_idx(b)) · c_alpha[c, b],
    ///   Σ_{b: carry=1} eq_low(low_idx(b)) · c_alpha[c, b]]`.
    /// Length = `num_claims`.
    pub challenge_block_summaries: &'a [[E; 2]],
    /// `tau1` equality weight for each `A`-row of `M` (the rows on which the
    /// SIS commitment matrix `A` for the `t` segment is enforced).
    /// Length = number of `A` rows.
    pub a_row_weights: &'a [E],
    /// Number of evaluation claims.
    pub num_claims: usize,
    /// Number of digits in the gadget decomposition of `w`
    /// (= `gadget_vector.len()`).
    pub num_digits: usize,
}

impl<F, E> SliceMleEvaluator<E> for TStructuredRowsEvaluator<'_, F, E>
where
    F: FieldCore,
    E: ExtField<F>,
{
    #[inline]
    fn num_outer_indices(&self) -> usize {
        self.num_claims * self.num_digits * self.a_row_weights.len()
    }

    #[inline]
    fn get_high_challenges(&self) -> &[E] {
        self.high_challenges
    }

    #[inline]
    fn get_offset_high(&self) -> usize {
        self.offset_high
    }

    #[inline]
    fn compute_inner_sum(&self, outer_index: usize) -> [E; POSSIBLE_CARRIES] {
        let claim_idx = outer_index % self.num_claims;
        let compound = outer_index / self.num_claims;
        let digit = compound % self.num_digits;
        let a_row_idx = compound / self.num_digits;
        let [aggregated_challenge_carry0, aggregated_challenge_carry1] =
            self.challenge_block_summaries[claim_idx];
        [
            self.a_row_weights[a_row_idx].mul_base(self.gadget_vector[digit])
                * aggregated_challenge_carry0,
            self.a_row_weights[a_row_idx].mul_base(self.gadget_vector[digit])
                * aggregated_challenge_carry1,
        ]
    }
}

/// Evaluator for the **structured rows** of the `M`-table that contribute
/// to the *encoding* part of the witness — the `z` segment in the Hachi
/// paper.
///
/// "Structured" here means the `z`-segment's consistency summand admits
/// a separable decomposition: for each `(point_idx, df, dc)`, the
/// contribution is
///
/// ```text
/// - consistency_weight · g1_commit[dc] · fold_gadget[df] · opening_points[pt].a[blk]
/// ```
///
/// which reduces to a small precomputed `[CARRY0, CARRY1]` block summary
/// of `opening_points[pt].a` (length `block_len`). The matrix-A
/// contribution to the same `z` segment (formerly `ZMatrixRowsEvaluator`)
/// has been fused into [`compute_matrix_rows_via_patterns`] — it shares
/// `r_eval` with the `\hat w` / `\hat t` halves there.
///
/// `outer_index = pt + P · (df + DF · dc)`. One source per outer index
/// (the consistency-row contribution against `opening_points[pt].a`),
/// looked up from a precomputed `[CARRY0, CARRY1]` summary.
///
/// Note: this evaluator peels `block_len`, **not** `num_blocks` — the
/// `z` segment's inner block size differs from `\hat w` / `\hat t`. The
/// caller therefore supplies a separate `eq_low` table over the low
/// `log₂(block_len)` bits.
///
/// **Power-of-two requirement and dense fallback.** The peeled-block
/// trait machinery requires `block_len` to be a power of two (the carry
/// algebra is only well-defined when the inner block size aligns with a
/// bit boundary). At root levels and most recursive levels `block_len`
/// is power of two; at a few recursive levels it is not (e.g. 290 in
/// `D128Full` NV=12 level 1). When `dims_pow2` is `false` this evaluator
/// falls back to materialising `z_segment_struct` and calling
/// single-factor `eval_offset_eq_tensor`. The override of
/// [`SliceMleEvaluator::evaluate`] hides this dispatch from callers, so
/// the call site is identical to the four `\hat w` / `\hat t` evaluators.
pub struct ZStructuredRowsEvaluator<'a, F: FieldCore, E> {
    /// `full_vec_randomness[log₂(block_len)..]` — slice's high-bit randomness.
    /// Used by the trait (peeled) path.
    pub high_challenges: &'a [E],
    /// `offset_z >> log₂(block_len)` — slice's high-bit offset.
    /// Used by the trait (peeled) path.
    pub offset_high: usize,
    /// Gadget vector for the digit decomposition of the witness's
    /// commit-side basis. Length = `depth_commit`.
    pub g1_commit: &'a [F],
    /// Gadget vector for the digit decomposition of the witness's
    /// fold-side basis. Length = `depth_fold`.
    pub fold_gadget: &'a [F],
    /// For each opening point `p`, the precomputed carry summary
    /// `[Σ_{blk: carry=0} eq_low_z(low_idx_z(blk)) · opening_point[p].a[blk],
    ///   Σ_{blk: carry=1} eq_low_z(low_idx_z(blk)) · opening_point[p].a[blk]]`.
    /// Length = number of opening points (or 0 when `!dims_pow2`).
    pub a_block_summary: &'a [[E; 2]],
    /// `tau1` equality weight for the consistency-challenge row of `M`.
    pub consistency_weight: E,
    /// Number of opening points (`P`).
    pub num_points: usize,
    /// Number of digits in the commit gadget (`DC = depth_commit`).
    pub depth_commit: usize,
    /// Number of digits in the fold gadget (`DF = depth_fold`).
    pub depth_fold: usize,
    /// `true` iff `block_len.is_power_of_two()`. Selects between the
    /// trait (peeled-block) path and the materialised fallback.
    pub dims_pow2: bool,
    /// Used only when `!dims_pow2`. Borrowed for the fallback's
    /// `z_segment_struct` materialisation.
    pub opening_points: &'a [RingOpeningPoint<F>],
    /// Full multilinear evaluation point. Used by the dense fallback's
    /// single-factor `eval_offset_eq_tensor` call.
    pub full_vec_randomness: &'a [E],
    /// Start-of-slice offset of `z` inside `M`.
    pub offset_z: usize,
    /// Inner block size of the `z` segment (= `prepared.block_len`).
    pub block_len: usize,
}

impl<F, E> SliceMleEvaluator<E> for ZStructuredRowsEvaluator<'_, F, E>
where
    F: FieldCore,
    E: ExtField<F>,
{
    #[inline]
    fn num_outer_indices(&self) -> usize {
        self.num_points * self.depth_fold * self.depth_commit
    }

    #[inline]
    fn get_high_challenges(&self) -> &[E] {
        self.high_challenges
    }

    #[inline]
    fn get_offset_high(&self) -> usize {
        self.offset_high
    }

    #[inline]
    fn compute_inner_sum(&self, outer_index: usize) -> [E; POSSIBLE_CARRIES] {
        // outer_index = pt + P · (df + DF · dc) — bit layout above the
        // peeled `blk` axis is `[pt][df][dc]`. Only valid in the
        // power-of-two-`block_len` path; the dense fallback's
        // `evaluate` override skips this.
        let pt = outer_index % self.num_points;
        let q1 = outer_index / self.num_points;
        let df = q1 % self.depth_fold;
        let dc = q1 / self.depth_fold;

        let [a_carry0, a_carry1] = self.a_block_summary[pt];
        // Negate `consistency_weight` once, then fold the two base-field
        // gadget scalars via `mul_base` to keep the per-cell work small.
        let scale = (-self.consistency_weight)
            .mul_base(self.g1_commit[dc])
            .mul_base(self.fold_gadget[df]);
        [scale * a_carry0, scale * a_carry1]
    }

    fn evaluate(&self) -> E {
        if self.dims_pow2 {
            // Standard trait-default body: collect carry terms and run
            // the high-bit eq pass.
            let n = self.num_outer_indices();
            let carry_terms: Vec<[E; POSSIBLE_CARRIES]> =
                (0..n).map(|q| self.compute_inner_sum(q)).collect();
            self.compute_outer_sum(&carry_terms)
        } else {
            // Dense fallback: materialise the structured-only
            // `z_segment` slice in the legacy layout and run a
            // single-factor `eval_offset_eq_tensor`. Used at recursive
            // levels where `block_len` isn't a power of two.
            let p = self.num_points;
            let b = self.block_len;
            let dc = self.depth_commit;
            let df_size = self.depth_fold;
            let z_total_blocks = p * b;
            let z_len = df_size * dc * z_total_blocks;
            let z_segment_struct: Vec<E> = cfg_into_iter!(0..z_len)
                .map(|x| {
                    let compound_dig = x / z_total_blocks;
                    let global_blk = x % z_total_blocks;
                    let dc_idx = compound_dig / df_size;
                    let df = compound_dig % df_size;
                    let point_idx = global_blk / b;
                    let blk = global_blk % b;
                    let base_scale = self.opening_points[point_idx].a[blk] * self.g1_commit[dc_idx];
                    -self
                        .consistency_weight
                        .mul_base(base_scale)
                        .mul_base(self.fold_gadget[df])
                })
                .collect();
            eval_offset_eq_tensor(
                self.full_vec_randomness,
                self.offset_z,
                E::one(),
                &[z_segment_struct.as_slice()],
            )
        }
    }
}

// ---------------------------------------------------------------------------
// 3. Eval-at-point breakdown
// ---------------------------------------------------------------------------

/// Breakdown of [`RingSwitchDeferredRowEval::eval_at_point`] into its additive
/// contributions. Their sum is the full M-table evaluation.
///
/// The `b_blinding` and `d_blinding` contributions are always present in the
/// field layout but are zero unless the `zk` feature is enabled (they
/// capture the contribution of the per-group ZK blinding planes appended to
/// each commitment group's `B` input, and the global D-side ZK blinding
/// planes, respectively).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EvalAtPointParts<E> {
    /// Structured (consistency) contribution to the `z` slice. Built as a
    /// pure tensor product of `(opening_points.a, g1_commit, fold_gadget)`.
    /// Analogous to [`Self::w_sep`] / [`Self::t_sep`].
    pub z_sep: E,
    /// `A`-matrix contribution to the `z` slice. Used **only** as the
    /// non-pow2 `block_len` fallback bucket — at recursive levels where
    /// `block_len` isn't a power of two, the verifier evaluates `z_a`
    /// via dense materialisation and stashes the result here. At all
    /// other levels (root + most recursive levels) `z_a` is fused into
    /// [`Self::t_b`] via the materialised-`Eval` algorithm of
    /// `compute_matrix_rows_via_patterns`, and this field is `E::zero()`.
    pub z_a: E,
    /// Public + consistency contribution to `\hat w` (slice-MLE).
    pub w_sep: E,
    /// `D · \hat w` contribution to `\hat w`. Always `E::zero()` —
    /// fused into [`Self::t_b`] via
    /// `compute_matrix_rows_via_patterns`.
    pub w_d: E,
    /// Consistency contribution to `\hat t` (slice-MLE).
    pub t_sep: E,
    /// Combined `D · \hat w + B · \hat t + A · \hat z` contribution.
    /// All three SIS-matrix rows are evaluated in a single fused
    /// `<M_Flat, Eval>` inner product (see
    /// `compute_matrix_rows_via_patterns` and
    /// `docs/mflat-eval-fusion.md` §9). The non-pow2 `z_a` fallback
    /// bucket lives in [`Self::z_a`] instead.
    pub t_b: E,
    /// ZK B-blinding contribution (uses tensor evaluator). Always zero when
    /// the `zk` feature is disabled.
    pub b_blinding: E,
    /// ZK D-blinding contribution (uses tensor evaluator). Always zero when
    /// the `zk` feature is disabled; covers the D-row blinding planes
    /// appended after the W segment for `v`-hiding.
    pub d_blinding: E,
    /// Power-of-two `r`-tail contribution (uses tensor evaluator).
    pub r_sep: E,
    /// Non-power-of-two `r`-tail contribution (uses tensor evaluator).
    pub r_dense: E,
}

impl<E: FieldCore> EvalAtPointParts<E> {
    /// Total M-evaluation: sum of all contributions.
    pub fn sum(&self) -> E {
        self.z_sep
            + self.z_a
            + self.w_sep
            + self.w_d
            + self.t_sep
            + self.t_b
            + self.b_blinding
            + self.d_blinding
            + self.r_sep
            + self.r_dense
    }
}

/// Shared workspace used by [`eval_at_point_parts`].
struct EvalAtPointWorkspace<'a, F: FieldCore, E, const D: usize> {
    alpha_pows: Vec<E>,
    g1_open: Vec<F>,
    g1_commit: Vec<F>,
    fold_gadget: Vec<F>,
    r_gadget: Vec<F>,
    r_gadget_ext: Vec<E>,
    levels: usize,
    d_view: RingMatrixView<'a, F, D>,
    b_view: RingMatrixView<'a, F, D>,
    a_view: RingMatrixView<'a, F, D>,
    consistency_weight: E,
    public_weights: &'a [E],
    d_weights: &'a [E],
    b_start: usize,
    a_weights: &'a [E],
    r_tail_len: usize,
    inner_width: usize,
    offset_z: usize,
    offset_w: usize,
    offset_t: usize,
    offset_r: usize,
    #[cfg(feature = "zk")]
    b_blinding_segment_offset: usize,
    #[cfg(feature = "zk")]
    d_blinding_segment_offset: usize,
    offset_low_bits: usize,
    is_multi_point: bool,
    opening_point_block_summaries: Vec<[E; 2]>,
    challenge_block_summaries: Vec<[E; 2]>,
    /// Eq-polynomial table over the low `log₂(block_len)` bits of
    /// `full_vec_randomness`. Used by `Z*RowsEvaluator` for the peeled
    /// `blk` axis of the `z` segment, which has block size `block_len`
    /// (not `num_blocks` like `\hat w` / `\hat t`). Length = `block_len`.
    z_block_low_eq: Vec<E>,
    /// `log₂(block_len)` — the peeled-block bit-width for `Z*RowsEvaluator`.
    z_offset_low_bits: usize,
    /// `offset_z & (block_len - 1)` — the low-bit carry offset for the
    /// `z` segment. Mirrors `block_offset_low` for `\hat w` / `\hat t`
    /// but at the `block_len` block size.
    z_offset_low: usize,
    /// For each opening point `p`, the precomputed carry summary
    /// `[Σ_{blk: carry=0} z_block_low_eq(low_idx_z(blk)) · opening_point[p].a[blk],
    ///   Σ_{blk: carry=1} z_block_low_eq(low_idx_z(blk)) · opening_point[p].a[blk]]`
    /// over all block-len indices `blk`. Length = number of opening points
    /// when `z_dims_pow2`, empty otherwise.
    a_block_summary: Vec<[E; 2]>,
    /// `true` iff `block_len.is_power_of_two()`. When `false`,
    /// `compute_matrix_rows_via_patterns` falls back to materialising
    /// `z_segment_matrix` and running single-factor
    /// `eval_offset_eq_tensor` for the `A · \hat z` half (returned via
    /// the second tuple element); `ZStructuredRowsEvaluator` does the
    /// same for the structured `z` half. Same trade-off
    /// `r_tail_dims_pow2` makes.
    z_dims_pow2: bool,
    denom: E,
    r_tail_dims_pow2: bool,
}

impl<'a, F, E, const D: usize> EvalAtPointWorkspace<'a, F, E, D>
where
    F: FieldCore + CanonicalField,
    E: ExtField<F>,
{
    fn build(
        prepared: &'a RingSwitchDeferredRowEval<E>,
        full_vec_randomness: &'a [E],
        setup: &'a AkitaExpandedSetup<F>,
        opening_points: &'a [RingOpeningPoint<F>],
        alpha: E,
    ) -> Self {
        let alpha_pows = scalar_powers(alpha, D);
        let g1_open = gadget_row_scalars::<F>(prepared.depth_open, prepared.log_basis);
        let g1_commit = gadget_row_scalars::<F>(prepared.depth_commit, prepared.log_basis);
        let fold_gadget = gadget_row_scalars::<F>(prepared.depth_fold, prepared.log_basis);
        let levels = r_decomp_levels::<F>(prepared.log_basis);
        let r_gadget = gadget_row_scalars::<F>(levels, prepared.log_basis);
        let r_gadget_ext = r_gadget
            .iter()
            .copied()
            .map(E::lift_base)
            .collect::<Vec<_>>();

        let stride = setup.seed.max_stride;
        let d_view = setup.shared_matrix.ring_view::<D>(prepared.n_d, stride);
        let b_view = setup.shared_matrix.ring_view::<D>(prepared.n_b, stride);
        let a_view = setup.shared_matrix.ring_view::<D>(prepared.n_a, stride);

        let consistency_weight = prepared.eq_tau1[0];
        let public_weights = &prepared.eq_tau1[1..(1 + prepared.num_public_eval_rows)];
        let d_start = 1 + prepared.num_public_eval_rows;
        let commitment_row_count = prepared.n_b * prepared.num_commitment_groups;
        let b_start = d_start + prepared.n_d;
        let a_start = b_start + commitment_row_count;
        let a_weights = &prepared.eq_tau1[a_start..prepared.rows];
        let d_weights = &prepared.eq_tau1[d_start..(d_start + prepared.n_d)];

        let total_blocks = prepared.total_blocks;
        let num_blocks = prepared.num_blocks;
        let depth_open = prepared.depth_open;
        let depth_commit = prepared.depth_commit;
        let depth_fold = prepared.depth_fold;
        let inner_width = prepared.inner_width;
        let num_points = prepared.num_points;

        let w_len = depth_open * total_blocks;
        let t_len = depth_open * prepared.n_a * total_blocks;
        let z_total_blocks = num_points * prepared.block_len;
        let z_len = depth_fold * depth_commit * z_total_blocks;
        let r_tail_len = prepared.rows * levels;

        let is_multi_point = num_points > 1;

        // ZK appends two blinding segments to the layout, both placed
        // immediately after `t_len` (and before `z` / `r`): first
        // `b_blinding_segment_len` columns for the per-group `B`-side
        // blinding, then `d_blinding_segment_len` columns for the `D`-side
        // blinding that hides the wire-visible `v`. When the `zk` feature is
        // disabled both lengths are zero and the layout matches the non-ZK
        // case.
        #[cfg(feature = "zk")]
        let b_blinding_segment_len = prepared.b_blinding_segment_len;
        #[cfg(not(feature = "zk"))]
        let b_blinding_segment_len = 0usize;
        #[cfg(feature = "zk")]
        let d_blinding_segment_len = prepared.d_blinding_segment_len;
        #[cfg(not(feature = "zk"))]
        let d_blinding_segment_len = 0usize;

        let offset_z = if prepared.z_first {
            0
        } else {
            w_len + t_len + b_blinding_segment_len + d_blinding_segment_len
        };
        let offset_w = if prepared.z_first { z_len } else { 0 };
        let offset_t = if prepared.z_first {
            z_len + w_len
        } else {
            w_len
        };
        #[cfg(feature = "zk")]
        let b_blinding_segment_offset = offset_t + t_len;
        #[cfg(feature = "zk")]
        let d_blinding_segment_offset = b_blinding_segment_offset + b_blinding_segment_len;
        let offset_r = w_len + d_blinding_segment_len + t_len + b_blinding_segment_len + z_len;
        let offset_low_bits = num_blocks.trailing_zeros() as usize;

        let block_low_eq = EqPolynomial::evals(&full_vec_randomness[..offset_low_bits]);
        let block_offset_low = offset_w & (num_blocks - 1);
        debug_assert_eq!(block_offset_low, offset_t & (num_blocks - 1));

        let opening_point_block_summaries: Vec<[E; 2]> = opening_points
            .iter()
            .map(|opening_point| {
                summarize_pow2_block_carries_base::<F, E>(
                    &block_low_eq,
                    block_offset_low,
                    &opening_point.b,
                )
            })
            .collect();
        let challenge_block_summaries: Vec<[E; 2]> = (0..prepared.num_claims)
            .map(|claim_idx| {
                let start = claim_idx * num_blocks;
                summarize_pow2_block_carries(
                    &block_low_eq,
                    block_offset_low,
                    &prepared.c_alphas[start..(start + num_blocks)],
                )
            })
            .collect();

        // The `z` segment peels `block_len`, not `num_blocks`. Build its
        // own `eq_low_z` table and per-opening-point summary of
        // `opening_points[pt].a` (length `block_len`).
        let block_len = prepared.block_len;
        let z_offset_low_bits = block_len.trailing_zeros() as usize;
        let z_block_low_eq = EqPolynomial::evals(&full_vec_randomness[..z_offset_low_bits]);
        let z_offset_low = offset_z & (block_len - 1);
        // The peeled-block trait path requires `block_len` to be a power
        // of two. At root levels it always is; at some recursive levels
        // `block_len` is non-power-of-two (e.g. 290), and we fall back to
        // a materialised z_segment + single-factor MLE in
        // `compute_non_peeled_parts`. The summary is only built when
        // `block_len.is_power_of_two()`.
        let z_dims_pow2 = block_len.is_power_of_two();
        let a_block_summary: Vec<[E; 2]> = if z_dims_pow2 {
            opening_points
                .iter()
                .map(|opening_point| {
                    summarize_pow2_block_carries_base::<F, E>(
                        &z_block_low_eq,
                        z_offset_low,
                        &opening_point.a[..block_len],
                    )
                })
                .collect()
        } else {
            Vec::new()
        };

        let alpha_pow_d = alpha_pows[D - 1] * alpha;
        let denom = alpha_pow_d + E::one();
        let r_tail_dims_pow2 = levels.is_power_of_two();

        Self {
            alpha_pows,
            g1_open,
            g1_commit,
            fold_gadget,
            r_gadget,
            r_gadget_ext,
            levels,
            d_view,
            b_view,
            a_view,
            consistency_weight,
            public_weights,
            d_weights,
            b_start,
            a_weights,
            r_tail_len,
            inner_width,
            offset_z,
            offset_w,
            offset_t,
            offset_r,
            #[cfg(feature = "zk")]
            b_blinding_segment_offset,
            #[cfg(feature = "zk")]
            d_blinding_segment_offset,
            offset_low_bits,
            is_multi_point,
            opening_point_block_summaries,
            challenge_block_summaries,
            z_block_low_eq,
            z_offset_low_bits,
            z_offset_low,
            a_block_summary,
            z_dims_pow2,
            denom,
            r_tail_dims_pow2,
        }
    }
}

/// Compute the two `r`-tail contributions that don't participate in the
/// peeled-block slice-MLE abstraction:
///
/// - `r_sep` — power-of-two `r`-tail dims, evaluated via multi-factor
///   `eval_offset_eq_tensor`.
/// - `r_dense` — non-power-of-two `r`-tail dims, materialised + single-factor
///   `eval_offset_eq_tensor`.
///
/// `z_sep` is evaluated at the call site (in [`eval_at_point_parts`])
/// via [`build_z_structured_rows_evaluator`] — it implements
/// [`SliceMleEvaluator`] and dispatches internally between the
/// peeled-block trait path and a materialised fallback (see that type's
/// docs). `z_a` is fused into [`compute_matrix_rows_via_patterns`]
/// alongside `w_d` and `t_b`.
fn compute_r_tail_parts<F, E, const D: usize>(
    prepared: &RingSwitchDeferredRowEval<E>,
    full_vec_randomness: &[E],
    ws: &EvalAtPointWorkspace<'_, F, E, D>,
) -> (E, E)
where
    F: FieldCore + CanonicalField,
    E: ExtField<F>,
{
    let r_sep = if ws.r_tail_dims_pow2 {
        eval_offset_eq_tensor(
            full_vec_randomness,
            ws.offset_r,
            -ws.denom,
            &[&ws.r_gadget_ext, &prepared.eq_tau1[..prepared.rows]],
        )
    } else {
        E::zero()
    };
    let r_dense = if ws.r_tail_dims_pow2 {
        E::zero()
    } else {
        let _span = tracing::info_span!("m_eval_r_dense").entered();
        let r_tail: Vec<E> = cfg_into_iter!(0..ws.r_tail_len)
            .map(|idx| {
                let row_idx = idx / ws.levels;
                let level_idx = idx % ws.levels;
                -(prepared.eq_tau1[row_idx] * ws.denom).mul_base(ws.r_gadget[level_idx])
            })
            .collect();
        eval_offset_eq_tensor(
            full_vec_randomness,
            ws.offset_r,
            E::one(),
            &[r_tail.as_slice()],
        )
    };

    (r_sep, r_dense)
}

/// Compute the ZK B-blinding contribution. Returns `E::zero()` whenever the
/// `zk` feature is disabled or the layout has no blinding planes per group.
#[cfg(feature = "zk")]
fn compute_b_blinding_part<F, E, const D: usize>(
    prepared: &RingSwitchDeferredRowEval<E>,
    full_vec_randomness: &[E],
    ws: &EvalAtPointWorkspace<'_, F, E, D>,
) -> E
where
    F: FieldCore + CanonicalField,
    E: ExtField<F>,
{
    let group_stride = prepared.b_blinding_digit_planes_per_group;
    if group_stride == 0 {
        return E::zero();
    }
    let _span = tracing::info_span!("m_eval_b_blinding").entered();
    // Mirror the prover's group-local B input layout:
    // `[group t_hat || group blinding]` for each commitment group.
    let b_blinding_segment_len = prepared.b_blinding_segment_len;
    let t_cols_per_claim = prepared.num_blocks * prepared.n_a * prepared.depth_open;
    let b_blinding_segment: Vec<E> = cfg_into_iter!(0..b_blinding_segment_len)
        .map(|idx| {
            let group_idx = idx / group_stride;
            let local = idx % group_stride;
            let group_message_planes = prepared.group_poly_counts[group_idx] * t_cols_per_claim;
            let local_col = group_message_planes + local;
            let commitment_weights = &prepared.eq_tau1[(ws.b_start + group_idx * prepared.n_b)
                ..(ws.b_start + (group_idx + 1) * prepared.n_b)];
            let mut acc = E::zero();
            for (row_idx, &eq_i) in commitment_weights.iter().enumerate() {
                if !eq_i.is_zero() {
                    acc += eq_i
                        * eval_ring_at_pows(&ws.b_view.row(row_idx)[local_col], &ws.alpha_pows);
                }
            }
            acc
        })
        .collect();
    eval_offset_eq_tensor(
        full_vec_randomness,
        ws.b_blinding_segment_offset,
        E::one(),
        &[b_blinding_segment.as_slice()],
    )
}

#[cfg(not(feature = "zk"))]
fn compute_b_blinding_part<F, E, const D: usize>(
    _prepared: &RingSwitchDeferredRowEval<E>,
    _full_vec_randomness: &[E],
    _ws: &EvalAtPointWorkspace<'_, F, E, D>,
) -> E
where
    F: FieldCore + CanonicalField,
    E: ExtField<F>,
{
    E::zero()
}

/// Compute the ZK D-blinding contribution. Returns `E::zero()` whenever the
/// `zk` feature is disabled or the layout has no D-side blinding planes.
///
/// The D-blinding segment lives in columns `[w_len, w_len +
/// d_blinding_segment_len)` of the shared SIS matrix (read via `d_view`),
/// weighted by the global D-row `eq_tau1` weights (the same `d_weights`
/// that the D-half of `compute_matrix_rows_via_patterns` uses). It is
/// placed in the M-table layout
/// immediately after the B-blinding segment, at
/// `ws.d_blinding_segment_offset`.
#[cfg(feature = "zk")]
fn compute_d_blinding_part<F, E, const D: usize>(
    prepared: &RingSwitchDeferredRowEval<E>,
    full_vec_randomness: &[E],
    ws: &EvalAtPointWorkspace<'_, F, E, D>,
) -> E
where
    F: FieldCore + CanonicalField,
    E: ExtField<F>,
{
    let d_blinding_segment_len = prepared.d_blinding_segment_len;
    if d_blinding_segment_len == 0 {
        return E::zero();
    }
    let _span = tracing::info_span!("m_eval_d_blinding").entered();
    let w_len = prepared.depth_open * prepared.total_blocks;
    let d_blinding_segment: Vec<E> = cfg_into_iter!(0..d_blinding_segment_len)
        .map(|local| {
            let local_col = w_len + local;
            let mut acc = E::zero();
            for (row_idx, &eq_i) in ws.d_weights.iter().enumerate() {
                if !eq_i.is_zero() {
                    acc += eq_i
                        * eval_ring_at_pows(&ws.d_view.row(row_idx)[local_col], &ws.alpha_pows);
                }
            }
            acc
        })
        .collect();
    eval_offset_eq_tensor(
        full_vec_randomness,
        ws.d_blinding_segment_offset,
        E::one(),
        &[d_blinding_segment.as_slice()],
    )
}

#[cfg(not(feature = "zk"))]
fn compute_d_blinding_part<F, E, const D: usize>(
    _prepared: &RingSwitchDeferredRowEval<E>,
    _full_vec_randomness: &[E],
    _ws: &EvalAtPointWorkspace<'_, F, E, D>,
) -> E
where
    F: FieldCore + CanonicalField,
    E: ExtField<F>,
{
    E::zero()
}

/// Helpers that build evaluators from the workspace.
///
/// Each helper takes the slice-derived state — `(high_challenges, offset_high)`
/// for the outer-sum side, and (when the evaluator does a strided block
/// scan) `(eq_low, offset_low)` for the inner-sum side — so the same
/// workspace can be reused across slices at different offsets.
#[allow(clippy::too_many_arguments)]
fn build_w_structured_rows_evaluator<'a, F, E, const D: usize>(
    prepared: &'a RingSwitchDeferredRowEval<E>,
    ws: &'a EvalAtPointWorkspace<'a, F, E, D>,
    high_challenges: &'a [E],
    offset_high: usize,
) -> WStructuredRowsEvaluator<'a, F, E>
where
    F: FieldCore,
    E: FieldCore,
{
    WStructuredRowsEvaluator {
        high_challenges,
        offset_high,
        gadget_vector: &ws.g1_open,
        opening_point_block_summaries: &ws.opening_point_block_summaries,
        challenge_block_summaries: &ws.challenge_block_summaries,
        gamma: &prepared.gamma,
        claim_to_point: &prepared.claim_to_point,
        input_row_weights: ws.public_weights,
        challenge_weight: ws.consistency_weight,
        num_claims: prepared.num_claims,
        num_digits: prepared.depth_open,
        is_multi_point: ws.is_multi_point,
    }
}

#[allow(clippy::too_many_arguments)]
fn build_t_structured_rows_evaluator<'a, F, E, const D: usize>(
    prepared: &'a RingSwitchDeferredRowEval<E>,
    ws: &'a EvalAtPointWorkspace<'a, F, E, D>,
    high_challenges: &'a [E],
    offset_high: usize,
) -> TStructuredRowsEvaluator<'a, F, E>
where
    F: FieldCore,
    E: FieldCore,
{
    TStructuredRowsEvaluator {
        high_challenges,
        offset_high,
        gadget_vector: &ws.g1_open,
        challenge_block_summaries: &ws.challenge_block_summaries,
        a_row_weights: ws.a_weights,
        num_claims: prepared.num_claims,
        num_digits: prepared.depth_open,
    }
}

/// Assemble a [`ZStructuredRowsEvaluator`] from the workspace and call
/// site state. The evaluator's [`SliceMleEvaluator::evaluate`] override
/// dispatches between the peeled-block trait path (when
/// `block_len.is_power_of_two()`) and the dense materialised fallback,
/// so the call site is uniform with `build_w_structured_rows_evaluator`
/// / `build_t_structured_rows_evaluator`.
#[allow(clippy::too_many_arguments)]
fn build_z_structured_rows_evaluator<'a, F, E, const D: usize>(
    prepared: &'a RingSwitchDeferredRowEval<E>,
    ws: &'a EvalAtPointWorkspace<'a, F, E, D>,
    full_vec_randomness: &'a [E],
    opening_points: &'a [RingOpeningPoint<F>],
) -> ZStructuredRowsEvaluator<'a, F, E>
where
    F: FieldCore,
    E: ExtField<F>,
{
    let z_offset_high = ws.offset_z >> ws.z_offset_low_bits;
    let z_high_challenges = &full_vec_randomness[ws.z_offset_low_bits..];
    ZStructuredRowsEvaluator {
        high_challenges: z_high_challenges,
        offset_high: z_offset_high,
        g1_commit: &ws.g1_commit,
        fold_gadget: &ws.fold_gadget,
        a_block_summary: &ws.a_block_summary,
        consistency_weight: ws.consistency_weight,
        num_points: prepared.num_points,
        depth_commit: prepared.depth_commit,
        depth_fold: prepared.depth_fold,
        dims_pow2: ws.z_dims_pow2,
        opening_points,
        full_vec_randomness,
        offset_z: ws.offset_z,
        block_len: prepared.block_len,
    }
}

/// Compute `w_d + t_b + z_a` via the materialised-`Eval` algorithm of
/// `docs/mflat-eval-fusion.md` §9. This is the canonical verifier-side
/// path for the three SIS-matrix slice-MLE contributions, which all
/// read rows of the same shared SIS matrix and therefore share
/// `r_eval[r, c] = M_Flat[r, c] = eval_alpha(shared_matrix.row(r)[c])`.
///
/// Returns the fused scalar `w_d + t_b + z_a = <M_Flat, Eval>` (with
/// `z_a` only fused in when `block_len.is_power_of_two()`; otherwise
/// it's returned via the second tuple element and the caller routes it
/// to a separate field). The materialised-`Eval` form yields one inner
/// product per SIS row, so the three halves are not recoverable
/// separately without redoing the work; the caller writes the combined
/// result into `EvalAtPointParts::t_b` (with `w_d = z_a = 0`).
///
/// Algorithm:
///
/// 1. Precompute three `eq_hi` tables — one each for W, T, Z. The W and
///    T tables share `high_challenges = full_vec_randomness[log₂(B)..]`;
///    the Z table uses `high_challenges_z =
///    full_vec_randomness[log₂(block_len)..]` (different prefix size
///    because Z peels `block_len`, not `num_blocks`).
///
/// 2. Build the column-only patterns over `n_cols_total` cells
///    (= `max(c_W_range, c_T_range, z_range)` when `z_active`,
///    else `max(c_W_range, c_T_range)`):
///
///    - `w_pattern_padded[c]` — group-independent (W's row weights
///      don't depend on commitment group). Zero for `c ≥ w_len/D`.
///    - `t_pattern_per_group[g][c]` — one pattern per commitment group.
///      Non-zero only when `g`'s flat-claim range reaches
///      `claim_within_group(c)`.
///    - `z_pattern_padded[c]` — non-zero only for `c < z_range
///      = block_len · depth_commit` and only when `block_len` is power
///      of two. Built via the §9-style peeled-block decomposition:
///      ```text
///      z_pattern_padded[c = blk · DC + dc]
///         = z_block_low_eq[(z_offset_low + blk) mod B] ·
///           S_per_dc_per_carry[dc][(z_offset_low + blk) / B]
///      ```
///      where the small `S_per_dc_per_carry[dc][carry]` table bakes in
///      the `Σ_{pt, df} -fold_gadget[df] · eq_hi_z(...)` factor.
///
/// 3. Pad the row-weight slices to length `r_max` (= `max(n_d, n_b, n_a)`
///    when Z fuses, else `max(n_d, n_b)`):
///
///    - `d_w_padded[r] = d_weights[r]` for `r < n_d`, else `0`.
///    - `b_w_padded_per_group[r][g] = eq_tau1[b_start + g·n_b + r]` for
///      `r < n_b`, else `0`.
///    - `a_w_padded[r] = a_weights[r]` for `r < n_a` (and `block_len`
///      pow-of-two), else `0`.
///
/// 4. For each SIS row `r ∈ [0, r_max)`, build
///    `r_eval[c] = M_Flat[r, c]` for `c < row_range`, where
///    `row_range = n_cols_total` when W or T is active and
///    `row_range = z_range` when only Z is active (the latter
///    saves the ring evals on the W/T tail of `r_eval` for those
///    Z-only rows). Then fold:
///
///    ```text
///    m_eval[r, c] = d_w_padded[r] · w_pattern_padded[c]
///                + Σ_g b_w_padded_per_group[r, g] · t_pattern_per_group[g, c]
///                + a_w_padded[r] · z_pattern_padded[c]
///
///    row_contribution[r] = Σ_c r_eval[c] · m_eval[r, c]
///    ```
///
///    `m_eval[r, c]` is fused into the fold so it's never materialised.
///    No branching in the inner loop (W/T-active rows use the full
///    expression; Z-only rows take a leaner two-term path that skips
///    the W and T contributions entirely since their weights are zero).
///
/// 5. Sum `row_contribution[r]` across rows.
///
/// `r_eval` is shared across W, T, and Z for every row that participates
/// in more than one half — eliminating the redundant ring-eval work that
/// the previous separate `ZMatrixRowsEvaluator` did over the rows that
/// W and T already cover.
///
/// **Non-pow-of-two `block_len` fallback.** When `block_len` isn't power
/// of two the peeled-block construction for `z_pattern` doesn't apply.
/// In that case `z_pattern_padded` is empty, the per-row inner product
/// folds W and T only (matching the previous `compute_w_d_and_t_b_via_patterns`),
/// and `z_a` is computed via dense materialisation
/// (`matrix_a` + `z_segment_matrix` + single-factor `eval_offset_eq_tensor`)
/// at the end of this function. The `(combined, z_a_dense)` tuple is
/// returned so the caller can route `z_a_dense` to a distinct
/// `EvalAtPointParts` field — keeping the two paths cleanly separable
/// for tracing and tests, even though both halves are summed identically.
#[allow(clippy::too_many_arguments)]
fn compute_matrix_rows_via_patterns<F, E, const D: usize>(
    prepared: &RingSwitchDeferredRowEval<E>,
    ws: &EvalAtPointWorkspace<'_, F, E, D>,
    full_vec_randomness: &[E],
    high_challenges: &[E],
    eq_low: &[E],
    block_offset_low: usize,
    w_offset_high: usize,
    t_offset_high: usize,
) -> (E, E)
where
    F: FieldCore + CanonicalField,
    E: ExtField<F>,
{
    let block_bits = ws.offset_low_bits;
    let num_blocks = prepared.num_blocks;
    let num_claims = prepared.num_claims;
    let num_digits = prepared.depth_open;
    let n_a = prepared.n_a;
    let n_d = prepared.n_d;
    let n_b = prepared.n_b;
    let num_groups = prepared.num_commitment_groups;
    let stride_t = n_a * num_digits;
    let cols_per_claim_t = stride_t * num_blocks;
    let b_per_claim_w = num_blocks * num_digits;
    let n_cols_w = num_claims * b_per_claim_w;
    let num_q_w = num_claims * num_digits;
    let num_q_t = num_q_w * n_a;
    let block_mask = num_blocks.wrapping_sub(1);

    // ----- Group-shape derivation (multi-group support) ------------------
    //
    // W's row weights are flat (`d_weights[r]`, independent of group), so
    // the W half is unchanged from single-group.
    //
    // T's row weights are group-dependent: row `r` carries weight
    // `eq_tau1[b_start + g · n_b + r]` for commitment group `g`. The
    // c-axis for T uses `claim_within_group`, which lives in
    // `[0, max_claims_per_group)`. Different flat-claims (different
    // groups) can hit the same physical c by sharing a
    // `claim_within_group` value — they differ only in their row weight,
    // not in the matrix column.
    //
    // We invert `claim_to_group` so each `(g, claim_within_group)` pair
    // can quickly recover the flat-claim index that owns it (needed for
    // the q_T computation: `q_T = flat_claim + C·dig + C·L·a_row`).
    let mut claims_per_group = vec![0usize; num_groups.max(1)];
    for &(g, _) in &prepared.claim_to_group {
        claims_per_group[g] += 1;
    }
    let max_claims_per_group = claims_per_group.iter().copied().max().unwrap_or(0);
    let mut flat_claim_for_group: Vec<Vec<Option<usize>>> =
        vec![vec![None; max_claims_per_group]; num_groups];
    for (flat_idx, &(g, c_in_g)) in prepared.claim_to_group.iter().enumerate() {
        flat_claim_for_group[g][c_in_g] = Some(flat_idx);
    }

    let n_cols_t = max_claims_per_group * cols_per_claim_t;

    // ----- Z setup ---------------------------------------------------------
    //
    // The Z half (formerly `ZMatrixRowsEvaluator::evaluate`) reads cells
    // `[0, z_range = block_len · depth_commit)` of the same shared SIS
    // matrix as W and T, so its column-only pattern slots into the same
    // `m_eval[r, c]` formula. Power-of-two `block_len` is required for
    // the peeled-block construction below; the non-pow2 case falls
    // through to the dense fallback.
    let z_dims_pow2 = ws.z_dims_pow2;
    let block_len = prepared.block_len;
    let depth_commit = prepared.depth_commit;
    let depth_fold = prepared.depth_fold;
    let num_points = prepared.num_points;
    let z_range = ws.inner_width;
    let z_offset_low = ws.z_offset_low;
    let z_offset_low_bits = ws.z_offset_low_bits;
    let z_offset_high = ws.offset_z >> z_offset_low_bits;
    // `block_len.wrapping_sub(1)` is harmless when `block_len == 0` —
    // `n_a == 0` (or `!z_dims_pow2`) would then short-circuit the Z
    // path entirely. Used only inside the `z_dims_pow2 && n_a > 0`
    // guard.
    let z_block_mask = block_len.wrapping_sub(1);
    let z_active = z_dims_pow2 && n_a > 0;

    // Cover all three reshapings: W's range is `C · B · L`; T's range
    // is `max(k_g) · n_a · B · L`; Z's range is `block_len · DC`. They
    // are independent — at recursive levels the `block_len` axis grows
    // while the `num_blocks` axis shrinks, so `z_range` can exceed
    // `max(n_cols_w, n_cols_t)`. Pad all patterns and loop bounds to
    // the union so each `c ∈ [0, n_cols_total)` is safely indexable
    // by every reshaping.
    let n_cols_total = n_cols_w
        .max(n_cols_t)
        .max(if z_active { z_range } else { 0 });

    // S_per_dc_per_carry[dc][c]
    //   = -Σ_{pt, df} fold_gadget[df]
    //                · eq_hi_z[z_offset_high + (pt + P·df + P·DF·dc) + c]
    // This bakes in the `pt`/`df` summation that's independent of `blk`,
    // turning the per-cell `z_pattern[c]` build into an O(1) lookup.
    let s_per_dc_per_carry: Vec<[E; POSSIBLE_CARRIES]> = if z_active {
        let z_high_challenges = &full_vec_randomness[z_offset_low_bits..];
        let num_q_z = num_points * depth_fold * depth_commit;
        let eq_hi_z_table: Vec<E> = (0..=num_q_z)
            .map(|k| eq_eval_at_index(z_high_challenges, z_offset_high + k))
            .collect();
        (0..depth_commit)
            .map(|dc| {
                let mut s = [E::zero(); POSSIBLE_CARRIES];
                for (carry_slot, slot) in s.iter_mut().enumerate() {
                    let mut acc = E::zero();
                    for df in 0..depth_fold {
                        let fg = ws.fold_gadget[df];
                        for pt in 0..num_points {
                            let k =
                                pt + num_points * df + num_points * depth_fold * dc + carry_slot;
                            acc += eq_hi_z_table[k].mul_base(fg);
                        }
                    }
                    *slot = -acc;
                }
                s
            })
            .collect()
    } else {
        Vec::new()
    };

    // Outer-loop range over SIS matrix rows. When `z_active` we extend
    // up to `n_a` so Z-only rows participate; when not, Z is computed
    // via the dense fallback at the end of the function and we cap at
    // `max(n_d, n_b)` (the previous `compute_w_d_and_t_b_via_patterns`
    // shape).
    let r_max_wt = n_d.max(n_b);
    let r_max = if z_active {
        r_max_wt.max(n_a)
    } else {
        r_max_wt
    };

    let pow2_part = if n_cols_total > 0 && r_max > 0 {
        // ----- Precompute eq_hi tables -----------------------------------
        let eq_hi_w_table: Vec<E> = (0..=num_q_w)
            .map(|k| eq_eval_at_index(high_challenges, w_offset_high + k))
            .collect();
        let eq_hi_t_table: Vec<E> = (0..=num_q_t)
            .map(|k| eq_eval_at_index(high_challenges, t_offset_high + k))
            .collect();

        // ----- Build column-only patterns --------------------------------
        //
        // `w_pattern_padded` is single (W is group-independent). Zero
        // for `c >= n_cols_w`.
        let w_pattern_padded: Vec<E> = cfg_into_iter!(0..n_cols_total)
            .map(|c| {
                if c >= n_cols_w {
                    E::zero()
                } else {
                    let dig_w = c % num_digits;
                    let b_w = (c / num_digits) % num_blocks;
                    let claim_w = c / b_per_claim_w;
                    let q_w = dig_w * num_claims + claim_w;
                    let sum = block_offset_low + b_w;
                    let low_idx = sum & block_mask;
                    let carry = sum >> block_bits;
                    eq_low[low_idx] * eq_hi_w_table[q_w + carry]
                }
            })
            .collect();

        // `t_pattern_per_group[g][c]` — T contribution at cell `c` from
        // commitment group `g`, zero outside `g`'s flat-claim range or
        // outside T's column range.
        let t_pattern_per_group: Vec<Vec<E>> = (0..num_groups)
            .map(|g| {
                let k_g = claims_per_group[g];
                cfg_into_iter!(0..n_cols_total)
                    .map(|c| {
                        if c >= n_cols_t {
                            return E::zero();
                        }
                        let claim_within_group = c / cols_per_claim_t;
                        if claim_within_group >= k_g {
                            return E::zero();
                        }
                        match flat_claim_for_group[g][claim_within_group] {
                            Some(flat_claim) => {
                                let dig_t = c % num_digits;
                                let a_row = (c / num_digits) % n_a;
                                let b_t = (c / stride_t) % num_blocks;
                                let q_t = flat_claim
                                    + num_claims * dig_t
                                    + num_claims * num_digits * a_row;
                                let sum = block_offset_low + b_t;
                                let low_idx = sum & block_mask;
                                let carry = sum >> block_bits;
                                eq_low[low_idx] * eq_hi_t_table[q_t + carry]
                            }
                            None => E::zero(),
                        }
                    })
                    .collect()
            })
            .collect();

        // `z_pattern_padded[c]` — non-zero only for `c < z_range` and
        // only when `z_active`. Empty otherwise (the per-row loop checks
        // before indexing).
        let z_pattern_padded: Vec<E> = if z_active {
            cfg_into_iter!(0..n_cols_total)
                .map(|c| {
                    if c >= z_range {
                        E::zero()
                    } else {
                        let blk = c / depth_commit;
                        let dc = c % depth_commit;
                        let sum = z_offset_low + blk;
                        let low_idx = sum & z_block_mask;
                        let carry = sum >> z_offset_low_bits;
                        ws.z_block_low_eq[low_idx] * s_per_dc_per_carry[dc][carry]
                    }
                })
                .collect()
        } else {
            Vec::new()
        };

        // ----- Row weights, padded to r_max ------------------------------
        let d_weights_full = ws.d_weights;
        let d_w_padded: Vec<E> = (0..r_max)
            .map(|r| {
                if r < n_d {
                    d_weights_full[r]
                } else {
                    E::zero()
                }
            })
            .collect();
        let b_w_padded_per_group: Vec<E> = (0..r_max)
            .flat_map(|r| {
                (0..num_groups).map(move |g| {
                    if r < n_b {
                        prepared.eq_tau1[ws.b_start + g * n_b + r]
                    } else {
                        E::zero()
                    }
                })
            })
            .collect();
        let a_w_padded: Vec<E> = (0..r_max)
            .map(|r| {
                if r < n_a && z_active {
                    ws.a_weights[r]
                } else {
                    E::zero()
                }
            })
            .collect();

        // ----- Per-row inner products ------------------------------------
        //
        // Two regimes:
        //
        // 1. Rows where W or T is active (r < max(n_d, n_b)). Range
        //    `n_cols_total`. Inner-loop body fuses all three halves;
        //    W/T weights are non-zero, Z weight is non-zero iff
        //    `r < n_a` (and `z_active`).
        //
        // 2. Rows where only Z is active (r >= max(n_d, n_b),
        //    r < n_a). Range `z_range`. W/T weights are zero, so we
        //    skip them and run a leaner `r_eval[c] · z_pattern[c]`
        //    inner loop. Only reached when `z_active`.
        let row_contribs: Vec<E> = cfg_into_iter!(0..r_max)
            .map(|r| {
                let need_w = r < n_d;
                let need_t = r < n_b;
                let need_wt = need_w || need_t;

                // Pick a view that has row `r`. All three views alias
                // the same backing matrix; the choice only matters for
                // the Rust-side row-count check inside `RingMatrixView`.
                let row_slice = if r < n_b {
                    ws.b_view.row(r)
                } else if r < n_d {
                    ws.d_view.row(r)
                } else {
                    ws.a_view.row(r)
                };

                let row_range = if need_wt { n_cols_total } else { z_range };
                let r_eval: Vec<E> = cfg_into_iter!(0..row_range)
                    .map(|c| eval_ring_at_pows(&row_slice[c], &ws.alpha_pows))
                    .collect();

                let d_w = d_w_padded[r];
                let a_w = a_w_padded[r];
                let b_w_for_groups = &b_w_padded_per_group[r * num_groups..(r + 1) * num_groups];

                if need_wt {
                    // Branch hoisted out of the inner loop: skip the Z
                    // term entirely when `z_pattern_padded` is empty
                    // (`!z_active`), avoiding a per-cell length check.
                    if z_pattern_padded.is_empty() {
                        cfg_into_iter!(0..row_range)
                            .map(|c| {
                                let mut m = d_w * w_pattern_padded[c];
                                for g in 0..num_groups {
                                    m += b_w_for_groups[g] * t_pattern_per_group[g][c];
                                }
                                r_eval[c] * m
                            })
                            .sum::<E>()
                    } else {
                        cfg_into_iter!(0..row_range)
                            .map(|c| {
                                let mut m = d_w * w_pattern_padded[c];
                                for g in 0..num_groups {
                                    m += b_w_for_groups[g] * t_pattern_per_group[g][c];
                                }
                                m += a_w * z_pattern_padded[c];
                                r_eval[c] * m
                            })
                            .sum::<E>()
                    }
                } else {
                    // Z-only row: skip W/T terms entirely (their
                    // weights are zero, so the muls would just be
                    // wasted work over the Z-narrow `row_range`).
                    let inner: E = cfg_into_iter!(0..row_range)
                        .map(|c| r_eval[c] * z_pattern_padded[c])
                        .sum();
                    a_w * inner
                }
            })
            .collect();

        row_contribs.into_iter().sum::<E>()
    } else {
        E::zero()
    };

    let z_a_dense = if !z_dims_pow2 && n_a > 0 {
        let _span = tracing::info_span!("m_eval_z_a_dense").entered();
        let w_cols = ws.inner_width;
        let matrix_a: Vec<E> = cfg_into_iter!(0..w_cols)
            .map(|local_k| {
                let mut acc = E::zero();
                for (a_idx, &eq_i) in ws.a_weights.iter().enumerate() {
                    if !eq_i.is_zero() {
                        acc += eq_i
                            * eval_ring_at_pows(&ws.a_view.row(a_idx)[local_k], &ws.alpha_pows);
                    }
                }
                acc
            })
            .collect();
        let z_total_blocks = num_points * block_len;
        let z_len = depth_fold * depth_commit * z_total_blocks;
        let z_segment_matrix: Vec<E> = cfg_into_iter!(0..z_len)
            .map(|x| {
                let compound_dig = x / z_total_blocks;
                let global_blk = x % z_total_blocks;
                let dc_idx = compound_dig / depth_fold;
                let df = compound_dig % depth_fold;
                let blk = global_blk % block_len;
                let local_k = blk * depth_commit + dc_idx;
                -matrix_a[local_k].mul_base(ws.fold_gadget[df])
            })
            .collect();
        eval_offset_eq_tensor(
            full_vec_randomness,
            ws.offset_z,
            E::one(),
            &[z_segment_matrix.as_slice()],
        )
    } else {
        E::zero()
    };

    (pow2_part, z_a_dense)
}

/// Compute every additive contribution of `RingSwitchDeferredRowEval::eval_at_point`
/// separately, returning them as [`EvalAtPointParts`].
///
/// Three structured parts (`w_sep`, `t_sep`, `z_sep`) go through the
/// [`SliceMleEvaluator`] abstraction. The three matrix-row parts
/// (`w_d`, `t_b`, `z_a`) are jointly computed by
/// [`compute_matrix_rows_via_patterns`] — they all read rows of the
/// same shared SIS matrix, so sharing `r_eval` across them is a strict
/// win. The two `r`-tail parts (`r_sep`, `r_dense`) go through the
/// tensor-evaluator helper `compute_r_tail_parts`. ZK blinding parts
/// go through their own dedicated helpers.
///
/// `RingSwitchDeferredRowEval::eval_at_point` is a thin wrapper that calls this function
/// and sums the parts.
///
/// # Errors
///
/// Returns the same errors as `eval_at_point`.
pub fn eval_at_point_parts<F, E, const D: usize>(
    prepared: &RingSwitchDeferredRowEval<E>,
    full_vec_randomness: &[E],
    setup: &AkitaExpandedSetup<F>,
    opening_points: &[RingOpeningPoint<F>],
    alpha: E,
) -> Result<EvalAtPointParts<E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: ExtField<F>,
{
    let ws = EvalAtPointWorkspace::<F, E, D>::build(
        prepared,
        full_vec_randomness,
        setup,
        opening_points,
        alpha,
    );

    // The four slice-MLE parts share the same `(full_vec_randomness,
    // offset_low_bits)`, so derive the high/low pieces once and share them
    // across all four evaluators. `w_*` and `t_*` differ only by `offset`.
    let offset_low_bits = ws.offset_low_bits;
    let low_mask = (1usize << offset_low_bits) - 1;
    let eq_low = EqPolynomial::evals(&full_vec_randomness[..offset_low_bits]);
    let high_challenges = &full_vec_randomness[offset_low_bits..];
    let w_offset_high = ws.offset_w >> offset_low_bits;
    let w_offset_low = ws.offset_w & low_mask;
    let t_offset_high = ws.offset_t >> offset_low_bits;
    let t_offset_low = ws.offset_t & low_mask;
    debug_assert_eq!(w_offset_low, t_offset_low);

    let w_sep = {
        let _span = tracing::info_span!("m_eval_w_sep").entered();
        build_w_structured_rows_evaluator::<F, E, D>(prepared, &ws, high_challenges, w_offset_high)
            .evaluate()
    };
    let t_sep = {
        let _span = tracing::info_span!("m_eval_t_sep").entered();
        build_t_structured_rows_evaluator::<F, E, D>(prepared, &ws, high_challenges, t_offset_high)
            .evaluate()
    };

    // `w_d` + `t_b` + `z_a` are computed jointly via the materialised-
    // `Eval` algorithm of `docs/mflat-eval-fusion.md` §9 (extended to
    // include `z_a` per the same doc's "B/D fusion" section): precompute
    // three `eq_hi` slices and the W/T/Z column-only patterns once,
    // then for each SIS row share `r_eval` across all three halves (and
    // across all commitment groups within T). The `z_a` half is fused
    // in only when `block_len.is_power_of_two()`; at the few recursive
    // levels where it isn't, `z_a_dense` is computed via dense
    // materialisation inside the same function and routed to
    // `EvalAtPointParts::z_a` to keep the path-dispatch visible in the
    // breakdown.
    let (w_d_t_b_z_a_pow2, z_a_dense) = {
        let _span = tracing::info_span!("m_eval_w_d_t_b_z_a").entered();
        compute_matrix_rows_via_patterns::<F, E, D>(
            prepared,
            &ws,
            full_vec_randomness,
            high_challenges,
            &eq_low,
            w_offset_low,
            w_offset_high,
            t_offset_high,
        )
    };

    let z_sep = {
        let _span = tracing::info_span!("m_eval_z_sep").entered();
        build_z_structured_rows_evaluator::<F, E, D>(
            prepared,
            &ws,
            full_vec_randomness,
            opening_points,
        )
        .evaluate()
    };

    let (r_sep, r_dense) = compute_r_tail_parts::<F, E, D>(prepared, full_vec_randomness, &ws);
    let b_blinding = compute_b_blinding_part::<F, E, D>(prepared, full_vec_randomness, &ws);
    let d_blinding = compute_d_blinding_part::<F, E, D>(prepared, full_vec_randomness, &ws);
    Ok(EvalAtPointParts {
        z_sep,
        z_a: z_a_dense,
        w_sep,
        w_d: E::zero(),
        t_sep,
        t_b: w_d_t_b_z_a_pow2,
        b_blinding,
        d_blinding,
        r_sep,
        r_dense,
    })
}
