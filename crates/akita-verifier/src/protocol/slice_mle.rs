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
/// has been fused into `compute_matrix_rows_via_patterns` — it shares
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
//
// The state that used to live in `EvalAtPointWorkspace` is now computed
// inline at the top of [`compute_matrix_mle`], and each helper below
// takes only the specific fields it needs.

/// Compute the `r`-tail contribution that doesn't participate in the
/// peeled-block slice-MLE abstraction. Dispatches on `r_tail_dims_pow2`:
///
/// - **Power-of-two `r`-tail dims:** multi-factor `eval_offset_eq_tensor`
///   over `(r_gadget_ext, eq_tau1[..rows])`. O(L · rows) field ops.
/// - **Non-power-of-two `r`-tail dims:** materialise the `r`-tail vector
///   (`-eq_tau1[row] · denom · r_gadget[level]`), then single-factor
///   `eval_offset_eq_tensor`. O(L · rows + r_tail_len) field ops.
///
/// `z_structured_contribution` is evaluated at the call site (in
/// [`compute_matrix_mle`]) via [`build_z_structured_rows_evaluator`] — it
/// implements [`SliceMleEvaluator`] and dispatches internally between the
/// peeled-block trait path and a materialised fallback (see that type's
/// docs). The A/B/D setup-matrix contributions are fused into
/// [`compute_matrix_rows_via_patterns`] (the `setup_contribution` scalar).
#[allow(clippy::too_many_arguments)]
fn compute_r_contribution<F, E>(
    prepared: &RingSwitchDeferredRowEval<E>,
    full_vec_randomness: &[E],
    offset_r: usize,
    denom: E,
    r_gadget: &[F],
    r_gadget_ext: &[E],
    r_tail_len: usize,
    levels: usize,
    r_tail_dims_pow2: bool,
) -> E
where
    F: FieldCore + CanonicalField,
    E: ExtField<F>,
{
    if r_tail_dims_pow2 {
        let _span = tracing::info_span!("r_structured").entered();
        eval_offset_eq_tensor(
            full_vec_randomness,
            offset_r,
            -denom,
            &[r_gadget_ext, &prepared.eq_tau1[..prepared.rows]],
        )
    } else {
        let _span = tracing::info_span!("r_dense").entered();
        let r_tail: Vec<E> = cfg_into_iter!(0..r_tail_len)
            .map(|idx| {
                let row_idx = idx / levels;
                let level_idx = idx % levels;
                -(prepared.eq_tau1[row_idx] * denom).mul_base(r_gadget[level_idx])
            })
            .collect();
        eval_offset_eq_tensor(
            full_vec_randomness,
            offset_r,
            E::one(),
            &[r_tail.as_slice()],
        )
    }
}

/// Compute the ZK B-blinding contribution. Only compiled when the `zk`
/// feature is enabled. Returns `E::zero()` when the layout has no blinding
/// planes per group.
///
/// Self-contained: derives the `B` matrix view, the `b_start` row offset
/// into `eq_tau1`, and the witness-layout `b_blinding_segment_offset` from
/// `prepared` and `setup` directly — no workspace input required.
#[cfg(feature = "zk")]
fn compute_b_blinding_part<F, E, const D: usize>(
    prepared: &RingSwitchDeferredRowEval<E>,
    full_vec_randomness: &[E],
    setup: &AkitaExpandedSetup<F>,
    alpha: E,
) -> E
where
    F: FieldCore + CanonicalField,
    E: ExtField<F>,
{
    let group_stride = prepared.b_blinding_digit_planes_per_group;
    if group_stride == 0 {
        return E::zero();
    }
    let _span = tracing::info_span!("b_blinding").entered();

    // Layout offsets and SIS-matrix view derived directly from inputs.
    let alpha_pows = scalar_powers(alpha, D);
    let b_view = setup
        .shared_matrix
        .ring_view::<D>(prepared.n_b, setup.seed.max_stride);
    let b_start = 1 + prepared.num_public_eval_rows + prepared.n_d;
    let w_len = prepared.depth_open * prepared.total_blocks;
    let t_len = prepared.depth_open * prepared.n_a * prepared.total_blocks;
    let z_len =
        prepared.depth_fold * prepared.depth_commit * prepared.num_points * prepared.block_len;
    let offset_t = if prepared.z_first {
        z_len + w_len
    } else {
        w_len
    };
    let b_blinding_segment_offset = offset_t + t_len;

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
            let commitment_weights = &prepared.eq_tau1
                [(b_start + group_idx * prepared.n_b)..(b_start + (group_idx + 1) * prepared.n_b)];
            let mut acc = E::zero();
            for (row_idx, &eq_i) in commitment_weights.iter().enumerate() {
                if !eq_i.is_zero() {
                    acc += eq_i * eval_ring_at_pows(&b_view.row(row_idx)[local_col], &alpha_pows);
                }
            }
            acc
        })
        .collect();
    eval_offset_eq_tensor(
        full_vec_randomness,
        b_blinding_segment_offset,
        E::one(),
        &[b_blinding_segment.as_slice()],
    )
}

/// Compute the ZK D-blinding contribution. Only compiled when the `zk`
/// feature is enabled. Returns `E::zero()` when the layout has no D-side
/// blinding planes.
///
/// The D-blinding segment lives in columns `[w_len, w_len +
/// d_blinding_segment_len)` of the shared SIS matrix (read via `d_view`),
/// weighted by the global D-row `eq_tau1` weights (the same `d_weights`
/// that the D-half of `compute_matrix_rows_via_patterns` uses). It is
/// placed in the M-table layout immediately after the B-blinding segment,
/// at `d_blinding_segment_offset`.
///
/// Self-contained: derives the `D` matrix view, the `d_weights` slice into
/// `eq_tau1`, and the witness-layout `d_blinding_segment_offset` from
/// `prepared` and `setup` directly — no workspace input required.
#[cfg(feature = "zk")]
fn compute_d_blinding_part<F, E, const D: usize>(
    prepared: &RingSwitchDeferredRowEval<E>,
    full_vec_randomness: &[E],
    setup: &AkitaExpandedSetup<F>,
    alpha: E,
) -> E
where
    F: FieldCore + CanonicalField,
    E: ExtField<F>,
{
    let d_blinding_segment_len = prepared.d_blinding_segment_len;
    if d_blinding_segment_len == 0 {
        return E::zero();
    }
    let _span = tracing::info_span!("d_blinding").entered();

    // Layout offsets, SIS-matrix view, and D-row weights derived directly
    // from inputs.
    let alpha_pows = scalar_powers(alpha, D);
    let d_view = setup
        .shared_matrix
        .ring_view::<D>(prepared.n_d, setup.seed.max_stride);
    let d_start = 1 + prepared.num_public_eval_rows;
    let d_weights = &prepared.eq_tau1[d_start..(d_start + prepared.n_d)];
    let w_len = prepared.depth_open * prepared.total_blocks;
    let t_len = prepared.depth_open * prepared.n_a * prepared.total_blocks;
    let z_len =
        prepared.depth_fold * prepared.depth_commit * prepared.num_points * prepared.block_len;
    let offset_t = if prepared.z_first {
        z_len + w_len
    } else {
        w_len
    };
    let b_blinding_segment_offset = offset_t + t_len;
    let d_blinding_segment_offset = b_blinding_segment_offset + prepared.b_blinding_segment_len;

    let d_blinding_segment: Vec<E> = cfg_into_iter!(0..d_blinding_segment_len)
        .map(|local| {
            let local_col = w_len + local;
            let mut acc = E::zero();
            for (row_idx, &eq_i) in d_weights.iter().enumerate() {
                if !eq_i.is_zero() {
                    acc += eq_i * eval_ring_at_pows(&d_view.row(row_idx)[local_col], &alpha_pows);
                }
            }
            acc
        })
        .collect();
    eval_offset_eq_tensor(
        full_vec_randomness,
        d_blinding_segment_offset,
        E::one(),
        &[d_blinding_segment.as_slice()],
    )
}

/// Return the low/high eq-table indices for a D-column that stores a `w` cell.
///
/// During commit, `D · w` is produced in D-physical order with `digit` as the
/// innermost axis, then `block`, then `claim`; this makes the prover's commit
/// loop write consecutive digit outputs without shuffling. The witness `w`
/// itself uses the M-layout order with `block` innermost, then `claim`, then
/// `digit`, because a power-of-two block axis lets the verifier split eq
/// evaluation into a small low-bit table and a high-bit table.
///
/// Reordering the whole D matrix into M-layout would make the verifier's eq
/// lookups trivial, but it would add a large shuffle over ring-valued matrix
/// entries. Instead, we leave D in its commit-friendly order and translate the
/// current D-column into the corresponding M-layout eq indices.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn get_eq_indices_for_d(
    current_index: usize,
    num_digits: usize,
    num_blocks: usize,
    num_claims: usize,
    blocks_per_claim_w: usize,
    block_offset_low: usize,
    block_mask: usize,
    block_bits: usize,
) -> (usize, usize) {
    let digit_idx = current_index % num_digits;
    let block_idx = (current_index / num_digits) % num_blocks;
    let claim_idx = current_index / blocks_per_claim_w;
    let m_layout_high_idx = digit_idx * num_claims + claim_idx;
    let block_sum = block_offset_low + block_idx;
    let low_eq_idx = block_sum & block_mask;
    let block_carry = block_sum >> block_bits;
    let high_eq_idx = m_layout_high_idx + block_carry;
    (low_eq_idx, high_eq_idx)
}

/// Return the low/high eq-table indices for a B-column that stores a `t` cell.
///
/// T follows the same verifier-side bridge as W, but its SIS column has one
/// extra `a_row` axis and its physical claim is first resolved to a global flat
/// claim for the active commitment group. The returned pair indexes the shared
/// low block eq table and the high `(a_row, digit, flat_claim)` eq table.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn get_eq_indices_for_b(
    current_index: usize,
    flat_claim: usize,
    num_digits: usize,
    n_a: usize,
    num_blocks: usize,
    num_claims: usize,
    stride_t: usize,
    block_offset_low: usize,
    block_mask: usize,
    block_bits: usize,
) -> (usize, usize) {
    let digit_idx = current_index % num_digits;
    let a_row_idx = (current_index / num_digits) % n_a;
    let block_idx = (current_index / stride_t) % num_blocks;
    let m_layout_high_idx =
        flat_claim + num_claims * digit_idx + num_claims * num_digits * a_row_idx;
    let block_sum = block_offset_low + block_idx;
    let low_eq_idx = block_sum & block_mask;
    let block_carry = block_sum >> block_bits;
    let high_eq_idx = m_layout_high_idx + block_carry;
    (low_eq_idx, high_eq_idx)
}

/// Return the low eq index, A digit index, and carry for an A-column of `z`.
///
/// The A sub-matrix is stored as `(block, dc)` with `dc` innermost, while the
/// `z` witness keeps `block` innermost and folds the extra `(point, df)` axes
/// into `s_per_dc_per_carry`. This translates the A-column into the low-block
/// eq table index plus the small precomputed `(dc, carry)` table lookup.
#[inline(always)]
fn get_eq_indices_for_a(
    current_index: usize,
    depth_commit: usize,
    z_offset_low: usize,
    z_block_mask: usize,
    z_offset_low_bits: usize,
) -> (usize, usize, usize) {
    let block_idx = current_index / depth_commit;
    let depth_commit_idx = current_index % depth_commit;
    let block_sum = z_offset_low + block_idx;
    let low_eq_idx = block_sum & z_block_mask;
    let block_carry = block_sum >> z_offset_low_bits;
    (low_eq_idx, depth_commit_idx, block_carry)
}

/// Compute the fused setup-matrix contribution `D · \hat w + B · \hat t
/// + A · \hat z` via the materialised-`Eval` algorithm of
/// `docs/mflat-eval-fusion.md` §9. This is the canonical verifier-side
/// path for the three SIS-matrix slice-MLE contributions, which all
/// read rows of the same shared SIS matrix and therefore share
/// `r_eval[r, c] = M_Flat[r, c] = eval_alpha(shared_matrix.row(r)[c])`.
///
/// Returns the fused scalar `<M_Flat, Eval>` (the three halves are not
/// recoverable separately without redoing the work). The caller folds
/// this scalar into the total M-table evaluation alongside the structured
/// and `r`-tail contributions.
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
///    - `w_eq_slice_padded[c]` — group-independent (W's row weights
///      don't depend on commitment group). Zero for `c ≥ w_len/D`.
///    - `t_eq_slice_per_group_padded[g][c]` — one slice per commitment group.
///      Non-zero only when `g`'s flat-claim range reaches
///      `claim_within_group(c)`.
///    - `z_eq_slice_padded[c]` — non-zero only for `c < z_range
///      = block_len · depth_commit` and only when `block_len` is power
///      of two. Built via the §9-style peeled-block decomposition:
///      ```text
///      z_eq_slice_padded[c = blk · DC + dc]
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
///    m_eval[r, c] = d_w_padded[r] · w_eq_slice_padded[c]
///                + Σ_g b_w_padded_per_group[r, g] · t_eq_slice_per_group_padded[g, c]
///                + a_w_padded[r] · z_eq_slice_padded[c]
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
/// **Non-pow-of-two `block_len`.** When `block_len` isn't power of two the
/// peeled-block formula for `z_eq_slice_padded[c]` doesn't apply (the block
/// axis no longer aligns with a bit window). Instead the build switches to
/// a dense aggregation:
///
/// ```text
/// z_eq_slice_padded[c] = -Σ_{pt, df} fold_gadget[df]
///                        · eq(r, offset_z + j_M^Z(c, pt, df))
/// ```
///
/// fed by a one-shot peeled eq cache (`EqPolynomial::evals` on the low
/// log2(z_len) bits + a tiny high-bit factor table) so per-cell cost stays
/// O(P · DF). The resulting `Vec<E>` has the same shape as the pow2
/// version, so the per-row inner-product loop is *layout-agnostic*: it
/// folds W + T + Z identically in both modes. No post-loop dense
/// fallback — every α-eval of an A-matrix row happens exactly once,
/// inside the per-row loop's Z-only branch (for rows in
/// `[max(n_d, n_b), n_a)`).
#[allow(clippy::too_many_arguments)]
fn compute_matrix_rows_via_patterns<F, E, const D: usize>(
    prepared: &RingSwitchDeferredRowEval<E>,
    full_vec_randomness: &[E],
    high_challenges: &[E],
    eq_low: &[E],
    block_offset_low: usize,
    block_bits: usize,
    w_offset_high: usize,
    t_offset_high: usize,
    offset_z: usize,
    z_offset_low: usize,
    z_offset_low_bits: usize,
    z_range: usize,
    z_dims_pow2: bool,
    b_start: usize,
    alpha_pows: &[E],
    fold_gadget: &[F],
    z_block_low_eq: &[E],
    d_weights: &[E],
    a_weights: &[E],
    b_view: RingMatrixView<'_, F, D>,
    d_view: RingMatrixView<'_, F, D>,
    a_view: RingMatrixView<'_, F, D>,
) -> E
where
    F: FieldCore + CanonicalField,
    E: ExtField<F>,
{
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
    // `m_eval[r, c]` formula. The pattern build branches on
    // `block_len.is_power_of_two()`: pow2 uses the peeled-block
    // construction with the `S_per_dc_per_carry` table; non-pow2 uses
    // a dense aggregation over the witness's `(pt, df)` axes with a
    // precomputed eq lookup. Both paths produce the same Vec shape so
    // the per-row inner-product loop is layout-agnostic.
    let block_len = prepared.block_len;
    let depth_commit = prepared.depth_commit;
    let depth_fold = prepared.depth_fold;
    let num_points = prepared.num_points;
    let z_offset_high = offset_z >> z_offset_low_bits;
    // `block_len.wrapping_sub(1)` is harmless when `block_len == 0` —
    // `n_a == 0` (or `z_range == 0`) would then short-circuit the Z
    // path entirely. Used only inside the `z_active` (pow2) guard.
    let z_block_mask = block_len.wrapping_sub(1);
    // `z_used` enables the Z column-only pattern, the Z-only outer-row
    // range, and the `a_w` weight. Pow2 / non-pow2 only differ inside
    // the `z_eq_slice_padded` build.
    let z_used = n_a > 0 && z_range > 0;
    // `z_active` is the *pow2-only* gate, kept for the peeled
    // `S_per_dc_per_carry` precompute that the non-pow2 path doesn't use.
    let z_active = z_dims_pow2 && z_used;

    // Cover all three reshapings: W's range is `C · B · L`; T's range
    // is `max(k_g) · n_a · B · L`; Z's range is `block_len · DC`. They
    // are independent — at recursive levels the `block_len` axis grows
    // while the `num_blocks` axis shrinks, so `z_range` can exceed
    // `max(n_cols_w, n_cols_t)`. Pad all patterns and loop bounds to
    // the union so each `c ∈ [0, n_cols_total)` is safely indexable
    // by every reshaping.
    let n_cols_total = n_cols_w.max(n_cols_t).max(if z_used { z_range } else { 0 });

    // S_per_dc_per_carry[dc][c]
    //   = -Σ_{pt, df} fold_gadget[df]
    //                · eq_hi_z[z_offset_high + (pt + P·df + P·DF·dc) + c]
    // This bakes in the `pt`/`df` summation that's independent of `blk`,
    // turning the per-cell `z_eq_slice_padded[c]` build into an O(1) lookup.
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
                    for (df, &fg) in fold_gadget.iter().enumerate().take(depth_fold) {
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

    // Outer-loop range over SIS matrix rows. When `z_used` we extend
    // up to `n_a` so Z-only rows participate. This holds in *both*
    // pow2 and non-pow2 `block_len` modes — the A-row α-evals always
    // happen inside this loop now, so there is no separate post-loop
    // matrix-A scan.
    let r_max_wt = n_d.max(n_b);
    let r_max = if z_used { r_max_wt.max(n_a) } else { r_max_wt };

    assert!(
        n_cols_total > 0,
        "matrix-row pattern evaluation requires at least one SIS column"
    );
    assert!(
        r_max > 0,
        "matrix-row pattern evaluation requires at least one SIS row"
    );

    let setup_contribution = {
        let eq_hi_w_table: Vec<E> = (0..=num_claims * num_digits)
            .map(|k| eq_eval_at_index(high_challenges, w_offset_high + k))
            .collect();
        let eq_hi_t_table: Vec<E> = (0..=num_claims * num_digits * n_a)
            .map(|k| eq_eval_at_index(high_challenges, t_offset_high + k))
            .collect();

        let w_eq_slice_padded: Vec<E> = cfg_into_iter!(0..n_cols_total)
            .map(|current_index| {
                if current_index >= n_cols_w {
                    E::zero()
                } else {
                    let (low_eq_idx, high_eq_idx) = get_eq_indices_for_d(
                        current_index,
                        num_digits,
                        num_blocks,
                        num_claims,
                        b_per_claim_w,
                        block_offset_low,
                        block_mask,
                        block_bits,
                    );
                    eq_low[low_eq_idx] * eq_hi_w_table[high_eq_idx]
                }
            })
            .collect();

        let t_eq_slice_per_group_padded: Vec<Vec<E>> = (0..num_groups)
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
                                let (low_eq_idx, high_eq_idx) = get_eq_indices_for_b(
                                    c,
                                    flat_claim,
                                    num_digits,
                                    n_a,
                                    num_blocks,
                                    num_claims,
                                    stride_t,
                                    block_offset_low,
                                    block_mask,
                                    block_bits,
                                );
                                eq_low[low_eq_idx] * eq_hi_t_table[high_eq_idx]
                            }
                            None => E::zero(),
                        }
                    })
                    .collect()
            })
            .collect();

        // `z_eq_slice_padded[c]` — column-only eq pattern for the
        // A half of Z. Non-zero only for `c < z_range`; empty when
        // `!z_used` (i.e. `n_a == 0` or `z_range == 0`).
        //
        // Two construction modes, same output shape:
        //
        // * `z_dims_pow2`: peeled-block formula
        //     `z_block_low_eq[low_idx] · S_per_dc_per_carry[dc][carry]`,
        //   using the bit-aligned block axis. O(1) per cell.
        //
        // * `!z_dims_pow2`: dense aggregation
        //     `-Σ_{pt, df} fold_gadget[df] · eq(r, offset_z + j_M^Z(c, pt, df))`,
        //   which absorbs what used to be the post-loop `z_a_dense`
        //   path. Per-cell cost is O(P · DF) lookups plus the one-shot
        //   peeled-cache build (size O(z_len)), so total non-pow2 cost
        //   matches the old `eval_offset_eq_tensor` call asymptotically.
        let z_eq_slice_padded: Vec<E> = if !z_used {
            Vec::new()
        } else if z_dims_pow2 {
            cfg_into_iter!(0..n_cols_total)
                .map(|c| {
                    if c >= z_range {
                        E::zero()
                    } else {
                        let (low_eq_idx, depth_commit_idx, block_carry) = get_eq_indices_for_a(
                            c,
                            depth_commit,
                            z_offset_low,
                            z_block_mask,
                            z_offset_low_bits,
                        );
                        z_block_low_eq[low_eq_idx]
                            * s_per_dc_per_carry[depth_commit_idx][block_carry]
                    }
                })
                .collect()
        } else {
            // Non-pow2 dense path. Build a peeled eq cache so each
            // per-cell `eq(r, offset_z + j_M^Z)` lookup is O(1) rather
            // than O(|r|). Without this cache, the build would be
            // O(z_len · |r|) — a measurable regression relative to the
            // old `eval_offset_eq_tensor` post-loop call.
            let z_total_blocks_dense = block_len * num_points;
            let z_len_dense = depth_fold * depth_commit * z_total_blocks_dense;
            let n_rand = full_vec_randomness.len();
            let bits_for_zlen = z_len_dense
                .saturating_sub(1)
                .checked_next_power_of_two()
                .map(|p| p.trailing_zeros() as usize)
                .unwrap_or(0)
                .max(1)
                .min(n_rand);
            let k = bits_for_zlen;
            let mask = (1usize << k) - 1;
            let offset_z_dense_low = offset_z & mask;
            let offset_z_dense_high = offset_z >> k;
            let eq_low_z_dense = EqPolynomial::evals(&full_vec_randomness[..k]);
            // The largest witness coord we'll read is `offset_z + z_len - 1`.
            // Its high-bit value is `(offset_z + z_len - 1) >> k`; the
            // smallest is `offset_z_dense_high`. Tabulate the eq factor
            // for every high value in that small range.
            let max_high = (offset_z + z_len_dense - 1) >> k;
            let n_high = max_high - offset_z_dense_high + 1;
            let eq_high_z_dense: Vec<E> = (0..n_high)
                .map(|h| eq_eval_at_index(&full_vec_randomness[k..], offset_z_dense_high + h))
                .collect();

            cfg_into_iter!(0..n_cols_total)
                .map(|c| {
                    if c >= z_range {
                        E::zero()
                    } else {
                        let dc = c % depth_commit;
                        let blk = c / depth_commit;
                        let mut acc = E::zero();
                        for pt in 0..num_points {
                            for (df, &fg) in fold_gadget.iter().enumerate().take(depth_fold) {
                                // j_M^Z(c, pt, df) = blk + B·pt + B·P·df + B·P·DF·dc
                                let x = blk
                                    + block_len * pt
                                    + block_len * num_points * df
                                    + block_len * num_points * depth_fold * dc;
                                let sum = offset_z_dense_low + x;
                                let low_idx = sum & mask;
                                let high_idx = sum >> k;
                                let eq_val = eq_low_z_dense[low_idx]
                                    * eq_high_z_dense[high_idx - offset_z_dense_high];
                                acc += eq_val.mul_base(fg);
                            }
                        }
                        -acc
                    }
                })
                .collect()
        };

        // ----- Row weights, padded to r_max ------------------------------
        let d_w_padded: Vec<E> = (0..r_max)
            .map(|r| if r < n_d { d_weights[r] } else { E::zero() })
            .collect();
        let b_w_padded_per_group: Vec<E> = (0..r_max)
            .flat_map(|r| {
                (0..num_groups).map(move |g| {
                    if r < n_b {
                        prepared.eq_tau1[b_start + g * n_b + r]
                    } else {
                        E::zero()
                    }
                })
            })
            .collect();
        let a_w_padded: Vec<E> = (0..r_max)
            .map(|r| {
                if r < n_a && z_used {
                    a_weights[r]
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
        //    skip them and run a leaner `r_eval[c] · z_eq_slice_padded[c]`
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
                    b_view.row(r)
                } else if r < n_d {
                    d_view.row(r)
                } else {
                    a_view.row(r)
                };

                let row_range = if need_wt { n_cols_total } else { z_range };
                let r_eval: Vec<E> = cfg_into_iter!(0..row_range)
                    .map(|c| eval_ring_at_pows(&row_slice[c], alpha_pows))
                    .collect();

                let d_w = d_w_padded[r];
                let a_w = a_w_padded[r];
                let b_w_for_groups = &b_w_padded_per_group[r * num_groups..(r + 1) * num_groups];

                if need_wt {
                    // Branch hoisted out of the inner loop: skip the Z
                    // term entirely when `z_eq_slice_padded` is empty
                    // (`!z_active`), avoiding a per-cell length check.
                    if z_eq_slice_padded.is_empty() {
                        cfg_into_iter!(0..row_range)
                            .map(|c| {
                                let mut m = d_w * w_eq_slice_padded[c];
                                for g in 0..num_groups {
                                    m += b_w_for_groups[g] * t_eq_slice_per_group_padded[g][c];
                                }
                                r_eval[c] * m
                            })
                            .sum::<E>()
                    } else {
                        cfg_into_iter!(0..row_range)
                            .map(|c| {
                                let mut m = d_w * w_eq_slice_padded[c];
                                for g in 0..num_groups {
                                    m += b_w_for_groups[g] * t_eq_slice_per_group_padded[g][c];
                                }
                                m += a_w * z_eq_slice_padded[c];
                                r_eval[c] * m
                            })
                            .sum::<E>()
                    }
                } else {
                    // Z-only row: skip W/T terms entirely (their
                    // weights are zero, so the muls would just be
                    // wasted work over the Z-narrow `row_range`).
                    let inner: E = cfg_into_iter!(0..row_range)
                        .map(|c| r_eval[c] * z_eq_slice_padded[c])
                        .sum();
                    a_w * inner
                }
            })
            .collect();

        row_contribs.into_iter().sum::<E>()
    };

    setup_contribution
}

/// Compute the M-table MLE at `x_challenges` as the sum of every additive
/// contribution. Each contribution is produced by a dedicated helper that
/// matches its sumcheck-evaluator shape:
///
/// Three structured contributions (`w`, `t`, `z`) go through the
/// [`SliceMleEvaluator`] abstraction. The fused setup-matrix contribution
/// (`D·\hat w + B·\hat t + A·\hat z`) is produced by
/// `compute_matrix_rows_via_patterns` — all three SIS-matrix rows share
/// `r_eval`, so fusing them is a strict win. The single `r`-tail
/// contribution goes through the tensor-evaluator helper
/// `compute_r_contribution`, which internally dispatches between the
/// pow2 multi-factor path and the non-pow2 materialised single-factor
/// path. The ZK blinding contributions are computed (only under
/// `feature = "zk"`) by their own dedicated helpers and folded into the
/// returned scalar.
///
/// [`RingSwitchDeferredRowEval::eval_at_point`] is a thin wrapper that
/// calls this function.
///
/// # Errors
///
/// Returns the same errors as `eval_at_point`.
pub fn compute_matrix_mle<F, E, const D: usize>(
    prepared: &RingSwitchDeferredRowEval<E>,
    full_vec_randomness: &[E],
    setup: &AkitaExpandedSetup<F>,
    opening_points: &[RingOpeningPoint<F>],
    alpha: E,
) -> Result<E, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: ExtField<F>,
{
    // === Precomputed state (was `EvalAtPointWorkspace`) ===
    //
    // Each helper below takes only the specific fields it needs from this
    // block, so it's straightforward to see what every contribution
    // depends on.
    let alpha_pows = scalar_powers(alpha, D);
    let g1_open = gadget_row_scalars::<F>(prepared.depth_open, prepared.log_basis);
    let g1_commit = gadget_row_scalars::<F>(prepared.depth_commit, prepared.log_basis);
    let fold_gadget = gadget_row_scalars::<F>(prepared.depth_fold, prepared.log_basis);
    let levels = r_decomp_levels::<F>(prepared.log_basis);
    let r_gadget = gadget_row_scalars::<F>(levels, prepared.log_basis);
    let r_gadget_ext: Vec<E> = r_gadget.iter().copied().map(E::lift_base).collect();

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

    let num_blocks = prepared.num_blocks;
    let depth_open = prepared.depth_open;
    let depth_commit = prepared.depth_commit;
    let depth_fold = prepared.depth_fold;
    let inner_width = prepared.inner_width;
    let num_points = prepared.num_points;
    let block_len = prepared.block_len;

    let w_len = depth_open * prepared.total_blocks;
    let t_len = depth_open * prepared.n_a * prepared.total_blocks;
    let z_total_blocks = num_points * block_len;
    let z_len = depth_fold * depth_commit * z_total_blocks;
    let r_tail_len = prepared.rows * levels;
    let is_multi_point = num_points > 1;

    // ZK appends two blinding segments to the layout, both placed
    // immediately after `t_len` (and before `z` / `r`); when the `zk`
    // feature is disabled both lengths are zero.
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
    let offset_r = w_len + d_blinding_segment_len + t_len + b_blinding_segment_len + z_len;
    let offset_low_bits = num_blocks.trailing_zeros() as usize;

    // Shared eq table over the low `log₂(num_blocks)` bits, used by
    // peeled-block summaries and by the slice-MLE evaluators' eq lookups.
    let eq_low = EqPolynomial::evals(&full_vec_randomness[..offset_low_bits]);
    let block_offset_low = offset_w & (num_blocks - 1);
    debug_assert_eq!(block_offset_low, offset_t & (num_blocks - 1));

    let opening_point_block_summaries: Vec<[E; 2]> = opening_points
        .iter()
        .map(|opening_point| {
            summarize_pow2_block_carries_base::<F, E>(&eq_low, block_offset_low, &opening_point.b)
        })
        .collect();
    let challenge_block_summaries: Vec<[E; 2]> = (0..prepared.num_claims)
        .map(|claim_idx| {
            let start = claim_idx * num_blocks;
            summarize_pow2_block_carries(
                &eq_low,
                block_offset_low,
                &prepared.c_alphas[start..(start + num_blocks)],
            )
        })
        .collect();

    // The `z` segment peels `block_len`, not `num_blocks`. Build its own
    // `eq_low_z` table and per-opening-point summary of `opening_points[pt].a`
    // (length `block_len`).
    let z_offset_low_bits = block_len.trailing_zeros() as usize;
    let z_block_low_eq = EqPolynomial::evals(&full_vec_randomness[..z_offset_low_bits]);
    let z_offset_low = offset_z & block_len.wrapping_sub(1);
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

    let low_mask = (1usize << offset_low_bits) - 1;
    let high_challenges = &full_vec_randomness[offset_low_bits..];
    let w_offset_high = offset_w >> offset_low_bits;
    let w_offset_low = offset_w & low_mask;
    let t_offset_high = offset_t >> offset_low_bits;
    let t_offset_low = offset_t & low_mask;
    debug_assert_eq!(w_offset_low, t_offset_low);

    let w_structured_contribution = {
        let _span = tracing::info_span!("w_structured").entered();
        WStructuredRowsEvaluator {
            high_challenges,
            offset_high: w_offset_high,
            gadget_vector: &g1_open,
            opening_point_block_summaries: &opening_point_block_summaries,
            challenge_block_summaries: &challenge_block_summaries,
            gamma: &prepared.gamma,
            claim_to_point: &prepared.claim_to_point,
            input_row_weights: public_weights,
            challenge_weight: consistency_weight,
            num_claims: prepared.num_claims,
            num_digits: prepared.depth_open,
            is_multi_point,
        }
        .evaluate()
    };
    let t_structured_contribution = {
        let _span = tracing::info_span!("t_structured").entered();
        TStructuredRowsEvaluator {
            high_challenges,
            offset_high: t_offset_high,
            gadget_vector: &g1_open,
            challenge_block_summaries: &challenge_block_summaries,
            a_row_weights: a_weights,
            num_claims: prepared.num_claims,
            num_digits: prepared.depth_open,
        }
        .evaluate()
    };

    let setup_contribution = {
        let _span = tracing::info_span!("setup_contribution").entered();
        compute_matrix_rows_via_patterns::<F, E, D>(
            prepared,
            full_vec_randomness,
            high_challenges,
            &eq_low,
            w_offset_low,
            offset_low_bits,
            w_offset_high,
            t_offset_high,
            offset_z,
            z_offset_low,
            z_offset_low_bits,
            inner_width,
            z_dims_pow2,
            b_start,
            &alpha_pows,
            &fold_gadget,
            &z_block_low_eq,
            d_weights,
            a_weights,
            b_view,
            d_view,
            a_view,
        )
    };

    let z_structured_contribution = {
        let _span = tracing::info_span!("z_structured").entered();
        let z_offset_high = offset_z >> z_offset_low_bits;
        let z_high_challenges = &full_vec_randomness[z_offset_low_bits..];
        ZStructuredRowsEvaluator {
            high_challenges: z_high_challenges,
            offset_high: z_offset_high,
            g1_commit: &g1_commit,
            fold_gadget: &fold_gadget,
            a_block_summary: &a_block_summary,
            consistency_weight,
            num_points: prepared.num_points,
            depth_commit: prepared.depth_commit,
            depth_fold: prepared.depth_fold,
            dims_pow2: z_dims_pow2,
            opening_points,
            full_vec_randomness,
            offset_z,
            block_len: prepared.block_len,
        }
        .evaluate()
    };

    let r_contribution = compute_r_contribution(
        prepared,
        full_vec_randomness,
        offset_r,
        denom,
        &r_gadget,
        &r_gadget_ext,
        r_tail_len,
        levels,
        r_tail_dims_pow2,
    );

    #[allow(unused_mut)]
    let mut total = z_structured_contribution
        + w_structured_contribution
        + t_structured_contribution
        + setup_contribution
        + r_contribution;

    #[cfg(feature = "zk")]
    {
        let b_blinding =
            compute_b_blinding_part::<F, E, D>(prepared, full_vec_randomness, setup, alpha);
        let d_blinding =
            compute_d_blinding_part::<F, E, D>(prepared, full_vec_randomness, setup, alpha);
        total = total + b_blinding + d_blinding;
    }

    Ok(total)
}
