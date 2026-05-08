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

use crate::protocol::ring_switch::{
    summarize_pow2_block_carries_base, summarize_strided_pow2_block_carries, PreparedMEval,
};

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

/// Public/opening-point + consistency contribution to the `\hat w` slice.
///
/// `q = dig · num_claims + claim_idx`, two sources per `q` (public part,
/// consistency part), each summarized via a precomputed `[low0, low1]` table.
pub struct WSepEvaluator<'a, F, E> {
    /// `full_vec_randomness[offset_low_bits..]` — slice's high-bit randomness.
    pub high_challenges: &'a [E],
    /// `offset >> offset_low_bits` — slice's high-bit offset.
    pub offset_high: usize,
    /// Gadget vector `g_open[dig]`, base scalars.
    pub g1_open: &'a [F],
    /// `[low0, low1]` summary of `opening_point.b` for each opening point.
    pub opening_point_block_summaries: &'a [[E; 2]],
    /// `[low0, low1]` summary of `c_alpha[claim, ·]` for each claim.
    pub challenge_block_summaries: &'a [[E; 2]],
    /// Batching coefficient per claim.
    pub gamma: &'a [E],
    /// `claim_to_point[claim_idx] = point_idx` (or all-zero in single-point).
    pub claim_to_point: &'a [usize],
    /// `tau1` weight for each public-row entry (one per opening point).
    pub public_weights: &'a [E],
    /// `tau1` weight for the consistency row.
    pub consistency_weight: E,
    /// Number of evaluation claims.
    pub num_claims: usize,
    /// Decomposition depth for the opening direction.
    pub depth_open: usize,
    /// Whether the protocol uses multiple opening points.
    pub is_multi_point: bool,
}

impl<F, E> SliceMleEvaluator<E> for WSepEvaluator<'_, F, E>
where
    F: FieldCore,
    E: ExtField<F>,
{
    #[inline]
    fn num_outer_indices(&self) -> usize {
        self.num_claims * self.depth_open
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
        let dig = outer_index / self.num_claims;
        let claim_idx = outer_index % self.num_claims;
        let g_open = self.g1_open[dig];

        let point_idx = if self.is_multi_point {
            self.claim_to_point[claim_idx]
        } else {
            0
        };
        let [pub0, pub1] = self.opening_point_block_summaries[point_idx];
        let public_scale =
            (self.public_weights[point_idx] * self.gamma[claim_idx]).mul_base(g_open);

        let [c0, c1] = self.challenge_block_summaries[claim_idx];
        let challenge_scale = self.consistency_weight.mul_base(g_open);

        [
            public_scale * pub0 + challenge_scale * c0,
            public_scale * pub1 + challenge_scale * c1,
        ]
    }
}

/// `D · \hat w` contribution to the `\hat w` slice.
///
/// `q = dig · num_claims + claim_idx`, `n_d` sources per `q` (D rows),
/// summarized via [`summarize_strided_pow2_block_carries`] over the `D` row.
pub struct WdEvaluator<'a, F: FieldCore, E, const D: usize> {
    /// `full_vec_randomness[offset_low_bits..]` — slice's high-bit randomness.
    pub high_challenges: &'a [E],
    /// `offset >> offset_low_bits` — slice's high-bit offset.
    pub offset_high: usize,
    /// Precomputed `eq(full_vec_randomness[..offset_low_bits], ·)`,
    /// length `1 << offset_low_bits`.
    pub eq_low: &'a [E],
    /// `offset & ((1 << offset_low_bits) - 1)`.
    pub offset_low: usize,
    /// `tau1` weights for the `D` rows; one entry per row.
    pub d_weights: &'a [E],
    /// View of the `D` matrix at ring dimension `D`.
    pub d_view: RingMatrixView<'a, F, D>,
    /// Powers of the ring-switch challenge `alpha`.
    pub alpha_pows: &'a [E],
    /// Width of the block dimension (must be a power of two).
    pub num_blocks: usize,
    /// Number of evaluation claims.
    pub num_claims: usize,
    /// Decomposition depth for the opening direction.
    pub depth_open: usize,
    /// `num_blocks * depth_open`.
    pub per_claim_d_width: usize,
}

impl<F, E, const D: usize> SliceMleEvaluator<E> for WdEvaluator<'_, F, E, D>
where
    F: FieldCore,
    E: ExtField<F>,
{
    #[inline]
    fn num_outer_indices(&self) -> usize {
        self.num_claims * self.depth_open
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
    fn parallelize_outer(&self) -> bool {
        true
    }

    #[inline]
    fn compute_inner_sum(&self, outer_index: usize) -> [E; POSSIBLE_CARRIES] {
        let claim_idx = outer_index % self.num_claims;
        let dig = outer_index / self.num_claims;
        let lane_offset = claim_idx * self.per_claim_d_width + dig;
        let mut out = [E::zero(); POSSIBLE_CARRIES];
        for (di, &d_weight) in self.d_weights.iter().enumerate() {
            if d_weight.is_zero() {
                continue;
            }
            let [a, b] = summarize_strided_pow2_block_carries::<F, E, D>(
                self.eq_low,
                self.offset_low,
                self.d_view.row(di),
                self.alpha_pows,
                self.num_blocks,
                self.depth_open,
                lane_offset,
            );
            out[CARRY0] += d_weight * a;
            out[CARRY1] += d_weight * b;
        }
        out
    }
}

/// Consistency contribution to the `\hat t` slice.
///
/// `q = num_claims · (digit_idx + depth_open · a_idx) + claim_idx`,
/// one source per `q`, summarized via `challenge_block_summaries[claim_idx]`.
pub struct TSepEvaluator<'a, F, E> {
    /// `full_vec_randomness[offset_low_bits..]` — slice's high-bit randomness.
    pub high_challenges: &'a [E],
    /// `offset >> offset_low_bits` — slice's high-bit offset.
    pub offset_high: usize,
    /// Gadget vector `g_open[dig]`, base scalars.
    pub g1_open: &'a [F],
    /// `[low0, low1]` summary of `c_alpha[claim, ·]` for each claim.
    pub challenge_block_summaries: &'a [[E; 2]],
    /// `tau1` weights for the `A` rows; one entry per row.
    pub a_weights: &'a [E],
    /// Number of evaluation claims.
    pub num_claims: usize,
    /// Decomposition depth for the opening direction.
    pub depth_open: usize,
}

impl<F, E> SliceMleEvaluator<E> for TSepEvaluator<'_, F, E>
where
    F: FieldCore,
    E: ExtField<F>,
{
    #[inline]
    fn num_outer_indices(&self) -> usize {
        self.num_claims * self.depth_open * self.a_weights.len()
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
        let digit_idx = compound % self.depth_open;
        let a_idx = compound / self.depth_open;
        let scale = self.a_weights[a_idx].mul_base(self.g1_open[digit_idx]);
        let [c0, c1] = self.challenge_block_summaries[claim_idx];
        [scale * c0, scale * c1]
    }
}

/// `B · \hat t` contribution to the `\hat t` slice.
///
/// Same `q` decoding as [`TSepEvaluator`]; `n_b` sources per `q`, summarized
/// via [`summarize_strided_pow2_block_carries`] over the `B` row.
pub struct TbEvaluator<'a, F: FieldCore, E, const D: usize> {
    /// `full_vec_randomness[offset_low_bits..]` — slice's high-bit randomness.
    pub high_challenges: &'a [E],
    /// `offset >> offset_low_bits` — slice's high-bit offset.
    pub offset_high: usize,
    /// Precomputed `eq(full_vec_randomness[..offset_low_bits], ·)`,
    /// length `1 << offset_low_bits`.
    pub eq_low: &'a [E],
    /// `offset & ((1 << offset_low_bits) - 1)`.
    pub offset_low: usize,
    /// Full `tau1` equality table (the evaluator slices into the `b_start`
    /// region per call).
    pub eq_tau1: &'a [E],
    /// View of the `B` matrix at ring dimension `D`.
    pub b_view: RingMatrixView<'a, F, D>,
    /// Powers of the ring-switch challenge `alpha`.
    pub alpha_pows: &'a [E],
    /// `(group_idx, claim_idx_within_group)` for each claim.
    pub claim_to_group: &'a [(usize, usize)],
    /// Width of the block dimension (must be a power of two).
    pub num_blocks: usize,
    /// Number of evaluation claims.
    pub num_claims: usize,
    /// Decomposition depth for the opening direction.
    pub depth_open: usize,
    /// Number of `A` rows.
    pub n_a: usize,
    /// Number of `B` rows.
    pub n_b: usize,
    /// Index where the `B` weights begin inside `eq_tau1`.
    pub b_start: usize,
    /// `n_a * depth_open`.
    pub t_compound_per_block: usize,
    /// `n_a * depth_open * num_blocks`.
    pub t_cols_per_claim: usize,
}

impl<F, E, const D: usize> SliceMleEvaluator<E> for TbEvaluator<'_, F, E, D>
where
    F: FieldCore,
    E: ExtField<F>,
{
    #[inline]
    fn num_outer_indices(&self) -> usize {
        self.num_claims * self.depth_open * self.n_a
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
    fn parallelize_outer(&self) -> bool {
        true
    }

    #[inline]
    fn compute_inner_sum(&self, outer_index: usize) -> [E; POSSIBLE_CARRIES] {
        let claim_idx = outer_index % self.num_claims;
        let compound = outer_index / self.num_claims;
        let a_idx = compound / self.depth_open;
        let digit_idx = compound % self.depth_open;
        let (group_idx, claim_idx_within_group) = self.claim_to_group[claim_idx];
        let weights_start = self.b_start + group_idx * self.n_b;
        let commitment_weights = &self.eq_tau1[weights_start..(weights_start + self.n_b)];
        let lane_offset =
            claim_idx_within_group * self.t_cols_per_claim + a_idx * self.depth_open + digit_idx;
        let mut out = [E::zero(); POSSIBLE_CARRIES];
        for (row_idx, &eq_i) in commitment_weights.iter().enumerate() {
            if eq_i.is_zero() {
                continue;
            }
            let [a, b] = summarize_strided_pow2_block_carries::<F, E, D>(
                self.eq_low,
                self.offset_low,
                self.b_view.row(row_idx),
                self.alpha_pows,
                self.num_blocks,
                self.t_compound_per_block,
                lane_offset,
            );
            out[CARRY0] += eq_i * a;
            out[CARRY1] += eq_i * b;
        }
        out
    }
}

// ---------------------------------------------------------------------------
// 3. Eval-at-point breakdown
// ---------------------------------------------------------------------------

/// Breakdown of [`PreparedMEval::eval_at_point`] into its seven additive
/// contributions. Their sum is the full M-table evaluation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EvalAtPointParts<E> {
    /// Dense `z` slice contribution (uses tensor evaluator, not slice-MLE).
    pub z_dense: E,
    /// Public + consistency contribution to `\hat w` (slice-MLE).
    pub w_sep: E,
    /// `D · \hat w` contribution to `\hat w` (slice-MLE).
    pub w_d: E,
    /// Consistency contribution to `\hat t` (slice-MLE).
    pub t_sep: E,
    /// `B · \hat t` contribution to `\hat t` (slice-MLE).
    pub t_b: E,
    /// Power-of-two `r`-tail contribution (uses tensor evaluator).
    pub r_sep: E,
    /// Non-power-of-two `r`-tail contribution (uses tensor evaluator).
    pub r_dense: E,
}

impl<E: FieldCore> EvalAtPointParts<E> {
    /// Total M-evaluation: sum of all seven contributions.
    pub fn sum(&self) -> E {
        self.z_dense + self.w_sep + self.w_d + self.t_sep + self.t_b + self.r_sep + self.r_dense
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
    z_len: usize,
    r_tail_len: usize,
    z_total_blocks: usize,
    inner_width: usize,
    offset_z: usize,
    offset_w: usize,
    offset_t: usize,
    offset_r: usize,
    offset_low_bits: usize,
    is_multi_point: bool,
    opening_point_block_summaries: Vec<[E; 2]>,
    challenge_block_summaries: Vec<[E; 2]>,
    denom: E,
    r_tail_dims_pow2: bool,
}

impl<'a, F, E, const D: usize> EvalAtPointWorkspace<'a, F, E, D>
where
    F: FieldCore + CanonicalField,
    E: ExtField<F>,
{
    fn build(
        prepared: &'a PreparedMEval<E>,
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
        let public_weights = &prepared.eq_tau1[1..(1 + prepared.num_eval_rows)];
        let d_start = 1 + prepared.num_eval_rows;
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

        let offset_z = if prepared.z_first { 0 } else { w_len + t_len };
        let offset_w = if prepared.z_first { z_len } else { 0 };
        let offset_t = if prepared.z_first {
            z_len + w_len
        } else {
            w_len
        };
        let offset_r = w_len + t_len + z_len;
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
            z_len,
            r_tail_len,
            z_total_blocks,
            inner_width,
            offset_z,
            offset_w,
            offset_t,
            offset_r,
            offset_low_bits,
            is_multi_point,
            opening_point_block_summaries,
            challenge_block_summaries,
            denom,
            r_tail_dims_pow2,
        }
    }
}

/// Compute the three contributions that do NOT participate in the slice-MLE
/// abstraction (`z_dense`, `r_sep`, `r_dense`).
fn compute_non_peeled_parts<F, E, const D: usize>(
    prepared: &PreparedMEval<E>,
    full_vec_randomness: &[E],
    opening_points: &[RingOpeningPoint<F>],
    ws: &EvalAtPointWorkspace<'_, F, E, D>,
) -> (E, E, E)
where
    F: FieldCore + CanonicalField,
    E: ExtField<F>,
{
    let z_base_len = prepared.num_points * ws.inner_width;
    let z_base: Vec<E> = {
        let _span = tracing::info_span!("m_eval_z_base").entered();
        cfg_into_iter!(0..z_base_len)
            .map(|k| {
                let point_idx = if ws.is_multi_point {
                    k / ws.inner_width
                } else {
                    0
                };
                let local_k = if ws.is_multi_point {
                    k % ws.inner_width
                } else {
                    k
                };
                let block_idx = local_k / prepared.depth_commit;
                let digit_idx = local_k % prepared.depth_commit;
                let opening_point = &opening_points[point_idx];
                let base_scale = opening_point.a[block_idx] * ws.g1_commit[digit_idx];
                let mut acc = ws.consistency_weight.mul_base(base_scale);
                for (a_idx, eq_i) in ws.a_weights.iter().enumerate() {
                    if !eq_i.is_zero() {
                        acc += *eq_i
                            * eval_ring_at_pows(&ws.a_view.row(a_idx)[local_k], &ws.alpha_pows);
                    }
                }
                acc
            })
            .collect()
    };

    let z_dense = {
        let _span = tracing::info_span!("m_eval_z_dense").entered();
        let z_segment: Vec<E> = cfg_into_iter!(0..ws.z_len)
            .map(|x| {
                let compound_dig = x / ws.z_total_blocks;
                let global_blk = x % ws.z_total_blocks;
                let dc = compound_dig / prepared.depth_fold;
                let df = compound_dig % prepared.depth_fold;
                let point_idx = global_blk / prepared.block_len;
                let blk = global_blk % prepared.block_len;
                let phys_k = point_idx * ws.inner_width + blk * prepared.depth_commit + dc;
                -z_base[phys_k].mul_base(ws.fold_gadget[df])
            })
            .collect();
        eval_offset_eq_tensor(
            full_vec_randomness,
            ws.offset_z,
            E::one(),
            &[z_segment.as_slice()],
        )
    };

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

    (z_dense, r_sep, r_dense)
}

/// Helpers that build evaluators from the workspace.
///
/// Each helper takes the slice-derived state — `(high_challenges, offset_high)`
/// for the outer-sum side, and (when the evaluator does a strided block
/// scan) `(eq_low, offset_low)` for the inner-sum side — so the same
/// workspace can be reused across slices at different offsets.
#[allow(clippy::too_many_arguments)]
fn build_w_sep_evaluator<'a, F, E, const D: usize>(
    prepared: &'a PreparedMEval<E>,
    ws: &'a EvalAtPointWorkspace<'a, F, E, D>,
    high_challenges: &'a [E],
    offset_high: usize,
) -> WSepEvaluator<'a, F, E>
where
    F: FieldCore,
    E: FieldCore,
{
    WSepEvaluator {
        high_challenges,
        offset_high,
        g1_open: &ws.g1_open,
        opening_point_block_summaries: &ws.opening_point_block_summaries,
        challenge_block_summaries: &ws.challenge_block_summaries,
        gamma: &prepared.gamma,
        claim_to_point: &prepared.claim_to_point,
        public_weights: ws.public_weights,
        consistency_weight: ws.consistency_weight,
        num_claims: prepared.num_claims,
        depth_open: prepared.depth_open,
        is_multi_point: ws.is_multi_point,
    }
}

#[allow(clippy::too_many_arguments)]
fn build_w_d_evaluator<'a, F, E, const D: usize>(
    prepared: &'a PreparedMEval<E>,
    ws: &'a EvalAtPointWorkspace<'a, F, E, D>,
    high_challenges: &'a [E],
    offset_high: usize,
    eq_low: &'a [E],
    offset_low: usize,
) -> WdEvaluator<'a, F, E, D>
where
    F: FieldCore,
    E: FieldCore,
{
    WdEvaluator {
        high_challenges,
        offset_high,
        eq_low,
        offset_low,
        d_weights: ws.d_weights,
        d_view: ws.d_view,
        alpha_pows: &ws.alpha_pows,
        num_blocks: prepared.num_blocks,
        num_claims: prepared.num_claims,
        depth_open: prepared.depth_open,
        per_claim_d_width: prepared.num_blocks * prepared.depth_open,
    }
}

#[allow(clippy::too_many_arguments)]
fn build_t_sep_evaluator<'a, F, E, const D: usize>(
    prepared: &'a PreparedMEval<E>,
    ws: &'a EvalAtPointWorkspace<'a, F, E, D>,
    high_challenges: &'a [E],
    offset_high: usize,
) -> TSepEvaluator<'a, F, E>
where
    F: FieldCore,
    E: FieldCore,
{
    TSepEvaluator {
        high_challenges,
        offset_high,
        g1_open: &ws.g1_open,
        challenge_block_summaries: &ws.challenge_block_summaries,
        a_weights: ws.a_weights,
        num_claims: prepared.num_claims,
        depth_open: prepared.depth_open,
    }
}

#[allow(clippy::too_many_arguments)]
fn build_t_b_evaluator<'a, F, E, const D: usize>(
    prepared: &'a PreparedMEval<E>,
    ws: &'a EvalAtPointWorkspace<'a, F, E, D>,
    high_challenges: &'a [E],
    offset_high: usize,
    eq_low: &'a [E],
    offset_low: usize,
) -> TbEvaluator<'a, F, E, D>
where
    F: FieldCore,
    E: FieldCore,
{
    let t_compound_per_block = prepared.n_a * prepared.depth_open;
    TbEvaluator {
        high_challenges,
        offset_high,
        eq_low,
        offset_low,
        eq_tau1: &prepared.eq_tau1,
        b_view: ws.b_view,
        alpha_pows: &ws.alpha_pows,
        claim_to_group: &prepared.claim_to_group,
        num_blocks: prepared.num_blocks,
        num_claims: prepared.num_claims,
        depth_open: prepared.depth_open,
        n_a: prepared.n_a,
        n_b: prepared.n_b,
        b_start: ws.b_start,
        t_compound_per_block,
        t_cols_per_claim: t_compound_per_block * prepared.num_blocks,
    }
}

/// Compute every additive contribution of `PreparedMEval::eval_at_point`
/// separately, returning them as [`EvalAtPointParts`].
///
/// The four slice-MLE parts (`w_sep`, `w_d`, `t_sep`, `t_b`) go through the
/// [`SliceMleEvaluator`] abstraction; the three remaining parts
/// (`z_dense`, `r_sep`, `r_dense`) go through the existing tensor-evaluator
/// helpers in [`compute_non_peeled_parts`].
///
/// `PreparedMEval::eval_at_point` is a thin wrapper that calls this function
/// and sums the parts.
///
/// # Errors
///
/// Returns the same errors as `eval_at_point`.
pub fn eval_at_point_parts<F, E, const D: usize>(
    prepared: &PreparedMEval<E>,
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
        build_w_sep_evaluator::<F, E, D>(prepared, &ws, high_challenges, w_offset_high).evaluate()
    };
    let w_d = {
        let _span = tracing::info_span!("m_eval_w_d").entered();
        build_w_d_evaluator::<F, E, D>(
            prepared,
            &ws,
            high_challenges,
            w_offset_high,
            &eq_low,
            w_offset_low,
        )
        .evaluate()
    };
    let t_sep = {
        let _span = tracing::info_span!("m_eval_t_sep").entered();
        build_t_sep_evaluator::<F, E, D>(prepared, &ws, high_challenges, t_offset_high).evaluate()
    };
    let t_b = {
        let _span = tracing::info_span!("m_eval_t_b").entered();
        build_t_b_evaluator::<F, E, D>(
            prepared,
            &ws,
            high_challenges,
            t_offset_high,
            &eq_low,
            t_offset_low,
        )
        .evaluate()
    };
    let (z_dense, r_sep, r_dense) =
        compute_non_peeled_parts::<F, E, D>(prepared, full_vec_randomness, opening_points, &ws);
    Ok(EvalAtPointParts {
        z_dense,
        w_sep,
        w_d,
        t_sep,
        t_b,
        r_sep,
        r_dense,
    })
}
