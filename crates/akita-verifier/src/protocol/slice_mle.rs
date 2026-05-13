//! Evaluate the M-table MLE from its non-zero slices.
//!
//! The verifier needs the multilinear-extension evaluation of a virtual
//! table `M` at a random point `r`. The naive approach is to materialize
//! the full equality table `eq(r, ·)`: that costs `O(|M|)` field operations
//! and `O(|M|)` memory, where `|M|` is linear in the witness size. Both are
//! too expensive.
//!
//! `M` is mostly zero. Only a handful of contiguous **slices** of `M` are
//! non-trivial. The MLE evaluation decomposes additively over those slices,
//! so we can evaluate each slice in isolation against the same `r` and sum
//! the results — each slice is orders of magnitude smaller than `M`.
//!
//! See `specs/optimized_verifier.md` for the full derivation.

use akita_algebra::eq_poly::EqPolynomial;
use akita_algebra::offset_eq::{eq_eval_at_index, eval_offset_eq_tensor};
use akita_algebra::ring::eval_ring_at_pows;
#[cfg(feature = "zk")]
use akita_algebra::ring::scalar_powers;
use akita_field::parallel::*;
use akita_field::{CanonicalField, ExtField, FieldCore};
use akita_types::{AkitaExpandedSetup, RingOpeningPoint};

use crate::protocol::ring_switch::RingSwitchDeferredRowEval;

/// Number of carry buckets per outer index produced by
/// [`StructuredSliceMleEvaluator::compute_inner_sum`].
///
/// **Note:** This module is only tested and intended for the
/// `POSSIBLE_CARRIES = 2` case. Anything other than `2` would require the
/// outer-sum algebra to be reworked; do not change this constant.
pub const POSSIBLE_CARRIES: usize = 2;

/// Inner-sum slot for the no-carry bucket (`carry = 0`).
pub const CARRY0: usize = 0;

/// Inner-sum slot for the one-carry bucket (`carry = 1`).
pub const CARRY1: usize = 1;

/// Peeled-block MLE evaluator for one structured slice of `M`. See
/// `specs/optimized_verifier.md` for the full derivation.
pub trait StructuredSliceMleEvaluator<F: FieldCore>: Sync {
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

/// W-segment slice evaluator. See `specs/optimized_verifier.md`.
pub struct WStructuredSlicesEvaluator<'a, F, E> {
    /// `full_vec_randomness[offset_low_bits..]` — slice's high-bit randomness.
    pub high_challenges: &'a [E],
    /// `offset >> offset_low_bits` — slice's high-bit offset.
    pub offset_high: usize,
    /// Gadget vector for the digit decomposition of `w`. Length =
    /// `num_digits`.
    pub gadget_vector: &'a [F],
    /// Per-opening-point carry summary of `opening_point.b`. Length =
    /// number of opening points.
    pub opening_point_block_summaries: &'a [[E; 2]],
    /// Per-claim carry summary of `c_alpha`. Length = `num_claims`.
    pub challenge_block_summaries: &'a [[E; 2]],
    /// RLC weights batching opening claims. Length = `num_claims`.
    pub gamma: &'a [E],
    /// `claim_to_point[claim_idx] = point_idx` (or all-zero in single-point).
    pub claim_to_point: &'a [usize],
    /// `tau1` equality weight for each opening-point input row of `M`.
    pub input_row_weights: &'a [E],
    /// `tau1` equality weight for the consistency-challenge row of `M`.
    pub challenge_weight: E,
}

impl<F, E> StructuredSliceMleEvaluator<E> for WStructuredSlicesEvaluator<'_, F, E>
where
    F: FieldCore,
    E: ExtField<F>,
{
    #[inline]
    fn num_outer_indices(&self) -> usize {
        self.gamma.len() * self.gadget_vector.len()
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
        let num_claims = self.gamma.len();
        let digit = outer_index / num_claims;
        let claim_idx = outer_index % num_claims;

        let point_idx = if self.opening_point_block_summaries.len() > 1 {
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

/// T-segment slice evaluator. See `specs/optimized_verifier.md`.
pub struct TStructuredSlicesEvaluator<'a, F, E> {
    /// `full_vec_randomness[offset_low_bits..]` — slice's high-bit randomness.
    pub high_challenges: &'a [E],
    /// `offset >> offset_low_bits` — slice's high-bit offset.
    pub offset_high: usize,
    /// Gadget vector for the digit decomposition of `w`. Length =
    /// `num_digits`.
    pub gadget_vector: &'a [F],
    /// Per-claim carry summary of `c_alpha`. Length = `num_claims`.
    pub challenge_block_summaries: &'a [[E; 2]],
    /// `tau1` equality weight for each `A`-row of `M`. Length =
    /// number of `A` rows.
    pub a_row_weights: &'a [E],
}

impl<F, E> StructuredSliceMleEvaluator<E> for TStructuredSlicesEvaluator<'_, F, E>
where
    F: FieldCore,
    E: ExtField<F>,
{
    #[inline]
    fn num_outer_indices(&self) -> usize {
        self.challenge_block_summaries.len() * self.gadget_vector.len() * self.a_row_weights.len()
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
        let num_claims = self.challenge_block_summaries.len();
        let num_digits = self.gadget_vector.len();
        let claim_idx = outer_index % num_claims;
        let compound = outer_index / num_claims;
        let digit = compound % num_digits;
        let a_row_idx = compound / num_digits;
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

/// Z-segment slice evaluator. See `specs/optimized_verifier.md`.
pub struct ZStructuredSlicesEvaluator<'a, F: FieldCore, E> {
    /// `full_vec_randomness[log₂(block_len)..]` — slice's high-bit
    /// randomness. Used by the peeled path.
    pub high_challenges: &'a [E],
    /// `offset_z >> log₂(block_len)` — slice's high-bit offset. Used by
    /// the peeled path.
    pub offset_high: usize,
    /// Commit-side gadget. Length = `depth_commit`.
    pub g1_commit: &'a [F],
    /// Fold-side gadget. Length = `depth_fold`.
    pub fold_gadget: &'a [F],
    /// Per-opening-point carry summary of `opening_point.a[..block_len]`.
    /// Empty in the dense fallback (non-pow2 `block_len`).
    pub a_block_summary: &'a [[E; 2]],
    /// `tau1` equality weight for the consistency-challenge row of `M`.
    pub consistency_weight: E,
    /// Opening points; used by the dense fallback.
    pub opening_points: &'a [RingOpeningPoint<F>],
    /// Full multilinear evaluation point; used by the dense fallback.
    pub full_vec_randomness: &'a [E],
    /// Start-of-slice offset of `z` inside `M`.
    pub offset_z: usize,
    /// Inner block size of the `z` segment.
    pub block_len: usize,
}

impl<F, E> StructuredSliceMleEvaluator<E> for ZStructuredSlicesEvaluator<'_, F, E>
where
    F: FieldCore,
    E: ExtField<F>,
{
    #[inline]
    fn num_outer_indices(&self) -> usize {
        self.opening_points.len() * self.fold_gadget.len() * self.g1_commit.len()
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
        let num_points = self.opening_points.len();
        let depth_fold = self.fold_gadget.len();
        let pt = outer_index % num_points;
        let q1 = outer_index / num_points;
        let df = q1 % depth_fold;
        let dc = q1 / depth_fold;

        let [a_carry0, a_carry1] = self.a_block_summary[pt];
        let scale = (-self.consistency_weight)
            .mul_base(self.g1_commit[dc])
            .mul_base(self.fold_gadget[df]);
        [scale * a_carry0, scale * a_carry1]
    }

    fn evaluate(&self) -> E {
        if self.block_len.is_power_of_two() {
            let n = self.num_outer_indices();
            let carry_terms: Vec<[E; POSSIBLE_CARRIES]> =
                (0..n).map(|q| self.compute_inner_sum(q)).collect();
            self.compute_outer_sum(&carry_terms)
        } else {
            let z_total_blocks = self.opening_points.len() * self.block_len;
            let z_len = self.fold_gadget.len() * self.g1_commit.len() * z_total_blocks;
            let z_segment_struct: Vec<E> = cfg_into_iter!(0..z_len)
                .map(|x| {
                    let compound_dig = x / z_total_blocks;
                    let global_blk = x % z_total_blocks;
                    let dc_idx = compound_dig / self.fold_gadget.len();
                    let df = compound_dig % self.fold_gadget.len();
                    let point_idx = global_blk / self.block_len;
                    let blk = global_blk % self.block_len;
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

/// Compute the `r`-tail contribution. Power-of-two `levels` uses a
/// multi-factor `eval_offset_eq_tensor`; otherwise materialises the
/// `r`-tail vector and falls back to the single-factor path.
pub(super) fn compute_r_contribution<F, E>(
    prepared: &RingSwitchDeferredRowEval<E>,
    full_vec_randomness: &[E],
    offset_r: usize,
    denom: E,
    r_gadget: &[F],
) -> E
where
    F: FieldCore + CanonicalField,
    E: ExtField<F>,
{
    let levels = r_gadget.len();
    if levels.is_power_of_two() {
        let _span = tracing::info_span!("r_structured").entered();
        let r_gadget_ext: Vec<E> = r_gadget.iter().copied().map(E::lift_base).collect();
        eval_offset_eq_tensor(
            full_vec_randomness,
            offset_r,
            -denom,
            &[&r_gadget_ext, &prepared.eq_tau1[..prepared.rows]],
        )
    } else {
        let _span = tracing::info_span!("r_dense").entered();
        let r_tail: Vec<E> = cfg_into_iter!(0..prepared.rows * levels)
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

/// ZK B-blinding contribution. See `specs/optimized_verifier.md`.
#[cfg(feature = "zk")]
pub(super) fn compute_b_blinding_part<F, E, const D: usize>(
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

/// ZK D-blinding contribution. See `specs/optimized_verifier.md`.
#[cfg(feature = "zk")]
pub(super) fn compute_d_blinding_part<F, E, const D: usize>(
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

/// Translate a D-column (D-physical order `[digit, block, claim]`) into
/// the M-layout `(low_block_eq_idx, high_eq_idx)` pair.
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

/// Translate a B-column (B-physical order `[digit, a_row, block, claim]`)
/// into `(low_block_eq_idx, high_eq_idx)`. `flat_claim` resolves the
/// per-group claim index to the global flat claim used by the high index.
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

/// Translate an A-column (A-physical order `[dc, block]`) into the
/// `(low_block_eq_idx, dc_idx, block_carry)` triple used to index
/// `z_block_low_eq` and the precomputed `s_per_dc_per_carry` table.
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

/// Sum `Σ r_eval[c] · (Σ_{p ∈ active} weight_p · pattern_p[c])` over one
/// contiguous column slice. The const generics select which of `{W, T, Z}`
/// is active — the compiler strips the inactive arms at monomorphisation.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn slice_inner_sum<F, E, const HAS_W: bool, const HAS_T: bool, const HAS_Z: bool>(
    range: std::ops::Range<usize>,
    r_eval: &[E],
    d_w: E,
    w_eq: &[E],
    b_w_for_groups: &[E],
    t_eq_per_group: &[Vec<E>],
    num_groups: usize,
    a_w: E,
    z_eq: &[E],
) -> E
where
    F: FieldCore,
    E: ExtField<F>,
{
    cfg_into_iter!(range)
        .map(|c| {
            let mut m = E::zero();
            if HAS_W {
                m += d_w * w_eq[c];
            }
            if HAS_T {
                for g in 0..num_groups {
                    m += b_w_for_groups[g] * t_eq_per_group[g][c];
                }
            }
            if HAS_Z {
                m += a_w * z_eq[c];
            }
            r_eval[c] * m
        })
        .sum()
}

/// Compute the fused setup-matrix contribution `D · ŵ + B · t̂ + A · ẑ`
/// as a single `<M_Flat, Eval>` over the shared SIS matrix. W, T, and Z
/// share `r_eval[c] = M_Flat[row, c]` for every row that participates in
/// more than one half. Per-row, the column axis is partitioned into three
/// contiguous slices sorted by each pattern's endpoint; the active subset
/// of `{W, T, Z}` is constant inside each slice and selected at the type
/// level via `slice_inner_sum`'s const generics. See
/// `specs/optimized_verifier.md` for the full derivation.
#[allow(clippy::too_many_arguments)]
pub(super) fn compute_setup_contribution<F, E, const D: usize>(
    prepared: &RingSwitchDeferredRowEval<E>,
    full_vec_randomness: &[E],
    setup: &AkitaExpandedSetup<F>,
    eq_low: &[E],
    z_block_low_eq: &[E],
    alpha_pows: &[E],
    fold_gadget: &[F],
    offset_w: usize,
    offset_t: usize,
    offset_z: usize,
) -> E
where
    F: FieldCore + CanonicalField,
    E: ExtField<F>,
{
    let block_bits = prepared.num_blocks.trailing_zeros() as usize;
    let block_mask = prepared.num_blocks.wrapping_sub(1);
    let block_offset_low = offset_w & block_mask;
    let w_offset_high = offset_w >> block_bits;
    let t_offset_high = offset_t >> block_bits;
    let high_challenges = &full_vec_randomness[block_bits..];

    let z_offset_low_bits = prepared.block_len.trailing_zeros() as usize;
    let z_offset_low = offset_z & prepared.block_len.wrapping_sub(1);
    let z_range = prepared.inner_width;
    let z_used = prepared.n_a > 0 && z_range > 0;
    let z_dims_pow2 = prepared.block_len.is_power_of_two();

    let b_start = 1 + prepared.num_public_eval_rows + prepared.n_d;
    let d_start = 1 + prepared.num_public_eval_rows;
    let a_start = b_start + prepared.n_b * prepared.num_commitment_groups;
    let d_weights = &prepared.eq_tau1[d_start..(d_start + prepared.n_d)];
    let a_weights = &prepared.eq_tau1[a_start..prepared.rows];

    let stride_t = prepared.n_a * prepared.depth_open;
    let cols_per_claim_t = stride_t * prepared.num_blocks;
    let b_per_claim_w = prepared.num_blocks * prepared.depth_open;
    let n_cols_w = prepared.num_claims * b_per_claim_w;

    // Invert `claim_to_group`: T's row weight is group-dependent and its
    // c-axis indexes `claim_within_group`; we need `(g, c_in_g) →
    // flat_claim` to recompute the q_T high index per cell.
    let mut claims_per_group = vec![0usize; prepared.num_commitment_groups.max(1)];
    for &(g, _) in &prepared.claim_to_group {
        claims_per_group[g] += 1;
    }
    let max_claims_per_group = claims_per_group.iter().copied().max().unwrap_or(0);
    let mut flat_claim_for_group: Vec<Vec<Option<usize>>> =
        vec![vec![None; max_claims_per_group]; prepared.num_commitment_groups];
    for (flat_idx, &(g, c_in_g)) in prepared.claim_to_group.iter().enumerate() {
        flat_claim_for_group[g][c_in_g] = Some(flat_idx);
    }
    let n_cols_t = max_claims_per_group * cols_per_claim_t;

    // Row range covers every SIS row that any of W/T/Z touch. Z extends
    // it to `n_a` when active, so Z-only rows participate inside the loop
    // — no separate post-loop matrix-A scan.
    let r_max = if z_used {
        prepared.n_d.max(prepared.n_b).max(prepared.n_a)
    } else {
        prepared.n_d.max(prepared.n_b)
    };
    let n_cols_total = n_cols_w.max(n_cols_t).max(if z_used { z_range } else { 0 });
    assert!(
        n_cols_total > 0,
        "matrix-row pattern evaluation requires at least one SIS column"
    );
    assert!(
        r_max > 0,
        "matrix-row pattern evaluation requires at least one SIS row"
    );

    let eq_hi_w_table: Vec<E> = (0..=prepared.num_claims * prepared.depth_open)
        .map(|k| eq_eval_at_index(high_challenges, w_offset_high + k))
        .collect();
    let eq_hi_t_table: Vec<E> = (0..=prepared.num_claims * prepared.depth_open * prepared.n_a)
        .map(|k| eq_eval_at_index(high_challenges, t_offset_high + k))
        .collect();

    let w_eq_slice: Vec<E> = cfg_into_iter!(0..n_cols_w)
        .map(|current_index| {
            let (low_eq_idx, high_eq_idx) = get_eq_indices_for_d(
                current_index,
                prepared.depth_open,
                prepared.num_blocks,
                prepared.num_claims,
                b_per_claim_w,
                block_offset_low,
                block_mask,
                block_bits,
            );
            eq_low[low_eq_idx] * eq_hi_w_table[high_eq_idx]
        })
        .collect();

    let t_eq_slice_per_group: Vec<Vec<E>> = (0..prepared.num_commitment_groups)
        .map(|g| {
            let k_g = claims_per_group[g];
            cfg_into_iter!(0..n_cols_t)
                .map(|c| {
                    let claim_within_group = c / cols_per_claim_t;
                    if claim_within_group >= k_g {
                        return E::zero();
                    }
                    match flat_claim_for_group[g][claim_within_group] {
                        Some(flat_claim) => {
                            let (low_eq_idx, high_eq_idx) = get_eq_indices_for_b(
                                c,
                                flat_claim,
                                prepared.depth_open,
                                prepared.n_a,
                                prepared.num_blocks,
                                prepared.num_claims,
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

    // `z_eq_slice[c]` — column-only Z pattern. Length `z_range`, empty
    // when `!z_used`. Pow2: peeled-block lookup `z_block_low_eq[low] ·
    // S_per_dc_per_carry[dc][carry]`. Non-pow2: dense aggregation over
    // `(pt, df)` with a one-shot peeled eq cache so per-cell cost stays
    // O(P · DF).
    let z_eq_slice: Vec<E> = if !z_used {
        Vec::new()
    } else if z_dims_pow2 {
        // `S_per_dc_per_carry[dc][carry] = -Σ_{pt, df} fold_gadget[df]
        //   · eq_hi_z[z_offset_high + (pt + P·df + P·DF·dc) + carry]`
        let z_offset_high = offset_z >> z_offset_low_bits;
        let z_block_mask = prepared.block_len.wrapping_sub(1);
        let s_per_dc_per_carry: Vec<[E; POSSIBLE_CARRIES]> = {
            let z_high_challenges = &full_vec_randomness[z_offset_low_bits..];
            let num_q_z = prepared.num_points * prepared.depth_fold * prepared.depth_commit;
            let eq_hi_z_table: Vec<E> = (0..=num_q_z)
                .map(|k| eq_eval_at_index(z_high_challenges, z_offset_high + k))
                .collect();
            (0..prepared.depth_commit)
                .map(|dc| {
                    let mut s = [E::zero(); POSSIBLE_CARRIES];
                    for (carry_slot, slot) in s.iter_mut().enumerate() {
                        let mut acc = E::zero();
                        for (df, &fg) in fold_gadget.iter().enumerate().take(prepared.depth_fold) {
                            for pt in 0..prepared.num_points {
                                let k = pt
                                    + prepared.num_points * df
                                    + prepared.num_points * prepared.depth_fold * dc
                                    + carry_slot;
                                acc += eq_hi_z_table[k].mul_base(fg);
                            }
                        }
                        *slot = -acc;
                    }
                    s
                })
                .collect()
        };
        cfg_into_iter!(0..z_range)
            .map(|c| {
                let (low_eq_idx, depth_commit_idx, block_carry) = get_eq_indices_for_a(
                    c,
                    prepared.depth_commit,
                    z_offset_low,
                    z_block_mask,
                    z_offset_low_bits,
                );
                z_block_low_eq[low_eq_idx] * s_per_dc_per_carry[depth_commit_idx][block_carry]
            })
            .collect()
    } else {
        // Build a peeled eq cache so each per-cell `eq(r, offset_z +
        // j_M^Z)` is O(1) instead of O(|r|).
        let z_total_blocks_dense = prepared.block_len * prepared.num_points;
        let z_len_dense = prepared.depth_fold * prepared.depth_commit * z_total_blocks_dense;
        let n_rand = full_vec_randomness.len();
        let k = z_len_dense
            .saturating_sub(1)
            .checked_next_power_of_two()
            .map(|p| p.trailing_zeros() as usize)
            .unwrap_or(0)
            .max(1)
            .min(n_rand);
        let mask = (1usize << k) - 1;
        let offset_z_dense_low = offset_z & mask;
        let offset_z_dense_high = offset_z >> k;
        let eq_low_z_dense = EqPolynomial::evals(&full_vec_randomness[..k]);
        let max_high = (offset_z + z_len_dense - 1) >> k;
        let n_high = max_high - offset_z_dense_high + 1;
        let eq_high_z_dense: Vec<E> = (0..n_high)
            .map(|h| eq_eval_at_index(&full_vec_randomness[k..], offset_z_dense_high + h))
            .collect();

        cfg_into_iter!(0..z_range)
            .map(|c| {
                let dc = c % prepared.depth_commit;
                let blk = c / prepared.depth_commit;
                let mut acc = E::zero();
                for pt in 0..prepared.num_points {
                    for (df, &fg) in fold_gadget.iter().enumerate().take(prepared.depth_fold) {
                        // j_M^Z(c, pt, df) = blk + B·pt + B·P·df + B·P·DF·dc
                        let x = blk
                            + prepared.block_len * pt
                            + prepared.block_len * prepared.num_points * df
                            + prepared.block_len * prepared.num_points * prepared.depth_fold * dc;
                        let sum = offset_z_dense_low + x;
                        let low_idx = sum & mask;
                        let high_idx = sum >> k;
                        let eq_val = eq_low_z_dense[low_idx]
                            * eq_high_z_dense[high_idx - offset_z_dense_high];
                        acc += eq_val.mul_base(fg);
                    }
                }
                -acc
            })
            .collect()
    };

    // Per-row inner products. Each row's column axis splits into three
    // contiguous slices sorted by which pattern's endpoint comes first;
    // the active subset of {W, T, Z} is constant inside each slice and
    // dispatched via `slice_inner_sum`'s const generics. The B / D / A
    // sub-matrices alias the same backing storage.
    #[derive(Copy, Clone)]
    enum Pat {
        W,
        T,
        Z,
    }
    let shared_view = setup
        .shared_matrix
        .ring_view::<D>(r_max, setup.seed.max_stride);

    let row_contribs: Vec<E> = cfg_into_iter!(0..r_max)
        .map(|row| {
            let row_slice = shared_view.row(row);

            let e_w = if row < prepared.n_d { n_cols_w } else { 0 };
            let e_t = if row < prepared.n_b { n_cols_t } else { 0 };
            let e_z = if row < prepared.n_a && z_used {
                z_range
            } else {
                0
            };

            let mut ends = [(e_w, Pat::W), (e_t, Pat::T), (e_z, Pat::Z)];
            ends.sort_by_key(|&(e, _)| e);
            let [(e1, k1), (e2, _), (e3, k3)] = ends;
            if e3 == 0 {
                return E::zero();
            }

            let r_eval: Vec<E> = cfg_into_iter!(0..e3)
                .map(|c| eval_ring_at_pows(&row_slice[c], alpha_pows))
                .collect();

            // `b_w_for_groups` is only read when `HAS_T = true`, which
            // implies `e_t > 0 ⟹ row < n_b ⟹ Vec is non-empty`. Same
            // guarantee applies to `d_weights[row]` and `a_weights[row]`
            // — passed inline since each `HAS_X = true` implies `e_X > 0`.
            let b_w_for_groups: Vec<E> = if row < prepared.n_b {
                (0..prepared.num_commitment_groups)
                    .map(|g| prepared.eq_tau1[b_start + g * prepared.n_b + row])
                    .collect()
            } else {
                Vec::new()
            };

            // Slice 1: `[0, e1)` — all three active.
            let s1 = if e1 > 0 {
                slice_inner_sum::<F, E, true, true, true>(
                    0..e1,
                    &r_eval,
                    d_weights[row],
                    &w_eq_slice,
                    &b_w_for_groups,
                    &t_eq_slice_per_group,
                    prepared.num_commitment_groups,
                    a_weights[row],
                    &z_eq_slice,
                )
            } else {
                E::zero()
            };

            // Slice 2: `[e1, e2)` — drop the pattern at `e1`.
            let s2 = if e2 > e1 {
                match k1 {
                    Pat::W => slice_inner_sum::<F, E, false, true, true>(
                        e1..e2,
                        &r_eval,
                        E::zero(),
                        &w_eq_slice,
                        &b_w_for_groups,
                        &t_eq_slice_per_group,
                        prepared.num_commitment_groups,
                        a_weights[row],
                        &z_eq_slice,
                    ),
                    Pat::T => slice_inner_sum::<F, E, true, false, true>(
                        e1..e2,
                        &r_eval,
                        d_weights[row],
                        &w_eq_slice,
                        &b_w_for_groups,
                        &t_eq_slice_per_group,
                        prepared.num_commitment_groups,
                        a_weights[row],
                        &z_eq_slice,
                    ),
                    Pat::Z => slice_inner_sum::<F, E, true, true, false>(
                        e1..e2,
                        &r_eval,
                        d_weights[row],
                        &w_eq_slice,
                        &b_w_for_groups,
                        &t_eq_slice_per_group,
                        prepared.num_commitment_groups,
                        E::zero(),
                        &z_eq_slice,
                    ),
                }
            } else {
                E::zero()
            };

            // Slice 3: `[e2, e3)` — only `k3` is active.
            let s3 = if e3 > e2 {
                match k3 {
                    Pat::W => slice_inner_sum::<F, E, true, false, false>(
                        e2..e3,
                        &r_eval,
                        d_weights[row],
                        &w_eq_slice,
                        &b_w_for_groups,
                        &t_eq_slice_per_group,
                        prepared.num_commitment_groups,
                        E::zero(),
                        &z_eq_slice,
                    ),
                    Pat::T => slice_inner_sum::<F, E, false, true, false>(
                        e2..e3,
                        &r_eval,
                        E::zero(),
                        &w_eq_slice,
                        &b_w_for_groups,
                        &t_eq_slice_per_group,
                        prepared.num_commitment_groups,
                        E::zero(),
                        &z_eq_slice,
                    ),
                    Pat::Z => slice_inner_sum::<F, E, false, false, true>(
                        e2..e3,
                        &r_eval,
                        E::zero(),
                        &w_eq_slice,
                        &b_w_for_groups,
                        &t_eq_slice_per_group,
                        prepared.num_commitment_groups,
                        a_weights[row],
                        &z_eq_slice,
                    ),
                }
            } else {
                E::zero()
            };

            s1 + s2 + s3
        })
        .collect();

    row_contribs.into_iter().sum::<E>()
}
