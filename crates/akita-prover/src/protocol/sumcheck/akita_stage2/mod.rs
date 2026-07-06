//! Stage-2 fused sumcheck prover/verifier for the Akita PCS.
//!
//! This stage views the committed witness as a Boolean table
//! `w : {0,1}^{col_bits} x {0,1}^{ring_bits} -> F`, where `x` indexes the padded
//! witness columns and `y` indexes the coefficient inside a
//! `D = 2^{ring_bits}`-dimensional ring element. Let `a(y)` be the multilinear
//! extension of `alpha_evals_y = [1, alpha, ..., alpha^(D-1)]`, so on Boolean
//! inputs `a(y) = alpha^{bin(y)}`. Let `M_alpha` be the ring-switch matrix
//! after evaluating every ring entry at the transcript challenge `alpha`, and
//! define the `tau1`-weighted row combination
//!
//! `m_tau1(x) = sum_i eq(tau1, i) * M_alpha(i, x)`.
//!
//! The Boolean table stored in `m_evals_x` is exactly `x -> m_tau1(x)`.
//!
//! If
//!
//! `y_alpha = [0,`
//! `           v_0(alpha), ..., v_{N_D-1}(alpha),`
//! `           u_0(alpha), ..., u_{N_B-1}(alpha),`
//! `           0, ..., 0],`
//!
//! then the linear relation claim is
//!
//! `relation_claim = sum_i eq(tau1, i) * y_alpha[i]`
//! `               = sum_{x,y} w(x, y) * a(y) * m_tau1(x)`.
//!
//! There is no public-output `y_ring` row: the §3.1 fold-opening trace check is
//! internalized as the fused `gamma^2` term below rather than carried as an `M`
//! row, so `y_alpha` runs `consistency | A | B(u) | D(v)`.
//!
//! The fused trace term binds the committed fold witness to the public opening
//! through a fixed, public multilinear `TraceWeight(x, y)` (nonzero only on the
//! `e_hat` digit segment). Its input contribution is `gamma^2 * trace_target`,
//! where `trace_target` is the incoming opening claim (or the EOR final claim on
//! extension-opening-reduction paths). It reuses the stage-2 batching challenge
//! `gamma` (relation = `gamma^0`, range = `gamma^1`, trace = `gamma^2`), which
//! is sampled after the next-level witness is bound, so it adds no new
//! Fiat-Shamir challenge.
//!
//! Stage 1 supplies the carried virtual claim
//!
//! `s_claim = w(stage1_point) * (w(stage1_point) + 1)`
//! `        = sum_z eq(stage1_point, z) * w(z) * (w(z) + 1)`
//!
//! for the same multilinear witness table. With `gamma = batching_coeff`, the
//! exact identity established by this sumcheck is
//!
//! `gamma * s_claim + relation_claim + gamma^2 * trace_target =`
//! `sum_{x,y} [ gamma * eq(stage1_point, (x, y)) * w(x, y) * (w(x, y) + 1)`
//! `           + w(x, y) * a(y) * m_tau1(x)`
//! `           + gamma^2 * w(x, y) * TraceWeight(x, y) ]`.
//!
//! After all rounds, at `r_stage2 = (r_x, r_y)`, the verifier checks
//!
//! `gamma * eq(stage1_point, r_stage2) * w(r_stage2) * (w(r_stage2) + 1)`
//! `  + w(r_stage2) * a(r_y) * m_tau1(r_x)`
//! `  + gamma^2 * w(r_stage2) * TraceWeight(r_stage2)`,
//!
//! exactly the oracle returned by `expected_output_claim()`. The prover fuses
//! the virtual, relation, and trace terms around the same local `w0` / `dw`
//! scan so the witness-side work is shared between all three.

use super::fold_full_prefix_pair;
use super::round_batching::{
    build_stage2_initial_round_batch_grid, can_use_stage2_initial_round_batch,
    Stage2RoundBatchState,
};
use super::round_batching::{stage2_b4_w_digit, stage2_b8_w_digit};
use akita_algebra::poly::trim_trailing_zeros;
use akita_algebra::split_eq::GruenSplitEq;
use akita_field::parallel::*;
use akita_field::unreduced::{HasOptimizedFold, HasUnreducedOps};
use akita_field::{AkitaError, FieldCore, FromPrimitiveInt, Zero};
use akita_sumcheck::{
    fold_evals_in_place, reduce_signed_accum, CompactPairFoldLut, SumcheckInstanceProver, UniPoly,
};
use akita_types::RelationWeightPolynomial;
use std::mem;
use std::time::Instant;

enum WTable<E: FieldCore> {
    Compact(Vec<i8>),
    Full(Vec<E>),
}

struct Stage2InitialRoundBatch<E: FieldCore> {
    skip_state: Stage2RoundBatchState<E>,
    first_challenge: Option<E>,
}

#[derive(Clone, Copy)]
enum NormRoundTerms<E: FieldCore> {
    Full([E; 3]),
    SkipLinear([E; 2]),
}

type CompactVirtAccum<E> = [<E as HasUnreducedOps>::MulU64Accum; 4];
type CompactVirtSkipLinearAccum<E> = [<E as HasUnreducedOps>::MulU64Accum; 2];
type CompactRelAccum<E> = [<E as HasUnreducedOps>::MulU64Accum; 6];

#[inline]
fn coeffs_to_poly<E: FieldCore>(coeffs: [E; 3]) -> UniPoly<E> {
    let mut coeffs = vec![coeffs[0], coeffs[1], coeffs[2]];
    trim_trailing_zeros(&mut coeffs);
    UniPoly::from_coeffs(coeffs)
}

#[inline]
fn accum_small_signed<E: FieldCore + HasUnreducedOps>(
    accum: &mut [E::MulU64Accum],
    pos_idx: usize,
    coeff: E,
    signed: i64,
) {
    if signed == 0 {
        return;
    }
    let prod = coeff.mul_u64_unreduced(signed.unsigned_abs());
    if signed < 0 {
        accum[pos_idx + 1] += prod;
    } else {
        accum[pos_idx] += prod;
    }
}

#[inline]
fn reduce_compact_virt<E: FieldCore + HasUnreducedOps>(virt: CompactVirtAccum<E>) -> [E; 3] {
    [
        E::reduce_mul_u64_accum(virt[0]),
        reduce_signed_accum::<E>(virt[1], virt[2]),
        E::reduce_mul_u64_accum(virt[3]),
    ]
}

#[inline]
fn reduce_compact_virt_skip_linear<E: FieldCore + HasUnreducedOps>(
    virt: CompactVirtSkipLinearAccum<E>,
) -> [E; 2] {
    [
        E::reduce_mul_u64_accum(virt[0]),
        E::reduce_mul_u64_accum(virt[1]),
    ]
}

#[inline]
fn reduce_compact_rel<E: FieldCore + HasUnreducedOps>(rel: CompactRelAccum<E>) -> [E; 3] {
    [
        reduce_signed_accum::<E>(rel[0], rel[1]),
        reduce_signed_accum::<E>(rel[2], rel[3]),
        reduce_signed_accum::<E>(rel[4], rel[5]),
    ]
}

#[inline]
fn stage2_eq_block(
    j_base: usize,
    blk: usize,
    num_first: usize,
    first_bits: usize,
    block_size: usize,
    live_pairs: usize,
) -> (usize, usize) {
    debug_assert!(num_first.is_power_of_two());
    let j = j_base + blk;
    let j_high = j >> first_bits;
    let bucket_remaining = num_first - (j & (num_first - 1));
    let blk_end = (blk + block_size.min(bucket_remaining)).min(live_pairs);
    (j_high, blk_end)
}

#[inline]
pub(crate) fn accumulate_relation_coeffs<E: FieldCore>(
    rel: &mut [E; 3],
    w0: E,
    dw: E,
    p0: E,
    p1: E,
) {
    let dp = p1 - p0;
    rel[0] += w0 * p0;
    rel[1] += w0 * dp + dw * p0;
    rel[2] += dw * dp;
}

#[inline]
pub(crate) fn accumulate_relation_coeffs_signed<E: FieldCore + HasUnreducedOps>(
    rel: &mut [E::MulU64Accum; 6],
    w0: i64,
    dw: i64,
    p0: E,
    p1: E,
) {
    let dp = p1 - p0;
    accum_small_signed::<E>(rel, 0, p0, w0);
    accum_small_signed::<E>(rel, 2, dp, w0);
    accum_small_signed::<E>(rel, 2, p0, dw);
    accum_small_signed::<E>(rel, 4, dp, dw);
}

/// Stage-2 fused virtual-claim + relation sumcheck prover.
///
/// Holds a single `w_table` shared by both halves of stage 2. The virtual half
/// is pre-weighted by `batching_coeff` through `split_eq`, so the round
/// polynomial is:
/// `batching_coeff * virtual_round(t) + relation_round(t)`.
pub struct AkitaStage2Prover<E: FieldCore> {
    w_table: WTable<E>,
    b: usize,
    batching_coeff: E,
    s_claim: E,
    input_claim: E,
    split_eq: GruenSplitEq<E>,

    relation_weight: RelationWeightPolynomial<E>,
    live_x_cols: usize,
    col_bits: usize,
    num_vars: usize,
    prev_norm_claim: E,
    prev_norm_poly: Option<UniPoly<E>>,
    prefix_r_stage1: Option<Vec<E>>,
    initial_round_batch: Option<Stage2InitialRoundBatch<E>>,
    cached_round_poly: Option<UniPoly<E>>,

    scan_time_total: f64,
    fold_time_total: f64,
    rounds_completed: usize,
}

mod dense_terms;
mod lifecycle;
mod round2_prefix;
mod round_flow;
mod x_prefix;
mod y_prefix;

impl<E: FieldCore + FromPrimitiveInt + HasUnreducedOps> AkitaStage2Prover<E> {
    #[inline]
    pub(super) fn relation_weight_y_len(&self) -> usize {
        self.relation_weight.y_len()
    }

    #[inline]
    fn relation_weight_pair_columns(
        &self,
        x0: usize,
        x1: usize,
        y: usize,
    ) -> (E, E) {
        let y_len = self.relation_weight_y_len();
        let live_x_cols = self.relation_weight.live_x_cols();
        let evals = self.relation_weight.evals();
        let p0 = evals[x0 * y_len + y];
        let p1 = if x1 < live_x_cols {
            evals[x1 * y_len + y]
        } else {
            E::zero()
        };
        (p0, p1)
    }
}

impl<E: FieldCore + FromPrimitiveInt + HasUnreducedOps + HasOptimizedFold> AkitaStage2Prover<E> {
    pub(super) fn set_relation_weight_evals(
        &mut self,
        evals: Vec<E>,
        relation_x_cols: usize,
        y_len: usize,
    ) {
        let witness_len = relation_x_cols * y_len;
        self.relation_weight = RelationWeightPolynomial::from_evals(
            evals,
            y_len,
            relation_x_cols,
            witness_len,
        )
        .expect("relation weight fold preserves shape");
    }

    pub(super) fn fold_relation_weight_for_round(
        &mut self,
        r: E,
        folding_x_round: bool,
        use_prefix_x_round: bool,
        use_prefix_y_round: bool,
        in_y_round: bool,
    ) {
        let y_len = self.relation_weight_y_len();
        let relation_x_cols = self.relation_weight.live_x_cols();
        let (evals, next_relation_x_cols, next_y_len) = if folding_x_round {
            if use_prefix_x_round {
                (
                    Self::fold_relation_weight_x_column_major(
                        self.relation_weight.evals(),
                        relation_x_cols,
                        y_len,
                        r,
                    ),
                    relation_x_cols.div_ceil(2),
                    y_len,
                )
            } else {
                let mut evals = self.relation_weight.evals().to_vec();
                fold_evals_in_place(&mut evals, r);
                (evals, relation_x_cols.div_ceil(2), y_len)
            }
        } else if in_y_round && use_prefix_y_round {
            (
                Self::fold_relation_weight_prefix_y(
                    self.relation_weight.evals(),
                    relation_x_cols,
                    y_len,
                    r,
                ),
                relation_x_cols,
                y_len / 2,
            )
        } else {
            let mut evals = self.relation_weight.evals().to_vec();
            fold_evals_in_place(&mut evals, r);
            (evals, relation_x_cols, y_len / 2)
        };
        self.set_relation_weight_evals(evals, next_relation_x_cols, next_y_len);
    }
}

#[cfg(test)]
mod tests;
