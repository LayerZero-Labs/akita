//! Stage-2 fused sumcheck prover/verifier for the Akita PCS.
//!
//! This stage views the committed witness as one flat LSB-first Boolean table.
//! The current state machine splits its point after
//! `log2(common_relation_witness_coeff_count)` low coordinates. Those coordinates
//! index the largest coefficient block aligned for both every relation role and
//! the outgoing witness ring representation; the remaining coordinates index
//! relation lanes and padded witness capacity. Kernel names use only coefficient
//! and lane geometry.
//!
//! Let `common_alpha` be the multilinear extension of
//! `[1, alpha, ..., alpha^(common_relation_witness_coeff_count - 1)]`. Let the
//! relation matrix be evaluated at the transcript challenge `alpha`, and define
//! its `tau1`-weighted relation-lane combination
//!
//! `relation_lane_weight(lane) = sum_i eq(tau1, i) * M_alpha(i, lane)`.
//!
//! The table stored in `relation_lane_weights` is exactly this lane weight.
//!
//! If
//!
//! `y_alpha = [0,`
//! `           u_0(alpha), ..., u_{N_B-1}(alpha),`
//! `           v_0(alpha), ..., v_{N_D-1}(alpha)]`
//! `           for physical quotient rows only;`
//!
//! then the linear relation claim over physical quotient rows is
//!
//! `relation_claim = sum_i eq(tau1, i) * y_alpha[i]`
//! `               = sum_address digit_witness(address)`
//! `                   * common_alpha(coeff_within_common_block(address))`
//! `                   * relation_lane_weight(relation_lane(address))`.
//!
//! There is no public-output `y_ring` row: the fold-opening trace check is
//! internalized as the `EvaluationTrace` relation row (last padded logical row),
//! weighted by `eq(tau1, EvaluationTrace_row_index)`. Physical M rows are
//! `consistency | A | B(u) | D(v)`; EvaluationTrace is absent from physical M.
//! `y_alpha` runs `FoldEvaluation | A | B(u) | D(v)` for quotient rows; the
//! opening target enters the Stage-2 claim through EvaluationTrace.
//!
//! The fused EvaluationTrace term binds the committed fold witness to the public
//! opening through a fixed, public multilinear `TraceWeight(address)` (nonzero only
//! on the `e_hat` digit segment). Its input contribution is
//! `eq(tau1, EvaluationTrace_row_index) * trace_target`, where `trace_target` is
//! the incoming opening claim (or the EOR final claim on extension-opening-reduction
//! paths). It reuses the existing row-index challenge (`tau1`) and adds no extra
//! Fiat-Shamir challenge at terminal folds (`batching_coeff = 0` there).
//!
//! Stage 1 supplies the carried virtual claim
//!
//! `range_image_evaluation`
//! `  = sum_z eq(stage1_point, z) * [w(z) * (w(z) + 1)]`
//!
//! for the multilinear extension of the pointwise Boolean range-image table. Away from
//! Boolean points this is not generally `w(stage1_point) * (w(stage1_point) + 1)`.
//! With `gamma = batching_coeff`, the
//! exact identity established by this sumcheck is
//!
//! `gamma * range_image_evaluation + relation_claim + eq(tau1, EvaluationTrace_row_index) * trace_target =`
//! `sum_address [ gamma * eq(stage1_point, address)`
//! `                  * digit_witness(address) * (digit_witness(address) + 1)`
//! `           + digit_witness(address)`
//! `               * common_alpha(coeff_within_common_block(address))`
//! `               * relation_lane_weight(relation_lane(address))`
//! `           + eq(tau1, EvaluationTrace_row_index)`
//! `               * digit_witness(address) * TraceWeight(address) ]`.
//!
//! After all rounds, at the complete flat point `r_stage2`, the verifier checks
//!
//! `gamma * eq(stage1_point, r_stage2) * w(r_stage2) * (w(r_stage2) + 1)`
//! `  + w(r_stage2) * common_alpha(common_point)`
//! `      * relation_lane_weight(lane_point)`
//! `  + eq(tau1, EvaluationTrace_row_index) * w(r_stage2) * TraceWeight(r_stage2)`,
//!
//! exactly the oracle returned by `expected_output_claim()`. The prover fuses
//! the virtual, relation, and EvaluationTrace terms around the same local `w0` /
//! `dw` scan so the witness-side work is shared.

use super::fold_full_prefix_pair as fold_folded_lane_pair;
use super::two_round_prefix::{
    build_stage2_bivariate_skip_proof_from_m_compact, can_use_stage2_two_round_prefix,
    Stage2BivariateSkipState,
};
use super::two_round_prefix::{stage2_b4_w_digit, stage2_b8_w_digit};
use akita_algebra::poly::trim_trailing_zeros;
use akita_algebra::split_eq::GruenSplitEq;
use akita_field::parallel::*;
use akita_field::unreduced::{HasOptimizedFold, HasUnreducedOps};
use akita_field::{AkitaError, FieldCore, FromPrimitiveInt, Zero};
use akita_sumcheck::{
    fold_evals_in_place, reduce_signed_accum, CompactPairFoldLut, SumcheckInstanceProver, UniPoly,
};
use std::mem;
use std::time::Instant;

enum WitnessState<E: FieldCore> {
    CompactPrefix(std::sync::Arc<[i8]>),
    FoldedSuffix(Vec<E>),
}

struct TwoRoundCompactPrefix<E: FieldCore> {
    skip_state: Stage2BivariateSkipState<E>,
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
/// Holds one witness state shared by the range-image, relation, and evaluation-trace
/// terms. The compact prefix is materialized once into the folded field suffix.
/// The range-image term is pre-weighted by `batching_coeff` through `split_eq`, so
/// the round polynomial is:
/// `batching_coeff * virtual_round(t) + relation_round(t)`.
pub struct AkitaStage2Prover<E: FieldCore> {
    witness_state: WitnessState<E>,
    b: usize,
    batching_coeff: E,
    range_image_evaluation: E,
    input_claim: E,
    split_eq: GruenSplitEq<E>,

    common_alpha_factor: Vec<E>,
    relation_lane_weights: Vec<E>,
    evaluation_trace: PreparedProverEvaluationTrace<E>,
    live_lane_count: usize,
    lane_bits: usize,
    num_vars: usize,
    relation_trace_claim: E,
    prev_norm_claim: E,
    prev_norm_poly: Option<UniPoly<E>>,
    compact_prefix_stage1_point: Option<Vec<E>>,
    deferred_compact_prefix: Option<TwoRoundCompactPrefix<E>>,
    cached_round_poly: Option<UniPoly<E>>,

    scan_time_total: f64,
    fold_time_total: f64,
    rounds_completed: usize,
}

mod coefficient_prefix;
mod compact_prefix;
mod dense_terms;
mod evaluation_trace;
mod lane_prefix;
mod lifecycle;
mod round_flow;

pub(crate) use evaluation_trace::{build_evaluation_trace_weights, PreparedProverEvaluationTrace};

impl<E: FieldCore + FromPrimitiveInt + HasUnreducedOps> AkitaStage2Prover<E> {
    // Fused relation (`alpha * m`) + trace-weight addend for one witness
    // corner. `witness_idx0/1` are flat indices into the Boolean `w` table
    // (`lane * coeff_count + coefficient`). Coefficient-round kernels
    // pass `2*j` and `2*j+1`; lane-prefix fusion passes lane-relative indices
    // directly.

    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn accumulate_fused_relation_trace(
        &self,
        rel: &mut [E; 3],
        w0: E,
        dw: E,
        witness_idx0: usize,
        witness_idx1: usize,
        p0: E,
        p1: E,
    ) {
        accumulate_relation_coeffs(rel, w0, dw, p0, p1);
        let coeff_count = self.common_alpha_factor.len();
        let (t0, t1) = self
            .evaluation_trace
            .pair_flat(witness_idx0, witness_idx1, coeff_count);
        accumulate_relation_coeffs(rel, w0, dw, t0, t1);
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn accumulate_fused_relation_trace_signed(
        &self,
        rel: &mut [E::MulU64Accum; 6],
        w0: i64,
        dw: i64,
        witness_idx0: usize,
        witness_idx1: usize,
        p0: E,
        p1: E,
    ) {
        accumulate_relation_coeffs_signed(rel, w0, dw, p0, p1);
        let coeff_count = self.common_alpha_factor.len();
        let (t0, t1) = self
            .evaluation_trace
            .pair_flat(witness_idx0, witness_idx1, coeff_count);
        accumulate_relation_coeffs_signed(rel, w0, dw, t0, t1);
    }

    #[inline]
    pub(super) fn fold_evaluation_trace_for_current_round(&mut self, challenge: E) {
        if self.in_coefficient_round() {
            self.evaluation_trace.fold_coefficients(challenge);
        } else {
            self.evaluation_trace.fold_lanes(challenge);
        }
    }
}

#[cfg(test)]
mod tests;
