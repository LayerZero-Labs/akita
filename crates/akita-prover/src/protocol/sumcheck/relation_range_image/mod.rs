//! Fused relation, structured-opening, and range-image sumcheck prover.
//!
//! This sumcheck views the committed witness as one flat LSB-first Boolean table.
//! The current state machine splits its point after
//! `log2(relation_coefficient_block_len)` low coordinates. Those coordinates
//! index the largest coefficient block shared by every current relation role;
//! the remaining coordinates index relation lanes and padded flat witness
//! capacity. Outgoing ring packaging determines that flat capacity, but not the
//! split. Kernel names use only coefficient and lane geometry.
//!
//! Let `common_alpha` be the multilinear extension of
//! `[1, alpha, ..., alpha^(relation_coefficient_block_len - 1)]`. Let the
//! relation matrix be evaluated at the transcript challenge `alpha`, and define
//! its `tau1`-weighted relation-lane combination
//!
//! `relation_lane_weight(lane) = sum_i eq(tau1, i) * M_alpha(i, lane)`.
//!
//! The table stored in `relation_lane_weights` is exactly this lane weight.
//! This is the quotient-lift oracle. Reduced evaluation instead compiles the
//! complete ordinary-plus-compression relation weight `p(address)` over the
//! padded flat domain. The sumcheck engine folds either representation through
//! `RelationWeightOracle` while the witness and structured-linear terms keep
//! the same coefficient/lane geometry.
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
//! `               = sum_address digit_witness(address) * p(address)`.
//!
//! There is no public-output `y_ring` row: the fold-opening trace check is
//! internalized as the `EvaluationTrace` relation row (last padded logical row),
//! weighted by `eq(tau1, EvaluationTrace_row_index)`. Physical M rows are
//! `consistency | A | B(u) | D(v)`; EvaluationTrace is absent from physical M.
//! `y_alpha` runs `FoldEvaluation | A | B(u) | D(v)` for quotient rows; the
//! opening target enters the Stage-2 claim through EvaluationTrace.
//!
//! The structured linear term engine binds the committed fold witness to the public
//! opening through fixed public multilinear weights. EvaluationTrace contributes one
//! such weight on the `e_hat` digit segment. Coefficient packing contributes its direct
//! scalar-opening weight plus a separate packing-consistency weight on `z_hat`. The
//! EvaluationTrace input contribution is
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
//! `           + digit_witness(address) * p(address)`
//! `           + eq(tau1, EvaluationTrace_row_index)`
//! `               * digit_witness(address) * TraceWeight(address) ]`.
//!
//! After all rounds, at the complete flat point `r_stage2`, the verifier checks
//!
//! `gamma * eq(stage1_point, r_stage2) * w(r_stage2) * (w(r_stage2) + 1)`
//! `  + w(r_stage2) * p(r_stage2)`
//! `  + eq(tau1, EvaluationTrace_row_index) * w(r_stage2) * TraceWeight(r_stage2)`,
//!
//! exactly the oracle returned by `expected_output_claim()`. The prover fuses
//! the virtual, relation, and EvaluationTrace terms around the same local `w0` /
//! `dw` scan so the witness-side work is shared.

use super::fold_prefix_pair_with_zero_padding as fold_folded_lane_pair;
use super::two_round_prefix::{
    build_stage2_bivariate_skip_proof_from_m_compact, can_use_stage2_two_round_prefix,
    Stage2BivariateSkipState,
};
use super::two_round_prefix::{
    stage2_b4_lookup_index_from_digits, stage2_b4_w_digit, stage2_b8_lookup_index_from_digits,
    stage2_b8_w_digit,
};
use akita_algebra::poly::trim_trailing_zeros;
use akita_algebra::split_eq::GruenSplitEq;
use akita_error::AkitaError;
use akita_sumcheck::{
    fold_evals_in_place, reduce_signed_accum, CompactPairFoldLut, SumcheckInstanceProver, UniPoly,
};
use jolt_field::solinas::parallel::*;
use jolt_field::{Field, Ring, Zero};
use jolt_field::{Fold, Unreduced};
use std::mem;
use std::time::Instant;

use crate::backend::packed_digits::{
    PackedSignedDigitIter, PackedSignedDigitView, PackedSignedDigits,
};
use crate::protocol::ring_switch::RelationWeightFactorization;

enum WitnessState<E: Field> {
    CompactPrefix(PackedSignedDigits),
    FoldedSuffix(Vec<E>),
}

struct DeferredCompactPrefix<E: Field> {
    skip_state: Stage2BivariateSkipState<E>,
    phase: DeferredCompactPrefixPhase<E>,
}

enum DeferredCompactPrefixPhase<E: Field> {
    Round0,
    Round1 { first_challenge: E },
}

enum QuotientPrefixState<E: Field> {
    Disabled,
    Deferred(DeferredCompactPrefix<E>),
}

enum RelationRoundState<E: Field> {
    QuotientFactored {
        weights: RelationWeightFactorization<E>,
        prefix: QuotientPrefixState<E>,
    },
    ReducedDense {
        weights: DenseRelationWeights<E>,
    },
}

#[derive(Clone, Copy)]
enum NormRoundTerms<E: Field> {
    Full([E; 3]),
    SkipLinear([E; 2]),
}

type CompactVirtAccum<E> = [<E as Unreduced>::SmallProduct; 4];
type CompactVirtSkipLinearAccum<E> = [<E as Unreduced>::SmallProduct; 2];
type CompactRelAccum<E> = [<E as Unreduced>::SmallProduct; 6];

#[inline]
fn coeffs_to_poly<E: Field>(coeffs: [E; 3]) -> UniPoly<E> {
    let mut coeffs = vec![coeffs[0], coeffs[1], coeffs[2]];
    trim_trailing_zeros(&mut coeffs);
    UniPoly::from_coeffs(coeffs)
}

#[inline]
fn fold_two_round_quad<E: Field>(v00: E, v10: E, v01: E, v11: E, r0: E, r1: E) -> E {
    let x0 = v00 + r0 * (v10 - v00);
    let x1 = v01 + r0 * (v11 - v01);
    x0 + r1 * (x1 - x0)
}

#[inline]
fn accum_small_signed<E: Field + Unreduced>(
    accum: &mut [E::SmallProduct],
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
fn reduce_compact_virt<E: Field + Unreduced>(virt: CompactVirtAccum<E>) -> [E; 3] {
    [
        E::reduce_small_product(virt[0]),
        reduce_signed_accum::<E>(virt[1], virt[2]),
        E::reduce_small_product(virt[3]),
    ]
}

#[inline]
fn reduce_compact_virt_skip_linear<E: Field + Unreduced>(
    virt: CompactVirtSkipLinearAccum<E>,
) -> [E; 2] {
    [
        E::reduce_small_product(virt[0]),
        E::reduce_small_product(virt[1]),
    ]
}

#[inline]
fn reduce_compact_rel<E: Field + Unreduced>(rel: CompactRelAccum<E>) -> [E; 3] {
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
pub(crate) fn accumulate_relation_coeffs<E: Field>(rel: &mut [E; 3], w0: E, dw: E, p0: E, p1: E) {
    let dp = p1 - p0;
    rel[0] += w0 * p0;
    rel[1] += w0 * dp + dw * p0;
    rel[2] += dw * dp;
}

#[inline]
pub(crate) fn accumulate_relation_coeffs_signed<E: Field + Unreduced>(
    rel: &mut [E::SmallProduct; 6],
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

/// Fused relation, structured-linear, and range-image sumcheck prover.
///
/// Holds one witness state shared by the range-image, relation, and structured-linear
/// terms. The compact prefix is materialized once into the folded field suffix.
/// The range-image term is pre-weighted by `batching_coeff` through `split_eq`, so
/// the round polynomial is:
/// `batching_coeff * virtual_round(t) + relation_round(t)`.
pub struct RelationRangeImageProver<E: Field> {
    witness_state: WitnessState<E>,
    b: usize,
    input_claim: E,
    split_eq: GruenSplitEq<E>,

    relation_state: RelationRoundState<E>,
    additional_relation_terms: Option<AdditionalRelationTerms<E>>,
    linear_terms: PreparedProverLinearTerms<E>,
    live_lane_count: usize,
    lane_bits: usize,
    num_vars: usize,
    prev_norm_claim: E,
    prev_norm_poly: Option<UniPoly<E>>,
    cached_round_poly: Option<UniPoly<E>>,

    scan_time_total: f64,
    fold_time_total: f64,
    rounds_completed: usize,
}

mod additional_terms;
mod coefficient_packing_terms;
mod coefficient_prefix;
mod coefficient_round_fold;
mod compact_prefix;
mod dense_terms;
mod evaluation_trace;
mod lane_prefix;
mod lifecycle;
mod prepared_linear_lane;
mod round_flow;
mod weight_oracle;

pub(crate) use additional_terms::AdditionalRelationTerms;
pub(in crate::protocol) use coefficient_packing_terms::prepare_coefficient_packing_linear_terms;
pub(crate) use evaluation_trace::{build_evaluation_trace_weights, PreparedProverLinearTerms};
#[cfg(test)]
pub(crate) use evaluation_trace::{
    StructuredLinearSegment, StructuredLinearTerm, StructuredLinearWeights,
};
pub(crate) use prepared_linear_lane::PreparedLinearLane;
pub(crate) use weight_oracle::{DenseRelationWeights, RelationWeightOracle};

impl<E: Field + Ring + Unreduced> RelationRangeImageProver<E> {
    #[cfg(test)]
    #[inline]
    fn common_alpha_factor(&self) -> Option<&[E]> {
        self.quotient_weights()
            .map(RelationWeightFactorization::common_alpha_factor)
    }

    #[cfg(test)]
    #[inline]
    fn relation_lane_weights(&self) -> Option<&[E]> {
        self.quotient_weights()
            .map(RelationWeightFactorization::relation_lane_weights)
    }

    #[cfg(test)]
    #[inline]
    fn quotient_weights(&self) -> Option<&RelationWeightFactorization<E>> {
        match &self.relation_state {
            RelationRoundState::QuotientFactored { weights, .. } => Some(weights),
            RelationRoundState::ReducedDense { .. } => None,
        }
    }

    #[cfg(test)]
    fn replace_common_alpha_factor(&mut self, replacement: Vec<E>) {
        if let RelationRoundState::QuotientFactored { weights, .. } = &mut self.relation_state {
            *weights.components_mut().0 = replacement;
        }
    }

    #[cfg(test)]
    fn replace_relation_lane_weights(&mut self, replacement: Vec<E>) {
        if let RelationRoundState::QuotientFactored { weights, .. } = &mut self.relation_state {
            *weights.components_mut().1 = replacement;
        }
    }

    // Fused relation (`alpha * m`) + structured-linear addend for one witness
    // corner. `witness_idx0` is the first flat index of an adjacent pair in
    // the Boolean `w` table (`lane * coeff_count + coefficient`).

    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn accumulate_fused_relation_linear(
        &self,
        rel: &mut [E; 3],
        w0: E,
        dw: E,
        witness_idx0: usize,
        p0: E,
        p1: E,
    ) {
        let (t0, t1) = self.linear_terms.pair_from_flat_index(witness_idx0);
        accumulate_relation_coeffs(rel, w0, dw, p0 + t0, p1 + t1);
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn accumulate_fused_relation_linear_signed(
        &self,
        rel: &mut [E::SmallProduct; 6],
        w0: i64,
        dw: i64,
        witness_idx0: usize,
        p0: E,
        p1: E,
    ) {
        let (t0, t1) = self.linear_terms.pair_from_flat_index(witness_idx0);
        accumulate_relation_coeffs_signed(rel, w0, dw, p0 + t0, p1 + t1);
    }

    #[inline]
    pub(super) fn fold_linear_terms_for_current_round(&mut self, challenge: E) {
        if self.in_coefficient_round() {
            self.linear_terms.fold_coefficients(challenge);
        } else {
            self.linear_terms.fold_lanes(challenge);
        }
    }
}

#[cfg(test)]
mod tests;
