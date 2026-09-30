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
//! `           for quotient relation rows only;`
//!
//! then the linear relation claim over quotient relation rows is
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
//! The response-norm binding term contributes `C_bind` to the combined linear
//! claim. Its weights are absorbed into the reduced relation table, prepared as
//! a rank-one linear source, or retained as sparse additional terms. These are
//! equivalent representations of the same binding term.
//!
//! Stage 1 supplies the carried virtual claim
//!
//! `range_image_evaluation`
//! `  = sum_z eq(stage1_point, z) * [w(z) * (w(z) + 1)]`
//!
//! for the multilinear extension of the pointwise Boolean range-image table. Away from
//! Boolean points this is not generally `w(stage1_point) * (w(stage1_point) + 1)`.
//! With `gamma = batching_coeff`, the
//! identity below shows the EvaluationTrace case before adding the response-norm
//! binding and other optional terms:
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

use akita_algebra::poly::trim_trailing_zeros;
use akita_algebra::split_eq::GruenSplitEq;
use akita_error::AkitaError;
use akita_sumcheck::{
    fold_evals_in_place, reduce_signed_accum, CompactPairFoldLut, SumcheckInstanceProver,
};
use jolt_field::solinas::parallel::*;
use jolt_field::{Field, Ring, Zero};
use jolt_field::{Fold, Unreduced};
use jolt_poly::UnivariatePoly;
use std::mem;

use crate::opaque::relation_weights::RelationWeightFactorization;
use crate::opaque::sumcheck::add_assign_all;
use crate::sources::packed_digits::{PackedSignedDigitView, PackedSignedDigits};

enum WitnessState<E: Field> {
    CompactPrefix(PackedSignedDigits),
    FoldedSuffix(Vec<E>),
}

enum CoefficientRelation<E: Field> {
    Factored(RelationWeightFactorization<E>),
    ReducedDense(DenseRelationWeights<E>),
}

/// State required to serve exactly one kind of Stage 2 round.
enum Phase<E: Field> {
    /// Rounds served by the compact quotient prefix engine.
    CompactPrefix {
        witness: PackedSignedDigits,
        weights: RelationWeightFactorization<E>,
        engine: Box<CompactQuotientPrefix<E>>,
    },
    /// Coefficient rounds outside the compact prefix.
    Coefficient {
        witness: WitnessState<E>,
        relation: CoefficientRelation<E>,
        relation_moments: Option<CoefficientRelationMoments<E>>,
    },
    /// Lane rounds on the folded witness.
    Lane {
        witness: Vec<E>,
        lane: LaneProduct<E>,
    },
}

#[derive(Clone, Copy)]
enum NormRoundTerms<E: Field> {
    Full([E; 3]),
    SkipLinear([E; 2]),
}

impl<E: Field> NormRoundTerms<E> {
    #[inline(always)]
    fn from_totals<const SKIP_LINEAR: bool>(totals: [E; 3]) -> Self {
        if SKIP_LINEAR {
            Self::SkipLinear([totals[0], totals[2]])
        } else {
            Self::Full(totals)
        }
    }
}

type CompactRelAccum<E> = [<E as Unreduced>::SmallProduct; 4];

/// Internal round state stores the message value at one and its top two
/// coefficients. The sumcheck claim recovers the constant, and the value at
/// one then recovers the linear coefficient at emission.
#[derive(Clone, Copy)]
struct RoundMessage<E: Field> {
    at_one: E,
    quadratic: E,
    cubic: E,
}

impl<E: Field> RoundMessage<E> {
    #[inline(always)]
    fn zero() -> Self {
        Self {
            at_one: E::zero(),
            quadratic: E::zero(),
            cubic: E::zero(),
        }
    }

    #[inline(always)]
    fn add_assign(&mut self, other: Self) {
        self.at_one += other.at_one;
        self.quadratic += other.quadratic;
        self.cubic += other.cubic;
    }

    #[inline]
    fn from_polynomial(polynomial: &UnivariatePoly<E>) -> Self {
        Self::from_coefficients(polynomial.coefficients())
    }

    #[inline]
    fn from_coefficients(coefficients: &[E]) -> Self {
        debug_assert!(coefficients.len() <= 4);
        Self {
            at_one: coefficients.iter().copied().sum(),
            quadratic: coefficients.get(2).copied().unwrap_or_else(E::zero),
            cubic: coefficients.get(3).copied().unwrap_or_else(E::zero),
        }
    }

    #[inline]
    fn into_polynomial(self, previous_claim: E) -> UnivariatePoly<E> {
        let constant = previous_claim - self.at_one;
        let linear = self.at_one - constant - self.quadratic - self.cubic;
        let mut coefficients = vec![constant, linear, self.quadratic, self.cubic];
        trim_trailing_zeros(&mut coefficients);
        UnivariatePoly::new(coefficients)
    }
}

#[inline]
fn coeffs_to_poly<E: Field>(coeffs: [E; 3]) -> UnivariatePoly<E> {
    let mut coeffs = vec![coeffs[0], coeffs[1], coeffs[2]];
    trim_trailing_zeros(&mut coeffs);
    UnivariatePoly::new(coeffs)
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
fn reduce_compact_rel<E: Field + Unreduced>(rel: CompactRelAccum<E>) -> RoundMessage<E> {
    RoundMessage {
        at_one: reduce_signed_accum::<E>(rel[0], rel[1]),
        quadratic: reduce_signed_accum::<E>(rel[2], rel[3]),
        cubic: E::zero(),
    }
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

/// Sum of field products reduced once.
///
/// Every field's product accumulator reduces a sum of fewer than `2^61`
/// widening products exactly (`Fp128` slots hold `2^64 - 1`), far more terms
/// than a round sums.
#[derive(Clone, Copy)]
struct ProductSum<E: Unreduced>(E::Product);

impl<E: Unreduced> ProductSum<E> {
    #[inline(always)]
    fn zero() -> Self {
        Self(E::Product::zero())
    }

    #[inline(always)]
    fn add(&mut self, left: E, right: E) {
        self.0 += left.mul_unreduced(right);
    }

    #[inline(always)]
    fn finish(self) -> E {
        E::reduce_product(self.0)
    }
}

/// Relation message of a run of pairs `w(X) * q(X)` with `w = w0 + X dw` and
/// `q = q0 + X (q1 - q0)`, where `q` is the relation weight plus the
/// structured linear term.
///
/// Only the value at one, `sum w1 q1`, and the quadratic coefficient,
/// `sum dw (q1 - q0)`, are accumulated. The sumcheck claim recovers the
/// constant and the linear coefficient. Every field-valued relation kernel
/// (dense, compact and folded coefficient, lane rounds) accumulates through
/// this type and reduces it once per its own task or lane. The signed-digit
/// kernels keep [`accumulate_relation_eval_coeffs_signed`], whose accumulator
/// is a pair of unsigned small products per coefficient.
#[derive(Clone, Copy)]
struct RelationPairAccumulator<E: Unreduced>([ProductSum<E>; 2]);

impl<E: Unreduced> RelationPairAccumulator<E> {
    #[inline(always)]
    fn zero() -> Self {
        Self([ProductSum::zero(); 2])
    }

    /// Adds one pair. `dw = w1 - w0` is passed in because the norm
    /// accumulator of the same scan needs it too.
    #[inline(always)]
    fn add_pair(&mut self, w1: E, dw: E, q0: E, q1: E) {
        self.0[0].add(w1, q1);
        self.0[1].add(dw, q1 - q0);
    }

    #[inline(always)]
    fn finish(self) -> RoundMessage<E>
    where
        E: Field,
    {
        RoundMessage {
            at_one: self.0[0].finish(),
            quadratic: self.0[1].finish(),
            cubic: E::zero(),
        }
    }
}

fn add_round_terms<E: Field>(
    left: &mut ([E; 3], RoundMessage<E>),
    right: ([E; 3], RoundMessage<E>),
) {
    add_assign_all(&mut left.0, &right.0);
    left.1.add_assign(right.1);
}

#[inline]
pub(crate) fn accumulate_relation_eval_coeffs_signed<E: Field + Unreduced>(
    rel: &mut [E::SmallProduct; 4],
    w0: i64,
    dw: i64,
    p0: E,
    p1: E,
) {
    let dp = p1 - p0;
    accum_small_signed::<E>(rel, 0, p1, w0 + dw);
    accum_small_signed::<E>(rel, 2, dp, dw);
}

/// Fused relation, structured-linear, and range-image sumcheck prover.
///
/// Holds one witness state shared by the range-image, relation, and structured-linear
/// terms. The compact prefix is materialized once into the folded field suffix.
/// The range-image term is pre-weighted by `batching_coeff` through `split_eq`, so
/// the round polynomial is:
/// `batching_coeff * virtual_round(t) + relation_round(t)`.
pub(crate) struct RelationRangeImageProver<E: Field> {
    phase: Option<Phase<E>>,
    input_claim: E,
    split_eq: GruenSplitEq<E>,

    additional_relation_terms: Option<AdditionalRelationTerms<E>>,
    linear_terms: PreparedProverLinearTerms<E>,
    live_lane_count: usize,
    lane_bits: usize,
    num_vars: usize,
    prev_norm_claim: E,
    prev_norm_poly: Option<UnivariatePoly<E>>,
    cached_round_message: Option<RoundMessage<E>>,

    rounds_completed: usize,
}

mod additional_terms;
mod coefficient_packing_terms;
mod coefficient_prefix;
mod coefficient_round_fold;
mod dense_terms;
mod evaluation_trace;
mod lane_product;
mod lifecycle;
mod norm;
use norm::{CompactNorm, FieldNorm, ProductNorm};
mod prefix_cache;
mod prepared_linear_lane;
mod quotient_prefix;
mod round_flow;
mod weight_oracle;
mod wide_mass;

pub(crate) use additional_terms::AdditionalRelationTerms;
pub(crate) use coefficient_packing_terms::prepare_coefficient_packing_linear_terms;
#[allow(unused_imports)]
pub(crate) use evaluation_trace::{build_evaluation_trace_weights, PreparedProverLinearTerms};
#[cfg(test)]
pub(crate) use evaluation_trace::{
    StructuredLinearSegment, StructuredLinearTerm, StructuredLinearWeights,
};
use lane_product::LaneProduct;
pub(crate) use prepared_linear_lane::PreparedLinearLane;
use quotient_prefix::{CoefficientRelationMoments, CompactQuotientPrefix};
pub(crate) use weight_oracle::{DenseRelationWeights, RelationWeightOracle};

impl<E: Field + Ring + Unreduced> RelationRangeImageProver<E> {
    #[cfg(test)]
    #[inline]
    fn quotient_weights(&self) -> Option<&RelationWeightFactorization<E>> {
        match self.phase.as_ref()? {
            Phase::CompactPrefix { weights, .. } => Some(weights),
            Phase::Coefficient {
                relation: CoefficientRelation::Factored(weights),
                ..
            } => Some(weights),
            Phase::Coefficient {
                relation: CoefficientRelation::ReducedDense(_),
                ..
            }
            | Phase::Lane { .. } => None,
        }
    }

    // Fused relation (`alpha * m`) + structured-linear addend for one witness
    // corner. `witness_idx0` is the first flat index of an adjacent pair in
    // the Boolean `w` table (`lane * coeff_count + coefficient`).

    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn accumulate_fused_relation_linear_signed(
        &self,
        rel: &mut [E::SmallProduct; 4],
        w0: i64,
        dw: i64,
        witness_idx0: usize,
        p0: E,
        p1: E,
    ) {
        let (t0, t1) = self.linear_terms.pair_from_flat_index(witness_idx0);
        accumulate_relation_eval_coeffs_signed(rel, w0, dw, p0 + t0, p1 + t1);
    }
}

#[cfg(test)]
mod tests;
