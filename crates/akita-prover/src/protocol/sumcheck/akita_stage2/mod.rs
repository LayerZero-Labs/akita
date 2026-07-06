//! Stage-2 fused sumcheck prover/verifier for the Akita PCS.
//!
//! This stage views the committed witness as a live flat Boolean table
//! `w : {0,1}^{num_vars} -> F`, zero-extended outside the public live range.
//!
//! The relation side is a single multilinear
//! [`RelationWeightPolynomial`] over the
//! witness hypercube. Its evaluations are the field-level, `tau1`-batched
//! relation weights: every row of the ring-switched matrix, including the
//! [`EvaluationTrace`](akita_types::RelationRowFamily::EvaluationTrace) row
//! that internalizes the fold-opening trace check (no separate `gamma^2`
//! summand and no on-wire `y_ring`).
//!
//! Stage 1 supplies the carried virtual claim
//!
//! `s_claim = w(stage1_point) * (w(stage1_point) + 1)`
//! `        = sum_z eq(stage1_point, z) * w(z) * (w(z) + 1)`.
//!
//! With `gamma = batching_coeff`, the exact identity established by this
//! sumcheck is
//!
//! `relation_weight_claim + gamma * s_claim =`
//! `sum_z [ w(z) * RelationWeightPolynomial(z)`
//! `      + gamma * eq(stage1_point, z) * w(z) * (w(z) + 1) ]`.
//!
//! After all rounds, at `r_stage2`, the verifier checks
//!
//! `w(r_stage2) * RelationWeightPolynomial(r_stage2)`
//! `  + gamma * eq(stage1_point, r_stage2) * w(r_stage2) * (w(r_stage2) + 1)`,
//!
//! exactly the oracle returned by `expected_output_claim()`. The prover fuses
//! the relation-weight and virtual terms around the same flat pair scan so the
//! witness-side work is shared.

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

enum WitnessTable<E: FieldCore> {
    Compact(Vec<i8>),
    Full(Vec<E>),
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Stage2Layout {
    live_len: usize,
    num_vars: usize,
    uniform_tiling: Option<UniformStage2Tiling>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct UniformStage2Tiling {
    live_tiles: usize,
    tile_bits: usize,
    lane_bits: usize,
    lane_len: usize,
}

impl Stage2Layout {
    pub(crate) fn flat(live_len: usize, num_vars: usize) -> Result<Self, AkitaError> {
        if live_len == 0 {
            return Err(AkitaError::InvalidInput(
                "stage-2 live length must be at least 1".to_string(),
            ));
        }
        let domain_len = boolean_domain_len(num_vars)?;
        if live_len > domain_len {
            return Err(AkitaError::InvalidSize {
                expected: domain_len,
                actual: live_len,
            });
        }
        Ok(Self {
            live_len,
            num_vars,
            uniform_tiling: None,
        })
    }

    pub(crate) fn uniform(
        live_tiles: usize,
        tile_bits: usize,
        lane_bits: usize,
    ) -> Result<Self, AkitaError> {
        if live_tiles == 0 {
            return Err(AkitaError::InvalidInput(
                "stage-2 live tile count must be at least 1".to_string(),
            ));
        }
        let tile_capacity = boolean_domain_len(tile_bits)?;
        if live_tiles > tile_capacity {
            return Err(AkitaError::InvalidSize {
                expected: tile_capacity,
                actual: live_tiles,
            });
        }
        let lane_len = boolean_domain_len(lane_bits)?;
        let live_len = live_tiles
            .checked_mul(lane_len)
            .ok_or_else(|| AkitaError::InvalidInput("stage-2 live length overflow".to_string()))?;
        let num_vars = tile_bits.checked_add(lane_bits).ok_or_else(|| {
            AkitaError::InvalidInput("stage-2 challenge width overflow".to_string())
        })?;
        let layout = Self::flat(live_len, num_vars)?;
        Ok(Self {
            live_len: layout.live_len,
            num_vars: layout.num_vars,
            uniform_tiling: Some(UniformStage2Tiling {
                live_tiles,
                tile_bits,
                lane_bits,
                lane_len,
            }),
        })
    }

    #[inline]
    pub(crate) fn live_len(&self) -> usize {
        self.live_len
    }

    #[inline]
    pub(crate) fn num_vars(&self) -> usize {
        self.num_vars
    }

    #[inline]
    pub(crate) fn uniform_tiling(&self) -> Option<UniformStage2Tiling> {
        self.uniform_tiling
    }
}

impl UniformStage2Tiling {
    #[inline]
    fn coeff_bits(&self) -> usize {
        self.lane_bits
    }

    #[inline]
    fn coeff_len(&self) -> usize {
        self.lane_len
    }
}

fn boolean_domain_len(num_vars: usize) -> Result<usize, AkitaError> {
    let bits = u32::try_from(num_vars)
        .map_err(|_| AkitaError::InvalidInput("stage-2 width overflow".to_string()))?;
    1usize
        .checked_shl(bits)
        .ok_or_else(|| AkitaError::InvalidInput("stage-2 width overflow".to_string()))
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

fn fold_live_evals_zero_padded<E: HasOptimizedFold>(evals: &[E], r: E) -> Vec<E> {
    let ctx = E::precompute_fold(r);
    cfg_into_iter!(0..evals.len().div_ceil(2))
        .map(|i| {
            let left = 2 * i;
            let a = evals[left];
            let b = evals.get(left + 1).copied().unwrap_or_else(E::zero);
            E::fold_one(&ctx, a, b)
        })
        .collect()
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
/// Holds a single `witness_table` shared by both halves of stage 2. The virtual half
/// is pre-weighted by `batching_coeff` through `split_eq`, so the round
/// polynomial is:
/// `batching_coeff * virtual_round(t) + relation_round(t)`.
pub struct AkitaStage2Prover<E: FieldCore> {
    witness_table: WitnessTable<E>,
    b: usize,
    batching_coeff: E,
    s_claim: E,
    input_claim: E,
    split_eq: GruenSplitEq<E>,

    relation_weight: RelationWeightPolynomial<E>,
    layout: Stage2Layout,
    live_segments: usize,
    relation_coeff_len: usize,
    segment_bits: usize,
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

mod coefficient_prefix;
mod lifecycle;
mod pair_scan;
mod round2_prefix;
mod round_flow;
mod segment_prefix;

impl<E: FieldCore + FromPrimitiveInt + HasUnreducedOps> AkitaStage2Prover<E> {
    #[inline]
    pub(super) fn relation_weight_coeff_len(&self) -> usize {
        self.relation_coeff_len
    }

    #[inline]
    fn relation_weight_pair_tiles(&self, tile0: usize, tile1: usize, lane: usize) -> (E, E) {
        let coeff_len = self.relation_weight_coeff_len();
        let evals = self.relation_weight.evals();
        let p0 = evals
            .get(tile0 * coeff_len + lane)
            .copied()
            .unwrap_or_else(E::zero);
        let p1 = evals
            .get(tile1 * coeff_len + lane)
            .copied()
            .unwrap_or_else(E::zero);
        (p0, p1)
    }
}

impl<E: FieldCore + FromPrimitiveInt + HasUnreducedOps + HasOptimizedFold> AkitaStage2Prover<E> {
    pub(super) fn set_relation_weight_evals(
        &mut self,
        evals: Vec<E>,
        relation_segments: usize,
        coeff_len: usize,
    ) {
        let live_len = relation_segments * coeff_len;
        self.relation_weight = RelationWeightPolynomial::from_live_evals(evals, live_len)
            .expect("relation weight fold preserves shape");
        self.relation_coeff_len = coeff_len;
    }

    pub(super) fn fold_relation_weight_for_round(
        &mut self,
        r: E,
        folding_segment_round: bool,
        use_segment_prefix_round: bool,
        use_coefficient_prefix_round: bool,
        in_coefficient_round: bool,
    ) {
        let coeff_len = self.relation_weight_coeff_len();
        let relation_segments = self.live_segments;
        let (evals, next_relation_segments, next_coeff_len) = if folding_segment_round {
            if use_segment_prefix_round {
                (
                    Self::fold_relation_weight_segment_major(
                        self.relation_weight.evals(),
                        relation_segments,
                        coeff_len,
                        r,
                    ),
                    relation_segments.div_ceil(2),
                    coeff_len,
                )
            } else {
                let evals = if self.layout.uniform_tiling().is_some() {
                    let mut evals = self.relation_weight.evals().to_vec();
                    fold_evals_in_place(&mut evals, r);
                    evals
                } else {
                    fold_live_evals_zero_padded(self.relation_weight.evals(), r)
                };
                (evals, relation_segments.div_ceil(2), coeff_len)
            }
        } else if in_coefficient_round && use_coefficient_prefix_round {
            (
                Self::fold_relation_weight_coefficient_prefix(
                    self.relation_weight.evals(),
                    relation_segments,
                    coeff_len,
                    r,
                ),
                relation_segments,
                coeff_len / 2,
            )
        } else {
            let mut evals = self.relation_weight.evals().to_vec();
            fold_evals_in_place(&mut evals, r);
            (evals, relation_segments, coeff_len / 2)
        };
        self.set_relation_weight_evals(evals, next_relation_segments, next_coeff_len);
    }
}

#[cfg(test)]
mod tests;
