//! Single-stage combined norm + relation sumcheck for `b = 4`.
//!
//! This specialization keeps the low degree-3 structure by tracking both the
//! witness table `w` and the virtual table `S = w(w+1)`. For compact rounds the
//! norm side uses the tiny `S ∈ {0, 2}` domain directly, so the range-check
//! contribution is a 1-bit lookup-style transition instead of a generic dense
//! evaluation.

use super::hachi_stage1::range_check_eval_from_s;
use super::hachi_stage2::{
    accumulate_relation_coeffs, accumulate_relation_coeffs_signed, HachiStage2Verifier,
};
use super::{
    fold_evals_in_place, trim_trailing_zeros, CompactPairFoldLut, SumcheckInstanceProver,
    SumcheckInstanceVerifier, UniPoly,
};
use crate::algebra::fields::HasUnreducedOps;
use crate::algebra::CyclotomicRing;
use crate::error::HachiError;
#[cfg(feature = "parallel")]
use crate::parallel::*;
use crate::{cfg_fold_reduce, cfg_into_iter};
use crate::{AdditiveGroup, CanonicalField, FieldCore, FromSmallInt};

#[derive(Clone, Copy)]
enum NormRoundTerms<E: FieldCore> {
    Full([E; 3]),
    SkipLinear([E; 2]),
}

enum STable<E: FieldCore> {
    Compact(Vec<i32>),
    Full(Vec<E>),
}

enum WTable<E: FieldCore> {
    Compact(Vec<i8>),
    Full(Vec<E>),
}

#[inline]
fn coeffs_to_poly<E: FieldCore>(coeffs: [E; 3]) -> UniPoly<E> {
    let mut coeffs = vec![coeffs[0], coeffs[1], coeffs[2]];
    trim_trailing_zeros(&mut coeffs);
    UniPoly::from_coeffs(coeffs)
}

#[inline]
fn reduce_signed_accum<E: FieldCore + HasUnreducedOps>(
    pos: E::MulU64Accum,
    neg: E::MulU64Accum,
) -> E {
    E::reduce_mul_u64_accum(pos) - E::reduce_mul_u64_accum(neg)
}

#[inline]
fn reduce_compact_rel<E: FieldCore + HasUnreducedOps>(rel: [E::MulU64Accum; 6]) -> [E; 3] {
    [
        reduce_signed_accum::<E>(rel[0], rel[1]),
        reduce_signed_accum::<E>(rel[2], rel[3]),
        reduce_signed_accum::<E>(rel[4], rel[5]),
    ]
}

#[inline]
fn scale_and_add_polys<E: FieldCore>(scale: E, lhs: &UniPoly<E>, rhs: &UniPoly<E>) -> UniPoly<E> {
    let max_len = lhs.coeffs.len().max(rhs.coeffs.len());
    let mut coeffs = vec![E::zero(); max_len];
    for (idx, coeff) in lhs.coeffs.iter().enumerate() {
        coeffs[idx] += scale * *coeff;
    }
    for (idx, coeff) in rhs.coeffs.iter().enumerate() {
        coeffs[idx] += *coeff;
    }
    trim_trailing_zeros(&mut coeffs);
    UniPoly::from_coeffs(coeffs)
}

#[inline]
fn b4_compact_norm_pair_changes(s0: i32, s1: i32) -> bool {
    debug_assert!(matches!(s0, 0 | 2));
    debug_assert!(matches!(s1, 0 | 2));
    s0 != s1
}

#[inline]
fn b4_full_norm_coeffs<E: FieldCore + FromSmallInt>(s0: E, s1: E) -> [E; 3] {
    let ds = s1 - s0;
    let two = E::from_u64(2);
    [s0 * (s0 - two), ds * ((s0 + s0) - two), ds * ds]
}

/// Single-stage combined norm + relation prover for `b = 4`.
pub struct CombinedNormRelationProver<E: FieldCore> {
    batching_coeff: E,
    s_table: STable<E>,
    w_table: WTable<E>,
    split_eq: super::split_eq::GruenSplitEq<E>,
    alpha_compact: Vec<E>,
    m_compact: Vec<E>,
    live_x_cols: usize,
    num_u: usize,
    num_vars: usize,
    relation_claim: E,
    prev_norm_claim: E,
    pending_norm_poly: Option<UniPoly<E>>,
    rounds_completed: usize,
}

impl<E: FieldCore + FromSmallInt + CanonicalField + HasUnreducedOps> CombinedNormRelationProver<E> {
    /// Build the single-stage combined prover for `b = 4`.
    ///
    /// # Panics
    ///
    /// Panics if the compact witness table, equality table, or relation tables
    /// have inconsistent dimensions.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        batching_coeff: E,
        w_evals_compact: Vec<i8>,
        tau0: &[E],
        alpha_evals_y: Vec<E>,
        m_evals_x: Vec<E>,
        live_x_cols: usize,
        num_u: usize,
        num_l: usize,
        relation_claim: E,
    ) -> Self {
        let num_vars = num_u + num_l;
        let y_len = 1usize << num_l;
        assert_eq!(w_evals_compact.len(), live_x_cols * y_len);
        assert_eq!(tau0.len(), num_vars);
        assert_eq!(alpha_evals_y.len(), y_len);
        assert_eq!(m_evals_x.len(), 1usize << num_u);

        let s_table = w_evals_compact
            .iter()
            .map(|&w| {
                let w = i32::from(w);
                w * (w + 1)
            })
            .collect();

        Self {
            batching_coeff,
            s_table: STable::Compact(s_table),
            w_table: WTable::Compact(w_evals_compact),
            split_eq: super::split_eq::GruenSplitEq::new(tau0),
            alpha_compact: alpha_evals_y,
            m_compact: m_evals_x,
            live_x_cols,
            num_u,
            num_vars,
            relation_claim,
            prev_norm_claim: E::zero(),
            pending_norm_poly: None,
            rounds_completed: 0,
        }
    }

    /// Return the final claimed evaluation of `S = w(w+1)`.
    ///
    /// # Panics
    ///
    /// Panics if the prover has not been fully folded to a single `S` value.
    pub fn final_s_claim(&self) -> E {
        match &self.s_table {
            STable::Full(s_full) => {
                assert_eq!(s_full.len(), 1, "s_table not fully folded");
                s_full[0]
            }
            STable::Compact(_) => panic!("s_table remained compact after final fold"),
        }
    }

    /// Return the final claimed evaluation of the next witness `w`.
    ///
    /// # Panics
    ///
    /// Panics if the prover has not been fully folded to a single witness
    /// value.
    pub fn final_w_eval(&self) -> E {
        match &self.w_table {
            WTable::Full(w_full) => {
                assert_eq!(w_full.len(), 1, "w_table not fully folded");
                w_full[0]
            }
            WTable::Compact(_) => panic!("w_table remained compact after final fold"),
        }
    }

    #[inline]
    fn current_x_width(&self) -> usize {
        self.num_u.saturating_sub(self.rounds_completed)
    }

    #[inline]
    fn current_x_len(&self) -> usize {
        1usize << self.current_x_width()
    }

    #[inline]
    fn use_prefix_x_round(&self) -> bool {
        self.rounds_completed < self.num_u && self.live_x_cols < self.current_x_len()
    }

    #[inline]
    fn can_skip_norm_linear_coeff(&self) -> bool {
        self.split_eq.can_recover_linear_q_term_from_claim()
    }

    #[inline]
    fn norm_poly_from_terms(&self, norm_terms: NormRoundTerms<E>) -> UniPoly<E> {
        match norm_terms {
            NormRoundTerms::Full(q_coeffs) => self.split_eq.gruen_mul(&coeffs_to_poly(q_coeffs)),
            NormRoundTerms::SkipLinear([q_constant, q_quadratic]) => self
                .split_eq
                .try_gruen_poly_deg_3(q_constant, q_quadratic, self.prev_norm_claim)
                .expect("split-eq norm claim recovery should succeed"),
        }
    }

    fn compute_round_compact_dense_terms(
        &self,
        s_compact: &[i32],
        w_compact: &[i8],
    ) -> (NormRoundTerms<E>, [E; 3]) {
        let (e_first, e_second) = self.split_eq.remaining_eq_tables();
        let num_first = e_first.len();
        let num_second = e_second.len();
        let current_x_width = self.current_x_width();
        let current_x_mask = (1usize << current_x_width).wrapping_sub(1);
        let alpha_compact = &self.alpha_compact;
        let m_compact = &self.m_compact;
        debug_assert_eq!(s_compact.len(), w_compact.len());
        debug_assert_eq!(w_compact.len() / 2, num_first * num_second);

        if self.can_skip_norm_linear_coeff() {
            let (q2_coeff, rel_accum) = cfg_fold_reduce!(
                0..num_second,
                || (E::ProductAccum::ZERO, [E::MulU64Accum::ZERO; 6]),
                |(mut q2_outer, mut rel), j_high| {
                    let mut q2_inner = E::MulU64Accum::ZERO;
                    let base = j_high * num_first;

                    for (j_low, &e_in) in e_first.iter().enumerate() {
                        let j = base + j_low;
                        let s0 = s_compact[2 * j];
                        let s1 = s_compact[2 * j + 1];
                        if b4_compact_norm_pair_changes(s0, s1) {
                            q2_inner += e_in.mul_u64_unreduced(4);
                        }

                        let w0 = i64::from(w_compact[2 * j]);
                        let w1 = i64::from(w_compact[2 * j + 1]);
                        let dw = w1 - w0;
                        let a0 = alpha_compact[(2 * j) >> current_x_width];
                        let a1 = alpha_compact[(2 * j + 1) >> current_x_width];
                        let m0 = m_compact[(2 * j) & current_x_mask];
                        let m1 = m_compact[(2 * j + 1) & current_x_mask];
                        accumulate_relation_coeffs_signed::<E>(&mut rel, w0, dw, a0 * m0, a1 * m1);
                    }

                    let e_out = e_second[j_high];
                    q2_outer += e_out.mul_to_product_accum(E::reduce_mul_u64_accum(q2_inner));
                    (q2_outer, rel)
                },
                |(qa, mut ra), (qb, rb)| {
                    let q = qa + qb;
                    for (ai, bi) in ra.iter_mut().zip(rb.iter()) {
                        *ai += *bi;
                    }
                    (q, ra)
                }
            );
            (
                NormRoundTerms::SkipLinear([E::zero(), E::reduce_product_accum(q2_coeff)]),
                reduce_compact_rel(rel_accum),
            )
        } else {
            let (q1_coeff, q2_coeff, rel_accum) = cfg_fold_reduce!(
                0..num_second,
                || {
                    (
                        E::ProductAccum::ZERO,
                        E::ProductAccum::ZERO,
                        [E::MulU64Accum::ZERO; 6],
                    )
                },
                |(mut q1_outer, mut q2_outer, mut rel), j_high| {
                    let mut q1_inner_neg = E::MulU64Accum::ZERO;
                    let mut q2_inner = E::MulU64Accum::ZERO;
                    let base = j_high * num_first;

                    for (j_low, &e_in) in e_first.iter().enumerate() {
                        let j = base + j_low;
                        let s0 = s_compact[2 * j];
                        let s1 = s_compact[2 * j + 1];
                        if b4_compact_norm_pair_changes(s0, s1) {
                            q1_inner_neg += e_in.mul_u64_unreduced(4);
                            q2_inner += e_in.mul_u64_unreduced(4);
                        }

                        let w0 = i64::from(w_compact[2 * j]);
                        let w1 = i64::from(w_compact[2 * j + 1]);
                        let dw = w1 - w0;
                        let a0 = alpha_compact[(2 * j) >> current_x_width];
                        let a1 = alpha_compact[(2 * j + 1) >> current_x_width];
                        let m0 = m_compact[(2 * j) & current_x_mask];
                        let m1 = m_compact[(2 * j + 1) & current_x_mask];
                        accumulate_relation_coeffs_signed::<E>(&mut rel, w0, dw, a0 * m0, a1 * m1);
                    }

                    let e_out = e_second[j_high];
                    let q1_inner = E::zero() - E::reduce_mul_u64_accum(q1_inner_neg);
                    q1_outer += e_out.mul_to_product_accum(q1_inner);
                    q2_outer += e_out.mul_to_product_accum(E::reduce_mul_u64_accum(q2_inner));
                    (q1_outer, q2_outer, rel)
                },
                |(q1a, q2a, mut ra), (q1b, q2b, rb)| {
                    let q1 = q1a + q1b;
                    let q2 = q2a + q2b;
                    for (ai, bi) in ra.iter_mut().zip(rb.iter()) {
                        *ai += *bi;
                    }
                    (q1, q2, ra)
                }
            );
            (
                NormRoundTerms::Full([
                    E::zero(),
                    E::reduce_product_accum(q1_coeff),
                    E::reduce_product_accum(q2_coeff),
                ]),
                reduce_compact_rel(rel_accum),
            )
        }
    }

    fn compute_round_compact_prefix_x_terms(
        &self,
        s_compact: &[i32],
        w_compact: &[i8],
    ) -> (NormRoundTerms<E>, [E; 3]) {
        let (e_first, e_second) = self.split_eq.remaining_eq_tables();
        let num_first = e_first.len();
        let first_bits = num_first.trailing_zeros() as usize;
        let current_x_half = 1usize << (self.current_x_width() - 1);
        let live_pairs = self.live_x_cols.div_ceil(2);
        let block_size = num_first.min(live_pairs);
        let alpha_compact = &self.alpha_compact;
        let m_compact = &self.m_compact;

        if self.can_skip_norm_linear_coeff() {
            let (q2_coeff, rel_accum) = cfg_fold_reduce!(
                0..alpha_compact.len(),
                || (E::ProductAccum::ZERO, [E::MulU64Accum::ZERO; 6]),
                |(mut q2_outer, mut rel), y| {
                    let row_start = y * self.live_x_cols;
                    let s_row = &s_compact[row_start..row_start + self.live_x_cols];
                    let w_row = &w_compact[row_start..row_start + self.live_x_cols];
                    let alpha = alpha_compact[y];
                    let j_base = y * current_x_half;

                    let mut blk = 0usize;
                    while blk < live_pairs {
                        let blk_end = (blk + block_size).min(live_pairs);
                        let j_high = (j_base + blk) >> first_bits;
                        let mut q2_inner = E::MulU64Accum::ZERO;

                        for pair_x in blk..blk_end {
                            let j_low = (j_base + pair_x) & (num_first - 1);
                            let e_in = e_first[j_low];
                            let left = 2 * pair_x;
                            let s0 = s_row[left];
                            let s1 = if left + 1 < self.live_x_cols {
                                s_row[left + 1]
                            } else {
                                0
                            };
                            if b4_compact_norm_pair_changes(s0, s1) {
                                q2_inner += e_in.mul_u64_unreduced(4);
                            }

                            let w0 = i64::from(w_row[left]);
                            let w1 = if left + 1 < self.live_x_cols {
                                i64::from(w_row[left + 1])
                            } else {
                                0
                            };
                            let dw = w1 - w0;
                            accumulate_relation_coeffs_signed::<E>(
                                &mut rel,
                                w0,
                                dw,
                                alpha * m_compact[left],
                                alpha * m_compact[left + 1],
                            );
                        }

                        let e_out = e_second[j_high];
                        q2_outer += e_out.mul_to_product_accum(E::reduce_mul_u64_accum(q2_inner));
                        blk = blk_end;
                    }

                    (q2_outer, rel)
                },
                |(qa, mut ra), (qb, rb)| {
                    let q = qa + qb;
                    for (ai, bi) in ra.iter_mut().zip(rb.iter()) {
                        *ai += *bi;
                    }
                    (q, ra)
                }
            );
            (
                NormRoundTerms::SkipLinear([E::zero(), E::reduce_product_accum(q2_coeff)]),
                reduce_compact_rel(rel_accum),
            )
        } else {
            let (q1_coeff, q2_coeff, rel_accum) = cfg_fold_reduce!(
                0..alpha_compact.len(),
                || {
                    (
                        E::ProductAccum::ZERO,
                        E::ProductAccum::ZERO,
                        [E::MulU64Accum::ZERO; 6],
                    )
                },
                |(mut q1_outer, mut q2_outer, mut rel), y| {
                    let row_start = y * self.live_x_cols;
                    let s_row = &s_compact[row_start..row_start + self.live_x_cols];
                    let w_row = &w_compact[row_start..row_start + self.live_x_cols];
                    let alpha = alpha_compact[y];
                    let j_base = y * current_x_half;

                    let mut blk = 0usize;
                    while blk < live_pairs {
                        let blk_end = (blk + block_size).min(live_pairs);
                        let j_high = (j_base + blk) >> first_bits;
                        let mut q1_inner_neg = E::MulU64Accum::ZERO;
                        let mut q2_inner = E::MulU64Accum::ZERO;

                        for pair_x in blk..blk_end {
                            let j_low = (j_base + pair_x) & (num_first - 1);
                            let e_in = e_first[j_low];
                            let left = 2 * pair_x;
                            let s0 = s_row[left];
                            let s1 = if left + 1 < self.live_x_cols {
                                s_row[left + 1]
                            } else {
                                0
                            };
                            if b4_compact_norm_pair_changes(s0, s1) {
                                q1_inner_neg += e_in.mul_u64_unreduced(4);
                                q2_inner += e_in.mul_u64_unreduced(4);
                            }

                            let w0 = i64::from(w_row[left]);
                            let w1 = if left + 1 < self.live_x_cols {
                                i64::from(w_row[left + 1])
                            } else {
                                0
                            };
                            let dw = w1 - w0;
                            accumulate_relation_coeffs_signed::<E>(
                                &mut rel,
                                w0,
                                dw,
                                alpha * m_compact[left],
                                alpha * m_compact[left + 1],
                            );
                        }

                        let e_out = e_second[j_high];
                        let q1_inner = E::zero() - E::reduce_mul_u64_accum(q1_inner_neg);
                        q1_outer += e_out.mul_to_product_accum(q1_inner);
                        q2_outer += e_out.mul_to_product_accum(E::reduce_mul_u64_accum(q2_inner));
                        blk = blk_end;
                    }

                    (q1_outer, q2_outer, rel)
                },
                |(q1a, q2a, mut ra), (q1b, q2b, rb)| {
                    let q1 = q1a + q1b;
                    let q2 = q2a + q2b;
                    for (ai, bi) in ra.iter_mut().zip(rb.iter()) {
                        *ai += *bi;
                    }
                    (q1, q2, ra)
                }
            );
            (
                NormRoundTerms::Full([
                    E::zero(),
                    E::reduce_product_accum(q1_coeff),
                    E::reduce_product_accum(q2_coeff),
                ]),
                reduce_compact_rel(rel_accum),
            )
        }
    }

    fn compute_round_full_dense_terms(
        &self,
        s_full: &[E],
        w_full: &[E],
    ) -> (NormRoundTerms<E>, [E; 3]) {
        let (e_first, e_second) = self.split_eq.remaining_eq_tables();
        let num_first = e_first.len();
        let num_second = e_second.len();
        let current_x_width = self.current_x_width();
        let current_x_mask = (1usize << current_x_width).wrapping_sub(1);
        let alpha_compact = &self.alpha_compact;
        let m_compact = &self.m_compact;
        debug_assert_eq!(s_full.len(), w_full.len());
        debug_assert_eq!(w_full.len() / 2, num_first * num_second);

        if self.can_skip_norm_linear_coeff() {
            let (norm_terms, rel_coeffs) = cfg_fold_reduce!(
                0..num_second,
                || ([E::zero(); 2], [E::zero(); 3]),
                |(mut norm, mut rel), j_high| {
                    let base = j_high * num_first;
                    let mut inner_norm = [E::zero(); 2];

                    for (j_low, &e_in) in e_first.iter().enumerate() {
                        let j = base + j_low;
                        let [q0, _, q2] = b4_full_norm_coeffs(s_full[2 * j], s_full[2 * j + 1]);
                        inner_norm[0] += e_in * q0;
                        inner_norm[1] += e_in * q2;

                        let w0 = w_full[2 * j];
                        let w1 = w_full[2 * j + 1];
                        let dw = w1 - w0;
                        let a0 = alpha_compact[(2 * j) >> current_x_width];
                        let a1 = alpha_compact[(2 * j + 1) >> current_x_width];
                        let m0 = m_compact[(2 * j) & current_x_mask];
                        let m1 = m_compact[(2 * j + 1) & current_x_mask];
                        accumulate_relation_coeffs(&mut rel, w0, dw, a0 * m0, a1 * m1);
                    }

                    let e_out = e_second[j_high];
                    norm[0] += e_out * inner_norm[0];
                    norm[1] += e_out * inner_norm[1];
                    (norm, rel)
                },
                |(mut na, mut ra), (nb, rb)| {
                    for (ai, bi) in na.iter_mut().zip(nb.iter()) {
                        *ai += *bi;
                    }
                    for (ai, bi) in ra.iter_mut().zip(rb.iter()) {
                        *ai += *bi;
                    }
                    (na, ra)
                }
            );
            (NormRoundTerms::SkipLinear(norm_terms), rel_coeffs)
        } else {
            let (norm_terms, rel_coeffs) = cfg_fold_reduce!(
                0..num_second,
                || ([E::zero(); 3], [E::zero(); 3]),
                |(mut norm, mut rel), j_high| {
                    let base = j_high * num_first;
                    let mut inner_norm = [E::zero(); 3];

                    for (j_low, &e_in) in e_first.iter().enumerate() {
                        let j = base + j_low;
                        let [q0, q1, q2] = b4_full_norm_coeffs(s_full[2 * j], s_full[2 * j + 1]);
                        inner_norm[0] += e_in * q0;
                        inner_norm[1] += e_in * q1;
                        inner_norm[2] += e_in * q2;

                        let w0 = w_full[2 * j];
                        let w1 = w_full[2 * j + 1];
                        let dw = w1 - w0;
                        let a0 = alpha_compact[(2 * j) >> current_x_width];
                        let a1 = alpha_compact[(2 * j + 1) >> current_x_width];
                        let m0 = m_compact[(2 * j) & current_x_mask];
                        let m1 = m_compact[(2 * j + 1) & current_x_mask];
                        accumulate_relation_coeffs(&mut rel, w0, dw, a0 * m0, a1 * m1);
                    }

                    let e_out = e_second[j_high];
                    norm[0] += e_out * inner_norm[0];
                    norm[1] += e_out * inner_norm[1];
                    norm[2] += e_out * inner_norm[2];
                    (norm, rel)
                },
                |(mut na, mut ra), (nb, rb)| {
                    for (ai, bi) in na.iter_mut().zip(nb.iter()) {
                        *ai += *bi;
                    }
                    for (ai, bi) in ra.iter_mut().zip(rb.iter()) {
                        *ai += *bi;
                    }
                    (na, ra)
                }
            );
            (NormRoundTerms::Full(norm_terms), rel_coeffs)
        }
    }

    fn compute_round_full_prefix_x_terms(
        &self,
        s_full: &[E],
        w_full: &[E],
    ) -> (NormRoundTerms<E>, [E; 3]) {
        let (e_first, e_second) = self.split_eq.remaining_eq_tables();
        let num_first = e_first.len();
        let first_bits = num_first.trailing_zeros() as usize;
        let current_x_half = 1usize << (self.current_x_width() - 1);
        let live_pairs = self.live_x_cols.div_ceil(2);
        let block_size = num_first.min(live_pairs);
        let alpha_compact = &self.alpha_compact;
        let m_compact = &self.m_compact;

        if self.can_skip_norm_linear_coeff() {
            let (norm_terms, rel_coeffs) = cfg_fold_reduce!(
                0..alpha_compact.len(),
                || ([E::zero(); 2], [E::zero(); 3]),
                |(mut norm, mut rel), y| {
                    let row_start = y * self.live_x_cols;
                    let s_row = &s_full[row_start..row_start + self.live_x_cols];
                    let w_row = &w_full[row_start..row_start + self.live_x_cols];
                    let alpha = alpha_compact[y];
                    let j_base = y * current_x_half;

                    let mut blk = 0usize;
                    while blk < live_pairs {
                        let blk_end = (blk + block_size).min(live_pairs);
                        let j_high = (j_base + blk) >> first_bits;
                        let mut inner_norm = [E::zero(); 2];

                        for pair_x in blk..blk_end {
                            let j_low = (j_base + pair_x) & (num_first - 1);
                            let e_in = e_first[j_low];
                            let left = 2 * pair_x;
                            let s0 = s_row[left];
                            let s1 = if left + 1 < self.live_x_cols {
                                s_row[left + 1]
                            } else {
                                E::zero()
                            };
                            let [q0, _, q2] = b4_full_norm_coeffs(s0, s1);
                            inner_norm[0] += e_in * q0;
                            inner_norm[1] += e_in * q2;

                            let w0 = w_row[left];
                            let w1 = if left + 1 < self.live_x_cols {
                                w_row[left + 1]
                            } else {
                                E::zero()
                            };
                            let dw = w1 - w0;
                            accumulate_relation_coeffs(
                                &mut rel,
                                w0,
                                dw,
                                alpha * m_compact[left],
                                alpha * m_compact[left + 1],
                            );
                        }

                        let e_out = e_second[j_high];
                        norm[0] += e_out * inner_norm[0];
                        norm[1] += e_out * inner_norm[1];
                        blk = blk_end;
                    }
                    (norm, rel)
                },
                |(mut na, mut ra), (nb, rb)| {
                    for (ai, bi) in na.iter_mut().zip(nb.iter()) {
                        *ai += *bi;
                    }
                    for (ai, bi) in ra.iter_mut().zip(rb.iter()) {
                        *ai += *bi;
                    }
                    (na, ra)
                }
            );
            (NormRoundTerms::SkipLinear(norm_terms), rel_coeffs)
        } else {
            let (norm_terms, rel_coeffs) = cfg_fold_reduce!(
                0..alpha_compact.len(),
                || ([E::zero(); 3], [E::zero(); 3]),
                |(mut norm, mut rel), y| {
                    let row_start = y * self.live_x_cols;
                    let s_row = &s_full[row_start..row_start + self.live_x_cols];
                    let w_row = &w_full[row_start..row_start + self.live_x_cols];
                    let alpha = alpha_compact[y];
                    let j_base = y * current_x_half;

                    let mut blk = 0usize;
                    while blk < live_pairs {
                        let blk_end = (blk + block_size).min(live_pairs);
                        let j_high = (j_base + blk) >> first_bits;
                        let mut inner_norm = [E::zero(); 3];

                        for pair_x in blk..blk_end {
                            let j_low = (j_base + pair_x) & (num_first - 1);
                            let e_in = e_first[j_low];
                            let left = 2 * pair_x;
                            let s0 = s_row[left];
                            let s1 = if left + 1 < self.live_x_cols {
                                s_row[left + 1]
                            } else {
                                E::zero()
                            };
                            let [q0, q1, q2] = b4_full_norm_coeffs(s0, s1);
                            inner_norm[0] += e_in * q0;
                            inner_norm[1] += e_in * q1;
                            inner_norm[2] += e_in * q2;

                            let w0 = w_row[left];
                            let w1 = if left + 1 < self.live_x_cols {
                                w_row[left + 1]
                            } else {
                                E::zero()
                            };
                            let dw = w1 - w0;
                            accumulate_relation_coeffs(
                                &mut rel,
                                w0,
                                dw,
                                alpha * m_compact[left],
                                alpha * m_compact[left + 1],
                            );
                        }

                        let e_out = e_second[j_high];
                        norm[0] += e_out * inner_norm[0];
                        norm[1] += e_out * inner_norm[1];
                        norm[2] += e_out * inner_norm[2];
                        blk = blk_end;
                    }
                    (norm, rel)
                },
                |(mut na, mut ra), (nb, rb)| {
                    for (ai, bi) in na.iter_mut().zip(nb.iter()) {
                        *ai += *bi;
                    }
                    for (ai, bi) in ra.iter_mut().zip(rb.iter()) {
                        *ai += *bi;
                    }
                    (na, ra)
                }
            );
            (NormRoundTerms::Full(norm_terms), rel_coeffs)
        }
    }

    fn compute_current_round_poly_from_state(&mut self) -> UniPoly<E> {
        match (&self.s_table, &self.w_table) {
            (STable::Compact(s_compact), WTable::Compact(w_compact)) => {
                let (norm_terms, rel_coeffs) = if self.use_prefix_x_round() {
                    self.compute_round_compact_prefix_x_terms(s_compact, w_compact)
                } else {
                    self.compute_round_compact_dense_terms(s_compact, w_compact)
                };
                let norm_poly = self.norm_poly_from_terms(norm_terms);
                self.pending_norm_poly = Some(norm_poly.clone());
                let relation_poly = coeffs_to_poly(rel_coeffs);
                scale_and_add_polys(self.batching_coeff, &norm_poly, &relation_poly)
            }
            (STable::Full(s_full), WTable::Full(w_full)) => {
                let (norm_terms, rel_coeffs) = if self.use_prefix_x_round() {
                    self.compute_round_full_prefix_x_terms(s_full, w_full)
                } else {
                    self.compute_round_full_dense_terms(s_full, w_full)
                };
                let norm_poly = self.norm_poly_from_terms(norm_terms);
                self.pending_norm_poly = Some(norm_poly.clone());
                let relation_poly = coeffs_to_poly(rel_coeffs);
                scale_and_add_polys(self.batching_coeff, &norm_poly, &relation_poly)
            }
            _ => unreachable!("combined prover s/w table representations diverged"),
        }
    }

    #[inline]
    fn build_compact_s_fold_lut(r: E) -> CompactPairFoldLut<E> {
        CompactPairFoldLut::from_allowed_values(&[0, 2], r)
    }

    #[inline]
    fn build_compact_w_fold_lut(r: E) -> CompactPairFoldLut<E> {
        CompactPairFoldLut::from_contiguous_range(-2, 1, r)
    }

    fn fold_s_compact_to_full(s_compact: &[i32], fold_lut: &CompactPairFoldLut<E>) -> Vec<E> {
        cfg_into_iter!(0..s_compact.len() / 2)
            .map(|j| fold_lut.fold(s_compact[2 * j], s_compact[2 * j + 1]))
            .collect()
    }

    fn fold_w_compact_to_full(w_compact: &[i8], fold_lut: &CompactPairFoldLut<E>) -> Vec<E> {
        cfg_into_iter!(0..w_compact.len() / 2)
            .map(|j| fold_lut.fold(i32::from(w_compact[2 * j]), i32::from(w_compact[2 * j + 1])))
            .collect()
    }

    fn fold_s_compact_prefix_x(
        s_compact: &[i32],
        live_x_cols: usize,
        y_len: usize,
        fold_lut: &CompactPairFoldLut<E>,
    ) -> Vec<E> {
        let next_live_x_cols = live_x_cols.div_ceil(2);
        let mut out = vec![E::zero(); y_len * next_live_x_cols];
        #[cfg(feature = "parallel")]
        out.par_chunks_mut(next_live_x_cols)
            .enumerate()
            .for_each(|(y, row_out)| {
                let row = &s_compact[y * live_x_cols..(y + 1) * live_x_cols];
                for (pair_x, dst) in row_out.iter_mut().enumerate() {
                    let left = 2 * pair_x;
                    let s1 = row.get(left + 1).copied().unwrap_or_default();
                    *dst = fold_lut.fold(row[left], s1);
                }
            });
        #[cfg(not(feature = "parallel"))]
        for (y, row_out) in out.chunks_mut(next_live_x_cols).enumerate() {
            let row = &s_compact[y * live_x_cols..(y + 1) * live_x_cols];
            for (pair_x, dst) in row_out.iter_mut().enumerate() {
                let left = 2 * pair_x;
                let s1 = row.get(left + 1).copied().unwrap_or_default();
                *dst = fold_lut.fold(row[left], s1);
            }
        }
        out
    }

    fn fold_w_compact_prefix_x(
        w_compact: &[i8],
        live_x_cols: usize,
        y_len: usize,
        fold_lut: &CompactPairFoldLut<E>,
    ) -> Vec<E> {
        let next_live_x_cols = live_x_cols.div_ceil(2);
        let mut out = vec![E::zero(); y_len * next_live_x_cols];
        #[cfg(feature = "parallel")]
        out.par_chunks_mut(next_live_x_cols)
            .enumerate()
            .for_each(|(y, row_out)| {
                let row = &w_compact[y * live_x_cols..(y + 1) * live_x_cols];
                for (pair_x, dst) in row_out.iter_mut().enumerate() {
                    let left = 2 * pair_x;
                    let w1 = row.get(left + 1).copied().unwrap_or_default();
                    *dst = fold_lut.fold(i32::from(row[left]), i32::from(w1));
                }
            });
        #[cfg(not(feature = "parallel"))]
        for (y, row_out) in out.chunks_mut(next_live_x_cols).enumerate() {
            let row = &w_compact[y * live_x_cols..(y + 1) * live_x_cols];
            for (pair_x, dst) in row_out.iter_mut().enumerate() {
                let left = 2 * pair_x;
                let w1 = row.get(left + 1).copied().unwrap_or_default();
                *dst = fold_lut.fold(i32::from(row[left]), i32::from(w1));
            }
        }
        out
    }

    fn fold_full_prefix_x(full: &[E], live_x_cols: usize, y_len: usize, r: E) -> Vec<E> {
        let next_live_x_cols = live_x_cols.div_ceil(2);
        let mut out = vec![E::zero(); y_len * next_live_x_cols];
        #[cfg(feature = "parallel")]
        out.par_chunks_mut(next_live_x_cols)
            .enumerate()
            .for_each(|(y, row_out)| {
                let row = &full[y * live_x_cols..(y + 1) * live_x_cols];
                for (pair_x, dst) in row_out.iter_mut().enumerate() {
                    let left = 2 * pair_x;
                    let left_val = row[left];
                    let right_val = row.get(left + 1).copied().unwrap_or_else(E::zero);
                    *dst = left_val + r * (right_val - left_val);
                }
            });
        #[cfg(not(feature = "parallel"))]
        for (y, row_out) in out.chunks_mut(next_live_x_cols).enumerate() {
            let row = &full[y * live_x_cols..(y + 1) * live_x_cols];
            for (pair_x, dst) in row_out.iter_mut().enumerate() {
                let left = 2 * pair_x;
                let left_val = row[left];
                let right_val = row.get(left + 1).copied().unwrap_or_else(E::zero);
                *dst = left_val + r * (right_val - left_val);
            }
        }
        out
    }

    fn fold_m_prefix(m_compact: &[E], r: E) -> Vec<E> {
        cfg_into_iter!(0..(m_compact.len() / 2))
            .map(|pair_x| {
                let left = 2 * pair_x;
                let m0 = m_compact[left];
                let m1 = m_compact[left + 1];
                m0 + r * (m1 - m0)
            })
            .collect()
    }
}

impl<E: FieldCore + FromSmallInt + CanonicalField + HasUnreducedOps> SumcheckInstanceProver<E>
    for CombinedNormRelationProver<E>
{
    fn num_rounds(&self) -> usize {
        self.num_vars
    }

    fn degree_bound(&self) -> usize {
        3
    }

    fn input_claim(&self) -> E {
        self.relation_claim
    }

    fn compute_round_univariate(&mut self, _round: usize, _previous_claim: E) -> UniPoly<E> {
        self.compute_current_round_poly_from_state()
    }

    fn ingest_challenge(&mut self, _round: usize, r: E) {
        if let Some(norm_poly) = self.pending_norm_poly.take() {
            self.prev_norm_claim = norm_poly.evaluate(&r);
        }
        self.split_eq.bind(r);

        let folding_x_round = self.rounds_completed < self.num_u;
        let use_prefix_x_round = self.use_prefix_x_round();

        let y_len = match &self.s_table {
            STable::Compact(s_compact) => s_compact.len() / self.live_x_cols,
            STable::Full(s_full) => s_full.len() / self.live_x_cols,
        };

        self.s_table = match std::mem::replace(&mut self.s_table, STable::Full(Vec::new())) {
            STable::Compact(s_compact) => {
                let fold_lut = Self::build_compact_s_fold_lut(r);
                let s_full = if use_prefix_x_round {
                    Self::fold_s_compact_prefix_x(&s_compact, self.live_x_cols, y_len, &fold_lut)
                } else {
                    Self::fold_s_compact_to_full(&s_compact, &fold_lut)
                };
                STable::Full(s_full)
            }
            STable::Full(mut s_full) => {
                if use_prefix_x_round {
                    STable::Full(Self::fold_full_prefix_x(
                        &s_full,
                        self.live_x_cols,
                        y_len,
                        r,
                    ))
                } else {
                    fold_evals_in_place(&mut s_full, r);
                    STable::Full(s_full)
                }
            }
        };

        self.w_table = match std::mem::replace(&mut self.w_table, WTable::Full(Vec::new())) {
            WTable::Compact(w_compact) => {
                let fold_lut = Self::build_compact_w_fold_lut(r);
                let w_full = if use_prefix_x_round {
                    Self::fold_w_compact_prefix_x(&w_compact, self.live_x_cols, y_len, &fold_lut)
                } else {
                    Self::fold_w_compact_to_full(&w_compact, &fold_lut)
                };
                WTable::Full(w_full)
            }
            WTable::Full(mut w_full) => {
                if use_prefix_x_round {
                    WTable::Full(Self::fold_full_prefix_x(
                        &w_full,
                        self.live_x_cols,
                        y_len,
                        r,
                    ))
                } else {
                    fold_evals_in_place(&mut w_full, r);
                    WTable::Full(w_full)
                }
            }
        };

        if folding_x_round {
            if use_prefix_x_round {
                self.m_compact = Self::fold_m_prefix(&self.m_compact, r);
            } else {
                fold_evals_in_place(&mut self.m_compact, r);
            }
            self.live_x_cols = self.live_x_cols.div_ceil(2);
        } else {
            fold_evals_in_place(&mut self.alpha_compact, r);
        }

        self.rounds_completed += 1;
    }
}

/// Single-stage combined norm + relation verifier for `b = 4`.
pub struct CombinedNormRelationVerifier<F: FieldCore, const D: usize> {
    batching_coeff: F,
    tau0: Vec<F>,
    s_claim: F,
    relation: HachiStage2Verifier<F, D>,
}

impl<F: FieldCore + FromSmallInt + CanonicalField, const D: usize>
    CombinedNormRelationVerifier<F, D>
{
    /// Build the combined verifier when the verifier has the full witness.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_full_witness(
        batching_coeff: F,
        s_claim: F,
        w_evals: Vec<F>,
        tau0: Vec<F>,
        alpha_evals_y: Vec<F>,
        m_evals_x: Vec<F>,
        tau1: Vec<F>,
        v: Vec<CyclotomicRing<F, D>>,
        u: Vec<CyclotomicRing<F, D>>,
        y_ring: CyclotomicRing<F, D>,
        alpha: F,
        num_u: usize,
        num_l: usize,
    ) -> Self {
        Self {
            batching_coeff,
            tau0: tau0.clone(),
            s_claim,
            relation: HachiStage2Verifier::new_with_full_witness(
                F::zero(),
                F::zero(),
                w_evals,
                tau0,
                alpha_evals_y,
                m_evals_x,
                tau1,
                v,
                u,
                y_ring,
                alpha,
                num_u,
                num_l,
            ),
        }
    }

    /// Build the combined verifier when only the final witness evaluation is available.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_claimed_w_eval(
        batching_coeff: F,
        s_claim: F,
        w_eval: F,
        tau0: Vec<F>,
        alpha_evals_y: Vec<F>,
        m_evals_x: Vec<F>,
        tau1: Vec<F>,
        v: Vec<CyclotomicRing<F, D>>,
        u: Vec<CyclotomicRing<F, D>>,
        y_ring: CyclotomicRing<F, D>,
        alpha: F,
        num_u: usize,
        num_l: usize,
    ) -> Self {
        Self {
            batching_coeff,
            tau0: tau0.clone(),
            s_claim,
            relation: HachiStage2Verifier::new_with_claimed_w_eval(
                F::zero(),
                F::zero(),
                w_eval,
                tau0,
                alpha_evals_y,
                m_evals_x,
                tau1,
                v,
                u,
                y_ring,
                alpha,
                num_u,
                num_l,
            ),
        }
    }
}

impl<F: FieldCore + FromSmallInt + CanonicalField, const D: usize> SumcheckInstanceVerifier<F>
    for CombinedNormRelationVerifier<F, D>
{
    fn num_rounds(&self) -> usize {
        self.relation.num_rounds()
    }

    fn degree_bound(&self) -> usize {
        3
    }

    fn input_claim(&self) -> F {
        self.relation.input_claim()
    }

    fn expected_output_claim(&self, challenges: &[F]) -> Result<F, HachiError> {
        let w_eval = self.relation.witness_eval(challenges)?;
        let expected_s = w_eval * (w_eval + F::one());
        if expected_s != self.s_claim {
            return Err(HachiError::InvalidProof);
        }
        let eq_val = super::eq_poly::EqPolynomial::mle(&self.tau0, challenges);
        let relation_oracle = self.relation.expected_output_claim(challenges)?;
        Ok(self.batching_coeff * eq_val * range_check_eval_from_s(expected_s, 4) + relation_oracle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::Prime128M8M4M1M0;
    use crate::protocol::sumcheck::eq_poly::EqPolynomial;
    use crate::protocol::sumcheck::hachi_stage1::{range_check_eval_from_s, HachiStage1Prover};
    use crate::protocol::sumcheck::hachi_stage2::HachiStage2Prover;
    use crate::protocol::sumcheck::{multilinear_eval, prove_sumcheck};
    use crate::protocol::transcript::labels as tr_labels;
    use crate::protocol::transcript::{Blake2bTranscript, Transcript};

    type F = Prime128M8M4M1M0;

    fn relation_claim_from_compact_w(
        w_compact: &[i8],
        alpha_evals_y: &[F],
        m_evals_x: &[F],
        live_x_cols: usize,
        num_u: usize,
        num_l: usize,
    ) -> F {
        let x_len = 1usize << num_u;
        let y_len = 1usize << num_l;
        let mut acc = F::zero();
        for y in 0..y_len {
            let row = &w_compact[y * live_x_cols..(y + 1) * live_x_cols];
            for x in 0..x_len {
                let w = row.get(x).copied().unwrap_or_default();
                acc += F::from_i64(w as i64) * alpha_evals_y[y] * m_evals_x[x];
            }
        }
        acc
    }

    fn new_transcript() -> Blake2bTranscript<F> {
        <Blake2bTranscript<F> as Transcript<F>>::new(tr_labels::DOMAIN_HACHI_PROTOCOL)
    }

    fn sample_round(tr: &mut Blake2bTranscript<F>) -> F {
        tr.challenge_scalar(tr_labels::CHALLENGE_SUMCHECK_ROUND)
    }

    #[test]
    fn combined_round0_matches_scaled_norm_plus_relation() {
        let num_u = 3usize;
        let num_l = 2usize;
        let live_x_cols = 1usize << num_u;
        let n = live_x_cols << num_l;
        let w_compact: Vec<i8> = (0..n).map(|i| ((i * 5 + 3) % 4) as i8 - 2).collect();
        let tau0: Vec<F> = (0..(num_u + num_l))
            .map(|i| F::from_u64((i as u64) + 11))
            .collect();
        let alpha_evals_y: Vec<F> = (0..(1usize << num_l))
            .map(|i| F::from_u64((3 * i as u64) + 13))
            .collect();
        let m_evals_x: Vec<F> = (0..(1usize << num_u))
            .map(|i| F::from_u64((7 * i as u64) + 17))
            .collect();
        let batching_coeff = F::from_u64(19);
        let relation_claim = relation_claim_from_compact_w(
            &w_compact,
            &alpha_evals_y,
            &m_evals_x,
            live_x_cols,
            num_u,
            num_l,
        );

        let mut combined = CombinedNormRelationProver::new(
            batching_coeff,
            w_compact.clone(),
            &tau0,
            alpha_evals_y.clone(),
            m_evals_x.clone(),
            live_x_cols,
            num_u,
            num_l,
            relation_claim,
        );
        let mut norm = HachiStage1Prover::new(&w_compact, &tau0, 4, live_x_cols, num_u, num_l);
        let mut relation = HachiStage2Prover::new(
            F::zero(),
            w_compact.clone(),
            &tau0,
            F::zero(),
            4,
            alpha_evals_y,
            m_evals_x,
            live_x_cols,
            num_u,
            num_l,
            relation_claim,
        );

        let combined_round0 = combined.compute_round_univariate(0, relation_claim);
        let norm_round0 = norm.compute_round_univariate(0, F::zero());
        let relation_round0 = relation.compute_round_univariate(0, relation_claim);
        assert_eq!(
            combined_round0,
            scale_and_add_polys(batching_coeff, &norm_round0, &relation_round0)
        );
    }

    #[test]
    fn combined_sumcheck_final_claim_matches_expected_oracle() {
        let num_u = 3usize;
        let num_l = 2usize;
        let live_x_cols = 1usize << num_u;
        let n = live_x_cols << num_l;
        let w_compact: Vec<i8> = (0..n).map(|i| ((i * 7 + 1) % 4) as i8 - 2).collect();
        let tau0: Vec<F> = (0..(num_u + num_l))
            .map(|i| F::from_u64((i as u64) + 23))
            .collect();
        let alpha_evals_y: Vec<F> = (0..(1usize << num_l))
            .map(|i| F::from_u64((5 * i as u64) + 29))
            .collect();
        let m_evals_x: Vec<F> = (0..(1usize << num_u))
            .map(|i| F::from_u64((11 * i as u64) + 31))
            .collect();
        let batching_coeff = F::from_u64(37);
        let relation_claim = relation_claim_from_compact_w(
            &w_compact,
            &alpha_evals_y,
            &m_evals_x,
            live_x_cols,
            num_u,
            num_l,
        );

        let mut prover = CombinedNormRelationProver::new(
            batching_coeff,
            w_compact,
            &tau0,
            alpha_evals_y.clone(),
            m_evals_x.clone(),
            live_x_cols,
            num_u,
            num_l,
            relation_claim,
        );
        let mut transcript = new_transcript();
        let (_proof, challenges, final_claim) =
            prove_sumcheck::<F, _, F, _, _>(&mut prover, &mut transcript, sample_round).unwrap();
        let s_claim = prover.final_s_claim();
        let w_eval = prover.final_w_eval();
        let eq_val = EqPolynomial::mle(&tau0, &challenges);
        let (x_challenges, y_challenges) = challenges.split_at(num_u);
        let alpha_val = multilinear_eval(&alpha_evals_y, y_challenges).unwrap();
        let m_val = multilinear_eval(&m_evals_x, x_challenges).unwrap();
        let expected = batching_coeff * eq_val * range_check_eval_from_s(s_claim, 4)
            + w_eval * alpha_val * m_val;

        assert_eq!(final_claim, expected);
    }

    #[test]
    fn combined_verifier_rejects_inconsistent_s_claim() {
        const D: usize = 4;
        let tau0 = vec![F::from_u64(3), F::from_u64(5)];
        let alpha_evals_y = vec![F::from_u64(7), F::from_u64(11)];
        let m_evals_x = vec![F::from_u64(13), F::from_u64(17)];
        let tau1 = vec![F::from_u64(19)];
        let verifier = CombinedNormRelationVerifier::<F, D>::new_with_claimed_w_eval(
            F::from_u64(23),
            F::zero(),
            F::one(),
            tau0,
            alpha_evals_y,
            m_evals_x,
            tau1,
            Vec::new(),
            Vec::new(),
            CyclotomicRing::<F, D>::zero(),
            F::from_u64(29),
            1,
            1,
        );

        assert!(matches!(
            verifier.expected_output_claim(&[F::from_u64(31), F::from_u64(37)]),
            Err(HachiError::InvalidProof)
        ));
    }
}
