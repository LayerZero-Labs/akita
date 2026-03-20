//! Single-stage direct norm + relation sumcheck for `b = 4`.
//!
//! The committed witness is a Boolean table
//! `w : {0,1}^{num_u} x {0,1}^{num_l} -> {-2, -1, 0, 1}`. For `b = 4`, the
//! range-check polynomial from stage 1 is
//!
//! `Q(s) = s * (s - 2)`.
//!
//! This combined specialization proves the direct identity
//!
//! `relation_claim = sum_z [ gamma * eq(tau0, z) * Q(w(z) * (w(z) + 1))`
//! `                      + w(z) * a(y) * m(x) ]`.
//!
//! The norm half still sums to zero on an honest Boolean witness, so the
//! sumcheck input claim remains exactly `relation_claim`. Unlike the previous
//! unsound attempt, this prover never carries a separate multilinear `S`
//! oracle. Every norm contribution is derived directly from the same folded
//! witness state `W` that the verifier evaluates at the final random point.

use super::eq_poly::EqPolynomial;
use super::hachi_stage2::{accumulate_relation_coeffs, accumulate_relation_coeffs_signed};
use super::split_eq::GruenSplitEq;
use super::{
    fold_evals_in_place, multilinear_eval, trim_trailing_zeros, CompactPairFoldLut,
    SumcheckInstanceProver, SumcheckInstanceVerifier, UniPoly,
};
use crate::algebra::fields::HasUnreducedOps;
use crate::error::HachiError;
use crate::primitives::arithmetic::AdditiveGroup;
use crate::{CanonicalField, FieldCore, FromSmallInt};
use std::mem;

#[derive(Clone, Copy)]
enum NormRoundTerms<E: FieldCore> {
    Full([E; 5]),
    SkipLinear([E; 4]),
}

enum WTable<E: FieldCore> {
    Compact(Vec<i8>),
    Full(Vec<E>),
}

enum CombinedWitnessOracle<F: FieldCore> {
    Full(Vec<F>),
    ClaimedEval(F),
}

struct B4CompactNormPairLut {
    coeffs: [[i64; 5]; 16],
}

impl B4CompactNormPairLut {
    fn new() -> Self {
        let mut coeffs = [[0i64; 5]; 16];
        for w0 in -2i8..=1 {
            for w1 in -2i8..=1 {
                coeffs[Self::index(w0, w1)] = b4_direct_norm_coeffs_int(w0, w1);
            }
        }
        Self { coeffs }
    }

    #[inline]
    fn index(w0: i8, w1: i8) -> usize {
        debug_assert!((-2..=1).contains(&w0));
        debug_assert!((-2..=1).contains(&w1));
        ((w0 + 2) as usize) * 4 + (w1 + 2) as usize
    }

    #[inline]
    fn coeffs(&self, w0: i8, w1: i8) -> &[i64; 5] {
        &self.coeffs[Self::index(w0, w1)]
    }
}

#[inline]
fn coeffs_to_poly<E: FieldCore>(coeffs: &[E]) -> UniPoly<E> {
    let mut coeffs = coeffs.to_vec();
    trim_trailing_zeros(&mut coeffs);
    UniPoly::from_coeffs(coeffs)
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
fn direct_b4_range_check_from_w<E: FieldCore + FromSmallInt>(w: E) -> E {
    let s = w * (w + E::one());
    let two = E::from_u64(2);
    s * (s - two)
}

#[inline]
fn b4_direct_norm_coeffs<E: FieldCore + FromSmallInt>(w0: E, w1: E) -> [E; 5] {
    let dw = w1 - w0;
    let two = E::from_u64(2);
    let s0 = w0 * (w0 + E::one());
    let s1 = dw * (w0 + w0 + E::one());
    let s2 = dw * dw;
    let two_s0_minus_two = (s0 + s0) - two;
    [
        s0 * (s0 - two),
        s1 * two_s0_minus_two,
        s1 * s1 + s2 * two_s0_minus_two,
        (s1 + s1) * s2,
        s2 * s2,
    ]
}

#[inline]
fn b4_direct_norm_coeffs_int(w0: i8, w1: i8) -> [i64; 5] {
    let w0 = i64::from(w0);
    let w1 = i64::from(w1);
    let dw = w1 - w0;
    let s0 = w0 * (w0 + 1);
    let s1 = dw * (2 * w0 + 1);
    let s2 = dw * dw;
    let two_s0_minus_two = 2 * s0 - 2;
    [
        s0 * (s0 - 2),
        s1 * two_s0_minus_two,
        s1 * s1 + s2 * two_s0_minus_two,
        2 * s1 * s2,
        s2 * s2,
    ]
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

/// Single-stage combined norm + relation prover for `b = 4`.
pub struct CombinedNormRelationProver<E: FieldCore> {
    batching_coeff: E,
    w_table: WTable<E>,
    split_eq: GruenSplitEq<E>,
    alpha_compact: Vec<E>,
    m_compact: Vec<E>,
    live_x_cols: usize,
    num_u: usize,
    num_vars: usize,
    relation_claim: E,
    prev_norm_claim: E,
    pending_norm_poly: Option<UniPoly<E>>,
    rounds_completed: usize,
    norm_pair_lut: B4CompactNormPairLut,
}

impl<E: FieldCore + FromSmallInt + CanonicalField + HasUnreducedOps> CombinedNormRelationProver<E> {
    /// Build the single-stage direct prover for `b = 4`.
    ///
    /// # Panics
    ///
    /// Panics if the provided tables do not match the dimensions implied by
    /// `tau0`, `num_u`, `num_l`, and `live_x_cols`.
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

        Self {
            batching_coeff,
            w_table: WTable::Compact(w_evals_compact),
            split_eq: GruenSplitEq::new(tau0),
            alpha_compact: alpha_evals_y,
            m_compact: m_evals_x,
            live_x_cols,
            num_u,
            num_vars,
            relation_claim,
            prev_norm_claim: E::zero(),
            pending_norm_poly: None,
            rounds_completed: 0,
            norm_pair_lut: B4CompactNormPairLut::new(),
        }
    }

    /// Return the final claimed evaluation of the next witness `w`.
    ///
    /// # Panics
    ///
    /// Panics if the witness table has not been fully folded to a single full
    /// evaluation.
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
            NormRoundTerms::Full(q_coeffs) => self.split_eq.gruen_mul(&coeffs_to_poly(&q_coeffs)),
            NormRoundTerms::SkipLinear(q_except_linear) => self
                .split_eq
                .try_gruen_poly_from_coeffs_except_linear(&q_except_linear, self.prev_norm_claim)
                .expect("split-eq norm claim recovery should succeed"),
        }
    }

    fn compute_round_compact_dense_terms(&self, w_compact: &[i8]) -> (NormRoundTerms<E>, [E; 3]) {
        let (e_first, e_second) = self.split_eq.remaining_eq_tables();
        let num_first = e_first.len();
        let num_second = e_second.len();
        let current_x_width = self.current_x_width();
        let current_x_mask = (1usize << current_x_width).wrapping_sub(1);
        let alpha_compact = &self.alpha_compact;
        let m_compact = &self.m_compact;
        debug_assert_eq!(w_compact.len() / 2, num_first * num_second);

        let mut rel = [E::MulU64Accum::ZERO; 6];
        if self.can_skip_norm_linear_coeff() {
            let mut norm = [E::zero(); 4];
            for (j_high, &e_out) in e_second.iter().enumerate() {
                let base = j_high * num_first;
                let mut inner_norm = [E::zero(); 4];
                for (j_low, &e_in) in e_first.iter().enumerate() {
                    let j = base + j_low;
                    let coeffs = self
                        .norm_pair_lut
                        .coeffs(w_compact[2 * j], w_compact[2 * j + 1]);
                    inner_norm[0] += e_in * E::from_i64(coeffs[0]);
                    inner_norm[1] += e_in * E::from_i64(coeffs[2]);
                    inner_norm[2] += e_in * E::from_i64(coeffs[3]);
                    inner_norm[3] += e_in * E::from_i64(coeffs[4]);

                    let w0 = i64::from(w_compact[2 * j]);
                    let w1 = i64::from(w_compact[2 * j + 1]);
                    let dw = w1 - w0;
                    let a0 = alpha_compact[(2 * j) >> current_x_width];
                    let a1 = alpha_compact[(2 * j + 1) >> current_x_width];
                    let m0 = m_compact[(2 * j) & current_x_mask];
                    let m1 = m_compact[(2 * j + 1) & current_x_mask];
                    accumulate_relation_coeffs_signed::<E>(&mut rel, w0, dw, a0 * m0, a1 * m1);
                }
                for idx in 0..4 {
                    norm[idx] += e_out * inner_norm[idx];
                }
            }
            (NormRoundTerms::SkipLinear(norm), reduce_compact_rel(rel))
        } else {
            let mut norm = [E::zero(); 5];
            for (j_high, &e_out) in e_second.iter().enumerate() {
                let base = j_high * num_first;
                let mut inner_norm = [E::zero(); 5];
                for (j_low, &e_in) in e_first.iter().enumerate() {
                    let j = base + j_low;
                    let coeffs = self
                        .norm_pair_lut
                        .coeffs(w_compact[2 * j], w_compact[2 * j + 1]);
                    for idx in 0..5 {
                        inner_norm[idx] += e_in * E::from_i64(coeffs[idx]);
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
                for idx in 0..5 {
                    norm[idx] += e_out * inner_norm[idx];
                }
            }
            (NormRoundTerms::Full(norm), reduce_compact_rel(rel))
        }
    }

    fn compute_round_compact_prefix_x_terms(
        &self,
        w_compact: &[i8],
    ) -> (NormRoundTerms<E>, [E; 3]) {
        debug_assert!(self.rounds_completed < self.num_u);
        debug_assert_eq!(w_compact.len(), self.live_x_cols * self.alpha_compact.len());

        let (e_first, e_second) = self.split_eq.remaining_eq_tables();
        let num_first = e_first.len();
        let first_bits = num_first.trailing_zeros() as usize;
        let current_x_half = 1usize << (self.current_x_width() - 1);
        let live_pairs = self.live_x_cols.div_ceil(2);
        let alpha_compact = &self.alpha_compact;
        let m_compact = &self.m_compact;

        let mut rel = [E::MulU64Accum::ZERO; 6];
        if self.can_skip_norm_linear_coeff() {
            let mut norm = [E::zero(); 4];
            for (y, &alpha) in alpha_compact.iter().enumerate() {
                let row_start = y * self.live_x_cols;
                let row = &w_compact[row_start..row_start + self.live_x_cols];
                let j_base = y * current_x_half;

                for pair_x in 0..live_pairs {
                    let j_low = (j_base + pair_x) & (num_first - 1);
                    let j_high = (j_base + pair_x) >> first_bits;
                    let eq_rem = e_first[j_low] * e_second[j_high];
                    let left = 2 * pair_x;
                    let w0 = row[left];
                    let w1 = row.get(left + 1).copied().unwrap_or_default();
                    let coeffs = self.norm_pair_lut.coeffs(w0, w1);
                    norm[0] += eq_rem * E::from_i64(coeffs[0]);
                    norm[1] += eq_rem * E::from_i64(coeffs[2]);
                    norm[2] += eq_rem * E::from_i64(coeffs[3]);
                    norm[3] += eq_rem * E::from_i64(coeffs[4]);

                    let w0 = i64::from(w0);
                    let w1 = i64::from(w1);
                    let dw = w1 - w0;
                    accumulate_relation_coeffs_signed::<E>(
                        &mut rel,
                        w0,
                        dw,
                        alpha * m_compact[left],
                        alpha * m_compact[left + 1],
                    );
                }
            }
            (NormRoundTerms::SkipLinear(norm), reduce_compact_rel(rel))
        } else {
            let mut norm = [E::zero(); 5];
            for (y, &alpha) in alpha_compact.iter().enumerate() {
                let row_start = y * self.live_x_cols;
                let row = &w_compact[row_start..row_start + self.live_x_cols];
                let j_base = y * current_x_half;

                for pair_x in 0..live_pairs {
                    let j_low = (j_base + pair_x) & (num_first - 1);
                    let j_high = (j_base + pair_x) >> first_bits;
                    let eq_rem = e_first[j_low] * e_second[j_high];
                    let left = 2 * pair_x;
                    let w0 = row[left];
                    let w1 = row.get(left + 1).copied().unwrap_or_default();
                    let coeffs = self.norm_pair_lut.coeffs(w0, w1);
                    for idx in 0..5 {
                        norm[idx] += eq_rem * E::from_i64(coeffs[idx]);
                    }

                    let w0 = i64::from(w0);
                    let w1 = i64::from(w1);
                    let dw = w1 - w0;
                    accumulate_relation_coeffs_signed::<E>(
                        &mut rel,
                        w0,
                        dw,
                        alpha * m_compact[left],
                        alpha * m_compact[left + 1],
                    );
                }
            }
            (NormRoundTerms::Full(norm), reduce_compact_rel(rel))
        }
    }

    fn compute_round_full_dense_terms(&self, w_full: &[E]) -> (NormRoundTerms<E>, [E; 3]) {
        let (e_first, e_second) = self.split_eq.remaining_eq_tables();
        let num_first = e_first.len();
        let num_second = e_second.len();
        let current_x_width = self.current_x_width();
        let current_x_mask = (1usize << current_x_width).wrapping_sub(1);
        let alpha_compact = &self.alpha_compact;
        let m_compact = &self.m_compact;
        debug_assert_eq!(w_full.len() / 2, num_first * num_second);

        if self.can_skip_norm_linear_coeff() {
            let mut norm = [E::zero(); 4];
            let mut rel = [E::zero(); 3];
            for (j_high, &e_out) in e_second.iter().enumerate() {
                let base = j_high * num_first;
                let mut inner_norm = [E::zero(); 4];
                for (j_low, &e_in) in e_first.iter().enumerate() {
                    let j = base + j_low;
                    let coeffs = b4_direct_norm_coeffs(w_full[2 * j], w_full[2 * j + 1]);
                    inner_norm[0] += e_in * coeffs[0];
                    inner_norm[1] += e_in * coeffs[2];
                    inner_norm[2] += e_in * coeffs[3];
                    inner_norm[3] += e_in * coeffs[4];

                    let w0 = w_full[2 * j];
                    let w1 = w_full[2 * j + 1];
                    let dw = w1 - w0;
                    let a0 = alpha_compact[(2 * j) >> current_x_width];
                    let a1 = alpha_compact[(2 * j + 1) >> current_x_width];
                    let m0 = m_compact[(2 * j) & current_x_mask];
                    let m1 = m_compact[(2 * j + 1) & current_x_mask];
                    accumulate_relation_coeffs(&mut rel, w0, dw, a0 * m0, a1 * m1);
                }
                for idx in 0..4 {
                    norm[idx] += e_out * inner_norm[idx];
                }
            }
            (NormRoundTerms::SkipLinear(norm), rel)
        } else {
            let mut norm = [E::zero(); 5];
            let mut rel = [E::zero(); 3];
            for (j_high, &e_out) in e_second.iter().enumerate() {
                let base = j_high * num_first;
                let mut inner_norm = [E::zero(); 5];
                for (j_low, &e_in) in e_first.iter().enumerate() {
                    let j = base + j_low;
                    let coeffs = b4_direct_norm_coeffs(w_full[2 * j], w_full[2 * j + 1]);
                    for idx in 0..5 {
                        inner_norm[idx] += e_in * coeffs[idx];
                    }

                    let w0 = w_full[2 * j];
                    let w1 = w_full[2 * j + 1];
                    let dw = w1 - w0;
                    let a0 = alpha_compact[(2 * j) >> current_x_width];
                    let a1 = alpha_compact[(2 * j + 1) >> current_x_width];
                    let m0 = m_compact[(2 * j) & current_x_mask];
                    let m1 = m_compact[(2 * j + 1) & current_x_mask];
                    accumulate_relation_coeffs(&mut rel, w0, dw, a0 * m0, a1 * m1);
                }
                for idx in 0..5 {
                    norm[idx] += e_out * inner_norm[idx];
                }
            }
            (NormRoundTerms::Full(norm), rel)
        }
    }

    fn compute_round_full_prefix_x_terms(&self, w_full: &[E]) -> (NormRoundTerms<E>, [E; 3]) {
        debug_assert!(self.rounds_completed < self.num_u);
        debug_assert_eq!(w_full.len(), self.live_x_cols * self.alpha_compact.len());

        let (e_first, e_second) = self.split_eq.remaining_eq_tables();
        let num_first = e_first.len();
        let first_bits = num_first.trailing_zeros() as usize;
        let current_x_half = 1usize << (self.current_x_width() - 1);
        let live_pairs = self.live_x_cols.div_ceil(2);
        let alpha_compact = &self.alpha_compact;
        let m_compact = &self.m_compact;

        if self.can_skip_norm_linear_coeff() {
            let mut norm = [E::zero(); 4];
            let mut rel = [E::zero(); 3];
            for (y, &alpha) in alpha_compact.iter().enumerate() {
                let row_start = y * self.live_x_cols;
                let row = &w_full[row_start..row_start + self.live_x_cols];
                let j_base = y * current_x_half;

                for pair_x in 0..live_pairs {
                    let j_low = (j_base + pair_x) & (num_first - 1);
                    let j_high = (j_base + pair_x) >> first_bits;
                    let eq_rem = e_first[j_low] * e_second[j_high];
                    let left = 2 * pair_x;
                    let w0 = row[left];
                    let w1 = row.get(left + 1).copied().unwrap_or_else(E::zero);
                    let coeffs = b4_direct_norm_coeffs(w0, w1);
                    norm[0] += eq_rem * coeffs[0];
                    norm[1] += eq_rem * coeffs[2];
                    norm[2] += eq_rem * coeffs[3];
                    norm[3] += eq_rem * coeffs[4];

                    let dw = w1 - w0;
                    accumulate_relation_coeffs(
                        &mut rel,
                        w0,
                        dw,
                        alpha * m_compact[left],
                        alpha * m_compact[left + 1],
                    );
                }
            }
            (NormRoundTerms::SkipLinear(norm), rel)
        } else {
            let mut norm = [E::zero(); 5];
            let mut rel = [E::zero(); 3];
            for (y, &alpha) in alpha_compact.iter().enumerate() {
                let row_start = y * self.live_x_cols;
                let row = &w_full[row_start..row_start + self.live_x_cols];
                let j_base = y * current_x_half;

                for pair_x in 0..live_pairs {
                    let j_low = (j_base + pair_x) & (num_first - 1);
                    let j_high = (j_base + pair_x) >> first_bits;
                    let eq_rem = e_first[j_low] * e_second[j_high];
                    let left = 2 * pair_x;
                    let w0 = row[left];
                    let w1 = row.get(left + 1).copied().unwrap_or_else(E::zero);
                    let coeffs = b4_direct_norm_coeffs(w0, w1);
                    for idx in 0..5 {
                        norm[idx] += eq_rem * coeffs[idx];
                    }

                    let dw = w1 - w0;
                    accumulate_relation_coeffs(
                        &mut rel,
                        w0,
                        dw,
                        alpha * m_compact[left],
                        alpha * m_compact[left + 1],
                    );
                }
            }
            (NormRoundTerms::Full(norm), rel)
        }
    }

    fn compute_current_round_poly_from_state(&mut self) -> UniPoly<E> {
        match &self.w_table {
            WTable::Compact(w_compact) => {
                let (norm_terms, rel_coeffs) = if self.use_prefix_x_round() {
                    self.compute_round_compact_prefix_x_terms(w_compact)
                } else {
                    self.compute_round_compact_dense_terms(w_compact)
                };
                let norm_poly = self.norm_poly_from_terms(norm_terms);
                self.pending_norm_poly = Some(norm_poly.clone());
                let relation_poly = coeffs_to_poly(&rel_coeffs);
                scale_and_add_polys(self.batching_coeff, &norm_poly, &relation_poly)
            }
            WTable::Full(w_full) => {
                let (norm_terms, rel_coeffs) = if self.use_prefix_x_round() {
                    self.compute_round_full_prefix_x_terms(w_full)
                } else {
                    self.compute_round_full_dense_terms(w_full)
                };
                let norm_poly = self.norm_poly_from_terms(norm_terms);
                self.pending_norm_poly = Some(norm_poly.clone());
                let relation_poly = coeffs_to_poly(&rel_coeffs);
                scale_and_add_polys(self.batching_coeff, &norm_poly, &relation_poly)
            }
        }
    }

    #[inline]
    fn build_compact_w_fold_lut(r: E) -> CompactPairFoldLut<E> {
        CompactPairFoldLut::from_contiguous_range(-2i16, 1i16, r)
    }

    fn fold_w_compact_to_full(w_compact: &[i8], fold_lut: &CompactPairFoldLut<E>) -> Vec<E> {
        (0..w_compact.len() / 2)
            .map(|j| fold_lut.fold(i16::from(w_compact[2 * j]), i16::from(w_compact[2 * j + 1])))
            .collect()
    }

    fn fold_w_compact_prefix_x(
        w_compact: &[i8],
        live_x_cols: usize,
        y_len: usize,
        fold_lut: &CompactPairFoldLut<E>,
    ) -> Vec<E> {
        let next_live_x_cols = live_x_cols.div_ceil(2);
        let mut out = vec![E::zero(); y_len * next_live_x_cols];
        for (y, row_out) in out.chunks_mut(next_live_x_cols).enumerate() {
            let row = &w_compact[y * live_x_cols..(y + 1) * live_x_cols];
            for (pair_x, dst) in row_out.iter_mut().enumerate() {
                let left = 2 * pair_x;
                let w1 = row.get(left + 1).copied().unwrap_or_default();
                *dst = fold_lut.fold(i16::from(row[left]), i16::from(w1));
            }
        }
        out
    }

    fn fold_full_prefix_x(full: &[E], live_x_cols: usize, y_len: usize, r: E) -> Vec<E> {
        let next_live_x_cols = live_x_cols.div_ceil(2);
        let mut out = vec![E::zero(); y_len * next_live_x_cols];
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
        (0..(m_compact.len() / 2))
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
        5
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
        let y_len = match &self.w_table {
            WTable::Compact(w_compact) => w_compact.len() / self.live_x_cols,
            WTable::Full(w_full) => w_full.len() / self.live_x_cols,
        };

        self.w_table = match mem::replace(&mut self.w_table, WTable::Full(Vec::new())) {
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
pub struct CombinedNormRelationVerifier<F: FieldCore> {
    batching_coeff: F,
    tau0: Vec<F>,
    relation_claim: F,
    witness_oracle: CombinedWitnessOracle<F>,
    alpha_evals_y: Vec<F>,
    m_evals_x: Vec<F>,
    num_u: usize,
}

impl<F: FieldCore> CombinedNormRelationVerifier<F> {
    fn witness_eval(&self, challenges: &[F]) -> Result<F, HachiError> {
        match &self.witness_oracle {
            CombinedWitnessOracle::Full(w_evals) => multilinear_eval(w_evals, challenges),
            CombinedWitnessOracle::ClaimedEval(w_eval) => Ok(*w_eval),
        }
    }
}

impl<F: FieldCore + FromSmallInt> CombinedNormRelationVerifier<F> {
    /// Build the combined verifier when the verifier has the full witness.
    pub fn new_with_full_witness(
        batching_coeff: F,
        relation_claim: F,
        w_evals: Vec<F>,
        tau0: Vec<F>,
        alpha_evals_y: Vec<F>,
        m_evals_x: Vec<F>,
        num_u: usize,
    ) -> Self {
        Self {
            batching_coeff,
            tau0,
            relation_claim,
            witness_oracle: CombinedWitnessOracle::Full(w_evals),
            alpha_evals_y,
            m_evals_x,
            num_u,
        }
    }

    /// Build the combined verifier when only the final witness evaluation is available.
    pub fn new_with_claimed_w_eval(
        batching_coeff: F,
        relation_claim: F,
        w_eval: F,
        tau0: Vec<F>,
        alpha_evals_y: Vec<F>,
        m_evals_x: Vec<F>,
        num_u: usize,
    ) -> Self {
        Self {
            batching_coeff,
            tau0,
            relation_claim,
            witness_oracle: CombinedWitnessOracle::ClaimedEval(w_eval),
            alpha_evals_y,
            m_evals_x,
            num_u,
        }
    }
}

impl<F: FieldCore + FromSmallInt> SumcheckInstanceVerifier<F> for CombinedNormRelationVerifier<F> {
    fn num_rounds(&self) -> usize {
        self.tau0.len()
    }

    fn degree_bound(&self) -> usize {
        5
    }

    fn input_claim(&self) -> F {
        self.relation_claim
    }

    fn expected_output_claim(&self, challenges: &[F]) -> Result<F, HachiError> {
        let w_eval = self.witness_eval(challenges)?;
        let eq_val = EqPolynomial::mle(&self.tau0, challenges);
        let (x_challenges, y_challenges) = challenges.split_at(self.num_u);
        let alpha_val = multilinear_eval(&self.alpha_evals_y, y_challenges)?;
        let m_val = multilinear_eval(&self.m_evals_x, x_challenges)?;
        let relation_oracle = w_eval * alpha_val * m_val;
        Ok(self.batching_coeff * eq_val * direct_b4_range_check_from_w(w_eval) + relation_oracle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::Prime128M8M4M1M0;
    use crate::protocol::sumcheck::{prove_sumcheck, CompressedUniPoly};
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
            for (x, &m_eval_x) in m_evals_x.iter().enumerate().take(x_len) {
                let w = row.get(x).copied().unwrap_or_default();
                acc += F::from_i64(w as i64) * alpha_evals_y[y] * m_eval_x;
            }
        }
        acc
    }

    fn padded_w_evals(w_compact: &[i8], live_x_cols: usize, num_u: usize, num_l: usize) -> Vec<F> {
        let x_len = 1usize << num_u;
        let y_len = 1usize << num_l;
        let mut out = vec![F::zero(); x_len * y_len];
        for y in 0..y_len {
            let row = &w_compact[y * live_x_cols..(y + 1) * live_x_cols];
            for x in 0..live_x_cols {
                out[y * x_len + x] = F::from_i64(i64::from(row[x]));
            }
        }
        out
    }

    fn new_transcript() -> Blake2bTranscript<F> {
        <Blake2bTranscript<F> as Transcript<F>>::new(tr_labels::DOMAIN_HACHI_PROTOCOL)
    }

    fn sample_round(tr: &mut Blake2bTranscript<F>) -> F {
        tr.challenge_scalar(tr_labels::CHALLENGE_SUMCHECK_ROUND)
    }

    fn split_eq_line_value(split_eq: &GruenSplitEq<F>, t: F) -> F {
        let tau = split_eq.current_tau();
        let scalar = split_eq.current_scalar();
        scalar * (tau * t + (F::one() - tau) * (F::one() - t))
    }

    fn reference_round_eval_from_compact_dense(
        prover: &CombinedNormRelationProver<F>,
        w_compact: &[i8],
        t: F,
    ) -> F {
        let (e_first, e_second) = prover.split_eq.remaining_eq_tables();
        let num_first = e_first.len();
        let num_second = e_second.len();
        let current_x_width = prover.current_x_width();
        let current_x_mask = (1usize << current_x_width).wrapping_sub(1);
        let mut norm_inner = F::zero();
        let mut relation = F::zero();
        for (j_high, &eq_second) in e_second.iter().enumerate().take(num_second) {
            let base = j_high * num_first;
            for (j_low, _) in e_first.iter().enumerate() {
                let j = base + j_low;
                let w0 = F::from_i64(i64::from(w_compact[2 * j]));
                let w1 = F::from_i64(i64::from(w_compact[2 * j + 1]));
                let w_t = w0 + t * (w1 - w0);
                norm_inner += e_first[j_low] * eq_second * direct_b4_range_check_from_w(w_t);

                let p0 = prover.alpha_compact[(2 * j) >> current_x_width]
                    * prover.m_compact[(2 * j) & current_x_mask];
                let p1 = prover.alpha_compact[(2 * j + 1) >> current_x_width]
                    * prover.m_compact[(2 * j + 1) & current_x_mask];
                let p_t = p0 + t * (p1 - p0);
                relation += w_t * p_t;
            }
        }
        prover.batching_coeff * split_eq_line_value(&prover.split_eq, t) * norm_inner + relation
    }

    fn reference_round_eval_from_compact_prefix(
        prover: &CombinedNormRelationProver<F>,
        w_compact: &[i8],
        t: F,
    ) -> F {
        let (e_first, e_second) = prover.split_eq.remaining_eq_tables();
        let num_first = e_first.len();
        let first_bits = num_first.trailing_zeros() as usize;
        let current_x_half = 1usize << (prover.current_x_width() - 1);
        let live_pairs = prover.live_x_cols.div_ceil(2);
        let mut norm_inner = F::zero();
        let mut relation = F::zero();

        for y in 0..prover.alpha_compact.len() {
            let row_start = y * prover.live_x_cols;
            let row = &w_compact[row_start..row_start + prover.live_x_cols];
            let alpha = prover.alpha_compact[y];
            let j_base = y * current_x_half;
            for pair_x in 0..live_pairs {
                let j_low = (j_base + pair_x) & (num_first - 1);
                let j_high = (j_base + pair_x) >> first_bits;
                let eq_rem = e_first[j_low] * e_second[j_high];
                let left = 2 * pair_x;
                let w0 = F::from_i64(i64::from(row[left]));
                let w1 = F::from_i64(i64::from(row.get(left + 1).copied().unwrap_or_default()));
                let w_t = w0 + t * (w1 - w0);
                norm_inner += eq_rem * direct_b4_range_check_from_w(w_t);

                let p0 = alpha * prover.m_compact[left];
                let p1 = alpha * prover.m_compact[left + 1];
                let p_t = p0 + t * (p1 - p0);
                relation += w_t * p_t;
            }
        }

        prover.batching_coeff * split_eq_line_value(&prover.split_eq, t) * norm_inner + relation
    }

    fn reference_round_eval_from_full_dense(
        prover: &CombinedNormRelationProver<F>,
        w_full: &[F],
        t: F,
    ) -> F {
        let (e_first, e_second) = prover.split_eq.remaining_eq_tables();
        let num_first = e_first.len();
        let num_second = e_second.len();
        let current_x_width = prover.current_x_width();
        let current_x_mask = (1usize << current_x_width).wrapping_sub(1);
        let mut norm_inner = F::zero();
        let mut relation = F::zero();
        for (j_high, &eq_second) in e_second.iter().enumerate().take(num_second) {
            let base = j_high * num_first;
            for (j_low, _) in e_first.iter().enumerate() {
                let j = base + j_low;
                let w0 = w_full[2 * j];
                let w1 = w_full[2 * j + 1];
                let w_t = w0 + t * (w1 - w0);
                norm_inner += e_first[j_low] * eq_second * direct_b4_range_check_from_w(w_t);

                let p0 = prover.alpha_compact[(2 * j) >> current_x_width]
                    * prover.m_compact[(2 * j) & current_x_mask];
                let p1 = prover.alpha_compact[(2 * j + 1) >> current_x_width]
                    * prover.m_compact[(2 * j + 1) & current_x_mask];
                let p_t = p0 + t * (p1 - p0);
                relation += w_t * p_t;
            }
        }
        prover.batching_coeff * split_eq_line_value(&prover.split_eq, t) * norm_inner + relation
    }

    fn reference_round_eval_from_full_prefix(
        prover: &CombinedNormRelationProver<F>,
        w_full: &[F],
        t: F,
    ) -> F {
        let (e_first, e_second) = prover.split_eq.remaining_eq_tables();
        let num_first = e_first.len();
        let first_bits = num_first.trailing_zeros() as usize;
        let current_x_half = 1usize << (prover.current_x_width() - 1);
        let live_pairs = prover.live_x_cols.div_ceil(2);
        let mut norm_inner = F::zero();
        let mut relation = F::zero();

        for y in 0..prover.alpha_compact.len() {
            let row_start = y * prover.live_x_cols;
            let row = &w_full[row_start..row_start + prover.live_x_cols];
            let alpha = prover.alpha_compact[y];
            let j_base = y * current_x_half;
            for pair_x in 0..live_pairs {
                let j_low = (j_base + pair_x) & (num_first - 1);
                let j_high = (j_base + pair_x) >> first_bits;
                let eq_rem = e_first[j_low] * e_second[j_high];
                let left = 2 * pair_x;
                let w0 = row[left];
                let w1 = row.get(left + 1).copied().unwrap_or_else(F::zero);
                let w_t = w0 + t * (w1 - w0);
                norm_inner += eq_rem * direct_b4_range_check_from_w(w_t);

                let p0 = alpha * prover.m_compact[left];
                let p1 = alpha * prover.m_compact[left + 1];
                let p_t = p0 + t * (p1 - p0);
                relation += w_t * p_t;
            }
        }

        prover.batching_coeff * split_eq_line_value(&prover.split_eq, t) * norm_inner + relation
    }

    fn reference_round_poly_from_state(prover: &CombinedNormRelationProver<F>) -> UniPoly<F> {
        let evals: Vec<F> = (0..=5)
            .map(|i| {
                let t = F::from_u64(i as u64);
                match &prover.w_table {
                    WTable::Compact(w_compact) => {
                        if prover.use_prefix_x_round() {
                            reference_round_eval_from_compact_prefix(prover, w_compact, t)
                        } else {
                            reference_round_eval_from_compact_dense(prover, w_compact, t)
                        }
                    }
                    WTable::Full(w_full) => {
                        if prover.use_prefix_x_round() {
                            reference_round_eval_from_full_prefix(prover, w_full, t)
                        } else {
                            reference_round_eval_from_full_dense(prover, w_full, t)
                        }
                    }
                }
            })
            .collect();
        UniPoly::from_evals(&evals)
    }

    #[test]
    fn direct_w_semantics_differs_from_pointwise_s_mle_off_cube() {
        let w_evals = [F::from_i64(-2), F::from_i64(-1)];
        let s_evals = [F::from_u64(2), F::zero()];
        let r = F::from_u64(5);
        let w_eval = multilinear_eval(&w_evals, &[r]).unwrap();
        let s_eval = multilinear_eval(&s_evals, &[r]).unwrap();
        assert_ne!(s_eval, w_eval * (w_eval + F::one()));
    }

    #[test]
    fn combined_rounds_match_direct_reference_dense() {
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
        let mut prover = CombinedNormRelationProver::new(
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

        let challenges = [
            F::from_u64(23),
            F::from_u64(29),
            F::from_u64(31),
            F::from_u64(37),
            F::from_u64(41),
        ];
        let mut claim = relation_claim;
        for (round, &challenge) in challenges.iter().enumerate() {
            let actual = prover.compute_round_univariate(round, claim);
            let expected = reference_round_poly_from_state(&prover);
            assert_eq!(actual, expected, "round {round}");
            claim = actual.evaluate(&challenge);
            prover.ingest_challenge(round, challenge);
        }

        let padded_w = padded_w_evals(&w_compact, live_x_cols, num_u, num_l);
        assert_eq!(
            prover.final_w_eval(),
            multilinear_eval(&padded_w, &challenges).unwrap()
        );
    }

    #[test]
    fn combined_rounds_match_direct_reference_prefix_x() {
        let num_u = 3usize;
        let num_l = 2usize;
        let live_x_cols = 5usize;
        let n = live_x_cols << num_l;
        let w_compact: Vec<i8> = (0..n).map(|i| ((i * 7 + 1) % 4) as i8 - 2).collect();
        let tau0: Vec<F> = (0..(num_u + num_l))
            .map(|i| F::from_u64((i as u64) + 43))
            .collect();
        let alpha_evals_y: Vec<F> = (0..(1usize << num_l))
            .map(|i| F::from_u64((5 * i as u64) + 47))
            .collect();
        let m_evals_x: Vec<F> = (0..(1usize << num_u))
            .map(|i| F::from_u64((11 * i as u64) + 53))
            .collect();
        let batching_coeff = F::from_u64(59);
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
            w_compact.clone(),
            &tau0,
            alpha_evals_y.clone(),
            m_evals_x.clone(),
            live_x_cols,
            num_u,
            num_l,
            relation_claim,
        );

        let challenges = [
            F::from_u64(61),
            F::from_u64(67),
            F::from_u64(71),
            F::from_u64(73),
            F::from_u64(79),
        ];
        let mut claim = relation_claim;
        for (round, &challenge) in challenges.iter().enumerate() {
            let actual = prover.compute_round_univariate(round, claim);
            let expected = reference_round_poly_from_state(&prover);
            assert_eq!(actual, expected, "round {round}");
            claim = actual.evaluate(&challenge);
            prover.ingest_challenge(round, challenge);
        }

        let padded_w = padded_w_evals(&w_compact, live_x_cols, num_u, num_l);
        assert_eq!(
            prover.final_w_eval(),
            multilinear_eval(&padded_w, &challenges).unwrap()
        );
    }

    #[test]
    fn combined_sumcheck_final_claim_matches_direct_expected_oracle() {
        let num_u = 3usize;
        let num_l = 2usize;
        let live_x_cols = 1usize << num_u;
        let n = live_x_cols << num_l;
        let w_compact: Vec<i8> = (0..n).map(|i| ((i * 9 + 1) % 4) as i8 - 2).collect();
        let tau0: Vec<F> = (0..(num_u + num_l))
            .map(|i| F::from_u64((i as u64) + 83))
            .collect();
        let alpha_evals_y: Vec<F> = (0..(1usize << num_l))
            .map(|i| F::from_u64((13 * i as u64) + 89))
            .collect();
        let m_evals_x: Vec<F> = (0..(1usize << num_u))
            .map(|i| F::from_u64((17 * i as u64) + 97))
            .collect();
        let batching_coeff = F::from_u64(101);
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
        let w_eval = prover.final_w_eval();
        let eq_val = EqPolynomial::mle(&tau0, &challenges);
        let (x_challenges, y_challenges) = challenges.split_at(num_u);
        let alpha_val = multilinear_eval(&alpha_evals_y, y_challenges).unwrap();
        let m_val = multilinear_eval(&m_evals_x, x_challenges).unwrap();
        let expected = batching_coeff * eq_val * direct_b4_range_check_from_w(w_eval)
            + w_eval * alpha_val * m_val;
        assert_eq!(final_claim, expected);
    }

    #[test]
    fn combined_verifier_uses_direct_w_only_oracle() {
        let tau0 = vec![F::from_u64(3), F::from_u64(5)];
        let alpha_evals_y = vec![F::from_u64(7), F::from_u64(11)];
        let m_evals_x = vec![F::from_u64(13), F::from_u64(17)];
        let verifier = CombinedNormRelationVerifier::new_with_claimed_w_eval(
            F::from_u64(19),
            F::from_u64(23),
            F::from_u64(29),
            tau0.clone(),
            alpha_evals_y.clone(),
            m_evals_x.clone(),
            1,
        );
        let challenges = [F::from_u64(31), F::from_u64(37)];
        let w_eval = F::from_u64(29);
        let eq_val = EqPolynomial::mle(&tau0, &challenges);
        let alpha_val = multilinear_eval(&alpha_evals_y, &challenges[1..]).unwrap();
        let m_val = multilinear_eval(&m_evals_x, &challenges[..1]).unwrap();
        let expected = F::from_u64(19) * eq_val * direct_b4_range_check_from_w(w_eval)
            + w_eval * alpha_val * m_val;
        assert_eq!(
            verifier.expected_output_claim(&challenges).unwrap(),
            expected
        );
        assert_eq!(verifier.input_claim(), F::from_u64(23));
    }

    #[test]
    fn combined_round_degree_bound_is_five() {
        let poly = UniPoly::from_evals(&[
            F::from_u64(1),
            F::from_u64(4),
            F::from_u64(9),
            F::from_u64(16),
            F::from_u64(25),
            F::from_u64(36),
        ]);
        let compressed: CompressedUniPoly<F> = poly.compress();
        assert!(compressed.degree() <= 5);
    }
}
