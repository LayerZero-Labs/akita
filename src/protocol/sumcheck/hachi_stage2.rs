//! Stage-2 fused sumcheck prover/verifier for the Hachi PCS.
//!
//! This stage combines the virtual claim over `w(w+1)` with the relation claim
//! over `w * alpha * m`. The inner scan is fused around a shared local basis so
//! the same `w0` / `dw` work is reused for both halves.

use super::eq_poly::EqPolynomial;
use super::split_eq::GruenSplitEq;
use super::{fold_evals_in_place, multilinear_eval, trim_trailing_zeros};
use super::{SumcheckInstanceProver, SumcheckInstanceVerifier, UniPoly};
use crate::algebra::fields::HasUnreducedOps;
use crate::algebra::CyclotomicRing;
use crate::error::HachiError;
#[cfg(feature = "parallel")]
use crate::parallel::*;
use crate::protocol::ring_switch::eval_ring_at;
use crate::{cfg_fold_reduce, cfg_into_iter};
use crate::{AdditiveGroup, CanonicalField, FieldCore, FromSmallInt};
use std::marker::PhantomData;
use std::mem;
use std::time::Instant;

enum WTable<E: FieldCore> {
    Compact(Vec<i8>),
    Full(Vec<E>),
}

#[derive(Clone, Copy)]
struct FieldLocalBasis<E: FieldCore> {
    eq_rem: E,
    w0: E,
    dw: E,
    p0: E,
    dp: E,
}

#[derive(Clone, Copy)]
struct CompactLocalBasis<E: FieldCore> {
    eq_rem: E,
    w0: i32,
    dw: i32,
    p0: E,
    dp: E,
}

type CompactVirtAccum<E> = [<E as HasUnreducedOps>::MulU64Accum; 4];
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
fn reduce_signed_accum<E: FieldCore + HasUnreducedOps>(
    pos: E::MulU64Accum,
    neg: E::MulU64Accum,
) -> E {
    E::reduce_mul_u64_accum(pos) - E::reduce_mul_u64_accum(neg)
}

#[inline]
fn absorb_field_basis<E: FieldCore>(
    virt: &mut [E; 3],
    rel: &mut [E; 3],
    basis: FieldLocalBasis<E>,
) {
    let two_w0_plus_one = basis.w0 + basis.w0 + E::one();
    virt[0] += basis.eq_rem * (basis.w0 * (basis.w0 + E::one()));
    virt[1] += basis.eq_rem * (basis.dw * two_w0_plus_one);
    virt[2] += basis.eq_rem * (basis.dw * basis.dw);

    rel[0] += basis.w0 * basis.p0;
    rel[1] += basis.w0 * basis.dp + basis.dw * basis.p0;
    rel[2] += basis.dw * basis.dp;
}

#[inline]
fn absorb_compact_basis<E: FieldCore + HasUnreducedOps>(
    virt: &mut CompactVirtAccum<E>,
    rel: &mut CompactRelAccum<E>,
    basis: CompactLocalBasis<E>,
) {
    let w0 = basis.w0 as i64;
    let dw = basis.dw as i64;

    let q0 = w0 * (w0 + 1);
    if q0 != 0 {
        virt[0] += basis.eq_rem.mul_u64_unreduced(q0 as u64);
    }
    let q1 = dw * (2 * w0 + 1);
    accum_small_signed::<E>(virt, 1, basis.eq_rem, q1);
    let q2 = dw * dw;
    if q2 != 0 {
        virt[3] += basis.eq_rem.mul_u64_unreduced(q2 as u64);
    }

    accum_small_signed::<E>(rel, 0, basis.p0, w0);
    accum_small_signed::<E>(rel, 2, basis.dp, w0);
    accum_small_signed::<E>(rel, 2, basis.p0, dw);
    accum_small_signed::<E>(rel, 4, basis.dp, dw);
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
fn reduce_compact_rel<E: FieldCore + HasUnreducedOps>(rel: CompactRelAccum<E>) -> [E; 3] {
    [
        reduce_signed_accum::<E>(rel[0], rel[1]),
        reduce_signed_accum::<E>(rel[2], rel[3]),
        reduce_signed_accum::<E>(rel[4], rel[5]),
    ]
}

#[inline]
pub(crate) fn relation_claim_from_rows<F: FieldCore + CanonicalField, const D: usize>(
    tau1: &[F],
    alpha: F,
    v: &[CyclotomicRing<F, D>],
    u: &[CyclotomicRing<F, D>],
    y_ring: &CyclotomicRing<F, D>,
) -> F {
    let eq_tau1 = EqPolynomial::evals(tau1);
    let mut acc = F::zero();
    let mut row_idx = 0usize;

    for r in v {
        if row_idx >= eq_tau1.len() {
            return acc;
        }
        acc += eq_tau1[row_idx] * eval_ring_at(r, &alpha);
        row_idx += 1;
    }
    for r in u {
        if row_idx >= eq_tau1.len() {
            return acc;
        }
        acc += eq_tau1[row_idx] * eval_ring_at(r, &alpha);
        row_idx += 1;
    }
    if row_idx < eq_tau1.len() {
        acc += eq_tau1[row_idx] * eval_ring_at(y_ring, &alpha);
    }
    acc
}

/// Stage-2 fused virtual-claim + relation sumcheck prover.
///
/// Holds a single `w_table` shared by both halves of stage 2, weighted by
/// `batching_coeff`. The round polynomial is:
/// `batching_coeff * virtual_round(t) + relation_round(t)`.
pub struct HachiStage2Prover<E: FieldCore> {
    w_table: WTable<E>,
    batching_coeff: E,
    s_claim: E,
    split_eq: GruenSplitEq<E>,

    alpha_compact: Vec<E>,
    m_compact: Vec<E>,
    live_x_cols: usize,
    num_u: usize,
    num_vars: usize,
    relation_claim: E,

    scan_time_total: f64,
    fold_time_total: f64,
    rounds_completed: usize,
}

impl<E: FieldCore + FromSmallInt + CanonicalField + HasUnreducedOps> HachiStage2Prover<E> {
    /// Create a fused stage-2 virtual-claim + relation sumcheck prover.
    #[allow(clippy::too_many_arguments)]
    #[tracing::instrument(skip_all, name = "HachiStage2Prover::new")]
    pub fn new(
        batching_coeff: E,
        w_evals_compact: Vec<i8>,
        r_stage1: &[E],
        s_claim: E,
        alpha_evals_y: Vec<E>,
        m_evals_x: Vec<E>,
        live_x_cols: usize,
        num_u: usize,
        num_l: usize,
        relation_claim: E,
    ) -> Self {
        let num_vars = num_u + num_l;
        assert!(live_x_cols >= 1, "live_x_cols must be at least 1");
        assert!(
            live_x_cols <= (1usize << num_u),
            "live_x_cols exceeds x width"
        );
        let y_len = 1usize << num_l;
        assert_eq!(w_evals_compact.len(), live_x_cols * y_len);
        assert_eq!(r_stage1.len(), num_vars);
        assert_eq!(alpha_evals_y.len(), y_len);
        assert_eq!(m_evals_x.len(), 1 << num_u);

        Self {
            w_table: WTable::Compact(w_evals_compact),
            batching_coeff,
            s_claim,
            split_eq: GruenSplitEq::new(r_stage1),
            alpha_compact: alpha_evals_y,
            m_compact: m_evals_x[..live_x_cols].to_vec(),
            live_x_cols,
            num_u,
            num_vars,
            relation_claim,
            scan_time_total: 0.0,
            fold_time_total: 0.0,
            rounds_completed: 0,
        }
    }

    /// Return the fully folded witness evaluation after the final round.
    ///
    /// # Panics
    ///
    /// Panics if called before the witness table has been fully folded to a
    /// single field element.
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
    fn polys_from_terms(
        &self,
        virt_q_coeffs: [E; 3],
        rel_coeffs: [E; 3],
    ) -> (UniPoly<E>, UniPoly<E>) {
        let virt_poly = self.split_eq.gruen_mul(&coeffs_to_poly(virt_q_coeffs));
        let rel_poly = coeffs_to_poly(rel_coeffs);
        (virt_poly, rel_poly)
    }

    #[inline]
    fn combine_polys(&self, virt_poly: &UniPoly<E>, relation_poly: &UniPoly<E>) -> UniPoly<E> {
        let max_len = virt_poly.coeffs.len().max(relation_poly.coeffs.len());
        let mut combined = vec![E::zero(); max_len];
        for (i, c) in virt_poly.coeffs.iter().enumerate() {
            combined[i] += self.batching_coeff * *c;
        }
        for (i, c) in relation_poly.coeffs.iter().enumerate() {
            combined[i] += *c;
        }
        UniPoly::from_coeffs(combined)
    }

    #[inline]
    fn combine_terms(&self, virt_q_coeffs: [E; 3], rel_coeffs: [E; 3]) -> UniPoly<E> {
        let (virt_poly, relation_poly) = self.polys_from_terms(virt_q_coeffs, rel_coeffs);
        self.combine_polys(&virt_poly, &relation_poly)
    }

    #[tracing::instrument(
        skip_all,
        name = "HachiStage2Prover::compute_round_compact_dense_terms"
    )]
    fn compute_round_compact_dense_terms(&self, w_compact: &[i8]) -> ([E; 3], [E; 3]) {
        let half = w_compact.len() / 2;
        let (e_first, e_second) = self.split_eq.remaining_eq_tables();
        let num_first = e_first.len();
        let first_bits = num_first.trailing_zeros();
        let current_x_width = self.current_x_width();
        let current_x_mask = (1usize << current_x_width).wrapping_sub(1);
        let alpha_compact = &self.alpha_compact;
        let m_compact = &self.m_compact;

        let (virt_accum, rel_accum) = cfg_fold_reduce!(
            0..half,
            || ([E::MulU64Accum::ZERO; 4], [E::MulU64Accum::ZERO; 6]),
            |(mut virt, mut rel), j| {
                let w0 = w_compact[2 * j] as i32;
                let w1 = w_compact[2 * j + 1] as i32;

                let j_low = j & (num_first - 1);
                let j_high = j >> first_bits;
                let eq_rem = e_first[j_low] * e_second[j_high];

                let a0 = alpha_compact[(2 * j) >> current_x_width];
                let a1 = alpha_compact[(2 * j + 1) >> current_x_width];
                let m0 = m_compact[(2 * j) & current_x_mask];
                let m1 = m_compact[(2 * j + 1) & current_x_mask];
                let p0 = a0 * m0;
                let p1 = a1 * m1;
                absorb_compact_basis(
                    &mut virt,
                    &mut rel,
                    CompactLocalBasis {
                        eq_rem,
                        w0,
                        dw: w1 - w0,
                        p0,
                        dp: p1 - p0,
                    },
                );

                (virt, rel)
            },
            |(mut va, mut ra), (vb, rb)| {
                for (ai, bi) in va.iter_mut().zip(vb.iter()) {
                    *ai += *bi;
                }
                for (ai, bi) in ra.iter_mut().zip(rb.iter()) {
                    *ai += *bi;
                }
                (va, ra)
            }
        );

        (
            reduce_compact_virt(virt_accum),
            reduce_compact_rel(rel_accum),
        )
    }

    #[tracing::instrument(
        skip_all,
        name = "HachiStage2Prover::compute_round_compact_prefix_x_terms"
    )]
    fn compute_round_compact_prefix_x_terms(&self, w_compact: &[i8]) -> ([E; 3], [E; 3]) {
        debug_assert!(self.rounds_completed < self.num_u);
        debug_assert_eq!(w_compact.len(), self.live_x_cols * self.alpha_compact.len());

        let (e_first, e_second) = self.split_eq.remaining_eq_tables();
        let num_first = e_first.len();
        let first_bits = num_first.trailing_zeros();
        let current_x_half = 1usize << (self.current_x_width() - 1);
        let live_pairs = self.live_x_cols.div_ceil(2);
        let alpha_compact = &self.alpha_compact;
        let m_compact = &self.m_compact;

        let (virt_accum, rel_accum) = cfg_fold_reduce!(
            0..alpha_compact.len(),
            || ([E::MulU64Accum::ZERO; 4], [E::MulU64Accum::ZERO; 6]),
            |(mut virt, mut rel), y| {
                let row_start = y * self.live_x_cols;
                let row = &w_compact[row_start..row_start + self.live_x_cols];
                let alpha = alpha_compact[y];
                for pair_x in 0..live_pairs {
                    let j = y * current_x_half + pair_x;
                    let j_low = j & (num_first - 1);
                    let j_high = j >> first_bits;
                    let eq_rem = e_first[j_low] * e_second[j_high];

                    let left = 2 * pair_x;
                    let w0 = row[left] as i32;
                    let w1 = if left + 1 < self.live_x_cols {
                        row[left + 1] as i32
                    } else {
                        0
                    };
                    let m0 = m_compact[left];
                    let m1 = if left + 1 < self.live_x_cols {
                        m_compact[left + 1]
                    } else {
                        E::zero()
                    };
                    let p0 = alpha * m0;
                    let p1 = alpha * m1;
                    absorb_compact_basis(
                        &mut virt,
                        &mut rel,
                        CompactLocalBasis {
                            eq_rem,
                            w0,
                            dw: w1 - w0,
                            p0,
                            dp: p1 - p0,
                        },
                    );
                }
                (virt, rel)
            },
            |(mut va, mut ra), (vb, rb)| {
                for (ai, bi) in va.iter_mut().zip(vb.iter()) {
                    *ai += *bi;
                }
                for (ai, bi) in ra.iter_mut().zip(rb.iter()) {
                    *ai += *bi;
                }
                (va, ra)
            }
        );

        (
            reduce_compact_virt(virt_accum),
            reduce_compact_rel(rel_accum),
        )
    }

    #[tracing::instrument(
        skip_all,
        name = "HachiStage2Prover::compute_round_full_prefix_x_terms"
    )]
    fn compute_round_full_prefix_x_terms(&self, w_full: &[E]) -> ([E; 3], [E; 3]) {
        debug_assert!(self.rounds_completed < self.num_u);
        debug_assert_eq!(w_full.len(), self.live_x_cols * self.alpha_compact.len());

        let (e_first, e_second) = self.split_eq.remaining_eq_tables();
        let num_first = e_first.len();
        let first_bits = num_first.trailing_zeros();
        let current_x_half = 1usize << (self.current_x_width() - 1);
        let live_pairs = self.live_x_cols.div_ceil(2);
        let alpha_compact = &self.alpha_compact;
        let m_compact = &self.m_compact;

        cfg_fold_reduce!(
            0..alpha_compact.len(),
            || ([E::zero(); 3], [E::zero(); 3]),
            |(mut virt_coeffs, mut rel_coeffs), y| {
                let row_start = y * self.live_x_cols;
                let row = &w_full[row_start..row_start + self.live_x_cols];
                let alpha = alpha_compact[y];
                for pair_x in 0..live_pairs {
                    let j = y * current_x_half + pair_x;
                    let j_low = j & (num_first - 1);
                    let j_high = j >> first_bits;
                    let eq_rem = e_first[j_low] * e_second[j_high];

                    let left = 2 * pair_x;
                    let w0 = row[left];
                    let w1 = if left + 1 < self.live_x_cols {
                        row[left + 1]
                    } else {
                        E::zero()
                    };
                    let m0 = m_compact[left];
                    let m1 = if left + 1 < self.live_x_cols {
                        m_compact[left + 1]
                    } else {
                        E::zero()
                    };
                    let p0 = alpha * m0;
                    let p1 = alpha * m1;
                    absorb_field_basis(
                        &mut virt_coeffs,
                        &mut rel_coeffs,
                        FieldLocalBasis {
                            eq_rem,
                            w0,
                            dw: w1 - w0,
                            p0,
                            dp: p1 - p0,
                        },
                    );
                }
                (virt_coeffs, rel_coeffs)
            },
            |(mut va, mut ra), (vb, rb)| {
                for (ai, bi) in va.iter_mut().zip(vb.iter()) {
                    *ai += *bi;
                }
                for (ai, bi) in ra.iter_mut().zip(rb.iter()) {
                    *ai += *bi;
                }
                (va, ra)
            }
        )
    }

    #[tracing::instrument(skip_all, name = "HachiStage2Prover::compute_round_full_dense_terms")]
    fn compute_round_full_dense_terms(&self, w_full: &[E]) -> ([E; 3], [E; 3]) {
        let half = w_full.len() / 2;
        let (e_first, e_second) = self.split_eq.remaining_eq_tables();
        let num_first = e_first.len();
        let first_bits = num_first.trailing_zeros();
        let current_x_width = self.current_x_width();
        let current_x_mask = (1usize << current_x_width).wrapping_sub(1);
        let alpha_compact = &self.alpha_compact;
        let m_compact = &self.m_compact;

        cfg_fold_reduce!(
            0..half,
            || ([E::zero(); 3], [E::zero(); 3]),
            |(mut virt_coeffs, mut rel_coeffs), j| {
                let w0 = w_full[2 * j];
                let w1 = w_full[2 * j + 1];

                let j_low = j & (num_first - 1);
                let j_high = j >> first_bits;
                let eq_rem = e_first[j_low] * e_second[j_high];

                let a0 = alpha_compact[(2 * j) >> current_x_width];
                let a1 = alpha_compact[(2 * j + 1) >> current_x_width];
                let m0 = m_compact[(2 * j) & current_x_mask];
                let m1 = m_compact[(2 * j + 1) & current_x_mask];
                let p0 = a0 * m0;
                let p1 = a1 * m1;
                absorb_field_basis(
                    &mut virt_coeffs,
                    &mut rel_coeffs,
                    FieldLocalBasis {
                        eq_rem,
                        w0,
                        dw: w1 - w0,
                        p0,
                        dp: p1 - p0,
                    },
                );

                (virt_coeffs, rel_coeffs)
            },
            |(mut va, mut ra), (vb, rb)| {
                for (ai, bi) in va.iter_mut().zip(vb.iter()) {
                    *ai += *bi;
                }
                for (ai, bi) in ra.iter_mut().zip(rb.iter()) {
                    *ai += *bi;
                }
                (va, ra)
            }
        )
    }

    fn compute_round_compact_prefix_x_polys(&self, w_compact: &[i8]) -> (UniPoly<E>, UniPoly<E>) {
        let (virt_q_coeffs, rel_coeffs) = self.compute_round_compact_prefix_x_terms(w_compact);
        self.polys_from_terms(virt_q_coeffs, rel_coeffs)
    }

    #[cfg(test)]
    fn compute_round_compact_dense_polys(&self, w_compact: &[i8]) -> (UniPoly<E>, UniPoly<E>) {
        let (virt_q_coeffs, rel_coeffs) = self.compute_round_compact_dense_terms(w_compact);
        self.polys_from_terms(virt_q_coeffs, rel_coeffs)
    }

    fn fold_compact_prefix_x(w_compact: &[i8], live_x_cols: usize, y_len: usize, r: E) -> Vec<E> {
        let next_live_x_cols = live_x_cols.div_ceil(2);
        let mut out = vec![E::zero(); y_len * next_live_x_cols];

        #[cfg(feature = "parallel")]
        out.par_chunks_mut(next_live_x_cols)
            .enumerate()
            .for_each(|(y, row_out)| {
                let row_start = y * live_x_cols;
                let row = &w_compact[row_start..row_start + live_x_cols];
                for (pair_x, dst) in row_out.iter_mut().enumerate() {
                    let left = 2 * pair_x;
                    let w_0 = E::from_i64(row[left] as i64);
                    let w_1 = if left + 1 < live_x_cols {
                        E::from_i64(row[left + 1] as i64)
                    } else {
                        E::zero()
                    };
                    *dst = w_0 + r * (w_1 - w_0);
                }
            });

        #[cfg(not(feature = "parallel"))]
        for (y, row_out) in out.chunks_mut(next_live_x_cols).enumerate() {
            let row_start = y * live_x_cols;
            let row = &w_compact[row_start..row_start + live_x_cols];
            for (pair_x, dst) in row_out.iter_mut().enumerate() {
                let left = 2 * pair_x;
                let w_0 = E::from_i64(row[left] as i64);
                let w_1 = if left + 1 < live_x_cols {
                    E::from_i64(row[left + 1] as i64)
                } else {
                    E::zero()
                };
                *dst = w_0 + r * (w_1 - w_0);
            }
        }

        out
    }

    fn fold_full_prefix_x(w_full: &[E], live_x_cols: usize, y_len: usize, r: E) -> Vec<E> {
        let next_live_x_cols = live_x_cols.div_ceil(2);
        let mut out = vec![E::zero(); y_len * next_live_x_cols];

        #[cfg(feature = "parallel")]
        out.par_chunks_mut(next_live_x_cols)
            .enumerate()
            .for_each(|(y, row_out)| {
                let row_start = y * live_x_cols;
                let row = &w_full[row_start..row_start + live_x_cols];
                for (pair_x, dst) in row_out.iter_mut().enumerate() {
                    let left = 2 * pair_x;
                    let w_0 = row[left];
                    let w_1 = if left + 1 < live_x_cols {
                        row[left + 1]
                    } else {
                        E::zero()
                    };
                    *dst = w_0 + r * (w_1 - w_0);
                }
            });

        #[cfg(not(feature = "parallel"))]
        for (y, row_out) in out.chunks_mut(next_live_x_cols).enumerate() {
            let row_start = y * live_x_cols;
            let row = &w_full[row_start..row_start + live_x_cols];
            for (pair_x, dst) in row_out.iter_mut().enumerate() {
                let left = 2 * pair_x;
                let w_0 = row[left];
                let w_1 = if left + 1 < live_x_cols {
                    row[left + 1]
                } else {
                    E::zero()
                };
                *dst = w_0 + r * (w_1 - w_0);
            }
        }

        out
    }

    fn fold_m_prefix(m_compact: &[E], live_x_cols: usize, r: E) -> Vec<E> {
        let next_live_x_cols = live_x_cols.div_ceil(2);
        cfg_into_iter!(0..next_live_x_cols)
            .map(|pair_x| {
                let left = 2 * pair_x;
                let m_0 = m_compact[left];
                let m_1 = if left + 1 < live_x_cols {
                    m_compact[left + 1]
                } else {
                    E::zero()
                };
                m_0 + r * (m_1 - m_0)
            })
            .collect()
    }

    fn fold_compact_to_full(w_compact: &[i8], r: E) -> Vec<E> {
        cfg_into_iter!(0..w_compact.len() / 2)
            .map(|j| {
                let w_0 = E::from_i64(w_compact[2 * j] as i64);
                let delta = w_compact[2 * j + 1] as i32 - w_compact[2 * j] as i32;
                let delta_abs = delta.unsigned_abs() as u64;
                let r_delta = E::reduce_mul_u64_accum(r.mul_u64_unreduced(delta_abs));
                if delta < 0 {
                    w_0 - r_delta
                } else {
                    w_0 + r_delta
                }
            })
            .collect()
    }
}

impl<E: FieldCore + FromSmallInt + CanonicalField + HasUnreducedOps> SumcheckInstanceProver<E>
    for HachiStage2Prover<E>
{
    fn num_rounds(&self) -> usize {
        self.num_vars
    }

    fn degree_bound(&self) -> usize {
        3
    }

    fn input_claim(&self) -> E {
        self.batching_coeff * self.s_claim + self.relation_claim
    }

    fn compute_round_univariate(&mut self, _round: usize, _previous_claim: E) -> UniPoly<E> {
        let t_scan = Instant::now();
        let poly = match &self.w_table {
            WTable::Compact(w_compact) => {
                if self.use_prefix_x_round() {
                    let (virt_poly, rel_poly) =
                        self.compute_round_compact_prefix_x_polys(w_compact);
                    self.combine_polys(&virt_poly, &rel_poly)
                } else {
                    let (virt_q_coeffs, rel_coeffs) =
                        self.compute_round_compact_dense_terms(w_compact);
                    self.combine_terms(virt_q_coeffs, rel_coeffs)
                }
            }
            WTable::Full(w_full) => {
                if self.use_prefix_x_round() {
                    let (virt_q_coeffs, rel_coeffs) =
                        self.compute_round_full_prefix_x_terms(w_full);
                    self.combine_terms(virt_q_coeffs, rel_coeffs)
                } else {
                    let (virt_q_coeffs, rel_coeffs) = self.compute_round_full_dense_terms(w_full);
                    self.combine_terms(virt_q_coeffs, rel_coeffs)
                }
            }
        };
        self.scan_time_total += t_scan.elapsed().as_secs_f64();
        poly
    }

    fn ingest_challenge(&mut self, _round: usize, r: E) {
        let t_fold = Instant::now();
        let _span = tracing::info_span!("HachiStage2Prover::fold_round").entered();
        self.split_eq.bind(r);
        let folding_x_round = self.rounds_completed < self.num_u;
        let use_prefix_x_round = self.use_prefix_x_round();
        let y_len = self.alpha_compact.len();

        self.w_table = match mem::replace(&mut self.w_table, WTable::Full(Vec::new())) {
            WTable::Compact(w_compact) => {
                let w_full = if use_prefix_x_round {
                    Self::fold_compact_prefix_x(&w_compact, self.live_x_cols, y_len, r)
                } else {
                    Self::fold_compact_to_full(&w_compact, r)
                };
                WTable::Full(w_full)
            }
            WTable::Full(mut w_full) => {
                if use_prefix_x_round {
                    w_full = Self::fold_full_prefix_x(&w_full, self.live_x_cols, y_len, r);
                } else {
                    fold_evals_in_place(&mut w_full, r);
                }
                WTable::Full(w_full)
            }
        };

        if folding_x_round {
            if use_prefix_x_round {
                self.m_compact = Self::fold_m_prefix(&self.m_compact, self.live_x_cols, r);
            } else {
                fold_evals_in_place(&mut self.m_compact, r);
            }
            self.live_x_cols = self.live_x_cols.div_ceil(2);
        } else {
            fold_evals_in_place(&mut self.alpha_compact, r);
        }

        drop(_span);
        self.fold_time_total += t_fold.elapsed().as_secs_f64();
        self.rounds_completed += 1;

        if self.rounds_completed == self.num_vars {
            tracing::debug!(
                rounds = self.num_vars,
                scan_s = self.scan_time_total,
                fold_s = self.fold_time_total,
                "stage2 sumcheck rounds complete"
            );
        }
    }
}

/// Verifier for the stage-2 fused virtual-claim + relation sumcheck.
pub struct HachiStage2Verifier<F: FieldCore, const D: usize> {
    batching_coeff: F,
    s_claim: F,
    w_evals: Vec<F>,
    /// When set, overrides the `w_eval` computed from `w_evals` in
    /// `expected_output_claim`. Used at intermediate fold levels where the
    /// full `w` vector is not available.
    w_eval_override: Option<F>,
    r_stage1: Vec<F>,
    alpha_evals_y: Vec<F>,
    m_evals_x: Vec<F>,
    num_u: usize,
    num_l: usize,
    relation_claim: F,
    _marker: PhantomData<[F; D]>,
}

impl<F: FieldCore + FromSmallInt + CanonicalField, const D: usize> HachiStage2Verifier<F, D> {
    /// Create a fused verifier for the stage-2 virtual-claim + relation sumcheck.
    #[allow(clippy::too_many_arguments)]
    #[tracing::instrument(skip_all, name = "HachiStage2Verifier::new")]
    pub fn new(
        batching_coeff: F,
        s_claim: F,
        w_evals: Vec<F>,
        r_stage1: Vec<F>,
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
        let relation_claim = relation_claim_from_rows::<F, D>(&tau1, alpha, &v, &u, &y_ring);
        Self {
            batching_coeff,
            s_claim,
            w_evals,
            w_eval_override: None,
            r_stage1,
            alpha_evals_y,
            m_evals_x,
            num_u,
            num_l,
            relation_claim,
            _marker: PhantomData,
        }
    }

    /// Set the `w_eval` override for intermediate fold levels where the
    /// full `w` vector is not available.
    pub fn with_w_eval_override(mut self, w_eval: F) -> Self {
        self.w_eval_override = Some(w_eval);
        self
    }
}

impl<F: FieldCore + FromSmallInt + CanonicalField, const D: usize> SumcheckInstanceVerifier<F>
    for HachiStage2Verifier<F, D>
{
    fn num_rounds(&self) -> usize {
        self.num_u + self.num_l
    }

    fn degree_bound(&self) -> usize {
        3
    }

    fn input_claim(&self) -> F {
        self.batching_coeff * self.s_claim + self.relation_claim
    }

    fn expected_output_claim(&self, challenges: &[F]) -> Result<F, HachiError> {
        let eq_val = EqPolynomial::mle(&self.r_stage1, challenges);
        let w_eval = match self.w_eval_override {
            Some(v) => v,
            None => multilinear_eval(&self.w_evals, challenges)?,
        };
        let virtual_oracle = eq_val * w_eval * (w_eval + F::one());

        let (x_challenges, y_challenges) = challenges.split_at(self.num_u);
        let alpha_val = multilinear_eval(&self.alpha_evals_y, y_challenges)?;
        let m_val = multilinear_eval(&self.m_evals_x, x_challenges)?;
        let relation_oracle = w_eval * alpha_val * m_val;

        Ok(self.batching_coeff * virtual_oracle + relation_oracle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::Prime128M8M4M1M0;

    type F = Prime128M8M4M1M0;

    struct Stage2Params<'a> {
        r_stage1: &'a [F],
        live_x_cols: usize,
        num_u: usize,
        num_l: usize,
    }

    fn new_stage2_test_prover(
        batching_coeff: F,
        s_claim: F,
        w_compact: Vec<i8>,
        alpha_evals_y: Vec<F>,
        m_evals_x: Vec<F>,
        params: Stage2Params<'_>,
    ) -> HachiStage2Prover<F> {
        HachiStage2Prover::new(
            batching_coeff,
            w_compact,
            params.r_stage1,
            s_claim,
            alpha_evals_y,
            m_evals_x,
            params.live_x_cols,
            params.num_u,
            params.num_l,
            F::zero(),
        )
    }

    fn relation_round_reference(
        w_compact: &[i8],
        alpha_compact: &[F],
        m_compact: &[F],
        num_u: usize,
    ) -> UniPoly<F> {
        let half = w_compact.len() / 2;
        let current_x_mask = (1usize << num_u).wrapping_sub(1);
        let mut evals = [F::zero(); 3];
        for j in 0..half {
            let w_0 = F::from_i64(w_compact[2 * j] as i64);
            let w_1 = F::from_i64(w_compact[2 * j + 1] as i64);
            let a_0 = alpha_compact[(2 * j) >> num_u];
            let a_1 = alpha_compact[(2 * j + 1) >> num_u];
            let m_0 = m_compact[(2 * j) & current_x_mask];
            let m_1 = m_compact[(2 * j + 1) & current_x_mask];
            evals[0] += w_0 * a_0 * m_0;
            evals[1] += w_1 * a_1 * m_1;
            let w_2 = w_1 + w_1 - w_0;
            let a_2 = a_1 + a_1 - a_0;
            let m_2 = m_1 + m_1 - m_0;
            evals[2] += w_2 * a_2 * m_2;
        }
        UniPoly::from_evals(&evals)
    }

    fn virtual_round_reference(split_eq: &GruenSplitEq<F>, w_compact: &[i8]) -> UniPoly<F> {
        let half = w_compact.len() / 2;
        let (e_first, e_second) = split_eq.remaining_eq_tables();
        let num_first = e_first.len();
        let first_bits = num_first.trailing_zeros();
        let mut evals = [F::zero(); 3];
        for j in 0..half {
            let j_low = j & (num_first - 1);
            let j_high = j >> first_bits;
            let eq_rem = e_first[j_low] * e_second[j_high];
            let w_0 = F::from_i64(w_compact[2 * j] as i64);
            let w_1 = F::from_i64(w_compact[2 * j + 1] as i64);
            let w_2 = w_1 + w_1 - w_0;
            evals[0] += eq_rem * w_0 * (w_0 + F::one());
            evals[1] += eq_rem * w_1 * (w_1 + F::one());
            evals[2] += eq_rem * w_2 * (w_2 + F::one());
        }
        split_eq.gruen_mul(&UniPoly::from_evals(&evals))
    }

    fn pad_compact_rows(
        w_prefix: &[i8],
        live_x_cols: usize,
        num_u: usize,
        num_l: usize,
    ) -> Vec<i8> {
        let x_len = 1usize << num_u;
        let y_len = 1usize << num_l;
        let mut padded = vec![0i8; x_len * y_len];
        for y in 0..y_len {
            let src_start = y * live_x_cols;
            let dst_start = y * x_len;
            padded[dst_start..dst_start + live_x_cols]
                .copy_from_slice(&w_prefix[src_start..src_start + live_x_cols]);
        }
        padded
    }

    #[test]
    fn stage2_compact_round0_matches_unfused_reference() {
        let num_u = 3usize;
        let num_l = 2usize;
        let b = 8usize;
        let n = 1usize << (num_u + num_l);
        let half = (b / 2) as i8;
        let w_compact: Vec<i8> = (0..n).map(|i| ((i * 5 + 3) % b) as i8 - half).collect();
        let r_stage1: Vec<F> = (0..(num_u + num_l))
            .map(|i| F::from_u64((i as u64) + 2))
            .collect();
        let alpha_evals_y: Vec<F> = (0..(1usize << num_l))
            .map(|i| F::from_u64((3 * i as u64) + 5))
            .collect();
        let m_evals_x: Vec<F> = (0..(1usize << num_u))
            .map(|i| F::from_u64((7 * i as u64) + 11))
            .collect();

        let prover = new_stage2_test_prover(
            F::from_u64(13),
            F::from_u64(17),
            w_compact.clone(),
            alpha_evals_y.clone(),
            m_evals_x.clone(),
            Stage2Params {
                r_stage1: &r_stage1,
                live_x_cols: 1usize << num_u,
                num_u,
                num_l,
            },
        );
        let (virt_poly, relation_poly) = prover.compute_round_compact_dense_polys(&w_compact);
        let virt_ref = virtual_round_reference(&prover.split_eq, &w_compact);
        let relation_ref = relation_round_reference(&w_compact, &alpha_evals_y, &m_evals_x, num_u);

        assert_eq!(virt_poly, virt_ref, "compact virtual round mismatch");
        assert_eq!(
            relation_poly, relation_ref,
            "compact relation round mismatch"
        );
    }

    #[test]
    fn stage2_prefix_aware_rounds_match_explicit_zero_padding() {
        let num_l = 2usize;
        let b = 8usize;
        let half = (b / 2) as i8;

        for live_x_cols in [5usize, 6usize] {
            let num_u = live_x_cols.next_power_of_two().trailing_zeros() as usize;
            let x_len = 1usize << num_u;
            let y_len = 1usize << num_l;
            let w_prefix: Vec<i8> = (0..(live_x_cols * y_len))
                .map(|i| ((i * 7 + 5) % b) as i8 - half)
                .collect();
            let w_padded = pad_compact_rows(&w_prefix, live_x_cols, num_u, num_l);
            let r_stage1: Vec<F> = (0..(num_u + num_l))
                .map(|i| F::from_u64((i as u64) + 31))
                .collect();
            let alpha_evals_y: Vec<F> = (0..y_len)
                .map(|i| F::from_u64((5 * i as u64) + 7))
                .collect();
            let mut m_evals_x: Vec<F> = (0..live_x_cols)
                .map(|i| F::from_u64((11 * i as u64) + 13))
                .collect();
            m_evals_x.resize(x_len, F::zero());

            let mut prefix_prover = new_stage2_test_prover(
                F::from_u64(17),
                F::from_u64(23),
                w_prefix.clone(),
                alpha_evals_y.clone(),
                m_evals_x.clone(),
                Stage2Params {
                    r_stage1: &r_stage1,
                    live_x_cols,
                    num_u,
                    num_l,
                },
            );
            let mut padded_prover = new_stage2_test_prover(
                F::from_u64(17),
                F::from_u64(23),
                w_padded.clone(),
                alpha_evals_y.clone(),
                m_evals_x.clone(),
                Stage2Params {
                    r_stage1: &r_stage1,
                    live_x_cols: 1usize << num_u,
                    num_u,
                    num_l,
                },
            );

            for round in 0..(num_u + num_l) {
                let prefix_poly = prefix_prover.compute_round_univariate(round, F::zero());
                let padded_poly = padded_prover.compute_round_univariate(round, F::zero());
                assert_eq!(
                    prefix_poly, padded_poly,
                    "round {round} polynomial mismatch live_x_cols={live_x_cols}"
                );

                let challenge = F::from_u64((round as u64) + 37);
                prefix_prover.ingest_challenge(round, challenge);
                padded_prover.ingest_challenge(round, challenge);
            }

            assert_eq!(prefix_prover.final_w_eval(), padded_prover.final_w_eval());
        }
    }

    #[test]
    fn stage2_zero_gated_round0_matches_reference() {
        let num_u = 3usize;
        let num_l = 1usize;
        let w_compact = vec![-1, 0, -1, 0, 0, -1, 0, -1, -1, 0, -1, 0, 0, -1, 0, -1];
        let r_stage1: Vec<F> = (0..(num_u + num_l))
            .map(|i| F::from_u64((i as u64) + 41))
            .collect();
        let alpha_evals_y: Vec<F> = (0..(1usize << num_l))
            .map(|i| F::from_u64((3 * i as u64) + 43))
            .collect();
        let m_evals_x: Vec<F> = (0..(1usize << num_u))
            .map(|i| F::from_u64((5 * i as u64) + 47))
            .collect();

        let prover = new_stage2_test_prover(
            F::from_u64(19),
            F::from_u64(29),
            w_compact.clone(),
            alpha_evals_y.clone(),
            m_evals_x.clone(),
            Stage2Params {
                r_stage1: &r_stage1,
                live_x_cols: 1usize << num_u,
                num_u,
                num_l,
            },
        );
        let (virt_poly, relation_poly) = prover.compute_round_compact_dense_polys(&w_compact);
        assert_eq!(
            virt_poly,
            virtual_round_reference(&prover.split_eq, &w_compact)
        );
        assert_eq!(
            relation_poly,
            relation_round_reference(&w_compact, &alpha_evals_y, &m_evals_x, num_u)
        );
    }
}
