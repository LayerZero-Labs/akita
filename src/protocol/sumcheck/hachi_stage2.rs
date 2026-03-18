//! Stage-2 fused sumcheck prover/verifier for the Hachi PCS.
//!
//! This stage views the committed witness as a Boolean table
//! `w : {0,1}^{num_u} x {0,1}^{num_l} -> F`, where `x` indexes the padded
//! witness columns and `y` indexes the coefficient inside a
//! `D = 2^{num_l}`-dimensional ring element. Let `a(y)` be the multilinear
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
//! `y_alpha = [v_0(alpha), ..., v_{N_D-1}(alpha),`
//! `           u_0(alpha), ..., u_{N_B-1}(alpha),`
//! `           y_ring(alpha), 0, ..., 0],`
//!
//! then the linear relation claim is
//!
//! `relation_claim = sum_i eq(tau1, i) * y_alpha[i]`
//! `               = sum_{x,y} w(x, y) * a(y) * m_tau1(x)`.
//!
//! Stage 1 supplies the carried virtual claim
//!
//! `s_claim = w(r_stage1) * (w(r_stage1) + 1)`
//! `        = sum_z eq(r_stage1, z) * w(z) * (w(z) + 1)`
//!
//! for the same multilinear witness table. With `gamma = batching_coeff`, the
//! exact identity established by this sumcheck is
//!
//! `gamma * s_claim + relation_claim =`
//! `sum_{x,y} [ gamma * eq(r_stage1, (x, y)) * w(x, y) * (w(x, y) + 1)`
//! `           + w(x, y) * a(y) * m_tau1(x) ]`.
//!
//! After all rounds, at `r_stage2 = (r_x, r_y)`, the verifier checks
//!
//! `gamma * eq(r_stage1, r_stage2) * w(r_stage2) * (w(r_stage2) + 1)`
//! `  + w(r_stage2) * a(r_y) * m_tau1(r_x)`,
//!
//! exactly the oracle returned by `expected_output_claim()`. The prover fuses
//! both halves around the same local `w0` / `dw` scan so the witness-side work
//! is shared between the virtual and relation terms.

use super::eq_poly::EqPolynomial;
use super::split_eq::GruenSplitEq;
use super::{fold_evals_in_place, multilinear_eval, trim_trailing_zeros, CompactPairFoldLut};
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
/// Holds a single `w_table` shared by both halves of stage 2. The virtual half
/// is pre-weighted by `batching_coeff` through `split_eq`, so the round
/// polynomial is:
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
            split_eq: GruenSplitEq::with_initial_scalar(r_stage1, batching_coeff),
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
            combined[i] += *c;
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
        let (e_first, e_second) = self.split_eq.remaining_eq_tables();
        let num_first = e_first.len();
        let num_second = e_second.len();
        let current_x_width = self.current_x_width();
        let current_x_mask = (1usize << current_x_width).wrapping_sub(1);
        let alpha_compact = &self.alpha_compact;
        let m_compact = &self.m_compact;
        debug_assert_eq!(w_compact.len() / 2, num_first * num_second);

        let (virt_coeffs, rel_accum) = cfg_fold_reduce!(
            0..num_second,
            || ([E::zero(); 3], [E::MulU64Accum::ZERO; 6]),
            |(mut virt, mut rel), j_high| {
                let mut inner_virt = [E::MulU64Accum::ZERO; 4];
                let base = j_high * num_first;

                for (j_low, &e_in) in e_first.iter().enumerate() {
                    let j = base + j_low;
                    let w0 = w_compact[2 * j] as i32;
                    let w1 = w_compact[2 * j + 1] as i32;
                    let dw = w1 - w0;
                    let w0_i64 = w0 as i64;
                    let dw_i64 = dw as i64;

                    let q0 = w0_i64 * (w0_i64 + 1);
                    if q0 != 0 {
                        inner_virt[0] += e_in.mul_u64_unreduced(q0 as u64);
                    }
                    let q1 = dw_i64 * (2 * w0_i64 + 1);
                    accum_small_signed::<E>(&mut inner_virt, 1, e_in, q1);
                    let q2 = dw_i64 * dw_i64;
                    if q2 != 0 {
                        inner_virt[3] += e_in.mul_u64_unreduced(q2 as u64);
                    }

                    let a0 = alpha_compact[(2 * j) >> current_x_width];
                    let a1 = alpha_compact[(2 * j + 1) >> current_x_width];
                    let m0 = m_compact[(2 * j) & current_x_mask];
                    let m1 = m_compact[(2 * j + 1) & current_x_mask];
                    let p0 = a0 * m0;
                    let p1 = a1 * m1;
                    let dp = p1 - p0;
                    accum_small_signed::<E>(&mut rel, 0, p0, w0_i64);
                    accum_small_signed::<E>(&mut rel, 2, dp, w0_i64);
                    accum_small_signed::<E>(&mut rel, 2, p0, dw_i64);
                    accum_small_signed::<E>(&mut rel, 4, dp, dw_i64);
                }

                let reduced_inner: [E; 3] = reduce_compact_virt(inner_virt);
                let e_out = e_second[j_high];
                virt[0] += e_out * reduced_inner[0];
                virt[1] += e_out * reduced_inner[1];
                virt[2] += e_out * reduced_inner[2];

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

        (virt_coeffs, reduce_compact_rel(rel_accum))
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
        let first_bits = num_first.trailing_zeros() as usize;
        let current_x_half = 1usize << (self.current_x_width() - 1);
        let live_pairs = self.live_x_cols.div_ceil(2);
        let block_size = num_first.min(live_pairs);
        let alpha_compact = &self.alpha_compact;
        let m_compact = &self.m_compact;

        let (virt_coeffs, rel_accum) = cfg_fold_reduce!(
            0..alpha_compact.len(),
            || ([E::zero(); 3], [E::MulU64Accum::ZERO; 6]),
            |(mut virt, mut rel), y| {
                let row_start = y * self.live_x_cols;
                let row = &w_compact[row_start..row_start + self.live_x_cols];
                let alpha = alpha_compact[y];
                let j_base = y * current_x_half;

                let mut blk = 0usize;
                while blk < live_pairs {
                    let blk_end = (blk + block_size).min(live_pairs);
                    let j_high = (j_base + blk) >> first_bits;
                    let mut inner_virt = [E::MulU64Accum::ZERO; 4];

                    for pair_x in blk..blk_end {
                        let j_low = (j_base + pair_x) & (num_first - 1);
                        let e_in = e_first[j_low];
                        let left = 2 * pair_x;
                        let w0 = row[left] as i32;
                        let w1 = if left + 1 < self.live_x_cols {
                            row[left + 1] as i32
                        } else {
                            0
                        };
                        let dw = w1 - w0;
                        let w0_i64 = w0 as i64;
                        let dw_i64 = dw as i64;

                        let q0 = w0_i64 * (w0_i64 + 1);
                        if q0 != 0 {
                            inner_virt[0] += e_in.mul_u64_unreduced(q0 as u64);
                        }
                        let q1 = dw_i64 * (2 * w0_i64 + 1);
                        accum_small_signed::<E>(&mut inner_virt, 1, e_in, q1);
                        let q2 = dw_i64 * dw_i64;
                        if q2 != 0 {
                            inner_virt[3] += e_in.mul_u64_unreduced(q2 as u64);
                        }

                        let m0 = m_compact[left];
                        let m1 = if left + 1 < self.live_x_cols {
                            m_compact[left + 1]
                        } else {
                            E::zero()
                        };
                        let p0 = alpha * m0;
                        let p1 = alpha * m1;
                        let dp = p1 - p0;
                        accum_small_signed::<E>(&mut rel, 0, p0, w0_i64);
                        accum_small_signed::<E>(&mut rel, 2, dp, w0_i64);
                        accum_small_signed::<E>(&mut rel, 2, p0, dw_i64);
                        accum_small_signed::<E>(&mut rel, 4, dp, dw_i64);
                    }

                    let reduced_inner: [E; 3] = reduce_compact_virt(inner_virt);
                    let e_out = e_second[j_high];
                    virt[0] += e_out * reduced_inner[0];
                    virt[1] += e_out * reduced_inner[1];
                    virt[2] += e_out * reduced_inner[2];

                    blk = blk_end;
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

        (virt_coeffs, reduce_compact_rel(rel_accum))
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
        let first_bits = num_first.trailing_zeros() as usize;
        let current_x_half = 1usize << (self.current_x_width() - 1);
        let live_pairs = self.live_x_cols.div_ceil(2);
        let block_size = num_first.min(live_pairs);
        let alpha_compact = &self.alpha_compact;
        let m_compact = &self.m_compact;

        cfg_fold_reduce!(
            0..alpha_compact.len(),
            || ([E::zero(); 3], [E::zero(); 3]),
            |(mut virt, mut rel), y| {
                let row_start = y * self.live_x_cols;
                let row = &w_full[row_start..row_start + self.live_x_cols];
                let alpha = alpha_compact[y];
                let j_base = y * current_x_half;

                let mut blk = 0usize;
                while blk < live_pairs {
                    let blk_end = (blk + block_size).min(live_pairs);
                    let j_high = (j_base + blk) >> first_bits;
                    let mut inner_virt = [E::zero(); 3];

                    for pair_x in blk..blk_end {
                        let j_low = (j_base + pair_x) & (num_first - 1);
                        let e_in = e_first[j_low];
                        let left = 2 * pair_x;
                        let w0 = row[left];
                        let w1 = if left + 1 < self.live_x_cols {
                            row[left + 1]
                        } else {
                            E::zero()
                        };
                        let dw = w1 - w0;
                        let two_w0_plus_one = w0 + w0 + E::one();

                        inner_virt[0] += e_in * (w0 * (w0 + E::one()));
                        inner_virt[1] += e_in * (dw * two_w0_plus_one);
                        inner_virt[2] += e_in * (dw * dw);

                        let m0 = m_compact[left];
                        let m1 = if left + 1 < self.live_x_cols {
                            m_compact[left + 1]
                        } else {
                            E::zero()
                        };
                        let p0 = alpha * m0;
                        let p1 = alpha * m1;
                        let dp = p1 - p0;
                        rel[0] += w0 * p0;
                        rel[1] += w0 * dp + dw * p0;
                        rel[2] += dw * dp;
                    }

                    let e_out = e_second[j_high];
                    virt[0] += e_out * inner_virt[0];
                    virt[1] += e_out * inner_virt[1];
                    virt[2] += e_out * inner_virt[2];

                    blk = blk_end;
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
        )
    }

    #[tracing::instrument(skip_all, name = "HachiStage2Prover::compute_round_full_dense_terms")]
    fn compute_round_full_dense_terms(&self, w_full: &[E]) -> ([E; 3], [E; 3]) {
        let (e_first, e_second) = self.split_eq.remaining_eq_tables();
        let num_first = e_first.len();
        let num_second = e_second.len();
        let current_x_width = self.current_x_width();
        let current_x_mask = (1usize << current_x_width).wrapping_sub(1);
        let alpha_compact = &self.alpha_compact;
        let m_compact = &self.m_compact;
        debug_assert_eq!(w_full.len() / 2, num_first * num_second);

        cfg_fold_reduce!(
            0..num_second,
            || ([E::zero(); 3], [E::zero(); 3]),
            |(mut virt, mut rel), j_high| {
                let mut inner_virt = [E::zero(); 3];
                let base = j_high * num_first;

                for (j_low, &e_in) in e_first.iter().enumerate() {
                    let j = base + j_low;
                    let w0 = w_full[2 * j];
                    let w1 = w_full[2 * j + 1];
                    let dw = w1 - w0;
                    let two_w0_plus_one = w0 + w0 + E::one();

                    inner_virt[0] += e_in * (w0 * (w0 + E::one()));
                    inner_virt[1] += e_in * (dw * two_w0_plus_one);
                    inner_virt[2] += e_in * (dw * dw);

                    let a0 = alpha_compact[(2 * j) >> current_x_width];
                    let a1 = alpha_compact[(2 * j + 1) >> current_x_width];
                    let m0 = m_compact[(2 * j) & current_x_mask];
                    let m1 = m_compact[(2 * j + 1) & current_x_mask];
                    let p0 = a0 * m0;
                    let p1 = a1 * m1;
                    let dp = p1 - p0;
                    rel[0] += w0 * p0;
                    rel[1] += w0 * dp + dw * p0;
                    rel[2] += dw * dp;
                }

                let e_out = e_second[j_high];
                virt[0] += e_out * inner_virt[0];
                virt[1] += e_out * inner_virt[1];
                virt[2] += e_out * inner_virt[2];

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

    #[inline]
    fn build_compact_w_fold_lut(w_compact: &[i8], r: E) -> CompactPairFoldLut<E> {
        let min_w = w_compact
            .iter()
            .copied()
            .map(i32::from)
            .min()
            .unwrap_or(0)
            .min(0);
        let max_w = w_compact
            .iter()
            .copied()
            .map(i32::from)
            .max()
            .unwrap_or(0)
            .max(0);
        CompactPairFoldLut::from_contiguous_range(min_w, max_w, r)
    }

    fn fold_compact_prefix_x(
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
                let row_start = y * live_x_cols;
                let row = &w_compact[row_start..row_start + live_x_cols];
                for (pair_x, dst) in row_out.iter_mut().enumerate() {
                    let left = 2 * pair_x;
                    let w_1 = if left + 1 < live_x_cols {
                        row[left + 1] as i32
                    } else {
                        0
                    };
                    *dst = fold_lut.fold(row[left] as i32, w_1);
                }
            });

        #[cfg(not(feature = "parallel"))]
        for (y, row_out) in out.chunks_mut(next_live_x_cols).enumerate() {
            let row_start = y * live_x_cols;
            let row = &w_compact[row_start..row_start + live_x_cols];
            for (pair_x, dst) in row_out.iter_mut().enumerate() {
                let left = 2 * pair_x;
                let w_1 = if left + 1 < live_x_cols {
                    row[left + 1] as i32
                } else {
                    0
                };
                *dst = fold_lut.fold(row[left] as i32, w_1);
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

    fn fold_compact_to_full(w_compact: &[i8], fold_lut: &CompactPairFoldLut<E>) -> Vec<E> {
        cfg_into_iter!(0..w_compact.len() / 2)
            .map(|j| fold_lut.fold(w_compact[2 * j] as i32, w_compact[2 * j + 1] as i32))
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
                let fold_lut = Self::build_compact_w_fold_lut(&w_compact, r);
                let w_full = if use_prefix_x_round {
                    Self::fold_compact_prefix_x(&w_compact, self.live_x_cols, y_len, &fold_lut)
                } else {
                    Self::fold_compact_to_full(&w_compact, &fold_lut)
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

    fn fold_compact_prefix_x_reference(
        w_compact: &[i8],
        live_x_cols: usize,
        y_len: usize,
        r: F,
    ) -> Vec<F> {
        let next_live_x_cols = live_x_cols.div_ceil(2);
        let mut out = vec![F::zero(); y_len * next_live_x_cols];
        for (y, row_out) in out.chunks_mut(next_live_x_cols).enumerate() {
            let row_start = y * live_x_cols;
            let row = &w_compact[row_start..row_start + live_x_cols];
            for (pair_x, dst) in row_out.iter_mut().enumerate() {
                let left = 2 * pair_x;
                let w_0 = F::from_i64(row[left] as i64);
                let w_1 = if left + 1 < live_x_cols {
                    F::from_i64(row[left + 1] as i64)
                } else {
                    F::zero()
                };
                *dst = w_0 + r * (w_1 - w_0);
            }
        }
        out
    }

    fn fold_compact_to_full_reference(w_compact: &[i8], r: F) -> Vec<F> {
        (0..w_compact.len() / 2)
            .map(|j| {
                let w_0 = F::from_i64(w_compact[2 * j] as i64);
                let w_1 = F::from_i64(w_compact[2 * j + 1] as i64);
                w_0 + r * (w_1 - w_0)
            })
            .collect()
    }

    #[test]
    fn stage2_compact_fold_lookup_matches_direct_formula() {
        let r = F::from_u64(53);

        let w_prefix = vec![1, 2, 3, 1, 2, 3, 1, 2, 3, 1];
        let fold_lut = HachiStage2Prover::<F>::build_compact_w_fold_lut(&w_prefix, r);
        assert_eq!(
            HachiStage2Prover::<F>::fold_compact_prefix_x(&w_prefix, 5, 2, &fold_lut),
            fold_compact_prefix_x_reference(&w_prefix, 5, 2, r)
        );

        let w_dense = vec![1, 2, 3, 1, 2, 3];
        let dense_lut = HachiStage2Prover::<F>::build_compact_w_fold_lut(&w_dense, r);
        assert_eq!(
            HachiStage2Prover::<F>::fold_compact_to_full(&w_dense, &dense_lut),
            fold_compact_to_full_reference(&w_dense, r)
        );
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
