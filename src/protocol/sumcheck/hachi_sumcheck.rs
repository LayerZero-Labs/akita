//! Fused norm+relation sumcheck prover/verifier for the Hachi PCS.
//!
//! Eliminates the redundant `w_evals` clone by sharing a single `w_table`
//! across both the norm (F_0) and relation (F_α) sumcheck computations.
//! Supports compact `Vec<i8>` storage for round 0 (all entries in [-b/2, b/2)),
//! transitioning to `Vec<F>` at half size after the first fold.

use super::eq_poly::EqPolynomial;
use super::norm_sumcheck::{
    choose_round_kernel, compute_entry_coeffs, compute_entry_coeffs_x4, field_from_i128,
    range_check_eval_i128, range_check_eval_precomputed, trim_trailing_zeros, NormRoundKernel,
    PointEvalPrecomp, RangeAffinePrecomp, MAX_AFFINE_COEFFS,
};
use super::split_eq::GruenSplitEq;
use super::{fold_evals_in_place, multilinear_eval, range_check_eval};
use super::{SumcheckInstanceProver, SumcheckInstanceVerifier, UniPoly};
use crate::algebra::fields::HasUnreducedOps;
use crate::algebra::CyclotomicRing;
use crate::error::HachiError;
#[cfg(feature = "parallel")]
use crate::parallel::*;
use crate::protocol::ring_switch::eval_ring_at;
use std::marker::PhantomData;

use crate::{cfg_fold_reduce, cfg_into_iter};
use std::iter;
use std::mem;
use std::time::Instant;

use crate::{AdditiveGroup, CanonicalField, FieldCore, FromSmallInt};

enum WTable<E: FieldCore> {
    Compact(Vec<i8>),
    Full(Vec<E>),
}

/// Fused norm+relation sumcheck prover.
///
/// Holds a single `w_table` shared by both sumcheck instances, weighted
/// by `batching_coeff`. The round polynomial is
/// `batching_coeff * norm_round(t) + relation_round(t)`.
///
/// Alpha and m are stored in compact form (sizes `2^num_l` and `2^num_u`
/// respectively) and folded only during rounds where their variables are active.
pub struct HachiSumcheckProver<E: FieldCore> {
    w_table: WTable<E>,
    batching_coeff: E,

    // Norm state
    split_eq: GruenSplitEq<E>,
    round_kernel: NormRoundKernel,
    point_precomp: Option<PointEvalPrecomp<E>>,
    range_precomp: Option<RangeAffinePrecomp<E>>,
    b: usize,

    // Relation state (compact — not expanded to full domain)
    alpha_compact: Vec<E>,
    m_compact: Vec<E>,
    num_u: usize,

    num_vars: usize,
    relation_claim: E,

    norm_time_total: f64,
    relation_time_total: f64,
    fold_time_total: f64,
    rounds_completed: usize,
}

impl<E: FieldCore + FromSmallInt + CanonicalField + HasUnreducedOps> HachiSumcheckProver<E> {
    /// Create a fused norm+relation sumcheck prover.
    ///
    /// # Panics
    ///
    /// Panics if table sizes are inconsistent with `num_u` and `num_l`.
    #[allow(clippy::too_many_arguments)]
    #[tracing::instrument(skip_all, name = "HachiSumcheckProver::new")]
    pub fn new(
        batching_coeff: E,
        w_evals_compact: Vec<i8>,
        tau0: &[E],
        b: usize,
        alpha_evals_y: Vec<E>,
        m_evals_x: Vec<E>,
        num_u: usize,
        num_l: usize,
        relation_claim: E,
    ) -> Self {
        assert!(b >= 1, "b must be at least 1");
        let num_vars = num_u + num_l;
        let n = 1usize << num_vars;
        assert_eq!(w_evals_compact.len(), n);
        assert_eq!(tau0.len(), num_vars);
        assert_eq!(alpha_evals_y.len(), 1 << num_l);
        assert_eq!(m_evals_x.len(), 1 << num_u);

        let round_kernel = choose_round_kernel(b);
        let point_precomp = match round_kernel {
            NormRoundKernel::PointEvalInterpolation => Some(PointEvalPrecomp::new(b)),
            NormRoundKernel::AffineCoeffComposition => None,
        };
        let range_precomp = match round_kernel {
            NormRoundKernel::PointEvalInterpolation => None,
            NormRoundKernel::AffineCoeffComposition => Some(RangeAffinePrecomp::new(b)),
        };

        Self {
            w_table: WTable::Compact(w_evals_compact),
            batching_coeff,
            split_eq: GruenSplitEq::new(tau0),
            round_kernel,
            point_precomp,
            range_precomp,
            b,
            alpha_compact: alpha_evals_y,
            m_compact: m_evals_x,
            num_u,
            num_vars,
            relation_claim,
            norm_time_total: 0.0,
            relation_time_total: 0.0,
            fold_time_total: 0.0,
            rounds_completed: 0,
        }
    }

    /// Accumulate `am * w_int` into split pos/neg accumulators.
    /// `accum[pos_idx]` gets the product when `w_int >= 0`,
    /// `accum[pos_idx + 1]` gets it when `w_int < 0`.
    #[inline]
    fn accum_signed_mul(accum: &mut [E::MulU64Accum], pos_idx: usize, am: E, w_int: i32) {
        let prod = am.mul_u64_unreduced(w_int.unsigned_abs() as u64);
        if w_int < 0 {
            accum[pos_idx + 1] += prod;
        } else {
            accum[pos_idx] += prod;
        }
    }

    /// Reduce a (positive, negative) accumulator pair to a single field element.
    #[inline]
    fn reduce_signed_accum(pos: E::MulU64Accum, neg: E::MulU64Accum) -> E {
        E::reduce_mul_u64_accum(pos) - E::reduce_mul_u64_accum(neg)
    }

    /// Fused compact round 0: computes both the norm and relation round
    /// polynomials in a single pass over `w_compact`, using i128/LUT
    /// arithmetic for the norm and unreduced small-int multiplies for the
    /// relation. Relation uses split pos/neg accumulators to avoid
    /// wrapping-neg overflow in the unsigned limbed accumulators.
    #[tracing::instrument(skip_all, name = "HachiSumcheckProver::compute_round_compact_fused")]
    fn compute_round_compact_fused(&self, w_compact: &[i8]) -> (UniPoly<E>, UniPoly<E>) {
        let half = w_compact.len() / 2;
        let (e_first, e_second) = self.split_eq.remaining_eq_tables();
        let num_first = e_first.len();
        let first_bits = num_first.trailing_zeros();
        let current_x_width = self.num_u.saturating_sub(self.rounds_completed);
        let current_x_mask = (1usize << current_x_width).wrapping_sub(1);
        let alpha_compact = &self.alpha_compact;
        let m_compact = &self.m_compact;
        let b = self.b;

        // 6-element array: [pos0, neg0, pos1, neg1, pos2, neg2]
        type RelAccum<E> = [<E as HasUnreducedOps>::MulU64Accum; 6];
        let rel_zero = || -> RelAccum<E> { [E::MulU64Accum::ZERO; 6] };
        #[allow(unused_variables)]
        let rel_combine = |a: &mut RelAccum<E>, b: &RelAccum<E>| {
            for i in 0..6 {
                a[i] += b[i];
            }
        };
        let rel_reduce = |r: RelAccum<E>| -> [E; 3] {
            [
                Self::reduce_signed_accum(r[0], r[1]),
                Self::reduce_signed_accum(r[2], r[3]),
                Self::reduce_signed_accum(r[4], r[5]),
            ]
        };

        match self.round_kernel {
            NormRoundKernel::PointEvalInterpolation if b <= 10 => {
                let degree_q = 2 * b - 1;
                let num_points_q = degree_q + 1;

                let _span = tracing::info_span!("fused_compact_point_eval").entered();
                let (q_evals, rel_accum) = cfg_fold_reduce!(
                    0..half,
                    || (vec![E::zero(); num_points_q], rel_zero()),
                    |(mut norm_evals, mut rel), j| {
                        let w0_i = w_compact[2 * j] as i32;
                        let w1_i = w_compact[2 * j + 1] as i32;
                        let delta_i = w1_i - w0_i;

                        let j_low = j & (num_first - 1);
                        let j_high = j >> first_bits;
                        let eq_rem = e_first[j_low] * e_second[j_high];
                        let mut w_t_i = w0_i;
                        for eval in norm_evals.iter_mut() {
                            let rc = range_check_eval_i128(w_t_i, b);
                            *eval += eq_rem * field_from_i128::<E>(rc);
                            w_t_i += delta_i;
                        }

                        let a_0 = alpha_compact[(2 * j) >> current_x_width];
                        let a_1 = alpha_compact[(2 * j + 1) >> current_x_width];
                        let m_0 = m_compact[(2 * j) & current_x_mask];
                        let m_1 = m_compact[(2 * j + 1) & current_x_mask];
                        let am_0 = a_0 * m_0;
                        let am_1 = a_1 * m_1;
                        let w2_i = 2 * w1_i - w0_i;
                        let am_2 = (a_1 + a_1 - a_0) * (m_1 + m_1 - m_0);

                        Self::accum_signed_mul(&mut rel, 0, am_0, w0_i);
                        Self::accum_signed_mul(&mut rel, 2, am_1, w1_i);
                        Self::accum_signed_mul(&mut rel, 4, am_2, w2_i);

                        (norm_evals, rel)
                    },
                    |(mut na, mut ra), (nb, rb)| {
                        for (ai, bi) in na.iter_mut().zip(nb.iter()) {
                            *ai += *bi;
                        }
                        rel_combine(&mut ra, &rb);
                        (na, ra)
                    }
                );

                let q_poly = UniPoly::from_evals(&q_evals);
                let norm_poly = self.split_eq.gruen_mul(&q_poly);
                let rel_evals = rel_reduce(rel_accum);
                (norm_poly, UniPoly::from_evals(&rel_evals))
            }
            NormRoundKernel::AffineCoeffComposition => {
                let rp = self.range_precomp.as_ref().unwrap();
                let num_coeffs_q = rp.degree_q + 1;

                let _span = tracing::info_span!("fused_compact_affine_coeff").entered();
                let (mut q_coeffs, rel_accum) = cfg_fold_reduce!(
                    0..e_second.len(),
                    || (vec![E::ProductAccum::ZERO; num_coeffs_q], rel_zero()),
                    |(mut outer_accum, mut rel), j_high| {
                        debug_assert!(num_coeffs_q <= MAX_AFFINE_COEFFS);
                        let mut inner_accum = [E::ProductAccum::ZERO; MAX_AFFINE_COEFFS];
                        for (j_low, &e_in) in e_first.iter().enumerate() {
                            let j = j_high * num_first + j_low;
                            let w0_int = w_compact[2 * j];
                            let w1_int = w_compact[2 * j + 1];

                            let w_1 = E::from_i64(w1_int as i64);
                            let a = w_1 - E::from_i64(w0_int as i64);
                            let mut a_pow = E::one();
                            for (i, acc) in inner_accum[..num_coeffs_q].iter_mut().enumerate() {
                                let h_i_w0 = rp.h_i_lut(w0_int, i);
                                let val = a_pow * h_i_w0;
                                *acc += e_in.mul_to_product_accum(val);
                                a_pow = a_pow * a;
                            }

                            let a_0 = alpha_compact[(2 * j) >> current_x_width];
                            let a_1 = alpha_compact[(2 * j + 1) >> current_x_width];
                            let m_0 = m_compact[(2 * j) & current_x_mask];
                            let m_1 = m_compact[(2 * j + 1) & current_x_mask];
                            let am_0 = a_0 * m_0;
                            let am_1 = a_1 * m_1;
                            let w2_i = 2 * w1_int as i32 - w0_int as i32;
                            let am_2 = (a_1 + a_1 - a_0) * (m_1 + m_1 - m_0);

                            Self::accum_signed_mul(&mut rel, 0, am_0, w0_int as i32);
                            Self::accum_signed_mul(&mut rel, 2, am_1, w1_int as i32);
                            Self::accum_signed_mul(&mut rel, 4, am_2, w2_i);
                        }
                        let e_out = e_second[j_high];
                        for k in 0..num_coeffs_q {
                            let inner_reduced = E::reduce_product_accum(inner_accum[k]);
                            outer_accum[k] += e_out.mul_to_product_accum(inner_reduced);
                        }
                        (outer_accum, rel)
                    },
                    |(mut ca, mut ra), (cb, rb)| {
                        for (ai, bi) in ca.iter_mut().zip(cb.iter()) {
                            *ai += *bi;
                        }
                        rel_combine(&mut ra, &rb);
                        (ca, ra)
                    }
                );

                let q_coeffs_reduced: Vec<E> =
                    q_coeffs.drain(..).map(E::reduce_product_accum).collect();
                let mut q_coeffs = q_coeffs_reduced;
                trim_trailing_zeros(&mut q_coeffs);
                let q_poly = UniPoly::from_coeffs(q_coeffs);
                let norm_poly = self.split_eq.gruen_mul(&q_poly);
                let rel_evals = rel_reduce(rel_accum);
                (norm_poly, UniPoly::from_evals(&rel_evals))
            }
            _ => {
                // b > 10 with point-eval: fall back to separate passes
                let _span = tracing::info_span!("compact_fallback").entered();
                use super::norm_sumcheck::compute_norm_round_poly_compact;
                let np = compute_norm_round_poly_compact(
                    &self.split_eq,
                    w_compact,
                    b,
                    self.round_kernel,
                    self.point_precomp.as_ref(),
                    self.range_precomp.as_ref(),
                );
                let pair = |j: usize| {
                    (
                        E::from_i64(w_compact[2 * j] as i64),
                        E::from_i64(w_compact[2 * j + 1] as i64),
                    )
                };
                let rel_evals = cfg_fold_reduce!(
                    0..half,
                    || [E::zero(); 3],
                    |mut evals, j| {
                        let (w_0, w_1) = pair(j);
                        let a_0 = alpha_compact[(2 * j) >> current_x_width];
                        let a_1 = alpha_compact[(2 * j + 1) >> current_x_width];
                        let m_0 = m_compact[(2 * j) & current_x_mask];
                        let m_1 = m_compact[(2 * j + 1) & current_x_mask];
                        evals[0] += w_0 * a_0 * m_0;
                        evals[1] += w_1 * a_1 * m_1;
                        let w_2 = w_1 + w_1 - w_0;
                        let a_2 = a_1 + a_1 - a_0;
                        let m_2 = m_1 + m_1 - m_0;
                        evals[2] += w_2 * a_2 * m_2;
                        evals
                    },
                    |mut a, b| {
                        for (ai, bi) in a.iter_mut().zip(b.iter()) {
                            *ai += *bi;
                        }
                        a
                    }
                );
                (np, UniPoly::from_evals(&rel_evals))
            }
        }
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
    for HachiSumcheckProver<E>
{
    fn num_rounds(&self) -> usize {
        self.num_vars
    }

    fn degree_bound(&self) -> usize {
        2 * self.b
    }

    fn input_claim(&self) -> E {
        self.relation_claim
    }

    fn compute_round_univariate(&mut self, _round: usize, _previous_claim: E) -> UniPoly<E> {
        let t_norm = Instant::now();
        let (norm_poly, relation_poly) = match &self.w_table {
            WTable::Compact(w_compact) => {
                let result = self.compute_round_compact_fused(w_compact);
                self.norm_time_total += t_norm.elapsed().as_secs_f64();
                result
            }
            WTable::Full(w_full) => {
                let half = w_full.len() / 2;
                let (e_first, e_second) = self.split_eq.remaining_eq_tables();
                let num_first = e_first.len();
                let first_bits = num_first.trailing_zeros();
                let current_x_width = self.num_u.saturating_sub(self.rounds_completed);
                let current_x_mask = (1usize << current_x_width).wrapping_sub(1);
                let alpha_compact = &self.alpha_compact;
                let m_compact = &self.m_compact;

                let _span = tracing::info_span!("fused_norm_relation").entered();

                let (np, rp) = match self.round_kernel {
                    NormRoundKernel::PointEvalInterpolation => {
                        let degree_q = 2 * self.b - 1;
                        let num_points_q = degree_q + 1;
                        let offsets_sq = &self.point_precomp.as_ref().unwrap().range_offsets_sq;

                        let (q_evals, rel_evals) = cfg_fold_reduce!(
                            0..half,
                            || (vec![E::zero(); num_points_q], [E::zero(); 3]),
                            |(mut norm_evals, mut rel_evals), j| {
                                let w_0 = w_full[2 * j];
                                let w_1 = w_full[2 * j + 1];

                                let j_low = j & (num_first - 1);
                                let j_high = j >> first_bits;
                                let eq_rem = e_first[j_low] * e_second[j_high];
                                let delta = w_1 - w_0;
                                let mut w_t = w_0;
                                for eval in norm_evals.iter_mut() {
                                    *eval += eq_rem * range_check_eval_precomputed(w_t, offsets_sq);
                                    w_t += delta;
                                }

                                let a_0 = alpha_compact[(2 * j) >> current_x_width];
                                let a_1 = alpha_compact[(2 * j + 1) >> current_x_width];
                                let m_0 = m_compact[(2 * j) & current_x_mask];
                                let m_1 = m_compact[(2 * j + 1) & current_x_mask];
                                rel_evals[0] += w_0 * a_0 * m_0;
                                rel_evals[1] += w_1 * a_1 * m_1;
                                let w_2 = w_1 + w_1 - w_0;
                                let a_2 = a_1 + a_1 - a_0;
                                let m_2 = m_1 + m_1 - m_0;
                                rel_evals[2] += w_2 * a_2 * m_2;

                                (norm_evals, rel_evals)
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

                        let q_poly = UniPoly::from_evals(&q_evals);
                        (
                            self.split_eq.gruen_mul(&q_poly),
                            UniPoly::from_evals(&rel_evals),
                        )
                    }
                    NormRoundKernel::AffineCoeffComposition => {
                        let range_pc = self.range_precomp.as_ref().unwrap();
                        let num_coeffs_q = range_pc.degree_q + 1;
                        debug_assert!(num_coeffs_q <= MAX_AFFINE_COEFFS);

                        let (mut q_coeffs, rel_evals) = cfg_fold_reduce!(
                            0..e_second.len(),
                            || (vec![E::ProductAccum::ZERO; num_coeffs_q], [E::zero(); 3]),
                            |(mut outer_accum, mut rel_evals), j_high| {
                                let mut inner_accum = [E::ProductAccum::ZERO; MAX_AFFINE_COEFFS];
                                let base_j = j_high * num_first;
                                let full_chunks = num_first / 4;
                                let mut batch_out = [[E::zero(); MAX_AFFINE_COEFFS]; 4];

                                for chunk in 0..full_chunks {
                                    let jl = chunk * 4;
                                    let w = [
                                        (w_full[2 * (base_j + jl)], w_full[2 * (base_j + jl) + 1]),
                                        (
                                            w_full[2 * (base_j + jl + 1)],
                                            w_full[2 * (base_j + jl + 1) + 1],
                                        ),
                                        (
                                            w_full[2 * (base_j + jl + 2)],
                                            w_full[2 * (base_j + jl + 2) + 1],
                                        ),
                                        (
                                            w_full[2 * (base_j + jl + 3)],
                                            w_full[2 * (base_j + jl + 3) + 1],
                                        ),
                                    ];
                                    compute_entry_coeffs_x4(
                                        &mut batch_out,
                                        range_pc,
                                        [w[0].0, w[1].0, w[2].0, w[3].0],
                                        [
                                            w[0].1 - w[0].0,
                                            w[1].1 - w[1].0,
                                            w[2].1 - w[2].0,
                                            w[3].1 - w[3].0,
                                        ],
                                    );
                                    for (b, bo) in batch_out.iter().enumerate() {
                                        let e_in = e_first[jl + b];
                                        for (acc, &entry) in inner_accum[..num_coeffs_q]
                                            .iter_mut()
                                            .zip(bo[..num_coeffs_q].iter())
                                        {
                                            *acc += e_in.mul_to_product_accum(entry);
                                        }
                                    }
                                    for (b, &(w_0, w_1)) in w.iter().enumerate() {
                                        let j = base_j + jl + b;
                                        let a_0 = alpha_compact[(2 * j) >> current_x_width];
                                        let a_1 = alpha_compact[(2 * j + 1) >> current_x_width];
                                        let m_0 = m_compact[(2 * j) & current_x_mask];
                                        let m_1 = m_compact[(2 * j + 1) & current_x_mask];
                                        rel_evals[0] += w_0 * a_0 * m_0;
                                        rel_evals[1] += w_1 * a_1 * m_1;
                                        let w_2 = w_1 + w_1 - w_0;
                                        let a_2 = a_1 + a_1 - a_0;
                                        let m_2 = m_1 + m_1 - m_0;
                                        rel_evals[2] += w_2 * a_2 * m_2;
                                    }
                                }

                                let mut entry_buf = [E::zero(); MAX_AFFINE_COEFFS];
                                let mut w_pows_buf = [E::zero(); MAX_AFFINE_COEFFS];
                                for (tail_idx, &e_in) in
                                    e_first[full_chunks * 4..].iter().enumerate()
                                {
                                    let j = base_j + full_chunks * 4 + tail_idx;
                                    let w_0 = w_full[2 * j];
                                    let w_1 = w_full[2 * j + 1];
                                    compute_entry_coeffs(
                                        &mut entry_buf,
                                        &mut w_pows_buf,
                                        range_pc,
                                        w_0,
                                        w_1 - w_0,
                                    );
                                    for (acc, &entry) in inner_accum[..num_coeffs_q]
                                        .iter_mut()
                                        .zip(entry_buf[..num_coeffs_q].iter())
                                    {
                                        *acc += e_in.mul_to_product_accum(entry);
                                    }
                                    let a_0 = alpha_compact[(2 * j) >> current_x_width];
                                    let a_1 = alpha_compact[(2 * j + 1) >> current_x_width];
                                    let m_0 = m_compact[(2 * j) & current_x_mask];
                                    let m_1 = m_compact[(2 * j + 1) & current_x_mask];
                                    rel_evals[0] += w_0 * a_0 * m_0;
                                    rel_evals[1] += w_1 * a_1 * m_1;
                                    let w_2 = w_1 + w_1 - w_0;
                                    let a_2 = a_1 + a_1 - a_0;
                                    let m_2 = m_1 + m_1 - m_0;
                                    rel_evals[2] += w_2 * a_2 * m_2;
                                }

                                let e_out = e_second[j_high];
                                for k in 0..num_coeffs_q {
                                    let inner_reduced = E::reduce_product_accum(inner_accum[k]);
                                    outer_accum[k] += e_out.mul_to_product_accum(inner_reduced);
                                }
                                (outer_accum, rel_evals)
                            },
                            |(mut ca, mut ra), (cb, rb)| {
                                for (ai, bi) in ca.iter_mut().zip(cb.iter()) {
                                    *ai += *bi;
                                }
                                for (ai, bi) in ra.iter_mut().zip(rb.iter()) {
                                    *ai += *bi;
                                }
                                (ca, ra)
                            }
                        );

                        let mut q_coeffs: Vec<E> =
                            q_coeffs.drain(..).map(E::reduce_product_accum).collect();
                        trim_trailing_zeros(&mut q_coeffs);
                        let q_poly = UniPoly::from_coeffs(q_coeffs);
                        (
                            self.split_eq.gruen_mul(&q_poly),
                            UniPoly::from_evals(&rel_evals),
                        )
                    }
                };

                self.norm_time_total += t_norm.elapsed().as_secs_f64();
                (np, rp)
            }
        };

        let max_len = norm_poly.coeffs.len().max(relation_poly.coeffs.len());
        let mut combined = vec![E::zero(); max_len];
        for (i, c) in norm_poly.coeffs.iter().enumerate() {
            combined[i] += self.batching_coeff * *c;
        }
        for (i, c) in relation_poly.coeffs.iter().enumerate() {
            combined[i] += *c;
        }
        UniPoly::from_coeffs(combined)
    }

    fn ingest_challenge(&mut self, _round: usize, r: E) {
        let t_fold = Instant::now();
        let _span = tracing::info_span!("fold_round").entered();
        self.split_eq.bind(r);

        self.w_table = match mem::replace(&mut self.w_table, WTable::Full(Vec::new())) {
            WTable::Compact(w_compact) => WTable::Full(Self::fold_compact_to_full(&w_compact, r)),
            WTable::Full(mut w_full) => {
                fold_evals_in_place(&mut w_full, r);
                WTable::Full(w_full)
            }
        };

        if self.rounds_completed < self.num_u {
            fold_evals_in_place(&mut self.m_compact, r);
        } else {
            fold_evals_in_place(&mut self.alpha_compact, r);
        }

        drop(_span);
        self.fold_time_total += t_fold.elapsed().as_secs_f64();
        self.rounds_completed += 1;

        if self.rounds_completed == self.num_vars {
            eprintln!(
                "    [fused_sc] {} rounds: norm={:.2}s, relation={:.2}s, fold={:.2}s",
                self.num_vars, self.norm_time_total, self.relation_time_total, self.fold_time_total
            );
        }
    }
}

/// Fused norm+relation sumcheck verifier.
pub struct HachiSumcheckVerifier<F: FieldCore, const D: usize> {
    batching_coeff: F,
    w_evals: Vec<F>,
    /// When set, overrides the `w_val` computed from `w_evals` in
    /// `expected_output_claim`. Used at intermediate fold levels where
    /// the full w vector is not available.
    w_val_override: Option<F>,
    tau0: Vec<F>,
    b: usize,
    alpha_evals_y: Vec<F>,
    m_evals_x: Vec<F>,
    num_u: usize,
    num_l: usize,
    relation_claim: F,
    _marker: PhantomData<[F; D]>,
}

impl<F: FieldCore + FromSmallInt, const D: usize> HachiSumcheckVerifier<F, D> {
    /// Create a fused verifier for the norm + relation sumcheck.
    #[allow(clippy::too_many_arguments)]
    #[tracing::instrument(skip_all, name = "HachiSumcheckVerifier::new")]
    pub fn new(
        batching_coeff: F,
        w_evals: Vec<F>,
        tau0: Vec<F>,
        b: usize,
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
        let y_a: Vec<F> = v
            .iter()
            .chain(u.iter())
            .chain(iter::once(&y_ring))
            .map(|r| eval_ring_at(r, &alpha))
            .collect();
        let eq_tau1 = EqPolynomial::evals(&tau1);
        let mut relation_claim = F::zero();
        for (i, eq_i) in eq_tau1.iter().enumerate() {
            let y_i = if i < y_a.len() { y_a[i] } else { F::zero() };
            relation_claim += *eq_i * y_i;
        }

        Self {
            batching_coeff,
            w_evals,
            w_val_override: None,
            tau0,
            b,
            alpha_evals_y,
            m_evals_x,
            num_u,
            num_l,
            relation_claim,
            _marker: PhantomData,
        }
    }

    /// Set the w_val override for intermediate fold levels where the
    /// full w vector is not available.
    pub fn with_w_val_override(mut self, w_val: F) -> Self {
        self.w_val_override = Some(w_val);
        self
    }
}

impl<F: FieldCore + FromSmallInt, const D: usize> SumcheckInstanceVerifier<F>
    for HachiSumcheckVerifier<F, D>
{
    fn num_rounds(&self) -> usize {
        self.num_u + self.num_l
    }

    fn degree_bound(&self) -> usize {
        2 * self.b
    }

    fn input_claim(&self) -> F {
        self.relation_claim
    }

    fn expected_output_claim(&self, challenges: &[F]) -> Result<F, HachiError> {
        let eq_val = EqPolynomial::mle(&self.tau0, challenges);
        let w_val = match self.w_val_override {
            Some(v) => v,
            None => multilinear_eval(&self.w_evals, challenges)?,
        };
        let norm_oracle = eq_val * range_check_eval(w_val, self.b);

        let (x_challenges, y_challenges) = challenges.split_at(self.num_u);
        let alpha_val = multilinear_eval(&self.alpha_evals_y, y_challenges)?;
        let m_val = multilinear_eval(&self.m_evals_x, x_challenges)?;
        let relation_oracle = w_val * alpha_val * m_val;

        eprintln!(
            "  [expected_output] num_u={}, num_l={}, w_override={}, b={}, tau0.len={}, m_evals_x.len={}, alpha_evals_y.len={}",
            self.num_u, self.num_l, self.w_val_override.is_some(), self.b,
            self.tau0.len(), self.m_evals_x.len(), self.alpha_evals_y.len()
        );

        Ok(self.batching_coeff * norm_oracle + relation_oracle)
    }
}
