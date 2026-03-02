//! Fused norm+relation sumcheck prover/verifier for the Hachi PCS.
//!
//! Eliminates the redundant `w_evals` clone by sharing a single `w_table`
//! across both the norm (F_0) and relation (F_α) sumcheck computations.
//! Supports compact `Vec<i8>` storage for round 0 (all entries in [-8, 7]),
//! transitioning to `Vec<F>` at half size after the first fold.

use super::eq_poly::EqPolynomial;
use super::norm_sumcheck::{
    accumulate_affine_range_coeffs, range_check_eval_precomputed, trim_trailing_zeros,
    NormRoundKernel, PointEvalPrecomp, RangeAffinePrecomp,
};
use super::split_eq::GruenSplitEq;
use super::{fold_evals_in_place, multilinear_eval, range_check_eval};
use super::{SumcheckInstanceProver, SumcheckInstanceVerifier, UniPoly};
use crate::algebra::CyclotomicRing;
use crate::cfg_into_iter;
use crate::error::HachiError;
#[cfg(feature = "parallel")]
use crate::parallel::*;
use crate::protocol::ring_switch::eval_ring_at;
use crate::{FieldCore, FromSmallInt};

enum WTable<E: FieldCore> {
    Compact(Vec<i8>),
    Full(Vec<E>),
}

/// Fused norm+relation sumcheck prover.
///
/// Holds a single `w_table` shared by both sumcheck instances, weighted
/// by `batching_coeff`. The round polynomial is
/// `batching_coeff * norm_round(t) + relation_round(t)`.
pub struct HachiSumcheckProver<E: FieldCore> {
    w_table: WTable<E>,
    batching_coeff: E,

    // Norm state
    split_eq: GruenSplitEq<E>,
    round_kernel: NormRoundKernel,
    point_precomp: Option<PointEvalPrecomp<E>>,
    range_precomp: Option<RangeAffinePrecomp<E>>,
    b: usize,

    // Relation state
    alpha_table: Vec<E>,
    m_table: Vec<E>,

    num_vars: usize,
    relation_claim: E,
}

impl<E: FieldCore + FromSmallInt> HachiSumcheckProver<E> {
    /// Create a fused norm+relation sumcheck prover.
    ///
    /// # Panics
    ///
    /// Panics if table sizes are inconsistent with `num_u` and `num_l`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        batching_coeff: E,
        w_evals_compact: Vec<i8>,
        tau0: &[E],
        b: usize,
        alpha_evals_y: &[E],
        m_evals_x: &[E],
        num_u: usize,
        num_l: usize,
    ) -> Self {
        let num_vars = num_u + num_l;
        let n = 1usize << num_vars;
        assert_eq!(w_evals_compact.len(), n);
        assert_eq!(tau0.len(), num_vars);
        assert_eq!(alpha_evals_y.len(), 1 << num_l);
        assert_eq!(m_evals_x.len(), 1 << num_u);

        let x_mask = (1usize << num_u) - 1;
        let alpha_table: Vec<E> = cfg_into_iter!(0..n)
            .map(|idx| alpha_evals_y[idx >> num_u])
            .collect();
        let m_table: Vec<E> = cfg_into_iter!(0..n)
            .map(|idx| m_evals_x[idx & x_mask])
            .collect();

        let relation_claim =
            Self::compute_relation_claim_compact(&w_evals_compact, &alpha_table, &m_table);

        let round_kernel = if b <= 8 {
            NormRoundKernel::PointEvalInterpolation
        } else {
            NormRoundKernel::AffineCoeffComposition
        };
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
            alpha_table,
            m_table,
            num_vars,
            relation_claim,
        }
    }

    fn compute_relation_claim_compact(w_compact: &[i8], alpha_table: &[E], m_table: &[E]) -> E {
        w_compact
            .iter()
            .zip(alpha_table.iter())
            .zip(m_table.iter())
            .fold(E::zero(), |acc, ((&w, &a), &m)| {
                acc + E::from_i64(w as i64) * a * m
            })
    }

    fn lift_i8(v: i8) -> E {
        E::from_i64(v as i64)
    }

    fn compute_round_norm_compact(&self, w_compact: &[i8]) -> UniPoly<E> {
        let half = w_compact.len() / 2;

        let (e_first, e_second) = self.split_eq.remaining_eq_tables();
        let num_first = e_first.len();
        let first_bits = num_first.trailing_zeros();

        match self.round_kernel {
            NormRoundKernel::PointEvalInterpolation => {
                let degree_q = 2 * self.b - 1;
                let num_points_q = degree_q + 1;
                let range_offsets = &self.point_precomp.as_ref().unwrap().range_offsets;

                let mut q_evals = vec![E::zero(); num_points_q];
                for j in 0..half {
                    let j_low = j & (num_first - 1);
                    let j_high = j >> first_bits;
                    let eq_rem = e_first[j_low] * e_second[j_high];
                    let w_0 = Self::lift_i8(w_compact[2 * j]);
                    let w_1 = Self::lift_i8(w_compact[2 * j + 1]);
                    let delta = w_1 - w_0;
                    let mut w_t = w_0;
                    for eval in q_evals.iter_mut() {
                        *eval = *eval + eq_rem * range_check_eval_precomputed(w_t, range_offsets);
                        w_t = w_t + delta;
                    }
                }
                let q_poly = UniPoly::from_evals(&q_evals);
                self.split_eq.gruen_mul(&q_poly)
            }
            NormRoundKernel::AffineCoeffComposition => {
                let range_precomp = self.range_precomp.as_ref().unwrap();
                let num_coeffs_q = range_precomp.degree_q + 1;
                let coeff_mix = &range_precomp.coeff_mix;

                let mut q_coeffs = vec![E::zero(); num_coeffs_q];
                for j in 0..half {
                    let j_low = j & (num_first - 1);
                    let j_high = j >> first_bits;
                    let eq_rem = e_first[j_low] * e_second[j_high];
                    let w_0 = Self::lift_i8(w_compact[2 * j]);
                    let w_1 = Self::lift_i8(w_compact[2 * j + 1]);
                    let a = w_1 - w_0;
                    accumulate_affine_range_coeffs(&mut q_coeffs, coeff_mix, w_0, a, eq_rem);
                }
                trim_trailing_zeros(&mut q_coeffs);
                let q_poly = UniPoly::from_coeffs(q_coeffs);
                self.split_eq.gruen_mul(&q_poly)
            }
        }
    }

    fn compute_round_relation_compact(&self, w_compact: &[i8]) -> UniPoly<E> {
        let half = w_compact.len() / 2;
        let num_points = 3;

        let mut evals = vec![E::zero(); num_points];
        for j in 0..half {
            let w_0 = Self::lift_i8(w_compact[2 * j]);
            let w_1 = Self::lift_i8(w_compact[2 * j + 1]);
            let a_0 = self.alpha_table[2 * j];
            let a_1 = self.alpha_table[2 * j + 1];
            let m_0 = self.m_table[2 * j];
            let m_1 = self.m_table[2 * j + 1];
            for (t, eval) in evals.iter_mut().enumerate() {
                let t_e = E::from_u64(t as u64);
                let w_t = w_0 + t_e * (w_1 - w_0);
                let a_t = a_0 + t_e * (a_1 - a_0);
                let m_t = m_0 + t_e * (m_1 - m_0);
                *eval = *eval + w_t * a_t * m_t;
            }
        }
        UniPoly::from_evals(&evals)
    }

    fn compute_round_norm_full(&self, w_full: &[E]) -> UniPoly<E> {
        let half = w_full.len() / 2;

        let (e_first, e_second) = self.split_eq.remaining_eq_tables();
        let num_first = e_first.len();
        let first_bits = num_first.trailing_zeros();

        match self.round_kernel {
            NormRoundKernel::PointEvalInterpolation => {
                let degree_q = 2 * self.b - 1;
                let num_points_q = degree_q + 1;
                let range_offsets = &self.point_precomp.as_ref().unwrap().range_offsets;

                #[cfg(feature = "parallel")]
                let q_evals = {
                    (0..half)
                        .into_par_iter()
                        .fold(
                            || vec![E::zero(); num_points_q],
                            |mut evals, j| {
                                let j_low = j & (num_first - 1);
                                let j_high = j >> first_bits;
                                let eq_rem = e_first[j_low] * e_second[j_high];
                                let w_0 = w_full[2 * j];
                                let w_1 = w_full[2 * j + 1];
                                let delta = w_1 - w_0;
                                let mut w_t = w_0;
                                for eval in evals.iter_mut() {
                                    *eval = *eval
                                        + eq_rem * range_check_eval_precomputed(w_t, range_offsets);
                                    w_t = w_t + delta;
                                }
                                evals
                            },
                        )
                        .reduce(
                            || vec![E::zero(); num_points_q],
                            |mut a, b_vec| {
                                for (ai, bi) in a.iter_mut().zip(b_vec.iter()) {
                                    *ai = *ai + *bi;
                                }
                                a
                            },
                        )
                };
                #[cfg(not(feature = "parallel"))]
                let q_evals = {
                    let mut evals = vec![E::zero(); num_points_q];
                    for j in 0..half {
                        let j_low = j & (num_first - 1);
                        let j_high = j >> first_bits;
                        let eq_rem = e_first[j_low] * e_second[j_high];
                        let w_0 = w_full[2 * j];
                        let w_1 = w_full[2 * j + 1];
                        let delta = w_1 - w_0;
                        let mut w_t = w_0;
                        for eval in evals.iter_mut() {
                            *eval =
                                *eval + eq_rem * range_check_eval_precomputed(w_t, range_offsets);
                            w_t = w_t + delta;
                        }
                    }
                    evals
                };

                let q_poly = UniPoly::from_evals(&q_evals);
                self.split_eq.gruen_mul(&q_poly)
            }
            NormRoundKernel::AffineCoeffComposition => {
                let range_precomp = self.range_precomp.as_ref().unwrap();
                let num_coeffs_q = range_precomp.degree_q + 1;
                let coeff_mix = &range_precomp.coeff_mix;

                #[cfg(feature = "parallel")]
                let mut q_coeffs = {
                    (0..half)
                        .into_par_iter()
                        .fold(
                            || vec![E::zero(); num_coeffs_q],
                            |mut coeffs, j| {
                                let j_low = j & (num_first - 1);
                                let j_high = j >> first_bits;
                                let eq_rem = e_first[j_low] * e_second[j_high];
                                let w_0 = w_full[2 * j];
                                let w_1 = w_full[2 * j + 1];
                                let a = w_1 - w_0;
                                accumulate_affine_range_coeffs(
                                    &mut coeffs,
                                    coeff_mix,
                                    w_0,
                                    a,
                                    eq_rem,
                                );
                                coeffs
                            },
                        )
                        .reduce(
                            || vec![E::zero(); num_coeffs_q],
                            |mut a, b_vec| {
                                for (ai, bi) in a.iter_mut().zip(b_vec.iter()) {
                                    *ai = *ai + *bi;
                                }
                                a
                            },
                        )
                };
                #[cfg(not(feature = "parallel"))]
                let mut q_coeffs = {
                    let mut coeffs = vec![E::zero(); num_coeffs_q];
                    for j in 0..half {
                        let j_low = j & (num_first - 1);
                        let j_high = j >> first_bits;
                        let eq_rem = e_first[j_low] * e_second[j_high];
                        let w_0 = w_full[2 * j];
                        let w_1 = w_full[2 * j + 1];
                        let a = w_1 - w_0;
                        accumulate_affine_range_coeffs(&mut coeffs, coeff_mix, w_0, a, eq_rem);
                    }
                    coeffs
                };

                trim_trailing_zeros(&mut q_coeffs);
                let q_poly = UniPoly::from_coeffs(q_coeffs);
                self.split_eq.gruen_mul(&q_poly)
            }
        }
    }

    fn compute_round_relation_full(&self, w_full: &[E]) -> UniPoly<E> {
        let half = w_full.len() / 2;
        let num_points = 3;

        #[cfg(feature = "parallel")]
        let round_evals = {
            (0..half)
                .into_par_iter()
                .fold(
                    || vec![E::zero(); num_points],
                    |mut evals, j| {
                        let w_0 = w_full[2 * j];
                        let w_1 = w_full[2 * j + 1];
                        let a_0 = self.alpha_table[2 * j];
                        let a_1 = self.alpha_table[2 * j + 1];
                        let m_0 = self.m_table[2 * j];
                        let m_1 = self.m_table[2 * j + 1];
                        for (t, eval) in evals.iter_mut().enumerate() {
                            let t_e = E::from_u64(t as u64);
                            let w_t = w_0 + t_e * (w_1 - w_0);
                            let a_t = a_0 + t_e * (a_1 - a_0);
                            let m_t = m_0 + t_e * (m_1 - m_0);
                            *eval = *eval + w_t * a_t * m_t;
                        }
                        evals
                    },
                )
                .reduce(
                    || vec![E::zero(); num_points],
                    |mut a, b| {
                        for (ai, bi) in a.iter_mut().zip(b.iter()) {
                            *ai = *ai + *bi;
                        }
                        a
                    },
                )
        };
        #[cfg(not(feature = "parallel"))]
        let round_evals = {
            let mut evals = vec![E::zero(); num_points];
            for j in 0..half {
                let w_0 = w_full[2 * j];
                let w_1 = w_full[2 * j + 1];
                let a_0 = self.alpha_table[2 * j];
                let a_1 = self.alpha_table[2 * j + 1];
                let m_0 = self.m_table[2 * j];
                let m_1 = self.m_table[2 * j + 1];
                for (t, eval) in evals.iter_mut().enumerate() {
                    let t_e = E::from_u64(t as u64);
                    let w_t = w_0 + t_e * (w_1 - w_0);
                    let a_t = a_0 + t_e * (a_1 - a_0);
                    let m_t = m_0 + t_e * (m_1 - m_0);
                    *eval = *eval + w_t * a_t * m_t;
                }
            }
            evals
        };

        UniPoly::from_evals(&round_evals)
    }

    fn fold_compact_to_full(w_compact: &[i8], r: E) -> Vec<E> {
        let half = w_compact.len() / 2;
        let mut out = Vec::with_capacity(half);
        for j in 0..half {
            let w_0 = Self::lift_i8(w_compact[2 * j]);
            let w_1 = Self::lift_i8(w_compact[2 * j + 1]);
            out.push(w_0 + r * (w_1 - w_0));
        }
        out
    }
}

impl<E: FieldCore + FromSmallInt> SumcheckInstanceProver<E> for HachiSumcheckProver<E> {
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
        let (norm_poly, relation_poly) = match &self.w_table {
            WTable::Compact(w_compact) => {
                let norm = self.compute_round_norm_compact(w_compact);
                let relation = self.compute_round_relation_compact(w_compact);
                (norm, relation)
            }
            WTable::Full(w_full) => {
                let norm = self.compute_round_norm_full(w_full);
                let relation = self.compute_round_relation_full(w_full);
                (norm, relation)
            }
        };

        let max_len = norm_poly.coeffs.len().max(relation_poly.coeffs.len());
        let mut combined = vec![E::zero(); max_len];
        for (i, c) in norm_poly.coeffs.iter().enumerate() {
            combined[i] = combined[i] + self.batching_coeff * *c;
        }
        for (i, c) in relation_poly.coeffs.iter().enumerate() {
            combined[i] = combined[i] + *c;
        }
        UniPoly::from_coeffs(combined)
    }

    fn ingest_challenge(&mut self, _round: usize, r: E) {
        self.split_eq.bind(r);

        self.w_table = match std::mem::replace(&mut self.w_table, WTable::Full(Vec::new())) {
            WTable::Compact(w_compact) => WTable::Full(Self::fold_compact_to_full(&w_compact, r)),
            WTable::Full(mut w_full) => {
                fold_evals_in_place(&mut w_full, r);
                WTable::Full(w_full)
            }
        };

        fold_evals_in_place(&mut self.alpha_table, r);
        fold_evals_in_place(&mut self.m_table, r);
    }
}

/// Fused norm+relation sumcheck verifier.
pub struct HachiSumcheckVerifier<F: FieldCore, const D: usize> {
    batching_coeff: F,
    w_evals: Vec<F>,
    tau0: Vec<F>,
    b: usize,
    alpha_evals_y: Vec<F>,
    m_evals_x: Vec<F>,
    num_u: usize,
    num_l: usize,
    relation_claim: F,
    _marker: std::marker::PhantomData<[F; D]>,
}

impl<F: FieldCore + FromSmallInt, const D: usize> HachiSumcheckVerifier<F, D> {
    /// Create a fused verifier for the norm + relation sumcheck.
    #[allow(clippy::too_many_arguments)]
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
            .chain(std::iter::once(&y_ring))
            .map(|r| eval_ring_at(r, &alpha))
            .collect();
        let eq_tau1 = EqPolynomial::evals(&tau1);
        let mut relation_claim = F::zero();
        for (i, eq_i) in eq_tau1.iter().enumerate() {
            let y_i = if i < y_a.len() { y_a[i] } else { F::zero() };
            relation_claim = relation_claim + *eq_i * y_i;
        }

        Self {
            batching_coeff,
            w_evals,
            tau0,
            b,
            alpha_evals_y,
            m_evals_x,
            num_u,
            num_l,
            relation_claim,
            _marker: std::marker::PhantomData,
        }
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
        let w_val = multilinear_eval(&self.w_evals, challenges)?;
        let norm_oracle = eq_val * range_check_eval(w_val, self.b);

        let (x_challenges, y_challenges) = challenges.split_at(self.num_u);
        let alpha_val = multilinear_eval(&self.alpha_evals_y, y_challenges)?;
        let m_val = multilinear_eval(&self.m_evals_x, x_challenges)?;
        let relation_oracle = w_val * alpha_val * m_val;

        Ok(self.batching_coeff * norm_oracle + relation_oracle)
    }
}
