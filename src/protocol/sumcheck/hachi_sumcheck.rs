//! Fused norm+relation sumcheck prover/verifier for the Hachi PCS.
//!
//! Eliminates the redundant `w_evals` clone by sharing a single `w_table`
//! across both the norm (F_0) and relation (F_α) sumcheck computations.
//! Supports compact `Vec<i8>` storage for round 0 (all entries in [-b/2, b/2)),
//! transitioning to `Vec<F>` at half size after the first fold.

use super::eq_poly::EqPolynomial;
use super::norm_sumcheck::{
    compute_norm_round_poly, compute_norm_round_poly_compact, NormRoundKernel, PointEvalPrecomp,
    RangeAffinePrecomp,
};
use super::split_eq::GruenSplitEq;
use super::{fold_evals_in_place, multilinear_eval, range_check_eval};
use super::{SumcheckInstanceProver, SumcheckInstanceVerifier, UniPoly};
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

use crate::{CanonicalField, FieldCore, FromSmallInt};

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

    norm_time_total: f64,
    relation_time_total: f64,
    fold_time_total: f64,
    rounds_completed: usize,
}

impl<E: FieldCore + FromSmallInt + CanonicalField> HachiSumcheckProver<E> {
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
        alpha_evals_y: &[E],
        m_evals_x: &[E],
        num_u: usize,
        num_l: usize,
    ) -> Self {
        assert!(b >= 1, "b must be at least 1");
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

        use super::norm_sumcheck::choose_round_kernel;
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
            alpha_table,
            m_table,
            num_vars,
            relation_claim,
            norm_time_total: 0.0,
            relation_time_total: 0.0,
            fold_time_total: 0.0,
            rounds_completed: 0,
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

    /// Unified norm sumcheck round. `w_pair(j)` returns `(w_{2j}, w_{2j+1})`.
    fn compute_round_norm(
        &self,
        half: usize,
        w_pair: impl Fn(usize) -> (E, E) + Sync,
    ) -> UniPoly<E> {
        compute_norm_round_poly(
            &self.split_eq,
            half,
            self.b,
            self.round_kernel,
            self.point_precomp.as_ref(),
            self.range_precomp.as_ref(),
            w_pair,
        )
    }

    /// Unified relation sumcheck round. `w_pair(j)` returns `(w_{2j}, w_{2j+1})`.
    fn compute_round_relation(
        &self,
        half: usize,
        w_pair: impl Fn(usize) -> (E, E) + Sync,
    ) -> UniPoly<E> {
        let evals = cfg_fold_reduce!(
            0..half,
            || [E::zero(); 3],
            |mut evals, j| {
                let (w_0, w_1) = w_pair(j);
                let a_0 = self.alpha_table[2 * j];
                let a_1 = self.alpha_table[2 * j + 1];
                let m_0 = self.m_table[2 * j];
                let m_1 = self.m_table[2 * j + 1];
                evals[0] = evals[0] + w_0 * a_0 * m_0;
                evals[1] = evals[1] + w_1 * a_1 * m_1;
                let w_2 = w_1 + w_1 - w_0;
                let a_2 = a_1 + a_1 - a_0;
                let m_2 = m_1 + m_1 - m_0;
                evals[2] = evals[2] + w_2 * a_2 * m_2;
                evals
            },
            |mut a, b| {
                for (ai, bi) in a.iter_mut().zip(b.iter()) {
                    *ai = *ai + *bi;
                }
                a
            }
        );
        UniPoly::from_evals(&evals)
    }

    fn fold_compact_to_full(w_compact: &[i8], r: E) -> Vec<E> {
        cfg_into_iter!(0..w_compact.len() / 2)
            .map(|j| {
                let w_0 = Self::lift_i8(w_compact[2 * j]);
                let w_1 = Self::lift_i8(w_compact[2 * j + 1]);
                w_0 + r * (w_1 - w_0)
            })
            .collect()
    }
}

impl<E: FieldCore + FromSmallInt + CanonicalField> SumcheckInstanceProver<E>
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
                let half = w_compact.len() / 2;
                let np = {
                    let _span = tracing::info_span!("norm_round").entered();
                    compute_norm_round_poly_compact(
                        &self.split_eq,
                        w_compact,
                        self.b,
                        self.round_kernel,
                        self.point_precomp.as_ref(),
                        self.range_precomp.as_ref(),
                    )
                };
                let norm_elapsed = t_norm.elapsed().as_secs_f64();
                self.norm_time_total += norm_elapsed;

                let t_rel = Instant::now();
                let pair = |j: usize| {
                    (
                        Self::lift_i8(w_compact[2 * j]),
                        Self::lift_i8(w_compact[2 * j + 1]),
                    )
                };
                let rp = {
                    let _span = tracing::info_span!("relation_round").entered();
                    self.compute_round_relation(half, pair)
                };
                self.relation_time_total += t_rel.elapsed().as_secs_f64();
                (np, rp)
            }
            WTable::Full(w_full) => {
                let half = w_full.len() / 2;
                let pair = |j: usize| (w_full[2 * j], w_full[2 * j + 1]);
                let np = {
                    let _span = tracing::info_span!("norm_round").entered();
                    self.compute_round_norm(half, pair)
                };
                let norm_elapsed = t_norm.elapsed().as_secs_f64();
                self.norm_time_total += norm_elapsed;

                let t_rel = Instant::now();
                let rp = {
                    let _span = tracing::info_span!("relation_round").entered();
                    self.compute_round_relation(half, pair)
                };
                self.relation_time_total += t_rel.elapsed().as_secs_f64();
                (np, rp)
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

        fold_evals_in_place(&mut self.alpha_table, r);
        fold_evals_in_place(&mut self.m_table, r);
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
            relation_claim = relation_claim + *eq_i * y_i;
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

        Ok(self.batching_coeff * norm_oracle + relation_oracle)
    }
}
