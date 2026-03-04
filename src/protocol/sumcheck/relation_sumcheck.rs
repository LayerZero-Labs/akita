//! Evaluation-relation sumcheck instance (F_α).
//!
//! **F_{α,τ₁}(x, y)** = w̃(x,y) · α̃(y) · m(x)
//! where m(x) = Σ_i ẽq(τ₁,i) · M̃_α(i,x).
//!
//! Proves the evaluation relation; sum equals `a = Σ_i ẽq(τ₁,i) · y_i(α)`.

use super::eq_poly::EqPolynomial;
use super::{fold_evals_in_place, multilinear_eval};
use super::{SumcheckInstanceProver, SumcheckInstanceVerifier, UniPoly};
use crate::algebra::ring::CyclotomicRing;
use crate::error::HachiError;
#[cfg(feature = "parallel")]
use crate::parallel::*;
use crate::protocol::ring_switch::eval_ring_at;
use crate::{FieldCore, FromSmallInt};
use std::iter;

/// Prover for `F_{α,τ₁}(x,y) = w̃(x,y) · α̃(y) · m(x)`.
///
/// Alpha and m are stored in compact form (sizes `2^num_l` and `2^num_u`)
/// and folded only during rounds where their variables are active.
///
/// Round polynomial degree is 2 (product of at most two multilinear factors
/// depending on any single variable).
pub struct RelationSumcheckProver<E> {
    w_table: Vec<E>,
    alpha_compact: Vec<E>,
    m_compact: Vec<E>,
    num_u: usize,
    num_vars: usize,
    rounds_completed: usize,
}

impl<E: FieldCore + FromSmallInt> RelationSumcheckProver<E> {
    /// Construct from the three constituent evaluation tables.
    ///
    /// - `w_evals`: evaluations of `w̃` over `{0,1}^{num_u + num_l}` (full domain).
    /// - `alpha_evals_y`: evaluations of `α̃` over `{0,1}^{num_l}` (compact).
    /// - `m_evals_x`: evaluations of `m` over `{0,1}^{num_u}` (compact).
    ///
    /// # Panics
    ///
    /// Panics if table sizes don't match `2^num_u`, `2^num_l`, or `2^(num_u+num_l)`.
    pub fn new(
        w_evals: Vec<E>,
        alpha_evals_y: &[E],
        m_evals_x: &[E],
        num_u: usize,
        num_l: usize,
    ) -> Self {
        let num_vars = num_u + num_l;
        let n = 1usize << num_vars;
        assert_eq!(w_evals.len(), n);
        assert_eq!(alpha_evals_y.len(), 1 << num_l);
        assert_eq!(m_evals_x.len(), 1 << num_u);

        Self {
            w_table: w_evals,
            alpha_compact: alpha_evals_y.to_vec(),
            m_compact: m_evals_x.to_vec(),
            num_u,
            num_vars,
            rounds_completed: 0,
        }
    }
}

impl<E: FieldCore + FromSmallInt> SumcheckInstanceProver<E> for RelationSumcheckProver<E> {
    fn num_rounds(&self) -> usize {
        self.num_vars
    }

    fn degree_bound(&self) -> usize {
        2
    }

    fn input_claim(&self) -> E {
        let x_mask = (1usize << self.num_u) - 1;
        let alpha_compact = &self.alpha_compact;
        let m_compact = &self.m_compact;
        let num_u = self.num_u;

        #[cfg(feature = "parallel")]
        {
            self.w_table
                .par_iter()
                .enumerate()
                .fold(
                    || E::zero(),
                    |acc, (idx, &w)| {
                        acc + w * alpha_compact[idx >> num_u] * m_compact[idx & x_mask]
                    },
                )
                .reduce(|| E::zero(), |a, b| a + b)
        }
        #[cfg(not(feature = "parallel"))]
        {
            self.w_table
                .iter()
                .enumerate()
                .fold(E::zero(), |acc, (idx, &w)| {
                    acc + w * alpha_compact[idx >> num_u] * m_compact[idx & x_mask]
                })
        }
    }

    fn compute_round_univariate(&mut self, _round: usize, _previous_claim: E) -> UniPoly<E> {
        let half = self.w_table.len() / 2;
        let num_points = 3;
        let current_x_width = self.num_u.saturating_sub(self.rounds_completed);
        let current_x_mask = (1usize << current_x_width).wrapping_sub(1);
        let alpha_compact = &self.alpha_compact;
        let m_compact = &self.m_compact;

        #[cfg(feature = "parallel")]
        let round_evals = {
            (0..half)
                .into_par_iter()
                .fold(
                    || vec![E::zero(); num_points],
                    |mut evals, j| {
                        let w_0 = self.w_table[2 * j];
                        let w_1 = self.w_table[2 * j + 1];
                        let a_0 = alpha_compact[(2 * j) >> current_x_width];
                        let a_1 = alpha_compact[(2 * j + 1) >> current_x_width];
                        let m_0 = m_compact[(2 * j) & current_x_mask];
                        let m_1 = m_compact[(2 * j + 1) & current_x_mask];
                        for (t, eval) in evals.iter_mut().enumerate() {
                            let t_e = E::from_u64(t as u64);
                            let w_t = w_0 + t_e * (w_1 - w_0);
                            let a_t = a_0 + t_e * (a_1 - a_0);
                            let m_t = m_0 + t_e * (m_1 - m_0);
                            *eval += w_t * a_t * m_t;
                        }
                        evals
                    },
                )
                .reduce(
                    || vec![E::zero(); num_points],
                    |mut a, b| {
                        for (ai, bi) in a.iter_mut().zip(b.iter()) {
                            *ai += *bi;
                        }
                        a
                    },
                )
        };
        #[cfg(not(feature = "parallel"))]
        let round_evals = {
            let mut evals = vec![E::zero(); num_points];
            for j in 0..half {
                let w_0 = self.w_table[2 * j];
                let w_1 = self.w_table[2 * j + 1];
                let a_0 = alpha_compact[(2 * j) >> current_x_width];
                let a_1 = alpha_compact[(2 * j + 1) >> current_x_width];
                let m_0 = m_compact[(2 * j) & current_x_mask];
                let m_1 = m_compact[(2 * j + 1) & current_x_mask];
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

    fn ingest_challenge(&mut self, _round: usize, r: E) {
        fold_evals_in_place(&mut self.w_table, r);
        if self.rounds_completed < self.num_u {
            fold_evals_in_place(&mut self.m_compact, r);
        } else {
            fold_evals_in_place(&mut self.alpha_compact, r);
        }
        self.rounds_completed += 1;
    }
}

/// Verifier for the evaluation-relation sumcheck `F_{α,τ₁}`.
pub struct RelationSumcheckVerifier<F: FieldCore, const D: usize> {
    w_evals: Vec<F>,
    alpha_evals_y: Vec<F>,
    m_evals_x: Vec<F>,
    tau: Vec<F>,
    v: Vec<CyclotomicRing<F, D>>,
    u: Vec<CyclotomicRing<F, D>>,
    y_ring: CyclotomicRing<F, D>,
    alpha: F,
    num_u: usize,
    num_l: usize,
}

impl<F: FieldCore, const D: usize> RelationSumcheckVerifier<F, D> {
    /// Create a new evaluation-relation sumcheck verifier.
    ///
    /// # Panics
    ///
    /// Panics if table sizes don't match `2^num_u`, `2^num_l`, or `2^(num_u+num_l)`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        w_evals: Vec<F>,
        alpha_evals_y: Vec<F>,
        m_evals_x: Vec<F>,
        tau: Vec<F>,
        v: Vec<CyclotomicRing<F, D>>,
        u: Vec<CyclotomicRing<F, D>>,
        y_ring: CyclotomicRing<F, D>,
        alpha: F,
        num_u: usize,
        num_l: usize,
    ) -> Self {
        assert_eq!(w_evals.len(), 1 << (num_u + num_l));
        assert_eq!(alpha_evals_y.len(), 1 << num_l);
        assert_eq!(m_evals_x.len(), 1 << num_u);
        Self {
            w_evals,
            alpha_evals_y,
            m_evals_x,
            tau,
            v,
            u,
            y_ring,
            alpha,
            num_u,
            num_l,
        }
    }
}

impl<F: FieldCore, const D: usize> SumcheckInstanceVerifier<F> for RelationSumcheckVerifier<F, D> {
    fn num_rounds(&self) -> usize {
        self.num_u + self.num_l
    }

    fn degree_bound(&self) -> usize {
        2
    }

    fn input_claim(&self) -> F {
        let y_a: Vec<F> = self
            .v
            .iter()
            .chain(self.u.iter())
            .chain(iter::once(&self.y_ring))
            .map(|r| eval_ring_at(r, &self.alpha))
            .collect();

        let eq_tau = EqPolynomial::evals(&self.tau);
        let mut acc = F::zero();
        for (i, eq_i) in eq_tau.iter().enumerate() {
            let y_i = if i < y_a.len() { y_a[i] } else { F::zero() };
            acc += *eq_i * y_i;
        }
        acc
    }

    fn expected_output_claim(&self, challenges: &[F]) -> Result<F, HachiError> {
        let (x_challenges, y_challenges) = challenges.split_at(self.num_u);
        let w_val = multilinear_eval(&self.w_evals, challenges)?;
        let alpha_val = multilinear_eval(&self.alpha_evals_y, y_challenges)?;
        let m_val = multilinear_eval(&self.m_evals_x, x_challenges)?;
        Ok(w_val * alpha_val * m_val)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::Fp64;
    use crate::protocol::commitment_scheme::rederive_alpha_and_m_a;
    use crate::protocol::hachi_poly_ops::DensePoly;
    use crate::protocol::opening_point::BasisMode;
    use crate::protocol::sumcheck::eq_poly::EqPolynomial;
    use crate::protocol::transcript::labels;
    use crate::protocol::{
        prove_sumcheck, verify_sumcheck, Blake2bTranscript, CommitmentConfig, CommitmentScheme,
        HachiCommitmentScheme, SmallTestCommitmentConfig, Transcript,
    };
    use crate::{FieldCore, FromSmallInt};

    type F = Fp64<4294967197>;
    type Cfg = SmallTestCommitmentConfig;
    const D: usize = Cfg::D;
    type Scheme = HachiCommitmentScheme<D, Cfg>;

    #[test]
    fn relation_sumcheck_uses_prove_w_evals() {
        let alpha_bits = D.trailing_zeros() as usize;
        let layout = SmallTestCommitmentConfig::commitment_layout(8).unwrap();
        let num_vars = layout.m_vars + layout.r_vars + alpha_bits;
        let len = 1usize << num_vars;
        let evals: Vec<F> = (0..len).map(|i| F::from_u64(i as u64)).collect();
        let poly = DensePoly::<F, D>::from_field_evals(num_vars, &evals).unwrap();

        let setup = Scheme::setup_prover(num_vars);
        let (commitment, hint) = Scheme::commit(&poly, &setup, &layout).unwrap();

        let opening_point: Vec<F> = (0..num_vars).map(|i| F::from_u64((i + 2) as u64)).collect();
        let mut prover_transcript = Blake2bTranscript::<F>::new(labels::DOMAIN_HACHI_PROTOCOL);
        let proof = Scheme::prove(
            &setup,
            &poly,
            &opening_point,
            hint,
            &mut prover_transcript,
            &commitment,
            BasisMode::Lagrange,
            &layout,
        )
        .unwrap();

        let (alpha, m_a_vec) = rederive_alpha_and_m_a::<F, { Cfg::D }, Cfg>(
            &proof,
            &Scheme::setup_verifier(&setup),
            &opening_point,
            &commitment,
        )
        .unwrap();

        let final_w: Vec<F> = proof.final_w.to_field_elems();
        let d = SmallTestCommitmentConfig::D;
        assert_eq!(final_w.len() % d, 0);
        let w_u = final_w.len() / d;
        let rows = SmallTestCommitmentConfig::N_D
            + SmallTestCommitmentConfig::N_B
            + 1
            + 1
            + SmallTestCommitmentConfig::N_A;
        assert!(rows > 0);
        assert_eq!(m_a_vec.len() % rows, 0);
        let cols = m_a_vec.len() / rows;
        assert_eq!(w_u, cols);

        let num_u = cols.next_power_of_two().trailing_zeros() as usize;
        let num_l = alpha_bits;
        let n = 1usize << (num_u + num_l);

        let mut w_evals = vec![F::zero(); n];
        let y_len = 1usize << num_l;
        let x_len = 1usize << num_u;
        for x in 0..x_len {
            for y in 0..y_len {
                let src = y + (x << num_l);
                if src < final_w.len() {
                    let dst = x + (y << num_u);
                    w_evals[dst] = final_w[src];
                }
            }
        }

        let num_i = rows.next_power_of_two().trailing_zeros() as usize;
        let tau1: Vec<F> = (0..num_i).map(|i| F::from_u64((i + 5) as u64)).collect();
        let eq_tau1 = EqPolynomial::evals(&tau1);

        let mut m_evals_x = vec![F::zero(); x_len];
        for x in 0..x_len {
            let mut acc = F::zero();
            for i in 0..(1usize << num_i) {
                let row_val = if i < rows && x < cols {
                    m_a_vec[i * cols + x]
                } else {
                    F::zero()
                };
                acc += eq_tau1[i] * row_val;
            }
            m_evals_x[x] = acc;
        }

        let mut alpha_evals_y = vec![F::zero(); y_len];
        let mut power = F::one();
        for val in alpha_evals_y.iter_mut() {
            *val = power;
            power = power * alpha;
        }

        let x_mask = x_len - 1;
        let alpha_full: Vec<F> = (0..n).map(|idx| alpha_evals_y[idx >> num_u]).collect();
        let m_full: Vec<F> = (0..n).map(|idx| m_evals_x[idx & x_mask]).collect();
        let _claim: F = (0..n)
            .map(|i| w_evals[i] * alpha_full[i] * m_full[i])
            .fold(F::zero(), |a, v| a + v);

        let mut prover =
            RelationSumcheckProver::new(w_evals.clone(), &alpha_evals_y, &m_evals_x, num_u, num_l);
        let mut pt = Blake2bTranscript::<F>::new(labels::DOMAIN_HACHI_PROTOCOL);
        let (proof_sc, prover_challenges, final_claim) =
            prove_sumcheck::<F, _, F, _, _>(&mut prover, &mut pt, |tr| {
                tr.challenge_scalar(labels::CHALLENGE_SUMCHECK_ROUND)
            })
            .unwrap();

        let (x_ch, y_ch) = prover_challenges.split_at(num_u);
        let oracle = multilinear_eval(&w_evals, &prover_challenges).unwrap()
            * multilinear_eval(&alpha_evals_y, y_ch).unwrap()
            * multilinear_eval(&m_evals_x, x_ch).unwrap();
        assert_eq!(final_claim, oracle, "prover final claim != oracle eval");

        let verifier = RelationSumcheckVerifier::new(
            w_evals,
            alpha_evals_y,
            m_evals_x,
            tau1,
            proof.levels[0].v.clone(),
            commitment.u.clone(),
            proof.levels[0].y_ring,
            alpha,
            num_u,
            num_l,
        );
        let mut vt = Blake2bTranscript::<F>::new(labels::DOMAIN_HACHI_PROTOCOL);
        let verifier_challenges =
            verify_sumcheck::<F, _, F, _, _>(&proof_sc, &verifier, &mut vt, |tr| {
                tr.challenge_scalar(labels::CHALLENGE_SUMCHECK_ROUND)
            })
            .unwrap();

        assert_eq!(prover_challenges, verifier_challenges);
    }
}
