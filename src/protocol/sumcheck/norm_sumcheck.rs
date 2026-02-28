//! Norm (range-check) sumcheck instance (F_0).
//!
//! **F_{0,τ₀}(x, y)** = ẽq(τ₀,(x,y)) · w̃(x,y) · (w̃−1)(w̃+1)···(w̃−b+1)(w̃+b−1)
//!
//! Proves that all entries of w̃ lie in {−(b−1), …, b−1}; the sum over the
//! boolean hypercube should equal zero when the range constraint holds.

use super::eq_poly::EqPolynomial;
use super::split_eq::GruenSplitEq;
use super::{fold_evals, multilinear_eval, range_check_eval};
use super::{SumcheckInstanceProver, SumcheckInstanceVerifier, UniPoly};
#[cfg(feature = "parallel")]
use crate::parallel::*;
use crate::{CanonicalField, FieldCore};

/// Prover for `F_{0,τ₀}(x,y) = ẽq(τ₀,(x,y)) · w̃(x,y) · range_check(w̃(x,y), b)`.
///
/// Uses the Gruen/Dao-Thaler optimization: the eq polynomial is factored into
/// a running scalar and split tables instead of being stored as a full table
/// and folded each round. The round polynomial is computed as `l(X) · q(X)`
/// where `l(X)` is the linear eq factor and `q(X)` is the inner sum without
/// the current-variable eq contribution.
pub struct NormSumcheckProver<E: FieldCore> {
    split_eq: GruenSplitEq<E>,
    w_table: Vec<E>,
    num_vars: usize,
    b: usize,
}

impl<E: FieldCore + CanonicalField> NormSumcheckProver<E> {
    /// Create a new norm (range-check) sumcheck prover.
    ///
    /// # Panics
    ///
    /// Panics if `w_evals.len() != 2^tau.len()`.
    pub fn new(tau: &[E], w_evals: Vec<E>, b: usize) -> Self {
        let num_vars = tau.len();
        assert_eq!(w_evals.len(), 1 << num_vars);
        Self {
            split_eq: GruenSplitEq::new(tau),
            w_table: w_evals,
            num_vars,
            b,
        }
    }
}

impl<E: FieldCore + CanonicalField> SumcheckInstanceProver<E> for NormSumcheckProver<E> {
    fn num_rounds(&self) -> usize {
        self.num_vars
    }

    fn degree_bound(&self) -> usize {
        2 * self.b
    }

    fn input_claim(&self) -> E {
        E::zero()
    }

    fn compute_round_univariate(&mut self, _round: usize, _previous_claim: E) -> UniPoly<E> {
        let half = self.w_table.len() / 2;
        let degree_q = 2 * self.b - 1;
        let num_points_q = degree_q + 1;

        let (e_first, e_second) = self.split_eq.remaining_eq_tables();
        let num_first = e_first.len();
        let first_bits = num_first.trailing_zeros();
        let b = self.b;

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
                        let w_0 = self.w_table[2 * j];
                        let w_1 = self.w_table[2 * j + 1];
                        for (t, eval) in evals.iter_mut().enumerate() {
                            let t_e = E::from_u64(t as u64);
                            let w_t = w_0 + t_e * (w_1 - w_0);
                            *eval = *eval + eq_rem * range_check_eval(w_t, b);
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
                let w_0 = self.w_table[2 * j];
                let w_1 = self.w_table[2 * j + 1];
                for (t, eval) in evals.iter_mut().enumerate() {
                    let t_e = E::from_u64(t as u64);
                    let w_t = w_0 + t_e * (w_1 - w_0);
                    *eval = *eval + eq_rem * range_check_eval(w_t, b);
                }
            }
            evals
        };

        let q_poly = UniPoly::from_evals(&q_evals);
        self.split_eq.gruen_mul(&q_poly)
    }

    fn ingest_challenge(&mut self, _round: usize, r: E) {
        self.split_eq.bind(r);
        self.w_table = fold_evals(&self.w_table, r);
    }
}

/// Verifier for the norm (range-check) sumcheck `F_{0,τ₀}`.
pub struct NormSumcheckVerifier<E> {
    tau: Vec<E>,
    w_evals: Vec<E>,
    num_vars: usize,
    b: usize,
}

impl<E: FieldCore + CanonicalField> NormSumcheckVerifier<E> {
    /// Create a new norm (range-check) sumcheck verifier.
    ///
    /// # Panics
    ///
    /// Panics if `w_evals.len() != 2^tau.len()`.
    pub fn new(tau: Vec<E>, w_evals: Vec<E>, b: usize) -> Self {
        let num_vars = tau.len();
        assert_eq!(w_evals.len(), 1 << num_vars);
        Self {
            tau,
            w_evals,
            num_vars,
            b,
        }
    }
}

impl<E: FieldCore + CanonicalField> SumcheckInstanceVerifier<E> for NormSumcheckVerifier<E> {
    fn num_rounds(&self) -> usize {
        self.num_vars
    }

    fn degree_bound(&self) -> usize {
        2 * self.b
    }

    fn input_claim(&self) -> E {
        E::zero()
    }

    fn expected_output_claim(&self, challenges: &[E]) -> E {
        let eq_val = EqPolynomial::mle(&self.tau, challenges);
        let w_val = multilinear_eval(&self.w_evals, challenges);
        eq_val * range_check_eval(w_val, self.b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::ring::CyclotomicRing;
    use crate::algebra::Fp64;
    use crate::primitives::multilinear_evals::DenseMultilinearEvals;
    use crate::protocol::ring_switch::build_w_coeffs;
    use crate::protocol::sumcheck::eq_poly::EqPolynomial;
    use crate::protocol::transcript::labels;
    use crate::protocol::{
        prove_sumcheck, verify_sumcheck, Blake2bTranscript, CommitmentConfig, CommitmentScheme,
        DefaultCommitmentConfig, HachiCommitmentScheme, Transcript,
    };
    use crate::{CanonicalField, FieldCore};

    type F = Fp64<4294967197>;
    const D: usize = 8;

    fn ring_with_small_coeff(value: u64) -> CyclotomicRing<F, D> {
        let coeffs = std::array::from_fn(|_| F::from_u64(value));
        CyclotomicRing::from_coefficients(coeffs)
    }

    #[test]
    fn norm_sumcheck_uses_commitment_w_evals() {
        let z = vec![
            ring_with_small_coeff(1),
            ring_with_small_coeff(2),
            ring_with_small_coeff(3),
        ];
        let r = vec![ring_with_small_coeff(0), ring_with_small_coeff(1)];
        let mut w_evals = build_w_coeffs::<F, D, DefaultCommitmentConfig>(&z, &r);

        let target_len = w_evals.len().next_power_of_two();
        w_evals.resize(target_len, F::zero());
        let num_vars = target_len.trailing_zeros() as usize;
        let tau: Vec<F> = (0..num_vars).map(|i| F::from_u64((i + 2) as u64)).collect();
        let b = 1usize << DefaultCommitmentConfig::LOG_BASIS;

        let eq_table = EqPolynomial::evals(&tau);
        let _claim: F = (0..w_evals.len())
            .map(|i| eq_table[i] * range_check_eval(w_evals[i], b))
            .fold(F::zero(), |a, v| a + v);

        let mut prover = NormSumcheckProver::new(&tau, w_evals.clone(), b);
        let mut pt = Blake2bTranscript::<F>::new(labels::DOMAIN_HACHI_PROTOCOL);
        let (proof, prover_challenges, final_claim) =
            prove_sumcheck::<F, _, F, _, _>(&mut prover, &mut pt, |tr| {
                tr.challenge_scalar(labels::CHALLENGE_SUMCHECK_ROUND)
            })
            .unwrap();

        let oracle = EqPolynomial::mle(&tau, &prover_challenges)
            * range_check_eval(multilinear_eval(&w_evals, &prover_challenges), b);
        assert_eq!(final_claim, oracle, "prover final claim != oracle eval");

        let verifier = NormSumcheckVerifier::new(tau, w_evals, b);
        let mut vt = Blake2bTranscript::<F>::new(labels::DOMAIN_HACHI_PROTOCOL);
        let verifier_challenges =
            verify_sumcheck::<F, _, F, _, _>(&proof, &verifier, &mut vt, |tr| {
                tr.challenge_scalar(labels::CHALLENGE_SUMCHECK_ROUND)
            })
            .unwrap();

        assert_eq!(prover_challenges, verifier_challenges);
    }

    #[test]
    fn norm_sumcheck_uses_prove_w_evals() {
        let alpha = DefaultCommitmentConfig::D.trailing_zeros() as usize;
        let num_vars = DefaultCommitmentConfig::R + DefaultCommitmentConfig::M + alpha;
        let len = 1usize << num_vars;
        let evals: Vec<F> = (0..len).map(|i| F::from_u64(i as u64)).collect();
        let poly = DenseMultilinearEvals::new_padded(evals);

        let setup = HachiCommitmentScheme::setup_prover(num_vars);
        let (commitment, hint) = HachiCommitmentScheme::commit(&poly, &setup).unwrap();

        let opening_point: Vec<F> = (0..num_vars).map(|i| F::from_u64((i + 2) as u64)).collect();
        let mut prover_transcript = Blake2bTranscript::<F>::new(labels::DOMAIN_HACHI_PROTOCOL);
        let proof = HachiCommitmentScheme::prove(
            &setup,
            &poly,
            &opening_point,
            Some(hint),
            &mut prover_transcript,
            &commitment,
        )
        .unwrap();

        let mut w_evals = proof.sumcheck_aux.w.clone();
        let target_len = w_evals.len().next_power_of_two();
        w_evals.resize(target_len, F::zero());
        let num_sumcheck_vars = target_len.trailing_zeros() as usize;
        let tau: Vec<F> = (0..num_sumcheck_vars)
            .map(|i| F::from_u64((i + 3) as u64))
            .collect();
        let b = 1usize << DefaultCommitmentConfig::LOG_BASIS;

        let eq_table = EqPolynomial::evals(&tau);
        let _claim: F = (0..w_evals.len())
            .map(|i| eq_table[i] * range_check_eval(w_evals[i], b))
            .fold(F::zero(), |a, v| a + v);

        let mut prover = NormSumcheckProver::new(&tau, w_evals.clone(), b);
        let mut pt = Blake2bTranscript::<F>::new(labels::DOMAIN_HACHI_PROTOCOL);
        let (proof_sc, prover_challenges, final_claim) =
            prove_sumcheck::<F, _, F, _, _>(&mut prover, &mut pt, |tr| {
                tr.challenge_scalar(labels::CHALLENGE_SUMCHECK_ROUND)
            })
            .unwrap();

        let oracle = EqPolynomial::mle(&tau, &prover_challenges)
            * range_check_eval(multilinear_eval(&w_evals, &prover_challenges), b);
        assert_eq!(final_claim, oracle, "prover final claim != oracle eval");

        let verifier = NormSumcheckVerifier::new(tau, w_evals, b);
        let mut vt = Blake2bTranscript::<F>::new(labels::DOMAIN_HACHI_PROTOCOL);
        let verifier_challenges =
            verify_sumcheck::<F, _, F, _, _>(&proof_sc, &verifier, &mut vt, |tr| {
                tr.challenge_scalar(labels::CHALLENGE_SUMCHECK_ROUND)
            })
            .unwrap();

        assert_eq!(prover_challenges, verifier_challenges);
    }
}
