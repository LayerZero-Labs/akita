#![allow(dead_code)]
//! Sumcheck for the setup-backed claim `Y_setup = <shared_matrix, matrix_weight>`.
//!
//! After stage 1 determines the algebraic contribution `alg(r_x)`, the
//! delegated setup residue is:
//!
//! ```text
//!   Y_setup = relation_target − λ · alg(r_x) = λ · setup(r_x)
//!           = λ · Σ_{row, col, k} shared_matrix[row, col, k] · matrix_weight[row, col, k]
//! ```
//!
//! This module provides `SetupClaimProver` / `SetupClaimVerifier` that
//! implement the standard `SumcheckInstance{Prover,Verifier}` traits for the
//! product sumcheck over the flat shared matrix and the weight tensor.

use crate::algebra::poly::fold_evals_in_place;
use crate::algebra::uni_poly::UniPoly;
use crate::error::HachiError;
use crate::protocol::sumcheck::traits::{SumcheckInstanceProver, SumcheckInstanceVerifier};
use crate::{FieldCore, FromSmallInt};

/// Prover for the setup-claim product sumcheck.
///
/// Holds the materialized shared matrix and matrix weight tables (both as dense
/// MLE evaluation vectors) and folds them one variable at a time.
pub(crate) struct SetupClaimProver<E> {
    shared_matrix: Vec<E>,
    matrix_weight: Vec<E>,
    num_vars: usize,
    claim: E,
}

impl<E: FieldCore> SetupClaimProver<E> {
    /// Construct a new setup claim prover.
    ///
    /// `shared_matrix` and `matrix_weight` must have the same power-of-two
    /// length `2^num_vars`.
    /// `claim` is the asserted inner product
    /// `Σ shared_matrix[i] * matrix_weight[i]`.
    pub(crate) fn new(
        shared_matrix: Vec<E>,
        matrix_weight: Vec<E>,
        num_vars: usize,
        claim: E,
    ) -> Self {
        debug_assert_eq!(shared_matrix.len(), 1 << num_vars);
        debug_assert_eq!(matrix_weight.len(), 1 << num_vars);
        Self {
            shared_matrix,
            matrix_weight,
            num_vars,
            claim,
        }
    }
}

impl<E: FieldCore + FromSmallInt> SumcheckInstanceProver<E> for SetupClaimProver<E> {
    fn num_rounds(&self) -> usize {
        self.num_vars
    }

    fn degree_bound(&self) -> usize {
        2
    }

    fn input_claim(&self) -> E {
        self.claim
    }

    fn compute_round_univariate(&mut self, _round: usize, _previous_claim: E) -> UniPoly<E> {
        let half = self.shared_matrix.len() / 2;
        let mut eval_0 = E::zero();
        let mut eval_1 = E::zero();
        let mut eval_2 = E::zero();
        for j in 0..half {
            let s0 = self.shared_matrix[2 * j];
            let s1 = self.shared_matrix[2 * j + 1];
            let w0 = self.matrix_weight[2 * j];
            let w1 = self.matrix_weight[2 * j + 1];
            eval_0 += s0 * w0;
            eval_1 += s1 * w1;
            let s2 = s1 + s1 - s0;
            let w2 = w1 + w1 - w0;
            eval_2 += s2 * w2;
        }
        UniPoly::from_evals(&[eval_0, eval_1, eval_2])
    }

    fn ingest_challenge(&mut self, _round: usize, r: E) {
        fold_evals_in_place(&mut self.shared_matrix, r);
        fold_evals_in_place(&mut self.matrix_weight, r);
    }
}

/// Verifier for the setup-claim product sumcheck.
///
/// The verifier's `expected_output_claim` is
/// `shared_matrix(r) · matrix_weight(r)`.
/// `matrix_weight(r)` is computed via the weight oracle;
/// `shared_matrix(r)` is the prover-claimed evaluation, verified later via PCS
/// opening.
#[allow(dead_code)]
pub(crate) struct SetupClaimVerifier<E> {
    num_vars: usize,
    claim: E,
    shared_matrix_eval: E,
    matrix_weight_eval: E,
}

#[allow(dead_code)]
impl<E: FieldCore> SetupClaimVerifier<E> {
    /// Construct the verifier.
    ///
    /// `claim` is the alleged inner product.
    /// `shared_matrix_eval` is the prover-provided evaluation (to be
    /// PCS-verified later).
    /// `matrix_weight_eval` is the verifier-computed evaluation from
    /// `eval_matrix_weight_at_point`.
    pub(crate) fn new(
        num_vars: usize,
        claim: E,
        shared_matrix_eval: E,
        matrix_weight_eval: E,
    ) -> Self {
        Self {
            num_vars,
            claim,
            shared_matrix_eval,
            matrix_weight_eval,
        }
    }
}

impl<E: FieldCore> SumcheckInstanceVerifier<E> for SetupClaimVerifier<E> {
    fn num_rounds(&self) -> usize {
        self.num_vars
    }

    fn degree_bound(&self) -> usize {
        2
    }

    fn input_claim(&self) -> E {
        self.claim
    }

    fn expected_output_claim(&self, _challenges: &[E]) -> Result<E, HachiError> {
        Ok(self.shared_matrix_eval * self.matrix_weight_eval)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::poly::multilinear_eval;
    use crate::algebra::Prime128Offset275;
    use crate::protocol::sumcheck::drivers::{prove_sumcheck, verify_sumcheck};
    use crate::protocol::transcript::labels;
    use crate::protocol::transcript::{Blake2bTranscript, Transcript};

    type F = Prime128Offset275;

    #[test]
    fn setup_claim_sumcheck_prove_verify() {
        let num_vars = 6;
        let n = 1usize << num_vars;

        let shared_matrix: Vec<F> = (0..n)
            .map(|i| F::from_u64((3 * i as u64 + 7) % 97))
            .collect();
        let matrix_weight: Vec<F> = (0..n)
            .map(|i| F::from_u64((11 * i as u64 + 5) % 101))
            .collect();
        let claim: F = shared_matrix
            .iter()
            .zip(matrix_weight.iter())
            .map(|(s, w)| *s * *w)
            .fold(F::zero(), |a, b| a + b);

        let mut prover = SetupClaimProver::new(
            shared_matrix.clone(),
            matrix_weight.clone(),
            num_vars,
            claim,
        );

        let mut prover_transcript = Blake2bTranscript::<F>::new(labels::DOMAIN_HACHI_PROTOCOL);
        let (proof, prover_challenges, final_claim) =
            prove_sumcheck::<F, _, F, _, _>(&mut prover, &mut prover_transcript, |tr| {
                tr.challenge_scalar(labels::CHALLENGE_SUMCHECK_ROUND)
            })
            .unwrap();

        assert_eq!(proof.round_polys.len(), num_vars);

        let sm_eval =
            multilinear_eval(&shared_matrix, &prover_challenges).expect("shared_matrix eval");
        let mw_eval =
            multilinear_eval(&matrix_weight, &prover_challenges).expect("matrix_weight eval");
        assert_eq!(
            final_claim,
            sm_eval * mw_eval,
            "final claim must equal product of evaluations"
        );

        let verifier = SetupClaimVerifier::new(num_vars, claim, sm_eval, mw_eval);
        let mut verifier_transcript = Blake2bTranscript::<F>::new(labels::DOMAIN_HACHI_PROTOCOL);
        let verifier_challenges =
            verify_sumcheck::<F, _, F, _, _>(&proof, &verifier, &mut verifier_transcript, |tr| {
                tr.challenge_scalar(labels::CHALLENGE_SUMCHECK_ROUND)
            })
            .unwrap();

        assert_eq!(prover_challenges, verifier_challenges);
    }

    #[test]
    fn setup_claim_sumcheck_wrong_claim_rejected() {
        let num_vars = 4;
        let n = 1usize << num_vars;

        let shared_matrix: Vec<F> = (0..n).map(|i| F::from_u64(i as u64 + 1)).collect();
        let matrix_weight: Vec<F> = (0..n).map(|i| F::from_u64(2 * i as u64 + 3)).collect();
        let correct_claim: F = shared_matrix
            .iter()
            .zip(matrix_weight.iter())
            .map(|(s, w)| *s * *w)
            .fold(F::zero(), |a, b| a + b);
        let wrong_claim = correct_claim + F::one();

        let mut prover = SetupClaimProver::new(
            shared_matrix.clone(),
            matrix_weight.clone(),
            num_vars,
            correct_claim,
        );
        let mut prover_transcript = Blake2bTranscript::<F>::new(labels::DOMAIN_HACHI_PROTOCOL);
        let (proof, prover_challenges, _) =
            prove_sumcheck::<F, _, F, _, _>(&mut prover, &mut prover_transcript, |tr| {
                tr.challenge_scalar(labels::CHALLENGE_SUMCHECK_ROUND)
            })
            .unwrap();

        let sm_eval =
            multilinear_eval(&shared_matrix, &prover_challenges).expect("shared_matrix eval");
        let mw_eval =
            multilinear_eval(&matrix_weight, &prover_challenges).expect("matrix_weight eval");

        let verifier = SetupClaimVerifier::new(num_vars, wrong_claim, sm_eval, mw_eval);
        let mut verifier_transcript = Blake2bTranscript::<F>::new(labels::DOMAIN_HACHI_PROTOCOL);
        let result =
            verify_sumcheck::<F, _, F, _, _>(&proof, &verifier, &mut verifier_transcript, |tr| {
                tr.challenge_scalar(labels::CHALLENGE_SUMCHECK_ROUND)
            });

        assert!(result.is_err(), "wrong claim must be rejected");
    }
}
