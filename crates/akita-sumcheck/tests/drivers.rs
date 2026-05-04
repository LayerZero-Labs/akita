#![allow(missing_docs)]

use akita_algebra::poly::{fold_evals_in_place, multilinear_eval};
use akita_algebra::split_eq::GruenSplitEq;
use akita_algebra::Prime128Offset275;
use akita_field::{AkitaError, FieldCore, FromSmallInt};
use akita_sumcheck::{
    prove_eq_factored_sumcheck, prove_sumcheck, prove_sumcheck_with_omitted_prefix_rounds,
    verify_eq_factored_sumcheck, verify_sumcheck_with_prefix_rounds,
    EqFactoredSumcheckInstanceProver, EqFactoredSumcheckInstanceVerifier, EqFactoredUniPoly,
    SumcheckInstanceProver, SumcheckInstanceVerifier, SumcheckProof, UniPoly,
};
use akita_transcript::labels as tr_labels;
use akita_transcript::{Blake2bTranscript, Transcript};

type F = Prime128Offset275;

#[derive(Clone)]
struct ToyMlInstance {
    original: Vec<F>,
    current: Vec<F>,
    num_rounds: usize,
}

impl ToyMlInstance {
    fn new(evals: Vec<F>) -> Self {
        let len = evals.len();
        let num_rounds = len.trailing_zeros() as usize;
        debug_assert_eq!(1usize << num_rounds, len);
        Self {
            original: evals.clone(),
            current: evals,
            num_rounds,
        }
    }
}

impl SumcheckInstanceProver<F> for ToyMlInstance {
    fn num_rounds(&self) -> usize {
        self.num_rounds
    }

    fn degree_bound(&self) -> usize {
        1
    }

    fn input_claim(&self) -> F {
        self.original
            .iter()
            .copied()
            .fold(F::zero(), |acc, x| acc + x)
    }

    fn compute_round_univariate(&mut self, round: usize, previous_claim: F) -> UniPoly<F> {
        debug_assert_eq!(self.current.len(), 1usize << (self.num_rounds - round));
        let half = self.current.len() / 2;
        let mut at_zero = F::zero();
        let mut slope = F::zero();
        for j in 0..half {
            let left = self.current[2 * j];
            let right = self.current[2 * j + 1];
            at_zero += left;
            slope += right - left;
        }
        let poly = UniPoly::from_coeffs(vec![at_zero, slope]);
        debug_assert_eq!(
            poly.evaluate(&F::zero()) + poly.evaluate(&F::one()),
            previous_claim
        );
        poly
    }

    fn ingest_challenge(&mut self, _round: usize, r_round: F) {
        fold_evals_in_place(&mut self.current, r_round);
    }
}

impl SumcheckInstanceVerifier<F> for ToyMlInstance {
    fn num_rounds(&self) -> usize {
        self.num_rounds
    }

    fn degree_bound(&self) -> usize {
        1
    }

    fn input_claim(&self) -> F {
        self.original
            .iter()
            .copied()
            .fold(F::zero(), |acc, x| acc + x)
    }

    fn expected_output_claim(&self, challenges: &[F]) -> Result<F, AkitaError> {
        multilinear_eval(&self.original, challenges)
    }
}

fn new_transcript() -> Blake2bTranscript<F> {
    <Blake2bTranscript<F> as Transcript<F>>::new(tr_labels::DOMAIN_AKITA_PROTOCOL)
}

fn sample_round(tr: &mut Blake2bTranscript<F>) -> F {
    tr.challenge_scalar(tr_labels::CHALLENGE_SUMCHECK_ROUND)
}

#[test]
fn prove_sumcheck_with_omitted_prefix_rounds_matches_full_proof_tail() {
    let evals: Vec<F> = (0..16).map(|i| F::from_u64((7 * i as u64) + 3)).collect();
    let mut full = ToyMlInstance::new(evals.clone());
    let mut full_tr = new_transcript();
    let (full_proof, full_challenges, full_final_claim) =
        prove_sumcheck::<F, _, F, _, _>(&mut full, &mut full_tr, sample_round).unwrap();

    let mut omitted = ToyMlInstance::new(evals);
    let mut omitted_tr = new_transcript();
    let (suffix_proof, challenges, suffix_final_claim) =
        prove_sumcheck_with_omitted_prefix_rounds::<F, _, F, _, _, _>(
            &mut omitted,
            &mut omitted_tr,
            sample_round,
            2,
            |_, _, _| Ok(()),
        )
        .unwrap();

    assert_eq!(challenges, full_challenges);
    assert_eq!(
        suffix_proof.round_polys.as_slice(),
        &full_proof.round_polys[2..]
    );
    assert_eq!(suffix_final_claim, full_final_claim);
}

#[test]
fn verify_sumcheck_with_prefix_rounds_matches_full_verification_tail() {
    let evals: Vec<F> = (0..16).map(|i| F::from_u64((11 * i as u64) + 5)).collect();
    let mut prover = ToyMlInstance::new(evals.clone());
    let mut proof_tr = new_transcript();
    let (full_proof, full_challenges, full_final_claim) =
        prove_sumcheck::<F, _, F, _, _>(&mut prover, &mut proof_tr, sample_round).unwrap();

    let verifier = ToyMlInstance::new(evals);
    let suffix_proof = SumcheckProof {
        round_polys: full_proof.round_polys[2..].to_vec(),
    };
    let prefix_rounds = full_proof.round_polys[..2].to_vec();
    let mut verify_tr = new_transcript();
    let challenges = verify_sumcheck_with_prefix_rounds::<F, _, F, _, _, _, _>(
        &suffix_proof,
        &verifier,
        &mut verify_tr,
        sample_round,
        2,
        |_, _| Ok(()),
        |round, _, _| prefix_rounds[round].clone(),
    )
    .unwrap();

    assert_eq!(challenges, full_challenges);
    assert_eq!(
        verifier.expected_output_claim(&challenges).unwrap(),
        full_final_claim
    );
}

struct ToyEqFactoredInstance {
    tau: F,
    split_eq: GruenSplitEq<F>,
    q_coeffs: Vec<F>,
}

impl ToyEqFactoredInstance {
    fn new(tau: F, q_coeffs: Vec<F>) -> Self {
        Self {
            tau,
            split_eq: GruenSplitEq::new(&[tau]),
            q_coeffs,
        }
    }

    fn q_poly(&self) -> UniPoly<F> {
        UniPoly::from_coeffs(self.q_coeffs.clone())
    }

    fn input_claim_from_tau(&self) -> F {
        let g = GruenSplitEq::new(&[self.tau]).gruen_mul(&self.q_poly());
        g.evaluate(&F::zero()) + g.evaluate(&F::one())
    }
}

impl EqFactoredSumcheckInstanceProver<F> for ToyEqFactoredInstance {
    fn num_rounds(&self) -> usize {
        1
    }

    fn degree_bound(&self) -> usize {
        self.q_coeffs.len().saturating_sub(1)
    }

    fn input_claim(&self) -> F {
        self.input_claim_from_tau()
    }

    fn current_linear_factor_evals(&self) -> (F, F) {
        self.split_eq.linear_factor_evals()
    }

    fn compute_round_eq_factored(&mut self, _round: usize) -> EqFactoredUniPoly<F> {
        EqFactoredUniPoly::from_q_coeffs(self.q_coeffs.clone())
    }

    fn ingest_challenge(&mut self, _round: usize, r_round: F) {
        self.split_eq.bind(r_round);
    }
}

impl EqFactoredSumcheckInstanceVerifier<F> for ToyEqFactoredInstance {
    type RoundState = GruenSplitEq<F>;

    fn num_rounds(&self) -> usize {
        1
    }

    fn degree_bound(&self) -> usize {
        self.q_coeffs.len().saturating_sub(1)
    }

    fn input_claim(&self) -> F {
        self.input_claim_from_tau()
    }

    fn start_round_state(&self) -> Self::RoundState {
        GruenSplitEq::new(&[self.tau])
    }

    fn expected_output_claim(
        &self,
        round_state: &Self::RoundState,
        challenges: &[F],
    ) -> Result<F, AkitaError> {
        Ok(round_state.current_scalar() * self.q_poly().evaluate(&challenges[0]))
    }
}

#[test]
fn eq_factored_sumcheck_prove_verify_roundtrip() {
    let tau = F::from_u64(17);
    let q_coeffs = vec![
        F::from_u64(3),
        F::from_u64(5),
        F::from_u64(7),
        F::from_u64(11),
        F::from_u64(13),
    ];
    let mut prover = ToyEqFactoredInstance::new(tau, q_coeffs.clone());
    let mut prover_tr = new_transcript();
    let (proof, prover_challenges, _) =
        prove_eq_factored_sumcheck::<F, _, F, _, _>(&mut prover, &mut prover_tr, sample_round)
            .unwrap();

    assert_eq!(proof.round_polys.len(), 1);
    assert_eq!(
        proof.round_polys[0],
        EqFactoredUniPoly::from_q_coeffs(q_coeffs.clone())
    );

    let verifier = ToyEqFactoredInstance::new(tau, q_coeffs);
    let mut verify_tr = new_transcript();
    let verifier_challenges = verify_eq_factored_sumcheck::<F, _, F, _, _>(
        &proof,
        &verifier,
        &mut verify_tr,
        sample_round,
    )
    .unwrap();

    assert_eq!(verifier_challenges, prover_challenges);
}
