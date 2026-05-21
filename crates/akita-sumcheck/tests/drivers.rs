#![allow(missing_docs)]
#![cfg(not(feature = "zk"))]

use akita_algebra::split_eq::GruenSplitEq;
use akita_field::AkitaError;
use akita_field::Prime128Offset275;
use akita_sumcheck::{
    EqFactoredSumcheckInstanceProver, EqFactoredSumcheckInstanceProverExt,
    EqFactoredSumcheckInstanceVerifier, EqFactoredSumcheckInstanceVerifierExt, EqFactoredUniPoly,
    UniPoly,
};
use akita_transcript::labels as tr_labels;
use akita_transcript::{AkitaTranscript, Transcript};

type F = Prime128Offset275;

fn new_transcript() -> AkitaTranscript<F> {
    <AkitaTranscript<F> as Transcript<F>>::new(tr_labels::DOMAIN_AKITA_PROTOCOL)
}

fn sample_round(tr: &mut AkitaTranscript<F>) -> F {
    tr.challenge_scalar(tr_labels::CHALLENGE_SUMCHECK_ROUND)
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
            split_eq: GruenSplitEq::new(&[tau]).unwrap(),
            q_coeffs,
        }
    }

    fn q_poly(&self) -> UniPoly<F> {
        UniPoly::from_coeffs(self.q_coeffs.clone())
    }

    fn input_claim_from_tau(&self) -> F {
        let g = GruenSplitEq::new(&[self.tau])
            .unwrap()
            .gruen_mul(&self.q_poly());
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

    fn start_round_state(&self) -> Result<Self::RoundState, AkitaError> {
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
    let (proof, prover_challenges, _) = prover
        .prove::<F, _, _>(&mut prover_tr, sample_round)
        .unwrap();

    assert_eq!(proof.round_polys.len(), 1);
    assert_eq!(
        proof.round_polys[0],
        EqFactoredUniPoly::from_q_coeffs(q_coeffs.clone())
    );

    let verifier = ToyEqFactoredInstance::new(tau, q_coeffs);
    let mut verify_tr = new_transcript();
    let verifier_challenges = verifier
        .verify::<F, _, _>(&proof, &mut verify_tr, sample_round)
        .unwrap();

    assert_eq!(verifier_challenges, prover_challenges);
}
