#![allow(missing_docs)]

use akita_algebra::split_eq::GruenSplitEq;
use akita_error::AkitaError;
use akita_serialization::{AkitaDeserialize, AkitaSerialize};
use akita_sumcheck::{
    EqFactoredSumcheckInstanceProver, EqFactoredSumcheckInstanceProverExt,
    EqFactoredSumcheckInstanceVerifier, EqFactoredSumcheckInstanceVerifierExt, EqFactoredUniPoly,
    UniPoly,
};
use akita_transcript::labels as tr_labels;
use akita_transcript::{AkitaTranscript, Transcript};
use jolt_field::{Field, One, Prime128Offset275, Ring, Zero};

type F = Prime128Offset275;

fn new_transcript() -> AkitaTranscript<F> {
    <AkitaTranscript<F> as akita_transcript::TranscriptFactory<F>>::new(
        tr_labels::DOMAIN_AKITA_PROTOCOL,
    )
}

fn sample_round(tr: &mut AkitaTranscript<F>) -> Result<F, AkitaError> {
    Ok(tr.challenge_scalar(tr_labels::CHALLENGE_SUMCHECK_ROUND))
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

    fn current_tau(&self) -> F {
        self.split_eq.current_tau()
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
        _round_state: &Self::RoundState,
        challenges: &[F],
    ) -> Result<F, AkitaError> {
        Ok(self.q_poly().evaluate(&challenges[0]))
    }
}

#[test]
fn eq_factored_sumcheck_prove_verify_roundtrip() {
    let q_coeffs = vec![
        F::from_u64(3),
        F::from_u64(5),
        F::from_u64(7),
        F::from_u64(11),
        F::from_u64(13),
    ];
    let half = F::from_u64(2).inverse().unwrap();
    for tau in [F::zero(), F::one(), half, F::from_u64(17)] {
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
        assert_eq!(
            proof.round_polys[0].coeffs_except_constant_term,
            q_coeffs[1..]
        );

        let verifier = ToyEqFactoredInstance::new(tau, q_coeffs.clone());
        let mut verify_tr = new_transcript();
        let verifier_challenges = verifier
            .verify::<F, _, _>(&proof, &mut verify_tr, sample_round)
            .unwrap();

        assert_eq!(verifier_challenges, prover_challenges);
    }
}

#[test]
fn eq_factored_sumcheck_rejects_tampering_when_tau_is_zero() {
    let q_coeffs = vec![F::from_u64(3), F::from_u64(5), F::from_u64(7)];
    let verifier = ToyEqFactoredInstance::new(F::zero(), q_coeffs);
    let proof = akita_sumcheck::EqFactoredSumcheckProof {
        round_polys: vec![EqFactoredUniPoly {
            coeffs_except_constant_term: vec![F::from_u64(99), F::from_u64(101)],
        }],
    };
    let mut transcript = new_transcript();

    assert_eq!(
        verifier.verify::<F, _, _>(&proof, &mut transcript, |_| Ok(F::from_u64(11))),
        Err(AkitaError::InvalidProof)
    );
}

#[test]
fn eq_factored_round_wire_contains_every_nonconstant_coefficient() {
    let q_coeffs = vec![
        F::from_u64(3),
        F::from_u64(5),
        F::from_u64(7),
        F::from_u64(11),
    ];
    let poly = EqFactoredUniPoly::from_q_coeffs(q_coeffs.clone());
    let mut encoded = Vec::new();
    poly.serialize_uncompressed(&mut encoded).unwrap();

    let mut expected = Vec::new();
    for coefficient in &q_coeffs[1..] {
        coefficient.serialize_uncompressed(&mut expected).unwrap();
    }
    assert_eq!(encoded, expected);
    assert_eq!(EqFactoredUniPoly::<F>::stored_coeff_count_for_degree(3), 3);
    assert_eq!(
        EqFactoredUniPoly::<F>::deserialize_uncompressed(&encoded[..], &3).unwrap(),
        poly
    );
}

struct ToyTwoRoundEqFactoredInstance {
    tau: [F; 2],
    split_eq: GruenSplitEq<F>,
    coefficients: [F; 4],
    first_challenge: Option<F>,
}

impl ToyTwoRoundEqFactoredInstance {
    fn new(tau: [F; 2], coefficients: [F; 4]) -> Self {
        Self {
            tau,
            split_eq: GruenSplitEq::new(&tau).unwrap(),
            coefficients,
            first_challenge: None,
        }
    }

    fn evaluate(&self, x_0: F, x_1: F) -> F {
        let [a, b, c, d] = self.coefficients;
        a + b * x_0 + c * x_1 + d * x_0 * x_1
    }
}

impl EqFactoredSumcheckInstanceProver<F> for ToyTwoRoundEqFactoredInstance {
    fn num_rounds(&self) -> usize {
        2
    }

    fn degree_bound(&self) -> usize {
        1
    }

    fn input_claim(&self) -> F {
        self.evaluate(self.tau[0], self.tau[1])
    }

    fn current_linear_factor_evals(&self) -> (F, F) {
        self.split_eq.linear_factor_evals()
    }

    fn current_tau(&self) -> F {
        self.split_eq.current_tau()
    }

    fn compute_round_eq_factored(&mut self, round: usize) -> EqFactoredUniPoly<F> {
        let [a, b, c, d] = self.coefficients;
        let coefficients = if round == 0 {
            vec![a + c * self.tau[1], b + d * self.tau[1]]
        } else {
            let r_0 = self.first_challenge.unwrap();
            vec![a + b * r_0, c + d * r_0]
        };
        EqFactoredUniPoly::from_q_coeffs(coefficients)
    }

    fn ingest_challenge(&mut self, round: usize, challenge: F) {
        if round == 0 {
            self.first_challenge = Some(challenge);
        }
        self.split_eq.bind(challenge);
    }
}

impl EqFactoredSumcheckInstanceVerifier<F> for ToyTwoRoundEqFactoredInstance {
    type RoundState = GruenSplitEq<F>;

    fn num_rounds(&self) -> usize {
        2
    }

    fn degree_bound(&self) -> usize {
        1
    }

    fn input_claim(&self) -> F {
        self.evaluate(self.tau[0], self.tau[1])
    }

    fn start_round_state(&self) -> Result<Self::RoundState, AkitaError> {
        GruenSplitEq::new(&self.tau)
    }

    fn expected_output_claim(
        &self,
        _round_state: &Self::RoundState,
        challenges: &[F],
    ) -> Result<F, AkitaError> {
        Ok(self.evaluate(challenges[0], challenges[1]))
    }
}

#[test]
fn eq_factored_sumcheck_rejects_later_tampering_after_eq_factor_vanishes() {
    let tau = [F::from_u64(2), F::from_u64(5)];
    let coefficients = [
        F::from_u64(3),
        F::from_u64(7),
        F::from_u64(11),
        F::from_u64(13),
    ];
    let r_0 = F::from_u64(3).inverse().unwrap();
    let r_1 = F::from_u64(17);
    let vanished_factor = tau[0] * r_0 + (F::one() - tau[0]) * (F::one() - r_0);
    assert!(vanished_factor.is_zero());
    let mut prover = ToyTwoRoundEqFactoredInstance::new(tau, coefficients);
    let mut prover_transcript = new_transcript();
    let mut prover_round = 0;
    let (mut proof, _, _) = prover
        .prove::<F, _, _>(&mut prover_transcript, |_| {
            let challenge = [r_0, r_1][prover_round];
            prover_round += 1;
            Ok(challenge)
        })
        .unwrap();

    let honest_verifier = ToyTwoRoundEqFactoredInstance::new(tau, coefficients);
    let mut honest_transcript = new_transcript();
    let mut honest_round = 0;
    let honest_result = honest_verifier.verify::<F, _, _>(&proof, &mut honest_transcript, |_| {
        let challenge = [r_0, r_1][honest_round];
        honest_round += 1;
        Ok(challenge)
    });
    assert_eq!(honest_result, Ok(vec![r_0, r_1]));

    proof.round_polys[1].coeffs_except_constant_term[0] += F::one();

    let verifier = ToyTwoRoundEqFactoredInstance::new(tau, coefficients);
    let mut verifier_transcript = new_transcript();
    let mut verifier_round = 0;
    let result = verifier.verify::<F, _, _>(&proof, &mut verifier_transcript, |_| {
        let challenge = [r_0, r_1][verifier_round];
        verifier_round += 1;
        Ok(challenge)
    });

    assert_eq!(result, Err(AkitaError::InvalidProof));
}
