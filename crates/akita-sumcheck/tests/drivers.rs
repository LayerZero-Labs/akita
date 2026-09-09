#![allow(missing_docs)]

use akita_algebra::split_eq::GruenSplitEq;
use akita_error::AkitaError;
use akita_serialization::{AkitaDeserialize, AkitaSerialize};
use akita_sumcheck::{
    CompressedUniPoly, EqFactoredSumcheckInstanceProver, EqFactoredSumcheckInstanceProverExt,
    EqFactoredSumcheckInstanceVerifier, EqFactoredSumcheckInstanceVerifierExt, EqFactoredUniPoly,
    SumcheckInstanceProver, SumcheckInstanceProverExt, SumcheckInstanceVerifier,
    SumcheckInstanceVerifierExt, SumcheckProof, UniPoly,
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
fn eq_factored_sumcheck_rejects_old_wire_forgery_when_tau_is_zero() {
    let q_coeffs = vec![F::from_u64(3), F::from_u64(5), F::from_u64(7)];
    let verifier = ToyEqFactoredInstance::new(F::zero(), q_coeffs);
    let old_wire_forged_q_0 = verifier.input_claim_from_tau();
    let proof = akita_sumcheck::EqFactoredSumcheckProof {
        round_polys: vec![EqFactoredUniPoly {
            // Under the old `[q_0, q_2]` wire convention, choosing `q_0 = T`
            // collapsed the scaled claim to zero and made `q_2` unconstrained.
            coeffs_except_constant_term: vec![old_wire_forged_q_0, F::from_u64(101)],
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
    assert_eq!(
        EqFactoredUniPoly::<F>::deserialize_uncompressed(&encoded[..], &3).unwrap(),
        poly
    );
}

#[test]
fn eq_factored_degree_zero_round_has_an_empty_message() {
    // Empty eq-factored messages preserve the normalized claim, including at
    // Boolean equality points and challenges, unlike empty ordinary messages.
    for tau in [F::zero(), F::one(), F::from_u64(7)] {
        for challenge in [F::zero(), F::one(), F::from_u64(11)] {
            let q_coeffs = vec![F::from_u64(23)];
            let mut prover = ToyEqFactoredInstance::new(tau, q_coeffs.clone());
            let (proof, _, final_claim) = prover
                .prove::<F, _, _>(&mut new_transcript(), |_| Ok(challenge))
                .unwrap();

            assert!(proof.round_polys[0].coeffs_except_constant_term.is_empty());
            let mut encoded = Vec::new();
            proof.round_polys[0]
                .serialize_uncompressed(&mut encoded)
                .unwrap();
            assert!(encoded.is_empty());
            assert_eq!(final_claim, q_coeffs[0]);

            let verifier = ToyEqFactoredInstance::new(tau, q_coeffs);
            assert_eq!(
                verifier.verify::<F, _, _>(&proof, &mut new_transcript(), |_| Ok(challenge)),
                Ok(vec![challenge])
            );
        }
    }
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

/// An inconsistent claim for the identically zero polynomial.
struct FalseClaimInstance;

impl SumcheckInstanceVerifier<F> for FalseClaimInstance {
    fn num_rounds(&self) -> usize {
        4
    }

    fn degree_bound(&self) -> usize {
        3
    }

    fn input_claim(&self) -> F {
        F::one()
    }

    fn expected_output_claim(&self, _challenges: &[F]) -> Result<F, AkitaError> {
        Ok(F::zero())
    }
}

fn assert_rounds_rejected_before_replay(proof: &SumcheckProof<F>, expected: AkitaError) {
    let verifier = FalseClaimInstance;
    for raw_driver in [false, true] {
        let mut transcript = new_transcript();
        let mut samples = 0;
        let sample = |_: &mut AkitaTranscript<F>| {
            samples += 1;
            Ok(F::one())
        };
        let result = if raw_driver {
            proof
                .verify::<F, _, _>(
                    verifier.input_claim(),
                    verifier.num_rounds(),
                    verifier.degree_bound(),
                    &mut transcript,
                    sample,
                )
                .map(|_| ())
        } else {
            verifier
                .verify::<F, _, _>(proof, &mut transcript, sample)
                .map(|_| ())
        };
        assert_eq!(result, Err(expected.clone()));
        assert_eq!(samples, 0);
        assert_eq!(
            transcript.challenge_bytes(b"test/rejected-round-state", 32),
            new_transcript().challenge_bytes(b"test/rejected-round-state", 32)
        );
    }
}

#[test]
fn standard_sumcheck_rejects_malformed_messages_at_every_round() {
    let verifier = FalseClaimInstance;
    for round in 0..verifier.num_rounds() {
        for stored_coefficients in [0, verifier.degree_bound() + 1] {
            let mut proof = SumcheckProof {
                round_polys: vec![
                    UniPoly::from_coeffs(vec![F::zero()]).compress();
                    verifier.num_rounds()
                ],
            };
            proof.round_polys[round] = CompressedUniPoly {
                coeffs_except_linear_term: vec![F::one(); stored_coefficients],
            };
            let error = if stored_coefficients == 0 {
                AkitaError::InvalidProof
            } else {
                AkitaError::InvalidInput("sumcheck round poly degree 4 exceeds bound 3".into())
            };
            assert_rounds_rejected_before_replay(&proof, error);
        }
    }
}

#[test]
fn standard_sumcheck_rejects_wrong_round_counts_before_replay() {
    let expected = FalseClaimInstance.num_rounds();
    for actual in [0, expected - 1, expected + 1] {
        let proof = SumcheckProof {
            round_polys: vec![UniPoly::from_coeffs(vec![F::zero()]).compress(); actual],
        };
        assert_rounds_rejected_before_replay(&proof, AkitaError::InvalidSize { expected, actual });
    }
}

#[test]
fn zero_round_sumcheck_preserves_the_claim() {
    let proof = SumcheckProof::<F> {
        round_polys: Vec::new(),
    };
    let claim = F::from_u64(7);
    assert_eq!(
        proof.verify::<F, _, _>(claim, 0, 0, &mut new_transcript(), |_| panic!(
            "no round to sample"
        )),
        Ok((claim, Vec::new()))
    );
}

#[test]
fn batched_sumcheck_rejects_an_empty_last_round() {
    let verifier = FalseClaimInstance;
    let mut proof = SumcheckProof {
        round_polys: vec![UniPoly::from_coeffs(vec![F::zero()]).compress(); verifier.num_rounds()],
    };
    proof
        .round_polys
        .last_mut()
        .unwrap()
        .coeffs_except_linear_term
        .clear();
    assert_eq!(
        akita_sumcheck::verify_batched_sumcheck::<F, _, F, _>(
            &proof,
            vec![&verifier],
            &mut new_transcript(),
            |tr| tr.challenge_scalar(tr_labels::CHALLENGE_SUMCHECK_ROUND),
        ),
        Err(AkitaError::InvalidProof)
    );
}

#[test]
fn compressed_constant_and_linear_rounds_require_degree_bound_one() {
    for coeffs in [vec![F::from_u64(3)], vec![F::from_u64(3), F::from_u64(5)]] {
        let polynomial = UniPoly::from_coeffs(coeffs);
        let claim = polynomial.evaluate(&F::zero()) + polynomial.evaluate(&F::one());
        let challenge = F::from_u64(7);
        let compressed = polynomial.compress();
        assert_eq!(compressed.degree(), 1);
        let proof = SumcheckProof {
            round_polys: vec![compressed],
        };

        assert_eq!(
            proof.verify::<F, _, _>(claim, 1, 1, &mut new_transcript(), |_| Ok(challenge)),
            Ok((polynomial.evaluate(&challenge), vec![challenge]))
        );
        assert!(matches!(
            proof.verify::<F, _, _>(claim, 1, 0, &mut new_transcript(), |_| panic!(
                "invalid degree must reject before sampling"
            )),
            Err(AkitaError::InvalidInput(_))
        ));
    }
}

struct ZeroSumcheckInstance;

impl SumcheckInstanceProver<F> for ZeroSumcheckInstance {
    fn num_rounds(&self) -> usize {
        4
    }

    fn degree_bound(&self) -> usize {
        1
    }

    fn input_claim(&self) -> F {
        F::zero()
    }

    fn compute_round_univariate(&mut self, _round: usize, _previous_claim: F) -> UniPoly<F> {
        UniPoly::from_coeffs(Vec::new())
    }

    fn ingest_challenge(&mut self, _round: usize, _challenge: F) {}
}

impl SumcheckInstanceVerifier<F> for ZeroSumcheckInstance {
    fn num_rounds(&self) -> usize {
        SumcheckInstanceProver::num_rounds(self)
    }

    fn degree_bound(&self) -> usize {
        SumcheckInstanceProver::degree_bound(self)
    }

    fn input_claim(&self) -> F {
        F::zero()
    }

    fn expected_output_claim(&self, _challenges: &[F]) -> Result<F, AkitaError> {
        Ok(F::zero())
    }
}

#[test]
fn empty_zero_polynomial_round_trips_through_standard_and_batched_sumcheck() {
    let (proof, challenges, final_claim) = ZeroSumcheckInstance
        .prove::<F, _, _>(&mut new_transcript(), sample_round)
        .unwrap();
    assert_eq!(final_claim, F::zero());
    assert!(proof
        .round_polys
        .iter()
        .all(|poly| poly.coeffs_except_linear_term == [F::zero()]));
    assert_eq!(
        ZeroSumcheckInstance.verify::<F, _, _>(&proof, &mut new_transcript(), sample_round),
        Ok(challenges)
    );

    let (proof, challenges) = akita_sumcheck::prove_batched_sumcheck::<F, _, F, _>(
        vec![&mut ZeroSumcheckInstance],
        &mut new_transcript(),
        |tr| tr.challenge_scalar(tr_labels::CHALLENGE_SUMCHECK_ROUND),
    )
    .unwrap();
    assert_eq!(
        akita_sumcheck::verify_batched_sumcheck::<F, _, F, _>(
            &proof,
            vec![&ZeroSumcheckInstance],
            &mut new_transcript(),
            |tr| tr.challenge_scalar(tr_labels::CHALLENGE_SUMCHECK_ROUND),
        ),
        Ok(challenges)
    );
}
