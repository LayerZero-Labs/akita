#![allow(missing_docs)]

use akita_algebra::split_eq::GruenSplitEq;
use akita_error::AkitaError;
use akita_serialization::{AkitaDeserialize, AkitaSerialize};
use akita_sumcheck::{
    advance_eq_factored_claim, CompressedUniPoly, EqFactoredSumcheckInstanceProver,
    EqFactoredSumcheckInstanceProverExt, EqFactoredSumcheckInstanceVerifier,
    EqFactoredSumcheckInstanceVerifierExt, EqFactoredUniPoly, SumcheckInstanceVerifier,
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
    let q_coeffs = vec![F::from_u64(23)];
    let mut prover = ToyEqFactoredInstance::new(F::from_u64(7), q_coeffs.clone());
    let mut prover_transcript = new_transcript();
    let (proof, _, final_claim) = prover
        .prove::<F, _, _>(&mut prover_transcript, |_| Ok(F::from_u64(11)))
        .unwrap();

    assert!(proof.round_polys[0].coeffs_except_constant_term.is_empty());
    let mut encoded = Vec::new();
    proof.round_polys[0]
        .serialize_uncompressed(&mut encoded)
        .unwrap();
    assert!(encoded.is_empty());
    assert_eq!(final_claim, q_coeffs[0]);

    let verifier = ToyEqFactoredInstance::new(F::from_u64(7), q_coeffs);
    let mut verifier_transcript = new_transcript();
    assert_eq!(
        verifier.verify::<F, _, _>(&proof, &mut verifier_transcript, |_| Ok(F::from_u64(11))),
        Ok(vec![F::from_u64(11)])
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

/// Standard-driver instance whose input claim is deliberately inconsistent with
/// its output claim: no honest proof exists, so any acceptance is a soundness
/// break.
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

fn empty_round_proof(num_rounds: usize) -> SumcheckProof<F> {
    SumcheckProof {
        round_polys: vec![
            CompressedUniPoly {
                coeffs_except_linear_term: Vec::new(),
            };
            num_rounds
        ],
    }
}

/// An empty compressed round message has `degree() == 0`, so it passes any
/// degree bound, and `eval_from_hint` evaluates it to zero without reading the
/// hint. Before the round-message validation, a proof of all-empty rounds drove
/// the running claim to zero regardless of `input_claim`, and any instance whose
/// expected output claim is zero accepted it.
#[test]
fn standard_sumcheck_rejects_empty_round_messages() {
    let verifier = FalseClaimInstance;
    let proof = empty_round_proof(verifier.num_rounds());
    let mut transcript = new_transcript();

    let result = verifier.verify::<F, _, _>(&proof, &mut transcript, sample_round);

    assert_eq!(result, Err(AkitaError::InvalidProof));
}

/// The raw [`SumcheckProof::verify`] driver shares the same validation, so it
/// rejects the same message before absorbing anything into the transcript.
#[test]
fn raw_sumcheck_driver_rejects_empty_round_messages() {
    let proof = empty_round_proof(4);
    let mut transcript = new_transcript();

    let result = proof.verify::<F, _, _>(F::one(), 4, 3, &mut transcript, sample_round);

    assert_eq!(result, Err(AkitaError::InvalidProof));
}

/// A single stored coefficient is a legitimate linear round message, even though
/// `CompressedUniPoly::degree` reports `0` for it. Rejecting on `degree() == 0`
/// instead of on an empty coefficient vector would reject honest proofs.
#[test]
fn sumcheck_validation_accepts_linear_round_messages() {
    let linear = UniPoly::from_coeffs(vec![F::from_u64(3), F::from_u64(5)]).compress();
    assert_eq!(linear.coeffs_except_linear_term.len(), 1);
    assert_eq!(linear.degree(), 0);

    let proof = SumcheckProof {
        round_polys: vec![linear],
    };

    assert_eq!(proof.validate_round_messages(1, 3), Ok(()));
}

/// The eq-factored driver is not vulnerable and so is not guarded: an empty
/// `q` message leaves the normalized claim untouched rather than zeroing it,
/// because both the `tau`-scaled correction and the evaluation are zero.
#[test]
fn eq_factored_empty_round_message_preserves_the_claim() {
    let empty = EqFactoredUniPoly::<F> {
        coeffs_except_constant_term: Vec::new(),
    };
    let claim = F::from_u64(7);

    let advanced = advance_eq_factored_claim(claim, F::from_u64(2), &empty, F::from_u64(9));

    assert_eq!(advanced, claim);
}
