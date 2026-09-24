use super::*;
use jolt_poly::{CompressedPoly, NormalizedPoly, UnivariatePoly};

use akita_algebra::poly::multilinear_eval;
use akita_serialization::{AkitaDeserialize, AkitaSerialize};
use akita_transcript::AkitaTranscript;
use jolt_field::{Field, One, Prime128Offset275 as F, Ring, Zero};

fn transcript() -> AkitaTranscript<F> {
    AkitaTranscript::prover(b"single-sumcheck", b"fixture")
}

fn sample(transcript: &mut AkitaTranscript<F>) -> Result<F, AkitaError> {
    Ok(transcript.challenge_scalar(labels::CHALLENGE_SUMCHECK_ROUND))
}

struct DenseInstance {
    evaluations: Vec<F>,
    rounds: usize,
    claim: F,
}

impl DenseInstance {
    fn new(evaluations: Vec<F>, rounds: usize, claim: F) -> Self {
        Self {
            evaluations,
            rounds,
            claim,
        }
    }
}

impl SumcheckInstanceProver<F> for DenseInstance {
    fn num_rounds(&self) -> usize {
        self.rounds
    }

    fn degree_bound(&self) -> usize {
        1
    }

    fn input_claim(&self) -> F {
        self.claim
    }

    fn compute_round_univariate(&mut self, _round: usize, _claim: F) -> UnivariatePoly<F> {
        let half = self.evaluations.len() / 2;
        let (zero, one) = (0..half).fold((F::zero(), F::zero()), |(zero, one), index| {
            (
                zero + self.evaluations[2 * index],
                one + self.evaluations[2 * index + 1],
            )
        });
        UnivariatePoly::new(vec![zero, one - zero])
    }

    fn ingest_challenge(&mut self, _round: usize, challenge: F) {
        let half = self.evaluations.len() / 2;
        for index in 0..half {
            let zero = self.evaluations[2 * index];
            self.evaluations[index] = zero + challenge * (self.evaluations[2 * index + 1] - zero);
        }
        self.evaluations.truncate(half);
    }
}

impl SumcheckInstanceVerifier<F> for DenseInstance {
    fn num_rounds(&self) -> usize {
        self.rounds
    }

    fn degree_bound(&self) -> usize {
        1
    }

    fn input_claim(&self) -> F {
        self.claim
    }

    fn expected_output_claim(&self, challenges: &[F]) -> Result<F, AkitaError> {
        multilinear_eval(&self.evaluations, challenges)
    }
}

#[test]
fn standard_roundtrip_and_terminal_claim_check() {
    let evaluations = (1..=16).map(F::from_u64).collect::<Vec<_>>();
    let claim = evaluations.iter().copied().fold(F::zero(), |a, b| a + b);
    let mut prover = DenseInstance::new(evaluations.clone(), 4, claim);
    let (proof, prover_point, _) = prove_sumcheck::<F, _, F, _, _>(
        &mut crate::InfallibleSumcheck(&mut prover),
        &mut transcript(),
        sample,
    )
    .unwrap();

    let verifier = DenseInstance::new(evaluations.clone(), 4, claim);
    assert_eq!(
        verify_sumcheck::<F, _, F, _, _>(&verifier, &proof, &mut transcript(), sample).unwrap(),
        prover_point
    );
    let wrong = DenseInstance::new(evaluations, 4, claim + F::one());
    assert_eq!(
        verify_sumcheck::<F, _, F, _, _>(&wrong, &proof, &mut transcript(), sample),
        Err(AkitaError::InvalidProof)
    );
}

#[test]
fn malformed_standard_rounds_fail_before_sampling() {
    let cases = [
        (
            SumcheckProof {
                round_polys: vec![],
            },
            AkitaError::InvalidSize {
                expected: 1,
                actual: 0,
            },
        ),
        (
            SumcheckProof {
                round_polys: vec![CompressedPoly::new(vec![])],
            },
            AkitaError::InvalidProof,
        ),
        (
            SumcheckProof {
                round_polys: vec![CompressedPoly::new(vec![F::zero(); 2])],
            },
            AkitaError::InvalidInput("sumcheck round poly degree 2 exceeds bound 1".into()),
        ),
    ];
    for (proof, expected) in cases {
        for verify_full in [false, true] {
            let mut verifier_transcript = transcript();
            let mut samples = 0;
            let result = if verify_full {
                let verifier = DenseInstance::new(vec![F::zero(); 2], 1, F::zero());
                verify_sumcheck::<F, _, F, _, _>(
                    &verifier,
                    &proof,
                    &mut verifier_transcript,
                    |_| {
                        samples += 1;
                        Ok(F::zero())
                    },
                )
                .map(|_| ())
            } else {
                verify_sumcheck_rounds::<F, _, F, _>(
                    &proof,
                    F::zero(),
                    1,
                    1,
                    &mut verifier_transcript,
                    |_| {
                        samples += 1;
                        Ok(F::zero())
                    },
                )
                .map(|_| ())
            };
            assert_eq!(result, Err(expected.clone()));
            assert_eq!(samples, 0);
            assert_eq!(
                verifier_transcript.challenge_bytes(b"test/rejected-round-state", 32),
                transcript().challenge_bytes(b"test/rejected-round-state", 32)
            );
        }
    }
}

struct OneRoundEqInstance {
    tau: F,
    split: GruenSplitEq<F>,
    q_coeffs: Vec<F>,
}

impl OneRoundEqInstance {
    fn new(tau: F, q_coeffs: Vec<F>) -> Self {
        Self {
            tau,
            split: GruenSplitEq::new(&[tau]).unwrap(),
            q_coeffs,
        }
    }

    fn q_at(&self, point: F) -> F {
        UnivariatePoly::new(self.q_coeffs.clone()).evaluate(point)
    }

    fn claim(&self) -> F {
        let q_zero = self.q_at(F::zero());
        let q_one = self.q_at(F::one());
        (F::one() - self.tau) * q_zero + self.tau * q_one
    }
}

impl EqFactoredSumcheckInstanceProver<F> for OneRoundEqInstance {
    fn num_rounds(&self) -> usize {
        1
    }

    fn degree_bound(&self) -> usize {
        self.q_coeffs.len().saturating_sub(1)
    }

    fn input_claim(&self) -> F {
        self.claim()
    }

    fn current_tau(&self) -> F {
        self.split.current_tau()
    }

    fn compute_round_eq_factored(&mut self, _round: usize) -> NormalizedPoly<F> {
        NormalizedPoly::from_q_coefficients(self.q_coeffs.clone())
    }

    fn ingest_challenge(&mut self, _round: usize, challenge: F) {
        self.split.bind(challenge);
    }
}

#[test]
fn equality_factored_rejects_old_wire_forgery_when_tau_is_zero() {
    let q_coeffs = vec![F::from_u64(3), F::from_u64(5), F::from_u64(7)];
    let instance = OneRoundEqInstance::new(F::zero(), q_coeffs);
    let proof = EqFactoredSumcheckProof {
        round_polys: vec![NormalizedPoly::new(
            // Under the old `[q_0, q_2]` convention, choosing `q_0 = T`
            // collapsed the scaled claim to zero and left `q_2` unconstrained.
            vec![instance.claim(), F::from_u64(101)],
        )],
    };
    let challenge = F::from_u64(11);

    assert_eq!(
        verify_eq_factored_sumcheck::<F, _, F, _, _>(
            &proof,
            &[instance.tau],
            instance.claim(),
            instance.degree_bound(),
            &mut transcript(),
            |_| Ok(challenge),
            |_| Ok(instance.q_at(challenge)),
        ),
        Err(AkitaError::InvalidProof)
    );
}

#[test]
fn equality_factored_wire_contains_every_nonconstant_coefficient() {
    let q_coeffs = vec![
        F::from_u64(3),
        F::from_u64(5),
        F::from_u64(7),
        F::from_u64(11),
    ];
    let poly = NormalizedPoly::from_q_coefficients(q_coeffs.clone());
    let mut encoded = Vec::new();
    poly.serialize_uncompressed(&mut encoded).unwrap();

    let mut expected = Vec::new();
    for coefficient in &q_coeffs[1..] {
        coefficient.serialize_uncompressed(&mut expected).unwrap();
    }
    assert_eq!(encoded, expected);
    assert_eq!(
        NormalizedPoly::<F>::deserialize_uncompressed(&encoded[..], &3).unwrap(),
        poly
    );
}

#[test]
fn equality_factored_degree_zero_round_has_an_empty_message() {
    for tau in [F::zero(), F::one(), F::from_u64(7)] {
        for challenge in [F::zero(), F::one(), F::from_u64(11)] {
            let q_coeffs = vec![F::from_u64(23)];
            let mut prover = OneRoundEqInstance::new(tau, q_coeffs.clone());
            let input_claim = prover.claim();
            let (proof, _, final_claim) =
                prove_eq_factored_sumcheck::<F, _, F, _, _>(&mut prover, &mut transcript(), |_| {
                    Ok(challenge)
                })
                .unwrap();

            assert!(proof.round_polys[0].coefficients().is_empty());
            let mut encoded = Vec::new();
            proof.round_polys[0]
                .serialize_uncompressed(&mut encoded)
                .unwrap();
            assert!(encoded.is_empty());
            assert_eq!(final_claim, q_coeffs[0]);
            assert_eq!(
                verify_eq_factored_sumcheck::<F, _, F, _, _>(
                    &proof,
                    &[tau],
                    input_claim,
                    0,
                    &mut transcript(),
                    |_| Ok(challenge),
                    |_| Ok(q_coeffs[0]),
                ),
                Ok(vec![challenge])
            );
        }
    }
}

struct EqInstance {
    equality: [F; 2],
    split: GruenSplitEq<F>,
    coefficients: [F; 4],
    first_challenge: Option<F>,
}

impl EqInstance {
    fn new(equality: [F; 2], coefficients: [F; 4]) -> Self {
        Self {
            equality,
            split: GruenSplitEq::new(&equality).unwrap(),
            coefficients,
            first_challenge: None,
        }
    }

    fn evaluate(&self, x: F, y: F) -> F {
        let [a, b, c, d] = self.coefficients;
        a + b * x + c * y + d * x * y
    }
}

impl EqFactoredSumcheckInstanceProver<F> for EqInstance {
    fn num_rounds(&self) -> usize {
        2
    }

    fn degree_bound(&self) -> usize {
        1
    }

    fn input_claim(&self) -> F {
        self.evaluate(self.equality[0], self.equality[1])
    }

    fn current_tau(&self) -> F {
        self.split.current_tau()
    }

    fn compute_round_eq_factored(&mut self, round: usize) -> NormalizedPoly<F> {
        let [a, b, c, d] = self.coefficients;
        let coefficients = if round == 0 {
            vec![a + c * self.equality[1], b + d * self.equality[1]]
        } else {
            let challenge = self.first_challenge.unwrap();
            vec![a + b * challenge, c + d * challenge]
        };
        NormalizedPoly::from_q_coefficients(coefficients)
    }

    fn ingest_challenge(&mut self, round: usize, challenge: F) {
        if round == 0 {
            self.first_challenge = Some(challenge);
        }
        self.split.bind(challenge);
    }
}

#[test]
fn equality_factored_replay_rejects_late_tampering_after_a_vanished_factor() {
    let equality = [F::from_u64(2), F::from_u64(5)];
    let coefficients = [
        F::from_u64(3),
        F::from_u64(7),
        F::from_u64(11),
        F::from_u64(13),
    ];
    let point = [F::from_u64(3).inverse().unwrap(), F::from_u64(17)];
    let mut prover = EqInstance::new(equality, coefficients);
    let mut round = 0;
    let (mut proof, _, _) =
        prove_eq_factored_sumcheck::<F, _, F, _, _>(&mut prover, &mut transcript(), |_| {
            let challenge = point[round];
            round += 1;
            Ok(challenge)
        })
        .unwrap();
    let expected = EqInstance::new(equality, coefficients).evaluate(point[0], point[1]);
    let replay = |proof: &EqFactoredSumcheckProof<F>| {
        let mut round = 0;
        verify_eq_factored_sumcheck::<F, _, F, _, _>(
            proof,
            &equality,
            EqInstance::new(equality, coefficients).input_claim(),
            1,
            &mut transcript(),
            |_| {
                let challenge = point[round];
                round += 1;
                Ok(challenge)
            },
            |_| Ok(expected),
        )
    };
    assert_eq!(replay(&proof), Ok(point.to_vec()));
    let mut coefficients = proof.round_polys[1].coefficients().to_vec();
    coefficients[0] += F::one();
    proof.round_polys[1] = NormalizedPoly::new(coefficients);
    assert_eq!(replay(&proof), Err(AkitaError::InvalidProof));
}

#[test]
fn arithmetic_cutover_preserves_proof_and_transcript_bytes() {
    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|byte| format!("{byte:02x}")).collect()
    }

    // Frozen from Akita a5f3629b, before the jolt-poly arithmetic cutover.
    let evaluations = (1..=16).map(F::from_u64).collect::<Vec<_>>();
    let claim = evaluations.iter().copied().fold(F::zero(), |a, b| a + b);
    let mut standard = DenseInstance::new(evaluations, 4, claim);
    let (standard_proof, standard_point, _) = prove_sumcheck::<F, _, F, _, _>(
        &mut crate::InfallibleSumcheck(&mut standard),
        &mut transcript(),
        sample,
    )
    .unwrap();
    let mut standard_bytes = Vec::new();
    standard_proof
        .serialize_uncompressed(&mut standard_bytes)
        .unwrap();
    let mut standard_challenges = Vec::new();
    for point in standard_point {
        point
            .serialize_uncompressed(&mut standard_challenges)
            .unwrap();
    }
    assert_eq!(
        hex(&standard_bytes),
        "4000000000000000000000000000000056be51b170e69ddb9503b89d67900b5a6910991f8c4c6e05f8cd2fe54ecc91c47f4db5773aa6f5b93fd459b4fca014b4"
    );
    assert_eq!(
        hex(&standard_challenges),
        "056f542c9c79e776e5006ee719e48296c70bbcf154d6e7450bf394c50601e3a56330fa19fd9fcfed507b7050b5ee72f4ed551a9d2c9f4d1e798da70c51a60eac"
    );

    let mut normalized = OneRoundEqInstance::new(
        F::from_u64(3),
        vec![F::from_u64(3), F::from_u64(5), F::from_u64(7)],
    );
    let (normalized_proof, normalized_point, _) =
        prove_eq_factored_sumcheck::<F, _, F, _, _>(&mut normalized, &mut transcript(), sample)
            .unwrap();
    let mut normalized_bytes = Vec::new();
    normalized_proof
        .serialize_uncompressed(&mut normalized_bytes)
        .unwrap();
    let mut normalized_challenges = Vec::new();
    for point in normalized_point {
        point
            .serialize_uncompressed(&mut normalized_challenges)
            .unwrap();
    }
    assert_eq!(
        hex(&normalized_bytes),
        "0500000000000000000000000000000007000000000000000000000000000000"
    );
    assert_eq!(
        hex(&normalized_challenges),
        "bd7304a00babde122b20294ee65a3f54"
    );
}
