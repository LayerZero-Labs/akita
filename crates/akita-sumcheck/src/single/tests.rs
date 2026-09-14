use super::*;
use crate::{CompressedUniPoly, UniPoly};
use akita_algebra::poly::multilinear_eval;
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

    fn compute_round_univariate(&mut self, _round: usize, _claim: F) -> UniPoly<F> {
        let half = self.evaluations.len() / 2;
        let (zero, one) = (0..half).fold((F::zero(), F::zero()), |(zero, one), index| {
            (
                zero + self.evaluations[2 * index],
                one + self.evaluations[2 * index + 1],
            )
        });
        UniPoly::from_coeffs(vec![zero, one - zero])
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
    let (proof, prover_point, _) =
        prove_sumcheck::<F, _, F, _, _>(&mut prover, &mut transcript(), sample).unwrap();

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
                round_polys: vec![CompressedUniPoly {
                    coeffs_except_linear_term: vec![],
                }],
            },
            AkitaError::InvalidProof,
        ),
        (
            SumcheckProof {
                round_polys: vec![CompressedUniPoly {
                    coeffs_except_linear_term: vec![F::zero(); 2],
                }],
            },
            AkitaError::InvalidInput("sumcheck round poly degree 2 exceeds bound 1".into()),
        ),
    ];
    for (proof, expected) in cases {
        let mut samples = 0;
        let result = verify_sumcheck_rounds::<F, _, F, _>(
            &proof,
            F::zero(),
            1,
            1,
            &mut transcript(),
            |_| {
                samples += 1;
                Ok(F::zero())
            },
        );
        assert_eq!(result, Err(expected));
        assert_eq!(samples, 0);
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

    fn compute_round_eq_factored(&mut self, round: usize) -> EqFactoredUniPoly<F> {
        let [a, b, c, d] = self.coefficients;
        let coefficients = if round == 0 {
            vec![a + c * self.equality[1], b + d * self.equality[1]]
        } else {
            let challenge = self.first_challenge.unwrap();
            vec![a + b * challenge, c + d * challenge]
        };
        EqFactoredUniPoly::from_q_coeffs(coefficients)
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
    proof.round_polys[1].coeffs_except_constant_term[0] += F::one();
    assert_eq!(replay(&proof), Err(AkitaError::InvalidProof));
}
