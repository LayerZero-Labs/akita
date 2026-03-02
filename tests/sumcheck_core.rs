#![allow(missing_docs)]

use std::time::Instant;

use hachi_pcs::algebra::poly::multilinear_eval;
use hachi_pcs::algebra::Fp64;
use hachi_pcs::error::HachiError;
use hachi_pcs::protocol::transcript::labels;
use hachi_pcs::protocol::{
    prove_sumcheck, verify_sumcheck, Blake2bTranscript, CompressedUniPoly, SumcheckInstanceProver,
    SumcheckInstanceVerifier, SumcheckProof, Transcript, UniPoly,
};
use hachi_pcs::{FieldCore, FieldSampling, FromSmallInt};
use rand::rngs::StdRng;
use rand::RngCore;
use rand::SeedableRng;

type F = Fp64<4294967197>;

#[test]
fn compressed_unipoly_round_trip_and_eval() {
    let mut rng = StdRng::seed_from_u64(123);

    for degree in 0..8usize {
        let coeffs: Vec<F> = (0..=degree).map(|_| F::sample(&mut rng)).collect();
        let poly = UniPoly::from_coeffs(coeffs);

        // Hint is g(0) + g(1).
        let hint = poly.evaluate(&F::zero()) + poly.evaluate(&F::one());

        let compressed = poly.compress();
        let decompressed = compressed.decompress(&hint);

        // Decompression should be functionally equivalent (it may materialize
        // a trailing zero linear term for constant polynomials).
        for x_u64 in [0u64, 1, 2, 3, 17] {
            let x = F::from_u64(x_u64);
            let direct = poly.evaluate(&x);
            let decompressed_direct = decompressed.evaluate(&x);
            let via_hint = compressed.eval_from_hint(&hint, &x);
            assert_eq!(direct, decompressed_direct);
            assert_eq!(direct, via_hint);
        }
    }
}

#[test]
fn sumcheck_proof_verifier_driver_is_transcript_deterministic() {
    // This test checks that the verifier driver absorbs messages and samples challenges
    // consistently, and that the returned (final_claim, r_vec) matches a manual replay.
    let mut rng = StdRng::seed_from_u64(999);

    let num_rounds = 5usize;
    let degree_bound = 7usize;

    // Build random per-round univariates (degree <= degree_bound), compress them.
    let round_polys: Vec<CompressedUniPoly<F>> = (0..num_rounds)
        .map(|_| {
            let deg = (rng.next_u32() as usize) % (degree_bound + 1);
            let coeffs: Vec<F> = (0..=deg).map(|_| F::sample(&mut rng)).collect();
            UniPoly::from_coeffs(coeffs).compress()
        })
        .collect();

    let proof = SumcheckProof { round_polys };
    let claim0 = F::sample(&mut rng);

    // Verifier run.
    let mut t1 = Blake2bTranscript::<F>::new(labels::DOMAIN_HACHI_PROTOCOL);
    let (final_claim_1, r_1) = proof
        .verify::<F, _, _>(claim0, num_rounds, degree_bound, &mut t1, |tr| {
            tr.challenge_scalar(labels::CHALLENGE_SUMCHECK_ROUND)
        })
        .unwrap();

    // Manual replay with a fresh transcript (must match).
    let mut t2 = Blake2bTranscript::<F>::new(labels::DOMAIN_HACHI_PROTOCOL);
    let mut claim = claim0;
    let mut r_manual = Vec::with_capacity(num_rounds);
    for poly in &proof.round_polys {
        t2.append_serde(labels::ABSORB_SUMCHECK_ROUND, poly);
        let r_i = t2.challenge_scalar(labels::CHALLENGE_SUMCHECK_ROUND);
        r_manual.push(r_i);
        claim = poly.eval_from_hint(&claim, &r_i);
    }

    assert_eq!(r_1, r_manual);
    assert_eq!(final_claim_1, claim);
}

struct DenseSumcheckProver<E> {
    evals: Vec<E>,
    num_vars: usize,
}

impl<E: FieldCore> SumcheckInstanceProver<E> for DenseSumcheckProver<E> {
    fn num_rounds(&self) -> usize {
        self.num_vars
    }

    fn degree_bound(&self) -> usize {
        1
    }

    fn input_claim(&self) -> E {
        self.evals.iter().copied().fold(E::zero(), |a, b| a + b)
    }

    fn compute_round_univariate(&mut self, _round: usize, _previous_claim: E) -> UniPoly<E> {
        let half = self.evals.len() / 2;
        let mut eval_0 = E::zero();
        let mut eval_1 = E::zero();
        for i in 0..half {
            eval_0 = eval_0 + self.evals[2 * i];
            eval_1 = eval_1 + self.evals[2 * i + 1];
        }
        UniPoly::from_coeffs(vec![eval_0, eval_1 - eval_0])
    }

    fn ingest_challenge(&mut self, _round: usize, r: E) {
        let half = self.evals.len() / 2;
        let mut new_evals = Vec::with_capacity(half);
        for i in 0..half {
            new_evals.push(self.evals[2 * i] + r * (self.evals[2 * i + 1] - self.evals[2 * i]));
        }
        self.evals = new_evals;
    }
}

struct DenseSumcheckVerifier<E> {
    evals: Vec<E>,
    num_vars: usize,
    claim: E,
}

impl<E: FieldCore> SumcheckInstanceVerifier<E> for DenseSumcheckVerifier<E> {
    fn num_rounds(&self) -> usize {
        self.num_vars
    }

    fn degree_bound(&self) -> usize {
        1
    }

    fn input_claim(&self) -> E {
        self.claim
    }

    fn expected_output_claim(&self, challenges: &[E]) -> Result<E, HachiError> {
        multilinear_eval(&self.evals, challenges)
    }
}

#[test]
fn prove_and_verify_single_sumcheck() {
    let num_vars = 4;
    let n = 1 << num_vars;

    let evals: Vec<F> = (1..=n).map(|i| F::from_u64(i as u64)).collect();
    let claim: F = evals.iter().copied().fold(F::zero(), |a, b| a + b);

    let mut prover = DenseSumcheckProver {
        evals: evals.clone(),
        num_vars,
    };

    let mut prover_transcript = Blake2bTranscript::<F>::new(labels::DOMAIN_HACHI_PROTOCOL);

    let (proof, prover_challenges, _final_claim) =
        prove_sumcheck::<F, _, F, _, _>(&mut prover, &mut prover_transcript, |tr| {
            tr.challenge_scalar(labels::CHALLENGE_SUMCHECK_ROUND)
        })
        .unwrap();

    let verifier = DenseSumcheckVerifier {
        evals,
        num_vars,
        claim,
    };

    let mut verifier_transcript = Blake2bTranscript::<F>::new(labels::DOMAIN_HACHI_PROTOCOL);

    let verifier_challenges =
        verify_sumcheck::<F, _, F, _, _>(&proof, &verifier, &mut verifier_transcript, |tr| {
            tr.challenge_scalar(labels::CHALLENGE_SUMCHECK_ROUND)
        })
        .unwrap();

    assert_eq!(prover_challenges, verifier_challenges);
}

#[test]
fn verify_rejects_wrong_claim() {
    let num_vars = 3;
    let n = 1 << num_vars;

    let evals: Vec<F> = (1..=n).map(|i| F::from_u64(i as u64)).collect();
    let correct_claim: F = evals.iter().copied().fold(F::zero(), |a, b| a + b);
    let wrong_claim = correct_claim + F::one();

    // Prove with correct claim.
    let mut prover = DenseSumcheckProver {
        evals: evals.clone(),
        num_vars,
    };
    let mut pt = Blake2bTranscript::<F>::new(labels::DOMAIN_HACHI_PROTOCOL);

    let (proof, _, _) = prove_sumcheck::<F, _, F, _, _>(&mut prover, &mut pt, |tr| {
        tr.challenge_scalar(labels::CHALLENGE_SUMCHECK_ROUND)
    })
    .unwrap();

    // Verify with *wrong* claim — should fail.
    let verifier = DenseSumcheckVerifier {
        evals,
        num_vars,
        claim: wrong_claim,
    };
    let mut vt = Blake2bTranscript::<F>::new(labels::DOMAIN_HACHI_PROTOCOL);

    let result = verify_sumcheck::<F, _, F, _, _>(&proof, &verifier, &mut vt, |tr| {
        tr.challenge_scalar(labels::CHALLENGE_SUMCHECK_ROUND)
    });

    assert!(result.is_err());
}

/// End-to-end sumcheck over 2^20 random field elements.
///
/// The prover holds a multilinear polynomial f with 2^20 evaluations and
/// proves that Σ_{b ∈ {0,1}^20} f(b) = claimed_sum.  The verifier checks the
/// proof using only the proof transcript and the oracle evaluation f(r).
#[test]
fn e2e_sumcheck_2_pow_20() {
    let num_vars = 20;
    let n: usize = 1 << num_vars; // 1,048,576

    let mut rng = StdRng::seed_from_u64(42);
    let evals: Vec<F> = (0..n).map(|_| F::sample(&mut rng)).collect();
    let claim: F = evals.iter().copied().fold(F::zero(), |a, b| a + b);

    let t0 = Instant::now();

    let mut prover = DenseSumcheckProver {
        evals: evals.clone(),
        num_vars,
    };
    let mut prover_transcript = Blake2bTranscript::<F>::new(labels::DOMAIN_HACHI_PROTOCOL);

    let (proof, prover_challenges, final_claim) =
        prove_sumcheck::<F, _, F, _, _>(&mut prover, &mut prover_transcript, |tr| {
            tr.challenge_scalar(labels::CHALLENGE_SUMCHECK_ROUND)
        })
        .unwrap();

    let prove_time = t0.elapsed();

    // Proof is just 20 compressed univariate polynomials (degree 1 each).
    assert_eq!(proof.round_polys.len(), num_vars);

    // Sanity: final claim must equal f evaluated at the challenge point.
    let oracle_eval = multilinear_eval(&evals, &prover_challenges).unwrap();
    assert_eq!(final_claim, oracle_eval);

    let t1 = Instant::now();

    let verifier = DenseSumcheckVerifier {
        evals,
        num_vars,
        claim,
    };
    let mut verifier_transcript = Blake2bTranscript::<F>::new(labels::DOMAIN_HACHI_PROTOCOL);

    let verifier_challenges =
        verify_sumcheck::<F, _, F, _, _>(&proof, &verifier, &mut verifier_transcript, |tr| {
            tr.challenge_scalar(labels::CHALLENGE_SUMCHECK_ROUND)
        })
        .unwrap();

    let verify_time = t1.elapsed();

    assert_eq!(prover_challenges, verifier_challenges);

    eprintln!(
        "[e2e_sumcheck_2_pow_20] n=2^{num_vars}={n}  \
         prove={prove_time:.2?}  verify={verify_time:.2?}  \
         rounds={} degree=1",
        proof.round_polys.len()
    );
}
