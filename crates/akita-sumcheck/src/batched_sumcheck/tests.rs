use super::*;
use akita_transcript::AkitaTranscript;
use jolt_field::{Fp64, One, Zero};

type F = Fp64<4294967197>;

struct DummyVerifier {
    rounds: usize,
}

impl SumcheckInstanceVerifier<F> for DummyVerifier {
    fn num_rounds(&self) -> usize {
        self.rounds
    }

    fn degree_bound(&self) -> usize {
        2
    }

    fn input_claim(&self) -> F {
        F::zero()
    }

    fn expected_output_claim(&self, _challenges: &[F]) -> Result<F, AkitaError> {
        Ok(F::one())
    }
}

#[test]
fn batched_expected_claim_rejects_malformed_shapes() {
    let verifier = DummyVerifier { rounds: 3 };
    let verifiers: Vec<&dyn SumcheckInstanceVerifier<F>> = vec![&verifier];
    assert!(
        compute_batched_expected_output_claim(verifiers.clone(), &[], 2, &[F::zero(); 2]).is_err()
    );
    assert!(
        compute_batched_expected_output_claim(verifiers, &[F::one()], 2, &[F::zero(); 2],).is_err()
    );
}

// ---- Determinism fixture: sum over {0,1}^n of a(x) * b(x). ----------------

/// Deterministic pseudo-random table of `2^rounds` field elements.
fn deterministic_evals(rounds: usize, seed: u64) -> Vec<F> {
    let mut state = seed;
    (0..1usize << rounds)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            F::from_u64(state >> 11)
        })
        .collect()
}

/// Degree-2 round univariate of `sum_x a(x) * b(x)`, binding the top
/// variable: `g(X) = sum_i (a_lo + X (a_hi - a_lo)) (b_lo + X (b_hi - b_lo))`.
fn product_round_univariate(a: &[F], b: &[F]) -> UniPoly<F> {
    let half = a.len() / 2;
    let mut c0 = F::zero();
    let mut c1 = F::zero();
    let mut c2 = F::zero();
    for i in 0..half {
        let (a_lo, a_hi) = (a[i], a[half + i]);
        let (b_lo, b_hi) = (b[i], b[half + i]);
        let da = a_hi - a_lo;
        let db = b_hi - b_lo;
        c0 += a_lo * b_lo;
        c1 += a_lo * db + b_lo * da;
        c2 += da * db;
    }
    UniPoly::from_coeffs(vec![c0, c1, c2])
}

fn fold_top_variable(evals: &mut Vec<F>, r: F) {
    let half = evals.len() / 2;
    for i in 0..half {
        let lo = evals[i];
        let hi = evals[half + i];
        evals[i] = lo + r * (hi - lo);
    }
    evals.truncate(half);
}

fn product_claim(a: &[F], b: &[F]) -> F {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| *x * *y)
        .fold(F::zero(), |acc, v| acc + v)
}

struct ProductInstance {
    a: Vec<F>,
    b: Vec<F>,
    rounds: usize,
}

impl ProductInstance {
    fn new(rounds: usize, seed: u64) -> Self {
        Self {
            a: deterministic_evals(rounds, seed),
            b: deterministic_evals(rounds, seed ^ 0x9e3779b97f4a7c15),
            rounds,
        }
    }
}

impl SumcheckInstanceProver<F> for ProductInstance {
    fn num_rounds(&self) -> usize {
        self.rounds
    }

    fn degree_bound(&self) -> usize {
        2
    }

    fn input_claim(&self) -> F {
        product_claim(&self.a, &self.b)
    }

    fn compute_round_univariate(&mut self, _round: usize, _previous_claim: F) -> UniPoly<F> {
        product_round_univariate(&self.a, &self.b)
    }

    fn ingest_challenge(&mut self, _round: usize, r_round: F) {
        fold_top_variable(&mut self.a, r_round);
        fold_top_variable(&mut self.b, r_round);
    }
}

impl SumcheckInstanceVerifier<F> for ProductInstance {
    fn num_rounds(&self) -> usize {
        self.rounds
    }

    fn degree_bound(&self) -> usize {
        2
    }

    fn input_claim(&self) -> F {
        product_claim(&self.a, &self.b)
    }

    fn expected_output_claim(&self, challenges: &[F]) -> Result<F, AkitaError> {
        let mut a = self.a.clone();
        let mut b = self.b.clone();
        for r in challenges {
            fold_top_variable(&mut a, *r);
            fold_top_variable(&mut b, *r);
        }
        Ok(a[0] * b[0])
    }
}

/// Mixed round counts so several instances hit the front-loaded padding
/// path. Every round of this batch stays far below
/// `PARALLEL_MIN_ROUND_WORK`, so under `--features parallel` the driver
/// takes the serial small-work branch.
const SMALL_DETERMINISM_SHAPES: [(usize, u64); 4] = [(3, 11), (2, 22), (3, 33), (1, 44)];
/// Early rounds of this batch carry 2^14 + 2^13 + 2^14 + 2^12 live points,
/// above `PARALLEL_MIN_ROUND_WORK`, so under `--features parallel` the
/// driver fans out over rayon; late rounds fall back below the cutoff,
/// exercising the mid-proof switch.
const LARGE_DETERMINISM_SHAPES: [(usize, u64); 4] = [(14, 5), (13, 6), (14, 7), (12, 8)];
const DETERMINISM_DOMAIN: &[u8] = b"test/batched-sumcheck-determinism";
const DETERMINISM_CHALLENGE: &[u8] = b"sumcheck-round";

/// The batched driver must be deterministic and instance-order-preserving
/// regardless of the `parallel` feature: the transcript inputs here are
/// fixed, so the proof, challenges, and claims produced under
/// `--features parallel` must be byte-identical to the serial ones. The
/// naive reference recomputation below is written serially in instance
/// order; any driver-side reordering or nondeterminism fails the
/// round-by-round equality.
fn run_determinism_check(shapes: &[(usize, u64)]) {
    let mut provers: Vec<ProductInstance> = shapes
        .iter()
        .map(|&(rounds, seed)| ProductInstance::new(rounds, seed))
        .collect();
    let max_num_rounds = shapes.iter().map(|&(r, _)| r).max().unwrap();

    let mut transcript = AkitaTranscript::<F>::new(DETERMINISM_DOMAIN);
    let instances: Vec<&mut (dyn SumcheckInstanceProver<F> + Send)> = provers
        .iter_mut()
        .map(|inst| inst as &mut (dyn SumcheckInstanceProver<F> + Send))
        .collect();
    let (proof, challenges) = prove_batched_sumcheck(instances, &mut transcript, |t| {
        t.challenge_scalar(DETERMINISM_CHALLENGE)
    })
    .expect("batched proving succeeds");
    assert_eq!(proof.round_polys.len(), max_num_rounds);
    assert_eq!(challenges.len(), max_num_rounds);

    // Verifier replay: same transcript inputs, so it must re-derive the
    // prover's challenges, and the final oracle check must pass.
    let verifiers_owned: Vec<ProductInstance> = shapes
        .iter()
        .map(|&(rounds, seed)| ProductInstance::new(rounds, seed))
        .collect();
    let verifiers: Vec<&dyn SumcheckInstanceVerifier<F>> = verifiers_owned
        .iter()
        .map(|v| v as &dyn SumcheckInstanceVerifier<F>)
        .collect();
    let mut verifier_transcript = AkitaTranscript::<F>::new(DETERMINISM_DOMAIN);
    let round_result =
        verify_batched_sumcheck_rounds(&proof, verifiers.clone(), &mut verifier_transcript, |t| {
            t.challenge_scalar(DETERMINISM_CHALLENGE)
        })
        .expect("round replay succeeds");
    assert_eq!(round_result.r_sumcheck, challenges);
    let expected_output_claim = compute_batched_expected_output_claim(
        verifiers,
        &round_result.batching_coeffs,
        round_result.max_num_rounds,
        &round_result.r_sumcheck,
    )
    .expect("expected output claim");
    check_batched_output_claim(round_result.output_claim, expected_output_claim)
        .expect("final oracle check");

    // Naive serial reference: recompute every round's batched compressed
    // polynomial from fresh tables, strictly in instance order, and
    // compare against the transcript-visible proof round by round.
    let mut tables: Vec<ProductInstance> = shapes
        .iter()
        .map(|&(rounds, seed)| ProductInstance::new(rounds, seed))
        .collect();
    let mut reference_claims: Vec<F> = tables
        .iter()
        .map(|inst| {
            let mut claim = product_claim(&inst.a, &inst.b);
            for _ in 0..max_num_rounds - inst.rounds {
                claim = claim + claim;
            }
            claim
        })
        .collect();
    for (round, (r_j, driver_round_poly)) in challenges
        .iter()
        .copied()
        .zip(proof.round_polys.iter())
        .enumerate()
    {
        let univariates: Vec<UniPoly<F>> = tables
            .iter()
            .zip(reference_claims.iter())
            .map(|(inst, claim)| {
                let offset = max_num_rounds - inst.rounds;
                if round >= offset {
                    product_round_univariate(&inst.a, &inst.b)
                } else {
                    UniPoly::from_coeffs(vec![claim.half()])
                }
            })
            .collect();
        let max_len = univariates.iter().map(|p| p.coeffs.len()).max().unwrap();
        let mut batched = vec![F::zero(); max_len];
        for (poly, coeff) in univariates.iter().zip(round_result.batching_coeffs.iter()) {
            for (i, c) in poly.coeffs.iter().enumerate() {
                batched[i] += *c * *coeff;
            }
        }
        assert_eq!(
            &UniPoly::from_coeffs(batched).compress(),
            driver_round_poly,
            "round {round}: driver output differs from the serial reference"
        );
        for ((inst, claim), poly) in tables
            .iter_mut()
            .zip(reference_claims.iter_mut())
            .zip(univariates.iter())
        {
            *claim = poly.evaluate(&r_j);
            let offset = max_num_rounds - inst.rounds;
            if round >= offset {
                fold_top_variable(&mut inst.a, r_j);
                fold_top_variable(&mut inst.b, r_j);
            }
        }
    }

    // The batched final claim from the reference must equal the
    // verifier-side output claim.
    let reference_output_claim: F = reference_claims
        .iter()
        .zip(round_result.batching_coeffs.iter())
        .map(|(claim, coeff)| *claim * *coeff)
        .fold(F::zero(), |acc, v| acc + v);
    assert_eq!(reference_output_claim, round_result.output_claim);
}

#[test]
fn batched_prove_matches_serial_reference_below_parallel_cutoff() {
    run_determinism_check(&SMALL_DETERMINISM_SHAPES);
}

#[test]
fn batched_prove_matches_serial_reference_above_parallel_cutoff() {
    run_determinism_check(&LARGE_DETERMINISM_SHAPES);
}
