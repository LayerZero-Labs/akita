//! Batched sumcheck protocol.
//!
//! Implements the standard technique for batching parallel sumchecks to reduce
//! verifier cost and proof size.
//!
//! For details, refer to Jim Posen's ["Perspectives on Sumcheck Batching"](https://hackmd.io/s/HyxaupAAA).
//! We do what they describe as "front-loaded" batch sumcheck.
//!
//! Adapted from Jolt's `BatchedSumcheck` implementation.

use crate::{SumcheckInstanceProver, SumcheckInstanceVerifier, SumcheckProof, UniPoly};
use akita_field::AkitaError;
use akita_field::{CanonicalField, FieldCore, FromPrimitiveInt, HalvingField};
use akita_serialization::AkitaSerialize;
use akita_transcript::labels;
use akita_transcript::Transcript;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

fn mul_pow_2<E: FieldCore>(x: E, k: usize) -> E {
    let mut result = x;
    for _ in 0..k {
        result = result + result;
    }
    result
}

fn linear_combination<E: FieldCore>(polys: &[UniPoly<E>], coeffs: &[E]) -> UniPoly<E> {
    let max_len = polys.iter().map(|p| p.coeffs.len()).max().unwrap_or(0);
    let mut result = vec![E::zero(); max_len];
    for (poly, coeff) in polys.iter().zip(coeffs.iter()) {
        for (i, c) in poly.coeffs.iter().enumerate() {
            result[i] += *c * *coeff;
        }
    }
    UniPoly::from_coeffs(result)
}

/// Verifier-side output of the batched sumcheck round replay.
///
/// This carries all transcript-derived values needed for the final oracle check,
/// which is intentionally split out so callers can compute the expected output
/// claim through an external reduction (e.g. Greyhound) before enforcing
/// equality.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchedSumcheckRoundResult<E: FieldCore> {
    /// Final claim produced by replaying all sumcheck rounds.
    pub output_claim: E,
    /// Challenge vector sampled during replay.
    pub r_sumcheck: Vec<E>,
    /// Front-loaded batching coefficient per verifier instance.
    pub batching_coeffs: Vec<E>,
    /// Maximum number of rounds among batched instances.
    pub max_num_rounds: usize,
}

/// Produce a batched sumcheck proof for multiple instances sharing the same
/// variable space, driving the Fiat–Shamir transcript.
///
/// This function:
/// - absorbs each instance's initial claim,
/// - samples batching coefficients (one per instance),
/// - computes a single batched round polynomial per round as a linear
///   combination of the individual round polynomials,
/// - returns a single [`SumcheckProof`] and the derived challenge vector.
///
/// Instances with fewer rounds than the maximum are padded with constant
/// "dummy" round polynomials (the Jolt "front-loaded" approach).
///
/// # Panics
///
/// Panics if `instances` is empty or if 2 is not invertible in the field.
///
/// # Errors
///
/// Returns an error if the field inverse of 2 does not exist.
#[tracing::instrument(skip_all, name = "prove_batched_sumcheck")]
pub fn prove_batched_sumcheck<F, T, E, S>(
    mut instances: Vec<&mut (dyn SumcheckInstanceProver<E> + Send)>,
    transcript: &mut T,
    mut sample_challenge: S,
) -> Result<(SumcheckProof<E>, Vec<E>), AkitaError>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
    E: FieldCore + FromPrimitiveInt + HalvingField + AkitaSerialize + Send + Sync,
    S: FnMut(&mut T) -> E,
{
    if instances.is_empty() {
        return Err(AkitaError::InvalidInput(
            "no sumcheck instances provided".into(),
        ));
    }

    let max_num_rounds = instances
        .iter()
        .map(|inst| inst.num_rounds())
        .max()
        .unwrap(); // safe: non-empty checked above

    // Absorb individual input claims.
    for inst in instances.iter() {
        let claim = inst.input_claim();
        transcript.append_serde(labels::ABSORB_SUMCHECK_CLAIM, &claim);
    }

    // Sample one batching coefficient per instance.
    let batching_coeffs: Vec<E> = (0..instances.len())
        .map(|_| sample_challenge(transcript))
        .collect();

    // To see why we may need to scale by a power of two, consider a batch of
    // two sumchecks:
    //   claim_a = \sum_x P(x)             where x \in {0, 1}^M
    //   claim_b = \sum_{x, y} Q(x, y)     where x \in {0, 1}^M, y \in {0, 1}^N
    // Then the batched sumcheck is:
    //   \sum_{x, y} A * P(x) + B * Q(x, y)  where A and B are batching coefficients
    //   = A * \sum_y \sum_x P(x) + B * \sum_{x, y} Q(x, y)
    //   = A * \sum_y claim_a + B * claim_b
    //   = A * 2^N * claim_a + B * claim_b
    let mut individual_claims: Vec<E> = instances
        .iter()
        .map(|inst| {
            let n = inst.num_rounds();
            let claim = inst.input_claim();
            mul_pow_2(claim, max_num_rounds - n)
        })
        .collect();

    let mut round_polys = Vec::with_capacity(max_num_rounds);
    let mut challenges = Vec::with_capacity(max_num_rounds);

    for round in 0..max_num_rounds {
        let compute_univariate =
            |(inst, previous_claim): (&mut &mut (dyn SumcheckInstanceProver<E> + Send), &E)| {
                let n = inst.num_rounds();
                let offset = max_num_rounds - n;
                let active = round >= offset && round < offset + n;
                if active {
                    inst.compute_round_univariate(round - offset, *previous_claim)
                } else {
                    UniPoly::from_coeffs(vec![previous_claim.half()])
                }
            };
        // With many instances (the fused selector batch carries dozens), the
        // per-instance round computations dominate late rounds whose domains
        // are too small for intra-instance parallelism; fan the instances out.
        #[cfg(feature = "parallel")]
        let univariate_polys: Vec<UniPoly<E>> = instances
            .par_iter_mut()
            .zip(individual_claims.par_iter())
            .map(compute_univariate)
            .collect();
        #[cfg(not(feature = "parallel"))]
        let univariate_polys: Vec<UniPoly<E>> = instances
            .iter_mut()
            .zip(individual_claims.iter())
            .map(compute_univariate)
            .collect();

        let batched_poly = linear_combination(&univariate_polys, &batching_coeffs);

        #[cfg(debug_assertions)]
        {
            let g0 = batched_poly.evaluate(&E::zero());
            let g1 = batched_poly.evaluate(&E::one());
            let batched_claim: E = individual_claims
                .iter()
                .zip(batching_coeffs.iter())
                .map(|(c, b)| *c * *b)
                .fold(E::zero(), |a, v| a + v);
            debug_assert!(
                g0 + g1 == batched_claim,
                "round {round}: H(0) + H(1) != batched claim"
            );
        }

        let compressed = batched_poly.compress();
        transcript.append_serde(labels::ABSORB_SUMCHECK_ROUND, &compressed);
        let r_j = sample_challenge(transcript);
        challenges.push(r_j);

        // Update individual claims from each instance's own univariate.
        for (claim, poly) in individual_claims.iter_mut().zip(univariate_polys.iter()) {
            *claim = poly.evaluate(&r_j);
        }

        // Ingest challenge into each active instance.
        let ingest = |inst: &mut &mut (dyn SumcheckInstanceProver<E> + Send)| {
            let n = inst.num_rounds();
            let offset = max_num_rounds - n;
            let active = round >= offset && round < offset + n;
            if active {
                inst.ingest_challenge(round - offset, r_j);
            }
        };
        #[cfg(feature = "parallel")]
        instances.par_iter_mut().for_each(ingest);
        #[cfg(not(feature = "parallel"))]
        instances.iter_mut().for_each(ingest);

        round_polys.push(compressed);
    }

    for inst in instances.iter_mut() {
        inst.finalize();
    }

    Ok((SumcheckProof { round_polys }, challenges))
}

/// Verify a batched sumcheck proof.
///
/// This function:
/// - absorbs each verifier instance's initial claim,
/// - re-derives the batching coefficients,
/// - computes the batched initial claim,
/// - verifies the proof against the batched claim.
///
/// Returns transcript-derived verifier data for the caller to perform the final
/// expected-output equality check.
///
/// # Panics
///
/// Panics if `verifiers` is empty.
///
/// # Errors
///
/// Propagates per-round verification errors.
pub fn verify_batched_sumcheck_rounds<F, T, E, S>(
    proof: &SumcheckProof<E>,
    verifiers: Vec<&dyn SumcheckInstanceVerifier<E>>,
    transcript: &mut T,
    mut sample_challenge: S,
) -> Result<BatchedSumcheckRoundResult<E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
    E: FieldCore + AkitaSerialize,
    S: FnMut(&mut T) -> E,
{
    if verifiers.is_empty() {
        return Err(AkitaError::InvalidInput(
            "no sumcheck instances provided".into(),
        ));
    }

    let max_degree = verifiers.iter().map(|v| v.degree_bound()).max().unwrap(); // safe: non-empty
    let max_num_rounds = verifiers.iter().map(|v| v.num_rounds()).max().unwrap(); // safe: non-empty

    // Absorb individual input claims.
    for v in verifiers.iter() {
        let claim = v.input_claim();
        transcript.append_serde(labels::ABSORB_SUMCHECK_CLAIM, &claim);
    }

    // Re-derive batching coefficients.
    let batching_coeffs: Vec<E> = (0..verifiers.len())
        .map(|_| sample_challenge(transcript))
        .collect();

    // Compute the combined initial claim with power-of-two scaling.
    let batched_claim: E = verifiers
        .iter()
        .zip(batching_coeffs.iter())
        .map(|(v, coeff)| {
            let n = v.num_rounds();
            let claim = v.input_claim();
            mul_pow_2(claim, max_num_rounds - n) * *coeff
        })
        .fold(E::zero(), |a, v| a + v);

    let (output_claim, r_sumcheck) = proof.verify::<F, T, _>(
        batched_claim,
        max_num_rounds,
        max_degree,
        transcript,
        &mut sample_challenge,
    )?;

    Ok(BatchedSumcheckRoundResult {
        output_claim,
        r_sumcheck,
        batching_coeffs,
        max_num_rounds,
    })
}

/// Compute the expected batched output claim from verifier instances and
/// transcript-derived batching data.
///
/// # Errors
///
/// Returns an error if batching metadata is inconsistent, or propagates errors
/// from verifier `expected_output_claim` calls.
pub fn compute_batched_expected_output_claim<E: FieldCore>(
    verifiers: Vec<&dyn SumcheckInstanceVerifier<E>>,
    batching_coeffs: &[E],
    max_num_rounds: usize,
    r_sumcheck: &[E],
) -> Result<E, AkitaError> {
    if batching_coeffs.len() != verifiers.len() {
        return Err(AkitaError::InvalidSize {
            expected: verifiers.len(),
            actual: batching_coeffs.len(),
        });
    }
    if r_sumcheck.len() != max_num_rounds {
        return Err(AkitaError::InvalidSize {
            expected: max_num_rounds,
            actual: r_sumcheck.len(),
        });
    }
    let expected_output_claim: E = verifiers
        .iter()
        .zip(batching_coeffs.iter())
        .map(|(v, coeff)| {
            let verifier_rounds = v.num_rounds();
            if verifier_rounds > max_num_rounds {
                return Err(AkitaError::InvalidSize {
                    expected: max_num_rounds,
                    actual: verifier_rounds,
                });
            }
            let offset = max_num_rounds - verifier_rounds;
            let r_slice = &r_sumcheck[offset..offset + verifier_rounds];
            v.expected_output_claim(r_slice).map(|val| val * *coeff)
        })
        .try_fold(E::zero(), |a, v| v.map(|val| a + val))?;

    Ok(expected_output_claim)
}

/// Enforce final batched output-claim equality.
///
/// # Errors
///
/// Returns an error if `output_claim != expected_output_claim`.
pub fn check_batched_output_claim<E: FieldCore>(
    output_claim: E,
    expected_output_claim: E,
) -> Result<(), AkitaError> {
    if output_claim != expected_output_claim {
        return Err(AkitaError::InvalidProof);
    }

    Ok(())
}

/// Verify a batched sumcheck proof, including final expected-output equality.
///
/// This convenience wrapper preserves the previous behavior. Callers that need
/// to inject an external reduction should use [`verify_batched_sumcheck_rounds`]
/// and [`check_batched_output_claim`] directly.
///
/// # Errors
///
/// Propagates errors from round verification and output-claim equality check.
#[tracing::instrument(skip_all, name = "verify_batched_sumcheck")]
pub fn verify_batched_sumcheck<F, T, E, S>(
    proof: &SumcheckProof<E>,
    verifiers: Vec<&dyn SumcheckInstanceVerifier<E>>,
    transcript: &mut T,
    mut sample_challenge: S,
) -> Result<Vec<E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
    E: FieldCore + AkitaSerialize,
    S: FnMut(&mut T) -> E,
{
    let round_result = verify_batched_sumcheck_rounds::<F, T, E, _>(
        proof,
        verifiers.clone(),
        transcript,
        &mut sample_challenge,
    )?;
    let expected_output_claim = compute_batched_expected_output_claim(
        verifiers,
        &round_result.batching_coeffs,
        round_result.max_num_rounds,
        &round_result.r_sumcheck,
    )?;
    check_batched_output_claim(round_result.output_claim, expected_output_claim)?;
    Ok(round_result.r_sumcheck)
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_field::Fp64;
    use akita_transcript::AkitaTranscript;

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
            compute_batched_expected_output_claim(verifiers.clone(), &[], 2, &[F::zero(); 2])
                .is_err()
        );
        assert!(
            compute_batched_expected_output_claim(verifiers, &[F::one()], 2, &[F::zero(); 2],)
                .is_err()
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

    /// Mixed round counts so several instances hit the front-loaded padding path.
    const DETERMINISM_SHAPES: [(usize, u64); 4] = [(3, 11), (2, 22), (3, 33), (1, 44)];
    const DETERMINISM_DOMAIN: &[u8] = b"test/batched-sumcheck-determinism";
    const DETERMINISM_CHALLENGE: &[u8] = b"sumcheck-round";

    /// The batched driver must be deterministic and instance-order-preserving
    /// regardless of the `parallel` feature: the transcript inputs here are
    /// fixed, so the proof, challenges, and claims produced under
    /// `--features parallel` must be byte-identical to the serial ones. The
    /// naive reference recomputation below is written serially in instance
    /// order; any driver-side reordering or nondeterminism fails the
    /// round-by-round equality.
    #[test]
    fn batched_prove_matches_serial_reference_under_either_feature() {
        let mut provers: Vec<ProductInstance> = DETERMINISM_SHAPES
            .iter()
            .map(|&(rounds, seed)| ProductInstance::new(rounds, seed))
            .collect();
        let max_num_rounds = DETERMINISM_SHAPES.iter().map(|&(r, _)| r).max().unwrap();

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
        let verifiers_owned: Vec<ProductInstance> = DETERMINISM_SHAPES
            .iter()
            .map(|&(rounds, seed)| ProductInstance::new(rounds, seed))
            .collect();
        let verifiers: Vec<&dyn SumcheckInstanceVerifier<F>> = verifiers_owned
            .iter()
            .map(|v| v as &dyn SumcheckInstanceVerifier<F>)
            .collect();
        let mut verifier_transcript = AkitaTranscript::<F>::new(DETERMINISM_DOMAIN);
        let round_result = verify_batched_sumcheck_rounds(
            &proof,
            verifiers.clone(),
            &mut verifier_transcript,
            |t| t.challenge_scalar(DETERMINISM_CHALLENGE),
        )
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
        let mut tables: Vec<ProductInstance> = DETERMINISM_SHAPES
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
}
