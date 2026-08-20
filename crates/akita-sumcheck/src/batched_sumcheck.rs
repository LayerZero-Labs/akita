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
use akita_error::AkitaError;
use akita_field::{CanonicalField, FieldCore, FromPrimitiveInt, HalvingField};
use akita_serialization::AkitaSerialize;
use akita_transcript::labels;
use akita_transcript::Transcript;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Minimum estimated live work in a round (total hypercube points still held
/// by active instances) before the per-instance fan-out pays for rayon's
/// per-round dispatch. Measured on an Apple M3 Max (14 rayon threads) with
/// product-of-multilinears instances: a 24-instance batch of 2^8-point tables
/// regresses ~1.7x under unconditional fan-out, while rounds carrying 2^13 or
/// more live points break even or win; Aerie's fused selector batch enters at
/// ~23 instances x 2^19 points and gains ~1.9x.
#[cfg(feature = "parallel")]
const PARALLEL_MIN_ROUND_WORK: u64 = 1 << 13;

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
        // Skip the fan-out when the round's live work is too small to amortize
        // rayon's dispatch: an active instance at round `round` still holds
        // 2^(n - (round - offset)) hypercube points, and that table size is
        // what its round univariate and fold each traverse.
        #[cfg(feature = "parallel")]
        let fan_out = instances.len() > 1 && {
            let mut live_points: u64 = 0;
            for inst in instances.iter() {
                let n = inst.num_rounds();
                let offset = max_num_rounds - n;
                if round >= offset && round < offset + n {
                    let remaining = n - (round - offset);
                    live_points = live_points.saturating_add(1u64 << remaining.min(63));
                }
            }
            live_points >= PARALLEL_MIN_ROUND_WORK
        };
        #[cfg(feature = "parallel")]
        let univariate_polys: Vec<UniPoly<E>> = if fan_out {
            instances
                .par_iter_mut()
                .zip(individual_claims.par_iter())
                .map(compute_univariate)
                .collect()
        } else {
            instances
                .iter_mut()
                .zip(individual_claims.iter())
                .map(compute_univariate)
                .collect()
        };
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
        if fan_out {
            instances.par_iter_mut().for_each(ingest);
        } else {
            instances.iter_mut().for_each(ingest);
        }
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
mod tests;
