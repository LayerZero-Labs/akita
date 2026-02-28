//! Batched sumcheck protocol.
//!
//! Implements the standard technique for batching parallel sumchecks to reduce
//! verifier cost and proof size.
//!
//! For details, refer to Jim Posen's ["Perspectives on Sumcheck Batching"](https://hackmd.io/s/HyxaupAAA).
//! We do what they describe as "front-loaded" batch sumcheck.
//!
//! Adapted from Jolt's `BatchedSumcheck` implementation.

use super::{SumcheckInstanceProver, SumcheckInstanceVerifier, SumcheckProof, UniPoly};
use crate::error::HachiError;
use crate::protocol::transcript::labels;
use crate::protocol::transcript::Transcript;
use crate::{CanonicalField, FieldCore};

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
            result[i] = result[i] + *c * *coeff;
        }
    }
    UniPoly::from_coeffs(result)
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
pub fn prove_batched_sumcheck<F, T, E, S>(
    mut instances: Vec<&mut dyn SumcheckInstanceProver<E>>,
    transcript: &mut T,
    mut sample_challenge: S,
) -> Result<(SumcheckProof<E>, Vec<E>), HachiError>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
    E: FieldCore + CanonicalField,
    S: FnMut(&mut T) -> E,
{
    assert!(!instances.is_empty());

    let max_num_rounds = instances
        .iter()
        .map(|inst| inst.num_rounds())
        .max()
        .unwrap();

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

    let two_inv = E::from_u64(2)
        .inv()
        .expect("2 must be invertible in the field");

    let mut round_polys = Vec::with_capacity(max_num_rounds);
    let mut challenges = Vec::with_capacity(max_num_rounds);

    for round in 0..max_num_rounds {
        let univariate_polys: Vec<UniPoly<E>> = instances
            .iter_mut()
            .zip(individual_claims.iter())
            .map(|(inst, previous_claim)| {
                let n = inst.num_rounds();
                let offset = max_num_rounds - n;
                let active = round >= offset && round < offset + n;
                if active {
                    inst.compute_round_univariate(round - offset, *previous_claim)
                } else {
                    // Variable is "dummy" for this instance: polynomial is independent of it,
                    // so the round univariate is constant with H(0)=H(1)=previous_claim/2.
                    UniPoly::from_coeffs(vec![*previous_claim * two_inv])
                }
            })
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
        for inst in instances.iter_mut() {
            let n = inst.num_rounds();
            let offset = max_num_rounds - n;
            let active = round >= offset && round < offset + n;
            if active {
                inst.ingest_challenge(round - offset, r_j);
            }
        }

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
/// - verifies the proof against the batched claim,
/// - checks that the final output matches the batched expected output claims.
///
/// Returns the challenge vector on success.
///
/// # Panics
///
/// Panics if `verifiers` is empty.
///
/// # Errors
///
/// Returns [`HachiError::InvalidProof`] if the batched output claim does not
/// match the expected value, or propagates per-round verification errors.
pub fn verify_batched_sumcheck<F, T, E, S>(
    proof: &SumcheckProof<E>,
    verifiers: Vec<&dyn SumcheckInstanceVerifier<E>>,
    transcript: &mut T,
    mut sample_challenge: S,
) -> Result<Vec<E>, HachiError>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
    E: FieldCore + CanonicalField,
    S: FnMut(&mut T) -> E,
{
    assert!(!verifiers.is_empty());

    let max_degree = verifiers.iter().map(|v| v.degree_bound()).max().unwrap();
    let max_num_rounds = verifiers.iter().map(|v| v.num_rounds()).max().unwrap();

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

    // Compute the expected batched output claim from each verifier instance.
    let expected_output_claim: E = verifiers
        .iter()
        .zip(batching_coeffs.iter())
        .map(|(v, coeff)| {
            let offset = max_num_rounds - v.num_rounds();
            let r_slice = &r_sumcheck[offset..offset + v.num_rounds()];
            v.expected_output_claim(r_slice) * *coeff
        })
        .fold(E::zero(), |a, v| a + v);

    if output_claim != expected_output_claim {
        return Err(HachiError::InvalidProof);
    }

    Ok(r_sumcheck)
}
