//! Direct single-instance sumcheck protocol functions.

#[cfg(test)]
use crate::SumcheckInstanceProver;
use crate::{
    EqFactoredSumcheckInstanceProver, EqFactoredSumcheckProof, EqFactoredUniPoly,
    SumcheckInstanceVerifier, SumcheckProof,
};
use akita_algebra::split_eq::GruenSplitEq;
use akita_error::AkitaError;
use akita_serialization::AkitaSerialize;
use akita_transcript::{labels, Transcript};
use jolt_field::{CanonicalEncoding, Field};

pub(crate) fn validate_sumcheck_round_messages<E: Field>(
    proof: &SumcheckProof<E>,
    num_rounds: usize,
    degree_bound: usize,
) -> Result<(), AkitaError> {
    if proof.round_polys.len() != num_rounds {
        return Err(AkitaError::InvalidSize {
            expected: num_rounds,
            actual: proof.round_polys.len(),
        });
    }
    for poly in &proof.round_polys {
        if poly.coeffs_except_linear_term.is_empty() {
            return Err(AkitaError::InvalidProof);
        }
        if poly.degree() > degree_bound {
            return Err(AkitaError::InvalidInput(format!(
                "sumcheck round poly degree {} exceeds bound {}",
                poly.degree(),
                degree_bound
            )));
        }
    }
    Ok(())
}

fn replay_validated_sumcheck_rounds<F, T, E, S>(
    proof: &SumcheckProof<E>,
    mut claim: E,
    transcript: &mut T,
    mut sample_challenge: S,
) -> Result<(E, Vec<E>), AkitaError>
where
    F: Field + CanonicalEncoding,
    T: Transcript<F>,
    E: Field + AkitaSerialize,
    S: FnMut(&mut T) -> Result<E, AkitaError>,
{
    let mut challenges = Vec::with_capacity(proof.round_polys.len());
    for poly in &proof.round_polys {
        transcript.append_serde(labels::ABSORB_SUMCHECK_ROUND, poly);
        let challenge = sample_challenge(transcript)?;
        challenges.push(challenge);
        claim = poly.eval_from_hint(&claim, &challenge);
    }
    Ok((claim, challenges))
}

/// Prove one standard sumcheck instance.
#[tracing::instrument(skip_all, name = "prove_sumcheck")]
#[inline(never)]
pub fn prove_sumcheck<F, T, E, S, P>(
    prover: &mut P,
    transcript: &mut T,
    mut sample_challenge: S,
) -> Result<(SumcheckProof<E>, Vec<E>, E), AkitaError>
where
    F: Field + CanonicalEncoding,
    T: Transcript<F>,
    E: Field + AkitaSerialize,
    S: FnMut(&mut T) -> Result<E, AkitaError>,
    P: crate::SumcheckKernel<E> + ?Sized,
{
    let num_rounds = prover.num_rounds();
    let mut claim = prover.input_claim();
    tracing::debug!(
        is_zero = claim.is_zero(),
        num_rounds,
        "prove_sumcheck input_claim"
    );
    transcript.append_serde(labels::ABSORB_SUMCHECK_CLAIM, &claim);

    let degree_bound = prover.degree_bound();
    let mut round_polys = Vec::with_capacity(num_rounds);
    let mut challenges = Vec::with_capacity(num_rounds);
    for round in 0..num_rounds {
        let _round_span = tracing::info_span!(
            "sumcheck_round",
            round,
            table_len = 1usize << (num_rounds - round)
        )
        .entered();
        let poly = {
            let _span = tracing::info_span!("sumcheck_round_univariate").entered();
            prover.round_polynomial(round, claim)?
        };
        if poly.evaluate(&E::zero()) + poly.evaluate(&E::one()) != claim {
            return Err(AkitaError::InvalidInput(
                "sumcheck round polynomial does not match its input claim".into(),
            ));
        }
        let compressed = poly.compress();
        if compressed.degree() > degree_bound {
            return Err(AkitaError::InvalidInput(format!(
                "sumcheck round poly degree {} exceeds bound {}",
                compressed.degree(),
                degree_bound
            )));
        }
        transcript.append_serde(labels::ABSORB_SUMCHECK_ROUND, &compressed);
        let challenge = sample_challenge(transcript)?;
        claim = compressed.eval_from_hint(&claim, &challenge);
        {
            let _span = tracing::info_span!("sumcheck_round_fold").entered();
            prover.bind_challenge(round, challenge)?;
        }
        challenges.push(challenge);
        round_polys.push(compressed);
    }
    prover.finish()?;
    Ok((SumcheckProof { round_polys }, challenges, claim))
}

/// Validate and replay standard sumcheck rounds without a terminal oracle check.
pub fn verify_sumcheck_rounds<F, T, E, S>(
    proof: &SumcheckProof<E>,
    claim: E,
    num_rounds: usize,
    degree_bound: usize,
    transcript: &mut T,
    sample_challenge: S,
) -> Result<(E, Vec<E>), AkitaError>
where
    F: Field + CanonicalEncoding,
    T: Transcript<F>,
    E: Field + AkitaSerialize,
    S: FnMut(&mut T) -> Result<E, AkitaError>,
{
    validate_sumcheck_round_messages(proof, num_rounds, degree_bound)?;
    replay_validated_sumcheck_rounds::<F, T, E, _>(proof, claim, transcript, sample_challenge)
}

/// Verify one standard sumcheck instance, including its terminal oracle claim.
#[tracing::instrument(skip_all, name = "verify_sumcheck")]
#[inline(never)]
pub fn verify_sumcheck<F, T, E, S, V>(
    verifier: &V,
    proof: &SumcheckProof<E>,
    transcript: &mut T,
    sample_challenge: S,
) -> Result<Vec<E>, AkitaError>
where
    F: Field + CanonicalEncoding,
    T: Transcript<F>,
    E: Field + AkitaSerialize,
    S: FnMut(&mut T) -> Result<E, AkitaError>,
    V: SumcheckInstanceVerifier<E> + ?Sized,
{
    let num_rounds = verifier.num_rounds();
    let degree_bound = verifier.degree_bound();
    validate_sumcheck_round_messages(proof, num_rounds, degree_bound)?;
    let input_claim = verifier.input_claim();
    tracing::debug!(
        is_zero = input_claim.is_zero(),
        num_rounds,
        "verify_sumcheck input_claim"
    );
    transcript.append_serde(labels::ABSORB_SUMCHECK_CLAIM, &input_claim);
    let (final_claim, challenges) = replay_validated_sumcheck_rounds::<F, T, E, _>(
        proof,
        input_claim,
        transcript,
        sample_challenge,
    )?;
    let expected = verifier.expected_output_claim(&challenges)?;
    if final_claim != expected {
        tracing::error!(
            rounds = num_rounds,
            degree_bound,
            diff_is_zero = (final_claim - expected).is_zero(),
            "verify_sumcheck MISMATCH"
        );
        return Err(AkitaError::InvalidProof);
    }
    Ok(challenges)
}

/// Advance the normalized claim for one equality-factored round.
pub fn advance_eq_factored_claim<E: Field>(
    claim: E,
    tau: E,
    poly: &EqFactoredUniPoly<E>,
    challenge: E,
) -> E {
    let constant = claim - tau * poly.nonconstant_term_sum_at_one();
    constant + poly.eval_nonconstant_terms(&challenge)
}

/// Prove one normalized equality-factored sumcheck instance.
#[tracing::instrument(skip_all, name = "prove_eq_factored_sumcheck")]
#[inline(never)]
pub fn prove_eq_factored_sumcheck<F, T, E, S, P>(
    prover: &mut P,
    transcript: &mut T,
    mut sample_challenge: S,
) -> Result<(EqFactoredSumcheckProof<E>, Vec<E>, E), AkitaError>
where
    F: Field + CanonicalEncoding,
    T: Transcript<F>,
    E: Field + AkitaSerialize,
    S: FnMut(&mut T) -> Result<E, AkitaError>,
    P: EqFactoredSumcheckInstanceProver<E> + ?Sized,
{
    let num_rounds = prover.num_rounds();
    let degree_bound = prover.degree_bound();
    let mut claim = prover.input_claim();
    let mut round_polys = Vec::with_capacity(num_rounds);
    let mut challenges = Vec::with_capacity(num_rounds);
    transcript.append_serde(labels::ABSORB_SUMCHECK_CLAIM, &claim);
    for round in 0..num_rounds {
        let poly = prover.compute_round_eq_factored(round);
        if poly.degree() > degree_bound {
            return Err(AkitaError::InvalidInput(format!(
                "eq-factored sumcheck round poly degree {} exceeds bound {}",
                poly.degree(),
                degree_bound
            )));
        }
        transcript.append_serde(labels::ABSORB_SUMCHECK_ROUND, &poly);
        let challenge = sample_challenge(transcript)?;
        claim = advance_eq_factored_claim(claim, prover.current_tau(), &poly, challenge);
        challenges.push(challenge);
        prover.ingest_challenge(round, challenge);
        round_polys.push(poly);
    }
    prover.finalize();
    Ok((EqFactoredSumcheckProof { round_polys }, challenges, claim))
}

/// Verify one normalized equality-factored sumcheck instance.
#[tracing::instrument(skip_all, name = "verify_eq_factored_sumcheck")]
#[inline(never)]
pub fn verify_eq_factored_sumcheck<F, T, E, S, C>(
    proof: &EqFactoredSumcheckProof<E>,
    equality_point: &[E],
    input_claim: E,
    degree_bound: usize,
    transcript: &mut T,
    mut sample_challenge: S,
    expected_output_claim: C,
) -> Result<Vec<E>, AkitaError>
where
    F: Field + CanonicalEncoding,
    T: Transcript<F>,
    E: Field + AkitaSerialize,
    S: FnMut(&mut T) -> Result<E, AkitaError>,
    C: FnOnce(&[E]) -> Result<E, AkitaError>,
{
    if proof.round_polys.len() != equality_point.len() {
        return Err(AkitaError::InvalidSize {
            expected: equality_point.len(),
            actual: proof.round_polys.len(),
        });
    }
    let mut equality = GruenSplitEq::new(equality_point)?;
    let mut claim = input_claim;
    let mut challenges = Vec::with_capacity(equality_point.len());
    transcript.append_serde(labels::ABSORB_SUMCHECK_CLAIM, &claim);
    for (round, poly) in proof.round_polys.iter().enumerate() {
        if poly.degree() > degree_bound {
            return Err(AkitaError::InvalidInput(format!(
                "eq-factored sumcheck round poly degree {} exceeds bound {}",
                poly.degree(),
                degree_bound
            )));
        }
        transcript.append_serde(labels::ABSORB_SUMCHECK_ROUND, poly);
        let challenge = sample_challenge(transcript)?;
        claim = advance_eq_factored_claim(claim, equality.current_tau(), poly, challenge);
        challenges.push(challenge);
        equality.bind(challenge);
        debug_assert_eq!(round + 1, challenges.len());
    }
    if claim != expected_output_claim(&challenges)? {
        return Err(AkitaError::InvalidProof);
    }
    Ok(challenges)
}

#[cfg(test)]
mod fallible_tests;
#[cfg(test)]
mod tests;
