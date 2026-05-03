//! Sumcheck proof driver functions.
//!
//! Contains the generic prove/verify loops for standard and eq-factored
//! sumchecks, including prefix-round omission variants used by the
//! bivariate-skip optimization.

use super::traits::{
    EqFactoredSumcheckInstanceProver, EqFactoredSumcheckInstanceVerifier,
    EqFactoredSumcheckRoundState, SumcheckInstanceProver, SumcheckInstanceVerifier,
};
use super::types::{EqFactoredSumcheckProof, EqFactoredUniPoly, SumcheckProof};
use crate::{CanonicalField, FieldCore};
use akita_algebra::uni_poly::CompressedUniPoly;
use akita_field::HachiError;
use akita_transcript::labels;
use akita_transcript::Transcript;

#[inline]
pub(crate) fn advance_eq_factored_claim<E: FieldCore>(
    scaled_claim: E,
    claim_scale: E,
    l_at_0: E,
    l_at_1: E,
    poly: &EqFactoredUniPoly<E>,
    r_round: E,
) -> (E, E) {
    let q_0 = poly.constant_term();
    let q_higher_sum = poly.higher_term_sum_at_one();
    let q_known_at_r = poly.eval_known_terms(&r_round);
    let current_scalar = l_at_0 + l_at_1;
    let scaled_linear_term =
        scaled_claim - claim_scale * current_scalar * q_0 - claim_scale * l_at_1 * q_higher_sum;
    let l_at_r = l_at_0 + (l_at_1 - l_at_0) * r_round;
    let next_claim_scale = claim_scale * l_at_1;
    let next_scaled_claim =
        next_claim_scale * l_at_r * q_known_at_r + l_at_r * r_round * scaled_linear_term;
    (next_scaled_claim, next_claim_scale)
}

/// Produce an eq-factored sumcheck proof.
///
/// The prover sends the inner polynomial `q(X)` with its linear coefficient
/// omitted in every round, while the driver maintains the verifier-equivalent
/// scaled claim update.
#[tracing::instrument(skip_all, name = "prove_eq_factored_sumcheck")]
#[inline(never)]
pub(crate) fn prove_eq_factored_sumcheck<F, T, E, S, Inst>(
    instance: &mut Inst,
    transcript: &mut T,
    mut sample_challenge: S,
) -> Result<(EqFactoredSumcheckProof<E>, Vec<E>, E), HachiError>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
    E: FieldCore,
    S: FnMut(&mut T) -> E,
    Inst: EqFactoredSumcheckInstanceProver<E>,
{
    let num_rounds = instance.num_rounds();
    let degree_bound = instance.degree_bound();
    let mut scaled_claim = instance.input_claim();
    let mut claim_scale = E::one();
    let mut round_polys = Vec::with_capacity(num_rounds);
    let mut challenges = Vec::with_capacity(num_rounds);

    transcript.append_serde(labels::ABSORB_SUMCHECK_CLAIM, &scaled_claim);

    for round in 0..num_rounds {
        let poly = instance.compute_round_eq_factored(round);
        if poly.degree() > degree_bound {
            return Err(HachiError::InvalidInput(format!(
                "eq-factored sumcheck round poly degree {} exceeds bound {}",
                poly.degree(),
                degree_bound
            )));
        }

        transcript.append_serde(labels::ABSORB_SUMCHECK_ROUND, &poly);
        let r_i = sample_challenge(transcript);
        let (l_at_0, l_at_1) = instance.current_linear_factor_evals();
        (scaled_claim, claim_scale) =
            advance_eq_factored_claim(scaled_claim, claim_scale, l_at_0, l_at_1, &poly, r_i);
        challenges.push(r_i);
        instance.ingest_challenge(round, r_i);
        round_polys.push(poly);
    }

    instance.finalize();
    Ok((
        EqFactoredSumcheckProof { round_polys },
        challenges,
        scaled_claim,
    ))
}

/// Verify an eq-factored sumcheck proof.
///
/// The verifier absorbs each round message, samples the corresponding
/// challenge, updates the scaled running claim from the current eq-factor
/// evaluations and the transmitted `q(X)` data, and finally checks the
/// expected folded oracle value at the full challenge point.
///
/// This creates and owns the mutable eq-factored round state locally, while
/// keeping `verifier` itself immutable.
#[tracing::instrument(skip_all, name = "verify_eq_factored_sumcheck")]
#[inline(never)]
pub(crate) fn verify_eq_factored_sumcheck<F, T, E, S, V>(
    proof: &EqFactoredSumcheckProof<E>,
    verifier: &V,
    transcript: &mut T,
    mut sample_challenge: S,
) -> Result<Vec<E>, HachiError>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
    E: FieldCore,
    S: FnMut(&mut T) -> E,
    V: EqFactoredSumcheckInstanceVerifier<E>,
{
    let num_rounds = verifier.num_rounds();
    if proof.round_polys.len() != num_rounds {
        return Err(HachiError::InvalidSize {
            expected: num_rounds,
            actual: proof.round_polys.len(),
        });
    }

    let degree_bound = verifier.degree_bound();
    let mut scaled_claim = verifier.input_claim();
    let mut claim_scale = E::one();
    let mut challenges = Vec::with_capacity(num_rounds);
    let mut round_state = verifier.start_round_state();

    transcript.append_serde(labels::ABSORB_SUMCHECK_CLAIM, &scaled_claim);

    for (round, poly) in proof.round_polys.iter().enumerate() {
        if poly.degree() > degree_bound {
            return Err(HachiError::InvalidInput(format!(
                "eq-factored sumcheck round poly degree {} exceeds bound {}",
                poly.degree(),
                degree_bound
            )));
        }

        transcript.append_serde(labels::ABSORB_SUMCHECK_ROUND, poly);
        let r_i = sample_challenge(transcript);
        let (l_at_0, l_at_1) = round_state.current_linear_factor_evals();
        (scaled_claim, claim_scale) =
            advance_eq_factored_claim(scaled_claim, claim_scale, l_at_0, l_at_1, poly, r_i);
        challenges.push(r_i);
        round_state.ingest_challenge(round, r_i);
    }

    let expected = verifier.expected_output_claim(&round_state, &challenges)?;
    if scaled_claim != claim_scale * expected {
        return Err(HachiError::InvalidProof);
    }
    Ok(challenges)
}

/// Produce a sumcheck proof while omitting the first `omitted_prefix_rounds`
/// transcript rounds from the stored proof.
///
/// This still drives the prover in the ordinary strict pipeline
/// `compute message -> absorb challenge -> ingest challenge -> ...`; it only
/// changes which compressed univariates are retained in the returned
/// [`SumcheckProof`]. Callers can use this to serialize early rounds via a
/// stage-local bivariate-skip proof instead of directly in the sumcheck proof.
///
/// # Errors
///
/// Returns an error if `omitted_prefix_rounds` exceeds the instance round
/// count, or if any per-round polynomial exceeds the instance's degree bound.
#[tracing::instrument(skip_all, name = "prove_sumcheck")]
#[inline(never)]
pub(crate) fn prove_sumcheck_with_omitted_prefix_rounds<F, T, E, S, Inst, A>(
    instance: &mut Inst,
    transcript: &mut T,
    mut sample_challenge: S,
    omitted_prefix_rounds: usize,
    mut absorb_after_compute: A,
) -> Result<(SumcheckProof<E>, Vec<E>, E), HachiError>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
    E: FieldCore,
    S: FnMut(&mut T) -> E,
    Inst: SumcheckInstanceProver<E>,
    A: FnMut(usize, &Inst, &mut T) -> Result<(), HachiError>,
{
    let num_rounds = instance.num_rounds();
    if omitted_prefix_rounds > num_rounds {
        return Err(HachiError::InvalidInput(format!(
            "sumcheck omitted_prefix_rounds {omitted_prefix_rounds} exceeds num_rounds {num_rounds}"
        )));
    }

    let mut claim = instance.input_claim();
    tracing::debug!(
        is_zero = claim.is_zero(),
        num_rounds,
        omitted_prefix_rounds,
        "prove_sumcheck input_claim"
    );
    transcript.append_serde(labels::ABSORB_SUMCHECK_CLAIM, &claim);

    let degree_bound = instance.degree_bound();
    let mut round_polys = Vec::with_capacity(num_rounds - omitted_prefix_rounds);
    let mut r = Vec::with_capacity(num_rounds);

    for round in 0..num_rounds {
        let g = instance.compute_round_univariate(round, claim);
        let round_sum = g.evaluate(&E::zero()) + g.evaluate(&E::one());
        debug_assert!(
            round_sum == claim,
            "sumcheck round {round} univariate does not match previous claim hint"
        );

        let compressed = g.compress();
        if compressed.degree() > degree_bound {
            return Err(HachiError::InvalidInput(format!(
                "sumcheck round poly degree {} exceeds bound {}",
                compressed.degree(),
                degree_bound
            )));
        }

        absorb_after_compute(round, instance, transcript)?;
        transcript.append_serde(labels::ABSORB_SUMCHECK_ROUND, &compressed);
        let r_i = sample_challenge(transcript);
        r.push(r_i);

        claim = compressed.eval_from_hint(&claim, &r_i);
        instance.ingest_challenge(round, r_i);
        if round >= omitted_prefix_rounds {
            round_polys.push(compressed);
        }
    }

    instance.finalize();
    Ok((SumcheckProof { round_polys }, r, claim))
}

/// Verify a sumcheck proof whose first `prefix_rounds` rounds are reconstructed by
/// a caller-supplied generator instead of being stored in `proof`.
///
/// The verifier still follows the ordinary transcript pipeline, sampling each
/// challenge only after absorbing that round's compressed univariate. For
/// rounds `round < prefix_rounds`, the compressed univariate is provided by
/// `prefix_round_poly`; later rounds are read from `proof`.
///
/// Returns the full challenge point `r` on success.
///
/// # Errors
///
/// Returns an error if `prefix_rounds` exceeds the verifier round count, if the
/// suffix proof length is inconsistent, if a generated/stored round polynomial
/// exceeds the degree bound, or if the final oracle check fails.
#[tracing::instrument(skip_all, name = "verify_sumcheck")]
#[inline(never)]
pub(crate) fn verify_sumcheck_with_prefix_rounds<F, T, E, S, V, A, P>(
    proof: &SumcheckProof<E>,
    verifier: &V,
    transcript: &mut T,
    mut sample_challenge: S,
    prefix_rounds: usize,
    mut absorb_before_round: A,
    mut prefix_round_poly: P,
) -> Result<Vec<E>, HachiError>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
    E: FieldCore,
    S: FnMut(&mut T) -> E,
    V: SumcheckInstanceVerifier<E>,
    A: FnMut(usize, &mut T) -> Result<(), HachiError>,
    P: FnMut(usize, E, &[E]) -> CompressedUniPoly<E>,
{
    let num_rounds = verifier.num_rounds();
    if prefix_rounds > num_rounds {
        return Err(HachiError::InvalidInput(format!(
            "sumcheck prefix_rounds {prefix_rounds} exceeds num_rounds {num_rounds}"
        )));
    }
    let expected_suffix_rounds = num_rounds - prefix_rounds;
    if proof.round_polys.len() != expected_suffix_rounds {
        return Err(HachiError::InvalidSize {
            expected: expected_suffix_rounds,
            actual: proof.round_polys.len(),
        });
    }

    let mut claim = verifier.input_claim();
    tracing::debug!(
        is_zero = claim.is_zero(),
        num_rounds,
        prefix_rounds,
        "verify_sumcheck input_claim"
    );
    transcript.append_serde(labels::ABSORB_SUMCHECK_CLAIM, &claim);

    let degree_bound = verifier.degree_bound();
    let mut challenges = Vec::with_capacity(num_rounds);
    let mut suffix_iter = proof.round_polys.iter();

    for round in 0..num_rounds {
        absorb_before_round(round, transcript)?;
        let poly = if round < prefix_rounds {
            prefix_round_poly(round, claim, &challenges)
        } else {
            suffix_iter
                .next()
                .cloned()
                .expect("suffix proof length checked above")
        };
        if poly.degree() > degree_bound {
            return Err(HachiError::InvalidInput(format!(
                "sumcheck round poly degree {} exceeds bound {}",
                poly.degree(),
                degree_bound
            )));
        }

        transcript.append_serde(labels::ABSORB_SUMCHECK_ROUND, &poly);
        let r_i = sample_challenge(transcript);
        challenges.push(r_i);
        claim = poly.eval_from_hint(&claim, &r_i);
    }
    debug_assert!(suffix_iter.next().is_none());

    check_sumcheck_output_claim(claim, verifier, &challenges)?;
    Ok(challenges)
}

/// Enforce the final sumcheck oracle equality for the provided challenge point.
///
/// This is useful when some prefix rounds are reconstructed outside the generic
/// verifier driver and the caller needs to check the final oracle value against
/// the full concatenated challenge vector.
///
/// # Errors
///
/// Returns any error produced by `verifier.expected_output_claim`, or
/// [`HachiError::InvalidProof`] if the final claim does not match the oracle
/// evaluation at `challenges`.
pub fn check_sumcheck_output_claim<E, V>(
    final_claim: E,
    verifier: &V,
    challenges: &[E],
) -> Result<(), HachiError>
where
    E: FieldCore,
    V: SumcheckInstanceVerifier<E>,
{
    let expected = verifier.expected_output_claim(challenges)?;
    if final_claim != expected {
        tracing::error!(
            rounds = verifier.num_rounds(),
            degree_bound = verifier.degree_bound(),
            diff_is_zero = (final_claim - expected).is_zero(),
            "verify_sumcheck MISMATCH"
        );
        return Err(HachiError::InvalidProof);
    }
    Ok(())
}

/// Produce a sumcheck proof for a single instance, driving the Fiat-Shamir transcript.
///
/// This method:
/// - does **not** absorb the initial claim into the transcript (callers should do so),
/// - appends each round message under `labels::ABSORB_SUMCHECK_ROUND`,
/// - samples one challenge per round via `sample_challenge`,
/// - updates the running claim using the per-round hint (`g(0)+g(1)`).
///
/// It returns the proof, the derived point `r`, and the final claimed value at `r`.
///
/// # Errors
///
/// Returns an error if any per-round polynomial exceeds the instance's degree bound.
pub fn prove_sumcheck<F, T, E, S, Inst>(
    instance: &mut Inst,
    transcript: &mut T,
    sample_challenge: S,
) -> Result<(SumcheckProof<E>, Vec<E>, E), HachiError>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
    E: FieldCore,
    S: FnMut(&mut T) -> E,
    Inst: SumcheckInstanceProver<E>,
{
    prove_sumcheck_with_omitted_prefix_rounds::<F, T, E, S, Inst, _>(
        instance,
        transcript,
        sample_challenge,
        0,
        |_, _, _| Ok(()),
    )
}

/// Verify a single-instance sumcheck proof.
///
/// This function:
/// - absorbs the initial claim into the transcript,
/// - delegates round-by-round verification to [`SumcheckProof::verify`],
/// - performs the final oracle check: `final_claim == verifier.expected_output_claim(r)`.
///
/// Returns the challenge point `r` on success.
///
/// # Errors
///
/// Returns [`HachiError::InvalidProof`] if the final sumcheck claim does not
/// match the oracle evaluation, or propagates any error from the per-round
/// verification (e.g. degree-bound violation, round-count mismatch).
pub fn verify_sumcheck<F, T, E, S, V>(
    proof: &SumcheckProof<E>,
    verifier: &V,
    transcript: &mut T,
    sample_challenge: S,
) -> Result<Vec<E>, HachiError>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
    E: FieldCore,
    S: FnMut(&mut T) -> E,
    V: SumcheckInstanceVerifier<E>,
{
    verify_sumcheck_with_prefix_rounds::<F, T, E, S, V, _, _>(
        proof,
        verifier,
        transcript,
        sample_challenge,
        0,
        |_, _| Ok(()),
        |_, _, _| unreachable!("no prefix rounds requested"),
    )
}
