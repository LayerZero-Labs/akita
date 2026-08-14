//! Standard sumcheck transcript drivers.

use crate::traits::{SumcheckInstanceProver, SumcheckInstanceVerifier};
use crate::types::{SharedChallengeSumcheckProof, SumcheckProof};
use akita_field::AkitaError;
use akita_field::{CanonicalField, FieldCore};
use akita_serialization::AkitaSerialize;
use akita_transcript::labels;
use akita_transcript::Transcript;

/// Plain extension for standard sumcheck provers.
pub trait SumcheckInstanceProverExt<E>: SumcheckInstanceProver<E> + Sized
where
    E: FieldCore,
{
    /// Produce a sumcheck proof for a single instance.
    ///
    /// It returns the proof, the derived point `r`, and the final claimed value
    /// at `r`.
    ///
    /// # Errors
    ///
    /// Returns an error if any per-round polynomial exceeds the instance's degree bound.
    #[tracing::instrument(skip_all, name = "prove_sumcheck")]
    #[inline(never)]
    fn prove<F, T, S>(
        &mut self,
        transcript: &mut T,
        mut sample_challenge: S,
    ) -> Result<(SumcheckProof<E>, Vec<E>, E), AkitaError>
    where
        F: FieldCore + CanonicalField,
        T: Transcript<F>,
        E: AkitaSerialize,
        S: FnMut(&mut T) -> E,
    {
        let num_rounds = self.num_rounds();
        let mut claim = self.input_claim();
        tracing::debug!(
            is_zero = claim.is_zero(),
            num_rounds,
            "prove_sumcheck input_claim"
        );
        transcript.append_serde(labels::ABSORB_SUMCHECK_CLAIM, &claim);

        let degree_bound = self.degree_bound();
        let mut round_polys = Vec::with_capacity(num_rounds);
        let mut r = Vec::with_capacity(num_rounds);

        for round in 0..num_rounds {
            let _round_span = tracing::info_span!(
                "sumcheck_round",
                round,
                table_len = 1usize << (num_rounds - round)
            )
            .entered();
            let g = {
                let _s = tracing::info_span!("sumcheck_round_univariate").entered();
                self.compute_round_univariate(round, claim)
            };
            let round_sum = g.evaluate(&E::zero()) + g.evaluate(&E::one());
            debug_assert!(
                round_sum == claim,
                "sumcheck round {round} univariate does not match previous claim hint"
            );

            let compressed = g.compress();
            if compressed.degree() > degree_bound {
                return Err(AkitaError::InvalidInput(format!(
                    "sumcheck round poly degree {} exceeds bound {}",
                    compressed.degree(),
                    degree_bound
                )));
            }

            transcript.append_serde(labels::ABSORB_SUMCHECK_ROUND, &compressed);
            let r_i = sample_challenge(transcript);
            r.push(r_i);

            claim = compressed.eval_from_hint(&claim, &r_i);
            {
                let _s = tracing::info_span!("sumcheck_round_fold").entered();
                self.ingest_challenge(round, r_i);
            }
            round_polys.push(compressed);
        }

        self.finalize();
        Ok((SumcheckProof { round_polys }, r, claim))
    }
}

impl<E, Inst> SumcheckInstanceProverExt<E> for Inst
where
    E: FieldCore,
    Inst: SumcheckInstanceProver<E>,
{
}

/// Output of a shared-challenge vector sumcheck prover.
pub struct SharedChallengeSumcheckProverOutput<E: FieldCore> {
    /// Round-major proof messages.
    pub proof: SharedChallengeSumcheckProof<E>,
    /// Challenge point shared by every claim.
    pub challenges: Vec<E>,
    /// One final running claim per input claim.
    pub final_claims: Vec<E>,
}

/// Prove several sumcheck instances with one challenge per round.
///
/// Every claim remains independent. The prover emits and absorbs all claim
/// polynomials for a round before sampling the challenge shared by that round.
pub fn prove_shared_challenge_sumcheck<F, T, E, P, S>(
    provers: &mut [P],
    transcript: &mut T,
    mut sample_challenge: S,
) -> Result<SharedChallengeSumcheckProverOutput<E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
    E: FieldCore + AkitaSerialize,
    P: SumcheckInstanceProver<E>,
    S: FnMut(&mut T) -> E,
{
    let first = provers.first().ok_or_else(|| {
        AkitaError::InvalidInput(
            "shared-challenge sumcheck requires at least one claim".to_string(),
        )
    })?;
    let num_rounds = first.num_rounds();
    let degree_bound = first.degree_bound();
    if provers
        .iter()
        .any(|prover| prover.num_rounds() != num_rounds || prover.degree_bound() != degree_bound)
    {
        return Err(AkitaError::InvalidInput(
            "shared-challenge sumcheck instances must have equal shapes".to_string(),
        ));
    }

    let mut claims = provers
        .iter()
        .map(SumcheckInstanceProver::input_claim)
        .collect::<Vec<_>>();
    for claim in &claims {
        transcript.append_serde(labels::ABSORB_SUMCHECK_CLAIM, claim);
    }
    let mut rounds = Vec::with_capacity(num_rounds);
    let mut challenges = Vec::with_capacity(num_rounds);
    for round_index in 0..num_rounds {
        let mut round = Vec::with_capacity(provers.len());
        for (prover, &claim) in provers.iter_mut().zip(&claims) {
            let poly = prover
                .compute_round_univariate(round_index, claim)
                .compress();
            if poly.degree() > degree_bound {
                return Err(AkitaError::InvalidInput(format!(
                    "sumcheck round poly degree {} exceeds bound {}",
                    poly.degree(),
                    degree_bound
                )));
            }
            transcript.append_serde(labels::ABSORB_SUMCHECK_ROUND, &poly);
            round.push(poly);
        }
        let challenge = sample_challenge(transcript);
        challenges.push(challenge);
        for ((prover, claim), poly) in provers.iter_mut().zip(&mut claims).zip(&round) {
            *claim = poly.eval_from_hint(claim, &challenge);
            prover.ingest_challenge(round_index, challenge);
        }
        rounds.push(round);
    }
    for prover in provers {
        prover.finalize();
    }
    Ok(SharedChallengeSumcheckProverOutput {
        proof: SharedChallengeSumcheckProof {
            round_polys: rounds,
        },
        challenges,
        final_claims: claims,
    })
}

/// Plain extension for standard sumcheck verifiers.
pub trait SumcheckInstanceVerifierExt<E>: SumcheckInstanceVerifier<E> + Sized
where
    E: FieldCore,
{
    /// Verify a single-instance sumcheck proof.
    ///
    /// Returns the challenge point `r` on success.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidProof`] if the final sumcheck claim does not
    /// match the oracle evaluation, or propagates any error from the per-round
    /// verification.
    #[tracing::instrument(skip_all, name = "verify_sumcheck")]
    #[inline(never)]
    fn verify<F, T, S>(
        &self,
        proof: &SumcheckProof<E>,
        transcript: &mut T,
        mut sample_challenge: S,
    ) -> Result<Vec<E>, AkitaError>
    where
        F: FieldCore + CanonicalField,
        T: Transcript<F>,
        E: AkitaSerialize,
        S: FnMut(&mut T) -> E,
    {
        let num_rounds = self.num_rounds();
        if proof.round_polys.len() != num_rounds {
            return Err(AkitaError::InvalidSize {
                expected: num_rounds,
                actual: proof.round_polys.len(),
            });
        }

        let mut claim = self.input_claim();
        tracing::debug!(
            is_zero = claim.is_zero(),
            num_rounds,
            "verify_sumcheck input_claim"
        );
        transcript.append_serde(labels::ABSORB_SUMCHECK_CLAIM, &claim);

        let degree_bound = self.degree_bound();
        let mut challenges = Vec::with_capacity(num_rounds);

        for poly in &proof.round_polys {
            if poly.degree() > degree_bound {
                return Err(AkitaError::InvalidInput(format!(
                    "sumcheck round poly degree {} exceeds bound {}",
                    poly.degree(),
                    degree_bound
                )));
            }

            transcript.append_serde(labels::ABSORB_SUMCHECK_ROUND, poly);
            let r_i = sample_challenge(transcript);
            challenges.push(r_i);
            claim = poly.eval_from_hint(&claim, &r_i);
        }

        check_sumcheck_output_claim(claim, self, &challenges)?;
        Ok(challenges)
    }
}

impl<E, Inst> SumcheckInstanceVerifierExt<E> for Inst
where
    E: FieldCore,
    Inst: SumcheckInstanceVerifier<E>,
{
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
/// [`AkitaError::InvalidProof`] if the final claim does not match the oracle
/// evaluation at `challenges`.
pub fn check_sumcheck_output_claim<E, V>(
    final_claim: E,
    verifier: &V,
    challenges: &[E],
) -> Result<(), AkitaError>
where
    E: FieldCore + AkitaSerialize,
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
        return Err(AkitaError::InvalidProof);
    }
    Ok(())
}
