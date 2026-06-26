//! Eq-factored sumcheck transcript drivers.

use crate::traits::{
    EqFactoredSumcheckInstanceProver, EqFactoredSumcheckInstanceVerifier,
    EqFactoredSumcheckRoundState,
};
use crate::types::{EqFactoredSumcheckProof, EqFactoredUniPoly};
use akita_field::AkitaError;
use akita_field::{CanonicalField, FieldCore};
use akita_serialization::AkitaSerialize;
use akita_transcript::labels;
use akita_transcript::Transcript;

/// Advance the scaled claim state for one eq-factored sumcheck round.
#[doc(hidden)]
#[inline]
pub fn advance_eq_factored_claim<E: FieldCore>(
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

/// Plain extension for eq-factored sumcheck provers.
pub trait EqFactoredSumcheckInstanceProverExt<E>:
    EqFactoredSumcheckInstanceProver<E> + Sized
where
    E: FieldCore,
{
    /// Produce an eq-factored sumcheck proof.
    ///
    /// The prover sends the inner polynomial `q(X)` with its linear coefficient
    /// omitted in every round, while the driver maintains the verifier-equivalent
    /// scaled claim update.
    ///
    /// # Errors
    ///
    /// Returns an error if any generated round polynomial exceeds the instance's
    /// degree bound.
    #[tracing::instrument(skip_all, name = "prove_eq_factored_sumcheck")]
    #[inline(never)]
    fn prove<F, T, S>(
        &mut self,
        transcript: &mut T,
        mut sample_challenge: S,
    ) -> Result<(EqFactoredSumcheckProof<E>, Vec<E>, E), AkitaError>
    where
        F: FieldCore + CanonicalField,
        T: Transcript<F>,
        E: AkitaSerialize,
        S: FnMut(&mut T) -> E,
    {
        let num_rounds = self.num_rounds();
        let degree_bound = self.degree_bound();
        let mut scaled_claim = self.input_claim();
        let mut claim_scale = E::one();
        let mut round_polys = Vec::with_capacity(num_rounds);
        let mut challenges = Vec::with_capacity(num_rounds);

        transcript.append_serde(labels::ABSORB_SUMCHECK_CLAIM, &scaled_claim);

        for round in 0..num_rounds {
            let poly = self.compute_round_eq_factored(round);
            if poly.degree() > degree_bound {
                return Err(AkitaError::InvalidInput(format!(
                    "eq-factored sumcheck round poly degree {} exceeds bound {}",
                    poly.degree(),
                    degree_bound
                )));
            }

            transcript.append_serde(labels::ABSORB_SUMCHECK_ROUND, &poly);
            let r_i = sample_challenge(transcript);
            let (l_at_0, l_at_1) = self.current_linear_factor_evals();
            (scaled_claim, claim_scale) =
                advance_eq_factored_claim(scaled_claim, claim_scale, l_at_0, l_at_1, &poly, r_i);
            challenges.push(r_i);
            self.ingest_challenge(round, r_i);
            round_polys.push(poly);
        }

        self.finalize();
        Ok((
            EqFactoredSumcheckProof { round_polys },
            challenges,
            scaled_claim,
        ))
    }
}

impl<E, Inst> EqFactoredSumcheckInstanceProverExt<E> for Inst
where
    E: FieldCore,
    Inst: EqFactoredSumcheckInstanceProver<E>,
{
}

/// Plain extension for eq-factored sumcheck verifiers.
pub trait EqFactoredSumcheckInstanceVerifierExt<E>:
    EqFactoredSumcheckInstanceVerifier<E> + Sized
where
    E: FieldCore,
{
    /// Verify an eq-factored sumcheck proof.
    ///
    /// The verifier absorbs each round message, samples the corresponding
    /// challenge, updates the scaled running claim from the current eq-factor
    /// evaluations and the transmitted `q(X)` data, and finally checks the
    /// expected folded oracle value at the full challenge point.
    ///
    /// This creates and owns the mutable eq-factored round state locally, while
    /// keeping `self` immutable.
    ///
    /// # Errors
    ///
    /// Returns an error if the proof length is invalid, a round polynomial exceeds
    /// the verifier degree bound, or the final folded oracle value does not match.
    #[tracing::instrument(skip_all, name = "verify_eq_factored_sumcheck")]
    #[inline(never)]
    fn verify<F, T, S>(
        &self,
        proof: &EqFactoredSumcheckProof<E>,
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

        let degree_bound = self.degree_bound();
        let mut scaled_claim = self.input_claim();
        let mut claim_scale = E::one();
        let mut challenges = Vec::with_capacity(num_rounds);
        let mut round_state = self.start_round_state()?;

        transcript.append_serde(labels::ABSORB_SUMCHECK_CLAIM, &scaled_claim);

        for (round, poly) in proof.round_polys.iter().enumerate() {
            if poly.degree() > degree_bound {
                return Err(AkitaError::InvalidInput(format!(
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

        let expected = self.expected_output_claim(&round_state, &challenges)?;
        if scaled_claim != claim_scale * expected {
            return Err(AkitaError::InvalidProof);
        }
        Ok(challenges)
    }
}

impl<E, Inst> EqFactoredSumcheckInstanceVerifierExt<E> for Inst
where
    E: FieldCore,
    Inst: EqFactoredSumcheckInstanceVerifier<E>,
{
}
