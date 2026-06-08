//! Eq-factored sumcheck transcript drivers.

use crate::traits::{
    EqFactoredSumcheckInstanceProver, EqFactoredSumcheckInstanceVerifier,
    EqFactoredSumcheckRoundState,
};
#[cfg(feature = "zk")]
use crate::types::EqFactoredSumcheckProofMasked;
use crate::types::{EqFactoredSumcheckProof, EqFactoredUniPoly};
use akita_field::AkitaError;
#[cfg(feature = "zk")]
use akita_field::ExtField;
use akita_field::{CanonicalField, FieldCore};
#[cfg(feature = "zk")]
use akita_r1cs::{ZkR1csLinearCombination, ZkRelationAccumulator};
use akita_serialization::AkitaSerialize;
use akita_transcript::labels;
use akita_transcript::Transcript;

#[cfg(feature = "zk")]
/// Prover output for an eq-factored sumcheck with plain-opening round masks.
///
/// The final element is the accumulated mask on the verifier's final scaled
/// claim, following the same omitted-linear-term recurrence used by verifier
/// relation generation.
pub type EqFactoredMaskedProveOutput<E> = (EqFactoredSumcheckProofMasked<E>, Vec<E>, E);

#[cfg(feature = "zk")]
/// Per-instance ZK final-relation emitter for eq-factored sumchecks.
pub trait ZkEqFactoredFinalRelation<E: FieldCore>: EqFactoredSumcheckInstanceVerifier<E> {
    /// Return the mask inherited by this sumcheck's input claim.
    fn initial_claim_mask(&self) -> ZkR1csLinearCombination<E> {
        ZkR1csLinearCombination::zero()
    }

    /// Record the instance-specific final check as deferred relations.
    ///
    /// # Errors
    ///
    /// Returns an error if the instance cannot evaluate the relation data at
    /// the sampled challenge point.
    fn record_final_relation(
        &self,
        round_state: &Self::RoundState,
        challenges: &[E],
        scaled_claim: ZkR1csLinearCombination<E>,
        claim_scale: E,
        handoff_mask: ZkR1csLinearCombination<E>,
        relations: &mut ZkRelationAccumulator<E>,
    ) -> Result<(), AkitaError>;
}

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

#[cfg(feature = "zk")]
fn eq_factored_claim_transition_coeffs<E: FieldCore>(
    claim_scale: E,
    l_at_0: E,
    l_at_1: E,
    r_round: E,
    stored_coeff_count: usize,
) -> (E, Vec<E>) {
    let current_scalar = l_at_0 + l_at_1;
    let l_at_r = l_at_0 + (l_at_1 - l_at_0) * r_round;
    let previous_coeff = l_at_r * r_round;
    let mut coeffs = Vec::with_capacity(stored_coeff_count);
    if stored_coeff_count == 0 {
        return (previous_coeff, coeffs);
    }

    coeffs.push(claim_scale * l_at_r * (l_at_1 - r_round * current_scalar));
    let higher_coeff_base = claim_scale * l_at_1 * l_at_r;
    let mut r_power = r_round * r_round;
    for _ in 1..stored_coeff_count {
        coeffs.push(higher_coeff_base * (r_power - r_round));
        r_power *= r_round;
    }
    (previous_coeff, coeffs)
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
        sample_challenge: S,
    ) -> Result<(EqFactoredSumcheckProof<E>, Vec<E>, E), AkitaError>
    where
        F: FieldCore + CanonicalField,
        T: Transcript<F>,
        E: AkitaSerialize,
        S: FnMut(&mut T) -> E,
    {
        crate::sink::prove_clear_eq_factored(self, transcript, sample_challenge)
    }
}

impl<E, Inst> EqFactoredSumcheckInstanceProverExt<E> for Inst
where
    E: FieldCore,
    Inst: EqFactoredSumcheckInstanceProver<E>,
{
}

#[cfg(feature = "zk")]
fn mask_eq_factored_poly<E>(
    poly: &EqFactoredUniPoly<E>,
    pad_poly: &EqFactoredUniPoly<E>,
    degree_bound: usize,
) -> Result<EqFactoredUniPoly<E>, AkitaError>
where
    E: FieldCore,
{
    let stored_coeffs = EqFactoredUniPoly::<E>::stored_coeff_count_for_degree(degree_bound);
    if pad_poly.coeffs_except_linear_term.len() != stored_coeffs {
        return Err(AkitaError::InvalidProof);
    }
    let mut masked_coeffs = Vec::with_capacity(stored_coeffs);
    for idx in 0..stored_coeffs {
        let true_coeff = poly
            .coeffs_except_linear_term
            .get(idx)
            .copied()
            .unwrap_or_else(E::zero);
        let pad = pad_poly.coeffs_except_linear_term[idx];
        masked_coeffs.push(true_coeff + pad);
    }
    Ok(EqFactoredUniPoly {
        coeffs_except_linear_term: masked_coeffs,
    })
}

#[cfg(feature = "zk")]
fn advance_eq_factored_claim_mask<E: FieldCore>(
    previous_mask: E,
    claim_scale: E,
    l_at_0: E,
    l_at_1: E,
    pad_poly: &EqFactoredUniPoly<E>,
    r_round: E,
) -> E {
    let (previous_coeff, transition_coeffs) = eq_factored_claim_transition_coeffs(
        claim_scale,
        l_at_0,
        l_at_1,
        r_round,
        pad_poly.coeffs_except_linear_term.len(),
    );
    transition_coeffs
        .iter()
        .zip(pad_poly.coeffs_except_linear_term.iter())
        .fold(previous_coeff * previous_mask, |acc, (&weight, &pad)| {
            acc + weight * pad
        })
}

#[cfg(feature = "zk")]
/// ZK extension for eq-factored sumcheck provers.
///
/// This mirrors the ordinary high-level sumcheck driver, but the transcript and
/// returned proof payload carry masked round messages only.
pub trait ZkEqFactoredSumcheckInstanceProverExt<E>: EqFactoredSumcheckInstanceProver<E>
where
    E: FieldCore,
{
    /// Prove with precommitted pad polynomials from the plain-opening hiding
    /// witness.
    ///
    /// # Errors
    ///
    /// Returns an error if pad shape is invalid or a round exceeds the degree
    /// bound.
    #[tracing::instrument(skip_all, name = "prove_zk_eq_factored_sumcheck")]
    #[inline(never)]
    fn prove_zk<F, T, S>(
        &mut self,
        transcript: &mut T,
        sample_challenge: S,
        pre_sampled_pads: Vec<EqFactoredUniPoly<E>>,
    ) -> Result<EqFactoredMaskedProveOutput<E>, AkitaError>
    where
        F: FieldCore + CanonicalField,
        T: Transcript<F>,
        E: AkitaSerialize,
        S: FnMut(&mut T) -> E,
    {
        self.prove_zk_with_public_claim::<F, T, S>(
            self.input_claim(),
            transcript,
            sample_challenge,
            pre_sampled_pads,
        )
    }

    /// Prove with a transcript-visible masked input claim while keeping the
    /// instance's private input claim for omitted-linear-coefficient updates.
    ///
    /// # Errors
    ///
    /// Returns an error if pad shape is invalid or a round exceeds the degree
    /// bound.
    #[tracing::instrument(skip_all, name = "prove_zk_eq_factored_sumcheck_public_claim")]
    #[inline(never)]
    fn prove_zk_with_public_claim<F, T, S>(
        &mut self,
        public_input_claim: E,
        transcript: &mut T,
        mut sample_challenge: S,
        pre_sampled_pads: Vec<EqFactoredUniPoly<E>>,
    ) -> Result<EqFactoredMaskedProveOutput<E>, AkitaError>
    where
        F: FieldCore + CanonicalField,
        T: Transcript<F>,
        E: AkitaSerialize,
        S: FnMut(&mut T) -> E,
    {
        let num_rounds = self.num_rounds();
        if pre_sampled_pads.len() != num_rounds {
            return Err(AkitaError::InvalidSize {
                expected: num_rounds,
                actual: pre_sampled_pads.len(),
            });
        }
        let degree_bound = self.degree_bound();
        let input_claim = self.input_claim();
        let mut scaled_claim = input_claim;
        let mut claim_mask = public_input_claim - input_claim;
        let mut claim_scale = E::one();
        let mut masked_round_polys = Vec::with_capacity(num_rounds);
        let mut challenges = Vec::with_capacity(num_rounds);

        transcript.append_serde(labels::ABSORB_SUMCHECK_CLAIM, &public_input_claim);

        for (round, pad_poly) in pre_sampled_pads.into_iter().enumerate() {
            let poly = self.compute_round_eq_factored(round);
            if poly.degree() > degree_bound {
                return Err(AkitaError::InvalidInput(format!(
                    "eq-factored sumcheck round poly degree {} exceeds bound {}",
                    poly.degree(),
                    degree_bound
                )));
            }
            // Eq-factored messages store q_i(X) without its linear coefficient:
            // [q_0, q_2, q_3, ...]. The ZK proof sends the masked stored part
            // q~_j = q_j + rho_j. The omitted q_1 is still determined by the
            // incoming true claim, so the prover advances its private
            // `scaled_claim` with the unmasked `poly`.
            let masked_poly = mask_eq_factored_poly(&poly, &pad_poly, degree_bound)?;

            transcript.append_serde(labels::ABSORB_SUMCHECK_ROUND, &masked_poly);
            let r_i = sample_challenge(transcript);
            let (l_at_0, l_at_1) = self.current_linear_factor_evals();
            claim_mask = advance_eq_factored_claim_mask(
                claim_mask,
                claim_scale,
                l_at_0,
                l_at_1,
                &pad_poly,
                r_i,
            );
            (scaled_claim, claim_scale) =
                advance_eq_factored_claim(scaled_claim, claim_scale, l_at_0, l_at_1, &poly, r_i);
            challenges.push(r_i);
            self.ingest_challenge(round, r_i);
            masked_round_polys.push(masked_poly);
        }

        self.finalize();
        Ok((
            EqFactoredSumcheckProofMasked { masked_round_polys },
            challenges,
            claim_mask,
        ))
    }
}

#[cfg(feature = "zk")]
impl<E, Inst> ZkEqFactoredSumcheckInstanceProverExt<E> for Inst
where
    E: FieldCore,
    Inst: EqFactoredSumcheckInstanceProver<E>,
{
}

#[cfg(feature = "zk")]
/// ZK extension for eq-factored sumcheck verifiers.
pub trait ZkEqFactoredSumcheckInstanceVerifierExt<E>:
    EqFactoredSumcheckInstanceVerifier<E> + ZkEqFactoredFinalRelation<E>
where
    E: FieldCore,
{
    /// Verify masked round messages and record deferred round residuals.
    ///
    /// # Errors
    ///
    /// Returns an error if the masked round count is invalid or a round exceeds
    /// the degree bound.
    #[tracing::instrument(skip_all, name = "verify_zk_eq_factored_sumcheck")]
    #[inline(never)]
    fn verify_zk<F, T, S>(
        &self,
        masks: &EqFactoredSumcheckProofMasked<E>,
        transcript: &mut T,
        relations: &mut ZkRelationAccumulator<E>,
        hiding_cursor: &mut usize,
        mut sample_challenge: S,
    ) -> Result<(Vec<E>, ZkR1csLinearCombination<E>), AkitaError>
    where
        F: FieldCore + CanonicalField,
        T: Transcript<F>,
        E: AkitaSerialize + ExtField<F>,
        S: FnMut(&mut T) -> E,
    {
        let num_rounds = self.num_rounds();
        if masks.masked_round_polys.len() != num_rounds {
            return Err(AkitaError::InvalidSize {
                expected: num_rounds,
                actual: masks.masked_round_polys.len(),
            });
        }

        let degree_bound = self.degree_bound();
        let scaled_claim = self.input_claim();
        let mut scaled_claim_handle = scaled_claim;
        let mut scaled_claim_mask = self.initial_claim_mask();
        let mut masked_scaled_claim = scaled_claim;
        let mut masked_claim_scale = E::one();
        let mut challenges = Vec::with_capacity(num_rounds);
        let mut round_state = self.start_round_state()?;

        transcript.append_serde(labels::ABSORB_SUMCHECK_CLAIM, &scaled_claim);

        for round in 0..num_rounds {
            let masked_poly = &masks.masked_round_polys[round];
            if masked_poly.degree() > degree_bound {
                return Err(AkitaError::InvalidInput(format!(
                    "eq-factored sumcheck round poly exceeds degree bound {degree_bound}"
                )));
            }

            transcript.append_serde(labels::ABSORB_SUMCHECK_ROUND, masked_poly);
            let r_i = sample_challenge(transcript);
            let (l_at_0, l_at_1) = round_state.current_linear_factor_evals();
            // The eq-factored verifier never receives q_1. Instead the claim
            // transition is linear in the previous claim and the stored
            // coefficients:
            //
            //   C_i = a_prev * C_{i-1} + sum_j a_j * q_j,
            //
            // where j ranges over stored coefficients [0, 2, 3, ...]. The same
            // coefficients apply to masks, giving
            //
            //   eta_i = a_prev * eta_{i-1} + sum_j a_j * rho_j.
            //
            // `coeffs_except_linear` are the a_j values; `previous_coeff` is
            // a_prev.
            let (previous_coeff, coeffs_except_linear) = eq_factored_claim_transition_coeffs(
                masked_claim_scale,
                l_at_0,
                l_at_1,
                r_i,
                masked_poly.coeffs_except_linear_term.len(),
            );
            // Advance the public masked claim using q~. This transition is
            // verifier-local once q~ is on the transcript; the deferred
            // relation only needs to carry the matching hidden mask transition
            // for the final unmasked claim relation.
            (masked_scaled_claim, masked_claim_scale) = advance_eq_factored_claim(
                masked_scaled_claim,
                masked_claim_scale,
                l_at_0,
                l_at_1,
                masked_poly,
                r_i,
            );
            scaled_claim_handle = masked_scaled_claim;
            // `next_claim_mask` is eta_i for the scaled running claim. This
            // recurrence includes the omitted-linear-term contribution induced
            // by the previous masked claim, so final handoffs must use the
            // accumulated mask rather than only the final round's stored terms.
            let next_claim_mask = ZkRelationAccumulator::<E>::masked_eq_factored_claim_mask::<F>(
                &scaled_claim_mask,
                previous_coeff,
                &coeffs_except_linear,
                hiding_cursor,
            );
            scaled_claim_mask = next_claim_mask;
            challenges.push(r_i);
            round_state.ingest_challenge(round, r_i);
        }

        let final_claim_lc = relations.push_masked_claim_relation(
            "eq-factored sumcheck final claim",
            scaled_claim_handle,
            &scaled_claim_mask,
        );
        self.record_final_relation(
            &round_state,
            &challenges,
            final_claim_lc,
            masked_claim_scale,
            scaled_claim_mask.clone(),
            relations,
        )?;
        Ok((challenges, scaled_claim_mask))
    }
}

#[cfg(feature = "zk")]
impl<E, Inst> ZkEqFactoredSumcheckInstanceVerifierExt<E> for Inst
where
    E: FieldCore,
    Inst: EqFactoredSumcheckInstanceVerifier<E> + ZkEqFactoredFinalRelation<E>,
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
