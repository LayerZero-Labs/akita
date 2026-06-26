//! Standard sumcheck transcript drivers.

use crate::traits::{SumcheckInstanceProver, SumcheckInstanceVerifier};
use crate::types::SumcheckProof;
#[cfg(feature = "zk")]
use crate::types::SumcheckProofMasked;
#[cfg(feature = "zk")]
use akita_algebra::uni_poly::CompressedUniPoly;
use akita_field::AkitaError;
#[cfg(feature = "zk")]
use akita_field::ExtField;
use akita_field::{CanonicalField, FieldCore};
use akita_serialization::AkitaSerialize;
use akita_transcript::labels;
use akita_transcript::Transcript;

#[cfg(feature = "zk")]
/// Prover output for a standard sumcheck with plain-opening round masks.
pub type MaskedProveOutput<E> = (SumcheckProofMasked<E>, Vec<E>);

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

#[cfg(feature = "zk")]
fn pad_compressed_coeffs_to_degree<E: FieldCore>(
    coeffs_except_linear: &[E],
    degree_bound: usize,
) -> Vec<E> {
    let mut out = coeffs_except_linear.to_vec();
    out.resize(degree_bound, E::zero());
    out
}

#[cfg(feature = "zk")]
fn mask_compressed_poly<E>(
    poly: &CompressedUniPoly<E>,
    pad_poly: CompressedUniPoly<E>,
    degree_bound: usize,
) -> Result<CompressedUniPoly<E>, AkitaError>
where
    E: FieldCore,
{
    let true_coeffs =
        pad_compressed_coeffs_to_degree(&poly.coeffs_except_linear_term, degree_bound);
    if pad_poly.coeffs_except_linear_term.len() != true_coeffs.len() {
        return Err(AkitaError::InvalidProof);
    }
    let mut masked_coeffs = Vec::with_capacity(true_coeffs.len());
    for (idx, true_coeff) in true_coeffs.into_iter().enumerate() {
        let pad = pad_poly.coeffs_except_linear_term[idx];
        masked_coeffs.push(true_coeff + pad);
    }
    Ok(CompressedUniPoly {
        coeffs_except_linear_term: masked_coeffs,
    })
}

#[cfg(feature = "zk")]
/// ZK extension for standard sumcheck provers.
pub trait ZkSumcheckInstanceProverExt<E>: SumcheckInstanceProver<E>
where
    E: FieldCore,
{
    /// Prove with a transcript-visible masked input claim while retaining the
    /// instance's private true input claim for round-polynomial construction.
    ///
    /// # Errors
    ///
    /// Returns an error if pad shape is invalid or a round exceeds the degree
    /// bound.
    #[tracing::instrument(skip_all, name = "prove_zk_sumcheck")]
    #[inline(never)]
    fn prove_zk<F, T, S>(
        &mut self,
        public_input_claim: E,
        transcript: &mut T,
        mut sample_challenge: S,
        pre_sampled_pads: Vec<CompressedUniPoly<E>>,
    ) -> Result<MaskedProveOutput<E>, AkitaError>
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
        let mut claim = self.input_claim();
        transcript.append_serde(labels::ABSORB_SUMCHECK_CLAIM, &public_input_claim);

        let degree_bound = self.degree_bound();
        let mut masked_round_polys = Vec::with_capacity(num_rounds);
        let mut r = Vec::with_capacity(num_rounds);

        for (round, pad_poly) in pre_sampled_pads.into_iter().enumerate() {
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
            let compressed = g.compress();
            if compressed.degree() > degree_bound {
                return Err(AkitaError::InvalidInput(format!(
                    "sumcheck round poly degree {} exceeds bound {}",
                    compressed.degree(),
                    degree_bound
                )));
            }
            let masked_poly = mask_compressed_poly(&compressed, pad_poly, degree_bound)?;
            transcript.append_serde(labels::ABSORB_SUMCHECK_ROUND, &masked_poly);
            let r_i = sample_challenge(transcript);
            r.push(r_i);

            claim = compressed.eval_from_hint(&claim, &r_i);
            {
                let _s = tracing::info_span!("sumcheck_round_fold").entered();
                self.ingest_challenge(round, r_i);
            }
            masked_round_polys.push(masked_poly);
        }

        self.finalize();
        Ok((SumcheckProofMasked { masked_round_polys }, r))
    }
}

#[cfg(feature = "zk")]
impl<E, Inst> ZkSumcheckInstanceProverExt<E> for Inst
where
    E: FieldCore,
    Inst: SumcheckInstanceProver<E>,
{
}
