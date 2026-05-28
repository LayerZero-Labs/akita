use super::*;

/// Verifier state for the degree-two extension-opening reduction sumcheck.
#[derive(Debug, Clone)]
pub struct ExtensionOpeningReductionVerifier<E: FieldCore> {
    witness_evals: Vec<E>,
    factor_evals: Vec<E>,
    input_claim: E,
    num_rounds: usize,
}

impl<E: FieldCore> ExtensionOpeningReductionVerifier<E> {
    /// Construct a verifier from transformed-witness and transparent-factor
    /// Boolean-hypercube evaluation tables.
    ///
    /// # Errors
    ///
    /// Returns an error if the tables do not have the same nonzero power-of-two
    /// length.
    pub fn new(witness_evals: Vec<E>, factor_evals: Vec<E>) -> Result<Self, AkitaError> {
        let input_claim = extension_opening_reduction_claim(&witness_evals, &factor_evals)?;
        let num_rounds = num_rounds_from_table_len(witness_evals.len())?;
        Ok(Self {
            witness_evals,
            factor_evals,
            input_claim,
            num_rounds,
        })
    }
}

impl<E: FieldCore> SumcheckInstanceVerifier<E> for ExtensionOpeningReductionVerifier<E> {
    fn num_rounds(&self) -> usize {
        self.num_rounds
    }

    fn degree_bound(&self) -> usize {
        EXTENSION_OPENING_REDUCTION_DEGREE
    }

    fn input_claim(&self) -> E {
        self.input_claim
    }

    fn expected_output_claim(&self, challenges: &[E]) -> Result<E, AkitaError> {
        extension_opening_reduction_eval_at_point(
            &self.witness_evals,
            &self.factor_evals,
            challenges,
        )
    }
}

/// Transcript driver for an extension-opening reduction sumcheck.
///
/// Unlike [`ExtensionOpeningReductionVerifier`], this object only verifies the
/// round chain and returns the detached final claim. The caller must still check
/// that final claim against the separately opened transformed witness.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExtensionOpeningReductionSumcheck<E: FieldCore> {
    input_claim: E,
    num_rounds: usize,
}

impl<E: FieldCore> ExtensionOpeningReductionSumcheck<E> {
    /// Construct a detached extension-opening reduction sumcheck driver.
    #[must_use]
    pub fn new(input_claim: E, num_rounds: usize) -> Self {
        Self {
            input_claim,
            num_rounds,
        }
    }

    /// Initial transcript-visible claim.
    #[must_use]
    pub fn input_claim(&self) -> E {
        self.input_claim
    }

    /// Number of sumcheck rounds.
    #[must_use]
    pub fn num_rounds(&self) -> usize {
        self.num_rounds
    }

    /// Degree bound for each round polynomial.
    #[must_use]
    pub fn degree_bound(&self) -> usize {
        EXTENSION_OPENING_REDUCTION_DEGREE
    }

    /// Prove an extension-opening reduction sumcheck.
    ///
    /// # Errors
    ///
    /// Returns an error if the prover instance shape does not match this driver
    /// or any produced round polynomial exceeds the fixed degree bound.
    pub fn prove<F, T, S>(
        &self,
        prover: &mut ExtensionOpeningReductionProver<E>,
        transcript: &mut T,
        sample_challenge: S,
    ) -> Result<(SumcheckProof<E>, ExtensionOpeningReductionRoundResult<E>), AkitaError>
    where
        F: FieldCore + CanonicalField,
        T: Transcript<F>,
        E: AkitaSerialize,
        S: FnMut(&mut T) -> E,
    {
        self.check_prover_shape(prover)?;
        let (proof, challenges, final_claim) =
            SumcheckInstanceProverExt::prove::<F, T, S>(prover, transcript, sample_challenge)?;
        Ok((
            proof,
            ExtensionOpeningReductionRoundResult {
                final_claim,
                challenges,
            },
        ))
    }

    /// Replay extension-opening reduction sumcheck rounds without doing the
    /// final witness-opening check.
    ///
    /// # Errors
    ///
    /// Returns an error if the proof shape is inconsistent or a round polynomial
    /// exceeds the fixed degree bound.
    pub fn verify<F, T, S>(
        &self,
        proof: &SumcheckProof<E>,
        transcript: &mut T,
        mut sample_challenge: S,
    ) -> Result<ExtensionOpeningReductionRoundResult<E>, AkitaError>
    where
        F: FieldCore + CanonicalField,
        T: Transcript<F>,
        E: AkitaSerialize,
        S: FnMut(&mut T) -> E,
    {
        if proof.round_polys.len() != self.num_rounds {
            return Err(AkitaError::InvalidSize {
                expected: self.num_rounds,
                actual: proof.round_polys.len(),
            });
        }

        transcript.append_serde(labels::ABSORB_SUMCHECK_CLAIM, &self.input_claim);
        let mut claim = self.input_claim;
        let mut challenges = Vec::with_capacity(self.num_rounds);
        for poly in &proof.round_polys {
            if poly.degree() > self.degree_bound() {
                return Err(AkitaError::InvalidInput(format!(
                    "extension-opening reduction round poly exceeds degree bound {}",
                    self.degree_bound()
                )));
            }
            transcript.append_serde(labels::ABSORB_SUMCHECK_ROUND, poly);
            let r_i = sample_challenge(transcript);
            challenges.push(r_i);
            claim = poly.eval_from_hint(&claim, &r_i);
        }

        Ok(ExtensionOpeningReductionRoundResult {
            final_claim: claim,
            challenges,
        })
    }

    fn check_prover_shape(
        &self,
        prover: &ExtensionOpeningReductionProver<E>,
    ) -> Result<(), AkitaError> {
        if prover.num_rounds() != self.num_rounds {
            return Err(AkitaError::InvalidSize {
                expected: self.num_rounds,
                actual: prover.num_rounds(),
            });
        }
        Ok(())
    }
}

#[cfg(feature = "zk")]
impl<E: FieldCore> ExtensionOpeningReductionSumcheck<E> {
    /// Prove an extension-opening reduction sumcheck with ZK round masks.
    ///
    /// # Errors
    ///
    /// Returns an error if the prover/pad shape is invalid or any produced round
    /// polynomial exceeds the fixed degree bound.
    pub fn prove_zk<F, T, S>(
        &self,
        prover: &mut ExtensionOpeningReductionProver<E>,
        transcript: &mut T,
        sample_challenge: S,
        pre_sampled_pads: Vec<CompressedUniPoly<E>>,
    ) -> Result<
        (
            SumcheckProofMasked<E>,
            ExtensionOpeningReductionRoundResult<E>,
        ),
        AkitaError,
    >
    where
        F: FieldCore + CanonicalField,
        T: Transcript<F>,
        E: AkitaSerialize,
        S: FnMut(&mut T) -> E,
    {
        self.check_prover_shape(prover)?;
        let (proof, challenges) = ZkSumcheckInstanceProverExt::prove_zk::<F, T, S>(
            prover,
            self.input_claim,
            transcript,
            sample_challenge,
            pre_sampled_pads,
        )?;
        let (final_witness, final_factor) =
            prover.final_witness_and_factor_evals().ok_or_else(|| {
                AkitaError::InvalidInput(
                    "extension-opening reduction has not reached a final point".to_string(),
                )
            })?;
        Ok((
            proof,
            ExtensionOpeningReductionRoundResult {
                final_claim: final_witness * final_factor,
                challenges,
            },
        ))
    }

    /// Replay masked extension-opening reduction sumcheck rounds and return the
    /// unmasked final claim as a deferred R1CS linear combination.
    ///
    /// # Errors
    ///
    /// Returns an error if the proof shape is inconsistent or any round
    /// polynomial exceeds the fixed degree bound.
    #[allow(clippy::too_many_arguments)]
    pub fn verify_zk<F, T, S>(
        &self,
        proof: &SumcheckProofMasked<E>,
        input_claim_mask: ZkR1csLinearCombination<E>,
        transcript: &mut T,
        mut sample_challenge: S,
        relations: &mut ZkRelationAccumulator<E>,
        hiding_cursor: &mut usize,
    ) -> Result<(ZkR1csLinearCombination<E>, Vec<E>), AkitaError>
    where
        F: FieldCore + CanonicalField,
        T: Transcript<F>,
        E: AkitaSerialize + ExtField<F>,
        S: FnMut(&mut T) -> E,
    {
        if proof.masked_round_polys.len() != self.num_rounds {
            return Err(AkitaError::InvalidSize {
                expected: self.num_rounds,
                actual: proof.masked_round_polys.len(),
            });
        }

        transcript.append_serde(labels::ABSORB_SUMCHECK_CLAIM, &self.input_claim);
        let mut masked_claim = self.input_claim;
        let mut claim_mask = input_claim_mask;
        let mut challenges = Vec::with_capacity(self.num_rounds);
        for masked_poly in &proof.masked_round_polys {
            if masked_poly.degree() > self.degree_bound() {
                return Err(AkitaError::InvalidInput(format!(
                    "extension-opening reduction round poly exceeds degree bound {}",
                    self.degree_bound()
                )));
            }
            transcript.append_serde(labels::ABSORB_SUMCHECK_ROUND, masked_poly);
            let r_i = sample_challenge(transcript);
            challenges.push(r_i);
            let next_claim_mask = zk_masked_compressed_round_claim_mask::<F, E>(
                &claim_mask,
                &masked_poly.coeffs_except_linear_term,
                r_i,
                hiding_cursor,
            );
            masked_claim = masked_poly.eval_from_hint(&masked_claim, &r_i);
            claim_mask = next_claim_mask;
        }

        Ok((
            relations.push_masked_claim_relation(
                "extension-opening reduction final claim",
                masked_claim,
                &claim_mask,
            ),
            challenges,
        ))
    }
}
