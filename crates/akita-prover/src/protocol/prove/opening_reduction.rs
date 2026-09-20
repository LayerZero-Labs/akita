use super::*;
pub(crate) struct ProvedExtensionOpeningReduction<E: Field> {
    pub(crate) reduction: ExtensionOpeningReduction<E>,
    pub(crate) protocol_points: Vec<Vec<E>>,
}

/// Drive one aggregate EOR session; individual contributions stay inside the backend.
#[allow(clippy::too_many_arguments)]
pub(crate) fn prove_extension_opening_reduction<F, E, T, B>(
    backend: &B,
    context: &crate::backend::ProofContext,
    opening_batch: &OpeningClaimsLayout,
    group_inputs: &[crate::backend::EorGroupRequest<
        '_,
        E,
        B::CommitmentHandle,
        B::WitnessHandle,
    >],
    transcript: &mut T,
    level: u32,
    expected_openings: &[E],
) -> Result<ProvedExtensionOpeningReduction<E>, AkitaError>
where
    F: Field + CanonicalEncoding + Ring + Unreduced + AkitaSerialize + 'static,
    <F as Unreduced>::Wide: From<F>,
    E: ExtField<F> + Unreduced + Fold + MulBaseUnreduced<F> + AkitaSerialize + 'static,
    T: akita_types::ProverTranscriptGrinding<F>,
    B: crate::backend::OpaqueEorKernel<F, E>,
{
    let (split_bits, width) = tensor_opening_split::<F, E>()?;
    let max_tail_vars = opening_batch
        .max_num_vars()
        .checked_sub(split_bits)
        .ok_or(AkitaError::InvalidProof)?;
    let prepared = backend.prepare_eor(context, opening_batch, group_inputs)?;
    let num_claims = opening_batch.num_total_polynomials();
    if prepared.openings != expected_openings
        || prepared.openings.len() != num_claims
        || prepared.proof_partials.len()
            != width
                .checked_mul(num_claims)
                .ok_or(AkitaError::InvalidProof)?
    {
        return Err(AkitaError::InvalidProof);
    }
    append_claim_values_to_transcript::<F, E, T>(&prepared.openings, transcript);
    for partial in &prepared.proof_partials {
        append_ext_field::<F, E, T>(transcript, ABSORB_EVALUATION_CLAIMS, partial);
    }
    transcript.grind_query(akita_types::GrindingSite::ExtensionOpeningPoint { level })?;
    let eta = (0..split_bits)
        .map(|_| sample_ext_challenge::<F, E, T>(transcript, CHALLENGE_SUMCHECK_BATCH))
        .collect::<Vec<_>>();
    let claim_coefficients = akita_types::sample_row_coefficients::<F, E, T>(
        opening_batch,
        akita_types::GrindingSite::ExtensionOpeningClaimBatch { level },
        transcript,
    )?;
    let (input_claim, mut session) =
        backend.begin_eor(prepared.handle, &eta, &claim_coefficients)?;
    // Reconstruct the aggregate input from scheduled public partials.
    let expected = prepared
        .proof_partials
        .chunks_exact(width)
        .zip(&claim_coefficients)
        .try_fold(E::zero(), |sum, (columns, coefficient)| {
            let rows = tensor_row_partials_from_columns::<F, E>(columns)?;
            Ok::<_, AkitaError>(
                sum + *coefficient * tensor_reduction_claim_from_rows::<F, E>(&rows, &eta)?,
            )
        })?;
    if input_claim != expected {
        return Err(AkitaError::InvalidProof);
    }
    let mut kernel = EorSumcheck::<F, E, B> {
        backend,
        session: &mut session,
        claim: input_claim,
        rounds: max_tail_vars,
        field: std::marker::PhantomData,
    };
    let mut round = 0u32;
    let (sumcheck, rho, claim) =
        akita_sumcheck::prove_sumcheck::<F, T, E, _, _>(&mut kernel, transcript, |tr| {
            let challenge = akita_types::sample_grinded_sumcheck_challenge::<F, E, T>(
                tr,
                akita_types::SumcheckProtocol::ExtensionOpeningReduction,
                level,
                0,
                round,
            )?;
            round = round.checked_add(1).ok_or(AkitaError::InvalidProof)?;
            Ok(challenge)
        })?;
    let final_claims = backend.finish_eor(session)?;
    if final_claims.len() != num_claims
        || final_claims
            .iter()
            .zip(&claim_coefficients)
            .fold(E::zero(), |sum, (value, weight)| sum + *value * *weight)
            != claim
    {
        return Err(AkitaError::InvalidProof);
    }
    let mut final_factors = Vec::with_capacity(group_inputs.len());
    let mut protocol_points = Vec::with_capacity(group_inputs.len());
    for input in group_inputs {
        let tail = input
            .point
            .get(split_bits..)
            .ok_or(AkitaError::InvalidProof)?;
        let local = rho.get(..tail.len()).ok_or(AkitaError::InvalidProof)?;
        let mut factor = tensor_equality_factor_eval_at_point::<F, E>(tail, &eta, local)?;
        for extra in rho.get(tail.len()..).ok_or(AkitaError::InvalidProof)? {
            factor *= E::one() - *extra;
        }
        let point = dispatch_for_field!(
            ProtocolDispatchSlot::Role(RingRole::Inner),
            F,
            input.ring_dimension,
            |D| ring_subfield_packed_extension_opening_point::<F, E, D>(local.len(), local)
        )?;
        final_factors.push(factor);
        protocol_points.push(point);
    }
    for value in &final_claims {
        append_ext_field::<F, E, T>(transcript, ABSORB_EOR_FINAL_CLAIM, value);
    }
    Ok(ProvedExtensionOpeningReduction {
        reduction: ExtensionOpeningReduction {
            proof: ExtensionOpeningReductionProof {
                partials: prepared.proof_partials,
                sumcheck,
                final_claims,
            },
            final_factors,
        },
        protocol_points,
    })
}

struct EorSumcheck<
    'a,
    F: Field + CanonicalEncoding,
    E: Field,
    B: crate::backend::OpaqueEorKernel<F, E>,
> {
    backend: &'a B,
    session: &'a mut B::EorSessionHandle,
    claim: E,
    rounds: usize,
    field: std::marker::PhantomData<F>,
}
impl<F: Field + CanonicalEncoding, E: Field, B: crate::backend::OpaqueEorKernel<F, E>>
    akita_sumcheck::SumcheckKernel<E> for EorSumcheck<'_, F, E, B>
{
    fn num_rounds(&self) -> usize {
        self.rounds
    }
    fn degree_bound(&self) -> usize {
        akita_types::EXTENSION_OPENING_REDUCTION_DEGREE
    }
    fn input_claim(&self) -> E {
        self.claim
    }
    fn round_polynomial(
        &mut self,
        round: usize,
        claim: E,
    ) -> Result<akita_algebra::uni_poly::UniPoly<E>, AkitaError> {
        self.backend.eor_round(self.session, round, claim)
    }
    fn bind_challenge(&mut self, round: usize, challenge: E) -> Result<(), AkitaError> {
        self.backend.bind_eor_round(self.session, round, challenge)
    }
    fn finish(&mut self) -> Result<(), AkitaError> {
        Ok(())
    }
}
