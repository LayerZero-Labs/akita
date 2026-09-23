//! Aggregate EOR arithmetic and validation, including all private group tuples.
use super::openings::RetainedOpeningSource;
use super::source::PreparedExtensionOpeningGroup;
use super::OperationBinding;
use crate::opaque::*;
use akita_algebra::uni_poly::UniPoly;
use akita_error::AkitaError;
use akita_serialization::AkitaSerialize;
use akita_types::*;
use jolt_field::{CanonicalEncoding, ExtField, Field, Fold, MulBaseUnreduced, Ring, Unreduced};

pub(crate) trait ExtensionOpeningSession<E: Field>: Send {
    fn num_rounds(&self) -> usize;
    fn num_terms(&self) -> usize;
    fn round_polynomial(
        &mut self,
        round: usize,
        previous_claim: E,
    ) -> Result<UniPoly<E>, AkitaError>;
    fn bind_challenge(&mut self, round: usize, challenge: E) -> Result<(), AkitaError>;
    fn finish(self: Box<Self>) -> Result<Vec<(E, E, E)>, AkitaError>;
}
struct PreparedGroup<F, E, Cfg>
where
    F: Field + CanonicalEncoding,
    E: Field,
    Cfg: akita_config::CommitmentConfig<Field = F, ExtField = E>,
{
    source: RetainedOpeningSource<F, E, Cfg>,
    point: Vec<E>,
    ring_dimension: usize,
    rows: Vec<Vec<E>>,
    witness_opening: Option<crate::opaque::CpuWitnessOpeningHandle<E>>,
}
pub struct CpuEorPreparation<F, E, Cfg>
where
    F: Field + CanonicalEncoding,
    E: Field,
    Cfg: akita_config::CommitmentConfig<Field = F, ExtField = E>,
{
    binding: OperationBinding,
    layout: OpeningClaimsLayout,
    groups: Vec<PreparedGroup<F, E, Cfg>>,
}
pub struct CpuEorSession<E: Field> {
    binding: OperationBinding,
    lease: crate::opaque::ScopeLease,
    groups: Vec<Box<dyn ExtensionOpeningSession<E>>>,
    coefficients: Vec<E>,
    /// Per-group tail points retained for the final transparent-factor
    /// cross-check.
    ///
    /// The factor is evaluated once in `finish_eor` from these points and the
    /// accumulated challenges, in `O(rounds)`. Materializing the `2^rounds`
    /// Lagrange table here and folding it every round would duplicate the
    /// compact table each group already owns, and would defeat
    /// `ExtensionOpeningReductionGroup::extend_cylindrically`, which exists
    /// precisely to avoid expanding a group over its high variables.
    expected_tails: Vec<Vec<E>>,
    eta: Vec<E>,
    num_rounds: usize,
    round: usize,
    claim: E,
    pending: Option<UniPoly<E>>,
    group_claims: Vec<E>,
    group_pending: Vec<UniPoly<E>>,
    challenges: Vec<E>,
}
impl<F, E, Cfg> OpaqueEorKernel<F, E> for CpuBackend<Cfg>
where
    Cfg: akita_config::CommitmentConfig<Field = F, ExtField = E>,
    F: Field
        + CanonicalEncoding
        + AkitaSerialize
        + jolt_field::PseudoMersenne
        + Ring
        + Unreduced
        + 'static,
    F::Wide: From<F> + jolt_field::AdditiveGroup,
    E: ExtField<F>
        + FpExtEncoding<F>
        + MulBaseUnreduced<F>
        + Unreduced
        + Fold
        + Ring
        + AkitaSerialize
        + 'static,
{
    fn prepare_eor(
        &self,
        session: &Self::ProofSessionHandle,
        context: &ProofContext,
        layout: &OpeningClaimsLayout,
        groups: &[EorGroupRequest<'_, E, Self::CommitmentHandle, Self::WitnessHandle>],
    ) -> Result<PreparedEor<E, Self::EorPreparationHandle>, AkitaError> {
        let binding = self.binding(session, context)?;
        match binding.scope_lease().proof_plan() {
            Ok((schedule, root_layout)) => {
                let expected = if context.fold_level() == 0 {
                    root_layout
                } else if context.fold_level() as usize == schedule.recursive_folds.len() + 1 {
                    let num_vars =
                        akita_error::checked::ceil_log2(schedule.terminal.input_witness_len)
                            .ok_or(AkitaError::InvalidProof)?;
                    OpeningClaimsLayout::new(num_vars, 1)?
                } else {
                    let step = schedule
                        .recursive_folds
                        .get(context.fold_level() as usize - 1)
                        .ok_or(AkitaError::InvalidProof)?;
                    OpeningClaimsLayout::from_groups(
                        step.params
                            .groups()
                            .iter()
                            .map(|group| group.profile.group)
                            .collect(),
                    )?
                };
                if *layout != expected {
                    return Err(AkitaError::InvalidProof);
                }
            }
            #[cfg(test)]
            Err(_) => {}
            #[cfg(not(test))]
            Err(error) => return Err(error),
        }
        if groups.len() != layout.num_groups() {
            return Err(AkitaError::InvalidProof);
        }
        let mut retained = Vec::with_capacity(groups.len());
        let mut openings = Vec::new();
        let mut proof_partials = Vec::new();
        let (split_bits, width) = tensor_opening_split::<F, E>()?;
        for (index, group) in groups.iter().enumerate() {
            let expected = layout.group_layout(index)?;
            if group.point.len() != expected.num_vars() || group.point.len() < split_bits {
                return Err(AkitaError::InvalidProof);
            }
            let (source, prepared, witness_opening) = match &group.source {
                OpeningSource::Commitment(handle) => {
                    binding.scope_lease().validate_commitment(
                        &context.for_group(index),
                        handle.committed.commitment_id,
                    )?;
                    if handle.owner != self.owner().backend_id()
                        || handle.committed.metadata.num_polynomials() != expected.num_polynomials()
                        || handle.committed.metadata.num_vars() != expected.num_vars()
                        || handle.committed.parameters.inner.matrix.ring_dimension()
                            != group.ring_dimension
                    {
                        return Err(AkitaError::InvalidProof);
                    }
                    (
                        RetainedOpeningSource::Commitment(handle.committed.clone()),
                        handle.committed.source.extension(
                            self,
                            group.ring_dimension,
                            group.point,
                        )?,
                        None,
                    )
                }
                OpeningSource::Witness(witness) => {
                    self.validate_binding(&witness.operation_binding())?;
                    binding.validate_lineage(&witness.operation_binding())?;
                    witness
                        .operation_binding()
                        .validate_group(index, layout.num_groups())?;
                    if expected.num_polynomials() != 1
                        || witness.manifest().num_vars()? != expected.num_vars()
                        || witness.manifest().commitment_ring_dimension() != group.ring_dimension
                    {
                        return Err(AkitaError::InvalidProof);
                    }
                    let plan = ValidatedWitnessOpeningPlan::new(
                        group.point,
                        group.ring_dimension,
                        witness.manifest().logical_len(),
                    );
                    let prepared=crate::opaque::consumer_kernels::CpuWitnessOpeningKernel::<F,E>::prepare_witness_opening(self,Some(self.prepared()?),witness,&plan)?;
                    let (openings, proof_partials, handle) = prepared.into_parts();
                    let rows = proof_partials
                        .chunks_exact(width)
                        .map(tensor_row_partials_from_columns::<F, E>)
                        .collect::<Result<Vec<_>, _>>()?;
                    (
                        RetainedOpeningSource::Witness(Box::new(witness.snapshot())),
                        PreparedExtensionOpeningGroup {
                            openings,
                            proof_partials,
                            row_partials_by_claim: rows,
                        },
                        Some(handle),
                    )
                }
            };
            if prepared.openings.len() != expected.num_polynomials()
                || prepared.row_partials_by_claim.len() != expected.num_polynomials()
            {
                return Err(AkitaError::InvalidProof);
            }
            openings.extend(prepared.openings);
            proof_partials.extend(prepared.proof_partials);
            retained.push(PreparedGroup {
                source,
                point: group.point.to_vec(),
                ring_dimension: group.ring_dimension,
                rows: prepared.row_partials_by_claim,
                witness_opening,
            });
        }
        Ok(PreparedEor {
            openings,
            proof_partials,
            handle: CpuEorPreparation {
                binding,
                layout: layout.clone(),
                groups: retained,
            },
        })
    }
    fn begin_eor(
        &self,
        preparation: Self::EorPreparationHandle,
        eta: &[E],
        coefficients: &[E],
    ) -> Result<(E, Self::EorSessionHandle), AkitaError> {
        self.validate_binding(&preparation.binding)?;
        let (split_bits, _) = tensor_opening_split::<F, E>()?;
        if eta.len() != split_bits
            || coefficients.len() != preparation.layout.num_total_polynomials()
        {
            return Err(AkitaError::InvalidProof);
        }
        let rounds = preparation
            .layout
            .max_num_vars()
            .checked_sub(split_bits)
            .ok_or(AkitaError::InvalidProof)?;
        let mut sessions = Vec::with_capacity(preparation.groups.len());
        let mut claims = Vec::new();
        let mut tails: Vec<Vec<E>> = Vec::with_capacity(preparation.groups.len());
        for (index, group) in preparation.groups.into_iter().enumerate() {
            let range = preparation.layout.root_group_claim_range(index)?;
            let weights = coefficients.get(range).ok_or(AkitaError::InvalidProof)?;
            let claim =
                group
                    .rows
                    .iter()
                    .zip(weights)
                    .try_fold(E::zero(), |sum, (row, weight)| {
                        Ok::<_, AkitaError>(
                            sum + *weight * tensor_reduction_claim_from_rows::<F, E>(row, eta)?,
                        )
                    })?;
            let tail = group
                .point
                .get(split_bits..)
                .ok_or(AkitaError::InvalidProof)?;
            let extra = rounds
                .checked_sub(tail.len())
                .ok_or(AkitaError::InvalidProof)?;
            tails.push(tail.to_vec());
            let session = match group.source {
                RetainedOpeningSource::Commitment(source) => source.source.begin_eor(
                    self,
                    group.ring_dimension,
                    weights,
                    tail,
                    eta,
                    vec![E::zero(); extra],
                    claim,
                )?,
                RetainedOpeningSource::Witness(witness) => {
                    let plan = ValidatedWitnessEorPlan::new(
                        weights,
                        tail,
                        eta,
                        vec![E::zero(); extra],
                        claim,
                        group.ring_dimension,
                    );
                    let handle=crate::opaque::consumer_kernels::CpuWitnessOpeningKernel::<F,E>::begin_witness_eor(self,Some(self.prepared()?),&witness,group.witness_opening.ok_or(AkitaError::InvalidProof)?,&plan)?;
                    Box::new(handle) as Box<dyn ExtensionOpeningSession<E>>
                }
            };
            if session.num_rounds() != rounds || session.num_terms() != weights.len() {
                return Err(AkitaError::InvalidProof);
            }
            sessions.push(session);
            claims.push(claim);
        }
        let claim = claims.iter().copied().fold(E::zero(), |sum, c| sum + c);
        let lease = preparation.binding.scope_lease().clone();
        Ok((
            claim,
            CpuEorSession {
                binding: preparation.binding,
                lease,
                groups: sessions,
                coefficients: coefficients.to_vec(),
                expected_tails: tails,
                eta: eta.to_vec(),
                num_rounds: rounds,
                round: 0,
                claim,
                pending: None,
                group_claims: claims,
                group_pending: Vec::new(),
                challenges: Vec::with_capacity(rounds),
            },
        ))
    }
    fn eor_round(
        &self,
        session: &mut Self::EorSessionHandle,
        round: usize,
        claim: E,
    ) -> Result<UniPoly<E>, AkitaError> {
        self.validate_leased_binding(&session.binding, &session.lease)?;
        if round != session.round
            || round >= session.num_rounds
            || claim != session.claim
            || session.pending.is_some()
        {
            return Err(AkitaError::InvalidProof);
        }
        let mut coefficients = vec![E::zero(); EXTENSION_OPENING_REDUCTION_DEGREE + 1];
        for (group, claim) in session.groups.iter_mut().zip(&session.group_claims) {
            let polynomial = group.round_polynomial(round, *claim)?;
            if polynomial.coeffs.len() > coefficients.len() {
                return Err(AkitaError::InvalidProof);
            }
            for (sum, value) in coefficients.iter_mut().zip(&polynomial.coeffs) {
                *sum += *value;
            }
            session.group_pending.push(polynomial);
        }
        let polynomial = UniPoly::from_coeffs(coefficients);
        if polynomial.evaluate(&E::zero()) + polynomial.evaluate(&E::one()) != claim {
            return Err(AkitaError::InvalidProof);
        }
        session.pending = Some(polynomial.clone());
        Ok(polynomial)
    }
    fn bind_eor_round(
        &self,
        session: &mut Self::EorSessionHandle,
        round: usize,
        challenge: E,
    ) -> Result<(), AkitaError> {
        self.validate_leased_binding(&session.binding, &session.lease)?;
        if round != session.round || round >= session.num_rounds {
            return Err(AkitaError::InvalidProof);
        }
        let polynomial = session.pending.take().ok_or(AkitaError::InvalidProof)?;
        // Each group advances its own claim from its already pending polynomial.
        for ((group, claim), polynomial) in session
            .groups
            .iter_mut()
            .zip(&mut session.group_claims)
            .zip(session.group_pending.drain(..))
        {
            *claim = polynomial.evaluate(&challenge);
            group.bind_challenge(round, challenge)?;
        }
        session.claim = polynomial.evaluate(&challenge);
        session.challenges.push(challenge);
        session.round += 1;
        Ok(())
    }
    fn finish_eor(&self, session: Self::EorSessionHandle) -> Result<Vec<E>, AkitaError> {
        self.validate_leased_binding(&session.binding, &session.lease)?;
        if session.round != session.num_rounds || session.pending.is_some() {
            return Err(AkitaError::InvalidProof);
        }
        let CpuEorSession {
            groups,
            coefficients,
            expected_tails,
            eta,
            challenges,
            claim,
            ..
        } = session;
        let mut values = Vec::with_capacity(coefficients.len());
        for (group, tail) in groups.into_iter().zip(&expected_tails) {
            // Same value the verifier derives: eq(tail, local) projected
            // through eta, then one `(1 - r)` per cylindrical high variable.
            let local = challenges
                .get(..tail.len())
                .ok_or(AkitaError::InvalidProof)?;
            let mut factor = tensor_equality_factor_eval_at_point::<F, E>(tail, &eta, local)?;
            for extra in challenges
                .get(tail.len()..)
                .ok_or(AkitaError::InvalidProof)?
            {
                factor *= E::one() - *extra;
            }
            for (coefficient, witness, actual_factor) in group.finish()? {
                let index = values.len();
                if actual_factor != factor || coefficients.get(index) != Some(&coefficient) {
                    return Err(AkitaError::InvalidProof);
                }
                values.push(witness * actual_factor);
            }
        }
        if values.len() != coefficients.len()
            || values
                .iter()
                .zip(&coefficients)
                .fold(E::zero(), |sum, (value, weight)| sum + *value * *weight)
                != claim
        {
            return Err(AkitaError::InvalidProof);
        }
        Ok(values)
    }
}
