//! Bind reusable commitments to a validated public proof request.
use super::CpuCommitmentMaterialHandle;
use crate::opaque::lifecycle::CpuProofSessionHandle;
use crate::opaque::CpuBackend;
use akita_config::{CommitmentConfig, TrustedScheduleCatalog};
use akita_error::AkitaError;
use akita_prover::backend::{ProofAdmission, ProofContext};
use akita_serialization::{AkitaSerialize, Valid};
use akita_types::{
    AkitaSetupDescriptor, Commitment, FoldSchedule, GroupCommitPhaseParams, OpeningClaimsLayout,
};
use jolt_field::{CanonicalEncoding, Field};

impl<F: Field, E> akita_prover::backend::ProofScopeConsumer for CpuBackend<F, E> {
    type ProofSessionHandle = CpuProofSessionHandle;

    fn finish_scope(&self, session: &Self::ProofSessionHandle) -> Result<(), AkitaError> {
        let proof = session.validate_owner(self.owner())?;
        self.owner().finish_scope(proof)
    }
    fn abort_scope_best_effort(&self, session: &Self::ProofSessionHandle) {
        if session.belongs_to(self.owner()) {
            self.owner().abort_scope(&session.scope_lease());
        }
    }
}

impl<F: Field, E> CpuBackend<F, E> {
    pub(crate) fn admitted_group(
        &self,
        binding: &crate::opaque::OperationBinding,
    ) -> Result<(akita_types::GroupOpenPhaseParams, usize), AkitaError> {
        self.validate_binding(binding)?;
        let (schedule, _) = binding.scope_lease().proof_plan()?;
        let level = if binding.fold_level() == 0 {
            &schedule.root.params
        } else {
            &schedule
                .recursive_folds
                .get(binding.fold_level() as usize - 1)
                .ok_or(AkitaError::InvalidProof)?
                .params
        };
        let index = binding.group_index().ok_or(AkitaError::InvalidProof)?;
        Ok((
            *level.groups().get(index).ok_or(AkitaError::InvalidProof)?,
            level.witness_chunk.num_chunks,
        ))
    }
}

impl<F, E> ProofAdmission<F, E> for CpuBackend<F, E>
where
    F: Field + CanonicalEncoding + AkitaSerialize + Send + Sync + 'static,
    E: Field + jolt_field::ExtField<F> + Send + Sync + 'static,
{
    fn begin_proof<Cfg>(
        &self,
        setup: &AkitaSetupDescriptor,
        schedules: &TrustedScheduleCatalog<Cfg>,
        plan: &FoldSchedule,
        layout: &OpeningClaimsLayout,
    ) -> Result<Self::ProofSessionHandle, AkitaError>
    where
        Cfg: CommitmentConfig<Field = F, ExtField = E>,
    {
        let prepared = self.prepared()?;
        setup
            .check()
            .map_err(|error| AkitaError::InvalidSetup(error.to_string()))?;
        if setup != &prepared.expanded.descriptor {
            return Err(AkitaError::InvalidSetup(
                "proof setup differs from backend setup".into(),
            ));
        }
        Self::validate_proof_configuration(schedules, plan, layout)?;
        plan.validate_structure()?;
        plan.validate_nonterminal_opening_execution(E::DEGREE)?;
        plan.root.params.validate_opening_batch(layout)?;
        let required = akita_types::setup_matrix_field_elements_for_schedule(plan)?;
        if required > prepared.expanded.shared_matrix.as_field_slice().len() {
            return Err(AkitaError::InvalidSetup(
                "proof plan exceeds backend setup capacity".into(),
            ));
        }
        for parameters in std::iter::once(&plan.root.params)
            .chain(plan.recursive_folds.iter().map(|fold| &fold.params))
        {
            parameters.witness_chunk.validate()?;
            for group in parameters.groups() {
                akita_types::validate_role_dims_for_field::<F>(
                    group.role_dims(parameters.open().matrix.ring_dimension()),
                )?;
            }
        }
        akita_types::dispatch_for_field!(
            akita_types::ProtocolDispatchSlot::Role(akita_types::RingRole::Inner),
            F,
            plan.terminal.d_a(),
            |D| {
                let _ = D;
                Ok::<(), AkitaError>(())
            }
        )?;
        let proof = self.owner().begin_proof(plan, layout)?;
        Ok(CpuProofSessionHandle::new(
            std::sync::Arc::clone(self.owner()),
            proof,
        ))
    }

    fn proof_context(
        &self,
        session: &Self::ProofSessionHandle,
        level: u32,
    ) -> Result<ProofContext, AkitaError> {
        session.validate_owner(self.owner())?;
        let context = ProofContext::new(
            self.owner_id(),
            self.owner().setup_digest(),
            session.scope_id(),
            level,
        );
        self.validate_context(session, &context)?;
        Ok(context)
    }

    fn validate_commitment(
        &self,
        session: &Self::ProofSessionHandle,
        context: &ProofContext,
        handle: &Self::CommitmentHandle,
        parameters: &GroupCommitPhaseParams,
        commitment: &Commitment<F>,
    ) -> Result<Self::CommitmentMaterialHandle, AkitaError> {
        let binding = self.binding(session, context)?;
        if handle.owner != self.owner_id()
            || handle.committed.parameters != *parameters
            || handle.committed.public.0.coeffs() != commitment.0.coeffs()
            || (commitment.0.ring_dim() != 0
                && commitment.0.ring_dim() != handle.committed.public.0.ring_dim())
        {
            return Err(AkitaError::InvalidInput(
                "commitment does not match its retained source, owner, or statement".into(),
            ));
        }
        let group_index = context.group_index().ok_or_else(|| {
            AkitaError::InvalidInput("commitment admission requires an ordered group".into())
        })?;
        let (schedule, root_layout) = binding.scope_lease().proof_plan()?;
        let (level, layout) = if context.fold_level() == 0 {
            (&schedule.root.params, root_layout)
        } else {
            let step = schedule
                .recursive_folds
                .get(context.fold_level() as usize - 1)
                .ok_or_else(|| {
                    AkitaError::InvalidInput(
                        "terminal state does not admit reusable commitments".into(),
                    )
                })?;
            let layout = OpeningClaimsLayout::from_groups(
                step.params
                    .groups()
                    .iter()
                    .map(|group| group.profile.group)
                    .collect(),
            )?;
            (&step.params, layout)
        };
        let expected = level.group_params(&layout, group_index)?;
        if expected.profile != *parameters {
            return Err(AkitaError::InvalidInput(
                "commitment parameters do not match the admitted ordered group".into(),
            ));
        }
        let plan = crate::commitment::CommitmentExecutionPlan::for_relation_group(
            level,
            &layout,
            group_index,
            context.fold_level() as usize,
        )?;
        let mut material = CpuCommitmentMaterialHandle::from_state(
            handle.committed.retained.clone(),
            &plan,
            parameters.group.num_polynomials(),
        )?;
        material.bind(binding.clone());
        material.bind_commitment(
            handle.committed.commitment_id,
            Some(handle.committed.public.0.clone()),
        );
        binding
            .scope_lease()
            .admit_commitment(context, handle.committed.commitment_id)?;
        Ok(material)
    }
}
