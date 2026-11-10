use super::{
    compile_commitment_request, AcceptedReach, BackendKindId, CommitmentExecutionOutput,
    CommitmentExecutionPlan, CommitmentExecutorBuilder, CommitmentRequestCapabilities,
    CommitmentSource, CommitmentStateBinding, CommitmentStatePolicy, CompiledCommitmentRequest,
    CompressionOperationCapabilities, FullCommitmentOutput, InnerCommitOutput, PolynomialType,
    PreparedCommitmentResources, PreparedCompression, PreparedFusedCommitment,
    PreparedInnerCommitment, PreparedOuterCommitment, ResidentStatePolicy,
    StageDimensionCapabilities, StageResources, UncompressedCommitmentOutput,
};
use super::{InnerRelationStateMaterial, PortableCommitmentHandle, PortableStatePolicy};
use crate::opaque::{
    ComputeBackendSetup, CpuBackend, CpuCompressionOperation, CpuInnerCommitOperation,
    CpuOuterCommitOperation, CpuPreparedSetup,
};
use akita_error::AkitaError;
use akita_params::GroupCommitPhaseParams;
use akita_types::{AkitaExpandedSetup, CommittedGroup};
use jolt_algebra::{CanonicalEncoding, Field, Unreduced, WithCommitAccumulator};
use std::sync::Arc;
use zeroize::Zeroizing;

use super::state_policy::CommitmentStateExporters;

#[cfg(test)]
struct CpuInnerContext;

/// Checked commitment-stage executor.
///
/// Registrations keep each operation with its capabilities and explicit
/// export edge. Stage outputs directly own their backend values, and the
/// selected state policy assembles those values without a state registry.
pub(super) enum PreparedInnerOuterRoute<'a, F>
where
    F: Field + CanonicalEncoding,
{
    InnerOnly {
        inner: PreparedInnerCommitment<'a, F>,
    },
    Split {
        inner: PreparedInnerCommitment<'a, F>,
        outer: PreparedOuterCommitment<'a, F>,
    },
    Fused {
        fused: PreparedFusedCommitment<'a, F>,
        terminal_inner: Option<PreparedInnerCommitment<'a, F>>,
    },
}

pub(crate) struct CommitmentExecutor<
    'a,
    F: Field + CanonicalEncoding,
    SP: CommitmentStatePolicy<F> = ResidentStatePolicy,
> {
    pub(super) setup: akita_types::AkitaSetupDescriptor,
    pub(super) route: PreparedInnerOuterRoute<'a, F>,
    pub(super) compression: Option<PreparedCompression<'a, F>>,
    pub(super) state_policy: SP,
    /// Source of B hiding randomness; sourceless unless the executor belongs
    /// to a backend.
    pub(super) hiding_rng: Arc<crate::hiding::HidingRng>,
}

impl<'a, F, SP> CommitmentExecutor<'a, F, SP>
where
    F: Field + CanonicalEncoding,
    SP: CommitmentStatePolicy<F>,
{
    pub(super) fn inner(&self) -> Option<&PreparedInnerCommitment<'a, F>> {
        match &self.route {
            PreparedInnerOuterRoute::InnerOnly { inner }
            | PreparedInnerOuterRoute::Split { inner, .. } => Some(inner),
            PreparedInnerOuterRoute::Fused { terminal_inner, .. } => terminal_inner.as_ref(),
        }
    }

    pub(super) fn outer(&self) -> Option<&PreparedOuterCommitment<'a, F>> {
        match &self.route {
            PreparedInnerOuterRoute::Split { outer, .. } => Some(outer),
            PreparedInnerOuterRoute::InnerOnly { .. } | PreparedInnerOuterRoute::Fused { .. } => {
                None
            }
        }
    }

    pub(super) fn fused(&self) -> Option<&PreparedFusedCommitment<'a, F>> {
        match &self.route {
            PreparedInnerOuterRoute::Fused { fused, .. } => Some(fused),
            PreparedInnerOuterRoute::InnerOnly { .. } | PreparedInnerOuterRoute::Split { .. } => {
                None
            }
        }
    }

    #[cfg(test)]
    fn inner_mut(&mut self) -> Option<&mut PreparedInnerCommitment<'a, F>> {
        match &mut self.route {
            PreparedInnerOuterRoute::InnerOnly { inner }
            | PreparedInnerOuterRoute::Split { inner, .. } => Some(inner),
            PreparedInnerOuterRoute::Fused { terminal_inner, .. } => terminal_inner.as_mut(),
        }
    }
}

impl<'a, F, SP> CommitmentExecutor<'a, F, SP>
where
    F: Field + CanonicalEncoding + Unreduced + WithCommitAccumulator + 'static,
    SP: CommitmentStatePolicy<F>,
{
    /// Build a CPU executor with an explicit standard-representation registry.
    pub(crate) fn cpu<E>(
        backend: &'a CpuBackend<F, E>,
        prepared: &'a CpuPreparedSetup<F>,
        expanded: &AkitaExpandedSetup<F>,
        standard_types: Vec<PolynomialType>,
        state_policy: SP,
    ) -> Result<Self, AkitaError> {
        if backend.prepared_expanded_setup(prepared).descriptor() != expanded.descriptor() {
            return Err(AkitaError::InvalidSetup(
                "CPU commitment executor setup descriptor mismatch".into(),
            ));
        }
        let mut builder = CommitmentExecutorBuilder::new(expanded, state_policy);
        let mut inner_capabilities = CommitmentRequestCapabilities::split::<CpuPreparedSetup<F>>(
            BackendKindId::of::<super::external::CpuBackendKind>("cpu")?,
            standard_types,
        );
        inner_capabilities.accept_any_standard_type();
        let inner_operation = Arc::new(CpuInnerCommitOperation::new(backend, prepared));
        let outer_operation = Arc::new(CpuOuterCommitOperation::new(
            backend,
            prepared,
            inner_operation.as_ref(),
        ));
        let compression_operation =
            Arc::new(CpuCompressionOperation::new(backend, prepared, expanded)?);
        let inner_owner = inner_operation.owner().clone();
        let outer_owner = inner_owner.clone();
        let backend_instance = builder.issue_backend_instance();
        let resources = StageResources::controlled(PreparedCommitmentResources::new(
            backend, prepared, expanded,
        )?);
        let inner_context =
            builder.operation_context(backend_instance, "cpu-inner", resources.clone())?;
        let outer_context =
            builder.operation_context(backend_instance, "cpu-outer", resources.clone())?;
        let compression_context =
            builder.operation_context(backend_instance, "cpu-compression", resources)?;
        builder.register_inner(PreparedInnerCommitment::new(
            inner_operation.clone(),
            inner_owner,
            inner_context,
            inner_capabilities,
            StageDimensionCapabilities::cpu_role::<F>(akita_params::RingRole::Inner),
            Some(inner_operation.portable_exporter()),
        )?)?;
        builder.register_outer(PreparedOuterCommitment::new(
            outer_operation,
            outer_owner,
            outer_context,
            StageDimensionCapabilities::cpu_role::<F>(akita_params::RingRole::Outer),
        ))?;
        builder.register_compression(PreparedCompression::new(
            compression_operation.clone(),
            compression_operation.owner().clone(),
            compression_context,
            CompressionOperationCapabilities::cpu::<F>(),
            Some(compression_operation.portable_exporter()),
        ))?;
        let mut executor = builder.build()?;
        executor.hiding_rng = backend.hiding_rng.clone();
        Ok(executor)
    }

    /// Draw B hiding randomness from `hiding_rng` instead of the backend's
    /// source.
    pub(crate) fn with_hiding_rng(mut self, hiding_rng: Arc<crate::hiding::HidingRng>) -> Self {
        self.hiding_rng = hiding_rng;
        self
    }
}

impl<'a, F, SP> CommitmentExecutor<'a, F, SP>
where
    F: Field + CanonicalEncoding,
    SP: CommitmentStatePolicy<F>,
{
    /// Validate that this executor and all of its prepared stages belong to
    /// the expanded setup used by the caller.
    pub(crate) fn validate_setup(
        &self,
        expanded: &AkitaExpandedSetup<F>,
    ) -> Result<(), AkitaError> {
        if self.setup != *expanded.descriptor() {
            return Err(AkitaError::InvalidSetup(
                "commitment executor setup descriptor mismatch".into(),
            ));
        }
        Ok(())
    }

    fn split_inner(&self) -> Result<&PreparedInnerCommitment<'a, F>, AkitaError> {
        self.inner().ok_or_else(|| {
            AkitaError::InvalidInput("commitment route has no split inner operation".into())
        })
    }

    fn split_outer(&self) -> Result<&PreparedOuterCommitment<'a, F>, AkitaError> {
        self.outer().ok_or_else(|| {
            AkitaError::InvalidInput("commitment route has no split outer operation".into())
        })
    }

    fn compression(&self) -> Result<&PreparedCompression<'a, F>, AkitaError> {
        self.compression.as_ref().ok_or_else(|| {
            AkitaError::InvalidInput("commitment route has no compression operation".into())
        })
    }

    fn prepare_request_resources(
        &self,
        plan: &CommitmentExecutionPlan,
        compiled: &CompiledCommitmentRequest<'_, F>,
    ) -> Result<(), AkitaError> {
        self.validate_plan_capabilities(plan)?;
        for requirement in self.request_ntt_requirements(plan, compiled)? {
            if let Some(fused) = self.fused_for_plan(plan) {
                fused.stage.ensure_ntt_slot(requirement)?;
            } else {
                match requirement.stage() {
                    super::CommitmentNttStage::Inner => {
                        self.split_inner()?.stage.ensure_ntt_slot(requirement)?
                    }
                    super::CommitmentNttStage::Outer => {
                        self.split_outer()?.stage.ensure_ntt_slot(requirement)?
                    }
                }
            }
        }
        Ok(())
    }

    fn fused_for_plan(
        &self,
        plan: &CommitmentExecutionPlan,
    ) -> Option<&PreparedFusedCommitment<'a, F>> {
        plan.uncompressed().is_some().then_some(())?;
        self.fused()
    }

    fn request_capabilities(
        &self,
        plan: &CommitmentExecutionPlan,
    ) -> Result<&CommitmentRequestCapabilities, AkitaError> {
        match self.fused_for_plan(plan) {
            Some(fused) => Ok(&fused.capabilities),
            None => Ok(&self.split_inner()?.capabilities),
        }
    }

    fn state_exporters(
        &self,
        plan: &CommitmentExecutionPlan,
    ) -> Result<CommitmentStateExporters<F>, AkitaError> {
        Ok(CommitmentStateExporters {
            inner: match self.fused_for_plan(plan) {
                Some(fused) => fused.exporter.clone(),
                None => self.split_inner()?.exporter.clone(),
            },
            compression: self
                .compression
                .as_ref()
                .and_then(|compression| compression.exporter.clone()),
        })
    }

    fn stage_resources(
        &self,
        plan: &CommitmentExecutionPlan,
        stage: super::CommitmentNttStage,
    ) -> Result<&StageResources<'a, F>, AkitaError> {
        if let Some(fused) = self.fused_for_plan(plan) {
            return Ok(&fused.stage.resources);
        }
        match stage {
            super::CommitmentNttStage::Inner => Ok(&self.split_inner()?.stage.resources),
            super::CommitmentNttStage::Outer => Ok(&self.split_outer()?.stage.resources),
        }
    }

    fn validate_plan_capabilities(&self, plan: &CommitmentExecutionPlan) -> Result<(), AkitaError> {
        self.validate_setup_capacity(plan)?;
        if let Some(fused) = self.fused_for_plan(plan) {
            let uncompressed = plan.uncompressed().ok_or_else(|| {
                AkitaError::InvalidInput("fused route requires an outer-stage plan".into())
            })?;
            if !fused.dimensions.supports(plan.inner().ring_dimension)
                || !fused
                    .dimensions
                    .supports(uncompressed.outer().ring_dimension())
            {
                return Err(AkitaError::InvalidInput(
                    "fused operation does not support the selected inner and outer dimensions"
                        .into(),
                ));
            }
        } else {
            if !self
                .split_inner()?
                .dimensions
                .supports(plan.inner().ring_dimension)
            {
                return Err(AkitaError::InvalidInput(
                    "inner operation does not support the selected ring dimension".into(),
                ));
            }
            if let Some(uncompressed) = plan.uncompressed() {
                if !self
                    .split_outer()?
                    .dimensions
                    .supports(uncompressed.outer().ring_dimension())
                {
                    return Err(AkitaError::InvalidInput(
                        "outer operation does not support the selected ring dimension".into(),
                    ));
                }
            }
        }
        if let (Some(compression), Some(relation_mode)) = (plan.compression(), plan.relation_mode())
        {
            if !self.compression()?.capabilities.supports(
                compression.maps().iter().map(|map| map.ring_dimension()),
                relation_mode,
            ) {
                return Err(AkitaError::InvalidInput(
                    "compression operation does not support the selected dimensions or relation mode"
                        .into(),
                ));
            }
        }
        Ok(())
    }

    fn validate_setup_capacity(&self, plan: &CommitmentExecutionPlan) -> Result<(), AkitaError> {
        let required = plan.max_setup_field_elements()?;
        if required > self.setup.num_field_elements {
            return Err(AkitaError::InvalidSetup(format!(
                "commitment execution requires {required} setup field elements, but setup has {}",
                self.setup.num_field_elements
            )));
        }
        Ok(())
    }

    fn request_ntt_requirements(
        &self,
        plan: &CommitmentExecutionPlan,
        compiled: &CompiledCommitmentRequest<'_, F>,
    ) -> Result<Vec<super::CommitmentNttRequirement>, AkitaError> {
        let mut requirements = Vec::new();
        let inner_resources = self.stage_resources(plan, super::CommitmentNttStage::Inner)?;
        if inner_resources.is_controlled() {
            for selected in compiled.selected_polynomial_types() {
                if let Some(requirement) = plan.inner_ntt_requirement(selected)? {
                    if !requirements.contains(&requirement) {
                        requirements.push(requirement);
                    }
                }
            }
        }
        if plan.uncompressed().is_some() {
            let outer_resources = self.stage_resources(plan, super::CommitmentNttStage::Outer)?;
            if outer_resources.is_controlled() {
                if let Some(requirement) = plan.outer_ntt_requirement()? {
                    requirements.push(requirement);
                }
            }
        }
        Ok(requirements)
    }

    /// Prewarm the retained NTT slots for a checked commitment request.
    ///
    /// Source discovery runs, but source representations and external
    /// operations are not materialized.
    #[cfg(test)]
    pub(crate) fn prewarm_request(
        &self,
        plan: &CommitmentExecutionPlan,
        sources: &[&dyn CommitmentSource<F>],
    ) -> Result<(), AkitaError> {
        let compiled =
            compile_commitment_request(plan.inner(), sources, self.request_capabilities(plan)?)?;
        self.prepare_request_resources(plan, &compiled)
    }

    /// Validate a complete portable-export route before commitment arithmetic.
    pub(crate) fn preflight_portable_export(
        &self,
        plan: &CommitmentExecutionPlan,
        sources: &[&dyn CommitmentSource<F>],
    ) -> Result<(), AkitaError> {
        let _compiled =
            compile_commitment_request(plan.inner(), sources, self.request_capabilities(plan)?)?;
        self.validate_plan_capabilities(plan)?;
        super::state_policy::validate_portable_export_route(
            &self.state_exporters(plan)?,
            plan.mode(),
        )
    }

    /// Validate the state-consumer edges needed by later proving stages.
    ///
    /// This check does not discover or materialize commitment sources. It is
    /// intended for proof-wide validation before transcript mutation.
    #[cfg(test)]
    pub(crate) fn preflight_prover_state_consumers(
        &self,
        plan: &CommitmentExecutionPlan,
    ) -> Result<(), AkitaError> {
        self.validate_plan_capabilities(plan)?;
        super::state_policy::validate_prover_state_consumer_route(
            &self.state_exporters(plan)?,
            plan.mode(),
        )
    }

    #[cfg(test)]
    pub(crate) fn release_built_ntt_slots_deduplicated(
        &self,
        owners: &mut Vec<crate::opaque::NttCacheOwnerId>,
    ) -> Result<usize, AkitaError> {
        let mut freed = 0usize;
        let mut registrations = Vec::with_capacity(4);
        if let Some(inner) = self.inner() {
            registrations.push(&inner.stage.resources);
        }
        if let Some(outer) = self.outer() {
            registrations.push(&outer.stage.resources);
        }
        if let Some(compression) = &self.compression {
            registrations.push(&compression.stage.resources);
        }
        if let Some(fused) = self.fused() {
            registrations.push(&fused.stage.resources);
        }
        for resources in registrations {
            let Some(owner) = resources.cache_owner_id() else {
                continue;
            };
            if owners.contains(&owner) {
                continue;
            }
            freed = freed
                .checked_add(resources.release_built_ntt_slots()?)
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("released commitment NTT bytes overflow".into())
                })?;
            owners.push(owner);
        }
        Ok(freed)
    }

    fn execute_inner_stages(
        &self,
        plan: &CommitmentExecutionPlan,
        sources: &[&dyn CommitmentSource<F>],
    ) -> Result<InnerCommitOutput, AkitaError> {
        let inner_registration = self.split_inner()?;
        let compiled =
            compile_commitment_request(plan.inner(), sources, &inner_registration.capabilities)?;
        self.prepare_request_resources(plan, &compiled)?;
        let resolved = compiled.materialize(None)?;
        let binding = CommitmentStateBinding::new(
            self.setup.clone(),
            *plan.inner(),
            sources.len(),
            plan.relation_mode(),
            plan.compression().cloned(),
        )?;
        let output =
            inner_registration
                .stage
                .operation
                .commit_inner(&binding, plan.inner(), &resolved)?;
        Self::validate_output_state(&binding, output.image(), &inner_registration.owner, "inner")?;
        Ok(output)
    }

    /// Execute an inner-only route and bind its policy-selected state.
    pub(crate) fn execute_inner(
        &self,
        plan: &CommitmentExecutionPlan,
        sources: &[&dyn CommitmentSource<F>],
    ) -> Result<SP::State, AkitaError> {
        if plan.mode() != super::CommitmentExecutionMode::InnerOnly {
            return Err(AkitaError::InvalidInput(
                "inner-only execution requires an inner-only commitment plan".into(),
            ));
        }
        self.trace_route(plan.mode());
        let inner = self.execute_inner_stages(plan, sources)?;
        let components = super::CommitmentStateComponents::new(
            super::CommitmentExecutionMode::InnerOnly,
            inner.into_image(),
            None,
            Zeroizing::new(Vec::new()),
            self.state_exporters(plan)?,
        )?;
        self.state_policy.bind(components)
    }

    pub(crate) fn execute_uncompressed_stages(
        &self,
        plan: &CommitmentExecutionPlan,
        sources: &[&dyn CommitmentSource<F>],
        accepted: Option<AcceptedReach>,
    ) -> Result<UncompressedCommitmentOutput<F>, AkitaError> {
        let uncompressed = plan.uncompressed().ok_or_else(|| {
            AkitaError::InvalidInput(
                "commitment execution plan does not contain an outer stage".into(),
            )
        })?;
        let compiled = compile_commitment_request(
            uncompressed.inner(),
            sources,
            self.request_capabilities(plan)?,
        )?;
        self.prepare_request_resources(plan, &compiled)?;
        let resolved = compiled.materialize(accepted)?;
        let binding = CommitmentStateBinding::new(
            self.setup.clone(),
            *uncompressed.inner(),
            sources.len(),
            plan.relation_mode(),
            plan.compression().cloned(),
        )?;
        // The randomness width is part of the plan, so a clear plan samples
        // nothing and keeps the commitment deterministic.
        let randomness = match uncompressed.outer().randomness_coefficient_len()? {
            0 => Zeroizing::new(Vec::new()),
            len => {
                let log_basis = uncompressed.outer().randomness_log_basis().ok_or_else(|| {
                    AkitaError::UnsupportedSchedule(
                        "hiding commitment has no consuming opening alphabet".into(),
                    )
                })?;
                self.hiding_rng.sample(len, log_basis)?
            }
        };
        if let Some(fused) = self.fused_for_plan(plan) {
            if !randomness.is_empty() {
                return Err(AkitaError::UnsupportedSchedule(
                    "fused commitment routes do not support hiding randomness".into(),
                ));
            }
            let output =
                fused
                    .stage
                    .operation
                    .commit_inner_outer(&binding, uncompressed, &resolved)?;
            Self::validate_output_state(
                &binding,
                output.image(),
                &fused.owner,
                "fused inner/outer",
            )?;
            return Ok(output);
        }
        let inner_registration = self.split_inner()?;
        let outer_registration = self.split_outer()?;
        let inner = inner_registration.stage.operation.commit_inner(
            &binding,
            uncompressed.inner(),
            &resolved,
        )?;
        Self::validate_output_state(&binding, inner.image(), &inner_registration.owner, "inner")?;
        let u = if inner_registration
            .owner
            .same_owner(&outer_registration.owner)
        {
            outer_registration.stage.operation.commit_outer(
                uncompressed,
                super::InnerImageInput::Owned(inner.image()),
                &randomness,
            )?
        } else {
            let rows = self
                .split_inner()?
                .exporter
                .as_ref()
                .ok_or_else(|| {
                    AkitaError::InvalidInput(
                        "cross-owner outer route has no inner-image export operation".into(),
                    )
                })?
                .export_inner_rows(uncompressed.inner(), inner.image())?;
            outer_registration.stage.operation.commit_outer(
                uncompressed,
                super::InnerImageInput::HostRows(&rows),
                &randomness,
            )?
        };
        UncompressedCommitmentOutput::new(inner.into_image(), u, uncompressed)?
            .with_outer_randomness(randomness, uncompressed)
    }

    /// Execute an uncompressed A/B route and bind its policy-selected state.
    pub(crate) fn execute_uncompressed(
        &self,
        plan: &CommitmentExecutionPlan,
        sources: &[&dyn CommitmentSource<F>],
    ) -> Result<CommitmentExecutionOutput<F, SP::State>, AkitaError> {
        if plan.mode() != super::CommitmentExecutionMode::Uncompressed {
            return Err(AkitaError::InvalidInput(
                "uncompressed execution requires an uncompressed commitment plan".into(),
            ));
        }
        self.trace_route(plan.mode());
        let output = self.execute_uncompressed_stages(plan, sources, None)?;
        CommitmentExecutionOutput::from_uncompressed(
            output,
            &self.state_policy,
            self.state_exporters(plan)?,
        )
    }

    /// Execute the complete checked split CPU commitment route.
    pub(crate) fn execute_full(
        &self,
        plan: &CommitmentExecutionPlan,
        sources: &[&dyn CommitmentSource<F>],
    ) -> Result<CommitmentExecutionOutput<F, SP::State>, AkitaError> {
        self.trace_route(plan.mode());
        let compression_plan = plan.compression().ok_or_else(|| {
            AkitaError::InvalidInput(
                "commitment execution plan does not contain a compression stage".into(),
            )
        })?;
        let relation_mode = plan.relation_mode().ok_or_else(|| {
            AkitaError::InvalidInput(
                "compressed commitment execution plan has no relation mode".into(),
            )
        })?;
        let uncompressed = self.execute_uncompressed_stages(plan, sources, None)?;
        let (image, u, outer_randomness) = uncompressed.into_parts();
        let expected_binding = image.binding().clone();
        let compression_registration = self.compression()?;
        let compression = compression_registration.stage.operation.compress(
            &expected_binding,
            compression_plan,
            relation_mode,
            u,
        )?;
        Self::validate_output_state(
            &expected_binding,
            compression.state(),
            &compression_registration.owner,
            "compression",
        )?;
        let output = FullCommitmentOutput::new(image, compression, outer_randomness)?;
        CommitmentExecutionOutput::from_full(
            output,
            &self.state_policy,
            self.state_exporters(plan)?,
        )
    }

    fn trace_route(&self, mode: super::CommitmentExecutionMode) {
        let fused = self.fused();
        let inner = self.inner();
        let outer = self.outer();
        let compression = self.compression.as_ref();
        tracing::debug!(
            ?mode,
            inner = inner.map(|registration| registration.stage.name),
            outer = outer.map(|registration| registration.stage.name),
            compression = compression.map(|registration| registration.stage.name),
            inner_operation = ?inner.map(|registration| registration.stage.operation_id),
            outer_operation = ?outer.map(|registration| registration.stage.operation_id),
            compression_operation = ?compression.map(|registration| registration.stage.operation_id),
            inner_backend = ?inner.map(|registration| registration.stage.backend_instance),
            outer_backend = ?outer.map(|registration| registration.stage.backend_instance),
            compression_backend = ?compression.map(|registration| registration.stage.backend_instance),
            inner_cache_owner = ?inner.and_then(|registration| registration.stage.resources.cache_owner_id()),
            outer_cache_owner = ?outer.and_then(|registration| registration.stage.resources.cache_owner_id()),
            compression_cache_owner = ?compression.and_then(|registration| registration.stage.resources.cache_owner_id()),
            fused = fused.is_some() && mode != super::CommitmentExecutionMode::InnerOnly,
            fused_operation = ?fused.map(|registration| registration.stage.operation_id),
            fused_backend = ?fused.map(|registration| registration.stage.backend_instance),
            fused_cache_owner = ?fused.and_then(|registration| registration.stage.resources.cache_owner_id()),
            "resolved commitment executor route"
        );
    }

    fn validate_output_state<K>(
        expected: &CommitmentStateBinding,
        actual: &super::BackendStateRef<K>,
        owner: &super::StateOwnerCapability<K>,
        stage: &'static str,
    ) -> Result<(), AkitaError> {
        if actual.binding() != expected {
            return Err(AkitaError::Internal(format!(
                "{stage} operation returned state bound to a different commitment request"
            )));
        }
        if !owner.owns(actual) {
            return Err(AkitaError::Internal(format!(
                "{stage} operation returned state owned by a different prepared implementation"
            )));
        }
        Ok(())
    }
}

impl<'a, F> CommitmentExecutor<'a, F, PortableStatePolicy>
where
    F: Field
        + CanonicalEncoding
        + Unreduced
        + WithCommitAccumulator
        + akita_serialization::Valid
        + akita_serialization::AkitaSerialize
        + 'static,
{
    /// Complete a CPU root through the public outer-image compression route,
    /// retaining the A-stage rows required by later proving stages.
    pub(crate) fn execute_full_via_outer_image<E>(
        &self,
        backend: &CpuBackend<F, E>,
        profile: GroupCommitPhaseParams,
        randomness_log_basis: Option<u32>,
        sources: &[&dyn CommitmentSource<F>],
        accepted: Option<AcceptedReach>,
    ) -> Result<(CommittedGroup<F>, PortableCommitmentHandle<F>), AkitaError>
where {
        let mut plan = CommitmentExecutionPlan::for_root(&profile)?;
        if let Some(log_basis) = randomness_log_basis {
            plan = plan.with_randomness_log_basis(log_basis);
        }
        self.trace_route(plan.mode());
        let (image, outer_image, outer_randomness) = self
            .execute_uncompressed_stages(&plan, sources, accepted)?
            .into_parts();
        let binding = image.binding().clone();
        let exporter = self.state_exporters(&plan)?.inner.ok_or_else(|| {
            AkitaError::InvalidInput("CPU commitment route has no inner-image exporter".into())
        })?;
        let rows = exporter.consume_inner_rows(binding.inner_plan(), image)?;
        let rows = InnerRelationStateMaterial::from_binding(&binding, rows)?.into_rows();
        let (committed_group, compression_state) =
            backend.compress_root_image(profile, outer_image)?;
        let retained = super::state_policy::assemble_portable_hint(
            binding,
            super::CommitmentExecutionMode::Full,
            rows,
            Some(compression_state),
            outer_randomness,
        )?;
        Ok((committed_group, retained))
    }
}

#[cfg(test)]
#[path = "tests/executor.rs"]
mod tests;

#[cfg(test)]
#[path = "tests/executor_binding.rs"]
mod binding_tests;

#[cfg(test)]
#[path = "tests/executor_resources.rs"]
mod resource_tests;
