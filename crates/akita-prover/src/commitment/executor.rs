use super::{
    compile_commitment_request, BackendKindId, CommitmentExecutionOutput, CommitmentExecutionPlan,
    CommitmentExecutorBuilder, CommitmentRequestCapabilities, CommitmentSource,
    CommitmentStateBinding, CommitmentStateOutput, CommitmentStatePolicy,
    CompiledCommitmentRequest, CompressionOperationCapabilities, FullCommitmentOutput,
    InnerCommitOutput, PolynomialType, PreparedCommitmentResources, PreparedCompression,
    PreparedFusedCommitment, PreparedInnerCommitment, PreparedOuterCommitment, ResidentStatePolicy,
    StageDimensionCapabilities, StageResources, UncompressedCommitmentOutput,
};
use crate::compute::{
    ComputeBackendSetup, CpuBackend, CpuCompressionOperation, CpuInnerCommitOperation,
    CpuOuterCommitOperation, CpuPreparedSetup, PlannedNttCacheOwnerMetric,
};
use akita_error::AkitaError;
use akita_types::AkitaExpandedSetup;
use jolt_field::{CanonicalEncoding, Field, Unreduced, WithCommitAccumulator};
use std::sync::Arc;

use super::state_policy::CommitmentStateExporters;

#[cfg(test)]
struct CpuInnerContext;

/// Checked commitment-stage executor.
///
/// Registrations keep each operation with its capabilities and explicit
/// export edge. Stage outputs directly own their backend values, and the
/// selected state policy assembles those values without a state registry.
pub struct CommitmentExecutor<
    'a,
    F: Field + CanonicalEncoding,
    SP: CommitmentStatePolicy<F> = ResidentStatePolicy,
> {
    pub(super) setup: akita_types::AkitaSetupDescriptor,
    pub(super) inner: Option<PreparedInnerCommitment<'a, F>>,
    pub(super) outer: Option<PreparedOuterCommitment<'a, F>>,
    pub(super) compression: Option<PreparedCompression<'a, F>>,
    pub(super) fused: Option<PreparedFusedCommitment<'a, F>>,
    pub(super) state_policy: SP,
}

impl<'a, F, SP> CommitmentExecutor<'a, F, SP>
where
    F: Field + CanonicalEncoding + Unreduced + WithCommitAccumulator + 'static,
    SP: CommitmentStatePolicy<F>,
{
    /// Build a CPU executor with an explicit standard-representation registry.
    pub fn cpu(
        backend: &'a CpuBackend,
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
            StageDimensionCapabilities::cpu_role::<F>(akita_types::RingRole::Inner),
            Some(inner_operation.portable_exporter()),
        )?)?;
        builder.register_outer(PreparedOuterCommitment::new(
            outer_operation,
            outer_owner,
            outer_context,
            StageDimensionCapabilities::cpu_role::<F>(akita_types::RingRole::Outer),
        ))?;
        builder.register_compression(PreparedCompression::new(
            compression_operation.clone(),
            compression_operation.owner().clone(),
            compression_context,
            CompressionOperationCapabilities::cpu::<F>(),
            Some(compression_operation.portable_exporter()),
        ))?;
        builder.build()
    }
}

impl<'a, F, SP> CommitmentExecutor<'a, F, SP>
where
    F: Field + CanonicalEncoding,
    SP: CommitmentStatePolicy<F>,
{
    pub(super) fn matches_inner_outer_kind(&self, kind: super::InnerOuterRouteKind) -> bool {
        match kind {
            super::InnerOuterRouteKind::Fused => self.fused.is_some(),
            super::InnerOuterRouteKind::Split => {
                self.fused.is_none() && self.inner.is_some() && self.outer.is_some()
            }
        }
    }

    fn split_inner(&self) -> Result<&PreparedInnerCommitment<'a, F>, AkitaError> {
        self.inner.as_ref().ok_or_else(|| {
            AkitaError::InvalidInput("commitment route has no split inner operation".into())
        })
    }

    fn split_outer(&self) -> Result<&PreparedOuterCommitment<'a, F>, AkitaError> {
        self.outer.as_ref().ok_or_else(|| {
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
        self.fused.as_ref()
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
        let inner = plan.inner();
        let inner_width =
            akita_error::checked::product([inner.num_positions_per_block, inner.num_digits_inner])
                .ok_or_else(|| AkitaError::InvalidSetup("commitment A width overflow".into()))?;
        let mut required =
            akita_error::checked::product([inner.n_a, inner_width, inner.ring_dimension])
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("commitment A setup footprint overflow".into())
                })?;
        if let Some(uncompressed) = plan.uncompressed() {
            let outer = uncompressed.outer();
            let outer_required = akita_error::checked::product([
                outer.n_b(),
                outer.geometry().physical_input_width(),
                outer.ring_dimension(),
            ])
            .ok_or_else(|| {
                AkitaError::InvalidSetup("commitment B setup footprint overflow".into())
            })?;
            required = required.max(outer_required);
        }
        if let Some(compression) = plan.compression() {
            required = required.max(compression.max_setup_field_elements()?);
        }
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
    pub fn prewarm_request(
        &self,
        plan: &CommitmentExecutionPlan,
        sources: &[&dyn CommitmentSource<F>],
    ) -> Result<(), AkitaError> {
        let compiled =
            compile_commitment_request(plan.inner(), sources, self.request_capabilities(plan)?)?;
        self.prepare_request_resources(plan, &compiled)
    }

    /// Validate source admission and every selected stage capability without
    /// materializing a source or invoking an arithmetic operation.
    pub fn preflight_request(
        &self,
        plan: &CommitmentExecutionPlan,
        sources: &[&dyn CommitmentSource<F>],
    ) -> Result<(), AkitaError> {
        let _compiled =
            compile_commitment_request(plan.inner(), sources, self.request_capabilities(plan)?)?;
        self.validate_plan_capabilities(plan)
    }

    /// Validate a complete portable-export route before commitment arithmetic.
    pub fn preflight_portable_export(
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
    pub fn preflight_prover_state_consumers(
        &self,
        plan: &CommitmentExecutionPlan,
    ) -> Result<(), AkitaError> {
        self.validate_plan_capabilities(plan)?;
        super::state_policy::validate_prover_state_consumer_route(
            &self.state_exporters(plan)?,
            plan.mode(),
        )
    }

    /// Planned retained NTT cache metrics after stage routing and owner aliasing.
    pub fn planned_request_ntt_cache_metrics(
        &self,
        plan: &CommitmentExecutionPlan,
        sources: &[&dyn CommitmentSource<F>],
    ) -> Result<Vec<PlannedNttCacheOwnerMetric>, AkitaError> {
        let compiled =
            compile_commitment_request(plan.inner(), sources, self.request_capabilities(plan)?)?;
        let mut metrics = Vec::<PlannedNttCacheOwnerMetric>::new();
        let mut entry_bytes = Vec::<Vec<usize>>::new();
        for requirement in self.request_ntt_requirements(plan, &compiled)? {
            let registration = self.stage_resources(plan, requirement.stage())?;
            if !registration.requirement_is_cached(requirement)? {
                continue;
            }
            let owner_id = registration.cache_owner_id().ok_or_else(|| {
                AkitaError::InvalidSetup("cached commitment requirement has no owner".into())
            })?;
            let bytes = registration.planned_ntt_cache_entry_bytes(requirement)?;
            let owner_index = metrics
                .iter()
                .position(|metric| metric.owner_id == owner_id)
                .unwrap_or_else(|| {
                    metrics.push(PlannedNttCacheOwnerMetric {
                        owner_id,
                        keys: Vec::new(),
                        cache_bytes: 0,
                    });
                    entry_bytes.push(Vec::new());
                    metrics.len() - 1
                });
            let key = requirement.key();
            match metrics[owner_index]
                .keys
                .iter()
                .position(|stored| stored.ring_d == key.ring_d && stored.domain == key.domain)
            {
                Some(index)
                    if key.num_ring_elements
                        > metrics[owner_index].keys[index].num_ring_elements =>
                {
                    metrics[owner_index].keys[index] = key;
                    entry_bytes[owner_index][index] = bytes;
                }
                Some(_) => {}
                None => {
                    metrics[owner_index].keys.push(key);
                    entry_bytes[owner_index].push(bytes);
                }
            }
        }
        for (metric, bytes) in metrics.iter_mut().zip(entry_bytes) {
            metric.cache_bytes = bytes.into_iter().try_fold(0usize, |total, entry| {
                total.checked_add(entry).ok_or_else(|| {
                    AkitaError::InvalidSetup("planned commitment NTT bytes overflow".into())
                })
            })?;
        }
        Ok(metrics)
    }

    /// Release each physical stage resource owner at most once.
    pub fn release_built_ntt_slots(&self) -> Result<usize, AkitaError> {
        self.release_built_ntt_slots_deduplicated(&mut Vec::new())
    }

    pub(crate) fn release_built_ntt_slots_deduplicated(
        &self,
        owners: &mut Vec<crate::compute::NttCacheOwnerId>,
    ) -> Result<usize, AkitaError> {
        let mut freed = 0usize;
        let mut registrations = Vec::with_capacity(4);
        if let Some(inner) = &self.inner {
            registrations.push(&inner.stage.resources);
        }
        if let Some(outer) = &self.outer {
            registrations.push(&outer.stage.resources);
        }
        if let Some(compression) = &self.compression {
            registrations.push(&compression.stage.resources);
        }
        if let Some(fused) = &self.fused {
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
        let resolved = compiled.materialize()?;
        let binding = CommitmentStateBinding::new(
            self.setup.clone(),
            *plan.inner(),
            sources.len(),
            plan.relation_mode(),
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
    pub fn execute_inner(
        &self,
        plan: &CommitmentExecutionPlan,
        sources: &[&dyn CommitmentSource<F>],
    ) -> Result<CommitmentStateOutput<SP::State>, AkitaError> {
        if plan.mode() != super::CommitmentExecutionMode::InnerOnly {
            return Err(AkitaError::InvalidInput(
                "inner-only execution requires an inner-only commitment plan".into(),
            ));
        }
        self.trace_route(plan.mode());
        let inner = self.execute_inner_stages(plan, sources)?;
        CommitmentStateOutput::from_inner(
            inner.into_image(),
            &self.state_policy,
            self.state_exporters(plan)?,
        )
    }

    pub(crate) fn execute_uncompressed_stages(
        &self,
        plan: &CommitmentExecutionPlan,
        sources: &[&dyn CommitmentSource<F>],
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
        let resolved = compiled.materialize()?;
        let binding = CommitmentStateBinding::new(
            self.setup.clone(),
            *uncompressed.inner(),
            sources.len(),
            plan.relation_mode(),
        )?;
        if let Some(fused) = self.fused_for_plan(plan) {
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
            outer_registration
                .stage
                .operation
                .commit_outer(uncompressed, super::InnerImageInput::Owned(inner.image()))?
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
            outer_registration
                .stage
                .operation
                .commit_outer(uncompressed, super::InnerImageInput::HostRows(&rows))?
        };
        UncompressedCommitmentOutput::new(inner.into_image(), u, uncompressed)
    }

    /// Execute an uncompressed A/B route and bind its policy-selected state.
    pub fn execute_uncompressed(
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
        let output = self.execute_uncompressed_stages(plan, sources)?;
        CommitmentExecutionOutput::from_uncompressed(
            output,
            &self.state_policy,
            self.state_exporters(plan)?,
        )
    }

    /// Execute the complete checked split CPU commitment route.
    pub fn execute_full(
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
        let uncompressed = self.execute_uncompressed_stages(plan, sources)?;
        let (image, u) = uncompressed.into_parts();
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
        let output = FullCommitmentOutput::new(image, compression)?;
        CommitmentExecutionOutput::from_full(
            output,
            &self.state_policy,
            self.state_exporters(plan)?,
        )
    }

    fn trace_route(&self, mode: super::CommitmentExecutionMode) {
        let fused = self.fused.as_ref();
        let inner = self.inner.as_ref();
        let outer = self.outer.as_ref();
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
            return Err(AkitaError::InvalidInput(format!(
                "{stage} operation returned state bound to a different commitment request"
            )));
        }
        if !owner.owns(actual) {
            return Err(AkitaError::InvalidInput(format!(
                "{stage} operation returned state owned by a different prepared implementation"
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commitment::external::CpuBackendKind;
    use crate::commitment::{
        AvailablePolynomialTypes, CommitSourceClass, CommitSourceDescriptor, CompressionOperation,
        CompressionState, DenseCoefficientSource, DenseRepresentation, DenseType,
        NoRetainedStatePolicy, OuterCommitOperation, PolynomialRepresentation,
        PolynomialTypeSelection, PortableStatePolicy, StateOwnerCapability,
    };
    use crate::{AkitaProverSetup, DensePoly};
    use akita_challenges::SparseChallengeConfig;
    use akita_types::{
        CommittedGroupParams, SetupMatrixCapacity, SisModulusProfileId, TerminalFoldParams,
    };
    use jolt_field::{Prime64Offset59, Ring};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    type F = Prime64Offset59;

    fn plan() -> CommitmentExecutionPlan {
        let params = CommittedGroupParams::params_only(
            SisModulusProfileId::Q64Offset59,
            64,
            2,
            1,
            1,
            1,
            SparseChallengeConfig::pm1_only(1),
        )
        .with_decomp(4, 8, 1, 2, 2)
        .unwrap();
        let terminal = TerminalFoldParams::from_expanded_group(params);
        CommitmentExecutionPlan::for_terminal(&terminal).unwrap()
    }

    fn split_capabilities() -> CommitmentRequestCapabilities {
        CommitmentRequestCapabilities::split::<CpuInnerContext>(
            BackendKindId::of::<CpuBackendKind>("cpu").unwrap(),
            vec![PolynomialType::Dense(DenseType::Coefficients)],
        )
    }

    struct CountingDense {
        poly: DensePoly<F>,
        materializations: AtomicUsize,
    }

    impl DenseCoefficientSource<F> for CountingDense {
        fn coefficients(&self) -> &[F] {
            self.poly.field_coeffs()
        }
    }

    impl CommitmentSource<F> for CountingDense {
        fn descriptor(&self) -> Result<CommitSourceDescriptor, AkitaError> {
            CommitSourceDescriptor::new(9, 512, 512, CommitSourceClass::Dense, "counting-dense")
        }

        fn committed_centered_reach(
            &self,
            _modulus: u128,
            _centering_threshold: u128,
        ) -> Result<(u128, u128), AkitaError> {
            Ok((0, 1))
        }

        fn available_polynomial_types(
            &self,
            _plan: &crate::compute::CommitInnerPlan,
        ) -> Result<AvailablePolynomialTypes, AkitaError> {
            AvailablePolynomialTypes::new(vec![PolynomialType::Dense(DenseType::Coefficients)])
        }

        fn represent_as(
            &self,
            _selected: PolynomialTypeSelection,
            _plan: &crate::compute::CommitInnerPlan,
        ) -> Result<PolynomialRepresentation<'_, F>, AkitaError> {
            self.materializations.fetch_add(1, Ordering::SeqCst);
            Ok(PolynomialRepresentation::Dense(
                DenseRepresentation::Coefficients(self),
            ))
        }
    }

    #[test]
    fn cpu_executor_prepares_resources_before_source_materialization() {
        let plan = plan();
        let setup = AkitaProverSetup::<F>::generate_with_capacity(
            9,
            1,
            SetupMatrixCapacity {
                num_field_elements: 3 * 64,
            },
        )
        .unwrap();
        let backend = CpuBackend::DEFAULT;
        let prepared = backend.prepare_setup(&setup).unwrap();
        let executor = CommitmentExecutor::cpu(
            &backend,
            &prepared,
            setup.expanded.as_ref(),
            vec![PolynomialType::Dense(DenseType::Coefficients)],
            ResidentStatePolicy,
        )
        .unwrap();
        let source = CountingDense {
            poly: DensePoly::from_field_evals(9, vec![F::from_u64(1); 512]).unwrap(),
            materializations: AtomicUsize::new(0),
        };
        let sources: [&dyn CommitmentSource<F>; 1] = [&source];

        let error = executor
            .execute_inner(&plan, &sources)
            .err()
            .expect("undersized setup must fail before materialization");
        assert!(matches!(
            error,
            AkitaError::InvalidSetup(message) if message.contains("setup field elements")
        ));
        assert_eq!(source.materializations.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn unsupported_inner_dimension_is_rejected_before_materialization() {
        let plan = plan();
        let setup = AkitaProverSetup::<F>::generate_with_capacity(
            9,
            1,
            SetupMatrixCapacity {
                num_field_elements: 128 * 64,
            },
        )
        .unwrap();
        let backend = CpuBackend::DEFAULT;
        let prepared = backend.prepare_setup(&setup).unwrap();
        let mut executor = CommitmentExecutor::cpu(
            &backend,
            &prepared,
            setup.expanded.as_ref(),
            vec![PolynomialType::Dense(DenseType::Coefficients)],
            ResidentStatePolicy,
        )
        .unwrap();
        executor.inner.as_mut().unwrap().dimensions =
            StageDimensionCapabilities::new(vec![128]).unwrap();
        let source = CountingDense {
            poly: DensePoly::from_field_evals(9, vec![F::from_u64(1); 512]).unwrap(),
            materializations: AtomicUsize::new(0),
        };
        let sources: [&dyn CommitmentSource<F>; 1] = [&source];

        assert!(executor.execute_inner(&plan, &sources).is_err());
        assert_eq!(source.materializations.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn unsupported_compression_mode_is_rejected_before_materialization() {
        let params = CommittedGroupParams::params_only(
            SisModulusProfileId::Q64Offset59,
            64,
            2,
            1,
            1,
            1,
            SparseChallengeConfig::pm1_only(1),
        )
        .with_decomp(4, 8, 1, 2, 2)
        .unwrap();
        let plan = CommitmentExecutionPlan::for_root(&params.own_group().profile).unwrap();
        let setup = AkitaProverSetup::<F>::generate_with_capacity(
            9,
            1,
            SetupMatrixCapacity {
                num_field_elements: 128 * 64,
            },
        )
        .unwrap();
        let backend = CpuBackend::DEFAULT;
        let prepared = backend.prepare_setup(&setup).unwrap();
        let mut executor = CommitmentExecutor::cpu(
            &backend,
            &prepared,
            setup.expanded.as_ref(),
            vec![PolynomialType::Dense(DenseType::Coefficients)],
            ResidentStatePolicy,
        )
        .unwrap();
        executor.compression.as_mut().unwrap().capabilities =
            CompressionOperationCapabilities::new(
                StageDimensionCapabilities::new(vec![32, 16]).unwrap(),
                vec![akita_types::RingRelationMode::ReducedEvaluation],
            )
            .unwrap();
        let source = CountingDense {
            poly: DensePoly::from_field_evals(9, vec![F::from_u64(1); 512]).unwrap(),
            materializations: AtomicUsize::new(0),
        };
        let sources: [&dyn CommitmentSource<F>; 1] = [&source];

        assert!(executor.execute_full(&plan, &sources).is_err());
        assert_eq!(source.materializations.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn cpu_executor_reports_prewarms_and_releases_shared_resources() {
        let plan = plan();
        let setup = AkitaProverSetup::<F>::generate_with_capacity(
            9,
            1,
            SetupMatrixCapacity {
                num_field_elements: 4 * 64,
            },
        )
        .unwrap();
        let backend = CpuBackend::DEFAULT;
        let prepared = backend.prepare_setup(&setup).unwrap();
        let executor = CommitmentExecutor::cpu(
            &backend,
            &prepared,
            setup.expanded.as_ref(),
            vec![PolynomialType::Dense(DenseType::Coefficients)],
            ResidentStatePolicy,
        )
        .unwrap();
        let source = CountingDense {
            poly: DensePoly::from_field_evals(9, vec![F::from_u64(1); 512]).unwrap(),
            materializations: AtomicUsize::new(0),
        };
        let sources: [&dyn CommitmentSource<F>; 1] = [&source];

        let metrics = executor
            .planned_request_ntt_cache_metrics(&plan, &sources)
            .unwrap();
        assert_eq!(metrics.len(), 1);
        assert_eq!(metrics[0].keys.len(), 1);
        assert!(metrics[0].cache_bytes > 0);
        executor.prewarm_request(&plan, &sources).unwrap();
        assert_eq!(source.materializations.load(Ordering::SeqCst), 0);
        let cached = prepared.shared_ntt_cache_bytes();
        assert_eq!(cached, metrics[0].cache_bytes);
        assert_eq!(executor.release_built_ntt_slots().unwrap(), cached);
        assert_eq!(prepared.shared_ntt_cache_bytes(), 0);
    }

    struct RecordingOuter {
        host_calls: Arc<AtomicUsize>,
    }

    impl OuterCommitOperation<F> for RecordingOuter {
        fn commit_outer(
            &self,
            plan: &super::super::UncompressedCommitPlan,
            inner: super::super::InnerImageInput<'_, F>,
        ) -> Result<akita_types::RingVec<F>, AkitaError> {
            if !matches!(inner, super::super::InnerImageInput::HostRows(_)) {
                return Err(AkitaError::InvalidInput(
                    "recording outer expected an explicit host transfer".into(),
                ));
            }
            self.host_calls.fetch_add(1, Ordering::SeqCst);
            akita_types::RingVec::from_coeffs_with_ring_dim(
                vec![F::from_u64(0); plan.outer().output_coefficient_len()?],
                plan.outer().ring_dimension(),
            )
        }
    }

    struct RecordingCompression {
        calls: Arc<AtomicUsize>,
        owner: StateOwnerCapability<CompressionState>,
    }

    impl CompressionOperation<F> for RecordingCompression {
        fn compress(
            &self,
            binding: &CommitmentStateBinding,
            plan: &akita_types::CompressionChainPlan,
            relation_mode: akita_types::RingRelationMode,
            _u: akita_types::RingVec<F>,
        ) -> Result<super::super::CompressionStageOutput<F>, AkitaError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let map = plan
                .maps()
                .last()
                .ok_or_else(|| AkitaError::InvalidSetup("empty compression chain".into()))?;
            let terminal = akita_types::RingVec::from_coeffs_with_ring_dim(
                vec![F::from_u64(0); map.output_coefficients()],
                map.ring_dimension(),
            )?;
            let state = self.owner.bind(binding.clone(), 0, ());
            super::super::CompressionStageOutput::new(terminal, state, plan, relation_mode)
        }
    }

    #[test]
    fn builder_routes_independent_outer_and_compression_operations() {
        let params = CommittedGroupParams::params_only(
            SisModulusProfileId::Q64Offset59,
            64,
            2,
            1,
            1,
            1,
            SparseChallengeConfig::pm1_only(1),
        )
        .with_decomp(4, 8, 1, 2, 2)
        .unwrap();
        let plan = CommitmentExecutionPlan::for_root(&params.own_group().profile).unwrap();
        let setup = AkitaProverSetup::<F>::generate_with_capacity(
            9,
            1,
            SetupMatrixCapacity {
                num_field_elements: 128 * 64,
            },
        )
        .unwrap();
        let backend = CpuBackend::DEFAULT;
        let prepared = backend.prepare_setup(&setup).unwrap();
        let mut builder =
            CommitmentExecutorBuilder::new(setup.expanded.as_ref(), ResidentStatePolicy);
        let inner = Arc::new(CpuInnerCommitOperation::new(&backend, &prepared));
        let inner_owner = inner.owner().clone();
        let inner_resources = StageResources::controlled(
            PreparedCommitmentResources::new(&backend, &prepared, setup.expanded.as_ref()).unwrap(),
        );
        let inner_context = builder
            .operation_context(
                builder.issue_backend_instance(),
                "cpu-inner",
                inner_resources,
            )
            .unwrap();
        builder
            .register_inner(
                PreparedInnerCommitment::new(
                    inner.clone(),
                    inner_owner,
                    inner_context,
                    split_capabilities(),
                    StageDimensionCapabilities::new(vec![64]).unwrap(),
                    Some(inner.portable_exporter()),
                )
                .unwrap(),
            )
            .unwrap();
        let host_calls = Arc::new(AtomicUsize::new(0));
        let outer_context = builder
            .operation_context(
                builder.issue_backend_instance(),
                "recording-outer",
                StageResources::none(),
            )
            .unwrap();
        builder
            .register_outer(PreparedOuterCommitment::new(
                Arc::new(RecordingOuter {
                    host_calls: host_calls.clone(),
                }),
                StateOwnerCapability::new(),
                outer_context,
                StageDimensionCapabilities::new(vec![64]).unwrap(),
            ))
            .unwrap();
        let compression_calls = Arc::new(AtomicUsize::new(0));
        let compression_owner = StateOwnerCapability::new();
        let compression_context = builder
            .operation_context(
                builder.issue_backend_instance(),
                "recording-compression",
                StageResources::none(),
            )
            .unwrap();
        builder
            .register_compression(PreparedCompression::new(
                Arc::new(RecordingCompression {
                    calls: compression_calls.clone(),
                    owner: compression_owner.clone(),
                }),
                compression_owner,
                compression_context,
                CompressionOperationCapabilities::new(
                    StageDimensionCapabilities::new(vec![32, 16]).unwrap(),
                    vec![akita_types::RingRelationMode::QuotientLift],
                )
                .unwrap(),
                None,
            ))
            .unwrap();
        let executor = builder.build().unwrap();
        let poly = DensePoly::from_field_evals(9, vec![F::from_u64(1); 512]).unwrap();
        let sources: [&dyn CommitmentSource<F>; 1] = [&poly];

        let output = executor.execute_full(&plan, &sources).unwrap();
        assert_eq!(host_calls.load(Ordering::SeqCst), 1);
        assert_eq!(compression_calls.load(Ordering::SeqCst), 1);
        assert_eq!(
            output.terminal_payload().coeff_len(),
            plan.compression().unwrap().terminal_coefficients()
        );
    }

    #[test]
    fn builder_rejects_cross_owner_route_without_exporter() {
        let setup = AkitaProverSetup::<F>::generate_with_capacity(
            9,
            1,
            SetupMatrixCapacity {
                num_field_elements: 4 * 64,
            },
        )
        .unwrap();
        let backend = CpuBackend::DEFAULT;
        let prepared = backend.prepare_setup(&setup).unwrap();
        let mut builder =
            CommitmentExecutorBuilder::new(setup.expanded.as_ref(), ResidentStatePolicy);
        let inner = Arc::new(CpuInnerCommitOperation::new(&backend, &prepared));
        let inner_context = builder
            .operation_context(
                builder.issue_backend_instance(),
                "cpu-inner",
                StageResources::none(),
            )
            .unwrap();
        builder
            .register_inner(
                PreparedInnerCommitment::new(
                    inner.clone(),
                    inner.owner().clone(),
                    inner_context,
                    split_capabilities(),
                    StageDimensionCapabilities::new(vec![64]).unwrap(),
                    None,
                )
                .unwrap(),
            )
            .unwrap();
        let outer_context = builder
            .operation_context(
                builder.issue_backend_instance(),
                "recording-outer",
                StageResources::none(),
            )
            .unwrap();
        builder
            .register_outer(PreparedOuterCommitment::new(
                Arc::new(RecordingOuter {
                    host_calls: Arc::new(AtomicUsize::new(0)),
                }),
                StateOwnerCapability::new(),
                outer_context,
                StageDimensionCapabilities::new(vec![64]).unwrap(),
            ))
            .unwrap();
        let compression_owner = StateOwnerCapability::new();
        let compression_context = builder
            .operation_context(
                builder.issue_backend_instance(),
                "recording-compression",
                StageResources::none(),
            )
            .unwrap();
        builder
            .register_compression(PreparedCompression::new(
                Arc::new(RecordingCompression {
                    calls: Arc::new(AtomicUsize::new(0)),
                    owner: compression_owner.clone(),
                }),
                compression_owner,
                compression_context,
                CompressionOperationCapabilities::new(
                    StageDimensionCapabilities::new(vec![32, 16]).unwrap(),
                    vec![akita_types::RingRelationMode::QuotientLift],
                )
                .unwrap(),
                None,
            ))
            .unwrap();

        assert!(builder.build().is_err());
    }

    #[test]
    fn cpu_executor_inner_path_binds_setup_plan_and_source() {
        let plan = plan();
        let setup = AkitaProverSetup::<F>::generate_with_capacity(
            9,
            1,
            SetupMatrixCapacity {
                num_field_elements: 4 * 64,
            },
        )
        .unwrap();
        let backend = CpuBackend::DEFAULT;
        let prepared = backend.prepare_setup(&setup).unwrap();
        let executor = CommitmentExecutor::cpu(
            &backend,
            &prepared,
            setup.expanded.as_ref(),
            vec![PolynomialType::Dense(DenseType::Coefficients)],
            ResidentStatePolicy,
        )
        .unwrap();
        assert_eq!(
            executor.inner.as_ref().unwrap().stage.backend_instance,
            executor.outer.as_ref().unwrap().stage.backend_instance
        );
        assert_eq!(
            executor.outer.as_ref().unwrap().stage.backend_instance,
            executor
                .compression
                .as_ref()
                .unwrap()
                .stage
                .backend_instance
        );
        assert_ne!(
            executor.inner.as_ref().unwrap().stage.operation_id,
            executor.outer.as_ref().unwrap().stage.operation_id
        );
        assert_ne!(
            executor.outer.as_ref().unwrap().stage.operation_id,
            executor.compression.as_ref().unwrap().stage.operation_id
        );
        assert_eq!(
            executor
                .inner
                .as_ref()
                .unwrap()
                .stage
                .resources
                .cache_owner_id(),
            executor
                .outer
                .as_ref()
                .unwrap()
                .stage
                .resources
                .cache_owner_id()
        );
        assert_eq!(
            executor
                .outer
                .as_ref()
                .unwrap()
                .stage
                .resources
                .cache_owner_id(),
            executor
                .compression
                .as_ref()
                .unwrap()
                .stage
                .resources
                .cache_owner_id()
        );
        let poly = DensePoly::from_field_evals(9, vec![F::from_u64(1); 512]).unwrap();
        let sources: [&dyn CommitmentSource<F>; 1] = [&poly];
        let output = executor.execute_inner_stages(&plan, &sources).unwrap();
        let rows = executor
            .inner
            .as_ref()
            .unwrap()
            .exporter
            .as_ref()
            .unwrap()
            .export_inner_rows(plan.inner(), output.image())
            .unwrap();

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].ring_dim(), 64);
        drop(output);

        let output = executor.execute_inner(&plan, &sources).unwrap();
        assert_eq!(output.prover_state().binding().relation_mode(), None);
        drop(output);
    }

    #[test]
    fn cpu_executor_rejects_prepared_setup_from_another_descriptor() {
        let setup_a = AkitaProverSetup::<F>::generate_with_capacity(
            9,
            1,
            SetupMatrixCapacity {
                num_field_elements: 4 * 64,
            },
        )
        .unwrap();
        let setup_b = AkitaProverSetup::<F>::generate_with_capacity(
            10,
            1,
            SetupMatrixCapacity {
                num_field_elements: 4 * 64,
            },
        )
        .unwrap();
        let backend = CpuBackend::DEFAULT;
        let prepared_a = backend.prepare_setup(&setup_a).unwrap();

        assert!(CommitmentExecutor::cpu(
            &backend,
            &prepared_a,
            setup_b.expanded.as_ref(),
            vec![PolynomialType::Dense(DenseType::Coefficients)],
            ResidentStatePolicy,
        )
        .is_err());
    }

    #[test]
    fn cpu_executor_split_path_keeps_inner_resident_and_returns_checked_u() {
        let params = CommittedGroupParams::params_only(
            SisModulusProfileId::Q64Offset59,
            64,
            2,
            1,
            1,
            1,
            SparseChallengeConfig::pm1_only(1),
        )
        .with_decomp(4, 8, 1, 2, 2)
        .unwrap();
        let plan = CommitmentExecutionPlan::for_root(&params.own_group().profile).unwrap();
        let compression_capacity = plan
            .compression()
            .unwrap()
            .maps()
            .iter()
            .map(|map| map.input_width() * map.ring_dimension())
            .max()
            .unwrap();
        let setup = AkitaProverSetup::<F>::generate_with_capacity(
            9,
            1,
            SetupMatrixCapacity {
                num_field_elements: compression_capacity.max(4 * 64),
            },
        )
        .unwrap();
        let backend = CpuBackend::DEFAULT;
        let prepared = backend.prepare_setup(&setup).unwrap();
        let executor = CommitmentExecutor::cpu(
            &backend,
            &prepared,
            setup.expanded.as_ref(),
            vec![PolynomialType::Dense(DenseType::Coefficients)],
            ResidentStatePolicy,
        )
        .unwrap();
        let poly = DensePoly::from_field_evals(9, vec![F::from_u64(1); 512]).unwrap();
        let sources: [&dyn CommitmentSource<F>; 1] = [&poly];
        let output = executor
            .execute_uncompressed_stages(&plan, &sources)
            .unwrap();
        let uncompressed = plan.uncompressed().unwrap();

        assert_eq!(output.u().ring_dim(), uncompressed.outer().ring_dimension());
        assert_eq!(
            output.u().coeff_len(),
            uncompressed.outer().output_coefficient_len().unwrap()
        );
        assert_eq!(output.image().binding().inner_plan(), uncompressed.inner());
    }

    #[test]
    fn cpu_executor_full_path_retains_mode_bound_state() {
        let params = CommittedGroupParams::params_only(
            SisModulusProfileId::Q64Offset59,
            64,
            2,
            1,
            1,
            1,
            SparseChallengeConfig::pm1_only(1),
        )
        .with_decomp(4, 8, 1, 2, 2)
        .unwrap();
        let plan = CommitmentExecutionPlan::for_root(&params.own_group().profile).unwrap();
        let compression_capacity = plan
            .compression()
            .unwrap()
            .maps()
            .iter()
            .map(|map| map.input_width() * map.ring_dimension())
            .max()
            .unwrap();
        let setup = AkitaProverSetup::<F>::generate_with_capacity(
            9,
            1,
            SetupMatrixCapacity {
                num_field_elements: compression_capacity.max(4 * 64),
            },
        )
        .unwrap();
        let backend = CpuBackend::DEFAULT;
        let prepared = backend.prepare_setup(&setup).unwrap();
        let executor = CommitmentExecutor::cpu(
            &backend,
            &prepared,
            setup.expanded.as_ref(),
            vec![PolynomialType::Dense(DenseType::Coefficients)],
            ResidentStatePolicy,
        )
        .unwrap();
        let poly = DensePoly::from_field_evals(9, vec![F::from_u64(1); 512]).unwrap();
        let sources: [&dyn CommitmentSource<F>; 1] = [&poly];
        let output = executor.execute_full(&plan, &sources).unwrap();

        assert_eq!(
            output.prover_state().binding().relation_mode(),
            plan.relation_mode()
        );
        assert_eq!(
            output.terminal_payload().coeff_len(),
            plan.compression().unwrap().terminal_coefficients()
        );
        drop(output);

        let no_state_executor = CommitmentExecutor::cpu(
            &backend,
            &prepared,
            setup.expanded.as_ref(),
            vec![PolynomialType::Dense(DenseType::Coefficients)],
            NoRetainedStatePolicy,
        )
        .unwrap();
        let output = no_state_executor.execute_full(&plan, &sources).unwrap();
        let expected_payload = output.terminal_payload().clone();
        assert_eq!(output.prover_state(), &());

        let portable_executor = CommitmentExecutor::cpu(
            &backend,
            &prepared,
            setup.expanded.as_ref(),
            vec![PolynomialType::Dense(DenseType::Coefficients)],
            PortableStatePolicy,
        )
        .unwrap();
        let inner = portable_executor
            .execute_inner_stages(&plan, &sources)
            .unwrap();
        let expected_rows = portable_executor
            .inner
            .as_ref()
            .unwrap()
            .exporter
            .as_ref()
            .unwrap()
            .export_inner_rows(plan.inner(), inner.image())
            .unwrap();
        drop(inner);
        let output = portable_executor.execute_full(&plan, &sources).unwrap();
        let hint = output.prover_state();

        assert_eq!(output.terminal_payload(), &expected_payload);
        assert_eq!(hint.inner_rows(), expected_rows);
        assert_eq!(
            hint.outer_compression_witness(plan.compression().unwrap())
                .unwrap()
                .plan(),
            plan.compression().unwrap()
        );
        assert_eq!(
            hint.outer_compression_quotients(plan.compression().unwrap())
                .unwrap()
                .len(),
            plan.compression().unwrap().maps().len()
        );
    }
}

#[cfg(test)]
#[path = "tests/executor_binding.rs"]
mod binding_tests;

#[cfg(test)]
#[path = "tests/executor_resources.rs"]
mod resource_tests;
