//! Accelerated commitment stages contributed to the owning CPU backend.

use super::{
    BackendKindId, CommitmentExecutionPlan, CommitmentExecutor, CommitmentExecutorBuilder,
    CommitmentRequestCapabilities, CommitmentSource, CommitmentStatePolicy,
    CompressionOperationCapabilities, PolynomialType, PortableStatePolicy,
    PreparedCommitmentResources, PreparedCompression, PreparedInnerCommitment,
    PreparedOuterCommitment, StageDimensionCapabilities, StageResources,
};
use crate::opaque::{
    ComputeBackendSetup, CpuBackend, CpuCompressionOperation, CpuInnerCommitOperation,
    CpuOuterCommitOperation, CpuPreparedSetup,
};
use akita_error::AkitaError;
use akita_types::{AkitaExpandedSetup, RingRole};
use jolt_field::{CanonicalEncoding, Field, Unreduced, WithCommitAccumulator};
use std::sync::Arc;

/// Extension seam: commitment stages that a compute backend contributes to the
/// owning CPU backend, for the Metal track in
/// `specs/akita-compute-backend-metal.md`.
///
/// A provider supplies prepared inner and outer stage operations for one
/// expanded setup. [`CpuBackend`] remains the only owner of handles and proof
/// state: the CPU reference operations fill every stage the provider leaves
/// empty, and compression always runs on the CPU. The route is selected per
/// request by a side-effect-free preflight, and selection never changes plans,
/// schedules, transcript order, or bytes.
///
/// Install a provider at runtime with
/// [`CpuBackend::with_commitment_stage_provider`]. No Cargo feature selects it.
pub trait CommitmentStageProvider<F: Field>: Send + Sync {
    /// Prepared stage operations for `expanded`.
    ///
    /// The backend calls this once per commitment request, so it must be
    /// cheap; keep device state inside the provider. Return empty stages to
    /// decline the setup. An error is returned to the committing caller and
    /// does not select the CPU route.
    fn stages<'a>(
        &'a self,
        expanded: &AkitaExpandedSetup<F>,
    ) -> Result<CommitmentStages<'a, F>, AkitaError>
    where
        F: CanonicalEncoding;
}

/// Optional accelerated inner and outer stages for one commitment executor.
///
/// A `None` stage runs on the CPU reference operation. An accelerated inner
/// stage paired with the CPU outer stage crosses owners, so it must register
/// an inner-image exporter; the portable retained state needs that exporter in
/// every route.
pub struct CommitmentStages<'a, F: Field + CanonicalEncoding> {
    /// Accelerated A-stage operation.
    pub inner: Option<PreparedInnerCommitment<'a, F>>,
    /// Accelerated B-stage operation.
    pub outer: Option<PreparedOuterCommitment<'a, F>>,
}

impl<F: Field + CanonicalEncoding> CommitmentStages<'_, F> {
    /// Whether every stage runs on the CPU.
    pub const fn is_empty(&self) -> bool {
        self.inner.is_none() && self.outer.is_none()
    }
}

impl<F: Field + CanonicalEncoding> Default for CommitmentStages<'_, F> {
    fn default() -> Self {
        Self {
            inner: None,
            outer: None,
        }
    }
}

impl<'a, F, SP> CommitmentExecutor<'a, F, SP>
where
    F: Field + CanonicalEncoding + Unreduced + WithCommitAccumulator + 'static,
    SP: CommitmentStatePolicy<F>,
{
    /// Build the owning CPU backend's executor.
    ///
    /// `accelerated` stages are registered as given. The CPU reference
    /// operations fill every omitted stage, and compression always runs on the
    /// CPU. With empty stages this is the plain CPU executor, whose inner stage
    /// accepts every standard polynomial type.
    pub(crate) fn new<E>(
        backend: &'a CpuBackend<F, E>,
        prepared: &'a CpuPreparedSetup<F>,
        expanded: &AkitaExpandedSetup<F>,
        standard_types: Vec<PolynomialType>,
        accelerated: CommitmentStages<'a, F>,
        state_policy: SP,
    ) -> Result<Self, AkitaError> {
        if backend.prepared_expanded_setup(prepared).descriptor() != expanded.descriptor() {
            return Err(AkitaError::InvalidSetup(
                "CPU commitment executor setup descriptor mismatch".into(),
            ));
        }
        let mut builder = CommitmentExecutorBuilder::new(expanded, state_policy);
        let cpu_inner = Arc::new(CpuInnerCommitOperation::new(backend, prepared));
        let backend_instance = builder.issue_backend_instance();
        let resources = StageResources::controlled(PreparedCommitmentResources::new(
            backend, prepared, expanded,
        )?);
        let inner = match accelerated.inner {
            Some(inner) => inner,
            None => {
                let mut capabilities = CommitmentRequestCapabilities::split::<CpuPreparedSetup<F>>(
                    BackendKindId::of::<super::external::CpuBackendKind>("cpu")?,
                    standard_types,
                );
                capabilities.accept_any_standard_type();
                PreparedInnerCommitment::new(
                    cpu_inner.clone(),
                    cpu_inner.owner().clone(),
                    builder.operation_context(backend_instance, "cpu-inner", resources.clone())?,
                    capabilities,
                    StageDimensionCapabilities::cpu_role::<F>(RingRole::Inner),
                    Some(cpu_inner.portable_exporter()),
                )?
            }
        };
        builder.register_inner(inner)?;
        let outer = match accelerated.outer {
            Some(outer) => outer,
            None => PreparedOuterCommitment::new(
                Arc::new(CpuOuterCommitOperation::new(
                    backend,
                    prepared,
                    cpu_inner.as_ref(),
                )),
                cpu_inner.owner().clone(),
                builder.operation_context(backend_instance, "cpu-outer", resources.clone())?,
                StageDimensionCapabilities::cpu_role::<F>(RingRole::Outer),
            ),
        };
        builder.register_outer(outer)?;
        let compression = Arc::new(CpuCompressionOperation::new(backend, prepared, expanded)?);
        builder.register_compression(PreparedCompression::new(
            compression.clone(),
            compression.owner().clone(),
            builder.operation_context(backend_instance, "cpu-compression", resources)?,
            CompressionOperationCapabilities::cpu::<F>(),
            Some(compression.portable_exporter()),
        ))?;
        builder.build()
    }
}

impl<F, E> CpuBackend<F, E>
where
    F: Field + CanonicalEncoding + Unreduced + WithCommitAccumulator + 'static,
{
    /// Select the portable-state commitment executor for one request.
    ///
    /// Without an installed [`CommitmentStageProvider`], or when it contributes
    /// no stage, this is the CPU executor. Otherwise the accelerated executor
    /// is kept only if `plan` and `sources` pass its portable-export
    /// preflight. Preflight compiles the request and checks capabilities,
    /// including [`super::InnerCommitOperation::supports_plan`] and
    /// [`super::OuterCommitOperation::supports_plan`], but materializes no
    /// source representation, so a declined request falls back to the CPU
    /// executor before any commitment work. A failure after selection is
    /// returned to the caller and never retried on the CPU.
    pub(crate) fn commitment_executor(
        &self,
        standard_types: Vec<PolynomialType>,
        plan: &CommitmentExecutionPlan,
        sources: &[&dyn CommitmentSource<F>],
    ) -> Result<CommitmentExecutor<'_, F, PortableStatePolicy>, AkitaError> {
        let prepared = self.prepared()?;
        let expanded = prepared.expanded.as_ref();
        let accelerated = match self.commitment_stage_provider() {
            Some(provider) => provider.stages(expanded)?,
            None => CommitmentStages::default(),
        };
        if accelerated.is_empty() {
            return CommitmentExecutor::new(
                self,
                prepared,
                expanded,
                standard_types,
                accelerated,
                PortableStatePolicy,
            );
        }
        let executor = CommitmentExecutor::new(
            self,
            prepared,
            expanded,
            standard_types.clone(),
            accelerated,
            PortableStatePolicy,
        )?;
        match executor.preflight_portable_export(plan, sources) {
            Ok(()) => Ok(executor),
            Err(reason) => {
                tracing::debug!(
                    %reason,
                    "commitment stage provider declined the request; using CPU stages"
                );
                CommitmentExecutor::new(
                    self,
                    prepared,
                    expanded,
                    standard_types,
                    CommitmentStages::default(),
                    PortableStatePolicy,
                )
            }
        }
    }
}
