use super::{
    executor::{
        CompressionRegistration, FusedRegistration, InnerRegistration, OuterRegistration,
        StageRegistration,
    },
    BackendInstanceId, BackendKindId, CommitmentExecutor, CommitmentOperationContext,
    CommitmentRequestCapabilities, CommitmentStatePolicy, CompressionOperation,
    CompressionOperationCapabilities, FusedInnerOuterOperation, InnerCommitOperation, InnerImage,
    InnerImageExportOperation, OuterCommitOperation, PolynomialType,
    PortableCompressionStateExport, StageDimensionCapabilities, StageResources,
    StateOwnerCapability,
};
use akita_error::AkitaError;
use akita_types::AkitaExpandedSetup;
use jolt_field::{CanonicalEncoding, Field};
use std::sync::Arc;

/// Builder for independently registered commitment stage operations.
pub struct CommitmentExecutorBuilder<'a, F, SP>
where
    F: Field + CanonicalEncoding,
    SP: CommitmentStatePolicy<F>,
{
    setup: akita_types::AkitaSetupDescriptor,
    state_policy: SP,
    inner_capabilities: Option<CommitmentRequestCapabilities>,
    inner: Option<InnerRegistration<'a, F>>,
    outer: Option<OuterRegistration<'a, F>>,
    compression: Option<CompressionRegistration<'a, F>>,
    fused: Option<FusedRegistration<'a, F>>,
}

impl<'a, F, SP> CommitmentExecutorBuilder<'a, F, SP>
where
    F: Field + CanonicalEncoding + 'static,
    SP: CommitmentStatePolicy<F>,
{
    /// Start a route builder bound to one explicit expanded setup.
    pub fn new<C: 'static>(
        expanded: &AkitaExpandedSetup<F>,
        inner_backend: BackendKindId,
        standard_types: Vec<PolynomialType>,
        state_policy: SP,
    ) -> Self {
        Self {
            setup: expanded.descriptor().clone(),
            state_policy,
            inner_capabilities: Some(CommitmentRequestCapabilities::split::<C>(
                inner_backend,
                standard_types,
            )),
            inner: None,
            outer: None,
            compression: None,
            fused: None,
        }
    }

    /// Start a fused-only route builder bound to one explicit expanded setup.
    ///
    /// The fused operation supplies its own request capabilities when it is
    /// registered. No unused split inner or outer operation is required.
    pub fn new_fused(expanded: &AkitaExpandedSetup<F>, state_policy: SP) -> Self {
        Self {
            setup: expanded.descriptor().clone(),
            state_policy,
            inner_capabilities: None,
            inner: None,
            outer: None,
            compression: None,
            fused: None,
        }
    }

    pub(super) fn accept_any_standard_type(&mut self) {
        if let Some(capabilities) = &mut self.inner_capabilities {
            capabilities.accept_any_standard_type();
        }
    }

    /// Issue an opaque identity shared by registrations on one backend instance.
    pub fn issue_backend_instance(&self) -> BackendInstanceId {
        BackendInstanceId::issue()
    }

    /// Validate and bind reusable registration metadata to this executor setup.
    pub fn operation_context(
        &self,
        backend_instance: BackendInstanceId,
        name: &'static str,
        resources: StageResources<'a, F>,
    ) -> Result<CommitmentOperationContext<'a, F>, AkitaError> {
        if name.is_empty() {
            return Err(AkitaError::InvalidSetup(
                "commitment stage registration requires a diagnostic name".into(),
            ));
        }
        resources.validate_setup(&self.setup)?;
        Ok(CommitmentOperationContext {
            backend_instance,
            name,
            resources,
        })
    }

    /// Register the inner operation and its optional host-export edge.
    pub fn register_inner<O>(
        &mut self,
        operation: Arc<O>,
        owner: StateOwnerCapability<InnerImage>,
        context: CommitmentOperationContext<'a, F>,
        dimensions: StageDimensionCapabilities,
        exporter: Option<Arc<dyn InnerImageExportOperation<F>>>,
    ) -> Result<(), AkitaError>
    where
        O: InnerCommitOperation<F> + 'a,
    {
        if self.inner.is_some() {
            return Err(AkitaError::InvalidSetup(
                "commitment executor inner operation is already registered".into(),
            ));
        }
        let operation: Arc<dyn InnerCommitOperation<F> + 'a> = operation;
        let capabilities = self.inner_capabilities.take().ok_or_else(|| {
            AkitaError::InvalidSetup("commitment executor inner capabilities are absent".into())
        })?;
        self.inner = Some(InnerRegistration {
            stage: StageRegistration::new(operation, context),
            owner,
            capabilities,
            dimensions,
            exporter,
        });
        Ok(())
    }

    /// Register the independently selected outer operation.
    pub fn register_outer<O>(
        &mut self,
        operation: Arc<O>,
        owner: StateOwnerCapability<InnerImage>,
        context: CommitmentOperationContext<'a, F>,
        dimensions: StageDimensionCapabilities,
    ) -> Result<(), AkitaError>
    where
        O: OuterCommitOperation<F> + 'a,
    {
        if self.outer.is_some() {
            return Err(AkitaError::InvalidSetup(
                "commitment executor outer operation is already registered".into(),
            ));
        }
        let operation: Arc<dyn OuterCommitOperation<F> + 'a> = operation;
        self.outer = Some(OuterRegistration {
            stage: StageRegistration::new(operation, context),
            owner,
            dimensions,
        });
        Ok(())
    }

    /// Register the independently selected compression operation.
    pub fn register_compression<O>(
        &mut self,
        operation: Arc<O>,
        context: CommitmentOperationContext<'a, F>,
        capabilities: CompressionOperationCapabilities,
        exporter: Option<Arc<dyn PortableCompressionStateExport<F>>>,
    ) -> Result<(), AkitaError>
    where
        O: CompressionOperation<F> + 'a,
    {
        if self.compression.is_some() {
            return Err(AkitaError::InvalidSetup(
                "commitment executor compression operation is already registered".into(),
            ));
        }
        let operation: Arc<dyn CompressionOperation<F> + 'a> = operation;
        self.compression = Some(CompressionRegistration {
            stage: StageRegistration::new(operation, context),
            capabilities,
            exporter,
        });
        Ok(())
    }

    /// Explicitly select one fused inner-plus-outer operation for routes that
    /// contain both stages. Inner-only execution continues to use `inner`.
    pub fn register_fused<O>(
        &mut self,
        operation: Arc<O>,
        context: CommitmentOperationContext<'a, F>,
        capabilities: CommitmentRequestCapabilities,
        dimensions: StageDimensionCapabilities,
        exporter: Option<Arc<dyn InnerImageExportOperation<F>>>,
    ) -> Result<(), AkitaError>
    where
        O: FusedInnerOuterOperation<F> + 'a,
    {
        if self.fused.is_some() {
            return Err(AkitaError::InvalidSetup(
                "commitment executor fused operation is already registered".into(),
            ));
        }
        if !capabilities.is_fused() {
            return Err(AkitaError::InvalidSetup(
                "fused registration requires fused request capabilities".into(),
            ));
        }
        capabilities.validate()?;
        let operation: Arc<dyn FusedInnerOuterOperation<F> + 'a> = operation;
        self.fused = Some(FusedRegistration {
            stage: StageRegistration::new(operation, context),
            capabilities,
            dimensions,
            exporter,
        });
        Ok(())
    }

    /// Finish after compression and either a split or fused A/B route exist.
    pub fn build(self) -> Result<CommitmentExecutor<'a, F, SP>, AkitaError> {
        let inner = self.inner;
        let outer = self.outer;
        let compression = self.compression.ok_or_else(|| {
            AkitaError::InvalidSetup("commitment executor has no compression operation".into())
        })?;
        match (&inner, &outer, &self.fused) {
            (Some(inner), Some(outer), _) => {
                inner.capabilities.validate()?;
                if !inner.owner.same_owner(&outer.owner) && inner.exporter.is_none() {
                    return Err(AkitaError::InvalidSetup(
                        "cross-owner outer route requires an inner-image exporter".into(),
                    ));
                }
            }
            (None, None, Some(_)) => {}
            (None, None, None) => {
                return Err(AkitaError::InvalidSetup(
                    "commitment executor has neither a split nor fused A/B route".into(),
                ));
            }
            _ => {
                return Err(AkitaError::InvalidSetup(
                    "split commitment route requires both inner and outer operations".into(),
                ));
            }
        }
        Ok(CommitmentExecutor {
            setup: self.setup,
            inner,
            outer,
            compression,
            fused: self.fused,
            state_policy: self.state_policy,
        })
    }
}

#[cfg(test)]
#[path = "builder_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "fusion_tests.rs"]
mod fusion_tests;

#[cfg(test)]
#[path = "external_fusion_tests.rs"]
mod external_fusion_tests;

#[cfg(test)]
#[path = "source_selection_tests.rs"]
mod source_selection_tests;
