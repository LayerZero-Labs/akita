use super::{
    BackendInstanceId, CommitmentOperationContext, CommitmentOperationId,
    CommitmentRequestCapabilities, CompressionOperation, CompressionOperationCapabilities,
    CompressionState, FusedInnerOuterOperation, InnerCommitOperation, InnerImage,
    InnerImageExportOperation, OuterCommitOperation, PortableCompressionStateExport,
    StageDimensionCapabilities, StageResources, StateOwnerCapability,
};
use akita_error::AkitaError;
use jolt_field::{CanonicalEncoding, Field};
use std::sync::Arc;

pub(super) struct PreparedStage<'a, F, O: ?Sized>
where
    F: Field + CanonicalEncoding,
{
    pub(super) operation: Arc<O>,
    setup: akita_types::AkitaSetupDescriptor,
    pub(super) operation_id: CommitmentOperationId,
    pub(super) backend_instance: BackendInstanceId,
    pub(super) name: &'static str,
    pub(super) resources: StageResources<'a, F>,
}

impl<'a, F, O: ?Sized> PreparedStage<'a, F, O>
where
    F: Field + CanonicalEncoding,
{
    pub(super) fn new(operation: Arc<O>, context: CommitmentOperationContext<'a, F>) -> Self {
        Self {
            operation,
            setup: context.setup,
            operation_id: CommitmentOperationId::issue(),
            backend_instance: context.backend_instance,
            name: context.name,
            resources: context.resources,
        }
    }

    pub(super) fn validate_setup(
        &self,
        setup: &akita_types::AkitaSetupDescriptor,
    ) -> Result<(), AkitaError> {
        if self.setup != *setup {
            return Err(AkitaError::InvalidSetup(
                "commitment stage was prepared for a different setup".into(),
            ));
        }
        self.resources.validate_setup(setup)
    }

    pub(super) fn ensure_ntt_slot(
        &self,
        requirement: super::CommitmentNttRequirement,
    ) -> Result<(), AkitaError> {
        if !self.resources.requirement_is_cached(requirement)? {
            return Ok(());
        }
        self.resources.ensure_ntt_slot(requirement)
    }
}

/// Prepared split-inner implementation and all metadata used to route it.
pub struct PreparedInnerCommitment<'a, F>
where
    F: Field + CanonicalEncoding,
{
    pub(super) stage: PreparedStage<'a, F, dyn InnerCommitOperation<F> + 'a>,
    pub(super) owner: StateOwnerCapability<InnerImage>,
    pub(super) capabilities: CommitmentRequestCapabilities,
    pub(super) dimensions: StageDimensionCapabilities,
    pub(super) exporter: Option<Arc<dyn InnerImageExportOperation<F>>>,
}

impl<'a, F> PreparedInnerCommitment<'a, F>
where
    F: Field + CanonicalEncoding,
{
    /// Bind one prepared operation to its ownership, routing, and export contract.
    pub fn new<O>(
        operation: Arc<O>,
        owner: StateOwnerCapability<InnerImage>,
        context: CommitmentOperationContext<'a, F>,
        capabilities: CommitmentRequestCapabilities,
        dimensions: StageDimensionCapabilities,
        exporter: Option<Arc<dyn InnerImageExportOperation<F>>>,
    ) -> Result<Self, AkitaError>
    where
        O: InnerCommitOperation<F> + 'a,
    {
        if capabilities.is_fused() {
            return Err(AkitaError::InvalidSetup(
                "split inner registration received fused capabilities".into(),
            ));
        }
        capabilities.validate()?;
        let operation: Arc<dyn InnerCommitOperation<F> + 'a> = operation;
        Ok(Self {
            stage: PreparedStage::new(operation, context),
            owner,
            capabilities,
            dimensions,
            exporter,
        })
    }
}

/// Prepared split-outer implementation and its resident-state ownership contract.
pub struct PreparedOuterCommitment<'a, F>
where
    F: Field + CanonicalEncoding,
{
    pub(super) stage: PreparedStage<'a, F, dyn OuterCommitOperation<F> + 'a>,
    pub(super) owner: StateOwnerCapability<InnerImage>,
    pub(super) dimensions: StageDimensionCapabilities,
}

impl<'a, F> PreparedOuterCommitment<'a, F>
where
    F: Field + CanonicalEncoding,
{
    /// Bind one prepared outer operation to its ownership and routing metadata.
    pub fn new<O>(
        operation: Arc<O>,
        owner: StateOwnerCapability<InnerImage>,
        context: CommitmentOperationContext<'a, F>,
        dimensions: StageDimensionCapabilities,
    ) -> Self
    where
        O: OuterCommitOperation<F> + 'a,
    {
        let operation: Arc<dyn OuterCommitOperation<F> + 'a> = operation;
        Self {
            stage: PreparedStage::new(operation, context),
            owner,
            dimensions,
        }
    }
}

/// Prepared compression implementation and its portable-state contract.
pub struct PreparedCompression<'a, F>
where
    F: Field + CanonicalEncoding,
{
    pub(super) stage: PreparedStage<'a, F, dyn CompressionOperation<F> + 'a>,
    pub(super) owner: StateOwnerCapability<CompressionState>,
    pub(super) capabilities: CompressionOperationCapabilities,
    pub(super) exporter: Option<Arc<dyn PortableCompressionStateExport<F>>>,
}

impl<'a, F> PreparedCompression<'a, F>
where
    F: Field + CanonicalEncoding,
{
    /// Bind one prepared compression operation to its routing and export metadata.
    pub fn new<O>(
        operation: Arc<O>,
        owner: StateOwnerCapability<CompressionState>,
        context: CommitmentOperationContext<'a, F>,
        capabilities: CompressionOperationCapabilities,
        exporter: Option<Arc<dyn PortableCompressionStateExport<F>>>,
    ) -> Self
    where
        O: CompressionOperation<F> + 'a,
    {
        let operation: Arc<dyn CompressionOperation<F> + 'a> = operation;
        Self {
            stage: PreparedStage::new(operation, context),
            owner,
            capabilities,
            exporter,
        }
    }
}

/// Prepared fused inner/outer implementation and all metadata used to route it.
pub struct PreparedFusedCommitment<'a, F>
where
    F: Field + CanonicalEncoding,
{
    pub(super) stage: PreparedStage<'a, F, dyn FusedInnerOuterOperation<F> + 'a>,
    pub(super) owner: StateOwnerCapability<InnerImage>,
    pub(super) capabilities: CommitmentRequestCapabilities,
    pub(super) dimensions: StageDimensionCapabilities,
    pub(super) exporter: Option<Arc<dyn InnerImageExportOperation<F>>>,
}

impl<'a, F> PreparedFusedCommitment<'a, F>
where
    F: Field + CanonicalEncoding,
{
    /// Bind one fused operation to its routing and export contract.
    pub fn new<O>(
        operation: Arc<O>,
        owner: StateOwnerCapability<InnerImage>,
        context: CommitmentOperationContext<'a, F>,
        capabilities: CommitmentRequestCapabilities,
        dimensions: StageDimensionCapabilities,
        exporter: Option<Arc<dyn InnerImageExportOperation<F>>>,
    ) -> Result<Self, AkitaError>
    where
        O: FusedInnerOuterOperation<F> + 'a,
    {
        if !capabilities.is_fused() {
            return Err(AkitaError::InvalidSetup(
                "fused registration requires fused request capabilities".into(),
            ));
        }
        capabilities.validate()?;
        let operation: Arc<dyn FusedInnerOuterOperation<F> + 'a> = operation;
        Ok(Self {
            stage: PreparedStage::new(operation, context),
            owner,
            capabilities,
            dimensions,
            exporter,
        })
    }
}
