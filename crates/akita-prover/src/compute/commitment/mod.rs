//! Composable commitment execution contracts.
//!
//! This module is introduced in stages. Source discovery and borrowed
//! representations land first; stage operations and the executor build on
//! these checked, ring-dimension-free inputs.

mod builder;
mod capabilities;
mod executor;
mod external;
mod plan;
mod registration;
mod resources;
mod schedule;
mod source;
mod stages;
mod state_policy;

pub use builder::CommitmentExecutorBuilder;
pub use capabilities::{
    CommitmentRequestCapabilities, CompressionOperationCapabilities, StageDimensionCapabilities,
};
pub use executor::CommitmentExecutor;
pub use external::{
    cpu_external_inner_commitment_capability, cpu_external_inner_prepared_setup, BackendKindId,
    ExternalFusedInnerCommitmentEncoder, ExternalInnerCommitmentCapability,
    ExternalInnerCommitmentInput, ExternalInnerCommitmentOperation, ExternalOperationIdentity,
    PreparedExternalInnerCommitment,
};
pub use plan::{
    CommitmentExecutionMode, CommitmentExecutionPlan, OuterCommitPlan, UncompressedCommitPlan,
};
pub use registration::{
    BackendStateRef, CommitmentStateBinding, CompressionState, InnerImage, StateOwnerCapability,
};
pub use resources::{
    BackendInstanceId, CommitmentNttRequirement, CommitmentNttStage, CommitmentOperationContext,
    CommitmentOperationId, CommitmentResourceControl, PreparedCommitmentResources, StageResources,
};
pub use schedule::{
    CommitmentExecutionSchedule, CommitmentExecutionScheduleBuilder, CommitmentRoundStep,
    InnerOuterRouteKind,
};

pub use source::{
    compile_commitment_request, AvailablePolynomialTypes, CommitSourceClass,
    CommitSourceDescriptor, CommitmentSource, CompiledCommitmentRequest, DenseCoefficientSource,
    DenseRepresentation, DenseType, OneHotIndexWidth, OneHotRepresentation, OneHotType,
    PolynomialRepresentation, PolynomialType, PolynomialTypeSelection, PredecomposedDigitPlanes,
    ResolvedCommitSource, ShortNormRepresentation, ShortNormType, UnitPositionSlice,
};
pub use stages::{
    CompressionOperation, CompressionStageOutput, FullCommitmentOutput, FusedInnerOuterOperation,
    InnerCommitOperation, InnerCommitOutput, InnerImageExportOperation, InnerImageInput,
    OuterCommitOperation, UncompressedCommitmentOutput,
};
pub use state_policy::{
    CommitmentExecutionOutput, CommitmentStateComponents, CommitmentStateOutput,
    CommitmentStatePolicy, InnerRelationState, InnerRelationStateMaterial,
    IntoPortableCommitmentState, NoRetainedStatePolicy, OuterCompressionState,
    PortableCommitmentState, PortableCompressionState, PortableCompressionStateExport,
    PortableStatePolicy, ResidentCommitmentState, ResidentStatePolicy, TerminalBindingState,
    TerminalTFieldsMessage,
};
