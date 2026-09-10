//! Commitment API, source contracts, execution plans, backend stages, routing,
//! resources, and retained state.
//!
//! This is the canonical home for commitment orchestration. General compute
//! backend traits and shared CPU primitives remain in [`crate::compute`].

mod api;
mod builder;
mod capabilities;
mod executor;
mod external;
mod outer_slices;
mod plan;
mod prepared;
mod registration;
mod resources;
mod schedule;
mod source;
mod stages;
mod state_policy;

pub use api::{commit, resolve_polynomial_group_layout, CommitOutput, GroupContext};
pub use outer_slices::for_each_outer_slice_input;

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
pub use prepared::{
    PreparedCompression, PreparedFusedCommitment, PreparedInnerCommitment, PreparedOuterCommitment,
};
pub use registration::{
    BackendStateRef, CommitmentStateBinding, CompressionState, InnerImage, StateOwnerCapability,
};
pub use resources::{
    BackendInstanceId, CommitmentNttRequirement, CommitmentNttRoute, CommitmentNttStage,
    CommitmentOperationContext, CommitmentOperationId, CommitmentResourceControl,
    PreparedCommitmentResources, StageResources,
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
