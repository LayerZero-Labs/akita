//! Commitment API, source contracts, execution plans, backend stages, routing,
//! resources, and retained state.
//!
//! This is the canonical home for commitment orchestration. General compute
//! backend traits and shared CPU primitives remain in [`crate::opaque`].

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
mod source;
mod stages;
mod state_policy;

pub use api::GroupContext;
pub(crate) use api::{commit, resolve_polynomial_group_layout};
pub(crate) use outer_slices::for_each_outer_slice_input;

pub(crate) use builder::CommitmentExecutorBuilder;
pub(crate) use capabilities::{
    CommitmentRequestCapabilities, CompressionOperationCapabilities, StageDimensionCapabilities,
};
pub(crate) use executor::CommitmentExecutor;
pub(crate) use external::{
    BackendKindId, ExternalInnerCommitmentCapability, PreparedExternalInnerCommitment,
};
pub(crate) use plan::{
    CommitmentExecutionMode, CommitmentExecutionPlan, OuterCommitPlan, UncompressedCommitPlan,
};
pub(crate) use prepared::{
    PreparedCompression, PreparedFusedCommitment, PreparedInnerCommitment, PreparedOuterCommitment,
};
pub(crate) use registration::{
    BackendStateRef, CommitmentStateBinding, CompressionState, InnerImage, StateOwnerCapability,
};
pub(crate) use resources::{
    BackendInstanceId, CommitmentNttRequirement, CommitmentNttRoute, CommitmentNttStage,
    CommitmentOperationContext, CommitmentOperationId, PreparedCommitmentResources, StageResources,
};

pub(crate) use source::{
    compile_commitment_request, AvailablePolynomialTypes, CommitSourceClass,
    CommitSourceDescriptor, CommitmentSource, CompiledCommitmentRequest, DenseCoefficientSource,
    DenseRepresentation, DenseType, OneHotIndexWidth, PolynomialRepresentation, PolynomialType,
    PolynomialTypeSelection, PredecomposedDigitPlanes, ResolvedCommitSource, UnitPositionSlice,
};
pub(crate) use stages::{
    CompressionOperation, CompressionStageOutput, FullCommitmentOutput, FusedInnerOuterOperation,
    InnerCommitOperation, InnerCommitOutput, InnerImageExportOperation, InnerImageInput,
    OuterCommitOperation, UncompressedCommitmentOutput,
};
pub(crate) use state_policy::{
    CommitmentExecutionOutput, CommitmentStateComponents, CommitmentStatePolicy,
    InnerRelationMaterial, InnerRelationState, InnerRelationStateMaterial,
    IntoPortableCommitmentState, OuterCompressionMaterial, OuterCompressionState,
    PortableCompressionState, PortableCompressionStateExport, PortableStatePolicy,
    ResidentStatePolicy, TerminalTFieldsMessage,
};

mod portable;
mod setup_prefix;
pub(crate) use portable::PortableCommitmentHandle;
pub use setup_prefix::{SetupPrefixProverRegistry, SetupPrefixSlot};

#[cfg(test)]
pub(crate) use external::{
    ExternalFusedInnerCommitmentEncoder, ExternalInnerCommitmentInput,
    ExternalInnerCommitmentOperation, ExternalOperationIdentity,
};
#[cfg(test)]
pub(crate) use resources::CommitmentResourceControl;
#[cfg(test)]
pub(crate) use source::{OneHotRepresentation, OneHotType, ShortNormRepresentation, ShortNormType};
#[cfg(test)]
pub(crate) use state_policy::{NoRetainedStatePolicy, PortableCommitmentState};
