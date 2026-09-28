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
mod imported_outer;
mod outer_slices;
mod plan;
mod prepared;
mod provider;
mod registration;
mod resources;
mod source;
mod stages;
mod state_policy;

#[cfg(test)]
pub(crate) use api::commit;
pub use api::GroupContext;
pub(crate) use api::{resolve_commit_params, resolve_polynomial_group_layout};
pub(crate) use outer_slices::for_each_outer_slice_input;
pub use state_policy::PortableCompressionState;

pub(crate) use builder::CommitmentExecutorBuilder;
pub(crate) use capabilities::CompressionOperationCapabilities;
pub use capabilities::{CommitmentRequestCapabilities, StageDimensionCapabilities};
pub(crate) use executor::CommitmentExecutor;
pub(crate) use plan::{CommitmentExecutionMode, CommitmentExecutionPlan};
pub use plan::{OuterCommitPlan, UncompressedCommitPlan};
pub(crate) use prepared::{PreparedCompression, PreparedFusedCommitment};
pub use prepared::{PreparedInnerCommitment, PreparedOuterCommitment};
pub use provider::{CommitmentStageProvider, CommitmentStages};
pub(crate) use registration::CompressionState;
pub use registration::{BackendStateRef, CommitmentStateBinding, InnerImage, StateOwnerCapability};
pub use resources::{
    BackendInstanceId, CommitmentNttRequirement, CommitmentNttStage, CommitmentOperationContext,
    CommitmentResourceControl, StageResources,
};
pub(crate) use resources::{
    CommitmentNttRoute, CommitmentOperationId, PreparedCommitmentResources,
};

pub(crate) use source::{compile_commitment_request, CompiledCommitmentRequest};
pub use source::{
    DenseCoefficientSource, DenseRepresentation, DenseType, OneHotIndexWidth, OneHotRepresentation,
    OneHotType, PolynomialType, PredecomposedDigitPlanes, ResolvedCommitSource,
    ShortNormRepresentation, ShortNormType, UnitPositionSlice,
};

pub use external::{
    cpu_external_inner_commitment_capability, cpu_external_inner_prepared_setup, BackendKindId,
    ExternalInnerCommitmentCapability, ExternalInnerCommitmentInput,
    ExternalInnerCommitmentOperation, ExternalOperationIdentity, PreparedExternalInnerCommitment,
};
pub use source::{
    AvailablePolynomialTypes, CommitSourceClass, CommitSourceDescriptor, CommitmentSource,
    PolynomialRepresentation, PolynomialTypeSelection,
};
pub(crate) use stages::{
    CompressionOperation, CompressionStageOutput, FullCommitmentOutput, FusedInnerOuterOperation,
    UncompressedCommitmentOutput,
};
pub use stages::{
    InnerCommitOperation, InnerCommitOutput, InnerImageExportOperation, InnerImageInput,
    OuterCommitOperation,
};
pub(crate) use state_policy::{
    CommitmentExecutionOutput, CommitmentStateComponents, CommitmentStatePolicy,
    InnerRelationMaterial, InnerRelationState, InnerRelationStateMaterial,
    IntoPortableCommitmentState, OuterCompressionMaterial, OuterCompressionState,
    PortableCompressionStateExport, PortableStatePolicy, ResidentStatePolicy,
    TerminalTFieldsMessage,
};

mod portable;
mod setup_prefix;
pub(crate) use portable::PortableCommitmentHandle;
pub use setup_prefix::{SetupPrefixProverRegistry, SetupPrefixSlot};

#[cfg(test)]
pub(crate) use external::ExternalFusedInnerCommitmentEncoder;
#[cfg(test)]
pub(crate) use state_policy::{NoRetainedStatePolicy, PortableCommitmentState};
