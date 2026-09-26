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
pub(crate) use capabilities::{
    CommitmentRequestCapabilities, CompressionOperationCapabilities, StageDimensionCapabilities,
};
pub(crate) use executor::CommitmentExecutor;
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
    compile_commitment_request, CompiledCommitmentRequest, DenseCoefficientSource,
    DenseRepresentation, DenseType, OneHotIndexWidth, PolynomialType, PredecomposedDigitPlanes,
    ResolvedCommitSource, UnitPositionSlice,
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
    InnerCommitOperation, InnerCommitOutput, InnerImageExportOperation, InnerImageInput,
    OuterCommitOperation, UncompressedCommitmentOutput,
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
pub(crate) use resources::CommitmentResourceControl;
#[cfg(test)]
pub(crate) use source::{OneHotRepresentation, OneHotType, ShortNormRepresentation, ShortNormType};
#[cfg(test)]
pub(crate) use state_policy::{NoRetainedStatePolicy, PortableCommitmentState};
