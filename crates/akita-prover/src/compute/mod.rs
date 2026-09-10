//! Prover compute backend boundary.
//!
//! The first backend is the existing CPU/Rayon implementation. The boundary is
//! intentionally operation-shaped: migrated prover code asks the backend to run
//! named commit/protocol kernels, and does not reach through prepared setup for
//! raw CPU matrices or NTT slots.
//!
//! # Module layout
//!
//! Split by stable capability cluster (see `akita-polyops-cutover` spec), not by
//! call-site helper. Representation-specific views and kernel impls stay in
//! `backend/*`; this directory owns traits, scalar operation plans, and shared
//! CPU arithmetic.
//!
//! | Sibling module | Role |
//! | --- | --- |
//! | `plans` | Internal CPU inputs and named operation outputs |
//! | `backend` | Prepared setup, digit-row, compression, and ring-switch capabilities |
//! | `cpu` | `CpuBackend` / `CpuPreparedSetup` and standard row-kernel impls |
//! | `operation_plans` | PO1 scalar operation parameters (`CommitInnerPlan`, `OpeningFoldPlan`, …) |
//! | `kernels` | Source-typed operation kernel traits generic over view `S` |
//! | `poly` | Root polynomial opening, tensor, and shape capability traits |
//! | `stack` | Per-fold [`LevelProveStacks`] + per-cluster [`OperationCtx`] / [`ProverComputeStack`] |

mod backend;
pub mod commitment;
pub(crate) mod compression;
mod cpu;
pub mod delegating_cpu;
mod kernels;
mod operation_plans;
mod plans;
mod poly;
mod requirements;
mod runtime_capabilities;
mod stack;

pub use backend::{
    CompressionComputeBackend, CompressionRowsProducts, ComputeBackendSetup,
    CyclicRowsComputeBackend, DigitRowsComputeBackend, NttCacheOwnerId,
};
pub use commitment::{
    compile_commitment_request, cpu_external_inner_commitment_capability,
    cpu_external_inner_prepared_setup, AvailablePolynomialTypes, BackendInstanceId, BackendKindId,
    BackendStateRef, CommitSourceClass, CommitSourceDescriptor, CommitmentExecutionMode,
    CommitmentExecutionOutput, CommitmentExecutionPlan, CommitmentExecutionSchedule,
    CommitmentExecutionScheduleBuilder, CommitmentExecutor, CommitmentExecutorBuilder,
    CommitmentNttRequirement, CommitmentNttRoute, CommitmentNttStage, CommitmentOperationContext,
    CommitmentOperationId, CommitmentRequestCapabilities, CommitmentResourceControl,
    CommitmentRoundStep, CommitmentSource, CommitmentStateBinding, CommitmentStateComponents,
    CommitmentStateOutput, CommitmentStatePolicy, CompiledCommitmentRequest, CompressionOperation,
    CompressionOperationCapabilities, CompressionStageOutput, CompressionState,
    DenseCoefficientSource, DenseRepresentation, DenseType, ExternalFusedInnerCommitmentEncoder,
    ExternalInnerCommitmentCapability, ExternalInnerCommitmentInput,
    ExternalInnerCommitmentOperation, ExternalOperationIdentity, FullCommitmentOutput,
    FusedInnerOuterOperation, InnerCommitOperation, InnerCommitOutput, InnerImage,
    InnerImageExportOperation, InnerImageInput, InnerOuterRouteKind, InnerRelationState,
    InnerRelationStateMaterial, IntoPortableCommitmentState, NoRetainedStatePolicy,
    OneHotIndexWidth, OneHotRepresentation, OneHotType, OuterCommitOperation, OuterCommitPlan,
    OuterCompressionState, PolynomialRepresentation, PolynomialType, PolynomialTypeSelection,
    PortableCommitmentState, PortableCompressionState, PortableCompressionStateExport,
    PortableStatePolicy, PredecomposedDigitPlanes, PreparedCommitmentResources,
    PreparedCompression, PreparedExternalInnerCommitment, PreparedFusedCommitment,
    PreparedInnerCommitment, PreparedOuterCommitment, ResidentCommitmentState, ResidentStatePolicy,
    ResolvedCommitSource, ShortNormRepresentation, ShortNormType, StageDimensionCapabilities,
    StageResources, StateOwnerCapability, TerminalBindingState, TerminalTFieldsMessage,
    UncompressedCommitPlan, UncompressedCommitmentOutput, UnitPositionSlice,
};
pub use cpu::{
    CpuBackend, CpuCompressionOperation, CpuInnerCommitOperation, CpuOuterCommitOperation,
    CpuPreparedSetup, PreparedCrtNttProfile, PreparedNttCacheMetric,
};
pub use delegating_cpu::{CommitCluster, OpeningCluster, RingSwitchCluster, TensorCluster};
pub use kernels::{
    BatchDecomposeFoldOutcome, OpeningBatchKernel, OpeningFoldKernel, RingSwitchRelationKernel,
    SubringCoefficientPackingBatchKernel, TensorProjectionBatchKernel, TensorProjectionKernel,
};
pub use operation_plans::{
    CommitInnerPlan, DecomposeFoldBatchPlan, DecomposeFoldPlan, OpeningFoldOutput, OpeningFoldPlan,
    RingSwitchRelationPlan, SubringCoefficientPackingPartials, SubringCoefficientPackingPlan,
};
pub use plans::RingSwitchRelationRows;
pub use requirements::{NttExecutionRequirements, NttOperationCluster, RoutedNttRequirement};

pub use poly::{
    centered_reach_of_field_coeffs, OpeningProveBackendFor, ProveFlowBackendFor, ProveStackFor,
    RecursiveProveBackend, RingSwitchProveBackend, RootOpeningSource, RootPolyMeta, RootPolyShape,
    RootProveBackend, RootProvePoly, RootTensorSource, TensorBackendFor,
};
pub use runtime_capabilities::{
    RootProveFlowBackend, RuntimeCoefficientPackingBackendFor, RuntimeOpeningProveBackendFor,
    RuntimeOpeningSource, RuntimeRecursiveWitnessProveBackend, RuntimeRingSwitchProveBackend,
    RuntimeRootProvePoly, RuntimeTensorBackendFor, RuntimeTensorSource, SuffixOpeningProveBackend,
    SuffixTensorProveBackend,
};
pub use stack::{
    planned_ntt_cache_metrics, prewarm_ntt_requirements, LevelProveStacks, OperationCtx,
    PlannedNttCacheOwnerMetric, ProverComputeStack, ReleaseRootNttAfterFold, TieredProveStacks,
    UniformProverStack,
};
