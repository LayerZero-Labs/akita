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
//! `backend/*`; this directory owns traits, shared plans, and the CPU row
//! helpers.
//!
//! | Sibling module | Role |
//! | --- | --- |
//! | `plans` | Legacy row/commit plan structs and `FlatBlockTable` |
//! | `backend` | Internal trait ladder (`ComputeBackendSetup` … `ProverComputeBackend`); not re-exported at crate root |
//! | `cpu` | `CpuBackend` / `CpuPreparedSetup` and standard row-kernel impls |
//! | `operation_plans` | PO1 scalar operation parameters (`CommitInnerPlan`, `OpeningFoldPlan`, …) |
//! | `kernels` | Source-typed operation kernel traits generic over view `S` |
//! | `poly` | Root polynomial capability traits (`RootPolyShape`, `RootCommitSource`, …) |
//! | `stack` | Per-fold [`LevelProveStacks`] + per-cluster [`OperationCtx`] / [`ProverComputeStack`] |

mod backend;
pub(crate) mod compression;
mod cpu;
pub mod delegating_cpu;
mod dispatch;
mod kernels;
mod operation_plans;
mod plans;
mod poly;
mod requirements;
mod stack;

pub use backend::{
    CommitmentComputeBackend, CompressionComputeBackend, CompressionRowsProducts,
    ComputeBackendSetup, CyclicRowsComputeBackend, DigitRowsComputeBackend, NttCacheOwnerId,
    ProverComputeBackend, RingSwitchComputeBackend,
};
pub use cpu::{CpuBackend, CpuPreparedSetup, PreparedCrtNttProfile, PreparedNttCacheMetric};
pub use delegating_cpu::{CommitCluster, OpeningCluster, RingSwitchCluster, TensorCluster};
pub(crate) use dispatch::tensor_root_projection;
pub use kernels::{
    BatchDecomposeFoldOutcome, OpeningBatchKernel, OpeningFoldKernel, RingSwitchQuotientKernel,
    RingSwitchRelationKernel, RootCommitKernel, TensorPackedWitness, TensorProjectionBatchKernel,
    TensorProjectionKernel,
};
pub use operation_plans::{
    CommitInnerPlan, DecomposeFoldBatchPlan, DecomposeFoldPlan, OpeningFoldOutput, OpeningFoldPlan,
    RingSwitchQuotientPlan, RingSwitchRelationPlan,
};
pub use crate::backend::onehot::LazyOneHotBlocks;
pub use plans::{
    DenseCommitInput, DenseCommitRowsPlan, FlatBlockTable, OneHotCommitBlocks,
    OneHotCommitRowsPlan, RecursiveWitnessCommitRowsPlan, RingSwitchQuotientRowsPlan,
    RingSwitchRelationRows, RingSwitchRelationRowsPlan, SparseRingCommitRowsPlan,
};
pub use requirements::{NttExecutionRequirements, NttOperationCluster, RoutedNttRequirement};

pub use poly::{
    CommitBackendFor, OpeningProveBackendFor, ProjectBackendFor, ProveBackendFor,
    ProveFlowBackendFor, ProveStackFor, RecursiveProveBackend, RingSwitchProveBackend,
    RootCommitBackend, RootCommitPoly, RootCommitPolys, RootCommitSource, RootOpeningSource,
    RootPolyMeta, RootPolyShape, RootProveBackend, RootProveFlowBackend, RootProvePoly,
    RootTensorSource, RuntimeCommitBackendFor, RuntimeOpeningProveBackendFor,
    RuntimeProveBackendFor, RuntimeRecursiveWitnessProveBackend, RuntimeRingSwitchProveBackend,
    RuntimeRootCommitBackend, RuntimeRootCommitPoly, RuntimeRootProvePoly, RuntimeTensorBackendFor,
    SuffixOpeningProveBackend, SuffixTensorProveBackend, TensorBackendFor,
    RECURSIVE_SUFFIX_RING_DIMENSIONS,
};
pub use stack::{
    planned_ntt_cache_metrics, prewarm_ntt_requirements, LevelProveStacks, OperationCtx,
    PlannedNttCacheOwnerMetric, ProverComputeStack, TieredProveStacks, UniformProverStack,
};
