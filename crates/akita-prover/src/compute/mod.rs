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
//! | `stack` | `OperationCtx` and heterogeneous `ProverComputeStack` |

mod backend;
mod cpu;
mod dispatch;
mod kernels;
mod operation_plans;
mod plans;
mod poly;
mod stack;

pub use backend::{
    CommitmentComputeBackend, ComputeBackendSetup, CyclicRowsComputeBackend,
    DigitRowsComputeBackend, ProverComputeBackend, RingSwitchComputeBackend,
};
pub use cpu::{CpuBackend, CpuPreparedSetup, PreparedCrtNttProfile};
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
pub use plans::{
    DenseCommitInput, DenseCommitRowsPlan, FlatBlockTable, OneHotCommitBlocks,
    OneHotCommitRowsPlan, RecursiveWitnessCommitRowsPlan, RingSwitchQuotientRowsPlan,
    RingSwitchRelationRows, RingSwitchRelationRowsPlan, SparseRingCommitRowsPlan,
};
pub use poly::ZkHidingCommitBackend;
pub use poly::{
    DirectRootWitnessSource, RecursiveProveBackend, RootCommitBackend, RootCommitPoly,
    RootCommitPolys, RootCommitSource, RootOpeningSource, RootPolyShape, RootProveBackend,
    RootProveFlowBackend, RootProvePoly, RootTensorProjectionCommitKernels,
    RootTensorProjectionProveKernels, RootTensorSource,
};
pub use stack::{OperationCtx, ProverComputeStack, UniformProverStack};
