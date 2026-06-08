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
//! | Module | Role |
//! | --- | --- |
//! | [`plans`](plans) | Legacy row/commit plan structs and [`FlatBlockTable`] |
//! | [`backend`](backend) | Fixed trait ladder (`ComputeBackendSetup` … `ProverComputeBackend`); removed at PO4 |
//! | [`cpu`](cpu) | [`CpuBackend`] / [`CpuPreparedSetup`] and standard row-kernel impls |
//! | [`operation_plans`](operation_plans) | PO1 scalar operation parameters (`CommitInnerPlan`, `OpeningFoldPlan`, …) |
//! | [`kernels`](kernels) | Source-typed operation kernel traits generic over view `S` |
//! | [`poly`](poly) | Root polynomial capability traits (`RootPolyShape`, `RootCommitSource`, …) |
//! | [`stack`](stack) | [`OperationCtx`] and heterogeneous [`ProverComputeStack`] |

mod backend;
mod cpu;
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
pub use kernels::{
    OpeningBatchKernel, OpeningFoldKernel, RingSwitchQuotientKernel, RingSwitchRelationKernel,
    RootCommitKernel, TensorPackedWitness, TensorProjectionBatchKernel, TensorProjectionKernel,
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
pub use poly::{
    AkitaRootPoly, DirectRootWitnessSource, RootCommitSource, RootOpeningSource, RootPolyShape,
    RootTensorSource,
};
pub use stack::{OperationCtx, ProverComputeStack};
