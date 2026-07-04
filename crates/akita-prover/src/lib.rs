//! Prover-facing API surface for the Akita PCS.
//!
//! This crate owns prover-side polynomial backends, setup artifacts, recursive
//! witness construction, ring-switch handoff, and Akita-specific sumcheck
//! provers. Config and schedule policy live in `akita-config`.

pub mod api;
pub mod backend;
pub mod compute;
pub mod kernels;
pub mod protocol;
pub mod types;
mod validation;

use akita_algebra::CyclotomicRing;
use akita_field::FieldCore;
use akita_types::FlatDigitBlocks;

pub use types::ProverOpeningData;

pub use api::{
    batched_commit, batched_commit_with_params, commit_final_group, commit_setup_prefix,
    prepare_batched_commit_inputs, AkitaProverSetup, CommitmentProver, CommitmentWithHint,
};

pub use backend::{
    tensor_pack_recursive_witness, DensePoly, MultiChunkEntry, MultilinearPolynomial, OneHotIndex,
    OneHotPoly, RecursiveCommitmentHintCache, RecursiveWitnessFlat, RootTensorProjectionPoly,
    SingleChunkEntry, SparseRingBlockEntry, SparseRingPoly, SuffixWitnessBatchView,
    SuffixWitnessView,
};
pub use compute::{
    BatchDecomposeFoldOutcome, CommitBackendFor, CommitCluster, CommitmentComputeBackend,
    ComputeBackendSetup, CpuBackend, CpuPreparedSetup, CyclicRowsComputeBackend, DenseCommitInput,
    DenseCommitRowsPlan, DigitRowsComputeBackend, FlatBlockTable, LevelProveStacks,
    OneHotCommitBlocks, OneHotCommitRowsPlan, OpeningCluster, OpeningProveBackendFor, OperationCtx,
    PreparedCrtNttProfile, ProveBackendFor, ProveFlowBackendFor, ProveStackFor, ProverComputeStack,
    RecursiveProveBackend, RecursiveWitnessCommitRowsPlan, RingSwitchCluster,
    RingSwitchComputeBackend, RingSwitchProveBackend, RingSwitchQuotientRowsPlan,
    RingSwitchRelationRows, RingSwitchRelationRowsPlan, RootCommitBackend, RootCommitSource,
    RootOpeningSource, RootPolyShape, RootProveBackend, RootProvePoly, RootTensorSource,
    SparseRingCommitRowsPlan, SuffixDispatchOpeningProveBackendFor,
    SuffixDispatchTensorProveBackendFor, SuffixRingSwitchProveBackend, TensorBackendFor,
    TensorCluster, TieredProveStacks, UniformProverStack, RECURSIVE_SUFFIX_RING_DIMENSIONS,
};
pub use protocol::fold_grind::ProverTranscriptGrind;
pub use protocol::fold_grind_observer::{FoldGrindObservation, FoldGrindObserverGuard};
pub use protocol::sumcheck::{AkitaStage1Prover, AkitaStage2Prover};
pub use protocol::{
    batched_prove, commit_next_w, prove, prove_root, prove_root_direct, prove_suffix,
    prove_terminal_root_fold_with_params, ProveLevelOutput, RecursiveSuffixOutcome,
    RingSwitchOutput, SuffixProverState,
};
pub use protocol::{RingRelationInstance, RingRelationProver, RingRelationWitness};

/// Prover-side output of the decompose + challenge-fold step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecomposeFoldWitness<F: FieldCore, const D: usize> {
    /// Semantic folded witness rows in ring form (`z`).
    pub z_folded_rings: Vec<CyclotomicRing<F, D>>,
    /// Semantic centered integer coefficients for each `z_folded_rings` row.
    pub centered_coeffs: Vec<[i32; D]>,
    /// Infinity norm of `centered_coeffs`.
    pub centered_inf_norm: u32,
    /// Logarithm of the digit basis used for `committed_digits`.
    pub log_basis: u32,
    /// Number of folded-response digit planes per committed coordinate.
    pub num_digits_fold: usize,
    /// Public shift `eta` with committed coordinates `z_comm = z - eta`.
    pub committed_shift: u128,
    /// Balanced digit planes for the shifted coordinates committed in the recursive witness.
    pub committed_digits: Vec<[i8; D]>,
}

/// Prover-side output of the inner Ajtai commit step.
pub struct CommitInnerWitness<F: FieldCore, const D: usize> {
    /// Recombined inner `A * s_i` rows, grouped by block.
    pub recomposed_inner_rows: Vec<Vec<CyclotomicRing<F, D>>>,
    /// Digit decompositions of `A * s_i` in flat column-major order plus
    /// explicit block boundaries.
    pub decomposed_inner_rows: FlatDigitBlocks<D>,
}
