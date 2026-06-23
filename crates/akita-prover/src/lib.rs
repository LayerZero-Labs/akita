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
mod validation;

use akita_algebra::CyclotomicRing;
use akita_field::FieldCore;
use akita_types::{FlatDigitBlocks, OpeningPoints};

pub use api::{
    batched_commit, batched_commit_with_params, commit, commit_setup_prefix, commit_with_params,
    prepare_batched_commit_inputs, prepare_commit_inputs, AkitaProverSetup, CommitmentProver,
};

pub use backend::{
    tensor_pack_recursive_witness, DensePoly, MultiChunkEntry, MultilinearPolynomial, OneHotIndex,
    OneHotPoly, RecursiveCommitmentHintCache, RecursiveWitnessFlat, RootTensorProjectionPoly,
    SingleChunkEntry, SparseRingBlockEntry, SparseRingPoly, SuffixWitnessBatchView,
    SuffixWitnessView,
};
pub use compute::{
    BatchDecomposeFoldOutcome, CommitBackendFor, CommitmentComputeBackend, ComputeBackendSetup,
    CpuBackend, CpuPreparedSetup, CyclicRowsComputeBackend, DenseCommitInput, DenseCommitRowsPlan,
    DigitRowsComputeBackend, FlatBlockTable, LevelProveStacks, OneHotCommitBlocks,
    OneHotCommitRowsPlan, OpeningProveBackendFor, OperationCtx, PreparedCrtNttProfile,
    ProveBackendFor, ProveFlowBackendFor, ProveStackFor, ProverComputeStack, RecursiveProveBackend,
    RecursiveWitnessCommitRowsPlan, RingSwitchComputeBackend, RingSwitchQuotientRowsPlan,
    RingSwitchRelationRows, RingSwitchRelationRowsPlan, RootCommitBackend, RootCommitSource,
    RootOpeningSource, RootPolyShape, RootProveBackend, RootProvePoly, RootTensorSource,
    SparseRingCommitRowsPlan, TensorBackendFor, TieredProveStacks, UniformProverStack,
    RECURSIVE_SUFFIX_RING_DIMENSIONS,
};
pub use protocol::fold_grind::ProverTranscriptGrind;
pub use protocol::fold_grind_observer::{FoldGrindObservation, FoldGrindObserverGuard};
pub use protocol::sumcheck::{AkitaStage1Prover, AkitaStage2Prover};
pub use protocol::{
    batched_prove, commit_next_w, prepare_batched_prove_inputs, prove, prove_root,
    prove_root_direct, prove_suffix, prove_terminal_root_fold_with_params,
    PreparedBatchedProveInputs, ProveLevelOutput, RecursiveSuffixOutcome, RingSwitchOutput,
    SuffixProverState,
};
pub use protocol::{RingRelationInstance, RingRelationProver, RingRelationWitness};
/// One PCS commitment and the polynomials it bundles, all opened at the batch's
/// shared opening point.
///
/// `polynomials` is the exact bundle committed by the prover commitment API;
/// `commitment` and `hint` are the corresponding outputs for that bundle.
#[derive(Debug, Clone)]
pub struct CommittedPolynomials<'a, P, C, H> {
    /// Polynomials addressable by claim `poly_idx` values at this point.
    pub polynomials: &'a [&'a P],
    /// Commitment for `polynomials`.
    pub commitment: &'a C,
    /// Prover-side hint for `commitment`.
    pub hint: H,
}

impl<'a, P, C, H> CommittedPolynomials<'a, P, C, H> {
    /// Number of polynomials addressable by opening-batch claims at this point.
    pub fn poly_count(&self) -> usize {
        self.polynomials.len()
    }
}

/// Batched prover input: one shared opening point plus commitment bundles.
///
/// Mirror of [`akita_types::VerifierClaims`]: `(shared_point, Vec<CommittedPolynomials>)`.
/// See `akita_types::proof::scheme` for the single-point batching contract.
pub type ProverClaims<'a, F, P, C, H> =
    (OpeningPoints<'a, F>, Vec<CommittedPolynomials<'a, P, C, H>>);

/// Prover-side output of the decompose + challenge-fold step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecomposeFoldWitness<F: FieldCore, const D: usize> {
    /// Folded witness rows in ring form.
    pub z_folded_rings: Vec<CyclotomicRing<F, D>>,
    /// Centered integer coefficients for each `z_folded_rings` row.
    pub centered_coeffs: Vec<[i32; D]>,
    /// Infinity norm of `centered_coeffs`.
    pub centered_inf_norm: u32,
}

/// Prover-side output of the inner Ajtai commit step.
pub struct CommitInnerWitness<F: FieldCore, const D: usize> {
    /// Recombined inner `A * s_i` rows, grouped by block.
    pub recomposed_inner_rows: Vec<Vec<CyclotomicRing<F, D>>>,
    /// Digit decompositions of `A * s_i` in flat column-major order plus
    /// explicit block boundaries.
    pub decomposed_inner_rows: FlatDigitBlocks<D>,
}
