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
    OneHotPoly, RecursiveCommitmentHintCache, RecursiveWitnessFlat, RecursiveWitnessView,
    RootTensorProjectionPoly, SingleChunkEntry, SparseRingBlockEntry, SparseRingPoly,
};
pub use compute::{
    ComputeBackendSetup, CpuBackend, CpuPreparedSetup, DenseCommitInput, DenseCommitRowsPlan,
    FlatBlockTable, OneHotCommitBlocks, OneHotCommitRowsPlan, PreparedCrtNttProfile,
    ProverComputeStack, RecursiveWitnessCommitRowsPlan, RingSwitchQuotientRowsPlan,
    RingSwitchRelationRows, RingSwitchRelationRowsPlan, RootCommitPolys, RootProveBackend,
    RootProveFlowBackend, RootProvePoly, RootTensorSource, SparseRingCommitRowsPlan,
    TensorProjectionBatchKernel, UniformProverStack,
};
pub use protocol::sumcheck::{AkitaStage1Prover, AkitaStage2Prover};
pub use protocol::{
    build_terminal_root_batched_proof, commit_next_w, prepare_batched_prove_inputs, prove_batched,
    prove_folded_batched, prove_recursive_suffix, prove_root_direct,
    prove_root_fold_from_ring_relation, prove_root_fold_with_params,
    prove_terminal_root_fold_from_ring_relation, prove_terminal_root_fold_with_params,
    PreparedBatchedProveInputs, ProveLevelOutput, RecursiveProverState, RecursiveSuffixOutcome,
    RingSwitchOutput, RootLevelProverOutput, RootLevelRawOutput,
};
pub use protocol::{RingRelationInstance, RingRelationProver, RingRelationWitness};
/// One commitment plus the polynomials it bundles, opened at one point.
///
/// `polynomials` is the exact bundle committed together by the prover
/// commitment API; `commitment` and `hint` are the corresponding outputs for
/// that bundle. Each opening point cites exactly one `CommittedPolynomials`.
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
    /// Number of polynomials addressable by incidence claims at this point.
    pub fn poly_count(&self) -> usize {
        self.polynomials.len()
    }
}

/// Batched prover input: one commitment plus its polynomials per point.
pub type ProverClaims<'a, F, P, C, H> =
    Vec<(OpeningPoints<'a, F>, CommittedPolynomials<'a, P, C, H>)>;

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
