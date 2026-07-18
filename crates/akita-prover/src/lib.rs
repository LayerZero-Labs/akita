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
use akita_field::{AkitaError, FieldCore};
use akita_types::{DigitBlocks, RingVec};

pub use api::{
    batched_commit, batched_commit_with_params, commit, commit_final_group, commit_group,
    commit_setup_prefix, commit_with_params, prepare_batched_commit_inputs, prepare_commit_inputs,
    AkitaProverSetup, CommitmentProver, CommitmentWithHint, CommittedGroupWithHint,
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
    RootOpeningSource, RootPolyMeta, RootPolyShape, RootProveBackend, RootProvePoly,
    RootTensorSource, RuntimeCommitBackendFor, RuntimeOpeningProveBackendFor,
    RuntimeProveBackendFor, RuntimeRecursiveWitnessProveBackend, RuntimeRingSwitchProveBackend,
    RuntimeRootCommitBackend, RuntimeRootCommitPoly, RuntimeRootProvePoly, RuntimeTensorBackendFor,
    SparseRingCommitRowsPlan, SuffixOpeningProveBackend, SuffixTensorProveBackend,
    TensorBackendFor, TensorCluster, TieredProveStacks, UniformProverStack,
    RECURSIVE_SUFFIX_RING_DIMENSIONS,
};
pub use protocol::fold_grind::ProverTranscriptGrind;
pub use protocol::sumcheck::{AkitaStage1Prover, AkitaStage2Prover};
pub use protocol::{
    batched_prove, commit_w, prove, prove_root, prove_suffix, ProveLevelOutput,
    RecursiveSuffixOutcome, RingSwitchOutput, SuffixProverState,
};
pub use protocol::{RingRelationInstance, RingRelationProver, RingRelationWitness};
pub use types::ProverOpeningData;

/// Prover-side output of the decompose + challenge-fold step.
///
/// Ring dimension is stored at runtime; hot paths inside `dispatch_ring_dim`
/// closures borrow typed ring rows via [`Self::z_folded_rings_trusted`] and
/// [`Self::centered_coeffs_trusted`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecomposeFoldWitness<F: FieldCore> {
    /// Folded witness rows in flat ring storage.
    pub z_folded_rings: RingVec<F>,
    /// Centered integer coefficients for each [`z_folded_rings`] row, stored row-major flat.
    ///
    /// Hot paths borrow typed rows via [`Self::centered_coeffs_trusted`].
    centered_coeffs_flat: Vec<i32>,
    /// Infinity norm of the flat centered coefficient storage above.
    pub centered_inf_norm: u32,
    /// Ring dimension (field coefficients per ring element), fixed at construction.
    ring_dim: usize,
}

impl<F: FieldCore> DecomposeFoldWitness<F> {
    /// Construct from typed ring rows at a kernel boundary.
    pub fn from_parts<const D: usize>(
        z_folded_rings: Vec<CyclotomicRing<F, D>>,
        centered_coeffs: Vec<[i32; D]>,
        centered_inf_norm: u32,
    ) -> Self {
        debug_assert_eq!(z_folded_rings.len(), centered_coeffs.len());
        Self {
            z_folded_rings: RingVec::from_ring_elems(&z_folded_rings),
            centered_coeffs_flat: centered_coeffs
                .iter()
                .flat_map(|row| row.iter().copied())
                .collect(),
            centered_inf_norm,
            ring_dim: D,
        }
    }

    /// Stored ring dimension (coefficients per ring element).
    pub fn ring_dim(&self) -> usize {
        self.ring_dim
    }

    /// Number of folded witness rows.
    pub fn row_count(&self) -> usize {
        self.centered_coeffs_flat
            .len()
            .checked_div(self.ring_dim)
            .unwrap_or(0)
    }

    /// # Errors
    ///
    /// Returns an error if the requested ring dimension does not match storage.
    pub fn ensure_ring_dim<const D: usize>(&self) -> Result<(), AkitaError> {
        if self.ring_dim != D {
            return Err(AkitaError::InvalidInput(format!(
                "decompose fold witness ring_d={} does not match requested D={D}",
                self.ring_dim
            )));
        }
        if !self.centered_coeffs_flat.len().is_multiple_of(D) {
            return Err(AkitaError::InvalidSize {
                expected: D,
                actual: self.centered_coeffs_flat.len(),
            });
        }
        if !self.z_folded_rings.can_decode_vec(D) {
            return Err(AkitaError::InvalidSize {
                expected: D,
                actual: self.z_folded_rings.coeff_len(),
            });
        }
        let ring_count = self.z_folded_rings.count();
        let row_count = self.centered_coeffs_flat.len() / D;
        if ring_count != row_count {
            return Err(AkitaError::InvalidInput(
                "decompose fold witness ring row count mismatch".to_string(),
            ));
        }
        Ok(())
    }

    /// Borrow folded ring rows after [`Self::ensure_ring_dim`].
    pub fn z_folded_rings_trusted<const D: usize>(&self) -> &[CyclotomicRing<F, D>] {
        debug_assert_eq!(self.ring_dim, D);
        self.z_folded_rings.as_ring_slice_trusted::<D>()
    }

    /// Borrow the centered coefficients as row-major flat storage (D-free).
    pub fn centered_coeffs_flat(&self) -> &[i32] {
        &self.centered_coeffs_flat
    }

    /// Borrow centered coefficient rows after [`Self::ensure_ring_dim`].
    pub fn centered_coeffs_trusted<const D: usize>(&self) -> &[[i32; D]] {
        debug_assert_eq!(self.ring_dim, D);
        let (chunks, rem) = self.centered_coeffs_flat.as_chunks::<D>();
        debug_assert!(rem.is_empty());
        chunks
    }

    /// Owned copy of centered coefficient rows after [`Self::ensure_ring_dim`].
    pub fn centered_coeffs_owned<const D: usize>(&self) -> Vec<[i32; D]> {
        self.centered_coeffs_trusted::<D>().to_vec()
    }
}

/// Prover-side output of the inner Ajtai commit step.
///
/// Ring dimension is stored at runtime; hot paths inside `dispatch_ring_dim`
/// closures borrow typed ring rows via [`Self::recomposed_block_trusted`] and
/// typed digit planes via [`Self::decomposed_inner_rows_trusted`].
pub struct CommitInnerWitness<F: FieldCore> {
    /// Recombined inner `A * s_i` rows per block, each block in flat ring storage.
    pub recomposed_inner_rows: Vec<RingVec<F>>,
    /// Digit decompositions of `A * s_i` in D-free protocol storage.
    pub decomposed_inner_rows: DigitBlocks,
    /// Ring dimension (coefficients per ring element), fixed at construction.
    ring_dim: usize,
}

impl<F: FieldCore> CommitInnerWitness<F> {
    /// Construct from typed kernel output at a commit boundary.
    pub fn from_parts<const D: usize>(
        recomposed_inner_rows: Vec<Vec<CyclotomicRing<F, D>>>,
        decomposed_inner_rows: DigitBlocks,
    ) -> Result<Self, AkitaError> {
        decomposed_inner_rows.ensure_stride::<D>()?;
        Ok(Self {
            recomposed_inner_rows: recomposed_inner_rows
                .into_iter()
                .map(|block| RingVec::from_ring_elems(&block))
                .collect(),
            decomposed_inner_rows,
            ring_dim: D,
        })
    }

    /// Stored ring dimension (coefficients per ring element).
    pub fn ring_dim(&self) -> usize {
        self.ring_dim
    }

    /// Number of inner commitment blocks.
    pub fn block_count(&self) -> usize {
        self.recomposed_inner_rows.len()
    }

    /// # Errors
    ///
    /// Returns an error if the requested ring dimension does not match storage.
    pub fn ensure_ring_dim<const D: usize>(&self) -> Result<(), AkitaError> {
        if self.ring_dim != D {
            return Err(AkitaError::InvalidInput(format!(
                "commit inner witness ring_d={} does not match requested D={D}",
                self.ring_dim
            )));
        }
        if self.decomposed_inner_rows.digit_stride() != D {
            return Err(AkitaError::InvalidSize {
                expected: D,
                actual: self.decomposed_inner_rows.digit_stride(),
            });
        }
        for block in &self.recomposed_inner_rows {
            if !block.can_decode_vec(D) {
                return Err(AkitaError::InvalidSize {
                    expected: D,
                    actual: block.coeff_len(),
                });
            }
        }
        Ok(())
    }

    /// Borrow recomposed rows for one block after [`Self::ensure_ring_dim`].
    pub fn recomposed_block_trusted<const D: usize>(
        &self,
        block: usize,
    ) -> Result<&[CyclotomicRing<F, D>], AkitaError> {
        self.ensure_ring_dim::<D>()?;
        self.recomposed_inner_rows
            .get(block)
            .ok_or_else(|| {
                AkitaError::InvalidInput(format!(
                    "commit inner witness block index {block} out of range"
                ))
            })
            .map(|rows| rows.as_ring_slice_trusted::<D>())
    }

    /// Borrow decomposed digit planes after [`Self::ensure_ring_dim`].
    pub fn decomposed_inner_rows_trusted<const D: usize>(
        &self,
    ) -> Result<&DigitBlocks, AkitaError> {
        self.ensure_ring_dim::<D>()?;
        Ok(&self.decomposed_inner_rows)
    }
}
