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
use akita_field::parallel::*;
use akita_field::{AkitaError, FieldCore};
use akita_types::RingVec;

pub use api::{
    commit, commit_setup_prefix, prepare_commit_inputs, AkitaProverSetup, CommitOutput,
    GroupContext, GroupParameterSource, PreparedGroupProveOps, PreparedProverGroup,
    PriorGroupContext,
};

pub use backend::{
    tensor_pack_recursive_witness, DensePoly, MultilinearPolynomial, OneHotIndex, OneHotPoly,
    RecursiveFoldSource, RecursiveWitnessFlat, RootTensorProjectionPoly, SparseRingBlockEntry,
    SparseRingPoly, SuffixWitnessBatchView, SuffixWitnessView,
};
pub use compute::{
    planned_ntt_cache_metrics, prewarm_ntt_requirements, BatchDecomposeFoldOutcome,
    CommitBackendFor, CommitCluster, ComputeBackendSetup, CpuBackend, CpuPreparedSetup,
    CyclicRowsComputeBackend, DigitRowsComputeBackend, LevelProveStacks, NttCacheOwnerId,
    NttExecutionRequirements, NttOperationCluster, OpeningCluster, OpeningProveBackendFor,
    OperationCtx, PlannedNttCacheOwnerMetric, PreparedCrtNttProfile, PreparedNttCacheMetric,
    ProveBackendFor, ProveFlowBackendFor, ProveStackFor, ProverComputeStack, RecursiveProveBackend,
    ReleaseRootNttAfterFold, RingSwitchCluster, RingSwitchProveBackend, RingSwitchRelationRows,
    RootCommitBackend, RootCommitSource, RootOpeningSource, RootPolyMeta, RootPolyShape,
    RootProveBackend, RootProvePoly, RootTensorSource, RoutedNttRequirement,
    RuntimeCommitBackendFor, RuntimeCommitSource, RuntimeOpeningProveBackendFor,
    RuntimeProveBackendFor, RuntimeRecursiveWitnessProveBackend, RuntimeRingSwitchProveBackend,
    RuntimeRootCommitBackend, RuntimeRootCommitPoly, RuntimeRootProvePoly, RuntimeTensorBackendFor,
    SuffixOpeningProveBackend, SuffixTensorProveBackend, TensorBackendFor, TensorCluster,
    TieredProveStacks, UniformProverStack,
};
pub use protocol::fold_grind::ProverTranscriptGrind;
pub use protocol::sumcheck::{DigitRangeProver, RelationRangeImageProver};
pub use protocol::{
    batched_prove, build_relation_weight_events, commit_terminal_w, commit_w, prove, prove_suffix,
    ProveLevelOutput, RecursiveSuffixOutcome, RelationSetupSource, RelationWeightContribution,
    RelationWeightEvent, RelationWeightEventInputs, RelationWeightEvents,
    RelationWeightFactorization, RingSwitchOutput, SuffixProverState,
};
pub use protocol::{RingRelationInstance, RingRelationProver, RingRelationWitness};
pub use types::{ProverOpeningData, SelectedProverOpeningData};

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
    /// Smallest signed centered coefficient.
    centered_min: i32,
    /// Largest signed centered coefficient.
    centered_max: i32,
    /// Ring dimension (field coefficients per ring element), fixed at construction.
    ring_dim: usize,
}

impl<F: FieldCore> DecomposeFoldWitness<F> {
    /// Construct from owned coefficient rows at a kernel boundary.
    pub fn from_coefficient_parts<const D: usize>(
        z_folded_coeffs: Vec<[F; D]>,
        centered_coeffs: Vec<[i32; D]>,
    ) -> Self {
        debug_assert_eq!(z_folded_coeffs.len(), centered_coeffs.len());
        let (centered_min, centered_max) = centered_coefficient_bounds(&centered_coeffs);
        Self {
            z_folded_rings: RingVec::from_coefficient_rows(z_folded_coeffs),
            centered_coeffs_flat: centered_coeffs.into_flattened(),
            centered_min,
            centered_max,
            ring_dim: D,
        }
    }

    /// Construct from typed ring rows at a kernel boundary.
    pub fn from_parts<const D: usize>(
        z_folded_rings: Vec<CyclotomicRing<F, D>>,
        centered_coeffs: Vec<[i32; D]>,
    ) -> Self {
        debug_assert_eq!(z_folded_rings.len(), centered_coeffs.len());
        let (centered_min, centered_max) = centered_coefficient_bounds(&centered_coeffs);
        Self {
            z_folded_rings: RingVec::from_ring_elems(&z_folded_rings),
            centered_coeffs_flat: centered_coeffs.into_flattened(),
            centered_min,
            centered_max,
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

    /// Infinity norm derived from the centered coefficient buffer.
    pub fn centered_inf_norm(&self) -> u32 {
        self.centered_min
            .unsigned_abs()
            .max(self.centered_max.unsigned_abs())
    }

    /// Signed extrema derived from the centered coefficient buffer.
    pub fn centered_signed_extrema(&self) -> (i32, i32) {
        (self.centered_min, self.centered_max)
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

fn centered_coefficient_bounds<const D: usize>(rows: &[[i32; D]]) -> (i32, i32) {
    if rows.is_empty() || D == 0 {
        return (0, 0);
    }
    cfg_fold_reduce!(
        rows,
        || (i32::MAX, i32::MIN),
        |(mut min, mut max), row| {
            for &coefficient in row {
                min = min.min(coefficient);
                max = max.max(coefficient);
            }
            (min, max)
        },
        |(left_min, left_max), (right_min, right_max)| {
            (left_min.min(right_min), left_max.max(right_max))
        }
    )
}

/// Prover-side output of the inner Ajtai commit step.
///
/// Ring dimension is stored by the single flat A-ring buffer. Public commit
/// parameters own the source-block and row boundaries.
pub struct CommitInnerWitness<F: FieldCore> {
    /// Recombined inner `A * s_i` rows in `[block][A row][coefficient]` order.
    pub inner_rows: RingVec<F>,
}

impl<F: FieldCore> CommitInnerWitness<F> {
    /// Construct from typed kernel output at a commit boundary.
    pub fn from_rows<const D: usize>(
        recomposed_inner_rows: Vec<Vec<CyclotomicRing<F, D>>>,
    ) -> Self {
        let coefficient_count = recomposed_inner_rows
            .iter()
            .map(Vec::len)
            .sum::<usize>()
            .checked_mul(D)
            .expect("trusted inner commitment output length must fit usize");
        let mut coefficients = Vec::with_capacity(coefficient_count);
        for block in recomposed_inner_rows {
            for row in block {
                coefficients.extend_from_slice(row.coefficients());
            }
        }
        Self {
            inner_rows: RingVec::from_coeffs_with_ring_dim(coefficients, D)
                .expect("typed inner commitment rows have valid ring storage"),
        }
    }

    /// Stored ring dimension (coefficients per ring element).
    pub fn ring_dim(&self) -> usize {
        self.inner_rows.ring_dim()
    }

    /// # Errors
    ///
    /// Returns an error if the requested ring dimension does not match storage.
    pub fn ensure_ring_dim<const D: usize>(&self) -> Result<(), AkitaError> {
        if self.ring_dim() != D {
            return Err(AkitaError::InvalidInput(format!(
                "commit inner witness ring_d={} does not match requested D={D}",
                self.ring_dim()
            )));
        }
        if !self.inner_rows.can_decode_vec(D) {
            return Err(AkitaError::InvalidSize {
                expected: D,
                actual: self.inner_rows.coeff_len(),
            });
        }
        Ok(())
    }

    /// Borrow one source block using the row count from public parameters.
    pub fn block_rows<const D: usize>(
        &self,
        block: usize,
        rows_per_block: usize,
    ) -> Result<&[CyclotomicRing<F, D>], AkitaError> {
        self.ensure_ring_dim::<D>()?;
        let start = block
            .checked_mul(rows_per_block)
            .ok_or_else(|| AkitaError::InvalidSetup("inner row offset overflow".into()))?;
        let end = start
            .checked_add(rows_per_block)
            .ok_or_else(|| AkitaError::InvalidSetup("inner row end overflow".into()))?;
        self.inner_rows
            .as_ring_slice_trusted::<D>()
            .get(start..end)
            .ok_or_else(|| AkitaError::InvalidInput("inner row block is out of range".into()))
    }

    /// Consume the inner commitment output as the persistent A-native rows.
    pub fn into_inner_rows(self) -> RingVec<F> {
        self.inner_rows
    }
}
