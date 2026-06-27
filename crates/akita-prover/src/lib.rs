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
use akita_field::{AkitaError, CanonicalField, ExtField, FieldCore};
use akita_transcript::Transcript;
use akita_types::{
    padded_scalar_batch_num_vars, validate_scalar_point_matches_poly_arity, AppendToTranscript,
    FlatDigitBlocks, FlatRingVec, OpeningBatchShape, OpeningGroupShape, OpeningPoints,
    PointVariableSelection, RingBuf, RingCommitment,
};

pub use api::{
    batched_commit, batched_commit_with_params, commit, commit_group, commit_setup_prefix,
    commit_with_params, prepare_batched_commit_inputs, prepare_commit_inputs, AkitaProverSetup,
    CommitmentWithHint, CommittedGroupHandle, CommittedGroupScheduleMeta, CommittedGroupWithHint,
    TypedCommitmentProver,
};

pub use backend::{
    tensor_pack_recursive_witness, DensePoly, MultiChunkEntry, MultilinearPolynomial, OneHotIndex,
    OneHotPoly, RecursiveWitnessFlat, RootTensorProjectionPoly, SingleChunkEntry,
    SparseRingBlockEntry, SparseRingPoly, SuffixWitnessBatchView, SuffixWitnessView,
};
pub use compute::{
    BatchDecomposeFoldOutcome, CommitBackendFor, CommitCluster, CommitmentComputeBackend,
    ComputeBackendSetup, CpuBackend, CpuPreparedSetup, CyclicRowsComputeBackend, DenseCommitInput,
    DenseCommitRowsPlan, DigitRowsComputeBackend, FlatBlockTable, LevelProveStacks,
    NttSlotCacheAny, OneHotCommitBlocks, OneHotCommitRowsPlan, OpeningCluster,
    OpeningProveBackendFor, OperationCtx, PreparedCrtNttProfile, ProveBackendFor,
    ProveFlowBackendFor, ProveStackFor, ProverComputeStack, RecursiveWitnessCommitRowsPlan,
    RecursiveWitnessProveFlowBackend, RingSwitchCluster, RingSwitchComputeBackend,
    RingSwitchProveBackend, RingSwitchQuotientRowsPlan, RingSwitchRelationRows,
    RingSwitchRelationRowsPlan, RootCommitBackend, RootCommitSource, RootOpeningSource,
    RootPolyShape, RootProveBackend, RootProveFlowBackend, RootProvePoly, RootTensorSource,
    SparseRingCommitRowsPlan, SuffixOpeningProveBackend, SuffixRingSwitchProveBackend,
    SuffixTensorProveBackend, TensorBackendFor, TensorCluster, TieredProveStacks,
    UniformProverStack,
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
/// One prover commitment group and the polynomials it bundles.
///
/// `polynomials` is the exact bundle committed by the prover commitment API;
/// `commitment` is the corresponding commitment output plus prover-side hint
/// for that bundle.
#[derive(Debug, Clone)]
#[doc(hidden)]
pub struct TypedProverCommitmentGroup<'a, P, F: FieldCore, const D: usize> {
    /// Coordinates of [`ProverOpeningBatch::point`] used by every polynomial in this group.
    pub point_vars: PointVariableSelection,
    /// Polynomials addressable by claim `poly_idx` values at this point.
    pub polynomials: &'a [&'a P],
    /// Commitment output for `polynomials`.
    pub commitment: api::commitment::TypedCommitmentWithHint<F, D>,
}

/// Batched prover input: one shared opening point plus prover commitment groups.
#[derive(Debug, Clone)]
#[doc(hidden)]
pub struct TypedProverOpeningBatch<'a, PointF: Clone, P, CommitF: FieldCore, const D: usize> {
    /// Padded/shared opening point.
    pub point: OpeningPoints<'a, PointF>,
    /// Commitment groups in transcript order.
    pub groups: Vec<TypedProverCommitmentGroup<'a, P, CommitF, D>>,
}

/// Prover commitment group and the polynomials it bundles.
///
/// Internal proving code converts this to [`TypedProverCommitmentGroup`] after the
/// A scheme preset fixes the root ring dimension only inside typed dispatch code.
#[derive(Debug, Clone)]
pub struct ProverCommitmentGroup<'a, P, F: FieldCore> {
    /// Coordinates of [`ProverOpeningBatch::point`] used by every polynomial in this group.
    pub point_vars: PointVariableSelection,
    /// Polynomials addressable by claim `poly_idx` values at this point.
    pub polynomials: &'a [&'a P],
    /// Commitment output for `polynomials`.
    pub commitment: CommitmentWithHint<F>,
}

/// Batched prover input: one shared opening point plus prover commitment groups.
#[derive(Debug, Clone)]
pub struct ProverOpeningBatch<'a, PointF: Clone, P, CommitF: FieldCore> {
    /// Padded/shared opening point.
    pub point: OpeningPoints<'a, PointF>,
    /// Commitment groups in transcript order.
    pub groups: Vec<ProverCommitmentGroup<'a, P, CommitF>>,
}

impl<'a, P, F: FieldCore> ProverCommitmentGroup<'a, P, F> {
    /// Number of polynomials addressable by opening-batch claims at this point.
    pub fn poly_count(&self) -> usize {
        self.polynomials.len()
    }
}

impl<'a, PointF: Clone, P, CommitF: FieldCore> ProverOpeningBatch<'a, PointF, P, CommitF> {
    /// Shared opening point.
    pub fn point(&self) -> &[PointF] {
        self.point.as_ref()
    }

    /// Commitment groups in transcript order.
    pub fn groups(&self) -> &[ProverCommitmentGroup<'a, P, CommitF>] {
        &self.groups
    }

    /// Number of polynomials in each commitment group.
    pub fn group_sizes(&self) -> Vec<usize> {
        self.groups
            .iter()
            .map(ProverCommitmentGroup::poly_count)
            .collect()
    }

    /// Convert this batch into the typed root batch consumed by kernels.
    ///
    /// # Errors
    ///
    /// Returns an error if any commitment cannot be decoded at `D`.
    pub fn into_typed<const D: usize>(
        self,
    ) -> Result<TypedProverOpeningBatch<'a, PointF, P, CommitF, D>, AkitaError> {
        let ProverOpeningBatch { point, groups } = self;
        let typed_groups = groups
            .into_iter()
            .map(|group| {
                let (commitment, hint) = group.commitment;
                Ok(TypedProverCommitmentGroup {
                    point_vars: group.point_vars,
                    polynomials: group.polynomials,
                    commitment: (commitment.try_to_ring_commitment::<D>()?, hint),
                })
            })
            .collect::<Result<Vec<_>, AkitaError>>()?;
        Ok(TypedProverOpeningBatch {
            point,
            groups: typed_groups,
        })
    }
}

impl<'a, P, F: FieldCore, const D: usize> TypedProverCommitmentGroup<'a, P, F, D> {
    /// Number of polynomials addressable by opening-batch claims at this point.
    pub fn poly_count(&self) -> usize {
        self.polynomials.len()
    }
}

impl<'a, PointF: Clone, P, CommitF: FieldCore, const D: usize>
    TypedProverOpeningBatch<'a, PointF, P, CommitF, D>
{
    /// Shared opening point.
    pub fn point(&self) -> &[PointF] {
        self.point.as_ref()
    }

    /// Commitment groups in transcript order.
    pub fn groups(&self) -> &[TypedProverCommitmentGroup<'a, P, CommitF, D>] {
        &self.groups
    }

    /// Number of polynomials in each commitment group.
    pub fn group_sizes(&self) -> Vec<usize> {
        self.groups
            .iter()
            .map(TypedProverCommitmentGroup::poly_count)
            .collect()
    }

    /// Shape-only opening batch with the padded domain selected from prover polynomials.
    pub fn to_opening_shape<PolyF>(&self) -> Result<OpeningBatchShape, AkitaError>
    where
        PolyF: FieldCore,
        P: RootPolyShape<PolyF, D>,
    {
        let padded_num_vars = padded_scalar_batch_num_vars(
            self.groups()
                .iter()
                .flat_map(|group| group.polynomials.iter().map(|poly| poly.num_vars())),
        )?;
        validate_scalar_point_matches_poly_arity(self.point().len(), padded_num_vars)?;
        OpeningBatchShape::from_groups(
            padded_num_vars,
            self.groups
                .iter()
                .map(|group| OpeningGroupShape {
                    point_vars: group.point_vars.clone(),
                    num_polynomials: group.poly_count(),
                })
                .collect(),
        )
    }

    /// Polynomials flattened in canonical claim order.
    pub fn flat_polys(&self) -> Vec<&'a P> {
        self.groups
            .iter()
            .flat_map(|group| group.polynomials.iter().copied())
            .collect()
    }

    /// Commitments in commitment-group order.
    pub fn commitments(&self) -> Vec<&RingCommitment<CommitF, D>> {
        self.groups
            .iter()
            .map(|group| &group.commitment.0)
            .collect()
    }

    /// Absorb the normalized batch shape, commitments, and shared point.
    pub fn append_to_transcript<T>(&self, transcript: &mut T) -> Result<(), AkitaError>
    where
        CommitF: CanonicalField,
        PointF: ExtField<CommitF>,
        P: RootPolyShape<CommitF, D>,
        T: Transcript<CommitF>,
    {
        let shape = self.to_opening_shape::<CommitF>()?;
        let commitments = self.commitments();
        shape.append_to_transcript::<CommitF, T>(transcript)?;
        for commitment in commitments {
            commitment
                .append_to_transcript(akita_transcript::labels::ABSORB_COMMITMENT, transcript);
        }
        for coord in self.point() {
            akita_transcript::append_ext_field::<CommitF, PointF, T>(
                transcript,
                akita_transcript::labels::ABSORB_EVALUATION_CLAIMS,
                coord,
            );
        }
        Ok(())
    }

    /// Return the only group when the current implementation's single-group path applies.
    pub fn single_group(&self) -> Option<&TypedProverCommitmentGroup<'a, P, CommitF, D>> {
        self.groups.first().filter(|_| self.groups.len() == 1)
    }

    /// Consume the batch and return its only group when the current single-group path applies.
    pub fn into_single_group(self) -> Option<TypedProverCommitmentGroup<'a, P, CommitF, D>> {
        let Self { mut groups, .. } = self;
        if groups.len() != 1 {
            return None;
        }
        groups.pop()
    }

    /// Borrow the current single-group fold batch's commitment rows as flat proof storage.
    pub(crate) fn single_fold_commitment(&self) -> Result<FlatRingVec<CommitF>, AkitaError> {
        let group = self.single_group().ok_or_else(|| {
            AkitaError::InvalidInput("multi-group fold proving is not supported yet".to_string())
        })?;
        Ok(FlatRingVec::from_ring_elems(&group.commitment.0.u))
    }

    /// Preserve this batch's grouping metadata while replacing its flat polynomial stream.
    pub(crate) fn regroup_polynomial_refs<'b, Q>(
        self,
        polynomials: &'b [&'b Q],
    ) -> Result<TypedProverOpeningBatch<'b, PointF, Q, CommitF, D>, AkitaError>
    where
        'a: 'b,
    {
        let TypedProverOpeningBatch { point, groups } = self;
        let mut input_offset = 0usize;
        let mut regrouped = Vec::with_capacity(groups.len());
        for group in groups {
            let group_len = group.polynomials.len();
            let input_end = input_offset.checked_add(group_len).ok_or_else(|| {
                AkitaError::InvalidInput("fold input group offset overflow".to_string())
            })?;
            let replacement_polynomials =
                polynomials.get(input_offset..input_end).ok_or_else(|| {
                    AkitaError::InvalidInput("fold input group shape mismatch".to_string())
                })?;
            regrouped.push(TypedProverCommitmentGroup {
                point_vars: group.point_vars,
                polynomials: replacement_polynomials,
                commitment: group.commitment,
            });
            input_offset = input_end;
        }
        if input_offset != polynomials.len() {
            return Err(AkitaError::InvalidInput(
                "fold input group coverage mismatch".to_string(),
            ));
        }
        Ok(TypedProverOpeningBatch {
            point,
            groups: regrouped,
        })
    }
}

/// Prover-side output of the decompose + challenge-fold step.
///
/// Ring dimension is stored at runtime; hot paths inside `dispatch_ring_dim`
/// closures borrow typed ring rows via [`Self::z_folded_rings_trusted`] and
/// [`Self::centered_coeffs_trusted`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecomposeFoldWitness<F: FieldCore> {
    /// Folded witness rows in flat ring storage.
    pub z_folded_rings: RingBuf<F>,
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
            z_folded_rings: RingBuf::from_ring_elems(&z_folded_rings),
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
pub struct CommitInnerWitness<F: FieldCore, const D: usize> {
    /// Recombined inner `A * s_i` rows, grouped by block.
    pub recomposed_inner_rows: Vec<Vec<CyclotomicRing<F, D>>>,
    /// Digit decompositions of `A * s_i` in flat column-major order plus
    /// explicit block boundaries.
    pub decomposed_inner_rows: FlatDigitBlocks,
}
