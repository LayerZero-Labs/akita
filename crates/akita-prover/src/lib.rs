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
    padded_scalar_batch_num_vars, validate_scalar_point_matches_poly_arity, Commitment,
    DigitBlocks, OpeningBatchShape, OpeningGroupShape, OpeningPoints, PointVariableSelection,
    RingVec,
};

pub use api::{
    batched_commit, batched_commit_with_params, commit, commit_group, commit_setup_prefix,
    commit_with_params, prepare_batched_commit_inputs, prepare_commit_inputs, AkitaProverSetup,
    CommitmentProver, CommitmentWithHint, CommittedGroupHandle, CommittedGroupScheduleMeta,
    CommittedGroupWithHint,
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
    DenseCommitRowsPlan, DigitRowsComputeBackend, FlatBlockTable, FlatDigitBlocks,
    LevelProveStacks, OneHotCommitBlocks, OneHotCommitRowsPlan, OpeningCluster,
    OpeningProveBackendFor, OperationCtx, PreparedCrtNttProfile, ProveBackendFor,
    ProveFlowBackendFor, ProveStackFor, ProverComputeStack, RecursiveProveBackend,
    RecursiveWitnessCommitRowsPlan, RingSwitchCluster, RingSwitchComputeBackend,
    RingSwitchProveBackend, RingSwitchQuotientRowsPlan, RingSwitchRelationRows,
    RingSwitchRelationRowsPlan, RootCommitBackend, RootCommitSource, RootOpeningSource,
    RootPolyMeta, RootPolyShape, RootProveBackend, RootProvePoly, RootTensorSource,
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
/// One prover commitment group and the polynomials it bundles.
///
/// `polynomials` is the exact bundle committed by the prover commitment API;
/// `commitment` is the corresponding commitment output plus prover-side hint
/// for that bundle.
#[derive(Debug, Clone)]
pub struct ProverCommitmentGroup<'a, P, F: FieldCore> {
    /// Coordinates of [`ProverOpeningBatch::point`] used by every polynomial in this group.
    pub point_vars: PointVariableSelection,
    /// Polynomials addressable by claim `poly_idx` values at this point.
    pub polynomials: &'a [&'a P],
    /// Commitment output for `polynomials` (D-free flat storage plus hint).
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

    /// Shape-only opening batch with the padded domain selected from prover polynomials.
    pub fn to_opening_shape<PolyF>(&self) -> Result<OpeningBatchShape, AkitaError>
    where
        PolyF: FieldCore,
        P: RootPolyMeta<PolyF>,
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

    /// Commitments in commitment-group order (D-free flat storage).
    pub fn commitments(&self) -> Vec<&Commitment<CommitF>> {
        self.groups
            .iter()
            .map(|group| &group.commitment.0)
            .collect()
    }

    /// Absorb the normalized batch shape, commitments, and shared point.
    ///
    /// Commitments are absorbed through the D-free flat coefficient encoder
    /// keyed on the schedule-derived `ring_dim`; this is byte-identical to the
    /// former typed `RingCommitment::append_to_transcript` path for the matching
    /// dimension (S2 byte-identity test).
    pub fn append_to_transcript<T>(
        &self,
        ring_dim: usize,
        transcript: &mut T,
    ) -> Result<(), AkitaError>
    where
        CommitF: CanonicalField,
        PointF: ExtField<CommitF>,
        P: RootPolyMeta<CommitF>,
        T: Transcript<CommitF>,
    {
        let shape = self.to_opening_shape::<CommitF>()?;
        let commitments = self.commitments();
        shape.append_to_transcript::<CommitF, T>(transcript)?;
        for commitment in commitments {
            commitment.append_to_transcript(
                akita_transcript::labels::ABSORB_COMMITMENT,
                ring_dim,
                transcript,
            )?;
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
    pub fn single_group(&self) -> Option<&ProverCommitmentGroup<'a, P, CommitF>> {
        self.groups.first().filter(|_| self.groups.len() == 1)
    }

    /// Consume the batch and return its only group when the current single-group path applies.
    pub fn into_single_group(self) -> Option<ProverCommitmentGroup<'a, P, CommitF>> {
        let Self { mut groups, .. } = self;
        if groups.len() != 1 {
            return None;
        }
        groups.pop()
    }

    /// Borrow the current single-group fold batch's commitment rows as flat proof storage.
    pub(crate) fn single_fold_commitment(&self) -> Result<RingVec<CommitF>, AkitaError> {
        let group = self.single_group().ok_or_else(|| {
            AkitaError::InvalidInput("multi-group fold proving is not supported yet".to_string())
        })?;
        Ok(group.commitment.0.rows().clone())
    }

    /// Preserve this batch's grouping metadata while replacing its flat polynomial stream.
    pub(crate) fn regroup_polynomial_refs<'b, Q>(
        self,
        polynomials: &'b [&'b Q],
    ) -> Result<ProverOpeningBatch<'b, PointF, Q, CommitF>, AkitaError>
    where
        'a: 'b,
    {
        let ProverOpeningBatch { point, groups } = self;
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
            regrouped.push(ProverCommitmentGroup {
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
        Ok(ProverOpeningBatch {
            point,
            groups: regrouped,
        })
    }

    /// Build the single-claim batch used by recursive suffix fold levels.
    pub(crate) fn new_suffix(
        opening_point: &[PointF],
        recursive_num_vars: usize,
        polynomials: &'a [&'a P],
        commitment: CommitmentWithHint<CommitF>,
    ) -> Result<Self, AkitaError>
    where
        PointF: FieldCore,
    {
        let opening_batch = OpeningBatchShape::new(recursive_num_vars, 1)?;
        let point_vars = opening_batch
            .groups()
            .first()
            .ok_or_else(|| {
                AkitaError::InvalidInput("recursive opening batch requires one group".to_string())
            })?
            .point_vars
            .clone();
        let mut padded_point = opening_point.to_vec();
        padded_point.resize(recursive_num_vars, PointF::zero());
        Ok(Self {
            point: padded_point.into(),
            groups: vec![ProverCommitmentGroup {
                point_vars,
                polynomials,
                commitment,
            }],
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
        decomposed_inner_rows: crate::compute::FlatDigitBlocks<D>,
    ) -> Self {
        Self {
            recomposed_inner_rows: recomposed_inner_rows
                .into_iter()
                .map(|block| RingVec::from_ring_elems(&block))
                .collect(),
            decomposed_inner_rows: decomposed_inner_rows.into_digit_blocks(),
            ring_dim: D,
        }
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

    /// Rebuild typed digit planes after [`Self::ensure_ring_dim`].
    pub fn decomposed_inner_rows_trusted<const D: usize>(
        &self,
    ) -> Result<crate::compute::FlatDigitBlocks<D>, AkitaError> {
        self.ensure_ring_dim::<D>()?;
        crate::compute::FlatDigitBlocks::from_digit_blocks(&self.decomposed_inner_rows)
    }
}
