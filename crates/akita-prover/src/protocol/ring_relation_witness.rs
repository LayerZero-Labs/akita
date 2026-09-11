//! Prover-only secret witness for the negacyclic-ring relation.

use crate::protocol::ring_relation::CompressionWitnessMaterialization;
use crate::DecomposeFoldWitness;
use akita_algebra::CyclotomicRing;
use akita_error::AkitaError;
use akita_types::{
    AkitaCommitmentHint, CoefficientPackingFoldProduct, CommitmentRingDims, DigitBlocks,
    OpeningFamily, RingRole, RingVec,
};
use jolt_field::Field;

/// Method-typed folded opening retained for quotient construction.
pub(crate) type GroupFoldedOpening<F> = OpeningFamily<RingVec<F>, CoefficientPackingFoldProduct<F>>;

/// One distributed fold window's centered coefficients and signed extrema.
pub(crate) struct CenteredFoldChunk {
    coefficients: Vec<i32>,
    min: i32,
    max: i32,
}

impl CenteredFoldChunk {
    /// Retain one chunk's centered coefficients and the extrema computed by
    /// its canonical fold-witness constructor.
    pub(crate) fn from_witness<F: Field>(witness: &DecomposeFoldWitness<F>) -> Self {
        let (min, max) = witness.centered_signed_extrema();
        Self {
            coefficients: witness.centered_coeffs_flat().to_vec(),
            min,
            max,
        }
    }

    pub(crate) fn coefficients(&self) -> &[i32] {
        &self.coefficients
    }

    pub(crate) fn signed_extrema(&self) -> (i32, i32) {
        (self.min, self.max)
    }
}

/// Centered fold coefficients retained for ring-switch witness emission.
///
/// A single fold reuses the global centered buffer in [`DecomposeFoldWitness`].
/// A distributed fold owns at least two independently bounded chunk buffers.
/// Every chunk keeps the same full ambient `z` width, including chunks assigned
/// no live witness blocks; only the corresponding `e` and `t` material may be
/// shorter.
pub(crate) struct FoldChunkCoefficients {
    storage: FoldChunkStorage,
}

enum FoldChunkStorage {
    Single,
    Chunked(Vec<CenteredFoldChunk>),
}

impl FoldChunkCoefficients {
    pub(crate) fn single() -> Self {
        Self {
            storage: FoldChunkStorage::Single,
        }
    }

    pub(crate) fn chunked(chunks: Vec<CenteredFoldChunk>) -> Result<Self, AkitaError> {
        if chunks.len() < 2 {
            return Err(AkitaError::InvalidInput(
                "distributed fold must retain at least two coefficient chunks".into(),
            ));
        }
        let expected_len = chunks[0].coefficients.len();
        if let Some(chunk) = chunks
            .iter()
            .find(|chunk| chunk.coefficients.len() != expected_len)
        {
            return Err(AkitaError::InvalidSize {
                expected: expected_len,
                actual: chunk.coefficients.len(),
            });
        }
        Ok(Self {
            storage: FoldChunkStorage::Chunked(chunks),
        })
    }

    pub(crate) fn num_chunks(&self) -> usize {
        match &self.storage {
            FoldChunkStorage::Single => 1,
            FoldChunkStorage::Chunked(chunks) => chunks.len(),
        }
    }

    #[cfg(feature = "response-model-diagnostics")]
    pub(crate) fn coefficient_count(&self, global: &[i32]) -> usize {
        match &self.storage {
            FoldChunkStorage::Single => global.len(),
            FoldChunkStorage::Chunked(chunks) => {
                chunks.iter().map(|chunk| chunk.coefficients.len()).sum()
            }
        }
    }

    pub(crate) fn all_extrema_within(
        &self,
        global: &DecomposeFoldWitness<impl Field>,
        mut accepts: impl FnMut(i32, i32) -> bool,
    ) -> bool {
        match &self.storage {
            FoldChunkStorage::Single => {
                let (min, max) = global.centered_signed_extrema();
                accepts(min, max)
            }
            FoldChunkStorage::Chunked(chunks) => chunks.iter().all(|chunk| {
                let (min, max) = chunk.signed_extrema();
                accepts(min, max)
            }),
        }
    }

    fn ensure_ring_dim<const D: usize>(&self) -> Result<(), AkitaError> {
        if let FoldChunkStorage::Chunked(chunks) = &self.storage {
            for chunk in chunks {
                if !chunk.coefficients.len().is_multiple_of(D) {
                    return Err(AkitaError::InvalidSize {
                        expected: D,
                        actual: chunk.coefficients.len(),
                    });
                }
            }
        }
        Ok(())
    }

    pub(crate) fn try_for_each(
        &self,
        global: &[i32],
        expected_chunks: usize,
        mut visit: impl FnMut(&[i32]) -> Result<(), AkitaError>,
    ) -> Result<(), AkitaError> {
        let actual_chunks = match &self.storage {
            FoldChunkStorage::Single => 1,
            FoldChunkStorage::Chunked(chunks) => chunks.len(),
        };
        if actual_chunks != expected_chunks {
            return Err(AkitaError::InvalidSize {
                expected: expected_chunks,
                actual: actual_chunks,
            });
        }
        match &self.storage {
            FoldChunkStorage::Single => visit(global),
            FoldChunkStorage::Chunked(chunks) => {
                for chunk in chunks {
                    visit(chunk.coefficients())?;
                }
                Ok(())
            }
        }
    }

    pub(crate) fn chunk<'a>(
        &'a self,
        global: &'a [i32],
        expected_chunks: usize,
        index: usize,
    ) -> Result<&'a [i32], AkitaError> {
        let actual_chunks = self.num_chunks();
        if actual_chunks != expected_chunks {
            return Err(AkitaError::InvalidSize {
                expected: expected_chunks,
                actual: actual_chunks,
            });
        }
        match &self.storage {
            FoldChunkStorage::Single if index == 0 => Ok(global),
            FoldChunkStorage::Single => Err(AkitaError::InvalidSize {
                expected: 1,
                actual: index + 1,
            }),
            FoldChunkStorage::Chunked(chunks) => chunks
                .get(index)
                .map(CenteredFoldChunk::coefficients)
                .ok_or(AkitaError::InvalidSize {
                    expected: chunks.len(),
                    actual: index + 1,
                }),
        }
    }
}

/// Per-group secret witness for the ring relation at one fold level.
pub struct RingRelationGroupWitness<F: Field> {
    pub z_folded_rings: DecomposeFoldWitness<F>,
    pub(crate) z_folded_coefficients: FoldChunkCoefficients,
    pub e_hat: DigitBlocks,
    pub(crate) folded_opening: GroupFoldedOpening<F>,
    pub hint: AkitaCommitmentHint<F>,
    role_dims: CommitmentRingDims,
}

impl<F: Field> RingRelationGroupWitness<F> {
    /// Construct one group witness from D-free carriers.
    pub(crate) fn from_parts(
        z_folded_rings: DecomposeFoldWitness<F>,
        z_folded_coefficients: FoldChunkCoefficients,
        e_hat: DigitBlocks,
        e_folded: RingVec<F>,
        hint: AkitaCommitmentHint<F>,
        role_dims: CommitmentRingDims,
    ) -> Self {
        Self {
            z_folded_rings,
            z_folded_coefficients,
            e_hat,
            folded_opening: OpeningFamily::EvaluationTrace(e_folded),
            hint,
            role_dims,
        }
    }

    /// Construct one coefficient-packing group witness from checked physical coordinates.
    pub(crate) fn from_coefficient_packing_parts(
        z_folded_rings: DecomposeFoldWitness<F>,
        z_folded_coefficients: FoldChunkCoefficients,
        e_hat: DigitBlocks,
        product: CoefficientPackingFoldProduct<F>,
        hint: AkitaCommitmentHint<F>,
        role_dims: CommitmentRingDims,
    ) -> Self {
        Self {
            z_folded_rings,
            z_folded_coefficients,
            e_hat,
            folded_opening: OpeningFamily::SubringCoefficientPacking(product),
            hint,
            role_dims,
        }
    }

    /// Per-role ring dimensions for this group witness.
    pub fn role_dims(&self) -> CommitmentRingDims {
        self.role_dims
    }

    /// Validate one role carrier against dispatch `D`.
    pub fn ensure_role_dim<const D: usize>(&self, role: RingRole) -> Result<(), AkitaError> {
        let expected = self.role_dims.dim_for(role);
        if D != expected {
            return Err(AkitaError::InvalidInput(format!(
                "ring relation witness role {role:?} expects d={expected}, requested D={D}"
            )));
        }
        match role {
            RingRole::Inner => {
                self.z_folded_rings.ensure_ring_dim::<D>()?;
                if let OpeningFamily::EvaluationTrace(e_folded) = &self.folded_opening {
                    if !e_folded.can_decode_vec(D) {
                        return Err(AkitaError::InvalidSize {
                            expected: D,
                            actual: e_folded.coeff_len(),
                        });
                    }
                }
                self.z_folded_coefficients.ensure_ring_dim::<D>()?;
            }
            RingRole::Opening => {
                if self.e_hat.digit_stride() != D {
                    return Err(AkitaError::InvalidSize {
                        expected: D,
                        actual: self.e_hat.digit_stride(),
                    });
                }
            }
            RingRole::Outer => {}
        }
        Ok(())
    }

    /// Validate that all role carriers match a single uniform dimension `D`.
    pub fn ensure_ring_dim<const D: usize>(&self) -> Result<(), AkitaError> {
        let uniform = self.role_dims.uniform_dim()?;
        if uniform != D {
            return Err(AkitaError::InvalidInput(format!(
                "ring relation witness uniform dim {uniform} does not match requested D={D}"
            )));
        }
        self.ensure_role_dim::<D>(RingRole::Inner)?;
        self.ensure_role_dim::<D>(RingRole::Opening)?;
        self.ensure_role_dim::<D>(RingRole::Outer)?;
        Ok(())
    }

    /// Rebuild typed `e_hat` digit planes after [`Self::ensure_role_dim`].
    pub fn e_hat_trusted<const D: usize>(&self) -> Result<&DigitBlocks, AkitaError> {
        self.ensure_role_dim::<D>(RingRole::Opening)?;
        self.e_hat.ensure_stride::<D>()?;
        Ok(&self.e_hat)
    }

    /// Borrow folded `e` rows after [`Self::ensure_role_dim`].
    pub fn e_folded_trusted<const D: usize>(&self) -> Result<&[CyclotomicRing<F, D>], AkitaError> {
        self.ensure_role_dim::<D>(RingRole::Inner)?;
        match &self.folded_opening {
            OpeningFamily::EvaluationTrace(e_folded) => e_folded.as_ring_slice::<D>(),
            OpeningFamily::SubringCoefficientPacking(_) => Err(AkitaError::InvalidSetup(
                "coefficient-packing folded opening is not an A-ring vector".into(),
            )),
        }
    }
}

/// Prover secret for the per-fold ring relation (never built on the verifier).
pub(crate) enum RelationDQuotientWitness<F: Field> {
    /// Quotient-lift mode retains the D-role quotient rows.
    QuotientLift(RingVec<F>),
    /// Reduced-evaluation mode has no D-role quotient rows.
    ReducedEvaluation,
}

pub struct RingRelationWitness<F: Field> {
    pub groups: Vec<RingRelationGroupWitness<F>>,
    /// Level-owned D-role quotient rows retained after transcript-time `v` construction.
    pub(crate) d_quotients: RelationDQuotientWitness<F>,
    pub(crate) compression: Option<CompressionWitnessMaterialization<F>>,
}

impl<F: Field> RingRelationWitness<F> {
    /// Construct from already-grouped witnesses.
    pub(crate) fn from_groups(
        groups: Vec<RingRelationGroupWitness<F>>,
        d_quotients: RelationDQuotientWitness<F>,
        compression: Option<CompressionWitnessMaterialization<F>>,
    ) -> Self {
        Self {
            groups,
            d_quotients,
            compression,
        }
    }

    /// Borrow one group's witness.
    pub fn group(&self, g: usize) -> Result<&RingRelationGroupWitness<F>, AkitaError> {
        self.groups.get(g).ok_or_else(|| {
            AkitaError::InvalidInput(format!(
                "ring relation witness group index {g} out of range ({} groups)",
                self.groups.len()
            ))
        })
    }

    /// Public terminal payload of the shared opening-compression chain.
    pub(crate) fn opening_payload(&self) -> Result<RingVec<F>, AkitaError>
    where
        F: jolt_field::CanonicalEncoding,
    {
        let source = self
            .compression
            .as_ref()
            .ok_or(AkitaError::InvalidProof)?
            .source(crate::protocol::ring_relation::CompressionSourceId::Opening)?;
        let ring_dim = source
            .witness
            .plan()
            .maps()
            .last()
            .ok_or(AkitaError::InvalidProof)?
            .ring_dimension();
        RingVec::from_coeffs_with_ring_dim(source.terminal.coefficients().to_vec(), ring_dim)
    }

    /// Validate one role carrier against dispatch `D` for every group.
    pub fn ensure_role_dim<const D: usize>(&self, role: RingRole) -> Result<(), AkitaError> {
        for group in &self.groups {
            group.ensure_role_dim::<D>(role)?;
        }
        Ok(())
    }

    /// Validate that all role carriers match a single uniform dimension `D`.
    pub fn ensure_ring_dim<const D: usize>(&self) -> Result<(), AkitaError> {
        for group in &self.groups {
            group.ensure_ring_dim::<D>()?;
        }
        Ok(())
    }
}
