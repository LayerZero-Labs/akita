//! Prover-only secret witness for the negacyclic-ring relation.

use crate::protocol::ring_relation::CompressionWitnessMaterialization;
use crate::DecomposeFoldWitness;
use akita_algebra::CyclotomicRing;
use akita_field::{AkitaError, FieldCore};
use akita_types::{AkitaCommitmentHint, CommitmentRingDims, DigitBlocks, RingRole, RingVec};

/// One distributed fold window's centered coefficients and signed extrema.
pub(crate) struct CenteredFoldChunk {
    coefficients: Vec<i32>,
    min: i32,
    max: i32,
}

impl CenteredFoldChunk {
    /// Retain one chunk's centered coefficients and the extrema computed by
    /// its canonical fold-witness constructor.
    pub(crate) fn from_witness<F: FieldCore>(witness: &DecomposeFoldWitness<F>) -> Self {
        Self {
            coefficients: witness.centered_coeffs_flat().to_vec(),
            min: witness.centered_min,
            max: witness.centered_max,
        }
    }

    pub(crate) fn coefficients(&self) -> &[i32] {
        &self.coefficients
    }

    pub(crate) fn signed_extrema(&self) -> (i32, i32) {
        (self.min, self.max)
    }
}

/// Per-group secret witness for the ring relation at one fold level.
pub struct RingRelationGroupWitness<F: FieldCore> {
    pub z_folded_rings: DecomposeFoldWitness<F>,
    /// Per-window centered fold responses for chunked witnesses. `None` means
    /// the one chunk is the global centered buffer in `z_folded_rings`.
    pub(crate) z_folded_centered_per_chunk: Option<Vec<CenteredFoldChunk>>,
    pub e_hat: DigitBlocks,
    pub e_folded: RingVec<F>,
    pub hint: AkitaCommitmentHint<F>,
    role_dims: CommitmentRingDims,
}

impl<F: FieldCore> RingRelationGroupWitness<F> {
    /// Construct one group witness from D-free carriers.
    pub(crate) fn from_parts(
        z_folded_rings: DecomposeFoldWitness<F>,
        z_folded_centered_per_chunk: Option<Vec<CenteredFoldChunk>>,
        e_hat: DigitBlocks,
        e_folded: RingVec<F>,
        hint: AkitaCommitmentHint<F>,
        role_dims: CommitmentRingDims,
    ) -> Self {
        Self {
            z_folded_rings,
            z_folded_centered_per_chunk,
            e_hat,
            e_folded,
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
                if !self.e_folded.can_decode_vec(D) {
                    return Err(AkitaError::InvalidSize {
                        expected: D,
                        actual: self.e_folded.coeff_len(),
                    });
                }
                if let Some(chunks) = &self.z_folded_centered_per_chunk {
                    for chunk in chunks {
                        if !chunk.coefficients.len().is_multiple_of(D) {
                            return Err(AkitaError::InvalidSize {
                                expected: D,
                                actual: chunk.coefficients.len(),
                            });
                        }
                    }
                }
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
        Ok(self.e_folded.as_ring_slice_trusted::<D>())
    }
}

/// Prover secret for the per-fold ring relation (never built on the verifier).
pub struct RingRelationWitness<F: FieldCore> {
    pub fold_grind_nonce: u32,
    pub groups: Vec<RingRelationGroupWitness<F>>,
    /// Level-owned D-role quotient rows retained after transcript-time `v` construction.
    pub(crate) d_quotients: RingVec<F>,
    pub(crate) compression: Option<CompressionWitnessMaterialization<F>>,
}

impl<F: FieldCore> RingRelationWitness<F> {
    /// Construct from D-free fold outputs under schedule-derived role dimensions.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_flat_parts(
        z_folded_rings: DecomposeFoldWitness<F>,
        z_folded_centered_per_chunk: Option<Vec<CenteredFoldChunk>>,
        fold_grind_nonce: u32,
        e_hat: DigitBlocks,
        e_folded: RingVec<F>,
        hint: AkitaCommitmentHint<F>,
        role_dims: CommitmentRingDims,
        d_quotients: RingVec<F>,
        compression: Option<CompressionWitnessMaterialization<F>>,
    ) -> Self {
        Self {
            fold_grind_nonce,
            groups: vec![RingRelationGroupWitness::from_parts(
                z_folded_rings,
                z_folded_centered_per_chunk,
                e_hat,
                e_folded,
                hint,
                role_dims,
            )],
            d_quotients,
            compression,
        }
    }

    /// Construct from already-grouped witnesses.
    pub(crate) fn from_groups(
        fold_grind_nonce: u32,
        groups: Vec<RingRelationGroupWitness<F>>,
        d_quotients: RingVec<F>,
        compression: Option<CompressionWitnessMaterialization<F>>,
    ) -> Self {
        Self {
            fold_grind_nonce,
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
        F: akita_field::CanonicalField,
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
