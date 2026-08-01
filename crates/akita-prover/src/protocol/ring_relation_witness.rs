//! Prover-only secret witness for the negacyclic-ring relation.

use crate::protocol::ring_relation::CompressionWitnessMaterialization;
use crate::DecomposeFoldWitness;
use akita_algebra::CyclotomicRing;
use akita_field::{AkitaError, FieldCore};
use akita_types::{AkitaCommitmentHint, CommitmentRingDims, DigitBlocks, RingRole, RingVec};

/// Per-group secret witness for the ring relation at one fold level.
pub struct RingRelationGroupWitness<F: FieldCore> {
    pub z_folded_rings: DecomposeFoldWitness<F>,
    /// Per-window centered fold responses `z_i = Σ_{j∈I_i} c_j s_j` emitted
    /// z-first per chunk by `build_w_coeffs`. Length `num_chunks` (one element
    /// equal to the global centered fold response for the single-chunk case).
    pub z_folded_centered_per_chunk: Vec<Vec<Vec<i32>>>,
    pub e_hat: DigitBlocks,
    pub e_folded: RingVec<F>,
    pub hint: AkitaCommitmentHint<F>,
    role_dims: CommitmentRingDims,
}

impl<F: FieldCore> RingRelationGroupWitness<F> {
    /// Construct one group witness from D-free carriers.
    pub fn from_parts(
        z_folded_rings: DecomposeFoldWitness<F>,
        z_folded_centered_per_chunk: Vec<Vec<Vec<i32>>>,
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
                for chunk in &self.z_folded_centered_per_chunk {
                    for row in chunk {
                        if row.len() != D {
                            return Err(AkitaError::InvalidSize {
                                expected: D,
                                actual: row.len(),
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

    /// Borrow per-chunk centered fold responses after [`Self::ensure_role_dim`].
    pub fn z_folded_centered_per_chunk_trusted<const D: usize>(
        &self,
    ) -> Result<Vec<Vec<[i32; D]>>, AkitaError> {
        self.ensure_role_dim::<D>(RingRole::Inner)?;
        self.z_folded_centered_per_chunk
            .iter()
            .map(|chunk| {
                chunk
                    .iter()
                    .map(|row| {
                        let arr: [i32; D] =
                            row.as_slice()
                                .try_into()
                                .map_err(|_| AkitaError::InvalidSize {
                                    expected: D,
                                    actual: row.len(),
                                })?;
                        Ok(arr)
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect()
    }
}

/// Prover secret for the per-fold ring relation (never built on the verifier).
pub struct RingRelationWitness<F: FieldCore> {
    pub fold_grind_nonce: u32,
    pub groups: Vec<RingRelationGroupWitness<F>>,
    pub(crate) compression: CompressionWitnessMaterialization<F>,
}

impl<F: FieldCore> RingRelationWitness<F> {
    /// Construct from D-free fold outputs under schedule-derived role dimensions.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_flat_parts(
        z_folded_rings: DecomposeFoldWitness<F>,
        z_folded_centered_per_chunk: Vec<Vec<Vec<i32>>>,
        fold_grind_nonce: u32,
        e_hat: DigitBlocks,
        e_folded: RingVec<F>,
        hint: AkitaCommitmentHint<F>,
        role_dims: CommitmentRingDims,
        compression: CompressionWitnessMaterialization<F>,
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
            compression,
        }
    }

    /// Construct from already-grouped witnesses.
    pub(crate) fn from_groups(
        fold_grind_nonce: u32,
        groups: Vec<RingRelationGroupWitness<F>>,
        compression: CompressionWitnessMaterialization<F>,
    ) -> Self {
        Self {
            fold_grind_nonce,
            groups,
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
