//! Prover-only secret witness for the negacyclic-ring relation.

use crate::DecomposeFoldWitness;
use akita_field::{AkitaError, FieldCore};
use akita_types::{AkitaCommitmentHint, CommitmentRingDims, DigitBlocks, RingRole, RingVec};

/// Prover secret for the per-fold ring relation (never built on the verifier).
///
/// `hint` is the D-free [`AkitaCommitmentHint`] (decomposed digit stream only);
/// recomposed inner rows are recomputed on demand from it (see
/// [`crate::compute::recompose_flat_hint_inner_rows`]).
pub struct RingRelationWitness<F: FieldCore> {
    pub z_folded_rings: DecomposeFoldWitness<F>,
    /// Per-window centered fold responses `z_i = Σ_{j∈I_i} c_j s_j` emitted
    /// z-first per chunk by `build_w_coeffs`. Length `num_chunks` (one element
    /// equal to the global centered fold response for the single-chunk case).
    pub z_folded_centered_per_chunk: Vec<Vec<Vec<i32>>>,
    pub fold_grind_nonce: u32,
    pub e_hat: DigitBlocks,
    pub e_folded: RingVec<F>,
    pub hint: AkitaCommitmentHint<F>,
    role_dims: CommitmentRingDims,
}

impl<F: FieldCore> RingRelationWitness<F> {
    /// Construct from D-free fold outputs under schedule-derived role dimensions.
    pub fn from_flat_parts(
        z_folded_rings: DecomposeFoldWitness<F>,
        z_folded_centered_per_chunk: Vec<Vec<Vec<i32>>>,
        fold_grind_nonce: u32,
        e_hat: DigitBlocks,
        e_folded: RingVec<F>,
        hint: AkitaCommitmentHint<F>,
        role_dims: CommitmentRingDims,
    ) -> Self {
        Self {
            z_folded_rings,
            z_folded_centered_per_chunk,
            fold_grind_nonce,
            e_hat,
            e_folded,
            hint,
            role_dims,
        }
    }

    /// Per-role ring dimensions for this witness.
    pub fn role_dims(&self) -> CommitmentRingDims {
        self.role_dims
    }

    /// Validate one role carrier against dispatch `D`.
    ///
    /// # Errors
    ///
    /// Returns an error if the requested ring dimension does not match the role.
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

    /// Borrow folded `e` rows after [`Self::ensure_role_dim`].
    pub fn e_folded_trusted<const D: usize>(
        &self,
    ) -> Result<&[akita_algebra::CyclotomicRing<F, D>], AkitaError> {
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
