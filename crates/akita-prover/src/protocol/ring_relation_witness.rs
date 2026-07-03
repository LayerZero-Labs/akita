//! Prover-only secret witness for the negacyclic-ring relation.

use crate::compute::FlatDigitBlocks;
use crate::DecomposeFoldWitness;
use akita_field::{AkitaError, FieldCore};
use akita_types::{AkitaCommitmentHint, DigitBlocks, RingVec};

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
    ring_dim: usize,
}

impl<F: FieldCore> RingRelationWitness<F> {
    /// Construct from D-free fold outputs under a schedule-derived ring
    /// dimension.
    pub fn from_flat_parts(
        z_folded_rings: DecomposeFoldWitness<F>,
        z_folded_centered_per_chunk: Vec<Vec<Vec<i32>>>,
        fold_grind_nonce: u32,
        e_hat: DigitBlocks,
        e_folded: RingVec<F>,
        hint: AkitaCommitmentHint<F>,
        ring_dim: usize,
    ) -> Self {
        Self {
            z_folded_rings,
            z_folded_centered_per_chunk,
            fold_grind_nonce,
            e_hat,
            e_folded,
            hint,
            ring_dim,
        }
    }

    /// Stored ring dimension (coefficients per ring element).
    pub fn ring_dim(&self) -> usize {
        self.ring_dim
    }

    /// # Errors
    ///
    /// Returns an error if the requested ring dimension does not match storage.
    pub fn ensure_ring_dim<const D: usize>(&self) -> Result<(), AkitaError> {
        if self.ring_dim != D {
            return Err(AkitaError::InvalidInput(format!(
                "ring relation witness ring_d={} does not match requested D={D}",
                self.ring_dim
            )));
        }
        self.z_folded_rings.ensure_ring_dim::<D>()?;
        if self.e_hat.digit_stride() != D {
            return Err(AkitaError::InvalidSize {
                expected: D,
                actual: self.e_hat.digit_stride(),
            });
        }
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
        Ok(())
    }

    /// Rebuild typed `e_hat` digit planes after [`Self::ensure_ring_dim`].
    pub fn e_hat_trusted<const D: usize>(&self) -> Result<FlatDigitBlocks<D>, AkitaError> {
        self.ensure_ring_dim::<D>()?;
        FlatDigitBlocks::from_digit_blocks(&self.e_hat)
    }

    /// Borrow folded `e` rows after [`Self::ensure_ring_dim`].
    pub fn e_folded_trusted<const D: usize>(
        &self,
    ) -> Result<&[akita_algebra::CyclotomicRing<F, D>], AkitaError> {
        self.ensure_ring_dim::<D>()?;
        Ok(self.e_folded.as_ring_slice_trusted::<D>())
    }

    /// Borrow per-chunk centered fold responses after [`Self::ensure_ring_dim`].
    pub fn z_folded_centered_per_chunk_trusted<const D: usize>(
        &self,
    ) -> Result<Vec<Vec<[i32; D]>>, AkitaError> {
        self.ensure_ring_dim::<D>()?;
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
