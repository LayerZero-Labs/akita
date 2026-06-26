//! Prover-only secret witness for the negacyclic-ring relation.

use crate::DecomposeFoldWitness;
use akita_field::{AkitaError, FieldCore};
use akita_types::{AkitaCommitmentHint, ErasedCommitmentHint, FlatDigitBlocks, RingBuf};

#[derive(Debug, Clone, PartialEq, Eq)]
struct ErasedFlatDigitBlocks {
    digits: Vec<i8>,
    block_sizes: Vec<usize>,
    ring_dim: usize,
}

impl ErasedFlatDigitBlocks {
    fn from_typed<const D: usize>(blocks: FlatDigitBlocks<D>) -> Self {
        let mut digits = Vec::with_capacity(blocks.flat_digits().len().saturating_mul(D));
        for plane in blocks.flat_digits() {
            digits.extend_from_slice(plane);
        }
        Self {
            digits,
            block_sizes: blocks.block_sizes().to_vec(),
            ring_dim: D,
        }
    }

    fn rebuild<const D: usize>(&self) -> Result<FlatDigitBlocks<D>, AkitaError> {
        if self.ring_dim != D {
            return Err(AkitaError::InvalidInput(format!(
                "erased flat digit blocks ring_d={} does not match requested D={D}",
                self.ring_dim
            )));
        }
        let total_planes: usize = self.block_sizes.iter().sum();
        if !self.digits.len().is_multiple_of(D) || self.digits.len() / D != total_planes {
            return Err(AkitaError::InvalidSize {
                expected: total_planes.saturating_mul(D),
                actual: self.digits.len(),
            });
        }
        let mut flat_digits = Vec::with_capacity(total_planes);
        for chunk in self.digits.chunks_exact(D) {
            let mut plane = [0i8; D];
            plane.copy_from_slice(chunk);
            flat_digits.push(plane);
        }
        FlatDigitBlocks::new(flat_digits, self.block_sizes.clone())
    }
}

/// Prover secret for the per-fold ring relation (never built on the verifier).
pub struct RingRelationWitness<F: FieldCore> {
    pub z_folded_rings: DecomposeFoldWitness<F>,
    pub fold_grind_nonce: u32,
    e_hat: ErasedFlatDigitBlocks,
    pub e_folded: RingBuf<F>,
    pub hint: ErasedCommitmentHint<F>,
}

type TypedRingRelationWitnessParts<F, const D: usize> = (
    DecomposeFoldWitness<F>,
    u32,
    FlatDigitBlocks<D>,
    Vec<akita_algebra::CyclotomicRing<F, D>>,
    AkitaCommitmentHint<F, D>,
);

impl<F: FieldCore> RingRelationWitness<F> {
    /// Capture typed fold-relation witness parts into D-free storage.
    pub fn from_typed<const D: usize>(
        z_folded_rings: DecomposeFoldWitness<F>,
        fold_grind_nonce: u32,
        e_hat: FlatDigitBlocks<D>,
        e_folded: Vec<akita_algebra::CyclotomicRing<F, D>>,
        hint: AkitaCommitmentHint<F, D>,
    ) -> Self {
        debug_assert_eq!(z_folded_rings.ring_dim(), D);
        Self {
            z_folded_rings,
            fold_grind_nonce,
            e_hat: ErasedFlatDigitBlocks::from_typed(e_hat),
            e_folded: RingBuf::from_ring_elems(&e_folded),
            hint: ErasedCommitmentHint::from_typed(hint),
        }
    }

    pub fn into_typed<const D: usize>(self) -> Result<TypedRingRelationWitnessParts<F, D>, AkitaError> {
        self.z_folded_rings.ensure_ring_dim::<D>()?;
        Ok((
            self.z_folded_rings,
            self.fold_grind_nonce,
            self.e_hat.rebuild::<D>()?,
            self.e_folded.as_ring_slice_trusted::<D>().to_vec(),
            self.hint.to_typed::<D>()?,
        ))
    }
}
