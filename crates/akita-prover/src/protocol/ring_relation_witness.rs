//! Prover-only secret witness for the negacyclic-ring relation.

use crate::DecomposeFoldWitness;
use akita_field::FieldCore;
use akita_types::{AkitaCommitmentHint, ErasedCommitmentHint, FlatDigitBlocks, RingBuf};

/// Prover secret for the per-fold ring relation (never built on the verifier).
pub struct RingRelationWitness<F: FieldCore> {
    pub z_folded_rings: DecomposeFoldWitness<F>,
    pub fold_grind_nonce: u32,
    pub e_hat: FlatDigitBlocks,
    pub e_folded: RingBuf<F>,
    pub hint: ErasedCommitmentHint<F>,
}

impl<F: FieldCore> RingRelationWitness<F> {
    /// Assemble D-free fold-relation witness storage from prover-side parts.
    pub fn new(
        z_folded_rings: DecomposeFoldWitness<F>,
        fold_grind_nonce: u32,
        e_hat: FlatDigitBlocks,
        e_folded: RingBuf<F>,
        hint: ErasedCommitmentHint<F>,
    ) -> Self {
        Self {
            z_folded_rings,
            fold_grind_nonce,
            e_hat,
            e_folded,
            hint,
        }
    }

    /// Assemble D-free fold-relation witness storage from typed kernel outputs.
    pub fn from_typed<const D: usize>(
        z_folded_rings: DecomposeFoldWitness<F>,
        fold_grind_nonce: u32,
        e_hat: FlatDigitBlocks,
        e_folded: Vec<akita_algebra::CyclotomicRing<F, D>>,
        hint: AkitaCommitmentHint<F, D>,
    ) -> Self {
        debug_assert_eq!(z_folded_rings.ring_dim(), D);
        debug_assert_eq!(e_hat.ring_dim(), D);
        Self {
            z_folded_rings,
            fold_grind_nonce,
            e_hat,
            e_folded: RingBuf::from_ring_elems(&e_folded),
            hint: ErasedCommitmentHint::from_typed(hint),
        }
    }
}
