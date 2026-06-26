//! Prover-only secret witness for the negacyclic-ring relation.

use crate::DecomposeFoldWitness;
use akita_field::FieldCore;
use akita_types::{AkitaCommitmentHint, FlatDigitBlocks, RingBuf};

/// Prover secret for the per-fold ring relation (never built on the verifier).
pub struct RingRelationWitness<F: FieldCore> {
    pub z_folded_rings: DecomposeFoldWitness<F>,
    pub fold_grind_nonce: u32,
    pub e_hat: FlatDigitBlocks,
    pub e_folded: RingBuf<F>,
    pub hint: AkitaCommitmentHint<F>,
}

impl<F: FieldCore> RingRelationWitness<F> {
    /// Assemble D-free fold-relation witness storage from prover-side parts.
    pub fn new(
        z_folded_rings: DecomposeFoldWitness<F>,
        fold_grind_nonce: u32,
        e_hat: FlatDigitBlocks,
        e_folded: RingBuf<F>,
        hint: AkitaCommitmentHint<F>,
    ) -> Self {
        Self {
            z_folded_rings,
            fold_grind_nonce,
            e_hat,
            e_folded,
            hint,
        }
    }
}
