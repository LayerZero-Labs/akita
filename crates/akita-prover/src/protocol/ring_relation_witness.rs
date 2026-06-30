//! Prover-only secret witness for the negacyclic-ring relation.

use crate::compute::FlatDigitBlocks;
use crate::DecomposeFoldWitness;
use akita_field::FieldCore;
use akita_types::AkitaCommitmentHint;

/// Prover secret for the per-fold ring relation (never built on the verifier).
///
/// `hint` is the D-free [`AkitaCommitmentHint`] (decomposed digit stream only);
/// recomposed inner rows are recomputed on demand from it (see
/// [`crate::compute::recompose_flat_hint_inner_rows`]).
pub struct RingRelationWitness<F: FieldCore, const D: usize> {
    pub z_folded_rings: DecomposeFoldWitness<F>,
    pub fold_grind_nonce: u32,
    pub e_hat: FlatDigitBlocks<D>,
    pub e_folded: Vec<akita_algebra::CyclotomicRing<F, D>>,
    pub hint: AkitaCommitmentHint<F>,
}
