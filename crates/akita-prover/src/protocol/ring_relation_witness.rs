//! Prover-only secret witness for the negacyclic-ring relation.

use crate::DecomposeFoldWitness;
use akita_field::FieldCore;
use akita_types::AkitaCommitmentHint;
use akita_types::FlatDigitBlocks;

/// Prover secret for the per-fold ring relation (never built on the verifier).
pub struct RingRelationWitness<F: FieldCore, const D: usize> {
    /// Global folded response `z = Σ_j c_j s_j` (used by the value-identical
    /// relation quotient). For the chunked layout this equals `Σ_i z_i` (mod q).
    pub z_folded_rings: DecomposeFoldWitness<F, D>,
    /// Per-window centered fold responses `z_i = Σ_{j∈I_i} c_j s_j` emitted
    /// z-first per chunk by `build_w_coeffs`. Length `num_chunks` (one element
    /// equal to `z_folded_rings.centered_coeffs` for the single-chunk case).
    pub z_folded_centered_per_chunk: Vec<Vec<[i32; D]>>,
    pub fold_grind_nonce: u32,
    pub e_hat: FlatDigitBlocks<D>,
    pub e_folded: Vec<akita_algebra::CyclotomicRing<F, D>>,
    pub hint: AkitaCommitmentHint<F, D>,
}
