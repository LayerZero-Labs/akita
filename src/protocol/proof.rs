//! Proof structures for the Hachi protocol.

use crate::algebra::ring::CyclotomicRing;
use crate::primitives::serialization::Compress;
use crate::protocol::sumcheck::SumcheckProof;
use crate::{FieldCore, HachiSerialize};

/// Temporary auxiliary data the verifier needs for sumcheck output verification.
///
/// Will be removed once recursive PCS evaluation proofs replace the direct
/// oracle check at the end of each sumcheck instance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SumcheckAux<F: FieldCore> {
    /// `w` coefficients (z and r coefficients, concatenated). The verifier
    /// reshapes this into sumcheck evaluation form to compute the expected
    /// output claims for F_0 and F_alpha.
    pub w: Vec<F>,
}

/// Hachi Proof for One Iteration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HachiProof<F: FieldCore, const D: usize> {
    /// `y_ring` from the §3.1 reduction.
    pub y_ring: CyclotomicRing<F, D>,
    /// `v = D · ŵ`.
    pub v: Vec<CyclotomicRing<F, D>>,
    /// `u_eval = Σ_i b_i (a^T f_i)` from the ring opening point.
    pub u_eval: CyclotomicRing<F, D>,
    /// Range-check sumcheck proof (§4.3, F_0).
    pub f0_proof: SumcheckProof<F>,
    /// Evaluation-relation sumcheck proof (§4.3, F_alpha).
    pub f_alpha_proof: SumcheckProof<F>,
    /// Temporary verifier auxiliary (will be removed with recursive PCS).
    pub sumcheck_aux: SumcheckAux<F>,
}

impl<F: FieldCore + HachiSerialize, const D: usize> HachiProof<F, D> {
    /// Returns the proof size in bytes (uncompressed).
    pub fn size(&self) -> usize {
        self.v.serialized_size(Compress::No)
            + self.y_ring.serialized_size(Compress::No)
            + self.sumcheck_aux.w.serialized_size(Compress::No)
            + self.u_eval.serialized_size(Compress::No)
            + self.f0_proof.serialized_size(Compress::No)
            + self.f_alpha_proof.serialized_size(Compress::No)
    }
}
