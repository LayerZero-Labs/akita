//! Proof structures for the Hachi protocol.

use crate::algebra::ring::CyclotomicRing;
use crate::primitives::serialization::Compress;
use crate::protocol::commitment::RingCommitment;
use crate::protocol::sumcheck::SumcheckProof;
use crate::{FieldCore, HachiSerialize};

/// Prover-side hint produced at commitment time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HachiCommitmentHint<F: FieldCore, const D: usize> {
    /// Decomposed `s_i` blocks from the commitment phase.
    pub s: Vec<Vec<CyclotomicRing<F, D>>>,
    /// Decomposed `t̂_i` blocks from the commitment phase.
    pub t_hat: Vec<Vec<CyclotomicRing<F, D>>>,
    /// Ring coefficients from the §3.1 reduction (evaluation table).
    pub ring_coeffs: Vec<CyclotomicRing<F, D>>,
}

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
    /// Batched sumcheck proof (F_0 norm + F_α relation, §4.3).
    pub sumcheck_proof: SumcheckProof<F>,
    /// Temporary verifier auxiliary (will be removed with recursive PCS).
    pub sumcheck_aux: SumcheckAux<F>,
    /// Commitment to the sumcheck witness `w`.
    pub w_commitment: RingCommitment<F, D>,
}

impl<F: FieldCore + HachiSerialize, const D: usize> HachiProof<F, D> {
    /// Returns the proof size in bytes (uncompressed).
    pub fn size(&self) -> usize {
        self.v.serialized_size(Compress::No)
            + self.y_ring.serialized_size(Compress::No)
            + self.sumcheck_aux.w.serialized_size(Compress::No)
            + self.sumcheck_proof.serialized_size(Compress::No)
            + self.w_commitment.serialized_size(Compress::No)
    }
}
