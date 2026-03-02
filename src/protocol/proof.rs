//! Proof structures for the Hachi protocol.

use crate::algebra::ring::CyclotomicRing;
use crate::protocol::commitment::RingCommitment;
use crate::protocol::greyhound::GreyhoundEvalProof;
use crate::protocol::labrador::LabradorProof;
use crate::protocol::sumcheck::SumcheckProof;
use crate::FieldCore;

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

/// One Hachi fold record (`ring-switch + batched sumcheck + next-w commitment`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HachiFoldProof<F: FieldCore, const D: usize> {
    /// `y_ring` from the §3.1 reduction.
    pub y_ring: CyclotomicRing<F, D>,
    /// `v = D · ŵ`.
    pub v: Vec<CyclotomicRing<F, D>>,
    /// Batched sumcheck proof (F_0 norm + F_α relation, §4.3).
    pub sumcheck_proof: SumcheckProof<F>,
    /// Commitment to the sumcheck witness `w`.
    pub w_commitment: RingCommitment<F, D>,
}

/// Full Hachi proof: folds + Greyhound base case + Labrador recursion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HachiProof<F: FieldCore, const D: usize> {
    /// Recursive Hachi folds.
    pub folds: Vec<HachiFoldProof<F, D>>,
    /// Greyhound evaluation proof at handoff boundary.
    pub greyhound_eval_proof: GreyhoundEvalProof<F, D>,
    /// Labrador recursive proof with final clear witness.
    pub labrador_proof: LabradorProof<F, D>,
}
