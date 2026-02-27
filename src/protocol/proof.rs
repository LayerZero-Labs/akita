//! Proof structures for the Hachi protocol.

use crate::algebra::ring::CyclotomicRing;
use crate::primitives::serialization::Compress;
use crate::{FieldCore, HachiSerialize};

/// Hachi Proof for One Iteration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HachiProof<F: FieldCore, const D: usize> {
    /// `y_ring` from the §3.1 reduction.
    pub y_ring: CyclotomicRing<F, D>,
    /// `v = D · ŵ`.
    pub v: Vec<CyclotomicRing<F, D>>,
    /// `w` coefficients used by sumcheck (z and r coefficients, concatenated).
    pub w: Vec<F>,
    /// Ring-switching challenge `alpha`.
    pub alpha: F,
    /// Flattened `M_a` vector (row-major).
    pub m_a: Vec<F>,
    /// `u_eval = Σ_i b_i (a^T f_i)` from the ring opening point.
    pub u_eval: CyclotomicRing<F, D>,
    /// Public `y` vector for ring-switch relation.
    pub y_vec: Vec<CyclotomicRing<F, D>>,
    /// Public `y(α)` values for ring-switch sumcheck.
    pub y_a: Vec<F>,
}

impl<F: FieldCore + HachiSerialize, const D: usize> HachiProof<F, D> {
    /// Returns the proof size in bytes (uncompressed).
    pub fn size(&self) -> usize {
        self.v.serialized_size(Compress::No)
            + self.y_ring.serialized_size(Compress::No)
            + self.w.serialized_size(Compress::No)
            + self.alpha.serialized_size(Compress::No)
            + self.m_a.serialized_size(Compress::No)
            + self.u_eval.serialized_size(Compress::No)
            + self.y_vec.serialized_size(Compress::No)
            + self.y_a.serialized_size(Compress::No)
    }
}
