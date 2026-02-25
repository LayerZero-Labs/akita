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
}

impl<F: FieldCore + HachiSerialize, const D: usize> HachiProof<F, D> {
    /// Returns the proof size in bytes (uncompressed).
    pub fn size(&self) -> usize {
        self.v.serialized_size(Compress::No) + self.y_ring.serialized_size(Compress::No)
    }
}
