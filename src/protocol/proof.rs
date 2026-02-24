//! Proof structures for the Hachi protocol.

use crate::algebra::ring::CyclotomicRing;
use crate::primitives::serialization::Compress;
use crate::{FieldCore, HachiSerialize};

/// Ring-native proof: contains only the verifier-facing messages.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HachiProof<F: FieldCore, const D: usize> {
    /// `v = D · ŵ`.
    pub v: Vec<CyclotomicRing<F, D>>,
}

impl<F: FieldCore + HachiSerialize, const D: usize> HachiProof<F, D> {
    /// Returns the proof size in bytes (uncompressed).
    pub fn size(&self) -> usize {
        self.v.serialized_size(Compress::No)
    }
}
