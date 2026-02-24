//! Ring-native opening point for the Hachi protocol.

use crate::algebra::ring::CyclotomicRing;
use crate::FieldCore;

/// Ring-native opening point
///
/// Contains the two vectors used by the §4.2 prover:
/// - `a`: evaluation vector of length `2^m` (inner-block coordinates).
/// - `b`: block-select vector of length `2^r` (outer coordinates).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RingOpeningPoint<F: FieldCore, const D: usize> {
    /// Evaluation vector of length `2^m`.
    pub a: Vec<CyclotomicRing<F, D>>,
    /// Block-select vector of length `2^r`.
    pub b: Vec<CyclotomicRing<F, D>>,
}
