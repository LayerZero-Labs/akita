//! Ring-native opening point for the Hachi protocol.

use crate::FieldCore;

/// Ring-native opening point storing field scalars (Lagrange weights).
///
/// Contains the two vectors used by the §4.2 prover:
/// - `a`: evaluation vector of length `2^m` (inner-block coordinates).
/// - `b`: block-select vector of length `2^r` (outer coordinates).
///
/// These are raw field scalars, not ring elements — they originate from
/// multilinear Lagrange basis evaluations and are always constant (scalar)
/// ring elements when embedded into the ring.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RingOpeningPoint<F: FieldCore> {
    /// Evaluation vector of length `2^m` (field scalars).
    pub a: Vec<F>,
    /// Block-select vector of length `2^r` (field scalars).
    pub b: Vec<F>,
}
