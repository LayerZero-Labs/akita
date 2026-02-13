//! Coefficient-domain representation boundary.

use crate::algebra::ring::CyclotomicRing;

/// Coefficient-domain ring representation.
pub type CoeffDomain<F, const D: usize> = CyclotomicRing<F, D>;
