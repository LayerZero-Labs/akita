//! CRT+NTT-domain representation boundary.

use crate::algebra::ring::CyclotomicCrtNtt;

/// CRT+NTT-domain ring representation.
pub type CrtNttDomain<const K: usize, const D: usize> = CyclotomicCrtNtt<K, D>;
