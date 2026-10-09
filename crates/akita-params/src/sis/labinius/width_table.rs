//! Runtime admission by domination of an exact certified SIS cutoff.

use super::{LabiniusCoefficientPrime, LabiniusRingDegree};

pub use super::generated_width_table::{LABINIUS_WIDTH_TABLE, LABINIUS_WIDTH_TABLE_DIGEST};

/// One exact certified positive-width cell under quantum 128-bit ADPS16.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LabiniusWidthCell {
    /// Exact coefficient prime.
    pub coefficient_prime: LabiniusCoefficientPrime,
    /// Scalarized commitment-ring degree.
    pub ring_degree: LabiniusRingDegree,
    /// Module rank.
    pub rank: u32,
    /// Certified coefficient infinity-norm bound.
    pub coeff_linf_bound: u64,
    /// Largest certified number of ring columns.
    pub max_width: u64,
}

/// Smallest certified rank dominating both the requested norm and width.
///
/// Only cells with the same prime and degree participate. There is no
/// interpolation or extrapolation; an uncovered request returns `None`.
pub fn labinius_min_secure_rank(
    prime: LabiniusCoefficientPrime,
    degree: LabiniusRingDegree,
    coeff_linf_bound: u128,
    width: u64,
) -> Option<u32> {
    LABINIUS_WIDTH_TABLE
        .iter()
        .filter(|cell| {
            cell.coefficient_prime == prime
                && cell.ring_degree == degree
                && u128::from(cell.coeff_linf_bound) >= coeff_linf_bound
                && cell.max_width >= width
        })
        .map(|cell| cell.rank)
        .min()
}

#[cfg(test)]
#[path = "width_table_tests.rs"]
mod tests;
