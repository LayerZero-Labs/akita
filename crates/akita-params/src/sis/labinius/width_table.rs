//! Runtime admission by domination of an exact certified SIS cutoff.

use super::{LabiniusCoefficientPrime, LabiniusCommitmentModulus, LabiniusRingDegree};

pub use super::generated_small_modulus_width_table::{
    LABINIUS_SMALL_MODULUS_WIDTH_TABLE, LABINIUS_SMALL_MODULUS_WIDTH_TABLE_DIGEST,
};

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

/// Provenance of a certified runtime width limit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LabiniusWidthCutoff {
    /// Exact positive cutoff with a rejected successor.
    Exact,
    /// Acceptance at the search cap; the true cutoff remains unknown.
    SearchCap,
}

/// A certified small-modulus admission cell under quantum 128-bit ADPS16.
///
/// A search-cap cell certifies a lower bound on the secure width, rather than
/// its exact cutoff. Acceptance at the cap and the stated prefix-monotonic
/// boundary precondition extend admission to smaller widths. Neither kind
/// permits admission above `max_width`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LabiniusSmallModulusWidthCell {
    /// Exact commitment modulus, independently of the opening prime.
    pub commitment_modulus: LabiniusCommitmentModulus,
    /// Scalarized commitment-ring degree.
    pub ring_degree: LabiniusRingDegree,
    /// Module rank.
    pub rank: u32,
    /// Certified coefficient infinity-norm bound.
    pub coeff_linf_bound: u64,
    /// Largest width covered by this certificate.
    pub max_width: u64,
    /// Whether this is an exact cutoff or a search-cap lower bound.
    pub cutoff: LabiniusWidthCutoff,
}

/// Smallest certified rank dominating the requested small-modulus bound and width.
///
/// Only cells with the same commitment modulus and degree participate. There
/// is no interpolation or extrapolation; an uncovered request returns `None`.
pub fn labinius_small_modulus_min_secure_rank(
    modulus: LabiniusCommitmentModulus,
    degree: LabiniusRingDegree,
    coeff_linf_bound: u128,
    width: u64,
) -> Option<u32> {
    LABINIUS_SMALL_MODULUS_WIDTH_TABLE
        .iter()
        .filter(|cell| {
            cell.commitment_modulus == modulus
                && cell.ring_degree == degree
                && dominates(
                    cell.coeff_linf_bound,
                    cell.max_width,
                    coeff_linf_bound,
                    width,
                )
        })
        .map(|cell| cell.rank)
        .min()
}

fn dominates(bound: u64, max_width: u64, requested_bound: u128, width: u64) -> bool {
    u128::from(bound) >= requested_bound && max_width >= width
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
                && dominates(
                    cell.coeff_linf_bound,
                    cell.max_width,
                    coeff_linf_bound,
                    width,
                )
        })
        .map(|cell| cell.rank)
        .min()
}

#[cfg(test)]
#[path = "width_table_tests.rs"]
mod tests;
