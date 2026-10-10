//! Runtime admission by domination of a certified SIS cell.

use super::{LabiniusCommitmentModulus, LabiniusRingDegree};

pub use super::generated_commitment_prime_width_table::{
    LABINIUS_COMMITMENT_PRIME_WIDTH_TABLE, LABINIUS_COMMITMENT_PRIME_WIDTH_TABLE_DIGEST,
};

/// Provenance of a certified runtime width limit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LabiniusWidthCutoff {
    /// Exact positive cutoff with a rejected successor.
    Exact,
    /// Acceptance at the search cap; the true cutoff remains unknown.
    SearchCap,
}

/// A certified admission cell under quantum 128-bit ADPS16.
///
/// A search-cap cell certifies a lower bound on the secure width, not its
/// exact cutoff. Acceptance at the cap and the stated prefix-monotonic
/// boundary precondition extend admission to smaller widths. Neither kind
/// permits admission above `max_width`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LabiniusWidthCell {
    /// Exact commitment modulus.
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

/// Smallest certified rank dominating both the requested bound and width.
///
/// Only cells with the same commitment modulus and degree participate. There
/// is no interpolation or extrapolation; an uncovered request returns `None`.
pub fn labinius_min_secure_rank(
    modulus: LabiniusCommitmentModulus,
    degree: LabiniusRingDegree,
    coeff_linf_bound: u128,
    width: u64,
) -> Option<u32> {
    LABINIUS_COMMITMENT_PRIME_WIDTH_TABLE
        .iter()
        .filter(|cell| {
            cell.commitment_modulus == modulus
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
