//! Narrow SIS coverage for production compressed commitments.
//!
//! This coverage is deliberately separate from the production A/B/D matrix
//! roles and schedule identity. It prices only the six rank-one F/H cells used
//! by the fixed two-map protocol.

use super::{SisModulusProfileId, SisSecurityPolicyId};

/// Coefficient infinity norm of a negative-binary compression matrix.
pub const COMPRESSION_SIS_COEFF_LINF_BOUND: u128 = 1;

/// One exact cell in the compressed-commitment SIS surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompressionSisCell {
    /// Exact SIS modulus profile.
    pub modulus_profile: SisModulusProfileId,
    /// Ring dimension.
    pub ring_dimension: u32,
    /// Largest ADPS16-quantum-secure input width at rank one.
    pub sis_max_width: u64,
}

/// Six rank-one cells: `(profile, ring_dimension, sis_max_width)`.
const COMPRESSION_SIS_CELLS: &[(SisModulusProfileId, u32, u64)] = &[
    (SisModulusProfileId::Q128OffsetA7F7, 8, 508),
    (SisModulusProfileId::Q128OffsetA7F7, 16, 7_077),
    (SisModulusProfileId::Q64Offset59, 16, 254),
    (SisModulusProfileId::Q64Offset59, 32, 3_538),
    (SisModulusProfileId::Q32Offset99, 32, 127),
    (SisModulusProfileId::Q32Offset99, 64, 1_769),
];

/// Enumerate the exact production compression coverage cells.
pub fn compression_sis_cells() -> impl ExactSizeIterator<Item = CompressionSisCell> {
    COMPRESSION_SIS_CELLS
        .iter()
        .copied()
        .map(
            |(modulus_profile, ring_dimension, sis_max_width)| CompressionSisCell {
                modulus_profile,
                ring_dimension,
                sis_max_width,
            },
        )
}

/// Return the exact production compression cell, if it is in scope.
#[must_use]
pub fn compression_sis_cell(
    modulus_profile: SisModulusProfileId,
    ring_dimension: u32,
    coeff_linf_bound: u128,
) -> Option<CompressionSisCell> {
    if coeff_linf_bound != COMPRESSION_SIS_COEFF_LINF_BOUND {
        return None;
    }
    COMPRESSION_SIS_CELLS
        .iter()
        .copied()
        .find(|&(profile, dimension, _)| profile == modulus_profile && dimension == ring_dimension)
        .map(
            |(modulus_profile, ring_dimension, sis_max_width)| CompressionSisCell {
                modulus_profile,
                ring_dimension,
                sis_max_width,
            },
        )
}

/// Minimum ADPS16-quantum-secure module rank for one compression matrix.
///
/// The compression protocol is structurally rank one, so this returns `Some(1)`
/// iff `width` is nonzero and at most the cell's SIS-certified max width.
#[must_use]
pub fn min_compression_secure_rank(
    policy: SisSecurityPolicyId,
    modulus_profile: SisModulusProfileId,
    ring_dimension: u32,
    coeff_linf_bound: u128,
    width: u64,
) -> Option<usize> {
    if policy != SisSecurityPolicyId::Quantum128BitADPS16 {
        return None;
    }
    let cell = compression_sis_cell(modulus_profile, ring_dimension, coeff_linf_bound)?;
    (width > 0 && width <= cell.sis_max_width).then_some(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn coverage_is_exactly_the_six_rank_one_compression_cells() {
        assert_eq!(COMPRESSION_SIS_CELLS.len(), 6);
        for &(profile, d, _) in COMPRESSION_SIS_CELLS {
            assert!(compression_sis_cell(profile, d, 1).is_some());
        }

        assert!(compression_sis_cell(SisModulusProfileId::Q128OffsetA7F7, 64, 1).is_none());
        assert!(compression_sis_cell(SisModulusProfileId::Q128OffsetA7F7, 32, 1).is_none());
        assert!(compression_sis_cell(SisModulusProfileId::Q64Offset59, 8, 1).is_none());
        assert!(compression_sis_cell(SisModulusProfileId::Q64Offset59, 64, 1).is_none());
        assert!(compression_sis_cell(SisModulusProfileId::Q32Offset99, 16, 1).is_none());
        assert!(compression_sis_cell(SisModulusProfileId::Q32Offset99, 128, 1).is_none());
        assert!(compression_sis_cell(SisModulusProfileId::Q128OffsetA7F7, 8, 2).is_none());
        assert_eq!(
            compression_sis_cell(SisModulusProfileId::Q128OffsetA7F7, 16, 1)
                .expect("first q128 map")
                .sis_max_width,
            7_077
        );
        assert_eq!(
            compression_sis_cell(SisModulusProfileId::Q32Offset99, 32, 1)
                .expect("terminal q32 map")
                .sis_max_width,
            127
        );
    }

    #[test]
    fn every_reachable_width_has_a_rank_in_the_narrow_table() {
        for &(profile, d, sis_max_width) in COMPRESSION_SIS_CELLS {
            assert_eq!(
                min_compression_secure_rank(
                    SisSecurityPolicyId::Quantum128BitADPS16,
                    profile,
                    d,
                    1,
                    sis_max_width
                ),
                Some(1)
            );
            assert!(min_compression_secure_rank(
                SisSecurityPolicyId::Quantum128BitADPS16,
                profile,
                d,
                1,
                sis_max_width + 1
            )
            .is_none());
        }
    }
}
