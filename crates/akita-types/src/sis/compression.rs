//! Narrow SIS coverage for diagnostic compressed commitments.
//!
//! This coverage is deliberately separate from the production A/B/D matrix
//! roles and schedule identity. It prices only the F/H matrices exercised by
//! the shadow compression path.

use super::generated_compression_sis_table::sis_max_widths;
use super::{SisModulusProfileId, SisSecurityPolicyId};

/// Coefficient infinity norm of a negative-binary compression matrix.
pub const COMPRESSION_SIS_COEFF_LINF_BOUND: u128 = 1;

/// Largest module rank generated for diagnostic compression matrices.
pub const COMPRESSION_SIS_MAX_MODULE_RANK: u32 = 2;

/// One exact cell in the diagnostic compressed-commitment SIS surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompressionSisCell {
    /// Exact SIS modulus profile.
    pub modulus_profile: SisModulusProfileId,
    /// Ring dimension.
    pub ring_dimension: u32,
    /// Coefficient infinity bound.
    pub coeff_linf_bound: u128,
    /// Largest generated module rank.
    pub max_module_rank: u32,
    /// Largest required input width.
    pub required_max_width: u64,
}

/// Return the exact diagnostic compression cell, if it is in scope.
#[must_use]
pub fn compression_sis_cell(
    modulus_profile: SisModulusProfileId,
    ring_dimension: u32,
    coeff_linf_bound: u128,
) -> Option<CompressionSisCell> {
    if coeff_linf_bound != COMPRESSION_SIS_COEFF_LINF_BOUND {
        return None;
    }
    let required_max_width = match (modulus_profile, ring_dimension) {
        (SisModulusProfileId::Q128OffsetA7F7, 16) => 8_192,
        (SisModulusProfileId::Q128OffsetA7F7, 8) => 512,
        (SisModulusProfileId::Q64Offset59, 32) => 4_096,
        (SisModulusProfileId::Q64Offset59, 16) => 256,
        (SisModulusProfileId::Q32Offset99, 64) => 2_048,
        (SisModulusProfileId::Q32Offset99, 32) => 128,
        _ => return None,
    };
    Some(CompressionSisCell {
        modulus_profile,
        ring_dimension,
        coeff_linf_bound,
        max_module_rank: COMPRESSION_SIS_MAX_MODULE_RANK,
        required_max_width,
    })
}

/// Minimum ADPS16-quantum-secure module rank for one compression matrix.
#[must_use]
pub fn min_compression_secure_rank(
    policy: SisSecurityPolicyId,
    modulus_profile: SisModulusProfileId,
    ring_dimension: u32,
    coeff_linf_bound: u128,
    width: u64,
) -> Option<usize> {
    let cell = compression_sis_cell(modulus_profile, ring_dimension, coeff_linf_bound)?;
    if width == 0 || width > cell.required_max_width {
        return None;
    }
    sis_max_widths(policy, modulus_profile, ring_dimension, coeff_linf_bound)?
        .iter()
        .take(usize::try_from(cell.max_module_rank).ok()?)
        .position(|&max_width| width <= max_width)
        .map(|index| index + 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coverage_is_exactly_the_six_compression_cells() {
        for (profile, supported) in [
            (SisModulusProfileId::Q128OffsetA7F7, [8, 16]),
            (SisModulusProfileId::Q64Offset59, [16, 32]),
            (SisModulusProfileId::Q32Offset99, [32, 64]),
        ] {
            for d in supported {
                assert!(compression_sis_cell(profile, d, 1).is_some());
            }
        }

        assert!(compression_sis_cell(SisModulusProfileId::Q128OffsetA7F7, 32, 1).is_none());
        assert!(compression_sis_cell(SisModulusProfileId::Q64Offset59, 8, 1).is_none());
        assert!(compression_sis_cell(SisModulusProfileId::Q32Offset99, 16, 1).is_none());
        assert!(compression_sis_cell(SisModulusProfileId::Q128OffsetA7F7, 8, 2).is_none());
        assert_eq!(
            compression_sis_cell(SisModulusProfileId::Q128OffsetA7F7, 16, 1)
                .expect("first q128 map")
                .required_max_width,
            8_192
        );
        assert_eq!(
            compression_sis_cell(SisModulusProfileId::Q32Offset99, 32, 1)
                .expect("terminal q32 map")
                .required_max_width,
            128
        );
    }

    #[test]
    fn every_reachable_width_has_a_rank_in_the_narrow_table() {
        for (profile, d) in [
            (SisModulusProfileId::Q128OffsetA7F7, 8),
            (SisModulusProfileId::Q128OffsetA7F7, 16),
            (SisModulusProfileId::Q64Offset59, 16),
            (SisModulusProfileId::Q64Offset59, 32),
            (SisModulusProfileId::Q32Offset99, 32),
            (SisModulusProfileId::Q32Offset99, 64),
        ] {
            let cell = compression_sis_cell(profile, d, 1).expect("cell");
            assert!(min_compression_secure_rank(
                SisSecurityPolicyId::Quantum128BitADPS16,
                profile,
                d,
                1,
                cell.required_max_width
            )
            .is_some());
            assert!(min_compression_secure_rank(
                SisSecurityPolicyId::Quantum128BitADPS16,
                profile,
                d,
                1,
                cell.required_max_width + 1
            )
            .is_none());
        }
    }
}
