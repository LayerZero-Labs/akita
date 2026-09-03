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
    use crate::ntt_cache_requires_exactness_tail;
    use jolt_field::{Prime128OffsetA7F7, Prime32Offset99, Prime64Offset59};

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

    #[test]
    fn reachable_negative_binary_widths_need_no_exactness_tail() {
        use crate::{prepare_compression_ntt_cache, FlatMatrix};
        use akita_algebra::CyclotomicRing;

        // First-map dimensions sit in both the protocol NTT band and the compression ladder.
        assert!(!ntt_cache_requires_exactness_tail::<Prime128OffsetA7F7, 16>(4_096, 1).unwrap());
        assert!(!ntt_cache_requires_exactness_tail::<Prime64Offset59, 32>(2_048, 1).unwrap());
        assert!(!ntt_cache_requires_exactness_tail::<Prime32Offset99, 64>(1_024, 1).unwrap());

        // Compression-only dims must use the purpose-aware prep path.
        let q128_d8 = FlatMatrix::from_ring_slice(&vec![
            CyclotomicRing::<Prime128OffsetA7F7, 8>::zero();
            256
        ]);
        let q128_cache =
            prepare_compression_ntt_cache(q128_d8.ring_view::<8>(1, 256).expect("view"))
                .expect("q128/D8 cache");
        assert!(!q128_cache.has_exactness_tail());
        assert!(q128_cache.has_cyclic());

        let q64_d16 =
            FlatMatrix::from_ring_slice(&vec![CyclotomicRing::<Prime64Offset59, 16>::zero(); 128]);
        let q64_cache =
            prepare_compression_ntt_cache(q64_d16.ring_view::<16>(1, 128).expect("view"))
                .expect("q64/D16 cache");
        assert!(!q64_cache.has_exactness_tail());
        assert!(q64_cache.has_cyclic());

        let q32_d32 =
            FlatMatrix::from_ring_slice(&vec![CyclotomicRing::<Prime32Offset99, 32>::zero(); 64]);
        let q32_cache =
            prepare_compression_ntt_cache(q32_d32.ring_view::<32>(1, 64).expect("view"))
                .expect("q32/D32 cache");
        assert!(!q32_cache.has_exactness_tail());
        assert!(q32_cache.has_cyclic());
    }
}
