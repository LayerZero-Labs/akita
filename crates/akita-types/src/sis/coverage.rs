//! Exact role, modulus, dimension, and coefficient-bound coverage for SIS rows.

use super::ajtai_key::{SisMatrixRole, SisModulusProfileId};
use super::norm_bound::role_a_collision_inf_norm_for_response_difference;
use crate::dispatch::{protocol_dispatch_tier_for_sis_profile, role_ring_dimensions_for_tier};
use crate::RingRole;
use std::sync::OnceLock;

/// One reachable role coverage cell used by generation and runtime checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SisRoleCell {
    /// Matrix role.
    pub role: SisMatrixRole,
    /// Exact modulus profile.
    pub modulus_profile: SisModulusProfileId,
    /// Ring dimension.
    pub ring_dimension: u32,
    /// Exact role coefficient bound cell.
    pub coeff_linf_bound: u128,
    /// Maximum supported module rank.
    pub max_module_rank: u32,
    /// Largest required ring width from the planner domain.
    pub required_max_width: u64,
}

/// Exact gadget anchors used by B and D.
pub const GADGET_COEFF_LINF_ANCHORS: &[u128] = &[3, 7, 15, 31, 63, 127, 255];

/// Opening-response exponents covered by the exact A-role infinity table.
///
/// Every accepted response difference has diameter `2^(ell * delta) - 1`.
/// Production opening bases `ell in 3..=6` and the explicit cap
/// `ell * delta <= 33` yield exactly these targets.
pub const INNER_RESPONSE_DIFFERENCE_EXPONENTS: &[u32] = &[
    3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 21, 24, 25, 27, 28, 30, 32, 33,
];

/// Largest exact accepted-response interval diameter covered by A-role rows.
pub const MAX_INNER_RESPONSE_DIFFERENCE: u128 = (1u128 << 33) - 1;

/// Production matrix roles with checked-in coverage.
pub const SIS_MATRIX_ROLES: &[SisMatrixRole] = &[
    SisMatrixRole::Inner,
    SisMatrixRole::Outer,
    SisMatrixRole::Open,
];

/// Maximum module rank searched for every production scalar SIS cell.
pub const SIS_MAX_MODULE_RANK: u32 = 20;

/// Per-cell scalar-width search cap used by the production generator.
pub const SIS_REQUIRED_MAX_WIDTH: u64 = 6_400_000_000_000;

const Q32_MAX_INNER_COEFF_LINF_BOUND: u128 = 268_435_455;
const Q64_MAX_INNER_COEFF_LINF_BOUND: u128 = 2_199_023_255_551;
const Q128_MAX_INNER_COEFF_LINF_BOUND: u128 = 17_592_186_044_415;

const fn dispatch_role(role: SisMatrixRole) -> RingRole {
    match role {
        SisMatrixRole::Inner => RingRole::Inner,
        SisMatrixRole::Outer => RingRole::Outer,
        SisMatrixRole::Open => RingRole::Opening,
    }
}

const fn max_inner_coeff_linf_bound(modulus_profile: SisModulusProfileId) -> u128 {
    match modulus_profile {
        SisModulusProfileId::Q32Offset99 => Q32_MAX_INNER_COEFF_LINF_BOUND,
        SisModulusProfileId::Q64Offset59 => Q64_MAX_INNER_COEFF_LINF_BOUND,
        SisModulusProfileId::Q128OffsetA7F7 => Q128_MAX_INNER_COEFF_LINF_BOUND,
    }
}

const fn challenge_extension_degree(modulus_profile: SisModulusProfileId) -> usize {
    match modulus_profile {
        SisModulusProfileId::Q32Offset99 => 4,
        SisModulusProfileId::Q64Offset59 => 2,
        SisModulusProfileId::Q128OffsetA7F7 => 1,
    }
}

fn exact_inner_collision_bound(challenge_l1: u128, exponent: u32) -> Option<u128> {
    role_a_collision_inf_norm_for_response_difference(
        challenge_l1,
        1u128.checked_shl(exponent)?.checked_sub(1)?,
    )
}

fn inner_challenge_mass_supported(
    modulus_profile: SisModulusProfileId,
    ring_dimension: u32,
    challenge_dimension: usize,
) -> bool {
    let Ok(ring_dimension) = usize::try_from(ring_dimension) else {
        return false;
    };
    if challenge_dimension == ring_dimension {
        return true;
    }
    challenge_extension_degree(modulus_profile)
        .checked_mul(challenge_dimension)
        .is_some_and(|partial_width| ring_dimension.is_multiple_of(partial_width))
}

fn inner_bound_supported(
    modulus_profile: SisModulusProfileId,
    ring_dimension: u32,
    coeff_linf_bound: u128,
) -> bool {
    cached_inner_coeff_linf_bounds(modulus_profile, ring_dimension)
        .is_some_and(|bounds| bounds.binary_search(&coeff_linf_bound).is_ok())
}

const INNER_RING_DIMENSION_COUNT: usize =
    akita_challenges::PRODUCTION_FOLD_CHALLENGE_RING_DIMS.len();
static Q32_INNER_COEFF_LINF_BOUNDS: [OnceLock<Vec<u128>>; INNER_RING_DIMENSION_COUNT] =
    [const { OnceLock::new() }; INNER_RING_DIMENSION_COUNT];
static Q64_INNER_COEFF_LINF_BOUNDS: [OnceLock<Vec<u128>>; INNER_RING_DIMENSION_COUNT] =
    [const { OnceLock::new() }; INNER_RING_DIMENSION_COUNT];
static Q128_INNER_COEFF_LINF_BOUNDS: [OnceLock<Vec<u128>>; INNER_RING_DIMENSION_COUNT] =
    [const { OnceLock::new() }; INNER_RING_DIMENSION_COUNT];

fn cached_inner_coeff_linf_bounds(
    modulus_profile: SisModulusProfileId,
    ring_dimension: u32,
) -> Option<&'static [u128]> {
    let ring_dimension_usize = usize::try_from(ring_dimension).ok()?;
    let index = akita_challenges::PRODUCTION_FOLD_CHALLENGE_RING_DIMS
        .iter()
        .position(|&dimension| dimension == ring_dimension_usize)?;
    let cache = match modulus_profile {
        SisModulusProfileId::Q32Offset99 => &Q32_INNER_COEFF_LINF_BOUNDS[index],
        SisModulusProfileId::Q64Offset59 => &Q64_INNER_COEFF_LINF_BOUNDS[index],
        SisModulusProfileId::Q128OffsetA7F7 => &Q128_INNER_COEFF_LINF_BOUNDS[index],
    };
    Some(cache.get_or_init(|| derive_inner_coeff_linf_bounds(modulus_profile, ring_dimension)))
}

/// Enumerate the exact A-role collision bounds for one supported profile and
/// ring dimension.
///
/// Targets are derived from the protocol's challenge configurations and the
/// complete response-difference exponent sweep. No independently maintained
/// coefficient-bound catalog exists.
#[must_use]
pub fn inner_coeff_linf_bounds(
    modulus_profile: SisModulusProfileId,
    ring_dimension: u32,
) -> Vec<u128> {
    if let Some(bounds) = cached_inner_coeff_linf_bounds(modulus_profile, ring_dimension) {
        return bounds.to_vec();
    }
    derive_inner_coeff_linf_bounds(modulus_profile, ring_dimension)
}

fn derive_inner_coeff_linf_bounds(
    modulus_profile: SisModulusProfileId,
    ring_dimension: u32,
) -> Vec<u128> {
    let mut bounds = Vec::new();
    for &challenge_dimension in akita_challenges::PRODUCTION_FOLD_CHALLENGE_RING_DIMS {
        if !inner_challenge_mass_supported(modulus_profile, ring_dimension, challenge_dimension) {
            continue;
        }
        let Some(config) =
            akita_challenges::SparseChallengeConfig::production_for_ring_dim(challenge_dimension)
        else {
            continue;
        };
        bounds.extend(
            INNER_RESPONSE_DIFFERENCE_EXPONENTS
                .iter()
                .filter_map(|&exponent| {
                    exact_inner_collision_bound(config.l1_norm() as u128, exponent)
                }),
        );
    }
    if ring_dimension == 64 {
        bounds.extend(
            INNER_RESPONSE_DIFFERENCE_EXPONENTS
                .iter()
                .filter_map(|&exponent| {
                    exact_inner_collision_bound(
                        akita_challenges::D64_SELECTIVE_L2_CHALLENGE_CONFIG.l1_norm() as u128,
                        exponent,
                    )
                }),
        );
    }
    let max_bound = max_inner_coeff_linf_bound(modulus_profile);
    bounds.retain(|&bound| bound <= max_bound);
    bounds.sort_unstable();
    bounds.dedup();
    bounds
}

fn role_bound_supported(
    role: SisMatrixRole,
    modulus_profile: SisModulusProfileId,
    ring_dimension: u32,
    coeff_linf_bound: u128,
) -> bool {
    match role {
        SisMatrixRole::Inner => {
            inner_bound_supported(modulus_profile, ring_dimension, coeff_linf_bound)
        }
        SisMatrixRole::Outer | SisMatrixRole::Open => {
            GADGET_COEFF_LINF_ANCHORS.contains(&coeff_linf_bound)
        }
    }
}

/// Whether generated SIS security floors cover one role/profile/dimension.
#[must_use]
pub fn sis_role_dimension_supported(
    role: SisMatrixRole,
    modulus_profile: SisModulusProfileId,
    ring_dimension: u32,
) -> bool {
    role_ring_dimensions_for_tier(
        protocol_dispatch_tier_for_sis_profile(modulus_profile),
        dispatch_role(role),
    )
    .contains(&(ring_dimension as usize))
}

/// Return whether the exact role cell is part of the canonical coverage.
///
/// The function is deliberately role aware. It does not form a product of
/// independent dimension and bound lists for one shared table.
#[must_use]
pub fn sis_role_cell(
    role: SisMatrixRole,
    modulus_profile: SisModulusProfileId,
    ring_dimension: u32,
    coeff_linf_bound: u128,
) -> Option<SisRoleCell> {
    let trivial_collision_bound = (modulus_profile.modulus() - 1) / 2;
    if !sis_role_dimension_supported(role, modulus_profile, ring_dimension)
        || !role_bound_supported(role, modulus_profile, ring_dimension, coeff_linf_bound)
        || coeff_linf_bound >= trivial_collision_bound
    {
        return None;
    }
    Some(SisRoleCell {
        role,
        modulus_profile,
        ring_dimension,
        coeff_linf_bound,
        max_module_rank: SIS_MAX_MODULE_RANK,
        required_max_width: SIS_REQUIRED_MAX_WIDTH,
    })
}

/// Enumerate every exact production matrix-role coverage cell.
pub fn sis_role_cells() -> Vec<SisRoleCell> {
    let profiles = [
        SisModulusProfileId::Q32Offset99,
        SisModulusProfileId::Q64Offset59,
        SisModulusProfileId::Q128OffsetA7F7,
    ];
    let mut cells = Vec::new();
    for role in SIS_MATRIX_ROLES.iter().copied() {
        for profile in profiles {
            for &dimension in role_ring_dimensions_for_tier(
                protocol_dispatch_tier_for_sis_profile(profile),
                dispatch_role(role),
            ) {
                let bounds = match role {
                    SisMatrixRole::Inner => inner_coeff_linf_bounds(profile, dimension as u32),
                    SisMatrixRole::Outer | SisMatrixRole::Open => {
                        GADGET_COEFF_LINF_ANCHORS.to_vec()
                    }
                };
                cells.extend(
                    bounds
                        .into_iter()
                        .filter_map(|bound| sis_role_cell(role, profile, dimension as u32, bound)),
                );
            }
        }
    }
    cells
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_inner_bound_union_is_sorted_complete_and_capped() {
        assert_eq!(MAX_INNER_RESPONSE_DIFFERENCE, (1u128 << 33) - 1);

        let mut expected = Vec::new();
        for challenge_l1 in [14, 16, 19, 23, 31, 51, 53] {
            for &exponent in INNER_RESPONSE_DIFFERENCE_EXPONENTS {
                expected.push(
                    exact_inner_collision_bound(challenge_l1, exponent)
                        .expect("production exact A bound"),
                );
            }
        }
        expected.sort_unstable();
        expected.dedup();
        let mut actual = sis_role_cells()
            .into_iter()
            .filter(|cell| {
                cell.role == SisMatrixRole::Inner
                    && cell.modulus_profile == SisModulusProfileId::Q64Offset59
            })
            .map(|cell| cell.coeff_linf_bound)
            .collect::<Vec<_>>();
        actual.sort_unstable();
        actual.dedup();
        assert!(actual.is_sorted());
        assert_eq!(actual.len(), 140);
        assert_eq!(expected, actual);
    }

    #[test]
    fn dimension_aware_exact_inner_coverage_has_the_expected_size() {
        let cells = sis_role_cells();
        let count = |profile| {
            cells
                .iter()
                .filter(|cell| cell.role == SisMatrixRole::Inner && cell.modulus_profile == profile)
                .count()
        };
        assert_eq!(count(SisModulusProfileId::Q32Offset99), 215);
        assert_eq!(count(SisModulusProfileId::Q64Offset59), 440);
        assert_eq!(count(SisModulusProfileId::Q128OffsetA7F7), 320);
    }

    #[test]
    fn selective_l2_l1_mass_is_only_an_exact_d64_inner_target() {
        let selective_t33 = exact_inner_collision_bound(53, 33).unwrap();
        for profile in [
            SisModulusProfileId::Q32Offset99,
            SisModulusProfileId::Q64Offset59,
            SisModulusProfileId::Q128OffsetA7F7,
        ] {
            let expected = profile != SisModulusProfileId::Q32Offset99;
            assert_eq!(
                sis_role_cell(SisMatrixRole::Inner, profile, 64, selective_t33).is_some(),
                expected,
            );
            assert!(sis_role_cell(SisMatrixRole::Inner, profile, 128, selective_t33).is_none());
        }
    }
}
