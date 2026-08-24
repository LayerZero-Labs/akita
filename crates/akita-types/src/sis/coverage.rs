//! Exact role, modulus, dimension, and coefficient-bound coverage for SIS rows.

use super::ajtai_key::{SisMatrixRole, SisModulusProfileId, COEFF_LINF_BUCKETS};
use crate::dispatch::{protocol_dispatch_tier_for_sis_profile, role_ring_dimensions_for_tier};
use crate::RingRole;

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
const LARGE_FIELD_MAX_INNER_COEFF_LINF_BOUND: u128 = 4_294_967_295;

const fn dispatch_role(role: SisMatrixRole) -> RingRole {
    match role {
        SisMatrixRole::Inner => RingRole::Inner,
        SisMatrixRole::Outer => RingRole::Outer,
        SisMatrixRole::Open => RingRole::Opening,
    }
}

fn role_bounds(role: SisMatrixRole) -> &'static [u128] {
    match role {
        SisMatrixRole::Inner => COEFF_LINF_BUCKETS,
        SisMatrixRole::Outer | SisMatrixRole::Open => GADGET_COEFF_LINF_ANCHORS,
    }
}

const fn max_inner_coeff_linf_bound(modulus_profile: SisModulusProfileId) -> u128 {
    match modulus_profile {
        SisModulusProfileId::Q32Offset99 => Q32_MAX_INNER_COEFF_LINF_BOUND,
        SisModulusProfileId::Q64Offset59 | SisModulusProfileId::Q128OffsetA7F7 => {
            LARGE_FIELD_MAX_INNER_COEFF_LINF_BOUND
        }
    }
}

fn role_bound_supported(
    role: SisMatrixRole,
    modulus_profile: SisModulusProfileId,
    coeff_linf_bound: u128,
) -> bool {
    role_bounds(role).contains(&coeff_linf_bound)
        && (role != SisMatrixRole::Inner
            || coeff_linf_bound <= max_inner_coeff_linf_bound(modulus_profile))
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
        || !role_bound_supported(role, modulus_profile, coeff_linf_bound)
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
                let bounds = role_bounds(role);
                cells.extend(
                    bounds
                        .iter()
                        .filter_map(|&bound| sis_role_cell(role, profile, dimension as u32, bound)),
                );
            }
        }
    }
    cells
}
