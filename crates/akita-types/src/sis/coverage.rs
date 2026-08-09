//! Exact role, modulus, dimension, and coefficient-bound coverage for SIS rows.

use super::ajtai_key::{SisMatrixRole, SisModulusProfileId, COEFF_LINF_BUCKETS};

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

/// Ring dimensions supported by A for every SIS modulus profile.
///
/// Q128 has the additional profile-specific `D = 512` cell enforced by
/// [`sis_role_cell`].
pub const A_ROLE_RING_DIMS: &[u32] = &[64, 128, 256];

/// Admitted B/D commitment-matrix dimensions.
pub const BD_ROLE_RING_DIMS: &[u32] = &[64, 128, 256];

/// Production matrix roles with checked-in coverage.
pub const SIS_MATRIX_ROLES: &[SisMatrixRole] = &[
    SisMatrixRole::Inner,
    SisMatrixRole::Outer,
    SisMatrixRole::Open,
];

/// Whether generated SIS security floors cover one role/profile/dimension.
#[must_use]
pub fn sis_role_dimension_supported(
    role: SisMatrixRole,
    modulus_profile: SisModulusProfileId,
    ring_dimension: u32,
) -> bool {
    match role {
        SisMatrixRole::Inner => {
            A_ROLE_RING_DIMS.contains(&ring_dimension)
                || (modulus_profile == SisModulusProfileId::Q128OffsetA7F7 && ring_dimension == 512)
        }
        SisMatrixRole::Outer | SisMatrixRole::Open => BD_ROLE_RING_DIMS.contains(&ring_dimension),
    }
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
    let bounds = match role {
        SisMatrixRole::Inner => COEFF_LINF_BUCKETS,
        SisMatrixRole::Outer | SisMatrixRole::Open => GADGET_COEFF_LINF_ANCHORS,
    };
    let trivial_collision_bound = (modulus_profile.modulus() - 1) / 2;
    if !sis_role_dimension_supported(role, modulus_profile, ring_dimension)
        || !bounds.contains(&coeff_linf_bound)
        || coeff_linf_bound >= trivial_collision_bound
    {
        return None;
    }
    Some(SisRoleCell {
        role,
        modulus_profile,
        ring_dimension,
        coeff_linf_bound,
        max_module_rank: 20,
        required_max_width: 6_400_000_000_000,
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
            let dimensions = match role {
                SisMatrixRole::Inner => A_ROLE_RING_DIMS,
                SisMatrixRole::Outer | SisMatrixRole::Open => BD_ROLE_RING_DIMS,
            };
            for dimension in dimensions.iter().copied().chain(
                (role == SisMatrixRole::Inner && profile == SisModulusProfileId::Q128OffsetA7F7)
                    .then_some(512),
            ) {
                let bounds = match role {
                    SisMatrixRole::Inner => COEFF_LINF_BUCKETS,
                    SisMatrixRole::Outer | SisMatrixRole::Open => GADGET_COEFF_LINF_ANCHORS,
                };
                cells.extend(
                    bounds
                        .iter()
                        .filter_map(|&bound| sis_role_cell(role, profile, dimension, bound)),
                );
            }
        }
    }
    cells
}
