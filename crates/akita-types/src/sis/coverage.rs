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

/// Maximum module rank searched for every production scalar SIS cell.
/// This is the audited ADPS16 table-generation domain from
/// `specs/sis-quantum128-scalar-n-table.md`.
pub const SIS_MAX_MODULE_RANK: u32 = 20;

/// Per-cell scalar-width search cap used by the production generator.
/// A required runtime width above this cap fails generation; the value is not
/// itself a security boundary.
pub const SIS_REQUIRED_MAX_WIDTH: u64 = 6_400_000_000_000;

fn role_dimensions(
    role: SisMatrixRole,
    modulus_profile: SisModulusProfileId,
) -> impl Iterator<Item = u32> {
    let base = match role {
        SisMatrixRole::Inner => A_ROLE_RING_DIMS,
        SisMatrixRole::Outer | SisMatrixRole::Open => BD_ROLE_RING_DIMS,
    };
    let q128_inner_extension = (role == SisMatrixRole::Inner
        && modulus_profile == SisModulusProfileId::Q128OffsetA7F7)
        .then_some(512);
    base.iter().copied().chain(q128_inner_extension)
}

fn role_bounds(role: SisMatrixRole) -> &'static [u128] {
    match role {
        SisMatrixRole::Inner => COEFF_LINF_BUCKETS,
        SisMatrixRole::Outer | SisMatrixRole::Open => GADGET_COEFF_LINF_ANCHORS,
    }
}

/// Whether generated SIS security floors cover one role/profile/dimension.
#[must_use]
pub fn sis_role_dimension_supported(
    role: SisMatrixRole,
    modulus_profile: SisModulusProfileId,
    ring_dimension: u32,
) -> bool {
    role_dimensions(role, modulus_profile).any(|dimension| dimension == ring_dimension)
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
    let bounds = role_bounds(role);
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
            for dimension in role_dimensions(role, profile) {
                let bounds = role_bounds(role);
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
