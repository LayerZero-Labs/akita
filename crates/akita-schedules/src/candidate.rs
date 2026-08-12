//! Shared candidate-construction helpers.

use crate::runtime::PlannerPolicy;
use akita_types::sis::{projected_role_ring_count, rounded_up_collision_inf_norm, SisTableKey};
use akita_types::SisMatrixRole;

/// Construct the canonical SIS-table key for one role and ring dimension.
pub fn sis_key_at_dimension(
    policy: &PlannerPolicy,
    role: SisMatrixRole,
    ring_dimension: usize,
    coeff_linf_bound: u128,
) -> SisTableKey {
    SisTableKey {
        policy: policy.sis_security_policy,
        table_digest: policy.sis_table_digest,
        modulus_profile: policy.sis_modulus_profile,
        role,
        ring_dimension: ring_dimension as u32,
        coeff_linf_bound,
    }
}

/// Price one projected B/D collision role using canonical physical width and
/// coefficient bounds.
pub fn projected_collision_role_price(
    policy: &PlannerPolicy,
    role: SisMatrixRole,
    source_dimension: usize,
    role_dimension: usize,
    native_width: usize,
    log_basis: u32,
) -> Option<(SisTableKey, usize)> {
    if role == SisMatrixRole::Inner
        || role_dimension == 0
        || !source_dimension.is_multiple_of(role_dimension)
    {
        return None;
    }
    let coeff_linf_bound = rounded_up_collision_inf_norm(
        policy.sis_security_policy,
        policy.sis_modulus_profile,
        role,
        role_dimension,
        log_basis,
    )?;
    let physical_width = projected_role_ring_count(source_dimension, role_dimension, native_width)?;
    Some((
        sis_key_at_dimension(policy, role, role_dimension, coeff_linf_bound),
        physical_width,
    ))
}
