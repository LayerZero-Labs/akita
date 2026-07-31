use super::*;

pub(super) fn sis_key_at_dimension(
    policy: &PlannerPolicy,
    role: akita_types::SisMatrixRole,
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

pub(super) fn projected_collision_role_price(
    policy: &PlannerPolicy,
    role: akita_types::SisMatrixRole,
    source_dimension: usize,
    role_dimension: usize,
    native_width: usize,
    log_basis: u32,
) -> Option<(SisTableKey, usize)> {
    if role == akita_types::SisMatrixRole::Inner
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
    let physical_width = akita_types::sis::projected_role_ring_count(
        source_dimension,
        role_dimension,
        native_width,
    )?;
    Some((
        sis_key_at_dimension(policy, role, role_dimension, coeff_linf_bound),
        physical_width,
    ))
}

#[cfg(all(test, feature = "catalog-gen"))]
mod tests {
    use super::*;
    use akita_config::{policy_of, proof_optimized::fp128::D256OneHot};

    #[test]
    fn projected_width_uses_exact_source_ratio() {
        let policy = policy_of::<D256OneHot>();
        let (_, width) = projected_collision_role_price(
            &policy,
            akita_types::SisMatrixRole::Outer,
            256,
            64,
            7,
            policy.decomposition.log_basis,
        )
        .expect("current SIS policy covers the projected role");
        assert_eq!(width, 28);
    }
}
