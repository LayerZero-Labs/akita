use super::*;

/// Inputs for synthetic setup-prefix commitment planning.
pub struct SetupPrefixPlanRequest<'a> {
    pub policy: &'a PlannerPolicy,
    pub ring_challenge: &'a SparseChallengeConfig,
    pub fold_shape: TensorChallengeShape,
    pub log_basis_outer: u32,
    pub log_basis_open: u32,
    pub prefix_field_elements: usize,
    pub num_chunks: usize,
    pub outer_ring_dimension: usize,
}

/// Plan one synthetic setup-prefix commitment used by test and profile fixtures.
///
/// # Errors
///
/// Returns an error for malformed policy or dimensions, or when no audited
/// secure setup-prefix geometry exists.
pub fn plan_setup_prefix_commitment(
    request: SetupPrefixPlanRequest<'_>,
) -> Result<PrecommittedLevelParams, AkitaError> {
    validate_policy(request.policy)?;
    candidate::derive_setup_prefix_group(
        request.policy,
        request.ring_challenge,
        request.fold_shape,
        request.log_basis_outer,
        request.log_basis_open,
        request.prefix_field_elements,
        request.num_chunks,
        request.outer_ring_dimension,
    )?
    .ok_or_else(|| {
        AkitaError::UnsupportedSchedule(format!(
            "no setup-prefix commitment at A{}/B{} for n_prefix={}",
            request.policy.ring_dimension,
            request.outer_ring_dimension,
            request.prefix_field_elements
        ))
    })
}
