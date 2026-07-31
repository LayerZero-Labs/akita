//! Canonical security audit for one fully expanded schedule row.

use akita_field::AkitaError;
use akita_types::sis::{
    decomposed_t_ring_count, decomposed_w_ring_count, fold_witness_digit_plan, num_digits_inner,
    num_digits_open, rounded_up_collision_inf_norm, rounded_up_role_a_inf_norm, FoldChallengeNorms,
    FoldWitnessLinfCapConfig, FoldWitnessNorms, InnerCommitMatrixParams, OpenCommitMatrixParams,
    OuterCommitMatrixParams, SisMatrixRole, SisTableKey,
};
use akita_types::{
    shared_d_digit_log_basis, validate_role_dims, CommittedGroupBatchProfile, CommittedGroupParams,
    DecompositionParams, FoldSchedule, PrecommittedLevelParams, TerminalCommittedGroupParams,
    TerminalResponseShape,
};

use crate::runtime::validate_policy;
use crate::PlannerPolicy;

fn invalid(label: &str, detail: &str) -> AkitaError {
    AkitaError::InvalidSetup(format!("{label}: {detail}"))
}

fn audit_sis_key(
    label: &str,
    key: SisTableKey,
    expected_role: SisMatrixRole,
    policy: &PlannerPolicy,
) -> Result<(), AkitaError> {
    if key.policy != policy.sis_security_policy
        || key.table_digest != policy.sis_table_digest
        || key.modulus_profile != policy.sis_modulus_profile
        || key.role != expected_role
    {
        return Err(invalid(
            label,
            "matrix SIS policy, table, modulus profile, or role disagrees with the catalog policy",
        ));
    }
    Ok(())
}

fn audit_inner_matrix(
    label: &str,
    matrix: &InnerCommitMatrixParams,
    policy: &PlannerPolicy,
) -> Result<(), AkitaError> {
    matrix.validate()?;
    audit_sis_key(label, matrix.sis_table_key(), SisMatrixRole::Inner, policy)
}

fn audit_outer_matrix(
    label: &str,
    matrix: &OuterCommitMatrixParams,
    policy: &PlannerPolicy,
) -> Result<(), AkitaError> {
    matrix.validate()?;
    audit_sis_key(label, matrix.sis_table_key(), SisMatrixRole::Outer, policy)
}

fn audit_open_matrix(
    label: &str,
    matrix: &OpenCommitMatrixParams,
    policy: &PlannerPolicy,
) -> Result<(), AkitaError> {
    matrix.validate()?;
    audit_sis_key(label, matrix.sis_table_key(), SisMatrixRole::Open, policy)
}

fn checked_projected_width(
    label: &str,
    width: usize,
    source_ring_dimension: usize,
    target_ring_dimension: usize,
) -> Result<usize, AkitaError> {
    if target_ring_dimension == 0 || !source_ring_dimension.is_multiple_of(target_ring_dimension) {
        return Err(invalid(label, "invalid matrix carrier projection"));
    }
    width
        .checked_mul(source_ring_dimension / target_ring_dimension)
        .ok_or_else(|| invalid(label, "projected matrix width overflow"))
}

fn audit_bound(label: &str, declared: u128, required: Option<u128>) -> Result<(), AkitaError> {
    let required = required.ok_or_else(|| invalid(label, "accepted envelope has no SIS row"))?;
    if declared < required {
        return Err(invalid(
            label,
            &format!("declared coefficient bound {declared} is below required bound {required}"),
        ));
    }
    Ok(())
}

fn audit_precommitted_group(
    label: &str,
    params: &PrecommittedLevelParams,
    policy: &PlannerPolicy,
) -> Result<(), AkitaError> {
    params.validate()?;
    audit_inner_matrix(label, &params.layout.inner_commit_matrix, policy)?;
    audit_outer_matrix(label, &params.layout.outer_commit_matrix, policy)?;

    let expected_open_digits = num_digits_open(DecompositionParams {
        log_basis: params.log_basis_open,
        ..policy.decomposition
    });
    if params.num_digits_open != expected_open_digits {
        return Err(invalid(
            label,
            "opening digit depth is not canonical for the field and basis",
        ));
    }

    audit_bound(
        label,
        params.layout.inner_commit_matrix.coeff_linf_bound(),
        rounded_up_role_a_inf_norm(
            policy.sis_security_policy,
            policy.sis_table_digest,
            policy.sis_modulus_profile,
            params.layout.inner_commit_matrix.ring_dimension(),
            params.log_basis_open,
            &params.fold_challenge_config,
            akita_challenges::TensorChallengeShape::Flat,
            params.num_digits_fold,
            policy.ring_subfield_norm_bound,
        ),
    )?;
    audit_bound(
        label,
        params.layout.outer_commit_matrix.coeff_linf_bound(),
        rounded_up_collision_inf_norm(
            policy.sis_security_policy,
            policy.sis_modulus_profile,
            SisMatrixRole::Outer,
            params.layout.outer_commit_matrix.ring_dimension(),
            params.log_basis_open,
        ),
    )
}

fn expected_d_width(
    label: &str,
    params: &CommittedGroupParams,
    num_claims: usize,
) -> Result<usize, AkitaError> {
    let dims = params.role_dims();
    let main_width =
        decomposed_w_ring_count(params.num_digits_open, params.num_live_blocks, num_claims)
            .ok_or_else(|| invalid(label, "main D width overflow"))?;
    let mut width = checked_projected_width(label, main_width, dims.d_a(), dims.d_d())?;

    for group in &params.precommitted_groups {
        width = width
            .checked_add(group.d_segment_width(dims.d_d())?)
            .ok_or_else(|| invalid(label, "precommitted D width overflow"))?;
    }
    if let Some(prefix) = &params.setup_prefix {
        width = width
            .checked_add(prefix.commitment_params.d_segment_width(dims.d_d())?)
            .ok_or_else(|| invalid(label, "setup-prefix D width overflow"))?;
    }
    Ok(width)
}

fn audit_committed_params(
    label: &str,
    params: &CommittedGroupParams,
    num_claims: usize,
    policy: &PlannerPolicy,
) -> Result<(), AkitaError> {
    if num_claims == 0 || params.num_fold_claims != num_claims {
        return Err(invalid(
            label,
            "fold claim count does not match the committed group",
        ));
    }
    if params.field_bits_hint != policy.decomposition.field_bits() {
        return Err(invalid(
            label,
            "field-width hint disagrees with the catalog policy",
        ));
    }
    params.validate_block_geometry()?;
    params.witness_chunk.validate()?;
    validate_role_dims(params.role_dims())?;
    params
        .fold_challenge_config
        .validate_for_ring_dim(params.d_a())
        .map_err(|message| invalid(label, message))?;
    audit_inner_matrix(label, &params.inner_commit_matrix, policy)?;
    audit_outer_matrix(label, &params.outer_commit_matrix, policy)?;
    audit_open_matrix(label, &params.open_commit_matrix, policy)?;

    let expected_outer_digits = num_digits_open(DecompositionParams {
        log_basis: params.log_basis_outer,
        ..policy.decomposition
    });
    let expected_open_digits = num_digits_open(DecompositionParams {
        log_basis: params.log_basis_open,
        ..policy.decomposition
    });
    if params.num_digits_inner == 0
        || params.num_digits_fold == 0
        || params.num_digits_outer != expected_outer_digits
        || params.num_digits_open != expected_open_digits
    {
        return Err(invalid(label, "digit depths are missing or noncanonical"));
    }

    let expected_cap_config = FoldWitnessLinfCapConfig::for_fold_level(
        &params.fold_challenge_config,
        params.fold_challenge_shape,
        params.d_a(),
        params.inner_width(),
    )?;
    if params.fold_linf_cap_config != expected_cap_config {
        return Err(invalid(
            label,
            "fold L-infinity cap configuration is not canonical",
        ));
    }

    let dims = params.role_dims();
    let expected_a_width = params
        .num_positions_per_block
        .checked_mul(params.num_digits_inner)
        .ok_or_else(|| invalid(label, "A width overflow"))?;
    let native_b_width = decomposed_t_ring_count(
        params.inner_commit_matrix.output_rank(),
        params.num_digits_outer,
        params.num_live_blocks,
        num_claims,
    )
    .ok_or_else(|| invalid(label, "B width overflow"))?;
    let expected_b_width = checked_projected_width(label, native_b_width, dims.d_a(), dims.d_b())?;
    let expected_d_width = expected_d_width(label, params, num_claims)?;
    if params.inner_commit_matrix.input_width() != expected_a_width
        || params.outer_commit_matrix.input_width() != expected_b_width
        || params.open_commit_matrix.input_width() != expected_d_width
    {
        return Err(invalid(
            label,
            "A, B, or D width disagrees with the accepted digit geometry",
        ));
    }

    audit_bound(
        label,
        params.inner_commit_matrix.coeff_linf_bound(),
        rounded_up_role_a_inf_norm(
            policy.sis_security_policy,
            policy.sis_table_digest,
            policy.sis_modulus_profile,
            dims.d_a(),
            params.log_basis_open,
            &params.fold_challenge_config,
            params.fold_challenge_shape,
            params.num_digits_fold,
            policy.ring_subfield_norm_bound,
        ),
    )?;
    audit_bound(
        label,
        params.outer_commit_matrix.coeff_linf_bound(),
        rounded_up_collision_inf_norm(
            policy.sis_security_policy,
            policy.sis_modulus_profile,
            SisMatrixRole::Outer,
            dims.d_b(),
            params.log_basis_outer,
        ),
    )?;
    audit_bound(
        label,
        params.open_commit_matrix.coeff_linf_bound(),
        rounded_up_collision_inf_norm(
            policy.sis_security_policy,
            policy.sis_modulus_profile,
            SisMatrixRole::Open,
            dims.d_d(),
            shared_d_digit_log_basis(params.log_basis_open, &params.precommitted_groups),
        ),
    )
}

fn audit_terminal(
    params: &TerminalCommittedGroupParams,
    sparse: &akita_challenges::SparseChallengeConfig,
    response_shape: &TerminalResponseShape,
    policy: &PlannerPolicy,
) -> Result<(), AkitaError> {
    let label = "terminal fold";
    audit_inner_matrix(label, &params.inner_commit_matrix, policy)?;
    sparse
        .validate_for_ring_dim(params.d_a())
        .map_err(|message| invalid(label, message))?;
    if params.num_live_ring_elements_per_claim == 0
        || params.num_positions_per_block == 0
        || !params.num_positions_per_block.is_power_of_two()
        || params.num_live_blocks
            != params
                .num_live_ring_elements_per_claim
                .div_ceil(params.num_positions_per_block)
    {
        return Err(invalid(label, "invalid terminal block geometry"));
    }

    let expected_digits = num_digits_inner(
        DecompositionParams {
            log_basis: params.log_basis_inner,
            ..policy.decomposition
        },
        false,
    );
    let expected_width = params
        .num_positions_per_block
        .checked_mul(expected_digits)
        .ok_or_else(|| invalid(label, "A width overflow"))?;
    if params.num_digits_inner != expected_digits
        || params.inner_commit_matrix.input_width() != expected_width
    {
        return Err(invalid(
            label,
            "terminal digits or A width are not canonical",
        ));
    }

    let expected_cap_config = FoldWitnessLinfCapConfig::for_fold_level(
        sparse,
        akita_challenges::TensorChallengeShape::Flat,
        params.d_a(),
        params.inner_width(),
    )?;
    if params.fold_linf_cap_config != expected_cap_config {
        return Err(invalid(
            label,
            "terminal fold L-infinity cap configuration is not canonical",
        ));
    }
    let (fold_digit_depth, _) = fold_witness_digit_plan(
        params.num_live_blocks,
        1,
        policy.decomposition.field_bits(),
        params.log_basis_inner,
        FoldChallengeNorms::new(sparse, akita_challenges::TensorChallengeShape::Flat),
        FoldWitnessNorms::bounded(params.log_basis_inner, params.d_a()),
        &expected_cap_config,
    )?;
    audit_bound(
        label,
        params.inner_commit_matrix.coeff_linf_bound(),
        rounded_up_role_a_inf_norm(
            policy.sis_security_policy,
            policy.sis_table_digest,
            policy.sis_modulus_profile,
            params.d_a(),
            params.log_basis_inner,
            sparse,
            akita_challenges::TensorChallengeShape::Flat,
            fold_digit_depth,
            policy.ring_subfield_norm_bound,
        ),
    )?;

    let response_policy = params.response_linf_policy(sparse)?;
    if *response_shape != TerminalResponseShape::derive(params, response_policy.admission_cap)? {
        return Err(invalid(
            label,
            "terminal response shape disagrees with the accepted response cap",
        ));
    }
    Ok(())
}

/// Re-audit one complete expanded row against the policy the verifier trusts.
pub(crate) fn audit_resolved_schedule(
    profiles: &CommittedGroupBatchProfile,
    schedule: &FoldSchedule,
    policy: &PlannerPolicy,
) -> Result<(), AkitaError> {
    validate_policy(policy)?;
    profiles.validate(policy.decomposition.field_bits())?;
    schedule.validate_structure()?;

    let root = &schedule.root.params;
    let final_params = &root.final_group.commitment;
    if profiles.final_group
        != akita_types::CommittedGroupProfile::from_params(profiles.final_group.group, final_params)
        || profiles.precommitteds.len() != root.precommitted_groups.len()
        || root.open_commit_matrix != final_params.open_commit_matrix
        || final_params.precommitted_groups.len() != root.precommitted_groups.len()
    {
        return Err(invalid(
            "root fold",
            "ordered profiles or shared D metadata disagree with the expanded row",
        ));
    }

    for (index, ((profile, root_group), params_group)) in profiles
        .precommitteds
        .iter()
        .zip(&root.precommitted_groups)
        .zip(&final_params.precommitted_groups)
        .enumerate()
    {
        if profile != &root_group.descriptor
            || profile != &root_group.commitment.layout
            || root_group.commitment != *params_group
        {
            return Err(invalid(
                &format!("root precommitted group {index}"),
                "profile and consuming parameters disagree",
            ));
        }
        audit_precommitted_group(
            &format!("root precommitted group {index}"),
            &root_group.commitment,
            policy,
        )?;
    }

    audit_committed_params(
        "root final group",
        final_params,
        profiles.final_group.group.num_polynomials(),
        policy,
    )?;
    for (index, step) in schedule.recursive_folds.iter().enumerate() {
        audit_committed_params(
            &format!("recursive fold {index}"),
            &step.params.witness,
            step.params.witness.num_fold_claims,
            policy,
        )?;
        if step.params.open_commit_matrix != step.params.witness.open_commit_matrix {
            return Err(invalid(
                &format!("recursive fold {index}"),
                "shared D matrix disagrees with the witness parameters",
            ));
        }
    }
    audit_terminal(
        &schedule.terminal.params.witness,
        &schedule.terminal.params.sparse_challenge_config,
        &schedule.terminal.params.response_shape,
        policy,
    )
}
