//! Runtime helpers for materializing cataloged multi-group root precommits.

use akita_challenges::SparseChallengeConfig;
use akita_field::AkitaError;
use akita_types::sis::{
    decomposed_t_ring_count, num_digits_open, rounded_up_collision_inf_norm,
    rounded_up_role_a_inf_norm, InnerCommitMatrixParams, OuterCommitMatrixParams, SisMatrixRole,
};
use akita_types::{
    AkitaScheduleLookupKey, CommittedGroupProfile, DecompositionParams, PrecommittedLevelParams,
};

use crate::generated::GeneratedRootPrecommittedGroup;
use crate::PlannerPolicy;

#[derive(Clone, Debug)]
struct PrecommittedGroupSeed {
    layout: CommittedGroupProfile,
    num_digits_fold: usize,
    inner_commit_matrix: InnerCommitMatrixParams,
    outer_commit_matrix: OuterCommitMatrixParams,
}

fn freeze_precommitted_group_layout(
    layout: &CommittedGroupProfile,
    generated: &GeneratedRootPrecommittedGroup,
    policy: &PlannerPolicy,
) -> Result<PrecommittedGroupSeed, AkitaError> {
    layout.validate_frozen_precommit(policy.decomposition.field_bits())?;

    let d_a = layout.inner_commit_matrix.ring_dimension();
    let d_b = layout.outer_commit_matrix.ring_dimension();
    let inner_commit_matrix = layout.inner_commit_matrix;

    let norm_t = rounded_up_collision_inf_norm(
        policy.sis_security_policy,
        policy.sis_modulus_profile,
        SisMatrixRole::Outer,
        d_b,
        layout.log_basis_outer,
    )
    .ok_or_else(|| AkitaError::InvalidSetup("no multi-group B-role norm".to_string()))?;
    let width_t = decomposed_t_ring_count(
        layout.inner_commit_matrix.output_rank(),
        layout.num_digits_outer,
        layout.num_live_blocks,
        layout.group.num_polynomials(),
    )
    .and_then(|width| width.checked_mul(d_a / d_b))
    .ok_or_else(|| AkitaError::InvalidSetup("setup B width overflow".to_string()))?;
    if layout.outer_commit_matrix.coeff_linf_bound() < norm_t {
        return Err(AkitaError::InvalidSetup(
            "precommitted group B bound is below the selected opening requirement".to_string(),
        ));
    }
    if width_t != layout.outer_commit_matrix.input_width() {
        return Err(AkitaError::InvalidSetup(
            "precommitted profile B width does not match its exact matrix".to_string(),
        ));
    }
    let outer_commit_matrix = layout.outer_commit_matrix;

    Ok(PrecommittedGroupSeed {
        layout: *layout,
        num_digits_fold: usize::try_from(generated.num_digits_fold).map_err(|_| {
            AkitaError::InvalidSetup(
                "generated precommitted fold depth does not fit the target platform".to_string(),
            )
        })?,
        inner_commit_matrix,
        outer_commit_matrix,
    })
}

fn materialize_precommitted_group_for_open_basis(
    group: &PrecommittedGroupSeed,
    policy: &PlannerPolicy,
    ring_challenge_cfg: &SparseChallengeConfig,
    log_basis_open: u32,
) -> Result<PrecommittedLevelParams, AkitaError> {
    if log_basis_open < group.layout.log_basis_outer {
        return Err(AkitaError::InvalidSetup(
            "certified opening basis must dominate the precommitted outer basis".to_string(),
        ));
    }
    let open_decomp = DecompositionParams {
        log_basis: log_basis_open,
        ..policy.decomposition
    };
    let num_digits_open = num_digits_open(open_decomp);
    let num_digits_fold = group.num_digits_fold;
    let required_a_bound = rounded_up_role_a_inf_norm(
        policy.sis_security_policy,
        policy.sis_table_digest,
        policy.sis_modulus_profile,
        group.layout.inner_commit_matrix.ring_dimension(),
        log_basis_open,
        ring_challenge_cfg,
        num_digits_fold,
        policy.ring_subfield_norm_bound,
    )
    .ok_or_else(|| AkitaError::InvalidSetup("no precommitted A-role norm".to_string()))?;
    if required_a_bound > group.inner_commit_matrix.coeff_linf_bound() {
        return Err(AkitaError::InvalidSetup(
            "precommitted A bound does not cover the certified opening basis".to_string(),
        ));
    }
    let required_b_bound = rounded_up_collision_inf_norm(
        policy.sis_security_policy,
        policy.sis_modulus_profile,
        SisMatrixRole::Outer,
        group.layout.outer_commit_matrix.ring_dimension(),
        log_basis_open,
    )
    .ok_or_else(|| AkitaError::InvalidSetup("no precommitted B-role norm".to_string()))?;
    if required_b_bound > group.outer_commit_matrix.coeff_linf_bound() {
        return Err(AkitaError::InvalidSetup(
            "precommitted B bound does not cover the certified opening basis".to_string(),
        ));
    }
    Ok(PrecommittedLevelParams {
        layout: group.layout,
        log_basis_open,
        fold_challenge_config: *ring_challenge_cfg,
        num_digits_open,
        num_digits_fold,
    })
}

fn multi_group_root_precommitted_group_seeds(
    key: &AkitaScheduleLookupKey,
    generated_groups: &[GeneratedRootPrecommittedGroup],
    policy: &PlannerPolicy,
) -> Result<Vec<PrecommittedGroupSeed>, AkitaError> {
    if key.precommitteds.is_empty() {
        return Err(AkitaError::InvalidSetup(
            "multi-group root params require at least one precommitted group".to_string(),
        ));
    }

    if key.precommitteds.len() != generated_groups.len() {
        return Err(AkitaError::InvalidSetup(
            "generated precommitted group count does not match the schedule key".to_string(),
        ));
    }
    key.precommitteds
        .iter()
        .zip(generated_groups)
        .map(|(layout, generated)| freeze_precommitted_group_layout(layout, generated, policy))
        .collect::<Result<Vec<_>, _>>()
}

pub(crate) fn multi_group_root_precommitted_groups_for_open_basis(
    key: &AkitaScheduleLookupKey,
    generated_groups: &[GeneratedRootPrecommittedGroup],
    policy: &PlannerPolicy,
    ring_challenge_config: &dyn Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    log_basis_open: u32,
    open_ring_dimension: usize,
) -> Result<(Vec<PrecommittedLevelParams>, usize), AkitaError> {
    let seeds = multi_group_root_precommitted_group_seeds(key, generated_groups, policy)?;
    let groups = seeds
        .iter()
        .map(|group| {
            let ring_challenge_cfg =
                ring_challenge_config(group.layout.inner_commit_matrix.ring_dimension())?;
            materialize_precommitted_group_for_open_basis(
                group,
                policy,
                &ring_challenge_cfg,
                log_basis_open,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut d_width = 0usize;
    for group in &groups {
        d_width = d_width
            .checked_add(group.d_segment_width(open_ring_dimension)?)
            .ok_or_else(|| AkitaError::InvalidSetup("multi-group D width overflow".to_string()))?;
    }
    Ok((groups, d_width))
}
