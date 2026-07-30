//! Runtime helpers for materializing cataloged multi-group root precommits.

use akita_challenges::{SparseChallengeConfig, TensorChallengeShape};
use akita_field::AkitaError;
use akita_types::sis::{
    decomposed_s_block_ring_count, decomposed_t_ring_count, fold_witness_digit_plan,
    num_digits_inner, num_digits_open, rounded_up_collision_inf_norm, rounded_up_role_a_inf_norm,
    FoldChallengeNorms, FoldWitnessLinfCapConfig, FoldWitnessNorms, InnerCommitMatrixParams,
    OuterCommitMatrixParams, SisMatrixRole,
};
use akita_types::{
    AkitaScheduleLookupKey, CommittedGroupDescriptor, DecompositionParams, PrecommittedLevelParams,
};

use crate::PlannerPolicy;

#[derive(Clone, Debug)]
struct PrecommittedGroupSeed {
    layout: CommittedGroupDescriptor,
    inner_commit_matrix: InnerCommitMatrixParams,
    outer_commit_matrix: OuterCommitMatrixParams,
}

fn freeze_precommitted_group_layout(
    layout: &CommittedGroupDescriptor,
    policy: &PlannerPolicy,
) -> Result<PrecommittedGroupSeed, AkitaError> {
    layout.validate_frozen_precommit(policy.decomposition.field_bits())?;

    let d_a = layout.inner_commit_matrix.ring_dimension();
    let d_b = layout.outer_commit_matrix.ring_dimension();
    let source = akita_types::GroupSource::from_encoding(layout.encoding);
    let witness_decomp = source.decomposition(DecompositionParams {
        log_basis: layout.log_basis_inner,
        ..policy.decomposition
    });
    let outer_decomp = DecompositionParams {
        log_basis: layout.log_basis_outer,
        ..policy.decomposition
    };
    let num_digits_inner = num_digits_inner(witness_decomp, true);
    let num_digits_outer = num_digits_open(outer_decomp);
    let width_s =
        decomposed_s_block_ring_count(layout.num_positions_per_block, num_digits_inner)
            .ok_or_else(|| AkitaError::InvalidSetup("multi-group A width overflow".to_string()))?;
    if num_digits_inner != layout.num_digits_inner
        || num_digits_outer != layout.num_digits_outer
        || width_s != layout.inner_commit_matrix.input_width()
    {
        return Err(AkitaError::InvalidSetup(
            "precommitted profile digit depths do not match its source encoding".to_string(),
        ));
    }
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
        num_digits_outer,
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
    if log_basis_open < group.layout.log_basis_inner
        || log_basis_open < group.layout.log_basis_outer
    {
        return Err(AkitaError::InvalidSetup(
            "certified opening basis must dominate precommitted inner/outer bases".to_string(),
        ));
    }
    let open_decomp = DecompositionParams {
        log_basis: log_basis_open,
        ..policy.decomposition
    };
    let num_digits_open = num_digits_open(open_decomp);
    let source = akita_types::GroupSource::from_encoding(group.layout.encoding);
    let onehot_chunk_size = source.sparse_chunk_size();
    let challenge_shape = TensorChallengeShape::Flat;
    let challenge = FoldChallengeNorms {
        infinity_norm: challenge_shape.effective_infinity_norm(ring_challenge_cfg) as u128,
        l1_norm: challenge_shape.effective_l1_mass(ring_challenge_cfg) as u128,
    };
    let witness = FoldWitnessNorms::new(
        group.layout.log_basis_inner,
        group.layout.inner_commit_matrix.ring_dimension(),
        if onehot_chunk_size == 0 {
            1
        } else {
            onehot_chunk_size
        },
        onehot_chunk_size > 0,
    );
    let cap_config = FoldWitnessLinfCapConfig::for_fold_level(
        ring_challenge_cfg,
        challenge_shape,
        group.layout.inner_commit_matrix.ring_dimension(),
        group.inner_commit_matrix.input_width(),
    )?;
    let (num_digits_fold_one, _) = fold_witness_digit_plan(
        group.layout.num_live_blocks,
        group.layout.group.num_polynomials(),
        policy.decomposition.field_bits(),
        log_basis_open,
        challenge,
        witness,
        &cap_config,
    )?;
    let witness_decomposition = source.decomposition(DecompositionParams {
        log_basis: group.layout.log_basis_inner,
        ..policy.decomposition
    });
    let required_a_bound = rounded_up_role_a_inf_norm(
        policy.sis_security_policy,
        policy.sis_table_digest,
        policy.sis_modulus_profile,
        group.layout.inner_commit_matrix.ring_dimension(),
        witness_decomposition,
        log_basis_open,
        ring_challenge_cfg,
        challenge_shape,
        true,
        onehot_chunk_size,
        policy.ring_subfield_norm_bound,
        group.layout.num_live_blocks,
        group.layout.group.num_polynomials(),
        group.inner_commit_matrix.input_width() as u64,
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
        source,
        log_basis_open,
        fold_challenge_config: *ring_challenge_cfg,
        num_digits_open,
        num_digits_fold_one,
    })
}

fn multi_group_root_precommitted_group_seeds(
    key: &AkitaScheduleLookupKey,
    policy: &PlannerPolicy,
) -> Result<Vec<PrecommittedGroupSeed>, AkitaError> {
    if key.precommitteds.is_empty() {
        return Err(AkitaError::InvalidSetup(
            "multi-group root params require at least one precommitted group".to_string(),
        ));
    }

    key.precommitteds
        .iter()
        .map(|layout| freeze_precommitted_group_layout(layout, policy))
        .collect::<Result<Vec<_>, _>>()
}

pub(crate) fn multi_group_root_precommitted_groups_for_open_basis(
    key: &AkitaScheduleLookupKey,
    policy: &PlannerPolicy,
    ring_challenge_config: &dyn Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    log_basis_open: u32,
) -> Result<(Vec<PrecommittedLevelParams>, usize), AkitaError> {
    let seeds = multi_group_root_precommitted_group_seeds(key, policy)?;
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
            .checked_add(group.d_segment_width(policy.ring_dimension)?)
            .ok_or_else(|| AkitaError::InvalidSetup("multi-group D width overflow".to_string()))?;
    }
    Ok((groups, d_width))
}
