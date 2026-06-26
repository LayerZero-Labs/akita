//! Grouped root-batch schedule planning.

use akita_challenges::{SparseChallengeConfig, TensorChallengeShape};
use akita_field::AkitaError;
use akita_types::sis::{
    choose_op_norm_rejection_for_a_role, decomposed_s_block_ring_count, decomposed_t_ring_count,
    decomposed_w_ring_count, min_secure_rank, num_digits_fold, num_digits_open,
    num_digits_s_commit, rounded_up_collision_norm_t, rounded_up_collision_norm_w, AjtaiKeyParams,
    FoldChallengeNorms, FoldWitnessLinfCapConfig, FoldWitnessNorms,
};
use akita_types::{
    direct_witness_bytes, AkitaScheduleInputs, CleartextWitnessShape, CommitmentGroupLayout,
    DecompositionParams, DirectStep, GroupBatchAkitaScheduleLookupKey, GroupRootParams,
    LevelParams, Schedule, Step,
};

use crate::schedule_params::RingChallengeConfigFn;
use crate::PlannerPolicy;

fn group_root_params_from_layout(
    layout: &CommitmentGroupLayout,
    policy: &PlannerPolicy,
    ring_challenge_config: RingChallengeConfigFn<'_>,
    fold_challenge_shape: TensorChallengeShape,
    conservative_b_rank: bool,
) -> Result<GroupRootParams, AkitaError> {
    if conservative_b_rank {
        layout.validate_frozen_precommit(policy.ring_dimension, policy.basis_range.0)?;
    } else {
        layout.validate()?;
        layout.validate_root_geometry(policy.ring_dimension)?;
    }
    if policy.tiered {
        return Err(AkitaError::InvalidSetup(
            "tiered grouped roots are not supported; see specs/multi-group-batching.md".to_string(),
        ));
    }

    let ring_challenge_cfg = ring_challenge_config(policy.ring_dimension)?;
    let d = policy.ring_dimension;
    let family = policy.sis_family;
    let level_decomp = DecompositionParams {
        log_basis: layout.log_basis,
        ..policy.decomposition
    };
    let num_digits_commit = num_digits_s_commit(level_decomp, true);
    let num_digits_open = num_digits_open(level_decomp);
    let num_blocks = 1usize
        .checked_shl(layout.r_vars as u32)
        .ok_or_else(|| AkitaError::InvalidSetup("grouped root num_blocks overflow".to_string()))?;
    let block_len = 1usize
        .checked_shl(layout.m_vars as u32)
        .ok_or_else(|| AkitaError::InvalidSetup("grouped root block_len overflow".to_string()))?;

    let width_s = decomposed_s_block_ring_count(block_len, num_digits_commit)
        .ok_or_else(|| AkitaError::InvalidSetup("grouped A width overflow".to_string()))?;
    let (op_norm_rejection, norm_s, min_n_a) = choose_op_norm_rejection_for_a_role(
        family,
        d,
        level_decomp,
        &ring_challenge_cfg,
        fold_challenge_shape,
        true,
        policy.onehot_chunk_size,
        policy.ring_subfield_norm_bound,
        layout.r_vars,
        layout.key.num_polynomials,
        width_s as u64,
    )
    .ok_or_else(|| AkitaError::InvalidSetup("no grouped A-role norm".to_string()))?;
    if layout.n_a < min_n_a {
        return Err(AkitaError::InvalidSetup(
            "precommitted group A rank is below grouped root requirement".to_string(),
        ));
    }
    let a_key = AjtaiKeyParams::try_new(family, layout.n_a, width_s, norm_s, d)?;

    let b_norm_basis = if conservative_b_rank {
        policy.basis_range.1
    } else {
        layout.log_basis
    };
    let norm_t = rounded_up_collision_norm_t(family, d, b_norm_basis)
        .ok_or_else(|| AkitaError::InvalidSetup("no grouped B-role norm".to_string()))?;
    let width_t = decomposed_t_ring_count(
        layout.n_a,
        num_digits_open,
        num_blocks,
        layout.key.num_polynomials,
    )
    .ok_or_else(|| AkitaError::InvalidSetup("grouped B width overflow".to_string()))?;
    let min_n_b = min_secure_rank(family, d as u32, norm_t, width_t as u64)
        .ok_or_else(|| AkitaError::InvalidSetup("no grouped B-role rank".to_string()))?;
    let n_b = if conservative_b_rank {
        if layout.conservative_n_b < min_n_b {
            return Err(AkitaError::InvalidSetup(
                "precommitted group conservative B rank is below grouped root requirement"
                    .to_string(),
            ));
        }
        layout.conservative_n_b
    } else {
        min_n_b
    };
    let b_key = AjtaiKeyParams::try_new(family, n_b, width_t, norm_t, d)?;

    let fold_linf_cap_config = FoldWitnessLinfCapConfig::for_fold_level(
        &ring_challenge_cfg,
        fold_challenge_shape,
        d,
        op_norm_rejection,
        width_s,
    )?;
    let challenge = FoldChallengeNorms {
        infinity_norm: fold_challenge_shape.effective_infinity_norm(&ring_challenge_cfg) as u128,
        l1_norm: fold_challenge_shape.effective_l1_mass(&ring_challenge_cfg) as u128,
    };
    let onehot_chunk_size = if policy.decomposition.log_commit_bound == 1 {
        policy.onehot_chunk_size
    } else {
        0
    };
    let witness = FoldWitnessNorms::new(
        layout.log_basis,
        d,
        if onehot_chunk_size == 0 {
            1
        } else {
            onehot_chunk_size
        },
        onehot_chunk_size > 0,
    );
    let num_digits_fold_one = num_digits_fold(
        layout.r_vars,
        layout.key.num_polynomials,
        policy.decomposition.field_bits(),
        layout.log_basis,
        challenge,
        witness,
        fold_linf_cap_config,
    )?;

    Ok(GroupRootParams {
        layout: layout.clone(),
        a_key,
        b_key,
        num_blocks,
        block_len,
        num_digits_commit,
        num_digits_open,
        num_digits_fold_one,
    })
}

struct GroupedRootDirectCandidateCtx<'a> {
    policy: &'a PlannerPolicy,
    ring_challenge_cfg: &'a SparseChallengeConfig,
    fold_challenge_shape: TensorChallengeShape,
    precommitted_d_width: usize,
    precommitted_groups: &'a [GroupRootParams],
}

fn checked_score_add(lhs: u128, rhs: u128, context: &'static str) -> Result<u128, AkitaError> {
    lhs.checked_add(rhs)
        .ok_or_else(|| AkitaError::InvalidSetup(format!("{context} overflow")))
}

fn checked_score_mul(lhs: u128, rhs: usize, context: &'static str) -> Result<u128, AkitaError> {
    lhs.checked_mul(rhs as u128)
        .ok_or_else(|| AkitaError::InvalidSetup(format!("{context} overflow")))
}

fn root_direct_split_cost(
    n_a: usize,
    num_blocks: usize,
    block_len: usize,
    num_digits_commit: usize,
    num_digits_open: usize,
    num_digits_fold: usize,
    context: &'static str,
) -> Result<u128, AkitaError> {
    // Match `optimal_m_r_split`: opening `(1 + n_a) * delta_open * 2^r`
    // plus folded witness `delta_commit * delta_fold * 2^m`.
    let e_hat_cost = checked_score_mul(num_digits_open as u128, num_blocks, context)?;
    let t_hat_cost = checked_score_mul(num_digits_open as u128, n_a, context)?;
    let t_hat_cost = checked_score_mul(t_hat_cost, num_blocks, context)?;
    let opening_cost = checked_score_add(e_hat_cost, t_hat_cost, context)?;

    let z_hat_cost = checked_score_mul(num_digits_commit as u128, num_digits_fold, context)?;
    let z_hat_cost = checked_score_mul(z_hat_cost, block_len, context)?;

    checked_score_add(opening_cost, z_hat_cost, context)
}

fn grouped_root_direct_cost_score(
    params: &LevelParams,
    main_num_polys: usize,
    field_bits: u32,
) -> Result<u128, AkitaError> {
    let main_num_digits_fold = params.num_digits_fold(main_num_polys, field_bits)?;
    let mut total = root_direct_split_cost(
        params.a_key.row_len(),
        params.num_blocks,
        params.block_len,
        params.num_digits_commit,
        params.num_digits_open,
        main_num_digits_fold,
        "grouped main root-direct score",
    )?;

    for group in &params.precommitted_groups {
        let group_cost = root_direct_split_cost(
            group.a_key.row_len(),
            group.num_blocks,
            group.block_len,
            group.num_digits_commit,
            group.num_digits_open,
            group.num_digits_fold_one,
            "grouped precommitted root-direct score",
        )?;
        total = checked_score_add(total, group_cost, "grouped root-direct score total")?;
    }

    Ok(total)
}

fn grouped_root_direct_witness_len(
    key: &GroupBatchAkitaScheduleLookupKey,
) -> Result<usize, AkitaError> {
    let group_len = |num_polys: usize, num_vars: usize| -> Result<usize, AkitaError> {
        let per_poly_len = 1usize.checked_shl(num_vars as u32).ok_or_else(|| {
            AkitaError::InvalidSetup("grouped root-direct witness length overflow".to_string())
        })?;
        per_poly_len.checked_mul(num_polys).ok_or_else(|| {
            AkitaError::InvalidSetup("grouped root-direct witness length overflow".to_string())
        })
    };

    let mut total = group_len(key.main.num_polynomials, key.main.num_vars)?;
    for layout in &key.precommitteds {
        let precommitted_len = group_len(layout.key.num_polynomials, layout.key.num_vars)?;
        total = total.checked_add(precommitted_len).ok_or_else(|| {
            AkitaError::InvalidSetup("grouped root-direct witness length overflow".to_string())
        })?;
    }
    Ok(total)
}

fn grouped_root_direct_main_candidate(
    ctx: &GroupedRootDirectCandidateCtx<'_>,
    main_num_polys: usize,
    log_basis: u32,
    m_vars: usize,
    r_vars: usize,
) -> Result<Option<LevelParams>, AkitaError> {
    let policy = ctx.policy;
    let d = policy.ring_dimension;
    let family = policy.sis_family;
    let decomp = policy.decomposition;
    let level_decomp = DecompositionParams {
        log_basis,
        ..decomp
    };
    let num_digits_commit = num_digits_s_commit(level_decomp, true);
    let num_digits_open = num_digits_open(level_decomp);
    let Some(num_blocks) = 1usize.checked_shl(r_vars as u32) else {
        return Ok(None);
    };
    let Some(block_len) = 1usize.checked_shl(m_vars as u32) else {
        return Ok(None);
    };

    let Some(width_s) = decomposed_s_block_ring_count(block_len, num_digits_commit) else {
        return Ok(None);
    };
    let Some((op_norm_rejection, norm_s, n_a)) = choose_op_norm_rejection_for_a_role(
        family,
        d,
        level_decomp,
        ctx.ring_challenge_cfg,
        ctx.fold_challenge_shape,
        true,
        policy.onehot_chunk_size,
        policy.ring_subfield_norm_bound,
        r_vars,
        main_num_polys,
        width_s as u64,
    ) else {
        return Ok(None);
    };
    let a_key = AjtaiKeyParams::try_new(family, n_a, width_s, norm_s, d)?;

    let Some(norm_t) = rounded_up_collision_norm_t(family, d, log_basis) else {
        return Ok(None);
    };
    let Some(width_t) = decomposed_t_ring_count(n_a, num_digits_open, num_blocks, main_num_polys)
    else {
        return Ok(None);
    };
    let Some(n_b) = min_secure_rank(family, d as u32, norm_t, width_t as u64) else {
        return Ok(None);
    };
    let b_key = AjtaiKeyParams::try_new(family, n_b, width_t, norm_t, d)?;

    // Grouped D uses one `w_hat_g` segment per commitment group, not per polynomial.
    let Some(main_d_width) = decomposed_w_ring_count(num_digits_open, num_blocks, 1) else {
        return Ok(None);
    };
    let d_width = main_d_width
        .checked_add(ctx.precommitted_d_width)
        .ok_or_else(|| AkitaError::InvalidSetup("grouped D width overflow".to_string()))?;
    let Some(norm_w) = rounded_up_collision_norm_w(family, d, log_basis) else {
        return Ok(None);
    };
    let Some(n_d) = min_secure_rank(family, d as u32, norm_w, d_width as u64) else {
        return Ok(None);
    };
    let d_key = AjtaiKeyParams::try_new(family, n_d, d_width, norm_w, d)?;

    let onehot_chunk_size = if decomp.log_commit_bound == 1 {
        policy.onehot_chunk_size
    } else {
        0
    };
    let params = LevelParams {
        ring_dimension: d,
        log_basis,
        a_key,
        b_key,
        d_key,
        num_blocks,
        block_len,
        m_vars,
        r_vars,
        stage1_config: ctx.ring_challenge_cfg.clone(),
        op_norm_rejection,
        fold_challenge_shape: ctx.fold_challenge_shape,
        num_digits_commit,
        num_digits_open,
        onehot_chunk_size,
        tier_split: 1,
        f_key: None,
        fold_linf_cap_config: FoldWitnessLinfCapConfig::worst_case_beta_only(),
        num_digits_fold_one: 1,
        field_bits_hint: 0,
        cached_num_digits_fold_claims: 0,
        cached_num_digits_fold_value: 1,
        // Grouped root-direct ships raw witnesses; chunked layout is orthogonal
        // and not used by the grouped precommit path.
        witness_chunk: akita_types::ChunkedWitnessCfg::default(),
        precommitted_groups: ctx.precommitted_groups.to_vec(),
    }
    .with_fold_linf_cap_config(decomp.field_bits(), main_num_polys)?;

    Ok(Some(params))
}

fn compute_grouped_root_direct_level_params(
    key: &GroupBatchAkitaScheduleLookupKey,
    policy: &PlannerPolicy,
    ring_challenge_config: RingChallengeConfigFn<'_>,
    fold_challenge_shape: TensorChallengeShape,
) -> Result<LevelParams, AkitaError> {
    key.validate()?;
    if key.precommitteds.is_empty() {
        return Err(AkitaError::InvalidSetup(
            "grouped root params require at least one precommitted group".to_string(),
        ));
    }

    let precommitted_groups = key
        .precommitteds
        .iter()
        .map(|layout| {
            group_root_params_from_layout(
                layout,
                policy,
                ring_challenge_config,
                fold_challenge_shape,
                true,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut precommitted_d_width = 0usize;
    for group in &precommitted_groups {
        precommitted_d_width = precommitted_d_width
            .checked_add(group.d_segment_width()?)
            .ok_or_else(|| AkitaError::InvalidSetup("grouped D width overflow".to_string()))?;
    }

    let ring_challenge_cfg = ring_challenge_config(policy.ring_dimension)?;
    let main_num_polys = key.main.num_polynomials;
    let main_num_vars = key.main.num_vars;
    let candidate_ctx = GroupedRootDirectCandidateCtx {
        policy,
        ring_challenge_cfg: &ring_challenge_cfg,
        fold_challenge_shape,
        precommitted_d_width,
        precommitted_groups: &precommitted_groups,
    };

    let mut best: Option<(u128, LevelParams)> = None;
    let alpha = (policy.ring_dimension as u32).trailing_zeros() as usize;
    let candidates = if main_num_vars <= alpha {
        vec![(0, 0)]
    } else {
        let reduced_vars = main_num_vars - alpha;
        if reduced_vars <= 2 || reduced_vars >= 53 {
            let r_vars = reduced_vars / 2;
            vec![(reduced_vars - r_vars, r_vars)]
        } else {
            (1..reduced_vars)
                .rev()
                .map(|r_vars| (reduced_vars - r_vars, r_vars))
                .collect()
        }
    };
    let (min_log_basis, max_log_basis) = policy.basis_range;
    for candidate_log_basis in min_log_basis..=max_log_basis {
        for &(m_vars, r_vars) in &candidates {
            let Some(candidate) = grouped_root_direct_main_candidate(
                &candidate_ctx,
                main_num_polys,
                candidate_log_basis,
                m_vars,
                r_vars,
            )?
            else {
                continue;
            };
            let score = grouped_root_direct_cost_score(
                &candidate,
                main_num_polys,
                policy.decomposition.field_bits(),
            )?;
            if best
                .as_ref()
                .is_none_or(|(best_score, _)| score < *best_score)
            {
                best = Some((score, candidate));
            }
        }
    }

    best.map(|(_, params)| params)
        .ok_or_else(|| AkitaError::InvalidSetup("main grouped root is not committable".to_string()))
}

/// Build the phase-1 grouped-root schedule from the full grouped key.
///
/// This intentionally emits only a root-direct schedule for `G > 1`: the params
/// carry true per-precommitted-group metadata and a shared D key, while folded
/// grouped proving remains guarded in scalar-only consumers.
pub fn find_group_batch_schedule(
    key: &GroupBatchAkitaScheduleLookupKey,
    policy: &PlannerPolicy,
    ring_challenge_config: impl Fn(usize) -> Result<akita_challenges::SparseChallengeConfig, AkitaError>,
    fold_challenge_shape_at_level: impl Fn(AkitaScheduleInputs) -> TensorChallengeShape,
) -> Result<Schedule, AkitaError> {
    key.validate()?;
    if key.num_commitment_groups() == 1 {
        return Err(AkitaError::InvalidSetup(
            "single-group grouped root schedules are not supported yet".to_string(),
        ));
    }
    if policy.tiered {
        return Err(AkitaError::InvalidSetup(
            "tiered multi-group root batching is not supported; see specs/multi-group-batching.md"
                .to_string(),
        ));
    }
    if policy.decomposition.log_commit_bound != 1 {
        return Err(AkitaError::InvalidSetup(
            "dense multi-group root batching is not supported; see specs/multi-group-batching.md"
                .to_string(),
        ));
    }
    let current_w_len = grouped_root_direct_witness_len(key)?;
    let fold_shape = fold_challenge_shape_at_level(AkitaScheduleInputs {
        num_vars: key.main.num_vars,
        level: 0,
        current_w_len,
    });
    let params =
        compute_grouped_root_direct_level_params(key, policy, &ring_challenge_config, fold_shape)?;
    let witness_shape = CleartextWitnessShape::FieldElements(current_w_len);
    let direct_bytes = direct_witness_bytes(policy.decomposition.field_bits(), &witness_shape);
    Ok(Schedule {
        steps: vec![Step::Direct(DirectStep {
            current_w_len,
            witness_shape,
            direct_bytes,
            params: Some(params),
        })],
        total_bytes: direct_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_types::{AkitaScheduleLookupKey, DecompositionParams, SisModulusFamily};

    fn flat_policy() -> PlannerPolicy {
        PlannerPolicy {
            ring_dimension: 64,
            decomposition: DecompositionParams {
                log_basis: 3,
                log_commit_bound: 1,
                log_open_bound: Some(8),
            },
            sis_family: SisModulusFamily::Q128,
            ring_subfield_norm_bound: 1,
            claim_ext_degree: 4,
            chal_ext_degree: 4,
            basis_range: (3, 4),
            onehot_chunk_size: 1,
            tiered: false,
            witness_chunk: akita_types::ChunkedWitnessCfg::default(),
        }
    }

    fn ring_challenge_config(_: usize) -> Result<SparseChallengeConfig, AkitaError> {
        Ok(SparseChallengeConfig::Uniform {
            weight: 1,
            nonzero_coeffs: vec![-1, 1],
        })
    }

    fn fold_shape(_: AkitaScheduleInputs) -> TensorChallengeShape {
        TensorChallengeShape::Flat
    }

    fn precommitted(num_polys: usize, num_vars: usize) -> CommitmentGroupLayout {
        let alpha = flat_policy().ring_dimension.trailing_zeros() as usize;
        let outer = num_vars - alpha;
        let r_vars = outer / 2;
        let m_vars = outer - r_vars;
        CommitmentGroupLayout {
            key: AkitaScheduleLookupKey::new(num_vars, num_polys),
            m_vars,
            r_vars,
            log_basis: 3,
            n_a: 1,
            conservative_n_b: 1,
        }
    }

    fn precommitted_from_policy(
        key: AkitaScheduleLookupKey,
        policy: &PlannerPolicy,
    ) -> CommitmentGroupLayout {
        let schedule =
            crate::find_schedule(key, policy, ring_challenge_config, fold_shape).expect("schedule");
        let params = match schedule.steps.first().expect("schedule step") {
            Step::Fold(fold) => fold.params.clone(),
            Step::Direct(direct) => direct.params.clone().expect("root-direct params"),
        };
        CommitmentGroupLayout::from_params(key, &params)
    }

    #[test]
    fn grouped_root_direct_witness_len_sums_mixed_polynomial_counts() {
        let key = GroupBatchAkitaScheduleLookupKey {
            main: AkitaScheduleLookupKey::new(20, 3),
            precommitteds: vec![precommitted(1, 20), precommitted(2, 20)],
        };

        let expected_len = 3 * (1usize << 20) + (1usize << 20) + 2 * (1usize << 20);
        assert_eq!(
            grouped_root_direct_witness_len(&key).expect("witness length"),
            expected_len
        );
    }

    #[test]
    fn grouped_main_d_width_uses_per_group_w_segment_not_polynomial_count() {
        let main_polys = 4usize;
        let num_blocks = 8usize;
        let num_digits_open = 3usize;
        let per_group_w = decomposed_w_ring_count(num_digits_open, num_blocks, 1).expect("w width");
        let scalar_w =
            decomposed_w_ring_count(num_digits_open, num_blocks, main_polys).expect("scalar w");
        assert_ne!(per_group_w, scalar_w);
        assert_eq!(per_group_w * main_polys, scalar_w);
    }

    #[test]
    fn find_group_batch_schedule_rejects_single_group() {
        let key = GroupBatchAkitaScheduleLookupKey {
            main: AkitaScheduleLookupKey::new(12, 1),
            precommitteds: Vec::new(),
        };

        let err =
            find_group_batch_schedule(&key, &flat_policy(), ring_challenge_config, fold_shape)
                .expect_err("single-group grouped schedule is disabled");
        assert!(matches!(err, AkitaError::InvalidSetup(_)));
    }

    #[test]
    fn grouped_root_direct_searches_policy_basis_range() {
        let mut policy = flat_policy();
        policy.decomposition.log_basis = 3;
        policy.basis_range = (4, 4);
        let pre_key = AkitaScheduleLookupKey::new(20, 1);
        let key = GroupBatchAkitaScheduleLookupKey {
            main: AkitaScheduleLookupKey::new(20, 2),
            precommitteds: vec![precommitted_from_policy(pre_key, &policy)],
        };

        let schedule = find_group_batch_schedule(&key, &policy, ring_challenge_config, fold_shape)
            .expect("grouped schedule");
        let params = match schedule.steps.first().expect("grouped step") {
            Step::Direct(direct) => direct.params.as_ref().expect("grouped root params"),
            Step::Fold(_) => panic!("phase-1 grouped schedule should be root-direct"),
        };

        assert_eq!(params.log_basis, 4);
    }

    #[test]
    fn find_group_batch_schedule_rejects_dense_policy() {
        let mut policy = flat_policy();
        policy.decomposition.log_commit_bound = 8;
        let key = GroupBatchAkitaScheduleLookupKey {
            main: AkitaScheduleLookupKey::new(20, 2),
            precommitteds: vec![precommitted(1, 20)],
        };

        let err = find_group_batch_schedule(&key, &policy, ring_challenge_config, fold_shape)
            .expect_err("dense grouped root schedules are phase-1 unsupported");

        assert!(matches!(err, AkitaError::InvalidSetup(_)));
    }
}
