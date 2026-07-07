//! Grouped root-batch schedule planning.

use akita_challenges::{SparseChallengeConfig, TensorChallengeShape};
use akita_field::AkitaError;
use akita_types::sis::{
    committed_fold_a_role_rank, compute_num_digits_full_field, decomposed_s_block_ring_count,
    decomposed_t_ring_count, decomposed_w_ring_count, min_secure_rank, num_digits_fold,
    num_digits_open, num_digits_s_commit, rounded_up_collision_linf_t, rounded_up_collision_linf_w,
    AjtaiKeyParams, FoldChallengeNorms, FoldWitnessLinfCapConfig, FoldWitnessNorms, SisTableKey,
};
use akita_types::{
    direct_witness_bytes, extension_opening_reduction_level_bytes, level_proof_bytes,
    AkitaScheduleInputs, AkitaScheduleLookupKey, CleartextWitnessShape, CommitmentRingDims,
    DecompositionParams, DirectStep, FoldStep, LevelParams, MRowLayout, PolynomialGroupLayout,
    PrecommittedGroupParams, PrecommittedLevelParams, Schedule, Step,
};

use crate::schedule_params::{
    derive_optimal_suffix_schedule, find_schedule, RingChallengeConfigFn, ScheduleMemo, SuffixCtx,
};
use crate::PlannerPolicy;

fn sis_key(policy: &PlannerPolicy, coeff_linf_bound: u128) -> SisTableKey {
    SisTableKey {
        min_security_bits: policy.min_sis_security_bits,
        family: policy.sis_family,
        ring_dimension: policy.ring_dimension as u32,
        coeff_linf_bound,
    }
}

pub(crate) fn group_root_params_from_layout(
    layout: &PrecommittedGroupParams,
    policy: &PlannerPolicy,
    ring_challenge_config: RingChallengeConfigFn<'_>,
    fold_challenge_shape: TensorChallengeShape,
    conservative_b_rank: bool,
) -> Result<PrecommittedLevelParams, AkitaError> {
    if conservative_b_rank {
        layout.validate_frozen_precommit(policy.ring_dimension, policy.basis_range.0)?;
    } else {
        layout.validate()?;
        layout.validate_root_geometry(policy.ring_dimension)?;
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
    let (norm_s, min_n_a) = committed_fold_a_role_rank(
        policy.min_sis_security_bits,
        family,
        d,
        level_decomp,
        &ring_challenge_cfg,
        fold_challenge_shape,
        true,
        policy.onehot_chunk_size,
        policy.ring_subfield_norm_bound,
        layout.r_vars,
        layout.group.num_polynomials(),
        width_s as u64,
    )
    .ok_or_else(|| AkitaError::InvalidSetup("no grouped A-role norm".to_string()))?;
    if layout.n_a < min_n_a {
        return Err(AkitaError::InvalidSetup(
            "precommitted group A rank is below grouped root requirement".to_string(),
        ));
    }
    let a_key = AjtaiKeyParams::try_new(
        policy.min_sis_security_bits,
        family,
        layout.n_a,
        width_s,
        norm_s,
        d,
    )?;

    let b_norm_basis = if conservative_b_rank {
        policy.basis_range.1
    } else {
        layout.log_basis
    };
    let norm_t = rounded_up_collision_linf_t(policy.min_sis_security_bits, family, d, b_norm_basis)
        .ok_or_else(|| AkitaError::InvalidSetup("no grouped B-role norm".to_string()))?;
    let width_t = decomposed_t_ring_count(
        layout.n_a,
        num_digits_open,
        num_blocks,
        layout.group.num_polynomials(),
    )
    .ok_or_else(|| AkitaError::InvalidSetup("grouped B width overflow".to_string()))?;
    let min_n_b = min_secure_rank(sis_key(policy, norm_t), width_t as u64)
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
    let b_key = AjtaiKeyParams::try_new(
        policy.min_sis_security_bits,
        family,
        n_b,
        width_t,
        norm_t,
        d,
    )?;

    let fold_linf_cap_config = FoldWitnessLinfCapConfig::for_fold_level(
        &ring_challenge_cfg,
        fold_challenge_shape,
        d,
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
        layout.group.num_polynomials(),
        policy.decomposition.field_bits(),
        layout.log_basis,
        challenge,
        witness,
        fold_linf_cap_config,
    )?;

    Ok(PrecommittedLevelParams {
        layout: *layout,
        a_key,
        b_key,
        num_blocks,
        block_len,
        num_digits_commit,
        num_digits_open,
        num_digits_fold_one,
    })
}

struct GroupedRootCandidateCtx<'a> {
    policy: &'a PlannerPolicy,
    ring_challenge_cfg: &'a SparseChallengeConfig,
    fold_challenge_shape: TensorChallengeShape,
    precommitted_d_width: usize,
    precommitted_groups: &'a [PrecommittedLevelParams],
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

pub(crate) fn grouped_root_precommitted_groups(
    key: &AkitaScheduleLookupKey,
    policy: &PlannerPolicy,
    ring_challenge_config: RingChallengeConfigFn<'_>,
    fold_challenge_shape: TensorChallengeShape,
) -> Result<(Vec<PrecommittedLevelParams>, usize), AkitaError> {
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

    Ok((precommitted_groups, precommitted_d_width))
}

fn grouped_root_segment_rings(
    num_polys: usize,
    num_blocks: usize,
    block_len: usize,
    n_a: usize,
    num_digits_commit: usize,
    num_digits_open: usize,
    num_digits_fold: usize,
) -> Result<usize, AkitaError> {
    let e_hat = num_polys
        .checked_mul(num_blocks)
        .and_then(|n| n.checked_mul(num_digits_open))
        .ok_or_else(|| {
            AkitaError::InvalidSetup("grouped root e-hat witness overflow".to_string())
        })?;
    let t_hat = num_polys
        .checked_mul(num_blocks)
        .and_then(|n| n.checked_mul(n_a))
        .and_then(|n| n.checked_mul(num_digits_open))
        .ok_or_else(|| {
            AkitaError::InvalidSetup("grouped root t-hat witness overflow".to_string())
        })?;
    let z_hat = block_len
        .checked_mul(num_digits_commit)
        .and_then(|n| n.checked_mul(num_digits_fold))
        .ok_or_else(|| {
            AkitaError::InvalidSetup("grouped root z-hat witness overflow".to_string())
        })?;

    e_hat
        .checked_add(t_hat)
        .and_then(|n| n.checked_add(z_hat))
        .ok_or_else(|| AkitaError::InvalidSetup("grouped root witness overflow".to_string()))
}

pub(crate) fn grouped_root_next_w_len(
    field_bits: u32,
    params: &LevelParams,
    main_num_polys: usize,
    layout: MRowLayout,
) -> Result<usize, AkitaError> {
    if params.precommitted_groups.is_empty() {
        return Err(AkitaError::InvalidSetup(
            "grouped root witness sizing requires precommitted groups".to_string(),
        ));
    }

    let mut total = grouped_root_segment_rings(
        main_num_polys,
        params.num_blocks,
        params.block_len,
        params.a_key.row_len(),
        params.num_digits_commit,
        params.num_digits_open,
        params.num_digits_fold(main_num_polys, field_bits)?,
    )?;
    for group in &params.precommitted_groups {
        let group_rings = grouped_root_segment_rings(
            group.layout.group.num_polynomials(),
            group.num_blocks,
            group.block_len,
            group.a_key.row_len(),
            group.num_digits_commit,
            group.num_digits_open,
            group.num_digits_fold_one,
        )?;
        total = total
            .checked_add(group_rings)
            .ok_or_else(|| AkitaError::InvalidSetup("grouped root witness overflow".to_string()))?;
    }

    let r_rows = params.m_row_count_for(params.precommitted_groups.len() + 1, layout)?;
    let r_count = r_rows
        .checked_mul(compute_num_digits_full_field(field_bits, params.log_basis))
        .ok_or_else(|| {
            AkitaError::InvalidSetup("grouped root r-tail witness overflow".to_string())
        })?;

    let rings = total
        .checked_add(r_count)
        .ok_or_else(|| AkitaError::InvalidSetup("grouped root witness overflow".to_string()))?;

    rings.checked_mul(params.ring_dimension).ok_or_else(|| {
        AkitaError::InvalidSetup("grouped root next witness length overflow".to_string())
    })
}

fn grouped_root_main_level_params_candidate(
    ctx: &GroupedRootCandidateCtx<'_>,
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
    let Some((norm_s, n_a)) = committed_fold_a_role_rank(
        policy.min_sis_security_bits,
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
    let a_key = AjtaiKeyParams::try_new(
        policy.min_sis_security_bits,
        family,
        n_a,
        width_s,
        norm_s,
        d,
    )?;

    let Some(norm_t) =
        rounded_up_collision_linf_t(policy.min_sis_security_bits, family, d, log_basis)
    else {
        return Ok(None);
    };
    let Some(width_t) = decomposed_t_ring_count(n_a, num_digits_open, num_blocks, main_num_polys)
    else {
        return Ok(None);
    };
    let Some(n_b) = min_secure_rank(sis_key(policy, norm_t), width_t as u64) else {
        return Ok(None);
    };
    let b_key = AjtaiKeyParams::try_new(
        policy.min_sis_security_bits,
        family,
        n_b,
        width_t,
        norm_t,
        d,
    )?;

    let Some(main_d_width) = decomposed_w_ring_count(num_digits_open, num_blocks, main_num_polys)
    else {
        return Ok(None);
    };
    let d_width = main_d_width
        .checked_add(ctx.precommitted_d_width)
        .ok_or_else(|| AkitaError::InvalidSetup("grouped D width overflow".to_string()))?;
    let Some(norm_w) =
        rounded_up_collision_linf_w(policy.min_sis_security_bits, family, d, log_basis)
    else {
        return Ok(None);
    };
    let Some(n_d) = min_secure_rank(sis_key(policy, norm_w), d_width as u64) else {
        return Ok(None);
    };
    let d_key = AjtaiKeyParams::try_new(
        policy.min_sis_security_bits,
        family,
        n_d,
        d_width,
        norm_w,
        d,
    )?;

    let onehot_chunk_size = if decomp.log_commit_bound == 1 {
        policy.onehot_chunk_size
    } else {
        0
    };
    let mut params = LevelParams {
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
        fold_challenge_shape: ctx.fold_challenge_shape,
        num_digits_commit,
        num_digits_open,
        onehot_chunk_size,
        fold_linf_cap_config: FoldWitnessLinfCapConfig::worst_case_beta_only(),
        num_digits_fold_one: 1,
        field_bits_hint: 0,
        cached_num_digits_fold_claims: 0,
        cached_num_digits_fold_value: 1,
        // Grouped root-direct ships raw witnesses; chunked layout is orthogonal
        // and not used by the grouped precommit path.
        witness_chunk: akita_types::ChunkedWitnessCfg::default(),
        precommitted_groups: ctx.precommitted_groups.to_vec(),
        role_dims: CommitmentRingDims::uniform(d),
    }
    .with_fold_linf_cap_config(decomp.field_bits(), main_num_polys)?;

    params.stamp_role_dims_from_keys();
    Ok(Some(params))
}

fn compute_grouped_root_direct_level_params(
    key: &AkitaScheduleLookupKey,
    policy: &PlannerPolicy,
    ring_challenge_config: RingChallengeConfigFn<'_>,
    fold_challenge_shape: TensorChallengeShape,
) -> Result<Option<LevelParams>, AkitaError> {
    key.validate()?;
    let (precommitted_groups, precommitted_d_width) =
        grouped_root_precommitted_groups(key, policy, ring_challenge_config, fold_challenge_shape)?;

    let ring_challenge_cfg = ring_challenge_config(policy.ring_dimension)?;
    let main_num_polys = key.final_group.num_polynomials();
    let main_num_vars = key.final_group.num_vars();
    let candidate_ctx = GroupedRootCandidateCtx {
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
            let Some(candidate) = grouped_root_main_level_params_candidate(
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

    Ok(best.map(|(_, params)| params))
}

/// Build the phase-1 grouped-root schedule from the full grouped key.
pub fn find_group_batch_schedule(
    key: &AkitaScheduleLookupKey,
    policy: &PlannerPolicy,
    ring_challenge_config: impl Fn(usize) -> Result<akita_challenges::SparseChallengeConfig, AkitaError>,
    fold_challenge_shape_at_level: impl Fn(AkitaScheduleInputs) -> TensorChallengeShape,
) -> Result<Schedule, AkitaError> {
    key.validate()?;
    if key.precommitteds.is_empty() {
        return find_schedule(
            key.final_group,
            policy,
            ring_challenge_config,
            fold_challenge_shape_at_level,
        );
    }
    if policy.decomposition.log_commit_bound != 1 {
        return Err(AkitaError::InvalidSetup(
            "dense multi-group root batching is not supported; see specs/multi-group-batching.md"
                .to_string(),
        ));
    }
    if policy.witness_chunk.uses_multi_chunk() {
        return Err(AkitaError::InvalidSetup(
            akita_types::GROUPED_ROOT_MULTI_CHUNK_UNSUPPORTED.to_string(),
        ));
    }

    let ring_challenge_config: RingChallengeConfigFn<'_> = &ring_challenge_config;
    let field_bits = policy.decomposition.field_bits();
    let challenge_field_bits = field_bits * policy.chal_ext_degree as u32;
    let direct_current_w_len = key.opening_layout()?.root_direct_witness_len()?;
    let direct_fold_shape = fold_challenge_shape_at_level(AkitaScheduleInputs {
        num_vars: key.final_group.num_vars(),
        level: 0,
        current_w_len: direct_current_w_len,
    });
    let mut best: Option<(usize, Vec<Step>)> = if let Some(params) =
        compute_grouped_root_direct_level_params(
            key,
            policy,
            ring_challenge_config,
            direct_fold_shape,
        )? {
        let witness_shape = CleartextWitnessShape::FieldElements(direct_current_w_len);
        let direct_bytes = direct_witness_bytes(field_bits, &witness_shape);
        Some((
            direct_bytes,
            vec![Step::Direct(DirectStep {
                current_w_len: direct_current_w_len,
                witness_shape,
                direct_bytes,
                params: Some(params),
            })],
        ))
    } else {
        None
    };

    let root_current_w_len = 1usize
        .checked_shl(key.final_group.num_vars() as u32)
        .ok_or_else(|| {
            AkitaError::InvalidSetup("grouped root-fold witness length overflow".to_string())
        })?;
    let fold_challenge_shape = fold_challenge_shape_at_level(AkitaScheduleInputs {
        num_vars: key.final_group.num_vars(),
        level: 0,
        current_w_len: root_current_w_len,
    });
    let alpha = (policy.ring_dimension as u32).trailing_zeros() as usize;
    let reduced_vars = key.final_group.num_vars().saturating_sub(alpha);
    if reduced_vars == 0 {
        let Some((total_bytes, steps)) = best else {
            return Err(AkitaError::InvalidSetup(
                "main grouped root is not committable".to_string(),
            ));
        };
        return Ok(Schedule { steps, total_bytes });
    }

    let (precommitted_groups, precommitted_d_width) =
        grouped_root_precommitted_groups(key, policy, ring_challenge_config, fold_challenge_shape)?;
    let ring_challenge_cfg = ring_challenge_config(policy.ring_dimension)?;
    let candidate_ctx = GroupedRootCandidateCtx {
        policy,
        ring_challenge_cfg: &ring_challenge_cfg,
        fold_challenge_shape,
        precommitted_d_width,
        precommitted_groups: &precommitted_groups,
    };
    let suffix_ctx = SuffixCtx {
        policy,
        ring_challenge_config,
        num_vars: key.final_group.num_vars(),
        key: PolynomialGroupLayout::singleton(key.final_group.num_vars()),
    };
    let mut memo = ScheduleMemo::new();
    let total_polys = key.num_polynomials()?;
    let root_eor_key = PolynomialGroupLayout::new(key.final_group.num_vars(), total_polys);
    let initial_witness_len_bits = root_current_w_len
        .checked_mul(field_bits as usize)
        .ok_or_else(|| {
            AkitaError::InvalidSetup("grouped root witness bit length overflow".into())
        })?;
    let min_r_vars: usize = if reduced_vars >= 3 { 1 } else { 0 };
    let max_r_vars: usize = (reduced_vars - 1).min(usize::BITS as usize - 1);
    let (min_log_basis, max_log_basis) = policy.basis_range;

    for candidate_log_basis in min_log_basis..=max_log_basis {
        for r_vars in (min_r_vars..=max_r_vars).rev() {
            let m_vars = reduced_vars - r_vars;
            let Some(candidate_params) = grouped_root_main_level_params_candidate(
                &candidate_ctx,
                key.final_group.num_polynomials(),
                candidate_log_basis,
                m_vars,
                r_vars,
            )?
            else {
                continue;
            };
            let next_w_len = grouped_root_next_w_len(
                field_bits,
                &candidate_params,
                key.final_group.num_polynomials(),
                MRowLayout::WithDBlock,
            )?;
            let next_w_len_terminal = grouped_root_next_w_len(
                field_bits,
                &candidate_params,
                key.final_group.num_polynomials(),
                MRowLayout::WithoutDBlock,
            )?;
            if next_w_len
                .checked_mul(candidate_log_basis as usize)
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("grouped root next witness bit length overflow".into())
                })?
                >= initial_witness_len_bits
            {
                continue;
            }

            let suffix = derive_optimal_suffix_schedule(
                &suffix_ctx,
                &mut memo,
                1,
                next_w_len,
                next_w_len_terminal,
                candidate_log_basis,
                0,
            )?;
            if suffix.is_empty() {
                continue;
            }
            let Ok(eor_bytes) = extension_opening_reduction_level_bytes(
                policy.decomposition.field_bits() * policy.chal_ext_degree as u32,
                policy.claim_ext_degree,
                0,
                root_eor_key,
                root_current_w_len,
            ) else {
                continue;
            };

            // A grouped root that is immediately terminal needs a grouped
            // segment-typed witness layout; keep phase-1 schedules on the
            // singleton recursive suffix path.
            if suffix.best_direct.is_some() && suffix.best_fold_per_lb.is_empty() {
                return Err(AkitaError::InvalidSetup(
                    "grouped terminal root folds are not supported yet; grouped folded schedules require a singleton recursive suffix".to_string(),
                ));
            }
            for suffix_fold in suffix.best_fold_per_lb.values() {
                let root_proof_size = level_proof_bytes(
                    field_bits,
                    challenge_field_bits,
                    &candidate_params,
                    Some(&suffix_fold.first_fold_params),
                    next_w_len,
                    1,
                    MRowLayout::WithDBlock,
                ) + eor_bytes;
                let total = root_proof_size + suffix_fold.total_bytes;
                if best
                    .as_ref()
                    .is_none_or(|(best_total, _)| total < *best_total)
                {
                    let mut steps = Vec::with_capacity(1 + suffix_fold.steps.len());
                    steps.push(Step::Fold(FoldStep {
                        params: candidate_params.clone(),
                        current_w_len: root_current_w_len,
                        next_w_len,
                        level_bytes: root_proof_size,
                    }));
                    steps.extend(suffix_fold.steps.iter().cloned());
                    best = Some((total, steps));
                }
            }
        }
    }

    let Some((total_bytes, steps)) = best else {
        return Err(AkitaError::InvalidSetup(
            "main grouped root is not committable".to_string(),
        ));
    };
    Ok(Schedule { steps, total_bytes })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::find_schedule;
    use akita_field::Prime128OffsetA7F7;
    use akita_types::{
        AkitaScheduleLookupKey, DecompositionParams, MRowLayout, PolynomialGroupLayout,
        SisModulusFamily, DEFAULT_SIS_SECURITY_BITS,
    };

    fn flat_policy() -> PlannerPolicy {
        PlannerPolicy {
            ring_dimension: 64,
            decomposition: DecompositionParams {
                log_basis: 3,
                log_commit_bound: 1,
                log_open_bound: Some(8),
            },
            sis_family: SisModulusFamily::Q128,
            min_sis_security_bits: DEFAULT_SIS_SECURITY_BITS,
            ring_subfield_norm_bound: 1,
            claim_ext_degree: 4,
            chal_ext_degree: 4,
            basis_range: (3, 4),
            onehot_chunk_size: 1,
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

    fn precommitted(num_polys: usize, num_vars: usize) -> PrecommittedGroupParams {
        let alpha = flat_policy().ring_dimension.trailing_zeros() as usize;
        let outer = num_vars - alpha;
        let r_vars = outer / 2;
        let m_vars = outer - r_vars;
        PrecommittedGroupParams {
            group: PolynomialGroupLayout::new(num_vars, num_polys),
            m_vars,
            r_vars,
            log_basis: 3,
            n_a: 1,
            conservative_n_b: 1,
        }
    }

    fn precommitted_from_policy(
        key: PolynomialGroupLayout,
        policy: &PlannerPolicy,
    ) -> PrecommittedGroupParams {
        let schedule =
            crate::find_schedule(key, policy, ring_challenge_config, fold_shape).expect("schedule");
        let params = match schedule.steps.first().expect("schedule step") {
            Step::Fold(fold) => fold.params.clone(),
            Step::Direct(direct) => direct.params.clone().expect("root-direct params"),
        };
        PrecommittedGroupParams::from_params(key, &params)
    }

    #[test]
    fn grouped_root_direct_witness_len_sums_mixed_polynomial_counts() {
        let key = AkitaScheduleLookupKey {
            final_group: PolynomialGroupLayout::new(20, 3),
            precommitteds: vec![precommitted(1, 20), precommitted(2, 20)],
        };

        let expected_len = 3 * (1usize << 20) + (1usize << 20) + 2 * (1usize << 20);
        assert_eq!(
            key.opening_layout()
                .expect("layout")
                .root_direct_witness_len()
                .expect("witness length"),
            expected_len
        );
    }

    #[test]
    fn decomposed_w_ring_count_scales_with_polynomial_count() {
        let main_polys = 4usize;
        let num_blocks = 8usize;
        let num_digits_open = 3usize;
        let per_group_w = decomposed_w_ring_count(num_digits_open, num_blocks, 1).expect("w width");
        let scalar_w =
            decomposed_w_ring_count(num_digits_open, num_blocks, main_polys).expect("scalar w");
        assert_ne!(per_group_w, scalar_w);
        assert_eq!(per_group_w * main_polys, scalar_w);
    }

    fn assert_grouped_fold_sizing_matches_runtime(final_polys: usize, pre_polys: usize) {
        let mut policy = flat_policy();
        policy.decomposition.log_open_bound = Some(128);
        policy.basis_range = (4, 4);
        let pre_key = PolynomialGroupLayout::new(20, pre_polys);
        let key = AkitaScheduleLookupKey {
            final_group: PolynomialGroupLayout::new(40, final_polys),
            precommitteds: vec![precommitted_from_policy(pre_key, &policy)],
        };
        let opening_batch = key.opening_layout().expect("opening layout");
        let schedule = find_group_batch_schedule(&key, &policy, ring_challenge_config, fold_shape)
            .expect("grouped schedule");
        let Step::Fold(root) = schedule.steps.first().expect("grouped root step") else {
            panic!("expected grouped root fold");
        };

        let runtime_next_w_len = root
            .params
            .root_next_w_len::<Prime128OffsetA7F7>(&opening_batch, MRowLayout::WithDBlock)
            .expect("runtime next w len");
        assert_eq!(root.next_w_len, runtime_next_w_len);

        let expected_d_width = root
            .params
            .num_digits_open
            .checked_mul(root.params.num_blocks)
            .and_then(|n| n.checked_mul(final_polys))
            .expect("main D width")
            + root
                .params
                .precommitted_groups
                .iter()
                .map(|group| group.d_segment_width().expect("precommitted D width"))
                .sum::<usize>();
        assert_eq!(root.params.d_key.col_len(), expected_d_width);
    }

    #[test]
    fn grouped_fold_sizing_matches_runtime_for_one_three() {
        assert_grouped_fold_sizing_matches_runtime(1, 3);
    }

    #[test]
    fn grouped_fold_sizing_matches_runtime_for_two_one() {
        assert_grouped_fold_sizing_matches_runtime(2, 1);
    }

    #[test]
    fn find_group_batch_schedule_delegates_single_group_to_scalar() {
        let final_group = PolynomialGroupLayout::new(12, 1);
        let key = AkitaScheduleLookupKey::single(final_group);
        let policy = flat_policy();

        let via_grouped =
            find_group_batch_schedule(&key, &policy, ring_challenge_config, fold_shape)
                .expect("single-group grouped key should delegate to scalar DP");
        let via_scalar =
            find_schedule(final_group, &policy, ring_challenge_config, fold_shape).expect("scalar");

        assert_eq!(via_grouped.total_bytes, via_scalar.total_bytes);
        assert_eq!(via_grouped.steps.len(), via_scalar.steps.len());
    }

    #[test]
    fn grouped_root_schedule_searches_policy_basis_range() {
        let mut policy = flat_policy();
        policy.decomposition.log_basis = 3;
        policy.basis_range = (4, 4);
        let pre_key = PolynomialGroupLayout::new(20, 1);
        let key = AkitaScheduleLookupKey {
            final_group: PolynomialGroupLayout::new(40, 2),
            precommitteds: vec![precommitted_from_policy(pre_key, &policy)],
        };

        let schedule = find_group_batch_schedule(&key, &policy, ring_challenge_config, fold_shape)
            .expect("grouped schedule");
        let params = match schedule.steps.first().expect("grouped step") {
            Step::Direct(direct) => direct.params.as_ref().expect("grouped root params"),
            Step::Fold(fold) => &fold.params,
        };

        assert_eq!(params.log_basis, 4);
    }

    #[test]
    fn grouped_schedule_can_start_with_fold() {
        let mut policy = flat_policy();
        policy.basis_range = (4, 4);
        let pre_key = PolynomialGroupLayout::new(20, 1);
        let key = AkitaScheduleLookupKey {
            final_group: PolynomialGroupLayout::new(40, 2),
            precommitteds: vec![precommitted_from_policy(pre_key, &policy)],
        };

        let schedule = find_group_batch_schedule(&key, &policy, ring_challenge_config, fold_shape)
            .expect("grouped schedule");
        let Step::Fold(root) = schedule.steps.first().expect("grouped root step") else {
            panic!("expected grouped root fold");
        };

        assert_eq!(root.current_w_len, 1usize << key.final_group.num_vars());
        assert_eq!(
            root.params.precommitted_groups.len(),
            key.precommitteds.len()
        );
        assert!(root.next_w_len > 0);
    }

    #[test]
    fn find_group_batch_schedule_rejects_dense_policy() {
        let mut policy = flat_policy();
        policy.decomposition.log_commit_bound = 8;
        let key = AkitaScheduleLookupKey {
            final_group: PolynomialGroupLayout::new(40, 2),
            precommitteds: vec![precommitted(1, 20)],
        };

        let err = find_group_batch_schedule(&key, &policy, ring_challenge_config, fold_shape)
            .expect_err("dense grouped root schedules are phase-1 unsupported");

        assert!(matches!(err, AkitaError::InvalidSetup(_)));
    }
}
