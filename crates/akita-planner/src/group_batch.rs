//! Multi-group root-batch schedule planning.

use akita_challenges::{SparseChallengeConfig, TensorChallengeShape};
use akita_field::{AkitaError, Prime128OffsetA7F7};
use akita_types::sis::{
    compute_num_digits_full_field, decomposed_s_block_ring_count, decomposed_t_ring_count,
    decomposed_w_ring_count, fold_witness_digit_plan, min_secure_rank, num_digits_open,
    num_digits_s_commit, rounded_up_collision_inf_norm, rounded_up_role_a_inf_norm, AjtaiKeyParams,
    FoldChallengeNorms, FoldWitnessLinfCapConfig, FoldWitnessNorms, SisTableKey,
};
use akita_types::{
    active_setup_field_len, direct_witness_bytes, extension_opening_reduction_level_bytes,
    level_proof_bytes, padded_setup_prefix_len, AkitaScheduleInputs, AkitaScheduleLookupKey,
    CleartextWitnessShape, CommitmentRingDims, DecompositionParams, DirectStep, FoldStep,
    LevelParams, PolynomialGroupLayout, PrecommittedGroupParams, PrecommittedLevelParams,
    RelationMatrixRowLayout, Schedule, SetupContributionMode, Step, SETUP_OFFLOAD_D_SETUP,
    SETUP_OFFLOAD_MIN_PREFIX_FIELD_LEN,
};

use crate::schedule_params::{
    derive_optimal_suffix_schedule, find_schedule, RingChallengeConfigFn, ScheduleMemo, SuffixCtx,
    SuffixState,
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
    let num_blocks = 1usize.checked_shl(layout.r_vars as u32).ok_or_else(|| {
        AkitaError::InvalidSetup("multi-group root num_blocks overflow".to_string())
    })?;
    let block_len = 1usize.checked_shl(layout.m_vars as u32).ok_or_else(|| {
        AkitaError::InvalidSetup("multi-group root block_len overflow".to_string())
    })?;

    let width_s = decomposed_s_block_ring_count(block_len, num_digits_commit)
        .ok_or_else(|| AkitaError::InvalidSetup("multi-group A width overflow".to_string()))?;
    let norm_s = rounded_up_role_a_inf_norm(
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
    .ok_or_else(|| AkitaError::InvalidSetup("no multi-group A-role norm".to_string()))?;
    let min_n_a = min_secure_rank(
        SisTableKey {
            min_security_bits: policy.min_sis_security_bits,
            family,
            ring_dimension: d as u32,
            coeff_linf_bound: norm_s,
        },
        width_s as u64,
    )
    .ok_or_else(|| AkitaError::InvalidSetup("no multi-group A-role rank".to_string()))?;
    if layout.n_a < min_n_a {
        return Err(AkitaError::InvalidSetup(
            "precommitted group A rank is below multi-group root requirement".to_string(),
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
    let norm_t =
        rounded_up_collision_inf_norm(policy.min_sis_security_bits, family, d, b_norm_basis)
            .ok_or_else(|| AkitaError::InvalidSetup("no multi-group B-role norm".to_string()))?;
    let width_t = decomposed_t_ring_count(
        layout.n_a,
        num_digits_open,
        num_blocks,
        layout.group.num_polynomials(),
    )
    .ok_or_else(|| AkitaError::InvalidSetup("setup B width overflow".to_string()))?;
    let min_n_b = min_secure_rank(sis_key(policy, norm_t), width_t as u64)
        .ok_or_else(|| AkitaError::InvalidSetup("no multi-group B-role rank".to_string()))?;
    let n_b = if conservative_b_rank {
        if layout.conservative_n_b < min_n_b {
            return Err(AkitaError::InvalidSetup(
                "precommitted group conservative B rank is below multi-group root requirement"
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
    let (num_digits_fold_one, _) = fold_witness_digit_plan(
        layout.r_vars,
        layout.group.num_polynomials(),
        policy.decomposition.field_bits(),
        layout.log_basis,
        challenge,
        witness,
        &fold_linf_cap_config,
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

struct MultiGroupRootCandidateCtx<'a> {
    policy: &'a PlannerPolicy,
    ring_challenge_cfg: &'a SparseChallengeConfig,
    fold_challenge_shape: TensorChallengeShape,
    precommitted_d_width: usize,
    precommitted_groups: &'a [PrecommittedLevelParams],
}

pub(crate) fn multi_group_root_precommitted_groups(
    key: &AkitaScheduleLookupKey,
    policy: &PlannerPolicy,
    ring_challenge_config: RingChallengeConfigFn<'_>,
    fold_challenge_shape: TensorChallengeShape,
) -> Result<(Vec<PrecommittedLevelParams>, usize), AkitaError> {
    if key.precommitteds.is_empty() {
        return Err(AkitaError::InvalidSetup(
            "multi-group root params require at least one precommitted group".to_string(),
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
            .ok_or_else(|| AkitaError::InvalidSetup("multi-group D width overflow".to_string()))?;
    }

    Ok((precommitted_groups, precommitted_d_width))
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

fn multi_group_root_direct_cost_score(
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
        "multi-group main root-direct score",
    )?;

    for group in &params.precommitted_groups {
        let group_cost = root_direct_split_cost(
            group.a_key.row_len(),
            group.num_blocks,
            group.block_len,
            group.num_digits_commit,
            group.num_digits_open,
            group.num_digits_fold_one,
            "multi-group precommitted root-direct score",
        )?;
        total = checked_score_add(total, group_cost, "multi-group root-direct score total")?;
    }

    Ok(total)
}

#[allow(dead_code)]
fn multi_group_root_segment_rings(
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
            AkitaError::InvalidSetup("multi-group root e-hat witness overflow".to_string())
        })?;
    let t_hat = num_polys
        .checked_mul(num_blocks)
        .and_then(|n| n.checked_mul(n_a))
        .and_then(|n| n.checked_mul(num_digits_open))
        .ok_or_else(|| {
            AkitaError::InvalidSetup("multi-group root t-hat witness overflow".to_string())
        })?;
    let z_hat = block_len
        .checked_mul(num_digits_commit)
        .and_then(|n| n.checked_mul(num_digits_fold))
        .ok_or_else(|| {
            AkitaError::InvalidSetup("multi-group root z-hat witness overflow".to_string())
        })?;

    e_hat
        .checked_add(t_hat)
        .and_then(|n| n.checked_add(z_hat))
        .ok_or_else(|| AkitaError::InvalidSetup("multi-group root witness overflow".to_string()))
}

#[allow(dead_code)]
fn multi_group_root_next_w_len(
    field_bits: u32,
    params: &LevelParams,
    main_num_polys: usize,
    layout: RelationMatrixRowLayout,
) -> Result<usize, AkitaError> {
    if params.precommitted_groups.is_empty() {
        return Err(AkitaError::InvalidSetup(
            "multi-group root witness sizing requires precommitted groups".to_string(),
        ));
    }

    let mut total = multi_group_root_segment_rings(
        main_num_polys,
        params.num_blocks,
        params.block_len,
        params.a_key.row_len(),
        params.num_digits_commit,
        params.num_digits_open,
        params.num_digits_fold(main_num_polys, field_bits)?,
    )?;
    for group in &params.precommitted_groups {
        let group_rings = multi_group_root_segment_rings(
            group.layout.group.num_polynomials(),
            group.num_blocks,
            group.block_len,
            group.a_key.row_len(),
            group.num_digits_commit,
            group.num_digits_open,
            group.num_digits_fold_one,
        )?;
        total = total.checked_add(group_rings).ok_or_else(|| {
            AkitaError::InvalidSetup("multi-group root witness overflow".to_string())
        })?;
    }

    let r_rows =
        params.relation_matrix_row_count_for(params.precommitted_groups.len() + 1, layout)?;
    let r_count = r_rows
        .checked_mul(compute_num_digits_full_field(field_bits, params.log_basis))
        .ok_or_else(|| {
            AkitaError::InvalidSetup("multi-group root r-tail witness overflow".to_string())
        })?;

    let rings = total
        .checked_add(r_count)
        .ok_or_else(|| AkitaError::InvalidSetup("multi-group root witness overflow".to_string()))?;

    rings.checked_mul(params.ring_dimension).ok_or_else(|| {
        AkitaError::InvalidSetup("multi-group root next witness length overflow".to_string())
    })
}

fn multi_group_root_main_level_params_candidate(
    ctx: &MultiGroupRootCandidateCtx<'_>,
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
    let Some(norm_s) = rounded_up_role_a_inf_norm(
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
    let Ok(a_key) = AjtaiKeyParams::try_new_with_min_rank(sis_key(policy, norm_s), width_s) else {
        return Ok(None);
    };
    let n_a = a_key.row_len();

    let Some(norm_t) =
        rounded_up_collision_inf_norm(policy.min_sis_security_bits, family, d, log_basis)
    else {
        return Ok(None);
    };
    let Some(width_t) = decomposed_t_ring_count(n_a, num_digits_open, num_blocks, main_num_polys)
    else {
        return Ok(None);
    };
    let Ok(b_key) = AjtaiKeyParams::try_new_with_min_rank(sis_key(policy, norm_t), width_t) else {
        return Ok(None);
    };

    let Some(main_d_width) = decomposed_w_ring_count(num_digits_open, num_blocks, main_num_polys)
    else {
        return Ok(None);
    };
    let d_width = main_d_width
        .checked_add(ctx.precommitted_d_width)
        .ok_or_else(|| AkitaError::InvalidSetup("multi-group D width overflow".to_string()))?;
    let Some(norm_w) =
        rounded_up_collision_inf_norm(policy.min_sis_security_bits, family, d, log_basis)
    else {
        return Ok(None);
    };
    let Ok(d_key) = AjtaiKeyParams::try_new_with_min_rank(sis_key(policy, norm_w), d_width) else {
        return Ok(None);
    };

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
        fold_challenge_config: *ctx.ring_challenge_cfg,
        fold_challenge_shape: ctx.fold_challenge_shape,
        num_digits_commit,
        num_digits_open,
        onehot_chunk_size,
        fold_linf_cap_config: FoldWitnessLinfCapConfig::worst_case_beta_only(),
        num_digits_fold_one: 1,
        field_bits_hint: 0,
        cached_num_digits_fold_claims: 0,
        cached_num_digits_fold_value: 1,
        // Multi-group root-direct ships raw witnesses; chunked layout is orthogonal
        // and not used by the multi-group precommit path.
        witness_chunk: akita_types::ChunkedWitnessCfg::default(),
        precommitted_groups: ctx.precommitted_groups.to_vec(),
        setup_prefix: None,
        role_dims: CommitmentRingDims::uniform(d),
        setup_contribution_mode: SetupContributionMode::Direct,
    }
    .with_fold_linf_cap_config(decomp.field_bits(), main_num_polys)?;

    params.stamp_role_dims_from_keys();
    Ok(Some(params))
}

fn compute_multi_group_root_direct_level_params(
    key: &AkitaScheduleLookupKey,
    policy: &PlannerPolicy,
    ring_challenge_config: RingChallengeConfigFn<'_>,
    fold_challenge_shape: TensorChallengeShape,
) -> Result<Option<LevelParams>, AkitaError> {
    key.validate()?;
    let (precommitted_groups, precommitted_d_width) = multi_group_root_precommitted_groups(
        key,
        policy,
        ring_challenge_config,
        fold_challenge_shape,
    )?;

    let ring_challenge_cfg = ring_challenge_config(policy.ring_dimension)?;
    let main_num_polys = key.final_group.num_polynomials();
    let main_num_vars = key.final_group.num_vars();
    let candidate_ctx = MultiGroupRootCandidateCtx {
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
            let Some(candidate) = multi_group_root_main_level_params_candidate(
                &candidate_ctx,
                main_num_polys,
                candidate_log_basis,
                m_vars,
                r_vars,
            )?
            else {
                continue;
            };
            let score = multi_group_root_direct_cost_score(
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

/// Build the phase-1 multi-group-root schedule from the full multi-group key.
pub fn find_group_batch_schedule(
    key: &AkitaScheduleLookupKey,
    policy: &PlannerPolicy,
    ring_challenge_config: impl Fn(usize) -> Result<akita_challenges::SparseChallengeConfig, AkitaError>,
    fold_challenge_shape_at_level: impl Fn(AkitaScheduleInputs) -> TensorChallengeShape,
) -> Result<Schedule, AkitaError> {
    key.validate()?;
    if key.precommitteds.is_empty() {
        // Genuine multi-group roots only. Empty-precommit keys are scalar and
        // must not enter recursion-enabled grouped planning.
        let mut scalar_policy = *policy;
        scalar_policy.recursive_setup_planning = false;
        return find_schedule(
            key.final_group,
            &scalar_policy,
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
            akita_types::MULTI_GROUP_ROOT_MULTI_CHUNK_UNSUPPORTED.to_string(),
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
        compute_multi_group_root_direct_level_params(
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
            AkitaError::InvalidSetup("multi-group root-fold witness length overflow".to_string())
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
                "main multi-group root is not committable".to_string(),
            ));
        };
        return Ok(Schedule { steps, total_bytes });
    }

    let (precommitted_groups, precommitted_d_width) = multi_group_root_precommitted_groups(
        key,
        policy,
        ring_challenge_config,
        fold_challenge_shape,
    )?;
    let ring_challenge_cfg = ring_challenge_config(policy.ring_dimension)?;
    let candidate_ctx = MultiGroupRootCandidateCtx {
        policy,
        ring_challenge_cfg: &ring_challenge_cfg,
        fold_challenge_shape,
        precommitted_d_width,
        precommitted_groups: &precommitted_groups,
    };
    let suffix_ctx = SuffixCtx {
        policy,
        ring_challenge_cfg: &ring_challenge_cfg,
        num_vars: key.final_group.num_vars(),
        key: PolynomialGroupLayout::singleton(key.final_group.num_vars()),
    };
    let mut memo = ScheduleMemo::new();
    let total_polys = key.num_polynomials()?;
    let root_eor_key = PolynomialGroupLayout::new(key.final_group.num_vars(), total_polys);
    let initial_witness_len_bits = root_current_w_len
        .checked_mul(field_bits as usize)
        .ok_or_else(|| {
            AkitaError::InvalidSetup("multi-group root witness bit length overflow".into())
        })?;
    let min_r_vars: usize = if reduced_vars >= 3 { 1 } else { 0 };
    let max_r_vars: usize = (reduced_vars - 1).min(usize::BITS as usize - 1);
    let (min_log_basis, max_log_basis) = policy.basis_range;

    for candidate_log_basis in min_log_basis..=max_log_basis {
        for r_vars in (min_r_vars..=max_r_vars).rev() {
            let m_vars = reduced_vars - r_vars;
            let Some(candidate_params) = multi_group_root_main_level_params_candidate(
                &candidate_ctx,
                key.final_group.num_polynomials(),
                candidate_log_basis,
                m_vars,
                r_vars,
            )?
            else {
                continue;
            };
            let opening_batch = key.opening_layout()?;
            let next_w_len = candidate_params.next_w_len::<Prime128OffsetA7F7>(
                &opening_batch,
                RelationMatrixRowLayout::WithDBlock,
            )?;
            // Grouped terminal folds are rejected; suffix folds are scalar.
            let next_w_len_terminal = next_w_len;
            if next_w_len
                .checked_mul(candidate_log_basis as usize)
                .ok_or_else(|| {
                    AkitaError::InvalidSetup(
                        "multi-group root next witness bit length overflow".into(),
                    )
                })?
                >= initial_witness_len_bits
            {
                continue;
            }

            let natural_len =
                active_setup_field_len(&candidate_params, &opening_batch, SETUP_OFFLOAD_D_SETUP)?;
            let n_prefix = padded_setup_prefix_len(natural_len);
            let recursion_threshold_met =
                policy.recursive_setup_planning && n_prefix > SETUP_OFFLOAD_MIN_PREFIX_FIELD_LEN;
            let child_suffix_no_prefix = derive_optimal_suffix_schedule(
                &suffix_ctx,
                &mut memo,
                SuffixState {
                    level: 1,
                    current_witness_len: next_w_len,
                    current_witness_len_terminal: next_w_len_terminal,
                    current_lb: candidate_log_basis,
                    incoming_setup_prefix: None,
                },
                0,
            )?;
            let Ok(eor_bytes) = extension_opening_reduction_level_bytes(
                policy.decomposition.field_bits() * policy.chal_ext_degree as u32,
                policy.claim_ext_degree,
                0,
                root_eor_key,
                root_current_w_len,
            ) else {
                continue;
            };

            for suffix_fold in child_suffix_no_prefix.best_fold_per_lb.values() {
                let child_is_terminal = matches!(suffix_fold.steps.get(1), Some(Step::Direct(_)));
                let (fold_mode, suffix_fold) = if child_is_terminal {
                    (SetupContributionMode::Direct, suffix_fold.clone())
                } else if recursion_threshold_met {
                    let prefixed_child_suffix = derive_optimal_suffix_schedule(
                        &suffix_ctx,
                        &mut memo,
                        SuffixState {
                            level: 1,
                            current_witness_len: next_w_len,
                            current_witness_len_terminal: next_w_len_terminal,
                            current_lb: candidate_log_basis,
                            incoming_setup_prefix: Some(natural_len),
                        },
                        0,
                    )?;
                    let child_lb = suffix_fold.first_fold_params.log_basis;
                    let Some(prefixed_suffix_fold) =
                        prefixed_child_suffix.best_fold_per_lb.get(&child_lb)
                    else {
                        continue;
                    };
                    if matches!(prefixed_suffix_fold.steps.get(1), Some(Step::Direct(_))) {
                        continue;
                    }
                    (
                        SetupContributionMode::Recursive,
                        prefixed_suffix_fold.clone(),
                    )
                } else {
                    (SetupContributionMode::Direct, suffix_fold.clone())
                };

                let mut fold_candidate_params = candidate_params.clone();
                fold_candidate_params.setup_contribution_mode = fold_mode;
                let root_proof_size = level_proof_bytes(
                    field_bits,
                    challenge_field_bits,
                    &fold_candidate_params,
                    Some(&suffix_fold.first_fold_params),
                    next_w_len,
                    1,
                    RelationMatrixRowLayout::WithDBlock,
                ) + eor_bytes;
                let total = root_proof_size + suffix_fold.total_bytes;
                if best
                    .as_ref()
                    .is_none_or(|(best_total, _)| total < *best_total)
                {
                    let mut steps = Vec::with_capacity(1 + suffix_fold.steps.len());
                    steps.push(Step::Fold(FoldStep {
                        params: fold_candidate_params,
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
            "main multi-group root is not committable".to_string(),
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
        AkitaScheduleLookupKey, DecompositionParams, PolynomialGroupLayout,
        RelationMatrixRowLayout, SisModulusFamily, DEFAULT_SIS_SECURITY_BITS,
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
            recursive_setup_planning: false,
        }
    }

    fn ring_challenge_config(_: usize) -> Result<SparseChallengeConfig, AkitaError> {
        Ok(SparseChallengeConfig::pm1_only(1))
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
        let mut precommitted_policy = *policy;
        precommitted_policy.recursive_setup_planning = false;
        let schedule = find_schedule(key, &precommitted_policy, ring_challenge_config, fold_shape)
            .expect("schedule");
        let params = match schedule.steps.first().expect("schedule step") {
            Step::Fold(fold) => fold.params.clone(),
            Step::Direct(direct) => direct.params.clone().expect("root-direct params"),
        };
        PrecommittedGroupParams::from_params(key, &params)
    }

    #[test]
    fn multi_group_root_direct_witness_len_sums_mixed_polynomial_counts() {
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

    fn assert_multi_group_fold_sizing_matches_runtime(final_polys: usize, pre_polys: usize) {
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
            .expect("multi-group schedule");
        let Step::Fold(root) = schedule.steps.first().expect("multi-group root step") else {
            panic!("expected multi-group root fold");
        };

        let runtime_next_w_len = root
            .params
            .next_w_len::<Prime128OffsetA7F7>(&opening_batch, RelationMatrixRowLayout::WithDBlock)
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
    fn multi_group_fold_sizing_matches_runtime_for_one_three() {
        assert_multi_group_fold_sizing_matches_runtime(1, 3);
    }

    #[test]
    fn multi_group_fold_sizing_matches_runtime_for_two_one() {
        assert_multi_group_fold_sizing_matches_runtime(2, 1);
    }

    #[test]
    fn find_group_batch_schedule_delegates_single_group_to_scalar() {
        let final_group = PolynomialGroupLayout::new(12, 1);
        let key = AkitaScheduleLookupKey::single(final_group);
        let policy = flat_policy();

        let via_multi_group =
            find_group_batch_schedule(&key, &policy, ring_challenge_config, fold_shape)
                .expect("single-group multi-group key should delegate to scalar DP");
        let via_scalar =
            find_schedule(final_group, &policy, ring_challenge_config, fold_shape).expect("scalar");

        assert_eq!(via_multi_group.total_bytes, via_scalar.total_bytes);
        assert_eq!(via_multi_group.steps.len(), via_scalar.steps.len());
    }

    #[test]
    fn malformed_non_tiny_precommit_geometry_is_not_candidate_infeasibility() {
        let policy = flat_policy();
        let malformed = PrecommittedGroupParams {
            group: PolynomialGroupLayout::new(20, 1),
            m_vars: 1,
            r_vars: 1,
            log_basis: 3,
            n_a: 1,
            conservative_n_b: 1,
        };
        let ring_cfg = ring_challenge_config(policy.ring_dimension).expect("ring challenge");

        let error = group_root_params_from_layout(
            &malformed,
            &policy,
            &|_| Ok(ring_cfg.clone()),
            TensorChallengeShape::Flat,
            true,
        )
        .expect_err("malformed non-tiny geometry must propagate");

        assert!(error.to_string().contains("geometry does not match"));
    }

    #[test]
    fn recursive_policy_empty_precommit_dense_still_uses_scalar_planner() {
        let mut policy = flat_policy();
        policy.decomposition.log_commit_bound = 8;
        policy.recursive_setup_planning = true;
        let key = AkitaScheduleLookupKey::single(PolynomialGroupLayout::new(12, 1));

        let schedule = find_group_batch_schedule(&key, &policy, ring_challenge_config, fold_shape)
            .expect("empty-precommit recursive keys must still use the scalar planner");
        for fold in schedule.fold_steps() {
            assert_eq!(
                fold.params.setup_contribution_mode,
                SetupContributionMode::Direct
            );
        }
    }

    #[test]
    fn multi_group_root_schedule_searches_policy_basis_range() {
        let mut policy = flat_policy();
        policy.decomposition.log_basis = 3;
        policy.basis_range = (4, 4);
        let pre_key = PolynomialGroupLayout::new(20, 1);
        let key = AkitaScheduleLookupKey {
            final_group: PolynomialGroupLayout::new(40, 2),
            precommitteds: vec![precommitted_from_policy(pre_key, &policy)],
        };

        let schedule = find_group_batch_schedule(&key, &policy, ring_challenge_config, fold_shape)
            .expect("multi-group schedule");
        let Step::Fold(root) = schedule.steps.first().expect("multi-group root step") else {
            panic!("expected multi-group root fold");
        };

        assert_eq!(root.params.log_basis, 4);
    }

    #[test]
    fn multi_group_schedule_can_start_with_fold() {
        let mut policy = flat_policy();
        policy.basis_range = (4, 4);
        let pre_key = PolynomialGroupLayout::new(20, 1);
        let key = AkitaScheduleLookupKey {
            final_group: PolynomialGroupLayout::new(40, 2),
            precommitteds: vec![precommitted_from_policy(pre_key, &policy)],
        };

        let schedule = find_group_batch_schedule(&key, &policy, ring_challenge_config, fold_shape)
            .expect("multi-group schedule");
        let Step::Fold(root) = schedule.steps.first().expect("multi-group root step") else {
            panic!("expected multi-group root fold");
        };

        assert_eq!(root.current_w_len, 1usize << key.final_group.num_vars());
        assert_eq!(
            root.params.precommitted_groups.len(),
            key.precommitteds.len()
        );
        assert!(root.next_w_len > 0);
    }

    #[test]
    fn recursive_policy_marks_threshold_multi_group_root() {
        let mut policy = flat_policy();
        policy.basis_range = (4, 4);
        policy.recursive_setup_planning = true;
        let pre_key = PolynomialGroupLayout::new(20, 1);
        let key = AkitaScheduleLookupKey {
            final_group: PolynomialGroupLayout::new(40, 2),
            precommitteds: vec![precommitted_from_policy(pre_key, &policy)],
        };

        let schedule = find_group_batch_schedule(&key, &policy, ring_challenge_config, fold_shape)
            .expect("recursive multi-group schedule");
        let folds = schedule
            .steps
            .iter()
            .filter_map(|step| match step {
                Step::Fold(fold) => Some(fold),
                Step::Direct(_) => None,
            })
            .collect::<Vec<_>>();

        assert!(folds.len() >= 2, "test key must plan a fold-again edge");
        assert_eq!(
            folds[0].params.setup_contribution_mode,
            SetupContributionMode::Recursive
        );
        assert!(folds[1].params.setup_prefix.is_some());
        assert_eq!(
            folds[1]
                .params
                .setup_prefix
                .as_ref()
                .expect("setup prefix")
                .commitment_params
                .layout
                .group
                .num_polynomials(),
            1
        );
    }

    #[test]
    fn recursive_policy_delegates_empty_precommit_keys_to_scalar_planner() {
        let mut policy = flat_policy();
        policy.basis_range = (4, 4);
        policy.recursive_setup_planning = true;
        let key = AkitaScheduleLookupKey {
            final_group: PolynomialGroupLayout::new(40, 2),
            precommitteds: Vec::new(),
        };

        let schedule = find_group_batch_schedule(&key, &policy, ring_challenge_config, fold_shape)
            .expect("empty-precommit keys must use the scalar planner");
        for fold in schedule.fold_steps() {
            assert_eq!(
                fold.params.setup_contribution_mode,
                SetupContributionMode::Direct
            );
            assert!(fold.params.setup_prefix.is_none());
            assert!(fold.params.precommitted_groups.is_empty());
        }
    }

    #[test]
    fn recursive_policy_rejects_singleton_scheduler() {
        let mut policy = flat_policy();
        policy.recursive_setup_planning = true;
        let key = PolynomialGroupLayout::new(24, 1);

        let err = find_schedule(key, &policy, ring_challenge_config, fold_shape)
            .expect_err("recursive setup planning requires grouped batch");

        assert!(err
            .to_string()
            .contains("requires the grouped-batch scheduler"));
    }

    #[test]
    fn multi_group_schedule_allows_precommitted_group_larger_than_final_group() {
        let mut policy = flat_policy();
        policy.decomposition.log_open_bound = Some(128);
        policy.basis_range = (4, 4);
        let pre_key = PolynomialGroupLayout::new(24, 1);
        let key = AkitaScheduleLookupKey {
            final_group: PolynomialGroupLayout::new(20, 1),
            precommitteds: vec![precommitted_from_policy(pre_key, &policy)],
        };

        let schedule = find_group_batch_schedule(&key, &policy, ring_challenge_config, fold_shape)
            .expect("multi-group schedule should allow larger precommitted groups");
        let Step::Fold(root) = schedule.steps.first().expect("multi-group root step") else {
            panic!("expected multi-group root fold");
        };

        assert_eq!(root.params.precommitted_groups[0].layout.group, pre_key);
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
            .expect_err("dense multi-group root schedules are phase-1 unsupported");

        assert!(matches!(err, AkitaError::InvalidSetup(_)));
    }

    #[test]
    fn suffix_dp_omits_best_direct_when_incoming_setup_prefix_present() {
        let mut policy = flat_policy();
        policy.recursive_setup_planning = true;
        policy.basis_range = (4, 4);
        let key = PolynomialGroupLayout::new(24, 1);
        let ring_cfg = ring_challenge_config(policy.ring_dimension).expect("ring cfg");
        let ctx = SuffixCtx {
            policy: &policy,
            ring_challenge_cfg: &ring_cfg,
            num_vars: key.num_vars(),
            key,
        };
        let mut memo = ScheduleMemo::default();
        let suffix = derive_optimal_suffix_schedule(
            &ctx,
            &mut memo,
            SuffixState {
                level: 1,
                current_witness_len: 1 << 18,
                current_witness_len_terminal: 1 << 16,
                current_lb: 4,
                incoming_setup_prefix: Some(1 << 12),
            },
            0,
        )
        .expect("suffix with incoming prefix");
        assert!(suffix.best_direct.is_none());
    }

    #[test]
    fn recursive_policy_terminal_child_stays_direct_when_threshold_met() {
        let mut policy = flat_policy();
        policy.basis_range = (4, 4);
        policy.recursive_setup_planning = true;
        policy.decomposition.log_open_bound = Some(128);
        let pre_key = PolynomialGroupLayout::new(20, 1);
        let key = AkitaScheduleLookupKey {
            final_group: PolynomialGroupLayout::new(24, 1),
            precommitteds: vec![precommitted_from_policy(pre_key, &policy)],
        };

        let schedule = find_group_batch_schedule(&key, &policy, ring_challenge_config, fold_shape)
            .expect("schedule with terminal child");
        let folds = schedule
            .steps
            .iter()
            .filter_map(|step| match step {
                Step::Fold(fold) => Some(fold),
                Step::Direct(_) => None,
            })
            .collect::<Vec<_>>();
        if folds.len() == 1 {
            assert_eq!(
                folds[0].params.setup_contribution_mode,
                SetupContributionMode::Direct
            );
            return;
        }
        let terminal_fold = folds.last().expect("terminal fold");
        assert_eq!(
            terminal_fold.params.setup_contribution_mode,
            SetupContributionMode::Direct
        );
        assert!(!terminal_fold.params.has_precommitted_groups());
        let predecessor = folds[folds.len() - 2];
        if matches!(schedule.steps.get(folds.len()), Some(Step::Direct(_))) {
            // The fold immediately before Direct must remain Direct-mode when
            // its child is the terminal fold.
            if folds.len() == 2 {
                assert_eq!(
                    predecessor.params.setup_contribution_mode,
                    SetupContributionMode::Direct
                );
            }
        }
    }

    #[test]
    fn recursive_fold_successor_carries_only_setup_prefix_group() {
        let mut policy = flat_policy();
        policy.basis_range = (4, 4);
        policy.recursive_setup_planning = true;
        let pre_key = PolynomialGroupLayout::new(20, 1);
        let key = AkitaScheduleLookupKey {
            final_group: PolynomialGroupLayout::new(40, 2),
            precommitteds: vec![precommitted_from_policy(pre_key, &policy)],
        };

        let schedule = find_group_batch_schedule(&key, &policy, ring_challenge_config, fold_shape)
            .expect("recursive multi-group schedule");
        for (index, step) in schedule.steps.iter().enumerate() {
            let Step::Fold(fold) = step else {
                continue;
            };
            if fold.params.setup_contribution_mode != SetupContributionMode::Recursive {
                continue;
            }
            let Step::Fold(successor) = schedule
                .steps
                .get(index + 1)
                .expect("recursive fold must have a successor")
            else {
                panic!("recursive fold successor must be another fold");
            };
            assert!(successor.params.setup_prefix.is_some());
            assert!(successor.params.precommitted_groups.is_empty());
            assert_eq!(successor.params.precommitted_group_count(), 1);
        }
    }
}
