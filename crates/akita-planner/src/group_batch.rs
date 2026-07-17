//! Multi-group root-batch schedule planning.

use akita_challenges::{SparseChallengeConfig, TensorChallengeShape};
use akita_field::AkitaError;
use akita_types::sis::{
    compute_num_digits_full_field, decomposed_s_block_ring_count, decomposed_t_ring_count,
    decomposed_w_ring_count, fold_witness_digit_plan, min_secure_rank, num_digits_open,
    num_digits_s_commit, rounded_up_collision_inf_norm, rounded_up_role_a_inf_norm, AjtaiKeyParams,
    FoldChallengeNorms, FoldWitnessLinfCapConfig, FoldWitnessNorms, SisTableKey,
};
use akita_types::{
    active_setup_field_len, direct_witness_bytes, extension_opening_reduction_level_bytes,
    level_proof_bytes, padded_setup_prefix_len, shared_d_digit_log_basis, AkitaScheduleInputs,
    AkitaScheduleLookupKey, CleartextWitnessShape, CommitmentRingDims, DecompositionParams,
    DirectStep, FoldStep, LevelParams, OpeningClaimsLayout, PolynomialGroupLayout,
    PrecommittedGroupParams, PrecommittedLevelParams, RelationMatrixRowLayout, Schedule,
    SetupContributionMode, Step, WitnessLayout, SETUP_OFFLOAD_MIN_PREFIX_FIELD_LEN,
};

use crate::schedule_params::{
    derive_optimal_suffix_schedule, find_schedule, optimize_fold_challenge_shape,
    RingChallengeConfigFn, ScheduleMemo, SuffixCtx, SuffixState,
};
use crate::PlannerPolicy;

fn sis_key(
    policy: &PlannerPolicy,
    role: akita_types::SisMatrixRole,
    coeff_linf_bound: u128,
) -> SisTableKey {
    SisTableKey {
        policy: policy.sis_security_policy,
        table_digest: policy.sis_table_digest,
        modulus_profile: policy.sis_modulus_profile,
        role,
        ring_dimension: policy.ring_dimension as u32,
        coeff_linf_bound,
    }
}
pub(crate) fn group_root_params_from_layout(
    layout: &PrecommittedGroupParams,
    policy: &PlannerPolicy,
    ring_challenge_config: RingChallengeConfigFn<'_>,
) -> Result<PrecommittedLevelParams, AkitaError> {
    layout.validate_frozen_precommit(policy.ring_dimension)?;

    let ring_challenge_cfg = ring_challenge_config(policy.ring_dimension)?;
    let d = policy.ring_dimension;
    let family = policy.sis_modulus_profile;
    let level_decomp = DecompositionParams {
        log_basis: layout.log_basis,
        ..policy.decomposition
    };
    let num_digits_commit = num_digits_s_commit(level_decomp, true);
    let num_digits_open = num_digits_open(level_decomp);
    let num_live_blocks = layout.num_live_blocks;
    let num_positions_per_block = layout.num_positions_per_block;
    let fold_challenge_shape = layout.fold_challenge_shape;
    let width_s = decomposed_s_block_ring_count(num_positions_per_block, num_digits_commit)
        .ok_or_else(|| AkitaError::InvalidSetup("multi-group A width overflow".to_string()))?;
    let norm_s = rounded_up_role_a_inf_norm(
        policy.sis_security_policy,
        family,
        d,
        level_decomp,
        &ring_challenge_cfg,
        fold_challenge_shape,
        true,
        policy.onehot_chunk_size,
        policy.ring_subfield_norm_bound,
        num_live_blocks,
        layout.group.num_polynomials(),
        width_s as u64,
    )
    .ok_or_else(|| AkitaError::InvalidSetup("no multi-group A-role norm".to_string()))?;
    let min_n_a = min_secure_rank(
        SisTableKey {
            policy: policy.sis_security_policy,
            table_digest: policy.sis_table_digest,
            modulus_profile: family,
            role: akita_types::SisMatrixRole::A,
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
        policy.sis_security_policy,
        policy.sis_table_digest,
        family,
        akita_types::SisMatrixRole::A,
        layout.n_a,
        width_s,
        norm_s,
        d,
    )?;

    let b_norm_basis = policy.basis_range.1;
    let norm_t = rounded_up_collision_inf_norm(
        policy.sis_security_policy,
        family,
        akita_types::SisMatrixRole::B,
        d,
        b_norm_basis,
    )
    .ok_or_else(|| AkitaError::InvalidSetup("no multi-group B-role norm".to_string()))?;
    let width_t = decomposed_t_ring_count(
        layout.n_a,
        num_digits_open,
        num_live_blocks,
        layout.group.num_polynomials(),
    )
    .ok_or_else(|| AkitaError::InvalidSetup("setup B width overflow".to_string()))?;
    let min_n_b = min_secure_rank(
        sis_key(policy, akita_types::SisMatrixRole::B, norm_t),
        width_t as u64,
    )
    .ok_or_else(|| AkitaError::InvalidSetup("no multi-group B-role rank".to_string()))?;
    let n_b = if layout.conservative_n_b < min_n_b {
        return Err(AkitaError::InvalidSetup(
            "precommitted group conservative B rank is below multi-group root requirement"
                .to_string(),
        ));
    } else {
        layout.conservative_n_b
    };
    let b_key = AjtaiKeyParams::try_new(
        policy.sis_security_policy,
        policy.sis_table_digest,
        family,
        akita_types::SisMatrixRole::B,
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
        num_live_blocks,
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
        num_digits_commit,
        num_digits_open,
        num_digits_fold_one,
    })
}

struct MultiGroupRootCandidateCtx<'a> {
    policy: &'a PlannerPolicy,
    ring_challenge_cfg: &'a SparseChallengeConfig,
    requested_fold_shape: TensorChallengeShape,
    precommitted_d_width: usize,
    precommitted_groups: &'a [PrecommittedLevelParams],
}

pub(crate) fn multi_group_root_precommitted_groups(
    key: &AkitaScheduleLookupKey,
    policy: &PlannerPolicy,
    ring_challenge_config: RingChallengeConfigFn<'_>,
) -> Result<(Vec<PrecommittedLevelParams>, usize), AkitaError> {
    if key.precommitteds.is_empty() {
        return Err(AkitaError::InvalidSetup(
            "multi-group root params require at least one precommitted group".to_string(),
        ));
    }

    let precommitted_groups = key
        .precommitteds
        .iter()
        .map(|layout| group_root_params_from_layout(layout, policy, ring_challenge_config))
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
    num_live_blocks: usize,
    num_positions_per_block: usize,
    num_digits_commit: usize,
    num_digits_open: usize,
    num_digits_fold: usize,
    context: &'static str,
) -> Result<u128, AkitaError> {
    // Match `optimal_block_geometry_split`: opening `(1 + n_a) * delta_open * 2^r`
    // plus folded witness `delta_commit * delta_fold * 2^m`.
    let e_hat_cost = checked_score_mul(num_digits_open as u128, num_live_blocks, context)?;
    let t_hat_cost = checked_score_mul(num_digits_open as u128, n_a, context)?;
    let t_hat_cost = checked_score_mul(t_hat_cost, num_live_blocks, context)?;
    let opening_cost = checked_score_add(e_hat_cost, t_hat_cost, context)?;

    let z_hat_cost = checked_score_mul(num_digits_commit as u128, num_digits_fold, context)?;
    let z_hat_cost = checked_score_mul(z_hat_cost, num_positions_per_block, context)?;

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
        params.num_live_blocks,
        params.num_positions_per_block,
        params.num_digits_commit,
        params.num_digits_open,
        main_num_digits_fold,
        "multi-group main root-direct score",
    )?;

    for group in &params.precommitted_groups {
        let group_cost = root_direct_split_cost(
            group.a_key.row_len(),
            group.layout.num_live_blocks,
            group.layout.num_positions_per_block,
            group.num_digits_commit,
            group.num_digits_open,
            group.num_digits_fold_one,
            "multi-group precommitted root-direct score",
        )?;
        total = checked_score_add(total, group_cost, "multi-group root-direct score total")?;
    }

    Ok(total)
}

fn multi_group_root_next_w_len(
    field_bits: u32,
    params: &LevelParams,
    opening_batch: &OpeningClaimsLayout,
    layout: RelationMatrixRowLayout,
) -> Result<usize, AkitaError> {
    params.witness_chunk.validate()?;
    params.validate_opening_batch(opening_batch)?;
    let relation_rows = params.relation_matrix_row_count_for(opening_batch.num_groups(), layout)?;
    let witness_layout = WitnessLayout::new(
        params,
        opening_batch,
        params.witness_chunk.num_chunks,
        relation_rows,
        compute_num_digits_full_field(field_bits, params.log_basis),
    )?;
    witness_layout
        .total_len()
        .checked_mul(params.ring_dimension)
        .ok_or_else(|| AkitaError::InvalidSetup("multi-group next witness length overflow".into()))
}

fn multi_group_root_main_level_params_candidate(
    ctx: &MultiGroupRootCandidateCtx<'_>,
    main_num_polys: usize,
    log_basis: u32,
    position_index_bits: usize,
    block_index_bits: usize,
) -> Result<Option<LevelParams>, AkitaError> {
    let policy = ctx.policy;
    let d = policy.ring_dimension;
    let family = policy.sis_modulus_profile;
    let decomp = policy.decomposition;
    let level_decomp = DecompositionParams {
        log_basis,
        ..decomp
    };
    let num_digits_commit = num_digits_s_commit(level_decomp, true);
    let num_digits_open = num_digits_open(level_decomp);
    let Some(num_live_blocks) = 1usize.checked_shl(block_index_bits as u32) else {
        return Ok(None);
    };
    let Some(num_positions_per_block) = 1usize.checked_shl(position_index_bits as u32) else {
        return Ok(None);
    };
    let Some(num_live_ring_elements_per_claim) =
        num_live_blocks.checked_mul(num_positions_per_block)
    else {
        return Ok(None);
    };
    let fold_challenge_shape =
        optimize_fold_challenge_shape(ctx.requested_fold_shape, num_live_blocks)?;

    let Some(width_s) = decomposed_s_block_ring_count(num_positions_per_block, num_digits_commit)
    else {
        return Ok(None);
    };
    let Some(norm_s) = rounded_up_role_a_inf_norm(
        policy.sis_security_policy,
        family,
        d,
        level_decomp,
        ctx.ring_challenge_cfg,
        fold_challenge_shape,
        true,
        policy.onehot_chunk_size,
        policy.ring_subfield_norm_bound,
        num_live_blocks,
        main_num_polys,
        width_s as u64,
    ) else {
        return Ok(None);
    };
    let Ok(a_key) = AjtaiKeyParams::try_new_with_min_rank(
        sis_key(policy, akita_types::SisMatrixRole::A, norm_s),
        width_s,
    ) else {
        return Ok(None);
    };
    let n_a = a_key.row_len();

    let Some(norm_t) = rounded_up_collision_inf_norm(
        policy.sis_security_policy,
        family,
        akita_types::SisMatrixRole::B,
        d,
        log_basis,
    ) else {
        return Ok(None);
    };
    let Some(width_t) =
        decomposed_t_ring_count(n_a, num_digits_open, num_live_blocks, main_num_polys)
    else {
        return Ok(None);
    };
    let Ok(b_key) = AjtaiKeyParams::try_new_with_min_rank(
        sis_key(policy, akita_types::SisMatrixRole::B, norm_t),
        width_t,
    ) else {
        return Ok(None);
    };

    let Some(main_d_width) =
        decomposed_w_ring_count(num_digits_open, num_live_blocks, main_num_polys)
    else {
        return Ok(None);
    };
    let d_width = main_d_width
        .checked_add(ctx.precommitted_d_width)
        .ok_or_else(|| AkitaError::InvalidSetup("multi-group D width overflow".to_string()))?;
    let d_log_basis = shared_d_digit_log_basis(log_basis, ctx.precommitted_groups);
    let Some(norm_w) = rounded_up_collision_inf_norm(
        policy.sis_security_policy,
        family,
        akita_types::SisMatrixRole::D,
        d,
        d_log_basis,
    ) else {
        return Ok(None);
    };
    let Ok(d_key) = AjtaiKeyParams::try_new_with_min_rank(
        sis_key(policy, akita_types::SisMatrixRole::D, norm_w),
        d_width,
    ) else {
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
        num_live_ring_elements_per_claim,
        num_positions_per_block,
        num_live_blocks,
        fold_challenge_config: *ctx.ring_challenge_cfg,
        fold_challenge_shape,
        num_digits_commit,
        num_digits_open,
        onehot_chunk_size,
        fold_linf_cap_config: FoldWitnessLinfCapConfig::worst_case_beta_only(),
        num_digits_fold_one: 1,
        field_bits_hint: 0,
        cached_num_digits_block_claims: 0,
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
    let (precommitted_groups, precommitted_d_width) =
        multi_group_root_precommitted_groups(key, policy, ring_challenge_config)?;

    let ring_challenge_cfg = ring_challenge_config(policy.ring_dimension)?;
    let main_num_polys = key.final_group.num_polynomials();
    let main_num_vars = key.final_group.num_vars();
    let candidate_ctx = MultiGroupRootCandidateCtx {
        policy,
        ring_challenge_cfg: &ring_challenge_cfg,
        requested_fold_shape: fold_challenge_shape,
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
            let block_index_bits = reduced_vars / 2;
            vec![(reduced_vars - block_index_bits, block_index_bits)]
        } else {
            (1..reduced_vars)
                .rev()
                .map(|block_index_bits| (reduced_vars - block_index_bits, block_index_bits))
                .collect()
        }
    };
    let (min_log_basis, max_log_basis) = policy.basis_range;
    for candidate_log_basis in min_log_basis..=max_log_basis {
        for &(position_index_bits, block_index_bits) in &candidates {
            let Some(candidate) = multi_group_root_main_level_params_candidate(
                &candidate_ctx,
                main_num_polys,
                candidate_log_basis,
                position_index_bits,
                block_index_bits,
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
    let ring_challenge_config: RingChallengeConfigFn<'_> = &ring_challenge_config;
    let fold_shape_at_level = &fold_challenge_shape_at_level;
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

    let (precommitted_groups, precommitted_d_width) =
        multi_group_root_precommitted_groups(key, policy, ring_challenge_config)?;
    let ring_challenge_cfg = ring_challenge_config(policy.ring_dimension)?;
    let candidate_ctx = MultiGroupRootCandidateCtx {
        policy,
        ring_challenge_cfg: &ring_challenge_cfg,
        requested_fold_shape: fold_challenge_shape,
        precommitted_d_width,
        precommitted_groups: &precommitted_groups,
    };
    let suffix_ctx = SuffixCtx {
        policy,
        ring_challenge_cfg: &ring_challenge_cfg,
        fold_challenge_shape_at_level: fold_shape_at_level,
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
    let min_block_index_bits: usize = if reduced_vars >= 3 { 1 } else { 0 };
    let max_block_index_bits: usize = (reduced_vars - 1).min(usize::BITS as usize - 1);
    let (configured_min_log_basis, max_log_basis) = policy.basis_range;
    let min_log_basis = configured_min_log_basis
        .max(policy.decomposition.log_basis)
        .max(if policy.decomposition.field_bits() < 128 {
            5
        } else {
            0
        });

    for candidate_log_basis in min_log_basis..=max_log_basis {
        for block_index_bits in (min_block_index_bits..=max_block_index_bits).rev() {
            let position_index_bits = reduced_vars - block_index_bits;
            let Some(mut candidate_params) = multi_group_root_main_level_params_candidate(
                &candidate_ctx,
                key.final_group.num_polynomials(),
                candidate_log_basis,
                position_index_bits,
                block_index_bits,
            )?
            else {
                continue;
            };
            let root_num_chunks = policy.chunks_at_level(0);
            if candidate_params
                .precommitted_groups
                .iter()
                .any(|group| group.layout.num_live_blocks < root_num_chunks)
            {
                continue;
            }
            candidate_params.witness_chunk = policy.witness_chunk_for_level(0);
            let opening_batch = key.opening_layout()?;
            let next_w_len = multi_group_root_next_w_len(
                field_bits,
                &candidate_params,
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

            let natural_len = active_setup_field_len(&candidate_params, &opening_batch)?;
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
        RelationMatrixRowLayout, SisModulusProfileId, SisTableDigest, DEFAULT_SIS_SECURITY_POLICY,
    };

    fn flat_policy() -> PlannerPolicy {
        PlannerPolicy {
            ring_dimension: 64,
            decomposition: DecompositionParams {
                log_basis: 3,
                log_commit_bound: 1,
                log_open_bound: Some(8),
            },
            sis_modulus_profile: SisModulusProfileId::Q128OffsetA7F7,
            sis_security_policy: DEFAULT_SIS_SECURITY_POLICY,
            sis_table_digest: SisTableDigest::CURRENT,
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
        let block_index_bits = outer / 2;
        let position_index_bits = outer - block_index_bits;
        PrecommittedGroupParams {
            group: PolynomialGroupLayout::new(num_vars, num_polys),
            num_live_ring_elements_per_claim: 1usize << outer,
            num_positions_per_block: 1usize << position_index_bits,
            num_live_blocks: 1usize << block_index_bits,
            fold_challenge_shape: TensorChallengeShape::Flat,
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
    fn multi_group_root_direct_witness_len_sums_singleton_precommitted_groups() {
        let key = AkitaScheduleLookupKey {
            final_group: PolynomialGroupLayout::new(20, 3),
            precommitteds: vec![precommitted(1, 20), precommitted(1, 20)],
        };

        let expected_len = 3 * (1usize << 20) + (1usize << 20) + (1usize << 20);
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
        let num_live_blocks = 8usize;
        let num_digits_open = 3usize;
        let per_group_w =
            decomposed_w_ring_count(num_digits_open, num_live_blocks, 1).expect("w width");
        let scalar_w = decomposed_w_ring_count(num_digits_open, num_live_blocks, main_polys)
            .expect("scalar w");
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
            .checked_mul(root.params.num_live_blocks)
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
    fn find_group_batch_schedule_rejects_multi_polynomial_precommitted_group() {
        let policy = flat_policy();
        let key = AkitaScheduleLookupKey {
            final_group: PolynomialGroupLayout::new(40, 1),
            precommitteds: vec![precommitted(3, 20)],
        };

        let err = find_group_batch_schedule(&key, &policy, ring_challenge_config, fold_shape)
            .expect_err("multi-polynomial precommitted groups are not supported");
        assert!(matches!(err, AkitaError::InvalidSetup(_)));
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
            num_live_ring_elements_per_claim: 2,
            num_positions_per_block: 2,
            num_live_blocks: 1,
            fold_challenge_shape: TensorChallengeShape::Flat,
            log_basis: 3,
            n_a: 1,
            conservative_n_b: 1,
        };
        let ring_cfg = ring_challenge_config(policy.ring_dimension).expect("ring challenge");

        let error = group_root_params_from_layout(&malformed, &policy, &|_| Ok(ring_cfg))
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
        policy.decomposition.log_open_bound = Some(128);
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
    fn mixed_basis_d128_root_prices_shared_d_at_precommit_basis() {
        let mut precommit_policy = flat_policy();
        precommit_policy.ring_dimension = 128;
        precommit_policy.decomposition.log_basis = 3;
        precommit_policy.basis_range = (3, 3);
        let pre_key = PolynomialGroupLayout::new(20, 1);
        let frozen = precommitted_from_policy(pre_key, &precommit_policy);
        assert_eq!(frozen.log_basis, 3);

        let mut root_policy = precommit_policy;
        root_policy.decomposition.log_open_bound = Some(128);
        root_policy.decomposition.log_basis = 2;
        root_policy.basis_range = (2, 2);
        let key = AkitaScheduleLookupKey {
            final_group: PolynomialGroupLayout::new(40, 2),
            precommitteds: vec![frozen],
        };
        let schedule =
            find_group_batch_schedule(&key, &root_policy, ring_challenge_config, fold_shape)
                .expect("mixed-basis D128 root schedule");
        let Step::Fold(root) = schedule.steps.first().expect("mixed-basis root step") else {
            panic!("expected mixed-basis root fold");
        };

        assert_eq!(root.params.log_basis, 2);
        assert_eq!(root.params.shared_d_digit_log_basis(), 3);
        let expected_d_bound = rounded_up_collision_inf_norm(
            root_policy.sis_security_policy,
            root_policy.sis_modulus_profile,
            akita_types::SisMatrixRole::D,
            root_policy.ring_dimension,
            3,
        )
        .expect("D128 basis-3 D bound");
        assert_eq!(root.params.d_key.coeff_linf_bound(), expected_d_bound);
    }

    #[test]
    fn multi_group_schedule_can_start_with_fold() {
        let mut policy = flat_policy();
        policy.decomposition.log_open_bound = Some(128);
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
        policy.decomposition.log_open_bound = Some(128);
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
    fn multi_group_schedule_rejects_precommitted_group_larger_than_half_final() {
        let mut policy = flat_policy();
        policy.decomposition.log_open_bound = Some(128);
        policy.basis_range = (4, 4);
        let pre_key = PolynomialGroupLayout::new(24, 1);
        let key = AkitaScheduleLookupKey {
            final_group: PolynomialGroupLayout::new(20, 1),
            precommitteds: vec![precommitted_from_policy(pre_key, &policy)],
        };

        let err = find_group_batch_schedule(&key, &policy, ring_challenge_config, fold_shape)
            .expect_err("precommitted num_vars above half the final group must be rejected");
        assert!(matches!(err, AkitaError::InvalidInput(_)));
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
            fold_challenge_shape_at_level: &fold_shape,
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
        policy.decomposition.log_open_bound = Some(128);
        let pre_key = PolynomialGroupLayout::new(12, 1);
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
