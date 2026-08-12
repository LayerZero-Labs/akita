//! Root schedule planning.

use akita_challenges::SparseChallengeConfig;
use akita_field::AkitaError;
use akita_types::sis::{
    decomposed_s_block_ring_count, decomposed_t_ring_count, decomposed_w_ring_count,
    num_digits_open, rounded_up_collision_inf_norm, rounded_up_role_a_inf_norm, HonestFoldPolicy,
    HonestFoldPolicySpec, HonestFoldSizingQuery, InnerCommitMatrixParams, OpenCommitMatrixParams,
    OuterCommitMatrixParams, SisMatrixRole,
};
use akita_types::{
    AkitaScheduleLookupKey, CommitmentRingDims, CommittedGroupParams, CommittedGroupProfile,
    DecompositionParams, OpeningClaimsLayout, PlannedFoldSchedule, PolynomialGroupLayout,
    PrecommittedGroupAdmissionPolicy, PrecommittedLevelParams,
};

use akita_schedules::planner_support::{projected_collision_role_price, sis_key_at_dimension};

use crate::schedule_params::{
    derive_selected_suffix_schedule, materialize_candidate_schedule, recursive_split_search_domain,
    select_complete_candidate, RingChallengeConfigFn, ScheduleMemo, SuffixCtx, SuffixState,
};
use crate::PlannerPolicy;

type PrecommittedGroupSeed = (CommittedGroupProfile, HonestFoldPolicySpec);

fn materialize_precommitted_group_for_open_basis(
    (layout, honest_fold_policy): &PrecommittedGroupSeed,
    policy: &PlannerPolicy,
    ring_challenge_cfg: SparseChallengeConfig,
    log_basis_open: u32,
) -> Result<Option<PrecommittedLevelParams>, AkitaError> {
    let ring_dimension = layout.inner_commit_matrix.ring_dimension();
    let num_chunks = policy.chunks_at_level(0);
    let num_fold_coeffs = layout
        .inner_commit_matrix
        .input_width()
        .checked_mul(ring_dimension)
        .and_then(|count| count.checked_mul(num_chunks))
        .ok_or_else(|| AkitaError::InvalidSetup("precommitted fold width overflow".into()))?;
    let group_claims = layout.group.num_polynomials();
    let num_digits_fold = honest_fold_policy.num_digits_fold(HonestFoldSizingQuery {
        ring_dimension,
        num_claims: group_claims,
        num_live_blocks: layout.num_live_blocks,
        num_chunks,
        num_fold_coeffs,
        witness_norms: honest_fold_policy
            .witness_norms_for_inner_basis(layout.log_basis_inner, ring_dimension),
        log_basis_response: log_basis_open,
        challenge_config: &ring_challenge_cfg,
    })?;
    let Some(required_a_bound) = rounded_up_role_a_inf_norm(
        policy.sis_security_policy,
        policy.sis_table_digest,
        policy.sis_modulus_profile,
        ring_dimension,
        log_basis_open,
        &ring_challenge_cfg,
        num_digits_fold,
    ) else {
        return Ok(None);
    };
    let declared_a_bound = layout
        .inner_commit_matrix
        .coeff_linf_bound()
        .ok_or_else(|| AkitaError::InvalidSetup("precommitted A cannot use an L2 route".into()))?;
    let Some(required_b_bound) = rounded_up_collision_inf_norm(
        policy.sis_security_policy,
        policy.sis_modulus_profile,
        SisMatrixRole::Outer,
        layout.outer_commit_matrix.ring_dimension(),
        log_basis_open,
    ) else {
        return Ok(None);
    };
    if required_a_bound > declared_a_bound
        || required_b_bound > layout.outer_commit_matrix.coeff_linf_bound()
    {
        return Ok(None);
    }
    PrecommittedLevelParams::admit(
        *layout,
        num_digits_fold,
        PrecommittedGroupAdmissionPolicy {
            decomposition: policy.decomposition,
            sis_security_policy: policy.sis_security_policy,
            sis_table_digest: policy.sis_table_digest,
            sis_modulus_profile: policy.sis_modulus_profile,
        },
        ring_challenge_cfg,
        log_basis_open,
    )
    .map(Some)
}

struct MultiGroupRootCandidateCtx<'a> {
    policy: &'a PlannerPolicy,
    dimensions: CommitmentRingDims,
    ring_challenge_cfg: &'a SparseChallengeConfig,
    final_honest_fold_policy: HonestFoldPolicySpec,
    main_num_polys: usize,
    source: crate::InnerBasisSource,
}

fn precommitted_groups_for_open_basis(
    seeds: &[PrecommittedGroupSeed],
    policy: &PlannerPolicy,
    ring_challenge_config: &dyn Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    shared_opening_ring_dimension: usize,
    log_basis_open: u32,
) -> Result<Option<(Vec<PrecommittedLevelParams>, usize)>, AkitaError> {
    let mut groups = Vec::with_capacity(seeds.len());
    for group in seeds {
        let ring_challenge_cfg =
            ring_challenge_config(group.0.inner_commit_matrix.ring_dimension())?;
        let Some(materialized) = materialize_precommitted_group_for_open_basis(
            group,
            policy,
            ring_challenge_cfg,
            log_basis_open,
        )?
        else {
            return Ok(None);
        };
        groups.push(materialized);
    }
    let mut d_width = 0usize;
    for group in &groups {
        d_width = d_width
            .checked_add(group.d_segment_width(shared_opening_ring_dimension)?)
            .ok_or_else(|| AkitaError::InvalidSetup("root batch D width overflow".to_string()))?;
    }
    Ok(Some((groups, d_width)))
}

pub(crate) fn root_batch_next_w_len(
    field_bits: u32,
    params: &CommittedGroupParams,
    opening_batch: &OpeningClaimsLayout,
) -> Result<Option<usize>, AkitaError> {
    if !params.compression_sources_supported()? {
        return Ok(None);
    }
    params
        .output_witness_len_for_field_bits(field_bits, opening_batch)
        .map(Some)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn root_level_candidates_for_basis(
    key: &AkitaScheduleLookupKey,
    final_honest_fold_policy: HonestFoldPolicySpec,
    precommitted_honest_fold_policies: &[HonestFoldPolicySpec],
    policy: &PlannerPolicy,
    dimensions: CommitmentRingDims,
    ring_challenge_cfg: &SparseChallengeConfig,
    ring_challenge_config: &dyn Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    root_input_witness_len: usize,
    candidate_log_basis_inner: u32,
    candidate_log_basis_open: u32,
    require_witness_contraction: bool,
) -> Result<Vec<(CommittedGroupParams, usize)>, AkitaError> {
    dimensions.validate_role_projection()?;
    let field_bits = policy.decomposition.field_bits();
    let alpha = dimensions.d_a().trailing_zeros() as usize;
    let reduced_vars = key.final_group.num_vars().saturating_sub(alpha);
    if reduced_vars == 0 {
        return Err(AkitaError::UnsupportedSchedule(format!(
            "root batch num_vars={} does not exceed log2(ring_dimension)={alpha}",
            key.final_group.num_vars()
        )));
    }

    if precommitted_honest_fold_policies.len() != key.precommitteds.len() {
        return Err(AkitaError::InvalidSetup(
            "group-batch planning requires one honest fold policy per precommitted profile"
                .to_string(),
        ));
    }
    let precommitted_groups = key
        .precommitteds
        .iter()
        .copied()
        .zip(precommitted_honest_fold_policies.iter().copied())
        .collect::<Vec<PrecommittedGroupSeed>>();
    let candidate_ctx = MultiGroupRootCandidateCtx {
        policy,
        dimensions,
        ring_challenge_cfg,
        final_honest_fold_policy,
        main_num_polys: key.final_group.num_polynomials(),
        source: crate::schedule_params::root_inner_basis_source(
            final_honest_fold_policy,
            policy.decomposition.log_commit_bound,
        ),
    };
    let opening_batch = key.opening_layout()?;
    let initial_witness_len_bits = root_input_witness_len
        .checked_mul(field_bits as usize)
        .ok_or_else(|| AkitaError::InvalidSetup("root batch witness bit length overflow".into()))?;
    let min_block_index_bits: usize = if reduced_vars >= 3 { 1 } else { 0 };
    let max_block_index_bits: usize = (reduced_vars - 1).min(usize::BITS as usize - 1);
    let num_ring_elems = 1usize.checked_shl(reduced_vars as u32).ok_or_else(|| {
        AkitaError::InvalidSetup("root reduced-variable domain is too large".into())
    })?;
    let delta_commit = candidate_ctx
        .source
        .num_digits_inner(policy.decomposition, candidate_log_basis_inner)?;
    let delta_open = num_digits_open(DecompositionParams {
        log_basis: candidate_log_basis_open,
        ..policy.decomposition
    });
    let mut split_domain = recursive_split_search_domain(
        policy.recursive_split_search_policy,
        num_ring_elems,
        reduced_vars,
        delta_commit,
        delta_open,
        policy.chunks_at_level(0),
    );
    if min_block_index_bits == 0 {
        split_domain.push(0);
    }
    split_domain.retain(|&split| min_block_index_bits <= split && split <= max_block_index_bits);
    split_domain.sort_unstable_by(|left, right| right.cmp(left));
    split_domain.dedup();

    let mut candidates = Vec::new();
    let shared_opening_ring_dimension = dimensions.d_d();
    if precommitted_groups.iter().any(|group| {
        !group
            .0
            .inner_commit_matrix
            .ring_dimension()
            .is_multiple_of(shared_opening_ring_dimension)
    }) {
        return Ok(Vec::new());
    }
    let Some((candidate_precommitted_groups, candidate_precommitted_d_width)) =
        precommitted_groups_for_open_basis(
            &precommitted_groups,
            policy,
            ring_challenge_config,
            shared_opening_ring_dimension,
            candidate_log_basis_open,
        )?
    else {
        return Ok(Vec::new());
    };
    for block_index_bits in split_domain {
        let position_index_bits = reduced_vars - block_index_bits;
        let Some(mut candidate_params) = root_final_group_level_params_candidate(
            &candidate_ctx,
            candidate_log_basis_inner,
            candidate_log_basis_open,
            position_index_bits,
            block_index_bits,
            &candidate_precommitted_groups,
            candidate_precommitted_d_width,
        )?
        else {
            continue;
        };
        candidate_params.witness_chunk = crate::policy::witness_chunk_at_level(policy, 0);
        let Some(output_witness_len) =
            root_batch_next_w_len(field_bits, &candidate_params, &opening_batch)?
        else {
            continue;
        };
        if require_witness_contraction
            && output_witness_len
                .checked_mul(candidate_log_basis_open as usize)
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("root batch next witness bit length overflow".into())
                })?
                >= initial_witness_len_bits
        {
            continue;
        }
        candidates.push((candidate_params, output_witness_len));
    }

    Ok(candidates)
}

fn root_final_group_level_params_candidate(
    ctx: &MultiGroupRootCandidateCtx<'_>,
    log_basis_inner: u32,
    log_basis_open: u32,
    position_index_bits: usize,
    block_index_bits: usize,
    precommitted_groups: &[PrecommittedLevelParams],
    precommitted_d_width: usize,
) -> Result<Option<CommittedGroupParams>, AkitaError> {
    let policy = ctx.policy;
    let dimensions = ctx.dimensions;
    let d_a = dimensions.d_a();
    let family = policy.sis_modulus_profile;
    let decomp = ctx.policy.decomposition;
    let level_decomp = DecompositionParams {
        log_basis: log_basis_open,
        ..decomp
    };
    let num_digits_inner = ctx
        .source
        .num_digits_inner(ctx.policy.decomposition, log_basis_inner)?;
    let num_digits_outer = num_digits_open(level_decomp);
    let num_digits_open = num_digits_outer;
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
    let Some(width_s) = decomposed_s_block_ring_count(num_positions_per_block, num_digits_inner)
    else {
        return Ok(None);
    };
    let num_chunks = policy.chunks_at_level(0);
    let Some(num_fold_coeffs) = width_s
        .checked_mul(d_a)
        .and_then(|count| count.checked_mul(num_chunks))
    else {
        return Ok(None);
    };
    let Ok(num_digits_fold) = ctx
        .final_honest_fold_policy
        .num_digits_fold(HonestFoldSizingQuery {
            ring_dimension: d_a,
            num_claims: ctx.main_num_polys,
            num_live_blocks,
            num_chunks,
            num_fold_coeffs,
            witness_norms: ctx
                .final_honest_fold_policy
                .witness_norms_for_inner_basis(log_basis_inner, d_a),
            log_basis_response: log_basis_open,
            challenge_config: ctx.ring_challenge_cfg,
        })
    else {
        return Ok(None);
    };
    let Some(norm_s) = rounded_up_role_a_inf_norm(
        policy.sis_security_policy,
        policy.sis_table_digest,
        family,
        d_a,
        log_basis_open,
        ctx.ring_challenge_cfg,
        num_digits_fold,
    ) else {
        return Ok(None);
    };
    let Ok(inner_commit_matrix) = InnerCommitMatrixParams::try_new_with_min_rank(
        sis_key_at_dimension(policy, akita_types::SisMatrixRole::Inner, d_a, norm_s),
        width_s,
    ) else {
        return Ok(None);
    };
    let n_a = inner_commit_matrix.output_rank();

    let Some(width_t) =
        decomposed_t_ring_count(n_a, num_digits_outer, num_live_blocks, ctx.main_num_polys)
    else {
        return Ok(None);
    };
    let Some((outer_key, width_t)) = projected_collision_role_price(
        policy,
        akita_types::SisMatrixRole::Outer,
        d_a,
        dimensions.d_b(),
        width_t,
        log_basis_open,
    ) else {
        return Ok(None);
    };
    let Ok(outer_commit_matrix) =
        OuterCommitMatrixParams::try_new_with_min_rank(outer_key, width_t)
    else {
        return Ok(None);
    };

    let Some(main_d_width) =
        decomposed_w_ring_count(num_digits_open, num_live_blocks, ctx.main_num_polys)
    else {
        return Ok(None);
    };
    let Some((open_key, main_d_width)) = projected_collision_role_price(
        policy,
        akita_types::SisMatrixRole::Open,
        d_a,
        dimensions.d_d(),
        main_d_width,
        log_basis_open,
    ) else {
        return Ok(None);
    };
    // Frozen precommit segments are already projected to the root's shared D
    // dimension by `precommitted_groups_for_open_basis`; only the main native
    // width passes through the A-to-D projection above.
    let d_width = main_d_width
        .checked_add(precommitted_d_width)
        .ok_or_else(|| AkitaError::InvalidSetup("root batch D width overflow".to_string()))?;
    let Ok(open_commit_matrix) = OpenCommitMatrixParams::try_new_with_min_rank(open_key, d_width)
    else {
        return Ok(None);
    };

    let params = CommittedGroupParams {
        payload_mode: akita_types::CommitmentPayloadMode::Compressed,
        log_basis_inner,
        log_basis_outer: log_basis_open,
        log_basis_open,
        inner_commit_matrix,
        outer_commit_matrix,
        open_commit_matrix,
        num_live_ring_elements_per_claim,
        num_positions_per_block,
        num_live_blocks,
        fold_challenge_config: *ctx.ring_challenge_cfg,
        num_digits_inner,
        num_digits_outer,
        num_digits_open,
        num_digits_fold,
        // Root folds use the ordinary single-chunk precommit path before the
        // schedule-level chunk policy is applied.
        witness_chunk: akita_types::ChunkedWitnessCfg::default(),
        precommitted_groups: precommitted_groups.to_vec(),
        setup_prefix: None,
    };

    Ok(Some(params))
}

fn validate_direct_adaptive_dimension_schedule_request(
    policy: &PlannerPolicy,
) -> Result<(), AkitaError> {
    if policy.selection_policy
        != crate::SelectionPolicyId::MinSetupMatrixFieldElementsThenProofPayload
    {
        return Err(AkitaError::InvalidSetup(
            "adaptive search requires MinSetupMatrixFieldElementsThenProofPayload".into(),
        ));
    }
    Ok(())
}

/// Build the fold schedule selected by a full schedule lookup key.
pub fn find_schedule(
    key: &AkitaScheduleLookupKey,
    final_honest_fold_policy: HonestFoldPolicySpec,
    precommitted_honest_fold_policies: &[HonestFoldPolicySpec],
    policy: &PlannerPolicy,
    ring_challenge_config: impl Fn(usize) -> Result<akita_challenges::SparseChallengeConfig, AkitaError>,
) -> Result<PlannedFoldSchedule, AkitaError> {
    akita_schedules::planner_support::validate_policy(policy)?;
    key.validate(policy.decomposition.field_bits())?;
    if matches!(
        policy.ring_dimension_schedule_mode,
        crate::RingDimensionScheduleMode::AdaptiveDimension { .. }
    ) && !policy.recursive_setup_planning
    {
        validate_direct_adaptive_dimension_schedule_request(policy)?;
    }
    let ring_challenge_config: RingChallengeConfigFn<'_> = &ring_challenge_config;
    let scalar_policy;
    let active_policy = if key.precommitteds.is_empty() && !policy.recursive_setup_planning {
        // Ordinary scalar families use the direct objective. Recursive
        // companion families retain their setup-aware objective so a scalar
        // root may carry its setup opening into the first suffix fold.
        scalar_policy = crate::policy::direct_only_policy(*policy);
        &scalar_policy
    } else {
        policy
    };
    let setup_field_budget = if active_policy.recursive_setup_planning {
        active_policy.setup_field_budget
    } else {
        None
    };
    let precommitted_honest_fold_policies = if key.precommitteds.is_empty() {
        &[]
    } else {
        precommitted_honest_fold_policies
    };
    let root_input_witness_len = 1usize
        .checked_shl(key.final_group.num_vars() as u32)
        .ok_or_else(|| {
            AkitaError::InvalidSetup("multi-group root-fold witness length overflow".to_string())
        })?;
    let ring_challenge_cfg = ring_challenge_config(active_policy.uniform_ring_dimension)?;
    let suffix_ctx = SuffixCtx {
        policy: active_policy,
        default_ring_challenge_cfg: &ring_challenge_cfg,
        ring_challenge_config,
        num_vars: key.final_group.num_vars(),
        key: PolynomialGroupLayout::singleton(key.final_group.num_vars()),
        setup_field_budget,
        root_lookup_key: Some(key),
        root_honest_fold_policy: Some(final_honest_fold_policy),
        precommitted_honest_fold_policies,
        level_zero_is_root: true,
    };
    let mut memo = ScheduleMemo::new();
    let dimension_ceiling = match active_policy.ring_dimension_schedule_mode {
        crate::RingDimensionScheduleMode::UniformDimension { ring_dimension } => {
            CommitmentRingDims::uniform(ring_dimension)
        }
        crate::RingDimensionScheduleMode::AdaptiveDimension {
            potential_a_dimensions,
            ..
        } => CommitmentRingDims::uniform(
            potential_a_dimensions
                .last()
                .copied()
                .ok_or_else(|| AkitaError::InvalidSetup("adaptive A domain is empty".into()))?,
        ),
    };
    let suffix = derive_selected_suffix_schedule(
        &suffix_ctx,
        &mut memo,
        SuffixState {
            level: 0,
            current_witness_len: root_input_witness_len,
            current_lb: 0,
            source_moment: None,
            incoming_setup_prefix: None,
            dimension_ceiling,
            payload_phase: akita_types::CommitmentPayloadPhase::CompressedPrefix,
        },
        0,
    )?;
    let best = match active_policy.selection_policy {
        crate::SelectionPolicyId::MinEstimatedProofPayload => {
            select_complete_candidate(active_policy, suffix.payload_candidates())?
        }
        crate::SelectionPolicyId::MinSetupMatrixFieldElementsThenProofPayload => {
            select_complete_candidate(active_policy, &suffix.mixed_frontier)?
        }
        crate::SelectionPolicyId::MinFirstDirectSetupThenPayload => {
            select_complete_candidate(active_policy, suffix.setup_candidates())?
        }
    };

    let Some(best) = best.cloned() else {
        if key.precommitteds.is_empty()
            && matches!(
                active_policy.ring_dimension_schedule_mode,
                crate::RingDimensionScheduleMode::AdaptiveDimension { .. }
            )
        {
            return Err(AkitaError::UnsupportedSchedule(format!(
                "no mixed-D schedule with at least two folds for num_vars={}, num_polynomials={}",
                key.final_group.num_vars(),
                key.final_group.num_polynomials()
            )));
        }
        return Err(AkitaError::UnsupportedSchedule(format!(
            "no multi-group schedule with at least two folds for num_vars={}",
            key.final_group.num_vars()
        )));
    };
    let first_direct_setup_field_len = if active_policy.recursive_setup_planning {
        Some(
            best.first_direct_setup_field_len
                .ok_or_else(|| {
                    AkitaError::InvalidSetup(
                        "recursive setup schedule is missing its first direct setup size".into(),
                    )
                })?
                .get(),
        )
    } else {
        None
    };
    materialize_candidate_schedule(
        best.total_bytes,
        best.setup_field_elements,
        first_direct_setup_field_len,
        best.folds.to_vec(),
        best.terminal.as_ref().clone(),
    )
}
