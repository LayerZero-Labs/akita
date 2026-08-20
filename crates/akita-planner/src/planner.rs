//! Root schedule planning.

use std::time::Instant;

use akita_error::AkitaError;
use akita_types::sis::{
    decomposed_s_block_ring_count, num_digits_open, rounded_up_collision_inf_norm,
    rounded_up_role_a_inf_norm, HonestFoldPolicy, HonestFoldPolicySpec, HonestFoldSizingQuery,
    OpenCommitMatrixParams, SisMatrixRole,
};
use akita_types::{
    AkitaScheduleLookupKey, CommitmentRingDims, CommittedGroupParams, DecompositionParams,
    GroupCommitPhaseParams, GroupOpenPhaseParams, OpeningClaimsLayout, PlannedFoldSchedule,
    PolynomialGroupLayout, PrecommittedGroupAdmissionPolicy,
};

use akita_schedules::planner_support::projected_collision_role_price;

use crate::schedule_params::{
    derive_ab_commitment_candidate, derive_selected_suffix_schedule,
    materialize_candidate_schedule, recursive_split_search_domain, select_complete_candidate,
    AbCommitmentCandidateRequest, PlannerOpeningCandidate, RingChallengeConfigFn, ScheduleMemo,
    SuffixCtx, SuffixState,
};
use crate::PlannerPolicy;

type PrecommittedGroupSeed = (GroupCommitPhaseParams, HonestFoldPolicySpec);

fn materialize_precommitted_group_for_open_basis(
    (layout, honest_fold_policy): &PrecommittedGroupSeed,
    policy: &PlannerPolicy,
    opening: PlannerOpeningCandidate,
    shared_opening_ring_dimension: usize,
    log_basis_open: u32,
) -> Result<Option<GroupOpenPhaseParams>, AkitaError> {
    let ring_dimension = layout.inner.matrix.ring_dimension();
    opening.validate_for(
        0,
        policy.claim_ext_degree,
        CommitmentRingDims {
            inner: ring_dimension,
            outer: layout.outer.matrix.ring_dimension(),
            opening: shared_opening_ring_dimension,
        },
    )?;
    let num_chunks = policy.chunks_at_level(0);
    let num_fold_coeffs = layout
        .inner
        .matrix
        .input_width()
        .checked_mul(ring_dimension)
        .and_then(|count| count.checked_mul(num_chunks))
        .ok_or_else(|| AkitaError::InvalidSetup("precommitted fold width overflow".into()))?;
    let group_claims = layout.group.num_polynomials();
    let num_digits_fold = honest_fold_policy.num_digits_fold(HonestFoldSizingQuery {
        ring_dimension,
        challenge_dimension: opening.challenge_dimension(ring_dimension),
        num_claims: group_claims,
        num_live_ring_elements_per_claim: layout.blocks.live_ring_elements_per_claim,
        num_live_blocks: layout.blocks.live_blocks,
        num_positions_per_block: layout.blocks.positions_per_block,
        num_chunks,
        num_fold_coeffs,
        witness_norms: honest_fold_policy
            .witness_norms_for_inner_basis(layout.inner.digits.log_basis, ring_dimension)?,
        log_basis_response: log_basis_open,
        challenge_config: &opening.challenge_config(),
    })?;
    let Some(required_a_bound) = rounded_up_role_a_inf_norm(
        policy.sis_security_policy,
        policy.sis_table_digest,
        policy.sis_modulus_profile,
        ring_dimension,
        log_basis_open,
        &opening.challenge_config(),
        num_digits_fold,
    ) else {
        return Ok(None);
    };
    let declared_a_bound =
        layout.inner.matrix.coeff_linf_bound().ok_or_else(|| {
            AkitaError::InvalidSetup("precommitted A cannot use an L2 route".into())
        })?;
    let Some(required_b_bound) = rounded_up_collision_inf_norm(
        policy.sis_security_policy,
        policy.sis_modulus_profile,
        SisMatrixRole::Outer,
        layout.outer.matrix.ring_dimension(),
        log_basis_open,
    ) else {
        return Ok(None);
    };
    if required_a_bound > declared_a_bound
        || required_b_bound > layout.outer.matrix.coeff_linf_bound()
    {
        return Ok(None);
    }
    GroupOpenPhaseParams::admit(
        *layout,
        num_digits_fold,
        PrecommittedGroupAdmissionPolicy {
            decomposition: policy.decomposition,
            sis_security_policy: policy.sis_security_policy,
            sis_table_digest: policy.sis_table_digest,
            sis_modulus_profile: policy.sis_modulus_profile,
        },
        opening.method(),
        opening.challenge_config(),
        log_basis_open,
    )
    .map(Some)
}

struct MultiGroupRootCandidateCtx<'a> {
    policy: &'a PlannerPolicy,
    dimensions: CommitmentRingDims,
    opening: PlannerOpeningCandidate,
    final_honest_fold_policy: HonestFoldPolicySpec,
    final_num_vars: usize,
    main_num_polys: usize,
    source: crate::InnerBasisSource,
}

struct RootFinalGroupCandidateInput<'a> {
    log_basis_inner: u32,
    log_basis_open: u32,
    position_index_bits: usize,
    block_index_bits: usize,
    outer_slice_count: akita_types::CommitmentSliceCount,
    precommitted_groups: &'a [GroupOpenPhaseParams],
    precommitted_d_width: usize,
}

fn precommitted_groups_for_open_basis(
    seeds: &[PrecommittedGroupSeed],
    openings: &[PlannerOpeningCandidate],
    policy: &PlannerPolicy,
    shared_opening_ring_dimension: usize,
    log_basis_open: u32,
) -> Result<Option<(Vec<GroupOpenPhaseParams>, usize)>, AkitaError> {
    let mut groups = Vec::with_capacity(seeds.len());
    for (group, opening) in seeds.iter().zip(openings.iter().copied()) {
        let Some(materialized) = materialize_precommitted_group_for_open_basis(
            group,
            policy,
            opening,
            shared_opening_ring_dimension,
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
            .checked_add(
                group.d_segment_width(policy.claim_ext_degree, shared_opening_ring_dimension)?,
            )
            .ok_or_else(|| AkitaError::InvalidSetup("root batch D width overflow".to_string()))?;
    }
    Ok(Some((groups, d_width)))
}

pub(crate) fn root_batch_next_w_len(
    field_bits: u32,
    extension_degree: usize,
    params: &CommittedGroupParams,
    opening_batch: &OpeningClaimsLayout,
) -> Result<Option<usize>, AkitaError> {
    if !params.compression_sources_supported()? {
        return Ok(None);
    }
    params
        .output_witness_len_for_field_bits(field_bits, extension_degree, opening_batch)
        .map(Some)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn root_level_candidates_for_basis(
    key: &AkitaScheduleLookupKey,
    final_honest_fold_policy: HonestFoldPolicySpec,
    precommitted_honest_fold_policies: &[HonestFoldPolicySpec],
    policy: &PlannerPolicy,
    dimensions: CommitmentRingDims,
    opening: PlannerOpeningCandidate,
    precommitted_openings: &[PlannerOpeningCandidate],
    root_input_witness_len: usize,
    candidate_log_basis_inner: u32,
    candidate_log_basis_open: u32,
    require_witness_contraction: bool,
) -> Result<Vec<(CommittedGroupParams, usize)>, AkitaError> {
    dimensions.validate_role_projection()?;
    opening.validate_for(0, policy.claim_ext_degree, dimensions)?;
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
    if precommitted_openings.len() != key.precommitteds.len() {
        return Err(AkitaError::InvalidSetup(
            "root precommit opening candidate count mismatch".into(),
        ));
    }
    if precommitted_openings
        .iter()
        .any(|candidate| candidate.is_coefficient_packing() != opening.is_coefficient_packing())
    {
        return Ok(Vec::new());
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
        opening,
        final_honest_fold_policy,
        final_num_vars: key.final_group.num_vars(),
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
    if !crate::schedule_params::precommitted_groups_support_opening_dimension(
        key.precommitteds.iter(),
        shared_opening_ring_dimension,
    ) {
        return Ok(Vec::new());
    }
    let Some((candidate_precommitted_groups, candidate_precommitted_d_width)) =
        precommitted_groups_for_open_basis(
            &precommitted_groups,
            precommitted_openings,
            policy,
            shared_opening_ring_dimension,
            candidate_log_basis_open,
        )?
    else {
        return Ok(Vec::new());
    };
    for block_index_bits in split_domain {
        let position_index_bits = reduced_vars - block_index_bits;
        let num_live_blocks = 1usize << block_index_bits;
        let mut slice_candidates = Vec::new();
        for outer_slice_count in akita_types::CommitmentSliceCount::ALL {
            if outer_slice_count
                .validate_for_commitment(
                    0,
                    akita_types::CommitmentPayloadMode::Compressed,
                    num_live_blocks,
                )
                .is_err()
            {
                continue;
            }
            let Some(mut candidate_params) = root_final_group_level_params_candidate(
                &candidate_ctx,
                RootFinalGroupCandidateInput {
                    log_basis_inner: candidate_log_basis_inner,
                    log_basis_open: candidate_log_basis_open,
                    position_index_bits,
                    block_index_bits,
                    outer_slice_count,
                    precommitted_groups: &candidate_precommitted_groups,
                    precommitted_d_width: candidate_precommitted_d_width,
                },
            )?
            else {
                continue;
            };
            candidate_params.witness_chunk = crate::policy::witness_chunk_at_level(policy, 0);
            if !candidate_params.compression_sources_supported()? {
                continue;
            }
            slice_candidates.push(candidate_params);
        }
        for candidate_params in crate::schedule_params::prune_locally_unprofitable_slices(
            policy,
            &opening_batch,
            slice_candidates,
        )? {
            let Some(output_witness_len) = root_batch_next_w_len(
                field_bits,
                policy.claim_ext_degree,
                &candidate_params,
                &opening_batch,
            )?
            else {
                continue;
            };
            if require_witness_contraction
                && output_witness_len
                    .checked_mul(candidate_log_basis_open as usize)
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup(
                            "root batch next witness bit length overflow".into(),
                        )
                    })?
                    >= initial_witness_len_bits
            {
                continue;
            }
            candidates.push((candidate_params, output_witness_len));
        }
    }

    Ok(candidates)
}

fn root_final_group_level_params_candidate(
    ctx: &MultiGroupRootCandidateCtx<'_>,
    input: RootFinalGroupCandidateInput<'_>,
) -> Result<Option<CommittedGroupParams>, AkitaError> {
    let RootFinalGroupCandidateInput {
        log_basis_inner,
        log_basis_open,
        position_index_bits,
        block_index_bits,
        outer_slice_count,
        precommitted_groups,
        precommitted_d_width,
    } = input;
    let policy = ctx.policy;
    let dimensions = ctx.dimensions;
    let d_a = dimensions.d_a();
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
    let witness_norms = ctx
        .final_honest_fold_policy
        .witness_norms_for_inner_basis(log_basis_inner, d_a)?;
    let Some(ab_candidate) = derive_ab_commitment_candidate(AbCommitmentCandidateRequest {
        policy,
        fold_policy: &ctx.final_honest_fold_policy,
        ring_challenge_cfg: &ctx.opening.challenge_config(),
        challenge_dimension: ctx.opening.challenge_dimension(d_a),
        dimensions,
        payload_mode: akita_types::CommitmentPayloadMode::Compressed,
        num_claims: ctx.main_num_polys,
        num_live_ring_elements_per_claim,
        num_live_blocks,
        num_positions_per_block,
        num_chunks,
        outer_slice_count,
        witness_norms,
        log_basis_open,
        width_s,
        num_digits_outer,
        modeled_linf_cap: None,
    })?
    else {
        return Ok(None);
    };
    let num_digits_fold = ab_candidate.num_digits_fold;
    let inner_commit_matrix = ab_candidate.inner_commit_matrix;
    let outer_commit_matrix = ab_candidate.outer_commit_matrix;

    let Ok(main_d_width) = akita_types::opening_d_segment_width(
        ctx.opening.method(),
        policy.claim_ext_degree,
        d_a,
        dimensions.d_d(),
        num_digits_open,
        num_live_blocks,
        ctx.main_num_polys,
    ) else {
        return Ok(None);
    };
    let Some((open_key, main_d_width)) = projected_collision_role_price(
        policy,
        akita_types::SisMatrixRole::Open,
        dimensions.d_d(),
        dimensions.d_d(),
        main_d_width,
        log_basis_open,
    ) else {
        return Ok(None);
    };
    // Every group width is already expressed in shared D-native subcolumns.
    // The collision-role pricing above only selects the D key and bound.
    let d_width = main_d_width
        .checked_add(precommitted_d_width)
        .ok_or_else(|| AkitaError::InvalidSetup("root batch D width overflow".to_string()))?;
    let Ok(open_commit_matrix) = OpenCommitMatrixParams::try_new_with_min_rank(open_key, d_width)
    else {
        return Ok(None);
    };

    let params = CommittedGroupParams {
        payload_mode: akita_types::CommitmentPayloadMode::Compressed,
        source_encoding: akita_types::CommittedSourceEncoding::for_producer(
            ctx.opening.method(),
            policy.claim_ext_degree,
            d_a,
            ctx.final_num_vars,
            true,
        ),
        opening_method: ctx.opening.method(),
        log_basis_inner,
        log_basis_outer: log_basis_open,
        log_basis_open,
        inner_commit_matrix,
        outer_commit_matrix,
        open_commit_matrix,
        num_live_ring_elements_per_claim,
        num_positions_per_block,
        num_live_blocks,
        outer_slice_count,
        fold_challenge_config: ctx.opening.challenge_config(),
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

/// Build the fold schedule selected by a full schedule lookup key.
pub fn find_schedule(
    key: &AkitaScheduleLookupKey,
    final_honest_fold_policy: HonestFoldPolicySpec,
    precommitted_honest_fold_policies: &[HonestFoldPolicySpec],
    policy: &PlannerPolicy,
    ring_challenge_config: impl Fn(usize) -> Result<akita_challenges::SparseChallengeConfig, AkitaError>,
) -> Result<PlannedFoldSchedule, AkitaError> {
    let diagnostics = crate::diagnostics::active();
    let diagnostics = diagnostics.as_deref();
    akita_schedules::planner_support::validate_policy(policy)?;
    key.validate(policy.decomposition.field_bits())?;
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
    let setup_field_budget = if active_policy.selection_policy
        == crate::SelectionPolicyId::MinFirstDirectSetupThenPayload
    {
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
    let suffix_ctx = SuffixCtx {
        policy: active_policy,
        diagnostics,
        ring_challenge_config,
        key: PolynomialGroupLayout::singleton(key.final_group.num_vars()),
        setup_field_budget,
        root_lookup_key: Some(key),
        root_honest_fold_policy: Some(final_honest_fold_policy),
        precommitted_honest_fold_policies,
        level_zero_is_root: true,
    };
    let mut memo = ScheduleMemo::new();
    let dimension_ceiling = super::schedule_params::initial_dimension_ceiling(active_policy)?;
    let suffix_started = diagnostics.map(|_| Instant::now());
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
    );
    if let (Some(diagnostics), Some(started)) = (diagnostics, suffix_started) {
        diagnostics.add_suffix_dp_time(started.elapsed());
        let (hits, misses) = memo.setup_prefix_cache_diagnostics();
        diagnostics.record_setup_prefix_cache(hits, misses);
    }
    let suffix = suffix?;
    let best = match active_policy.selection_policy {
        crate::SelectionPolicyId::MinEstimatedProofPayload => {
            select_complete_candidate(active_policy, suffix.payload_candidates(), diagnostics)?
        }
        crate::SelectionPolicyId::MinFirstDirectSetupThenPayload => {
            select_complete_candidate(active_policy, suffix.setup_candidates(), diagnostics)?
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
    let first_direct_setup_field_len = if active_policy.selection_policy
        == crate::SelectionPolicyId::MinFirstDirectSetupThenPayload
    {
        Some(
            best.first_direct_setup_field_len
                .ok_or_else(|| {
                    AkitaError::InvalidSetup(
                        "setup-first schedule is missing its first direct setup size".into(),
                    )
                })?
                .get(),
        )
    } else {
        None
    };
    if let Some(diagnostics) = diagnostics {
        let metrics = best.metrics();
        diagnostics.record_selected(
            active_policy.selection_policy,
            metrics.proof_bytes,
            metrics.setup_field_elements,
            metrics.first_direct_setup_capacity.field_elements(),
            best.folds
                .to_vec()
                .iter()
                .map(|fold| fold.params.role_dims())
                .collect(),
        );
    }
    let materialization_started = diagnostics.map(|_| Instant::now());
    let root_layout = key.opening_layout()?;
    let planned = materialize_candidate_schedule(
        best.total_bytes,
        best.setup_field_elements,
        first_direct_setup_field_len,
        active_policy.selection_policy,
        &root_layout,
        best.folds.to_vec(),
        best.terminal.as_ref().clone(),
    );
    if let (Some(diagnostics), Some(started)) = (diagnostics, materialization_started) {
        diagnostics.add_final_materialization_time(started.elapsed());
    }
    planned
}
