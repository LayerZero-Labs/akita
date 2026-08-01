//! Multi-group root-batch schedule planning.

use akita_challenges::{SparseChallengeConfig, TensorChallengeShape};
use akita_field::AkitaError;
use akita_types::sis::{
    compute_num_digits_field_width, decomposed_s_block_ring_count, decomposed_t_ring_count,
    decomposed_w_ring_count, num_digits_inner, num_digits_open, rounded_up_collision_inf_norm,
    rounded_up_role_a_inf_norm, HonestFoldPolicy, HonestFoldPolicySpec, HonestFoldSizingQuery,
    InnerCommitMatrixParams, OpenCommitMatrixParams, OuterCommitMatrixParams, SisTableKey,
};
use akita_types::{
    AkitaScheduleInputs, AkitaScheduleLookupKey, CommittedGroupParams, CommittedGroupProfile,
    DecompositionParams, OpeningClaimsLayout, PlannedFoldSchedule, PolynomialGroupLayout,
    PrecommittedLevelParams, WitnessLayout,
};

use crate::schedule_params::{
    derive_optimal_suffix_schedule, find_schedule, materialize_candidate_schedule,
    optimize_fold_challenge_shape, validate_policy, RingChallengeConfigFn, ScheduleMemo, SuffixCtx,
    SuffixState,
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
        ring_dimension: policy.uniform_ring_dimension as u32,
        coeff_linf_bound,
    }
}

#[derive(Clone, Debug)]
struct PrecommittedGroupSeed {
    layout: CommittedGroupProfile,
    honest_fold_policy: HonestFoldPolicySpec,
    inner_commit_matrix: InnerCommitMatrixParams,
    outer_commit_matrix: OuterCommitMatrixParams,
}

/// Validate frozen standalone precommit metadata and reconstruct the immutable
/// group-local A/B key facts. This deliberately does not choose or certify a
/// multi-group root opening basis: `log_basis_open` is selected later by the
/// root candidate search.
fn freeze_precommitted_group_layout(
    layout: &CommittedGroupProfile,
    honest_fold_policy: HonestFoldPolicySpec,
    policy: &PlannerPolicy,
) -> Result<PrecommittedGroupSeed, AkitaError> {
    layout.validate_frozen_precommit(policy.decomposition.field_bits())?;

    let d_a = layout.inner_commit_matrix.ring_dimension();
    let d_b = layout.outer_commit_matrix.ring_dimension();
    let outer_decomp = DecompositionParams {
        log_basis: layout.log_basis_outer,
        ..policy.decomposition
    };
    let num_digits_outer = num_digits_open(outer_decomp);
    let num_live_blocks = layout.num_live_blocks;
    let num_positions_per_block = layout.num_positions_per_block;
    let width_s =
        decomposed_s_block_ring_count(num_positions_per_block, layout.num_digits_inner)
            .ok_or_else(|| AkitaError::InvalidSetup("multi-group A width overflow".to_string()))?;
    if num_digits_outer != layout.num_digits_outer
        || width_s != layout.inner_commit_matrix.input_width()
    {
        return Err(AkitaError::InvalidSetup(
            "precommitted profile digit depths do not match its frozen matrices".to_string(),
        ));
    }
    let inner_commit_matrix = layout.inner_commit_matrix;

    let norm_t = rounded_up_collision_inf_norm(
        policy.sis_security_policy,
        policy.sis_modulus_profile,
        akita_types::SisMatrixRole::Outer,
        d_b,
        layout.log_basis_outer,
    )
    .ok_or_else(|| AkitaError::InvalidSetup("no multi-group B-role norm".to_string()))?;
    let width_t = decomposed_t_ring_count(
        layout.inner_commit_matrix.output_rank(),
        num_digits_outer,
        num_live_blocks,
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
        honest_fold_policy,
        inner_commit_matrix,
        outer_commit_matrix,
    })
}

/// Materialize a frozen precommitted group for a candidate multi-group root
/// `log_basis_open`. This is the phase that assigns the opening basis, recomputes
/// open/fold digit depths from that basis, and checks the frozen A/B bounds still
/// cover the chosen response-basis envelopes.
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
    let challenge_shape = TensorChallengeShape::Flat;
    let ring_dimension = group.layout.inner_commit_matrix.ring_dimension();
    let num_fold_coeffs = group
        .inner_commit_matrix
        .input_width()
        .checked_mul(ring_dimension)
        .ok_or_else(|| AkitaError::InvalidSetup("precommitted fold width overflow".into()))?;
    let group_claims = group.layout.group.num_polynomials();
    let num_digits_fold = group
        .honest_fold_policy
        .num_digits_fold(HonestFoldSizingQuery {
            ring_dimension,
            num_claims: group_claims,
            num_live_blocks: group.layout.num_live_blocks,
            num_fold_coeffs,
            log_basis: log_basis_open,
            challenge_config: ring_challenge_cfg,
            challenge_shape,
        })?;
    let required_a_bound = rounded_up_role_a_inf_norm(
        policy.sis_security_policy,
        policy.sis_table_digest,
        policy.sis_modulus_profile,
        group.layout.inner_commit_matrix.ring_dimension(),
        log_basis_open,
        ring_challenge_cfg,
        challenge_shape,
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
        akita_types::SisMatrixRole::Outer,
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

struct MultiGroupRootCandidateCtx<'a> {
    policy: &'a PlannerPolicy,
    ring_challenge_cfg: &'a SparseChallengeConfig,
    requested_fold_shape: TensorChallengeShape,
    final_honest_fold_policy: HonestFoldPolicySpec,
}

fn multi_group_root_precommitted_group_seeds(
    key: &AkitaScheduleLookupKey,
    precommitted_honest_fold_policies: &[HonestFoldPolicySpec],
    policy: &PlannerPolicy,
) -> Result<Vec<PrecommittedGroupSeed>, AkitaError> {
    if key.precommitteds.is_empty() {
        return Err(AkitaError::InvalidSetup(
            "multi-group root params require at least one precommitted group".to_string(),
        ));
    }
    if precommitted_honest_fold_policies.len() != key.precommitteds.len() {
        return Err(AkitaError::InvalidSetup(
            "group-batch planning requires one honest fold policy per precommitted profile"
                .to_string(),
        ));
    }

    key.precommitteds
        .iter()
        .zip(precommitted_honest_fold_policies)
        .map(|(layout, honest_policy)| {
            freeze_precommitted_group_layout(layout, *honest_policy, policy)
        })
        .collect::<Result<Vec<_>, _>>()
}

fn precommitted_groups_for_open_basis(
    seeds: &[PrecommittedGroupSeed],
    policy: &PlannerPolicy,
    ring_challenge_config: &dyn Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    log_basis_open: u32,
) -> Result<(Vec<PrecommittedLevelParams>, usize), AkitaError> {
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
            .checked_add(group.d_segment_width(policy.uniform_ring_dimension)?)
            .ok_or_else(|| AkitaError::InvalidSetup("multi-group D width overflow".to_string()))?;
    }
    Ok((groups, d_width))
}

pub(crate) fn multi_group_root_next_w_len(
    field_bits: u32,
    params: &CommittedGroupParams,
    opening_batch: &OpeningClaimsLayout,
) -> Result<usize, AkitaError> {
    params.witness_chunk.validate()?;
    params.validate_opening_batch(opening_batch)?;
    let witness_layout = WitnessLayout::new(
        params,
        opening_batch,
        params.witness_chunk.num_chunks,
        compute_num_digits_field_width(field_bits, params.log_basis_open),
    )?;
    Ok(witness_layout.live_coeff_len())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn multi_group_root_level_candidates_for_basis(
    key: &AkitaScheduleLookupKey,
    final_honest_fold_policy: HonestFoldPolicySpec,
    precommitted_honest_fold_policies: &[HonestFoldPolicySpec],
    policy: &PlannerPolicy,
    ring_challenge_cfg: &SparseChallengeConfig,
    ring_challenge_config: &dyn Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    requested_fold_shape: TensorChallengeShape,
    root_input_witness_len: usize,
    candidate_log_basis: u32,
) -> Result<Vec<(CommittedGroupParams, usize)>, AkitaError> {
    let field_bits = policy.decomposition.field_bits();
    let alpha = (policy.uniform_ring_dimension as u32).trailing_zeros() as usize;
    let reduced_vars = key.final_group.num_vars().saturating_sub(alpha);
    if reduced_vars == 0 {
        return Err(AkitaError::UnsupportedSchedule(format!(
            "multi-group num_vars={} does not exceed log2(ring_dimension)={alpha}",
            key.final_group.num_vars()
        )));
    }

    let precommitted_groups =
        multi_group_root_precommitted_group_seeds(key, precommitted_honest_fold_policies, policy)?;
    let candidate_ctx = MultiGroupRootCandidateCtx {
        policy,
        ring_challenge_cfg,
        requested_fold_shape,
        final_honest_fold_policy,
    };
    let opening_batch = key.opening_layout()?;
    let initial_witness_len_bits = root_input_witness_len
        .checked_mul(field_bits as usize)
        .ok_or_else(|| {
            AkitaError::InvalidSetup("multi-group root witness bit length overflow".into())
        })?;
    let min_block_index_bits: usize = if reduced_vars >= 3 { 1 } else { 0 };
    let max_block_index_bits: usize = (reduced_vars - 1).min(usize::BITS as usize - 1);

    let mut candidates = Vec::new();
    let (candidate_precommitted_groups, candidate_precommitted_d_width) =
        precommitted_groups_for_open_basis(
            &precommitted_groups,
            policy,
            ring_challenge_config,
            candidate_log_basis,
        )?;
    for block_index_bits in (min_block_index_bits..=max_block_index_bits).rev() {
        let position_index_bits = reduced_vars - block_index_bits;
        let Some(mut candidate_params) = multi_group_root_main_level_params_candidate(
            &candidate_ctx,
            key.final_group.num_polynomials(),
            candidate_log_basis,
            position_index_bits,
            block_index_bits,
            &candidate_precommitted_groups,
            candidate_precommitted_d_width,
        )?
        else {
            continue;
        };
        let root_num_chunks = policy.chunks_at_level(0);
        // A chunked root fold distributes both the main folded witness and
        // every precommitted group's folded response across `num_chunks`
        // block windows, so each needs at least one live block per chunk.
        if candidate_params.num_live_blocks < root_num_chunks
            || candidate_params
                .precommitted_groups
                .iter()
                .any(|group| group.layout.num_live_blocks < root_num_chunks)
        {
            continue;
        }
        candidate_params.witness_chunk = policy.witness_chunk_for_level(0);
        let output_witness_len =
            multi_group_root_next_w_len(field_bits, &candidate_params, &opening_batch)?;
        if output_witness_len
            .checked_mul(candidate_log_basis as usize)
            .ok_or_else(|| {
                AkitaError::InvalidSetup("multi-group root next witness bit length overflow".into())
            })?
            >= initial_witness_len_bits
        {
            continue;
        }
        candidates.push((candidate_params, output_witness_len));
    }

    Ok(candidates)
}

fn multi_group_root_main_level_params_candidate(
    ctx: &MultiGroupRootCandidateCtx<'_>,
    main_num_polys: usize,
    log_basis: u32,
    position_index_bits: usize,
    block_index_bits: usize,
    precommitted_groups: &[PrecommittedLevelParams],
    precommitted_d_width: usize,
) -> Result<Option<CommittedGroupParams>, AkitaError> {
    let policy = ctx.policy;
    let d = policy.uniform_ring_dimension;
    let family = policy.sis_modulus_profile;
    let decomp = ctx.policy.decomposition;
    let level_decomp = DecompositionParams {
        log_basis,
        ..decomp
    };
    let log_basis_inner = log_basis;
    let witness_decomp = DecompositionParams {
        log_basis: log_basis_inner,
        ..decomp
    };
    let num_digits_inner = num_digits_inner(witness_decomp, true);
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
    let fold_challenge_shape =
        optimize_fold_challenge_shape(ctx.requested_fold_shape, num_live_blocks)?;

    let Some(width_s) = decomposed_s_block_ring_count(num_positions_per_block, num_digits_inner)
    else {
        return Ok(None);
    };
    let Some(num_fold_coeffs) = width_s.checked_mul(d) else {
        return Ok(None);
    };
    let Ok(num_digits_fold) = ctx
        .final_honest_fold_policy
        .num_digits_fold(HonestFoldSizingQuery {
            ring_dimension: d,
            num_claims: main_num_polys,
            num_live_blocks,
            num_fold_coeffs,
            log_basis,
            challenge_config: ctx.ring_challenge_cfg,
            challenge_shape: fold_challenge_shape,
        })
    else {
        return Ok(None);
    };
    let Some(norm_s) = rounded_up_role_a_inf_norm(
        policy.sis_security_policy,
        policy.sis_table_digest,
        family,
        d,
        log_basis,
        ctx.ring_challenge_cfg,
        fold_challenge_shape,
        num_digits_fold,
        policy.ring_subfield_norm_bound,
    ) else {
        return Ok(None);
    };
    let Ok(inner_commit_matrix) = InnerCommitMatrixParams::try_new_with_min_rank(
        sis_key(policy, akita_types::SisMatrixRole::Inner, norm_s),
        width_s,
    ) else {
        return Ok(None);
    };
    let n_a = inner_commit_matrix.output_rank();

    let Some(norm_t) = rounded_up_collision_inf_norm(
        policy.sis_security_policy,
        family,
        akita_types::SisMatrixRole::Outer,
        d,
        log_basis,
    ) else {
        return Ok(None);
    };
    let Some(width_t) =
        decomposed_t_ring_count(n_a, num_digits_outer, num_live_blocks, main_num_polys)
    else {
        return Ok(None);
    };
    let Ok(outer_commit_matrix) = OuterCommitMatrixParams::try_new_with_min_rank(
        sis_key(policy, akita_types::SisMatrixRole::Outer, norm_t),
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
        .checked_add(precommitted_d_width)
        .ok_or_else(|| AkitaError::InvalidSetup("multi-group D width overflow".to_string()))?;
    let Some(norm_w) = rounded_up_collision_inf_norm(
        policy.sis_security_policy,
        family,
        akita_types::SisMatrixRole::Open,
        d,
        log_basis,
    ) else {
        return Ok(None);
    };
    let Ok(open_commit_matrix) = OpenCommitMatrixParams::try_new_with_min_rank(
        sis_key(policy, akita_types::SisMatrixRole::Open, norm_w),
        d_width,
    ) else {
        return Ok(None);
    };

    let params = CommittedGroupParams {
        log_basis_inner,
        log_basis_outer: log_basis,
        log_basis_open: log_basis,
        inner_commit_matrix,
        outer_commit_matrix,
        open_commit_matrix,
        num_live_ring_elements_per_claim,
        num_positions_per_block,
        num_live_blocks,
        fold_challenge_config: *ctx.ring_challenge_cfg,
        fold_challenge_shape,
        num_digits_inner,
        num_digits_outer,
        num_digits_open,
        num_digits_fold,
        // Multi-group root folds use the ordinary single-chunk precommit path.
        witness_chunk: akita_types::ChunkedWitnessCfg::default(),
        precommitted_groups: precommitted_groups.to_vec(),
        setup_prefix: None,
    };

    Ok(Some(params))
}

/// Build the phase-1 multi-group-root schedule from the full multi-group key.
pub fn find_group_batch_schedule(
    key: &AkitaScheduleLookupKey,
    final_honest_fold_policy: HonestFoldPolicySpec,
    precommitted_honest_fold_policies: &[HonestFoldPolicySpec],
    policy: &PlannerPolicy,
    ring_challenge_config: impl Fn(usize) -> Result<akita_challenges::SparseChallengeConfig, AkitaError>,
    fold_challenge_shape_at_level: impl Fn(AkitaScheduleInputs) -> TensorChallengeShape,
) -> Result<PlannedFoldSchedule, AkitaError> {
    validate_policy(policy)?;
    let ring_challenge_config: RingChallengeConfigFn<'_> = &ring_challenge_config;
    let fold_challenge_shape_at_level = &fold_challenge_shape_at_level;
    if policy.recursive_setup_planning && !key.precommitteds.is_empty() {
        return find_group_batch_schedule_inner(
            key,
            final_honest_fold_policy,
            precommitted_honest_fold_policies,
            policy,
            ring_challenge_config,
            fold_challenge_shape_at_level,
            policy.setup_field_budget,
        );
    }
    find_group_batch_schedule_inner(
        key,
        final_honest_fold_policy,
        precommitted_honest_fold_policies,
        policy,
        ring_challenge_config,
        fold_challenge_shape_at_level,
        None,
    )
}

fn find_group_batch_schedule_inner(
    key: &AkitaScheduleLookupKey,
    final_honest_fold_policy: HonestFoldPolicySpec,
    precommitted_honest_fold_policies: &[HonestFoldPolicySpec],
    policy: &PlannerPolicy,
    ring_challenge_config: RingChallengeConfigFn<'_>,
    fold_challenge_shape_at_level: &dyn Fn(AkitaScheduleInputs) -> TensorChallengeShape,
    setup_field_budget: Option<usize>,
) -> Result<PlannedFoldSchedule, AkitaError> {
    key.validate(policy.decomposition.field_bits())?;
    if key.precommitteds.is_empty() {
        // Genuine multi-group roots only. Empty-precommit keys are scalar and
        // must not enter recursion-enabled grouped planning.
        let scalar_policy = policy.direct_only();
        let dimensions = crate::schedule_params::RingDimensionSearchDomain::new(
            scalar_policy.ring_dimension_candidates.iter().copied(),
        )?;
        return find_schedule(
            key.final_group,
            &scalar_policy,
            final_honest_fold_policy,
            &dimensions,
            ring_challenge_config,
            fold_challenge_shape_at_level,
        );
    }
    let root_input_witness_len = 1usize
        .checked_shl(key.final_group.num_vars() as u32)
        .ok_or_else(|| {
            AkitaError::InvalidSetup("multi-group root-fold witness length overflow".to_string())
        })?;
    let ring_challenge_cfg = ring_challenge_config(policy.uniform_ring_dimension)?;
    let suffix_ctx = SuffixCtx {
        policy,
        default_ring_challenge_cfg: &ring_challenge_cfg,
        ring_challenge_config,
        fold_challenge_shape_at_level,
        num_vars: key.final_group.num_vars(),
        key: PolynomialGroupLayout::singleton(key.final_group.num_vars()),
        setup_field_budget,
        root_lookup_key: Some(key),
        root_honest_fold_policy: Some(final_honest_fold_policy),
        precommitted_honest_fold_policies,
    };
    let mut memo = ScheduleMemo::new();
    let suffix = derive_optimal_suffix_schedule(
        &suffix_ctx,
        &mut memo,
        SuffixState {
            level: 0,
            current_witness_len: root_input_witness_len,
            current_lb: 0,
            incoming_setup_prefix: None,
        },
        0,
    )?;
    let best = match policy.selection_policy {
        crate::SelectionPolicyId::MinEstimatedProofPayload => suffix
            .best_by_payload_per_lb
            .values()
            .min_by_key(|candidate| candidate.total_bytes),
        crate::SelectionPolicyId::MinSetupMatrixFieldElementsThenProofPayload => {
            return Err(AkitaError::UnsupportedSchedule(
                "mixed ring-dimension selection is not supported for multi-group roots".to_string(),
            ));
        }
        crate::SelectionPolicyId::MinFirstDirectSetupThenPayload => suffix
            .best_by_first_direct_setup_per_lb
            .values()
            .min_by_key(|candidate| {
                (
                    candidate.first_direct_setup_field_len_or_max(),
                    candidate.total_bytes,
                )
            }),
    };

    let Some(best) = best.cloned() else {
        return Err(AkitaError::UnsupportedSchedule(format!(
            "no multi-group schedule with at least two folds for num_vars={}",
            key.final_group.num_vars()
        )));
    };
    let first_direct_setup_field_len = if policy.recursive_setup_planning {
        Some(best.first_direct_setup_field_len.ok_or_else(|| {
            AkitaError::InvalidSetup(
                "recursive setup schedule is missing its first direct setup size".into(),
            )
        })?)
    } else {
        None
    };
    materialize_candidate_schedule(
        best.total_bytes,
        best.setup_field_elements,
        first_direct_setup_field_len,
        best.folds,
        best.terminal,
    )
}
