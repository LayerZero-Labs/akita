//! Multi-group root-batch schedule planning.

use akita_challenges::{SparseChallengeConfig, TensorChallengeShape};
use akita_field::AkitaError;
use akita_types::sis::{
    compute_num_digits_full_field, decomposed_s_block_ring_count, decomposed_t_ring_count,
    decomposed_w_ring_count, fold_witness_digit_plan, num_digits_inner, num_digits_open,
    rounded_up_collision_inf_norm, rounded_up_role_a_inf_norm, AjtaiKeyParams, FoldChallengeNorms,
    FoldWitnessLinfCapConfig, FoldWitnessNorms, SisTableKey,
};
use akita_types::{
    active_setup_field_len, extension_opening_reduction_level_bytes, level_proof_bytes,
    padded_setup_prefix_len, AkitaScheduleInputs, AkitaScheduleLookupKey, CommitmentRingDims,
    DecompositionParams, FoldStep, LevelParams, OpeningClaimsLayout, PolynomialGroupLayout,
    PrecommittedGroupParams, PrecommittedLevelParams, RelationMatrixRowLayout, Schedule,
    SetupContributionMode, WitnessLayout, SETUP_OFFLOAD_MIN_PREFIX_FIELD_LEN,
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

#[derive(Clone, Debug)]
struct PrecommittedGroupSeed {
    layout: PrecommittedGroupParams,
    a_key: AjtaiKeyParams,
    b_key: AjtaiKeyParams,
    num_digits_inner: usize,
    num_digits_outer: usize,
}

/// Validate frozen standalone precommit metadata and reconstruct the immutable
/// group-local A/B key facts. This deliberately does not choose or certify a
/// multi-group root opening basis: `log_basis_open` is selected later by the
/// root candidate search.
fn freeze_precommitted_group_layout(
    layout: &PrecommittedGroupParams,
    policy: &PlannerPolicy,
) -> Result<PrecommittedGroupSeed, AkitaError> {
    layout.validate_frozen_precommit(policy.ring_dimension)?;

    let d = policy.ring_dimension;
    let family = policy.sis_modulus_profile;
    let witness_decomp = DecompositionParams {
        log_basis: layout.log_basis_inner,
        ..policy.decomposition
    };
    let outer_decomp = DecompositionParams {
        log_basis: layout.log_basis_outer,
        ..policy.decomposition
    };
    let num_digits_inner = num_digits_inner(witness_decomp, true);
    let num_digits_outer = num_digits_open(outer_decomp);
    let num_live_blocks = layout.num_live_blocks;
    let num_positions_per_block = layout.num_positions_per_block;
    let width_s = decomposed_s_block_ring_count(num_positions_per_block, num_digits_inner)
        .ok_or_else(|| AkitaError::InvalidSetup("multi-group A width overflow".to_string()))?;
    let a_key = AjtaiKeyParams::try_new(
        policy.sis_security_policy,
        policy.sis_table_digest,
        family,
        akita_types::SisMatrixRole::A,
        layout.n_a,
        width_s,
        layout.a_coeff_linf_bound,
        d,
    )?;

    let norm_t = rounded_up_collision_inf_norm(
        policy.sis_security_policy,
        family,
        akita_types::SisMatrixRole::B,
        d,
        layout.log_basis_outer,
    )
    .ok_or_else(|| AkitaError::InvalidSetup("no multi-group B-role norm".to_string()))?;
    let width_t = decomposed_t_ring_count(
        layout.n_a,
        num_digits_outer,
        num_live_blocks,
        layout.group.num_polynomials(),
    )
    .ok_or_else(|| AkitaError::InvalidSetup("setup B width overflow".to_string()))?;
    if layout.b_coeff_linf_bound < norm_t {
        return Err(AkitaError::InvalidSetup(
            "precommitted group B bound is below the selected opening requirement".to_string(),
        ));
    }
    let b_key = AjtaiKeyParams::try_new(
        policy.sis_security_policy,
        policy.sis_table_digest,
        family,
        akita_types::SisMatrixRole::B,
        layout.n_b,
        width_t,
        layout.b_coeff_linf_bound,
        d,
    )?;

    Ok(PrecommittedGroupSeed {
        layout: *layout,
        a_key,
        b_key,
        num_digits_inner,
        num_digits_outer,
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
    let onehot_chunk_size = if policy.decomposition.log_commit_bound == 1 {
        policy.onehot_chunk_size
    } else {
        0
    };
    let challenge = FoldChallengeNorms {
        infinity_norm: group
            .layout
            .fold_challenge_shape
            .effective_infinity_norm(ring_challenge_cfg) as u128,
        l1_norm: group
            .layout
            .fold_challenge_shape
            .effective_l1_mass(ring_challenge_cfg) as u128,
    };
    let witness = FoldWitnessNorms::new(
        group.layout.log_basis_inner,
        policy.ring_dimension,
        if onehot_chunk_size == 0 {
            1
        } else {
            onehot_chunk_size
        },
        onehot_chunk_size > 0,
    );
    let cap_config = FoldWitnessLinfCapConfig::for_fold_level(
        ring_challenge_cfg,
        group.layout.fold_challenge_shape,
        policy.ring_dimension,
        group.a_key.col_len(),
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
    let witness_decomposition = DecompositionParams {
        log_basis: group.layout.log_basis_inner,
        ..policy.decomposition
    };
    let required_a_bound = rounded_up_role_a_inf_norm(
        policy.sis_security_policy,
        policy.sis_modulus_profile,
        policy.ring_dimension,
        witness_decomposition,
        log_basis_open,
        ring_challenge_cfg,
        group.layout.fold_challenge_shape,
        true,
        policy.onehot_chunk_size,
        policy.ring_subfield_norm_bound,
        group.layout.num_live_blocks,
        group.layout.group.num_polynomials(),
        group.a_key.col_len() as u64,
    )
    .ok_or_else(|| AkitaError::InvalidSetup("no precommitted A-role norm".to_string()))?;
    if required_a_bound > group.a_key.coeff_linf_bound() {
        return Err(AkitaError::InvalidSetup(
            "precommitted A bound does not cover the certified opening basis".to_string(),
        ));
    }
    let required_b_bound = rounded_up_collision_inf_norm(
        policy.sis_security_policy,
        policy.sis_modulus_profile,
        akita_types::SisMatrixRole::B,
        policy.ring_dimension,
        log_basis_open,
    )
    .ok_or_else(|| AkitaError::InvalidSetup("no precommitted B-role norm".to_string()))?;
    if required_b_bound > group.b_key.coeff_linf_bound() {
        return Err(AkitaError::InvalidSetup(
            "precommitted B bound does not cover the certified opening basis".to_string(),
        ));
    }
    Ok(PrecommittedLevelParams {
        layout: group.layout,
        a_key: group.a_key.clone(),
        b_key: group.b_key.clone(),
        log_basis_open,
        num_digits_inner: group.num_digits_inner,
        num_digits_outer: group.num_digits_outer,
        num_digits_open,
        num_digits_fold_one,
    })
}

struct MultiGroupRootCandidateCtx<'a> {
    policy: &'a PlannerPolicy,
    ring_challenge_cfg: &'a SparseChallengeConfig,
    requested_fold_shape: TensorChallengeShape,
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
    ring_challenge_config: RingChallengeConfigFn<'_>,
    log_basis_open: u32,
) -> Result<(Vec<PrecommittedLevelParams>, usize), AkitaError> {
    let ring_challenge_cfg = ring_challenge_config(policy.ring_dimension)?;
    let commit_groups = multi_group_root_precommitted_group_seeds(key, policy)?;
    precommitted_groups_for_open_basis(&commit_groups, policy, &ring_challenge_cfg, log_basis_open)
}

fn precommitted_groups_for_open_basis(
    seeds: &[PrecommittedGroupSeed],
    policy: &PlannerPolicy,
    ring_challenge_cfg: &SparseChallengeConfig,
    log_basis_open: u32,
) -> Result<(Vec<PrecommittedLevelParams>, usize), AkitaError> {
    let groups = seeds
        .iter()
        .map(|group| {
            materialize_precommitted_group_for_open_basis(
                group,
                policy,
                ring_challenge_cfg,
                log_basis_open,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut d_width = 0usize;
    for group in &groups {
        d_width = d_width
            .checked_add(group.d_segment_width()?)
            .ok_or_else(|| AkitaError::InvalidSetup("multi-group D width overflow".to_string()))?;
    }
    Ok((groups, d_width))
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
        compute_num_digits_full_field(field_bits, params.log_basis_open),
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
    precommitted_groups: &[PrecommittedLevelParams],
    precommitted_d_width: usize,
) -> Result<Option<LevelParams>, AkitaError> {
    let policy = ctx.policy;
    let d = policy.ring_dimension;
    let family = policy.sis_modulus_profile;
    let decomp = policy.decomposition;
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
    let Some(norm_s) = rounded_up_role_a_inf_norm(
        policy.sis_security_policy,
        family,
        d,
        witness_decomp,
        log_basis,
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
        decomposed_t_ring_count(n_a, num_digits_outer, num_live_blocks, main_num_polys)
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
        .checked_add(precommitted_d_width)
        .ok_or_else(|| AkitaError::InvalidSetup("multi-group D width overflow".to_string()))?;
    let Some(norm_w) = rounded_up_collision_inf_norm(
        policy.sis_security_policy,
        family,
        akita_types::SisMatrixRole::D,
        d,
        log_basis,
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
        log_basis_inner,
        log_basis_outer: log_basis,
        log_basis_open: log_basis,
        a_key,
        b_key,
        d_key,
        num_live_ring_elements_per_claim,
        num_positions_per_block,
        num_live_blocks,
        fold_challenge_config: *ctx.ring_challenge_cfg,
        fold_challenge_shape,
        num_digits_inner,
        num_digits_outer,
        num_digits_open,
        onehot_chunk_size,
        fold_linf_cap_config: FoldWitnessLinfCapConfig::worst_case_beta_only(),
        num_digits_fold_one: 1,
        field_bits_hint: 0,
        cached_num_digits_block_claims: 0,
        cached_num_digits_fold_value: 1,
        // Multi-group root folds use the ordinary single-chunk precommit path.
        witness_chunk: akita_types::ChunkedWitnessCfg::default(),
        precommitted_groups: precommitted_groups.to_vec(),
        setup_prefix: None,
        role_dims: CommitmentRingDims::uniform(d),
        setup_contribution_mode: SetupContributionMode::Direct,
    }
    .with_fold_linf_cap_config(decomp.field_bits(), main_num_polys)?;

    params.stamp_role_dims_from_keys();
    Ok(Some(params))
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
    let mut best: Option<(usize, Vec<FoldStep>, akita_types::TerminalWitnessPlan)> = None;

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
        return Err(AkitaError::UnsupportedSchedule(format!(
            "multi-group num_vars={} does not exceed log2(ring_dimension)={alpha}",
            key.final_group.num_vars()
        )));
    }

    let precommitted_groups = multi_group_root_precommitted_group_seeds(key, policy)?;
    let ring_challenge_cfg = ring_challenge_config(policy.ring_dimension)?;
    let candidate_ctx = MultiGroupRootCandidateCtx {
        policy,
        ring_challenge_cfg: &ring_challenge_cfg,
        requested_fold_shape: fold_challenge_shape,
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
        let (candidate_precommitted_groups, candidate_precommitted_d_width) =
            precommitted_groups_for_open_basis(
                &precommitted_groups,
                policy,
                &ring_challenge_cfg,
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
                let child_is_terminal = suffix_fold.folds.len() == 1;
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
                    let child_lb = suffix_fold.first_fold_params.log_basis_open;
                    let Some(prefixed_suffix_fold) =
                        prefixed_child_suffix.best_fold_per_lb.get(&child_lb)
                    else {
                        continue;
                    };
                    if prefixed_suffix_fold.folds.len() == 1 {
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
                    RelationMatrixRowLayout::WithDBlock,
                    Some(akita_types::NextWitnessBindingPolicy::OuterCommitment),
                )? + eor_bytes;
                let total = root_proof_size + suffix_fold.total_bytes;
                if best
                    .as_ref()
                    .is_none_or(|(best_total, _, _)| total < *best_total)
                {
                    let mut folds = Vec::with_capacity(1 + suffix_fold.folds.len());
                    folds.push(FoldStep {
                        params: fold_candidate_params,
                        current_w_len: root_current_w_len,
                        next_w_len,
                        level_bytes: root_proof_size,
                    });
                    folds.extend(suffix_fold.folds.iter().cloned());
                    best = Some((total, folds, suffix_fold.terminal.clone()));
                }
            }
        }
    }

    let Some((total_bytes, folds, terminal)) = best else {
        return Err(AkitaError::UnsupportedSchedule(format!(
            "no multi-group schedule with at least two folds for num_vars={}",
            key.final_group.num_vars()
        )));
    };
    Ok(Schedule {
        folds,
        terminal,
        total_bytes,
    })
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
            claim_ext_degree: 1,
            chal_ext_degree: 1,
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
            log_basis_inner: 1,
            log_basis_outer: 3,
            n_a: 1,
            a_coeff_linf_bound: 1,
            n_b: 1,
            b_coeff_linf_bound: 1,
        }
    }

    fn frozen_precommitted_for_policy(
        policy: &PlannerPolicy,
        num_polys: usize,
        num_vars: usize,
        log_basis_inner: u32,
        log_basis_outer: u32,
        a_coeff_linf_bound: u128,
        b_coeff_linf_bound: u128,
    ) -> PrecommittedGroupParams {
        let alpha = policy.ring_dimension.trailing_zeros() as usize;
        let outer = num_vars - alpha;
        let block_index_bits = outer / 2;
        let position_index_bits = outer - block_index_bits;
        PrecommittedGroupParams {
            group: PolynomialGroupLayout::new(num_vars, num_polys),
            num_live_ring_elements_per_claim: 1usize << outer,
            num_positions_per_block: 1usize << position_index_bits,
            num_live_blocks: 1usize << block_index_bits,
            fold_challenge_shape: TensorChallengeShape::Flat,
            log_basis_inner,
            log_basis_outer,
            n_a: 10,
            a_coeff_linf_bound,
            n_b: 10,
            b_coeff_linf_bound,
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
        let params = schedule
            .folds
            .first()
            .expect("schedule root fold")
            .params
            .clone();
        PrecommittedGroupParams::from_params(key, &params)
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
        let root = schedule.folds.first().expect("multi-group root fold");

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
        let final_group = PolynomialGroupLayout::new(32, 1);
        let key = AkitaScheduleLookupKey::single(final_group);
        let policy = flat_policy();

        let via_multi_group =
            find_group_batch_schedule(&key, &policy, ring_challenge_config, fold_shape);
        let via_scalar = find_schedule(final_group, &policy, ring_challenge_config, fold_shape);

        match (via_multi_group, via_scalar) {
            (Ok(via_multi_group), Ok(via_scalar)) => {
                assert_eq!(via_multi_group.total_bytes, via_scalar.total_bytes);
                assert_eq!(via_multi_group.folds.len(), via_scalar.folds.len());
            }
            (Err(AkitaError::UnsupportedSchedule(_)), Err(AkitaError::UnsupportedSchedule(_))) => {}
            (via_multi_group, via_scalar) => panic!(
                "single-group dispatch diverged: grouped={via_multi_group:?}, scalar={via_scalar:?}"
            ),
        }
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
            log_basis_inner: 1,
            log_basis_outer: 3,
            n_a: 1,
            a_coeff_linf_bound: 1,
            n_b: 1,
            b_coeff_linf_bound: 1,
        };
        let error = freeze_precommitted_group_layout(&malformed, &policy)
            .expect_err("malformed non-tiny geometry must propagate");

        assert!(error.to_string().contains("geometry does not match"));
    }

    #[test]
    fn recursive_policy_empty_precommit_dense_still_uses_scalar_planner() {
        let mut policy = flat_policy();
        policy.decomposition.log_commit_bound = 8;
        policy.recursive_setup_planning = true;
        let key = AkitaScheduleLookupKey::single(PolynomialGroupLayout::new(32, 1));

        let grouped = find_group_batch_schedule(&key, &policy, ring_challenge_config, fold_shape);
        let scalar = find_schedule(
            key.final_group,
            &PlannerPolicy {
                recursive_setup_planning: false,
                ..policy
            },
            ring_challenge_config,
            fold_shape,
        );
        match (grouped, scalar) {
            (Ok(grouped), Ok(scalar)) => {
                assert_eq!(grouped.total_bytes, scalar.total_bytes);
                for fold in grouped.fold_steps() {
                    assert_eq!(
                        fold.params.setup_contribution_mode,
                        SetupContributionMode::Direct
                    );
                }
            }
            (Err(AkitaError::UnsupportedSchedule(_)), Err(AkitaError::UnsupportedSchedule(_))) => {}
            (grouped, scalar) => {
                panic!("empty-precommit dispatch diverged: grouped={grouped:?}, scalar={scalar:?}")
            }
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
        let root = schedule.folds.first().expect("multi-group root fold");

        assert_eq!(root.params.log_basis_open, 4);
    }

    #[test]
    fn mixed_basis_root_rejects_open_basis_above_frozen_without_recertified_bounds() {
        let mut precommit_policy = flat_policy();
        precommit_policy.ring_dimension = 128;
        precommit_policy.decomposition.log_basis = 2;
        precommit_policy.basis_range = (2, 2);
        let mut root_policy = precommit_policy;
        root_policy.decomposition.log_open_bound = Some(128);
        root_policy.decomposition.log_basis = 3;
        root_policy.basis_range = (3, 3);
        let pre_key = PolynomialGroupLayout::new(20, 1);
        let frozen = frozen_precommitted_for_policy(
            &root_policy,
            1,
            pre_key.num_vars(),
            2,
            2,
            1,
            rounded_up_collision_inf_norm(
                root_policy.sis_security_policy,
                root_policy.sis_modulus_profile,
                akita_types::SisMatrixRole::B,
                root_policy.ring_dimension,
                3,
            )
            .expect("B norm"),
        );
        let key = AkitaScheduleLookupKey {
            final_group: PolynomialGroupLayout::new(40, 2),
            precommitteds: vec![frozen],
        };
        let err = find_group_batch_schedule(&key, &root_policy, ring_challenge_config, fold_shape)
            .expect_err("higher root opening basis must not reuse lower certified bounds");
        assert!(err
            .to_string()
            .contains("precommitted A bound does not cover the certified opening basis"));
    }

    #[test]
    fn mixed_basis_root_rejects_open_basis_below_frozen_precommit() {
        let mut precommit_policy = flat_policy();
        precommit_policy.ring_dimension = 128;
        precommit_policy.decomposition.log_basis = 3;
        precommit_policy.basis_range = (3, 3);
        let mut root_policy = precommit_policy;
        root_policy.decomposition.log_open_bound = Some(128);
        root_policy.decomposition.log_basis = 2;
        root_policy.basis_range = (2, 2);
        let pre_key = PolynomialGroupLayout::new(20, 1);
        let frozen = frozen_precommitted_for_policy(
            &root_policy,
            1,
            pre_key.num_vars(),
            3,
            3,
            1,
            rounded_up_collision_inf_norm(
                root_policy.sis_security_policy,
                root_policy.sis_modulus_profile,
                akita_types::SisMatrixRole::B,
                root_policy.ring_dimension,
                3,
            )
            .expect("B norm"),
        );
        let key = AkitaScheduleLookupKey {
            final_group: PolynomialGroupLayout::new(40, 2),
            precommitteds: vec![frozen],
        };
        let err = find_group_batch_schedule(&key, &root_policy, ring_challenge_config, fold_shape)
            .expect_err("opening basis below frozen precommit must be rejected");
        assert!(err
            .to_string()
            .contains("certified opening basis must dominate precommitted inner/outer bases"));
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
        let root = schedule.folds.first().expect("multi-group root fold");

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
        let folds = &schedule.folds;

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
        policy.decomposition.log_basis = 4;
        policy.decomposition.log_open_bound = Some(128);
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
        policy.decomposition.log_basis = 4;
        policy.basis_range = (4, 4);
        policy.recursive_setup_planning = true;
        policy.decomposition.log_open_bound = Some(128);
        let pre_key = PolynomialGroupLayout::new(16, 1);
        let key = AkitaScheduleLookupKey {
            final_group: PolynomialGroupLayout::new(32, 1),
            precommitteds: vec![precommitted_from_policy(pre_key, &policy)],
        };

        let schedule = find_group_batch_schedule(&key, &policy, ring_challenge_config, fold_shape)
            .expect("schedule with terminal child");
        let folds = &schedule.folds;
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
        let predecessor = &folds[folds.len() - 2];
        // The fold immediately before the terminal fold must remain Direct-mode.
        if folds.len() == 2 {
            assert_eq!(
                predecessor.params.setup_contribution_mode,
                SetupContributionMode::Direct
            );
        }
    }

    #[test]
    fn recursive_fold_successor_carries_only_setup_prefix_group() {
        let mut policy = flat_policy();
        policy.decomposition.log_basis = 4;
        policy.decomposition.log_open_bound = Some(128);
        policy.basis_range = (4, 4);
        policy.recursive_setup_planning = true;
        let pre_key = PolynomialGroupLayout::new(20, 1);
        let mut frozen = precommitted_from_policy(pre_key, &policy);
        frozen.n_a = 10;
        frozen.n_b = 10;
        let key = AkitaScheduleLookupKey {
            final_group: PolynomialGroupLayout::new(40, 2),
            precommitteds: vec![frozen],
        };

        let schedule = find_group_batch_schedule(&key, &policy, ring_challenge_config, fold_shape)
            .expect("recursive multi-group schedule");
        for (index, fold) in schedule.folds.iter().enumerate() {
            if fold.params.setup_contribution_mode != SetupContributionMode::Recursive {
                continue;
            }
            let successor = schedule
                .folds
                .get(index + 1)
                .expect("recursive fold must have a successor");
            assert!(successor.params.setup_prefix.is_some());
            assert!(successor.params.precommitted_groups.is_empty());
            assert_eq!(successor.params.precommitted_group_count(), 1);
        }
    }
}
