use super::*;

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
/// Build one recursive-fold candidate for an explicit ring-element bucket and
/// split. Setup certification uses the maximum current length in each
/// `ceil(log2(ring_elems))` bucket, which dominates every shorter member for
/// the same split.
#[allow(clippy::too_many_arguments)]
pub(crate) fn recursive_fold_level_params_candidate(
    policy: &PlannerPolicy,
    ring_challenge_cfg: &akita_challenges::SparseChallengeConfig,
    num_ring_elems: usize,
    reduced_vars: usize,
    log_basis: u32,
    fold_level: usize,
    block_index_bits: usize,
    requested_fold_shape: TensorChallengeShape,
) -> Result<Option<LevelParams>, AkitaError> {
    if reduced_vars <= 2
        || reduced_vars >= 53
        || block_index_bits == 0
        || block_index_bits >= reduced_vars
    {
        return Ok(None);
    }
    let num_chunks = policy.chunks_at_level(fold_level);
    let num_positions_per_block = 1usize
        .checked_shl((reduced_vars - block_index_bits) as u32)
        .ok_or_else(|| {
            AkitaError::InvalidSetup("recursive candidate position count overflow".to_string())
        })?;
    let num_live_blocks = num_ring_elems.div_ceil(num_positions_per_block);
    if num_live_blocks < num_chunks {
        return Ok(None);
    }
    let fold_challenge_shape =
        optimize_fold_challenge_shape(requested_fold_shape, num_live_blocks)?;
    let decomp = DecompositionParams {
        log_basis,
        ..policy.decomposition
    };
    let delta_commit = num_digits_s_commit(decomp, false);
    let delta_open = num_digits_open(decomp);
    let Some(width_s) = decomposed_s_block_ring_count(num_positions_per_block, delta_commit) else {
        return Ok(None);
    };
    let Some(norm_s) = rounded_up_role_a_inf_norm(
        policy.sis_security_policy,
        policy.sis_modulus_profile,
        policy.ring_dimension,
        decomp,
        ring_challenge_cfg,
        fold_challenge_shape,
        false,
        policy.onehot_chunk_size,
        policy.ring_subfield_norm_bound,
        num_live_blocks,
        1,
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
    let Some(norm_t) = rounded_up_collision_inf_norm(
        policy.sis_security_policy,
        policy.sis_modulus_profile,
        akita_types::SisMatrixRole::B,
        policy.ring_dimension,
        log_basis,
    ) else {
        return Ok(None);
    };
    let Some(width_t) = decomposed_t_ring_count(a_key.row_len(), delta_open, num_live_blocks, 1)
    else {
        return Ok(None);
    };
    let Ok(b_key) = AjtaiKeyParams::try_new_with_min_rank(
        sis_key(policy, akita_types::SisMatrixRole::B, norm_t),
        width_t,
    ) else {
        return Ok(None);
    };
    let Some(norm_w) = rounded_up_collision_inf_norm(
        policy.sis_security_policy,
        policy.sis_modulus_profile,
        akita_types::SisMatrixRole::D,
        policy.ring_dimension,
        log_basis,
    ) else {
        return Ok(None);
    };
    let Some(width_w) = decomposed_w_ring_count(delta_open, num_live_blocks, 1) else {
        return Ok(None);
    };
    let Ok(d_key) = AjtaiKeyParams::try_new_with_min_rank(
        sis_key(policy, akita_types::SisMatrixRole::D, norm_w),
        width_w,
    ) else {
        return Ok(None);
    };
    let mut params = LevelParams {
        ring_dimension: policy.ring_dimension,
        log_basis,
        a_key,
        b_key,
        d_key,
        num_live_ring_elements_per_claim: num_ring_elems,
        num_positions_per_block,
        num_live_blocks,
        fold_challenge_config: *ring_challenge_cfg,
        fold_challenge_shape,
        num_digits_commit: delta_commit,
        num_digits_open: delta_open,
        onehot_chunk_size: 0,
        fold_linf_cap_config: FoldWitnessLinfCapConfig::worst_case_beta_only(),
        num_digits_fold_one: 1,
        field_bits_hint: 0,
        cached_num_digits_block_claims: 0,
        cached_num_digits_fold_value: 1,
        witness_chunk: policy.witness_chunk_for_level(fold_level),
        precommitted_groups: Vec::new(),
        setup_prefix: None,
        role_dims: CommitmentRingDims::uniform(policy.ring_dimension),
        setup_contribution_mode: SetupContributionMode::Direct,
    }
    .with_fold_linf_cap_config(policy.decomposition.field_bits(), 1)?;
    params.stamp_role_dims_from_keys();
    Ok(Some(params))
}

fn checked_power_of_two_vars(field_len: usize, context: &'static str) -> Result<usize, AkitaError> {
    if field_len == 0 {
        return Err(AkitaError::InvalidSetup(format!(
            "{context} must be nonzero"
        )));
    }
    let padded = field_len.checked_next_power_of_two().ok_or_else(|| {
        AkitaError::InvalidSetup(format!("{context} power-of-two padding overflow"))
    })?;
    Ok(padded.trailing_zeros() as usize)
}

pub fn suffix_opening_layout(
    current_witness_len: usize,
    incoming_setup_prefix: Option<usize>,
) -> Result<OpeningClaimsLayout, AkitaError> {
    let witness_vars = checked_power_of_two_vars(current_witness_len, "suffix witness length")?;
    let witness_group = PolynomialGroupLayout::singleton(witness_vars);
    match incoming_setup_prefix {
        Some(natural_len) => {
            let n_prefix = padded_setup_prefix_len(natural_len);
            if n_prefix == 0 || !n_prefix.is_power_of_two() {
                return Err(AkitaError::InvalidSetup(
                    "incoming setup prefix length must be a nonzero power of two".to_string(),
                ));
            }
            let prefix_vars = checked_power_of_two_vars(n_prefix, "incoming setup prefix length")?;
            OpeningClaimsLayout::from_groups(vec![
                PolynomialGroupLayout::singleton(prefix_vars),
                witness_group,
            ])
        }
        None => OpeningClaimsLayout::from_groups(vec![witness_group]),
    }
}

#[allow(clippy::too_many_arguments)]
fn grouped_segment_rings(
    num_polys: usize,
    num_live_blocks: usize,
    num_chunks: usize,
    num_positions_per_block: usize,
    n_a: usize,
    num_digits_commit: usize,
    num_digits_open: usize,
    num_digits_fold: usize,
) -> Result<usize, AkitaError> {
    let e_hat = num_polys
        .checked_mul(num_live_blocks)
        .and_then(|n| n.checked_mul(num_digits_open))
        .ok_or_else(|| AkitaError::InvalidSetup("group e-hat witness overflow".to_string()))?;
    let t_hat = num_polys
        .checked_mul(num_live_blocks)
        .and_then(|n| n.checked_mul(n_a))
        .and_then(|n| n.checked_mul(num_digits_open))
        .ok_or_else(|| AkitaError::InvalidSetup("group t-hat witness overflow".to_string()))?;
    let z_hat = num_positions_per_block
        .checked_mul(num_digits_commit)
        .and_then(|n| n.checked_mul(num_digits_fold))
        .and_then(|n| n.checked_mul(num_chunks))
        .ok_or_else(|| AkitaError::InvalidSetup("group z-hat witness overflow".to_string()))?;

    e_hat
        .checked_add(t_hat)
        .and_then(|n| n.checked_add(z_hat))
        .ok_or_else(|| AkitaError::InvalidSetup("group witness overflow".to_string()))
}

pub(crate) fn planned_next_witness_len(
    field_bits: u32,
    params: &LevelParams,
    final_num_polys: usize,
    layout: RelationMatrixRowLayout,
    num_chunks: usize,
) -> Result<usize, AkitaError> {
    if !params.precommitted_groups.is_empty() {
        return Err(AkitaError::InvalidSetup(
            "multi-group root witness sizing must use LevelParams::next_w_len".to_string(),
        ));
    }
    if params.setup_prefix.is_some() {
        return grouped_setup_prefix_next_witness_len(
            field_bits,
            params,
            final_num_polys,
            layout,
            num_chunks,
        );
    }

    w_ring_element_count_for_chunks(field_bits, params, final_num_polys, layout, num_chunks)?
        .checked_mul(params.ring_dimension)
        .ok_or_else(|| AkitaError::InvalidSetup("next witness length overflow".into()))
}

fn grouped_setup_prefix_next_witness_len(
    field_bits: u32,
    params: &LevelParams,
    final_num_polys: usize,
    layout: RelationMatrixRowLayout,
    num_chunks: usize,
) -> Result<usize, AkitaError> {
    let mut total = grouped_segment_rings(
        final_num_polys,
        params.num_live_blocks,
        num_chunks,
        params.num_positions_per_block,
        params.a_key.row_len(),
        params.num_digits_commit,
        params.num_digits_open,
        params.num_digits_fold(final_num_polys, field_bits)?,
    )?;
    for group in params.precommitted_group_iter() {
        let group_rings = grouped_segment_rings(
            group.layout.group.num_polynomials(),
            group.layout.num_live_blocks,
            num_chunks,
            group.layout.num_positions_per_block,
            group.a_key.row_len(),
            group.num_digits_commit,
            group.num_digits_open,
            group.num_digits_fold_one,
        )?;
        total = total
            .checked_add(group_rings)
            .ok_or_else(|| AkitaError::InvalidSetup("grouped witness overflow".to_string()))?;
    }

    let r_rows =
        params.relation_matrix_row_count_for(params.precommitted_group_count() + 1, layout)?;
    let r_count = r_rows
        .checked_mul(akita_types::sis::compute_num_digits_full_field(
            field_bits,
            params.log_basis,
        ))
        .ok_or_else(|| AkitaError::InvalidSetup("grouped r-tail witness overflow".to_string()))?;
    let rings = total
        .checked_add(r_count)
        .ok_or_else(|| AkitaError::InvalidSetup("grouped witness overflow".to_string()))?;

    rings
        .checked_mul(params.ring_dimension)
        .ok_or_else(|| AkitaError::InvalidSetup("grouped next witness length overflow".to_string()))
}

pub(crate) fn terminal_witness_shape_for_opening_layout(
    terminal_lp: &LevelParams,
    field_bits: u32,
    opening_layout: &OpeningClaimsLayout,
) -> Result<CleartextWitnessShape, AkitaError> {
    if !terminal_lp.precommitted_groups.is_empty() {
        return Err(AkitaError::InvalidSetup(
            "grouped terminal direct witness layout is unsupported".to_string(),
        ));
    }
    let order = opening_layout.root_group_order()?;
    let mut group_shapes: Vec<(&dyn akita_types::LevelParamsLike, usize, usize, usize)> =
        Vec::with_capacity(order.len());
    for &group_index in &order {
        let group_lp = terminal_lp.group_params(opening_layout, group_index)?;
        let group_polys = opening_layout.group_layout(group_index)?.num_polynomials();
        group_shapes.push((group_lp, group_polys, group_polys, 1));
    }
    segment_typed_witness_shape_from_groups(
        terminal_lp,
        field_bits,
        group_shapes,
        opening_layout.num_groups(),
        akita_types::TerminalQuotientMode::Omit,
    )
}

fn derive_setup_prefix_group(
    policy: &PlannerPolicy,
    ring_challenge_cfg: &SparseChallengeConfig,
    requested_fold_shape: TensorChallengeShape,
    log_basis: u32,
    n_prefix: usize,
    num_chunks: usize,
) -> Result<Option<PrecommittedLevelParams>, AkitaError> {
    if policy.ring_dimension != SETUP_OFFLOAD_D_SETUP {
        return Err(AkitaError::InvalidSetup(
            "recursive setup planning requires D64".to_string(),
        ));
    }
    if n_prefix == 0 || !n_prefix.is_power_of_two() {
        return Err(AkitaError::InvalidSetup(
            "setup prefix length must be a nonzero power of two".to_string(),
        ));
    }
    if !n_prefix.is_multiple_of(policy.ring_dimension) {
        return Err(AkitaError::InvalidSetup(
            "setup prefix length must be a multiple of the ring dimension".to_string(),
        ));
    }
    let ring_slots = n_prefix / policy.ring_dimension;
    let reduced_vars = checked_power_of_two_vars(ring_slots, "setup prefix ring slots")?;
    let prefix_num_vars = checked_power_of_two_vars(n_prefix, "setup prefix field length")?;
    let family = policy.sis_modulus_profile;
    let d = policy.ring_dimension;
    let decomp = DecompositionParams {
        log_basis,
        ..policy.decomposition
    };
    let num_digits_commit = num_digits_setup_prefix_commit(decomp);
    let num_digits_open_val = num_digits_open(decomp);
    let mut best: Option<(LayoutCandidateScore, PrecommittedLevelParams)> = None;

    for block_index_bits in (0..=reduced_vars).rev() {
        let Some(num_live_blocks) = 1usize.checked_shl(block_index_bits as u32) else {
            continue;
        };
        let position_index_bits = reduced_vars - block_index_bits;
        let Some(num_positions_per_block) = 1usize.checked_shl(position_index_bits as u32) else {
            continue;
        };
        let fold_shape = optimize_fold_challenge_shape(requested_fold_shape, num_live_blocks)?;
        if num_live_blocks < num_chunks {
            continue;
        }
        let Some(width_s) =
            decomposed_s_block_ring_count(num_positions_per_block, num_digits_commit)
        else {
            continue;
        };
        let Some(norm_s) = rounded_up_role_a_inf_norm(
            policy.sis_security_policy,
            family,
            d,
            decomp,
            ring_challenge_cfg,
            fold_shape,
            false,
            0,
            policy.ring_subfield_norm_bound,
            num_live_blocks,
            1,
            width_s as u64,
        ) else {
            continue;
        };
        let Ok(a_key) = AjtaiKeyParams::try_new_with_min_rank(
            sis_key(policy, akita_types::SisMatrixRole::A, norm_s),
            width_s,
        ) else {
            continue;
        };
        let Some(norm_t) = rounded_up_collision_inf_norm(
            policy.sis_security_policy,
            family,
            akita_types::SisMatrixRole::B,
            d,
            log_basis,
        ) else {
            continue;
        };
        let Some(width_t) =
            decomposed_t_ring_count(a_key.row_len(), num_digits_open_val, num_live_blocks, 1)
        else {
            continue;
        };
        let Ok(b_key) = AjtaiKeyParams::try_new_with_min_rank(
            sis_key(policy, akita_types::SisMatrixRole::B, norm_t),
            width_t,
        ) else {
            continue;
        };
        let fold_linf_cap_config =
            FoldWitnessLinfCapConfig::for_fold_level(ring_challenge_cfg, fold_shape, d, width_s)?;
        let challenge = FoldChallengeNorms {
            infinity_norm: fold_shape.effective_infinity_norm(ring_challenge_cfg) as u128,
            l1_norm: fold_shape.effective_l1_mass(ring_challenge_cfg) as u128,
        };
        let (num_digits_fold_one, _) = fold_witness_digit_plan(
            num_live_blocks,
            1,
            policy.decomposition.field_bits(),
            log_basis,
            challenge,
            FoldWitnessNorms::new(log_basis, d, 1, false),
            &fold_linf_cap_config,
        )?;
        let layout = PrecommittedGroupParams {
            group: PolynomialGroupLayout::singleton(prefix_num_vars),
            num_live_ring_elements_per_claim: ring_slots,
            num_positions_per_block,
            num_live_blocks,
            fold_challenge_shape: fold_shape,
            log_basis,
            n_a: a_key.row_len(),
            conservative_n_b: b_key.row_len(),
        };
        let params = PrecommittedLevelParams {
            layout,
            a_key,
            b_key,
            num_digits_commit,
            num_digits_open: num_digits_open_val,
            num_digits_fold_one,
        };
        let physical_width = grouped_segment_rings(
            1,
            num_live_blocks,
            num_chunks,
            num_positions_per_block,
            params.a_key.row_len(),
            num_digits_commit,
            num_digits_open_val,
            num_digits_fold_one,
        )?;
        let score =
            layout_candidate_score(physical_width, num_live_blocks, num_chunks, fold_shape)?;
        if best
            .as_ref()
            .is_none_or(|(best_score, _)| score < *best_score)
        {
            best = Some((score, params));
        }
    }

    Ok(best.map(|(_, params)| params))
}

/// Compute parameters that generate the smallest witness for the next
/// fold level. Note that this is not the optimum case: in the optimum
/// case (similar to `find_schedule`), we should check that current proof
/// size + suffix cost is the smallest. However, as time blows up, we
/// don't do that here.
pub(crate) fn derive_candidate_level_params(
    policy: &PlannerPolicy,
    ring_challenge_cfg: &akita_challenges::SparseChallengeConfig,
    current_witness_len: usize,
    log_basis: u32,
    fold_level: usize,
    incoming_setup_prefix: Option<usize>,
    requested_fold_shape: TensorChallengeShape,
) -> Result<Option<(LevelParams, usize, usize)>, AkitaError> {
    // Chunk count of the witness this level commits/produces (sized below as
    // `next_witness_len`). Equal for the metadata field and the width pricing so
    // a future verifier recomputing the size from `witness_chunk` agrees.
    let num_chunks = policy.chunks_at_level(fold_level);
    if !current_witness_len.is_multiple_of(policy.ring_dimension) {
        return Ok(None);
    }
    let num_ring_elems = current_witness_len / policy.ring_dimension;
    let reduced_vars = num_ring_elems
        .checked_next_power_of_two()
        .ok_or_else(|| AkitaError::InvalidSetup("recursive witness capacity overflow".to_string()))?
        .max(1)
        .trailing_zeros() as usize;

    if reduced_vars <= 2 || reduced_vars >= 53 {
        return Err(AkitaError::InvalidSetup(format!(
            "recursive fold candidate reduced_vars={reduced_vars} is outside \
             the optimizable range [3, 52]"
        )));
    }

    let setup_prefix = match incoming_setup_prefix {
        Some(natural_len) => {
            let n_prefix = padded_setup_prefix_len(natural_len);
            let Some(group) = derive_setup_prefix_group(
                policy,
                ring_challenge_cfg,
                requested_fold_shape,
                log_basis,
                n_prefix,
                num_chunks,
            )?
            else {
                return Ok(None);
            };
            Some(akita_types::setup_prefix_slot_id(
                SETUP_OFFLOAD_D_SETUP,
                natural_len,
                group,
            ))
        }
        None => None,
    };

    let mut best: Option<(LayoutCandidateScore, LevelParams, usize, usize)> = None;
    for r in (1..reduced_vars).rev() {
        let Some(candidate_params) = recursive_fold_level_params_candidate(
            policy,
            ring_challenge_cfg,
            num_ring_elems,
            reduced_vars,
            log_basis,
            fold_level,
            r,
            requested_fold_shape,
        )?
        else {
            continue;
        };
        let mut candidate_params = candidate_params;
        candidate_params.setup_prefix = setup_prefix.clone();
        candidate_params.setup_contribution_mode = SetupContributionMode::Direct;
        if let Some(prefix) = &candidate_params.setup_prefix {
            let prefix_d_width = prefix.commitment_params.d_segment_width()?;
            let total_d_width = candidate_params
                .d_key
                .col_len()
                .checked_add(prefix_d_width)
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("setup-prefix shared D width overflow".to_string())
                })?;
            candidate_params.d_key = AjtaiKeyParams::try_new_with_min_rank(
                candidate_params.d_key.sis_table_key(),
                total_d_width,
            )?;
        }
        let next_witness_len = planned_next_witness_len(
            policy.decomposition.field_bits(),
            &candidate_params,
            1,
            RelationMatrixRowLayout::WithDBlock,
            num_chunks,
        )?;
        let terminal_shape = segment_typed_witness_shape_from_groups(
            &candidate_params,
            policy.decomposition.field_bits(),
            [(&candidate_params as &dyn LevelParamsLike, 1, 1, 1)],
            1,
            akita_types::TerminalQuotientMode::Omit,
        )?;
        let next_witness_len_terminal = terminal_shape.logical_num_elems();

        let score = layout_candidate_score(
            next_witness_len,
            candidate_params.num_live_blocks,
            num_chunks,
            candidate_params.fold_challenge_shape,
        )?;
        if best
            .as_ref()
            .is_none_or(|(best_score, _, _, _)| score < *best_score)
        {
            best = Some((
                score,
                candidate_params,
                next_witness_len,
                next_witness_len_terminal,
            ));
        }
    }

    let Some((_, candidate_params, next_witness_len, next_witness_len_terminal)) = best else {
        return Ok(None);
    };

    if next_witness_len >= current_witness_len {
        return Ok(None);
    }

    Ok(Some((
        candidate_params,
        next_witness_len,
        next_witness_len_terminal,
    )))
}

/// Brute-forced root-direct commit `LevelParams` (optimal `(r_pos, r_blk)` split).
///
/// Root-direct schedules ship the cleartext witness on the wire, so they
/// don't run the relation fold (D unused). The planner brute-forces the
/// committed `(m, r, n_a, n_b, n_d)` here via the SIS-floor search and
/// stores it in `GeneratedDirectStep.commit`; the runtime reconstructs the
/// identical `LevelParams` with `GeneratedFoldStep::expand_to_level_params`.
///
/// This derives every value directly and assembles a single `LevelParams`:
///
/// - `a_collision` — the audited A-role SIS bucket (`2·β` base norm scaled
///   by the stage-1 infinity norm and the ring-subfield embedding norm).
/// - `bd_collision = 2^lb − 1` — the B/D digit-range bucket.
/// - `(position_index_bits, block_index_bits)` — `optimal_block_geometry_split` for a normal root, or `(0, 0)`
///   for a tiny root that fits inside one padded ring element.
/// - `(n_a, n_b, n_d)` — the tight SIS-floor ranks for the resulting
///   inner / outer / D-matrix widths.
///
/// - `(n_a, n_b, n_d)` — the tight SIS-floor ranks for the resulting
///   inner / outer / D-matrix widths, where the outer (B) and prover (D)
///   widths already carry the `num_claims` batch factor (the root commits
///   `num_claims` polynomials, so there is no separate per-claim-then-scale
///   step; `num_claims == 1` is the singleton root).
///
/// `fold_challenge_shape` is stamped onto the committed level (the level-0
/// shape; the `(r_pos, r_blk)` split itself is scored against the flat L1 mass).
///
/// Returns `Ok(None)` when any SIS-floor lookup or bound arithmetic rejects
/// the candidate (the uncommittable edge), matching the previous
/// `Result::ok()` fallback.
pub(crate) fn compute_root_direct_level_params(
    policy: &PlannerPolicy,
    ring_challenge_cfg: &akita_challenges::SparseChallengeConfig,
    num_vars: usize,
    log_basis: u32,
    requested_fold_shape: TensorChallengeShape,
    num_claims: usize,
) -> Result<Option<LevelParams>, AkitaError> {
    let d = policy.ring_dimension;
    let sis_modulus_profile = policy.sis_modulus_profile;
    let decomp = policy.decomposition;
    let alpha = (d as u32).trailing_zeros() as usize;

    let level_decomp = DecompositionParams {
        log_basis,
        ..decomp
    };
    // Root-direct commits against `log_commit_bound` (the root form of
    // `num_digits_s_commit`) and opens at `log_open_bound`.
    let depth_commit = num_digits_s_commit(level_decomp, true);
    let depth_open = num_digits_open(level_decomp);

    // Outer/inner variable split: brute-force the optimum for a normal root,
    // single-block `(0, 0)` for a tiny root (`num_vars <= log2(d)`). The
    // optimizer recomputes the fold-priced A collision per candidate
    // (it grows with the exact fold arity `num_claims · B`), so it needs the
    // batch factor and ring-subfield norm, not a single pre-baked bucket.
    let (position_index_bits, block_index_bits) = if num_vars > alpha {
        // The `(r_pos, r_blk)` split is scored against the flat L1 mass (the root fold
        // shape disambiguates the committed table, not the split search).
        let fold_challenge = akita_types::sis::FoldChallengeNorms::new(
            ring_challenge_cfg,
            TensorChallengeShape::Flat,
        );
        // One-hot root commits a sparse witness (`||s||_inf = 1`,
        // `nonzeros = ceil(D/K)`); dense roots use the balanced-digit norms.
        let is_onehot = decomp.log_commit_bound == 1;
        let fold_witness = FoldWitnessNorms::new(log_basis, d, policy.onehot_chunk_size, is_onehot);
        let (position_index_bits, block_index_bits, _scoring_n_a) = optimal_block_geometry_split(
            policy.sis_security_policy,
            sis_modulus_profile,
            d as u32,
            num_claims,
            policy.ring_subfield_norm_bound,
            fold_challenge,
            fold_witness,
            ring_challenge_cfg,
            TensorChallengeShape::Flat,
            decomp,
            policy.onehot_chunk_size,
            num_vars - alpha,
            0,
        );
        (position_index_bits, block_index_bits)
    } else {
        (0, 0)
    };

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
        optimize_fold_challenge_shape(requested_fold_shape, num_live_blocks)?;

    // The A/B/D keys, composed from the `akita_types::sis` primitives:
    // norm -> width -> tight SIS-secure rank -> key. `t_vectors = num_claims`
    // folds the batched-root scaling into the B/D widths (the root commits
    // `num_claims` polynomials) — no separate per-claim-then-scale pass.
    let Some(width_s) = decomposed_s_block_ring_count(num_positions_per_block, depth_commit) else {
        return Ok(None);
    };
    let Some(norm_s) = rounded_up_role_a_inf_norm(
        policy.sis_security_policy,
        sis_modulus_profile,
        d,
        level_decomp,
        ring_challenge_cfg,
        fold_challenge_shape,
        true,
        policy.onehot_chunk_size,
        policy.ring_subfield_norm_bound,
        num_live_blocks,
        num_claims,
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
        sis_modulus_profile,
        akita_types::SisMatrixRole::B,
        d,
        log_basis,
    ) else {
        return Ok(None);
    };
    let Some(width_t) = decomposed_t_ring_count(n_a, depth_open, num_live_blocks, num_claims)
    else {
        return Ok(None);
    };
    let Ok(b_key) = AjtaiKeyParams::try_new_with_min_rank(
        sis_key(policy, akita_types::SisMatrixRole::B, norm_t),
        width_t,
    ) else {
        return Ok(None);
    };
    let Some(norm_w) = rounded_up_collision_inf_norm(
        policy.sis_security_policy,
        sis_modulus_profile,
        akita_types::SisMatrixRole::D,
        d,
        log_basis,
    ) else {
        return Ok(None);
    };
    let Some(width_w) = decomposed_w_ring_count(depth_open, num_live_blocks, num_claims) else {
        return Ok(None);
    };
    let Ok(d_key) = AjtaiKeyParams::try_new_with_min_rank(
        sis_key(policy, akita_types::SisMatrixRole::D, norm_w),
        width_w,
    ) else {
        return Ok(None);
    };

    // A one-hot root (`log_commit_bound == 1`) commits a sparse witness; record
    // its chunk size so `num_digits_fold` and the binding norm size the folded
    // witness against `nonzeros = ceil(D/K)` instead of `D`.
    let onehot_chunk_size = if decomp.log_commit_bound == 1 {
        policy.onehot_chunk_size
    } else {
        0
    };

    let mut root_direct_params = LevelParams {
        ring_dimension: d,
        log_basis,
        a_key,
        b_key,
        d_key,
        num_live_ring_elements_per_claim,
        num_positions_per_block,
        num_live_blocks,
        fold_challenge_config: *ring_challenge_cfg,
        fold_challenge_shape,
        num_digits_commit: depth_commit,
        num_digits_open: depth_open,
        onehot_chunk_size,
        fold_linf_cap_config: FoldWitnessLinfCapConfig::worst_case_beta_only(),
        num_digits_fold_one: 1,
        field_bits_hint: 0,
        cached_num_digits_block_claims: 0,
        cached_num_digits_fold_value: 1,
        // Root-direct ships the raw polynomial on the wire (no chunked commitment).
        witness_chunk: ChunkedWitnessCfg::default(),
        precommitted_groups: Vec::new(),
        setup_prefix: None,
        role_dims: CommitmentRingDims::uniform(d),
        setup_contribution_mode: SetupContributionMode::Direct,
    }
    .with_fold_linf_cap_config(decomp.field_bits(), num_claims)?;
    root_direct_params.stamp_role_dims_from_keys();
    Ok(Some(root_direct_params))
}

/// Build one scalar root-fold candidate for an explicit basis and split.
///
/// `Ok(None)` is the canonical candidate-infeasibility signal used by both
/// schedule optimization and setup-capacity certification.
pub(crate) fn scalar_root_fold_level_params_candidate(
    policy: &PlannerPolicy,
    ring_challenge_cfg: &akita_challenges::SparseChallengeConfig,
    num_vars: usize,
    num_claims: usize,
    log_basis: u32,
    block_index_bits: usize,
    requested_fold_shape: TensorChallengeShape,
) -> Result<Option<LevelParams>, AkitaError> {
    let alpha = (policy.ring_dimension as u32).trailing_zeros() as usize;
    let reduced_vars = num_vars.saturating_sub(alpha);
    if reduced_vars == 0 || block_index_bits >= reduced_vars {
        return Ok(None);
    }
    let num_live_blocks = 1usize.checked_shl(block_index_bits as u32).ok_or_else(|| {
        AkitaError::InvalidSetup("root candidate num_live_blocks overflow".to_string())
    })?;
    let root_num_chunks = policy.chunks_at_level(0);
    if num_live_blocks < root_num_chunks {
        return Ok(None);
    }
    let fold_challenge_shape =
        optimize_fold_challenge_shape(requested_fold_shape, num_live_blocks)?;
    let position_index_bits = reduced_vars - block_index_bits;
    let num_positions_per_block =
        1usize
            .checked_shl(position_index_bits as u32)
            .ok_or_else(|| {
                AkitaError::InvalidSetup("root candidate position count overflow".to_string())
            })?;
    let num_live_ring_elements_per_claim = num_live_blocks
        .checked_mul(num_positions_per_block)
        .ok_or_else(|| AkitaError::InvalidSetup("root candidate source length overflow".into()))?;
    let level_decomp = DecompositionParams {
        log_basis,
        ..policy.decomposition
    };
    let num_digits_commit = num_digits_s_commit(level_decomp, true);
    let num_digits_open = num_digits_open(level_decomp);
    let Some(width_s) = decomposed_s_block_ring_count(num_positions_per_block, num_digits_commit)
    else {
        return Ok(None);
    };
    let Some(norm_s) = rounded_up_role_a_inf_norm(
        policy.sis_security_policy,
        policy.sis_modulus_profile,
        policy.ring_dimension,
        level_decomp,
        ring_challenge_cfg,
        fold_challenge_shape,
        true,
        policy.onehot_chunk_size,
        policy.ring_subfield_norm_bound,
        num_live_blocks,
        num_claims,
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
    let Some(norm_t) = rounded_up_collision_inf_norm(
        policy.sis_security_policy,
        policy.sis_modulus_profile,
        akita_types::SisMatrixRole::B,
        policy.ring_dimension,
        log_basis,
    ) else {
        return Ok(None);
    };
    let Some(width_t) = decomposed_t_ring_count(
        a_key.row_len(),
        num_digits_open,
        num_live_blocks,
        num_claims,
    ) else {
        return Ok(None);
    };
    let Ok(b_key) = AjtaiKeyParams::try_new_with_min_rank(
        sis_key(policy, akita_types::SisMatrixRole::B, norm_t),
        width_t,
    ) else {
        return Ok(None);
    };
    let Some(norm_w) = rounded_up_collision_inf_norm(
        policy.sis_security_policy,
        policy.sis_modulus_profile,
        akita_types::SisMatrixRole::D,
        policy.ring_dimension,
        log_basis,
    ) else {
        return Ok(None);
    };
    let Some(width_w) = decomposed_w_ring_count(num_digits_open, num_live_blocks, num_claims)
    else {
        return Ok(None);
    };
    let Ok(d_key) = AjtaiKeyParams::try_new_with_min_rank(
        sis_key(policy, akita_types::SisMatrixRole::D, norm_w),
        width_w,
    ) else {
        return Ok(None);
    };
    let onehot_chunk_size = if policy.decomposition.log_commit_bound == 1 {
        policy.onehot_chunk_size
    } else {
        0
    };
    let mut params = (LevelParams {
        ring_dimension: policy.ring_dimension,
        log_basis,
        a_key,
        b_key,
        d_key,
        num_live_ring_elements_per_claim,
        num_positions_per_block,
        num_live_blocks,
        fold_challenge_config: *ring_challenge_cfg,
        fold_challenge_shape,
        num_digits_commit,
        num_digits_open,
        onehot_chunk_size,
        fold_linf_cap_config: FoldWitnessLinfCapConfig::worst_case_beta_only(),
        num_digits_fold_one: 1,
        field_bits_hint: 0,
        cached_num_digits_block_claims: 0,
        cached_num_digits_fold_value: 1,
        witness_chunk: policy.witness_chunk_for_level(0),
        precommitted_groups: Vec::new(),
        setup_prefix: None,
        role_dims: CommitmentRingDims::uniform(policy.ring_dimension),
        setup_contribution_mode: SetupContributionMode::Direct,
    })
    .with_fold_linf_cap_config(policy.decomposition.field_bits(), num_claims)?;
    params.stamp_role_dims_from_keys();
    Ok(Some(params))
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_challenges::SparseChallengeConfig;
    use akita_types::{PolynomialGroupLayout, SisModulusProfileId};

    fn grouped_level_params() -> LevelParams {
        let fold_challenge_config = SparseChallengeConfig::pm1_only(3);
        let mut params = LevelParams::params_only(
            SisModulusProfileId::Q128OffsetA7F7,
            64,
            3,
            2,
            2,
            2,
            fold_challenge_config,
        )
        .with_decomp(2, 2, 2, 2)
        .expect("grouped params");
        let precommitted = LevelParams::params_only(
            SisModulusProfileId::Q128OffsetA7F7,
            64,
            3,
            2,
            2,
            2,
            fold_challenge_config,
        )
        .with_decomp(2, 2, 2, 2)
        .expect("precommitted params");
        params.precommitted_groups = vec![PrecommittedLevelParams {
            layout: PrecommittedGroupParams::from_params(
                PolynomialGroupLayout::new(6, 1),
                &precommitted,
            ),
            a_key: precommitted.a_key.clone(),
            b_key: precommitted.b_key.clone(),
            num_digits_commit: precommitted.num_digits_commit,
            num_digits_open: precommitted.num_digits_open,
            num_digits_fold_one: precommitted.num_digits_fold_one,
        }];
        params
    }

    #[test]
    fn planned_next_witness_len_rejects_multi_group_root_level_params() {
        let grouped = grouped_level_params();
        let err =
            planned_next_witness_len(128, &grouped, 1, RelationMatrixRowLayout::WithDBlock, 1)
                .expect_err("multi-group root suffix sizing must use next_w_len");
        assert!(matches!(err, AkitaError::InvalidSetup(_)));
    }

    #[test]
    fn terminal_witness_shape_rejects_multi_group_root_level_params() {
        let grouped = grouped_level_params();
        let layout = OpeningClaimsLayout::from_groups(vec![
            PolynomialGroupLayout::new(6, 1),
            PolynomialGroupLayout::new(8, 1),
        ])
        .expect("opening layout");
        let err = terminal_witness_shape_for_opening_layout(&grouped, 128, &layout)
            .expect_err("grouped terminal witness shape is unsupported");
        assert!(matches!(err, AkitaError::InvalidSetup(_)));
    }
}
