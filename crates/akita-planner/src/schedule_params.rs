//! Schedule planner that finds the global minimum proof size.
//!
//! Public entry: [`find_schedule`]. The search is `Cfg`-free: every
//! per-preset input is carried by the plain-value [`PlannerPolicy`] plus
//! the `ring_challenge_config` / `fold_challenge_shape_at_level` closures,
//! exactly the shape `crate::schedule_from_entry` already consumes. This keeps the
//! DP a pure function of `(policy, key)` so `akita-config` can call it
//! directly on a schedule-table miss without a dependency cycle.

use std::collections::{BTreeMap, HashMap};

use akita_challenges::{SparseChallengeConfig, TensorChallengeShape};
use akita_field::AkitaError;
use akita_types::layout::digit_math::optimal_m_r_split;
use akita_types::sis::{
    decomposed_s_block_ring_count, decomposed_t_ring_count, decomposed_w_ring_count,
    fold_witness_digit_plan, num_digits_open, num_digits_s_commit, num_digits_setup_prefix_commit,
    rounded_up_collision_inf_norm, rounded_up_role_a_inf_norm, AjtaiKeyParams, FoldChallengeNorms,
    FoldWitnessLinfCapConfig, FoldWitnessNorms, SisTableKey,
};
use akita_types::{
    active_setup_field_len, direct_witness_bytes, extension_opening_reduction_level_bytes,
    level_proof_bytes, padded_setup_prefix_len, segment_typed_witness_shape_from_groups,
    w_ring_element_count_for_chunks, AkitaScheduleInputs, ChunkedWitnessCfg, CleartextWitnessShape,
    CommitmentRingDims, DecompositionParams, DirectStep, FoldStep, LevelParams,
    OpeningClaimsLayout, PolynomialGroupLayout, PrecommittedGroupParams, PrecommittedLevelParams,
    RelationMatrixRowLayout, Schedule, SetupContributionMode, Step, SETUP_OFFLOAD_D_SETUP,
    SETUP_OFFLOAD_MIN_PREFIX_FIELD_LEN,
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
/// Validate the policy's multi-chunk witness settings at a planner entry point.
///
/// Layout-only rules live on [`ChunkedWitnessCfg::validate`]; the recursion-depth
/// bound (which needs the planner-private [`MAX_RECURSION_DEPTH`]) is enforced
/// here so `akita-types` stays free of planner internals.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidSetup`] for an invalid `ChunkedWitnessCfg`, or
/// `num_activated_levels` beyond the planner recursion cap. Verifier-reachable: never panics.
pub(crate) fn validate_policy_witness_chunk(policy: &PlannerPolicy) -> Result<(), AkitaError> {
    let mc = policy.witness_chunk;
    mc.validate()?;
    if mc.num_activated_levels > MAX_RECURSION_DEPTH {
        return Err(AkitaError::InvalidSetup(format!(
            "num_activated_levels={} exceeds the planner recursion cap {MAX_RECURSION_DEPTH}",
            mc.num_activated_levels
        )));
    }
    Ok(())
}

/// Stage-1 sparse-challenge closure shared by the planner entry points.
pub(crate) type RingChallengeConfigFn<'a> =
    &'a dyn Fn(usize) -> Result<akita_challenges::SparseChallengeConfig, AkitaError>;

/// Stage-1 fold-round challenge-shape closure (`level 0` root shape).
type FoldShapeFn<'a> = &'a dyn Fn(AkitaScheduleInputs) -> TensorChallengeShape;

// Suffix-DP depth cap. Schedules in our working parameter range never need
// more than this many recursive fold levels; deeper search only blows up
// memo state without changing emitted tables.
pub(crate) const MAX_RECURSION_DEPTH: usize = 12;

/// Build one recursive-fold candidate for an explicit ring-element bucket and
/// split. Setup certification uses the maximum current length in each
/// `ceil(log2(ring_elems))` bucket, which dominates every shorter member for
/// the same split.
pub(crate) fn recursive_fold_level_params_candidate(
    policy: &PlannerPolicy,
    ring_challenge_cfg: &akita_challenges::SparseChallengeConfig,
    num_ring_elems: usize,
    reduced_vars: usize,
    log_basis: u32,
    fold_level: usize,
    r_vars: usize,
) -> Result<Option<LevelParams>, AkitaError> {
    if reduced_vars <= 2 || reduced_vars >= 53 || r_vars == 0 || r_vars >= reduced_vars {
        return Ok(None);
    }
    let num_chunks = policy.chunks_at_level(fold_level);
    let num_blocks = 1usize.checked_shl(r_vars as u32).ok_or_else(|| {
        AkitaError::InvalidSetup("recursive candidate num_blocks overflow".to_string())
    })?;
    if num_chunks > 1 && !num_blocks.is_multiple_of(num_chunks) {
        return Ok(None);
    }
    let block_len = num_ring_elems.div_ceil(num_blocks);
    let decomp = DecompositionParams {
        log_basis,
        ..policy.decomposition
    };
    let delta_commit = num_digits_s_commit(decomp, false);
    let delta_open = num_digits_open(decomp);
    let Some(width_s) = decomposed_s_block_ring_count(block_len, delta_commit) else {
        return Ok(None);
    };
    let Some(norm_s) = rounded_up_role_a_inf_norm(
        policy.min_sis_security_bits,
        policy.sis_family,
        policy.ring_dimension,
        decomp,
        ring_challenge_cfg,
        TensorChallengeShape::Flat,
        false,
        policy.onehot_chunk_size,
        policy.ring_subfield_norm_bound,
        r_vars,
        1,
        width_s as u64,
    ) else {
        return Ok(None);
    };
    let Ok(a_key) = AjtaiKeyParams::try_new_with_min_rank(sis_key(policy, norm_s), width_s) else {
        return Ok(None);
    };
    let Some(norm_t) = rounded_up_collision_inf_norm(
        policy.min_sis_security_bits,
        policy.sis_family,
        policy.ring_dimension,
        log_basis,
    ) else {
        return Ok(None);
    };
    let Some(width_t) = decomposed_t_ring_count(a_key.row_len(), delta_open, num_blocks, 1) else {
        return Ok(None);
    };
    let Ok(b_key) = AjtaiKeyParams::try_new_with_min_rank(sis_key(policy, norm_t), width_t) else {
        return Ok(None);
    };
    let Some(width_w) = decomposed_w_ring_count(delta_open, num_blocks, 1) else {
        return Ok(None);
    };
    let Ok(d_key) = AjtaiKeyParams::try_new_with_min_rank(sis_key(policy, norm_t), width_w) else {
        return Ok(None);
    };
    let mut params = LevelParams {
        ring_dimension: policy.ring_dimension,
        log_basis,
        a_key,
        b_key,
        d_key,
        num_blocks,
        block_len,
        m_vars: reduced_vars - r_vars,
        r_vars,
        fold_challenge_config: *ring_challenge_cfg,
        fold_challenge_shape: TensorChallengeShape::Flat,
        num_digits_commit: delta_commit,
        num_digits_open: delta_open,
        onehot_chunk_size: 0,
        fold_linf_cap_config: FoldWitnessLinfCapConfig::worst_case_beta_only(),
        num_digits_fold_one: 1,
        field_bits_hint: 0,
        cached_num_digits_fold_claims: 0,
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

pub(crate) fn suffix_opening_layout(
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

fn grouped_segment_rings(
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
        .ok_or_else(|| AkitaError::InvalidSetup("group e-hat witness overflow".to_string()))?;
    let t_hat = num_polys
        .checked_mul(num_blocks)
        .and_then(|n| n.checked_mul(n_a))
        .and_then(|n| n.checked_mul(num_digits_open))
        .ok_or_else(|| AkitaError::InvalidSetup("group t-hat witness overflow".to_string()))?;
    let z_hat = block_len
        .checked_mul(num_digits_commit)
        .and_then(|n| n.checked_mul(num_digits_fold))
        .ok_or_else(|| AkitaError::InvalidSetup("group z-hat witness overflow".to_string()))?;

    e_hat
        .checked_add(t_hat)
        .and_then(|n| n.checked_add(z_hat))
        .ok_or_else(|| AkitaError::InvalidSetup("group witness overflow".to_string()))
}

fn grouped_next_witness_len(
    field_bits: u32,
    params: &LevelParams,
    final_num_polys: usize,
    layout: RelationMatrixRowLayout,
) -> Result<usize, AkitaError> {
    let mut total = grouped_segment_rings(
        final_num_polys,
        params.num_blocks,
        params.block_len,
        params.a_key.row_len(),
        params.num_digits_commit,
        params.num_digits_open,
        params.num_digits_fold(final_num_polys, field_bits)?,
    )?;
    for group in params.precommitted_group_iter() {
        let group_rings = grouped_segment_rings(
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

pub(crate) fn planned_next_witness_len(
    field_bits: u32,
    params: &LevelParams,
    final_num_polys: usize,
    layout: RelationMatrixRowLayout,
    num_chunks: usize,
) -> Result<usize, AkitaError> {
    if params.has_precommitted_groups() {
        if num_chunks > 1 {
            return Err(AkitaError::InvalidSetup(
                "setup-prefix grouped suffixes do not support multi-chunk witnesses".to_string(),
            ));
        }
        return grouped_next_witness_len(field_bits, params, final_num_polys, layout);
    }

    w_ring_element_count_for_chunks(field_bits, params, final_num_polys, layout, num_chunks)?
        .checked_mul(params.ring_dimension)
        .ok_or_else(|| AkitaError::InvalidSetup("next witness length overflow".into()))
}

pub(crate) fn terminal_witness_shape_for_opening_layout(
    terminal_lp: &LevelParams,
    field_bits: u32,
    opening_layout: &OpeningClaimsLayout,
) -> Result<CleartextWitnessShape, AkitaError> {
    let order = opening_layout.root_group_order()?;
    let mut group_shapes: Vec<(&dyn akita_types::LevelParamsLike, usize, usize, usize)> =
        Vec::with_capacity(order.len());
    for &group_index in &order {
        let group_lp = terminal_lp.root_group_params(opening_layout, group_index)?;
        let group_polys = opening_layout.group_layout(group_index)?.num_polynomials();
        group_shapes.push((group_lp, group_polys, group_polys, 1));
    }
    segment_typed_witness_shape_from_groups(
        terminal_lp,
        field_bits,
        group_shapes,
        opening_layout.num_groups(),
    )
}

fn derive_setup_prefix_group(
    policy: &PlannerPolicy,
    ring_challenge_cfg: &SparseChallengeConfig,
    fold_shape: TensorChallengeShape,
    log_basis: u32,
    n_prefix: usize,
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
    let family = policy.sis_family;
    let d = policy.ring_dimension;
    let decomp = DecompositionParams {
        log_basis,
        ..policy.decomposition
    };
    let num_digits_commit = num_digits_setup_prefix_commit(decomp);
    let num_digits_open_val = num_digits_open(decomp);
    let mut best: Option<(usize, PrecommittedLevelParams)> = None;

    for r_vars in (0..=reduced_vars).rev() {
        let Some(num_blocks) = 1usize.checked_shl(r_vars as u32) else {
            continue;
        };
        let m_vars = reduced_vars - r_vars;
        let Some(block_len) = 1usize.checked_shl(m_vars as u32) else {
            continue;
        };
        let Some(width_s) = decomposed_s_block_ring_count(block_len, num_digits_commit) else {
            continue;
        };
        let Some(norm_s) = rounded_up_role_a_inf_norm(
            policy.min_sis_security_bits,
            family,
            d,
            decomp,
            ring_challenge_cfg,
            fold_shape,
            false,
            0,
            policy.ring_subfield_norm_bound,
            r_vars,
            1,
            width_s as u64,
        ) else {
            continue;
        };
        let Ok(a_key) = AjtaiKeyParams::try_new_with_min_rank(sis_key(policy, norm_s), width_s)
        else {
            continue;
        };
        let Some(norm_t) =
            rounded_up_collision_inf_norm(policy.min_sis_security_bits, family, d, log_basis)
        else {
            continue;
        };
        let Some(width_t) =
            decomposed_t_ring_count(a_key.row_len(), num_digits_open_val, num_blocks, 1)
        else {
            continue;
        };
        let Ok(b_key) = AjtaiKeyParams::try_new_with_min_rank(sis_key(policy, norm_t), width_t)
        else {
            continue;
        };
        let fold_linf_cap_config =
            FoldWitnessLinfCapConfig::for_fold_level(ring_challenge_cfg, fold_shape, d, width_s)?;
        let challenge = FoldChallengeNorms {
            infinity_norm: fold_shape.effective_infinity_norm(ring_challenge_cfg) as u128,
            l1_norm: fold_shape.effective_l1_mass(ring_challenge_cfg) as u128,
        };
        let (num_digits_fold_one, _) = fold_witness_digit_plan(
            r_vars,
            1,
            policy.decomposition.field_bits(),
            log_basis,
            challenge,
            FoldWitnessNorms::new(log_basis, d, 1, false),
            &fold_linf_cap_config,
        )?;
        let layout = PrecommittedGroupParams {
            group: PolynomialGroupLayout::singleton(prefix_num_vars),
            m_vars,
            r_vars,
            log_basis,
            n_a: a_key.row_len(),
            conservative_n_b: b_key.row_len(),
        };
        let params = PrecommittedLevelParams {
            layout,
            a_key,
            b_key,
            num_blocks,
            block_len,
            num_digits_commit,
            num_digits_open: num_digits_open_val,
            num_digits_fold_one,
        };
        let score = grouped_segment_rings(
            1,
            num_blocks,
            block_len,
            params.a_key.row_len(),
            num_digits_commit,
            num_digits_open_val,
            num_digits_fold_one,
        )?;
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
fn derive_candidate_level_params(
    policy: &PlannerPolicy,
    ring_challenge_cfg: &akita_challenges::SparseChallengeConfig,
    current_witness_len: usize,
    log_basis: u32,
    fold_level: usize,
    incoming_setup_prefix: Option<usize>,
) -> Result<Option<(LevelParams, usize, usize)>, AkitaError> {
    // Chunk count of the witness this level commits/produces (sized below as
    // `next_witness_len`). Equal for the metadata field and the width pricing so
    // a future verifier recomputing the size from `witness_chunk` agrees.
    let num_chunks = policy.chunks_at_level(fold_level);
    if !current_witness_len.is_multiple_of(policy.ring_dimension) {
        return Ok(None);
    }
    let num_ring_elems = current_witness_len / policy.ring_dimension;
    let reduced_vars = num_ring_elems.next_power_of_two().max(1).trailing_zeros() as usize;

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
                &ring_challenge_cfg,
                TensorChallengeShape::Flat,
                log_basis,
                n_prefix,
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

    let mut best: Option<(LevelParams, usize, usize)> = None;
    for r in (1..reduced_vars).rev() {
        let Some(candidate_params) = recursive_fold_level_params_candidate(
            policy,
            ring_challenge_cfg,
            num_ring_elems,
            reduced_vars,
            log_basis,
            fold_level,
            r,
        )?
        else {
            continue;
        };
        let mut candidate_params = candidate_params;
        candidate_params.setup_prefix = setup_prefix.clone();
        candidate_params.setup_contribution_mode = SetupContributionMode::Direct;
        let next_witness_len = planned_next_witness_len(
            policy.decomposition.field_bits(),
            &candidate_params,
            1,
            RelationMatrixRowLayout::WithDBlock,
            num_chunks,
        )?;
        let next_witness_len_terminal = planned_next_witness_len(
            policy.decomposition.field_bits(),
            &candidate_params,
            1,
            RelationMatrixRowLayout::WithoutDBlock,
            num_chunks,
        )?;

        if best.as_ref().is_none_or(|(_, c, _)| next_witness_len < *c) {
            best = Some((
                candidate_params,
                next_witness_len,
                next_witness_len_terminal,
            ));
        }
    }

    let Some((candidate_params, next_witness_len, next_witness_len_terminal)) = best else {
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

/// A `Step::Fold`-first suffix schedule.
///
/// The parent's proof-size formula needs the child's first fold params
/// (`first_fold_params`), so the suffix carries it directly instead of
/// re-matching `steps[0]`.
#[derive(Clone)]
pub(crate) struct FoldSuffix {
    pub(crate) total_bytes: usize,
    pub(crate) first_fold_params: LevelParams,
    pub(crate) steps: Vec<Step>,
}

/// Best direct suffix at one DP state: witness length only. The terminal
/// `DirectStep` is materialized at stitch time from the predecessor fold's
/// committed `LevelParams`.
#[derive(Clone, Copy)]
pub(crate) struct DirectSuffix {
    pub(crate) current_w_len: usize,
}

/// Result of the suffix DP at one state. Both shape options are reported
/// because the parent's proof-size formula depends on the child's first
/// step:
///
/// - `best_direct` — best no-outgoing-prefix terminal schedule whose first
///   step is a `Step::Direct` (parent scores under
///   `RelationMatrixRowLayout::WithoutDBlock`). It ignores
///   `incoming_setup_prefix`, because a direct child means the parent did not
///   offload a new setup prefix into that child.
/// - `best_fold_per_lb` — best `Step::Fold`-first schedule per first-fold
///   `log_basis`, consuming `incoming_setup_prefix` when one is present.
#[derive(Clone)]
pub(crate) struct SuffixResult {
    pub(crate) best_direct: Option<DirectSuffix>,
    pub(crate) best_fold_per_lb: BTreeMap<u32, FoldSuffix>,
}

impl SuffixResult {
    pub(crate) fn is_empty(&self) -> bool {
        self.best_direct.is_none() && self.best_fold_per_lb.is_empty()
    }
}

fn make_terminal_direct_step(
    current_w_len: usize,
    terminal_lp: &LevelParams,
    field_bits: u32,
    num_polynomials: usize,
    opening_layout: Option<&OpeningClaimsLayout>,
) -> Result<DirectStep, AkitaError> {
    // The terminal-direct (cleartext) witness is single-chunk by construction:
    // the prover emits the global folded response and one shared `r̂` tail, so
    // chunking the cleartext tail is unsupported. The last fold level must be
    // single-chunk (only the leading activated levels are chunked). Reject here
    // to match `resolve.rs` and avoid a cryptic prover-side layout mismatch.
    if terminal_lp.witness_chunk.num_chunks > 1 {
        return Err(AkitaError::InvalidSetup(
            "terminal-direct witness does not support a multi-chunk last fold level".to_string(),
        ));
    }
    let witness_shape = match opening_layout {
        Some(layout) => terminal_witness_shape_for_opening_layout(terminal_lp, field_bits, layout)?,
        None => segment_typed_witness_shape_from_groups(
            terminal_lp,
            field_bits,
            [(
                terminal_lp as &dyn akita_types::LevelParamsLike,
                num_polynomials,
                num_polynomials,
                1,
            )],
            1,
        )?,
    };
    let direct_bytes = direct_witness_bytes(field_bits, &witness_shape);
    Ok(DirectStep {
        current_w_len,
        witness_shape,
        direct_bytes,
        params: None,
    })
}

/// Like [`terminal_direct_suffix_cost`], but returns `None` when the fold at
/// `terminal_fold_level` is multi-chunk. The suffix DP uses this to skip the
/// fold-then-direct branch without aborting fold-then-fold exploration.
fn try_terminal_direct_suffix_cost(
    current_w_len: usize,
    terminal_lp: &LevelParams,
    field_bits: u32,
    key: PolynomialGroupLayout,
    terminal_fold_level: usize,
    opening_layout: Option<&OpeningClaimsLayout>,
) -> Result<Option<(DirectStep, usize)>, AkitaError> {
    if terminal_lp.witness_chunk.num_chunks > 1 {
        return Ok(None);
    }
    let (direct, direct_bytes) = terminal_direct_suffix_cost(
        current_w_len,
        terminal_lp,
        field_bits,
        key,
        terminal_fold_level,
        opening_layout,
    )?;
    Ok(Some((direct, direct_bytes)))
}

pub(crate) fn terminal_direct_suffix_cost(
    current_w_len: usize,
    terminal_lp: &LevelParams,
    field_bits: u32,
    key: PolynomialGroupLayout,
    terminal_fold_level: usize,
    opening_layout: Option<&OpeningClaimsLayout>,
) -> Result<(DirectStep, usize), AkitaError> {
    // Scalar same-point root fold: polynomial count at the root, 1 recursively.
    let num_polynomials = if terminal_fold_level == 0 {
        key.num_polynomials()
    } else {
        1
    };
    let direct = make_terminal_direct_step(
        current_w_len,
        terminal_lp,
        field_bits,
        num_polynomials,
        opening_layout,
    )?;
    let direct_bytes = direct.direct_bytes;
    Ok((direct, direct_bytes))
}

pub(crate) type ScheduleMemo = HashMap<(usize, usize, usize, u32, usize), SuffixResult>;

/// DP-invariant inputs for the suffix search.
///
/// `policy`, `ring_challenge_cfg`, and `num_vars` are constant across the whole
/// recursion, so they are carried in one context value rather than as
/// per-call arguments (keeps the recursive signature small).
#[derive(Clone, Copy)]
pub(crate) struct SuffixCtx<'a> {
    pub(crate) policy: &'a PlannerPolicy,
    pub(crate) ring_challenge_cfg: &'a akita_challenges::SparseChallengeConfig,
    pub(crate) num_vars: usize,
    pub(crate) key: PolynomialGroupLayout,
}

#[derive(Clone, Copy)]
pub(crate) struct SuffixState {
    pub(crate) level: usize,
    pub(crate) current_witness_len: usize,
    pub(crate) current_witness_len_terminal: usize,
    pub(crate) current_lb: u32,
    pub(crate) incoming_setup_prefix: Option<usize>,
}

impl SuffixState {
    fn memo_key(self) -> (usize, usize, usize, u32, usize) {
        (
            self.level,
            self.current_witness_len,
            self.current_witness_len_terminal,
            self.current_lb,
            self.incoming_setup_prefix.unwrap_or(0),
        )
    }
}

/// Shared inputs for root-level `LevelParams` candidates.
/// Suffix DP for the optimal recursive schedule at
/// `(level, current_witness_len, current_witness_len_terminal, current_lb)`.
///
/// Two witness lengths are carried because the shape leaving a fold
/// depends on its successor: `current_witness_len` is the `Intermediate` shape
/// (used if level `L` folds again) and `current_witness_len_terminal` is the
/// `Terminal` shape (used if level `L` sends the witness directly — drops
/// the D-block and zk D-blinding, so it is `<= current_witness_len`).
///
/// At each state: `best_direct` ships the witness directly without consuming
/// or forwarding an incoming prefix; `best_fold` keeps one fold candidate per
/// `log_basis` (from [`derive_candidate_level_params`]) and consumes
/// `incoming_setup_prefix` when present.
pub(crate) fn derive_optimal_suffix_schedule(
    ctx: &SuffixCtx<'_>,
    memo: &mut ScheduleMemo,
    state: SuffixState,
    depth: usize,
) -> Result<SuffixResult, AkitaError> {
    let SuffixCtx {
        policy,
        ring_challenge_cfg,
        num_vars,
        key,
    } = *ctx;
    let SuffixState {
        level,
        current_witness_len,
        current_witness_len_terminal,
        current_lb,
        incoming_setup_prefix,
    } = state;
    let memo_key = state.memo_key();
    if depth <= MAX_RECURSION_DEPTH {
        if let Some(cached) = memo.get(&memo_key) {
            return Ok(cached.clone());
        }
    }

    let best_direct = if derive_candidate_level_params(
        policy,
        ring_challenge_cfg,
        current_witness_len,
        current_lb,
        level,
        None,
    )?
    .is_some()
    {
        Some(DirectSuffix {
            current_w_len: current_witness_len_terminal,
        })
    } else {
        None
    };

    if depth > MAX_RECURSION_DEPTH {
        let result = SuffixResult {
            best_direct,
            best_fold_per_lb: BTreeMap::new(),
        };
        memo.insert(memo_key, result.clone());
        return Ok(result);
    }

    let mut best_fold_per_lb: BTreeMap<u32, FoldSuffix> = BTreeMap::new();
    let (min_log_basis, max_log_basis) = policy.basis_range;
    for lb in min_log_basis..=max_log_basis {
        if lb < current_lb {
            continue;
        }
        let Some((candidate_params, next_witness_len, next_witness_len_terminal)) =
            derive_candidate_level_params(
                policy,
                ring_challenge_cfg,
                current_witness_len,
                lb,
                level,
                incoming_setup_prefix,
            )?
        else {
            continue;
        };
        let Ok(eor_bytes) = extension_opening_reduction_level_bytes(
            policy.decomposition.field_bits() * policy.chal_ext_degree as u32,
            policy.claim_ext_degree,
            level,
            PolynomialGroupLayout::singleton(num_vars),
            current_witness_len,
        ) else {
            continue;
        };

        let mut best_for_this_lb: Option<(usize, Vec<Step>)> = None;
        let try_update = |total: usize, steps: Vec<Step>, slot: &mut Option<(usize, Vec<Step>)>| {
            if slot.as_ref().map(|(c, _)| total < *c).unwrap_or(true) {
                *slot = Some((total, steps));
            }
        };

        let current_opening_layout =
            suffix_opening_layout(current_witness_len, incoming_setup_prefix)?;
        let natural_len = active_setup_field_len(
            &candidate_params,
            &current_opening_layout,
            SETUP_OFFLOAD_D_SETUP,
        )?;
        let n_prefix = padded_setup_prefix_len(natural_len);
        let must_recurse = policy.recursive_setup_planning
            && level <= 1
            && n_prefix > SETUP_OFFLOAD_MIN_PREFIX_FIELD_LEN;
        let child_incoming_setup_prefix = must_recurse.then_some(natural_len);

        let child_suffix = derive_optimal_suffix_schedule(
            ctx,
            memo,
            SuffixState {
                level: level + 1,
                current_witness_len: next_witness_len,
                current_witness_len_terminal: next_witness_len_terminal,
                current_lb: lb,
                incoming_setup_prefix: child_incoming_setup_prefix,
            },
            depth + 1,
        )?;

        // Branch A: suffix is a Direct at level+1. This is the no-offload
        // terminal alternative even when `child_suffix` was computed with a
        // hypothetical incoming prefix for fold-first alternatives.
        if let Some(direct_suffix) = child_suffix.best_direct {
            let field_bits = policy.decomposition.field_bits();
            let terminal_opening_layout = incoming_setup_prefix
                .map(|_| suffix_opening_layout(current_witness_len, incoming_setup_prefix))
                .transpose()?;
            if let Some((direct_step, suffix_cost)) = try_terminal_direct_suffix_cost(
                direct_suffix.current_w_len,
                &candidate_params,
                field_bits,
                key,
                level,
                terminal_opening_layout.as_ref(),
            )? {
                let level_proof_size = level_proof_bytes(
                    field_bits,
                    field_bits * policy.chal_ext_degree as u32,
                    &candidate_params,
                    None,
                    next_witness_len_terminal,
                    1,
                    RelationMatrixRowLayout::WithoutDBlock,
                ) + eor_bytes;
                let total = level_proof_size + suffix_cost;
                let steps = vec![
                    Step::Fold(FoldStep {
                        params: candidate_params.clone(),
                        current_w_len: current_witness_len,
                        next_w_len: next_witness_len_terminal,
                        level_bytes: level_proof_size,
                    }),
                    Step::Direct(direct_step),
                ];
                try_update(total, steps, &mut best_for_this_lb);
            }
        }
        // Branch B: suffix is a Fold at level+1.
        let mut fold_candidate_params = candidate_params.clone();
        fold_candidate_params.setup_contribution_mode = if must_recurse {
            SetupContributionMode::Recursive
        } else {
            SetupContributionMode::Direct
        };
        for suffix_fold in child_suffix.best_fold_per_lb.values() {
            let level_proof_size = level_proof_bytes(
                policy.decomposition.field_bits(),
                policy.decomposition.field_bits() * policy.chal_ext_degree as u32,
                &fold_candidate_params,
                Some(&suffix_fold.first_fold_params),
                next_witness_len,
                1,
                RelationMatrixRowLayout::WithDBlock,
            ) + eor_bytes;
            let total = level_proof_size + suffix_fold.total_bytes;
            let mut steps = Vec::with_capacity(1 + suffix_fold.steps.len());
            steps.push(Step::Fold(FoldStep {
                params: fold_candidate_params.clone(),
                current_w_len: current_witness_len,
                next_w_len: next_witness_len,
                level_bytes: level_proof_size,
            }));
            steps.extend(suffix_fold.steps.iter().cloned());
            try_update(total, steps, &mut best_for_this_lb);
        }

        if let Some((total_bytes, steps)) = best_for_this_lb {
            let first_fold_params = steps
                .first()
                .and_then(|step| match step {
                    Step::Fold(fold) => Some(fold.params.clone()),
                    Step::Direct(_) => None,
                })
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("fold suffix missing first fold params".to_string())
                })?;
            best_fold_per_lb.insert(
                lb,
                FoldSuffix {
                    total_bytes,
                    first_fold_params,
                    steps,
                },
            );
        }
    }

    let result = SuffixResult {
        best_direct,
        best_fold_per_lb,
    };
    memo.insert(memo_key, result.clone());
    Ok(result)
}

/// Brute-forced root-direct commit `LevelParams` (optimal `(m, r)` split).
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
/// - `(m_vars, r_vars)` — `optimal_m_r_split` for a normal root, or `(0, 0)`
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
/// shape; the `(m, r)` split itself is scored against the flat L1 mass).
///
/// Returns `Ok(None)` when any SIS-floor lookup or bound arithmetic rejects
/// the candidate (the uncommittable edge), matching the previous
/// `Result::ok()` fallback.
pub(crate) fn compute_root_direct_level_params(
    policy: &PlannerPolicy,
    ring_challenge_cfg: &akita_challenges::SparseChallengeConfig,
    num_vars: usize,
    log_basis: u32,
    fold_challenge_shape: TensorChallengeShape,
    num_claims: usize,
) -> Result<Option<LevelParams>, AkitaError> {
    let d = policy.ring_dimension;
    let sis_family = policy.sis_family;
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
    // optimizer recomputes the fold-priced A collision per `r` internally
    // (it grows with the fold arity `num_claims · 2^r`), so it needs the
    // batch factor and ring-subfield norm, not a single pre-baked bucket.
    let (m_vars, r_vars) = if num_vars > alpha {
        // The `(m, r)` split is scored against the flat L1 mass (the root fold
        // shape disambiguates the committed table, not the split search).
        let fold_challenge = akita_types::sis::FoldChallengeNorms::new(
            ring_challenge_cfg,
            TensorChallengeShape::Flat,
        );
        // One-hot root commits a sparse witness (`||s||_inf = 1`,
        // `nonzeros = ceil(D/K)`); dense roots use the balanced-digit norms.
        let is_onehot = decomp.log_commit_bound == 1;
        let fold_witness = FoldWitnessNorms::new(log_basis, d, policy.onehot_chunk_size, is_onehot);
        let (m_vars, r_vars, _scoring_n_a) = optimal_m_r_split(
            policy.min_sis_security_bits,
            sis_family,
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
        (m_vars, r_vars)
    } else {
        (0, 0)
    };

    let Some(num_blocks) = 1usize.checked_shl(r_vars as u32) else {
        return Ok(None);
    };
    let Some(block_len) = 1usize.checked_shl(m_vars as u32) else {
        return Ok(None);
    };

    // The A/B/D keys, composed from the `akita_types::sis` primitives:
    // norm -> width -> tight SIS-secure rank -> key. `t_vectors = num_claims`
    // folds the batched-root scaling into the B/D widths (the root commits
    // `num_claims` polynomials) — no separate per-claim-then-scale pass.
    let Some(width_s) = decomposed_s_block_ring_count(block_len, depth_commit) else {
        return Ok(None);
    };
    let Some(norm_s) = rounded_up_role_a_inf_norm(
        policy.min_sis_security_bits,
        sis_family,
        d,
        level_decomp,
        ring_challenge_cfg,
        fold_challenge_shape,
        true,
        policy.onehot_chunk_size,
        policy.ring_subfield_norm_bound,
        r_vars,
        num_claims,
        width_s as u64,
    ) else {
        return Ok(None);
    };
    let Ok(a_key) = AjtaiKeyParams::try_new_with_min_rank(sis_key(policy, norm_s), width_s) else {
        return Ok(None);
    };
    let n_a = a_key.row_len();
    let Some(norm_t) =
        rounded_up_collision_inf_norm(policy.min_sis_security_bits, sis_family, d, log_basis)
    else {
        return Ok(None);
    };
    let Some(width_t) = decomposed_t_ring_count(n_a, depth_open, num_blocks, num_claims) else {
        return Ok(None);
    };
    let Ok(b_key) = AjtaiKeyParams::try_new_with_min_rank(sis_key(policy, norm_t), width_t) else {
        return Ok(None);
    };
    let Some(norm_w) =
        rounded_up_collision_inf_norm(policy.min_sis_security_bits, sis_family, d, log_basis)
    else {
        return Ok(None);
    };
    let Some(width_w) = decomposed_w_ring_count(depth_open, num_blocks, num_claims) else {
        return Ok(None);
    };
    let Ok(d_key) = AjtaiKeyParams::try_new_with_min_rank(sis_key(policy, norm_w), width_w) else {
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
        num_blocks,
        block_len,
        m_vars,
        r_vars,
        fold_challenge_config: *ring_challenge_cfg,
        fold_challenge_shape,
        num_digits_commit: depth_commit,
        num_digits_open: depth_open,
        onehot_chunk_size,
        fold_linf_cap_config: FoldWitnessLinfCapConfig::worst_case_beta_only(),
        num_digits_fold_one: 1,
        field_bits_hint: 0,
        cached_num_digits_fold_claims: 0,
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
    r_vars: usize,
    fold_challenge_shape: TensorChallengeShape,
) -> Result<Option<LevelParams>, AkitaError> {
    let alpha = (policy.ring_dimension as u32).trailing_zeros() as usize;
    let reduced_vars = num_vars.saturating_sub(alpha);
    if reduced_vars == 0 || r_vars >= reduced_vars {
        return Ok(None);
    }
    let num_blocks = 1usize.checked_shl(r_vars as u32).ok_or_else(|| {
        AkitaError::InvalidSetup("root candidate num_blocks overflow".to_string())
    })?;
    let root_num_chunks = policy.chunks_at_level(0);
    if root_num_chunks > 1 && !num_blocks.is_multiple_of(root_num_chunks) {
        return Ok(None);
    }
    let m_vars = reduced_vars - r_vars;
    let block_len = 1usize
        .checked_shl(m_vars as u32)
        .ok_or_else(|| AkitaError::InvalidSetup("root candidate block_len overflow".to_string()))?;
    let level_decomp = DecompositionParams {
        log_basis,
        ..policy.decomposition
    };
    let num_digits_commit = num_digits_s_commit(level_decomp, true);
    let num_digits_open = num_digits_open(level_decomp);
    let Some(width_s) = decomposed_s_block_ring_count(block_len, num_digits_commit) else {
        return Ok(None);
    };
    let Some(norm_s) = rounded_up_role_a_inf_norm(
        policy.min_sis_security_bits,
        policy.sis_family,
        policy.ring_dimension,
        level_decomp,
        ring_challenge_cfg,
        fold_challenge_shape,
        true,
        policy.onehot_chunk_size,
        policy.ring_subfield_norm_bound,
        r_vars,
        num_claims,
        width_s as u64,
    ) else {
        return Ok(None);
    };
    let Ok(a_key) = AjtaiKeyParams::try_new_with_min_rank(sis_key(policy, norm_s), width_s) else {
        return Ok(None);
    };
    let Some(norm_t) = rounded_up_collision_inf_norm(
        policy.min_sis_security_bits,
        policy.sis_family,
        policy.ring_dimension,
        log_basis,
    ) else {
        return Ok(None);
    };
    let Some(width_t) =
        decomposed_t_ring_count(a_key.row_len(), num_digits_open, num_blocks, num_claims)
    else {
        return Ok(None);
    };
    let Ok(b_key) = AjtaiKeyParams::try_new_with_min_rank(sis_key(policy, norm_t), width_t) else {
        return Ok(None);
    };
    let Some(width_w) = decomposed_w_ring_count(num_digits_open, num_blocks, num_claims) else {
        return Ok(None);
    };
    let Ok(d_key) = AjtaiKeyParams::try_new_with_min_rank(sis_key(policy, norm_t), width_w) else {
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
        num_blocks,
        block_len,
        m_vars,
        r_vars,
        fold_challenge_config: *ring_challenge_cfg,
        fold_challenge_shape,
        num_digits_commit,
        num_digits_open,
        onehot_chunk_size,
        fold_linf_cap_config: FoldWitnessLinfCapConfig::worst_case_beta_only(),
        num_digits_fold_one: 1,
        field_bits_hint: 0,
        cached_num_digits_fold_claims: 0,
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

/// Find the optimal schedule for a root schedule lookup key under `policy`.
///
/// Runs an exhaustive DP that minimizes proof size. The result is a pure,
/// deterministic function of `(policy, key)` (plus the `ring_challenge_config` /
/// `fold_challenge_shape_at_level` closures, which presets derive from the same hooks the
/// generated tables were emitted from), so the prover and verifier
/// regenerate identical schedules on a table miss.
///
/// # Errors
///
/// Returns an error if vector counts are invalid or if the witness length
/// overflows. The function never panics on malformed input — it is
/// verifier-reachable and audited under the no-panic contract.
pub fn find_schedule(
    key: PolynomialGroupLayout,
    policy: &PlannerPolicy,
    ring_challenge_config: impl Fn(usize) -> Result<akita_challenges::SparseChallengeConfig, AkitaError>,
    fold_challenge_shape_at_level: impl Fn(AkitaScheduleInputs) -> TensorChallengeShape,
) -> Result<Schedule, AkitaError> {
    find_schedule_inner(
        key,
        policy,
        ring_challenge_config,
        fold_challenge_shape_at_level,
    )
}

fn find_schedule_inner(
    key: PolynomialGroupLayout,
    policy: &PlannerPolicy,
    ring_challenge_config: impl Fn(usize) -> Result<akita_challenges::SparseChallengeConfig, AkitaError>,
    fold_challenge_shape_at_level: impl Fn(AkitaScheduleInputs) -> TensorChallengeShape,
) -> Result<Schedule, AkitaError> {
    let ring_challenge_config: RingChallengeConfigFn<'_> = &ring_challenge_config;
    let fold_shape: FoldShapeFn<'_> = &fold_challenge_shape_at_level;

    key.validate()?;
    validate_policy_witness_chunk(policy)?;
    let ring_challenge_cfg = ring_challenge_config(policy.ring_dimension)?;
    let suffix_ctx = SuffixCtx {
        policy,
        ring_challenge_cfg: &ring_challenge_cfg,
        num_vars: key.num_vars(),
        key,
    };

    key.validate()?;
    validate_policy_witness_chunk(policy)?;
    if policy.recursive_setup_planning {
        return Err(AkitaError::InvalidSetup(
            "recursive setup planning requires the grouped-batch scheduler".to_string(),
        ));
    }
    let root_inputs = AkitaScheduleInputs::for_root(key)?;
    let witness_len = root_inputs.current_w_len;

    let field_bits = policy.decomposition.field_bits();

    let root_witness_shape = CleartextWitnessShape::FieldElements(witness_len);
    let mut best_cost = direct_witness_bytes(field_bits, &root_witness_shape);
    let fold_challenge_shape = fold_shape(root_inputs);
    // The level-0 fold-challenge shape and the `num_claims = num_polynomials`
    // batch factor are folded directly into the committed B/D widths, so a table
    // miss reproduces the exact root commit layout the table-hit expansion
    // (`expand_to_level_params`) builds — no separate per-claim-then-scale
    // pass. `Ok(None)` is the uncommittable (large-`num_vars`) edge.
    let root_direct_commit_params = compute_root_direct_level_params(
        policy,
        &ring_challenge_cfg,
        key.num_vars(),
        policy.decomposition.log_basis,
        fold_challenge_shape,
        key.num_polynomials(),
    )?;
    let mut best_steps: Vec<Step> = vec![Step::Direct(DirectStep {
        current_w_len: witness_len,
        witness_shape: root_witness_shape,
        direct_bytes: best_cost,
        params: root_direct_commit_params,
    })];
    let mut memo = ScheduleMemo::new();

    let alpha = (policy.ring_dimension as u32).trailing_zeros() as usize;
    let reduced_vars = key.num_vars().saturating_sub(alpha);

    if reduced_vars == 0 {
        return Ok(Schedule {
            steps: best_steps,
            total_bytes: best_cost,
        });
    }

    let min_r_vars: usize = if reduced_vars >= 3 { 1 } else { 0 };
    let max_r_vars: usize = (reduced_vars - 1).min(usize::BITS as usize - 1);

    // Chunk count of the witness committed at the root fold (absolute level 0).
    let root_num_chunks = policy.chunks_at_level(0);

    let (min_log_basis, max_log_basis) = policy.basis_range;
    for candidate_log_basis in min_log_basis..=max_log_basis {
        for r_vars in (min_r_vars..=max_r_vars).rev() {
            let Some(candidate_params) = scalar_root_fold_level_params_candidate(
                policy,
                &ring_challenge_cfg,
                key.num_vars(),
                key.num_polynomials(),
                candidate_log_basis,
                r_vars,
                fold_challenge_shape,
            )?
            else {
                continue;
            };

            let next_withness_len_impl = |layout| -> Result<usize, AkitaError> {
                let rings = w_ring_element_count_for_chunks(
                    field_bits,
                    &candidate_params,
                    key.num_polynomials(),
                    layout,
                    root_num_chunks,
                )?;
                rings.checked_mul(policy.ring_dimension).ok_or_else(|| {
                    AkitaError::InvalidSetup("root next witness length overflow".into())
                })
            };
            let next_w_len = next_withness_len_impl(RelationMatrixRowLayout::WithDBlock)?;
            let next_w_len_terminal =
                next_withness_len_impl(RelationMatrixRowLayout::WithoutDBlock)?;
            let initial_witness_len_bits = witness_len
                .checked_mul(field_bits as usize)
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("root witness bit length overflow".into())
                })?;
            if next_w_len
                .checked_mul(candidate_log_basis as usize)
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("root next witness bit length overflow".into())
                })?
                >= initial_witness_len_bits
            {
                continue;
            }

            let suffix = derive_optimal_suffix_schedule(
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
            if suffix.is_empty() {
                continue;
            }
            let Ok(eor_bytes) = extension_opening_reduction_level_bytes(
                policy.decomposition.field_bits() * policy.chal_ext_degree as u32,
                policy.claim_ext_degree,
                0,
                key,
                witness_len,
            ) else {
                continue;
            };

            // Branch A: suffix at level 1 is a Direct
            if let Some(direct_suffix) = suffix.best_direct {
                if let Some((direct_step, suffix_cost)) = try_terminal_direct_suffix_cost(
                    direct_suffix.current_w_len,
                    &candidate_params,
                    field_bits,
                    key,
                    0,
                    None,
                )? {
                    let root_proof_size = level_proof_bytes(
                        field_bits,
                        field_bits * policy.chal_ext_degree as u32,
                        &candidate_params,
                        None,
                        next_w_len_terminal,
                        1,
                        RelationMatrixRowLayout::WithoutDBlock,
                    ) + eor_bytes;
                    let total = root_proof_size + suffix_cost;
                    if total < best_cost {
                        best_cost = total;
                        best_steps = vec![
                            Step::Fold(FoldStep {
                                params: candidate_params.clone(),
                                current_w_len: witness_len,
                                next_w_len: next_w_len_terminal,
                                level_bytes: root_proof_size,
                            }),
                            Step::Direct(direct_step),
                        ];
                    }
                }
            }
            // Branch B: suffix at level 1 is a Fold
            for suffix_fold in suffix.best_fold_per_lb.values() {
                let root_proof_size = level_proof_bytes(
                    field_bits,
                    field_bits * policy.chal_ext_degree as u32,
                    &candidate_params,
                    Some(&suffix_fold.first_fold_params),
                    next_w_len,
                    1,
                    RelationMatrixRowLayout::WithDBlock,
                ) + eor_bytes;
                let total = root_proof_size + suffix_fold.total_bytes;
                if total < best_cost {
                    best_cost = total;
                    let mut steps = Vec::with_capacity(1 + suffix_fold.steps.len());
                    steps.push(Step::Fold(FoldStep {
                        params: candidate_params.clone(),
                        current_w_len: witness_len,
                        next_w_len,
                        level_bytes: root_proof_size,
                    }));
                    steps.extend(suffix_fold.steps.iter().cloned());
                    best_steps = steps;
                }
            }
        }
    }

    Ok(Schedule {
        steps: best_steps,
        total_bytes: best_cost,
    })
}
