//! Test-only layout helpers shared by the workspace's integration tests,
//! unit tests, and the `profile` example.
//!
//! Everything in this module is gated behind the `test-support` Cargo
//! feature, which production builds never enable: it is switched on only
//! through the dev-dependency edge of `akita-pcs`, so the helpers here are
//! compiled for test/example/bench targets and are
//! absent from every shipped artifact. Production callers size their
//! per-poly inputs through [`CommitmentConfig::get_params_for_batched_commitment`]
//! directly and never need this module.

use akita_challenges::{SparseChallengeConfig, TensorChallengeShape};
use akita_field::AkitaError;
use akita_planner::generated::{table_entry, GeneratedFoldStep, GeneratedStep};
use akita_planner::{generated_schedule_lookup_key, PlannerPolicy};
use akita_types::sis::{
    committed_fold_a_role_rank, decomposed_s_block_ring_count, decomposed_t_ring_count,
    decomposed_w_ring_count, min_secure_rank, num_digits_open, num_digits_s_commit,
    rounded_up_collision_norm_t, rounded_up_collision_norm_w,
};
use akita_types::{
    direct_witness_bytes, level_proof_bytes, segment_typed_witness_shape,
    w_ring_element_count_with_counts_for_layout_bits, AjtaiKeyParams, AkitaScheduleInputs,
    AkitaScheduleLookupKey, CommitmentGroupScheduleKey, DecompositionParams, DirectStep, FoldStep,
    LevelParams, MRowLayout, OpeningBatchShape, Schedule, Step,
};

use crate::{policy_of, CommitmentConfig};

/// Derive the per-polynomial commitment layout optimized for a batch of
/// `num_polynomials` polynomials with `num_vars` variables.
///
/// First reads the runtime schedule (table hit or DP fallback). When the
/// schedule is a root fold it returns that root layout; for a direct-only
/// schedule it falls back to the batched root commit layout
/// `Cfg::get_params_for_batched_commitment` derives for the same
/// `num_polynomials` (so the fallback layout is sized for the requested batch,
/// not a singleton).
///
/// Tests, benches, and the `profile` example use this to pre-size per-poly
/// inputs (e.g. `OneHotPoly`) so the `block_len` / `num_blocks` line up with
/// what `Scheme::commit` will use under the batched layout. Production
/// callers always go through `Cfg::get_params_for_batched_commitment(&opening_batch)`
/// instead.
///
/// # Errors
///
/// Returns an error if the layout parameters overflow or are invalid.
pub fn akita_batched_root_layout<Cfg>(
    num_vars: usize,
    num_polynomials: usize,
) -> Result<LevelParams, AkitaError>
where
    Cfg: CommitmentConfig,
{
    let lookup_key = CommitmentGroupScheduleKey::new(num_vars, num_polynomials);
    let schedule = Cfg::runtime_schedule(AkitaScheduleLookupKey::single(lookup_key))?;
    if let Some(root) = akita_types::schedule_root_fold_step(&schedule) {
        let layout = root.params.clone();
        tracing::info!(
            num_vars,
            num_polynomials,
            total_bytes = schedule.total_bytes,
            root_m = layout.log_block_len(),
            root_r = layout.log_num_blocks(),
            root_lb = layout.log_basis,
            "batched root split: read from runtime schedule"
        );
        return Ok(layout);
    }
    tracing::info!(
        num_vars,
        num_polynomials,
        "batched root split: schedule is direct-only, falling back to config root layout"
    );
    // Size the fallback for the requested batch (`num_polynomials`), not a
    // singleton — otherwise the per-poly inputs would be smaller than the
    // batched commit layout `Scheme::commit` actually uses.
    Cfg::get_params_for_batched_commitment(&OpeningBatchShape::new(num_vars, num_polynomials)?)
}

// ---------------------------------------------------------------------------
// Mixed-D per-level schedule stitching (runtime ring cutover acceptance
// fixture). Test-only: production schedules stay uniform-D until the planner
// grows a DP `ring_d` search (specs/mixed-row-ring-dimensions.md).
// ---------------------------------------------------------------------------

struct MixedSuffixFoldPlan {
    params: LevelParams,
    current_w_len: usize,
    next_w_len: usize,
    is_terminal: bool,
}

fn generated_fold_step<Cfg: CommitmentConfig>(
    key: CommitmentGroupScheduleKey,
    level: usize,
) -> Result<GeneratedFoldStep, AkitaError> {
    let catalog = Cfg::schedule_catalog().ok_or_else(|| {
        AkitaError::InvalidSetup(format!(
            "{} missing generated schedule catalog",
            std::any::type_name::<Cfg>()
        ))
    })?;
    let table_key = generated_schedule_lookup_key(key);
    let entry = table_entry(catalog, table_key).ok_or_else(|| {
        AkitaError::InvalidSetup(format!("missing generated schedule for {key:?}"))
    })?;
    let mut fold_idx = 0usize;
    for step in entry.steps {
        if let GeneratedStep::Fold(fold) = step {
            if fold_idx == level {
                return Ok(*fold);
            }
            fold_idx += 1;
        }
    }
    Err(AkitaError::InvalidSetup(format!(
        "fold level {level} missing from {} table entry {key:?}",
        std::any::type_name::<Cfg>()
    )))
}

/// Expand a compact generated fold step's witness geometry at a different
/// active ring dimension.
///
/// Mixed-D hand schedules keep the envelope `current_w_len` / block geometry
/// while halving `ring_d`. Ranks are recomputed for the target geometry
/// instead of reusing the stored compact tuple from either table (the same
/// role-pricing primitives `GeneratedFoldStep::expand_to_level_params` audits
/// stored ranks against). `extra_block_vars` is the additional `r_vars` added
/// when the previous fold executed at a larger ring dimension (typically `1`
/// on the first suffix level after a `128 → 64` drop, `0` thereafter).
#[allow(clippy::too_many_arguments)]
fn expand_envelope_witness_at_ring_d(
    step: &GeneratedFoldStep,
    policy: &PlannerPolicy,
    ring_challenge_config: impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    fold_level: usize,
    envelope_current_w_len: usize,
    target_ring_d: usize,
    fold_shape: TensorChallengeShape,
    num_claims: usize,
    extra_block_vars: usize,
    block_m_vars: Option<usize>,
    block_r_vars: Option<usize>,
) -> Result<LevelParams, AkitaError> {
    if target_ring_d == 0 {
        return Err(AkitaError::InvalidSetup(
            "mixed-D target ring dimension must be nonzero".into(),
        ));
    }
    let is_root = fold_level == 0;
    let log_basis = step.log_basis;
    let sis_family = policy.sis_family;
    let m_vars = block_m_vars.unwrap_or(step.m_vars as usize);
    let r_vars = block_r_vars
        .unwrap_or(step.r_vars as usize)
        .checked_add(extra_block_vars)
        .ok_or_else(|| AkitaError::InvalidSetup("mixed-D block variable count overflow".into()))?;
    let num_blocks = 1usize.checked_shl(r_vars as u32).ok_or_else(|| {
        AkitaError::InvalidSetup("generated schedule 2^r_vars overflows usize".to_string())
    })?;
    let block_len = if is_root {
        1usize.checked_shl(m_vars as u32).ok_or_else(|| {
            AkitaError::InvalidSetup("generated schedule 2^m_vars overflows usize".to_string())
        })?
    } else {
        (envelope_current_w_len / target_ring_d).div_ceil(num_blocks)
    };
    let no_layout = |role: &str| {
        AkitaError::InvalidSetup(format!(
            "no audited {role}-role layout for mixed-D schedule \
             (family={sis_family:?}, d={target_ring_d}, log_basis={log_basis})"
        ))
    };
    let decomp = DecompositionParams {
        log_basis,
        ..policy.decomposition
    };
    let ring_challenge_cfg = ring_challenge_config(target_ring_d)?;
    let num_digits_commit = num_digits_s_commit(decomp, is_root);
    let num_digits_open_val = num_digits_open(decomp);
    let inner_width = decomposed_s_block_ring_count(block_len, num_digits_commit)
        .ok_or_else(|| no_layout("A"))?;
    let (a_bucket, n_a) = committed_fold_a_role_rank(
        sis_family,
        target_ring_d,
        decomp,
        &ring_challenge_cfg,
        fold_shape,
        is_root,
        policy.onehot_chunk_size,
        policy.ring_subfield_norm_bound,
        r_vars,
        num_claims,
        inner_width as u64,
    )
    .ok_or_else(|| no_layout("A"))?;
    let b_bucket = rounded_up_collision_norm_t(sis_family, target_ring_d, log_basis)
        .ok_or_else(|| no_layout("B"))?;
    let outer_width = decomposed_t_ring_count(n_a, num_digits_open_val, num_blocks, num_claims)
        .ok_or_else(|| no_layout("B"))?;
    let d_bucket = rounded_up_collision_norm_w(sis_family, target_ring_d, log_basis)
        .ok_or_else(|| no_layout("D"))?;
    let d_matrix_width = decomposed_w_ring_count(num_digits_open_val, num_blocks, num_claims)
        .ok_or_else(|| no_layout("D"))?;
    if step.tier_split.is_some() || step.n_f.is_some() {
        return Err(AkitaError::InvalidSetup(
            "mixed-D envelope witness expansion does not support tiered layouts".into(),
        ));
    }
    let n_b = min_secure_rank(
        sis_family,
        target_ring_d as u32,
        b_bucket,
        outer_width as u64,
    )
    .ok_or_else(|| no_layout("B"))?;
    let n_d = min_secure_rank(
        sis_family,
        target_ring_d as u32,
        d_bucket,
        d_matrix_width as u64,
    )
    .ok_or_else(|| no_layout("D"))?;
    let onehot_chunk_size = if is_root && policy.decomposition.log_commit_bound == 1 {
        policy.onehot_chunk_size
    } else {
        0
    };
    let params = LevelParams {
        ring_dimension: target_ring_d,
        log_basis,
        a_key: AjtaiKeyParams::try_new(sis_family, n_a, inner_width, a_bucket, target_ring_d)?,
        b_key: AjtaiKeyParams::try_new(sis_family, n_b, outer_width, b_bucket, target_ring_d)?,
        d_key: AjtaiKeyParams::try_new(sis_family, n_d, d_matrix_width, d_bucket, target_ring_d)?,
        num_blocks,
        block_len,
        m_vars,
        r_vars,
        stage1_config: ring_challenge_cfg,
        fold_challenge_shape: fold_shape,
        num_digits_commit,
        num_digits_open: num_digits_open_val,
        onehot_chunk_size,
        tier_split: 1,
        f_key: None,
        fold_linf_cap_config: akita_types::sis::FoldWitnessLinfCapConfig::worst_case_beta_only(),
        num_digits_fold_one: 1,
        field_bits_hint: 0,
        cached_num_digits_fold_claims: 0,
        cached_num_digits_fold_value: 1,
        precommitted_groups: Vec::new(),
    };
    params.with_fold_linf_cap_config(policy.decomposition.field_bits(), num_claims)
}

fn mixed_continue_suffix_level<EnvelopeCfg, SuffixCfg>(
    key: CommitmentGroupScheduleKey,
    level: usize,
    current_w_len: usize,
    suffix_ring_d: usize,
    block_m_vars: usize,
    block_r_vars: usize,
) -> Result<LevelParams, AkitaError>
where
    EnvelopeCfg: CommitmentConfig,
    SuffixCfg: CommitmentConfig,
{
    let envelope_gen = generated_fold_step::<EnvelopeCfg>(key, level)?;
    let suffix_policy = policy_of::<SuffixCfg>();
    let fold_shape = SuffixCfg::fold_challenge_shape_at_level(AkitaScheduleInputs {
        num_vars: key.num_vars,
        level,
        current_w_len,
    });
    expand_envelope_witness_at_ring_d(
        &envelope_gen,
        &suffix_policy,
        SuffixCfg::ring_challenge_config,
        level,
        current_w_len,
        suffix_ring_d,
        fold_shape,
        1,
        0,
        Some(block_m_vars),
        Some(block_r_vars),
    )
}

fn mixed_level_params<EnvelopeCfg, SuffixCfg>(
    key: CommitmentGroupScheduleKey,
    level: usize,
    envelope_current_w_len: usize,
    envelope_params: &LevelParams,
    suffix_ring_d: usize,
    prev_ring_d: usize,
) -> Result<LevelParams, AkitaError>
where
    EnvelopeCfg: CommitmentConfig,
    SuffixCfg: CommitmentConfig,
{
    let envelope_gen = generated_fold_step::<EnvelopeCfg>(key, level)?;
    let extra_block_vars = extra_block_vars_for_drop(prev_ring_d, suffix_ring_d);
    if envelope_gen.ring_d as usize == suffix_ring_d && extra_block_vars == 0 {
        return Ok(envelope_params.clone());
    }
    let suffix_policy = policy_of::<SuffixCfg>();
    let num_claims = if level == 0 { key.num_polynomials } else { 1 };
    let fold_shape = EnvelopeCfg::fold_challenge_shape_at_level(AkitaScheduleInputs {
        num_vars: key.num_vars,
        level,
        current_w_len: envelope_current_w_len,
    });
    expand_envelope_witness_at_ring_d(
        &envelope_gen,
        &suffix_policy,
        SuffixCfg::ring_challenge_config,
        level,
        envelope_current_w_len,
        suffix_ring_d,
        fold_shape,
        num_claims,
        extra_block_vars,
        None,
        None,
    )
}

/// Extra block-select variables when dropping ring dimension by a
/// power-of-two factor.
pub fn extra_block_vars_for_drop(prev_ring_d: usize, suffix_ring_d: usize) -> usize {
    if prev_ring_d > suffix_ring_d && prev_ring_d.is_multiple_of(suffix_ring_d) {
        let downscale = prev_ring_d / suffix_ring_d;
        if downscale.is_power_of_two() {
            return downscale.trailing_zeros() as usize;
        }
    }
    0
}

/// Hand-built mixed-D schedule for the runtime ring cutover acceptance
/// fixture.
///
/// Fold levels `[0, switch_at_fold)` use [`EnvelopeCfg`]'s shipped table
/// unchanged. Levels `[switch_at_fold, …)` keep the envelope witness-length
/// chain while dropping the per-level ring dimension to [`SuffixCfg`]'s
/// (typically `128 → 64`): block counts and Ajtai ranks are recomputed for
/// the suffix geometry so `commit_next_w` dispatch at the suffix `D` matches
/// the envelope field-length ladder.
///
/// The fixture supports `num_polynomials == 1` only.
///
/// # Errors
///
/// Returns an error when either preset schedule cannot be resolved, the
/// suffix table has fewer fold levels than the envelope, or `switch_at_fold`
/// is out of range.
pub fn mixed_d_per_level_schedule<EnvelopeCfg, SuffixCfg>(
    num_vars: usize,
    num_polynomials: usize,
    switch_at_fold: usize,
) -> Result<Schedule, AkitaError>
where
    EnvelopeCfg: CommitmentConfig,
    SuffixCfg: CommitmentConfig,
{
    if num_polynomials != 1 {
        return Err(AkitaError::InvalidSetup(format!(
            "mixed-D fixture supports singleton batches only (got {num_polynomials})"
        )));
    }
    let lookup_key =
        AkitaScheduleLookupKey::single(CommitmentGroupScheduleKey::new(num_vars, num_polynomials));
    let envelope = EnvelopeCfg::runtime_schedule(lookup_key.clone())?;
    let suffix = SuffixCfg::runtime_schedule(lookup_key.clone())?;

    let envelope_folds: Vec<FoldStep> = envelope.fold_steps().cloned().collect();
    let suffix_folds: Vec<FoldStep> = suffix.fold_steps().cloned().collect();
    if envelope_folds.is_empty() {
        return Err(AkitaError::InvalidSetup(
            "mixed-D fixture requires a folded schedule".into(),
        ));
    }
    if suffix_folds.len() < envelope_folds.len() {
        return Err(AkitaError::InvalidSetup(format!(
            "suffix schedule has fewer fold levels than the envelope: {} < {}",
            suffix_folds.len(),
            envelope_folds.len()
        )));
    }
    if switch_at_fold > envelope_folds.len() {
        return Err(AkitaError::InvalidSetup(format!(
            "switch_at_fold={switch_at_fold} exceeds fold count {}",
            envelope_folds.len()
        )));
    }
    if switch_at_fold == 0 {
        return Err(AkitaError::InvalidSetup(
            "switch_at_fold=0 is unsupported; use switch_at_fold >= 1 for mixed-D fixtures".into(),
        ));
    }

    let mut mixed_folds: Vec<FoldStep> = envelope_folds
        .iter()
        .take(switch_at_fold)
        .cloned()
        .collect();

    if switch_at_fold < envelope_folds.len() {
        let suffix_policy = policy_of::<SuffixCfg>();
        let field_bits = suffix_policy.decomposition.field_bits();
        let challenge_field_bits = field_bits * suffix_policy.chal_ext_degree as u32;
        let suffix_ring_d = suffix_folds[switch_at_fold].params.ring_dimension;
        let num_fold_levels = envelope_folds.len();
        let mut w_len = envelope_folds[switch_at_fold - 1].next_w_len;
        let mut prev_ring_d = envelope_folds[switch_at_fold - 1].params.ring_dimension;
        let mut suffix_plan: Vec<MixedSuffixFoldPlan> = Vec::new();

        for (level, envelope_step) in envelope_folds.iter().enumerate().skip(switch_at_fold) {
            let params = if level == switch_at_fold {
                mixed_level_params::<EnvelopeCfg, SuffixCfg>(
                    lookup_key.final_group,
                    level,
                    w_len,
                    &envelope_step.params,
                    suffix_ring_d,
                    prev_ring_d,
                )?
            } else {
                let prev = mixed_folds
                    .last()
                    .expect("mixed suffix fold chain must be nonempty");
                mixed_continue_suffix_level::<EnvelopeCfg, SuffixCfg>(
                    lookup_key.final_group,
                    level,
                    w_len,
                    suffix_ring_d,
                    prev.params.m_vars,
                    prev.params.r_vars,
                )?
            };
            let is_terminal_fold = level + 1 == num_fold_levels;
            let layout = if is_terminal_fold {
                MRowLayout::WithoutDBlock
            } else {
                MRowLayout::WithDBlock
            };
            let ring = w_ring_element_count_with_counts_for_layout_bits(
                field_bits, &params, 1, 1, layout,
            )?;
            let next_w_len = ring.checked_mul(params.ring_dimension).ok_or_else(|| {
                AkitaError::InvalidSetup("mixed-D witness length overflow".into())
            })?;
            suffix_plan.push(MixedSuffixFoldPlan {
                params,
                current_w_len: w_len,
                next_w_len,
                is_terminal: is_terminal_fold,
            });
            w_len = next_w_len;
            prev_ring_d = suffix_ring_d;
        }

        for (idx, plan) in suffix_plan.iter().enumerate() {
            let layout = if plan.is_terminal {
                MRowLayout::WithoutDBlock
            } else {
                MRowLayout::WithDBlock
            };
            let next_lp = if plan.is_terminal {
                None
            } else {
                Some(&suffix_plan[idx + 1].params)
            };
            let level_bytes = level_proof_bytes(
                field_bits,
                challenge_field_bits,
                &plan.params,
                next_lp,
                plan.next_w_len,
                1,
                layout,
            );
            mixed_folds.push(FoldStep {
                params: plan.params.clone(),
                current_w_len: plan.current_w_len,
                next_w_len: plan.next_w_len,
                level_bytes,
            });
        }
    }

    let envelope_terminal = match envelope.steps.last() {
        Some(Step::Direct(step)) => step,
        _ => {
            return Err(AkitaError::InvalidSetup(
                "envelope schedule must end in a direct witness step".into(),
            ));
        }
    };
    let needs_terminal_override = mixed_folds
        .iter()
        .zip(envelope_folds.iter())
        .any(|(mixed, envelope)| mixed.params.ring_dimension != envelope.params.ring_dimension);

    let terminal_current_w_len = if needs_terminal_override {
        mixed_folds
            .last()
            .map(|fold| fold.next_w_len)
            .unwrap_or(envelope_terminal.current_w_len)
    } else {
        envelope_terminal.current_w_len
    };

    let terminal = if needs_terminal_override {
        let terminal_fold = mixed_folds.last().expect("mixed folds");
        let terminal_lp = &terminal_fold.params;
        let terminal_fold_level = mixed_folds.len() - 1;
        let field_bits = SuffixCfg::decomposition().field_bits();
        let terminal_num_polynomials = if terminal_fold_level == 0 {
            lookup_key.final_group.num_polynomials
        } else {
            1
        };
        let witness_shape = segment_typed_witness_shape(
            terminal_lp,
            field_bits,
            terminal_num_polynomials,
            terminal_num_polynomials,
            1,
            1,
        )?;
        let direct_bytes = direct_witness_bytes(field_bits, &witness_shape);
        DirectStep {
            current_w_len: terminal_current_w_len,
            witness_shape,
            direct_bytes,
            params: None,
        }
    } else {
        DirectStep {
            current_w_len: terminal_current_w_len,
            witness_shape: envelope_terminal.witness_shape.clone(),
            direct_bytes: envelope_terminal.direct_bytes,
            params: envelope_terminal.params.clone(),
        }
    };

    let total_bytes = mixed_folds
        .iter()
        .map(|fold| fold.level_bytes)
        .try_fold(0usize, |acc, bytes| {
            acc.checked_add(bytes)
                .ok_or_else(|| AkitaError::InvalidSetup("mixed-D total_bytes overflow".into()))
        })?
        .checked_add(terminal.direct_bytes)
        .ok_or_else(|| AkitaError::InvalidSetup("mixed-D total_bytes overflow".into()))?;

    let mut steps = mixed_folds.into_iter().map(Step::Fold).collect::<Vec<_>>();
    steps.push(Step::Direct(terminal));

    Ok(Schedule { steps, total_bytes })
}
