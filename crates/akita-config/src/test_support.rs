//! Mixed-D schedule stitching for Wave 0 fixtures and Phase 4 planner bring-up.
//!
//! Production callers size inputs through [`CommitmentConfig::get_params_for_batched_commitment`].
//! Phase 4 moves this module beside generated expansion in `akita-planner` once envelope
//! policy and DP `ring_d` search land.

use akita_field::AkitaError;
use akita_planner::generated::{table_entry, GeneratedFoldStep, GeneratedStep};
use akita_planner::generated_schedule_lookup_key;
use akita_types::{
    direct_witness_bytes, level_proof_bytes, terminal_direct_witness_shape_for_key,
    w_ring_element_count_with_counts_for_layout_bits, AkitaScheduleInputs, AkitaScheduleLookupKey,
    DirectStep, FoldStep, LevelParams, MRowLayout, OpeningBatchShape, Schedule, Step,
};

use crate::{policy_of, CommitmentConfig};

struct MixedSuffixFoldPlan {
    params: LevelParams,
    current_w_len: usize,
    next_w_len: usize,
    is_terminal: bool,
}

fn generated_fold_step<Cfg: CommitmentConfig>(
    key: AkitaScheduleLookupKey,
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

fn suffix_level_params<SuffixCfg: CommitmentConfig>(
    key: AkitaScheduleLookupKey,
    level: usize,
    current_w_len: usize,
) -> Result<LevelParams, AkitaError> {
    let suffix_gen = generated_fold_step::<SuffixCfg>(key, level)?;
    let policy = policy_of::<SuffixCfg>();
    let fold_shape = SuffixCfg::fold_challenge_shape_at_level(AkitaScheduleInputs {
        num_vars: key.num_vars,
        level,
        current_w_len,
    });
    suffix_gen.expand_to_level_params(
        &policy,
        SuffixCfg::ring_challenge_config,
        level,
        current_w_len,
        fold_shape,
        1,
    )
}

fn mixed_level_params<EnvelopeCfg, SuffixCfg>(
    key: AkitaScheduleLookupKey,
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
    envelope_gen.expand_envelope_witness_at_ring_d(
        &suffix_policy,
        SuffixCfg::ring_challenge_config,
        level,
        envelope_current_w_len,
        suffix_ring_d,
        fold_shape,
        num_claims,
        extra_block_vars,
    )
}

/// Extra block-select variables when dropping ring dimension by a power-of-two factor.
pub fn extra_block_vars_for_drop(prev_ring_d: usize, suffix_ring_d: usize) -> usize {
    if prev_ring_d > suffix_ring_d && prev_ring_d.is_multiple_of(suffix_ring_d) {
        let downscale = prev_ring_d / suffix_ring_d;
        if downscale.is_power_of_two() {
            return downscale.trailing_zeros() as usize;
        }
    }
    0
}

/// Hand-built mixed-D schedule for runtime ring cutover Wave 0.
///
/// Fold levels `[0, switch_at_fold)` use [`EnvelopeCfg`]'s shipped table unchanged.
/// Levels `[switch_at_fold, …)` keep the envelope witness-length chain while
/// halving the per-level ring dimension (typically `128 → 64`): block counts and
/// Ajtai keys are scaled from the suffix table so `commit_next_w` dispatch at
/// `D=64` matches the envelope field-length ladder.
///
/// Wave 0 fixtures use `num_polynomials == 1` only.
///
/// # Errors
///
/// Returns an error when either preset schedule cannot be resolved, fold counts
/// disagree, or `switch_at_fold` is out of range.
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
            "mixed-D Wave 0 fixture supports singleton batches only (got {num_polynomials})"
        )));
    }
    let lookup_key = AkitaScheduleLookupKey::new(num_vars, num_polynomials);
    let envelope = EnvelopeCfg::runtime_schedule(lookup_key)?;
    let suffix = SuffixCfg::runtime_schedule(lookup_key)?;

    let envelope_folds: Vec<FoldStep> = envelope.fold_steps().cloned().collect();
    let suffix_folds: Vec<FoldStep> = suffix.fold_steps().cloned().collect();
    if envelope_folds.is_empty() {
        return Err(AkitaError::InvalidSetup(
            "mixed-D fixture requires a folded schedule".into(),
        ));
    }
    if envelope_folds.len() != suffix_folds.len() {
        return Err(AkitaError::InvalidSetup(format!(
            "envelope and suffix schedules disagree on fold count: {} vs {}",
            envelope_folds.len(),
            suffix_folds.len()
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
                    lookup_key,
                    level,
                    w_len,
                    &envelope_step.params,
                    suffix_ring_d,
                    prev_ring_d,
                )?
            } else {
                suffix_level_params::<SuffixCfg>(lookup_key, level, w_len)?
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
    let needs_tail_grind_override = mixed_folds
        .iter()
        .zip(envelope_folds.iter())
        .any(|(mixed, envelope)| mixed.params.ring_dimension != envelope.params.ring_dimension);

    let terminal_current_w_len = if needs_tail_grind_override {
        mixed_folds
            .last()
            .map(|fold| fold.next_w_len)
            .unwrap_or(envelope_terminal.current_w_len)
    } else {
        envelope_terminal.current_w_len
    };

    let terminal = if needs_tail_grind_override {
        let terminal_fold = mixed_folds.last().expect("mixed folds");
        let terminal_lp = &terminal_fold.params;
        let terminal_fold_level = mixed_folds.len() - 1;
        let field_bits = SuffixCfg::decomposition().field_bits();
        let witness_shape = terminal_direct_witness_shape_for_key(
            terminal_lp,
            field_bits,
            lookup_key,
            terminal_fold_level,
            terminal_current_w_len,
            terminal_lp.log_basis,
        )?;
        let direct_bytes = direct_witness_bytes(field_bits, &witness_shape);
        DirectStep {
            current_w_len: terminal_current_w_len,
            witness_shape,
            direct_bytes,
            params: None,
            tail_grind_level_params: None,
        }
    } else {
        DirectStep {
            current_w_len: terminal_current_w_len,
            witness_shape: envelope_terminal.witness_shape.clone(),
            direct_bytes: envelope_terminal.direct_bytes,
            params: envelope_terminal.params.clone(),
            tail_grind_level_params: None,
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
    let lookup_key = AkitaScheduleLookupKey::new(num_vars, num_polynomials);
    let schedule = Cfg::runtime_schedule(lookup_key)?;
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
    Cfg::get_params_for_batched_commitment(&OpeningBatchShape::new(num_vars, num_polynomials)?)
}
