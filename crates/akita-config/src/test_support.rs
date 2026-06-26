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

use akita_field::AkitaError;
use akita_types::{
    AkitaScheduleLookupKey, DirectStep, FoldStep, LevelParams, OpeningBatchShape, Schedule, Step,
};

use crate::CommitmentConfig;

fn mixed_level_params(
    envelope_params: &LevelParams,
    suffix_params: &LevelParams,
) -> Result<LevelParams, AkitaError> {
    let envelope_d = envelope_params.ring_dimension;
    let suffix_d = suffix_params.ring_dimension;
    if envelope_d == suffix_d {
        return Ok(envelope_params.clone());
    }
    if !envelope_d.is_multiple_of(suffix_d) {
        return Err(AkitaError::InvalidSetup(format!(
            "envelope ring dimension {envelope_d} is not divisible by suffix ring dimension {suffix_d}"
        )));
    }
    let scale = envelope_d / suffix_d;
    let mut params = envelope_params.clone();
    params.ring_dimension = suffix_d;
    params.stage1_config = suffix_params.stage1_config.clone();
    params.num_blocks = params.num_blocks.checked_mul(scale).ok_or_else(|| {
        AkitaError::InvalidSetup("mixed-D num_blocks scale overflow".into())
    })?;
    params.a_key = suffix_params.a_key.clone();
    params.b_key = suffix_params.b_key.clone();
    params.d_key = suffix_params.d_key.clone();
    params.num_digits_commit = suffix_params.num_digits_commit;
    params.num_digits_open = suffix_params.num_digits_open;
    Ok(params)
}

/// Hand-built mixed-D schedule for runtime ring cutover Wave 0.
///
/// Fold levels `[0, switch_at_fold)` use [`EnvelopeCfg`]'s shipped table unchanged.
/// Levels `[switch_at_fold, …)` keep the envelope witness-length chain while
/// halving the per-level ring dimension (typically `128 → 64`): block counts and
/// Ajtai keys are scaled from the suffix table so `commit_next_w` dispatch at
/// `D=64` matches the envelope field-length ladder.
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

    let mixed_folds = envelope_folds
        .iter()
        .zip(suffix_folds.iter())
        .enumerate()
        .map(|(level, (envelope_step, suffix_step))| {
            if level < switch_at_fold {
                Ok(envelope_step.clone())
            } else {
                Ok(FoldStep {
                    params: mixed_level_params(&envelope_step.params, &suffix_step.params)?,
                    current_w_len: envelope_step.current_w_len,
                    next_w_len: envelope_step.next_w_len,
                    level_bytes: envelope_step.level_bytes,
                })
            }
        })
        .collect::<Result<Vec<_>, _>>()?;

    let envelope_terminal = match envelope.steps.last() {
        Some(Step::Direct(step)) => step,
        _ => {
            return Err(AkitaError::InvalidSetup(
                "envelope schedule must end in a direct witness step".into(),
            ));
        }
    };
    let terminal = DirectStep {
        current_w_len: envelope_terminal.current_w_len,
        witness_shape: envelope_terminal.witness_shape.clone(),
        direct_bytes: envelope_terminal.direct_bytes,
        params: envelope_terminal.params.clone(),
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

    let mut steps = mixed_folds
        .into_iter()
        .map(Step::Fold)
        .collect::<Vec<_>>();
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
    // Size the fallback for the requested batch (`num_polynomials`), not a
    // singleton — otherwise the per-poly inputs would be smaller than the
    // batched commit layout `Scheme::commit` actually uses.
    Cfg::get_params_for_batched_commitment(&OpeningBatchShape::new(num_vars, num_polynomials)?)
}
