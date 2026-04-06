//! DP-based adaptive schedule planner and debug cross-check helpers.
//!
//! The core DP planner (`best_recursive_suffix`) finds the minimum-proof-size
//! recursion schedule by dynamic programming over `(level, w_len, log_basis)`
//! states.  It is used at build time by the table generator and in debug
//! builds as a cross-check against the pre-generated schedule tables.

use super::config::CommitmentConfig;
use super::schedule::{
    current_level_layout_with_log_basis, direct_witness_bytes, field_bits, hachi_level_proof_bytes,
    planned_next_w_len, HachiPlannedDirectStep, HachiPlannedLevel, HachiPlannedState,
    HachiPlannedStep, HachiScheduleInputs,
};
use crate::error::HachiError;
use crate::protocol::proof::DirectWitnessShape;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) struct PlannerState {
    pub level: usize,
    pub current_w_len: usize,
    pub log_basis: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PlannedSuffix {
    steps: Vec<HachiPlannedStep>,
    no_wrapper_bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct PlannerConfig {
    pub max_num_vars: usize,
    pub min_log_basis: u32,
    pub max_log_basis: u32,
    pub field_bits: u32,
    pub half_field_bound: u128,
}

impl PlannerConfig {
    pub(super) fn from_cfg<Cfg: CommitmentConfig>(
        max_num_vars: usize,
        min_log_basis: u32,
        max_log_basis: u32,
    ) -> Self {
        Self {
            max_num_vars,
            min_log_basis,
            max_log_basis,
            field_bits: field_bits(Cfg::decomposition()),
            half_field_bound: Cfg::planner_half_field_bound(),
        }
    }
}

// ---------------------------------------------------------------------------
// Core DP planner
// ---------------------------------------------------------------------------

fn best_recursive_suffix<Cfg: CommitmentConfig>(
    cfg: PlannerConfig,
    memo: &mut HashMap<PlannerState, PlannedSuffix>,
    state: PlannerState,
) -> Result<PlannedSuffix, HachiError> {
    if let Some(existing) = memo.get(&state) {
        return Ok(existing.clone());
    }

    let direct_state = HachiPlannedState {
        level: state.level,
        current_w_len: state.current_w_len,
        log_basis: state.log_basis,
    };
    let witness_shape = DirectWitnessShape::PackedDigits((state.current_w_len, state.log_basis));
    let direct_bytes = direct_witness_bytes(cfg.field_bits, &witness_shape);
    let mut best = PlannedSuffix {
        steps: vec![HachiPlannedStep::Direct(HachiPlannedDirectStep {
            state: direct_state,
            witness_shape,
            direct_bytes,
        })],
        no_wrapper_bytes: direct_bytes,
    };

    let inputs = HachiScheduleInputs {
        max_num_vars: cfg.max_num_vars,
        level: state.level,
        current_w_len: state.current_w_len,
    };
    if let Ok((params, layout)) =
        current_level_layout_with_log_basis::<Cfg>(inputs, state.log_basis)
    {
        let next_w_len = planned_next_w_len(cfg.field_bits, cfg.half_field_bound, &params, layout);
        if next_w_len < state.current_w_len {
            let next_level = state.level + 1;
            let next_inputs = HachiScheduleInputs {
                max_num_vars: cfg.max_num_vars,
                level: next_level,
                current_w_len: next_w_len,
            };
            for next_log_basis in state.log_basis.max(cfg.min_log_basis)..=cfg.max_log_basis {
                let next_level_params =
                    Cfg::level_params_with_log_basis(next_inputs, next_log_basis);
                let level_bytes = hachi_level_proof_bytes(
                    cfg.field_bits,
                    &params,
                    layout,
                    &next_level_params,
                    next_w_len,
                );
                let suffix = best_recursive_suffix::<Cfg>(
                    cfg,
                    memo,
                    PlannerState {
                        level: next_level,
                        current_w_len: next_w_len,
                        log_basis: next_log_basis,
                    },
                )?;
                let candidate_bytes = level_bytes + suffix.no_wrapper_bytes;
                if candidate_bytes < best.no_wrapper_bytes {
                    let mut steps = Vec::with_capacity(suffix.steps.len() + 1);
                    steps.push(HachiPlannedStep::Fold(Box::new(HachiPlannedLevel {
                        inputs,
                        params: params.clone(),
                        layout,
                        next_inputs,
                        next_level_log_basis: next_log_basis,
                        next_commit_coeffs: next_level_params.n_b * next_level_params.d,
                        level_bytes,
                    })));
                    steps.extend(suffix.steps);
                    best = PlannedSuffix {
                        steps,
                        no_wrapper_bytes: candidate_bytes,
                    };
                }
            }
        }
    }

    memo.insert(state, best.clone());
    Ok(best)
}

/// Suffix byte estimate from the DP planner at a specific state.
pub(super) fn dp_suffix_bytes<Cfg: CommitmentConfig>(
    cfg: PlannerConfig,
    state: PlannerState,
) -> Result<usize, HachiError> {
    let mut memo = HashMap::new();
    let suffix = best_recursive_suffix::<Cfg>(cfg, &mut memo, state)?;
    Ok(suffix.no_wrapper_bytes)
}

#[cfg(test)]
pub(crate) fn planned_recursive_suffix_bytes<Cfg: CommitmentConfig>(
    max_num_vars: usize,
    level: usize,
    current_w_len: usize,
    min_log_basis: u32,
    max_log_basis: u32,
) -> Result<usize, HachiError> {
    use super::schedule::planned_recursive_suffix_bytes_from_schedule;

    let inputs = HachiScheduleInputs {
        max_num_vars,
        level,
        current_w_len,
    };
    if let Some(schedule) = Cfg::schedule_plan(max_num_vars)? {
        return planned_recursive_suffix_bytes_from_schedule::<Cfg>(
            &schedule,
            max_num_vars,
            level,
            current_w_len,
            min_log_basis,
            max_log_basis,
        );
    }
    let current_log_basis = Cfg::log_basis_at_level(inputs);
    let cfg = PlannerConfig::from_cfg::<Cfg>(max_num_vars, min_log_basis, max_log_basis);
    let state = PlannerState {
        level,
        current_w_len,
        log_basis: current_log_basis,
    };
    dp_suffix_bytes::<Cfg>(cfg, state)
}

// ---------------------------------------------------------------------------
// Debug cross-check helpers
// ---------------------------------------------------------------------------

/// Find the best log_basis (minimising suffix bytes) via the DP planner over
/// the given search range.
#[cfg(debug_assertions)]
pub(super) fn dp_best_basis<Cfg: CommitmentConfig>(
    inputs: HachiScheduleInputs,
    min_log_basis: u32,
    max_log_basis: u32,
    lower_bound: u32,
) -> Option<(u32, usize)> {
    let cfg = PlannerConfig::from_cfg::<Cfg>(
        inputs.max_num_vars,
        min_log_basis,
        max_log_basis,
    );
    let mut memo = HashMap::new();
    let mut best: Option<(u32, usize)> = None;
    for log_basis in lower_bound..=max_log_basis {
        if let Ok(suffix) = best_recursive_suffix::<Cfg>(
            cfg,
            &mut memo,
            PlannerState {
                level: inputs.level,
                current_w_len: inputs.current_w_len,
                log_basis,
            },
        ) {
            if best
                .as_ref()
                .is_none_or(|(_, b)| suffix.no_wrapper_bytes < *b)
            {
                best = Some((log_basis, suffix.no_wrapper_bytes));
            }
        }
    }
    best
}

/// Warn when the table-selected log_basis diverges from the DP planner's
/// optimal choice.
#[cfg(debug_assertions)]
pub(super) fn debug_check_dp_basis<Cfg: CommitmentConfig>(
    label: &str,
    inputs: HachiScheduleInputs,
    table_basis: u32,
    min_log_basis: u32,
    max_log_basis: u32,
    lower_bound: u32,
) {
    if let Some((dp_basis, _)) =
        dp_best_basis::<Cfg>(inputs, min_log_basis, max_log_basis, lower_bound)
    {
        if table_basis != dp_basis {
            tracing::warn!(
                level = inputs.level,
                w_len = inputs.current_w_len,
                table_basis,
                dp_basis,
                "{label}"
            );
        }
    }
}

/// Warn when the table suffix byte estimate diverges significantly from the
/// DP planner's value (ratio outside `[0.5, 2.0)`).
#[cfg(debug_assertions)]
pub(super) fn debug_check_dp_suffix_bytes<Cfg: CommitmentConfig>(
    label: &str,
    state: PlannerState,
    table_bytes: usize,
    planner_cfg: PlannerConfig,
) {
    let mut memo = HashMap::new();
    if let Ok(suffix) = best_recursive_suffix::<Cfg>(planner_cfg, &mut memo, state) {
        let dp_bytes = suffix.no_wrapper_bytes;
        if dp_bytes > 0 {
            let ratio = table_bytes as f64 / dp_bytes as f64;
            if !(0.5..2.0).contains(&ratio) {
                tracing::warn!(
                    level = state.level,
                    current_w_len = state.current_w_len,
                    table_bytes,
                    dp_bytes,
                    ratio = format_args!("{ratio:.2}"),
                    "{label}"
                );
            }
        }
    }
}
