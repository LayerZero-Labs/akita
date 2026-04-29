//! DP-based adaptive schedule planner and debug cross-check helpers.
//!
//! The core DP planner (`best_recursive_suffix`) finds the minimum-proof-size
//! recursion schedule by dynamic programming over `(level, w_len, log_basis)`
//! states. Recursive candidate levels are scored by exact serialized proof-body
//! size under the current wire format. It is used at build time by the table
//! generator and in debug builds as a cross-check against the pre-generated
//! schedule tables.

use super::config::CommitmentConfig;
use super::schedule::{
    current_level_layout_with_log_basis, direct_witness_bytes, exact_recursive_level_proof_bytes,
    field_bits, planned_next_w_len, HachiBatchPlanningEnvelope, HachiPlannedDirectStep,
    HachiPlannedLevel, HachiPlannedState, HachiPlannedStep, HachiScheduleInputs,
};
use crate::error::HachiError;

use crate::protocol::proof::DirectWitnessShape;
use std::any::TypeId;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) struct PlannerConfig {
    pub max_num_vars: usize,
    pub min_log_basis: u32,
    pub max_log_basis: u32,
    pub field_bits: u32,
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
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct CachedSuffixStateKey {
    cfg_type: TypeId,
    cfg: PlannerConfig,
    envelope: HachiBatchPlanningEnvelope,
    state: PlannerState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct CachedBasisChoiceKey {
    cfg_type: TypeId,
    cfg: PlannerConfig,
    envelope: HachiBatchPlanningEnvelope,
    inputs: HachiScheduleInputs,
    lower_bound: u32,
}

type ExactSuffixCache = HashMap<CachedSuffixStateKey, usize>;
type BestBasisCache = HashMap<CachedBasisChoiceKey, (u32, usize)>;

fn exact_suffix_cache() -> &'static Mutex<ExactSuffixCache> {
    static CACHE: OnceLock<Mutex<ExactSuffixCache>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn best_basis_cache() -> &'static Mutex<BestBasisCache> {
    static CACHE: OnceLock<Mutex<BestBasisCache>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
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

    let (direct_log_basis, witness_shape) = if state.level == 0 {
        (
            Cfg::decomposition().log_basis,
            DirectWitnessShape::FieldElements(state.current_w_len),
        )
    } else {
        (
            state.log_basis,
            DirectWitnessShape::PackedDigits((state.current_w_len, state.log_basis)),
        )
    };
    let direct_state = HachiPlannedState {
        level: state.level,
        current_w_len: state.current_w_len,
        log_basis: direct_log_basis,
    };
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
    if let Ok(level_lp) = current_level_layout_with_log_basis::<Cfg>(inputs, state.log_basis) {
        let next_w_len = planned_next_w_len(cfg.field_bits, &level_lp);
        if next_w_len < state.current_w_len {
            let next_level = state.level + 1;
            let next_inputs = HachiScheduleInputs {
                max_num_vars: cfg.max_num_vars,
                level: next_level,
                current_w_len: next_w_len,
            };
            for next_log_basis in state.log_basis.max(cfg.min_log_basis)..=cfg.max_log_basis {
                let next_lp =
                    current_level_layout_with_log_basis::<Cfg>(next_inputs, next_log_basis)?;
                let level_bytes = exact_recursive_level_proof_bytes::<Cfg::Field>(
                    &level_lp, &next_lp, next_w_len,
                )?;
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
                        lp: level_lp.clone(),
                        next_inputs,
                        next_level_log_basis: next_log_basis,
                        next_commit_coeffs: next_lp.b_key.row_len() * next_lp.ring_dimension,
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

pub(super) fn cached_dp_suffix_bytes<Cfg: CommitmentConfig>(
    cfg: PlannerConfig,
    envelope: HachiBatchPlanningEnvelope,
    state: PlannerState,
) -> Result<usize, HachiError> {
    let cache_key = CachedSuffixStateKey {
        cfg_type: TypeId::of::<Cfg>(),
        cfg,
        envelope,
        state,
    };
    if let Some(bytes) = exact_suffix_cache()
        .lock()
        .expect("exact suffix cache lock poisoned")
        .get(&cache_key)
        .copied()
    {
        return Ok(bytes);
    }

    let mut memo = HashMap::new();
    let suffix = best_recursive_suffix::<Cfg>(cfg, &mut memo, state)?;
    let mut cache = exact_suffix_cache()
        .lock()
        .expect("exact suffix cache lock poisoned");
    for (memo_state, planned_suffix) in memo {
        cache
            .entry(CachedSuffixStateKey {
                cfg_type: TypeId::of::<Cfg>(),
                cfg,
                envelope,
                state: memo_state,
            })
            .or_insert(planned_suffix.no_wrapper_bytes);
    }
    Ok(suffix.no_wrapper_bytes)
}

pub(super) fn cached_dp_best_basis<Cfg: CommitmentConfig>(
    cfg: PlannerConfig,
    envelope: HachiBatchPlanningEnvelope,
    inputs: HachiScheduleInputs,
    lower_bound: u32,
) -> Option<(u32, usize)> {
    let cache_key = CachedBasisChoiceKey {
        cfg_type: TypeId::of::<Cfg>(),
        cfg,
        envelope,
        inputs,
        lower_bound,
    };
    if let Some(best) = best_basis_cache()
        .lock()
        .expect("best basis cache lock poisoned")
        .get(&cache_key)
        .copied()
    {
        return Some(best);
    }

    let best = if lower_bound > cfg.max_log_basis {
        cached_dp_suffix_bytes::<Cfg>(
            cfg,
            envelope,
            PlannerState {
                level: inputs.level,
                current_w_len: inputs.current_w_len,
                log_basis: lower_bound,
            },
        )
        .ok()
        .map(|bytes| (lower_bound, bytes))
    } else {
        let mut best: Option<(u32, usize)> = None;
        for log_basis in lower_bound..=cfg.max_log_basis {
            let state = PlannerState {
                level: inputs.level,
                current_w_len: inputs.current_w_len,
                log_basis,
            };
            if let Ok(suffix_bytes) = cached_dp_suffix_bytes::<Cfg>(cfg, envelope, state) {
                if best.as_ref().is_none_or(|(_, bytes)| suffix_bytes < *bytes) {
                    best = Some((log_basis, suffix_bytes));
                }
            }
        }
        best
    }?;

    best_basis_cache()
        .lock()
        .expect("best basis cache lock poisoned")
        .insert(cache_key, best);
    Some(best)
}
