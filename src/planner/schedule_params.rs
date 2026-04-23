//! Schedule planner that finds the global minimum proof size.
//!
//! A single exhaustive DP over `(level, w_len, log_basis)` states.  At each
//! state, every feasible basis is tried; `level_proof_bytes` uses the
//! smallest `next_commit` across all next-level bases; the suffix is
//! recursed into unconstrained.
//!
//! Uses the protocol's own layout derivation, `planned_next_w_len`, and
//! `exact_recursive_level_proof_bytes` — no separate cost model, no
//! approximations.

use std::collections::HashMap;

use crate::error::HachiError;
use crate::planner::digit_math::compute_num_digits_fold_with_claims;
use crate::protocol::commitment::{
    current_level_layout_with_log_basis, derive_batched_root_level_derivation,
    direct_witness_bytes, field_bits, level_proof_bytes, planned_next_w_len,
    planned_w_ring_element_count, recursive_r_decomp_levels, CommitmentConfig, HachiPlannedStep,
    HachiRootBatchSummary, HachiScheduleInputs, HachiScheduleLookupKey, HachiSchedulePlan,
};
use crate::protocol::params::{AjtaiKeyParams, LevelParams};
use crate::protocol::proof::DirectWitnessShape;

const MAX_RECURSION_DEPTH: usize = 12;

// -----------------------------------------------------------------------
// Output types
// -----------------------------------------------------------------------

/// Parameters for one fold level in the computed schedule.
#[derive(Clone, Debug)]
pub struct FoldStep {
    /// Unified level parameters (ring dimension, Ajtai keys, block geometry,
    /// digit depths, challenge config).
    pub params: LevelParams,
    /// Witness length entering this level.
    pub current_w_len: usize,
    /// Per-polynomial fold digits (`num_claims=1`). Equal to
    /// `params.num_digits_fold` for singleton schedules; smaller for batched
    /// roots where the layout uses the batched bound.
    pub delta_fold_per_poly: usize,
    /// Ring-element count in the witness after ring-switching.
    pub w_ring: usize,
    /// Witness length leaving this level.
    pub next_w_len: usize,
    /// Proof bytes for this level.
    pub level_bytes: usize,
}

/// Terminal direct-send step.
#[derive(Clone, Debug)]
pub struct DirectStep {
    pub current_w_len: usize,
    pub bits_per_elem: u32,
    pub direct_bytes: usize,
}

/// A single step in the schedule.
#[derive(Clone, Debug)]
pub enum Step {
    Fold(FoldStep),
    Direct(DirectStep),
}

/// Complete schedule with step-by-step parameters.
#[derive(Clone, Debug)]
pub struct Schedule {
    pub steps: Vec<Step>,
    pub total_bytes: usize,
}

// -----------------------------------------------------------------------
// Single-level evaluation
// -----------------------------------------------------------------------

/// All layout data for one candidate fold level.
struct CandidateLevelParams {
    proof_lp: LevelParams,
    lp: LevelParams,
    next_w_len: usize,
    w_ring: usize,
}

/// Derive the layout for folding at `(level, w_len, log_basis)`.
/// Returns `None` if the layout is infeasible or doesn't shrink the witness.
fn derive_candidate_level_params<Cfg: CommitmentConfig>(
    max_num_vars: usize,
    level: usize,
    current_w_len: usize,
    log_basis: u32,
) -> Option<CandidateLevelParams> {
    let inputs = HachiScheduleInputs {
        max_num_vars,
        level,
        current_w_len,
    };

    let level_lp = if level == 0 {
        Cfg::root_level_layout_with_log_basis(inputs, log_basis).ok()?
    } else {
        current_level_layout_with_log_basis::<Cfg>(inputs, log_basis).ok()?
    };

    let fb = field_bits(Cfg::decomposition());
    let w_ring = planned_w_ring_element_count(fb, &level_lp);
    let next_w_len = planned_next_w_len(fb, &level_lp);

    let input_elem_bits = if level == 0 {
        fb as usize
    } else {
        log_basis as usize
    };
    if next_w_len * (log_basis as usize) >= current_w_len * input_elem_bits {
        return None;
    }

    Some(CandidateLevelParams {
        proof_lp: level_lp.clone(),
        lp: level_lp,
        next_w_len,
        w_ring,
    })
}

/// Compute the proof bytes for this fold level against a concrete successor.
fn compute_level_proof_size<Cfg: CommitmentConfig>(
    candidate: &CandidateLevelParams,
    next_level_params: &LevelParams,
    num_public_outputs: usize,
) -> usize {
    let fb = field_bits(Cfg::decomposition());
    level_proof_bytes(
        fb,
        &candidate.proof_lp,
        &candidate.lp,
        next_level_params,
        candidate.next_w_len,
        num_public_outputs,
    )
}

// -----------------------------------------------------------------------
// Step construction
// -----------------------------------------------------------------------

fn to_fold_step(c: &CandidateLevelParams, current_w_len: usize, level_bytes: usize) -> Step {
    let per_poly_fold = compute_num_digits_fold_with_claims(
        c.lp.r_vars,
        c.lp.challenge_l1_mass(),
        c.lp.log_basis,
        1,
    );
    Step::Fold(FoldStep {
        params: c.lp.clone(),
        current_w_len,
        delta_fold_per_poly: per_poly_fold,
        w_ring: c.w_ring,
        next_w_len: c.next_w_len,
        level_bytes,
    })
}

fn to_direct_step(current_w_len: usize, log_basis: u32) -> Step {
    Step::Direct(DirectStep {
        current_w_len,
        bits_per_elem: log_basis,
        direct_bytes: (current_w_len * log_basis as usize).div_ceil(8),
    })
}

/// Inclusive range of `log_basis` values to search at a given state.
fn basis_range<Cfg: CommitmentConfig>(
    max_num_vars: usize,
    level: usize,
    current_w_len: usize,
) -> std::ops::RangeInclusive<u32> {
    let (lo, hi) = Cfg::log_basis_search_range(HachiScheduleInputs {
        max_num_vars,
        level,
        current_w_len,
    });
    lo..=hi
}

fn level_params_from_fold_step<Cfg: CommitmentConfig>(step: &FoldStep) -> LevelParams {
    debug_assert_eq!(
        Cfg::stage1_challenge_config(step.params.ring_dimension).l1_mass(),
        step.params.challenge_l1_mass()
    );
    step.params.clone()
}

fn successor_level_params_from_schedule<Cfg: CommitmentConfig>(
    max_num_vars: usize,
    level: usize,
    current_w_len: usize,
    suffix_steps: &[Step],
) -> Result<LevelParams, HachiError> {
    match suffix_steps
        .first()
        .expect("optimal suffix schedule must contain at least one step")
    {
        Step::Fold(step) => Ok(level_params_from_fold_step::<Cfg>(step)),
        Step::Direct(step) => current_level_layout_with_log_basis::<Cfg>(
            HachiScheduleInputs {
                max_num_vars,
                level,
                current_w_len,
            },
            step.bits_per_elem,
        ),
    }
}

// -----------------------------------------------------------------------
// DP — suffix search
// -----------------------------------------------------------------------

/// Memo key: `(level, w_len, log_basis)`.
type ScheduleMemo = HashMap<(usize, usize, u32), (usize, Vec<Step>)>;

/// Find the minimum-cost suffix starting at `(level, current_w_len, current_lb)`,
/// returning both the total bytes and the step-by-step schedule.
fn derive_optimal_suffix_schedule<Cfg: CommitmentConfig>(
    memo: &mut ScheduleMemo,
    max_num_vars: usize,
    level: usize,
    current_w_len: usize,
    current_lb: u32,
    depth: usize,
) -> (usize, Vec<Step>) {
    let key = (level, current_w_len, current_lb);
    if depth <= MAX_RECURSION_DEPTH {
        if let Some(cached) = memo.get(&key) {
            return cached.clone();
        }
    }

    // Baseline: send the witness directly without folding.
    let fb = field_bits(Cfg::decomposition());
    let direct_bytes = direct_witness_bytes(
        fb,
        &DirectWitnessShape::PackedDigits((current_w_len, current_lb)),
    );
    let mut best_cost = direct_bytes;
    let mut best_schedule = vec![to_direct_step(current_w_len, current_lb)];

    // Try each feasible basis for one more fold level.
    if depth <= MAX_RECURSION_DEPTH {
        for lb in basis_range::<Cfg>(max_num_vars, level, current_w_len) {
            if lb < current_lb {
                continue;
            }
            let Some(candidate) =
                derive_candidate_level_params::<Cfg>(max_num_vars, level, current_w_len, lb)
            else {
                continue;
            };

            let (suffix_cost, suffix_steps) = derive_optimal_suffix_schedule::<Cfg>(
                memo,
                max_num_vars,
                level + 1,
                candidate.next_w_len,
                lb,
                depth + 1,
            );
            let Ok(next_level_params) = successor_level_params_from_schedule::<Cfg>(
                max_num_vars,
                level + 1,
                candidate.next_w_len,
                &suffix_steps,
            ) else {
                continue;
            };
            let level_proof_size =
                compute_level_proof_size::<Cfg>(&candidate, &next_level_params, 1);

            let total = level_proof_size + suffix_cost;
            if total < best_cost {
                best_cost = total;
                let mut steps = Vec::with_capacity(1 + suffix_steps.len());
                steps.push(to_fold_step(&candidate, current_w_len, level_proof_size));
                steps.extend(suffix_steps);
                best_schedule = steps;
            }
        }

        memo.insert(key, (best_cost, best_schedule.clone()));
    }

    (best_cost, best_schedule)
}

// -----------------------------------------------------------------------
// Witness shape
// -----------------------------------------------------------------------

/// Aggregate witness-shape inputs that determine root-level sizing.
///
/// The root-level witness ring count is, for any `(K, G, P)`:
///
/// ```text
///   W(lp; K, G, P) = K · 2^r · δ_open                       // |ŵ|
///                  + K · 2^r · n_A · δ_open                 // |t̂|
///                  + P · 2^m · δ_commit · δ_fold            // |z_pre|
///                  + (n_D + n_B·G + P + 1 + n_A) · δ_R(b)   // |r|
/// ```
///
/// Singleton openings are simply the `K = G = P = 1` special case of this
/// formula; the planner does not need to branch on "batched vs non-batched"
/// — only on this aggregate shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WitnessShape {
    /// `K` — total number of polynomial claims (drives `|ŵ|, |t̂|`).
    pub num_claims: usize,
    /// `G` — number of commitment groups (drives the `n_B·G` term in `|r|`).
    pub num_commitment_groups: usize,
    /// `P` — number of distinct opening points (drives `|z_pre|` and the
    /// `+P` term in `|r|`).
    pub num_points: usize,
}

impl WitnessShape {
    /// Build a witness shape from explicit `(K, G, P)`.
    pub const fn new(num_claims: usize, num_commitment_groups: usize, num_points: usize) -> Self {
        Self {
            num_claims,
            num_commitment_groups,
            num_points,
        }
    }

    /// Singleton shape: one polynomial, one group, one point.
    pub const fn singleton() -> Self {
        Self {
            num_claims: 1,
            num_commitment_groups: 1,
            num_points: 1,
        }
    }

    /// Build a witness shape from per-group opening-point counts.
    ///
    /// Interprets `points_per_group[g]` as the number of distinct opening
    /// points associated with commitment group `g`. The aggregates are:
    ///
    /// * `G = points_per_group.len()`
    /// * `P = sum(points_per_group)`  (treats each group's points as
    ///   distinct from other groups')
    /// * `K = sum(points_per_group)`  (one claim per `(group, point)` pair)
    pub fn from_points_per_group(points_per_group: &[usize]) -> Self {
        let num_commitment_groups = points_per_group.len();
        let total_points: usize = points_per_group.iter().copied().sum();
        Self {
            num_claims: total_points,
            num_commitment_groups,
            num_points: total_points,
        }
    }
}

/// Root-level witness ring-element count parameterized by aggregate shape.
///
/// Implements the single shape-agnostic formula
///
/// ```text
///   W(lp; K, G, P) = K · 2^r · δ_open                       // |ŵ|
///                  + K · 2^r · n_A · δ_open                 // |t̂|
///                  + P · 2^m · δ_commit · δ_fold            // |z_pre|
///                  + (n_D + n_B·G + P + 1 + n_A) · δ_R(b)   // |r|
/// ```
///
/// Mirrors `w_ring_element_count_with_counts` in `ring_switch.rs` but uses
/// field-erased helpers already available to the planner. The singleton
/// case is just `(K, G, P) = (1, 1, 1)` of this formula — there is no
/// branching on "batched vs non-batched".
fn root_w_ring_element_count<Cfg: CommitmentConfig>(
    lp: &LevelParams,
    shape: &WitnessShape,
) -> usize {
    let fb = field_bits(Cfg::decomposition());
    let r_decomp = recursive_r_decomp_levels(fb, lp.log_basis);

    let w_hat = shape.num_claims * lp.num_blocks * lp.num_digits_open;
    let t_hat = shape.num_claims * lp.num_blocks * lp.a_key.row_len() * lp.num_digits_open;
    let z_pre = shape.num_points * lp.inner_width() * lp.num_digits_fold;
    // One public y-row per distinct opening point.
    let r_rows = lp.m_row_count(shape.num_commitment_groups, shape.num_points);
    let r = r_rows * r_decomp;

    w_hat + t_hat + z_pre + r
}

// -----------------------------------------------------------------------
// Shape-agnostic root candidate + entry point
// -----------------------------------------------------------------------

/// Derive the optimal root candidate at `log_basis` for any witness shape.
///
/// Runs the full `(m, r)` block-split search using the shape-agnostic
/// witness sizing formula `root_w_ring_element_count`. Singleton openings
/// are `WitnessShape::singleton()`.
fn derive_root_candidate<Cfg: CommitmentConfig, const D: usize>(
    max_num_vars: usize,
    root_w_len: usize,
    log_basis: u32,
    shape: &WitnessShape,
) -> Option<CandidateLevelParams> {
    let inputs = HachiScheduleInputs {
        max_num_vars,
        level: 0,
        current_w_len: root_w_len,
    };

    let root_lp = Cfg::root_level_layout_with_log_basis(inputs, log_basis).ok()?;
    let fb = field_bits(Cfg::decomposition());

    let alpha = Cfg::D.trailing_zeros() as usize;
    let reduced_vars = max_num_vars.checked_sub(alpha)?;
    if reduced_vars < 1 {
        return None;
    }

    let mut best: Option<CandidateLevelParams> = None;

    // Try every feasible (m, r) split with m + r = reduced_vars.
    //
    // We allow r_vars = 0 as a candidate: this corresponds to a single
    // block (no row-direction folding). It is the natural fallback for
    // very small witnesses where `optimal_m_r_split` would otherwise pick
    // r = 0 in the singleton path (`reduced_vars <= 2`). For larger
    // witnesses the loop also covers all r in [1, reduced_vars - 1].
    let r_lo: usize = if reduced_vars >= 3 { 1 } else { 0 };
    let r_hi: usize = reduced_vars.saturating_sub(1).max(r_lo);

    for r_vars in r_lo..=r_hi {
        let m_vars = reduced_vars - r_vars;
        let per_poly_fold = compute_num_digits_fold_with_claims(
            r_vars,
            root_lp.challenge_l1_mass(),
            root_lp.log_basis,
            1,
        );

        let Some(num_blocks) = 1usize.checked_shl(r_vars as u32) else {
            continue;
        };
        let Some(block_len) = 1usize.checked_shl(m_vars as u32) else {
            continue;
        };
        let Some(inner_width) = block_len.checked_mul(root_lp.num_digits_commit) else {
            continue;
        };
        let Some(outer_width) = root_lp
            .a_key
            .row_len()
            .checked_mul(root_lp.num_digits_open)
            .and_then(|x| x.checked_mul(num_blocks))
        else {
            continue;
        };
        let Some(d_matrix_width) = root_lp.num_digits_open.checked_mul(num_blocks) else {
            continue;
        };

        let d = root_lp.ring_dimension;
        let candidate_lp = LevelParams {
            ring_dimension: d,
            log_basis: root_lp.log_basis,
            a_key: AjtaiKeyParams::new_unchecked(
                root_lp.a_key.row_len(),
                inner_width,
                root_lp.a_key.collision_inf(),
                d,
            ),
            b_key: AjtaiKeyParams::new_unchecked(
                root_lp.b_key.row_len(),
                outer_width,
                root_lp.b_key.collision_inf(),
                d,
            ),
            d_key: AjtaiKeyParams::new_unchecked(
                root_lp.d_key.row_len(),
                d_matrix_width,
                root_lp.d_key.collision_inf(),
                d,
            ),
            num_blocks,
            block_len,
            m_vars,
            r_vars,
            stage1_config: root_lp.stage1_config.clone(),
            num_digits_commit: root_lp.num_digits_commit,
            num_digits_open: root_lp.num_digits_open,
            num_digits_fold: per_poly_fold,
        };

        let Ok(derivation) = derive_batched_root_level_derivation::<Cfg, D>(
            max_num_vars,
            &candidate_lp,
            shape.num_claims,
        ) else {
            continue;
        };
        let level_lp = derivation.level_lp;
        let proof_lp = derivation.root_lp;
        let w_ring = root_w_ring_element_count::<Cfg>(&level_lp, shape);
        let next_w_len = w_ring * level_lp.ring_dimension;

        if next_w_len * (log_basis as usize) >= root_w_len * (fb as usize) {
            continue;
        }

        if best.as_ref().is_none_or(|b| next_w_len < b.next_w_len) {
            best = Some(CandidateLevelParams {
                proof_lp,
                lp: level_lp,
                next_w_len,
                w_ring,
            });
        }
    }

    best
}

/// Translate an offline [`HachiSchedulePlan`] into this planner's
/// [`Schedule`] format.
///
/// The offline schedule tables in `src/protocol/commitment/generated/*`
/// are the authoritative source of pre-computed optimal schedules for
/// every `(Cfg, max_num_vars, WitnessShape)` case that ships with the
/// crate, and the runtime converts each entry into a
/// [`HachiSchedulePlan`] via `Cfg::schedule_plan`. This helper maps that
/// plan back into a [`Schedule`] so the standalone planner API can hand
/// out the pre-computed answer without redoing the DP search.
fn schedule_from_plan<Cfg: CommitmentConfig>(plan: &HachiSchedulePlan) -> Schedule {
    let field_bits_u32 = field_bits(Cfg::decomposition());
    let mut steps = Vec::with_capacity(plan.steps.len());
    for step in &plan.steps {
        match step {
            HachiPlannedStep::Fold(level) => {
                let lp = level.lp.clone();
                let delta_fold_per_poly = compute_num_digits_fold_with_claims(
                    lp.r_vars,
                    lp.challenge_l1_mass(),
                    lp.log_basis,
                    1,
                );
                let ring_dim = lp.ring_dimension;
                let next_w_len = level.next_inputs.current_w_len;
                let w_ring = next_w_len / ring_dim;
                steps.push(Step::Fold(FoldStep {
                    params: lp,
                    current_w_len: level.inputs.current_w_len,
                    delta_fold_per_poly,
                    w_ring,
                    next_w_len,
                    level_bytes: level.level_bytes,
                }));
            }
            HachiPlannedStep::Direct(direct) => {
                let bits_per_elem = match direct.witness_shape {
                    DirectWitnessShape::PackedDigits((_, bits)) => bits,
                    DirectWitnessShape::FieldElements(_) => field_bits_u32,
                };
                steps.push(Step::Direct(DirectStep {
                    current_w_len: direct.state.current_w_len,
                    bits_per_elem,
                    direct_bytes: direct.direct_bytes,
                }));
            }
        }
    }
    Schedule {
        steps,
        total_bytes: plan.exact_proof_bytes,
    }
}

/// Consult the offline schedule tables for a pre-computed answer.
///
/// Returns `Ok(Some(schedule))` when the config ships an offline entry
/// whose [`HachiScheduleLookupKey`] exactly matches the requested
/// `(max_num_vars=num_vars, num_vars, layout_num_claims, batch)` case.
/// A [`WitnessShape`] is invalid as a [`HachiRootBatchSummary`] when
/// `G > K` or `P > K`, in which case no offline entry can exist and we
/// return `Ok(None)` so the caller falls through to the DP.
fn offline_schedule_for_shape<Cfg: CommitmentConfig>(
    num_vars: usize,
    shape: WitnessShape,
) -> Result<Option<Schedule>, HachiError> {
    let batch = match HachiRootBatchSummary::new(
        shape.num_claims,
        shape.num_commitment_groups,
        shape.num_points,
    ) {
        Ok(batch) => batch,
        Err(_) => return Ok(None),
    };
    let key = HachiScheduleLookupKey::with_batch(num_vars, num_vars, shape.num_claims, batch);
    Ok(Cfg::schedule_plan(key)?.map(|plan| schedule_from_plan::<Cfg>(&plan)))
}

/// Find the optimal schedule for any root opening shape.
///
/// The planner is shape-agnostic: it takes the aggregate witness shape
/// `(K, G, P)` (or, equivalently, per-group point counts) and always
/// returns the same optimum. Singleton openings are
/// `WitnessShape::singleton()`; there is no separate code path for
/// batched vs non-batched.
///
/// **Offline fast path.** Each `(Cfg, num_vars, shape)` that ships with
/// the crate has a pre-computed entry in `Cfg::schedule_plan` (the
/// generated schedule tables in
/// `src/protocol/commitment/generated/*`). Every such entry is keyed on
/// the full [`WitnessShape`] — i.e. each batching case is a distinct
/// row — so this function just returns the stored answer in O(1). Only
/// shapes without an offline entry fall back to the DP search.
///
/// # Errors
///
/// Returns an error if any of `K`, `G`, `P` is zero, if the witness
/// length overflows, or if the config's offline-table lookup fails.
pub fn find_optimal_schedule<Cfg: CommitmentConfig, const D: usize>(
    num_vars: usize,
    shape: WitnessShape,
) -> Result<Schedule, HachiError> {
    if shape.num_claims == 0 || shape.num_commitment_groups == 0 || shape.num_points == 0 {
        return Err(HachiError::InvalidSetup(
            "witness shape dimensions must be at least 1".into(),
        ));
    }

    if let Some(schedule) = offline_schedule_for_shape::<Cfg>(num_vars, shape)? {
        tracing::debug!(
            num_vars,
            num_claims = shape.num_claims,
            num_commitment_groups = shape.num_commitment_groups,
            num_points = shape.num_points,
            total_bytes = schedule.total_bytes,
            "schedule planner: served from offline schedule tables"
        );
        return Ok(schedule);
    }

    let root_w_len = 1usize
        .checked_shl(num_vars as u32)
        .ok_or_else(|| HachiError::InvalidSetup("witness too large".into()))?;

    let fb = field_bits(Cfg::decomposition());
    let mut best_cost = direct_witness_bytes(fb, &DirectWitnessShape::FieldElements(root_w_len));
    let mut best_steps: Vec<Step> = vec![to_direct_step(root_w_len, 128)];
    let mut memo = ScheduleMemo::new();

    for root_lb in basis_range::<Cfg>(num_vars, 0, root_w_len) {
        let Some(candidate) =
            derive_root_candidate::<Cfg, D>(num_vars, root_w_len, root_lb, &shape)
        else {
            continue;
        };
        let (suffix_cost, suffix_steps) = derive_optimal_suffix_schedule::<Cfg>(
            &mut memo,
            num_vars,
            1,
            candidate.next_w_len,
            root_lb,
            0,
        );
        let Ok(next_level_params) = successor_level_params_from_schedule::<Cfg>(
            num_vars,
            1,
            candidate.next_w_len,
            &suffix_steps,
        ) else {
            continue;
        };
        // Root-level proofs carry one public y-ring per distinct opening point.
        let root_proof_size =
            compute_level_proof_size::<Cfg>(&candidate, &next_level_params, shape.num_points);

        let total = root_proof_size + suffix_cost;
        if total < best_cost {
            best_cost = total;
            let mut steps = Vec::with_capacity(1 + suffix_steps.len());
            steps.push(to_fold_step(&candidate, root_w_len, root_proof_size));
            steps.extend(suffix_steps);
            best_steps = steps;
        }
    }

    let num_folds = best_steps
        .iter()
        .filter(|s| matches!(s, Step::Fold(_)))
        .count();
    tracing::info!(
        num_vars,
        num_claims = shape.num_claims,
        num_commitment_groups = shape.num_commitment_groups,
        num_points = shape.num_points,
        total_bytes = best_cost,
        fold_levels = num_folds,
        "schedule planner: computed from scratch (no offline entry)"
    );

    Ok(Schedule {
        steps: best_steps,
        total_bytes: best_cost,
    })
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::commitment::presets::fp128;
    use crate::protocol::commitment::{
        exact_planned_level_execution, exact_schedule_plan_for_lookup_key,
        hachi_root_runtime_plan_with_batch, CommitmentPreset, GeneratedAdaptivePolicy,
        HachiRootBatchSummary, HachiScheduleLookupKey,
    };
    use crate::protocol::ring_switch::w_ring_element_count_with_batch_summary;

    type D64OH = CommitmentPreset<fp128::Field, GeneratedAdaptivePolicy<fp128::Profile, 64, 1>>;
    type D128Full = fp128::D128Full;

    #[test]
    fn rejects_zero_shape_dimensions() {
        for shape in [
            WitnessShape::new(0, 1, 1),
            WitnessShape::new(1, 0, 1),
            WitnessShape::new(1, 1, 0),
        ] {
            assert!(find_optimal_schedule::<D64OH, 64>(20, shape).is_err());
        }
    }

    #[test]
    fn monotonic_in_claims() {
        for nv in [16, 20, 25] {
            let s1 = find_optimal_schedule::<D64OH, 64>(nv, WitnessShape::new(1, 1, 1));
            let s4 = find_optimal_schedule::<D64OH, 64>(nv, WitnessShape::new(4, 4, 1));
            let s8 = find_optimal_schedule::<D64OH, 64>(nv, WitnessShape::new(8, 8, 1));

            if let (Ok(s1), Ok(s4), Ok(s8)) = (&s1, &s4, &s8) {
                assert!(
                    s1.total_bytes <= s4.total_bytes,
                    "D64-oh nv={nv}: 1-claim ({}) > 4-claim ({})",
                    s1.total_bytes,
                    s4.total_bytes,
                );
                assert!(
                    s4.total_bytes <= s8.total_bytes,
                    "D64-oh nv={nv}: 4-claim ({}) > 8-claim ({})",
                    s4.total_bytes,
                    s8.total_bytes,
                );
            }
        }
    }

    /// `from_points_per_group` is the user-facing way to build a
    /// `WitnessShape`: assert it derives the expected `(K, G, P)`.
    #[test]
    fn from_points_per_group_builds_expected_shapes() {
        let cases: &[(&[usize], WitnessShape)] = &[
            (&[1], WitnessShape::singleton()),
            (&[2, 2], WitnessShape::new(4, 2, 4)),
            (&[1, 1, 1, 1], WitnessShape::new(4, 4, 4)),
            (&[3, 1], WitnessShape::new(4, 2, 4)),
        ];
        for (per_group, expected) in cases {
            let shape = WitnessShape::from_points_per_group(per_group);
            assert_eq!(
                shape, *expected,
                "from_points_per_group({per_group:?}) gave {shape:?}, expected {expected:?}"
            );
            // Smoke: the planner accepts the derived shape.
            let _ = find_optimal_schedule::<D64OH, 64>(20, shape)
                .expect("planner should succeed on derived shape");
        }
    }

    fn assert_standalone_root_matches_runtime<Cfg: CommitmentConfig, const D: usize>(
        num_vars: usize,
        shape: WitnessShape,
    ) {
        let schedule =
            find_optimal_schedule::<Cfg, D>(num_vars, shape).expect("planner should succeed");
        let Some(Step::Fold(root_step)) = schedule.steps.first() else {
            panic!("planner should start with a fold");
        };
        let batch_summary = HachiRootBatchSummary::new(
            shape.num_claims,
            shape.num_commitment_groups,
            shape.num_points,
        )
        .expect("valid batch summary");
        let runtime_root = hachi_root_runtime_plan_with_batch::<Cfg, D>(
            num_vars,
            num_vars,
            shape.num_claims,
            batch_summary,
        )
        .expect("runtime root plan should succeed");

        assert_eq!(root_step.next_w_len, runtime_root.next_w_len());
        assert_eq!(
            root_step.level_bytes,
            runtime_root.level_proof_bytes::<Cfg>()
        );
    }

    #[test]
    fn standalone_root_matches_runtime_bytes() {
        assert_standalone_root_matches_runtime::<D64OH, 64>(20, WitnessShape::new(4, 1, 1));
        assert_standalone_root_matches_runtime::<D128Full, 128>(20, WitnessShape::new(4, 1, 1));
    }

    fn assert_table_root_matches_runtime<Cfg: CommitmentConfig, const D: usize>(
        num_vars: usize,
        shape: WitnessShape,
    ) {
        let batch_summary = HachiRootBatchSummary::new(
            shape.num_claims,
            shape.num_commitment_groups,
            shape.num_points,
        )
        .expect("valid batch summary");
        let key =
            HachiScheduleLookupKey::with_batch(num_vars, num_vars, shape.num_claims, batch_summary);
        let plan =
            exact_schedule_plan_for_lookup_key::<Cfg, D>(key).expect("batched exact schedule");
        let root_log_basis = plan
            .fold_levels()
            .next()
            .expect("batched exact schedule should begin with a fold")
            .lp
            .log_basis;
        let planned_root = exact_planned_level_execution::<Cfg>(
            &plan,
            HachiScheduleInputs {
                max_num_vars: num_vars,
                level: 0,
                current_w_len: 1usize.checked_shl(num_vars as u32).unwrap_or(0),
            },
            root_log_basis,
        )
        .expect("batched exact root execution should resolve")
        .expect("batched exact root execution should match");
        let runtime_root = hachi_root_runtime_plan_with_batch::<Cfg, D>(
            num_vars,
            num_vars,
            shape.num_claims,
            batch_summary,
        )
        .expect("runtime root plan should succeed");
        let runtime_w_ring = w_ring_element_count_with_batch_summary::<Cfg::Field>(
            &planned_root.level.lp,
            batch_summary,
        );

        assert_eq!(
            planned_root.level.lp, runtime_root.level_lp,
            "planned/runtime root layout mismatch for batch={batch_summary:?}"
        );
        assert_eq!(
            planned_root.level.next_inputs.current_w_len,
            runtime_root.next_w_len(),
            "planned/runtime next_w_len mismatch for batch={batch_summary:?}"
        );
        assert_eq!(
            runtime_w_ring * planned_root.level.lp.ring_dimension,
            runtime_root.next_w_len(),
            "planner/runtime root witness sizing mismatch for batch={batch_summary:?}"
        );
        assert_eq!(
            planned_root.level.level_bytes,
            runtime_root.level_proof_bytes::<Cfg>(),
            "planned/runtime root proof bytes mismatch for batch={batch_summary:?}"
        );
        assert_eq!(
            planned_root.level.next_level_log_basis, runtime_root.next_level_params.log_basis,
            "planned/runtime next-level basis mismatch for batch={batch_summary:?}"
        );
        assert_eq!(
            planned_root.level.next_commit_coeffs,
            runtime_root.next_level_params.b_key.row_len()
                * runtime_root.next_level_params.ring_dimension,
            "planned/runtime next commitment shape mismatch for batch={batch_summary:?}"
        );
        assert_eq!(
            planned_root.level.lp.b_key.row_len(),
            runtime_root.root_lp.b_key.row_len(),
            "planned/runtime B-row rank mismatch for batch={batch_summary:?}"
        );
        assert_eq!(
            planned_root.level.lp.d_key.row_len(),
            runtime_root.root_lp.d_key.row_len(),
            "planned/runtime D-row rank mismatch for batch={batch_summary:?}"
        );
    }

    #[test]
    fn onehot_root_matches_runtime_for_group_and_point_counts() {
        for shape in [
            WitnessShape::new(6, 6, 1),
            WitnessShape::new(6, 3, 1),
            WitnessShape::new(6, 3, 2),
        ] {
            assert_table_root_matches_runtime::<D64OH, 64>(20, shape);
        }
    }

    #[test]
    fn dense_root_matches_runtime_for_group_and_point_counts() {
        for shape in [
            WitnessShape::new(6, 6, 1),
            WitnessShape::new(6, 3, 1),
            WitnessShape::new(6, 3, 2),
        ] {
            assert_table_root_matches_runtime::<D128Full, 128>(20, shape);
        }
    }
}
