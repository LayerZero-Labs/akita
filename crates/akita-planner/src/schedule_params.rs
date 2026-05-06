//! Schedule planner that finds the global minimum proof size.
//!
//! A single exhaustive DP over `(level, w_len, log_basis)` states.  At each
//! state, every feasible basis is tried; `level_proof_bytes` uses the
//! smallest `next_commit` across all next-level bases; the suffix is
//! recursed into unconstrained.
//!
//! Uses config-supplied protocol layout derivation and the same proof-size
//! formulas as runtime generated-schedule validation.

use std::collections::HashMap;

use crate::PlannerConfig;
use akita_field::AkitaError;
use akita_types::layout::digit_math::{
    compute_num_digits_fold_with_claims, compute_num_digits_full_field,
};
use akita_types::{
    direct_witness_bytes, level_proof_bytes, planned_next_w_len, planned_w_ring_element_count,
    root_current_w_len, scale_batched_root_layout, AjtaiKeyParams, AkitaScheduleInputs, DirectStep,
    DirectWitnessShape, FoldStep, LevelParams, Schedule, Step, WitnessShape,
};
#[cfg(not(feature = "zk"))]
use akita_types::{schedule_from_plan, AkitaRootBatchSummary, AkitaScheduleLookupKey};

const MAX_RECURSION_DEPTH: usize = 12;

fn derive_batched_root_level_derivation<Cfg>(
    max_num_vars: usize,
    root_lp: &LevelParams,
    num_claims: usize,
) -> Result<(LevelParams, LevelParams), AkitaError>
where
    Cfg: PlannerConfig,
{
    let inputs = AkitaScheduleInputs {
        max_num_vars,
        level: 0,
        current_w_len: root_current_w_len(root_lp),
    };
    let level_lp = scale_batched_root_layout(
        root_lp,
        num_claims,
        Cfg::planner_stage1_challenge_config(Cfg::PLANNER_D).l1_mass(),
    )?;
    let derived_root_lp =
        Cfg::planner_root_level_params_for_layout_with_log_basis(inputs, &level_lp)?;
    Ok((level_lp, derived_root_lp))
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
fn derive_candidate_level_params<Cfg>(
    max_num_vars: usize,
    level: usize,
    current_w_len: usize,
    log_basis: u32,
) -> Option<CandidateLevelParams>
where
    Cfg: PlannerConfig,
{
    let inputs = AkitaScheduleInputs {
        max_num_vars,
        level,
        current_w_len,
    };

    let level_lp = if level == 0 {
        Cfg::planner_root_level_layout_with_log_basis(inputs, log_basis).ok()?
    } else {
        Cfg::planner_current_level_layout_with_log_basis(inputs, log_basis).ok()?
    };

    let fb = Cfg::planner_field_bits();
    let w_ring = planned_w_ring_element_count::<Cfg::PlannerField>(fb, &level_lp);
    let next_w_len = planned_next_w_len::<Cfg::PlannerField>(fb, &level_lp);

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
fn compute_level_proof_size<Cfg: PlannerConfig>(
    candidate: &CandidateLevelParams,
    next_level_params: &LevelParams,
    num_public_outputs: usize,
) -> usize {
    let fb = Cfg::planner_field_bits();
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
fn basis_range<Cfg: PlannerConfig>(
    max_num_vars: usize,
    level: usize,
    current_w_len: usize,
) -> std::ops::RangeInclusive<u32> {
    let (lo, hi) = Cfg::planner_log_basis_search_range(AkitaScheduleInputs {
        max_num_vars,
        level,
        current_w_len,
    });
    lo..=hi
}

fn level_params_from_fold_step<Cfg: PlannerConfig>(step: &FoldStep) -> LevelParams {
    debug_assert_eq!(
        Cfg::planner_stage1_challenge_config(step.params.ring_dimension).l1_mass(),
        step.params.challenge_l1_mass()
    );
    step.params.clone()
}

fn successor_level_params_from_schedule<Cfg: PlannerConfig>(
    max_num_vars: usize,
    level: usize,
    current_w_len: usize,
    suffix_steps: &[Step],
) -> Result<LevelParams, AkitaError> {
    match suffix_steps
        .first()
        .expect("optimal suffix schedule must contain at least one step")
    {
        Step::Fold(step) => Ok(level_params_from_fold_step::<Cfg>(step)),
        Step::Direct(step) => Cfg::planner_current_level_layout_with_log_basis(
            AkitaScheduleInputs {
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
fn derive_optimal_suffix_schedule<Cfg>(
    memo: &mut ScheduleMemo,
    max_num_vars: usize,
    level: usize,
    current_w_len: usize,
    current_lb: u32,
    depth: usize,
) -> (usize, Vec<Step>)
where
    Cfg: PlannerConfig,
{
    let key = (level, current_w_len, current_lb);
    if depth <= MAX_RECURSION_DEPTH {
        if let Some(cached) = memo.get(&key) {
            return cached.clone();
        }
    }

    // Baseline: send the witness directly without folding.
    let fb = Cfg::planner_field_bits();
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

/// Root-level witness ring-element count parameterized by aggregate shape.
///
/// Implements the single shape-agnostic formula
///
/// ```text
///   W(lp; K, G, P) = K · 2^r · δ_open                       // |ŵ|
///                  + K · 2^r · n_A · δ_open                 // |t̂|
///                  + G · blind_cols(n_B, D, δ_open)          // |blind|
///                  + P · 2^m · δ_commit · δ_fold            // |z_pre|
///                  + (n_D + n_B·G + P + 1 + n_A) · δ_R(b)   // |r|
/// ```
///
/// Here `blind_cols` counts the decomposed B-matrix columns reserved for fresh
/// LHL blinding when compiled with `zk`; otherwise it is zero. The singleton
/// case is just `(K, G, P) = (1, 1, 1)` of this formula.
fn root_w_ring_element_count<Cfg>(lp: &LevelParams, shape: &WitnessShape) -> usize
where
    Cfg: PlannerConfig,
{
    let fb = Cfg::planner_field_bits();
    let r_decomp = compute_num_digits_full_field(fb, lp.log_basis);

    let w_hat = shape.num_claims * lp.num_blocks * lp.num_digits_open;
    let t_hat = shape.num_claims * lp.num_blocks * lp.a_key.row_len() * lp.num_digits_open;
    let z_pre = shape.num_points * lp.inner_width() * lp.num_digits_fold;
    // One public y-row per distinct opening point.
    let r_rows = lp.m_row_count(shape.num_commitment_groups, shape.num_points);
    let r = r_rows * r_decomp;

    #[cfg(feature = "zk")]
    {
        let blind = shape.num_commitment_groups
            * akita_types::zk::blind_column_count::<Cfg::PlannerField>(
                lp.b_key.row_len(),
                lp.ring_dimension,
                lp.num_digits_open,
            );
        w_hat + t_hat + blind + z_pre + r
    }
    #[cfg(not(feature = "zk"))]
    {
        w_hat + t_hat + z_pre + r
    }
}

// -----------------------------------------------------------------------
// Shape-agnostic root candidate + entry point
// -----------------------------------------------------------------------

/// Derive the optimal root candidate at `log_basis` for any witness shape.
///
/// Runs the full `(m, r)` block-split search using the shape-agnostic
/// witness sizing formula `root_w_ring_element_count`. Singleton openings
/// are `WitnessShape::singleton()`.
fn derive_root_candidate<Cfg>(
    max_num_vars: usize,
    root_w_len: usize,
    log_basis: u32,
    shape: &WitnessShape,
) -> Option<CandidateLevelParams>
where
    Cfg: PlannerConfig,
{
    let inputs = AkitaScheduleInputs {
        max_num_vars,
        level: 0,
        current_w_len: root_w_len,
    };

    let root_lp = Cfg::planner_root_level_layout_with_log_basis(inputs, log_basis).ok()?;
    let fb = Cfg::planner_field_bits();

    let alpha = Cfg::PLANNER_D.trailing_zeros() as usize;
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
        let Ok(a_key) = AjtaiKeyParams::try_new(
            root_lp.a_key.row_len(),
            inner_width,
            root_lp.a_key.collision_inf(),
            d,
        ) else {
            continue;
        };
        let Ok(b_key) = AjtaiKeyParams::try_new(
            root_lp.b_key.row_len(),
            outer_width,
            root_lp.b_key.collision_inf(),
            d,
        ) else {
            continue;
        };
        let Ok(d_key) = AjtaiKeyParams::try_new(
            root_lp.d_key.row_len(),
            d_matrix_width,
            root_lp.d_key.collision_inf(),
            d,
        ) else {
            continue;
        };

        let candidate_lp = LevelParams {
            ring_dimension: d,
            log_basis: root_lp.log_basis,
            a_key,
            b_key,
            d_key,
            num_blocks,
            block_len,
            m_vars,
            r_vars,
            stage1_config: root_lp.stage1_config.clone(),
            num_digits_commit: root_lp.num_digits_commit,
            num_digits_open: root_lp.num_digits_open,
            num_digits_fold: per_poly_fold,
        };

        let Ok((level_lp, proof_lp)) = derive_batched_root_level_derivation::<Cfg>(
            max_num_vars,
            &candidate_lp,
            shape.num_claims,
        ) else {
            continue;
        };
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

/// Consult the offline schedule tables for a pre-computed answer.
///
/// Returns `Ok(Some(schedule))` when the config ships an offline entry
/// whose [`AkitaScheduleLookupKey`] exactly matches the requested
/// `(max_num_vars=num_vars, num_vars, layout_num_claims, batch)` case.
/// A [`WitnessShape`] is invalid as a [`AkitaRootBatchSummary`] when
/// `G > K` or `P > K`, in which case no offline entry can exist and we
/// return `Ok(None)` so the caller falls through to the DP.
fn offline_schedule_for_shape<Cfg>(
    max_num_vars: usize,
    num_vars: usize,
    shape: WitnessShape,
) -> Result<Option<Schedule>, AkitaError>
where
    Cfg: PlannerConfig,
{
    #[cfg(feature = "zk")]
    {
        let _ = std::marker::PhantomData::<Cfg>;
        let _ = (max_num_vars, num_vars, shape);
        Ok(None)
    }
    #[cfg(not(feature = "zk"))]
    {
        let batch = match AkitaRootBatchSummary::new(
            shape.num_claims,
            shape.num_commitment_groups,
            shape.num_points,
        ) {
            Ok(batch) => batch,
            Err(_) => return Ok(None),
        };
        let key =
            AkitaScheduleLookupKey::with_batch(max_num_vars, num_vars, shape.num_claims, batch);
        Ok(Cfg::planner_schedule_plan(key)?
            .map(|plan| schedule_from_plan(&plan, Cfg::planner_field_bits())))
    }
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
/// generated schedule tables in `akita-types`). Every such entry is keyed
/// on the full [`WitnessShape`] — i.e. each batching case is a distinct
/// row — so this function just returns the stored answer in O(1). Only
/// shapes without an offline entry fall back to the DP search.
///
/// # Errors
///
/// Returns an error if any of `K`, `G`, `P` is zero, if the witness
/// length overflows, or if the config's offline-table lookup fails.
pub fn find_optimal_schedule<Cfg>(
    num_vars: usize,
    shape: WitnessShape,
) -> Result<Schedule, AkitaError>
where
    Cfg: PlannerConfig,
{
    find_optimal_schedule_with_max::<Cfg>(num_vars, num_vars, shape)
}

/// Find the optimal schedule for an opening with distinct setup capacity and
/// actual witness size.
///
/// `max_num_vars` is the setup/config capacity used for SIS-secure matrix
/// sizing; `num_vars` is the opened polynomial size and therefore determines
/// the root witness length.
///
/// # Errors
///
/// Returns an error if any of `K`, `G`, `P` is zero, if the witness
/// length overflows, or if the config's offline-table lookup fails.
pub fn find_optimal_schedule_with_max<Cfg>(
    max_num_vars: usize,
    num_vars: usize,
    shape: WitnessShape,
) -> Result<Schedule, AkitaError>
where
    Cfg: PlannerConfig,
{
    if shape.num_claims == 0 || shape.num_commitment_groups == 0 || shape.num_points == 0 {
        return Err(AkitaError::InvalidSetup(
            "witness shape dimensions must be at least 1".into(),
        ));
    }

    if let Some(schedule) = offline_schedule_for_shape::<Cfg>(max_num_vars, num_vars, shape)? {
        tracing::debug!(
            max_num_vars,
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
        .ok_or_else(|| AkitaError::InvalidSetup("witness too large".into()))?;

    let fb = Cfg::planner_field_bits();
    let mut best_cost = direct_witness_bytes(fb, &DirectWitnessShape::FieldElements(root_w_len));
    let mut best_steps: Vec<Step> = vec![to_direct_step(root_w_len, 128)];
    let mut memo = ScheduleMemo::new();

    for root_lb in basis_range::<Cfg>(max_num_vars, 0, root_w_len) {
        let Some(candidate) =
            derive_root_candidate::<Cfg>(max_num_vars, root_w_len, root_lb, &shape)
        else {
            continue;
        };
        let (suffix_cost, suffix_steps) = derive_optimal_suffix_schedule::<Cfg>(
            &mut memo,
            max_num_vars,
            1,
            candidate.next_w_len,
            root_lb,
            0,
        );
        let Ok(next_level_params) = successor_level_params_from_schedule::<Cfg>(
            max_num_vars,
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
        max_num_vars,
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
