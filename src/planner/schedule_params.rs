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
use crate::planner::digit_math::compute_num_digits_fold;
use crate::protocol::commitment::{
    current_level_layout_with_log_basis, derive_batched_root_level_derivation,
    direct_witness_bytes, field_bits, level_proof_bytes, planned_next_w_len,
    planned_w_ring_element_count, recursive_r_decomp_levels, CommitmentConfig, HachiScheduleInputs,
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
        &candidate.lp,
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
    let per_poly_fold =
        compute_num_digits_fold(c.lp.r_vars, c.lp.challenge_l1_mass(), c.lp.log_basis, 1);
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
// Batched mode
// -----------------------------------------------------------------------

/// Batch dimensions for a batched root opening.
#[derive(Clone, Copy, Debug)]
pub struct BatchConfig {
    /// Total number of polynomial claims being opened.
    pub num_claims: usize,
    /// Number of commitment groups (each group shares one commitment).
    pub num_commitment_groups: usize,
    /// Number of distinct opening points.
    pub num_points: usize,
}

impl BatchConfig {
    /// Singleton batch: one polynomial, one group, one point.
    pub const fn singleton() -> Self {
        Self {
            num_claims: 1,
            num_commitment_groups: 1,
            num_points: 1,
        }
    }
}

/// Batched root witness ring-element count.
///
/// Mirrors `w_ring_element_count_with_counts` in `ring_switch.rs` but uses
/// field-erased helpers already available to the planner.
/// `num_claims` scales the concatenated witness pieces (`w_hat`, `t_hat`),
/// while `num_points` scales only the public y rows and folded preimage rows.
fn batched_root_w_ring_element_count<Cfg: CommitmentConfig>(
    lp: &LevelParams,
    batch: &BatchConfig,
) -> usize {
    let fb = field_bits(Cfg::decomposition());
    let r_decomp = recursive_r_decomp_levels(fb, lp.log_basis);

    let w_hat = batch.num_claims * lp.num_blocks * lp.num_digits_open;
    let t_hat = batch.num_claims * lp.num_blocks * lp.a_key.row_len() * lp.num_digits_open;
    let z_pre = batch.num_points * lp.inner_width() * lp.num_digits_fold;
    let r_rows = if batch.num_points == 1 && batch.num_commitment_groups == 1 {
        lp.m_row_count()
    } else {
        lp.m_row_count_with_commitments_and_public_outputs(
            batch.num_commitment_groups,
            batch.num_points,
        )
    };
    let r = r_rows * r_decomp;

    w_hat + t_hat + z_pre + r
}

/// Derive the batched root candidate for a given `log_basis`.
///
/// Gets the per-poly `(params, layout)` from `Cfg`, then computes the
/// batched witness size and applies the shrink check.  For singleton
/// batches, uses the per-poly layout directly.  For multi-claim batches,
/// searches over `(m, r)` splits to find the split that minimises the
/// batched witness, since the optimal balance shifts toward fewer blocks
/// when each block is replicated across `num_claims` openings.
fn derive_batched_root_candidate<Cfg: CommitmentConfig, const D: usize>(
    max_num_vars: usize,
    root_w_len: usize,
    log_basis: u32,
    batch: &BatchConfig,
) -> Option<CandidateLevelParams> {
    let inputs = HachiScheduleInputs {
        max_num_vars,
        level: 0,
        current_w_len: root_w_len,
    };

    let root_lp = Cfg::root_level_layout_with_log_basis(inputs, log_basis).ok()?;

    let fb = field_bits(Cfg::decomposition());

    if batch.num_claims <= 1 && batch.num_commitment_groups <= 1 && batch.num_points <= 1 {
        let w_ring = planned_w_ring_element_count(fb, &root_lp);
        let next_w_len = planned_next_w_len(fb, &root_lp);

        if next_w_len * (log_basis as usize) >= root_w_len * (fb as usize) {
            return None;
        }
        return Some(CandidateLevelParams {
            lp: root_lp,
            next_w_len,
            w_ring,
        });
    }

    // Multi-claim: search over (m, r) splits.  More claims amplify the
    // per-block opening cost, shifting the optimum toward fewer blocks.
    let alpha = Cfg::D.trailing_zeros() as usize;
    let reduced_vars = max_num_vars.checked_sub(alpha)?;
    if reduced_vars < 2 {
        return None;
    }

    let mut best: Option<CandidateLevelParams> = None;

    for r_vars in 1..reduced_vars {
        let m_vars = reduced_vars - r_vars;
        let per_poly_fold =
            compute_num_digits_fold(r_vars, root_lp.challenge_l1_mass(), root_lp.log_basis, 1);

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
        let bd_collision = (1u32 << root_lp.log_basis) - 1;
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
                bd_collision,
                d,
            ),
            d_key: AjtaiKeyParams::new_unchecked(
                root_lp.d_key.row_len(),
                d_matrix_width,
                bd_collision,
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
            batch.num_claims,
        ) else {
            continue;
        };
        let mut planner_layout = candidate_lp.clone();
        planner_layout.num_digits_fold = derivation.level_lp.num_digits_fold;
        // Merge: rank from the batch-effective derivation (may differ for
        // adaptive configs), geometry from the per-claim candidate layout.
        let lp = derivation.root_lp.with_layout(&planner_layout);

        let w_ring = batched_root_w_ring_element_count::<Cfg>(&lp, batch);
        let next_w_len = w_ring * lp.ring_dimension;

        if next_w_len * (log_basis as usize) >= root_w_len * (fb as usize) {
            continue;
        }

        if best.as_ref().is_none_or(|b| next_w_len < b.next_w_len) {
            best = Some(CandidateLevelParams {
                lp,
                next_w_len,
                w_ring,
            });
        }
    }

    best
}

/// Find the optimal batched schedule with full step-by-step parameters.
///
/// Only the root level (level 0) differs from the single-poly planner:
/// the proof carries `num_claims` ring openings and the root witness is
/// larger. Recursive levels (1+) are identical.
pub fn find_optimal_batched_schedule<Cfg: CommitmentConfig, const D: usize>(
    num_vars: usize,
    batch: BatchConfig,
) -> Result<Schedule, HachiError> {
    if batch.num_claims == 0 || batch.num_commitment_groups == 0 || batch.num_points == 0 {
        return Err(HachiError::InvalidSetup(
            "batch dimensions must be at least 1".into(),
        ));
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
            derive_batched_root_candidate::<Cfg, D>(num_vars, root_w_len, root_lb, &batch)
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
        let root_proof_size =
            compute_level_proof_size::<Cfg>(&candidate, &next_level_params, batch.num_points);

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
        num_claims = batch.num_claims,
        num_commitment_groups = batch.num_commitment_groups,
        num_points = batch.num_points,
        total_bytes = best_cost,
        fold_levels = num_folds,
        "refactored planner: schedule computed from scratch (no pre-computed table)"
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

    fn assert_batched_root_parity<Cfg: CommitmentConfig, const D: usize>(
        num_vars: usize,
        batch: BatchConfig,
    ) {
        let batch_summary = HachiRootBatchSummary::new(
            batch.num_claims,
            batch.num_commitment_groups,
            batch.num_points,
        )
        .expect("valid batch summary");
        let key =
            HachiScheduleLookupKey::with_batch(num_vars, num_vars, batch.num_claims, batch_summary);
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
            batch.num_claims,
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
    fn batched_monotonic_in_claims() {
        for nv in [16, 20, 25] {
            let s1 = find_optimal_batched_schedule::<D64OH, 64>(
                nv,
                BatchConfig {
                    num_claims: 1,
                    num_commitment_groups: 1,
                    num_points: 1,
                },
            );
            let s4 = find_optimal_batched_schedule::<D64OH, 64>(
                nv,
                BatchConfig {
                    num_claims: 4,
                    num_commitment_groups: 4,
                    num_points: 1,
                },
            );
            let s8 = find_optimal_batched_schedule::<D64OH, 64>(
                nv,
                BatchConfig {
                    num_claims: 8,
                    num_commitment_groups: 8,
                    num_points: 1,
                },
            );

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

    #[test]
    fn batched_rejects_zero_dimensions() {
        let zero_claims = BatchConfig {
            num_claims: 0,
            num_commitment_groups: 1,
            num_points: 1,
        };
        assert!(find_optimal_batched_schedule::<D64OH, 64>(20, zero_claims).is_err());

        let zero_groups = BatchConfig {
            num_claims: 1,
            num_commitment_groups: 0,
            num_points: 1,
        };
        assert!(find_optimal_batched_schedule::<D64OH, 64>(20, zero_groups).is_err());

        let zero_points = BatchConfig {
            num_claims: 1,
            num_commitment_groups: 1,
            num_points: 0,
        };
        assert!(find_optimal_batched_schedule::<D64OH, 64>(20, zero_points).is_err());
    }

    #[test]
    fn onehot_batched_root_matches_runtime_for_group_and_point_counts() {
        for batch in [
            BatchConfig {
                num_claims: 6,
                num_commitment_groups: 6,
                num_points: 1,
            },
            BatchConfig {
                num_claims: 6,
                num_commitment_groups: 3,
                num_points: 1,
            },
            BatchConfig {
                num_claims: 6,
                num_commitment_groups: 3,
                num_points: 2,
            },
        ] {
            assert_batched_root_parity::<D64OH, 64>(20, batch);
        }
    }

    #[test]
    fn dense_batched_root_matches_runtime_for_group_and_point_counts() {
        for batch in [
            BatchConfig {
                num_claims: 6,
                num_commitment_groups: 6,
                num_points: 1,
            },
            BatchConfig {
                num_claims: 6,
                num_commitment_groups: 3,
                num_points: 1,
            },
            BatchConfig {
                num_claims: 6,
                num_commitment_groups: 3,
                num_points: 2,
            },
        ] {
            assert_batched_root_parity::<D128Full, 128>(20, batch);
        }
    }
}
