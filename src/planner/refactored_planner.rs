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
    batched_root_level_proof_bytes, derive_commitment_layout, direct_witness_bytes, field_bits,
    planned_next_w_len, planned_w_ring_element_count, recursive_level_decomposition_from_root,
    recursive_r_decomp_levels_for_bound, CommitmentConfig, HachiCommitmentLayout, HachiLevelParams,
    HachiScheduleInputs,
};
use crate::protocol::proof::DirectWitnessShape;

const MAX_RECURSION_DEPTH: usize = 12;

// -----------------------------------------------------------------------
// Output types
// -----------------------------------------------------------------------

/// Parameters for one fold level in the computed schedule.
#[derive(Clone, Debug)]
pub struct FoldStep {
    pub current_w_len: usize,
    pub d: u32,
    pub log_basis: u32,
    pub challenge_l1_mass: usize,
    pub m_vars: usize,
    pub r_vars: usize,
    pub n_a: usize,
    pub n_b: usize,
    pub n_d: usize,
    pub delta_open: usize,
    pub delta_commit: usize,
    pub delta_fold: usize,
    /// Per-polynomial fold digits (`num_claims=1`).  Equal to `delta_fold`
    /// for singleton schedules; smaller for batched roots where the layout
    /// uses the batched bound.
    pub delta_fold_per_poly: usize,
    pub w_ring: usize,
    pub next_w_len: usize,
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
    params: HachiLevelParams,
    layout: HachiCommitmentLayout,
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

    // Root: `root_level_layout_with_log_basis` runs a fixed-point n_a
    // convergence loop that may produce different params than
    // `level_params_with_log_basis` — we must use its returned params.
    // Recursive: params + derive_commitment_layout.
    let (params, layout) = if level == 0 {
        Cfg::root_level_layout_with_log_basis(inputs, log_basis).ok()?
    } else {
        let params = Cfg::level_params_with_log_basis(inputs, log_basis);
        if current_w_len % params.d != 0 {
            return None;
        }
        let num_ring = current_w_len / params.d;
        let reduced_vars = num_ring.next_power_of_two().max(1).trailing_zeros() as usize;
        let decomp = recursive_level_decomposition_from_root(Cfg::decomposition(), log_basis);
        let layout = derive_commitment_layout(&params, decomp, reduced_vars, num_ring).ok()?;
        (params, layout)
    };

    let fb = field_bits(Cfg::decomposition());
    let hfb = Cfg::planner_half_field_bound();
    let w_ring = planned_w_ring_element_count(fb, hfb, &params, layout);
    let next_w_len = planned_next_w_len(fb, hfb, &params, layout);

    let input_elem_bits = if level == 0 {
        fb as usize
    } else {
        log_basis as usize
    };
    if next_w_len * (log_basis as usize) >= current_w_len * input_elem_bits {
        return None;
    }

    Some(CandidateLevelParams {
        params,
        layout,
        next_w_len,
        w_ring,
    })
}

/// Compute the minimum proof bytes for this fold level, trying all feasible
/// next-level bases to find the smallest `next_commit`.
///
/// `num_claims` controls the batched `y_ring` size in the proof; pass `1`
/// for single-polynomial and recursive levels.
fn compute_level_proof_size<Cfg: CommitmentConfig>(
    candidate: &CandidateLevelParams,
    max_num_vars: usize,
    level: usize,
    num_claims: usize,
) -> usize {
    let next_inputs = HachiScheduleInputs {
        max_num_vars,
        level: level + 1,
        current_w_len: candidate.next_w_len,
    };

    let fb = field_bits(Cfg::decomposition());
    let proof_bytes_with = |next_params: &HachiLevelParams| {
        batched_root_level_proof_bytes(
            fb,
            &candidate.params,
            candidate.layout,
            next_params,
            candidate.next_w_len,
            num_claims,
        )
    };

    let (lo, hi) = Cfg::log_basis_search_range(next_inputs);
    (lo..=hi)
        .map(|next_lb| proof_bytes_with(&Cfg::level_params_with_log_basis(next_inputs, next_lb)))
        .min()
        .unwrap_or_else(|| proof_bytes_with(&Cfg::level_params(next_inputs)))
}

// -----------------------------------------------------------------------
// Step construction
// -----------------------------------------------------------------------

fn to_fold_step(c: &CandidateLevelParams, current_w_len: usize, level_bytes: usize) -> Step {
    let per_poly_fold = compute_num_digits_fold(
        c.layout.r_vars,
        c.params.challenge_l1_mass,
        c.layout.log_basis,
        1,
    );
    Step::Fold(FoldStep {
        current_w_len,
        d: c.params.d as u32,
        log_basis: c.layout.log_basis,
        challenge_l1_mass: c.params.challenge_l1_mass,
        m_vars: c.layout.m_vars,
        r_vars: c.layout.r_vars,
        n_a: c.params.n_a,
        n_b: c.params.n_b,
        n_d: c.params.n_d,
        delta_open: c.layout.num_digits_open,
        delta_commit: c.layout.num_digits_commit,
        delta_fold: c.layout.num_digits_fold,
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

            let level_proof_size =
                compute_level_proof_size::<Cfg>(&candidate, max_num_vars, level, 1);
            let (suffix_cost, suffix_steps) = derive_optimal_suffix_schedule::<Cfg>(
                memo,
                max_num_vars,
                level + 1,
                candidate.next_w_len,
                lb,
                depth + 1,
            );

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
fn batched_root_w_ring_element_count<Cfg: CommitmentConfig>(
    params: &HachiLevelParams,
    layout: HachiCommitmentLayout,
    batch: &BatchConfig,
) -> usize {
    let fb = field_bits(Cfg::decomposition());
    let hfb = Cfg::planner_half_field_bound();
    let r_decomp = recursive_r_decomp_levels_for_bound(fb, hfb, layout.log_basis);

    let batched_num_digits_fold = layout.num_digits_fold.max(compute_num_digits_fold(
        layout.r_vars,
        params.challenge_l1_mass,
        layout.log_basis,
        batch.num_claims,
    ));

    let w_hat = batch.num_claims * layout.num_blocks * layout.num_digits_open;
    let t_hat = batch.num_claims * layout.num_blocks * params.n_a * layout.num_digits_open;
    let z_pre = batch.num_points * layout.inner_width * batched_num_digits_fold;
    let r_rows = params.m_row_count_with_commitments_and_public_outputs(
        batch.num_commitment_groups,
        batch.num_claims,
    );
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
fn derive_batched_root_candidate<Cfg: CommitmentConfig>(
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

    let (params, layout) = Cfg::root_level_layout_with_log_basis(inputs, log_basis).ok()?;

    let fb = field_bits(Cfg::decomposition());

    if batch.num_claims <= 1 && batch.num_commitment_groups <= 1 && batch.num_points <= 1 {
        let hfb = Cfg::planner_half_field_bound();
        let w_ring = planned_w_ring_element_count(fb, hfb, &params, layout);
        let next_w_len = planned_next_w_len(fb, hfb, &params, layout);

        if next_w_len * (log_basis as usize) >= root_w_len * (fb as usize) {
            return None;
        }
        return Some(CandidateLevelParams {
            params,
            layout,
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
        let batched_fold = layout.num_digits_fold.max(compute_num_digits_fold(
            r_vars,
            params.challenge_l1_mass,
            layout.log_basis,
            batch.num_claims,
        ));

        let Ok(candidate_layout) = HachiCommitmentLayout::new_with_decomp(
            m_vars,
            r_vars,
            params.n_a,
            layout.num_digits_commit,
            layout.num_digits_open,
            batched_fold,
            layout.log_basis,
            0,
        ) else {
            continue;
        };

        let w_ring = batched_root_w_ring_element_count::<Cfg>(&params, candidate_layout, batch);
        let next_w_len = w_ring * params.d;

        if next_w_len * (log_basis as usize) >= root_w_len * (fb as usize) {
            continue;
        }

        if best.as_ref().is_none_or(|b| next_w_len < b.next_w_len) {
            best = Some(CandidateLevelParams {
                params: params.clone(),
                layout: candidate_layout,
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
            derive_batched_root_candidate::<Cfg>(num_vars, root_w_len, root_lb, &batch)
        else {
            continue;
        };

        let root_proof_size =
            compute_level_proof_size::<Cfg>(&candidate, num_vars, 0, batch.num_claims);
        let (suffix_cost, suffix_steps) = derive_optimal_suffix_schedule::<Cfg>(
            &mut memo,
            num_vars,
            1,
            candidate.next_w_len,
            root_lb,
            0,
        );

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
    use crate::protocol::commitment::generated::{
        self, GeneratedScheduleKey, GeneratedScheduleTable,
    };
    use crate::protocol::commitment::presets::fp128;
    use crate::protocol::commitment::{CommitmentPreset, GeneratedAdaptivePolicy};

    type D128OH = CommitmentPreset<fp128::Field, GeneratedAdaptivePolicy<fp128::Profile, 128, 1>>;
    type D128F = CommitmentPreset<fp128::Field, GeneratedAdaptivePolicy<fp128::Profile, 128, 128>>;
    type D64OH = CommitmentPreset<fp128::Field, GeneratedAdaptivePolicy<fp128::Profile, 64, 1>>;
    type D64F = CommitmentPreset<fp128::Field, GeneratedAdaptivePolicy<fp128::Profile, 64, 128>>;
    type D32OH = CommitmentPreset<fp128::Field, GeneratedAdaptivePolicy<fp128::Profile, 32, 1>>;
    type D32F = CommitmentPreset<fp128::Field, GeneratedAdaptivePolicy<fp128::Profile, 32, 128>>;

    struct Family {
        label: &'static str,
        table: GeneratedScheduleTable,
        run: fn(usize, BatchConfig) -> Result<Schedule, HachiError>,
    }

    fn families() -> Vec<Family> {
        vec![
            Family {
                label: "D128-oh",
                table: generated::fp128_d128_onehot_table(),
                run: |nv, b| find_optimal_batched_schedule::<D128OH, 128>(nv, b),
            },
            Family {
                label: "D128-f",
                table: generated::fp128_d128_full_table(),
                run: |nv, b| find_optimal_batched_schedule::<D128F, 128>(nv, b),
            },
            Family {
                label: "D64-oh",
                table: generated::fp128_d64_onehot_table(),
                run: |nv, b| find_optimal_batched_schedule::<D64OH, 64>(nv, b),
            },
            Family {
                label: "D64-f",
                table: generated::fp128_d64_full_table(),
                run: |nv, b| find_optimal_batched_schedule::<D64F, 64>(nv, b),
            },
            Family {
                label: "D32-oh",
                table: generated::fp128_d32_onehot_table(),
                run: |nv, b| find_optimal_batched_schedule::<D32OH, 32>(nv, b),
            },
            Family {
                label: "D32-f",
                table: generated::fp128_d32_full_table(),
                run: |nv, b| find_optimal_batched_schedule::<D32F, 32>(nv, b),
            },
        ]
    }

    #[test]
    fn never_worse_than_generated_tables() {
        let (mut n_total, mut n_same, mut n_better, mut n_worse) = (0, 0, 0usize, 0usize);
        let (mut total_saved, mut total_regressed) = (0i128, 0i64);

        for fam in families() {
            for nv in 1..=50usize {
                let key = GeneratedScheduleKey {
                    max_num_vars: nv,
                    num_vars: nv,
                    layout_num_claims: 1,
                    batch_num_claims: 1,
                    batch_num_commitment_groups: 1,
                    batch_num_points: 1,
                };
                let Some(entry) = generated::table_entry(fam.table, key) else {
                    continue;
                };
                n_total += 1;

                let sched = (fam.run)(nv, BatchConfig::singleton()).unwrap();
                let diff = sched.total_bytes as i64 - entry.total_bytes as i64;

                if diff < 0 {
                    n_better += 1;
                    total_saved += -(diff as i128);
                } else if diff > 0 {
                    n_worse += 1;
                    total_regressed += diff;
                    eprintln!(
                        "  WORSE {}: nv={nv} gen={} ours={} (+{diff}B)",
                        fam.label, entry.total_bytes, sched.total_bytes
                    );
                } else {
                    n_same += 1;
                }
            }
        }

        eprintln!("\n  Total: {n_total}  Same: {n_same}  Better: {n_better} (-{total_saved}B)  Worse: {n_worse} (+{total_regressed}B)");
        assert_eq!(n_worse, 0, "must never produce larger proofs");
    }

    #[test]
    fn timing_d32_onehot_nv32() {
        use std::time::Instant;

        let singleton = BatchConfig::singleton();

        // Warmup
        let _ = find_optimal_batched_schedule::<D32OH, 32>(32, singleton);

        let t = Instant::now();
        let sched = find_optimal_batched_schedule::<D32OH, 32>(32, singleton).unwrap();
        let elapsed = t.elapsed();

        eprintln!(
            "\n  D32-onehot nv=32: {} bytes, {} steps, {elapsed:?}",
            sched.total_bytes,
            sched.steps.len()
        );
    }

    // -------------------------------------------------------------------
    // Batched planner tests
    // -------------------------------------------------------------------

    #[test]
    fn batched_never_worse_than_generated_d64_onehot() {
        let table = generated::fp128_d64_onehot_table();

        for &(claims, groups, points) in &[(6, 3, 1), (6, 3, 2)] {
            let key = GeneratedScheduleKey {
                max_num_vars: 20,
                num_vars: 20,
                layout_num_claims: claims,
                batch_num_claims: claims,
                batch_num_commitment_groups: groups,
                batch_num_points: points,
            };
            let Some(entry) = generated::table_entry(table, key) else {
                continue;
            };

            let batch = BatchConfig {
                num_claims: claims,
                num_commitment_groups: groups,
                num_points: points,
            };
            let sched = find_optimal_batched_schedule::<D64OH, 64>(20, batch).unwrap();
            assert!(
                sched.total_bytes <= entry.total_bytes,
                "D64-oh batched claims={claims} groups={groups} points={points}: \
                 planner={} > generated={} (+{}B)",
                sched.total_bytes,
                entry.total_bytes,
                sched.total_bytes as i64 - entry.total_bytes as i64
            );
        }
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
}
