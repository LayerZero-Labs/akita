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
use akita_types::schedule_from_plan;
use akita_types::{
    direct_witness_bytes, level_proof_bytes, planned_next_w_len, root_current_w_len,
    scale_batched_root_layout, AjtaiKeyParams, AkitaScheduleInputs, AkitaScheduleLookupKey,
    DirectStep, DirectWitnessShape, FoldStep, LevelParams, Schedule, Step,
};

const MAX_RECURSION_DEPTH: usize = 12;

/// Root `z` protocol vectors represented by a schedule lookup key.
///
/// Incidence construction maps committed groups with identical opening-point
/// sets onto one `z`; the planner only consumes that already-projected count.
fn num_z_vectors(key: AkitaScheduleLookupKey) -> usize {
    key.num_z_vectors
}

fn derive_batched_root_level_derivation<Cfg>(
    num_vars: usize,
    root_lp: &LevelParams,
    num_claims: usize,
) -> Result<(LevelParams, LevelParams), AkitaError>
where
    Cfg: PlannerConfig,
{
    let inputs = AkitaScheduleInputs {
        num_vars,
        level: 0,
        current_w_len: root_current_w_len(root_lp),
    };
    let level_lp = scale_batched_root_layout(
        root_lp,
        num_claims,
        Cfg::planner_stage1_challenge_config(Cfg::PLANNER_D).l1_norm(),
        Cfg::planner_field_bits(),
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
    num_vars: usize,
    level: usize,
    current_w_len: usize,
    log_basis: u32,
) -> Option<CandidateLevelParams>
where
    Cfg: PlannerConfig,
{
    let inputs = AkitaScheduleInputs {
        num_vars,
        level,
        current_w_len,
    };

    let level_lp = if level == 0 {
        Cfg::planner_root_level_layout_with_log_basis(inputs, log_basis).ok()?
    } else {
        Cfg::planner_current_level_layout_with_log_basis(inputs, log_basis).ok()?
    };

    let fb = Cfg::planner_field_bits();
    let next_w_len = planned_next_w_len::<Cfg::PlannerField>(fb, &level_lp)
        .checked_mul(Cfg::planner_recursive_witness_expansion())
        .expect("recursive witness expansion overflow");
    let w_ring = next_w_len / level_lp.ring_dimension;

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

fn to_fold_step(
    c: &CandidateLevelParams,
    current_w_len: usize,
    level_bytes: usize,
    field_bits: u32,
) -> Step {
    let per_poly_fold = compute_num_digits_fold_with_claims(
        c.lp.r_vars,
        c.lp.challenge_l1_mass(),
        c.lp.log_basis,
        1,
        field_bits,
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

fn terminal_direct_witness_len<Cfg: PlannerConfig>(current_w_len: usize) -> usize {
    let expansion = Cfg::planner_recursive_witness_expansion();
    assert!(expansion > 0, "recursive witness expansion must be nonzero");
    assert_eq!(
        current_w_len % expansion,
        0,
        "terminal recursive witness length must be divisible by the extension expansion"
    );
    current_w_len / expansion
}

fn terminal_direct_witness_shape<Cfg: PlannerConfig>(
    current_w_len: usize,
    log_basis: u32,
) -> DirectWitnessShape {
    DirectWitnessShape::PackedDigits((terminal_direct_witness_len::<Cfg>(current_w_len), log_basis))
}

fn to_direct_step<Cfg: PlannerConfig>(current_w_len: usize, log_basis: u32) -> Step {
    let witness_shape = terminal_direct_witness_shape::<Cfg>(current_w_len, log_basis);
    let direct_bytes = direct_witness_bytes(Cfg::planner_field_bits(), &witness_shape);
    Step::Direct(DirectStep {
        current_w_len,
        witness_shape,
        direct_bytes,
    })
}

/// Inclusive range of `log_basis` values to search at a given state.
fn basis_range<Cfg: PlannerConfig>(
    num_vars: usize,
    level: usize,
    current_w_len: usize,
) -> std::ops::RangeInclusive<u32> {
    let (lo, hi) = Cfg::planner_log_basis_search_range(AkitaScheduleInputs {
        num_vars,
        level,
        current_w_len,
    });
    lo..=hi
}

fn level_params_from_fold_step<Cfg: PlannerConfig>(step: &FoldStep) -> LevelParams {
    debug_assert_eq!(
        Cfg::planner_stage1_challenge_config(step.params.ring_dimension).l1_norm(),
        step.params.challenge_l1_mass()
    );
    step.params.clone()
}

fn successor_level_params_from_schedule<Cfg: PlannerConfig>(
    num_vars: usize,
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
                num_vars,
                level,
                current_w_len,
            },
            step.log_basis(Cfg::planner_field_bits()),
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
    num_vars: usize,
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
        &terminal_direct_witness_shape::<Cfg>(current_w_len, current_lb),
    );
    let mut best_cost = direct_bytes;
    let mut best_schedule = vec![to_direct_step::<Cfg>(current_w_len, current_lb)];

    // Try each feasible basis for one more fold level.
    if depth <= MAX_RECURSION_DEPTH {
        for lb in basis_range::<Cfg>(num_vars, level, current_w_len) {
            if lb < current_lb {
                continue;
            }
            let Some(candidate) =
                derive_candidate_level_params::<Cfg>(num_vars, level, current_w_len, lb)
            else {
                continue;
            };

            let (suffix_cost, suffix_steps) = derive_optimal_suffix_schedule::<Cfg>(
                memo,
                num_vars,
                level + 1,
                candidate.next_w_len,
                lb,
                depth + 1,
            );
            let Ok(next_level_params) = successor_level_params_from_schedule::<Cfg>(
                num_vars,
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
                steps.push(to_fold_step(
                    &candidate,
                    current_w_len,
                    level_proof_size,
                    Cfg::planner_field_bits(),
                ));
                steps.extend(suffix_steps);
                best_schedule = steps;
            }
        }

        memo.insert(key, (best_cost, best_schedule.clone()));
    }

    (best_cost, best_schedule)
}

// -----------------------------------------------------------------------
// Key-derived root sizing
// -----------------------------------------------------------------------

/// Root-level witness ring-element count parameterized by schedule key.
///
/// ```text
///   W(lp; key) = W · 2^r · δ_open
///              + T · 2^r · n_A · δ_open
///              + Z · 2^m · δ_commit · δ_fold
///              + (n_D + n_B·Z + Z + 1 + n_A) · δ_R(b)
/// ```
fn root_w_ring_element_count<Cfg>(
    lp: &LevelParams,
    key: AkitaScheduleLookupKey,
) -> Result<usize, AkitaError>
where
    Cfg: PlannerConfig,
{
    let fb = Cfg::planner_field_bits();
    let r_decomp = compute_num_digits_full_field(fb, lp.log_basis);

    let t_vectors = key.num_t_vectors;
    let w_vectors = key.num_w_vectors;
    let z_vectors = num_z_vectors(key);

    let w_hat = w_vectors * lp.num_blocks * lp.num_digits_open;
    let t_hat = t_vectors * lp.num_blocks * lp.a_key.row_len() * lp.num_digits_open;
    let z_pre = z_vectors * lp.inner_width() * lp.num_digits_fold;
    let r_rows = lp.m_row_count(z_vectors, z_vectors);
    let r = r_rows * r_decomp;

    #[cfg(feature = "zk")]
    {
        let d_blinding = akita_types::zk::blinding_column_count_from_bits(
            lp.d_key.row_len(),
            lp.ring_dimension,
            lp.log_basis,
            fb as usize,
        );
        let b_blinding = z_vectors
            * akita_types::zk::blinding_column_count_from_bits(
                lp.b_key.row_len(),
                lp.ring_dimension,
                lp.log_basis,
                fb as usize,
            );
        Ok(w_hat + t_hat + b_blinding + d_blinding + z_pre + r)
    }
    #[cfg(not(feature = "zk"))]
    {
        Ok(w_hat + t_hat + z_pre + r)
    }
}

// -----------------------------------------------------------------------
// Key-driven root candidate + entry point
// -----------------------------------------------------------------------

/// Derive the optimal root candidate at `log_basis` for one schedule key.
///
/// Runs the full `(m, r)` block-split search using key-provided
/// `t`/`w`/`z` protocol-vector counts.
fn derive_root_candidate<Cfg>(
    num_vars: usize,
    root_w_len: usize,
    log_basis: u32,
    key: AkitaScheduleLookupKey,
) -> Result<Option<CandidateLevelParams>, AkitaError>
where
    Cfg: PlannerConfig,
{
    let inputs = AkitaScheduleInputs {
        num_vars,
        level: 0,
        current_w_len: root_w_len,
    };

    let root_lp = match Cfg::planner_root_level_layout_with_log_basis(inputs, log_basis) {
        Ok(root_lp) => root_lp,
        Err(_) => return Ok(None),
    };
    let fb = Cfg::planner_field_bits();

    let alpha = Cfg::PLANNER_D.trailing_zeros() as usize;
    let Some(reduced_vars) = num_vars.checked_sub(alpha) else {
        return Ok(None);
    };
    if reduced_vars < 1 {
        return Ok(None);
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
            fb,
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
            Cfg::planner_sis_modulus_family(),
            root_lp.a_key.row_len(),
            inner_width,
            root_lp.a_key.collision_inf(),
            d,
        ) else {
            continue;
        };
        let Ok(b_key) = AjtaiKeyParams::try_new(
            Cfg::planner_sis_modulus_family(),
            root_lp.b_key.row_len(),
            outer_width,
            root_lp.b_key.collision_inf(),
            d,
        ) else {
            continue;
        };
        let Ok(d_key) = AjtaiKeyParams::try_new(
            Cfg::planner_sis_modulus_family(),
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

        let Ok((level_lp, proof_lp)) =
            derive_batched_root_level_derivation::<Cfg>(num_vars, &candidate_lp, key.num_t_vectors)
        else {
            continue;
        };
        let raw_w_ring = root_w_ring_element_count::<Cfg>(&level_lp, key)?;
        let next_w_len = raw_w_ring
            .checked_mul(level_lp.ring_dimension)
            .and_then(|len| len.checked_mul(Cfg::planner_recursive_witness_expansion()))
            .expect("root recursive witness expansion overflow");
        let w_ring = next_w_len / level_lp.ring_dimension;

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

    Ok(best)
}

/// Consult the offline schedule tables for a pre-computed answer.
///
/// Returns `Ok(Some(schedule))` when the config ships an offline entry whose
/// [`AkitaScheduleLookupKey`] exactly matches the requested key.
fn offline_schedule_for_key<Cfg>(
    key: AkitaScheduleLookupKey,
) -> Result<Option<Schedule>, AkitaError>
where
    Cfg: PlannerConfig,
{
    Ok(Cfg::planner_schedule_plan(key)?
        .map(|plan| schedule_from_plan(&plan, Cfg::planner_field_bits())))
}

/// Find the optimal schedule for a root schedule lookup key.
///
/// **Offline fast path.** Each `(Cfg, num_vars, shape)` that ships with
/// the crate has a pre-computed entry in `Cfg::schedule_plan` (the
/// generated schedule tables in `akita-types`). Keys outside that generated
/// envelope fall back to the DP search.
///
/// # Errors
///
/// Returns an error if vector counts are invalid, if the witness length
/// overflows, or if the config's offline-table lookup fails.
pub fn find_optimal_schedule<Cfg>(key: AkitaScheduleLookupKey) -> Result<Schedule, AkitaError>
where
    Cfg: PlannerConfig,
{
    let t_vectors = key.num_t_vectors;
    let w_vectors = key.num_w_vectors;
    let z_vectors = num_z_vectors(key);
    if t_vectors == 0 || w_vectors == 0 || z_vectors == 0 {
        return Err(AkitaError::InvalidSetup(
            "schedule key planner dimensions must be at least 1".into(),
        ));
    }
    let num_vars = key.num_vars;

    if let Some(schedule) = offline_schedule_for_key::<Cfg>(key)? {
        tracing::debug!(
            num_vars,
            num_t_vectors = t_vectors,
            num_w_vectors = w_vectors,
            num_z_vectors = z_vectors,
            total_bytes = schedule.total_bytes,
            "schedule planner: served from offline schedule tables"
        );
        return Ok(schedule);
    }

    let root_w_len = 1usize
        .checked_shl(num_vars as u32)
        .ok_or_else(|| AkitaError::InvalidSetup("witness too large".into()))?;

    let fb = Cfg::planner_field_bits();
    let root_direct_shape = DirectWitnessShape::FieldElements(root_w_len);
    let mut best_cost = direct_witness_bytes(fb, &root_direct_shape);
    let mut best_steps: Vec<Step> = vec![Step::Direct(DirectStep {
        current_w_len: root_w_len,
        witness_shape: root_direct_shape,
        direct_bytes: best_cost,
    })];
    let mut memo = ScheduleMemo::new();

    for root_lb in basis_range::<Cfg>(num_vars, 0, root_w_len) {
        let Some(candidate) = derive_root_candidate::<Cfg>(num_vars, root_w_len, root_lb, key)?
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
            compute_level_proof_size::<Cfg>(&candidate, &next_level_params, z_vectors);

        let total = root_proof_size + suffix_cost;
        if total < best_cost {
            best_cost = total;
            let mut steps = Vec::with_capacity(1 + suffix_steps.len());
            steps.push(to_fold_step(
                &candidate,
                root_w_len,
                root_proof_size,
                Cfg::planner_field_bits(),
            ));
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
        num_t_vectors = t_vectors,
        num_w_vectors = w_vectors,
        num_z_vectors = z_vectors,
        total_bytes = best_cost,
        fold_levels = num_folds,
        "schedule planner: computed from scratch (no offline entry)"
    );

    Ok(Schedule {
        steps: best_steps,
        total_bytes: best_cost,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_challenges::SparseChallengeConfig;
    use akita_field::Prime128OffsetA7F7;
    use akita_types::{
        AkitaPlannedDirectStep, AkitaPlannedState, AkitaPlannedStep, AkitaSchedulePlan,
        SisModulusFamily,
    };

    #[test]
    fn planner_z_count_comes_from_schedule_key() {
        let key = AkitaScheduleLookupKey::new(2, 3, 4, 1);

        assert_eq!(key.num_t_vectors, 3);
        assert_eq!(key.num_w_vectors, 4);
        assert_eq!(num_z_vectors(key), 1);
    }

    #[derive(Clone)]
    struct OfflineOnlyConfig;

    impl PlannerConfig for OfflineOnlyConfig {
        type PlannerField = Prime128OffsetA7F7;

        const PLANNER_D: usize = 64;

        fn planner_field_bits() -> u32 {
            128
        }

        fn planner_sis_modulus_family() -> SisModulusFamily {
            SisModulusFamily::Q128
        }

        fn planner_stage1_challenge_config(_d: usize) -> SparseChallengeConfig {
            SparseChallengeConfig::Uniform {
                weight: 3,
                nonzero_coeffs: vec![-1, 1],
            }
        }

        fn planner_schedule_plan(
            key: AkitaScheduleLookupKey,
        ) -> Result<Option<AkitaSchedulePlan>, AkitaError> {
            Ok(Some(AkitaSchedulePlan {
                steps: vec![AkitaPlannedStep::Direct(AkitaPlannedDirectStep {
                    state: AkitaPlannedState {
                        level: 0,
                        current_w_len: 1usize << key.num_vars,
                        log_basis: 128,
                    },
                    witness_shape: DirectWitnessShape::FieldElements(1usize << key.num_vars),
                    direct_bytes: 16usize << key.num_vars,
                })],
                no_wrapper_bytes: 16usize << key.num_vars,
                exact_proof_bytes: 16usize << key.num_vars,
            }))
        }

        fn planner_root_level_layout_with_log_basis(
            _inputs: AkitaScheduleInputs,
            _log_basis: u32,
        ) -> Result<LevelParams, AkitaError> {
            panic!("offline test should not run DP root layout search")
        }

        fn planner_current_level_layout_with_log_basis(
            _inputs: AkitaScheduleInputs,
            _log_basis: u32,
        ) -> Result<LevelParams, AkitaError> {
            panic!("offline test should not run DP recursive layout search")
        }

        fn planner_root_level_params_for_layout_with_log_basis(
            _inputs: AkitaScheduleInputs,
            _lp: &LevelParams,
        ) -> Result<LevelParams, AkitaError> {
            panic!("offline test should not run DP root params search")
        }

        fn planner_log_basis_search_range(_inputs: AkitaScheduleInputs) -> (u32, u32) {
            panic!("offline test should not run DP basis search")
        }
    }

    #[test]
    fn planner_uses_generated_schedule_fast_path_for_all_features() {
        let key = AkitaScheduleLookupKey::new(4, 1, 1, 1);
        let schedule =
            find_optimal_schedule::<OfflineOnlyConfig>(key).expect("offline schedule lookup");

        assert_eq!(schedule.total_bytes, 256);
        assert!(matches!(schedule.steps.first(), Some(Step::Direct(_))));
    }
}
