use super::config::{
    compute_num_digits, compute_num_digits_fold, optimal_m_r_split_with_params, CommitmentConfig,
    DecompositionParams, HachiCommitmentLayout,
};
use crate::algebra::SparseChallengeConfig;
use crate::error::HachiError;
use std::collections::HashMap;
use std::fmt::Write;

/// Public inputs that deterministically select one level's active Hachi params.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HachiScheduleInputs {
    /// Root polynomial variable count.
    pub max_num_vars: usize,
    /// Fold level, where `0` is the original polynomial.
    pub level: usize,
    /// Current witness length in field elements before this level runs.
    pub current_w_len: usize,
}

/// Runtime source of truth for one Hachi level.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HachiLevelParams {
    /// Ring dimension at this level.
    pub d: usize,
    /// Gadget base exponent.
    pub log_basis: u32,
    /// Active inner Ajtai rank.
    pub n_a: usize,
    /// Active outer commitment rank.
    pub n_b: usize,
    /// Active D-matrix rank.
    pub n_d: usize,
    /// Conservative sparse-challenge L1 mass used by folded-norm bounds.
    pub challenge_l1_mass: usize,
    /// Stage-1 challenge family sampled at this level.
    pub stage1_config: SparseChallengeConfig,
}

impl HachiLevelParams {
    /// Total number of quotient / relation rows in `M`.
    pub fn m_row_count(&self) -> usize {
        self.n_d + self.n_b + 2 + self.n_a
    }
}

fn with_log_basis(mut decomp: DecompositionParams, log_basis: u32) -> DecompositionParams {
    decomp.log_basis = log_basis;
    decomp
}

pub(crate) fn main_level_decomposition_from_root(
    root_decomp: DecompositionParams,
    log_basis: u32,
) -> DecompositionParams {
    with_log_basis(root_decomp, log_basis)
}

fn main_level_decomposition<Cfg: CommitmentConfig>(
    params: &HachiLevelParams,
) -> DecompositionParams {
    main_level_decomposition_from_root(Cfg::decomposition(), params.log_basis)
}

pub(crate) fn recursive_level_decomposition_from_root(
    root_decomp: DecompositionParams,
    log_basis: u32,
) -> DecompositionParams {
    let parent_open = root_decomp
        .log_open_bound
        .unwrap_or(root_decomp.log_commit_bound);
    DecompositionParams {
        log_basis,
        log_commit_bound: log_basis,
        log_open_bound: Some(parent_open),
    }
}

pub(crate) fn recursive_level_decomposition<Cfg: CommitmentConfig>(
    params: &HachiLevelParams,
) -> DecompositionParams {
    recursive_level_decomposition_from_root(Cfg::decomposition(), params.log_basis)
}

fn layout_from_params(
    m_vars: usize,
    r_vars: usize,
    params: &HachiLevelParams,
    decomp: DecompositionParams,
) -> Result<HachiCommitmentLayout, HachiError> {
    let depth_commit = compute_num_digits(decomp.log_commit_bound, decomp.log_basis);
    let open_bound = decomp.log_open_bound.unwrap_or(decomp.log_commit_bound);
    let depth_open = compute_num_digits(open_bound, decomp.log_basis);
    let depth_fold = compute_num_digits_fold(r_vars, params.challenge_l1_mass, decomp.log_basis);
    HachiCommitmentLayout::new_with_decomp(
        m_vars,
        r_vars,
        params.n_a,
        depth_commit,
        depth_open,
        depth_fold,
        decomp.log_basis,
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct PlannerState {
    level: usize,
    current_w_len: usize,
    log_basis: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HachiPlannedLevel {
    pub inputs: HachiScheduleInputs,
    pub params: HachiLevelParams,
    pub layout: HachiCommitmentLayout,
    pub next_inputs: HachiScheduleInputs,
    pub next_level_log_basis: u32,
    pub level_bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HachiPlannedState {
    pub level: usize,
    pub current_w_len: usize,
    pub log_basis: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Deterministic level-by-level schedule selected from public inputs.
pub struct HachiSchedulePlan {
    /// Planned witness states, including the terminal direct-tail/handoff state.
    pub states: Vec<HachiPlannedState>,
    /// Planned Hachi proof levels in execution order.
    pub levels: Vec<HachiPlannedLevel>,
    /// Total proof bytes excluding the outer proof wrapper.
    pub no_wrapper_bytes: usize,
    /// Total proof bytes including the wrapper used by `HachiProof`.
    pub exact_proof_bytes: usize,
}

impl HachiSchedulePlan {
    /// Return the final witness state after all planned Hachi levels.
    ///
    /// # Panics
    ///
    /// Panics if the schedule was constructed without any states.
    pub fn terminal_state(&self) -> HachiPlannedState {
        *self
            .states
            .last()
            .expect("planned schedule always contains at least one state")
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PlannedSuffix {
    levels: Vec<HachiPlannedLevel>,
    no_wrapper_bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PlannerConfig {
    max_num_vars: usize,
    min_log_basis: u32,
    max_log_basis: u32,
    field_bits: u32,
    half_field_bound: u128,
}

fn field_bits(root_decomp: DecompositionParams) -> u32 {
    root_decomp
        .log_open_bound
        .unwrap_or(root_decomp.log_commit_bound)
}

fn field_bytes(field_bits: u32) -> usize {
    (field_bits as usize).div_ceil(8)
}

fn flat_ring_vec_bytes(ring_len: usize, ring_dim: usize, elem_bytes: usize) -> usize {
    4 + 8 + ring_len * ring_dim * elem_bytes
}

fn packed_digits_bytes(num_elems: usize, bits_per_elem: u32) -> usize {
    8 + 1 + (num_elems * bits_per_elem as usize).div_ceil(8)
}

fn compressed_unipoly_bytes(degree: usize, elem_bytes: usize) -> usize {
    8 + degree * elem_bytes
}

fn sumcheck_bytes(rounds: usize, degree: usize, elem_bytes: usize) -> usize {
    8 + rounds * compressed_unipoly_bytes(degree, elem_bytes)
}

pub(crate) fn recursive_r_decomp_levels_for_bound(
    field_bits: u32,
    half_field_bound: u128,
    log_basis: u32,
) -> usize {
    let bits = field_bits as usize;
    let lb = log_basis as usize;
    let mut levels = compute_num_digits(field_bits, log_basis);
    if levels == 0 {
        levels = 1;
    }

    let total_bits = levels * lb;
    if total_bits <= bits {
        let b = 1u128 << log_basis;
        let half_b_minus_1 = b / 2 - 1;
        let b_minus_1 = b - 1;
        let mut b_pow = 1u128;
        for _ in 0..levels {
            b_pow = b_pow.saturating_mul(b);
        }
        let max_positive = half_b_minus_1.saturating_mul((b_pow - 1) / b_minus_1);
        if max_positive < half_field_bound {
            levels += 1;
        }
    }

    levels
}

pub(crate) fn planned_w_ring_element_count(
    field_bits: u32,
    half_field_bound: u128,
    level_params: &HachiLevelParams,
    layout: HachiCommitmentLayout,
) -> usize {
    let w_hat_count = layout.num_blocks * layout.num_digits_open;
    let t_hat_count = layout.num_blocks * level_params.n_a * layout.num_digits_open;
    let z_pre_count = layout.inner_width * layout.num_digits_fold;
    let r_count = level_params.m_row_count()
        * recursive_r_decomp_levels_for_bound(field_bits, half_field_bound, layout.log_basis);
    w_hat_count + t_hat_count + z_pre_count + r_count
}

pub(crate) fn planned_next_w_len(
    field_bits: u32,
    half_field_bound: u128,
    level_params: &HachiLevelParams,
    layout: HachiCommitmentLayout,
) -> usize {
    planned_w_ring_element_count(field_bits, half_field_bound, level_params, layout)
        * level_params.d
}

fn sumcheck_rounds(level_d: usize, next_w_len: usize) -> usize {
    let num_l = level_d.trailing_zeros() as usize;
    let num_ring_elems = next_w_len / level_d;
    let num_u = num_ring_elems.next_power_of_two().trailing_zeros() as usize;
    num_u + num_l
}

fn hachi_level_proof_bytes(
    field_bits: u32,
    level_params: &HachiLevelParams,
    layout: HachiCommitmentLayout,
    next_level_params: &HachiLevelParams,
    next_w_len: usize,
) -> usize {
    let elem_bytes = field_bytes(field_bits);
    let y_bytes = flat_ring_vec_bytes(1, level_params.d, elem_bytes);
    let v_bytes = flat_ring_vec_bytes(level_params.n_d, level_params.d, elem_bytes);
    let next_commit_bytes =
        flat_ring_vec_bytes(next_level_params.n_b, next_level_params.d, elem_bytes);
    let next_eval_bytes = elem_bytes;
    let rounds = sumcheck_rounds(level_params.d, next_w_len);
    let b = 1usize << layout.log_basis;
    let stage1_degree = b / 2 + 1;

    // Every level now uses the same two-stage norm-check body, even for b = 4.
    y_bytes
        + v_bytes
        + sumcheck_bytes(rounds, stage1_degree, elem_bytes)
        + elem_bytes
        + sumcheck_bytes(rounds, 3, elem_bytes)
        + next_commit_bytes
        + next_eval_bytes
}

fn current_level_layout_with_log_basis<Cfg: CommitmentConfig>(
    inputs: HachiScheduleInputs,
    log_basis: u32,
) -> Result<(HachiLevelParams, HachiCommitmentLayout), HachiError> {
    let params = Cfg::level_params_with_log_basis(inputs, log_basis);
    let layout = if inputs.level == 0 {
        let alpha = params.d.trailing_zeros() as usize;
        let reduced_vars = inputs.max_num_vars.checked_sub(alpha).ok_or_else(|| {
            HachiError::InvalidSetup("max_num_vars is smaller than alpha".to_string())
        })?;
        if reduced_vars == 0 {
            return Err(HachiError::InvalidSetup(
                "max_num_vars must leave at least one outer variable".to_string(),
            ));
        }
        let decomp = main_level_decomposition_from_root(Cfg::decomposition(), log_basis);
        let (m_vars, r_vars) = optimal_m_r_split_with_params(&params, decomp, reduced_vars);
        layout_from_params(m_vars, r_vars, &params, decomp)?
    } else {
        hachi_recursive_level_layout_from_params::<Cfg>(&params, inputs.current_w_len)?
    };
    Ok((params, layout))
}

fn best_recursive_suffix<Cfg: CommitmentConfig>(
    cfg: PlannerConfig,
    memo: &mut HashMap<PlannerState, PlannedSuffix>,
    state: PlannerState,
) -> Result<PlannedSuffix, HachiError> {
    if let Some(existing) = memo.get(&state) {
        return Ok(existing.clone());
    }

    let mut best = PlannedSuffix {
        levels: Vec::new(),
        no_wrapper_bytes: packed_digits_bytes(state.current_w_len, state.log_basis),
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
                    let mut levels = Vec::with_capacity(suffix.levels.len() + 1);
                    levels.push(HachiPlannedLevel {
                        inputs,
                        params: params.clone(),
                        layout,
                        next_inputs,
                        next_level_log_basis: next_log_basis,
                        level_bytes,
                    });
                    levels.extend(suffix.levels);
                    best = PlannedSuffix {
                        levels,
                        no_wrapper_bytes: candidate_bytes,
                    };
                }
            }
        }
    }

    memo.insert(state, best.clone());
    Ok(best)
}

pub(crate) fn planned_schedule<Cfg: CommitmentConfig>(
    max_num_vars: usize,
    min_log_basis: u32,
    max_log_basis: u32,
) -> Result<HachiSchedulePlan, HachiError> {
    let root_current_w_len = 1usize
        .checked_shl(max_num_vars as u32)
        .ok_or_else(|| HachiError::InvalidSetup("root witness length overflow".to_string()))?;
    let root_inputs = HachiScheduleInputs {
        max_num_vars,
        level: 0,
        current_w_len: root_current_w_len,
    };
    let cfg = PlannerConfig {
        max_num_vars,
        min_log_basis,
        max_log_basis,
        field_bits: field_bits(Cfg::decomposition()),
        half_field_bound: Cfg::planner_half_field_bound(),
    };
    let mut memo = HashMap::new();
    let mut best: Option<PlannedSuffix> = None;

    for root_log_basis in min_log_basis..=max_log_basis {
        let Ok((root_params, root_layout)) =
            current_level_layout_with_log_basis::<Cfg>(root_inputs, root_log_basis)
        else {
            continue;
        };
        let next_w_len = planned_next_w_len(
            cfg.field_bits,
            cfg.half_field_bound,
            &root_params,
            root_layout,
        );

        let next_level = 1usize;
        let next_inputs = HachiScheduleInputs {
            max_num_vars,
            level: next_level,
            current_w_len: next_w_len,
        };
        for next_log_basis in root_log_basis.max(min_log_basis)..=max_log_basis {
            let next_level_params = Cfg::level_params_with_log_basis(next_inputs, next_log_basis);
            let level_bytes = hachi_level_proof_bytes(
                cfg.field_bits,
                &root_params,
                root_layout,
                &next_level_params,
                next_w_len,
            );
            let mut levels = Vec::new();
            levels.push(HachiPlannedLevel {
                inputs: root_inputs,
                params: root_params.clone(),
                layout: root_layout,
                next_inputs,
                next_level_log_basis: next_log_basis,
                level_bytes,
            });
            let candidate_bytes = if next_w_len < root_inputs.current_w_len {
                let suffix = best_recursive_suffix::<Cfg>(
                    cfg,
                    &mut memo,
                    PlannerState {
                        level: next_level,
                        current_w_len: next_w_len,
                        log_basis: next_log_basis,
                    },
                )?;
                let suffix_bytes = suffix.no_wrapper_bytes;
                levels.extend(suffix.levels);
                level_bytes + suffix_bytes
            } else {
                level_bytes + packed_digits_bytes(next_w_len, next_log_basis)
            };
            if best
                .as_ref()
                .is_none_or(|existing| candidate_bytes < existing.no_wrapper_bytes)
            {
                best = Some(PlannedSuffix {
                    levels,
                    no_wrapper_bytes: candidate_bytes,
                });
            }
        }
    }

    let best = best.ok_or_else(|| {
        HachiError::InvalidSetup("adaptive schedule search found no valid root level".to_string())
    })?;

    let mut states = Vec::with_capacity(best.levels.len() + 1);
    let first = best
        .levels
        .first()
        .expect("adaptive schedule always contains a root level");
    states.push(HachiPlannedState {
        level: first.inputs.level,
        current_w_len: first.inputs.current_w_len,
        log_basis: first.params.log_basis,
    });
    for level in &best.levels {
        states.push(HachiPlannedState {
            level: level.next_inputs.level,
            current_w_len: level.next_inputs.current_w_len,
            log_basis: level.next_level_log_basis,
        });
    }

    Ok(HachiSchedulePlan {
        states,
        levels: best.levels,
        no_wrapper_bytes: best.no_wrapper_bytes,
        exact_proof_bytes: best.no_wrapper_bytes + 4,
    })
}

pub(crate) fn planned_log_basis_at_level<Cfg: CommitmentConfig>(
    inputs: HachiScheduleInputs,
    min_log_basis: u32,
    max_log_basis: u32,
) -> Result<u32, HachiError> {
    let schedule = planned_schedule::<Cfg>(inputs.max_num_vars, min_log_basis, max_log_basis)?;
    let state = schedule
        .states
        .get(inputs.level)
        .copied()
        .unwrap_or_else(|| schedule.terminal_state());
    debug_assert_eq!(
        state.level,
        inputs.level.min(schedule.terminal_state().level)
    );
    if inputs.level > 0 {
        debug_assert_eq!(state.current_w_len, inputs.current_w_len);
    }
    Ok(state.log_basis)
}

pub(crate) fn planned_schedule_key<Cfg: CommitmentConfig>(
    max_num_vars: usize,
    min_log_basis: u32,
    max_log_basis: u32,
) -> Result<String, HachiError> {
    let schedule = planned_schedule::<Cfg>(max_num_vars, min_log_basis, max_log_basis)?;
    let mut key = String::from("planner_v2");
    for state in schedule.states {
        let _ = write!(key, "_l{}b{}", state.level, state.log_basis);
    }
    Ok(key)
}

/// Derive the root level's active params and layout.
///
/// # Errors
///
/// Returns an error if the root variable split is invalid or overflows.
pub fn hachi_root_level_layout<Cfg: CommitmentConfig>(
    max_num_vars: usize,
) -> Result<(HachiLevelParams, HachiCommitmentLayout), HachiError> {
    let params = Cfg::level_params(HachiScheduleInputs {
        max_num_vars,
        level: 0,
        current_w_len: 1usize.checked_shl(max_num_vars as u32).unwrap_or(0),
    });
    let alpha = params.d.trailing_zeros() as usize;
    let reduced_vars = max_num_vars.checked_sub(alpha).ok_or_else(|| {
        HachiError::InvalidSetup("max_num_vars is smaller than alpha".to_string())
    })?;
    if reduced_vars == 0 {
        return Err(HachiError::InvalidSetup(
            "max_num_vars must leave at least one outer variable".to_string(),
        ));
    }
    let decomp = main_level_decomposition::<Cfg>(&params);
    let (m_vars, r_vars) = optimal_m_r_split_with_params(&params, decomp, reduced_vars);
    let layout = layout_from_params(m_vars, r_vars, &params, decomp)?;
    Ok((params, layout))
}

/// Derive a recursive `w`-opening level's active params and layout.
///
/// # Errors
///
/// Returns an error if the recursive layout derivation overflows.
pub fn hachi_level_layout<Cfg: CommitmentConfig>(
    inputs: HachiScheduleInputs,
) -> Result<(HachiLevelParams, HachiCommitmentLayout), HachiError> {
    let params = Cfg::level_params(inputs);
    let layout = hachi_recursive_level_layout_from_params::<Cfg>(&params, inputs.current_w_len)?;
    Ok((params, layout))
}

/// Derive a recursive `w`-opening layout from the active level params.
///
/// # Errors
///
/// Returns an error if the witness length is incompatible with `params.d` or if
/// the recursive layout derivation overflows.
pub fn hachi_recursive_level_layout_from_params<Cfg: CommitmentConfig>(
    params: &HachiLevelParams,
    current_w_len: usize,
) -> Result<HachiCommitmentLayout, HachiError> {
    if current_w_len % params.d != 0 {
        return Err(HachiError::InvalidInput(format!(
            "witness length {current_w_len} is not divisible by D={}",
            params.d
        )));
    }
    let num_ring_elems = current_w_len / params.d;
    let total = num_ring_elems.next_power_of_two().max(1);
    let alpha = params.d.trailing_zeros() as usize;
    let reduced_vars = total.trailing_zeros() as usize;
    let max_num_vars = reduced_vars + alpha;
    let decomp = recursive_level_decomposition::<Cfg>(params);
    let (m_vars, r_vars) = optimal_m_r_split_with_params(params, decomp, reduced_vars);
    let layout = layout_from_params(m_vars, r_vars, params, decomp)?;
    debug_assert_eq!(layout.m_vars + layout.r_vars + alpha, max_num_vars);
    Ok(layout)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::Prime128M8M4M1M0;
    use crate::algebra::{CyclotomicRing, SparseChallengeConfig};
    use crate::primitives::serialization::{Compress, HachiSerialize};
    use crate::protocol::commitment::{
        Fp128AdaptiveBoundedCommitmentConfig, Fp128AdaptiveOneHotCommitmentConfig,
    };
    use crate::protocol::proof::{FlatRingVec, HachiLevelProof};
    use crate::protocol::ring_switch::w_ring_element_count;
    use crate::protocol::sumcheck::{CompressedUniPoly, SumcheckProof};
    use crate::FieldCore;

    type F = Prime128M8M4M1M0;

    fn dummy_sumcheck(rounds: usize, degree: usize) -> SumcheckProof<F> {
        SumcheckProof {
            round_polys: (0..rounds)
                .map(|_| CompressedUniPoly {
                    coeffs_except_linear_term: vec![F::zero(); degree],
                })
                .collect(),
        }
    }

    fn assert_plan_matches_runtime_w_sizes<Cfg: CommitmentConfig>(max_num_vars: usize) {
        let plan = Cfg::schedule_plan(max_num_vars)
            .expect("planner should succeed")
            .expect("config should provide a planner");
        for level in &plan.levels {
            let runtime_next_w_len =
                w_ring_element_count::<Prime128M8M4M1M0>(&level.params, level.layout)
                    * level.params.d;
            assert_eq!(
                runtime_next_w_len, level.next_inputs.current_w_len,
                "planner/runtime next_w_len mismatch at level {} for max_num_vars={max_num_vars}",
                level.inputs.level
            );
        }
    }

    #[test]
    fn adaptive_bounded_plan_matches_runtime_next_w_len() {
        for max_num_vars in [14, 20, 30] {
            assert_plan_matches_runtime_w_sizes::<Fp128AdaptiveBoundedCommitmentConfig<128>>(
                max_num_vars,
            );
        }
    }

    #[test]
    fn adaptive_onehot_plan_matches_runtime_next_w_len() {
        for max_num_vars in [15, 30, 44] {
            assert_plan_matches_runtime_w_sizes::<Fp128AdaptiveOneHotCommitmentConfig>(
                max_num_vars,
            );
        }
    }

    #[test]
    fn planned_level_bytes_match_two_stage_payload_at_all_bases() {
        const D: usize = 64;
        let stage1_config = SparseChallengeConfig::Uniform {
            weight: 3,
            nonzero_coeffs: vec![-1, 1],
        };
        let next_level_params = HachiLevelParams {
            d: D,
            log_basis: 2,
            n_a: 2,
            n_b: 3,
            n_d: 2,
            challenge_l1_mass: stage1_config.l1_mass(),
            stage1_config: stage1_config.clone(),
        };
        let next_w_len = D * 8;

        for log_basis in 2..=5 {
            let level_params = HachiLevelParams {
                d: D,
                log_basis,
                n_a: 2,
                n_b: 2,
                n_d: 2,
                challenge_l1_mass: stage1_config.l1_mass(),
                stage1_config: stage1_config.clone(),
            };
            let layout = HachiCommitmentLayout {
                m_vars: 0,
                r_vars: 0,
                num_blocks: 1,
                block_len: 1,
                inner_width: 1,
                outer_width: 1,
                d_matrix_width: 1,
                num_digits_commit: 1,
                num_digits_open: 1,
                num_digits_fold: 1,
                log_basis,
            };
            let rounds = sumcheck_rounds(D, next_w_len);
            let stage1_degree = (1usize << log_basis) / 2 + 1;
            let next_commitment = FlatRingVec::from_ring_elems(&vec![
                CyclotomicRing::<F, D>::zero();
                next_level_params.n_b
            ]);
            let level_proof = HachiLevelProof::new_two_stage::<D>(
                CyclotomicRing::<F, D>::zero(),
                vec![CyclotomicRing::<F, D>::zero(); level_params.n_d],
                dummy_sumcheck(rounds, stage1_degree),
                F::zero(),
                dummy_sumcheck(rounds, 3),
                next_commitment,
                F::zero(),
            );

            assert_eq!(
                hachi_level_proof_bytes(128, &level_params, layout, &next_level_params, next_w_len),
                level_proof.serialized_size(Compress::No),
                "planned level bytes should match the serialized two-stage body at log_basis={log_basis}"
            );
        }
    }
}
