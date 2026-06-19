//! Phase-0 offline byte model: per-fold ring-dimension ladders.
//!
//! Searches schedules that may use different power-of-two ring degrees per fold
//! (e.g. 128 → 64 → 32), scoring with the same [`level_proof_bytes`] helpers as
//! [`crate::find_schedule`].
//!
//! ## Divisibility when halving `D`
//!
//! Flat witness length after a fold is `ring_count × D_fold` (total digit slots).
//! If `D_next` divides `D_fold` (hold `D` or halve), any length divisible by
//! `D_fold` is divisible by `D_next`, so the recursive DP gate
//! `current_witness_len % ring_d == 0` never fails on a halving transition.

use std::collections::{BTreeMap, HashMap};

use akita_challenges::TensorChallengeShape;
use akita_field::AkitaError;
use akita_types::sis::{
    decomposed_s_block_ring_count, decomposed_t_ring_count, decomposed_w_ring_count,
    min_secure_rank, num_digits_open, num_digits_s_commit, rounded_up_collision_norm_s,
    rounded_up_collision_norm_t, rounded_up_collision_norm_w, AjtaiKeyParams,
    FoldWitnessLinfCapConfig,
};
use akita_types::{
    direct_witness_bytes, level_proof_bytes, w_ring_element_count_with_counts_for_layout_bits,
    AkitaScheduleInputs, AkitaScheduleLookupKey, CleartextWitnessShape, DecompositionParams,
    DirectStep, FoldStep, LevelParams, MRowLayout, Schedule, Step,
};

use crate::schedule_params::{
    derive_candidate_level_params, extension_opening_reduction_level_bytes, multi_tiered_keys,
    terminal_direct_suffix_cost, MAX_RECURSION_DEPTH, RingChallengeConfigFn,
};
use crate::PlannerPolicy;

/// Default fp128 onehot ladder candidates (high → low).
pub const FP128_ONEHOT_LADDER_DIMS: &[usize] = &[128, 64, 32];

/// Ring degrees allowed at the level after `current_d` when only holding or
/// stepping down along a power-of-two ladder.
pub fn monotone_ring_dim_successors(current_d: usize, allowed: &[usize]) -> Vec<usize> {
    allowed
        .iter()
        .copied()
        .filter(|&d| d <= current_d && current_d.is_multiple_of(d))
        .collect()
}

/// Extract the per-fold ring dimensions from a schedule's fold steps.
pub fn fold_ring_dimensions(schedule: &Schedule) -> Vec<usize> {
    schedule
        .steps
        .iter()
        .filter_map(|step| match step {
            Step::Fold(f) => Some(f.params.ring_dimension),
            Step::Direct(_) => None,
        })
        .collect()
}

/// True when adjacent fold levels use different ring dimensions.
pub fn is_mixed_ring_dimension_ladder(dims: &[usize]) -> bool {
    dims.windows(2).any(|w| w[0] != w[1])
}

/// Fold-level proof bytes (excludes terminal direct tail).
pub fn schedule_fold_bytes(schedule: &Schedule) -> usize {
    schedule
        .steps
        .iter()
        .filter_map(|s| match s {
            Step::Fold(f) => Some(f.level_bytes),
            Step::Direct(_) => None,
        })
        .sum()
}

/// Terminal direct witness bytes across all direct steps.
pub fn schedule_terminal_bytes(schedule: &Schedule) -> usize {
    schedule
        .steps
        .iter()
        .filter_map(|s| match s {
            Step::Direct(d) => Some(d.direct_bytes),
            Step::Fold(_) => None,
        })
        .sum()
}

#[derive(Clone, Debug)]
struct LadderFoldSuffix {
    total_bytes: usize,
    first_fold_params: LevelParams,
    steps: Vec<Step>,
}

#[derive(Clone, Copy)]
struct LadderDirectSuffix {
    current_w_len: usize,
}

#[derive(Clone)]
struct LadderSuffixResult {
    best_direct: Option<LadderDirectSuffix>,
    best_fold_per_lb: BTreeMap<u32, LadderFoldSuffix>,
}

impl LadderSuffixResult {
    fn is_empty(&self) -> bool {
        self.best_direct.is_none() && self.best_fold_per_lb.is_empty()
    }
}

type LadderMemo = HashMap<(usize, usize, usize, usize, u32), LadderSuffixResult>;

#[derive(Clone, Copy)]
struct LadderSuffixCtx<'a> {
    policy: &'a PlannerPolicy,
    ring_challenge_config: RingChallengeConfigFn<'a>,
    allowed_ring_dims: &'a [usize],
    num_vars: usize,
    key: AkitaScheduleLookupKey,
}

fn derive_ladder_suffix(
    ctx: &LadderSuffixCtx<'_>,
    memo: &mut LadderMemo,
    level: usize,
    current_d: usize,
    current_witness_len: usize,
    current_witness_len_terminal: usize,
    current_lb: u32,
    depth: usize,
) -> Result<LadderSuffixResult, AkitaError> {
    let LadderSuffixCtx {
        policy,
        ring_challenge_config,
        allowed_ring_dims,
        num_vars,
        key,
    } = *ctx;
    let memo_key = (
        level,
        current_d,
        current_witness_len,
        current_witness_len_terminal,
        current_lb,
    );
    if depth <= MAX_RECURSION_DEPTH {
        if let Some(cached) = memo.get(&memo_key) {
            return Ok(cached.clone());
        }
    }

    let best_direct = if derive_candidate_level_params(
        policy,
        ring_challenge_config,
        current_d,
        current_witness_len,
        current_lb,
    )?
    .is_some()
    {
        Some(LadderDirectSuffix {
            current_w_len: current_witness_len_terminal,
        })
    } else {
        None
    };

    if depth > MAX_RECURSION_DEPTH {
        let result = LadderSuffixResult {
            best_direct,
            best_fold_per_lb: BTreeMap::new(),
        };
        memo.insert(memo_key, result.clone());
        return Ok(result);
    }

    let mut best_fold_per_lb: BTreeMap<u32, LadderFoldSuffix> = BTreeMap::new();
    let (min_log_basis, max_log_basis) = policy.basis_range;
    for lb in min_log_basis..=max_log_basis {
        if lb < current_lb {
            continue;
        }
        let Some((candidate_params, next_witness_len, next_witness_len_terminal)) =
            derive_candidate_level_params(
                policy,
                ring_challenge_config,
                current_d,
                current_witness_len,
                lb,
            )?
        else {
            continue;
        };

        let mut best_for_this_lb: Option<(usize, Vec<Step>)> = None;
        let try_update =
            |total: usize, steps: Vec<Step>, slot: &mut Option<(usize, Vec<Step>)>| {
                if slot.as_ref().map(|(c, _)| total < *c).unwrap_or(true) {
                    *slot = Some((total, steps));
                }
            };

        for next_d in monotone_ring_dim_successors(current_d, allowed_ring_dims) {
            let suffix = derive_ladder_suffix(
                ctx,
                memo,
                level + 1,
                next_d,
                next_witness_len,
                next_witness_len_terminal,
                lb,
                depth + 1,
            )?;
            let Ok(eor_bytes) = extension_opening_reduction_level_bytes(
                policy,
                AkitaScheduleLookupKey::singleton(num_vars),
                level,
                current_witness_len,
            ) else {
                continue;
            };
            let field_bits = policy.decomposition.field_bits();
            let chal_bits = field_bits * policy.chal_ext_degree as u32;

            if let Some(direct_suffix) = &suffix.best_direct {
                let (direct_step, suffix_cost) = terminal_direct_suffix_cost(
                    direct_suffix.current_w_len,
                    &candidate_params,
                    field_bits,
                    key,
                    level,
                    lb,
                )?;
                let level_proof_size = level_proof_bytes(
                    field_bits,
                    chal_bits,
                    &candidate_params,
                    None,
                    next_witness_len_terminal,
                    1,
                    MRowLayout::WithoutDBlock,
                ) + eor_bytes;
                let total = level_proof_size + suffix_cost;
                let steps = vec![
                    Step::Fold(FoldStep {
                        params: candidate_params.clone(),
                        current_w_len: current_witness_len,
                        next_w_len: next_witness_len_terminal,
                        level_bytes: level_proof_size,
                    }),
                    Step::Direct(direct_step),
                ];
                try_update(total, steps, &mut best_for_this_lb);
            }

            for suffix_fold in suffix.best_fold_per_lb.values() {
                let level_proof_size = level_proof_bytes(
                    field_bits,
                    chal_bits,
                    &candidate_params,
                    Some(&suffix_fold.first_fold_params),
                    next_witness_len,
                    1,
                    MRowLayout::WithDBlock,
                ) + eor_bytes;
                let total = level_proof_size + suffix_fold.total_bytes;
                let mut steps = Vec::with_capacity(1 + suffix_fold.steps.len());
                steps.push(Step::Fold(FoldStep {
                    params: candidate_params.clone(),
                    current_w_len: current_witness_len,
                    next_w_len: next_witness_len,
                    level_bytes: level_proof_size,
                }));
                steps.extend(suffix_fold.steps.iter().cloned());
                try_update(total, steps, &mut best_for_this_lb);
            }
        }

        if let Some((total_bytes, steps)) = best_for_this_lb {
            best_fold_per_lb.insert(
                lb,
                LadderFoldSuffix {
                    total_bytes,
                    // Match [`find_schedule`]: the parent's `next_lp` is this level's
                    // `candidate_params`, not the grandchild's first fold.
                    first_fold_params: candidate_params.clone(),
                    steps,
                },
            );
        }
    }

    let result = LadderSuffixResult {
        best_direct,
        best_fold_per_lb,
    };
    memo.insert(memo_key, result.clone());
    Ok(result)
}

/// Find the minimum proof-size schedule over monotone ring-dimension ladders.
///
/// `allowed_ring_dims` must be strictly decreasing powers of two (e.g.
/// `[128, 64, 32]`). Each fold may hold its ring degree or step down to any
/// allowed divisor of the current degree.
///
/// # Errors
///
/// Same failure modes as [`crate::find_schedule`].
pub fn find_ladder_schedule(
    key: AkitaScheduleLookupKey,
    policy: &PlannerPolicy,
    allowed_ring_dims: &[usize],
    ring_challenge_config: impl Fn(usize) -> Result<akita_challenges::SparseChallengeConfig, AkitaError>,
    fold_shape: impl Fn(AkitaScheduleInputs) -> TensorChallengeShape,
) -> Result<Schedule, AkitaError> {
    if allowed_ring_dims.is_empty() {
        return Err(AkitaError::InvalidSetup(
            "ladder search requires at least one ring dimension".into(),
        ));
    }
    for window in allowed_ring_dims.windows(2) {
        let [high, low] = window else { continue };
        if low >= high || !high.is_multiple_of(*low) {
            return Err(AkitaError::InvalidSetup(format!(
                "allowed_ring_dims must be strictly decreasing powers of two, got {high} then {low}"
            )));
        }
    }

    let ring_challenge_config: RingChallengeConfigFn<'_> = &ring_challenge_config;
    let fold_shape = &fold_shape;
    let suffix_ctx = LadderSuffixCtx {
        policy,
        ring_challenge_config,
        allowed_ring_dims,
        num_vars: key.num_vars,
        key,
    };

    let t_vectors = key.num_t_vectors;
    let w_vectors = key.num_w_vectors;
    let z_vectors = key.num_z_vectors;
    if t_vectors == 0 || w_vectors == 0 || z_vectors == 0 {
        return Err(AkitaError::InvalidSetup(
            "schedule key planner dimensions must be at least 1".into(),
        ));
    }
    if z_vectors != 1 {
        return Err(AkitaError::InvalidSetup(
            "schedule key must describe one shared opening point and one public row".into(),
        ));
    }

    let witness_len = 1usize
        .checked_shl(key.num_vars as u32)
        .ok_or_else(|| AkitaError::InvalidSetup("witness too large".into()))?;

    let field_bits = policy.decomposition.field_bits();
    let root_witness_shape = CleartextWitnessShape::FieldElements(witness_len);
    let mut best_cost = direct_witness_bytes(field_bits, &root_witness_shape);
    let mut best_steps: Vec<Step> = vec![Step::Direct(DirectStep {
        current_w_len: witness_len,
        witness_shape: root_witness_shape,
        direct_bytes: best_cost,
        params: None,
    })];
    let mut memo = LadderMemo::new();

    let fold_challenge_shape = fold_shape(AkitaScheduleInputs {
        num_vars: key.num_vars,
        level: 0,
        current_w_len: witness_len,
    });

    for &root_d in allowed_ring_dims {
        let Ok(stage1_config) = ring_challenge_config(root_d) else {
            continue;
        };
        let alpha = (root_d as u32).trailing_zeros() as usize;
        let reduced_vars = key.num_vars.saturating_sub(alpha);
        if reduced_vars == 0 {
            continue;
        }

        let min_r_vars: usize = if reduced_vars >= 3 { 1 } else { 0 };
        let max_r_vars: usize = (reduced_vars - 1).min(usize::BITS as usize - 1);
        let (min_log_basis, max_log_basis) = policy.basis_range;

        for candidate_log_basis in min_log_basis..=max_log_basis {
            let level_decomp = DecompositionParams {
                log_basis: candidate_log_basis,
                ..policy.decomposition
            };
            let num_digits_commit = num_digits_s_commit(level_decomp, true);
            let num_digits_open = num_digits_open(level_decomp);

            for r_vars in (min_r_vars..=max_r_vars).rev() {
                let Some(num_blocks) = 1usize.checked_shl(r_vars as u32) else {
                    continue;
                };
                let m_vars = reduced_vars - r_vars;
                let Some(block_len) = 1usize.checked_shl(m_vars as u32) else {
                    continue;
                };

                let family = policy.sis_family;
                let d = root_d;
                let Some(norm_s) = rounded_up_collision_norm_s(
                    family,
                    d,
                    level_decomp,
                    &stage1_config,
                    fold_challenge_shape,
                    true,
                    policy.onehot_chunk_size,
                    policy.ring_subfield_norm_bound,
                    r_vars,
                    t_vectors,
                ) else {
                    continue;
                };
                let Some(width_s) = decomposed_s_block_ring_count(block_len, num_digits_commit)
                else {
                    continue;
                };
                let Some(n_a) = min_secure_rank(family, d as u32, norm_s, width_s as u64) else {
                    continue;
                };
                let a_key = AjtaiKeyParams::try_new(family, n_a, width_s, norm_s, d)?;
                let Some(norm_t) = rounded_up_collision_norm_t(family, d, candidate_log_basis)
                else {
                    continue;
                };
                let Some(width_t) =
                    decomposed_t_ring_count(n_a, num_digits_open, num_blocks, t_vectors)
                else {
                    continue;
                };
                let Some(n_b) = min_secure_rank(family, d as u32, norm_t, width_t as u64) else {
                    continue;
                };
                let b_key = AjtaiKeyParams::try_new(family, n_b, width_t, norm_t, d)?;
                let Some(norm_w) = rounded_up_collision_norm_w(family, d, candidate_log_basis)
                else {
                    continue;
                };
                let Some(width_w) =
                    decomposed_w_ring_count(num_digits_open, num_blocks, t_vectors)
                else {
                    continue;
                };
                let Some(n_d) = min_secure_rank(family, d as u32, norm_w, width_w as u64) else {
                    continue;
                };
                let d_key = AjtaiKeyParams::try_new(family, n_d, width_w, norm_w, d)?;

                let onehot_chunk_size = if policy.decomposition.log_commit_bound == 1 {
                    policy.onehot_chunk_size
                } else {
                    0
                };
                let (tier_split, b_key, f_key) = if policy.tiered {
                    let Some(a_matrix_size) = a_key.row_len().checked_mul(a_key.col_len()) else {
                        continue;
                    };
                    multi_tiered_keys(
                        a_matrix_size,
                        &b_key,
                        num_digits_open,
                        candidate_log_basis,
                        d,
                    )?
                } else {
                    (1, b_key, None)
                };

                let candidate_params = LevelParams {
                    ring_dimension: root_d,
                    log_basis: candidate_log_basis,
                    a_key,
                    b_key,
                    d_key,
                    num_blocks,
                    block_len,
                    m_vars,
                    r_vars,
                    stage1_config: stage1_config.clone(),
                    fold_challenge_shape,
                    num_digits_commit,
                    num_digits_open,
                    onehot_chunk_size,
                    tier_split,
                    f_key,
                    fold_linf_cap_config: FoldWitnessLinfCapConfig::worst_case_beta_only(),
                    num_digits_fold_one: 1,
                    field_bits_hint: 0,
                    cached_num_digits_fold_claims: 0,
                    cached_num_digits_fold_value: 1,
                }
                .with_fold_linf_cap_config(field_bits, key.num_t_vectors);

                let next_witness_len_impl = |layout| -> Result<usize, AkitaError> {
                    let rings = w_ring_element_count_with_counts_for_layout_bits(
                        field_bits,
                        &candidate_params,
                        1,
                        key.num_t_vectors,
                        key.num_w_vectors,
                        key.num_z_vectors,
                        layout,
                    )?;
                    rings
                        .checked_mul(root_d)
                        .ok_or_else(|| AkitaError::InvalidSetup("root next witness length overflow".into()))
                };
                let next_w_len = next_witness_len_impl(MRowLayout::WithDBlock)?;
                let next_w_len_terminal = next_witness_len_impl(MRowLayout::WithoutDBlock)?;
                let initial_witness_len_bits = witness_len
                    .checked_mul(field_bits as usize)
                    .ok_or_else(|| AkitaError::InvalidSetup("root witness bit length overflow".into()))?;
                if next_w_len
                    .checked_mul(candidate_log_basis as usize)
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup("root next witness bit length overflow".into())
                    })?
                    >= initial_witness_len_bits
                {
                    continue;
                }

                let Ok(eor_bytes) =
                    extension_opening_reduction_level_bytes(policy, key, 0, witness_len)
                else {
                    continue;
                };

                for next_d in monotone_ring_dim_successors(root_d, allowed_ring_dims) {
                    let suffix = derive_ladder_suffix(
                        &suffix_ctx,
                        &mut memo,
                        1,
                        next_d,
                        next_w_len,
                        next_w_len_terminal,
                        candidate_log_basis,
                        0,
                    )?;
                    if suffix.is_empty() {
                        continue;
                    }

                    if let Some(direct_suffix) = suffix.best_direct {
                        let (direct_step, suffix_cost) = terminal_direct_suffix_cost(
                            direct_suffix.current_w_len,
                            &candidate_params,
                            field_bits,
                            key,
                            0,
                            candidate_log_basis,
                        )?;
                        let root_proof_size = level_proof_bytes(
                            field_bits,
                            field_bits * policy.chal_ext_degree as u32,
                            &candidate_params,
                            None,
                            next_w_len_terminal,
                            z_vectors,
                            MRowLayout::WithoutDBlock,
                        ) + eor_bytes;
                        let total = root_proof_size + suffix_cost;
                        if total < best_cost {
                            best_cost = total;
                            best_steps = vec![
                                Step::Fold(FoldStep {
                                    params: candidate_params.clone(),
                                    current_w_len: witness_len,
                                    next_w_len: next_w_len_terminal,
                                    level_bytes: root_proof_size,
                                }),
                                Step::Direct(direct_step),
                            ];
                        }
                    }

                    for suffix_fold in suffix.best_fold_per_lb.values() {
                        let root_proof_size = level_proof_bytes(
                            field_bits,
                            field_bits * policy.chal_ext_degree as u32,
                            &candidate_params,
                            Some(&suffix_fold.first_fold_params),
                            next_w_len,
                            z_vectors,
                            MRowLayout::WithDBlock,
                        ) + eor_bytes;
                        let total = root_proof_size + suffix_fold.total_bytes;
                        if total < best_cost {
                            best_cost = total;
                            let mut steps = Vec::with_capacity(1 + suffix_fold.steps.len());
                            steps.push(Step::Fold(FoldStep {
                                params: candidate_params.clone(),
                                current_w_len: witness_len,
                                next_w_len,
                                level_bytes: root_proof_size,
                            }));
                            steps.extend(suffix_fold.steps.iter().cloned());
                            best_steps = steps;
                        }
                    }
                }
            }
        }
    }

    Ok(Schedule {
        steps: best_steps,
        total_bytes: best_cost,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn halving_ring_dim_preserves_flat_witness_divisibility() {
        for d_high in [128usize, 64] {
            for d_low in [64usize, 32, 16] {
                if !d_high.is_multiple_of(d_low) {
                    continue;
                }
                let flat_len = 1_024 * d_high;
                assert!(
                    flat_len.is_multiple_of(d_low),
                    "{flat_len} divisible by {d_high} but not {d_low}"
                );
            }
        }
    }

    #[test]
    fn monotone_successors_are_divisors() {
        let allowed = [128, 64, 32];
        assert_eq!(monotone_ring_dim_successors(128, &allowed), vec![128, 64, 32]);
        assert_eq!(monotone_ring_dim_successors(64, &allowed), vec![64, 32]);
        assert_eq!(monotone_ring_dim_successors(32, &allowed), vec![32]);
    }
}
