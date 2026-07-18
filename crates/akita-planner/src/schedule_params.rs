//! Schedule planner that finds the global minimum proof size.
//!
//! Public entry: [`find_schedule`]. The search is `Cfg`-free: every
//! per-preset input is carried by the plain-value [`PlannerPolicy`] plus
//! the `ring_challenge_config` / `fold_challenge_shape_at_level` closures,
//! exactly the shape `crate::schedule_from_entry` already consumes. This keeps the
//! DP a pure function of `(policy, key)` so `akita-config` can call it
//! directly on a schedule-table miss without a dependency cycle.

use akita_challenges::{SparseChallengeConfig, TensorChallengeShape};
use akita_field::AkitaError;
use akita_types::layout::digit_math::optimal_block_geometry_split;
use akita_types::sis::{
    decomposed_s_block_ring_count, decomposed_t_ring_count, decomposed_w_ring_count,
    fold_witness_digit_plan, num_digits_open, num_digits_s_commit, num_digits_setup_prefix_commit,
    rounded_up_collision_inf_norm, rounded_up_role_a_inf_norm, AjtaiKeyParams, FoldChallengeNorms,
    FoldWitnessLinfCapConfig, FoldWitnessNorms, SisTableKey,
};
use akita_types::{
    direct_witness_bytes, extension_opening_reduction_level_bytes,
    intermediate_w_ring_element_count_for_chunks, level_proof_bytes, padded_setup_prefix_len,
    segment_typed_witness_shape_from_groups, AkitaScheduleInputs, ChunkedWitnessCfg,
    CleartextWitnessShape, CommitmentRingDims, DecompositionParams, DirectStep, FoldStep,
    LevelParams, LevelParamsLike, OpeningClaimsLayout, PolynomialGroupLayout,
    PrecommittedGroupParams, PrecommittedLevelParams, RelationMatrixRowLayout, Schedule,
    SetupContributionMode, Step, WitnessLayout, SETUP_OFFLOAD_D_SETUP,
};

use crate::PlannerPolicy;

mod candidate;
mod suffix_dp;

pub use candidate::suffix_opening_layout;
pub(crate) use candidate::{
    compute_root_direct_level_params, derive_candidate_level_params, planned_next_witness_len,
    scalar_root_fold_level_params_candidate, terminal_witness_shape_for_opening_layout,
};
use suffix_dp::try_terminal_direct_suffix_cost;
pub(crate) use suffix_dp::{derive_optimal_suffix_schedule, ScheduleMemo, SuffixCtx, SuffixState};

/// Validate the policy's multi-chunk witness settings at a planner entry point.
///
/// Layout-only rules live on [`ChunkedWitnessCfg::validate`]; the recursion-depth
/// bound (which needs the planner-private [`MAX_RECURSION_DEPTH`]) is enforced
/// here so `akita-types` stays free of planner internals.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidSetup`] for an invalid `ChunkedWitnessCfg`, or
/// `num_activated_levels` beyond the planner recursion cap. Verifier-reachable: never panics.
pub(crate) fn validate_policy_witness_chunk(policy: &PlannerPolicy) -> Result<(), AkitaError> {
    let mc = policy.witness_chunk;
    mc.validate()?;
    if mc.num_activated_levels > MAX_RECURSION_DEPTH {
        return Err(AkitaError::InvalidSetup(format!(
            "num_activated_levels={} exceeds the planner recursion cap {MAX_RECURSION_DEPTH}",
            mc.num_activated_levels
        )));
    }
    Ok(())
}

/// Stage-1 sparse-challenge closure shared by the planner entry points.
pub(crate) type RingChallengeConfigFn<'a> =
    &'a dyn Fn(usize) -> Result<akita_challenges::SparseChallengeConfig, AkitaError>;

pub(crate) type LayoutCandidateScore = (usize, usize, usize, usize);

/// Resolve the tensor low length independently from the num_positions_per_block split.
/// A tensor-enabled policy selects the shape family; the planner enumerates
/// every power-of-two low length through the Boolean block-index domain size and chooses
/// the minimum exact `Q + ceil(F/Q)` verifier work.
pub(crate) fn optimize_fold_challenge_shape(
    requested: TensorChallengeShape,
    num_live_blocks: usize,
) -> Result<TensorChallengeShape, AkitaError> {
    if num_live_blocks == 0 {
        return Err(AkitaError::InvalidSetup(
            "fold-shape optimization requires a positive num_live_blocks".to_string(),
        ));
    }
    if matches!(requested, TensorChallengeShape::Flat) {
        return Ok(TensorChallengeShape::Flat);
    }

    let capacity = num_live_blocks.checked_next_power_of_two().ok_or_else(|| {
        AkitaError::InvalidSetup("tensor low-length capacity overflow".to_string())
    })?;
    let mut best = None;
    let mut low_len = 1usize;
    loop {
        let high_len = num_live_blocks.div_ceil(low_len);
        let work = high_len
            .checked_add(low_len)
            .ok_or_else(|| AkitaError::InvalidSetup("tensor verifier-work overflow".to_string()))?;
        if best.is_none_or(|(best_work, best_low)| (work, low_len) < (best_work, best_low)) {
            best = Some((work, low_len));
        }
        if low_len == capacity {
            break;
        }
        low_len = low_len.checked_mul(2).ok_or_else(|| {
            AkitaError::InvalidSetup("tensor low-length enumeration overflow".to_string())
        })?;
    }
    let (_, fold_low_len) = best.ok_or_else(|| {
        AkitaError::InvalidSetup("tensor low-length enumeration was empty".to_string())
    })?;
    Ok(TensorChallengeShape::Tensor { fold_low_len })
}

/// Combine exact physical width, challenge-factor work, chunk evaluator work,
/// and load imbalance when comparing `M` candidates. All terms count ring or
/// scalar work units; exact physical width remains an explicit tie-breaker.
pub(crate) fn layout_candidate_score(
    physical_width: usize,
    num_live_blocks: usize,
    num_chunks: usize,
    fold_shape: TensorChallengeShape,
) -> Result<LayoutCandidateScore, AkitaError> {
    let challenge_work = match fold_shape {
        TensorChallengeShape::Flat => num_live_blocks,
        TensorChallengeShape::Tensor { fold_low_len } => fold_low_len
            .checked_add(num_live_blocks.div_ceil(fold_low_len))
            .ok_or_else(|| AkitaError::InvalidSetup("challenge-work overflow".to_string()))?,
    };
    let chunk_ranges = WitnessLayout::resolve_chunk_block_ranges(num_live_blocks, num_chunks)?;
    let min_load = chunk_ranges
        .iter()
        .map(|range| range.len())
        .min()
        .ok_or_else(|| AkitaError::InvalidSetup("balanced chunk geometry is empty".to_string()))?;
    let max_load = chunk_ranges
        .iter()
        .map(|range| range.len())
        .max()
        .ok_or_else(|| AkitaError::InvalidSetup("balanced chunk geometry is empty".to_string()))?;
    let chunk_work = num_live_blocks;
    let imbalance = max_load - min_load;
    let combined = physical_width
        .checked_add(challenge_work)
        .and_then(|cost| cost.checked_add(chunk_work))
        .and_then(|cost| cost.checked_add(imbalance))
        .ok_or_else(|| AkitaError::InvalidSetup("layout candidate score overflow".to_string()))?;
    Ok((combined, physical_width, chunk_work, imbalance))
}

// Suffix-DP depth cap. Schedules in our working parameter range never need
// more than this many recursive fold levels; deeper search only blows up
// memo state without changing emitted tables.
pub(crate) const MAX_RECURSION_DEPTH: usize = 12;

/// Find the optimal schedule for a root schedule lookup key under `policy`.
///
/// Runs an exhaustive DP that minimizes proof size. The result is a pure,
/// deterministic function of `(policy, key)` (plus the `ring_challenge_config` /
/// `fold_challenge_shape_at_level` closures, which presets derive from the same hooks the
/// generated tables were emitted from), so the prover and verifier
/// regenerate identical schedules on a table miss.
///
/// # Errors
///
/// Returns an error if vector counts are invalid or if the witness length
/// overflows. The function never panics on malformed input — it is
/// verifier-reachable and audited under the no-panic contract.
pub fn find_schedule(
    key: PolynomialGroupLayout,
    policy: &PlannerPolicy,
    ring_challenge_config: impl Fn(usize) -> Result<akita_challenges::SparseChallengeConfig, AkitaError>,
    fold_challenge_shape_at_level: impl Fn(AkitaScheduleInputs) -> TensorChallengeShape,
) -> Result<Schedule, AkitaError> {
    find_schedule_inner(
        key,
        policy,
        ring_challenge_config,
        fold_challenge_shape_at_level,
    )
}

fn find_schedule_inner(
    key: PolynomialGroupLayout,
    policy: &PlannerPolicy,
    ring_challenge_config: impl Fn(usize) -> Result<akita_challenges::SparseChallengeConfig, AkitaError>,
    fold_challenge_shape_at_level: impl Fn(AkitaScheduleInputs) -> TensorChallengeShape,
) -> Result<Schedule, AkitaError> {
    let ring_challenge_config: RingChallengeConfigFn<'_> = &ring_challenge_config;
    let fold_shape = &fold_challenge_shape_at_level;

    key.validate()?;
    validate_policy_witness_chunk(policy)?;
    let ring_challenge_cfg = ring_challenge_config(policy.ring_dimension)?;
    let suffix_ctx = SuffixCtx {
        policy,
        ring_challenge_cfg: &ring_challenge_cfg,
        fold_challenge_shape_at_level: fold_shape,
        num_vars: key.num_vars(),
        key,
    };

    key.validate()?;
    validate_policy_witness_chunk(policy)?;
    if policy.recursive_setup_planning {
        return Err(AkitaError::InvalidSetup(
            "recursive setup planning requires the grouped-batch scheduler".to_string(),
        ));
    }
    let witness_len = 1usize
        .checked_shl(key.num_vars() as u32)
        .ok_or_else(|| AkitaError::InvalidSetup("witness too large".into()))?;

    let field_bits = policy.decomposition.field_bits();

    let root_witness_shape = CleartextWitnessShape::FieldElements(witness_len);
    let mut best_cost = direct_witness_bytes(field_bits, &root_witness_shape);
    let fold_challenge_shape = fold_shape(AkitaScheduleInputs {
        num_vars: key.num_vars(),
        level: 0,
        current_w_len: witness_len,
    });
    // The level-0 fold-challenge shape and the `num_claims = num_polynomials`
    // batch factor are folded directly into the committed B/D widths, so a table
    // miss reproduces the exact root commit layout the table-hit expansion
    // (`expand_to_level_params`) builds — no separate per-claim-then-scale
    // pass. `Ok(None)` is the uncommittable (large-`num_vars`) edge.
    let root_direct_commit_params = compute_root_direct_level_params(
        policy,
        &ring_challenge_cfg,
        key.num_vars(),
        policy.decomposition.log_basis,
        fold_challenge_shape,
        key.num_polynomials(),
    )?;
    let mut best_steps: Vec<Step> = vec![Step::Direct(DirectStep {
        current_w_len: witness_len,
        witness_shape: root_witness_shape,
        direct_bytes: best_cost,
        params: root_direct_commit_params,
    })];
    let mut memo = ScheduleMemo::new();

    let alpha = (policy.ring_dimension as u32).trailing_zeros() as usize;
    let reduced_vars = key.num_vars().saturating_sub(alpha);

    if reduced_vars == 0 {
        return Ok(Schedule {
            steps: best_steps,
            total_bytes: best_cost,
        });
    }

    let min_block_index_bits: usize = if reduced_vars >= 3 { 1 } else { 0 };
    let max_block_index_bits: usize = (reduced_vars - 1).min(usize::BITS as usize - 1);

    // Chunk count of the witness committed at the root fold (absolute level 0).
    let root_num_chunks = policy.chunks_at_level(0);

    let (configured_min_log_basis, max_log_basis) = policy.basis_range;
    let min_log_basis = configured_min_log_basis
        .max(policy.decomposition.log_basis)
        .max(if policy.decomposition.field_bits() < 128 {
            5
        } else {
            0
        });
    for candidate_log_basis in min_log_basis..=max_log_basis {
        for block_index_bits in (min_block_index_bits..=max_block_index_bits).rev() {
            let Some(candidate_params) = scalar_root_fold_level_params_candidate(
                policy,
                &ring_challenge_cfg,
                key.num_vars(),
                key.num_polynomials(),
                candidate_log_basis,
                block_index_bits,
                fold_challenge_shape,
            )?
            else {
                continue;
            };

            let next_w_len = intermediate_w_ring_element_count_for_chunks(
                field_bits,
                &candidate_params,
                key.num_polynomials(),
                root_num_chunks,
            )?
            .checked_mul(policy.ring_dimension)
            .ok_or_else(|| AkitaError::InvalidSetup("root next witness length overflow".into()))?;
            let terminal_shape = segment_typed_witness_shape_from_groups(
                &candidate_params,
                field_bits,
                [(
                    &candidate_params as &dyn akita_types::LevelParamsLike,
                    key.num_polynomials(),
                    key.num_polynomials(),
                    1,
                )],
            )?;
            let next_w_len_terminal = terminal_shape.logical_num_elems();
            let initial_witness_len_bits = witness_len
                .checked_mul(field_bits as usize)
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("root witness bit length overflow".into())
                })?;
            if next_w_len
                .checked_mul(candidate_log_basis as usize)
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("root next witness bit length overflow".into())
                })?
                >= initial_witness_len_bits
            {
                continue;
            }

            let suffix = derive_optimal_suffix_schedule(
                &suffix_ctx,
                &mut memo,
                SuffixState {
                    level: 1,
                    current_witness_len: next_w_len,
                    current_witness_len_terminal: next_w_len_terminal,
                    current_lb: candidate_log_basis,
                    incoming_setup_prefix: None,
                },
                0,
            )?;
            if suffix.is_empty() {
                continue;
            }
            let Ok(eor_bytes) = extension_opening_reduction_level_bytes(
                policy.decomposition.field_bits() * policy.chal_ext_degree as u32,
                policy.claim_ext_degree,
                0,
                key,
                witness_len,
            ) else {
                continue;
            };

            // Branch A: suffix at level 1 is a Direct
            if let Some(direct_suffix) = suffix.best_direct {
                if let Some((direct_step, suffix_cost)) = try_terminal_direct_suffix_cost(
                    direct_suffix.current_w_len,
                    &candidate_params,
                    field_bits,
                    key,
                    0,
                    None,
                )? {
                    let root_proof_size = level_proof_bytes(
                        field_bits,
                        field_bits * policy.chal_ext_degree as u32,
                        &candidate_params,
                        None,
                        next_w_len_terminal,
                        1,
                        RelationMatrixRowLayout::WithoutDBlock,
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
            }
            // Branch B: suffix at level 1 is a Fold
            for suffix_fold in suffix.best_fold_per_lb.values() {
                let root_proof_size = level_proof_bytes(
                    field_bits,
                    field_bits * policy.chal_ext_degree as u32,
                    &candidate_params,
                    Some(&suffix_fold.first_fold_params),
                    next_w_len,
                    1,
                    RelationMatrixRowLayout::WithDBlock,
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

    Ok(Schedule {
        steps: best_steps,
        total_bytes: best_cost,
    })
}

#[cfg(test)]
mod geometry_tests {
    use super::*;

    #[test]
    fn tensor_low_length_is_selected_independently() {
        assert_eq!(
            optimize_fold_challenge_shape(TensorChallengeShape::Tensor { fold_low_len: 1 }, 13,)
                .unwrap(),
            TensorChallengeShape::Tensor { fold_low_len: 4 },
        );
    }

    #[test]
    fn balanced_chunk_geometry_prices_exact_work_and_residual_imbalance() {
        let flat = TensorChallengeShape::Flat;
        assert_eq!(
            layout_candidate_score(100, 13, 3, flat).unwrap(),
            (127, 100, 13, 1)
        );
        assert_eq!(
            layout_candidate_score(100, 12, 3, flat).unwrap(),
            (124, 100, 12, 0)
        );
    }
}
