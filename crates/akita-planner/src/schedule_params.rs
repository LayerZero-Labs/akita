//! Schedule planner that finds the global minimum proof size.
//!
//! Public entry: [`find_schedule`]. The search is `Cfg`-free: every
//! per-preset input is carried by the plain-value [`PlannerPolicy`] plus
//! the `ring_challenge_config` / `fold_challenge_shape_at_level` closures,
//! exactly the shape `crate::schedule_from_entry` already consumes. This keeps the
//! DP a pure function of `(policy, key)` so `akita-config` can call it
//! directly on a schedule-table miss without a dependency cycle.

use akita_challenges::{SparseChallengeConfig, TensorChallengeShape};
use akita_error::AkitaError;
use akita_types::layout::digit_math::optimal_m_r_split;
use akita_types::sis::{
    decomposed_s_block_ring_count, decomposed_t_ring_count, decomposed_w_ring_count,
    fold_witness_digit_plan, num_digits_open, num_digits_s_commit, num_digits_setup_prefix_commit,
    rounded_up_collision_inf_norm, rounded_up_role_a_inf_norm, AjtaiKeyParams, FoldChallengeNorms,
    FoldWitnessLinfCapConfig, FoldWitnessNorms, SisTableKey,
};
use akita_types::{
    direct_witness_bytes, extension_opening_reduction_level_bytes, level_proof_bytes,
    padded_setup_prefix_len, segment_typed_witness_shape_from_groups,
    w_ring_element_count_for_chunks, AkitaScheduleInputs, ChunkedWitnessCfg, CleartextWitnessShape,
    CommitmentRingDims, DecompositionParams, DirectStep, FoldStep, LevelParams,
    OpeningClaimsLayout, PolynomialGroupLayout, PrecommittedGroupParams, PrecommittedLevelParams,
    RelationMatrixRowLayout, Schedule, SetupContributionMode, Step, SETUP_OFFLOAD_D_SETUP,
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

    let min_r_vars: usize = if reduced_vars >= 3 { 1 } else { 0 };
    let max_r_vars: usize = (reduced_vars - 1).min(usize::BITS as usize - 1);

    // Chunk count of the witness committed at the root fold (absolute level 0).
    let root_num_chunks = policy.chunks_at_level(0);

    let (min_log_basis, max_log_basis) = policy.basis_range;
    for candidate_log_basis in min_log_basis..=max_log_basis {
        for r_vars in (min_r_vars..=max_r_vars).rev() {
            let Some(candidate_params) = scalar_root_fold_level_params_candidate(
                policy,
                &ring_challenge_cfg,
                key.num_vars(),
                key.num_polynomials(),
                candidate_log_basis,
                r_vars,
                fold_challenge_shape,
            )?
            else {
                continue;
            };

            let next_withness_len_impl = |layout| -> Result<usize, AkitaError> {
                let rings = w_ring_element_count_for_chunks(
                    field_bits,
                    &candidate_params,
                    key.num_polynomials(),
                    layout,
                    root_num_chunks,
                )?;
                rings.checked_mul(policy.ring_dimension).ok_or_else(|| {
                    AkitaError::InvalidSetup("root next witness length overflow".into())
                })
            };
            let next_w_len = next_withness_len_impl(RelationMatrixRowLayout::WithDBlock)?;
            let next_w_len_terminal =
                next_withness_len_impl(RelationMatrixRowLayout::WithoutDBlock)?;
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
