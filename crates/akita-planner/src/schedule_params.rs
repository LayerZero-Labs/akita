//! FoldSchedule planner that finds the global minimum proof size. Recursive
//! grouped scheduling additionally minimizes the first direct setup footprint
//! before proof size.
//!
//! Scalar schedule search behind the public grouped gate. The search is
//! `Cfg`-free: every
//! per-preset input is carried by the plain-value [`PlannerPolicy`] plus
//! the `ring_challenge_config` / `fold_challenge_shape_at_level` closures,
//! exactly the shape generated catalog emission consumes. This keeps the DP a
//! pure function of `(policy, key)` for offline table generation.

use akita_challenges::{SparseChallengeConfig, TensorChallengeShape};
use akita_field::AkitaError;
use akita_types::sis::{
    decomposed_s_block_ring_count, decomposed_t_ring_count, decomposed_w_ring_count,
    fold_witness_digit_plan, num_digits_inner, num_digits_open, num_digits_setup_prefix_commit,
    rounded_up_collision_inf_norm, rounded_up_role_a_inf_norm, FoldChallengeNorms,
    FoldWitnessLinfCapConfig, FoldWitnessNorms, InnerCommitMatrixParams, OpenCommitMatrixParams,
    OuterCommitMatrixParams, SisTableKey,
};
use akita_types::{
    active_setup_field_len, level_proof_bytes, padded_setup_prefix_len,
    try_extension_opening_reduction_level_bytes, AkitaScheduleInputs, CommitmentRingDims,
    CommittedGroupParams, DecompositionParams, FoldSchedule, OpeningClaimsLayout,
    PlannedFoldSchedule, PolynomialGroupLayout, PrecommittedGroupDescriptor,
    PrecommittedLevelParams, WitnessLayout,
};

use crate::PlannerPolicy;

mod candidate;
mod mixed_search;
mod setup_score;
mod suffix_dp;
#[cfg(feature = "test-support")]
#[path = "schedule_params/tests/test_support.rs"]
pub(crate) mod test_support;
#[cfg(all(test, feature = "catalog-gen"))]
#[path = "schedule_params/tests/unpruned_search.rs"]
mod unpruned_search;

pub(crate) use akita_schedules::planner_support::{
    checked_power_of_two_vars, grouped_segment_rings, materialize_candidate_schedule,
    optimize_fold_challenge_shape, planned_next_witness_len, stage3_payload_bytes_for_successor,
    validate_policy, CandidateFoldStep, CandidateTerminalResponse,
};
pub use akita_schedules::suffix_opening_layout;
#[cfg(feature = "test-support")]
pub use candidate::derive_setup_prefix_group;
pub(crate) use candidate::{
    derive_candidate_level_params, derive_candidate_level_params_all_splits,
    scalar_root_fold_level_params_candidate,
};
pub(crate) use mixed_search::find_schedule_mixed_ring;
pub(crate) use setup_score::{
    level_setup_field_elements, terminal_setup_field_elements, MixedScore,
};
pub(crate) use suffix_dp::{derive_optimal_suffix_schedule, ScheduleMemo, SuffixCtx, SuffixState};

const MIXED_SEARCH_FOLD_LEVELS: usize = 2;
const MIXED_SEARCH_SUFFIX_RING_DIMENSION: usize = 64;

#[derive(Clone, Debug)]
pub(crate) struct ScheduleCandidate {
    pub(crate) first_direct_setup_field_len: Option<usize>,
    pub(crate) total_bytes: usize,
    pub(crate) setup_field_elements: usize,
    pub(crate) folds: Vec<CandidateFoldStep>,
    pub(crate) terminal: CandidateTerminalResponse,
}

impl ScheduleCandidate {
    pub(crate) fn first_fold_params(&self) -> Option<&CommittedGroupParams> {
        self.folds.first().map(|fold| &fold.params)
    }

    pub(crate) fn first_direct_setup_field_len_or_max(&self) -> usize {
        self.first_direct_setup_field_len.unwrap_or(usize::MAX)
    }
}

/// Stage-1 sparse-challenge closure shared by the planner entry points.
pub(crate) type RingChallengeConfigFn<'a> =
    &'a dyn Fn(usize) -> Result<akita_challenges::SparseChallengeConfig, AkitaError>;

pub(crate) type LayoutCandidateScore = (usize, usize, usize, usize);

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

fn componentwise_dimensions_at_most(
    dimensions: CommitmentRingDims,
    ceiling: CommitmentRingDims,
) -> bool {
    dimensions.d_a() <= ceiling.d_a()
        && dimensions.d_b() <= ceiling.d_b()
        && dimensions.d_d() <= ceiling.d_d()
}

pub(crate) fn find_schedule_singular(
    key: PolynomialGroupLayout,
    policy: &PlannerPolicy,
    ring_challenge_config: impl Fn(usize) -> Result<akita_challenges::SparseChallengeConfig, AkitaError>,
    fold_challenge_shape_at_level: impl Fn(AkitaScheduleInputs) -> TensorChallengeShape,
) -> Result<PlannedFoldSchedule, AkitaError> {
    key.validate()?;
    validate_policy(policy)?;
    if policy.recursive_setup_planning {
        return Err(AkitaError::InvalidSetup(
            "recursive setup planning requires the grouped-batch scheduler".to_string(),
        ));
    }
    let ring_challenge_config: RingChallengeConfigFn<'_> = &ring_challenge_config;
    let fold_shape = &fold_challenge_shape_at_level;

    let default_ring_challenge_cfg = ring_challenge_config(policy.ring_dimension)?;
    let suffix_ctx = SuffixCtx {
        policy,
        default_ring_challenge_cfg: &default_ring_challenge_cfg,
        ring_challenge_config,
        fold_challenge_shape_at_level: fold_shape,
        num_vars: key.num_vars(),
        key,
        setup_field_budget: None,
        root_lookup_key: None,
        level_zero_is_root: true,
    };
    let witness_len = 1usize
        .checked_shl(key.num_vars() as u32)
        .ok_or_else(|| AkitaError::InvalidSetup("witness too large".into()))?;

    if let crate::SelectionPolicyId::MinRootRankThenPayloadWithinSlack { slack_permille } =
        policy.selection_policy
    {
        return find_rank_aware_singular(
            key,
            policy,
            ring_challenge_config,
            fold_shape,
            &suffix_ctx,
            witness_len,
            slack_permille,
        );
    }

    let mut memo = ScheduleMemo::new();
    let suffix = derive_optimal_suffix_schedule(
        &suffix_ctx,
        &mut memo,
        SuffixState {
            level: 0,
            current_witness_len: witness_len,
            current_lb: 0,
            incoming_setup_prefix: None,
        },
        0,
    )?;
    let best = match policy.selection_policy {
        crate::SelectionPolicyId::MinEstimatedProofPayload => suffix
            .best_by_payload_per_lb
            .values()
            .min_by_key(|candidate| candidate.total_bytes),
        crate::SelectionPolicyId::MinFirstDirectSetupThenPayloadWithinSupportedEnvelope {
            ..
        } => suffix
            .best_by_first_direct_setup_per_lb
            .values()
            .min_by_key(|candidate| {
                (
                    candidate.first_direct_setup_field_len_or_max(),
                    candidate.total_bytes,
                )
            }),
        crate::SelectionPolicyId::MinRootRankThenPayloadWithinSlack { .. } => {
            return Err(AkitaError::InvalidSetup(
                "rank-aware singular planning did not use the root candidate search".to_string(),
            ));
        }
        crate::SelectionPolicyId::MinSetupMatrixFieldElementsThenProofPayload => {
            return Err(AkitaError::UnsupportedSchedule(
                "mixed ring-dimension selection is not supported for singular schedules"
                    .to_string(),
            ));
        }
    }
    .cloned();

    let Some(best) = best else {
        return Err(AkitaError::UnsupportedSchedule(format!(
            "no schedule with at least two folds for num_vars={}, num_polynomials={}",
            key.num_vars(),
            key.num_polynomials()
        )));
    };
    materialize_candidate_schedule(
        best.total_bytes,
        best.setup_field_elements,
        policy.ring_dimension,
        best.first_direct_setup_field_len,
        best.folds,
        best.terminal,
    )
}

fn find_rank_aware_singular(
    key: PolynomialGroupLayout,
    policy: &PlannerPolicy,
    ring_challenge_config: RingChallengeConfigFn<'_>,
    fold_shape: &dyn Fn(AkitaScheduleInputs) -> TensorChallengeShape,
    suffix_ctx: &SuffixCtx<'_>,
    witness_len: usize,
    slack_permille: u32,
) -> Result<PlannedFoldSchedule, AkitaError> {
    let dimensions = CommitmentRingDims::uniform(policy.ring_dimension);
    let ring_challenge_cfg = ring_challenge_config(dimensions.d_a())?;
    let alpha = dimensions.d_a().trailing_zeros() as usize;
    let reduced_vars = key.num_vars().saturating_sub(alpha);
    if reduced_vars == 0 {
        return Err(AkitaError::UnsupportedSchedule(format!(
            "num_vars={} does not exceed log2(ring_dimension)={alpha}",
            key.num_vars()
        )));
    }

    let field_bits = policy.decomposition.field_bits();
    let challenge_field_bits = policy.challenge_field_bits()?;
    let initial_witness_len_bits = witness_len
        .checked_mul(field_bits as usize)
        .ok_or_else(|| AkitaError::InvalidSetup("root witness bit length overflow".into()))?;
    let opening_layout = OpeningClaimsLayout::new(key.num_vars(), key.num_polynomials())?;
    let requested_fold_shape = fold_shape(AkitaScheduleInputs {
        num_vars: key.num_vars(),
        level: 0,
        input_witness_len: witness_len,
    });
    let eor_bytes = try_extension_opening_reduction_level_bytes(
        challenge_field_bits,
        policy.claim_ext_degree,
        0,
        key,
        witness_len,
        policy.ring_dimension,
    )?
    .ok_or_else(|| AkitaError::UnsupportedSchedule("root EOR geometry is unsupported".into()))?;
    let min_block_index_bits = if reduced_vars >= 3 { 1 } else { 0 };
    let max_block_index_bits = (reduced_vars - 1).min(usize::BITS as usize - 1);
    let mut memo = ScheduleMemo::new();
    let mut candidates = Vec::new();

    let (min_log_basis, max_log_basis) = policy.log_basis_search_range_at_level(0);
    for log_basis in min_log_basis..=max_log_basis {
        for block_index_bits in (min_block_index_bits..=max_block_index_bits).rev() {
            let Some(candidate_params) = scalar_root_fold_level_params_candidate(
                policy,
                &ring_challenge_cfg,
                dimensions,
                key.num_vars(),
                key.num_polynomials(),
                log_basis,
                block_index_bits,
                requested_fold_shape,
            )?
            else {
                continue;
            };
            let output_witness_len = planned_next_witness_len(
                field_bits,
                &candidate_params,
                key.num_polynomials(),
                policy.chunks_at_level(0),
            )?;
            if output_witness_len
                .checked_mul(log_basis as usize)
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("root next witness bit length overflow".into())
                })?
                >= initial_witness_len_bits
            {
                continue;
            }

            let suffix = derive_optimal_suffix_schedule(
                suffix_ctx,
                &mut memo,
                SuffixState {
                    level: 1,
                    current_witness_len: output_witness_len,
                    current_lb: log_basis,
                    incoming_setup_prefix: None,
                },
                0,
            )?;
            for suffix in suffix.best_by_payload_per_lb.values() {
                let child_is_terminal = suffix.folds.is_empty();
                let direct_payload_bytes = level_proof_bytes(
                    field_bits,
                    challenge_field_bits,
                    &candidate_params,
                    suffix.first_fold_params(),
                    output_witness_len,
                    Some(if child_is_terminal {
                        akita_types::NextWitnessBindingPolicy::TerminalInnerState
                    } else {
                        akita_types::NextWitnessBindingPolicy::OuterCommitment
                    }),
                )?
                .checked_add(eor_bytes)
                .ok_or_else(|| AkitaError::InvalidSetup("root proof size overflow".into()))?;
                let stage3_payload_bytes =
                    stage3_payload_bytes_for_successor(policy, suffix.first_fold_params())?;
                let total_bytes = direct_payload_bytes
                    .checked_add(stage3_payload_bytes)
                    .and_then(|bytes| bytes.checked_add(suffix.total_bytes))
                    .ok_or_else(|| AkitaError::InvalidSetup("root proof size overflow".into()))?;
                let mut folds = Vec::with_capacity(1 + suffix.folds.len());
                folds.push(CandidateFoldStep {
                    params: candidate_params.clone(),
                    input_witness_len: witness_len,
                    output_witness_len,
                    estimated_direct_payload_bytes: direct_payload_bytes,
                    estimated_stage3_payload_bytes: stage3_payload_bytes,
                });
                folds.extend(suffix.folds.iter().cloned());
                candidates.push(ScheduleCandidate {
                    first_direct_setup_field_len: Some(active_setup_field_len(
                        &candidate_params,
                        &opening_layout,
                    )?),
                    total_bytes,
                    setup_field_elements: level_setup_field_elements(&candidate_params)?
                        .max(suffix.setup_field_elements),
                    folds,
                    terminal: suffix.terminal.clone(),
                });
            }
        }
    }

    let min_payload = candidates
        .iter()
        .map(|candidate| candidate.total_bytes)
        .min()
        .ok_or_else(|| {
            AkitaError::UnsupportedSchedule(format!(
                "no rank-aware schedule for num_vars={}, num_polynomials={}",
                key.num_vars(),
                key.num_polynomials()
            ))
        })?;
    let max_payload = (min_payload as u128)
        .saturating_mul(1_000 + u128::from(slack_permille))
        .div_ceil(1_000)
        .min(usize::MAX as u128) as usize;
    let best = candidates
        .into_iter()
        .filter(|candidate| candidate.total_bytes <= max_payload)
        .min_by_key(|candidate| {
            (
                candidate.folds.first().map_or(usize::MAX, |fold| {
                    fold.params.inner_commit_matrix.output_rank()
                }),
                candidate.total_bytes,
            )
        })
        .ok_or_else(|| AkitaError::UnsupportedSchedule("rank-aware frontier is empty".into()))?;

    materialize_candidate_schedule(
        best.total_bytes,
        best.setup_field_elements,
        policy.ring_dimension,
        best.first_direct_setup_field_len,
        best.folds,
        best.terminal,
    )
}

#[cfg(test)]
mod tests;
