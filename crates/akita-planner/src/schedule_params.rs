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
#[cfg(test)]
use akita_types::extension_opening_reduction_level_bytes;
use akita_types::sis::{
    decomposed_s_block_ring_count, decomposed_t_ring_count, decomposed_w_ring_count,
    fold_witness_digit_plan, num_digits_inner, num_digits_open, num_digits_setup_prefix_commit,
    rounded_up_collision_inf_norm, rounded_up_role_a_inf_norm, FoldChallengeNorms,
    FoldWitnessLinfCapConfig, FoldWitnessNorms, InnerCommitMatrixParams, OpenCommitMatrixParams,
    OuterCommitMatrixParams, SisTableKey,
};
use akita_types::{
    intermediate_w_ring_element_count_for_chunks, level_proof_bytes, padded_setup_prefix_len,
    try_extension_opening_reduction_level_bytes, AkitaScheduleInputs, CommitmentRingDims,
    CommittedGroupParams, DecompositionParams, FoldSchedule, PlannedFoldSchedule,
    PolynomialGroupLayout, PrecommittedGroupDescriptor, PrecommittedLevelParams,
    TerminalResponseShape, WitnessLayout,
};

use crate::PlannerPolicy;

mod candidate;
mod mixed_search;
mod setup_score;
mod suffix_dp;
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

/// One recursive fold of an independently planned suffix
/// ([`plan_optimal_suffix`]).
#[derive(Clone, Debug)]
pub struct PlannedSuffixFold {
    /// Committed-group params for this fold level (already priced at
    /// `policy.ring_dimension`).
    pub params: CommittedGroupParams,
    /// Field-element witness length entering this fold.
    pub input_witness_len: usize,
    /// Field-element witness length produced for the next level.
    pub output_witness_len: usize,
}

/// Terminal (cleartext) response of an independently planned suffix.
#[derive(Clone, Debug)]
pub struct PlannedSuffixTerminal {
    /// Terminal committed-group params.
    pub params: akita_types::TerminalCommittedGroupParams,
    /// Short ring challenge family for the terminal fold.
    pub sparse_challenge_config: akita_challenges::SparseChallengeConfig,
    /// Field-element witness length entering the terminal fold.
    pub input_witness_len: usize,
    /// Cleartext response wire shape.
    pub response_shape: TerminalResponseShape,
}

/// Optimal recursive suffix planned from an intermediate witness.
#[derive(Clone, Debug)]
pub struct PlannedSuffix {
    /// Recursive fold levels, starting at `start_level`.
    pub folds: Vec<PlannedSuffixFold>,
    /// Terminal fold.
    pub terminal: PlannedSuffixTerminal,
    /// Header-stripped direct-mode proof bytes of the suffix (folds + terminal).
    pub total_bytes: usize,
}

/// Plan the proof-size-optimal recursive suffix that folds a witness of
/// `start_witness_len` field elements (produced by some predecessor fold at
/// `start_level - 1`) down to a cleartext terminal, at `policy.ring_dimension`.
///
/// This is the exact suffix DP [`crate::find_schedule`] runs after choosing a root,
/// exposed so callers can splice an optimal suffix onto a differently sized
/// predecessor — e.g. a mixed ring-dimension-per-level schedule whose root
/// folds at a larger ring dimension than the suffix. `start_lb` is the
/// predecessor level's `log_basis` (fold `log_basis` is non-decreasing), and
/// `num_vars` is the opening arity (used for the singleton opening layout the
/// suffix prices against).
///
/// # Errors
///
/// Returns [`AkitaError::UnsupportedSchedule`] if no terminating suffix exists
/// for the requested state, or propagates SIS-sizing / overflow failures.
pub fn plan_optimal_suffix(
    policy: &PlannerPolicy,
    ring_challenge_config: impl Fn(usize) -> Result<akita_challenges::SparseChallengeConfig, AkitaError>,
    fold_challenge_shape_at_level: impl Fn(AkitaScheduleInputs) -> TensorChallengeShape,
    num_vars: usize,
    start_level: usize,
    start_witness_len: usize,
    start_lb: u32,
) -> Result<PlannedSuffix, AkitaError> {
    validate_policy(policy)?;
    if policy.recursive_setup_planning {
        return Err(AkitaError::InvalidSetup(
            "recursive setup planning is not supported by plan_optimal_suffix".to_string(),
        ));
    }
    let ring_challenge_cfg = ring_challenge_config(policy.ring_dimension)?;
    let ctx = SuffixCtx {
        policy,
        default_ring_challenge_cfg: &ring_challenge_cfg,
        ring_challenge_config: &ring_challenge_config,
        fold_challenge_shape_at_level: &fold_challenge_shape_at_level,
        num_vars,
        key: PolynomialGroupLayout::singleton(num_vars),
        setup_field_budget: None,
        root_lookup_key: None,
    };
    let mut memo = ScheduleMemo::new();
    let result = derive_optimal_suffix_schedule(
        &ctx,
        &mut memo,
        SuffixState {
            level: start_level,
            current_witness_len: start_witness_len,
            current_lb: start_lb,
            incoming_setup_prefix: None,
        },
        0,
    )?;
    let best = result
        .best_by_payload_per_lb
        .values()
        .min_by_key(|suffix| suffix.total_bytes)
        .ok_or_else(|| {
            AkitaError::UnsupportedSchedule(format!(
                "no terminating suffix for witness_len={start_witness_len} at level {start_level}"
            ))
        })?;
    Ok(PlannedSuffix {
        folds: best
            .folds
            .iter()
            .map(|fold| PlannedSuffixFold {
                params: fold.params.clone(),
                input_witness_len: fold.input_witness_len,
                output_witness_len: fold.output_witness_len,
            })
            .collect(),
        terminal: PlannedSuffixTerminal {
            params: best.terminal.params.clone(),
            sparse_challenge_config: best.terminal.sparse_challenge_config,
            input_witness_len: best.terminal.input_witness_len,
            response_shape: best.terminal.response_shape.clone(),
        },
        total_bytes: best.total_bytes,
    })
}

pub(crate) fn find_schedule_singular(
    key: PolynomialGroupLayout,
    policy: &PlannerPolicy,
    ring_challenge_config: impl Fn(usize) -> Result<akita_challenges::SparseChallengeConfig, AkitaError>,
    fold_challenge_shape_at_level: impl Fn(AkitaScheduleInputs) -> TensorChallengeShape,
) -> Result<PlannedFoldSchedule, AkitaError> {
    key.validate()?;
    validate_policy(policy)?;
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
    };

    if policy.recursive_setup_planning {
        return Err(AkitaError::InvalidSetup(
            "recursive setup planning requires the grouped-batch scheduler".to_string(),
        ));
    }
    let witness_len = 1usize
        .checked_shl(key.num_vars() as u32)
        .ok_or_else(|| AkitaError::InvalidSetup("witness too large".into()))?;

    let field_bits = policy.decomposition.field_bits();
    let mut best: Option<ScheduleCandidate> = None;
    let fold_challenge_shape = fold_shape(AkitaScheduleInputs {
        num_vars: key.num_vars(),
        level: 0,
        input_witness_len: witness_len,
    });
    let mut memo = ScheduleMemo::new();

    // Chunk count of the witness committed at the root fold (absolute level 0).
    let root_num_chunks = policy.chunks_at_level(0);
    let (min_log_basis, max_log_basis) = policy.log_basis_search_range_at_level(0);
    let candidate_dimensions = CommitmentRingDims::uniform(policy.ring_dimension);
    for candidate_log_basis in min_log_basis..=max_log_basis {
        let mut root_candidates = Vec::new();
        {
            let alpha = (candidate_dimensions.d_a() as u32).trailing_zeros() as usize;
            let reduced_vars = key.num_vars().saturating_sub(alpha);
            if reduced_vars == 0 {
                continue;
            }
            let min_block_index_bits: usize = if reduced_vars >= 3 { 1 } else { 0 };
            let max_block_index_bits: usize = (reduced_vars - 1).min(usize::BITS as usize - 1);
            let Ok(ring_challenge_cfg) = ring_challenge_config(candidate_dimensions.d_a()) else {
                continue;
            };
            for block_index_bits in (min_block_index_bits..=max_block_index_bits).rev() {
                let Some(candidate_params) = scalar_root_fold_level_params_candidate(
                    policy,
                    &ring_challenge_cfg,
                    candidate_dimensions,
                    key.num_vars(),
                    key.num_polynomials(),
                    candidate_log_basis,
                    block_index_bits,
                    fold_challenge_shape,
                )?
                else {
                    continue;
                };

                let output_witness_len = intermediate_w_ring_element_count_for_chunks(
                    field_bits,
                    &candidate_params,
                    key.num_polynomials(),
                    root_num_chunks,
                )?
                .checked_mul(candidate_dimensions.d_a())
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("root next witness length overflow".into())
                })?;
                let initial_witness_len_bits = witness_len
                    .checked_mul(field_bits as usize)
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup("root witness bit length overflow".into())
                    })?;
                if output_witness_len
                    .checked_mul(candidate_log_basis as usize)
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup("root next witness bit length overflow".into())
                    })?
                    >= initial_witness_len_bits
                {
                    continue;
                }
                root_candidates.push((candidate_params, output_witness_len));
            }
        }
        for (candidate_params, output_witness_len) in root_candidates {
            let candidate_dimensions = candidate_params.role_dims();
            // Root projection is governed by the candidate's committed A dimension,
            // not the setup-generation ceiling.
            let Some(eor_bytes) = try_extension_opening_reduction_level_bytes(
                policy.challenge_field_bits()?,
                policy.claim_ext_degree,
                0,
                key,
                witness_len,
                candidate_dimensions.d_a(),
            )?
            else {
                continue;
            };
            let suffix = derive_optimal_suffix_schedule(
                &suffix_ctx,
                &mut memo,
                SuffixState {
                    level: 1,
                    current_witness_len: output_witness_len,
                    current_lb: candidate_log_basis,
                    incoming_setup_prefix: None,
                },
                0,
            )?;
            if suffix.is_empty() {
                continue;
            }
            // A supported root must recurse into at least one suffix fold.
            for suffix_fold in suffix.best_by_payload_per_lb.values() {
                let next_witness_binding = if suffix_fold.folds.is_empty() {
                    akita_types::NextWitnessBindingPolicy::TerminalInnerState
                } else {
                    akita_types::NextWitnessBindingPolicy::OuterCommitment
                };
                let root_proof_size = level_proof_bytes(
                    field_bits,
                    policy.challenge_field_bits()?,
                    &candidate_params,
                    suffix_fold.first_fold_params(),
                    output_witness_len,
                    Some(next_witness_binding),
                )? + eor_bytes;
                let total = root_proof_size + suffix_fold.total_bytes;
                let root_envelope = level_setup_field_elements(&candidate_params)?;
                let setup_envelope = root_envelope.max(suffix_fold.setup_field_elements);
                let mut folds = Vec::with_capacity(1 + suffix_fold.folds.len());
                folds.push(CandidateFoldStep {
                    params: candidate_params.clone(),
                    input_witness_len: witness_len,
                    output_witness_len,
                    estimated_direct_payload_bytes: root_proof_size,
                    estimated_stage3_payload_bytes: 0,
                });
                folds.extend(suffix_fold.folds.iter().cloned());
                let candidate = ScheduleCandidate {
                    first_direct_setup_field_len: None,
                    total_bytes: total,
                    setup_field_elements: setup_envelope,
                    folds,
                    terminal: suffix_fold.terminal.clone(),
                };
                let replace = match &best {
                    None => true,
                    Some(current) => candidate.total_bytes < current.total_bytes,
                };
                if replace {
                    best = Some(candidate);
                }
            }
        }
    }

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

#[cfg(test)]
mod tests;
