//! FoldSchedule planner that applies each catalog-bound selection objective.
//!
//! Public entry: [`crate::find_schedule`]. The search is `Cfg`-free: every
//! per-preset input is carried by the plain-value [`PlannerPolicy`] plus
//! the `ring_challenge_config` closure,
//! exactly the shape generated catalog emission consumes. This keeps the DP a
//! pure function of `(policy, key, dimension domain)` for offline table generation.

use akita_challenges::SparseChallengeConfig;
use akita_field::AkitaError;
use akita_types::sis::{
    decomposed_s_block_ring_count, decomposed_t_ring_count, decomposed_w_ring_count,
    num_digits_inner, num_digits_open, num_digits_setup_prefix_commit,
    rounded_up_collision_inf_norm, rounded_up_role_a_inf_norm, BalancedSignedDigitFoldPolicy,
    FoldWitnessNorms, HonestFoldPolicy, HonestFoldPolicySpec, HonestFoldSizingQuery,
    InnerCommitMatrixParams, OpenCommitMatrixParams, OuterCommitMatrixParams,
};
use akita_types::{
    dyadic_block_ranges, padded_setup_prefix_len, AkitaScheduleLookupKey, CommitmentRingDims,
    CommittedGroupParams, CommittedGroupProfile, DecompositionParams, PolynomialGroupLayout,
    PrecommittedLevelParams,
};
#[cfg(test)]
use akita_types::{
    level_proof_bytes, try_extension_opening_reduction_level_bytes, PlannedFoldSchedule,
};

use crate::PlannerPolicy;

mod candidate;
mod objective;
mod setup_score;
mod suffix_dp;
#[cfg(test)]
#[path = "test/unpruned_search.rs"]
mod unpruned_search;
pub(crate) use akita_schedules::planner_support::{
    materialize_candidate_schedule, stage3_payload_bytes_for_successor, CandidateFoldStep,
    CandidateTerminalResponse,
};
pub use akita_types::suffix_opening_layout;
pub(crate) use candidate::{
    derive_candidate_level_params, derive_candidate_level_params_all_splits,
};
pub(crate) use objective::select_complete_candidate;
pub(crate) use setup_score::{
    level_setup_field_elements, terminal_setup_field_elements, MixedScore,
};
pub(crate) use suffix_dp::{derive_optimal_suffix_schedule, ScheduleMemo, SuffixCtx, SuffixState};

fn dimension_candidates(
    policy: &PlannerPolicy,
    level: usize,
    ceiling: CommitmentRingDims,
) -> Result<Vec<akita_schedules::planner_support::RingDimensionCandidate<'_>>, AkitaError> {
    use akita_schedules::planner_support::RingDimensionCandidate;

    ceiling.validate_role_projection()?;
    let candidates = match policy.ring_dimension_schedule_mode {
        crate::RingDimensionScheduleMode::UniformDimension { ring_dimension } => {
            vec![RingDimensionCandidate::Fixed(CommitmentRingDims::uniform(
                ring_dimension,
            ))]
        }
        crate::RingDimensionScheduleMode::AdaptiveDimension {
            num_search_levels,
            uniform_suffix_dimension,
            potential_a_dimensions,
            potential_b_dimensions,
            potential_d_dimensions,
        } => {
            if level >= num_search_levels {
                vec![RingDimensionCandidate::Fixed(CommitmentRingDims::uniform(
                    uniform_suffix_dimension,
                ))]
            } else if policy.selection_policy
                == crate::SelectionPolicyId::MinSetupMatrixFieldElementsThenProofPayload
            {
                potential_a_dimensions
                    .iter()
                    .copied()
                    .filter(|&inner| inner <= ceiling.d_a())
                    .map(|inner| RingDimensionCandidate::Adaptive {
                        inner,
                        outer_dimensions: potential_b_dimensions,
                        opening_dimensions: potential_d_dimensions,
                        ceiling,
                    })
                    .collect()
            } else {
                let mut candidates = Vec::new();
                for &inner in potential_a_dimensions {
                    if inner > ceiling.d_a() {
                        continue;
                    }
                    for &outer in potential_b_dimensions {
                        if outer > ceiling.d_b() || !inner.is_multiple_of(outer) {
                            continue;
                        }
                        for &opening in potential_d_dimensions {
                            if opening > ceiling.d_d() || !inner.is_multiple_of(opening) {
                                continue;
                            }
                            candidates.push(RingDimensionCandidate::Fixed(CommitmentRingDims {
                                inner,
                                outer,
                                opening,
                            }));
                        }
                    }
                }
                candidates
            }
        }
    };
    Ok(candidates)
}

#[cfg(test)]
pub(crate) const ADAPTIVE_SUFFIX_RING_DIMENSION: usize = 64;

/// Explicit A/B/D dimensions admitted by mixed-D planner search.
///
/// The planner policy's uniform ring dimension defines only the implicit
/// singleton domain used by [`crate::find_schedule`]. Mixed-dimension search supplies
/// this explicit set of schedule-owned A/B/D tuples.
#[cfg(test)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct RingDimensionSearchDomain {
    candidates: Vec<CommitmentRingDims>,
}

#[cfg(test)]
impl RingDimensionSearchDomain {
    /// Construct and canonicalize a non-empty dimension domain.
    ///
    /// Every tuple must satisfy the schedule-local A-carrier invariant.
    pub(crate) fn new(
        candidates: impl IntoIterator<Item = CommitmentRingDims>,
    ) -> Result<Self, AkitaError> {
        let mut candidates = candidates.into_iter().collect::<Vec<_>>();
        candidates.sort_by_key(|dims| (dims.d_a(), dims.d_b(), dims.d_d()));
        candidates.dedup();
        if candidates.is_empty() {
            return Err(AkitaError::InvalidSetup(
                "ring-dimension search domain must be nonempty".into(),
            ));
        }
        for dims in &candidates {
            dims.validate_role_projection()?;
        }
        Ok(Self { candidates })
    }

    /// Construct the explicit singleton domain used by a uniform policy.
    pub(crate) fn uniform(ring_dimension: usize) -> Result<Self, AkitaError> {
        Self::new([CommitmentRingDims::uniform(ring_dimension)])
    }

    /// Canonically ordered admitted A/B/D tuples.
    pub(crate) fn candidates(&self) -> &[CommitmentRingDims] {
        &self.candidates
    }

    #[cfg(test)]
    pub(crate) fn validate_for_policy(&self, policy: &PlannerPolicy) -> Result<(), AkitaError> {
        akita_schedules::planner_support::validate_policy(policy)
    }
}

#[cfg(test)]
fn componentwise_dimensions_at_most(
    dimensions: CommitmentRingDims,
    ceiling: CommitmentRingDims,
) -> bool {
    dimensions.d_a() <= ceiling.d_a()
        && dimensions.d_b() <= ceiling.d_b()
        && dimensions.d_d() <= ceiling.d_d()
}

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

    pub(crate) fn direct_frontier_score(&self) -> (usize, usize) {
        (self.total_bytes, self.setup_field_elements)
    }

    pub(crate) fn recursive_setup_frontier_score(&self) -> (usize, usize, usize) {
        (
            self.first_direct_setup_field_len_or_max(),
            self.total_bytes,
            self.setup_field_elements,
        )
    }
}

pub(crate) fn candidate_schedule_descriptor_bytes(
    choice: &ScheduleCandidate,
) -> Result<Vec<u8>, AkitaError> {
    Ok(materialize_candidate_schedule(
        choice.total_bytes,
        choice.setup_field_elements,
        choice.first_direct_setup_field_len,
        choice.folds.clone(),
        choice.terminal.clone(),
    )?
    .schedule
    .canonical_descriptor_bytes())
}

/// Stage-1 sparse-challenge closure shared by the planner entry points.
pub(crate) type RingChallengeConfigFn<'a> =
    &'a dyn Fn(usize) -> Result<akita_challenges::SparseChallengeConfig, AkitaError>;

pub(crate) type LayoutCandidateScore = (usize, usize, usize, usize);

/// Combine exact physical width, challenge work, chunk evaluator work,
/// and load imbalance when comparing `M` candidates. All terms count ring or
/// scalar work units; exact physical width remains an explicit tie-breaker.
pub(crate) fn layout_candidate_score(
    physical_width: usize,
    num_live_blocks: usize,
    num_chunks: usize,
) -> Result<LayoutCandidateScore, AkitaError> {
    let challenge_work = num_live_blocks;
    let chunk_ranges = dyadic_block_ranges(num_live_blocks, num_chunks)?;
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

/// Offline canonical standalone precommit descriptor for one group.
///
/// Root precommits are selected independently of any future final-group
/// schedule. Adaptive policies use their uniform suffix dimension for both A
/// and B; final groups retain the full per-level adaptive search.
pub fn derive_standalone_precommit_profile(
    key: PolynomialGroupLayout,
    policy: &PlannerPolicy,
    honest_fold_policy: HonestFoldPolicySpec,
    ring_challenge_config: impl Fn(usize) -> Result<akita_challenges::SparseChallengeConfig, AkitaError>,
) -> Result<CommittedGroupProfile, AkitaError> {
    key.validate()?;
    let schedule_key = AkitaScheduleLookupKey::single(key);
    let mut direct_policy = policy.direct_only();
    let precommit_dimension = match policy.ring_dimension_schedule_mode {
        crate::RingDimensionScheduleMode::UniformDimension { ring_dimension } => ring_dimension,
        crate::RingDimensionScheduleMode::AdaptiveDimension {
            uniform_suffix_dimension,
            ..
        } => uniform_suffix_dimension,
    };
    direct_policy.uniform_ring_dimension = precommit_dimension;
    direct_policy.ring_dimension_schedule_mode =
        crate::RingDimensionScheduleMode::UniformDimension {
            ring_dimension: precommit_dimension,
        };
    direct_policy.selection_policy =
        crate::SelectionPolicyId::for_policy(false, direct_policy.ring_dimension_schedule_mode);
    direct_policy.basis_range = (direct_policy.basis_range.0, direct_policy.basis_range.0);
    akita_schedules::planner_support::validate_policy(&direct_policy)?;
    let witness_len = 1usize
        .checked_shl(key.num_vars() as u32)
        .ok_or_else(|| AkitaError::InvalidSetup("precommit witness too large".into()))?;
    let (min_log_basis, max_log_basis) = direct_policy.log_basis_search_range_at_level(0);
    let mut best: Option<(usize, CommittedGroupParams)> = None;
    for candidate_log_basis in min_log_basis..=max_log_basis {
        let dimensions = akita_schedules::planner_support::RingDimensionCandidate::Fixed(
            CommitmentRingDims::uniform(precommit_dimension),
        );
        let ring_challenge_cfg = ring_challenge_config(precommit_dimension)?;
        for (candidate_params, next_witness_len) in crate::planner::root_level_candidates_for_basis(
            &schedule_key,
            honest_fold_policy,
            &[],
            &direct_policy,
            dimensions,
            &ring_challenge_cfg,
            &ring_challenge_config,
            witness_len,
            candidate_log_basis,
            false,
        )? {
            match &best {
                Some((best_len, _)) if *best_len <= next_witness_len => {}
                _ => best = Some((next_witness_len, candidate_params)),
            }
        }
    }
    best.map(|(_, params)| CommittedGroupProfile::from_params(key, &params))
        .ok_or_else(|| {
            AkitaError::InvalidSetup(format!(
                "no standalone precommit profile found for layout {key:?} under this policy"
            ))
        })
}

#[cfg(test)]
#[path = "test/schedule_params.rs"]
mod tests;
