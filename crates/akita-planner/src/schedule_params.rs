//! FoldSchedule planner that applies each catalog-bound selection objective.
//!
//! Public entry: [`crate::find_schedule`]. The search is `Cfg`-free: every
//! per-preset input is carried by the plain-value [`PlannerPolicy`] plus
//! the `ring_challenge_config` closure,
//! exactly the shape generated catalog emission consumes. This keeps the DP a
//! pure function of `(policy, key, dimension domain)` for offline table generation.

use std::{num::NonZeroUsize, sync::Arc};

use akita_challenges::SparseChallengeConfig;
use akita_field::AkitaError;
use akita_types::sis::{
    decomposed_s_block_ring_count, decomposed_t_ring_count, decomposed_w_ring_count,
    num_digits_inner_for_bound, num_digits_open, rounded_up_collision_inf_norm,
    rounded_up_role_a_inf_norm, BalancedSignedDigitFoldPolicy, FoldWitnessNorms, HonestFoldPolicy,
    HonestFoldPolicySpec, HonestFoldSizingQuery, InnerCommitMatrixParams, OpenCommitMatrixParams,
    OuterCommitMatrixParams,
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

use crate::{InnerBasisSource, PlannerPolicy};

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
    derive_candidate_level_params, derive_candidate_level_params_split_frontier,
    SetupPrefixSearchCache,
};
pub(crate) use objective::select_complete_candidate;
pub(crate) use setup_score::{
    level_setup_field_elements, terminal_setup_field_elements, MixedScore,
};
pub(crate) use suffix_dp::{derive_optimal_suffix_schedule, ScheduleMemo, SuffixCtx, SuffixState};

pub(crate) fn root_inner_basis_source(
    honest_fold_policy: HonestFoldPolicySpec,
    log_bound: u32,
) -> InnerBasisSource {
    match honest_fold_policy {
        HonestFoldPolicySpec::UnitOneHot(_) => InnerBasisSource::UnitOneHot,
        HonestFoldPolicySpec::BalancedSignedDigit(_) => {
            InnerBasisSource::RawCoefficients { log_bound }
        }
    }
}

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

#[derive(Clone, Debug, Default)]
pub(crate) struct CandidateFoldChain {
    head: Option<Arc<CandidateFoldNode>>,
    len: usize,
}

#[derive(Debug)]
struct CandidateFoldNode {
    step: CandidateFoldStep,
    tail: Option<Arc<CandidateFoldNode>>,
}

impl CandidateFoldChain {
    pub(crate) fn is_empty(&self) -> bool {
        self.head.is_none()
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    pub(crate) fn first(&self) -> Option<&CandidateFoldStep> {
        self.head.as_deref().map(|node| &node.step)
    }

    pub(crate) fn prepend(&self, step: CandidateFoldStep) -> Self {
        Self {
            head: Some(Arc::new(CandidateFoldNode {
                step,
                tail: self.head.clone(),
            })),
            len: self.len + 1,
        }
    }

    pub(crate) fn to_vec(&self) -> Vec<CandidateFoldStep> {
        let mut folds = Vec::with_capacity(self.len);
        let mut node = self.head.as_deref();
        while let Some(current) = node {
            folds.push(current.step.clone());
            node = current.tail.as_deref();
        }
        folds
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ScheduleCandidate {
    pub(crate) first_direct_setup_field_len: Option<NonZeroUsize>,
    pub(crate) total_bytes: usize,
    pub(crate) setup_field_elements: usize,
    pub(crate) folds: CandidateFoldChain,
    pub(crate) terminal: Arc<CandidateTerminalResponse>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub(crate) struct SetupPrefixCapacity(usize);

impl SetupPrefixCapacity {
    pub(crate) const MAX: Self = Self(usize::MAX);

    pub(crate) fn for_natural_len(natural_len: usize) -> Self {
        Self(padded_setup_prefix_len(natural_len))
    }

    pub(crate) const fn field_elements(self) -> usize {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct CandidateMetrics {
    pub(crate) first_direct_setup_capacity: SetupPrefixCapacity,
    pub(crate) proof_bytes: usize,
    pub(crate) setup_field_elements: usize,
}

impl ScheduleCandidate {
    pub(crate) fn first_fold_params(&self) -> Option<&CommittedGroupParams> {
        self.folds.first().map(|fold| fold.params.as_ref())
    }

    pub(crate) fn metrics(&self) -> CandidateMetrics {
        CandidateMetrics {
            first_direct_setup_capacity: self
                .first_direct_setup_field_len
                .map_or(SetupPrefixCapacity::MAX, |natural_len| {
                    SetupPrefixCapacity::for_natural_len(natural_len.get())
                }),
            proof_bytes: self.total_bytes,
            setup_field_elements: self.setup_field_elements,
        }
    }
}

pub(crate) fn candidate_schedule_descriptor_bytes(
    choice: &ScheduleCandidate,
) -> Result<Vec<u8>, AkitaError> {
    Ok(materialize_candidate_schedule(
        choice.total_bytes,
        choice.setup_field_elements,
        choice.first_direct_setup_field_len.map(NonZeroUsize::get),
        choice.folds.to_vec(),
        choice.terminal.as_ref().clone(),
    )?
    .schedule
    .canonical_descriptor_bytes())
}

pub(crate) fn candidate_suffix_descriptor_bytes(choice: &ScheduleCandidate) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&choice.folds.len().to_le_bytes());
    let mut node = choice.folds.head.as_deref();
    while let Some(current) = node {
        let fold = &current.step;
        let params = fold.params.canonical_descriptor_bytes();
        bytes.extend_from_slice(&params.len().to_le_bytes());
        bytes.extend_from_slice(&params);
        bytes.extend_from_slice(&fold.input_witness_len.to_le_bytes());
        bytes.extend_from_slice(&fold.output_witness_len.to_le_bytes());
        node = current.tail.as_deref();
    }
    let terminal = choice.terminal.params.canonical_descriptor_bytes();
    bytes.extend_from_slice(&terminal.len().to_le_bytes());
    bytes.extend_from_slice(&terminal);
    bytes.extend_from_slice(&choice.terminal.input_witness_len.to_le_bytes());
    bytes
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StandalonePrecommitSelectionPolicy {
    MinPaddedSetupThenNextWitness,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StandalonePrecommitCandidate {
    pub profile: CommittedGroupProfile,
    pub next_witness_len: usize,
    pub ab_setup_field_elements: usize,
    pub padded_ab_setup_field_elements: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StandalonePrecommitPlan {
    pub selection_policy: StandalonePrecommitSelectionPolicy,
    pub selected: StandalonePrecommitCandidate,
    pub pareto_frontier: Vec<StandalonePrecommitCandidate>,
}

fn matrix_field_elements(
    output_rank: usize,
    input_width: usize,
    ring_dimension: usize,
) -> Result<usize, AkitaError> {
    output_rank
        .checked_mul(input_width)
        .and_then(|elements| elements.checked_mul(ring_dimension))
        .ok_or_else(|| AkitaError::InvalidSetup("standalone matrix size overflow".into()))
}

fn ab_setup_field_elements(params: &CommittedGroupParams) -> Result<usize, AkitaError> {
    let inner = matrix_field_elements(
        params.inner_commit_matrix.output_rank(),
        params.inner_commit_matrix.input_width(),
        params.inner_commit_matrix.ring_dimension(),
    )?;
    let outer = matrix_field_elements(
        params.outer_commit_matrix.output_rank(),
        params.outer_commit_matrix.input_width(),
        params.outer_commit_matrix.ring_dimension(),
    )?;
    Ok(inner.max(outer))
}

/// Exhaustively plan the standalone A/B geometry and retain its Pareto frontier.
pub fn plan_standalone_precommit(
    key: PolynomialGroupLayout,
    policy: &PlannerPolicy,
    honest_fold_policy: HonestFoldPolicySpec,
    ring_challenge_config: impl Fn(usize) -> Result<akita_challenges::SparseChallengeConfig, AkitaError>,
) -> Result<StandalonePrecommitPlan, AkitaError> {
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
    direct_policy.opening_basis_range = (
        direct_policy.opening_basis_range.0,
        direct_policy.opening_basis_range.0,
    );
    akita_schedules::planner_support::validate_policy(&direct_policy)?;
    let witness_len = 1usize
        .checked_shl(key.num_vars() as u32)
        .ok_or_else(|| AkitaError::InvalidSetup("precommit witness too large".into()))?;
    let (min_open_basis, max_open_basis) = direct_policy.log_basis_search_range_at_level(0);
    let inner_source = root_inner_basis_source(
        honest_fold_policy,
        direct_policy.decomposition.log_commit_bound,
    );
    let (min_inner_basis, max_inner_basis) =
        direct_policy.inner_basis_search_range(inner_source)?;
    let dimensions = akita_schedules::planner_support::RingDimensionCandidate::Fixed(
        CommitmentRingDims::uniform(precommit_dimension),
    );
    let ring_challenge_cfg = ring_challenge_config(precommit_dimension)?;
    let mut frontier: Vec<(StandalonePrecommitCandidate, Vec<u8>)> = Vec::new();

    for candidate_open_basis in min_open_basis..=max_open_basis {
        for candidate_inner_basis in min_inner_basis..=max_inner_basis {
            for (candidate_params, next_witness_len) in
                crate::planner::root_level_candidates_for_basis(
                    &schedule_key,
                    honest_fold_policy,
                    &[],
                    &direct_policy,
                    dimensions,
                    &ring_challenge_cfg,
                    &ring_challenge_config,
                    witness_len,
                    candidate_inner_basis,
                    candidate_open_basis,
                    false,
                )?
            {
                let setup = ab_setup_field_elements(&candidate_params)?;
                let candidate = StandalonePrecommitCandidate {
                    profile: CommittedGroupProfile::from_params(key, &candidate_params),
                    next_witness_len,
                    ab_setup_field_elements: setup,
                    padded_ab_setup_field_elements: padded_setup_prefix_len(setup),
                };
                let coords = (
                    candidate.next_witness_len,
                    candidate.padded_ab_setup_field_elements,
                    candidate.ab_setup_field_elements,
                );
                let descriptor = candidate.profile.canonical_descriptor_bytes();
                if frontier.iter().any(|(best, best_descriptor)| {
                    let best_coords = (
                        best.next_witness_len,
                        best.padded_ab_setup_field_elements,
                        best.ab_setup_field_elements,
                    );
                    best_coords.0 <= coords.0
                        && best_coords.1 <= coords.1
                        && best_coords.2 <= coords.2
                        && (best_coords != coords || best_descriptor <= &descriptor)
                }) {
                    continue;
                }
                frontier.retain(|(other, other_descriptor)| {
                    let other_coords = (
                        other.next_witness_len,
                        other.padded_ab_setup_field_elements,
                        other.ab_setup_field_elements,
                    );
                    !(coords.0 <= other_coords.0
                        && coords.1 <= other_coords.1
                        && coords.2 <= other_coords.2
                        && (coords != other_coords || descriptor < *other_descriptor))
                });
                frontier.push((candidate, descriptor));
            }
        }
    }

    frontier.sort_by(|(left, left_descriptor), (right, right_descriptor)| {
        (
            left.padded_ab_setup_field_elements,
            left.next_witness_len,
            left.ab_setup_field_elements,
            left_descriptor,
        )
            .cmp(&(
                right.padded_ab_setup_field_elements,
                right.next_witness_len,
                right.ab_setup_field_elements,
                right_descriptor,
            ))
    });
    let selected = frontier
        .first()
        .map(|(candidate, _)| candidate.clone())
        .ok_or_else(|| {
            AkitaError::InvalidSetup(format!(
                "no standalone precommit profile found for layout {key:?} under this policy"
            ))
        })?;
    Ok(StandalonePrecommitPlan {
        selection_policy: StandalonePrecommitSelectionPolicy::MinPaddedSetupThenNextWitness,
        selected,
        pareto_frontier: frontier
            .into_iter()
            .map(|(candidate, _)| candidate)
            .collect(),
    })
}

pub fn derive_standalone_precommit_profile(
    key: PolynomialGroupLayout,
    policy: &PlannerPolicy,
    honest_fold_policy: HonestFoldPolicySpec,
    ring_challenge_config: impl Fn(usize) -> Result<akita_challenges::SparseChallengeConfig, AkitaError>,
) -> Result<CommittedGroupProfile, AkitaError> {
    Ok(
        plan_standalone_precommit(key, policy, honest_fold_policy, ring_challenge_config)?
            .selected
            .profile,
    )
}

#[cfg(test)]
#[path = "test/schedule_params.rs"]
mod tests;
