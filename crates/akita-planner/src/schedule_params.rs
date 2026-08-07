//! FoldSchedule planner that finds the global minimum proof size. Recursive
//! grouped scheduling additionally minimizes the first direct setup footprint
//! before proof size.
//!
//! Public entry: [`crate::find_schedule`]. The search is `Cfg`-free: every
//! per-preset input is carried by the plain-value [`PlannerPolicy`] plus
//! the `ring_challenge_config` closure,
//! exactly the shape generated catalog emission consumes. This keeps the DP a
//! pure function of `(policy, key, dimension domain)` for offline table generation.

use std::sync::Arc;

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
    level_proof_bytes, padded_setup_prefix_len, try_extension_opening_reduction_level_bytes,
    AkitaScheduleLookupKey, CommitmentRingDims, CommittedGroupParams, CommittedGroupProfile,
    DecompositionParams, PlannedFoldSchedule, PolynomialGroupLayout, PrecommittedLevelParams,
    TerminalResponseShape, WitnessLayout,
};

use akita_schedules::planner_support::{
    materialize_candidate_schedule, stage3_payload_bytes_for_successor, MAX_RECURSION_DEPTH,
};

use crate::{InnerBasisSource, PlannerPolicy};

mod candidate;
pub(crate) mod mixed_search;
mod objective;
mod setup_score;
mod suffix_dp;
#[cfg(test)]
#[path = "test/unpruned_search.rs"]
mod unpruned_search;

pub use akita_types::suffix_opening_layout;
pub(in crate::schedule_params) use candidate::{
    derive_candidate_level_params, derive_candidate_level_params_all_splits, SetupPrefixSearchCache,
};
pub(crate) use objective::select_complete_candidate;
pub(crate) use setup_score::{
    level_setup_field_elements, terminal_setup_field_elements, MixedScore,
};
pub(crate) use suffix_dp::{
    derive_optimal_suffix_schedule, SuffixCtx, SuffixSearchCache, SuffixState,
};

pub(crate) const MIXED_SEARCH_FOLD_LEVELS: usize = 2;
pub(crate) const MIXED_SEARCH_SUFFIX_RING_DIMENSION: usize = 64;

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

#[derive(Clone, Debug)]
pub(crate) struct CandidateFoldStep {
    pub(crate) params: Arc<CommittedGroupParams>,
    pub(crate) input_witness_len: usize,
    pub(crate) output_witness_len: usize,
    pub(crate) estimated_direct_payload_bytes: usize,
    pub(crate) estimated_stage3_payload_bytes: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct CandidateTerminalResponse {
    pub(crate) params: akita_types::TerminalCommittedGroupParams,
    pub(crate) sparse_challenge_config: akita_challenges::SparseChallengeConfig,
    pub(crate) input_witness_len: usize,
    pub(crate) estimated_direct_payload_bytes: usize,
    pub(crate) response_shape: TerminalResponseShape,
    pub(crate) estimated_payload_bytes: usize,
}

/// Explicit A/B/D dimensions admitted by mixed-D planner search.
///
/// The planner policy's uniform ring dimension defines only the implicit
/// singleton domain used by [`crate::find_schedule`]. Mixed-dimension search supplies
/// this explicit set of schedule-owned A/B/D tuples.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RingDimensionSearchDomain {
    candidates: Vec<CommitmentRingDims>,
}

impl RingDimensionSearchDomain {
    /// Construct and canonicalize a non-empty dimension domain.
    ///
    /// Every tuple must satisfy the schedule-local A-carrier invariant.
    pub fn new(
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
    pub fn uniform(ring_dimension: usize) -> Result<Self, AkitaError> {
        Self::new([CommitmentRingDims::uniform(ring_dimension)])
    }

    /// Canonically ordered admitted A/B/D tuples.
    pub fn candidates(&self) -> &[CommitmentRingDims] {
        &self.candidates
    }

    pub(crate) fn validate_for_policy(&self, policy: &PlannerPolicy) -> Result<(), AkitaError> {
        if self.candidates.as_slice() != policy.ring_dimension_candidates {
            if policy.ring_dimension_candidates.len() == 1 {
                return Ok(());
            }
            return Err(AkitaError::InvalidSetup(
                "ring-dimension search domain disagrees with the catalog-bound policy domain"
                    .to_string(),
            ));
        }
        Ok(())
    }

    pub(crate) fn is_uniform_policy_domain(&self, policy: &PlannerPolicy) -> bool {
        self.candidates.as_slice() == [CommitmentRingDims::uniform(policy.uniform_ring_dimension)]
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ScheduleCandidate {
    pub(crate) first_direct_setup_field_len: Option<usize>,
    pub(crate) total_bytes: usize,
    pub(crate) setup_field_elements: usize,
    pub(crate) folds: CandidateFoldChain,
    pub(crate) terminal: Arc<CandidateTerminalResponse>,
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
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

impl PartialOrd for SetupPrefixCapacity {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SetupPrefixCapacity {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.cmp(&other.0)
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
            first_direct_setup_capacity: self.first_direct_setup_field_len.map_or(
                SetupPrefixCapacity::MAX,
                SetupPrefixCapacity::for_natural_len,
            ),
            proof_bytes: self.total_bytes,
            setup_field_elements: self.setup_field_elements,
        }
    }

    pub(crate) fn materialize(&self) -> Result<PlannedFoldSchedule, AkitaError> {
        materialize_candidate_schedule(
            self.total_bytes,
            self.setup_field_elements,
            self.first_direct_setup_field_len,
            self.folds
                .to_vec()
                .into_iter()
                .map(akita_schedules::planner_support::CandidateFoldStep::from)
                .collect(),
            akita_schedules::planner_support::CandidateTerminalResponse::from(
                self.terminal.as_ref().clone(),
            ),
        )
    }
}

impl From<CandidateFoldStep> for akita_schedules::planner_support::CandidateFoldStep {
    fn from(step: CandidateFoldStep) -> Self {
        Self {
            params: step.params.as_ref().clone(),
            input_witness_len: step.input_witness_len,
            output_witness_len: step.output_witness_len,
            estimated_direct_payload_bytes: step.estimated_direct_payload_bytes,
            estimated_stage3_payload_bytes: step.estimated_stage3_payload_bytes,
        }
    }
}

impl From<CandidateTerminalResponse>
    for akita_schedules::planner_support::CandidateTerminalResponse
{
    fn from(response: CandidateTerminalResponse) -> Self {
        Self {
            params: response.params,
            sparse_challenge_config: response.sparse_challenge_config,
            input_witness_len: response.input_witness_len,
            estimated_direct_payload_bytes: response.estimated_direct_payload_bytes,
            response_shape: response.response_shape,
            estimated_payload_bytes: response.estimated_payload_bytes,
        }
    }
}

pub(crate) fn candidate_schedule_descriptor_bytes(
    choice: &ScheduleCandidate,
) -> Result<Vec<u8>, AkitaError> {
    Ok(choice.materialize()?.schedule.canonical_descriptor_bytes())
}

pub(crate) fn candidate_suffix_descriptor_bytes(choice: &ScheduleCandidate) -> Vec<u8> {
    let mut bytes = Vec::new();
    let folds = choice.folds.to_vec();
    bytes.extend_from_slice(&folds.len().to_le_bytes());
    for fold in folds {
        let params = fold.params.canonical_descriptor_bytes();
        bytes.extend_from_slice(&params.len().to_le_bytes());
        bytes.extend_from_slice(&params);
        bytes.extend_from_slice(&fold.input_witness_len.to_le_bytes());
        bytes.extend_from_slice(&fold.output_witness_len.to_le_bytes());
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

/// Combine exact physical width, challenge-factor work, chunk evaluator work,
/// and load imbalance when comparing `M` candidates. All terms count ring or
/// scalar work units; exact physical width remains an explicit tie-breaker.
pub(crate) fn layout_candidate_score(
    physical_width: usize,
    num_live_blocks: usize,
    num_chunks: usize,
) -> Result<LayoutCandidateScore, AkitaError> {
    let challenge_work = num_live_blocks;
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

/// Explicit selection policy for the standalone-precommit Pareto frontier.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StandalonePrecommitSelectionPolicy {
    /// Prefer the smaller power-of-two A/B setup allocation, then the next witness.
    MinPaddedSetupThenNextWitness,
}

/// One non-dominated standalone A/B commitment candidate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StandalonePrecommitCandidate {
    pub profile: CommittedGroupProfile,
    pub next_witness_len: usize,
    pub ab_setup_field_elements: usize,
    pub padded_ab_setup_field_elements: usize,
}

/// Selected standalone descriptor together with every setup/witness Pareto point.
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

/// Exhaustively plan the standalone A/B commitment geometry for one group.
///
/// Runtime config code consumes the selected generated [`CommittedGroupProfile`].
/// The full frontier remains available to audits and reports so setup/witness
/// tradeoffs are never silently collapsed during geometry search.
pub fn plan_standalone_precommit(
    key: PolynomialGroupLayout,
    policy: &PlannerPolicy,
    honest_fold_policy: HonestFoldPolicySpec,
    ring_challenge_config: impl Fn(usize) -> Result<akita_challenges::SparseChallengeConfig, AkitaError>,
) -> Result<StandalonePrecommitPlan, AkitaError> {
    key.validate()?;
    let mut direct_policy = *policy;
    direct_policy.recursive_setup_planning = false;
    direct_policy.selection_policy = crate::SelectionPolicyId::MinEstimatedProofPayload;
    direct_policy.opening_basis_range = (
        direct_policy.opening_basis_range.0,
        direct_policy.opening_basis_range.0,
    );
    akita_schedules::planner_support::validate_policy(&direct_policy)?;

    let witness_len = 1usize
        .checked_shl(key.num_vars() as u32)
        .ok_or_else(|| AkitaError::InvalidSetup("precommit witness too large".into()))?;
    let (min_log_basis, max_log_basis) = direct_policy.log_basis_search_range_at_level(0);
    let inner_source = root_inner_basis_source(
        honest_fold_policy,
        direct_policy.decomposition.log_commit_bound,
    );
    let (min_inner_basis, max_inner_basis) =
        direct_policy.inner_basis_search_range(inner_source)?;
    let mut frontier: Vec<(StandalonePrecommitCandidate, Vec<u8>)> = Vec::new();
    let schedule_key = AkitaScheduleLookupKey::single(key);

    for candidate_log_basis in min_log_basis..=max_log_basis {
        for candidate_inner_basis in min_inner_basis..=max_inner_basis {
            for dimensions in direct_policy.ring_dimension_candidates.iter().copied() {
                let Ok(ring_challenge_cfg) = ring_challenge_config(dimensions.d_a()) else {
                    continue;
                };
                let alpha = (dimensions.d_a() as u32).trailing_zeros() as usize;
                let reduced_vars = key.num_vars().saturating_sub(alpha);
                if reduced_vars == 0 {
                    continue;
                }
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
                        candidate_log_basis,
                        false,
                    )?
                {
                    let ab_setup_field_elements = ab_setup_field_elements(&candidate_params)?;
                    let candidate = StandalonePrecommitCandidate {
                        profile: CommittedGroupProfile::from_params(key, &candidate_params),
                        next_witness_len,
                        ab_setup_field_elements,
                        padded_ab_setup_field_elements: padded_setup_prefix_len(
                            ab_setup_field_elements,
                        ),
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

fn componentwise_dimensions_at_most(
    dimensions: CommitmentRingDims,
    ceiling: CommitmentRingDims,
) -> bool {
    dimensions.d_a() <= ceiling.d_a()
        && dimensions.d_b() <= ceiling.d_b()
        && dimensions.d_d() <= ceiling.d_d()
}

#[cfg(test)]
#[path = "test/schedule_params.rs"]
mod tests;
