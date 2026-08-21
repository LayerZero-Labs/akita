//! FoldSchedule planner that applies each catalog-bound selection objective.
//!
//! Public entry: [`crate::find_schedule`]. The search is `Cfg`-free: every
//! per-preset input is carried by the plain-value [`PlannerPolicy`] plus
//! the `ring_challenge_config` closure,
//! exactly the shape generated catalog emission consumes. This keeps the DP a
//! pure function of `(policy, key, dimension domain)` for offline table generation.

use std::{num::NonZeroUsize, sync::Arc};

use akita_challenges::SparseChallengeConfig;
use akita_error::AkitaError;
use akita_types::sis::{
    decomposed_s_block_ring_count, num_digits_for_linf_cap, num_digits_inner_for_bound,
    num_digits_open, rounded_up_collision_inf_norm, rounded_up_role_a_inf_norm,
    BalancedSignedDigitFoldPolicy, FoldWitnessNorms, HonestFoldPolicy, HonestFoldPolicySpec,
    HonestFoldSizingQuery, InnerCommitMatrixParams, OpenCommitMatrixParams,
    OuterCommitMatrixParams,
};
use akita_types::{
    active_setup_field_len, dyadic_block_ranges, padded_setup_prefix_len, CommitmentRingDims,
    CommittedGroupParams, DecompositionParams, GroupCommitPhaseParams, GroupOpenPhaseParams,
    OpeningClaimsLayout, PolynomialGroupLayout,
};
#[cfg(test)]
use akita_types::{
    level_proof_bytes, try_extension_opening_reduction_level_bytes, PlannedFoldSchedule,
};

use crate::{InnerBasisSource, PlannerPolicy};

mod candidate;
mod objective;
mod pareto;
mod setup_score;
mod suffix_dp;
#[cfg(test)]
#[path = "test/unpruned_search.rs"]
mod unpruned_search;
pub(crate) use akita_schedules::planner_support::{
    materialize_candidate_schedule, CandidateFoldStep, CandidateTerminalResponse,
};
pub use akita_types::suffix_opening_layout;
pub(crate) use candidate::{
    derive_ab_commitment_candidate, derive_fold_candidates, derive_recursive_candidate_views,
    derive_terminal_candidates, recursive_split_search_domain, AbCommitmentCandidateRequest,
    FoldCandidatePolicy, PlannerOpeningCandidate, RecursiveCandidateRequest, RecursiveSetupPrefix,
    SetupPrefixSearchCache, SplitBoundPolicy,
};
pub(crate) use objective::{select_complete_candidate, CompleteObjectiveBound};
pub(crate) use setup_score::{level_setup_field_elements, terminal_setup_field_elements};
pub(crate) use suffix_dp::{derive_selected_suffix_schedule, ScheduleMemo, SuffixCtx, SuffixState};

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

pub(crate) fn precommitted_groups_support_opening_dimension<'a>(
    profiles: impl IntoIterator<Item = &'a GroupCommitPhaseParams>,
    opening_ring_dimension: usize,
) -> bool {
    profiles.into_iter().all(|profile| {
        profile
            .inner
            .matrix
            .ring_dimension()
            .is_multiple_of(opening_ring_dimension)
    })
}

pub(crate) fn dimension_candidates(
    policy: &PlannerPolicy,
    level: usize,
    ceiling: CommitmentRingDims,
) -> Result<Vec<CommitmentRingDims>, AkitaError> {
    ceiling.validate_role_projection()?;
    let candidates = match policy.ring_dimension_schedule_mode {
        crate::RingDimensionScheduleMode::UniformDimension { ring_dimension } => {
            vec![CommitmentRingDims::uniform(ring_dimension)]
        }
        crate::RingDimensionScheduleMode::AdaptiveDimension {
            num_search_levels,
            suffix_dimensions,
            potential_a_dimensions,
            potential_b_dimensions,
            potential_d_dimensions,
        } => {
            if level >= num_search_levels {
                let Some(maximum_suffix_dimension) =
                    suffix_dimension_ceiling(suffix_dimensions, ceiling)
                else {
                    return Ok(Vec::new());
                };
                suffix_dimensions
                    .iter()
                    .copied()
                    .take_while(|&dimension| dimension <= maximum_suffix_dimension)
                    .map(CommitmentRingDims::uniform)
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
                            candidates.push(CommitmentRingDims {
                                inner,
                                outer,
                                opening,
                            });
                        }
                    }
                }
                candidates
            }
        }
    };
    Ok(candidates)
}

pub(crate) fn initial_dimension_ceiling(
    policy: &PlannerPolicy,
) -> Result<CommitmentRingDims, AkitaError> {
    match policy.ring_dimension_schedule_mode {
        crate::RingDimensionScheduleMode::UniformDimension { ring_dimension } => {
            Ok(CommitmentRingDims::uniform(ring_dimension))
        }
        crate::RingDimensionScheduleMode::AdaptiveDimension {
            potential_a_dimensions,
            potential_b_dimensions,
            potential_d_dimensions,
            ..
        } => Ok(CommitmentRingDims {
            inner: potential_a_dimensions
                .last()
                .copied()
                .ok_or_else(|| AkitaError::InvalidSetup("adaptive A domain is empty".into()))?,
            outer: potential_b_dimensions
                .last()
                .copied()
                .ok_or_else(|| AkitaError::InvalidSetup("adaptive B domain is empty".into()))?,
            opening: potential_d_dimensions
                .last()
                .copied()
                .ok_or_else(|| AkitaError::InvalidSetup("adaptive D domain is empty".into()))?,
        }),
    }
}

fn suffix_dimension_ceiling(
    suffix_dimensions: &[usize],
    ceiling: CommitmentRingDims,
) -> Option<usize> {
    let role_ceiling = ceiling.d_a().min(ceiling.d_b()).min(ceiling.d_d());
    suffix_dimensions
        .iter()
        .rev()
        .copied()
        .find(|&dimension| dimension <= role_ceiling)
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

struct CandidateFoldIter<'a> {
    next: Option<&'a CandidateFoldNode>,
    remaining: usize,
}

impl<'a> Iterator for CandidateFoldIter<'a> {
    type Item = &'a CandidateFoldStep;

    fn next(&mut self) -> Option<Self::Item> {
        let node = self.next?;
        self.next = node.tail.as_deref();
        self.remaining -= 1;
        Some(&node.step)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl ExactSizeIterator for CandidateFoldIter<'_> {}

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

    fn iter(&self) -> impl ExactSizeIterator<Item = &CandidateFoldStep> {
        CandidateFoldIter {
            next: self.head.as_deref(),
            remaining: self.len,
        }
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
    pub(crate) payload_bytes: usize,
    pub(crate) nonce_bits: usize,
    pub(crate) proof_bytes: usize,
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
    pub(crate) payload_bytes: usize,
    pub(crate) nonce_bits: usize,
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
            payload_bytes: self.payload_bytes,
            nonce_bits: self.nonce_bits,
            proof_bytes: self.proof_bytes,
            setup_field_elements: self.setup_field_elements,
        }
    }
}

pub(crate) fn candidate_schedule_descriptor_bytes(
    choice: &ScheduleCandidate,
    diagnostics: Option<&crate::diagnostics::PlannerDiagnostics>,
) -> Result<Vec<u8>, AkitaError> {
    let started = diagnostics.map(|_| std::time::Instant::now());
    let result = (|| {
        if choice.folds.is_empty() {
            return Ok(choice.terminal.params.canonical_descriptor_bytes());
        }
        let carrier_prefix_len = choice.folds.len().min(2);
        let mut bytes = Vec::new();
        bytes.push(carrier_prefix_len as u8);
        bytes.extend(
            choice
                .folds
                .iter()
                .take(carrier_prefix_len)
                .map(|fold| fold.params.payload_mode.tag()),
        );
        let descriptor_steps = choice.folds.iter().enumerate().map(|(index, fold)| {
            akita_types::FoldScheduleDescriptorStep {
                params: &fold.params,
                payload_mode: if index < carrier_prefix_len {
                    akita_types::CommitmentPayloadMode::Compressed
                } else {
                    fold.params.payload_mode
                },
                input_witness_len: fold.input_witness_len,
                output_witness_len: fold.output_witness_len,
            }
        });
        let terminal = &choice.terminal.params;
        akita_types::FoldSchedule::append_descriptor_bytes_from_steps(
            &mut bytes,
            descriptor_steps,
            terminal,
        )?;
        Ok(bytes)
    })();
    if let (Some(diagnostics), Some(started)) = (diagnostics, started) {
        diagnostics.record_descriptor(started.elapsed());
    }
    result
}

/// Stage-1 sparse-challenge closure shared by the planner entry points.
pub(crate) type RingChallengeConfigFn<'a> =
    &'a dyn Fn(usize) -> Result<akita_challenges::SparseChallengeConfig, AkitaError>;

pub(crate) type LayoutCandidateScore = (usize, usize, usize, usize);

/// For setup-primary planning, retain the smallest slice count that reaches
/// the best local setup objective before witness sizing and suffix recursion.
pub(crate) fn prune_locally_unprofitable_slices(
    policy: &PlannerPolicy,
    opening_layout: &OpeningClaimsLayout,
    candidates: Vec<CommittedGroupParams>,
) -> Result<Vec<CommittedGroupParams>, AkitaError> {
    if policy.selection_policy == crate::SelectionPolicyId::MinEstimatedProofPayload
        || candidates.len() <= 1
    {
        return Ok(candidates);
    }
    let mut best: Option<((usize, usize), CommittedGroupParams)> = None;
    for params in candidates {
        let setup_score = padded_setup_prefix_len(active_setup_field_len(&params, opening_layout)?);
        let score = (setup_score, params.outer_slice_count().get());
        if best
            .as_ref()
            .is_none_or(|(best_score, _)| score < *best_score)
        {
            best = Some((score, params));
        }
    }
    Ok(best.map(|(_, params)| vec![params]).unwrap_or_default())
}

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

#[cfg(test)]
#[path = "test/schedule_params.rs"]
mod tests;

#[cfg(all(test, feature = "catalog-gen"))]
#[path = "test/adaptive_dimensions.rs"]
mod adaptive_dimension_tests;

#[cfg(all(test, feature = "catalog-gen"))]
#[path = "test/adaptive_search.rs"]
mod adaptive_search_tests;
