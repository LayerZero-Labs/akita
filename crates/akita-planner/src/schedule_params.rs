//! FoldSchedule planner that applies each catalog-bound selection objective.
//!
//! Public entry: [`crate::find_schedule`]. The search is `Cfg`-free: every
//! per-preset input is carried by the plain-value [`PlannerPolicy`] plus
//! the `ring_challenge_config` closure,
//! exactly the shape generated catalog emission consumes. This keeps the DP a
//! pure function of `(policy, key, dimension domain)` for offline table generation.

use std::{num::NonZeroUsize, sync::Arc};

/// One modeled proof byte has the same weight as this many folded witness elements.
pub(crate) const FOLD_WORK_ELEMENTS_PER_OBJECTIVE_BYTE: u128 = 1 << 18;

use akita_challenges::SparseChallengeConfig;
use akita_error::AkitaError;
use akita_types::sis::{
    decomposed_s_block_ring_count, num_digits_for_linf_cap, num_digits_inner_for_bound,
    num_digits_open, rounded_up_collision_inf_norm, rounded_up_role_a_inf_norm,
    BalancedSignedDigitFoldPolicy, FoldWitnessNorms, HonestFoldPolicy, HonestFoldSizingQuery,
    InnerCommitMatrixParams, OpenCommitMatrixParams, OuterCommitMatrixParams,
};
use akita_types::{
    active_setup_field_len, padded_setup_prefix_len, CommitmentRingDims, CommittedGroupParams,
    DecompositionParams, GroupCommitPhaseParams, GroupOpenPhaseParams, OpeningClaimsLayout,
    PolynomialGroupLayout, TranscriptGrindingCost,
};
#[cfg(all(test, feature = "catalog-gen"))]
use akita_types::{try_extension_opening_reduction_level_bytes, PlannedFoldSchedule};

use crate::{InnerBasisSource, PlannerPolicy};

mod candidate;
mod objective;
mod pareto;
mod relation_transition;
mod setup_score;
mod suffix_dp;
#[cfg(all(test, feature = "catalog-gen"))]
#[path = "test/unpruned_search.rs"]
mod unpruned_search;
pub(crate) use akita_schedules::planner_support::{
    materialize_candidate_schedule, CandidateFoldStep, CandidateMaterializationCost,
    CandidateTerminalResponse,
};
pub use akita_types::suffix_opening_layout;
pub(crate) use candidate::{
    derive_ab_commitment_candidate, derive_fold_candidates, derive_recursive_candidate_views,
    derive_terminal_candidates, recursive_split_search_domain, AbCommitmentCandidateRequest,
    CandidateInnerRoute, CandidateLayoutGuide, FoldCandidatePolicy, PlannerOpeningCandidate,
    RecursiveCandidateRequest, RecursiveFoldWork, SetupPrefixLayoutGuide, SetupPrefixSearchCache,
    SplitBoundPolicy,
};
#[cfg(all(test, feature = "catalog-gen"))]
pub(crate) use candidate::{
    derive_unpruned_fold_candidates_for_oracle, derive_unpruned_terminal_candidates_for_oracle,
};
pub(crate) use objective::{select_complete_candidate, CompleteObjectiveBound};
#[cfg(feature = "test-support")]
pub use relation_transition::TestRelationModeFilter;
pub(crate) use relation_transition::{
    ReducedTransitionRejection, RelationModeFilter, RelationSearchDomain, RelationTraversalOrder,
};
pub(crate) use setup_score::{level_setup_field_elements, terminal_setup_field_elements};
pub(crate) use suffix_dp::{
    derive_selected_suffix_schedule, QuerySearch, ScheduleMemo, SuffixCtx, SuffixState,
    SuffixTopology,
};

pub(crate) fn root_inner_basis_source(
    source: akita_types::sis::CommittedSourceContract,
) -> InnerBasisSource {
    match source.class() {
        akita_types::sis::CommittedSourceClass::UnitOneHot { .. } => InnerBasisSource::UnitOneHot,
        akita_types::sis::CommittedSourceClass::BalancedSignedDigit => {
            InnerBasisSource::RawCoefficients {
                log_bound: source.decomposition().log_commit_bound,
            }
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

#[cfg(all(test, feature = "catalog-gen"))]
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
    #[cfg(feature = "catalog-gen")]
    pub(crate) fn uniform(ring_dimension: usize) -> Result<Self, AkitaError> {
        Self::new([CommitmentRingDims::uniform(ring_dimension)])
    }

    /// Canonically ordered admitted A/B/D tuples.
    pub(crate) fn candidates(&self) -> &[CommitmentRingDims] {
        &self.candidates
    }

    #[cfg(feature = "catalog-gen")]
    pub(crate) fn validate_for_policy(&self, policy: &PlannerPolicy) -> Result<(), AkitaError> {
        akita_schedules::planner_support::validate_policy(policy)
    }
}

#[cfg(all(test, feature = "catalog-gen"))]
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

struct CandidateFoldPartsIter<'a> {
    first: Option<&'a CandidateFoldStep>,
    suffix: CandidateFoldIter<'a>,
    remaining: usize,
}

impl<'a> Iterator for CandidateFoldPartsIter<'a> {
    type Item = &'a CandidateFoldStep;

    fn next(&mut self) -> Option<Self::Item> {
        let step = self.first.take().or_else(|| self.suffix.next())?;
        self.remaining -= 1;
        Some(step)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl ExactSizeIterator for CandidateFoldPartsIter<'_> {}

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

    #[cfg(all(test, feature = "catalog-gen"))]
    fn iter(&self) -> impl ExactSizeIterator<Item = &CandidateFoldStep> {
        CandidateFoldIter {
            next: self.head.as_deref(),
            remaining: self.len,
        }
    }

    fn iter_with_prefix<'a>(
        &'a self,
        first: Option<&'a CandidateFoldStep>,
    ) -> impl ExactSizeIterator<Item = &'a CandidateFoldStep> {
        CandidateFoldPartsIter {
            first,
            suffix: CandidateFoldIter {
                next: self.head.as_deref(),
                remaining: self.len,
            },
            remaining: self.len + usize::from(first.is_some()),
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
    pub(crate) first_direct_output_witness_len: usize,
    pub(crate) cost: NativeProofCost,
    pub(crate) setup_field_elements: usize,
    pub(crate) folds: CandidateFoldChain,
    pub(crate) terminal: Arc<CandidateTerminalResponse>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct NativeProofCost {
    payload_bytes: usize,
    native_nonce_bytes: usize,
    nonce_bits: usize,
    expanded_query_count: u64,
    fold_work_elements: u128,
}

impl NativeProofCost {
    pub(crate) fn new(
        payload_bytes: usize,
        native_nonce_bytes: usize,
        expanded_query_count: u64,
        fold_work_elements: u128,
    ) -> Result<Self, AkitaError> {
        let cost = Self {
            payload_bytes,
            native_nonce_bytes,
            nonce_bits: 0,
            expanded_query_count,
            fold_work_elements,
        };
        cost.validate_objective()?;
        Ok(cost)
    }

    pub(crate) fn proof_bytes(self) -> usize {
        self.checked_proof_bytes()
            .expect("validated native proof cost")
    }

    pub(crate) fn exact_score(self) -> u128 {
        self.checked_exact_score()
            .expect("validated additive proof-and-work cost")
    }

    pub(crate) fn checked_prepend(
        self,
        payload_bytes: usize,
        native_nonce_bytes: usize,
        nonce_bits: usize,
        expanded_query_count: u64,
        fold_work_elements: usize,
    ) -> Result<Self, AkitaError> {
        let payload_bytes = self
            .payload_bytes
            .checked_add(payload_bytes)
            .ok_or_else(|| AkitaError::InvalidSetup("suffix proof payload overflow".into()))?;
        let nonce_bits = self.nonce_bits.checked_add(nonce_bits).ok_or_else(|| {
            AkitaError::InvalidSetup("candidate nonce bit length overflow".into())
        })?;
        let native_nonce_bytes = self
            .native_nonce_bytes
            .checked_add(native_nonce_bytes)
            .ok_or_else(|| AkitaError::InvalidSetup("native nonce byte length overflow".into()))?;
        let expanded_query_count = self
            .expanded_query_count
            .checked_add(expanded_query_count)
            .ok_or_else(|| AkitaError::InvalidSetup("candidate query count overflow".into()))?;
        let fold_work_elements = self
            .fold_work_elements
            .checked_add(fold_work_elements as u128)
            .ok_or_else(|| AkitaError::InvalidSetup("candidate fold work overflow".into()))?;
        let cost = Self {
            payload_bytes,
            native_nonce_bytes,
            nonce_bits,
            expanded_query_count,
            fold_work_elements,
        };
        cost.validate_objective()?;
        Ok(cost)
    }

    pub(crate) fn grinding_cost(self) -> TranscriptGrindingCost {
        TranscriptGrindingCost {
            total_nonce_bits: self.nonce_bits,
            native_nonce_max_bytes: self.native_nonce_bytes,
            expanded_query_count: self.expanded_query_count,
        }
    }

    pub(crate) const fn expanded_query_count(self) -> u64 {
        self.expanded_query_count
    }

    pub(crate) const fn fits_query_limit(self) -> bool {
        self.expanded_query_count < akita_types::TRANSCRIPT_GRINDING_QUERY_LIMIT
    }

    pub(crate) fn never_worse(self, other: Self) -> bool {
        (self.exact_score(), self.proof_bytes()) <= (other.exact_score(), other.proof_bytes())
    }

    pub(crate) fn strictly_better(self, other: Self) -> bool {
        (self.exact_score(), self.proof_bytes()) < (other.exact_score(), other.proof_bytes())
    }

    fn validate_objective(self) -> Result<(), AkitaError> {
        self.checked_exact_score()
            .ok_or_else(|| AkitaError::InvalidSetup("candidate objective overflow".into()))?;
        Ok(())
    }

    fn checked_exact_score(self) -> Option<u128> {
        (self.checked_proof_bytes()? as u128)
            .checked_mul(FOLD_WORK_ELEMENTS_PER_OBJECTIVE_BYTE)?
            .checked_add(self.fold_work_elements)
    }

    fn checked_proof_bytes(self) -> Option<usize> {
        self.payload_bytes.checked_add(self.native_nonce_bytes)
    }
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
    pub(crate) first_direct_output_witness_len: usize,
    pub(crate) cost: NativeProofCost,
    pub(crate) setup_field_elements: usize,
}

impl CandidateMetrics {
    pub(crate) fn proof_bytes(self) -> usize {
        self.cost.proof_bytes()
    }
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
            first_direct_output_witness_len: self.first_direct_output_witness_len,
            cost: self.cost,
            setup_field_elements: self.setup_field_elements,
        }
    }
}

pub(crate) fn candidate_schedule_descriptor_bytes(
    first_fold: Option<&CandidateFoldStep>,
    suffix_folds: &CandidateFoldChain,
    terminal: &akita_types::TerminalFoldParams,
    diagnostics: Option<&crate::diagnostics::PlannerDiagnostics>,
) -> Result<Vec<u8>, AkitaError> {
    let started = diagnostics.map(|_| std::time::Instant::now());
    let result = (|| {
        let fold_count = suffix_folds.len() + usize::from(first_fold.is_some());
        if fold_count == 0 {
            return Ok(terminal.canonical_descriptor_bytes());
        }
        let folds = || suffix_folds.iter_with_prefix(first_fold);
        let carrier_prefix_len = fold_count.min(2);
        let mut bytes = Vec::new();
        bytes.push(carrier_prefix_len as u8);
        bytes.extend(
            folds()
                .take(carrier_prefix_len)
                .map(|fold| fold.params.payload_mode.tag()),
        );
        let descriptor_steps =
            folds()
                .enumerate()
                .map(|(index, fold)| akita_types::FoldScheduleDescriptorStep {
                    params: &fold.params,
                    payload_mode: if index < carrier_prefix_len {
                        akita_types::CommitmentPayloadMode::Compressed
                    } else {
                        fold.params.payload_mode
                    },
                    input_witness_len: fold.input_witness_len,
                    output_witness_len: fold.output_witness_len,
                });
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

/// For setup-primary planning, retain every slice that reaches the best local
/// setup objective before witness sizing and suffix recursion. Equal setup
/// candidates can still differ in proof size or the complete descriptor.
pub(crate) fn prune_locally_unprofitable_slices(
    policy: &PlannerPolicy,
    opening_layout: &OpeningClaimsLayout,
    candidates: Vec<CommittedGroupParams>,
) -> Result<Vec<CommittedGroupParams>, AkitaError> {
    if policy.selection_policy == crate::SelectionPolicyId::MinEstimatedExactProofAndWorkV4
        || candidates.len() <= 1
    {
        return Ok(candidates);
    }
    let mut best_setup = None;
    let mut retained = Vec::new();
    for params in candidates {
        let setup_score = match policy.selection_policy {
            crate::SelectionPolicyId::MinFirstDirectSetupThenExactProofAndWorkV4 => {
                padded_setup_prefix_len(active_setup_field_len(&params, opening_layout)?)
            }
            crate::SelectionPolicyId::MinPaddedSetupEnvelopeThenFirstDirectThenExactProofAndWorkV5 => {
                padded_setup_prefix_len(level_setup_field_elements(&params)?)
            }
            crate::SelectionPolicyId::MinEstimatedExactProofAndWorkV4 => unreachable!(),
        };
        match best_setup.map(|best| setup_score.cmp(&best)) {
            None | Some(std::cmp::Ordering::Less) => {
                best_setup = Some(setup_score);
                retained.clear();
                retained.push(params);
            }
            Some(std::cmp::Ordering::Equal) => retained.push(params),
            Some(std::cmp::Ordering::Greater) => {}
        }
    }
    Ok(retained)
}

/// Combine exact physical width, challenge work, chunk evaluator work,
/// and load imbalance when comparing `M` candidates. All terms count ring or
/// scalar work units; exact physical width remains an explicit tie-breaker.
pub(crate) fn layout_candidate_score(
    physical_width: usize,
    num_live_blocks: usize,
    num_chunks: usize,
) -> Result<LayoutCandidateScore, AkitaError> {
    if num_live_blocks == 0
        || num_chunks == 0
        || num_chunks > akita_types::MAX_WITNESS_CHUNKS
        || !num_chunks.is_power_of_two()
    {
        return Err(AkitaError::InvalidSetup(
            "layout candidate chunk geometry is malformed".to_string(),
        ));
    }
    let challenge_work = num_live_blocks;
    let chunk_work = num_live_blocks;
    // Canonical proportional partitioning gives every chunk either
    // `floor(blocks / chunks)` or `ceil(blocks / chunks)` blocks.
    let imbalance = usize::from(!num_live_blocks.is_multiple_of(num_chunks));
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

pub(crate) use akita_types::{RelationCandidateTopology, RingRelationPhase};
