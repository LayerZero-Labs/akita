use std::{
    collections::{BTreeMap, HashMap},
    num::NonZeroUsize,
    sync::Arc,
};

use akita_field::AkitaError;
use akita_types::{
    active_setup_field_len, terminal_response_planner_bytes,
    try_extension_opening_reduction_level_bytes, AkitaScheduleLookupKey, CommitmentRingDims,
    CommittedGroupParams, OpeningClaimsLayout, PolynomialGroupLayout, TerminalResponseShape,
};

use crate::{planner::root_level_candidates_for_basis, PlannerPolicy};

use super::{
    derive_candidate_level_params, derive_candidate_level_params_split_frontier,
    derive_recursive_candidate_views, derive_terminal_candidate_params, dimension_candidates,
    level_setup_field_elements, suffix_opening_layout, terminal_setup_field_elements,
    CandidateFoldStep, CandidateTerminalResponse, CompleteObjectiveBound, MixedScore,
    ScheduleCandidate, SetupPrefixCapacity, SetupPrefixSearchCache,
};
use akita_schedules::planner_support::MAX_RECURSION_DEPTH;

mod frontier;
mod prune;
mod state;
mod terminal;

use frontier::{consider_child_suffixes, FrontierProjection, ProjectedFrontier};
use state::*;
pub(crate) use state::{ScheduleMemo, SuffixCtx, SuffixState};
pub(crate) use terminal::try_terminal_direct_suffix_cost;

fn offloaded_witness_contracts(
    input_witness_len: usize,
    input_log_basis: u32,
    setup_prefix_field_len: usize,
    field_bits: u32,
    output_witness_len: usize,
    output_log_basis: u32,
    minimum_contraction: usize,
) -> Result<bool, AkitaError> {
    let input_bits = input_witness_len
        .checked_mul(input_log_basis as usize)
        .and_then(|bits| {
            setup_prefix_field_len
                .checked_mul(field_bits as usize)
                .and_then(|prefix_bits| bits.checked_add(prefix_bits))
        })
        .ok_or_else(|| AkitaError::InvalidSetup("input witness bit length overflow".to_string()))?;
    let minimum_input_bits = output_witness_len
        .checked_mul(output_log_basis as usize)
        .and_then(|bits| bits.checked_mul(minimum_contraction))
        .ok_or_else(|| {
            AkitaError::InvalidSetup("offloaded witness contraction overflow".to_string())
        })?;
    Ok(input_bits >= minimum_input_bits)
}

struct ChildEdge<'a> {
    policy: &'a PlannerPolicy,
    diagnostics: Option<&'a crate::diagnostics::PlannerDiagnostics>,
    candidate_params: Arc<CommittedGroupParams>,
    current_witness_len: usize,
    next_witness_len: usize,
    natural_setup_field_len: usize,
    level_setup_field_elements: usize,
    offloaded: bool,
    require_child_fold: bool,
    setup_field_budget: Option<usize>,
}

struct PendingScheduleCandidate {
    first_direct_setup_field_len: Option<NonZeroUsize>,
    total_bytes: usize,
    setup_field_elements: usize,
    first_fold: CandidateFoldStep,
    suffix_folds: super::CandidateFoldChain,
    terminal: Arc<CandidateTerminalResponse>,
}

#[derive(Clone)]
struct OpeningWork {
    dimensions: CommitmentRingDims,
    opening: crate::schedule_params::PlannerOpeningCandidate,
    precommitted_openings: Vec<crate::schedule_params::PlannerOpeningCandidate>,
    opening_reduction_bytes: usize,
    allows_terminal: bool,
    allows_fold: bool,
}

type LevelCandidate = (
    CommittedGroupParams,
    usize,
    usize,
    Option<crate::response_model::SourceMomentEstimate>,
);

type GuidedLevelCandidate = (CompleteObjectiveBound, Option<usize>, LevelCandidate);

enum CandidateTraversal {
    Plain(std::vec::IntoIter<LevelCandidate>),
    Guided(std::vec::IntoIter<GuidedLevelCandidate>),
}

#[derive(Clone, Copy)]
enum GuideScope {
    CompleteRoot,
    RecursivePrefix,
}

impl GuideScope {
    fn for_state(
        policy: &PlannerPolicy,
        is_complete_root: bool,
        incoming_setup_prefix: Option<usize>,
    ) -> Option<Self> {
        if is_complete_root {
            Some(Self::CompleteRoot)
        } else if incoming_setup_prefix.is_some()
            && policy.selection_policy == crate::SelectionPolicyId::MinFirstDirectSetupThenPayload
        {
            Some(Self::RecursivePrefix)
        } else {
            None
        }
    }
}

impl Iterator for CandidateTraversal {
    type Item = (
        Option<(CompleteObjectiveBound, Option<usize>)>,
        LevelCandidate,
    );

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Plain(candidates) => candidates.next().map(|candidate| (None, candidate)),
            Self::Guided(candidates) => candidates
                .next()
                .map(|(bound, natural_len, candidate)| (Some((bound, natural_len)), candidate)),
        }
    }
}

pub(super) const fn state_allows_terminal_seed(
    is_root_level: bool,
    has_incoming_setup_prefix: bool,
) -> bool {
    !is_root_level && !has_incoming_setup_prefix
}

pub(super) fn packing_precommit_opening_products(
    policy: &PlannerPolicy,
    dimensions: CommitmentRingDims,
    key: &AkitaScheduleLookupKey,
) -> Result<Vec<Vec<crate::schedule_params::PlannerOpeningCandidate>>, AkitaError> {
    let mut products = vec![Vec::new()];
    for profile in &key.precommitteds {
        let domain = crate::schedule_params::PlannerOpeningCandidate::coefficient_packing_domain(
            0,
            policy.claim_ext_degree,
            CommitmentRingDims {
                inner: profile.inner_commit_matrix.ring_dimension(),
                outer: profile.outer_commit_matrix.ring_dimension(),
                opening: dimensions.d_d(),
            },
        )?;
        if domain.is_empty() {
            return Ok(Vec::new());
        }
        let next_len = products.len().checked_mul(domain.len()).ok_or_else(|| {
            AkitaError::InvalidSetup("root precommit opening search domain overflow".into())
        })?;
        let mut next = Vec::new();
        next.try_reserve_exact(next_len).map_err(|_| {
            AkitaError::InvalidSetup("root precommit opening search domain is too large".into())
        })?;
        for product in products {
            for &opening in &domain {
                let mut extended = product.clone();
                extended.push(opening);
                next.push(extended);
            }
        }
        products = next;
    }
    Ok(products)
}

impl PendingScheduleCandidate {
    fn metrics(&self) -> super::CandidateMetrics {
        super::CandidateMetrics {
            first_direct_setup_capacity: self
                .first_direct_setup_field_len
                .map_or(super::SetupPrefixCapacity::MAX, |natural_len| {
                    super::SetupPrefixCapacity::for_natural_len(natural_len.get())
                }),
            proof_bytes: self.total_bytes,
            setup_field_elements: self.setup_field_elements,
        }
    }

    fn into_candidate(self) -> ScheduleCandidate {
        ScheduleCandidate {
            first_direct_setup_field_len: self.first_direct_setup_field_len,
            total_bytes: self.total_bytes,
            setup_field_elements: self.setup_field_elements,
            folds: self.suffix_folds.prepend(self.first_fold),
            terminal: self.terminal,
        }
    }
}

fn child_choice(
    edge: &ChildEdge<'_>,
    suffix: &ScheduleCandidate,
) -> Result<Option<PendingScheduleCandidate>, AkitaError> {
    if !frontier::ParentAdmissionClass::for_candidate(suffix).is_admitted_by(
        edge.require_child_fold,
        edge.offloaded,
        edge.natural_setup_field_len,
    ) {
        return Ok(None);
    }

    let (direct_payload_bytes, stage3_payload_bytes) =
        akita_schedules::planner_support::nonterminal_level_payload_bytes(
            edge.policy,
            &edge.candidate_params,
            suffix.first_fold_params(),
            edge.current_witness_len,
            edge.next_witness_len,
        )?;
    if edge.offloaded != (stage3_payload_bytes != 0) {
        return Err(AkitaError::InvalidSetup(
            "setup edge topology disagrees with Stage-3 accounting".to_string(),
        ));
    }
    let total_bytes = direct_payload_bytes
        .checked_add(stage3_payload_bytes)
        .and_then(|value| value.checked_add(suffix.total_bytes))
        .ok_or_else(|| AkitaError::InvalidSetup("suffix proof size overflow".to_string()))?;
    let setup_field_elements = edge
        .level_setup_field_elements
        .max(suffix.setup_field_elements);
    if edge
        .setup_field_budget
        .is_some_and(|budget| setup_field_elements > budget)
    {
        return Ok(None);
    }
    let first_direct_setup_field_len = if edge.offloaded {
        suffix.first_direct_setup_field_len
    } else {
        Some(
            NonZeroUsize::new(edge.natural_setup_field_len).ok_or_else(|| {
                AkitaError::InvalidSetup("direct setup field length must be nonzero".into())
            })?,
        )
    };
    let first_fold = CandidateFoldStep {
        params: Arc::clone(&edge.candidate_params),
        input_witness_len: edge.current_witness_len,
        output_witness_len: edge.next_witness_len,
        estimated_direct_payload_bytes: direct_payload_bytes,
        estimated_stage3_payload_bytes: stage3_payload_bytes,
    };
    Ok(Some(PendingScheduleCandidate {
        first_direct_setup_field_len,
        total_bytes,
        setup_field_elements,
        first_fold,
        suffix_folds: suffix.folds.clone(),
        terminal: suffix.terminal.clone(),
    }))
}

fn consider_mixed_child_suffixes<'a>(
    edge: &ChildEdge<'_>,
    child_candidates: impl Iterator<Item = &'a ScheduleCandidate>,
    frontier: &mut MixedFrontier,
) -> Result<(), AkitaError> {
    for suffix in child_candidates {
        let Some(candidate) = child_choice(edge, suffix)? else {
            continue;
        };
        insert_mixed_frontier(edge.policy, frontier, candidate.into_candidate())?;
    }
    Ok(())
}

fn direct_edge_lower_bound(
    policy: &PlannerPolicy,
    params: &CommittedGroupParams,
    input_witness_len: usize,
    output_witness_len: usize,
    natural_setup_field_len: usize,
) -> Result<CompleteObjectiveBound, AkitaError> {
    let (proof_bytes, stage3_bytes) =
        akita_schedules::planner_support::nonterminal_level_payload_bytes(
            policy,
            params,
            None,
            input_witness_len,
            output_witness_len,
        )?;
    if stage3_bytes != 0 {
        return Err(AkitaError::InvalidSetup(
            "direct-edge lower bound unexpectedly includes Stage-3 bytes".into(),
        ));
    }
    Ok(CompleteObjectiveBound::for_direct_edge(
        policy,
        SetupPrefixCapacity::for_natural_len(natural_setup_field_len).field_elements(),
        proof_bytes,
        level_setup_field_elements(params)?,
    ))
}

fn complete_root_bound_is_strictly_worse(
    policy: &PlannerPolicy,
    lower_bound: CompleteObjectiveBound,
    frontier: &ProjectedFrontier,
    mixed_frontier: &MixedFrontier,
) -> bool {
    match policy.selection_policy {
        crate::SelectionPolicyId::MinEstimatedProofPayload => frontier
            .by_parent_cost
            .values()
            .flat_map(frontier::ObjectiveChoices::payload_candidates)
            .any(|candidate| lower_bound.is_strictly_worse_than(candidate.metrics())),
        crate::SelectionPolicyId::MinSetupMatrixFieldElementsThenProofPayload => mixed_frontier
            .candidates()
            .any(|candidate| lower_bound.is_strictly_worse_than(candidate.metrics())),
        crate::SelectionPolicyId::MinFirstDirectSetupThenPayload => frontier
            .by_parent_cost
            .values()
            .flat_map(frontier::ObjectiveChoices::setup_candidates)
            .any(|candidate| lower_bound.is_strictly_worse_than(candidate.metrics())),
    }
}

fn direct_edge_bound_is_strictly_worse(
    policy: &PlannerPolicy,
    guide_scope: GuideScope,
    params: &CommittedGroupParams,
    natural_setup_field_len: usize,
    lower_bound: CompleteObjectiveBound,
    frontier: &ProjectedFrontier,
    mixed_frontier: &MixedFrontier,
) -> Result<bool, AkitaError> {
    match guide_scope {
        GuideScope::CompleteRoot => Ok(complete_root_bound_is_strictly_worse(
            policy,
            lower_bound,
            frontier,
            mixed_frontier,
        )),
        GuideScope::RecursivePrefix => {
            let parent_cost = ParentObservableKey::new(policy, Some(params))?;
            Ok(frontier.recursive_direct_bound_is_strictly_worse(
                &parent_cost,
                SetupPrefixCapacity::for_natural_len(natural_setup_field_len),
                lower_bound,
            ))
        }
    }
}

fn candidate_traversal(
    policy: &PlannerPolicy,
    guide_scope: Option<GuideScope>,
    opening_layout: &OpeningClaimsLayout,
    current_witness_len: usize,
    candidates: Vec<LevelCandidate>,
) -> Result<CandidateTraversal, AkitaError> {
    if guide_scope.is_none() {
        return Ok(CandidateTraversal::Plain(candidates.into_iter()));
    }
    let mut guided = candidates
        .into_iter()
        .map(|candidate| {
            let natural_len = (policy.selection_policy
                == crate::SelectionPolicyId::MinFirstDirectSetupThenPayload)
                .then(|| active_setup_field_len(&candidate.0, opening_layout))
                .transpose()?;
            let lower_bound = direct_edge_lower_bound(
                policy,
                &candidate.0,
                current_witness_len,
                candidate.1,
                natural_len.unwrap_or_default(),
            )?;
            Ok((lower_bound, natural_len, candidate))
        })
        .collect::<Result<Vec<_>, AkitaError>>()?;
    guided.sort_by_key(|(lower_bound, _, (_, next_witness_len, _, _))| {
        (*lower_bound, *next_witness_len)
    });
    Ok(CandidateTraversal::Guided(guided.into_iter()))
}

#[allow(clippy::too_many_arguments)]
fn price_terminal_candidate(
    ctx: &SuffixCtx<'_>,
    state: SuffixState,
    candidate_params: &CommittedGroupParams,
    opening_reduction_bytes: usize,
    natural_len: usize,
    frontier: &mut ProjectedFrontier,
    mixed_frontier: &mut MixedFrontier,
) -> Result<(), AkitaError> {
    let policy = ctx.policy;
    let direct_projection =
        if state.incoming_setup_prefix.is_some() || (ctx.level_zero_is_root && state.level == 0) {
            FrontierProjection::Both
        } else {
            FrontierProjection::Payload
        };
    if (ctx.level_zero_is_root && state.level == 0)
        || state.incoming_setup_prefix.is_some()
        || candidate_params.has_precommitted_groups()
    {
        return Ok(());
    }
    let field_bits = policy.decomposition.field_bits();
    let Some((mut direct_step, suffix_cost)) = try_terminal_direct_suffix_cost(
        policy,
        state.current_witness_len,
        candidate_params,
        field_bits,
        ctx.key,
        state.level,
        None,
        state.source_moment,
    )?
    else {
        return Ok(());
    };
    let level_proof_size = akita_types::proof_size::FOLD_GRIND_NONCE_BYTES
        .checked_add(opening_reduction_bytes)
        .ok_or_else(|| AkitaError::InvalidSetup("terminal proof size overflow".into()))?;
    let total = level_proof_size
        .checked_add(suffix_cost)
        .ok_or_else(|| AkitaError::InvalidSetup("terminal proof size overflow".to_string()))?;
    direct_step.estimated_direct_payload_bytes = level_proof_size;
    let candidate = ScheduleCandidate {
        first_direct_setup_field_len: Some(NonZeroUsize::new(natural_len).ok_or_else(|| {
            AkitaError::InvalidSetup("direct setup field length must be nonzero".into())
        })?),
        total_bytes: total,
        setup_field_elements: terminal_setup_field_elements(&direct_step.params)?,
        folds: super::CandidateFoldChain::default(),
        terminal: Arc::new(direct_step),
    };
    frontier.consider_candidate(
        policy,
        ctx.diagnostics,
        candidate.clone(),
        direct_projection,
    )?;
    insert_mixed_frontier(policy, mixed_frontier, candidate)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn price_level_candidate_with_children(
    ctx: &SuffixCtx<'_>,
    state: SuffixState,
    candidate_params: &CommittedGroupParams,
    next_witness_len: usize,
    natural_len: usize,
    direct_child: Option<&SuffixResult>,
    offloaded_child: Option<&SuffixResult>,
    require_child_fold: bool,
    frontier: &mut ProjectedFrontier,
    mixed_frontier: &mut MixedFrontier,
) -> Result<(), AkitaError> {
    let policy = ctx.policy;
    // Only a prefix-consuming state is read through the setup projection by
    // an offloaded parent. The top-level recursive objective also reads the
    // root setup projection. Ordinary direct suffixes are consumed solely
    // through the payload projection, so retaining a parallel setup winner
    // there duplicates frontier work and memo ownership with no observer.
    let direct_projection =
        if state.incoming_setup_prefix.is_some() || (ctx.level_zero_is_root && state.level == 0) {
            FrontierProjection::Both
        } else {
            FrontierProjection::Payload
        };
    let level_setup_field_elements = level_setup_field_elements(candidate_params)?;
    let direct_edge = ChildEdge {
        policy,
        diagnostics: ctx.diagnostics,
        candidate_params: Arc::new(candidate_params.clone()),
        current_witness_len: state.current_witness_len,
        next_witness_len,
        natural_setup_field_len: natural_len,
        level_setup_field_elements,
        offloaded: false,
        require_child_fold,
        setup_field_budget: ctx.setup_field_budget,
    };
    if let Some(direct_child) = direct_child {
        consider_child_suffixes(
            &direct_edge,
            direct_child.payload_candidates(),
            state.incoming_setup_prefix,
            direct_projection,
            frontier,
        )?;
        if state.incoming_setup_prefix.is_none() {
            consider_mixed_child_suffixes(
                &direct_edge,
                direct_child.mixed_frontier.candidates(),
                mixed_frontier,
            )?;
        }
    }
    if let Some(offloaded_child) = offloaded_child {
        let offloaded_edge = ChildEdge {
            offloaded: true,
            ..direct_edge
        };
        consider_child_suffixes(
            &offloaded_edge,
            offloaded_child.setup_candidates(),
            state.incoming_setup_prefix,
            FrontierProjection::FirstDirectSetup,
            frontier,
        )?;
        consider_child_suffixes(
            &offloaded_edge,
            offloaded_child.payload_candidates(),
            state.incoming_setup_prefix,
            FrontierProjection::Payload,
            frontier,
        )?;
    }

    Ok(())
}

/// Shared inputs for root-level `CommittedGroupParams` candidates.
/// Suffix DP for the selected recursive schedule at
/// `(level, current_witness_len, current_lb)`.
///
/// At each state, the projected maps keep the setup and payload winners for
/// each parent-visible first-fold key (from
/// [`derive_candidate_level_params`]). A candidate may terminate on the current
/// witness when there is no incoming setup prefix, or fold again and consume
/// `incoming_setup_prefix` when present. Fold-again edges plan exactly one child
/// state: recursive setup edges pass the outgoing setup prefix to the child,
/// while direct edges plan the ordinary no-prefix child.
pub(crate) fn derive_selected_suffix_schedule(
    ctx: &SuffixCtx<'_>,
    memo: &mut ScheduleMemo,
    state: SuffixState,
    depth: usize,
) -> Result<Arc<SuffixResult>, AkitaError> {
    let SuffixCtx {
        policy,
        diagnostics,
        ring_challenge_config,
        key: _,
        setup_field_budget: _,
        root_lookup_key,
        root_honest_fold_policy,
        precommitted_honest_fold_policies,
        level_zero_is_root,
    } = *ctx;
    if let Some(diagnostics) = diagnostics {
        diagnostics.record_suffix_call();
    }
    let SuffixState {
        level,
        current_witness_len,
        current_lb,
        source_moment,
        incoming_setup_prefix,
        dimension_ceiling,
        payload_phase,
    } = state;
    let memo_key = state.memo_key(policy);
    if depth <= MAX_RECURSION_DEPTH {
        let cached = memo.get(&memo_key);
        if let Some(diagnostics) = diagnostics {
            diagnostics.record_memo_result(cached.is_some());
        }
        if let Some(cached) = cached {
            return Ok(Arc::clone(cached));
        }
    }

    if depth > MAX_RECURSION_DEPTH {
        // Depth-overflow states are never read from the memo: the lookup above
        // is deliberately restricted to admissible depths. Caching these
        // write-only empty results used to evict hot exact suffixes during wide
        // searches and could turn one catalog row into millions of redundant
        // recomputations.
        return Ok(empty_suffix_result());
    }
    if policy.selective_l2_response_model_enabled()
        && !(level_zero_is_root && level == 0)
        && source_moment.is_none()
    {
        return Err(AkitaError::InvalidSetup(
            "recursive suffix is missing its response source moment".into(),
        ));
    }
    let retains_setup_projection =
        incoming_setup_prefix.is_some() || (level_zero_is_root && level == 0);
    let mut payload_only = BTreeMap::new();
    let mut setup_and_payload: BTreeMap<ParentObservableKey, frontier::ObjectiveChoices> =
        BTreeMap::new();
    let mut mixed_frontier = MixedFrontier::new();
    let root_level_key = root_lookup_key.filter(|_| level == 0);
    if root_level_key.is_some() && incoming_setup_prefix.is_some() {
        return Err(AkitaError::InvalidSetup(
            "root batch cannot consume an incoming setup prefix".to_string(),
        ));
    }
    if level_zero_is_root && level == 0 && root_level_key.is_none() {
        return Err(AkitaError::InvalidSetup(
            "root-level suffix state is missing its opening lookup key".to_string(),
        ));
    }
    if payload_phase == akita_types::CommitmentPayloadPhase::RawSuffix
        && incoming_setup_prefix.is_some()
    {
        return Err(AkitaError::InvalidSetup(
            "raw commitment suffix cannot consume a recursive setup prefix".to_string(),
        ));
    }
    let root_opening_layout = root_level_key
        .map(AkitaScheduleLookupKey::opening_layout)
        .transpose()?;
    let scalar_opening_layout = if root_level_key.is_some() {
        None
    } else {
        Some(suffix_opening_layout(
            current_witness_len,
            incoming_setup_prefix,
        )?)
    };
    let eor_opening_shape = root_opening_layout
        .as_ref()
        .or(scalar_opening_layout.as_ref())
        .ok_or_else(|| AkitaError::InvalidSetup("opening layout is missing".into()))?
        .aggregate_polynomial_group_layout()?;
    let inner_source = if level_zero_is_root && level == 0 {
        super::root_inner_basis_source(
            root_honest_fold_policy.ok_or_else(|| {
                AkitaError::InvalidSetup("root batch is missing its honest fold policy".into())
            })?,
            policy.decomposition.log_commit_bound,
        )
    } else {
        crate::InnerBasisSource::BalancedDigits {
            log_basis: current_lb,
        }
    };
    let (min_inner_basis, max_inner_basis) = inner_source.search_range(policy)?;
    let (min_open_basis, max_open_basis) =
        crate::policy::log_basis_search_range_at_level(policy, level);
    let mut dimension_work = Vec::new();
    let mut early_packing_work = Vec::new();
    for dimensions in dimension_candidates(policy, level, dimension_ceiling)? {
        let early_packing_level = level <= 1;
        // A direct terminal response cannot consume an attached setup prefix:
        // that prefix must first participate in an emitted recursive fold.
        // Root batches likewise need their emitted root fold before terminal.
        let terminal_seed_is_relevant =
            state_allows_terminal_seed(root_level_key.is_some(), incoming_setup_prefix.is_some());
        let packing_domain = early_packing_level
            .then(|| {
                crate::schedule_params::PlannerOpeningCandidate::coefficient_packing_domain(
                    level,
                    policy.claim_ext_degree,
                    dimensions,
                )
            })
            .transpose()?
            .unwrap_or_default();
        let root_precommit_products = if early_packing_level {
            root_level_key
                .map(|root_key| packing_precommit_opening_products(policy, dimensions, root_key))
                .transpose()?
        } else {
            None
        };
        if let Ok(ring_challenge_cfg) = ring_challenge_config(dimensions.d_a()) {
            if let Some(opening_reduction_bytes) = try_extension_opening_reduction_level_bytes(
                policy.challenge_field_bits()?,
                policy.claim_ext_degree,
                eor_opening_shape,
            )? {
                let precommitted_openings = if let Some(root_key) = root_level_key {
                    let mut openings = Vec::with_capacity(root_key.precommitteds.len());
                    let mut valid = true;
                    for profile in &root_key.precommitteds {
                        let Ok(config) =
                            ring_challenge_config(profile.inner_commit_matrix.ring_dimension())
                        else {
                            valid = false;
                            break;
                        };
                        openings.push(
                            crate::schedule_params::PlannerOpeningCandidate::evaluation_trace(
                                config,
                            ),
                        );
                    }
                    valid.then_some(openings)
                } else {
                    Some(Vec::new())
                };
                if let Some(precommitted_openings) = precommitted_openings {
                    let trace_work = OpeningWork {
                        dimensions,
                        opening: crate::schedule_params::PlannerOpeningCandidate::evaluation_trace(
                            ring_challenge_cfg,
                        ),
                        precommitted_openings,
                        opening_reduction_bytes,
                        allows_terminal: terminal_seed_is_relevant,
                        allows_fold: !early_packing_level,
                    };
                    if early_packing_level {
                        if terminal_seed_is_relevant {
                            dimension_work.push(trace_work);
                        }
                    } else {
                        dimension_work.push(trace_work);
                    }
                }
            }
        }
        if !packing_domain.is_empty() {
            if let Some(precommit_products) = root_precommit_products.as_ref() {
                for opening in packing_domain {
                    for precommitted_openings in precommit_products {
                        early_packing_work.push(OpeningWork {
                            dimensions,
                            opening,
                            precommitted_openings: precommitted_openings.clone(),
                            opening_reduction_bytes: 0,
                            allows_terminal: false,
                            allows_fold: true,
                        });
                    }
                }
            } else {
                early_packing_work.extend(packing_domain.into_iter().map(|opening| OpeningWork {
                    dimensions,
                    opening,
                    precommitted_openings: Vec::new(),
                    opening_reduction_bytes: 0,
                    allows_terminal: false,
                    allows_fold: true,
                }));
            }
        }
    }
    if level <= 1 {
        dimension_work.extend(early_packing_work);
    }
    // Every opening basis contributes to one state frontier. In particular,
    // terminal-direct candidates have no first fold and therefore share the
    // `None` key; they must be compared by the canonical objective instead of
    // being overwritten by the last basis visited.
    let mut frontier = ProjectedFrontier::default();
    for open_lb in min_open_basis..=max_open_basis {
        if open_lb < current_lb {
            continue;
        }
        let current_opening_layout = if root_level_key.is_some() {
            root_opening_layout.as_ref().ok_or_else(|| {
                AkitaError::InvalidSetup("root batch opening layout is missing".to_string())
            })?
        } else {
            scalar_opening_layout.as_ref().ok_or_else(|| {
                AkitaError::InvalidSetup("scalar suffix opening layout is missing".to_string())
            })?
        };
        let require_child_fold =
            root_level_key.is_some_and(|root_key| !root_key.precommitteds.is_empty());
        let mut fold_candidates = Vec::new();
        let mut terminal_candidates = Vec::new();

        for inner_lb in min_inner_basis..=max_inner_basis {
            if let Some(root_key) = root_level_key {
                for work in &dimension_work {
                    let dimension_candidates = root_level_candidates_for_basis(
                        root_key,
                        root_honest_fold_policy.ok_or_else(|| {
                            AkitaError::InvalidSetup(
                                "root batch is missing its honest fold policy".to_string(),
                            )
                        })?,
                        precommitted_honest_fold_policies,
                        policy,
                        work.dimensions,
                        work.opening,
                        &work.precommitted_openings,
                        current_witness_len,
                        inner_lb,
                        open_lb,
                        true,
                    )?;
                    for (params, next_witness_len) in dimension_candidates {
                        if work.allows_terminal {
                            terminal_candidates.push((
                                params.clone(),
                                next_witness_len,
                                work.opening_reduction_bytes,
                            ));
                        }
                        if work.allows_fold {
                            fold_candidates.push((
                                params,
                                next_witness_len,
                                work.opening_reduction_bytes,
                            ));
                        }
                    }
                }
            } else {
                for work in &dimension_work {
                    for &mode in
                        payload_phase.candidate_modes(level, incoming_setup_prefix.is_some())
                    {
                        // Proof-first uniform search compares complete schedules across the
                        // same early split frontier as the bounds-disabled oracle.
                        let retain_split_frontier = incoming_setup_prefix.is_some()
                            || (policy.selection_policy
                                == crate::SelectionPolicyId::MinEstimatedProofPayload
                                && level < akita_schedules::ADAPTIVE_SEARCH_LEVELS)
                            || matches!(
                                policy.ring_dimension_schedule_mode,
                                crate::RingDimensionScheduleMode::AdaptiveDimension {
                                    num_search_levels,
                                    ..
                                } if level < num_search_levels
                            );
                        if work.allows_terminal
                            && work.allows_fold
                            && incoming_setup_prefix.is_none()
                        {
                            let views = derive_recursive_candidate_views(
                                policy,
                                mode,
                                work.opening,
                                work.dimensions,
                                current_witness_len,
                                inner_source,
                                inner_lb,
                                open_lb,
                                level,
                                source_moment,
                                retain_split_frontier,
                            )?;
                            terminal_candidates.extend(
                                views
                                    .terminal
                                    .into_iter()
                                    .map(|params| (params, 0, work.opening_reduction_bytes)),
                            );
                            fold_candidates.extend(views.folds.into_iter().map(
                                |(params, next_witness_len)| {
                                    (params, next_witness_len, work.opening_reduction_bytes)
                                },
                            ));
                            continue;
                        }
                        if work.allows_terminal {
                            terminal_candidates.extend(
                                derive_terminal_candidate_params(
                                    policy,
                                    mode,
                                    work.opening,
                                    work.dimensions,
                                    current_witness_len,
                                    inner_source,
                                    inner_lb,
                                    open_lb,
                                    level,
                                    source_moment,
                                )?
                                .into_iter()
                                .map(|params| (params, 0, work.opening_reduction_bytes)),
                            );
                        }
                        if !work.allows_fold {
                            continue;
                        }
                        let level_candidates = if retain_split_frontier {
                            derive_candidate_level_params_split_frontier(
                                Some(&mut memo.setup_prefixes),
                                policy,
                                mode,
                                work.opening,
                                work.dimensions,
                                current_witness_len,
                                inner_source,
                                inner_lb,
                                open_lb,
                                level,
                                incoming_setup_prefix,
                                source_moment,
                            )?
                        } else {
                            derive_candidate_level_params(
                                Some(&mut memo.setup_prefixes),
                                policy,
                                mode,
                                work.opening,
                                work.dimensions,
                                current_witness_len,
                                inner_source,
                                inner_lb,
                                open_lb,
                                level,
                                incoming_setup_prefix,
                                source_moment,
                            )?
                        };
                        for (params, next_witness_len) in level_candidates {
                            fold_candidates.push((
                                params,
                                next_witness_len,
                                work.opening_reduction_bytes,
                            ));
                        }
                    }
                }
            }
        }
        let attach_source_moments = |candidates: Vec<_>| -> Result<Vec<_>, AkitaError> {
            let mut candidates_with_source = Vec::with_capacity(candidates.len());
            for (candidate_params, next_witness_len, opening_reduction_bytes) in candidates {
                let next_source_moment = if policy.selective_l2_response_model_enabled() {
                    let source_groups = if root_level_key.is_some() {
                        crate::response_model::root_group_source_moments(
                            &candidate_params,
                            current_opening_layout,
                            root_honest_fold_policy.ok_or_else(|| {
                                AkitaError::InvalidSetup(
                                    "root batch is missing its response source policy".into(),
                                )
                            })?,
                            precommitted_honest_fold_policies,
                            policy.decomposition.field_bits(),
                        )?
                    } else if let Some(natural_prefix_len) = incoming_setup_prefix {
                        let prefix_params =
                            candidate_params.group_params(current_opening_layout, 0)?;
                        let prefix_moment = crate::response_model::uniform_field_source_moment(
                            natural_prefix_len,
                            policy.decomposition.field_bits(),
                            prefix_params.log_basis_inner(),
                            prefix_params.num_digits_inner(),
                        )?;
                        vec![
                            prefix_moment,
                            source_moment.ok_or_else(|| {
                                AkitaError::InvalidSetup(
                                    "recursive response source is missing".into(),
                                )
                            })?,
                        ]
                    } else {
                        vec![source_moment.ok_or_else(|| {
                            AkitaError::InvalidSetup("recursive response source is missing".into())
                        })?]
                    };
                    Some(crate::response_model::next_source_moment(
                        &candidate_params,
                        current_opening_layout,
                        &source_groups,
                        policy.decomposition.field_bits(),
                        policy.claim_ext_degree,
                    )?)
                } else {
                    None
                };
                candidates_with_source.push((
                    candidate_params,
                    next_witness_len,
                    opening_reduction_bytes,
                    next_source_moment,
                ));
            }
            Ok(candidates_with_source)
        };
        // Terminal projection discards B, D, and the unused successor witness.
        // Do not run fold-layout Pareto pruning here: its coordinates can
        // discard the A matrix or basis that is optimal after terminal
        // conversion. The terminal objective frontier below compares the
        // actual setup and response bytes.
        let generated_candidate_count = terminal_candidates
            .len()
            .saturating_add(fold_candidates.len());
        let terminal_candidate_count = terminal_candidates.len();
        for (candidate_params, _, opening_reduction_bytes) in terminal_candidates {
            let natural_len = active_setup_field_len(&candidate_params, current_opening_layout)?;
            price_terminal_candidate(
                ctx,
                state,
                &candidate_params,
                opening_reduction_bytes,
                natural_len,
                &mut frontier,
                &mut mixed_frontier,
            )?;
        }
        let candidates = prune::level_candidates(
            current_opening_layout,
            attach_source_moments(fold_candidates)?,
        )?;
        if let Some(diagnostics) = diagnostics {
            diagnostics.record_candidates(
                generated_candidate_count,
                terminal_candidate_count.saturating_add(candidates.len()),
            );
        }
        if candidates.is_empty() {
            continue;
        }

        let guide_scope =
            GuideScope::for_state(policy, root_level_key.is_some(), incoming_setup_prefix);
        let candidates = candidate_traversal(
            policy,
            guide_scope,
            current_opening_layout,
            current_witness_len,
            candidates,
        )?;

        for (guide, (candidate_params, next_witness_len, _, next_source_moment)) in candidates {
            if let Some(natural_prefix_len) = incoming_setup_prefix {
                let padded_prefix_len = akita_types::padded_setup_prefix_len(natural_prefix_len);
                if !offloaded_witness_contracts(
                    current_witness_len,
                    current_lb,
                    padded_prefix_len,
                    policy.decomposition.field_bits(),
                    next_witness_len,
                    open_lb,
                    policy.min_offloaded_witness_contraction,
                )? {
                    continue;
                }
            }
            let natural_len = guide.and_then(|(_, natural_len)| natural_len).map_or_else(
                || active_setup_field_len(&candidate_params, current_opening_layout),
                Ok,
            )?;
            let direct_edge_is_admissible = incoming_setup_prefix.is_none_or(|incoming_len| {
                akita_types::padded_setup_prefix_len(natural_len)
                    < akita_types::padded_setup_prefix_len(incoming_len)
            });
            let prune_direct_edge = if direct_edge_is_admissible {
                guide
                    .zip(guide_scope)
                    .map(|((lower_bound, _), guide_scope)| {
                        direct_edge_bound_is_strictly_worse(
                            policy,
                            guide_scope,
                            &candidate_params,
                            natural_len,
                            lower_bound,
                            &frontier,
                            &mixed_frontier,
                        )
                    })
                    .transpose()?
                    .unwrap_or(false)
            } else {
                false
            };
            if prune_direct_edge {
                if let Some(diagnostics) = diagnostics {
                    diagnostics.record_guided_direct_edge_prune();
                }
                if !policy.recursive_setup_planning {
                    continue;
                }
            }
            let direct_child = if !direct_edge_is_admissible || prune_direct_edge {
                None
            } else if depth == MAX_RECURSION_DEPTH {
                Some(empty_suffix_result())
            } else {
                Some(derive_selected_suffix_schedule(
                    ctx,
                    memo,
                    SuffixState {
                        level: level + 1,
                        current_witness_len: next_witness_len,
                        current_lb: open_lb,
                        source_moment: next_source_moment,
                        incoming_setup_prefix: None,
                        dimension_ceiling: candidate_params.role_dims(),
                        payload_phase: payload_phase.after(candidate_params.payload_mode),
                    },
                    depth + 1,
                )?)
            };
            let offloaded_child = if policy.recursive_setup_planning
                && candidate_params.payload_mode.is_compressed()
                // An offloaded edge accepts only a child suffix with at
                // least two folds. At the last two admissible depths that
                // topology cannot fit, so planning the child can only
                // produce results that `child_choice` rejects.
                && depth + 2 < MAX_RECURSION_DEPTH
            {
                Some(derive_selected_suffix_schedule(
                    ctx,
                    memo,
                    SuffixState {
                        level: level + 1,
                        current_witness_len: next_witness_len,
                        current_lb: open_lb,
                        source_moment: next_source_moment,
                        incoming_setup_prefix: Some(natural_len),
                        dimension_ceiling: candidate_params.role_dims(),
                        payload_phase,
                    },
                    depth + 1,
                )?)
            } else {
                None
            };
            price_level_candidate_with_children(
                ctx,
                state,
                &candidate_params,
                next_witness_len,
                natural_len,
                direct_child.as_deref(),
                offloaded_child.as_deref(),
                require_child_fold,
                &mut frontier,
                &mut mixed_frontier,
            )?;
        }
    }
    if let Some(diagnostics) = diagnostics {
        diagnostics.record_completed_state(
            frontier
                .candidate_count()
                .saturating_add(mixed_frontier.candidate_count()),
        );
    }
    for (key, choices) in frontier.by_parent_cost {
        if retains_setup_projection {
            setup_and_payload.insert(key, choices);
        } else {
            let candidates = choices.into_payload_candidates();
            if !candidates.is_empty() {
                payload_only.insert(key, candidates);
            }
        }
    }

    let result = Arc::new(SuffixResult {
        payload_only,
        setup_and_payload,
        mixed_frontier,
    });
    memo.insert(memo_key, Arc::clone(&result));
    Ok(result)
}

#[cfg(test)]
#[path = "../test/suffix_dp.rs"]
mod tests;
