use std::{
    collections::{hash_map::Entry, BTreeMap, HashMap, VecDeque},
    num::NonZeroUsize,
    sync::Arc,
};

use akita_error::AkitaError;
use akita_types::{
    active_setup_field_len, terminal_response_planner_bytes, AkitaScheduleLookupKey,
    CommitmentRingDims, CommittedGroupParams, OpeningClaimsLayout, PolynomialGroupLayout,
    TerminalResponseShape,
};

use crate::PlannerPolicy;

use super::{
    derive_fold_candidates, derive_recursive_candidate_views, derive_terminal_candidates,
    dimension_candidates, level_setup_field_elements, suffix_opening_layout,
    terminal_setup_field_elements, CandidateFoldStep, CandidateTerminalResponse,
    CompleteObjectiveBound, FoldCandidatePolicy, PackedProofCost, RecursiveCandidateRequest,
    RecursiveSetupPrefix, ScheduleCandidate, SetupPrefixCapacity, SetupPrefixSearchCache,
    SplitBoundPolicy,
};
use akita_schedules::planner_support::MAX_RECURSION_DEPTH;

mod candidates;
mod frontier;
mod prune;
mod state;
mod terminal;

#[cfg(test)]
pub(super) use candidates::{packing_precommit_opening_products, state_allows_terminal_seed};
use frontier::{consider_child_suffixes, ProjectedFrontier, Projection};
use state::*;
pub(crate) use state::{ScheduleMemo, SuffixCtx, SuffixState};
pub(crate) use terminal::try_terminal_direct_suffix_cost;

const SETUP_AND_PAYLOAD_PROJECTIONS: &[Projection] =
    &[Projection::FirstDirectSetup, Projection::Payload];
const PAYLOAD_PROJECTION: &[Projection] = &[Projection::Payload];
const SETUP_PROJECTION: &[Projection] = &[Projection::FirstDirectSetup];

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
    opening_layout: &'a OpeningClaimsLayout,
    level: u32,
    candidate_params: Arc<CommittedGroupParams>,
    current_witness_len: usize,
    next_witness_len: usize,
    natural_setup_field_len: usize,
    level_setup_field_elements: usize,
    offloaded: bool,
    require_child_fold: bool,
    setup_field_budget: Option<usize>,
}

#[derive(Clone, Copy)]
struct ChildEdgePrice {
    direct_payload_bytes: usize,
    stage3_payload_bytes: usize,
}

struct PendingScheduleCandidate {
    first_direct_setup_field_len: Option<NonZeroUsize>,
    cost: PackedProofCost,
    setup_field_elements: usize,
    first_fold: CandidateFoldStep,
    suffix_folds: super::CandidateFoldChain,
    terminal: Arc<CandidateTerminalResponse>,
}

struct StateFrontiers {
    projected: ProjectedFrontier,
}

impl StateFrontiers {
    fn new() -> Self {
        Self {
            projected: ProjectedFrontier::default(),
        }
    }

    fn candidate_count(&self) -> usize {
        self.projected.candidate_count()
    }
}

struct LevelCandidateEdge<'a> {
    params: &'a CommittedGroupParams,
    next_witness_len: usize,
    natural_setup_field_len: usize,
    require_child_fold: bool,
}

struct CandidateChildren<'a> {
    direct: Option<&'a SuffixResult>,
    offloaded: Option<&'a SuffixResult>,
}

type LevelCandidate = (
    CommittedGroupParams,
    usize,
    usize,
    Option<crate::response_model::SourceMomentEstimate>,
);

struct GuidedLevelCandidate {
    lower_bound: CompleteObjectiveBound,
    natural_len: Option<usize>,
    candidate: LevelCandidate,
}

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
            Self::Guided(candidates) => candidates.next().map(
                |GuidedLevelCandidate {
                     lower_bound,
                     natural_len,
                     candidate,
                     ..
                 }| { (Some((lower_bound, natural_len)), candidate) },
            ),
        }
    }
}

impl PendingScheduleCandidate {
    fn metrics(&self) -> super::CandidateMetrics {
        super::CandidateMetrics {
            first_direct_setup_capacity: self
                .first_direct_setup_field_len
                .map_or(super::SetupPrefixCapacity::MAX, |natural_len| {
                    super::SetupPrefixCapacity::for_natural_len(natural_len.get())
                }),
            cost: self.cost,
            setup_field_elements: self.setup_field_elements,
        }
    }

    fn into_candidate(self) -> ScheduleCandidate {
        ScheduleCandidate {
            first_direct_setup_field_len: self.first_direct_setup_field_len,
            cost: self.cost,
            setup_field_elements: self.setup_field_elements,
            folds: self.suffix_folds.prepend(self.first_fold),
            terminal: self.terminal,
        }
    }
}

fn child_edge_price(
    edge: &ChildEdge<'_>,
    successor: Option<&CommittedGroupParams>,
) -> Result<ChildEdgePrice, AkitaError> {
    let (direct_payload_bytes, stage3_payload_bytes) =
        akita_schedules::planner_support::nonterminal_level_payload_bytes(
            edge.policy,
            &edge.candidate_params,
            successor,
            edge.current_witness_len,
            edge.next_witness_len,
        )?;
    if edge.offloaded != (stage3_payload_bytes != 0) {
        return Err(AkitaError::InvalidSetup(
            "setup edge topology disagrees with Stage-3 accounting".to_string(),
        ));
    }
    Ok(ChildEdgePrice {
        direct_payload_bytes,
        stage3_payload_bytes,
    })
}

fn child_choice(
    edge: &ChildEdge<'_>,
    edge_price: ChildEdgePrice,
    edge_nonce_bits: usize,
    suffix: &ScheduleCandidate,
) -> Result<Option<PendingScheduleCandidate>, AkitaError> {
    if !frontier::ParentAdmissionClass::for_candidate(suffix).is_admitted_by(
        edge.require_child_fold,
        edge.offloaded,
        edge.natural_setup_field_len,
    ) {
        return Ok(None);
    }

    let edge_payload_bytes = edge_price
        .direct_payload_bytes
        .checked_add(edge_price.stage3_payload_bytes)
        .ok_or_else(|| AkitaError::InvalidSetup("edge proof payload overflow".to_string()))?;
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
        estimated_direct_payload_bytes: edge_price.direct_payload_bytes,
        estimated_stage3_payload_bytes: edge_price.stage3_payload_bytes,
    };
    let cost = suffix
        .cost
        .checked_prepend(edge_payload_bytes, edge_nonce_bits)?;
    Ok(Some(PendingScheduleCandidate {
        first_direct_setup_field_len,
        cost,
        setup_field_elements,
        first_fold,
        suffix_folds: suffix.folds.clone(),
        terminal: suffix.terminal.clone(),
    }))
}

fn edge_grinding_nonce_bits(
    edge: &ChildEdge<'_>,
    suffix: &ScheduleCandidate,
) -> Result<usize, AkitaError> {
    let fold = CandidateFoldStep {
        params: Arc::clone(&edge.candidate_params),
        input_witness_len: edge.current_witness_len,
        output_witness_len: edge.next_witness_len,
        estimated_direct_payload_bytes: 0,
        estimated_stage3_payload_bytes: 0,
    };
    let recursive_successor = suffix.folds.first();
    akita_schedules::planner_support::candidate_edge_grinding_nonce_bits(
        edge.policy,
        edge.opening_layout,
        &fold,
        recursive_successor,
        recursive_successor
            .is_none()
            .then_some(suffix.terminal.as_ref()),
        edge.level,
    )
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
) -> bool {
    match policy.selection_policy {
        crate::SelectionPolicyId::MinEstimatedProofPayload => frontier
            .by_parent_cost
            .values()
            .flat_map(frontier::ProjectedObjectiveChoices::payload_candidates)
            .any(|candidate| lower_bound.is_strictly_worse_than(candidate.metrics())),
        crate::SelectionPolicyId::MinFirstDirectSetupThenPayload => frontier
            .by_parent_cost
            .values()
            .flat_map(frontier::ProjectedObjectiveChoices::setup_candidates)
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
) -> Result<bool, AkitaError> {
    match guide_scope {
        GuideScope::CompleteRoot => Ok(complete_root_bound_is_strictly_worse(
            policy,
            lower_bound,
            frontier,
        )),
        GuideScope::RecursivePrefix => {
            let parent_cost = ParentObservableKey::new(policy, Some(params), None)?;
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
            Ok(GuidedLevelCandidate {
                lower_bound,
                natural_len,
                candidate,
            })
        })
        .collect::<Result<Vec<_>, AkitaError>>()?;
    guided.sort_by_cached_key(
        |GuidedLevelCandidate {
             lower_bound,
             candidate: (params, next_witness_len, _, _),
             ..
         }| {
            (
                *lower_bound,
                *next_witness_len,
                params.canonical_descriptor_bytes(),
            )
        },
    );
    Ok(CandidateTraversal::Guided(guided.into_iter()))
}

fn price_terminal_candidate(
    ctx: &SuffixCtx<'_>,
    state: SuffixState,
    candidate_params: &CommittedGroupParams,
    opening_reduction_bytes: usize,
    natural_len: usize,
    frontiers: &mut StateFrontiers,
) -> Result<(), AkitaError> {
    let policy = ctx.policy;
    let direct_projections =
        if state.incoming_setup_prefix.is_some() || (ctx.level_zero_is_root && state.level == 0) {
            SETUP_AND_PAYLOAD_PROJECTIONS
        } else {
            PAYLOAD_PROJECTION
        };
    if (ctx.level_zero_is_root && state.level == 0)
        || state.incoming_setup_prefix.is_some()
        || candidate_params.has_preceding_groups()
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
    let level_proof_size = opening_reduction_bytes;
    let total = level_proof_size
        .checked_add(suffix_cost)
        .ok_or_else(|| AkitaError::InvalidSetup("terminal proof size overflow".to_string()))?;
    direct_step.estimated_direct_payload_bytes = level_proof_size;
    let candidate = ScheduleCandidate {
        first_direct_setup_field_len: Some(NonZeroUsize::new(natural_len).ok_or_else(|| {
            AkitaError::InvalidSetup("direct setup field length must be nonzero".into())
        })?),
        cost: PackedProofCost::new(total, 0)?,
        setup_field_elements: terminal_setup_field_elements(&direct_step.params)?,
        folds: super::CandidateFoldChain::default(),
        terminal: Arc::new(direct_step),
    };
    frontiers.projected.consider_candidate(
        policy,
        ctx.diagnostics,
        candidate.clone(),
        direct_projections,
    )?;
    Ok(())
}

fn price_level_candidate_with_children(
    ctx: &SuffixCtx<'_>,
    state: SuffixState,
    opening_layout: &OpeningClaimsLayout,
    candidate: LevelCandidateEdge<'_>,
    children: CandidateChildren<'_>,
    frontiers: &mut StateFrontiers,
) -> Result<(), AkitaError> {
    let policy = ctx.policy;
    let LevelCandidateEdge {
        params: candidate_params,
        next_witness_len,
        natural_setup_field_len: natural_len,
        require_child_fold,
    } = candidate;
    // Only a prefix-consuming state is read through the setup projection by
    // an offloaded parent. The top-level setup-first objective also reads the
    // root setup projection. Ordinary direct suffixes are consumed solely
    // through the payload projection, so retaining a parallel setup winner
    // there duplicates frontier work and memo ownership with no observer.
    let direct_projections =
        if state.incoming_setup_prefix.is_some() || (ctx.level_zero_is_root && state.level == 0) {
            SETUP_AND_PAYLOAD_PROJECTIONS
        } else {
            PAYLOAD_PROJECTION
        };
    let level_setup_field_elements = level_setup_field_elements(candidate_params)?;
    let direct_edge = ChildEdge {
        policy,
        diagnostics: ctx.diagnostics,
        opening_layout,
        level: u32::try_from(state.level)
            .map_err(|_| AkitaError::InvalidSetup("grinding level exceeds u32".into()))?,
        candidate_params: Arc::new(candidate_params.clone()),
        current_witness_len: state.current_witness_len,
        next_witness_len,
        natural_setup_field_len: natural_len,
        level_setup_field_elements,
        offloaded: false,
        require_child_fold,
        setup_field_budget: ctx.setup_field_budget,
    };
    if let Some(direct_child) = children.direct {
        for candidates in direct_child.payload_only.values() {
            consider_child_suffixes(
                &direct_edge,
                candidates,
                state.incoming_setup_prefix,
                direct_projections,
                &mut frontiers.projected,
            )?;
        }
        for choices in direct_child.setup_and_payload.values() {
            consider_child_suffixes(
                &direct_edge,
                choices.payload_candidates(),
                state.incoming_setup_prefix,
                direct_projections,
                &mut frontiers.projected,
            )?;
        }
    }
    if let Some(offloaded_child) = children.offloaded {
        let offloaded_edge = ChildEdge {
            offloaded: true,
            ..direct_edge
        };
        for choices in offloaded_child.setup_and_payload.values() {
            consider_child_suffixes(
                &offloaded_edge,
                choices.setup_candidates(),
                state.incoming_setup_prefix,
                SETUP_PROJECTION,
                &mut frontiers.projected,
            )?;
            consider_child_suffixes(
                &offloaded_edge,
                choices.payload_candidates(),
                state.incoming_setup_prefix,
                PAYLOAD_PROJECTION,
                &mut frontiers.projected,
            )?;
        }
    }

    Ok(())
}

/// Derive the suffix frontier for the selected recursive schedule at
/// `(level, current_witness_len, current_lb)`.
///
/// At each state, the projected maps keep the setup and payload winners for
/// each parent-visible first-fold key (from
/// [`derive_fold_candidates`]). A candidate may terminate on the current
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
    let policy = ctx.policy;
    let diagnostics = ctx.diagnostics;
    let root_honest_fold_policy = ctx.root_honest_fold_policy;
    let precommitted_honest_fold_policies = ctx.precommitted_honest_fold_policies;
    let level_zero_is_root = ctx.level_zero_is_root;
    if let Some(diagnostics) = diagnostics {
        diagnostics.record_suffix_call();
    }
    let SuffixState {
        level,
        current_witness_len,
        current_lb,
        source_moment,
        incoming_setup_prefix,
        dimension_ceiling: _,
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
    let candidate_domain = candidates::CandidateDomain::prepare(ctx, state)?;
    let root_level_key = candidate_domain.root_level_key;
    let current_opening_layout = &candidate_domain.opening_layout;
    // Every opening basis contributes to one state frontier. In particular,
    // terminal-direct candidates have no first fold and therefore share the
    // `None` key; they must be compared by the canonical objective instead of
    // being overwritten by the last basis visited.
    let mut frontiers = StateFrontiers::new();
    for open_lb in candidate_domain.opening_basis_range.clone() {
        let candidates = candidate_domain.generate_for_opening_basis(
            ctx,
            state,
            open_lb,
            &mut memo.setup_prefixes,
        )?;
        let fold_candidates = candidates.folds;
        let terminal_candidates = candidates.terminal;
        let attach_source_moments =
            |candidates: Vec<candidates::RawLevelCandidate>| -> Result<Vec<_>, AkitaError> {
                let mut candidates_with_source = Vec::with_capacity(candidates.len());
                for candidate in candidates {
                    let candidates::RawLevelCandidate {
                        params: candidate_params,
                        next_witness_len,
                        opening_reduction_bytes,
                    } = candidate;
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
                                policy.decomposition,
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
                                AkitaError::InvalidSetup(
                                    "recursive response source is missing".into(),
                                )
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
        for candidate in terminal_candidates {
            let natural_len = active_setup_field_len(&candidate.params, current_opening_layout)?;
            price_terminal_candidate(
                ctx,
                state,
                &candidate.params,
                candidate.opening_reduction_bytes,
                natural_len,
                &mut frontiers,
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
                            &frontiers.projected,
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
                current_opening_layout,
                LevelCandidateEdge {
                    params: &candidate_params,
                    next_witness_len,
                    natural_setup_field_len: natural_len,
                    require_child_fold: candidate_domain.require_child_fold,
                },
                CandidateChildren {
                    direct: direct_child.as_deref(),
                    offloaded: offloaded_child.as_deref(),
                },
                &mut frontiers,
            )?;
        }
    }
    if let Some(diagnostics) = diagnostics {
        diagnostics.record_completed_state(frontiers.candidate_count());
    }
    for (key, choices) in frontiers.projected.by_parent_cost {
        if retains_setup_projection {
            setup_and_payload.insert(key, choices.into_objective_choices());
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
    });
    memo.insert(memo_key, Arc::clone(&result));
    Ok(result)
}

#[cfg(test)]
#[path = "../test/suffix_dp.rs"]
mod tests;
