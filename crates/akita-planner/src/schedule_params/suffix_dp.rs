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
    terminal_setup_field_elements, CandidateFoldStep, CandidateInnerRoute, CandidateLayoutGuide,
    CandidateTerminalResponse, CompleteObjectiveBound, FoldCandidatePolicy, PackedProofCost,
    RecursiveCandidateRequest, RecursiveFoldWork, RelationCandidateTopology, RelationModeFilter,
    RelationSearchDomain, RelationTraversalOrder, RingRelationPhase, ScheduleCandidate,
    SetupPrefixCapacity, SetupPrefixLayoutGuide, SetupPrefixSearchCache, SplitBoundPolicy,
};
use akita_schedules::planner_support::MAX_RECURSION_DEPTH;

mod candidates;
mod frontier;
mod prune;
mod search;
mod source;
mod state;
mod terminal;

#[cfg(all(test, feature = "catalog-gen"))]
pub(super) use candidates::packing_precommit_opening_products;
#[cfg(test)]
pub(super) use candidates::state_allows_terminal_seed;
use frontier::{consider_child_suffixes, price_child_edge, ProjectedFrontier, Projection};
pub(crate) use search::derive_selected_suffix_schedule;
use source::attach_source_moments;
use state::*;
pub(crate) use state::{ScheduleMemo, SuffixCtx, SuffixState, SuffixTopology};
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

impl ChildEdge<'_> {
    fn grinding_nonce_bits(
        &self,
        suffix: &ScheduleCandidate,
        relation_geometry: akita_types::RelationAddressGeometry,
    ) -> Result<usize, AkitaError> {
        let successor = suffix.folds.first().map_or_else(
            || akita_types::FoldSuccessor::Terminal(&suffix.terminal.params),
            |fold| akita_types::FoldSuccessor::Recursive(fold.params.as_ref()),
        );
        akita_types::transcript_grinding_nonce_bits_for_planner_edge(
            self.candidate_params.as_ref(),
            relation_geometry,
            self.opening_layout,
            successor,
            self.policy.decomposition.field_bits(),
            self.policy.claim_ext_degree,
            self.level,
        )
    }
}

#[derive(Clone, Copy)]
struct ChildEdgePrice {
    direct_payload_bytes: usize,
    stage3_payload_bytes: usize,
    relation_geometry: akita_types::RelationAddressGeometry,
}

struct PendingScheduleCandidate {
    first_direct_setup_field_len: Option<NonZeroUsize>,
    first_direct_output_witness_len: usize,
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

struct PlannedFoldCandidate {
    params: CommittedGroupParams,
    next_witness_len: usize,
    opening_reduction_bytes: usize,
    next_source_moment: Option<crate::response_model::SourceMomentEstimate>,
}

struct GuidedLevelCandidate {
    lower_bound: CompleteObjectiveBound,
    natural_len: Option<usize>,
    candidate: PlannedFoldCandidate,
}

enum CandidateTraversal {
    Plain(std::vec::IntoIter<PlannedFoldCandidate>),
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
            && matches!(
                policy.selection_policy,
                crate::SelectionPolicyId::MinFirstDirectSetupThenPayloadV2
                    | crate::SelectionPolicyId::MinPaddedSetupEnvelopeThenFirstDirectThenPayloadV3
            )
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
        PlannedFoldCandidate,
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
            first_direct_output_witness_len: self.first_direct_output_witness_len,
            cost: self.cost,
            setup_field_elements: self.setup_field_elements,
        }
    }

    fn into_candidate(self) -> ScheduleCandidate {
        ScheduleCandidate {
            first_direct_setup_field_len: self.first_direct_setup_field_len,
            first_direct_output_witness_len: self.first_direct_output_witness_len,
            cost: self.cost,
            setup_field_elements: self.setup_field_elements,
            folds: self.suffix_folds.prepend(self.first_fold),
            terminal: self.terminal,
        }
    }

    fn descriptor_bytes(
        &self,
        diagnostics: Option<&crate::diagnostics::PlannerDiagnostics>,
    ) -> Result<Vec<u8>, AkitaError> {
        super::candidate_schedule_descriptor_bytes(
            Some(&self.first_fold),
            &self.suffix_folds,
            &self.terminal.params,
            diagnostics,
        )
    }
}

fn child_edge_price(
    edge: &ChildEdge<'_>,
    suffix: &ScheduleCandidate,
) -> Result<ChildEdgePrice, AkitaError> {
    let successor = suffix.folds.first().map_or_else(
        || akita_types::FoldSuccessor::Terminal(&suffix.terminal.params),
        |fold| akita_types::FoldSuccessor::Recursive(fold.params.as_ref()),
    );
    let payload = akita_schedules::planner_support::nonterminal_level_payload_bytes(
        edge.policy,
        &edge.candidate_params,
        edge.opening_layout,
        successor,
        edge.next_witness_len,
    )?;
    let direct_payload_bytes = payload.direct;
    let stage3_payload_bytes = payload.stage3;
    if edge.offloaded != (stage3_payload_bytes != 0) {
        return Err(AkitaError::InvalidSetup(
            "setup edge topology disagrees with Stage-3 accounting".to_string(),
        ));
    }
    Ok(ChildEdgePrice {
        direct_payload_bytes,
        stage3_payload_bytes,
        relation_geometry: payload.relation_geometry,
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
    let first_direct_output_witness_len = if edge.offloaded {
        suffix.first_direct_output_witness_len
    } else {
        edge.next_witness_len
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
        first_direct_output_witness_len,
        cost,
        setup_field_elements,
        first_fold,
        suffix_folds: suffix.folds.clone(),
        terminal: suffix.terminal.clone(),
    }))
}

fn direct_edge_lower_bound(
    policy: &PlannerPolicy,
    params: &CommittedGroupParams,
    opening_layout: &OpeningClaimsLayout,
    output_witness_len: usize,
    natural_setup_field_len: usize,
) -> Result<CompleteObjectiveBound, AkitaError> {
    // Before a child is selected, price the smallest valid power-of-two
    // successor relation and omit its next payload. Every concrete successor
    // is at least this expensive, so the guide remains a conservative bound.
    let relation_geometry = params.relation_address_geometry(
        opening_layout,
        policy.claim_ext_degree,
        1,
        output_witness_len,
    )?;
    let proof_bytes = akita_types::level_proof_bytes(
        policy.decomposition.field_bits(),
        policy.challenge_field_bits()?,
        params,
        relation_geometry,
        None,
    )?;
    Ok(CompleteObjectiveBound::for_direct_edge(
        policy,
        SetupPrefixCapacity::for_natural_len(natural_setup_field_len).field_elements(),
        output_witness_len,
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
        crate::SelectionPolicyId::MinEstimatedProofPayloadV2 => frontier
            .by_parent_cost
            .values()
            .flat_map(frontier::ProjectedObjectiveChoices::payload_candidates)
            .any(|candidate| lower_bound.is_strictly_worse_than(candidate.metrics())),
        crate::SelectionPolicyId::MinFirstDirectSetupThenPayloadV2 => frontier
            .by_parent_cost
            .values()
            .flat_map(frontier::ProjectedObjectiveChoices::setup_candidates)
            .any(|candidate| lower_bound.is_strictly_worse_than(candidate.metrics())),
        crate::SelectionPolicyId::MinPaddedSetupEnvelopeThenFirstDirectThenPayloadV3 => frontier
            .by_parent_cost
            .values()
            .flat_map(frontier::ProjectedObjectiveChoices::setup_candidates)
            .any(|candidate| lower_bound.is_strictly_worse_than(candidate.metrics())),
    }
}

fn complete_root_setup_bound_is_strictly_worse(
    lower_bound: CompleteObjectiveBound,
    frontier: &ProjectedFrontier,
) -> bool {
    frontier
        .by_parent_cost
        .values()
        .flat_map(frontier::ProjectedObjectiveChoices::setup_candidates)
        .any(|candidate| lower_bound.setup_envelope_is_strictly_worse_than(candidate.metrics()))
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
    candidates: Vec<PlannedFoldCandidate>,
) -> Result<CandidateTraversal, AkitaError> {
    if guide_scope.is_none() {
        return Ok(CandidateTraversal::Plain(candidates.into_iter()));
    }
    let mut guided = candidates
        .into_iter()
        .map(|candidate| {
            let natural_len = (matches!(
                policy.selection_policy,
                crate::SelectionPolicyId::MinFirstDirectSetupThenPayloadV2
                    | crate::SelectionPolicyId::MinPaddedSetupEnvelopeThenFirstDirectThenPayloadV3
            ))
            .then(|| active_setup_field_len(&candidate.params, opening_layout))
            .transpose()?;
            let lower_bound = direct_edge_lower_bound(
                policy,
                &candidate.params,
                opening_layout,
                candidate.next_witness_len,
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
             candidate,
             ..
         }| {
            (
                *lower_bound,
                candidate.next_witness_len,
                candidate.params.canonical_descriptor_bytes(),
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
    let direct_projections = if state.topology.incoming_setup_prefix().is_some()
        || (ctx.level_zero_is_root && state.level == 0)
    {
        SETUP_AND_PAYLOAD_PROJECTIONS
    } else {
        PAYLOAD_PROJECTION
    };
    if (ctx.level_zero_is_root && state.level == 0)
        || state.topology.incoming_setup_prefix().is_some()
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
        ctx.adaptation_guide.map(|schedule| {
            CandidateInnerRoute::of(schedule.terminal.inner.matrix.security_route())
        }),
    )?
    else {
        return Ok(());
    };
    if let Some(guide) = ctx.adaptation_guide.map(|schedule| &schedule.terminal) {
        let candidate_route = direct_step.params.inner.matrix.security_route();
        let guide_route = guide.inner.matrix.security_route();
        let route_kind_matches = matches!(
            (candidate_route, guide_route),
            (
                akita_types::InnerCommitSecurityRoute::Linf(_),
                akita_types::InnerCommitSecurityRoute::Linf(_)
            ) | (
                akita_types::InnerCommitSecurityRoute::L2 { .. },
                akita_types::InnerCommitSecurityRoute::L2 { .. }
            )
        );
        if direct_step.params.d_a() != guide.d_a()
            || direct_step.params.blocks.positions_per_block != guide.blocks.positions_per_block
            || direct_step.params.inner.digits.log_basis != guide.inner.digits.log_basis
            || direct_step.params.fold.log_basis != guide.fold.log_basis
            || direct_step.sparse_challenge_config != guide.fold_challenge_config
            || !route_kind_matches
        {
            return Ok(());
        }
    }
    let level_proof_size = opening_reduction_bytes;
    let total = level_proof_size
        .checked_add(suffix_cost)
        .ok_or_else(|| AkitaError::InvalidSetup("terminal proof size overflow".to_string()))?;
    direct_step.estimated_direct_payload_bytes = level_proof_size;
    let candidate = ScheduleCandidate {
        first_direct_setup_field_len: Some(NonZeroUsize::new(natural_len).ok_or_else(|| {
            AkitaError::InvalidSetup("direct setup field length must be nonzero".into())
        })?),
        first_direct_output_witness_len: 0,
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
    let direct_projections = if state.topology.incoming_setup_prefix().is_some()
        || (ctx.level_zero_is_root && state.level == 0)
    {
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
    let parent_cost = ParentObservableKey::new(policy, Some(candidate_params), None)?;
    if let Some(direct_child) = children.direct {
        for (successor_class, candidates) in &direct_child.payload_only {
            let Some(representative) = candidates.first() else {
                continue;
            };
            let priced_edge = price_child_edge(&direct_edge, successor_class, representative)?;
            consider_child_suffixes(
                &direct_edge,
                candidates,
                priced_edge,
                &parent_cost,
                state.topology.incoming_setup_prefix(),
                direct_projections,
                &mut frontiers.projected,
            )?;
        }
        for (successor_class, choices) in &direct_child.setup_and_payload {
            let Some(representative) = choices
                .payload_candidates()
                .next()
                .or_else(|| choices.setup_candidates().next())
            else {
                continue;
            };
            let priced_edge = price_child_edge(&direct_edge, successor_class, representative)?;
            consider_child_suffixes(
                &direct_edge,
                choices.payload_candidates(),
                priced_edge,
                &parent_cost,
                state.topology.incoming_setup_prefix(),
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
        for (successor_class, choices) in &offloaded_child.setup_and_payload {
            let Some(representative) = choices
                .payload_candidates()
                .next()
                .or_else(|| choices.setup_candidates().next())
            else {
                continue;
            };
            let priced_edge = price_child_edge(&offloaded_edge, successor_class, representative)?;
            consider_child_suffixes(
                &offloaded_edge,
                choices.setup_candidates(),
                priced_edge,
                &parent_cost,
                state.topology.incoming_setup_prefix(),
                SETUP_PROJECTION,
                &mut frontiers.projected,
            )?;
            consider_child_suffixes(
                &offloaded_edge,
                choices.payload_candidates(),
                priced_edge,
                &parent_cost,
                state.topology.incoming_setup_prefix(),
                PAYLOAD_PROJECTION,
                &mut frontiers.projected,
            )?;
        }
    }

    Ok(())
}

#[cfg(test)]
#[path = "../test/suffix_dp.rs"]
mod tests;
