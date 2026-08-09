use std::{
    collections::{hash_map::Entry, BTreeMap, HashMap, VecDeque},
    sync::Arc,
};

use akita_field::AkitaError;
use akita_types::{
    active_setup_field_len, level_proof_bytes, terminal_response_bytes,
    try_extension_opening_reduction_level_bytes, AkitaScheduleLookupKey, CommitmentRingDims,
    CommittedGroupParams, OpeningClaimsLayout, PolynomialGroupLayout, TerminalResponseShape,
};

use crate::{planner::root_level_candidates_for_basis, PlannerPolicy};

use super::{
    derive_candidate_level_params, derive_candidate_level_params_all_splits, dimension_candidates,
    level_setup_field_elements, stage3_payload_bytes_for_successor, suffix_opening_layout,
    terminal_setup_field_elements, CandidateFoldStep, CandidateTerminalResponse, MixedScore,
    ScheduleCandidate, SetupPrefixSearchCache,
};
use akita_schedules::planner_support::MAX_RECURSION_DEPTH;

mod frontier;
mod prune;

use frontier::{consider_child_suffixes, FrontierProjection, ProjectedFrontier};

/// Result of the suffix DP at one state. Each selection objective retains the
/// candidates its parent needs because proof-size and setup-envelope pricing
/// depend on the child's first step:
///
/// - `best_by_first_direct_setup_per_lb` — lexicographically best fold-first
///   schedule by first direct setup scan and then proof payload, per first-fold
///   `log_basis`. An entry with no ordinary folds terminates directly on the
///   current witness; otherwise it consumes `incoming_setup_prefix` when one
///   is present.
/// - `best_by_payload_per_lb` — smallest-payload fold-first schedule per
///   first-fold `log_basis`, used after an earlier direct edge has fixed the
///   setup-size objective.
/// - `mixed_frontier` — nondominated setup-envelope/proof candidates for the
///   direct adaptive-dimension objective.
#[derive(Clone)]
pub(crate) struct SuffixResult {
    pub(crate) best_by_first_direct_setup_per_lb: BTreeMap<FirstFoldKey, ScheduleCandidate>,
    pub(crate) best_by_payload_per_lb: BTreeMap<FirstFoldKey, ScheduleCandidate>,
    /// Nondominated setup-envelope/proof candidates used by adaptive scalar
    /// planning. Candidates with different first folds remain distinct because
    /// the parent proof price and canonical descriptor can distinguish them.
    pub(crate) mixed_frontier: Vec<ScheduleCandidate>,
}

fn mixed_score(candidate: &ScheduleCandidate) -> MixedScore {
    MixedScore {
        setup_field_elements: candidate.setup_field_elements,
        proof_bytes: candidate.total_bytes,
    }
}

fn dominates_mixed_score(left: MixedScore, right: MixedScore) -> bool {
    left.setup_field_elements <= right.setup_field_elements
        // A setup-only improvement cannot prune `right`: a parent can mask
        // both setup footprints, leaving proof bytes and the descriptor to
        // decide the complete schedule.
        && left.proof_bytes < right.proof_bytes
}

fn insert_mixed_frontier(
    policy: &PlannerPolicy,
    frontier: &mut Vec<ScheduleCandidate>,
    candidate: ScheduleCandidate,
) {
    if policy.selection_policy
        != crate::SelectionPolicyId::MinSetupMatrixFieldElementsThenProofPayload
        || !policy.admits_setup_field_elements(candidate.setup_field_elements)
    {
        return;
    }
    let same_first_fold =
        |other: &ScheduleCandidate| other.first_fold_params() == candidate.first_fold_params();
    if frontier.iter().any(|other| {
        same_first_fold(other) && dominates_mixed_score(mixed_score(other), mixed_score(&candidate))
    }) {
        return;
    }
    frontier.retain(|other| {
        !same_first_fold(other)
            || !dominates_mixed_score(mixed_score(&candidate), mixed_score(other))
    });
    frontier.push(candidate);
}

/// Parent-visible first-fold class. A parent edge prices the child's outgoing
/// commitment payload, so suffixes with different first payload sizes are not
/// interchangeable even when they use the same digit basis.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct FirstFoldKey {
    log_basis: u32,
    outer_payload_bytes: usize,
    stage3_payload_bytes: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct ParentPayloadKey {
    outer_payload_bytes: usize,
    stage3_payload_bytes: usize,
}

/// Like [`terminal_direct_suffix_cost`], but returns `None` when the fold at
/// `terminal_fold_level` is multi-chunk. The suffix DP uses this to skip the
/// fold-then-direct branch without aborting fold-then-fold exploration.
pub(super) fn try_terminal_direct_suffix_cost(
    input_witness_len: usize,
    terminal_lp: &CommittedGroupParams,
    field_bits: u32,
    key: PolynomialGroupLayout,
    terminal_fold_level: usize,
    opening_layout: Option<&OpeningClaimsLayout>,
) -> Result<Option<(CandidateTerminalResponse, usize)>, AkitaError> {
    if terminal_lp.witness_chunk.num_chunks > 1 {
        return Ok(None);
    }
    let result = terminal_direct_suffix_cost(
        input_witness_len,
        terminal_lp,
        field_bits,
        key,
        terminal_fold_level,
        opening_layout,
    );
    match result {
        Ok(candidate) => Ok(Some(candidate)),
        // Candidate construction is an optimization search. A geometry whose
        // fixed inner matrix cannot admit the directly checked terminal response is
        // infeasible, not a fatal planner error.
        Err(AkitaError::InvalidSetup(_)) => Ok(None),
        Err(error) => Err(error),
    }
}

pub(crate) fn terminal_direct_suffix_cost(
    input_witness_len: usize,
    terminal_lp: &CommittedGroupParams,
    field_bits: u32,
    key: PolynomialGroupLayout,
    terminal_fold_level: usize,
    opening_layout: Option<&OpeningClaimsLayout>,
) -> Result<(CandidateTerminalResponse, usize), AkitaError> {
    // Scalar same-point root fold: polynomial count at the root, 1 recursively.
    let num_polynomials = if terminal_fold_level == 0 {
        key.num_polynomials()
    } else {
        1
    };
    // The terminal-direct (cleartext) witness is single-chunk by construction:
    // the prover emits the global folded response and one shared `r̂` tail, so
    // chunking the cleartext tail is unsupported. The last fold level must be
    // single-chunk (only the leading activated levels are chunked). Reject here
    // to match `resolve.rs` and avoid a cryptic prover-side layout mismatch.
    if terminal_lp.witness_chunk.num_chunks > 1 {
        return Err(AkitaError::InvalidSetup(
            "terminal-direct witness does not support a multi-chunk last fold level".to_string(),
        ));
    }
    if opening_layout.is_some() || num_polynomials != 1 || terminal_lp.has_precommitted_groups() {
        return Err(AkitaError::InvalidSetup(
            "terminal direct response must be a scalar flat fold".to_string(),
        ));
    }
    let (terminal_params, admission_cap) =
        akita_types::TerminalCommittedGroupParams::try_from_expanded_group(terminal_lp.clone())?;
    let witness_shape = TerminalResponseShape::derive(&terminal_params, admission_cap)?;
    let terminal_bytes = terminal_response_bytes(field_bits, &witness_shape);
    let direct = CandidateTerminalResponse {
        params: terminal_params,
        sparse_challenge_config: terminal_lp.fold_challenge_config,
        input_witness_len,
        estimated_direct_payload_bytes: 0,
        response_shape: witness_shape,
        estimated_payload_bytes: terminal_bytes,
    };
    Ok((direct, terminal_bytes))
}

type ScheduleMemoKey = (
    usize,
    usize,
    u32,
    usize,
    usize,
    usize,
    usize,
    akita_types::CommitmentPayloadPhase,
);

pub(crate) struct ScheduleMemo {
    entries: HashMap<ScheduleMemoKey, Arc<SuffixResult>>,
    direct_insertion_order: VecDeque<ScheduleMemoKey>,
    prefixed_insertion_order: VecDeque<ScheduleMemoKey>,
    setup_prefixes: SetupPrefixSearchCache,
}

const MAX_SUFFIX_SEARCH_CACHE_ENTRIES: usize = 262_144;
// Prefix layouts create a much wider stream of one-off states than ordinary
// suffixes. Separate FIFO quotas prevent that stream from evicting the direct
// states reused across basis, dimension, and split candidates while preserving
// the original hard bound on total cached results.
const MAX_DIRECT_SUFFIX_CACHE_ENTRIES: usize = 196_608;
const MAX_PREFIXED_SUFFIX_CACHE_ENTRIES: usize =
    MAX_SUFFIX_SEARCH_CACHE_ENTRIES - MAX_DIRECT_SUFFIX_CACHE_ENTRIES;

impl ScheduleMemo {
    pub(crate) fn new() -> Self {
        Self {
            entries: HashMap::new(),
            direct_insertion_order: VecDeque::new(),
            prefixed_insertion_order: VecDeque::new(),
            setup_prefixes: SetupPrefixSearchCache::default(),
        }
    }

    fn get(&mut self, key: &ScheduleMemoKey) -> Option<&Arc<SuffixResult>> {
        self.entries.get(key)
    }

    fn insert(&mut self, key: ScheduleMemoKey, result: Arc<SuffixResult>) {
        if let Entry::Occupied(mut existing) = self.entries.entry(key) {
            existing.insert(result);
            return;
        }
        let (insertion_order, capacity) = if key.3 == 0 {
            (
                &mut self.direct_insertion_order,
                MAX_DIRECT_SUFFIX_CACHE_ENTRIES,
            )
        } else {
            (
                &mut self.prefixed_insertion_order,
                MAX_PREFIXED_SUFFIX_CACHE_ENTRIES,
            )
        };
        if insertion_order.len() >= capacity {
            if let Some(evicted) = insertion_order.pop_front() {
                self.entries.remove(&evicted);
            }
        }
        insertion_order.push_back(key);
        self.entries.insert(key, result);
    }
}

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
    candidate_params: Arc<CommittedGroupParams>,
    current_witness_len: usize,
    next_witness_len: usize,
    natural_setup_field_len: usize,
    level_setup_field_elements: usize,
    eor_bytes: usize,
    offloaded: bool,
    require_child_fold: bool,
    setup_field_budget: Option<usize>,
}

struct PendingScheduleCandidate {
    first_direct_setup_field_len: Option<usize>,
    total_bytes: usize,
    setup_field_elements: usize,
    first_fold: CandidateFoldStep,
    suffix_folds: super::CandidateFoldChain,
    terminal: Arc<CandidateTerminalResponse>,
}

impl PendingScheduleCandidate {
    fn metrics(&self) -> super::CandidateMetrics {
        super::CandidateMetrics {
            first_direct_setup_capacity: self.first_direct_setup_field_len.map_or(
                super::SetupPrefixCapacity::MAX,
                super::SetupPrefixCapacity::for_natural_len,
            ),
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
    let child_is_terminal = suffix.folds.is_empty();
    if edge.require_child_fold && child_is_terminal {
        return Ok(None);
    }
    if edge.offloaded {
        if child_is_terminal || suffix.folds.len() == 1 {
            return Ok(None);
        }
        if suffix.metrics().first_direct_setup_capacity
            >= super::SetupPrefixCapacity::for_natural_len(edge.natural_setup_field_len)
        {
            return Ok(None);
        }
    }

    let direct_payload_bytes = level_proof_bytes(
        edge.policy.decomposition.field_bits(),
        edge.policy.challenge_field_bits()?,
        &edge.candidate_params,
        suffix.first_fold_params(),
        edge.next_witness_len,
        Some(if child_is_terminal {
            akita_types::NextWitnessBindingPolicy::TerminalInnerState
        } else {
            akita_types::NextWitnessBindingPolicy::OuterPayload
        }),
    )?
    .checked_add(edge.eor_bytes)
    .ok_or_else(|| AkitaError::InvalidSetup("level proof size overflow".to_string()))?;
    let stage3_payload_bytes =
        stage3_payload_bytes_for_successor(edge.policy, suffix.first_fold_params())?;
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
        Some(edge.natural_setup_field_len)
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

fn consider_mixed_child_suffixes(
    edge: &ChildEdge<'_>,
    child_candidates: &[ScheduleCandidate],
    frontier: &mut Vec<ScheduleCandidate>,
) -> Result<(), AkitaError> {
    for suffix in child_candidates {
        let Some(candidate) = child_choice(edge, suffix)? else {
            continue;
        };
        insert_mixed_frontier(edge.policy, frontier, candidate.into_candidate());
    }
    Ok(())
}

fn empty_suffix_result() -> Arc<SuffixResult> {
    Arc::new(SuffixResult {
        best_by_first_direct_setup_per_lb: BTreeMap::new(),
        best_by_payload_per_lb: BTreeMap::new(),
        mixed_frontier: Vec::new(),
    })
}

/// DP-invariant inputs for the suffix search.
///
/// `policy`, `ring_challenge_cfg`, and `num_vars` are constant across the whole
/// recursion, so they are carried in one context value rather than as
/// per-call arguments (keeps the recursive signature small).
#[derive(Clone, Copy)]
pub(crate) struct SuffixCtx<'a> {
    pub(crate) policy: &'a PlannerPolicy,
    pub(crate) default_ring_challenge_cfg: &'a akita_challenges::SparseChallengeConfig,
    pub(crate) ring_challenge_config:
        &'a dyn Fn(usize) -> Result<akita_challenges::SparseChallengeConfig, AkitaError>,
    pub(crate) num_vars: usize,
    pub(crate) key: PolynomialGroupLayout,
    pub(crate) setup_field_budget: Option<usize>,
    pub(crate) root_lookup_key: Option<&'a AkitaScheduleLookupKey>,
    pub(crate) root_honest_fold_policy: Option<akita_types::sis::HonestFoldPolicySpec>,
    pub(crate) precommitted_honest_fold_policies: &'a [akita_types::sis::HonestFoldPolicySpec],
    pub(crate) level_zero_is_root: bool,
}

#[derive(Clone, Copy)]
pub(crate) struct SuffixState {
    pub(crate) level: usize,
    pub(crate) current_witness_len: usize,
    pub(crate) current_lb: u32,
    pub(crate) incoming_setup_prefix: Option<usize>,
    pub(crate) dimension_ceiling: CommitmentRingDims,
    pub(crate) payload_phase: akita_types::CommitmentPayloadPhase,
}

impl SuffixState {
    fn memo_key(
        self,
        policy: &PlannerPolicy,
    ) -> (
        usize,
        usize,
        u32,
        usize,
        usize,
        usize,
        usize,
        akita_types::CommitmentPayloadPhase,
    ) {
        let memo_dimensions = match policy.ring_dimension_schedule_mode {
            crate::RingDimensionScheduleMode::AdaptiveDimension {
                num_search_levels,
                uniform_suffix_dimension,
                ..
            } if self.level >= num_search_levels => {
                CommitmentRingDims::uniform(uniform_suffix_dimension)
            }
            _ => self.dimension_ceiling,
        };
        (
            self.level,
            self.current_witness_len,
            self.current_lb,
            self.incoming_setup_prefix.unwrap_or(0),
            memo_dimensions.d_a(),
            memo_dimensions.d_b(),
            memo_dimensions.d_d(),
            self.payload_phase,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn price_level_candidate_with_children(
    ctx: &SuffixCtx<'_>,
    state: SuffixState,
    candidate_params: &CommittedGroupParams,
    next_witness_len: usize,
    eor_bytes: usize,
    natural_len: usize,
    direct_child: Option<&SuffixResult>,
    offloaded_child: Option<&SuffixResult>,
    require_child_fold: bool,
    frontier: &mut ProjectedFrontier,
    mixed_frontier: &mut Vec<ScheduleCandidate>,
) -> Result<(), AkitaError> {
    let policy = ctx.policy;
    // Only a prefix-consuming state is read through the setup projection by
    // an offloaded parent. The top-level recursive objective also reads the
    // root setup projection. Ordinary direct suffixes are consumed solely
    // through `best_by_payload_per_lb`, so retaining a parallel setup winner
    // there duplicates frontier work and memo ownership with no observer.
    let direct_projection =
        if state.incoming_setup_prefix.is_some() || (ctx.level_zero_is_root && state.level == 0) {
            FrontierProjection::Both
        } else {
            FrontierProjection::Payload
        };
    // Branch A: terminate directly on the witness entering this state.
    // There is no alternative terminal-shaped predecessor output: the
    // predecessor produces one canonical witness, and the terminal inner
    // commitment consumes that exact witness.
    let adaptive_terminal_is_allowed = !matches!(
        policy.ring_dimension_schedule_mode,
        crate::RingDimensionScheduleMode::AdaptiveDimension {
            num_search_levels,
            ..
        } if policy.selection_policy
            == crate::SelectionPolicyId::MinSetupMatrixFieldElementsThenProofPayload
            && state.level < num_search_levels
    );
    if adaptive_terminal_is_allowed
        && !(ctx.level_zero_is_root && state.level == 0)
        && state.incoming_setup_prefix.is_none()
        && !candidate_params.has_precommitted_groups()
    {
        let field_bits = policy.decomposition.field_bits();
        if let Some((mut direct_step, suffix_cost)) = try_terminal_direct_suffix_cost(
            state.current_witness_len,
            candidate_params,
            field_bits,
            ctx.key,
            state.level,
            None,
        )? {
            let level_proof_size = akita_types::proof_size::FOLD_GRIND_NONCE_BYTES
                .checked_add(eor_bytes)
                .ok_or_else(|| AkitaError::InvalidSetup("terminal proof size overflow".into()))?;
            let total = level_proof_size.checked_add(suffix_cost).ok_or_else(|| {
                AkitaError::InvalidSetup("terminal proof size overflow".to_string())
            })?;
            direct_step.estimated_direct_payload_bytes = level_proof_size;
            let candidate = ScheduleCandidate {
                first_direct_setup_field_len: Some(natural_len),
                total_bytes: total,
                setup_field_elements: terminal_setup_field_elements(&direct_step.params)?,
                folds: super::CandidateFoldChain::default(),
                terminal: Arc::new(direct_step),
            };
            frontier.consider_candidate(policy, candidate.clone(), direct_projection)?;
            insert_mixed_frontier(policy, mixed_frontier, candidate);
        }
    }

    let level_setup_field_elements = level_setup_field_elements(candidate_params)?;
    let direct_edge = ChildEdge {
        policy,
        candidate_params: Arc::new(candidate_params.clone()),
        current_witness_len: state.current_witness_len,
        next_witness_len,
        natural_setup_field_len: natural_len,
        level_setup_field_elements,
        eor_bytes,
        offloaded: false,
        require_child_fold,
        setup_field_budget: ctx.setup_field_budget,
    };
    if let Some(direct_child) = direct_child {
        consider_child_suffixes(
            &direct_edge,
            &direct_child.best_by_payload_per_lb,
            direct_projection,
            frontier,
        )?;
        consider_mixed_child_suffixes(&direct_edge, &direct_child.mixed_frontier, mixed_frontier)?;
    }
    if let Some(offloaded_child) = offloaded_child {
        let offloaded_edge = ChildEdge {
            offloaded: true,
            ..direct_edge
        };
        consider_child_suffixes(
            &offloaded_edge,
            &offloaded_child.best_by_first_direct_setup_per_lb,
            FrontierProjection::FirstDirectSetup,
            frontier,
        )?;
        consider_child_suffixes(
            &offloaded_edge,
            &offloaded_child.best_by_payload_per_lb,
            FrontierProjection::Payload,
            frontier,
        )?;
    }

    Ok(())
}

/// Shared inputs for root-level `CommittedGroupParams` candidates.
/// Suffix DP for the optimal recursive schedule at
/// `(level, current_witness_len, current_lb)`.
///
/// At each state, `best_by_first_direct_setup_per_lb` keeps one candidate per
/// `log_basis` (from
/// [`derive_candidate_level_params`]). A candidate may terminate on the current
/// witness when there is no incoming setup prefix, or fold again and consume
/// `incoming_setup_prefix` when present. Fold-again edges plan exactly one child
/// state: recursive setup edges pass the outgoing setup prefix to the child,
/// while direct edges plan the ordinary no-prefix child.
pub(crate) fn derive_optimal_suffix_schedule(
    ctx: &SuffixCtx<'_>,
    memo: &mut ScheduleMemo,
    state: SuffixState,
    depth: usize,
) -> Result<Arc<SuffixResult>, AkitaError> {
    let SuffixCtx {
        policy,
        default_ring_challenge_cfg,
        ring_challenge_config,
        num_vars,
        key,
        setup_field_budget: _,
        root_lookup_key,
        root_honest_fold_policy,
        precommitted_honest_fold_policies,
        level_zero_is_root,
    } = *ctx;
    let SuffixState {
        level,
        current_witness_len,
        current_lb,
        incoming_setup_prefix,
        dimension_ceiling,
        payload_phase,
    } = state;
    let memo_key = state.memo_key(policy);
    if depth <= MAX_RECURSION_DEPTH {
        if let Some(cached) = memo.get(&memo_key) {
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
    let mut best_by_first_direct_setup_per_lb: BTreeMap<FirstFoldKey, ScheduleCandidate> =
        BTreeMap::new();
    let mut best_by_payload_per_lb: BTreeMap<FirstFoldKey, ScheduleCandidate> = BTreeMap::new();
    let mut mixed_frontier = Vec::new();
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
    let root_eor_key = root_level_key
        .map(|root_key| {
            root_key
                .num_polynomials()
                .map(|total_polys| PolynomialGroupLayout::new(root_key.max_num_vars(), total_polys))
        })
        .transpose()?;
    let eor_key = root_eor_key.unwrap_or_else(|| {
        if level_zero_is_root && level == 0 {
            key
        } else {
            PolynomialGroupLayout::singleton(num_vars)
        }
    });
    let scalar_opening_layout = if root_level_key.is_some() {
        None
    } else {
        Some(suffix_opening_layout(
            current_witness_len,
            incoming_setup_prefix,
        )?)
    };
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
    let (min_inner_basis, max_inner_basis) = policy.inner_basis_search_range(inner_source)?;
    let (min_open_basis, max_open_basis) = policy.log_basis_search_range_at_level(level);
    for open_lb in min_open_basis..=max_open_basis {
        if open_lb < current_lb {
            continue;
        }
        let mut frontier = ProjectedFrontier::default();
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
        let mut candidates = Vec::new();

        for inner_lb in min_inner_basis..=max_inner_basis {
            if let Some(root_key) = root_level_key {
                for dimensions in dimension_candidates(policy, level, dimension_ceiling)? {
                    let Some(eor_bytes) = try_extension_opening_reduction_level_bytes(
                        policy.challenge_field_bits()?,
                        policy.claim_ext_degree,
                        level,
                        eor_key,
                        current_witness_len,
                        dimensions.inner(),
                    )?
                    else {
                        continue;
                    };
                    let ring_challenge_cfg = match dimensions {
                        akita_schedules::planner_support::RingDimensionCandidate::Fixed(value)
                            if value
                                == CommitmentRingDims::uniform(policy.uniform_ring_dimension) =>
                        {
                            *default_ring_challenge_cfg
                        }
                        _ => {
                            let Ok(config) = ring_challenge_config(dimensions.inner()) else {
                                continue;
                            };
                            config
                        }
                    };
                    let dimension_candidates = root_level_candidates_for_basis(
                        root_key,
                        root_honest_fold_policy.ok_or_else(|| {
                            AkitaError::InvalidSetup(
                                "root batch is missing its honest fold policy".to_string(),
                            )
                        })?,
                        precommitted_honest_fold_policies,
                        policy,
                        dimensions,
                        &ring_challenge_cfg,
                        ring_challenge_config,
                        current_witness_len,
                        inner_lb,
                        open_lb,
                        true,
                    )?;
                    candidates.extend(
                        dimension_candidates
                            .into_iter()
                            .map(|(params, next_witness_len)| {
                                (params, next_witness_len, eor_bytes)
                            }),
                    );
                }
            } else {
                for dimensions in dimension_candidates(policy, level, dimension_ceiling)? {
                    let Some(eor_bytes) = try_extension_opening_reduction_level_bytes(
                        policy.challenge_field_bits()?,
                        policy.claim_ext_degree,
                        level,
                        eor_key,
                        current_witness_len,
                        dimensions.inner(),
                    )?
                    else {
                        continue;
                    };
                    let Ok(ring_challenge_cfg) = ring_challenge_config(dimensions.inner()) else {
                        continue;
                    };
                    for &mode in
                        payload_phase.candidate_modes(level, incoming_setup_prefix.is_some())
                    {
                        let exhaustive_splits = incoming_setup_prefix.is_some()
                            || matches!(
                                policy.ring_dimension_schedule_mode,
                                crate::RingDimensionScheduleMode::AdaptiveDimension {
                                    num_search_levels,
                                    ..
                                } if level < num_search_levels
                            );
                        let level_candidates = if exhaustive_splits {
                            derive_candidate_level_params_all_splits(
                                Some(&mut memo.setup_prefixes),
                                policy,
                                mode,
                                &ring_challenge_cfg,
                                dimensions,
                                current_witness_len,
                                inner_source,
                                inner_lb,
                                open_lb,
                                level,
                                incoming_setup_prefix,
                            )?
                        } else {
                            derive_candidate_level_params(
                                Some(&mut memo.setup_prefixes),
                                policy,
                                mode,
                                &ring_challenge_cfg,
                                dimensions,
                                current_witness_len,
                                inner_source,
                                inner_lb,
                                open_lb,
                                level,
                                incoming_setup_prefix,
                            )?
                            .into_iter()
                            .collect()
                        };
                        candidates.extend(level_candidates.into_iter().map(
                            |(params, next_witness_len)| (params, next_witness_len, eor_bytes),
                        ));
                    }
                }
            }
        }
        let candidates = prune::level_candidates(current_opening_layout, candidates)?;
        if candidates.is_empty() {
            continue;
        }

        for (candidate_params, next_witness_len, eor_bytes) in candidates {
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
            let natural_len = active_setup_field_len(&candidate_params, current_opening_layout)?;
            let direct_child = if depth == MAX_RECURSION_DEPTH {
                Some(empty_suffix_result())
            } else {
                Some(derive_optimal_suffix_schedule(
                    ctx,
                    memo,
                    SuffixState {
                        level: level + 1,
                        current_witness_len: next_witness_len,
                        current_lb: open_lb,
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
                Some(derive_optimal_suffix_schedule(
                    ctx,
                    memo,
                    SuffixState {
                        level: level + 1,
                        current_witness_len: next_witness_len,
                        current_lb: open_lb,
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
                eor_bytes,
                natural_len,
                direct_child.as_deref(),
                offloaded_child.as_deref(),
                require_child_fold,
                &mut frontier,
                &mut mixed_frontier,
            )?;
        }

        for (parent_key, choices) in frontier.by_parent_cost {
            let key = FirstFoldKey {
                log_basis: open_lb,
                outer_payload_bytes: parent_key.outer_payload_bytes,
                stage3_payload_bytes: parent_key.stage3_payload_bytes,
            };
            if let Some(choice) = choices.setup {
                best_by_first_direct_setup_per_lb.insert(key, choice);
            }
            if let Some(choice) = choices.payload {
                best_by_payload_per_lb.insert(key, choice);
            }
        }
    }

    let result = Arc::new(SuffixResult {
        best_by_first_direct_setup_per_lb,
        best_by_payload_per_lb,
        mixed_frontier,
    });
    memo.insert(memo_key, Arc::clone(&result));
    Ok(result)
}

#[cfg(test)]
#[path = "../test/suffix_dp.rs"]
mod tests;
