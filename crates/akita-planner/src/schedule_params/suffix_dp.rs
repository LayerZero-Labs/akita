use std::{
    collections::{BTreeMap, HashMap},
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
    derive_candidate_level_params, level_setup_field_elements, stage3_payload_bytes_for_successor,
    suffix_opening_layout, terminal_setup_field_elements, CandidateFoldStep,
    CandidateTerminalResponse, ScheduleCandidate, MAX_RECURSION_DEPTH,
};

/// Result of the suffix DP at one state. Both shape options are reported
/// because the parent's proof-size formula depends on the child's first
/// step:
///
/// - `best_by_first_direct_setup_per_lb` — lexicographically best fold-first
///   schedule by first direct setup scan and then proof payload, per first-fold
///   `log_basis`. An entry with no ordinary folds terminates directly on the
///   current witness; otherwise it consumes `incoming_setup_prefix` when one
///   is present.
/// - `best_by_payload_per_lb` — smallest-payload fold-first schedule per
///   first-fold `log_basis`, used after an earlier direct edge has fixed the
///   setup-size objective.
#[derive(Clone)]
pub(crate) struct SuffixResult {
    pub(crate) best_by_first_direct_setup_per_lb: BTreeMap<FirstFoldKey, ScheduleCandidate>,
    pub(crate) best_by_payload_per_lb: BTreeMap<FirstFoldKey, ScheduleCandidate>,
}

/// Parent-visible first-fold class. A parent edge prices the child's outgoing
/// commitment payload, so suffixes with different first payload sizes are not
/// interchangeable even when they use the same digit basis.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct FirstFoldKey {
    log_basis: u32,
    outer_payload_bytes: usize,
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

pub(crate) type ScheduleMemo = HashMap<
    (
        usize,
        usize,
        u32,
        usize,
        akita_types::CommitmentPayloadPhase,
    ),
    Arc<SuffixResult>,
>;

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
    candidate_params: &'a CommittedGroupParams,
    current_witness_len: usize,
    next_witness_len: usize,
    natural_setup_field_len: usize,
    level_setup_field_elements: usize,
    eor_bytes: usize,
    offloaded: bool,
    require_child_fold: bool,
    setup_field_budget: Option<usize>,
}

#[derive(Clone, Copy)]
struct ChildObjectives {
    first_direct_setup_then_payload: bool,
    payload: bool,
}

fn consider_child_suffixes(
    edge: &ChildEdge<'_>,
    child_candidates: &BTreeMap<FirstFoldKey, ScheduleCandidate>,
    objectives: ChildObjectives,
    best_by_setup: &mut BTreeMap<usize, ScheduleCandidate>,
    best_by_payload: &mut BTreeMap<usize, ScheduleCandidate>,
) -> Result<(), AkitaError> {
    for suffix in child_candidates.values() {
        let Some(candidate) = child_choice(edge, suffix)? else {
            continue;
        };
        update_candidate_frontiers(
            edge.policy,
            candidate,
            objectives,
            best_by_setup,
            best_by_payload,
        )?;
    }
    Ok(())
}

fn first_outer_payload_bytes(
    policy: &PlannerPolicy,
    candidate: &ScheduleCandidate,
) -> Result<usize, AkitaError> {
    let Some(first) = candidate.first_fold_params() else {
        return Ok(0);
    };
    first
        .outer_payload_geometry()?
        .transmitted_coefficients()
        .checked_mul(akita_types::layout::field_bytes(
            policy.decomposition.field_bits(),
        ))
        .ok_or_else(|| AkitaError::InvalidSetup("first-fold payload size overflow".into()))
}

fn update_candidate_frontiers(
    policy: &PlannerPolicy,
    candidate: ScheduleCandidate,
    objectives: ChildObjectives,
    best_by_setup: &mut BTreeMap<usize, ScheduleCandidate>,
    best_by_payload: &mut BTreeMap<usize, ScheduleCandidate>,
) -> Result<(), AkitaError> {
    let outer_payload_bytes = first_outer_payload_bytes(policy, &candidate)?;
    let improves_setup = objectives.first_direct_setup_then_payload
        && best_by_setup.get(&outer_payload_bytes).is_none_or(|best| {
            candidate.recursive_setup_frontier_score() < best.recursive_setup_frontier_score()
        });
    let improves_payload = objectives.payload
        && best_by_payload
            .get(&outer_payload_bytes)
            .is_none_or(|best| candidate.direct_frontier_score() < best.direct_frontier_score());
    if improves_setup {
        best_by_setup.insert(outer_payload_bytes, candidate.clone());
    }
    if improves_payload {
        best_by_payload.insert(outer_payload_bytes, candidate);
    }
    Ok(())
}

fn child_choice(
    edge: &ChildEdge<'_>,
    suffix: &ScheduleCandidate,
) -> Result<Option<ScheduleCandidate>, AkitaError> {
    let child_is_terminal = suffix.folds.is_empty();
    if edge.require_child_fold && child_is_terminal {
        return Ok(None);
    }
    if edge.offloaded {
        if child_is_terminal || suffix.folds.len() == 1 {
            return Ok(None);
        }
        if suffix.first_direct_setup_field_len_or_max() >= edge.natural_setup_field_len {
            return Ok(None);
        }
    }

    let direct_payload_bytes = level_proof_bytes(
        edge.policy.decomposition.field_bits(),
        edge.policy.challenge_field_bits()?,
        edge.candidate_params,
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
    let mut folds = Vec::with_capacity(1 + suffix.folds.len());
    folds.push(CandidateFoldStep {
        params: edge.candidate_params.clone(),
        input_witness_len: edge.current_witness_len,
        output_witness_len: edge.next_witness_len,
        estimated_direct_payload_bytes: direct_payload_bytes,
        estimated_stage3_payload_bytes: stage3_payload_bytes,
    });
    folds.extend(suffix.folds.iter().cloned());
    Ok(Some(ScheduleCandidate {
        first_direct_setup_field_len,
        total_bytes,
        setup_field_elements,
        folds,
        terminal: suffix.terminal.clone(),
    }))
}

fn empty_suffix_result() -> Arc<SuffixResult> {
    Arc::new(SuffixResult {
        best_by_first_direct_setup_per_lb: BTreeMap::new(),
        best_by_payload_per_lb: BTreeMap::new(),
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
    pub(crate) payload_phase: akita_types::CommitmentPayloadPhase,
}

impl SuffixState {
    fn memo_key(
        self,
    ) -> (
        usize,
        usize,
        u32,
        usize,
        akita_types::CommitmentPayloadPhase,
    ) {
        (
            self.level,
            self.current_witness_len,
            self.current_lb,
            self.incoming_setup_prefix.unwrap_or(0),
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
    direct_child: &SuffixResult,
    offloaded_child: Option<&SuffixResult>,
    require_child_fold: bool,
    best_for_this_lb: &mut BTreeMap<usize, ScheduleCandidate>,
    best_payload_for_this_lb: &mut BTreeMap<usize, ScheduleCandidate>,
) -> Result<(), AkitaError> {
    let policy = ctx.policy;
    // Branch A: terminate directly on the witness entering this state.
    // There is no alternative terminal-shaped predecessor output: the
    // predecessor produces one canonical witness, and the terminal inner
    // commitment consumes that exact witness.
    if !(ctx.level_zero_is_root && state.level == 0)
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
                folds: Vec::new(),
                terminal: direct_step,
            };
            update_candidate_frontiers(
                policy,
                candidate,
                ChildObjectives {
                    first_direct_setup_then_payload: true,
                    payload: true,
                },
                best_for_this_lb,
                best_payload_for_this_lb,
            )?;
        }
    }

    let level_setup_field_elements = level_setup_field_elements(candidate_params)?;
    let direct_edge = ChildEdge {
        policy,
        candidate_params,
        current_witness_len: state.current_witness_len,
        next_witness_len,
        natural_setup_field_len: natural_len,
        level_setup_field_elements,
        eor_bytes,
        offloaded: false,
        require_child_fold,
        setup_field_budget: ctx.setup_field_budget,
    };
    consider_child_suffixes(
        &direct_edge,
        &direct_child.best_by_payload_per_lb,
        ChildObjectives {
            first_direct_setup_then_payload: true,
            payload: true,
        },
        best_for_this_lb,
        best_payload_for_this_lb,
    )?;
    if let Some(offloaded_child) = offloaded_child {
        let offloaded_edge = ChildEdge {
            offloaded: true,
            ..direct_edge
        };
        consider_child_suffixes(
            &offloaded_edge,
            &offloaded_child.best_by_first_direct_setup_per_lb,
            ChildObjectives {
                first_direct_setup_then_payload: true,
                payload: false,
            },
            best_for_this_lb,
            best_payload_for_this_lb,
        )?;
        consider_child_suffixes(
            &offloaded_edge,
            &offloaded_child.best_by_payload_per_lb,
            ChildObjectives {
                first_direct_setup_then_payload: false,
                payload: true,
            },
            best_for_this_lb,
            best_payload_for_this_lb,
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
        payload_phase,
    } = state;
    let memo_key = state.memo_key();
    if depth <= MAX_RECURSION_DEPTH {
        if let Some(cached) = memo.get(&memo_key) {
            return Ok(Arc::clone(cached));
        }
    }

    if depth > MAX_RECURSION_DEPTH {
        let result = empty_suffix_result();
        memo.insert(memo_key, Arc::clone(&result));
        return Ok(result);
    }

    let mut best_by_first_direct_setup_per_lb: BTreeMap<FirstFoldKey, ScheduleCandidate> =
        BTreeMap::new();
    let mut best_by_payload_per_lb: BTreeMap<FirstFoldKey, ScheduleCandidate> = BTreeMap::new();
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
    let Some(eor_bytes) = try_extension_opening_reduction_level_bytes(
        policy.challenge_field_bits()?,
        policy.claim_ext_degree,
        level,
        eor_key,
        current_witness_len,
        policy.uniform_ring_dimension,
    )?
    else {
        let result = empty_suffix_result();
        memo.insert(memo_key, Arc::clone(&result));
        return Ok(result);
    };
    let scalar_opening_layout = if root_level_key.is_some() {
        None
    } else {
        Some(suffix_opening_layout(
            current_witness_len,
            incoming_setup_prefix,
        )?)
    };
    let (min_log_basis, max_log_basis) = policy.log_basis_search_range_at_level(level);
    for lb in min_log_basis..=max_log_basis {
        if lb < current_lb {
            continue;
        }
        let mut best_for_this_lb = BTreeMap::<usize, ScheduleCandidate>::new();
        let mut best_payload_for_this_lb = BTreeMap::<usize, ScheduleCandidate>::new();

        let (current_opening_layout, candidates, require_child_fold) = if let Some(root_key) =
            root_level_key
        {
            let current_opening_layout = root_opening_layout.as_ref().ok_or_else(|| {
                AkitaError::InvalidSetup("root batch opening layout is missing".to_string())
            })?;
            let dimensions = CommitmentRingDims::uniform(policy.uniform_ring_dimension);
            let candidates = root_level_candidates_for_basis(
                root_key,
                root_honest_fold_policy.ok_or_else(|| {
                    AkitaError::InvalidSetup(
                        "root batch is missing its honest fold policy".to_string(),
                    )
                })?,
                precommitted_honest_fold_policies,
                policy,
                dimensions,
                default_ring_challenge_cfg,
                ring_challenge_config,
                current_witness_len,
                lb,
                true,
            )?;
            (
                current_opening_layout,
                candidates,
                !root_key.precommitteds.is_empty(),
            )
        } else {
            let mut candidates = Vec::new();
            let dimensions = CommitmentRingDims::uniform(policy.uniform_ring_dimension);
            let Ok(ring_challenge_cfg) = ring_challenge_config(dimensions.d_a()) else {
                continue;
            };
            for &mode in payload_phase.candidate_modes(level, incoming_setup_prefix.is_some()) {
                if let Some(candidate) = derive_candidate_level_params(
                    policy,
                    mode,
                    &ring_challenge_cfg,
                    dimensions,
                    current_witness_len,
                    lb,
                    level,
                    incoming_setup_prefix,
                )? {
                    candidates.push(candidate);
                }
            }
            if candidates.is_empty() {
                continue;
            }
            (
                scalar_opening_layout.as_ref().ok_or_else(|| {
                    AkitaError::InvalidSetup("scalar suffix opening layout is missing".to_string())
                })?,
                candidates,
                false,
            )
        };

        for (candidate_params, next_witness_len) in candidates {
            if let Some(natural_prefix_len) = incoming_setup_prefix {
                let padded_prefix_len = akita_types::padded_setup_prefix_len(natural_prefix_len);
                if !offloaded_witness_contracts(
                    current_witness_len,
                    current_lb,
                    padded_prefix_len,
                    policy.decomposition.field_bits(),
                    next_witness_len,
                    lb,
                    policy.min_offloaded_witness_contraction,
                )? {
                    continue;
                }
            }
            let natural_len = active_setup_field_len(&candidate_params, current_opening_layout)?;
            let direct_child = derive_optimal_suffix_schedule(
                ctx,
                memo,
                SuffixState {
                    level: level + 1,
                    current_witness_len: next_witness_len,
                    current_lb: lb,
                    incoming_setup_prefix: None,
                    payload_phase: payload_phase.after(candidate_params.payload_mode),
                },
                depth + 1,
            )?;
            let offloaded_child = if policy.recursive_setup_planning
                && candidate_params.payload_mode.is_compressed()
            {
                Some(derive_optimal_suffix_schedule(
                    ctx,
                    memo,
                    SuffixState {
                        level: level + 1,
                        current_witness_len: next_witness_len,
                        current_lb: lb,
                        incoming_setup_prefix: Some(natural_len),
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
                &direct_child,
                offloaded_child.as_deref(),
                require_child_fold,
                &mut best_for_this_lb,
                &mut best_payload_for_this_lb,
            )?;
        }

        for (outer_payload_bytes, choice) in best_for_this_lb {
            let ScheduleCandidate {
                first_direct_setup_field_len,
                total_bytes,
                setup_field_elements,
                folds,
                terminal,
            } = choice;
            best_by_first_direct_setup_per_lb.insert(
                FirstFoldKey {
                    log_basis: lb,
                    outer_payload_bytes,
                },
                ScheduleCandidate {
                    total_bytes,
                    setup_field_elements,
                    first_direct_setup_field_len,
                    folds,
                    terminal,
                },
            );
        }
        for (outer_payload_bytes, choice) in best_payload_for_this_lb {
            let ScheduleCandidate {
                first_direct_setup_field_len,
                total_bytes,
                setup_field_elements,
                folds,
                terminal,
            } = choice;
            best_by_payload_per_lb.insert(
                FirstFoldKey {
                    log_basis: lb,
                    outer_payload_bytes,
                },
                ScheduleCandidate {
                    total_bytes,
                    setup_field_elements,
                    first_direct_setup_field_len,
                    folds,
                    terminal,
                },
            );
        }
    }

    let result = Arc::new(SuffixResult {
        best_by_first_direct_setup_per_lb,
        best_by_payload_per_lb,
    });
    memo.insert(memo_key, Arc::clone(&result));
    Ok(result)
}

#[cfg(test)]
#[path = "../test/suffix_dp.rs"]
mod tests;
