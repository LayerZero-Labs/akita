use std::{collections::BTreeMap, sync::Arc};

use akita_field::AkitaError;
use akita_types::{
    active_setup_field_len, level_proof_bytes, try_extension_opening_reduction_level_bytes,
    AkitaScheduleLookupKey, CommitmentRingDims, CommittedGroupParams, PolynomialGroupLayout,
};

use crate::{planner::root_level_candidates_for_basis, PlannerPolicy};

use super::{
    derive_candidate_level_params_all_splits, level_setup_field_elements,
    stage3_payload_bytes_for_successor, suffix_opening_layout, terminal_setup_field_elements,
    CandidateFoldChain, CandidateFoldStep, CandidateTerminalResponse, ScheduleCandidate,
    MAX_RECURSION_DEPTH,
};

mod frontier;
mod prune;
mod state;
mod terminal;

use frontier::{consider_child_suffixes, FrontierProjection, ProjectedFrontier};
#[cfg(test)]
use state::MAX_SUFFIX_SEARCH_CACHE_ENTRIES;
use state::{FirstFoldKey, ParentVisibleCost};
pub(crate) use state::{SuffixCtx, SuffixResult, SuffixSearchCache, SuffixState};
pub(super) use terminal::try_direct_cost as try_terminal_direct_suffix_cost;

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
    suffix_folds: CandidateFoldChain,
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
        terminal: Arc::clone(&suffix.terminal),
    }))
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
    frontier: &mut ProjectedFrontier,
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
        if let Some((mut direct_step, suffix_cost)) = terminal::try_direct_cost(
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
                folds: CandidateFoldChain::default(),
                terminal: Arc::new(direct_step),
            };
            frontier.consider_candidate(policy, candidate, FrontierProjection::Both)?;
        }
    }

    let candidate_params = Arc::new(candidate_params.clone());
    let level_setup_field_elements = level_setup_field_elements(&candidate_params)?;
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
        FrontierProjection::Both,
        frontier,
    )?;
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
    memo: &mut SuffixSearchCache,
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
    let memo_key = state;
    if depth <= MAX_RECURSION_DEPTH {
        if let Some(cached) = memo.get(&memo_key) {
            return Ok(Arc::clone(cached));
        }
    }

    if depth > MAX_RECURSION_DEPTH {
        let result = state::empty_result();
        memo.insert(memo_key, &result);
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
        let result = state::empty_result();
        memo.insert(memo_key, &result);
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
        let mut frontier = ProjectedFrontier::default();
        let inner_source = if level_zero_is_root && level == 0 {
            super::root_inner_basis_source(
                root_honest_fold_policy.ok_or_else(|| {
                    AkitaError::InvalidSetup(
                        "root batch is missing its honest fold policy".to_string(),
                    )
                })?,
                policy.decomposition.log_commit_bound,
            )
        } else {
            crate::InnerBasisSource::BalancedDigits {
                log_basis: current_lb,
            }
        };
        let (min_inner_basis, max_inner_basis) = policy.inner_basis_search_range(inner_source)?;

        for inner_lb in min_inner_basis..=max_inner_basis {
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
                    inner_lb,
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
                    candidates.extend(derive_candidate_level_params_all_splits(
                        Some(&mut memo.setup_prefixes),
                        policy,
                        mode,
                        &ring_challenge_cfg,
                        dimensions,
                        current_witness_len,
                        crate::InnerBasisSource::BalancedDigits {
                            log_basis: current_lb,
                        },
                        inner_lb,
                        lb,
                        level,
                        incoming_setup_prefix,
                    )?);
                }
                if candidates.is_empty() {
                    continue;
                }
                (
                    scalar_opening_layout.as_ref().ok_or_else(|| {
                        AkitaError::InvalidSetup(
                            "scalar suffix opening layout is missing".to_string(),
                        )
                    })?,
                    candidates,
                    false,
                )
            };

            let candidates = prune::level_candidates(current_opening_layout, candidates)?;
            for (candidate_params, next_witness_len) in candidates {
                if let Some(natural_prefix_len) = incoming_setup_prefix {
                    let padded_prefix_len =
                        akita_types::padded_setup_prefix_len(natural_prefix_len);
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
                let natural_len =
                    active_setup_field_len(&candidate_params, current_opening_layout)?;
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
                    &mut frontier,
                )?;
            }
        }

        for (parent_cost, choices) in frontier.by_parent_cost {
            let key = FirstFoldKey {
                log_basis: lb,
                parent_cost,
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
    });
    memo.insert(memo_key, &result);
    Ok(result)
}

#[cfg(test)]
#[path = "../test/suffix_dp.rs"]
mod tests;
