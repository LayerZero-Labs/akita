use std::collections::{BTreeMap, HashMap};

use akita_field::AkitaError;
use akita_types::{
    active_setup_field_len, extension_opening_reduction_level_bytes, level_proof_bytes,
    terminal_response_bytes, CommittedGroupParams, OpeningClaimsLayout, PolynomialGroupLayout,
    TerminalResponseShape,
};

use crate::PlannerPolicy;

use super::{
    derive_candidate_level_params, stage3_payload_bytes_for_successor, suffix_opening_layout,
    CandidateFoldStep, CandidateTerminalResponse, MAX_RECURSION_DEPTH,
};

const MIN_OFFLOADED_WITNESS_CONTRACTION: usize = 3;

/// A fold-first suffix schedule.
///
/// The parent's proof-size formula needs the child's first fold params
/// (`first_fold_params`), so the suffix carries it directly instead of
/// re-reading `folds[0]`.
#[derive(Clone)]
pub(crate) struct FoldSuffix {
    pub(crate) total_bytes: usize,
    pub(crate) setup_envelope_ring_elements: usize,
    pub(crate) first_direct_setup_field_len: usize,
    pub(crate) first_fold_params: Option<CommittedGroupParams>,
    pub(crate) folds: Vec<CandidateFoldStep>,
    pub(crate) terminal: CandidateTerminalResponse,
}

/// Result of the suffix DP at one state. Both shape options are reported
/// because the parent's proof-size formula depends on the child's first
/// step:
///
/// - `best_fold_per_lb` — best fold-first schedule per first-fold
///   `log_basis`. An entry with no ordinary folds terminates directly on the
///   current witness; otherwise it consumes `incoming_setup_prefix` when one
///   is present.
/// - `best_proof_fold_per_lb` — best fold-first schedule per first-fold
///   `log_basis` after an earlier direct fold has already fixed the setup-size
///   objective, so only proof bytes matter for the remaining suffix.
#[derive(Clone)]
pub(crate) struct SuffixResult {
    pub(crate) best_fold_per_lb: BTreeMap<u32, FoldSuffix>,
    pub(crate) best_proof_fold_per_lb: BTreeMap<u32, FoldSuffix>,
}

impl SuffixResult {
    pub(crate) fn is_empty(&self) -> bool {
        self.best_fold_per_lb.is_empty()
    }
}

fn make_terminal_direct_step(
    input_witness_len: usize,
    terminal_lp: &CommittedGroupParams,
    field_bits: u32,
    num_polynomials: usize,
    opening_layout: Option<&OpeningClaimsLayout>,
) -> Result<CandidateTerminalResponse, AkitaError> {
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
    Ok(CandidateTerminalResponse {
        params: terminal_params,
        sparse_challenge_config: terminal_lp.fold_challenge_config,
        input_witness_len,
        estimated_direct_payload_bytes: 0,
        response_shape: witness_shape,
        estimated_payload_bytes: terminal_bytes,
    })
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
    let direct = make_terminal_direct_step(
        input_witness_len,
        terminal_lp,
        field_bits,
        num_polynomials,
        opening_layout,
    )?;
    let terminal_bytes = direct.estimated_payload_bytes;
    Ok((direct, terminal_bytes))
}

pub(crate) type ScheduleMemo = HashMap<(usize, usize, u32, usize), SuffixResult>;

#[derive(Clone)]
struct CandidateSuffixChoice {
    first_direct_setup_field_len: usize,
    total_bytes: usize,
    setup_envelope_ring_elements: usize,
    folds: Vec<CandidateFoldStep>,
    terminal: CandidateTerminalResponse,
}

fn level_setup_envelope(params: &CommittedGroupParams) -> Result<usize, AkitaError> {
    let mut envelope = 1;
    akita_types::accumulate_matrix_envelope_for_level(params, &mut envelope)?;
    Ok(envelope)
}

fn terminal_setup_envelope(
    params: &akita_types::TerminalCommittedGroupParams,
) -> Result<usize, AkitaError> {
    let mut envelope = 1;
    akita_types::accumulate_terminal_matrix_envelope(params, &mut envelope)?;
    Ok(envelope)
}

fn update_best_suffix_choices(
    first_direct_setup_field_len: usize,
    total_bytes: usize,
    setup_envelope_ring_elements: usize,
    folds: Vec<CandidateFoldStep>,
    terminal: CandidateTerminalResponse,
    best_by_setup: &mut Option<CandidateSuffixChoice>,
    best_by_proof: &mut Option<CandidateSuffixChoice>,
) {
    let candidate = CandidateSuffixChoice {
        first_direct_setup_field_len,
        total_bytes,
        setup_envelope_ring_elements,
        folds,
        terminal,
    };
    if best_by_setup
        .as_ref()
        .map(|best| {
            (first_direct_setup_field_len, total_bytes)
                < (best.first_direct_setup_field_len, best.total_bytes)
        })
        .unwrap_or(true)
    {
        *best_by_setup = Some(candidate.clone());
    }
    if best_by_proof
        .as_ref()
        .map(|best| total_bytes < best.total_bytes)
        .unwrap_or(true)
    {
        *best_by_proof = Some(candidate);
    }
}

fn offloaded_witness_contracts(
    input_witness_len: usize,
    input_log_basis: u32,
    setup_prefix_field_len: usize,
    field_bits: u32,
    output_witness_len: usize,
    output_log_basis: u32,
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
        .and_then(|bits| bits.checked_mul(MIN_OFFLOADED_WITNESS_CONTRACTION))
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
    eor_bytes: usize,
    offloaded: bool,
    setup_envelope_budget: Option<usize>,
}

fn consider_child_suffixes(
    edge: &ChildEdge<'_>,
    child_candidates: &BTreeMap<u32, FoldSuffix>,
    update_setup_choice: bool,
    update_proof_choice: bool,
    best_by_setup: &mut Option<CandidateSuffixChoice>,
    best_by_proof: &mut Option<CandidateSuffixChoice>,
) -> Result<(), AkitaError> {
    for suffix in child_candidates.values() {
        let child_is_terminal = suffix.folds.is_empty();
        if edge.offloaded {
            if child_is_terminal || suffix.folds.len() == 1 {
                continue;
            }
            if suffix.first_direct_setup_field_len >= edge.natural_setup_field_len {
                continue;
            }
        }

        let direct_payload_bytes = level_proof_bytes(
            edge.policy.decomposition.field_bits(),
            edge.policy.decomposition.field_bits() * edge.policy.chal_ext_degree as u32,
            edge.candidate_params,
            suffix.first_fold_params.as_ref(),
            edge.next_witness_len,
            Some(if child_is_terminal {
                akita_types::NextWitnessBindingPolicy::TerminalInnerState
            } else {
                akita_types::NextWitnessBindingPolicy::OuterCommitment
            }),
        )?
        .checked_add(edge.eor_bytes)
        .ok_or_else(|| AkitaError::InvalidSetup("level proof size overflow".to_string()))?;
        let stage3_payload_bytes = stage3_payload_bytes_for_successor(
            edge.policy,
            suffix.first_fold_params.as_ref(),
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
        let setup_envelope_ring_elements =
            level_setup_envelope(edge.candidate_params)?.max(suffix.setup_envelope_ring_elements);
        if edge
            .setup_envelope_budget
            .is_some_and(|budget| setup_envelope_ring_elements > budget)
        {
            continue;
        }
        let first_direct_setup_field_len = if edge.offloaded {
            suffix.first_direct_setup_field_len
        } else {
            edge.natural_setup_field_len
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
        let candidate = CandidateSuffixChoice {
            first_direct_setup_field_len,
            total_bytes,
            setup_envelope_ring_elements,
            folds,
            terminal: suffix.terminal.clone(),
        };

        if update_setup_choice
            && best_by_setup.as_ref().is_none_or(|best| {
                (first_direct_setup_field_len, total_bytes)
                    < (best.first_direct_setup_field_len, best.total_bytes)
            })
        {
            *best_by_setup = Some(candidate.clone());
        }
        if update_proof_choice
            && best_by_proof
                .as_ref()
                .is_none_or(|best| total_bytes < best.total_bytes)
        {
            *best_by_proof = Some(candidate);
        }
    }
    Ok(())
}

/// DP-invariant inputs for the suffix search.
///
/// `policy`, `ring_challenge_cfg`, and `num_vars` are constant across the whole
/// recursion, so they are carried in one context value rather than as
/// per-call arguments (keeps the recursive signature small).
#[derive(Clone, Copy)]
pub(crate) struct SuffixCtx<'a> {
    pub(crate) policy: &'a PlannerPolicy,
    pub(crate) ring_challenge_cfg: &'a akita_challenges::SparseChallengeConfig,
    pub(crate) fold_challenge_shape_at_level:
        &'a dyn Fn(akita_types::AkitaScheduleInputs) -> akita_challenges::TensorChallengeShape,
    pub(crate) num_vars: usize,
    pub(crate) key: PolynomialGroupLayout,
    pub(crate) setup_envelope_budget: Option<usize>,
}

#[derive(Clone, Copy)]
pub(crate) struct SuffixState {
    pub(crate) level: usize,
    pub(crate) current_witness_len: usize,
    pub(crate) current_lb: u32,
    pub(crate) incoming_setup_prefix: Option<usize>,
}

impl SuffixState {
    fn memo_key(self) -> (usize, usize, u32, usize) {
        (
            self.level,
            self.current_witness_len,
            self.current_lb,
            self.incoming_setup_prefix.unwrap_or(0),
        )
    }
}

/// Shared inputs for root-level `CommittedGroupParams` candidates.
/// Suffix DP for the optimal recursive schedule at
/// `(level, current_witness_len, current_lb)`.
///
/// At each state, `best_fold_per_lb` keeps one candidate per `log_basis` (from
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
) -> Result<SuffixResult, AkitaError> {
    let SuffixCtx {
        policy,
        ring_challenge_cfg,
        fold_challenge_shape_at_level,
        num_vars,
        key,
        setup_envelope_budget,
    } = *ctx;
    let SuffixState {
        level,
        current_witness_len,
        current_lb,
        incoming_setup_prefix,
    } = state;
    let memo_key = state.memo_key();
    let requested_fold_shape = fold_challenge_shape_at_level(akita_types::AkitaScheduleInputs {
        num_vars,
        level,
        input_witness_len: current_witness_len,
    });
    if depth <= MAX_RECURSION_DEPTH {
        if let Some(cached) = memo.get(&memo_key) {
            return Ok(cached.clone());
        }
    }

    if depth > MAX_RECURSION_DEPTH {
        let result = SuffixResult {
            best_fold_per_lb: BTreeMap::new(),
            best_proof_fold_per_lb: BTreeMap::new(),
        };
        memo.insert(memo_key, result.clone());
        return Ok(result);
    }

    let mut best_fold_per_lb: BTreeMap<u32, FoldSuffix> = BTreeMap::new();
    let mut best_proof_fold_per_lb: BTreeMap<u32, FoldSuffix> = BTreeMap::new();
    let (configured_min_log_basis, max_log_basis) = policy.basis_range;
    let min_log_basis = configured_min_log_basis
        .max(policy.decomposition.log_basis)
        .max(if policy.decomposition.field_bits() < 128 {
            5
        } else {
            0
        });
    for lb in min_log_basis..=max_log_basis {
        if lb < current_lb {
            continue;
        }
        let Some((candidate_params, next_witness_len)) = derive_candidate_level_params(
            policy,
            ring_challenge_cfg,
            current_witness_len,
            lb,
            level,
            incoming_setup_prefix,
            requested_fold_shape,
        )?
        else {
            continue;
        };
        if let Some(natural_prefix_len) = incoming_setup_prefix {
            let padded_prefix_len = akita_types::padded_setup_prefix_len(natural_prefix_len);
            if !offloaded_witness_contracts(
                current_witness_len,
                current_lb,
                padded_prefix_len,
                policy.decomposition.field_bits(),
                next_witness_len,
                lb,
            )? {
                continue;
            }
        }
        let Ok(eor_bytes) = extension_opening_reduction_level_bytes(
            policy.decomposition.field_bits() * policy.chal_ext_degree as u32,
            policy.claim_ext_degree,
            level,
            PolynomialGroupLayout::singleton(num_vars),
            current_witness_len,
        ) else {
            continue;
        };

        let mut best_for_this_lb: Option<CandidateSuffixChoice> = None;
        let mut best_proof_for_this_lb: Option<CandidateSuffixChoice> = None;

        let current_opening_layout =
            suffix_opening_layout(current_witness_len, incoming_setup_prefix)?;
        let natural_len = active_setup_field_len(&candidate_params, &current_opening_layout)?;

        // Branch A: terminate directly on the witness entering this state.
        // There is no alternative terminal-shaped predecessor output: the
        // predecessor produces one canonical witness, and the terminal inner
        // commitment consumes that exact witness.
        if incoming_setup_prefix.is_none() && !candidate_params.has_precommitted_groups() {
            let field_bits = policy.decomposition.field_bits();
            if let Some((mut direct_step, suffix_cost)) = try_terminal_direct_suffix_cost(
                current_witness_len,
                &candidate_params,
                field_bits,
                key,
                level,
                None,
            )? {
                let level_proof_size = akita_types::proof_size::FOLD_GRIND_NONCE_BYTES
                    .checked_add(eor_bytes)
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup("terminal proof size overflow".into())
                    })?;
                let total = level_proof_size.checked_add(suffix_cost).ok_or_else(|| {
                    AkitaError::InvalidSetup("terminal proof size overflow".to_string())
                })?;
                direct_step.estimated_direct_payload_bytes = level_proof_size;
                update_best_suffix_choices(
                    natural_len,
                    total,
                    terminal_setup_envelope(&direct_step.params)?,
                    Vec::new(),
                    direct_step,
                    &mut best_for_this_lb,
                    &mut best_proof_for_this_lb,
                );
            }
        }

        let direct_child = derive_optimal_suffix_schedule(
            ctx,
            memo,
            SuffixState {
                level: level + 1,
                current_witness_len: next_witness_len,
                current_lb: lb,
                incoming_setup_prefix: None,
            },
            depth + 1,
        )?;
        let direct_edge = ChildEdge {
            policy,
            candidate_params: &candidate_params,
            current_witness_len,
            next_witness_len,
            natural_setup_field_len: natural_len,
            eor_bytes,
            offloaded: false,
            setup_envelope_budget,
        };
        consider_child_suffixes(
            &direct_edge,
            &direct_child.best_proof_fold_per_lb,
            true,
            true,
            &mut best_for_this_lb,
            &mut best_proof_for_this_lb,
        )?;

        if policy.recursive_setup_planning {
            let offloaded_child = derive_optimal_suffix_schedule(
                ctx,
                memo,
                SuffixState {
                    level: level + 1,
                    current_witness_len: next_witness_len,
                    current_lb: lb,
                    incoming_setup_prefix: Some(natural_len),
                },
                depth + 1,
            )?;
            let offloaded_edge = ChildEdge {
                offloaded: true,
                ..direct_edge
            };
            consider_child_suffixes(
                &offloaded_edge,
                &offloaded_child.best_fold_per_lb,
                true,
                false,
                &mut best_for_this_lb,
                &mut best_proof_for_this_lb,
            )?;
            consider_child_suffixes(
                &offloaded_edge,
                &offloaded_child.best_proof_fold_per_lb,
                false,
                true,
                &mut best_for_this_lb,
                &mut best_proof_for_this_lb,
            )?;
        }

        if let Some(choice) = best_for_this_lb {
            let CandidateSuffixChoice {
                first_direct_setup_field_len,
                total_bytes,
                setup_envelope_ring_elements,
                folds,
                terminal,
            } = choice;
            let first_fold_params = folds.first().map(|fold| fold.params.clone());
            best_fold_per_lb.insert(
                lb,
                FoldSuffix {
                    total_bytes,
                    setup_envelope_ring_elements,
                    first_direct_setup_field_len,
                    first_fold_params,
                    folds,
                    terminal,
                },
            );
        }
        if let Some(choice) = best_proof_for_this_lb {
            let CandidateSuffixChoice {
                first_direct_setup_field_len,
                total_bytes,
                setup_envelope_ring_elements,
                folds,
                terminal,
            } = choice;
            let first_fold_params = folds.first().map(|fold| fold.params.clone());
            best_proof_fold_per_lb.insert(
                lb,
                FoldSuffix {
                    total_bytes,
                    setup_envelope_ring_elements,
                    first_direct_setup_field_len,
                    first_fold_params,
                    folds,
                    terminal,
                },
            );
        }
    }

    let result = SuffixResult {
        best_fold_per_lb,
        best_proof_fold_per_lb,
    };
    memo.insert(memo_key, result.clone());
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::offloaded_witness_contracts;

    #[test]
    fn offloaded_contraction_accepts_exact_threefold_boundary() {
        assert!(offloaded_witness_contracts(300, 2, 0, 128, 100, 2).unwrap());
        assert!(!offloaded_witness_contracts(299, 2, 0, 128, 100, 2).unwrap());
    }

    #[test]
    fn offloaded_contraction_prices_changed_digit_basis() {
        assert!(offloaded_witness_contracts(900, 2, 0, 128, 100, 6).unwrap());
        assert!(!offloaded_witness_contracts(899, 2, 0, 128, 100, 6).unwrap());
    }

    #[test]
    fn offloaded_contraction_includes_full_field_setup_prefix() {
        assert!(offloaded_witness_contracts(100, 2, 100, 128, 1000, 4).unwrap());
        assert!(!offloaded_witness_contracts(100, 2, 90, 128, 1000, 4).unwrap());
    }
}
