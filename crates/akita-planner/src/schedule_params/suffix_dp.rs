use std::collections::{BTreeMap, HashMap};

use akita_field::AkitaError;
use akita_types::{
    active_setup_field_len, extension_opening_reduction_level_bytes, level_proof_bytes,
    padded_setup_prefix_len, terminal_response_bytes, CommittedGroupParams, OpeningClaimsLayout,
    PolynomialGroupLayout, TerminalResponseShape, SETUP_OFFLOAD_MIN_PREFIX_FIELD_LEN,
};

use crate::PlannerPolicy;

use super::{
    derive_candidate_level_params, suffix_opening_layout, CandidateFoldStep,
    CandidateTerminalResponse, MAX_RECURSION_DEPTH,
};

/// A fold-first suffix schedule.
///
/// The parent's proof-size formula needs the child's first fold params
/// (`first_fold_params`), so the suffix carries it directly instead of
/// re-reading `folds[0]`.
#[derive(Clone)]
pub(crate) struct FoldSuffix {
    pub(crate) total_bytes: usize,
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
    let (terminal_params, honest_response_linf_cap) =
        akita_types::TerminalCommittedGroupParams::try_from_expanded_group(terminal_lp.clone())?;
    let witness_shape = TerminalResponseShape::derive(&terminal_params, honest_response_linf_cap)?;
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
        // fixed inner matrix cannot admit the unsnapped terminal response is
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

type CandidateSuffixChoice = (
    usize,
    usize,
    Vec<CandidateFoldStep>,
    CandidateTerminalResponse,
);

fn update_best_suffix_choices(
    first_direct_setup_field_len: usize,
    total_bytes: usize,
    folds: Vec<CandidateFoldStep>,
    terminal: CandidateTerminalResponse,
    best_by_setup: &mut Option<CandidateSuffixChoice>,
    best_by_proof: &mut Option<CandidateSuffixChoice>,
) {
    let candidate = (first_direct_setup_field_len, total_bytes, folds, terminal);
    if best_by_setup
        .as_ref()
        .map(|(best_setup, best_total, _, _)| {
            (first_direct_setup_field_len, total_bytes) < (*best_setup, *best_total)
        })
        .unwrap_or(true)
    {
        *best_by_setup = Some(candidate.clone());
    }
    if best_by_proof
        .as_ref()
        .map(|(_, best_total, _, _)| total_bytes < *best_total)
        .unwrap_or(true)
    {
        *best_by_proof = Some(candidate);
    }
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
        let n_prefix = padded_setup_prefix_len(natural_len);
        let recursion_threshold_met = policy.recursive_setup_planning
            && level <= 1
            && n_prefix > SETUP_OFFLOAD_MIN_PREFIX_FIELD_LEN;

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
                let total = level_proof_size + suffix_cost;
                direct_step.estimated_direct_payload_bytes = level_proof_size;
                update_best_suffix_choices(
                    natural_len,
                    total,
                    Vec::new(),
                    direct_step,
                    &mut best_for_this_lb,
                    &mut best_proof_for_this_lb,
                );
            }
        }

        let child_suffix = derive_optimal_suffix_schedule(
            ctx,
            memo,
            SuffixState {
                level: level + 1,
                current_witness_len: next_witness_len,
                current_lb: lb,
                incoming_setup_prefix: recursion_threshold_met.then_some(natural_len),
            },
            depth + 1,
        )?;
        // Branch B: suffix is a Fold at level+1. Recursive setup edges inherit
        // the child's first direct setup size; direct edges fix it here and use
        // the proof-optimal child suffix.
        let objective_child_folds = if recursion_threshold_met {
            &child_suffix.best_fold_per_lb
        } else {
            &child_suffix.best_proof_fold_per_lb
        };
        for suffix_fold in objective_child_folds.values() {
            let child_is_terminal = suffix_fold.folds.is_empty();
            assert!(
                !(recursion_threshold_met && child_is_terminal),
                "recursive setup planning produced a terminal child suffix at level {}",
                level + 1
            );
            if recursion_threshold_met && suffix_fold.folds.len() == 1 {
                continue;
            }
            let suffix_fold = suffix_fold.clone();
            let fold_candidate_params = candidate_params.clone();
            let level_proof_size = level_proof_bytes(
                policy.decomposition.field_bits(),
                policy.decomposition.field_bits() * policy.chal_ext_degree as u32,
                &fold_candidate_params,
                suffix_fold.first_fold_params.as_ref(),
                next_witness_len,
                Some(if child_is_terminal {
                    akita_types::NextWitnessBindingPolicy::TerminalInnerState
                } else {
                    akita_types::NextWitnessBindingPolicy::OuterCommitment
                }),
            )? + eor_bytes;
            let total = level_proof_size + suffix_fold.total_bytes;
            let mut folds = Vec::with_capacity(1 + suffix_fold.folds.len());
            folds.push(CandidateFoldStep {
                params: fold_candidate_params,
                input_witness_len: current_witness_len,
                output_witness_len: next_witness_len,
                estimated_direct_payload_bytes: level_proof_size,
            });
            folds.extend(suffix_fold.folds.iter().cloned());
            let first_direct_setup_field_len = if recursion_threshold_met {
                suffix_fold.first_direct_setup_field_len
            } else {
                natural_len
            };
            let candidate = (
                first_direct_setup_field_len,
                total,
                folds,
                suffix_fold.terminal.clone(),
            );
            if best_for_this_lb
                .as_ref()
                .map(|(best_setup, best_total, _, _)| {
                    (first_direct_setup_field_len, total) < (*best_setup, *best_total)
                })
                .unwrap_or(true)
            {
                best_for_this_lb = Some(candidate);
            }
        }

        for suffix_fold in child_suffix.best_proof_fold_per_lb.values() {
            let child_is_terminal = suffix_fold.folds.is_empty();
            assert!(
                !(recursion_threshold_met && child_is_terminal),
                "recursive setup planning produced a terminal child suffix at level {}",
                level + 1
            );
            if recursion_threshold_met && suffix_fold.folds.len() == 1 {
                continue;
            }
            let suffix_fold = suffix_fold.clone();
            let fold_candidate_params = candidate_params.clone();
            let level_proof_size = level_proof_bytes(
                policy.decomposition.field_bits(),
                policy.decomposition.field_bits() * policy.chal_ext_degree as u32,
                &fold_candidate_params,
                suffix_fold.first_fold_params.as_ref(),
                next_witness_len,
                Some(if child_is_terminal {
                    akita_types::NextWitnessBindingPolicy::TerminalInnerState
                } else {
                    akita_types::NextWitnessBindingPolicy::OuterCommitment
                }),
            )? + eor_bytes;
            let total = level_proof_size + suffix_fold.total_bytes;
            if best_proof_for_this_lb
                .as_ref()
                .map(|(_, best_total, _, _)| total < *best_total)
                .unwrap_or(true)
            {
                let mut folds = Vec::with_capacity(1 + suffix_fold.folds.len());
                folds.push(CandidateFoldStep {
                    params: fold_candidate_params,
                    input_witness_len: current_witness_len,
                    output_witness_len: next_witness_len,
                    estimated_direct_payload_bytes: level_proof_size,
                });
                folds.extend(suffix_fold.folds.iter().cloned());
                let first_direct_setup_field_len = if recursion_threshold_met {
                    suffix_fold.first_direct_setup_field_len
                } else {
                    natural_len
                };
                best_proof_for_this_lb = Some((
                    first_direct_setup_field_len,
                    total,
                    folds,
                    suffix_fold.terminal.clone(),
                ));
            }
        }

        if let Some((first_direct_setup_field_len, total_bytes, folds, terminal)) = best_for_this_lb
        {
            let first_fold_params = folds.first().map(|fold| fold.params.clone());
            best_fold_per_lb.insert(
                lb,
                FoldSuffix {
                    total_bytes,
                    first_direct_setup_field_len,
                    first_fold_params,
                    folds,
                    terminal,
                },
            );
        }
        if let Some((first_direct_setup_field_len, total_bytes, folds, terminal)) =
            best_proof_for_this_lb
        {
            let first_fold_params = folds.first().map(|fold| fold.params.clone());
            best_proof_fold_per_lb.insert(
                lb,
                FoldSuffix {
                    total_bytes,
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
