use std::collections::{BTreeMap, HashMap};

use akita_field::AkitaError;
use akita_types::{
    active_setup_field_len, extension_opening_reduction_level_bytes, level_proof_bytes,
    padded_setup_prefix_len, terminal_response_bytes, FoldStep, LevelParams, OpeningClaimsLayout,
    PolynomialGroupLayout, RelationMatrixRowLayout, SetupContributionMode, TerminalResponseShape,
    TerminalWitnessPlan, SETUP_OFFLOAD_MIN_PREFIX_FIELD_LEN,
};

use crate::PlannerPolicy;

use super::{
    derive_candidate_level_params, suffix_opening_layout,
    terminal_witness_shape_for_opening_layout, MAX_RECURSION_DEPTH,
};

/// A fold-first suffix schedule.
///
/// The parent's proof-size formula needs the child's first fold params
/// (`first_fold_params`), so the suffix carries it directly instead of
/// re-reading `folds[0]`.
#[derive(Clone)]
pub(crate) struct FoldSuffix {
    pub(crate) total_bytes: usize,
    pub(crate) first_fold_params: LevelParams,
    pub(crate) folds: Vec<FoldStep>,
    pub(crate) terminal: TerminalWitnessPlan,
}

/// Best direct suffix at one DP state: witness length only. The terminal
/// `TerminalWitnessPlan` is materialized at stitch time from the predecessor fold's
/// committed `LevelParams`.
#[derive(Clone, Copy)]
pub(crate) struct DirectSuffix {
    pub(crate) input_witness_len: usize,
}

/// Result of the suffix DP at one state. Both shape options are reported
/// because the parent's proof-size formula depends on the child's first
/// step:
///
/// - `best_direct` — best no-outgoing-prefix terminal schedule whose first
///   next operation is terminal direct. Omitted when
///   `incoming_setup_prefix` is present, because a direct child means the
///   parent did not offload a new setup prefix into that child.
/// - `best_fold_per_lb` — best fold-first schedule per first-fold
///   `log_basis`, consuming `incoming_setup_prefix` when one is present.
#[derive(Clone)]
pub(crate) struct SuffixResult {
    pub(crate) best_direct: Option<DirectSuffix>,
    pub(crate) best_fold_per_lb: BTreeMap<u32, FoldSuffix>,
}

impl SuffixResult {
    pub(crate) fn is_empty(&self) -> bool {
        self.best_direct.is_none() && self.best_fold_per_lb.is_empty()
    }
}

fn make_terminal_direct_step(
    input_witness_len: usize,
    terminal_lp: &LevelParams,
    field_bits: u32,
    num_polynomials: usize,
    opening_layout: Option<&OpeningClaimsLayout>,
) -> Result<TerminalWitnessPlan, AkitaError> {
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
    let witness_shape = match opening_layout {
        Some(layout) => terminal_witness_shape_for_opening_layout(terminal_lp, field_bits, layout)?,
        None => TerminalResponseShape::from_groups(
            terminal_lp,
            field_bits,
            [(
                terminal_lp as &dyn akita_types::LevelParamsLike,
                num_polynomials,
                num_polynomials,
                1,
            )],
        )?,
    };
    let terminal_bytes = terminal_response_bytes(field_bits, &witness_shape);
    Ok(TerminalWitnessPlan {
        input_witness_len,
        witness_shape,
        terminal_bytes,
    })
}

/// Like [`terminal_direct_suffix_cost`], but returns `None` when the fold at
/// `terminal_fold_level` is multi-chunk. The suffix DP uses this to skip the
/// fold-then-direct branch without aborting fold-then-fold exploration.
pub(super) fn try_terminal_direct_suffix_cost(
    input_witness_len: usize,
    terminal_lp: &LevelParams,
    field_bits: u32,
    key: PolynomialGroupLayout,
    terminal_fold_level: usize,
    opening_layout: Option<&OpeningClaimsLayout>,
) -> Result<Option<(TerminalWitnessPlan, usize)>, AkitaError> {
    if terminal_lp.witness_chunk.num_chunks > 1 {
        return Ok(None);
    }
    let (direct, terminal_bytes) = terminal_direct_suffix_cost(
        input_witness_len,
        terminal_lp,
        field_bits,
        key,
        terminal_fold_level,
        opening_layout,
    )?;
    Ok(Some((direct, terminal_bytes)))
}

pub(crate) fn terminal_direct_suffix_cost(
    input_witness_len: usize,
    terminal_lp: &LevelParams,
    field_bits: u32,
    key: PolynomialGroupLayout,
    terminal_fold_level: usize,
    opening_layout: Option<&OpeningClaimsLayout>,
) -> Result<(TerminalWitnessPlan, usize), AkitaError> {
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
    let terminal_bytes = direct.terminal_bytes;
    Ok((direct, terminal_bytes))
}

pub(crate) type ScheduleMemo = HashMap<(usize, usize, usize, u32, usize), SuffixResult>;

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
    pub(crate) current_witness_len_terminal: usize,
    pub(crate) current_lb: u32,
    pub(crate) incoming_setup_prefix: Option<usize>,
}

impl SuffixState {
    fn memo_key(self) -> (usize, usize, usize, u32, usize) {
        (
            self.level,
            self.current_witness_len,
            self.current_witness_len_terminal,
            self.current_lb,
            self.incoming_setup_prefix.unwrap_or(0),
        )
    }
}

/// Shared inputs for root-level `LevelParams` candidates.
/// Suffix DP for the optimal recursive schedule at
/// `(level, current_witness_len, current_witness_len_terminal, current_lb)`.
///
/// Two witness lengths are carried because the shape leaving a fold
/// depends on its successor: `current_witness_len` is the `Intermediate` shape
/// (used if level `L` folds again) and `current_witness_len_terminal` is the
/// `Terminal` shape (used if level `L` sends the witness directly — drops
/// the D-block and zk D-blinding, so it is `<= current_witness_len`).
///
/// At each state: `best_direct` ships the witness directly without consuming
/// or forwarding an incoming prefix; `best_fold` keeps one fold candidate per
/// `log_basis` (from [`derive_candidate_level_params`]) and consumes
/// `incoming_setup_prefix` when present. Fold-again edges always plan the child
/// without an incoming prefix first, then mark the current fold `Recursive` only
/// when the prefix threshold is met, the child suffix is nonterminal, and a
/// compatible prefixed child exists.
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
        current_witness_len_terminal,
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

    let best_direct = if incoming_setup_prefix.is_some() {
        None
    } else if derive_candidate_level_params(
        policy,
        ring_challenge_cfg,
        current_witness_len,
        current_lb,
        level,
        None,
        requested_fold_shape,
    )?
    .is_some()
    {
        Some(DirectSuffix {
            input_witness_len: current_witness_len_terminal,
        })
    } else {
        None
    };

    if depth > MAX_RECURSION_DEPTH {
        let result = SuffixResult {
            best_direct,
            best_fold_per_lb: BTreeMap::new(),
        };
        memo.insert(memo_key, result.clone());
        return Ok(result);
    }

    let mut best_fold_per_lb: BTreeMap<u32, FoldSuffix> = BTreeMap::new();
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
        let Some((candidate_params, next_witness_len, next_witness_len_terminal)) =
            derive_candidate_level_params(
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

        let mut best_for_this_lb: Option<(usize, Vec<FoldStep>, TerminalWitnessPlan)> = None;
        let try_update =
            |total: usize,
             folds: Vec<FoldStep>,
             terminal: TerminalWitnessPlan,
             slot: &mut Option<(usize, Vec<FoldStep>, TerminalWitnessPlan)>| {
                if slot.as_ref().map(|(c, _, _)| total < *c).unwrap_or(true) {
                    *slot = Some((total, folds, terminal));
                }
            };

        let current_opening_layout =
            suffix_opening_layout(current_witness_len, incoming_setup_prefix)?;
        let natural_len = active_setup_field_len(&candidate_params, &current_opening_layout)?;
        let n_prefix = padded_setup_prefix_len(natural_len);
        let recursion_threshold_met = policy.recursive_setup_planning
            && level <= 1
            && n_prefix > SETUP_OFFLOAD_MIN_PREFIX_FIELD_LEN;

        let child_suffix_no_prefix = derive_optimal_suffix_schedule(
            ctx,
            memo,
            SuffixState {
                level: level + 1,
                current_witness_len: next_witness_len,
                current_witness_len_terminal: next_witness_len_terminal,
                current_lb: lb,
                incoming_setup_prefix: None,
            },
            depth + 1,
        )?;

        // Branch A: suffix is a Direct at level+1. Scalar terminals only: grouped
        // folds must continue with another fold, and an incoming setup prefix
        // makes a terminal direct suffix infeasible.
        if !candidate_params.has_precommitted_groups() {
            if let Some(direct_suffix) = child_suffix_no_prefix.best_direct {
                let field_bits = policy.decomposition.field_bits();
                let terminal_opening_layout = incoming_setup_prefix
                    .map(|_| suffix_opening_layout(current_witness_len, incoming_setup_prefix))
                    .transpose()?;
                if let Some((direct_step, suffix_cost)) = try_terminal_direct_suffix_cost(
                    direct_suffix.input_witness_len,
                    &candidate_params,
                    field_bits,
                    key,
                    level,
                    terminal_opening_layout.as_ref(),
                )? {
                    let level_proof_size = level_proof_bytes(
                        field_bits,
                        field_bits * policy.chal_ext_degree as u32,
                        &candidate_params,
                        None,
                        next_witness_len_terminal,
                        RelationMatrixRowLayout::WithoutCommitmentBlocks,
                        None,
                    )? + eor_bytes;
                    let total = level_proof_size + suffix_cost;
                    let folds = vec![FoldStep {
                        params: candidate_params.clone(),
                        input_witness_len: current_witness_len,
                        output_witness_len: next_witness_len_terminal,
                        level_bytes: level_proof_size,
                    }];
                    try_update(total, folds, direct_step, &mut best_for_this_lb);
                }
            }
        }
        // Branch B: suffix is a Fold at level+1. Plan the child without an
        // incoming prefix first, then classify the edge from the child's
        // topology: terminal children stay direct; nonterminal children recurse
        // only when the prefix threshold is met and a compatible prefixed child
        // exists.
        for suffix_fold in child_suffix_no_prefix.best_fold_per_lb.values() {
            let child_is_terminal = suffix_fold.folds.len() == 1;
            let (fold_mode, suffix_fold) = if child_is_terminal {
                (SetupContributionMode::Direct, suffix_fold.clone())
            } else if recursion_threshold_met {
                let prefixed_child_suffix = derive_optimal_suffix_schedule(
                    ctx,
                    memo,
                    SuffixState {
                        level: level + 1,
                        current_witness_len: next_witness_len,
                        current_witness_len_terminal: next_witness_len_terminal,
                        current_lb: lb,
                        incoming_setup_prefix: Some(natural_len),
                    },
                    depth + 1,
                )?;
                let child_lb = suffix_fold.first_fold_params.log_basis_open;
                let Some(prefixed_suffix_fold) =
                    prefixed_child_suffix.best_fold_per_lb.get(&child_lb)
                else {
                    continue;
                };
                if prefixed_suffix_fold.folds.len() == 1 {
                    continue;
                }
                (
                    SetupContributionMode::Recursive,
                    prefixed_suffix_fold.clone(),
                )
            } else {
                (SetupContributionMode::Direct, suffix_fold.clone())
            };

            let mut fold_candidate_params = candidate_params.clone();
            fold_candidate_params.setup_contribution_mode = fold_mode;
            let level_proof_size = level_proof_bytes(
                policy.decomposition.field_bits(),
                policy.decomposition.field_bits() * policy.chal_ext_degree as u32,
                &fold_candidate_params,
                Some(&suffix_fold.first_fold_params),
                next_witness_len,
                RelationMatrixRowLayout::WithDBlock,
                Some(if child_is_terminal {
                    akita_types::NextWitnessBindingPolicy::TerminalInnerState
                } else {
                    akita_types::NextWitnessBindingPolicy::OuterCommitment
                }),
            )? + eor_bytes;
            let total = level_proof_size + suffix_fold.total_bytes;
            let mut folds = Vec::with_capacity(1 + suffix_fold.folds.len());
            folds.push(FoldStep {
                params: fold_candidate_params,
                input_witness_len: current_witness_len,
                output_witness_len: next_witness_len,
                level_bytes: level_proof_size,
            });
            folds.extend(suffix_fold.folds.iter().cloned());
            try_update(
                total,
                folds,
                suffix_fold.terminal.clone(),
                &mut best_for_this_lb,
            );
        }

        if let Some((total_bytes, folds, terminal)) = best_for_this_lb {
            let first_fold_params =
                folds
                    .first()
                    .map(|fold| fold.params.clone())
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup(
                            "fold suffix missing first fold params".to_string(),
                        )
                    })?;
            best_fold_per_lb.insert(
                lb,
                FoldSuffix {
                    total_bytes,
                    first_fold_params,
                    folds,
                    terminal,
                },
            );
        }
    }

    let result = SuffixResult {
        best_direct,
        best_fold_per_lb,
    };
    memo.insert(memo_key, result.clone());
    Ok(result)
}
