//! Schedule planner that finds the global minimum proof size.
//!
//! Public entry: [`find_schedule`]. The search is `Cfg`-free: every
//! per-preset input is carried by the plain-value [`PlannerPolicy`] plus
//! the `ring_challenge_config` / `fold_challenge_shape_at_level` closures,
//! exactly the shape `crate::schedule_from_entry` already consumes. This keeps the
//! DP a pure function of `(policy, key)` so `akita-config` can call it
//! directly on a schedule-table miss without a dependency cycle.

use std::collections::{BTreeMap, HashMap};

use akita_challenges::TensorChallengeShape;
use akita_field::AkitaError;
use akita_types::layout::digit_math::optimal_m_r_split;
use akita_types::sis::{
    choose_op_norm_rejection_for_a_role, decomposed_s_block_ring_count, decomposed_t_ring_count,
    decomposed_w_ring_count, min_secure_rank, num_digits_open, num_digits_s_commit,
    rounded_up_collision_norm_t, rounded_up_collision_norm_tiered_commitment,
    rounded_up_collision_norm_w, AjtaiKeyParams, FoldWitnessLinfCapConfig, FoldWitnessNorms,
};
use akita_types::{
    direct_witness_bytes, extension_opening_reduction_proof_bytes, level_proof_bytes,
    segment_typed_witness_shape, w_ring_element_count_for_chunks, AkitaScheduleInputs,
    AkitaScheduleLookupKey, ChunkedWitnessCfg, CleartextWitnessShape, DecompositionParams,
    DirectStep, FoldStep, LevelParams, MRowLayout, Schedule, Step,
};

use crate::PlannerPolicy;

/// Validate the policy's multi-chunk witness settings at a planner entry point.
///
/// Layout-only rules live on [`ChunkedWitnessCfg::validate`]; the tiered guard
/// and the recursion-depth bound (which needs the planner-private
/// [`MAX_RECURSION_DEPTH`]) are enforced here so `akita-types` stays free of
/// planner internals.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidSetup`] for an invalid `ChunkedWitnessCfg`, a
/// tiered + multi-chunk combination, or `num_activated_levels` beyond the
/// planner recursion cap. Verifier-reachable: never panics.
pub(crate) fn validate_policy_witness_chunk(policy: &PlannerPolicy) -> Result<(), AkitaError> {
    let mc = policy.witness_chunk;
    mc.validate()?;
    if policy.tiered && mc.uses_multi_chunk() {
        return Err(AkitaError::InvalidSetup(
            "multi-chunk witness layout is unsupported with tiered commitments".to_string(),
        ));
    }
    if mc.num_activated_levels > MAX_RECURSION_DEPTH {
        return Err(AkitaError::InvalidSetup(format!(
            "num_activated_levels={} exceeds the planner recursion cap {MAX_RECURSION_DEPTH}",
            mc.num_activated_levels
        )));
    }
    Ok(())
}

/// Stage-1 sparse-challenge closure shared by the planner entry points.
pub(crate) type RingChallengeConfigFn<'a> =
    &'a dyn Fn(usize) -> Result<akita_challenges::SparseChallengeConfig, AkitaError>;

/// Stage-1 fold-round challenge-shape closure (`level 0` root shape).
type FoldShapeFn<'a> = &'a dyn Fn(AkitaScheduleInputs) -> TensorChallengeShape;

// Suffix-DP depth cap. Schedules in our working parameter range never need
// more than this many recursive fold levels; deeper search only blows up
// memo state without changing emitted tables.
const MAX_RECURSION_DEPTH: usize = 12;

/// Largest tiered split factor `f` the planner will consider. `B'` matrix size
/// is monotone non-increasing in `f`, so scanning `f = 2..=MAX` ascending and
/// taking the first divisor of the `B` width whose size fits under `A` yields the
/// smallest feasible split. Kept small to bound the added relation rows / witness
/// / `F` size (and the verifier-reachable DP work).
const MAX_TIERED_SPLIT_FACTOR: usize = 16;

/// Tiered second matrix `F`. When the first-tier `B` matrix size exceeds the
/// inner `A` matrix size, shrink `B` to a `B'` reused across `f` equal flat
/// column-slices of `t̂`, and size the second-tier `F` that commits the decomposed
/// concatenation `u_1 ‖ … ‖ u_f`.
///
/// The split is a plain contiguous partition of the `B` width into `f` equal
/// slices (`f | width`). No column structure is preserved and `f` need not be a
/// power of two — the relation only enforces `B'·t_i = u_i` and
/// `F(u_1 ‖ … ‖ u_f) = u_final`, and every prover/verifier path slices `t̂` flatly.
/// `F` is sized at its own collision bucket ([`rounded_up_collision_norm_tiered_commitment`]).
///
/// `B'` matrix size is monotonically non-increasing in `f`, so the **smallest**
/// divisor `f` in `2..=MAX_TIERED_SPLIT_FACTOR` whose size drops to
/// `<= a_matrix_size` is optimal (smallest `F`, fewest relation rows, least
/// witness).
///
/// Returns `(tier_split, b_key, f_key)`: `(1, b_key.clone(), None)` when no split
/// is needed (`B` already fits under `A`) or none is feasible; `(f, B', Some(F))`
/// otherwise.
///
/// Verifier-reachable through the runtime DP fallback: never panics — every
/// product is checked and every emitted key passes [`AjtaiKeyParams::try_new`].
/// This is the DP's source of truth for the `B'`/`F` split; the shipped tiered
/// table stores the resulting layout directly, so expansion never re-runs it.
fn multi_tiered_keys(
    a_matrix_size: usize,
    b_key: &AjtaiKeyParams,
    delta_open: usize,
    log_basis: u32,
    ring_d: usize,
) -> Result<(usize, AjtaiKeyParams, Option<AjtaiKeyParams>), AkitaError> {
    let Some(b_matrix_size) = b_key.row_len().checked_mul(b_key.col_len()) else {
        return Ok((1, b_key.clone(), None));
    };
    if b_matrix_size <= a_matrix_size {
        // First-tier `B` already fits under `A`; no tiering needed.
        return Ok((1, b_key.clone(), None));
    }

    // `F` commits balanced base-`2^log_basis` digits of `u_concat`.
    let Some(norm_f) =
        rounded_up_collision_norm_tiered_commitment(b_key.sis_family(), ring_d, log_basis)
    else {
        return Ok((1, b_key.clone(), None));
    };

    // Smallest f in 2..=MAX dividing the B width whose shrunk B' matrix size fits
    // under A. Size is monotone non-increasing in f, so the first hit is optimal.
    for f in 2..=MAX_TIERED_SPLIT_FACTOR {
        if !b_key.col_len().is_multiple_of(f) {
            continue;
        }
        let shrunk_width = b_key.col_len() / f; // exact: f divides the B width
        let Some(n_b_small) = min_secure_rank(
            b_key.sis_family(),
            ring_d as u32,
            b_key.collision_l2_sq(),
            shrunk_width as u64,
        ) else {
            continue;
        };
        let Some(b_small_size) = n_b_small.checked_mul(shrunk_width) else {
            continue;
        };
        if b_small_size > a_matrix_size {
            continue;
        }
        // `F` commits `decompose(u_1 ‖ … ‖ u_f)`: `f · n_b' · δ_open` digit columns.
        let Some(width_f) = f
            .checked_mul(n_b_small)
            .and_then(|w| w.checked_mul(delta_open))
        else {
            continue;
        };
        let Some(n_f) = min_secure_rank(b_key.sis_family(), ring_d as u32, norm_f, width_f as u64)
        else {
            continue;
        };
        let f_key = AjtaiKeyParams::try_new(b_key.sis_family(), n_f, width_f, norm_f, ring_d)?;
        let tiered_b_key = AjtaiKeyParams::try_new(
            b_key.sis_family(),
            n_b_small,
            shrunk_width,
            b_key.collision_l2_sq(),
            ring_d,
        )?;
        return Ok((f, tiered_b_key, Some(f_key)));
    }

    // No split in 2..=MAX brings B' under A; stay single-tier.
    Ok((1, b_key.clone(), None))
}

/// Compute parameters that generate the smallest witness for the next
/// fold level. Note that this is not the optimum case: in the optimum
/// case (similar to `find_schedule`), we should check that current proof
/// size + suffix cost is the smallest. However, as time blows up, we
/// don't do that here.
fn derive_candidate_level_params(
    policy: &PlannerPolicy,
    ring_challenge_config: RingChallengeConfigFn<'_>,
    current_witness_len: usize,
    log_basis: u32,
    fold_level: usize,
) -> Result<Option<(LevelParams, usize, usize)>, AkitaError> {
    // Chunk count of the witness this level commits/produces (sized below as
    // `next_witness_len`). Equal for the metadata field and the width pricing so
    // a future verifier recomputing the size from `witness_chunk` agrees.
    let num_chunks = policy.chunks_at_level(fold_level);
    let Ok(ring_challenge_cfg) = ring_challenge_config(policy.ring_dimension) else {
        return Ok(None);
    };
    if !current_witness_len.is_multiple_of(policy.ring_dimension) {
        return Ok(None);
    }
    let num_ring_elems = current_witness_len / policy.ring_dimension;
    let reduced_vars = num_ring_elems.next_power_of_two().max(1).trailing_zeros() as usize;

    if reduced_vars <= 2 || reduced_vars >= 53 {
        return Err(AkitaError::InvalidSetup(format!(
            "recursive fold candidate reduced_vars={reduced_vars} is outside \
             the optimizable range [3, 52]"
        )));
    }

    let mut best: Option<(LevelParams, usize, usize)> = None;
    for r in (1..reduced_vars).rev() {
        let Some(num_blocks) = 1usize.checked_shl(r as u32) else {
            continue;
        };
        // Multi-chunk levels require an equal block window per chunk; skip splits
        // that do not divide evenly (not fixed up later).
        if num_chunks > 1 && !num_blocks.is_multiple_of(num_chunks) {
            continue;
        }
        let block_len = num_ring_elems.div_ceil(num_blocks);

        // Recursive levels commit a dense balanced-digit witness (`is_root =
        // false`, flat fold). Compose the three SIS-secure keys from the
        // `akita_types::sis` primitives: norm -> width -> rank -> key.
        let family = policy.sis_family;
        let d = policy.ring_dimension;
        let decomp = DecompositionParams {
            log_basis,
            ..policy.decomposition
        };
        let delta_commit = num_digits_s_commit(decomp, false);
        let delta_open = num_digits_open(decomp);
        let Some(width_s) = decomposed_s_block_ring_count(block_len, delta_commit) else {
            continue;
        };
        let Some((op_norm_rejection, norm_s, n_a)) = choose_op_norm_rejection_for_a_role(
            family,
            d,
            decomp,
            &ring_challenge_cfg,
            TensorChallengeShape::Flat,
            false,
            policy.onehot_chunk_size,
            policy.ring_subfield_norm_bound,
            r,
            1,
            width_s as u64,
        ) else {
            continue;
        };
        let a_key = AjtaiKeyParams::try_new(family, n_a, width_s, norm_s, d)?;
        let Some(norm_t) = rounded_up_collision_norm_t(family, d, log_basis) else {
            continue;
        };
        let Some(width_t) = decomposed_t_ring_count(n_a, delta_open, num_blocks, 1) else {
            continue;
        };
        let Some(n_b) = min_secure_rank(family, d as u32, norm_t, width_t as u64) else {
            continue;
        };
        let b_key = AjtaiKeyParams::try_new(family, n_b, width_t, norm_t, d)?;
        let Some(norm_w) = rounded_up_collision_norm_w(family, d, log_basis) else {
            continue;
        };
        let Some(width_w) = decomposed_w_ring_count(delta_open, num_blocks, 1) else {
            continue;
        };
        let Some(n_d) = min_secure_rank(family, d as u32, norm_w, width_w as u64) else {
            continue;
        };
        let d_key = AjtaiKeyParams::try_new(family, n_d, width_w, norm_w, d)?;

        let (tier_split, b_key, f_key) = if policy.tiered {
            let Some(a_matrix_size) = a_key.row_len().checked_mul(a_key.col_len()) else {
                continue;
            };
            multi_tiered_keys(a_matrix_size, &b_key, delta_open, log_basis, d)?
        } else {
            (1, b_key, None)
        };

        let Ok(candidate_params) = LevelParams {
            ring_dimension: policy.ring_dimension,
            log_basis,
            a_key,
            b_key,
            d_key,
            num_blocks,
            block_len,
            m_vars: reduced_vars - r,
            r_vars: r,
            stage1_config: ring_challenge_cfg.clone(),
            op_norm_rejection,
            fold_challenge_shape: TensorChallengeShape::Flat,
            num_digits_commit: delta_commit,
            num_digits_open: delta_open,
            // Recursive levels commit dense balanced-digit witnesses.
            onehot_chunk_size: 0,
            tier_split,
            f_key,
            fold_linf_cap_config: FoldWitnessLinfCapConfig::worst_case_beta_only(),
            num_digits_fold_one: 1,
            field_bits_hint: 0,
            cached_num_digits_fold_claims: 0,
            cached_num_digits_fold_value: 1,
            witness_chunk: policy.witness_chunk_for_level(fold_level),
            precommitted_groups: Vec::new(),
        }
        .with_fold_linf_cap_config(policy.decomposition.field_bits(), 1) else {
            continue;
        };

        let next_witness_len = w_ring_element_count_for_chunks(
            policy.decomposition.field_bits(),
            &candidate_params,
            1,
            MRowLayout::WithDBlock,
            num_chunks,
        )?
        .checked_mul(policy.ring_dimension)
        .ok_or_else(|| AkitaError::InvalidSetup("recursive witness length overflow".into()))?;
        let next_witness_len_terminal = w_ring_element_count_for_chunks(
            policy.decomposition.field_bits(),
            &candidate_params,
            1,
            MRowLayout::WithoutDBlock,
            num_chunks,
        )?
        .checked_mul(policy.ring_dimension)
        .ok_or_else(|| AkitaError::InvalidSetup("recursive witness length overflow".into()))?;

        if best.as_ref().is_none_or(|(_, c, _)| next_witness_len < *c) {
            best = Some((
                candidate_params,
                next_witness_len,
                next_witness_len_terminal,
            ));
        }
    }

    let Some((candidate_params, next_witness_len, next_witness_len_terminal)) = best else {
        return Ok(None);
    };

    if next_witness_len >= current_witness_len {
        return Ok(None);
    }

    Ok(Some((
        candidate_params,
        next_witness_len,
        next_witness_len_terminal,
    )))
}

fn padded_boolean_vars(len: usize) -> Result<usize, AkitaError> {
    let padded = len
        .checked_next_power_of_two()
        .ok_or_else(|| AkitaError::InvalidSetup("opening witness length overflow".to_string()))?;
    Ok(padded.trailing_zeros() as usize)
}

fn extension_opening_reduction_level_bytes(
    policy: &PlannerPolicy,
    key: AkitaScheduleLookupKey,
    fold_level: usize,
    current_w_len: usize,
) -> Result<usize, AkitaError> {
    let width = policy.claim_ext_degree;
    if width <= 1 {
        return Ok(0);
    }
    let (partials, opening_vars) = if fold_level == 0 {
        (width.saturating_mul(key.num_polynomials), key.num_vars)
    } else {
        (width, padded_boolean_vars(current_w_len)?)
    };
    extension_opening_reduction_proof_bytes(
        policy.decomposition.field_bits() * policy.chal_ext_degree as u32,
        partials,
        opening_vars,
        width,
    )
}

/// A `Step::Fold`-first suffix schedule.
///
/// The parent's proof-size formula needs the child's first fold params
/// (`first_fold_params`), so the suffix carries it directly instead of
/// re-matching `steps[0]`.
#[derive(Clone)]
struct FoldSuffix {
    total_bytes: usize,
    first_fold_params: LevelParams,
    steps: Vec<Step>,
}

/// Best direct suffix at one DP state: witness length only. The terminal
/// `DirectStep` is materialized at stitch time from the predecessor fold's
/// committed `LevelParams`.
#[derive(Clone, Copy)]
struct DirectSuffix {
    current_w_len: usize,
}

/// Result of the suffix DP at one state. Both shape options are reported
/// because the parent's proof-size formula depends on the child's first
/// step:
///
/// - `best_direct` — best schedule whose first step is a `Step::Direct`
///   (parent scores under `MRowLayout::WithoutDBlock`). `None` when infeasible.
/// - `best_fold_per_lb` — best `Step::Fold`-first schedule per first-fold
///   `log_basis`.
#[derive(Clone)]
struct SuffixResult {
    best_direct: Option<DirectSuffix>,
    best_fold_per_lb: BTreeMap<u32, FoldSuffix>,
}

impl SuffixResult {
    fn is_empty(&self) -> bool {
        self.best_direct.is_none() && self.best_fold_per_lb.is_empty()
    }
}

fn make_terminal_direct_step(
    current_w_len: usize,
    terminal_lp: &LevelParams,
    field_bits: u32,
    num_polynomials: usize,
) -> Result<DirectStep, AkitaError> {
    // Chunked terminal: `ê`/`t̂` partitioned (totals unchanged), `ẑ` replicated
    // and shared `r`-tail priced with `num_chunks` commitments — modeled via the
    // segment/public-row multiplicity. `num_chunks == 1` is byte-identical to the
    // single-chunk shape, so non-chunked levels are unaffected. Must match the
    // table-walk terminal pricing in `resolve.rs`.
    let terminal_num_chunks = terminal_lp.witness_chunk.num_chunks.max(1);
    let witness_shape = segment_typed_witness_shape(
        terminal_lp,
        field_bits,
        num_polynomials,
        num_polynomials,
        terminal_num_chunks,
        terminal_num_chunks,
    )?;
    let direct_bytes = direct_witness_bytes(field_bits, &witness_shape);
    Ok(DirectStep {
        current_w_len,
        witness_shape,
        direct_bytes,
        params: None,
    })
}

fn terminal_direct_suffix_cost(
    current_w_len: usize,
    terminal_lp: &LevelParams,
    field_bits: u32,
    key: AkitaScheduleLookupKey,
    terminal_fold_level: usize,
) -> Result<(DirectStep, usize), AkitaError> {
    // Scalar same-point root fold: polynomial count at the root, 1 recursively.
    let num_polynomials = if terminal_fold_level == 0 {
        key.num_polynomials
    } else {
        1
    };
    let direct =
        make_terminal_direct_step(current_w_len, terminal_lp, field_bits, num_polynomials)?;
    let direct_bytes = direct.direct_bytes;
    Ok((direct, direct_bytes))
}

type ScheduleMemo = HashMap<(usize, usize, usize, u32), SuffixResult>;

/// DP-invariant inputs for the suffix search.
///
/// `policy`, `ring_challenge_config`, and `num_vars` are constant across the whole
/// recursion, so they are carried in one context value rather than as
/// per-call arguments (keeps the recursive signature small).
#[derive(Clone, Copy)]
struct SuffixCtx<'a> {
    policy: &'a PlannerPolicy,
    ring_challenge_config: RingChallengeConfigFn<'a>,
    num_vars: usize,
    key: AkitaScheduleLookupKey,
}

/// Suffix DP for the optimal recursive schedule at
/// `(level, current_witness_len, current_witness_len_terminal, current_lb)`.
///
/// Two witness lengths are carried because the shape leaving a fold
/// depends on its successor: `current_witness_len` is the `Intermediate` shape
/// (used if level `L` folds again) and `current_witness_len_terminal` is the
/// `Terminal` shape (used if level `L` sends the witness directly — drops
/// the D-block and zk D-blinding, so it is `<= current_witness_len`).
///
/// At each state: `best_direct` ships the witness directly (Terminal, no
/// SIS audit, always present); `best_fold` keeps one fold candidate per
/// `log_basis` (from [`derive_candidate_level_params`]).
fn derive_optimal_suffix_schedule(
    ctx: &SuffixCtx<'_>,
    memo: &mut ScheduleMemo,
    level: usize,
    current_witness_len: usize,
    current_witness_len_terminal: usize,
    current_lb: u32,
    depth: usize,
) -> Result<SuffixResult, AkitaError> {
    let SuffixCtx {
        policy,
        ring_challenge_config,
        num_vars,
        key,
    } = *ctx;
    let memo_key = (
        level,
        current_witness_len,
        current_witness_len_terminal,
        current_lb,
    );
    if depth <= MAX_RECURSION_DEPTH {
        if let Some(cached) = memo.get(&memo_key) {
            return Ok(cached.clone());
        }
    }

    let best_direct = if derive_candidate_level_params(
        policy,
        ring_challenge_config,
        current_witness_len,
        current_lb,
        level,
    )?
    .is_some()
    {
        Some(DirectSuffix {
            current_w_len: current_witness_len_terminal,
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
    let (min_log_basis, max_log_basis) = policy.basis_range;
    for lb in min_log_basis..=max_log_basis {
        if lb < current_lb {
            continue;
        }
        let Some((candidate_params, next_witness_len, next_witness_len_terminal)) =
            derive_candidate_level_params(
                policy,
                ring_challenge_config,
                current_witness_len,
                lb,
                level,
            )?
        else {
            continue;
        };

        let suffix = derive_optimal_suffix_schedule(
            ctx,
            memo,
            level + 1,
            next_witness_len,
            next_witness_len_terminal,
            lb,
            depth + 1,
        )?;
        let Ok(eor_bytes) = extension_opening_reduction_level_bytes(
            policy,
            AkitaScheduleLookupKey::singleton(num_vars),
            level,
            current_witness_len,
        ) else {
            continue;
        };

        let mut best_for_this_lb: Option<(usize, Vec<Step>)> = None;
        let try_update = |total: usize, steps: Vec<Step>, slot: &mut Option<(usize, Vec<Step>)>| {
            if slot.as_ref().map(|(c, _)| total < *c).unwrap_or(true) {
                *slot = Some((total, steps));
            }
        };

        // Branch A: suffix is a Direct at level+1.
        if let Some(direct_suffix) = suffix.best_direct {
            let field_bits = policy.decomposition.field_bits();
            let (direct_step, suffix_cost) = terminal_direct_suffix_cost(
                direct_suffix.current_w_len,
                &candidate_params,
                field_bits,
                key,
                level,
            )?;
            let level_proof_size = level_proof_bytes(
                field_bits,
                field_bits * policy.chal_ext_degree as u32,
                &candidate_params,
                None,
                next_witness_len_terminal,
                1,
                MRowLayout::WithoutDBlock,
            ) + eor_bytes;
            let total = level_proof_size + suffix_cost;
            let steps = vec![
                Step::Fold(FoldStep {
                    params: candidate_params.clone(),
                    current_w_len: current_witness_len,
                    next_w_len: next_witness_len_terminal,
                    level_bytes: level_proof_size,
                }),
                Step::Direct(direct_step),
            ];
            try_update(total, steps, &mut best_for_this_lb);
        }
        // Branch B: suffix is a Fold at level+1.
        for suffix_fold in suffix.best_fold_per_lb.values() {
            let level_proof_size = level_proof_bytes(
                policy.decomposition.field_bits(),
                policy.decomposition.field_bits() * policy.chal_ext_degree as u32,
                &candidate_params,
                Some(&suffix_fold.first_fold_params),
                next_witness_len,
                1,
                MRowLayout::WithDBlock,
            ) + eor_bytes;
            let total = level_proof_size + suffix_fold.total_bytes;
            let mut steps = Vec::with_capacity(1 + suffix_fold.steps.len());
            steps.push(Step::Fold(FoldStep {
                params: candidate_params.clone(),
                current_w_len: current_witness_len,
                next_w_len: next_witness_len,
                level_bytes: level_proof_size,
            }));
            steps.extend(suffix_fold.steps.iter().cloned());
            try_update(total, steps, &mut best_for_this_lb);
        }

        if let Some((total_bytes, steps)) = best_for_this_lb {
            best_fold_per_lb.insert(
                lb,
                FoldSuffix {
                    total_bytes,
                    first_fold_params: candidate_params,
                    steps,
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

/// Brute-forced root-direct commit `LevelParams` (optimal `(m, r)` split).
///
/// Root-direct schedules ship the cleartext witness on the wire, so they
/// don't run the relation fold (D unused). The planner brute-forces the
/// committed `(m, r, n_a, n_b, n_d)` here via the SIS-floor search and
/// stores it in `GeneratedDirectStep.commit`; the runtime reconstructs the
/// identical `LevelParams` with `GeneratedFoldStep::expand_to_level_params`.
///
/// This derives every value directly and assembles a single `LevelParams`:
///
/// - `a_collision` — the audited A-role SIS bucket (`2·β` base norm scaled
///   by the stage-1 infinity norm and the ring-subfield embedding norm).
/// - `bd_collision = 2^lb − 1` — the B/D digit-range bucket.
/// - `(m_vars, r_vars)` — `optimal_m_r_split` for a normal root, or `(0, 0)`
///   for a tiny root that fits inside one padded ring element.
/// - `(n_a, n_b, n_d)` — the tight SIS-floor ranks for the resulting
///   inner / outer / D-matrix widths.
///
/// - `(n_a, n_b, n_d)` — the tight SIS-floor ranks for the resulting
///   inner / outer / D-matrix widths, where the outer (B) and prover (D)
///   widths already carry the `num_claims` batch factor (the root commits
///   `num_claims` polynomials, so there is no separate per-claim-then-scale
///   step; `num_claims == 1` is the singleton root).
///
/// `fold_challenge_shape` is stamped onto the committed level (the level-0
/// shape; the `(m, r)` split itself is scored against the flat L1 mass).
///
/// Returns `Ok(None)` when any SIS-floor lookup or bound arithmetic rejects
/// the candidate (the uncommittable edge), matching the previous
/// `Result::ok()` fallback.
fn compute_root_direct_level_params(
    policy: &PlannerPolicy,
    ring_challenge_config: RingChallengeConfigFn<'_>,
    num_vars: usize,
    log_basis: u32,
    fold_challenge_shape: TensorChallengeShape,
    num_claims: usize,
) -> Result<Option<LevelParams>, AkitaError> {
    let ring_challenge_cfg = ring_challenge_config(policy.ring_dimension)?;
    let d = policy.ring_dimension;
    let sis_family = policy.sis_family;
    let decomp = policy.decomposition;
    let alpha = (d as u32).trailing_zeros() as usize;

    let level_decomp = DecompositionParams {
        log_basis,
        ..decomp
    };
    // Root-direct commits against `log_commit_bound` (the root form of
    // `num_digits_s_commit`) and opens at `log_open_bound`.
    let depth_commit = num_digits_s_commit(level_decomp, true);
    let depth_open = num_digits_open(level_decomp);

    // Outer/inner variable split: brute-force the optimum for a normal root,
    // single-block `(0, 0)` for a tiny root (`num_vars <= log2(d)`). The
    // optimizer recomputes the fold-priced A collision per `r` internally
    // (it grows with the fold arity `num_claims · 2^r`), so it needs the
    // batch factor and ring-subfield norm, not a single pre-baked bucket.
    let (m_vars, r_vars) = if num_vars > alpha {
        // The `(m, r)` split is scored against the flat L1 mass (the root fold
        // shape disambiguates the committed table, not the split search).
        let fold_challenge =
            akita_types::sis::fold_challenge_norms(&ring_challenge_cfg, TensorChallengeShape::Flat);
        // One-hot root commits a sparse witness (`||s||_inf = 1`,
        // `nonzeros = ceil(D/K)`); dense roots use the balanced-digit norms.
        let is_onehot = decomp.log_commit_bound == 1;
        let fold_witness = FoldWitnessNorms::new(log_basis, d, policy.onehot_chunk_size, is_onehot);
        let (m_vars, r_vars, _scoring_n_a) = optimal_m_r_split(
            sis_family,
            d as u32,
            num_claims,
            policy.ring_subfield_norm_bound,
            fold_challenge,
            fold_witness,
            &ring_challenge_cfg,
            TensorChallengeShape::Flat,
            decomp,
            policy.onehot_chunk_size,
            num_vars - alpha,
            0,
        );
        (m_vars, r_vars)
    } else {
        (0, 0)
    };

    let Some(num_blocks) = 1usize.checked_shl(r_vars as u32) else {
        return Ok(None);
    };
    let Some(block_len) = 1usize.checked_shl(m_vars as u32) else {
        return Ok(None);
    };

    // The A/B/D keys, composed from the `akita_types::sis` primitives:
    // norm -> width -> tight SIS-secure rank -> key. `t_vectors = num_claims`
    // folds the batched-root scaling into the B/D widths (the root commits
    // `num_claims` polynomials) — no separate per-claim-then-scale pass.
    let Some(width_s) = decomposed_s_block_ring_count(block_len, depth_commit) else {
        return Ok(None);
    };
    let Some((op_norm_rejection, norm_s, n_a)) = choose_op_norm_rejection_for_a_role(
        sis_family,
        d,
        level_decomp,
        &ring_challenge_cfg,
        fold_challenge_shape,
        true,
        policy.onehot_chunk_size,
        policy.ring_subfield_norm_bound,
        r_vars,
        num_claims,
        width_s as u64,
    ) else {
        return Ok(None);
    };
    let a_key = AjtaiKeyParams::try_new(sis_family, n_a, width_s, norm_s, d)?;
    let Some(norm_t) = rounded_up_collision_norm_t(sis_family, d, log_basis) else {
        return Ok(None);
    };
    let Some(width_t) = decomposed_t_ring_count(n_a, depth_open, num_blocks, num_claims) else {
        return Ok(None);
    };
    let Some(n_b) = min_secure_rank(sis_family, d as u32, norm_t, width_t as u64) else {
        return Ok(None);
    };
    let b_key = AjtaiKeyParams::try_new(sis_family, n_b, width_t, norm_t, d)?;
    let Some(norm_w) = rounded_up_collision_norm_w(sis_family, d, log_basis) else {
        return Ok(None);
    };
    let Some(width_w) = decomposed_w_ring_count(depth_open, num_blocks, num_claims) else {
        return Ok(None);
    };
    let Some(n_d) = min_secure_rank(sis_family, d as u32, norm_w, width_w as u64) else {
        return Ok(None);
    };
    let d_key = AjtaiKeyParams::try_new(sis_family, n_d, width_w, norm_w, d)?;

    // A one-hot root (`log_commit_bound == 1`) commits a sparse witness; record
    // its chunk size so `num_digits_fold` and the binding norm size the folded
    // witness against `nonzeros = ceil(D/K)` instead of `D`.
    let onehot_chunk_size = if decomp.log_commit_bound == 1 {
        policy.onehot_chunk_size
    } else {
        0
    };

    let (tier_split, b_key, f_key) = if policy.tiered {
        let Some(a_matrix_size) = a_key.row_len().checked_mul(a_key.col_len()) else {
            return Ok(None);
        };
        multi_tiered_keys(a_matrix_size, &b_key, depth_open, log_basis, d)?
    } else {
        (1, b_key, None)
    };

    let root_direct_params = LevelParams {
        ring_dimension: d,
        log_basis,
        a_key,
        b_key,
        d_key,
        num_blocks,
        block_len,
        m_vars,
        r_vars,
        stage1_config: ring_challenge_cfg,
        op_norm_rejection,
        fold_challenge_shape,
        num_digits_commit: depth_commit,
        num_digits_open: depth_open,
        onehot_chunk_size,
        tier_split,
        f_key,
        fold_linf_cap_config: FoldWitnessLinfCapConfig::worst_case_beta_only(),
        num_digits_fold_one: 1,
        field_bits_hint: 0,
        cached_num_digits_fold_claims: 0,
        cached_num_digits_fold_value: 1,
        // Root-direct ships the raw polynomial on the wire (no chunked commitment).
        witness_chunk: ChunkedWitnessCfg::default(),
        precommitted_groups: Vec::new(),
    }
    .with_fold_linf_cap_config(decomp.field_bits(), num_claims)?;
    Ok(Some(root_direct_params))
}

/// Find the optimal schedule for a root schedule lookup key under `policy`.
///
/// Runs an exhaustive DP that minimizes proof size. The result is a pure,
/// deterministic function of `(policy, key)` (plus the `ring_challenge_config` /
/// `fold_challenge_shape_at_level` closures, which presets derive from the same hooks the
/// generated tables were emitted from), so the prover and verifier
/// regenerate identical schedules on a table miss.
///
/// # Errors
///
/// Returns an error if vector counts are invalid or if the witness length
/// overflows. The function never panics on malformed input — it is
/// verifier-reachable and audited under the no-panic contract.
pub fn find_schedule(
    key: AkitaScheduleLookupKey,
    policy: &PlannerPolicy,
    ring_challenge_config: impl Fn(usize) -> Result<akita_challenges::SparseChallengeConfig, AkitaError>,
    fold_challenge_shape_at_level: impl Fn(AkitaScheduleInputs) -> TensorChallengeShape,
) -> Result<Schedule, AkitaError> {
    find_schedule_inner(
        key,
        policy,
        ring_challenge_config,
        fold_challenge_shape_at_level,
    )
}

fn find_schedule_inner(
    key: AkitaScheduleLookupKey,
    policy: &PlannerPolicy,
    ring_challenge_config: impl Fn(usize) -> Result<akita_challenges::SparseChallengeConfig, AkitaError>,
    fold_challenge_shape_at_level: impl Fn(AkitaScheduleInputs) -> TensorChallengeShape,
) -> Result<Schedule, AkitaError> {
    let ring_challenge_config: RingChallengeConfigFn<'_> = &ring_challenge_config;
    let fold_shape: FoldShapeFn<'_> = &fold_challenge_shape_at_level;
    let suffix_ctx = SuffixCtx {
        policy,
        ring_challenge_config,
        num_vars: key.num_vars,
        key,
    };

    key.validate()?;
    validate_policy_witness_chunk(policy)?;

    let witness_len = 1usize
        .checked_shl(key.num_vars as u32)
        .ok_or_else(|| AkitaError::InvalidSetup("witness too large".into()))?;

    let field_bits = policy.decomposition.field_bits();

    let root_witness_shape = CleartextWitnessShape::FieldElements(witness_len);
    let mut best_cost = direct_witness_bytes(field_bits, &root_witness_shape);
    let fold_challenge_shape = fold_shape(AkitaScheduleInputs {
        num_vars: key.num_vars,
        level: 0,
        current_w_len: witness_len,
    });
    // The level-0 fold-challenge shape and the `num_claims = num_polynomials`
    // batch factor are folded directly into the committed B/D widths, so a table
    // miss reproduces the exact root commit layout the table-hit expansion
    // (`expand_to_level_params`) builds — no separate per-claim-then-scale
    // pass. `Ok(None)` is the uncommittable (large-`num_vars`) edge.
    let root_direct_commit_params = compute_root_direct_level_params(
        policy,
        ring_challenge_config,
        key.num_vars,
        policy.decomposition.log_basis,
        fold_challenge_shape,
        key.num_polynomials,
    )?;
    let mut best_steps: Vec<Step> = vec![Step::Direct(DirectStep {
        current_w_len: witness_len,
        witness_shape: root_witness_shape,
        direct_bytes: best_cost,
        params: root_direct_commit_params,
    })];
    let mut memo = ScheduleMemo::new();

    let ring_challenge_cfg = ring_challenge_config(policy.ring_dimension)?;
    let alpha = (policy.ring_dimension as u32).trailing_zeros() as usize;
    let reduced_vars = key.num_vars.saturating_sub(alpha);

    if reduced_vars == 0 {
        return Ok(Schedule {
            steps: best_steps,
            total_bytes: best_cost,
        });
    }

    let min_r_vars: usize = if reduced_vars >= 3 { 1 } else { 0 };
    let max_r_vars: usize = (reduced_vars - 1).min(usize::BITS as usize - 1);

    // Chunk count of the witness committed at the root fold (absolute level 0).
    let root_num_chunks = policy.chunks_at_level(0);
    let root_witness_chunk = policy.witness_chunk_for_level(0);

    let (min_log_basis, max_log_basis) = policy.basis_range;
    for candidate_log_basis in min_log_basis..=max_log_basis {
        let level_decomp = DecompositionParams {
            log_basis: candidate_log_basis,
            ..policy.decomposition
        };
        let num_digits_commit = num_digits_s_commit(level_decomp, true);
        let num_digits_open = num_digits_open(level_decomp);

        for r_vars in (min_r_vars..=max_r_vars).rev() {
            let Some(num_blocks) = 1usize.checked_shl(r_vars as u32) else {
                continue;
            };
            // Multi-chunk root candidates require an equal block window per chunk.
            if root_num_chunks > 1 && !num_blocks.is_multiple_of(root_num_chunks) {
                continue;
            }
            let m_vars = reduced_vars - r_vars;

            let Some(block_len) = 1usize.checked_shl(m_vars as u32) else {
                continue;
            };

            // Compose the three SIS-secure keys from the `akita_types::sis`
            // primitives: norm -> width -> tight rank -> key.
            let family = policy.sis_family;
            let d = policy.ring_dimension;
            let Some(width_s) = decomposed_s_block_ring_count(block_len, num_digits_commit) else {
                continue;
            };
            let Some((op_norm_rejection, norm_s, n_a)) = choose_op_norm_rejection_for_a_role(
                family,
                d,
                level_decomp,
                &ring_challenge_cfg,
                fold_challenge_shape,
                true,
                policy.onehot_chunk_size,
                policy.ring_subfield_norm_bound,
                r_vars,
                key.num_polynomials,
                width_s as u64,
            ) else {
                continue;
            };
            let a_key = AjtaiKeyParams::try_new(family, n_a, width_s, norm_s, d)?;
            let Some(norm_t) = rounded_up_collision_norm_t(family, d, candidate_log_basis) else {
                continue;
            };
            let Some(width_t) =
                decomposed_t_ring_count(n_a, num_digits_open, num_blocks, key.num_polynomials)
            else {
                continue;
            };
            let Some(n_b) = min_secure_rank(family, d as u32, norm_t, width_t as u64) else {
                continue;
            };
            let b_key = AjtaiKeyParams::try_new(family, n_b, width_t, norm_t, d)?;
            let Some(norm_w) = rounded_up_collision_norm_w(family, d, candidate_log_basis) else {
                continue;
            };
            let Some(width_w) =
                decomposed_w_ring_count(num_digits_open, num_blocks, key.num_polynomials)
            else {
                continue;
            };
            let Some(n_d) = min_secure_rank(family, d as u32, norm_w, width_w as u64) else {
                continue;
            };
            let d_key = AjtaiKeyParams::try_new(family, n_d, width_w, norm_w, d)?;

            let onehot_chunk_size = if policy.decomposition.log_commit_bound == 1 {
                policy.onehot_chunk_size
            } else {
                0
            };
            let (tier_split, b_key, f_key) = if policy.tiered {
                let Some(a_matrix_size) = a_key.row_len().checked_mul(a_key.col_len()) else {
                    continue;
                };
                multi_tiered_keys(
                    a_matrix_size,
                    &b_key,
                    num_digits_open,
                    candidate_log_basis,
                    d,
                )?
            } else {
                (1, b_key, None)
            };
            let Ok(candidate_params) = LevelParams {
                ring_dimension: policy.ring_dimension,
                log_basis: candidate_log_basis,
                a_key,
                b_key,
                d_key,
                num_blocks,
                block_len,
                m_vars,
                r_vars,
                stage1_config: ring_challenge_cfg.clone(),
                op_norm_rejection,
                fold_challenge_shape,
                num_digits_commit,
                num_digits_open,
                onehot_chunk_size,
                tier_split,
                f_key,
                fold_linf_cap_config: FoldWitnessLinfCapConfig::worst_case_beta_only(),
                num_digits_fold_one: 1,
                field_bits_hint: 0,
                cached_num_digits_fold_claims: 0,
                cached_num_digits_fold_value: 1,
                witness_chunk: root_witness_chunk,
                precommitted_groups: Vec::new(),
            }
            .with_fold_linf_cap_config(field_bits, key.num_polynomials) else {
                continue;
            };

            let next_withness_len_impl = |layout| -> Result<usize, AkitaError> {
                let rings = w_ring_element_count_for_chunks(
                    field_bits,
                    &candidate_params,
                    key.num_polynomials,
                    layout,
                    root_num_chunks,
                )?;
                rings.checked_mul(policy.ring_dimension).ok_or_else(|| {
                    AkitaError::InvalidSetup("root next witness length overflow".into())
                })
            };
            let next_w_len = next_withness_len_impl(MRowLayout::WithDBlock)?;
            let next_w_len_terminal = next_withness_len_impl(MRowLayout::WithoutDBlock)?;
            let initial_witness_len_bits = witness_len
                .checked_mul(field_bits as usize)
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("root witness bit length overflow".into())
                })?;
            if next_w_len
                .checked_mul(candidate_log_basis as usize)
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("root next witness bit length overflow".into())
                })?
                >= initial_witness_len_bits
            {
                continue;
            }

            let suffix = derive_optimal_suffix_schedule(
                &suffix_ctx,
                &mut memo,
                1,
                next_w_len,
                next_w_len_terminal,
                candidate_log_basis,
                0,
            )?;
            if suffix.is_empty() {
                continue;
            }
            let Ok(eor_bytes) =
                extension_opening_reduction_level_bytes(policy, key, 0, witness_len)
            else {
                continue;
            };

            // Branch A: suffix at level 1 is a Direct
            if let Some(direct_suffix) = suffix.best_direct {
                let (direct_step, suffix_cost) = terminal_direct_suffix_cost(
                    direct_suffix.current_w_len,
                    &candidate_params,
                    field_bits,
                    key,
                    0,
                )?;
                let root_proof_size = level_proof_bytes(
                    field_bits,
                    field_bits * policy.chal_ext_degree as u32,
                    &candidate_params,
                    None,
                    next_w_len_terminal,
                    1,
                    MRowLayout::WithoutDBlock,
                ) + eor_bytes;
                let total = root_proof_size + suffix_cost;
                if total < best_cost {
                    best_cost = total;
                    best_steps = vec![
                        Step::Fold(FoldStep {
                            params: candidate_params.clone(),
                            current_w_len: witness_len,
                            next_w_len: next_w_len_terminal,
                            level_bytes: root_proof_size,
                        }),
                        Step::Direct(direct_step),
                    ];
                }
            }
            // Branch B: suffix at level 1 is a Fold
            for suffix_fold in suffix.best_fold_per_lb.values() {
                let root_proof_size = level_proof_bytes(
                    field_bits,
                    field_bits * policy.chal_ext_degree as u32,
                    &candidate_params,
                    Some(&suffix_fold.first_fold_params),
                    next_w_len,
                    1,
                    MRowLayout::WithDBlock,
                ) + eor_bytes;
                let total = root_proof_size + suffix_fold.total_bytes;
                if total < best_cost {
                    best_cost = total;
                    let mut steps = Vec::with_capacity(1 + suffix_fold.steps.len());
                    steps.push(Step::Fold(FoldStep {
                        params: candidate_params.clone(),
                        current_w_len: witness_len,
                        next_w_len,
                        level_bytes: root_proof_size,
                    }));
                    steps.extend(suffix_fold.steps.iter().cloned());
                    best_steps = steps;
                }
            }
        }
    }

    Ok(Schedule {
        steps: best_steps,
        total_bytes: best_cost,
    })
}

#[cfg(test)]
mod tiering_tests {
    use super::*;
    use akita_types::sis::{min_secure_rank, AjtaiKeyParams, SisModulusFamily};

    const D: usize = 64;
    const FAMILY: SisModulusFamily = SisModulusFamily::Q128;
    const LOG_BASIS: u32 = 3;
    const DELTA_OPEN: usize = 43;

    fn tiered_collision() -> u128 {
        rounded_up_collision_norm_tiered_commitment(FAMILY, D, LOG_BASIS).unwrap()
    }

    fn b_key(n_b: usize, width: usize) -> AjtaiKeyParams {
        AjtaiKeyParams::new_unchecked(FAMILY, n_b, width, tiered_collision(), D)
    }

    #[test]
    fn tiering_skipped_when_b_fits_under_a() {
        // B size (1·86) already <= A size, so no split.
        let bk = b_key(1, 86);
        let (f, out_b, fk) = multi_tiered_keys(1_000_000, &bk, DELTA_OPEN, LOG_BASIS, D).unwrap();
        assert_eq!(f, 1);
        assert!(fk.is_none());
        assert_eq!(out_b.col_len(), bk.col_len());
    }

    #[test]
    fn tiering_fires_with_smallest_feasible_split() {
        // B size = 1·5504 = 5504 > A size = 2·1024 = 2048.
        let width_t = 5504;
        let a_matrix_size = 2 * 1024;
        let bk = b_key(1, width_t);

        let (f, out_b, fk) =
            multi_tiered_keys(a_matrix_size, &bk, DELTA_OPEN, LOG_BASIS, D).unwrap();
        let fk = fk.expect("expected tiering to fire");

        assert!(f > 1);
        // B' fits under A; width shrank by exactly the split factor.
        assert!(out_b.row_len() * out_b.col_len() <= a_matrix_size);
        assert_eq!(out_b.col_len(), width_t / f);
        // F width = f · n_b' · δ_open, same collision bucket as B.
        assert_eq!(fk.col_len(), f * out_b.row_len() * DELTA_OPEN);
        let norm = tiered_collision();
        assert_eq!(out_b.collision_l2_sq(), norm);
        assert_eq!(fk.collision_l2_sq(), norm);
        // Minimality: no smaller divisor of width_t (in 2..f) makes B' fit under A.
        for smaller in 2..f {
            if !width_t.is_multiple_of(smaller) {
                continue;
            }
            let w = width_t / smaller;
            let n = min_secure_rank(FAMILY, D as u32, norm, w as u64).unwrap();
            assert!(
                n * w > a_matrix_size,
                "split f={smaller} should not fit under A"
            );
        }
    }
}
