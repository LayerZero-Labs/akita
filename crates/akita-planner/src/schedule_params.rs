//! Schedule planner that finds the global minimum proof size.
//!
//! Public entry: [`find_schedule`]. The search is `Cfg`-free: every
//! per-preset input is carried by the plain-value [`PlannerPolicy`] plus
//! the `stage1` / `fold_shape` closures, exactly the shape
//! `crate::schedule_from_entry` already consumes. This keeps the
//! DP a pure function of `(policy, key)` so `akita-config` can call it
//! directly on a schedule-table miss without a dependency cycle.

use std::collections::{BTreeMap, HashMap};

use akita_challenges::TensorChallengeShape;
use akita_field::AkitaError;
use akita_types::layout::digit_math::optimal_m_r_split;
use akita_types::sis::{
    decomposed_s_block_ring_count, decomposed_t_ring_count, decomposed_w_ring_count,
    min_secure_rank, num_digits_open, num_digits_s_commit, rounded_up_collision_norm_s,
    rounded_up_collision_norm_t, rounded_up_collision_norm_w, AjtaiKeyParams, FoldChallengeNorms,
    FoldWitnessNorms,
};
use akita_types::{
    decomp_depths, direct_witness_bytes, extension_opening_reduction_proof_bytes,
    level_proof_bytes, root_extension_opening_partials,
    w_ring_element_count_with_counts_for_layout_bits, AkitaScheduleInputs, AkitaScheduleLookupKey,
    CleartextWitnessShape, DecompositionParams, DirectStep, FoldStep, LevelParams, MRowLayout,
    Schedule, Step,
};

use crate::PlannerPolicy;

/// Stage-1 sparse-challenge closure shared by the planner entry points.
pub(crate) type Stage1Fn<'a> =
    &'a dyn Fn(usize) -> Result<akita_challenges::SparseChallengeConfig, AkitaError>;

/// Stage-1 fold-round challenge-shape closure (`level 0` root shape).
type FoldShapeFn<'a> = &'a dyn Fn(AkitaScheduleInputs) -> TensorChallengeShape;

// Suffix-DP depth cap. Schedules in our working parameter range never need
// more than this many recursive fold levels; deeper search only blows up
// memo state without changing emitted tables.
const MAX_RECURSION_DEPTH: usize = 12;

/// SIS-secure `F` candidate for a tiered split into `f` slices, or `None` if
/// any width/bucket lookup or checked arithmetic rejects it.
///
/// Returns `(n_b', width_t', n_f, width_f, b_footprint, f_footprint)` where
/// `B'` is `n_b' × width_t'` and `F` is `n_f × width_f`.
#[allow(clippy::too_many_arguments)]
fn tiering_candidate(
    family: akita_types::SisModulusFamily,
    d: usize,
    norm_t: u32,
    norm_f: u32,
    width_t: usize,
    delta_open: usize,
    f: usize,
) -> Option<(usize, usize, usize, usize, usize, usize)> {
    // `f` divides `num_blocks · t_vectors`, hence divides `width_t`.
    let width_t_small = width_t.checked_div(f)?;
    if width_t_small == 0 {
        return None;
    }
    let n_b_small = min_secure_rank(family, d as u32, norm_t, width_t_small as u64)?;
    let b_footprint = n_b_small.checked_mul(width_t_small)?;
    // `F` commits `decompose(u_1 ‖ … ‖ u_f)`: `f · n_b' · δ_open` digit columns.
    let width_f = f.checked_mul(n_b_small)?.checked_mul(delta_open)?;
    let n_f = min_secure_rank(family, d as u32, norm_f, width_f as u64)?;
    let f_footprint = n_f.checked_mul(width_f)?;
    Some((
        n_b_small,
        width_t_small,
        n_f,
        width_f,
        b_footprint,
        f_footprint,
    ))
}

/// Apply the tiered-commitment second matrix `F` to a freshly-built level.
///
/// When [`PlannerPolicy::tiered`] is set and the first-tier `B` footprint
/// exceeds the inner `A` footprint, reuse a smaller `B'` across `f` equal
/// column-slices of `t̂` (`f` divides `num_blocks · t_vectors`) and size the
/// second-tier `F` that commits the decomposed concatenated slice images. The
/// chosen `f` is the feasible power of two minimizing
/// `max(B'_footprint, F_footprint)` subject to both being `<= A_footprint`
/// (tie-broken toward the smaller `f`, which adds fewer relation rows and less
/// witness). Returns the level unchanged (single-tier: `tier_split == 1`,
/// `f_key == None`) when tiering is off, unnecessary (`B <= A`), or infeasible.
///
/// Verifier-reachable through the runtime DP fallback: never panics — every
/// product is checked and every emitted key passes [`AjtaiKeyParams::try_new`].
///
/// Also replayed by [`crate::generated::GeneratedFoldStep::expand_to_level_params`]
/// so a shipped tiered table reconstructs the exact same `B'`/`F` split the DP
/// chose: the table stores the *un-tiered* `n_b`, and expansion re-applies this.
pub(crate) fn apply_tiering(
    policy: &PlannerPolicy,
    lp: LevelParams,
) -> Result<LevelParams, AkitaError> {
    if !policy.tiered {
        return Ok(lp);
    }
    let family = policy.sis_family;
    let d = policy.ring_dimension;

    let Some(a_footprint) = lp.a_key.row_len().checked_mul(lp.a_key.col_len()) else {
        return Ok(lp);
    };
    let width_t = lp.b_key.col_len();
    let Some(b_footprint) = lp.b_key.row_len().checked_mul(width_t) else {
        return Ok(lp);
    };
    if b_footprint <= a_footprint {
        // First-tier `B` already fits under `A`; no tiering needed.
        return Ok(lp);
    }

    // `width_t = n_a · δ_open · num_blocks · t_vectors`. Split only along the
    // "repeat" dimensions (`num_blocks · t_vectors`), keeping each slice's
    // inner `(n_a · δ_open)` structure intact so "the same B'" is well-defined.
    let delta_open = lp.num_digits_open;
    let Some(inner) = lp.a_key.row_len().checked_mul(delta_open) else {
        return Ok(lp);
    };
    if inner == 0 || !width_t.is_multiple_of(inner) {
        return Ok(lp);
    }
    let repeat = width_t / inner; // = num_blocks · t_vectors

    let norm_t = lp.b_key.collision_inf();
    // `F` consumes balanced base-`2^log_basis` digits of `u_concat`, so its
    // collision bucket is the digit range — the same bound as the `B`/`D` roles.
    let Some(norm_f) = rounded_up_collision_norm_t(family, d, lp.log_basis) else {
        return Ok(lp);
    };

    let mut best: Option<(usize, usize, usize, usize, usize, usize)> = None;
    let mut f = 2usize;
    while f <= repeat {
        if repeat.is_multiple_of(f) {
            if let Some((n_b_small, width_t_small, n_f, width_f, b_foot, f_foot)) =
                tiering_candidate(family, d, norm_t, norm_f, width_t, delta_open, f)
            {
                if b_foot <= a_footprint && f_foot <= a_footprint {
                    let max_foot = b_foot.max(f_foot);
                    let better = match best {
                        None => true,
                        Some((best_f, _, _, _, _, best_max)) => {
                            max_foot < best_max || (max_foot == best_max && f < best_f)
                        }
                    };
                    if better {
                        best = Some((f, n_b_small, width_t_small, n_f, width_f, max_foot));
                    }
                }
            }
        }
        f = match f.checked_mul(2) {
            Some(next) => next,
            None => break,
        };
    }

    let Some((f, n_b_small, width_t_small, n_f, width_f, _)) = best else {
        // No feasible split keeps both B' and F under A; stay single-tier.
        return Ok(lp);
    };

    let b_key = AjtaiKeyParams::try_new(family, n_b_small, width_t_small, norm_t, d)?;
    let f_key = AjtaiKeyParams::try_new(family, n_f, width_f, norm_f, d)?;
    Ok(LevelParams {
        b_key,
        f_key: Some(f_key),
        tier_split: f,
        ..lp
    })
}

/// Compute parameters that generate the smallest witness for the next
/// fold level. Note that this is not the optimum case: in the optimum
/// case (similar to `find_schedule`), we should check that current proof
/// size + suffix cost is the smallest. However, as time blows up, we
/// don't do that here.
fn derive_candidate_level_params(
    policy: &PlannerPolicy,
    stage1: Stage1Fn<'_>,
    current_witness_len: usize,
    log_basis: u32,
) -> Result<Option<(LevelParams, usize, usize)>, AkitaError> {
    let Ok(stage1_config) = stage1(policy.ring_dimension) else {
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
        let Some(norm_s) = rounded_up_collision_norm_s(
            family,
            d,
            decomp,
            &stage1_config,
            TensorChallengeShape::Flat,
            false,
            policy.onehot_chunk_size,
            policy.ring_subfield_norm_bound,
            r,
            1,
        ) else {
            continue;
        };
        let Some(width_s) = decomposed_s_block_ring_count(block_len, delta_commit) else {
            continue;
        };
        let Some(n_a) = min_secure_rank(family, d as u32, norm_s, width_s as u64) else {
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

        let candidate_params = LevelParams {
            ring_dimension: policy.ring_dimension,
            log_basis,
            a_key,
            b_key,
            d_key,
            num_blocks,
            block_len,
            m_vars: reduced_vars - r,
            r_vars: r,
            stage1_config: stage1_config.clone(),
            fold_challenge_shape: TensorChallengeShape::Flat,
            num_digits_commit: delta_commit,
            num_digits_open: delta_open,
            // Recursive levels commit dense balanced-digit witnesses.
            onehot_chunk_size: 0,
            tier_split: 1,
            f_key: None,
        };
        let candidate_params = apply_tiering(policy, candidate_params)?;

        let next_witness_len = w_ring_element_count_with_counts_for_layout_bits(
            policy.decomposition.field_bits(),
            &candidate_params,
            1,
            1,
            1,
            1,
            MRowLayout::WithDBlock,
        )?
        .checked_mul(policy.ring_dimension)
        .ok_or_else(|| AkitaError::InvalidSetup("recursive witness length overflow".into()))?;
        let next_witness_len_terminal = w_ring_element_count_with_counts_for_layout_bits(
            policy.decomposition.field_bits(),
            &candidate_params,
            1,
            1,
            1,
            1,
            MRowLayout::WithoutDBlock,
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
        (
            root_extension_opening_partials(width, key.num_w_vectors),
            key.num_vars,
        )
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
    best_direct: Option<(usize, Vec<Step>)>,
    best_fold_per_lb: BTreeMap<u32, FoldSuffix>,
}

impl SuffixResult {
    fn is_empty(&self) -> bool {
        self.best_direct.is_none() && self.best_fold_per_lb.is_empty()
    }
}

type ScheduleMemo = HashMap<(usize, usize, usize, u32), SuffixResult>;

/// DP-invariant inputs for the suffix search.
///
/// `policy`, `stage1`, and `num_vars` are constant across the whole
/// recursion, so they are carried in one context value rather than as
/// per-call arguments (keeps the recursive signature small).
#[derive(Clone, Copy)]
struct SuffixCtx<'a> {
    policy: &'a PlannerPolicy,
    stage1: Stage1Fn<'a>,
    num_vars: usize,
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
        stage1,
        num_vars,
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

    let best_direct = {
        let witness_shape =
            CleartextWitnessShape::PackedDigits((current_witness_len_terminal, current_lb));
        let direct_bytes = direct_witness_bytes(policy.decomposition.field_bits(), &witness_shape);
        let step = Step::Direct(DirectStep {
            current_w_len: current_witness_len_terminal,
            witness_shape,
            direct_bytes,
            params: None,
        });
        Some((direct_bytes, vec![step]))
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
            derive_candidate_level_params(policy, stage1, current_witness_len, lb)?
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
        if let Some((suffix_cost, suffix_sched)) = suffix.best_direct.as_ref() {
            let level_proof_size = level_proof_bytes(
                policy.decomposition.field_bits(),
                policy.decomposition.field_bits() * policy.chal_ext_degree as u32,
                &candidate_params,
                None,
                next_witness_len_terminal,
                1,
                MRowLayout::WithoutDBlock,
            ) + eor_bytes;
            let total = level_proof_size + suffix_cost;
            let mut steps = Vec::with_capacity(1 + suffix_sched.len());
            steps.push(Step::Fold(FoldStep {
                params: candidate_params.clone(),
                current_w_len: current_witness_len,
                next_w_len: next_witness_len_terminal,
                level_bytes: level_proof_size,
            }));
            steps.extend(suffix_sched.iter().cloned());
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
    stage1: Stage1Fn<'_>,
    num_vars: usize,
    log_basis: u32,
    fold_challenge_shape: TensorChallengeShape,
    num_claims: usize,
) -> Result<Option<LevelParams>, AkitaError> {
    let stage1_config = stage1(policy.ring_dimension)?;
    let d = policy.ring_dimension;
    let sis_family = policy.sis_family;
    let decomp = policy.decomposition;
    let alpha = (d as u32).trailing_zeros() as usize;

    let level_decomp = DecompositionParams {
        log_basis,
        ..decomp
    };
    let (depth_commit, depth_open) = decomp_depths(level_decomp);

    // Outer/inner variable split: brute-force the optimum for a normal root,
    // single-block `(0, 0)` for a tiny root (`num_vars <= log2(d)`). The
    // optimizer recomputes the fold-priced A collision per `r` internally
    // (it grows with the fold arity `num_claims · 2^r`), so it needs the
    // batch factor and ring-subfield norm, not a single pre-baked bucket.
    let (m_vars, r_vars) = if num_vars > alpha {
        // The `(m, r)` split is scored against the flat L1 mass (the root fold
        // shape disambiguates the committed table, not the split search).
        let fold_challenge = FoldChallengeNorms {
            infinity_norm: TensorChallengeShape::Flat.effective_infinity_norm(&stage1_config)
                as u128,
            l1_norm: TensorChallengeShape::Flat.effective_l1_mass(&stage1_config) as u128,
        };
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
            decomp.log_commit_bound,
            log_basis,
            num_vars - alpha,
            0,
            decomp.field_bits(),
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
    let Some(norm_s) = rounded_up_collision_norm_s(
        sis_family,
        d,
        level_decomp,
        &stage1_config,
        fold_challenge_shape,
        true,
        policy.onehot_chunk_size,
        policy.ring_subfield_norm_bound,
        r_vars,
        num_claims,
    ) else {
        return Ok(None);
    };
    let Some(width_s) = decomposed_s_block_ring_count(block_len, depth_commit) else {
        return Ok(None);
    };
    let Some(n_a) = min_secure_rank(sis_family, d as u32, norm_s, width_s as u64) else {
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
        stage1_config,
        fold_challenge_shape,
        num_digits_commit: depth_commit,
        num_digits_open: depth_open,
        onehot_chunk_size,
        tier_split: 1,
        f_key: None,
    };
    Ok(Some(apply_tiering(policy, root_direct_params)?))
}

/// Find the optimal schedule for a root schedule lookup key under `policy`.
///
/// Runs an exhaustive DP that minimizes proof size. The result is a pure,
/// deterministic function of `(policy, key)` (plus the `stage1` /
/// `fold_shape` closures, which presets derive from the same hooks the
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
    stage1: impl Fn(usize) -> Result<akita_challenges::SparseChallengeConfig, AkitaError>,
    fold_shape: impl Fn(AkitaScheduleInputs) -> TensorChallengeShape,
) -> Result<Schedule, AkitaError> {
    let stage1: Stage1Fn<'_> = &stage1;
    let fold_shape: FoldShapeFn<'_> = &fold_shape;
    let suffix_ctx = SuffixCtx {
        policy,
        stage1,
        num_vars: key.num_vars,
    };

    let t_vectors = key.num_t_vectors;
    let w_vectors = key.num_w_vectors;
    let z_vectors = key.num_z_vectors;
    let num_points = key.num_points;
    if num_points == 0 || t_vectors == 0 || w_vectors == 0 || z_vectors == 0 {
        return Err(AkitaError::InvalidSetup(
            "schedule key planner dimensions must be at least 1".into(),
        ));
    }
    if num_points > t_vectors || num_points > w_vectors {
        return Err(AkitaError::InvalidSetup(
            "schedule key opening-point count cannot exceed t or w vector counts".into(),
        ));
    }

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
    // The level-0 fold-challenge shape and the `num_claims = t_vectors` batch
    // factor are folded directly into the committed B/D widths, so a table
    // miss reproduces the exact root commit layout the table-hit expansion
    // (`expand_to_level_params`) builds — no separate per-claim-then-scale
    // pass. `Ok(None)` is the uncommittable (large-`num_vars`) edge.
    let root_direct_commit_params = compute_root_direct_level_params(
        policy,
        stage1,
        key.num_vars,
        policy.decomposition.log_basis,
        fold_challenge_shape,
        t_vectors,
    )?;
    let mut best_steps: Vec<Step> = vec![Step::Direct(DirectStep {
        current_w_len: witness_len,
        witness_shape: root_witness_shape,
        direct_bytes: best_cost,
        params: root_direct_commit_params,
    })];
    let mut memo = ScheduleMemo::new();

    let stage1_config = stage1(policy.ring_dimension)?;
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
            let m_vars = reduced_vars - r_vars;

            let Some(block_len) = 1usize.checked_shl(m_vars as u32) else {
                continue;
            };

            // Compose the three SIS-secure keys from the `akita_types::sis`
            // primitives: norm -> width -> tight rank -> key.
            let family = policy.sis_family;
            let d = policy.ring_dimension;
            let Some(norm_s) = rounded_up_collision_norm_s(
                family,
                d,
                level_decomp,
                &stage1_config,
                fold_challenge_shape,
                true,
                policy.onehot_chunk_size,
                policy.ring_subfield_norm_bound,
                r_vars,
                t_vectors,
            ) else {
                continue;
            };
            let Some(width_s) = decomposed_s_block_ring_count(block_len, num_digits_commit) else {
                continue;
            };
            let Some(n_a) = min_secure_rank(family, d as u32, norm_s, width_s as u64) else {
                continue;
            };
            let a_key = AjtaiKeyParams::try_new(family, n_a, width_s, norm_s, d)?;
            let Some(norm_t) = rounded_up_collision_norm_t(family, d, candidate_log_basis) else {
                continue;
            };
            let Some(width_t) =
                decomposed_t_ring_count(n_a, num_digits_open, num_blocks, t_vectors)
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
            let Some(width_w) = decomposed_w_ring_count(num_digits_open, num_blocks, t_vectors)
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
            let candidate_params = LevelParams {
                ring_dimension: policy.ring_dimension,
                log_basis: candidate_log_basis,
                a_key,
                b_key,
                d_key,
                num_blocks,
                block_len,
                m_vars,
                r_vars,
                stage1_config: stage1_config.clone(),
                fold_challenge_shape,
                num_digits_commit,
                num_digits_open,
                onehot_chunk_size,
                tier_split: 1,
                f_key: None,
            };
            let candidate_params = apply_tiering(policy, candidate_params)?;

            let next_withness_len_impl = |layout| -> Result<usize, AkitaError> {
                let rings = w_ring_element_count_with_counts_for_layout_bits(
                    field_bits,
                    &candidate_params,
                    key.num_points,
                    key.num_t_vectors,
                    key.num_w_vectors,
                    key.num_z_vectors,
                    layout,
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
            if let Some((suffix_cost, suffix_sched)) = suffix.best_direct.as_ref() {
                let root_proof_size = level_proof_bytes(
                    field_bits,
                    field_bits * policy.chal_ext_degree as u32,
                    &candidate_params,
                    None,
                    next_w_len_terminal,
                    z_vectors,
                    MRowLayout::WithoutDBlock,
                ) + eor_bytes;
                let total = root_proof_size + suffix_cost;
                if total < best_cost {
                    best_cost = total;
                    let mut steps = Vec::with_capacity(1 + suffix_sched.len());
                    steps.push(Step::Fold(FoldStep {
                        params: candidate_params.clone(),
                        current_w_len: witness_len,
                        next_w_len: next_w_len_terminal,
                        level_bytes: root_proof_size,
                    }));
                    steps.extend(suffix_sched.iter().cloned());
                    best_steps = steps;
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
                    z_vectors,
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
    use akita_challenges::SparseChallengeConfig;
    use akita_types::sis::{AjtaiKeyParams, SisModulusFamily};

    fn tiered_policy(tiered: bool) -> PlannerPolicy {
        PlannerPolicy {
            ring_dimension: 64,
            decomposition: DecompositionParams {
                log_basis: 3,
                log_commit_bound: 1,
                log_open_bound: Some(128),
            },
            sis_family: SisModulusFamily::Q128,
            ring_subfield_norm_bound: 1,
            claim_ext_degree: 1,
            chal_ext_degree: 1,
            basis_range: (2, 6),
            onehot_chunk_size: 256,
            tiered,
        }
    }

    /// A level whose first-tier `B` footprint exceeds `A`. `inner = n_a · δ_open
    /// = 2 · 43 = 86`, so `width_t` must be a multiple of 86 for the split to
    /// keep slices structurally intact.
    fn level_with_b_above_a(width_t: usize, a_cols: usize) -> LevelParams {
        let d = 64;
        let family = SisModulusFamily::Q128;
        LevelParams {
            ring_dimension: d,
            log_basis: 3,
            a_key: AjtaiKeyParams::new_unchecked(family, 2, a_cols, 8_388_607, d),
            b_key: AjtaiKeyParams::new_unchecked(family, 1, width_t, 7, d),
            d_key: AjtaiKeyParams::new_unchecked(family, 1, 43, 7, d),
            num_blocks: 8,
            block_len: a_cols,
            m_vars: 10,
            r_vars: 3,
            stage1_config: SparseChallengeConfig::Uniform {
                weight: 7,
                nonzero_coeffs: vec![-1, 1],
            },
            fold_challenge_shape: TensorChallengeShape::Flat,
            num_digits_commit: 1,
            num_digits_open: 43,
            onehot_chunk_size: 256,
            tier_split: 1,
            f_key: None,
        }
    }

    #[test]
    fn tiering_off_is_noop() {
        let lp = level_with_b_above_a(5504, 1024);
        let out = apply_tiering(&tiered_policy(false), lp.clone()).unwrap();
        assert_eq!(out.tier_split, 1);
        assert!(out.f_key.is_none());
        assert_eq!(out.b_key.col_len(), lp.b_key.col_len());
    }

    #[test]
    fn tiering_skipped_when_b_fits_under_a() {
        // Large A column count so the (small) B footprint already fits under A.
        let lp = level_with_b_above_a(86, 1_000_000);
        let out = apply_tiering(&tiered_policy(true), lp).unwrap();
        assert_eq!(out.tier_split, 1);
        assert!(out.f_key.is_none());
    }

    #[test]
    fn tiering_picks_min_max_split_and_fits_under_a() {
        // inner = 86, repeat = 5504/86 = 64, A footprint = 2 · 1024 = 2048.
        let a_cols = 1024;
        let width_t = 5504;
        let lp = level_with_b_above_a(width_t, a_cols);
        let a_footprint = lp.a_key.row_len() * lp.a_key.col_len();
        let orig_n_b = lp.b_key.row_len();

        let out = apply_tiering(&tiered_policy(true), lp).unwrap();

        let fk = out.f_key.as_ref().expect("expected tiering to fire");
        // Both B' and F fit under A (the planner invariant).
        assert!(out.b_key.row_len() * out.b_key.col_len() <= a_footprint);
        assert!(fk.row_len() * fk.col_len() <= a_footprint);
        // Re-derived rank never grows when the width shrinks.
        assert!(out.b_key.row_len() <= orig_n_b);
        // Width shrank by exactly the split factor.
        assert_eq!(out.b_key.col_len(), width_t / out.tier_split);
        // F width = f · n_b' · δ_open.
        assert_eq!(
            fk.col_len(),
            out.tier_split * out.b_key.row_len() * out.num_digits_open
        );
        // Deterministic min-max crossover for these dims: f = 8.
        assert_eq!(out.tier_split, 8);
        // Both keys must pass the SIS audit (try_new succeeded).
        assert_eq!(out.b_key.collision_inf(), 7);
        assert_eq!(fk.collision_inf(), 7);
    }
}
