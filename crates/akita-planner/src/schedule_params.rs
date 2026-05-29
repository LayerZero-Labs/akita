//! Schedule planner that finds the global minimum proof size.
//!
//! Public entry: [`find_schedule`], `<Cfg>`-generic. When `use_lookup` is
//! true the search consults `Cfg::schedule_table()` before running DP;
//! production callers pass `true`. The table-emitter binary passes
//! `false` to regenerate from scratch.

use std::collections::{BTreeMap, HashMap};

use akita_challenges::TensorChallengeShape;
use akita_config::CommitmentConfig;
use akita_field::AkitaError;
use akita_types::generated::sis_floor::{ceil_supported_collision, min_rank_for_secure_width};
use akita_types::generated::GeneratedScheduleTable;
use akita_types::layout::digit_math::{compute_num_digits_fold_with_claims, num_digits_for_bound};
use akita_types::{
    decomp_depths, direct_witness_bytes, extension_opening_reduction_proof_bytes,
    root_extension_opening_partials, schedule_from_plan,
    w_ring_element_count_with_counts_for_layout_bits, AjtaiKeyParams, AkitaScheduleInputs,
    AkitaScheduleLookupKey, DecompositionParams, DirectStep, DirectWitnessShape, FoldStep,
    LevelParams, MRowLayout, Schedule, Step,
};

use akita_derive::{schedule_plan_from_table, PlanPolicy};

use crate::ajtai_params::{
    compute_ajtai_key_params_a, compute_ajtai_key_params_b, compute_ajtai_key_params_d, WitnessType,
};
use crate::proof_size::level_proof_bytes;

// Suffix-DP depth cap. Schedules in our working parameter range never need
// more than this many recursive fold levels; deeper search only blows up
// memo state without changing emitted tables.
const MAX_RECURSION_DEPTH: usize = 12;

/// One recursive fold candidate: its level params plus the next-level
/// witness lengths under the Intermediate (`.1`) and Terminal (`.2`)
/// M-row layouts.
type CandidateLayout = (LevelParams, usize, usize);

/// Derive the layout for folding at `(level, w_len, log_basis)`.
/// Returns `None` if the layout is infeasible or doesn't shrink the witness.
///
/// Recursive levels (`level > 0`) only — the root candidate path in
/// `find_schedule` builds its own `LevelParams` directly from the
/// per-role SIS-floor lookups so it can enumerate `(num_blocks, log_basis)`
/// explicitly. Recursive levels pick the `(m, r)` split internally by a
/// cheap local witness-size estimate (the inlined `optimal_m_r_split`
/// sweep below) rather than enumerating `(num_blocks, log_basis)` and
/// recursing into the suffix DP per split: explicit enumeration here
/// would make every parent fold's `next_w_len` a fresh memo key and
/// expand the suffix-DP state space exponentially in depth, destroying
/// the dedup that makes the DP tractable.
///
/// Single-shot: the sweep derives the per-`r` SIS-secure A-rank `n_a`
/// from the floor table, scores each split by `|t̂| + |ŵ| + |ẑ|`, and
/// builds the full candidate (including the `(n_b, n_d)` lookups, both
/// M-row witness lengths, and the shrink check) for the winning split.
fn derive_candidate_level_params<Cfg: CommitmentConfig>(
    level: usize,
    current_w_len: usize,
    log_basis: u32,
) -> Result<Option<CandidateLayout>, AkitaError> {
    debug_assert!(
        level > 0,
        "derive_candidate_level_params is recursive-only; root candidates are built directly in find_schedule",
    );

    let sis_family = Cfg::sis_modulus_family();
    let d = Cfg::D;
    let Ok(stage1_config) = Cfg::stage1_challenge_config(d) else {
        return Ok(None);
    };
    if !current_w_len.is_multiple_of(d) {
        return Ok(None);
    }
    let num_ring_elems = current_w_len / d;
    let total = num_ring_elems.next_power_of_two().max(1);
    let reduced_vars = total.trailing_zeros() as usize;

    // Recursive-level decomposition: balanced-digit `w` collapses
    // `log_commit_bound` to `log_basis`; opening folds inherit the
    // parent's open bound. Matches `recursive_level_layout_from_params`.
    let root_decomp = Cfg::decomposition();
    let recursive_decomp = DecompositionParams {
        log_basis,
        log_commit_bound: log_basis,
        log_open_bound: Some(
            root_decomp
                .log_open_bound
                .unwrap_or(root_decomp.log_commit_bound),
        ),
    };
    let field_bits = recursive_decomp.field_bits();
    let (num_digits_commit, num_digits_open) = decomp_depths(recursive_decomp);
    let l1_mass = TensorChallengeShape::Flat.effective_l1_mass(&stage1_config);

    // Audited SIS collision buckets for this `(log_basis, family, d)`.
    let Some(collisions) = (|| -> Option<(u32, u32)> {
        let bd_collision = 1u32.checked_shl(log_basis)?.checked_sub(1)?;
        let a_collision_raw = bd_collision
            .checked_mul(stage1_config.infinity_norm())?
            .checked_mul(Cfg::ring_subfield_embedding_norm_bound())?;
        let a_collision = ceil_supported_collision(sis_family, d as u32, a_collision_raw)?;
        Some((a_collision, bd_collision))
    })() else {
        return Ok(None);
    };
    let (collision_a, collision_bd) = collisions;

    // Build the full candidate layout for one `(m_vars, r_vars, n_a)`
    // split, where `n_a` is the SIS-secure A-rank already chosen for the
    // split's `inner_width`. Returns `Ok(None)` when the B/D widths fall
    // off the audited SIS floor or the Intermediate witness fails to
    // shrink — the same rejections the post-split code applied before.
    //
    // Both M-row layouts are materialized so the suffix DP can cost
    // "fold then fold" (Intermediate) and "fold then direct" (Terminal)
    // against the same candidate without ever emitting a placeholder.
    //
    // The shrink check runs on the Intermediate shape (the witness
    // arrives in `log_basis`-bit digits and is re-emitted at the same
    // basis, so it reduces to "ring-element count strictly decreases",
    // kept in bit form to mirror the root's check and surface overflow).
    // We deliberately do NOT gate on the terminal shape: a candidate
    // whose Intermediate shape shrinks while its Terminal shape does not
    // is still useful for fold-then-...-then-direct chains.
    let build_candidate = |m_vars: usize,
                           r_vars: usize,
                           n_a: usize|
     -> Result<Option<CandidateLayout>, AkitaError> {
        let num_blocks = 1usize
            .checked_shl(r_vars as u32)
            .ok_or_else(|| AkitaError::InvalidSetup("2^r_vars does not fit usize".to_string()))?;
        let block_len = num_ring_elems.div_ceil(num_blocks);
        let inner_width = block_len
            .checked_mul(num_digits_commit)
            .ok_or_else(|| AkitaError::InvalidSetup("inner width overflow".to_string()))?;
        let outer_width = n_a
            .checked_mul(num_digits_open)
            .and_then(|w| w.checked_mul(num_blocks))
            .ok_or_else(|| AkitaError::InvalidSetup("outer width overflow".to_string()))?;
        let d_matrix_width = num_digits_open
            .checked_mul(num_blocks)
            .ok_or_else(|| AkitaError::InvalidSetup("D-matrix width overflow".to_string()))?;

        let Some(n_b) =
            min_rank_for_secure_width(sis_family, d as u32, collision_bd, outer_width as u64)
        else {
            return Ok(None);
        };
        let Some(n_d) =
            min_rank_for_secure_width(sis_family, d as u32, collision_bd, d_matrix_width as u64)
        else {
            return Ok(None);
        };

        let num_digits_fold =
            compute_num_digits_fold_with_claims(r_vars, l1_mass, log_basis, 1, field_bits);

        let level_lp = LevelParams {
            ring_dimension: d,
            log_basis,
            a_key: AjtaiKeyParams::new_unchecked(sis_family, n_a, inner_width, collision_a, d),
            b_key: AjtaiKeyParams::new_unchecked(sis_family, n_b, outer_width, collision_bd, d),
            d_key: AjtaiKeyParams::new_unchecked(sis_family, n_d, d_matrix_width, collision_bd, d),
            num_blocks,
            block_len,
            m_vars,
            r_vars,
            stage1_config: stage1_config.clone(),
            fold_challenge_shape: TensorChallengeShape::Flat,
            num_digits_commit,
            num_digits_open,
            num_digits_fold,
        };

        let next_w_len =
            recursive_next_witness_len(&level_lp, field_bits, MRowLayout::Intermediate)?;
        let next_w_len_terminal =
            recursive_next_witness_len(&level_lp, field_bits, MRowLayout::Terminal)?;

        let next_bits = next_w_len
            .checked_mul(log_basis as usize)
            .ok_or_else(|| AkitaError::InvalidSetup("next witness bit length overflow".into()))?;
        let current_bits = current_w_len
            .checked_mul(log_basis as usize)
            .ok_or_else(|| {
                AkitaError::InvalidSetup("current witness bit length overflow".into())
            })?;
        if next_bits >= current_bits {
            return Ok(None);
        }

        Ok(Some((level_lp, next_w_len, next_w_len_terminal)))
    };

    // Inlined `optimal_m_r_split`: brute-force the `(m, r)` split of
    // `reduced_vars` that minimizes the local next-witness-size estimate
    // `|t̂| + |ŵ| + |ẑ|`, building the full candidate for the winning
    // split in the same pass. Unlike `find_schedule`'s root loop we do
    // NOT recurse into the suffix DP per split — that would explode the
    // memo state space; the cheap witness-size proxy is the same
    // heuristic the standalone `optimal_m_r_split` used. `n_a` is the
    // per-`r` SIS-secure A-rank read straight from the floor table;
    // splits whose `inner_width` falls off the table are skipped.
    if reduced_vars <= 2 || reduced_vars >= 53 {
        // Too few vars to optimize, or `2^r` would overflow `u64`: use
        // the paper's symmetric split with the fallback rank `n_a = 1`.
        let r = reduced_vars / 2;
        return build_candidate(reduced_vars - r, r, 1);
    }

    // Cost-only digit counts (match the standalone optimizer): opening
    // entries are bounded by `max(log_commit_bound, field_bits)`, so the
    // cost `δ_open` is the full-field depth and is distinct from the
    // layout's `num_digits_open` (which tracks the inherited open bound).
    let cost_open_bound = recursive_decomp.log_commit_bound.max(field_bits);
    let cost_delta_open = num_digits_for_bound(cost_open_bound, field_bits, log_basis) as u64;
    let cost_delta_commit = num_digits_commit as u64;

    let mut best: Option<(u64, Option<CandidateLayout>)> = None;
    for r in 1..reduced_vars {
        let num_blocks = 1u64 << r;
        // `reduced_vars >= 3` here, so `num_ring_elems > 0` and the
        // block length is the tight `⌈num_ring / 2^r⌉`.
        let block_len = num_ring_elems.div_ceil(1usize << r) as u64;
        let m_eff = block_len;

        // Per-`r` SIS-secure A-rank for this split's inner width. Skip
        // the split when the floor table doesn't cover it.
        let Some(inner_width) = (block_len as usize).checked_mul(cost_delta_commit as usize) else {
            continue;
        };
        let Some(n_a) =
            min_rank_for_secure_width(sis_family, d as u32, collision_a, inner_width as u64)
        else {
            continue;
        };

        // δ_fold grows with r because β = 2^r · challenge_l1_mass · 2^(lb-1).
        let delta_fold =
            compute_num_digits_fold_with_claims(r, l1_mass, log_basis, 1, field_bits) as u64;
        // |t̂| + |ŵ|: each of the 2^r blocks contributes (1 + n_A) · δ_open.
        let per_block_cost =
            cost_delta_open.saturating_add((n_a as u64).saturating_mul(cost_delta_open));
        let opening_cost = per_block_cost.saturating_mul(num_blocks);
        // |ẑ|: folded-witness cost.
        let folding_cost = cost_delta_commit
            .saturating_mul(delta_fold)
            .saturating_mul(m_eff);
        let total = opening_cost.saturating_add(folding_cost);

        if best.as_ref().is_none_or(|(c, _)| total < *c) {
            let candidate = build_candidate(reduced_vars - r, r, n_a)?;
            best = Some((total, candidate));
        }
    }

    match best {
        Some((_, candidate)) => Ok(candidate),
        // No `r` had an audited A-rank: fall back to the paper's
        // symmetric split with `n_a = 1`, matching `optimal_m_r_split`.
        None => {
            let r = reduced_vars / 2;
            build_candidate(reduced_vars - r, r, 1)
        }
    }
}

/// Recursive next-level witness length under the given M-row layout.
///
/// Recursive folds carry a single witness opened at a single prepared
/// recursive point (all batching is absorbed into level 0), so the
/// multiplicity arguments to
/// [`w_ring_element_count_with_counts_for_layout_bits`] are all 1. The
/// only knob is the layout: `Intermediate` keeps the D-block in the
/// M-matrix (and the D-blinding under ZK); `Terminal` drops both.
fn recursive_next_witness_len(
    level_lp: &LevelParams,
    field_bits: u32,
    layout: MRowLayout,
) -> Result<usize, AkitaError> {
    let w_ring_elements =
        w_ring_element_count_with_counts_for_layout_bits(field_bits, level_lp, 1, 1, 1, 1, layout)?;
    w_ring_elements
        .checked_mul(level_lp.ring_dimension)
        .ok_or_else(|| AkitaError::InvalidSetup("recursive witness length overflow".into()))
}

/// Unified proof-size formula for one fold level under either M-row
/// layout. `next_lp` is consulted only when `layout =
/// MRowLayout::Intermediate`; terminal callers may pass any
/// `&LevelParams` (typically `lp`) as a placeholder.
fn compute_level_proof_size<Cfg: CommitmentConfig>(
    lp: &LevelParams,
    next_lp: &LevelParams,
    next_w_len: usize,
    num_public_outputs: usize,
    layout: MRowLayout,
) -> usize {
    let field_bits = Cfg::decomposition().field_bits();
    let challenge_bits = field_bits * Cfg::CHAL_EXT_DEGREE as u32;
    level_proof_bytes(
        field_bits,
        challenge_bits,
        lp,
        next_lp,
        next_w_len,
        num_public_outputs,
        layout,
    )
}

fn padded_boolean_vars(len: usize) -> Result<usize, AkitaError> {
    let padded = len
        .checked_next_power_of_two()
        .ok_or_else(|| AkitaError::InvalidSetup("opening witness length overflow".to_string()))?;
    Ok(padded.trailing_zeros() as usize)
}

fn extension_opening_reduction_level_bytes<Cfg: CommitmentConfig>(
    key: AkitaScheduleLookupKey,
    fold_level: usize,
    current_w_len: usize,
) -> Result<usize, AkitaError> {
    let width = Cfg::CLAIM_EXT_DEGREE;
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
        Cfg::decomposition().field_bits() * Cfg::CHAL_EXT_DEGREE as u32,
        partials,
        opening_vars,
        width,
    )
}

// -----------------------------------------------------------------------
// DP — suffix search
// -----------------------------------------------------------------------

/// Result of the suffix DP at one state.
///
/// The DP reports both shape options because the parent's proof-size
/// formula depends on the child's first step:
///
/// - `best_direct` — optimal schedule whose first step at this level is
///   a `Step::Direct`. Lets the parent score under
///   `compute_level_proof_size(..., MRowLayout::Terminal)` (drops the
///   v-rows and stage-1 sumcheck at the parent level). `None` when SIS
///   is infeasible at this state.
/// - `best_fold_per_lb` — for each candidate `log_basis` at this level,
///   the optimal schedule whose first step is `Step::Fold` with that
///   `log_basis`. Keyed by first fold's `log_basis` because the
///   parent's intermediate proof formula uses
///   `next_lp.b_key.row_len`, which depends on the first fold's
///   layout. A single "best fold" min would erase that dependence and
///   force the child into a locally-cheap choice that can balloon the
///   parent's intermediate proof beyond the savings. Listing one entry
///   per first-fold `log_basis` lets the parent enumerate child
///   options against its own proof formula. Empty when no fold
///   candidate produces a valid continuation.
#[derive(Clone)]
struct SuffixResult {
    best_direct: Option<(usize, Vec<Step>)>,
    best_fold_per_lb: BTreeMap<u32, (usize, Vec<Step>)>,
}

impl SuffixResult {
    fn is_empty(&self) -> bool {
        self.best_direct.is_none() && self.best_fold_per_lb.is_empty()
    }
}

type ScheduleMemo = HashMap<(usize, usize, usize, u32), SuffixResult>;

/// Suffix DP that searches for the optimal recursive schedule starting
/// at `(level, current_w_len, current_w_len_terminal, current_lb)`.
///
/// The DP carries **two** witness lengths through every state because
/// the witness shape leaving a fold depends on what its successor is:
///
/// - `current_w_len` — `MRowLayout::Intermediate` shape, the witness
///   length the prover ships into level `L` *if* level `L` is going to
///   fold further. Consumed by the fold branch.
/// - `current_w_len_terminal` — `MRowLayout::Terminal` shape, the
///   witness length the prover ships into level `L` *if* level `L` is
///   going to send the witness directly. The terminal layout drops the
///   D-block (and, under ZK, the D-blinding) from the parent's
///   M-matrix, so this is `<= current_w_len`.
///
/// The DP returns both [`SuffixResult::best_direct`] and
/// [`SuffixResult::best_fold`] separately, because the parent's
/// proof-size formula depends on which one is selected at this level:
/// a "fold-here-then-X" suffix is locally cheaper sometimes but forces
/// the parent into the larger intermediate proof formula, which can
/// dwarf the local savings. Letting the parent see both options
/// removes that bias.
///
/// At each state we evaluate:
///
/// - **`best_direct`**: ship the witness directly at this level under
///   `MRowLayout::Terminal`. Cost is the Terminal-shape witness bytes.
///   The terminal direct does not commit, so there is no SIS audit
///   here — `best_direct` is always present.
/// - **`best_fold`**: one fold candidate per `log_basis` (derived via
///   [`derive_candidate_level_params`], which threads
///   `optimal_m_r_split` internally to pick `(m_vars, r_vars)` and
///   computes both Intermediate and Terminal successor shapes). For
///   each candidate, we score both child-shape options (child is
///   Direct → `compute_level_proof_size(..., MRowLayout::Terminal)`;
///   child is Fold → `compute_level_proof_size(..., MRowLayout::Intermediate)`)
///   and keep the cheaper one.
///
/// The greedy "one candidate per `log_basis`" pre-selection is
/// structurally required for memo dedup: explicit `(num_blocks,
/// log_basis)` enumeration would produce a fresh `next_w_len` per
/// parent fold candidate and blow up the suffix-DP memo state space
/// exponentially in depth (the root path can afford explicit
/// enumeration only because it is entered once, not recursively).
fn derive_optimal_suffix_schedule<Cfg: CommitmentConfig>(
    memo: &mut ScheduleMemo,
    num_vars: usize,
    level: usize,
    current_w_len: usize,
    current_w_len_terminal: usize,
    current_lb: u32,
    depth: usize,
) -> Result<SuffixResult, AkitaError> {
    let memo_key = (level, current_w_len, current_w_len_terminal, current_lb);
    if depth <= MAX_RECURSION_DEPTH {
        if let Some(cached) = memo.get(&memo_key) {
            return Ok(cached.clone());
        }
    }

    let best_direct = {
        let witness_shape = DirectWitnessShape::PackedDigits((current_w_len_terminal, current_lb));
        let direct_bytes = direct_witness_bytes(Cfg::decomposition().field_bits(), &witness_shape);
        let step = Step::Direct(DirectStep {
            current_w_len: current_w_len_terminal,
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

    let mut best_fold_per_lb: BTreeMap<u32, (usize, Vec<Step>)> = BTreeMap::new();
    let (min_log_basis, max_log_basis) = Cfg::basis_range();
    for lb in min_log_basis..=max_log_basis {
        if lb < current_lb {
            continue;
        }
        let Some((lp, next_w_len, next_w_len_terminal)) =
            derive_candidate_level_params::<Cfg>(level, current_w_len, lb)?
        else {
            continue;
        };

        let child = derive_optimal_suffix_schedule::<Cfg>(
            memo,
            num_vars,
            level + 1,
            next_w_len,
            next_w_len_terminal,
            lb,
            depth + 1,
        )?;
        let Ok(eor_bytes) = extension_opening_reduction_level_bytes::<Cfg>(
            AkitaScheduleLookupKey::singleton(num_vars),
            level,
            current_w_len,
        ) else {
            continue;
        };

        // Best schedule with a Fold-at-this-level of `lb` as its first
        // step. Considers every grandchild-shape option (child=Direct
        // forces the terminal proof formula at this level; child=Fold
        // for each grandchild `first_lb` forces the intermediate proof
        // formula with the corresponding `next_lp`).
        let mut best_for_this_lb: Option<(usize, Vec<Step>)> = None;
        let try_update = |total: usize, steps: Vec<Step>, slot: &mut Option<(usize, Vec<Step>)>| {
            if slot.as_ref().map(|(c, _)| total < *c).unwrap_or(true) {
                *slot = Some((total, steps));
            }
        };

        // Branch A: child is a Direct at level+1. This level uses the
        // terminal proof formula.
        if let Some((child_cost, child_sched)) = child.best_direct.as_ref() {
            let level_proof_size = compute_level_proof_size::<Cfg>(
                &lp,
                &lp,
                next_w_len_terminal,
                1,
                MRowLayout::Terminal,
            ) + eor_bytes;
            let total = level_proof_size + child_cost;
            let mut steps = Vec::with_capacity(1 + child_sched.len());
            steps.push(Step::Fold(FoldStep {
                params: lp.clone(),
                current_w_len,
                next_w_len: next_w_len_terminal,
                level_bytes: level_proof_size,
            }));
            steps.extend(child_sched.iter().cloned());
            try_update(total, steps, &mut best_for_this_lb);
        }
        // Branch B: child is a Fold at level+1. Each child first_lb
        // gives a different `next_lp.b_key.row_len`, hence a different
        // intermediate proof at this level — so iterate them all
        // instead of pre-picking the child's local min.
        for (_child_first_lb, (child_cost, child_sched)) in child.best_fold_per_lb.iter() {
            let Step::Fold(child_fold_step) = &child_sched[0] else {
                unreachable!("best_fold_per_lb schedules start with Step::Fold");
            };
            let level_proof_size = compute_level_proof_size::<Cfg>(
                &lp,
                &child_fold_step.params,
                next_w_len,
                1,
                MRowLayout::Intermediate,
            ) + eor_bytes;
            let total = level_proof_size + child_cost;
            let mut steps = Vec::with_capacity(1 + child_sched.len());
            steps.push(Step::Fold(FoldStep {
                params: lp.clone(),
                current_w_len,
                next_w_len,
                level_bytes: level_proof_size,
            }));
            steps.extend(child_sched.iter().cloned());
            try_update(total, steps, &mut best_for_this_lb);
        }

        if let Some(entry) = best_for_this_lb {
            best_fold_per_lb.insert(lb, entry);
        }
    }

    let result = SuffixResult {
        best_direct,
        best_fold_per_lb,
    };
    memo.insert(memo_key, result.clone());
    Ok(result)
}

/// Consult the offline schedule tables for a pre-computed answer.
fn offline_schedule_for_key<Cfg: CommitmentConfig>(
    key: AkitaScheduleLookupKey,
    table: GeneratedScheduleTable,
) -> Result<Option<Schedule>, AkitaError> {
    // Materialization arithmetic is bit-width driven and uses
    // `root_decomp.field_bits()` everywhere, so the field marker is just
    // a phantom — pick a small canonical field that satisfies the bound.
    use akita_field::Prime128OffsetA7F7 as PhantomField;
    let plan = schedule_plan_from_table::<PhantomField, _>(
        key,
        table,
        PlanPolicy {
            sis_family: Cfg::sis_modulus_family(),
            ring_dimension: Cfg::D,
            root_decomp: Cfg::decomposition(),
            challenge_field_bits: Cfg::decomposition().field_bits() * Cfg::CHAL_EXT_DEGREE as u32,
            recursive_public_rows: 1,
            extension_opening_width: Cfg::CLAIM_EXT_DEGREE,
            stage1_challenge_config: Cfg::stage1_challenge_config,
            ring_subfield_norm_bound: Cfg::ring_subfield_embedding_norm_bound(),
            fold_challenge_shape: Cfg::fold_challenge_shape_at_level,
        },
    )?;
    Ok(plan.map(|plan| schedule_from_plan(&plan)))
}

/// Root-direct commitment `LevelParams` with a symmetric `(m, r)` split.
///
/// Root-direct schedules ship the cleartext witness on the wire, so
/// they don't run the relation fold (D unused). They still build the
/// outer Ajtai commitment `u = Σ B · t_i = Σ B · A · s_i`, so both A
/// (witness commit) and B (outer commit) need real SIS-secure sizing.
fn compute_root_direct_level_params<Cfg: CommitmentConfig>(
    num_vars: usize,
    log_basis: u32,
) -> Result<Option<LevelParams>, AkitaError> {
    let d = Cfg::D;
    let alpha = (d as u32).trailing_zeros() as usize;
    let reduced_vars = num_vars.saturating_sub(alpha);
    // Symmetric split. `r_vars = log_num_blocks`, `m_vars = log_block_len`.
    let r_vars = reduced_vars / 2;
    let m_vars = reduced_vars - r_vars;
    let Some(num_blocks) = 1usize.checked_shl(r_vars as u32) else {
        return Ok(None);
    };
    let Some(block_len) = 1usize.checked_shl(m_vars as u32) else {
        return Ok(None);
    };
    let Some(a_key) = compute_ajtai_key_params_a::<Cfg>(block_len, log_basis)? else {
        return Ok(None);
    };
    let Some(b_key) = compute_ajtai_key_params_b::<Cfg>(a_key.row_len(), num_blocks, 1, log_basis)?
    else {
        return Ok(None);
    };
    // D is unused at proof time for root-direct (no relation fold).
    // Placeholder zero-width key via `new_unchecked` to bypass the
    // strict SIS audit — never flows through `try_new` from here.
    let d_key = AjtaiKeyParams::new_unchecked(Cfg::sis_modulus_family(), 1, 0, 0, d);

    let witness_len = 1usize.checked_shl(num_vars as u32).unwrap_or(0);
    let stage1 = Cfg::stage1_challenge_config(d)?;
    let fold_shape = Cfg::fold_challenge_shape_at_level(AkitaScheduleInputs {
        num_vars,
        level: 0,
        current_w_len: witness_len,
    });
    Ok(Some(LevelParams {
        ring_dimension: d,
        log_basis,
        a_key,
        b_key,
        d_key,
        num_blocks,
        block_len,
        m_vars,
        r_vars,
        stage1_config: stage1.clone(),
        fold_challenge_shape: fold_shape,
        num_digits_commit: WitnessType::S.decomposed_num_digits::<Cfg>(log_basis),
        num_digits_open: WitnessType::T.decomposed_num_digits::<Cfg>(log_basis),
        // TODO: drop this fake fold-digit count once the ring-switch
        // validators are gated on schedule shape. Root-direct never
        // folds, but the verifier's ring-switch path currently rejects
        // any `LevelParams` with `num_digits_fold == 0`, so we compute
        // the canonical singleton value here just to keep that check
        // happy. The "real" value for root-direct is 0.
        num_digits_fold: WitnessType::Z.decomposed_fold_num_digits::<Cfg>(
            log_basis,
            r_vars,
            fold_shape.effective_l1_mass(&stage1),
            1,
        ),
    }))
}

/// Find the optimal schedule for a root schedule lookup key under `Cfg`.
///
/// When `use_lookup` is true the search consults `Cfg::schedule_table()`
/// as a fast path. Production callers pass `true`; the
/// `gen_schedule_tables` binary passes `false` so the output is a pure
/// function of `Cfg` itself.
///
/// # Errors
///
/// Returns an error if vector counts are invalid, if the witness length
/// overflows, or if generated-table materialization fails.
pub fn find_schedule<Cfg: CommitmentConfig>(
    key: AkitaScheduleLookupKey,
    use_lookup: bool,
) -> Result<Schedule, AkitaError> {
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

    if use_lookup {
        if let Some(table) = Cfg::schedule_table() {
            if let Some(schedule) = offline_schedule_for_key::<Cfg>(key, table)? {
                tracing::debug!(
                    ?key,
                    total_bytes = schedule.total_bytes,
                    "schedule planner: served from offline tables"
                );
                return Ok(schedule);
            }
        }
    }

    let witness_len = 1usize
        .checked_shl(key.num_vars as u32)
        .ok_or_else(|| AkitaError::InvalidSetup("witness too large".into()))?;

    let root_witness_shape = DirectWitnessShape::FieldElements(witness_len);
    let mut best_cost =
        direct_witness_bytes(Cfg::decomposition().field_bits(), &root_witness_shape);
    let root_direct_commit_params =
        compute_root_direct_level_params::<Cfg>(key.num_vars, Cfg::decomposition().log_basis)?;
    let mut best_steps: Vec<Step> = vec![Step::Direct(DirectStep {
        current_w_len: witness_len,
        witness_shape: root_witness_shape,
        direct_bytes: best_cost,
        params: root_direct_commit_params,
    })];
    let mut memo = ScheduleMemo::new();

    let stage1 = Cfg::stage1_challenge_config(Cfg::D)?;
    let fold_shape = Cfg::fold_challenge_shape_at_level(AkitaScheduleInputs {
        num_vars: key.num_vars,
        level: 0,
        current_w_len: witness_len,
    });
    let alpha = (Cfg::D as u32).trailing_zeros() as usize;
    let reduced_vars = key.num_vars.saturating_sub(alpha);

    if reduced_vars == 0 {
        return Ok(Schedule {
            steps: best_steps,
            total_bytes: best_cost,
        });
    }

    let min_r_vars: usize = if reduced_vars >= 3 { 1 } else { 0 };
    let max_r_vars: usize = (reduced_vars - 1).min(usize::BITS as usize - 1);

    let field_bits = Cfg::decomposition().field_bits();

    let (min_log_basis, max_log_basis) = Cfg::basis_range();
    for candidate_log_basis in min_log_basis..=max_log_basis {
        let num_digits_commit = WitnessType::S.decomposed_num_digits::<Cfg>(candidate_log_basis);
        let num_digits_open = WitnessType::T.decomposed_num_digits::<Cfg>(candidate_log_basis);

        for r_vars in min_r_vars..=max_r_vars {
            let num_blocks: usize = 1usize << r_vars;
            let m_vars = reduced_vars - r_vars;

            let Some(block_len) = 1usize.checked_shl(m_vars as u32) else {
                continue;
            };

            let Some(a_key) = compute_ajtai_key_params_a::<Cfg>(block_len, candidate_log_basis)?
            else {
                continue;
            };

            let Some(b_key) = compute_ajtai_key_params_b::<Cfg>(
                a_key.row_len(),
                num_blocks,
                t_vectors,
                candidate_log_basis,
            )?
            else {
                continue;
            };

            let Some(d_key) =
                compute_ajtai_key_params_d::<Cfg>(num_blocks, t_vectors, candidate_log_basis)?
            else {
                continue;
            };

            let num_digits_fold = WitnessType::Z.decomposed_fold_num_digits::<Cfg>(
                candidate_log_basis,
                r_vars,
                fold_shape.effective_l1_mass(&stage1),
                t_vectors,
            );

            let level_lp = LevelParams {
                ring_dimension: Cfg::D,
                log_basis: candidate_log_basis,
                a_key,
                b_key,
                d_key,
                num_blocks,
                block_len,
                m_vars,
                r_vars,
                stage1_config: stage1.clone(),
                fold_challenge_shape: fold_shape,
                num_digits_commit,
                num_digits_open,
                num_digits_fold,
            };

            let next_withness_len_impl = |layout| -> Result<usize, AkitaError> {
                let rings = w_ring_element_count_with_counts_for_layout_bits(
                    field_bits,
                    &level_lp,
                    key.num_points,
                    key.num_t_vectors,
                    key.num_w_vectors,
                    key.num_z_vectors,
                    layout,
                )?;
                rings.checked_mul(Cfg::D).ok_or_else(|| {
                    AkitaError::InvalidSetup("root next witness length overflow".into())
                })
            };
            let next_w_len = next_withness_len_impl(MRowLayout::Intermediate)?;
            let next_w_len_terminal = next_withness_len_impl(MRowLayout::Terminal)?;
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

            let child = derive_optimal_suffix_schedule::<Cfg>(
                &mut memo,
                key.num_vars,
                1,
                next_w_len,
                next_w_len_terminal,
                candidate_log_basis,
                0,
            )?;
            if child.is_empty() {
                continue;
            }
            let Ok(eor_bytes) = extension_opening_reduction_level_bytes::<Cfg>(key, 0, witness_len)
            else {
                continue;
            };

            // Branch A: suffix at level 1 is a Direct
            if let Some((child_cost, child_sched)) = child.best_direct.as_ref() {
                let root_proof_size = compute_level_proof_size::<Cfg>(
                    &level_lp,
                    &level_lp,
                    next_w_len_terminal,
                    z_vectors,
                    MRowLayout::Terminal,
                ) + eor_bytes;
                let total = root_proof_size + child_cost;
                if total < best_cost {
                    best_cost = total;
                    let mut steps = Vec::with_capacity(1 + child_sched.len());
                    steps.push(Step::Fold(FoldStep {
                        params: level_lp.clone(),
                        current_w_len: witness_len,
                        next_w_len: next_w_len_terminal,
                        level_bytes: root_proof_size,
                    }));
                    steps.extend(child_sched.iter().cloned());
                    best_steps = steps;
                }
            }
            // Branch B: suffix at level 1 is a Fold
            for (_child_first_lb, (child_cost, child_sched)) in child.best_fold_per_lb.iter() {
                let Step::Fold(child_fold_step) = &child_sched[0] else {
                    unreachable!("best_fold_per_lb schedules start with Step::Fold");
                };
                let root_proof_size = compute_level_proof_size::<Cfg>(
                    &level_lp,
                    &child_fold_step.params,
                    next_w_len,
                    z_vectors,
                    MRowLayout::Intermediate,
                ) + eor_bytes;
                let total = root_proof_size + child_cost;
                if total < best_cost {
                    best_cost = total;
                    let mut steps = Vec::with_capacity(1 + child_sched.len());
                    steps.push(Step::Fold(FoldStep {
                        params: level_lp.clone(),
                        current_w_len: witness_len,
                        next_w_len,
                        level_bytes: root_proof_size,
                    }));
                    steps.extend(child_sched.iter().cloned());
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
