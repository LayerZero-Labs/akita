//! Schedule planner that finds the global minimum proof size.
//!
//! A single exhaustive DP over `(level, w_len, log_basis)` states. At each
//! state, every feasible basis is tried; `level_proof_bytes` uses the
//! smallest `next_commit` across all next-level bases; the suffix is
//! recursed into unconstrained.
//!
//! Public entry: [`find_schedule`], `<Cfg>`-generic. When `use_lookup` is
//! true the search consults `Cfg::schedule_table()` (and seeds the DP with
//! the corresponding singleton plan / envelope floor) before running DP;
//! production callers pass `true`. The table-emitter binary passes
//! `false` to regenerate from scratch.

use std::collections::{BTreeMap, HashMap};

use akita_challenges::TensorChallengeShape;
use akita_config::CommitmentConfig;
use akita_field::AkitaError;
use akita_types::generated::sis_floor::{ceil_supported_collision, min_rank_for_secure_width};
use akita_types::generated::GeneratedScheduleTable;
use akita_types::layout::digit_math::{compute_num_digits_fold_with_claims, optimal_m_r_split};
use akita_types::{
    decomp_depths, direct_witness_bytes, extension_opening_reduction_proof_bytes,
    level_proof_bytes, root_extension_opening_partials, scale_batched_root_layout,
    schedule_from_plan, terminal_level_proof_bytes,
    w_ring_element_count_with_counts_for_layout_bits, AjtaiKeyParams, AkitaScheduleInputs,
    AkitaScheduleLookupKey, DecompositionParams, DirectStep, DirectWitnessShape, FoldStep,
    LevelParams, MRowLayout, Schedule, Step,
};

use akita_derive::{schedule_plan_from_table, PlanPolicy};

use crate::ajtai_params::{
    compute_ajtai_key_params_a, compute_ajtai_key_params_b, compute_ajtai_key_params_d, WitnessType,
};

const MAX_RECURSION_DEPTH: usize = 12;

// -----------------------------------------------------------------------
// Single-level evaluation
// -----------------------------------------------------------------------

/// All layout data for one candidate fold level.
struct CandidateLevelParams {
    /// Per-level layout used for both SIS-derived rank floors and
    /// proof-size accounting. Root candidates store the batched layout
    /// (widths scaled by `num_t_vectors`); recursive candidates store
    /// the per-level layout produced by [`derive_recursive_candidate_layout`].
    lp: LevelParams,
    /// Witness length entering the next level under `MRowLayout::Intermediate`,
    /// i.e. when the next level is another fold. Used to recurse into the
    /// suffix DP's fold branch.
    next_w_len: usize,
    /// Witness length entering the next level under `MRowLayout::Terminal`,
    /// i.e. when the next level is a direct-send. Always `<= next_w_len`
    /// because the terminal layout drops the D-block (and, under ZK, the
    /// D-blinding) from the M-matrix. Lets the suffix DP cost the direct
    /// branch correctly the first time it considers it, instead of emitting
    /// a placeholder under the intermediate shape and patching it later.
    next_w_len_terminal: usize,
}

/// Pick the recursive `LevelParams` for a `(current_w_len, log_basis)`
/// candidate as a single direct construction.
///
/// Single-shot: `optimal_m_r_split` derives `n_a` per candidate `r`
/// from the SIS-floor table and returns `(m_vars, r_vars, n_a)`
/// jointly. The remaining `(n_b, n_d)` widths are read straight off
/// the audited tables for the chosen geometry. All three Ajtai keys
/// and the layout coordinates flow into a single `LevelParams` struct
/// literal at the end.
///
/// Returns `Ok(None)` when the candidate is infeasible for any reason
/// (SIS-bound overflow, missing audited bucket / rank coverage,
/// non-divisible `current_w_len`, …). The planner DP treats this as
/// "skip this candidate".
fn derive_recursive_candidate_layout<Cfg: CommitmentConfig>(
    current_w_len: usize,
    log_basis: u32,
) -> Result<Option<LevelParams>, AkitaError> {
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

    let (m_vars, r_vars, picked_n_a) = optimal_m_r_split(
        sis_family,
        d as u32,
        collision_a,
        l1_mass,
        recursive_decomp.log_commit_bound,
        recursive_decomp.log_basis,
        reduced_vars,
        num_ring_elems,
        field_bits,
    );
    let num_blocks = 1usize
        .checked_shl(r_vars as u32)
        .ok_or_else(|| AkitaError::InvalidSetup("2^r_vars does not fit usize".to_string()))?;
    let block_len = num_ring_elems.div_ceil(num_blocks);
    let inner_width = block_len
        .checked_mul(num_digits_commit)
        .ok_or_else(|| AkitaError::InvalidSetup("inner width overflow".to_string()))?;
    let n_a = picked_n_a as usize;
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

    Ok(Some(LevelParams {
        ring_dimension: d,
        log_basis,
        a_key: AjtaiKeyParams::new_unchecked(sis_family, n_a, inner_width, collision_a, d),
        b_key: AjtaiKeyParams::new_unchecked(sis_family, n_b, outer_width, collision_bd, d),
        d_key: AjtaiKeyParams::new_unchecked(sis_family, n_d, d_matrix_width, collision_bd, d),
        num_blocks,
        block_len,
        m_vars,
        r_vars,
        stage1_config,
        fold_challenge_shape: TensorChallengeShape::Flat,
        num_digits_commit,
        num_digits_open,
        num_digits_fold,
    }))
}

/// Derive the layout for folding at `(level, w_len, log_basis)`.
/// Returns `None` if the layout is infeasible or doesn't shrink the witness.
///
/// Recursive levels (`level > 0`) only — the root candidate path in
/// `find_schedule` builds its own `LevelParams` directly from the
/// per-role SIS-floor lookups so it can enumerate `(num_blocks, log_basis)`
/// explicitly. Recursive levels go through
/// [`derive_recursive_candidate_layout`] (which threads
/// `optimal_m_r_split` internally) because explicit
/// `(num_blocks, log_basis)` enumeration here would expand the
/// suffix-DP memo state space exponentially in depth: every parent
/// fold's `next_w_len` becomes a fresh memo key, so the dedup that
/// makes the DP tractable disappears.
fn derive_candidate_level_params<Cfg: CommitmentConfig>(
    level: usize,
    current_w_len: usize,
    log_basis: u32,
) -> Result<Option<CandidateLevelParams>, AkitaError> {
    debug_assert!(
        level > 0,
        "derive_candidate_level_params is recursive-only; root candidates are built directly in find_schedule",
    );

    let Some(level_lp) = derive_recursive_candidate_layout::<Cfg>(current_w_len, log_basis)? else {
        return Ok(None);
    };

    let fb = Cfg::decomposition().field_bits();
    // Recursive folds carry one recursive witness and open it at one prepared
    // recursive point. Root batching is reflected only at level 0.
    //
    // We materialize both M-row layouts here so the suffix DP can cost
    // "fold then fold" (Intermediate) and "fold then direct" (Terminal)
    // against the same candidate without ever emitting a placeholder.
    let next_w_len = recursive_next_witness_len(&level_lp, fb, MRowLayout::Intermediate)?;
    let next_w_len_terminal = recursive_next_witness_len(&level_lp, fb, MRowLayout::Terminal)?;

    // Strict shrink check on bit count, evaluated on the Intermediate
    // shape. At recursive levels the witness arrives encoded in
    // `log_basis`-bit digits and is re-emitted at the same basis, so this
    // reduces to "field-element count strictly decreases" — but we keep
    // the bit form to mirror the root's check and to surface bit-length
    // overflows explicitly. We deliberately do NOT also gate on the
    // terminal shape: a candidate whose Intermediate shape fails to
    // shrink is structurally dominated by the direct baseline, but a
    // candidate whose Intermediate shape shrinks while its Terminal
    // shape does not is still useful for fold-then-fold-...-then-direct
    // chains.
    let next_bits = next_w_len
        .checked_mul(log_basis as usize)
        .ok_or_else(|| AkitaError::InvalidSetup("next witness bit length overflow".into()))?;
    let current_bits = current_w_len
        .checked_mul(log_basis as usize)
        .ok_or_else(|| AkitaError::InvalidSetup("current witness bit length overflow".into()))?;
    if next_bits >= current_bits {
        return Ok(None);
    }

    Ok(Some(CandidateLevelParams {
        lp: level_lp,
        next_w_len,
        next_w_len_terminal,
    }))
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

fn compute_level_proof_size<Cfg: CommitmentConfig>(
    candidate: &CandidateLevelParams,
    next_level_params: &LevelParams,
    num_public_outputs: usize,
) -> usize {
    level_proof_bytes(
        Cfg::decomposition().field_bits(),
        Cfg::decomposition().field_bits() * Cfg::CHAL_EXT_DEGREE as u32,
        &candidate.lp,
        &candidate.lp,
        next_level_params,
        candidate.next_w_len,
        num_public_outputs,
    )
}

fn compute_terminal_level_proof_size<Cfg: CommitmentConfig>(
    candidate: &CandidateLevelParams,
    terminal_next_w_len: usize,
    num_public_outputs: usize,
) -> usize {
    terminal_level_proof_bytes(
        Cfg::decomposition().field_bits(),
        Cfg::decomposition().field_bits() * Cfg::CHAL_EXT_DEGREE as u32,
        &candidate.lp,
        terminal_next_w_len,
        num_public_outputs,
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
// Step construction
// -----------------------------------------------------------------------

fn to_fold_step(
    c: &CandidateLevelParams,
    current_w_len: usize,
    level_bytes: usize,
    field_bits: u32,
    next_w_len_override: Option<usize>,
) -> Step {
    let per_poly_fold = compute_num_digits_fold_with_claims(
        c.lp.r_vars,
        c.lp.challenge_l1_mass(),
        c.lp.log_basis,
        1,
        field_bits,
    );
    let next_w_len = next_w_len_override.unwrap_or(c.next_w_len);
    let w_ring = next_w_len / c.lp.ring_dimension;
    Step::Fold(FoldStep {
        params: c.lp.clone(),
        current_w_len,
        delta_fold_per_poly: per_poly_fold,
        w_ring,
        next_w_len,
        level_bytes,
    })
}

/// Build a SIS-secure terminal direct step in one shot.
///
/// The DP at level `level` reaches this state with two witness lengths:
///
/// - `current_w_len` (Intermediate-layout) — the witness shape if the
///   next step is another fold; consumed by the fold branch.
/// - `current_w_len_terminal` (Terminal-layout) — the witness shape if
///   the next step is *this* direct send. The terminal layout drops the
///   D-block from the parent's M-matrix (and the D-blinding under ZK),
///   so `current_w_len_terminal <= current_w_len`.
///
/// This constructor uses the Terminal value to (a) cost the direct
/// bytes correctly and (b) derive SIS-secure `level_params` for the
/// direct's commitment. Returns `Ok(None)` when SIS is infeasible at
/// the audited floor; the DP treats this as "no direct option from
/// this state".
fn to_direct_step<Cfg: CommitmentConfig>(
    num_vars: usize,
    level: usize,
    current_w_len_terminal: usize,
    log_basis: u32,
) -> Result<Option<Step>, AkitaError> {
    let witness_shape = DirectWitnessShape::PackedDigits((current_w_len_terminal, log_basis));
    let direct_bytes = direct_witness_bytes(Cfg::decomposition().field_bits(), &witness_shape);
    let level_params = match akita_derive::direct_level_params_with_log_basis(
        Cfg::sis_modulus_family(),
        Cfg::D,
        Cfg::decomposition(),
        Cfg::stage1_challenge_config(Cfg::D)?,
        Cfg::ring_subfield_embedding_norm_bound(),
        AkitaScheduleInputs {
            num_vars,
            level,
            current_w_len: current_w_len_terminal,
        },
        log_basis,
    ) {
        Ok(lp) => lp,
        // SIS-infeasible at the audited floor for this geometry —
        // matches the prior lazy-finalize behavior of abandoning the
        // candidate path, only decided one level earlier.
        Err(_) => return Ok(None),
    };
    Ok(Some(Step::Direct(DirectStep {
        current_w_len: current_w_len_terminal,
        witness_shape,
        direct_bytes,
        // Terminal Direct after one or more folds: stores the SIS-secure
        // level params for this direct's commitment. (Root commit
        // layout lives on the first `FoldStep`, not on this step.)
        params: Some(level_params),
    })))
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
///   `compute_terminal_level_proof_size` (drops the v-rows and stage-1
///   sumcheck at the parent level). `None` when SIS is infeasible at
///   this state.
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
/// - **`best_direct`**: ship the witness directly at this level. Cost
///   is the Terminal-shape witness bytes; SIS feasibility is verified
///   eagerly by [`to_direct_step`]. An SIS-infeasible state has
///   `best_direct = None`.
/// - **`best_fold`**: one fold candidate per `log_basis` (derived via
///   [`derive_candidate_level_params`], which threads
///   `optimal_m_r_split` internally to pick `(m_vars, r_vars)` and
///   computes both Intermediate and Terminal successor shapes). For
///   each candidate, we score both child-shape options (child is
///   Direct → use `compute_terminal_level_proof_size`; child is Fold
///   → use `compute_level_proof_size`) and keep the cheaper one.
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

    // best_direct: try to build the SIS-secure direct step eagerly. If
    // SIS is infeasible the direct option simply does not exist at
    // this state.
    let best_direct =
        match to_direct_step::<Cfg>(num_vars, level, current_w_len_terminal, current_lb)? {
            Some(step) => {
                let bytes = match &step {
                    Step::Direct(d) => d.direct_bytes,
                    Step::Fold(_) => unreachable!("to_direct_step returns Step::Direct"),
                };
                Some((bytes, vec![step]))
            }
            None => None,
        };

    if depth > MAX_RECURSION_DEPTH {
        let result = SuffixResult {
            best_direct,
            best_fold_per_lb: BTreeMap::new(),
        };
        memo.insert(memo_key, result.clone());
        return Ok(result);
    }

    let field_bits = Cfg::decomposition().field_bits();
    let mut best_fold_per_lb: BTreeMap<u32, (usize, Vec<Step>)> = BTreeMap::new();
    let (min_log_basis, max_log_basis) = Cfg::basis_range();
    for lb in min_log_basis..=max_log_basis {
        if lb < current_lb {
            continue;
        }
        let Some(candidate) = derive_candidate_level_params::<Cfg>(level, current_w_len, lb)?
        else {
            continue;
        };

        let child = derive_optimal_suffix_schedule::<Cfg>(
            memo,
            num_vars,
            level + 1,
            candidate.next_w_len,
            candidate.next_w_len_terminal,
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
            let level_proof_size = compute_terminal_level_proof_size::<Cfg>(
                &candidate,
                candidate.next_w_len_terminal,
                1,
            ) + eor_bytes;
            let total = level_proof_size + child_cost;
            let mut steps = Vec::with_capacity(1 + child_sched.len());
            steps.push(to_fold_step(
                &candidate,
                current_w_len,
                level_proof_size,
                field_bits,
                Some(candidate.next_w_len_terminal),
            ));
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
            let level_proof_size =
                compute_level_proof_size::<Cfg>(&candidate, &child_fold_step.params, 1) + eor_bytes;
            let total = level_proof_size + child_cost;
            let mut steps = Vec::with_capacity(1 + child_sched.len());
            steps.push(to_fold_step(
                &candidate,
                current_w_len,
                level_proof_size,
                field_bits,
                None,
            ));
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
    Ok(plan.map(|plan| schedule_from_plan(&plan, Cfg::decomposition().field_bits())))
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
    // Populate `commit_params` so consumers don't re-derive the root
    // commit layout from the schedule shape. Large `num_vars` baselines
    // whose root layout exceeds the audited SIS-floor are still valid as
    // a proof-bytes upper bound for the DP comparator; they record
    // `commit_params: None`.
    let root_direct_commit_params = {
        let singleton = akita_derive::root_direct_commit_layout(
            Cfg::sis_modulus_family(),
            Cfg::D,
            Cfg::decomposition(),
            Cfg::stage1_challenge_config(Cfg::D)?,
            Cfg::ring_subfield_embedding_norm_bound(),
            key.num_vars,
            Cfg::decomposition().log_basis,
        )
        .ok();
        // Apply the same batched-root scaling as the Fold-root path so
        // `commit_params` carries the correct width for non-singleton
        // incidences.
        let root_is_batched = num_points != 1 || t_vectors != 1 || w_vectors != 1 || z_vectors != 1;
        match singleton {
            Some(lp) if root_is_batched => {
                // Scaling the singleton root layout multiplies the B/D
                // widths by `t_vectors` and re-runs `try_new` against
                // the original ranks. The audit (post-hardening) may
                // reject when those ranks no longer cover the batched
                // widths. That's not an error — it just means the
                // direct-baseline cannot be carried as a layout-typed
                // hint; the schedule still falls back to `None` and
                // the direct-witness bytes remain a valid DP upper
                // bound.
                scale_batched_root_layout(&lp, t_vectors, Cfg::decomposition().field_bits()).ok()
            }
            other => other,
        }
    };
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

            let candidate = CandidateLevelParams {
                lp: level_lp,
                next_w_len,
                next_w_len_terminal,
            };

            let child = derive_optimal_suffix_schedule::<Cfg>(
                &mut memo,
                key.num_vars,
                1,
                candidate.next_w_len,
                candidate.next_w_len_terminal,
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
                let root_proof_size = compute_terminal_level_proof_size::<Cfg>(
                    &candidate,
                    candidate.next_w_len_terminal,
                    z_vectors,
                ) + eor_bytes;
                let total = root_proof_size + child_cost;
                if total < best_cost {
                    best_cost = total;
                    let mut steps = Vec::with_capacity(1 + child_sched.len());
                    steps.push(to_fold_step(
                        &candidate,
                        witness_len,
                        root_proof_size,
                        field_bits,
                        Some(candidate.next_w_len_terminal),
                    ));
                    steps.extend(child_sched.iter().cloned());
                    best_steps = steps;
                }
            }
            // Branch B: suffix at level 1 is a Fold
            for (_child_first_lb, (child_cost, child_sched)) in child.best_fold_per_lb.iter() {
                let Step::Fold(child_fold_step) = &child_sched[0] else {
                    unreachable!("best_fold_per_lb schedules start with Step::Fold");
                };
                let root_proof_size =
                    compute_level_proof_size::<Cfg>(&candidate, &child_fold_step.params, z_vectors)
                        + eor_bytes;
                let total = root_proof_size + child_cost;
                if total < best_cost {
                    best_cost = total;
                    let mut steps = Vec::with_capacity(1 + child_sched.len());
                    steps.push(to_fold_step(
                        &candidate,
                        witness_len,
                        root_proof_size,
                        field_bits,
                        None,
                    ));
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
