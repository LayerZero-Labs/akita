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

use std::collections::HashMap;

use akita_challenges::TensorChallengeShape;
use akita_config::CommitmentConfig;
use akita_field::AkitaError;
use akita_types::generated::sis_floor::{
    ceil_supported_collision, min_rank_for_secure_width, sis_max_widths,
};
use akita_types::generated::GeneratedScheduleTable;
use akita_types::layout::digit_math::{
    compute_num_digits_fold_with_claims, compute_num_digits_full_field, num_digits_for_bound,
    optimal_m_r_split,
};
use akita_types::{
    decomp_depths, direct_witness_bytes, extension_opening_reduction_proof_bytes,
    level_proof_bytes, root_extension_opening_partials, scale_batched_root_layout,
    schedule_from_plan, terminal_level_proof_bytes, w_ring_element_count_with_counts_bits,
    w_ring_element_count_with_counts_for_layout_bits, AjtaiKeyParams, AjtaiRole,
    AkitaScheduleInputs, AkitaScheduleLookupKey, CommitmentEnvelope, DecompositionParams,
    DirectStep, DirectWitnessShape, FoldStep, LevelParams, MRowLayout, Schedule, Step,
};

use akita_derive::{schedule_plan_from_table, PlanPolicy};

const MAX_RECURSION_DEPTH: usize = 12;

/// Iteration cap for the recursive-level `n_a` fixed point inside
/// [`derive_recursive_candidate_layout`].
///
/// Each iteration grows `candidate_n_a` monotonically (both
/// `optimal_m_r_split`'s `block_len` and `min_rank_for_secure_width` are
/// monotone non-decreasing in `n_a`), and the audited SIS-floor rank
/// tables have at most ~32 rows per bucket, so the fixed point
/// converges well within this bound. Non-convergence is treated as "no
/// SIS-secure layout exists for this state" and falls through to the
/// envelope-rank cost-estimate fallback.
const RECURSIVE_RANK_ITERATION_CAP: usize = 64;

/// Build the [`CommitmentEnvelope`] the planner uses at `num_vars` under
/// `Cfg`.
///
/// When `use_lookup` is true, the envelope is the one `Cfg` ships for
/// production. When `use_lookup` is false (the `gen_schedule_tables`
/// idempotency path), the envelope is reconstructed from the audited
/// SIS-floor rank accessors alone so the planner cannot read back any
/// value that came from a stored schedule entry — otherwise table
/// regeneration would not be a fixed point.
fn planner_envelope<Cfg: CommitmentConfig>(
    num_vars: usize,
    use_lookup: bool,
) -> CommitmentEnvelope {
    if use_lookup {
        Cfg::envelope(num_vars)
    } else {
        let inner = Cfg::audited_root_rank(AjtaiRole::Inner, num_vars);
        let outer = Cfg::audited_root_rank(AjtaiRole::Outer, num_vars);
        CommitmentEnvelope {
            max_n_a: inner,
            max_n_b: outer,
            max_n_d: outer,
        }
    }
}

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
    next_w_len: usize,
}

/// Pick the recursive `LevelParams` for a `(current_w_len, log_basis)`
/// candidate as a single direct construction.
///
/// This is the fused inline of what used to be three chained
/// constructors:
///
/// - `sis_derived_recursive_params` (fixed-point on `n_a` to find a
///   self-consistent SIS-secure rank for the current `(num_blocks,
///   block_len)` choice),
/// - `LevelParams::params_only(envelope ranks)` (cost-estimate fallback
///   when SIS coverage is missing),
/// - `recursive_level_layout_from_params` (`optimal_m_r_split` →
///   `with_decomp` to populate layout fields).
///
/// Each of those steps allocated its own intermediate `LevelParams`;
/// they all collapse here into a single struct literal at the bottom.
///
/// **SIS-secure path** (preferred): iterate `n_a` until
/// `min_rank_for_secure_width(a_collision, inner_width(n_a)) <= n_a`.
/// Both `optimal_m_r_split`'s `block_len` and `min_rank_for_secure_width`
/// are monotone non-decreasing in `n_a`, so the fixed point converges
/// in a small constant number of iterations. After convergence, the
/// outer/D-matrix widths and the corresponding `n_b` / `n_d` are
/// derived from the same audited tables.
///
/// **Envelope fallback** (cost-estimate only): if any SIS-floor lookup
/// or shift overflows (or the fixed point fails to converge), build the
/// layout against `envelope.max_{n_a, n_b, n_d}` with `collision_inf =
/// 0`. The strict `AjtaiKeyParams::try_new` audit rejects this layout,
/// but the planner only consumes it as a proof-bytes upper bound for
/// the DP; if the schedule materializer or prover later picks a
/// candidate that landed on the fallback, they re-derive against the
/// SIS-strict tables.
fn derive_recursive_candidate_layout<Cfg: CommitmentConfig>(
    envelope: &CommitmentEnvelope,
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
    // Recursive levels always run with the flat-shape stage-1
    // challenge — `LevelParams::params_only` (the seed used by every
    // SIS-derivation helper) hard-codes `TensorChallengeShape::Flat`.
    let l1_mass = TensorChallengeShape::Flat.effective_l1_mass(&stage1_config);

    // Helper: given a candidate `n_a`, derive the block geometry that
    // an `optimal_m_r_split`-driven recursive layout would pick for it.
    let geometry_for = |n_a: usize| -> Result<(usize, usize, usize, usize, usize), AkitaError> {
        let (m_vars, r_vars) = optimal_m_r_split(
            n_a as u32,
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
        Ok((m_vars, r_vars, num_blocks, block_len, inner_width))
    };

    // Try the SIS-floor secure path. Success returns
    // `(n_a, n_b, n_d, a_collision, bd_collision)` for the converged
    // layout; `None` falls through to the envelope fallback below.
    let secure = (|| -> Option<(usize, usize, usize, u32, u32)> {
        let bd_collision = 1u32.checked_shl(log_basis)?.checked_sub(1)?;
        let a_collision_raw = bd_collision
            .checked_mul(stage1_config.infinity_norm())?
            .checked_mul(Cfg::ring_subfield_embedding_norm_bound())?;
        let a_collision = ceil_supported_collision(sis_family, d as u32, a_collision_raw)?;

        // Fixed point on `n_a`. Once `secure_n_a <= candidate_n_a` the
        // converged rank is `secure_n_a`; we then re-derive geometry
        // against this final `n_a` (matches the historical behaviour
        // where the outer `recursive_level_layout_from_params` re-ran
        // `optimal_m_r_split` with `derived.a_key.row_len()`).
        let mut candidate_n_a = envelope.max_n_a.max(1);
        let n_a = (0..RECURSIVE_RANK_ITERATION_CAP).find_map(|_| {
            let (_, _, _, _, inner_width) = geometry_for(candidate_n_a).ok()?;
            let secure_n_a =
                min_rank_for_secure_width(sis_family, d as u32, a_collision, inner_width as u64)?
                    .max(envelope.max_n_a);
            if secure_n_a <= candidate_n_a {
                Some(secure_n_a)
            } else {
                candidate_n_a = secure_n_a;
                None
            }
        })?;
        let (_, _, num_blocks, _, _) = geometry_for(n_a).ok()?;
        let outer_width = n_a.checked_mul(num_digits_open)?.checked_mul(num_blocks)?;
        let d_matrix_width = num_digits_open.checked_mul(num_blocks)?;
        let n_b =
            min_rank_for_secure_width(sis_family, d as u32, bd_collision, outer_width as u64)?
                .max(envelope.max_n_b);
        let n_d =
            min_rank_for_secure_width(sis_family, d as u32, bd_collision, d_matrix_width as u64)?
                .max(envelope.max_n_d);
        Some((n_a, n_b, n_d, a_collision, bd_collision))
    })();

    // `n_a` (and the geometry it induces) is the only thing the SIS
    // path has to fix-point on. The remaining ranks (`n_b`, `n_d`) and
    // both collision buckets are read straight off the audited tables
    // for the converged geometry — or zeroed when we fall back.
    let (n_a, n_b, n_d, collision_a, collision_bd) =
        secure.unwrap_or((envelope.max_n_a, envelope.max_n_b, envelope.max_n_d, 0, 0));
    let (m_vars, r_vars, num_blocks, block_len, inner_width) = geometry_for(n_a)?;
    let outer_width = n_a
        .checked_mul(num_digits_open)
        .and_then(|w| w.checked_mul(num_blocks))
        .ok_or_else(|| AkitaError::InvalidSetup("outer width overflow".to_string()))?;
    let d_matrix_width = num_digits_open
        .checked_mul(num_blocks)
        .ok_or_else(|| AkitaError::InvalidSetup("D-matrix width overflow".to_string()))?;
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
    envelope: &CommitmentEnvelope,
    level: usize,
    current_w_len: usize,
    log_basis: u32,
) -> Result<Option<CandidateLevelParams>, AkitaError> {
    debug_assert!(
        level > 0,
        "derive_candidate_level_params is recursive-only; root candidates are built directly in find_schedule",
    );

    let Some(level_lp) =
        derive_recursive_candidate_layout::<Cfg>(envelope, current_w_len, log_basis)?
    else {
        return Ok(None);
    };

    let fb = Cfg::decomposition().field_bits();
    // Recursive folds carry one recursive witness and open it at one prepared
    // recursive point. Root batching is reflected only at level 0.
    let w_ring_elements = w_ring_element_count_with_counts_bits(fb, &level_lp, 1, 1, 1, 1)?;
    let next_w_len = w_ring_elements
        .checked_mul(level_lp.ring_dimension)
        .ok_or_else(|| AkitaError::InvalidSetup("recursive witness length overflow".into()))?;

    // Strict shrink check on bit count: at recursive levels the witness
    // arrives encoded in `log_basis`-bit digits and is re-emitted at the
    // same basis, so this reduces to "field-element count strictly
    // decreases" — but we keep the bit form to mirror the root's check
    // and to surface bit-length overflows explicitly.
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
    }))
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

/// Initial (placeholder) witness shape for a terminal direct step recorded
/// during the suffix DP. The DP records this when transitioning into the
/// terminal base case using only `(current_w_len, log_basis)`; the FINAL
/// shape (computed from the last fold's `lp` under [`MRowLayout::Terminal`])
/// overwrites this once the enclosing fold candidate is known.
fn to_direct_step<Cfg: CommitmentConfig>(current_w_len: usize, log_basis: u32) -> Step {
    let witness_shape = DirectWitnessShape::PackedDigits((current_w_len, log_basis));
    let direct_bytes = direct_witness_bytes(Cfg::decomposition().field_bits(), &witness_shape);
    Step::Direct(DirectStep {
        current_w_len,
        witness_shape,
        direct_bytes,
        // Terminal Direct after one or more folds: root commit params live
        // on the first `FoldStep`, not on this terminal step.
        commit_params: None,
        // Populated by `finalize_terminal_direct_witness_shape` once the
        // enclosing fold candidate (and therefore `terminal_field_len`)
        // is known.
        level_params: None,
    })
}

#[allow(clippy::too_many_arguments)]
fn finalize_terminal_direct_witness_shape<Cfg: CommitmentConfig>(
    envelope: &CommitmentEnvelope,
    num_vars: usize,
    suffix_steps: &mut [Step],
    candidate: &CandidateLevelParams,
    num_points: usize,
    num_t_vectors: usize,
    num_w_vectors: usize,
    num_public_rows: usize,
    fold_level: usize,
) -> Result<(), AkitaError> {
    if suffix_steps.len() != 1 {
        return Err(AkitaError::InvalidSetup(
            "terminal direct finalizer expects exactly one suffix step".to_string(),
        ));
    }
    let first = suffix_steps.first_mut().ok_or_else(|| {
        AkitaError::InvalidSetup("terminal direct finalizer received empty suffix".to_string())
    })?;
    let Step::Direct(direct) = first else {
        return Err(AkitaError::InvalidSetup(
            "terminal direct finalizer expected a direct suffix step".to_string(),
        ));
    };
    let DirectWitnessShape::PackedDigits((_, log_basis)) = direct.witness_shape else {
        return Err(AkitaError::InvalidSetup(
            "terminal direct finalizer expected a packed-digit witness".to_string(),
        ));
    };
    let ring_count = w_ring_element_count_with_counts_for_layout_bits(
        Cfg::decomposition().field_bits(),
        &candidate.lp,
        num_points,
        num_t_vectors,
        num_w_vectors,
        num_public_rows,
        MRowLayout::Terminal,
    )
    .expect("terminal recursive witness length overflow");
    let terminal_field_len = ring_count
        .checked_mul(candidate.lp.ring_dimension)
        .expect("terminal recursive witness length overflow");
    let witness_shape = DirectWitnessShape::PackedDigits((terminal_field_len, log_basis));
    let direct_bytes = direct_witness_bytes(Cfg::decomposition().field_bits(), &witness_shape);
    // Bake the SIS-secure terminal-direct level params onto the step so
    // prover/verifier (and the materializer, when this candidate is
    // emitted via the offline table) can read them straight from the
    // schedule.
    let level_params = akita_derive::direct_level_params_with_log_basis(
        Cfg::sis_modulus_family(),
        Cfg::D,
        Cfg::decomposition(),
        Cfg::stage1_challenge_config(Cfg::D)?,
        Cfg::ring_subfield_embedding_norm_bound(),
        envelope,
        AkitaScheduleInputs {
            num_vars,
            level: fold_level + 1,
            current_w_len: terminal_field_len,
        },
        log_basis,
    )?;
    direct.current_w_len = terminal_field_len;
    direct.witness_shape = witness_shape;
    direct.direct_bytes = direct_bytes;
    direct.level_params = Some(level_params);
    Ok(())
}

fn level_params_from_fold_step<Cfg: CommitmentConfig>(step: &FoldStep) -> LevelParams {
    if let Ok(config) = Cfg::stage1_challenge_config(step.params.ring_dimension) {
        debug_assert_eq!(config.l1_norm(), step.params.challenge_l1_mass());
    }
    step.params.clone()
}

fn successor_level_params_from_schedule<Cfg: CommitmentConfig>(
    envelope: &CommitmentEnvelope,
    num_vars: usize,
    level: usize,
    current_w_len: usize,
    suffix_steps: &[Step],
) -> Result<LevelParams, AkitaError> {
    match suffix_steps
        .first()
        .expect("optimal suffix schedule must contain at least one step")
    {
        Step::Fold(step) => Ok(level_params_from_fold_step::<Cfg>(step)),
        Step::Direct(step) => akita_derive::direct_level_params_with_log_basis(
            Cfg::sis_modulus_family(),
            Cfg::D,
            Cfg::decomposition(),
            Cfg::stage1_challenge_config(Cfg::D)?,
            Cfg::ring_subfield_embedding_norm_bound(),
            envelope,
            AkitaScheduleInputs {
                num_vars,
                level,
                current_w_len,
            },
            step.log_basis(Cfg::decomposition().field_bits()),
        ),
    }
}

// -----------------------------------------------------------------------
// DP — suffix search
// -----------------------------------------------------------------------

type ScheduleMemo = HashMap<(usize, usize, u32), (usize, Vec<Step>)>;

#[allow(clippy::too_many_arguments)]
/// Suffix DP that searches for the optimal recursive schedule starting
/// at `(level, current_w_len, current_lb)`.
///
/// At each state we consider:
///
/// - the **direct fallback** — ship the witness directly at this level.
///   Cost is just the witness bytes; SIS feasibility is verified lazily
///   by the parent fold's call to
///   [`finalize_terminal_direct_witness_shape`] when this direct is the
///   chosen terminal suffix. No upfront
///   `direct_level_params_with_log_basis` probe — the same SIS-secure
///   derivation runs unconditionally during finalize, so probing here
///   would just duplicate the work and discard it.
/// - one **fold candidate per `log_basis`** — derived via
///   [`derive_candidate_level_params`], which threads
///   `optimal_m_r_split` internally to pick `(m_vars, r_vars)`. The
///   greedy pre-selection is structurally required for memo dedup:
///   explicit `(num_blocks, log_basis)` enumeration would produce a
///   fresh `next_w_len` per parent fold candidate and blow up the
///   suffix-DP memo state space exponentially in depth (the root path
///   can afford explicit enumeration only because it is entered once,
///   not recursively).
fn derive_optimal_suffix_schedule<Cfg: CommitmentConfig>(
    envelope: &CommitmentEnvelope,
    memo: &mut ScheduleMemo,
    num_vars: usize,
    level: usize,
    current_w_len: usize,
    current_lb: u32,
    depth: usize,
) -> Result<(usize, Vec<Step>), AkitaError> {
    let memo_key = (level, current_w_len, current_lb);
    if depth <= MAX_RECURSION_DEPTH {
        if let Some(cached) = memo.get(&memo_key) {
            return Ok(cached.clone());
        }
    }

    // Direct fallback baseline: cost is just the witness bytes. SIS
    // feasibility is checked lazily by the parent fold's call to
    // `finalize_terminal_direct_witness_shape` when this direct ends up
    // as the chosen terminal suffix.
    let placeholder = to_direct_step::<Cfg>(current_w_len, current_lb);
    let placeholder_cost = match &placeholder {
        Step::Direct(direct) => direct.direct_bytes,
        Step::Fold(_) => unreachable!("to_direct_step returns Step::Direct"),
    };
    let mut best_cost = placeholder_cost;
    let mut best_schedule = vec![placeholder];

    if depth > MAX_RECURSION_DEPTH {
        return Ok((best_cost, best_schedule));
    }

    let (min_log_basis, max_log_basis) = Cfg::basis_range();
    for lb in min_log_basis..=max_log_basis {
        if lb < current_lb {
            continue;
        }
        let Some(candidate) =
            derive_candidate_level_params::<Cfg>(envelope, level, current_w_len, lb)?
        else {
            continue;
        };

        let (mut suffix_cost, mut suffix_steps) = derive_optimal_suffix_schedule::<Cfg>(
            envelope,
            memo,
            num_vars,
            level + 1,
            candidate.next_w_len,
            lb,
            depth + 1,
        )?;
        if suffix_steps.is_empty() {
            continue;
        }
        let suffix_is_terminal = matches!(suffix_steps.first(), Some(Step::Direct(_)));
        let next_w_len_override = if suffix_is_terminal {
            let old_direct_bytes = match suffix_steps.first().expect("suffix non-empty") {
                Step::Direct(direct) => direct.direct_bytes,
                Step::Fold(_) => unreachable!("suffix_is_terminal guard"),
            };
            // Lazy SIS check: skip this fold candidate if the terminal
            // direct cannot be SIS-secured under the candidate's
            // `LevelParams`. Replaces the previous unconditional `?`,
            // which would have failed the entire search rather than
            // discarding one infeasible candidate, and replaces the
            // upfront `direct_level_params_with_log_basis` probe — that
            // probe was redundant work because
            // `finalize_terminal_direct_witness_shape` re-runs the same
            // SIS-secure derivation here.
            if finalize_terminal_direct_witness_shape::<Cfg>(
                envelope,
                num_vars,
                &mut suffix_steps,
                &candidate,
                1,
                1,
                1,
                1,
                level,
            )
            .is_err()
            {
                continue;
            }
            let (new_direct_bytes, terminal_field_len) =
                match suffix_steps.first().expect("suffix non-empty") {
                    Step::Direct(direct) => (direct.direct_bytes, direct.current_w_len),
                    Step::Fold(_) => unreachable!("suffix_is_terminal guard"),
                };
            suffix_cost = suffix_cost + new_direct_bytes - old_direct_bytes;
            Some(terminal_field_len)
        } else {
            None
        };
        let Ok(eor_bytes) = extension_opening_reduction_level_bytes::<Cfg>(
            AkitaScheduleLookupKey::singleton(num_vars),
            level,
            current_w_len,
        ) else {
            continue;
        };
        let level_proof_size = if suffix_is_terminal {
            let terminal_next_w_len = next_w_len_override
                .expect("suffix_is_terminal branch populates next_w_len_override above");
            compute_terminal_level_proof_size::<Cfg>(&candidate, terminal_next_w_len, 1) + eor_bytes
        } else {
            let Ok(next_level_params) = successor_level_params_from_schedule::<Cfg>(
                envelope,
                num_vars,
                level + 1,
                candidate.next_w_len,
                &suffix_steps,
            ) else {
                continue;
            };
            compute_level_proof_size::<Cfg>(&candidate, &next_level_params, 1) + eor_bytes
        };

        let total = level_proof_size + suffix_cost;
        if total < best_cost {
            best_cost = total;
            let mut steps = Vec::with_capacity(1 + suffix_steps.len());
            steps.push(to_fold_step(
                &candidate,
                current_w_len,
                level_proof_size,
                Cfg::decomposition().field_bits(),
                next_w_len_override,
            ));
            steps.extend(suffix_steps);
            best_schedule = steps;
        }
    }

    memo.insert(memo_key, (best_cost, best_schedule.clone()));
    Ok((best_cost, best_schedule))
}

// -----------------------------------------------------------------------
// Key-derived root sizing
// -----------------------------------------------------------------------

fn root_next_witness_len<Cfg: CommitmentConfig>(
    lp: &LevelParams,
    key: AkitaScheduleLookupKey,
) -> Result<usize, AkitaError> {
    let fb = Cfg::decomposition().field_bits();
    let r_decomp = compute_num_digits_full_field(fb, lp.log_basis);

    let t_vectors = key.num_t_vectors;
    let w_vectors = key.num_w_vectors;
    let z_vectors = key.num_z_vectors;
    let num_points = key.num_points;

    let w_hat = w_vectors * lp.num_blocks * lp.num_digits_open;
    let t_hat = t_vectors * lp.num_blocks * lp.a_key.row_len() * lp.num_digits_open;
    let z_pre = z_vectors * lp.inner_width() * lp.num_digits_fold;
    let r_rows = lp.m_row_count(num_points, z_vectors)?;
    let r = r_rows * r_decomp;

    #[cfg(feature = "zk")]
    let raw_w_ring = w_hat
        + t_hat
        + {
            let d_blinding = akita_types::zk::blinding_column_count_from_bits(
                lp.d_key.row_len(),
                lp.ring_dimension,
                lp.log_basis,
                fb as usize,
            );
            let b_blinding = num_points
                * akita_types::zk::blinding_column_count_from_bits(
                    lp.b_key.row_len(),
                    lp.ring_dimension,
                    lp.log_basis,
                    fb as usize,
                );
            b_blinding + d_blinding
        }
        + z_pre
        + r;
    #[cfg(not(feature = "zk"))]
    let raw_w_ring = w_hat + t_hat + z_pre + r;

    raw_w_ring
        .checked_mul(lp.ring_dimension)
        .ok_or_else(|| AkitaError::InvalidSetup("root recursive witness length overflow".into()))
}

// -----------------------------------------------------------------------
// Root-level candidate helpers
// -----------------------------------------------------------------------

/// Smallest SIS-secure rank for `width` under a pre-cached rank table.
///
/// Equivalent to [`akita_types::generated::sis_floor::min_rank_for_secure_width`]
/// but consumes a slice the caller already obtained from `sis_max_widths`,
/// avoiding a per-candidate re-lookup.
fn rank_floor_from_table(table: &[u64], width: usize) -> Option<usize> {
    let width_u64 = width as u64;
    for (i, &max_w) in table.iter().enumerate() {
        if width_u64 <= max_w {
            return Some(i + 1);
        }
    }
    None
}

/// Compute the A-role weak-binding collision norm for the SIS-floor lookup.
///
/// Implements the A-column-block argument of `2·ω̄·β̄` from
/// Lemma 7 ("Weak Binding") of the Hachi paper.
///
/// - `β̄` is the per-coefficient half-width of an honest commit
///   coefficient, so `2·β̄` is the diff bound across two honest openings;
/// - `ω̄` is the stage-1 sparse-challenge infinity norm.
///
/// An extra `ring_subfield_norm` factor accounts for embedding the
/// subfield challenge into the cyclotomic ring of integers via `ψ`; it is
/// not present in the paper's abstract lemma but is required by the
/// concrete Akita instantiation.
fn compute_weak_binding_norm_for_a(
    log_basis: u32,
    log_commit_bound: u32,
    stage1_inf_norm: u32,
    ring_subfield_norm: u32,
) -> Option<u32> {
    let beta = log_basis
        .checked_sub(1)
        .and_then(|s| 1u32.checked_shl(s))
        .and_then(|b| b.checked_sub(1))?;
    let beta: u32 = if log_commit_bound == 1 { 1 } else { beta };
    let omega: u32 = stage1_inf_norm;
    beta.checked_mul(2)
        .and_then(|v| v.checked_mul(omega))
        .and_then(|v| v.checked_mul(ring_subfield_norm))
}

/// Resolve the per-candidate digit count for the honest commit
/// coefficients from `Cfg` and the candidate `log_basis`.
fn num_digits_commit_for<Cfg: CommitmentConfig>(log_basis: u32) -> usize {
    let decomp = Cfg::decomposition();
    num_digits_for_bound(decomp.log_commit_bound, decomp.field_bits(), log_basis)
}

/// Resolve the per-candidate digit count for the opened coefficients
/// from `Cfg` and the candidate `log_basis`. Mirrors the call-site fall
/// back from `log_open_bound` to `log_commit_bound`.
fn num_digits_open_for<Cfg: CommitmentConfig>(log_basis: u32) -> usize {
    let decomp = Cfg::decomposition();
    let log_open_bound = decomp.log_open_bound.unwrap_or(decomp.log_commit_bound);
    num_digits_for_bound(log_open_bound, decomp.field_bits(), log_basis)
}

/// Resolve the raw B/D-role weak-binding collision norm
/// `2^log_basis − 1` — the digit-diff bound shared by the B and D
/// SIS-floor lookups. Returns `None` on `log_basis` shift overflow.
fn bd_collision_raw(log_basis: u32) -> Option<u32> {
    1u32.checked_shl(log_basis).and_then(|b| b.checked_sub(1))
}

/// Build the A-role `AjtaiKeyParams` for a candidate root tile.
///
/// Everything besides the block geometry is derived from `Cfg` and the
/// candidate `log_basis`:
///
/// - `num_digits_commit` via [`num_digits_commit_for`].
/// - `a_collision_raw` (the `2·ω̄·β̄` weak-binding norm from Hachi
///   Lemma 7) via [`compute_weak_binding_norm_for_a`].
///
/// Then the full A-role sizing pipeline runs end-to-end:
///
/// 1. `inner_width = block_len * num_digits_commit` (col_len of A).
/// 2. Round `a_collision_raw` up to the next audited bucket.
/// 3. Resolve the generated SIS-floor `max_widths` row for that bucket.
/// 4. Look up the smallest A-row count `matrix_a_rank` (row_len of A)
///    whose secure width covers `inner_width`.
/// 5. Hand `(matrix_a_rank, inner_width, a_bucket)` to
///    `AjtaiKeyParams::try_new` for the boundary audit.
///
/// Returns `Ok(None)` when steps 1-4 fall off the generated tables
/// (skip this candidate), `Err(...)` only on a hard audit / Cfg error,
/// and `Ok(Some(key))` on success.
fn compute_ajtai_key_params_a<Cfg: CommitmentConfig>(
    block_len: usize,
    log_basis: u32,
) -> Result<Option<AjtaiKeyParams>, AkitaError> {
    let decomp = Cfg::decomposition();
    let stage1 = Cfg::stage1_challenge_config(Cfg::D)?;
    let Some(a_collision_raw) = compute_weak_binding_norm_for_a(
        log_basis,
        decomp.log_commit_bound,
        stage1.infinity_norm(),
        Cfg::ring_subfield_embedding_norm_bound(),
    ) else {
        return Ok(None);
    };
    let num_digits_commit = num_digits_commit_for::<Cfg>(log_basis);
    let Some(inner_width) = block_len.checked_mul(num_digits_commit) else {
        return Ok(None);
    };
    let Some(a_bucket) =
        ceil_supported_collision(Cfg::sis_modulus_family(), Cfg::D as u32, a_collision_raw)
    else {
        return Ok(None);
    };
    let Some(a_table) = sis_max_widths(Cfg::sis_modulus_family(), Cfg::D as u32, a_bucket) else {
        return Ok(None);
    };
    let Some(matrix_a_rank) = rank_floor_from_table(a_table, inner_width) else {
        return Ok(None);
    };
    AjtaiKeyParams::try_new(
        Cfg::sis_modulus_family(),
        matrix_a_rank,
        inner_width,
        a_bucket,
        Cfg::D,
    )
    .map(Some)
}

/// Build the B-role `AjtaiKeyParams` for a candidate root tile.
///
/// Inputs are the truly call-site-dependent values (the rank flowing in
/// from the A role plus the block-fanout shape); everything else is
/// derived from `Cfg` and `log_basis`:
///
/// - `num_digits_open` via [`num_digits_open_for`].
/// - `bd_collision_raw = 2^log_basis − 1` via [`bd_collision_raw`].
///
/// Pipeline:
///
/// 1. `vector_t_len = matrix_a_rank * num_digits_open * num_blocks * t_vectors`
///    (col_len of B — the batched column count across all `t` vectors).
/// 2. Round `bd_collision_raw` up to the next audited bucket.
/// 3. Resolve the generated SIS-floor `max_widths` row for that bucket.
/// 4. Look up the smallest B-row count `matrix_b_rank` whose secure
///    width covers `vector_t_len`.
/// 5. Hand `(matrix_b_rank, vector_t_len, bd_bucket)` to
///    `AjtaiKeyParams::try_new`.
///
/// See [`compute_ajtai_key_params_a`] for the return contract.
fn compute_ajtai_key_params_b<Cfg: CommitmentConfig>(
    matrix_a_rank: usize,
    num_blocks: usize,
    t_vectors: usize,
    log_basis: u32,
) -> Result<Option<AjtaiKeyParams>, AkitaError> {
    let num_digits_open = num_digits_open_for::<Cfg>(log_basis);
    let Some(bd_raw) = bd_collision_raw(log_basis) else {
        return Ok(None);
    };
    let Some(vector_t_len) = matrix_a_rank
        .checked_mul(num_digits_open)
        .and_then(|w| w.checked_mul(num_blocks))
        .and_then(|w| w.checked_mul(t_vectors))
    else {
        return Ok(None);
    };
    let Some(bd_bucket) =
        ceil_supported_collision(Cfg::sis_modulus_family(), Cfg::D as u32, bd_raw)
    else {
        return Ok(None);
    };
    let Some(bd_table) = sis_max_widths(Cfg::sis_modulus_family(), Cfg::D as u32, bd_bucket) else {
        return Ok(None);
    };
    let Some(matrix_b_rank) = rank_floor_from_table(bd_table, vector_t_len) else {
        return Ok(None);
    };
    AjtaiKeyParams::try_new(
        Cfg::sis_modulus_family(),
        matrix_b_rank,
        vector_t_len,
        bd_bucket,
        Cfg::D,
    )
    .map(Some)
}

/// Build the D-role `AjtaiKeyParams` for a candidate root tile.
///
/// Pipeline:
///
/// 1. `d_width = num_digits_open * num_blocks * t_vectors` (col_len of
///    D — independent of `matrix_a_rank`).
/// 2. Round `bd_collision_raw = 2^log_basis − 1` up to the next audited
///    bucket — D shares the B/D-digit collision norm.
/// 3. Resolve the generated SIS-floor `max_widths` row for that bucket.
/// 4. Look up the smallest D-row count `matrix_d_rank` whose secure
///    width covers `d_width`.
/// 5. Hand `(matrix_d_rank, d_width, bd_bucket)` to
///    `AjtaiKeyParams::try_new`.
///
/// See [`compute_ajtai_key_params_a`] for the return contract.
fn compute_ajtai_key_params_d<Cfg: CommitmentConfig>(
    num_blocks: usize,
    t_vectors: usize,
    log_basis: u32,
) -> Result<Option<AjtaiKeyParams>, AkitaError> {
    let num_digits_open = num_digits_open_for::<Cfg>(log_basis);
    let Some(bd_raw) = bd_collision_raw(log_basis) else {
        return Ok(None);
    };
    let Some(d_width) = num_digits_open
        .checked_mul(num_blocks)
        .and_then(|w| w.checked_mul(t_vectors))
    else {
        return Ok(None);
    };
    let Some(bd_bucket) =
        ceil_supported_collision(Cfg::sis_modulus_family(), Cfg::D as u32, bd_raw)
    else {
        return Ok(None);
    };
    let Some(bd_table) = sis_max_widths(Cfg::sis_modulus_family(), Cfg::D as u32, bd_bucket) else {
        return Ok(None);
    };
    let Some(matrix_d_rank) = rank_floor_from_table(bd_table, d_width) else {
        return Ok(None);
    };
    AjtaiKeyParams::try_new(
        Cfg::sis_modulus_family(),
        matrix_d_rank,
        d_width,
        bd_bucket,
        Cfg::D,
    )
    .map(Some)
}

/// Consult the offline schedule tables for a pre-computed answer.
fn offline_schedule_for_key<Cfg: CommitmentConfig>(
    key: AkitaScheduleLookupKey,
    table: GeneratedScheduleTable,
    envelope: CommitmentEnvelope,
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
            envelope,
            ring_subfield_norm_bound: Cfg::ring_subfield_embedding_norm_bound(),
            fold_challenge_shape: Cfg::fold_challenge_shape_at_level,
        },
    )?;
    Ok(plan.map(|plan| schedule_from_plan(&plan, Cfg::decomposition().field_bits())))
}

/// Find the optimal schedule for a root schedule lookup key under `Cfg`.
///
/// When `use_lookup` is true the search consults `Cfg::schedule_table()`
/// as a fast path and seeds the DP from the corresponding singleton plan
/// / envelope floor. Production callers pass `true`; the
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

    // `envelope` is the only per-search input not derivable from `Cfg`
    // alone — it depends on `key.num_vars` and on `use_lookup`. Compute
    // it once and thread it through the helpers as a plain reference.
    //
    // We no longer thread `Cfg::schedule_plan(...)` into the suffix DP:
    // `current_level_layout_with_log_basis`'s exact-plan match was a
    // single-`(m, r)` shortcut that would fire on at most one of the
    // log_basis values per state; the DP already explores every
    // log_basis itself, so the shortcut never improves the optimum.
    let envelope = planner_envelope::<Cfg>(key.num_vars, use_lookup);

    if use_lookup {
        if let Some(table) = Cfg::schedule_table() {
            if let Some(schedule) = offline_schedule_for_key::<Cfg>(key, table, envelope)? {
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
        commit_params: root_direct_commit_params,
        // Root-direct has no next level; the prover walk stops here.
        level_params: None,
    })];
    let mut memo = ScheduleMemo::new();

    // Finding best candidate
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

    let min_num_blocks: usize = if reduced_vars >= 3 { 2 } else { 1 };
    let max_num_blocks: usize = 1usize
        .checked_shl((reduced_vars - 1) as u32)
        .unwrap_or(usize::MAX)
        .max(min_num_blocks);

    let (min_log_basis, max_log_basis) = Cfg::basis_range();
    for candidate_log_basis in min_log_basis..=max_log_basis {
        let decomp = Cfg::decomposition();
        let field_bits = decomp.field_bits();
        let log_open_bound = decomp.log_open_bound.unwrap_or(decomp.log_commit_bound);
        let num_digits_commit =
            num_digits_for_bound(decomp.log_commit_bound, field_bits, candidate_log_basis);
        let num_digits_open = num_digits_for_bound(log_open_bound, field_bits, candidate_log_basis);

        // The canonical derivation (`sis_derived_root_params_for_layout`)
        // passes the raw `bd = 2^candidate_log_basis − 1` into
        // `sis_secure_level_params` for both B and D, without
        // multiplying by the stage-1 / embedding norms (only the A role
        // amplifies). All per-role collision raws and SIS-floor
        // bucket/table lookups now live inside
        // [`compute_ajtai_key_params_a`] /
        // [`compute_ajtai_key_params_b`] /
        // [`compute_ajtai_key_params_d`].

        let mut num_blocks = min_num_blocks;
        loop {
            let r_vars = num_blocks.trailing_zeros() as usize;
            let m_vars = reduced_vars - r_vars;

            // Each candidate evaluates inside a labeled block so a
            // `continue`-equivalent rejection (`break 'candidate;`)
            // still falls through to the `num_blocks <<= 1` advance
            // below. Without the label the bare `loop` would infinite-
            // loop on any rejected candidate.
            'candidate: {
                // (1) `(m, r)` → block geometry. `num_blocks` is the
                // loop variable; only `block_len` needs derivation.
                let Some(block_len) = 1usize.checked_shl(m_vars as u32) else {
                    break 'candidate;
                };

                // (2) Build the A-role key. `Ok(None)` = candidate falls
                // off the SIS-floor tables (skip), `Err(...)` = audit
                // invariant violation (propagate hard).
                let Some(a_key) =
                    compute_ajtai_key_params_a::<Cfg>(block_len, candidate_log_basis)?
                else {
                    break 'candidate;
                };

                // (3) Build the B-role key. `matrix_a_rank` flows in via
                // `a_key.row_len()` since B's `vector_t_len` depends on
                // the A rank.
                let Some(b_key) = compute_ajtai_key_params_b::<Cfg>(
                    a_key.row_len(),
                    num_blocks,
                    t_vectors,
                    candidate_log_basis,
                )?
                else {
                    break 'candidate;
                };

                // (4) Build the D-role key. `d_width` is independent of
                // `matrix_a_rank` and shares the B/D bucket with B.
                let Some(d_key) =
                    compute_ajtai_key_params_d::<Cfg>(num_blocks, t_vectors, candidate_log_basis)?
                else {
                    break 'candidate;
                };

                // (5) Per-level fold-digit count for the batched layout.
                // Matches `akita_types::scale_batched_root_layout`: per-claim
                // the fold weight has L1 ≤ `effective_l1_mass(stage1)`
                // (which squares `l1_norm` for `TensorChallengeShape` since
                // the per-block challenge is the ring product `left · right`);
                // batching across `t_vectors` claims sums linearly. The tight
                // worst case is therefore `effective_l1_mass · t_vectors`.
                let num_digits_fold = compute_num_digits_fold_with_claims(
                    r_vars,
                    fold_shape.effective_l1_mass(&stage1),
                    candidate_log_basis,
                    t_vectors,
                    field_bits,
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

                // (7) Derived witness length for the next level + shrink
                // check. A `(log_basis, r_vars)` that doesn't strictly
                // shrink the witness in bits cannot be on the optimal
                // path — the direct-witness baseline always beats it.
                // `initial_witness_len` is the candidate's pre-fold
                // witness budget (in bits); defined here next to its
                // only use.
                let next_w_len = root_next_witness_len::<Cfg>(&level_lp, key)?;
                let next_witness_len = next_w_len
                    .checked_mul(candidate_log_basis as usize)
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup("root next witness bit length overflow".into())
                    })?;
                let initial_witness_len =
                    witness_len
                        .checked_mul(field_bits as usize)
                        .ok_or_else(|| {
                            AkitaError::InvalidSetup("root witness bit length overflow".into())
                        })?;
                if next_witness_len >= initial_witness_len {
                    break 'candidate;
                }

                let candidate = CandidateLevelParams {
                    lp: level_lp,
                    next_w_len,
                };

                // (8) Suffix DP + scoring. Every surviving candidate is
                // scored on `root_proof_size + suffix_cost` (no greedy
                // pre-selection), so the planner output is monotone in the
                // candidate set: a previously-rejected `(m, r)` with a
                // worse `next_w_len` but a smaller total proof can no
                // longer be silently dropped.
                let (mut suffix_cost, mut suffix_steps) = derive_optimal_suffix_schedule::<Cfg>(
                    &envelope,
                    &mut memo,
                    key.num_vars,
                    1,
                    candidate.next_w_len,
                    candidate_log_basis,
                    0,
                )?;
                if suffix_steps.is_empty() {
                    break 'candidate;
                }
                let suffix_is_terminal = matches!(suffix_steps.first(), Some(Step::Direct(_)));
                let Ok(eor_bytes) =
                    extension_opening_reduction_level_bytes::<Cfg>(key, 0, witness_len)
                else {
                    break 'candidate;
                };
                let next_w_len_override = if suffix_is_terminal {
                    let old_direct_bytes = match suffix_steps.first().expect("suffix non-empty") {
                        Step::Direct(direct) => direct.direct_bytes,
                        Step::Fold(_) => unreachable!("suffix_is_terminal guard"),
                    };
                    // Lazy SIS check: skip this root candidate if its
                    // terminal direct cannot be SIS-secured under the
                    // candidate's batched `LevelParams`. Mirrors the
                    // recursive suffix DP's behaviour (replaces the
                    // previous unconditional `?`, which would have
                    // failed the entire search rather than discarding
                    // one infeasible candidate).
                    if finalize_terminal_direct_witness_shape::<Cfg>(
                        &envelope,
                        key.num_vars,
                        &mut suffix_steps,
                        &candidate,
                        num_points,
                        t_vectors,
                        w_vectors,
                        z_vectors,
                        0,
                    )
                    .is_err()
                    {
                        break 'candidate;
                    }
                    let (new_direct_bytes, terminal_field_len) =
                        match suffix_steps.first().expect("suffix non-empty") {
                            Step::Direct(direct) => (direct.direct_bytes, direct.current_w_len),
                            Step::Fold(_) => unreachable!("suffix_is_terminal guard"),
                        };
                    suffix_cost = suffix_cost + new_direct_bytes - old_direct_bytes;
                    Some(terminal_field_len)
                } else {
                    None
                };
                let root_proof_size = if suffix_is_terminal {
                    let terminal_next_w_len = next_w_len_override
                        .expect("suffix_is_terminal branch populates next_w_len_override above");
                    compute_terminal_level_proof_size::<Cfg>(
                        &candidate,
                        terminal_next_w_len,
                        z_vectors,
                    ) + eor_bytes
                } else {
                    let Ok(next_level_params) = successor_level_params_from_schedule::<Cfg>(
                        &envelope,
                        key.num_vars,
                        1,
                        candidate.next_w_len,
                        &suffix_steps,
                    ) else {
                        break 'candidate;
                    };
                    compute_level_proof_size::<Cfg>(&candidate, &next_level_params, z_vectors)
                        + eor_bytes
                };

                let total = root_proof_size + suffix_cost;
                if total < best_cost {
                    best_cost = total;
                    let mut steps = Vec::with_capacity(1 + suffix_steps.len());
                    steps.push(to_fold_step(
                        &candidate,
                        witness_len,
                        root_proof_size,
                        field_bits,
                        next_w_len_override,
                    ));
                    steps.extend(suffix_steps);
                    best_steps = steps;
                }
            }

            // Advance to the next power-of-two `num_blocks`. When we've
            // already evaluated `max_num_blocks`, stop iterating.
            if num_blocks >= max_num_blocks {
                break;
            }
            num_blocks <<= 1;
        }
    }

    let num_folds = best_steps
        .iter()
        .filter(|s| matches!(s, Step::Fold(_)))
        .count();
    tracing::info!(
        ?key,
        total_bytes = best_cost,
        fold_levels = num_folds,
        "schedule planner: computed from scratch (no offline entry)"
    );

    Ok(Schedule {
        steps: best_steps,
        total_bytes: best_cost,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn planner_z_count_comes_from_schedule_key() {
        let key = AkitaScheduleLookupKey::new(2, 3, 4, 1);

        assert_eq!(key.num_t_vectors, 3);
        assert_eq!(key.num_w_vectors, 4);
        assert_eq!(key.num_z_vectors, 1);
    }

    #[test]
    #[cfg(not(feature = "zk"))]
    fn planner_uses_generated_schedule_fast_path() {
        // Pick a real preset and a singleton key its table covers. The fast
        // path should short-circuit before DP runs.
        use akita_config::proof_optimized::fp128;
        let key = AkitaScheduleLookupKey::singleton(8);
        let schedule = find_schedule::<fp128::D64Full>(key, true).expect("offline schedule lookup");
        assert!(!schedule.steps.is_empty());
    }
}
