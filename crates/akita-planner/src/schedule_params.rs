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
use std::marker::PhantomData;

use akita_config::CommitmentConfig;
use akita_field::AkitaError;
use akita_types::generated::sis_floor::{ceil_supported_collision, sis_max_widths};
use akita_types::generated::GeneratedScheduleTable;
use akita_types::layout::digit_math::{
    compute_num_digits_fold_with_claims, compute_num_digits_full_field,
};
use akita_types::AkitaSchedulePlan;
use akita_types::{
    decomp_depths, direct_witness_bytes, extension_opening_reduction_proof_bytes,
    level_proof_bytes, root_extension_opening_partials, scale_batched_root_layout,
    schedule_from_plan, terminal_level_proof_bytes, w_ring_element_count_with_counts_bits,
    w_ring_element_count_with_counts_for_layout_bits, AjtaiKeyParams, AjtaiRole,
    AkitaScheduleInputs, AkitaScheduleLookupKey, CommitmentEnvelope, DirectStep,
    DirectWitnessShape, FoldStep, LevelParams, MRowLayout, Schedule, Step,
};

use akita_derive::{schedule_plan_from_table, PlanPolicy};

const MAX_RECURSION_DEPTH: usize = 12;

/// Per-search inputs that aren't derivable from `Cfg`.
///
/// Everything else (ring dimension, decomposition, SIS family, stage-1
/// challenge config, basis range, fold-challenge shape, etc.) is read
/// directly off `Cfg`.
pub(crate) struct SearchOptions<Cfg: CommitmentConfig> {
    pub key: AkitaScheduleLookupKey,
    /// Offline schedule table consulted before DP when `use_lookup` is set.
    pub table: Option<GeneratedScheduleTable>,
    /// Pre-computed singleton schedule plan for `key.num_vars`.
    pub schedule_plan: Option<AkitaSchedulePlan>,
    /// Pre-computed commitment envelope for `key.num_vars`.
    pub envelope: CommitmentEnvelope,
    _cfg: PhantomData<Cfg>,
}

fn search_options<Cfg: CommitmentConfig>(
    key: AkitaScheduleLookupKey,
    use_lookup: bool,
) -> SearchOptions<Cfg> {
    // When regenerating tables (`use_lookup = false`), drop the singleton
    // plan and offline table so the DP cannot read back any value that
    // came from a stored schedule entry — otherwise `gen_schedule_tables`
    // would not be idempotent. Reconstruct the envelope from the audited
    // ranks alone, matching what a from-scratch search would compute on
    // an empty table.
    let (schedule_plan, table, envelope) = if use_lookup {
        (
            Cfg::schedule_plan(AkitaScheduleLookupKey::singleton(key.num_vars))
                .ok()
                .flatten(),
            Cfg::schedule_table(),
            Cfg::envelope(key.num_vars),
        )
    } else {
        let inner = Cfg::audited_root_rank(AjtaiRole::Inner, key.num_vars);
        let outer = Cfg::audited_root_rank(AjtaiRole::Outer, key.num_vars);
        (
            None,
            None,
            CommitmentEnvelope {
                max_n_a: inner,
                max_n_b: outer,
                max_n_d: outer,
            },
        )
    };
    SearchOptions {
        key,
        table,
        schedule_plan,
        envelope,
        _cfg: PhantomData,
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
    /// the per-level layout as derived by
    /// `current_level_layout_with_log_basis`.
    lp: LevelParams,
    next_w_len: usize,
}

/// Derive the layout for folding at `(level, w_len, log_basis)`.
/// Returns `None` if the layout is infeasible or doesn't shrink the witness.
fn derive_candidate_level_params<Cfg: CommitmentConfig>(
    opts: &SearchOptions<Cfg>,
    num_vars: usize,
    level: usize,
    current_w_len: usize,
    log_basis: u32,
) -> Result<Option<CandidateLevelParams>, AkitaError> {
    let inputs = AkitaScheduleInputs {
        num_vars,
        level,
        current_w_len,
    };

    let level_lp = match akita_derive::current_level_layout_with_log_basis(
        Cfg::sis_modulus_family(),
        Cfg::D,
        Cfg::decomposition(),
        Cfg::ring_subfield_embedding_norm_bound(),
        opts.schedule_plan.as_ref(),
        &opts.envelope,
        Cfg::stage1_challenge_config,
        inputs,
        log_basis,
    ) {
        Ok(level_lp) => level_lp,
        Err(_) => return Ok(None),
    };

    let fb = Cfg::decomposition().field_bits();
    // Recursive folds carry one recursive witness and open it at one prepared
    // recursive point. Root batching is reflected only at level 0.
    let w_ring_elements = w_ring_element_count_with_counts_bits(fb, &level_lp, 1, 1, 1, 1)?;
    let next_w_len = w_ring_elements
        .checked_mul(level_lp.ring_dimension)
        .ok_or_else(|| AkitaError::InvalidSetup("recursive witness length overflow".into()))?;

    let input_elem_bits = if level == 0 {
        fb as usize
    } else {
        log_basis as usize
    };
    let next_bits = next_w_len
        .checked_mul(log_basis as usize)
        .ok_or_else(|| AkitaError::InvalidSetup("next witness bit length overflow".into()))?;
    let current_bits = current_w_len
        .checked_mul(input_elem_bits)
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
    opts: &SearchOptions<Cfg>,
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
        &opts.envelope,
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
    opts: &SearchOptions<Cfg>,
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
            &opts.envelope,
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

fn derive_optimal_suffix_schedule<Cfg: CommitmentConfig>(
    opts: &SearchOptions<Cfg>,
    memo: &mut ScheduleMemo,
    num_vars: usize,
    level: usize,
    current_w_len: usize,
    current_lb: u32,
    depth: usize,
) -> Result<(usize, Vec<Step>), AkitaError> {
    let key = (level, current_w_len, current_lb);
    if depth <= MAX_RECURSION_DEPTH {
        if let Some(cached) = memo.get(&key) {
            return Ok(cached.clone());
        }
    }

    let direct_allowed = level == 0
        || akita_derive::direct_level_params_with_log_basis(
            Cfg::sis_modulus_family(),
            Cfg::D,
            Cfg::decomposition(),
            match Cfg::stage1_challenge_config(Cfg::D) {
                Ok(s) => s,
                Err(_) => return Ok((usize::MAX, Vec::new())),
            },
            Cfg::ring_subfield_embedding_norm_bound(),
            &opts.envelope,
            AkitaScheduleInputs {
                num_vars,
                level,
                current_w_len,
            },
            current_lb,
        )
        .is_ok();
    let mut best_cost = usize::MAX;
    let mut best_schedule = Vec::new();
    if direct_allowed {
        let placeholder = to_direct_step::<Cfg>(current_w_len, current_lb);
        let Step::Direct(direct) = &placeholder else {
            unreachable!("to_direct_step returns Step::Direct");
        };
        best_cost = direct.direct_bytes;
        best_schedule = vec![placeholder];
    }

    if depth <= MAX_RECURSION_DEPTH {
        let (min_log_basis, max_log_basis) = Cfg::basis_range();
        for lb in min_log_basis..=max_log_basis {
            if lb < current_lb {
                continue;
            }
            let Some(candidate) =
                derive_candidate_level_params(opts, num_vars, level, current_w_len, lb)?
            else {
                continue;
            };

            let (mut suffix_cost, mut suffix_steps) = derive_optimal_suffix_schedule(
                opts,
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
                finalize_terminal_direct_witness_shape(
                    opts,
                    num_vars,
                    &mut suffix_steps,
                    &candidate,
                    1,
                    1,
                    1,
                    1,
                    level,
                )?;
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
                compute_terminal_level_proof_size::<Cfg>(&candidate, terminal_next_w_len, 1)
                    + eor_bytes
            } else {
                let Ok(next_level_params) = successor_level_params_from_schedule(
                    opts,
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

        memo.insert(key, (best_cost, best_schedule.clone()));
    }

    Ok((best_cost, best_schedule))
}

// -----------------------------------------------------------------------
// Key-derived root sizing
// -----------------------------------------------------------------------

fn root_w_ring_element_count<Cfg: CommitmentConfig>(
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
    {
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
        Ok(w_hat + t_hat + b_blinding + d_blinding + z_pre + r)
    }
    #[cfg(not(feature = "zk"))]
    {
        Ok(w_hat + t_hat + z_pre + r)
    }
}

// -----------------------------------------------------------------------
// Key-driven root candidate + entry point
// -----------------------------------------------------------------------

/// Cached, loop-invariant inputs for [`derive_root_candidate`].
///
/// All of these are pure functions of `Cfg` + `log_basis` and can be
/// computed once before iterating over `(m_vars, r_vars)` splits.
struct RootCandidateContext<'a> {
    sis_family: akita_types::SisModulusFamily,
    d: usize,
    log_basis: u32,
    fb: u32,
    stage1: akita_challenges::SparseChallengeConfig,
    fold_shape: akita_challenges::TensorChallengeShape,
    num_claims: usize,
    reduced_vars: usize,
    num_digits_commit: usize,
    num_digits_open: usize,
    /// Pre-rounded SIS-floor collision bucket for the root A role.
    a_bucket: u32,
    /// Pre-rounded SIS-floor collision bucket for the root B/D roles.
    bd_bucket: u32,
    /// Cached A-role rank table at `a_bucket`. Index `i` ↔ rank `i + 1`,
    /// value is the maximum SIS-secure width at that rank.
    a_table: &'a [u64],
    /// Cached B/D-role rank table at `bd_bucket`.
    bd_table: &'a [u64],
}

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

/// Compose A-role collision norm before SIS-floor bucket rounding.
///
/// Mirrors the formula in `akita_derive::derivation` so the planner and
/// the verifier-reachable derivation agree on the bucket. The honest
/// commit-coefficient bound when `log_commit_bound == 1` is the tight
/// constant `2`; otherwise it equals the digit / B-D bound.
fn a_role_collision_raw(
    log_commit_bound: u32,
    bd_raw: u32,
    stage1_inf_norm: u32,
    ring_subfield_norm: u32,
) -> Option<u32> {
    let a_raw = if log_commit_bound == 1 { 2 } else { bd_raw };
    a_raw
        .checked_mul(stage1_inf_norm)?
        .checked_mul(ring_subfield_norm)
}

/// Build [`RootCandidateContext`] for one `log_basis`, or return `None`
/// when the audited SIS-floor tables don't cover this configuration.
fn build_root_candidate_context<Cfg: CommitmentConfig>(
    num_vars: usize,
    log_basis: u32,
) -> Result<Option<RootCandidateContext<'static>>, AkitaError> {
    let sis_family = Cfg::sis_modulus_family();
    let d = Cfg::D;
    let decomp = Cfg::decomposition();
    let fb = decomp.field_bits();
    let stage1 = Cfg::stage1_challenge_config(d)?;
    let ring_subfield = Cfg::ring_subfield_embedding_norm_bound();
    let fold_shape = Cfg::fold_challenge_shape_at_level(AkitaScheduleInputs {
        num_vars,
        level: 0,
        current_w_len: 1usize.checked_shl(num_vars as u32).unwrap_or(0),
    });

    let alpha = (d as u32).trailing_zeros() as usize;
    let Some(reduced_vars) = num_vars.checked_sub(alpha) else {
        return Ok(None);
    };
    if reduced_vars < 1 {
        return Ok(None);
    }

    // `decomp_depths` is sensitive to `decomp.log_basis`, but the planner
    // iterates `log_basis` independently of the Cfg default. Override the
    // basis on a local copy of `decomp` so the digit depths reflect the
    // candidate basis — this matches what
    // `akita_derive::derived_root_commitment_layout_from_params` does
    // before deriving widths.
    let candidate_decomp = akita_types::DecompositionParams {
        log_basis,
        ..decomp
    };
    let (num_digits_commit, num_digits_open) = decomp_depths(candidate_decomp);

    let Some(bd_raw) = 1u32.checked_shl(log_basis).and_then(|b| b.checked_sub(1)) else {
        return Ok(None);
    };

    let Some(a_collision_raw) = a_role_collision_raw(
        decomp.log_commit_bound,
        bd_raw,
        stage1.infinity_norm(),
        ring_subfield,
    ) else {
        return Ok(None);
    };

    // The canonical derivation (`sis_derived_root_params_for_layout`)
    // passes the raw `bd = 2^log_basis − 1` into `sis_secure_level_params`
    // for both B and D, without multiplying by the stage-1 / embedding
    // norms (only the A role amplifies). Pre-rounding through
    // `ceil_supported_collision` here is a defensive no-op for
    // `log_basis >= 2` (since `2^lb − 1` already equals a generated
    // bucket) but lets `log_basis == 1` round up to the smallest
    // audited bucket instead of falling off the table.
    let Some(a_bucket) = ceil_supported_collision(sis_family, d as u32, a_collision_raw) else {
        return Ok(None);
    };
    let Some(bd_bucket) = ceil_supported_collision(sis_family, d as u32, bd_raw) else {
        return Ok(None);
    };

    let Some(a_table) = sis_max_widths(sis_family, d as u32, a_bucket) else {
        return Ok(None);
    };
    let Some(bd_table) = sis_max_widths(sis_family, d as u32, bd_bucket) else {
        return Ok(None);
    };

    Ok(Some(RootCandidateContext {
        sis_family,
        d,
        log_basis,
        fb,
        stage1,
        fold_shape,
        num_claims: 1,
        reduced_vars,
        num_digits_commit,
        num_digits_open,
        a_bucket,
        bd_bucket,
        a_table,
        bd_table,
    }))
}

/// Enumerate every SIS-secure root candidate at `log_basis` whose
/// folded recursion shrinks the witness.
///
/// For each `(m_vars, r_vars)` split the rank–width chain is one-shot:
///
/// 1. `inner_width = block_len · num_digits_commit` → `n_a` from the
///    cached A-role rank table.
/// 2. `outer_width = n_a · num_digits_open · num_blocks · num_t_vectors`
///    (already batched) → `n_b` from the cached B/D-role rank table.
/// 3. `d_width = num_digits_open · num_blocks · num_t_vectors` (no
///    `n_a` dependency) → `n_d` from the B/D-role rank table.
/// 4. Build a single SIS-secure [`LevelParams`].
///
/// No fixed-point iteration is required because `(m, r)` are fixed for
/// the body and the width→rank→width chain has no back-edge.
///
/// Returns every candidate that passes the per-`(m, r)` SIS-floor lookup
/// and the per-candidate shrink check (`next_bits < root_bits`). The
/// caller (`find_schedule`) scores each candidate by total proof bytes
/// — root proof bytes plus suffix DP — and keeps the global minimum
/// across all `(log_basis, m, r)` triples.
///
/// Returning `Vec<_>` instead of pre-selecting one candidate is what
/// makes the planner output strictly monotone in candidate set: a
/// previously-rejected `(m, r)` split that turns out to have a smaller
/// `total = root_proof_size + suffix_cost` than the
/// `next_w_len`-greedy pick can no longer be silently dropped before
/// scoring.
fn enumerate_root_candidates<Cfg: CommitmentConfig>(
    opts: &SearchOptions<Cfg>,
    num_vars: usize,
    witness_len: usize,
    log_basis: u32,
) -> Result<Vec<CandidateLevelParams>, AkitaError> {
    let mut ctx = match build_root_candidate_context::<Cfg>(num_vars, log_basis)? {
        Some(ctx) => ctx,
        None => return Ok(Vec::new()),
    };
    ctx.num_claims = opts.key.num_t_vectors;

    let root_bits = witness_len
        .checked_mul(ctx.fb as usize)
        .ok_or_else(|| AkitaError::InvalidSetup("root witness bit length overflow".into()))?;

    let r_lo: usize = if ctx.reduced_vars >= 3 { 1 } else { 0 };
    let r_hi: usize = ctx.reduced_vars.saturating_sub(1).max(r_lo);

    let mut candidates: Vec<CandidateLevelParams> =
        Vec::with_capacity(r_hi.saturating_sub(r_lo) + 1);

    for r_vars in r_lo..=r_hi {
        let m_vars = ctx.reduced_vars - r_vars;

        let Some(candidate) =
            build_root_candidate_for_split::<Cfg>(&ctx, m_vars, r_vars, opts.key)?
        else {
            continue;
        };

        let next_bits = candidate
            .next_w_len
            .checked_mul(log_basis as usize)
            .ok_or_else(|| {
                AkitaError::InvalidSetup("root next witness bit length overflow".into())
            })?;
        if next_bits >= root_bits {
            continue;
        }

        candidates.push(candidate);
    }

    Ok(candidates)
}

/// Build the SIS-secure root candidate for one `(m_vars, r_vars)` split,
/// or return `None` when no audited SIS rank covers the required width.
fn build_root_candidate_for_split<Cfg: CommitmentConfig>(
    ctx: &RootCandidateContext<'_>,
    m_vars: usize,
    r_vars: usize,
    key: AkitaScheduleLookupKey,
) -> Result<Option<CandidateLevelParams>, AkitaError> {
    let Some(num_blocks) = 1usize.checked_shl(r_vars as u32) else {
        return Ok(None);
    };
    let Some(block_len) = 1usize.checked_shl(m_vars as u32) else {
        return Ok(None);
    };

    // (1) inner_width → n_a
    let Some(inner_width) = block_len.checked_mul(ctx.num_digits_commit) else {
        return Ok(None);
    };
    let Some(n_a) = rank_floor_from_table(ctx.a_table, inner_width) else {
        return Ok(None);
    };

    // (2) n_a → outer_width (already batched by num_claims) → n_b
    let Some(outer_width) = n_a
        .checked_mul(ctx.num_digits_open)
        .and_then(|w| w.checked_mul(num_blocks))
        .and_then(|w| w.checked_mul(ctx.num_claims))
    else {
        return Ok(None);
    };
    let Some(n_b) = rank_floor_from_table(ctx.bd_table, outer_width) else {
        return Ok(None);
    };

    // (3) d_width (independent of n_a, also batched) → n_d
    let Some(d_width) = ctx
        .num_digits_open
        .checked_mul(num_blocks)
        .and_then(|w| w.checked_mul(ctx.num_claims))
    else {
        return Ok(None);
    };
    let Some(n_d) = rank_floor_from_table(ctx.bd_table, d_width) else {
        return Ok(None);
    };

    // (4) Per-level fold-digit count for the batched layout.
    //
    // Matches the semantics `akita_types::scale_batched_root_layout`
    // applies to the per-poly `LevelParams` the old planner built: take
    // the maximum of
    //
    //   - the per-poly fold-digit count using the fold-shape effective
    //     L1 mass (which squares `l1_norm` for `TensorChallengeShape`),
    //     evaluated at `num_claims = 1`, and
    //   - the batched fold-digit count using the raw stage-1 `l1_norm`
    //     at the actual `num_claims`.
    //
    // For `Flat` shapes the per-poly branch is dominated by the batched
    // one and the `max` collapses to the batched value; for `Tensor`
    // shapes the per-poly branch can be larger and the `max` is
    // significant.
    let per_poly_fold = compute_num_digits_fold_with_claims(
        r_vars,
        ctx.fold_shape.effective_l1_mass(&ctx.stage1),
        ctx.log_basis,
        1,
        ctx.fb,
    );
    let batched_fold = compute_num_digits_fold_with_claims(
        r_vars,
        ctx.stage1.l1_norm(),
        ctx.log_basis,
        ctx.num_claims,
        ctx.fb,
    );
    let num_digits_fold = per_poly_fold.max(batched_fold);

    let a_key = AjtaiKeyParams::try_new(ctx.sis_family, n_a, inner_width, ctx.a_bucket, ctx.d)?;
    let b_key = AjtaiKeyParams::try_new(ctx.sis_family, n_b, outer_width, ctx.bd_bucket, ctx.d)?;
    let d_key = AjtaiKeyParams::try_new(ctx.sis_family, n_d, d_width, ctx.bd_bucket, ctx.d)?;

    let level_lp = LevelParams {
        ring_dimension: ctx.d,
        log_basis: ctx.log_basis,
        a_key,
        b_key,
        d_key,
        num_blocks,
        block_len,
        m_vars,
        r_vars,
        stage1_config: ctx.stage1.clone(),
        fold_challenge_shape: ctx.fold_shape,
        num_digits_commit: ctx.num_digits_commit,
        num_digits_open: ctx.num_digits_open,
        num_digits_fold,
    };

    let raw_w_ring = root_w_ring_element_count::<Cfg>(&level_lp, key)?;
    let next_w_len = raw_w_ring
        .checked_mul(level_lp.ring_dimension)
        .ok_or_else(|| AkitaError::InvalidSetup("root recursive witness length overflow".into()))?;

    Ok(Some(CandidateLevelParams {
        lp: level_lp,
        next_w_len,
    }))
}

/// Consult the offline schedule tables for a pre-computed answer.
fn offline_schedule_for_key<Cfg: CommitmentConfig>(
    opts: &SearchOptions<Cfg>,
) -> Result<Option<Schedule>, AkitaError> {
    let Some(table) = opts.table else {
        return Ok(None);
    };
    // Materialization arithmetic is bit-width driven and uses
    // `root_decomp.field_bits()` everywhere, so the field marker is just
    // a phantom — pick a small canonical field that satisfies the bound.
    use akita_field::Prime128OffsetA7F7 as PhantomField;
    let plan = schedule_plan_from_table::<PhantomField, _>(
        opts.key,
        table,
        PlanPolicy {
            sis_family: Cfg::sis_modulus_family(),
            ring_dimension: Cfg::D,
            root_decomp: Cfg::decomposition(),
            challenge_field_bits: Cfg::decomposition().field_bits() * Cfg::CHAL_EXT_DEGREE as u32,
            recursive_public_rows: 1,
            extension_opening_width: Cfg::CLAIM_EXT_DEGREE,
            stage1_challenge_config: Cfg::stage1_challenge_config,
            envelope: opts.envelope,
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

    let opts = search_options::<Cfg>(key, use_lookup);

    if use_lookup {
        if let Some(schedule) = offline_schedule_for_key(&opts)? {
            tracing::debug!(
                ?key,
                total_bytes = schedule.total_bytes,
                "schedule planner: served from offline tables"
            );
            return Ok(schedule);
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
                scale_batched_root_layout(
                    &lp,
                    t_vectors,
                    Cfg::stage1_challenge_config(Cfg::D)?.l1_norm(),
                    Cfg::decomposition().field_bits(),
                )
                .ok()
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

    let (min_log_basis, max_log_basis) = Cfg::basis_range();
    for root_lb in min_log_basis..=max_log_basis {
        let candidates = enumerate_root_candidates(&opts, key.num_vars, witness_len, root_lb)?;
        // Score every `(root_lb, m, r)` candidate by total proof bytes
        // and keep the global minimum across all candidate triples. The
        // per-`(m, r)` `next_w_len`-greedy selection used previously
        // was a heuristic that could discard a candidate with a worse
        // `next_w_len` but a smaller `root_proof_size + suffix_cost`;
        // scoring every candidate makes the planner output monotone in
        // candidate set.
        for candidate in candidates {
            let (mut suffix_cost, mut suffix_steps) = derive_optimal_suffix_schedule(
                &opts,
                &mut memo,
                key.num_vars,
                1,
                candidate.next_w_len,
                root_lb,
                0,
            )?;
            if suffix_steps.is_empty() {
                continue;
            }
            let suffix_is_terminal = matches!(suffix_steps.first(), Some(Step::Direct(_)));
            let Ok(eor_bytes) = extension_opening_reduction_level_bytes::<Cfg>(key, 0, witness_len)
            else {
                continue;
            };
            let next_w_len_override = if suffix_is_terminal {
                let old_direct_bytes = match suffix_steps.first().expect("suffix non-empty") {
                    Step::Direct(direct) => direct.direct_bytes,
                    Step::Fold(_) => unreachable!("suffix_is_terminal guard"),
                };
                finalize_terminal_direct_witness_shape(
                    &opts,
                    key.num_vars,
                    &mut suffix_steps,
                    &candidate,
                    num_points,
                    t_vectors,
                    w_vectors,
                    z_vectors,
                    0,
                )?;
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
                compute_terminal_level_proof_size::<Cfg>(&candidate, terminal_next_w_len, z_vectors)
                    + eor_bytes
            } else {
                let Ok(next_level_params) = successor_level_params_from_schedule(
                    &opts,
                    key.num_vars,
                    1,
                    candidate.next_w_len,
                    &suffix_steps,
                ) else {
                    continue;
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
                    Cfg::decomposition().field_bits(),
                    next_w_len_override,
                ));
                steps.extend(suffix_steps);
                best_steps = steps;
            }
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
