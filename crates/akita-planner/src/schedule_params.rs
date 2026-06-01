//! Schedule planner that finds the global minimum proof size.
//!
//! A single exhaustive DP over `(level, w_len, log_basis)` states.  At each
//! state, every feasible basis is tried; `level_proof_bytes` uses the
//! smallest `next_commit` across all next-level bases; the suffix is
//! recursed into unconstrained.
//!
//! Public entry: [`find_optimal_schedule`], `<Cfg>`-generic, takes a
//! [`ScheduleSearchMode`] argument that controls whether the search
//! consults `Cfg::schedule_table()` (and seeds the DP with the
//! corresponding singleton plan / envelope floor) before running DP.
//! Production callers pass [`ScheduleSearchMode::RuntimeTableSeeded`];
//! the table-emitter binary regenerates from scratch with
//! [`ScheduleSearchMode::RegenerateFromScratch`].

use std::collections::HashMap;

use akita_challenges::SparseChallengeConfig;
use akita_config::CommitmentConfig;
use akita_field::AkitaError;
use akita_types::generated::GeneratedScheduleTable;
use akita_types::layout::digit_math::{
    compute_num_digits_fold_with_claims, compute_num_digits_full_field,
};
use akita_types::AkitaSchedulePlan;
use akita_types::{
    direct_witness_bytes, extension_opening_reduction_proof_bytes, level_proof_bytes,
    root_current_w_len, root_extension_opening_partials, scale_batched_root_layout,
    schedule_from_plan, terminal_level_proof_bytes, w_ring_element_count_with_counts_bits,
    w_ring_element_count_with_counts_for_layout_bits, AjtaiKeyParams, AjtaiRole,
    AkitaScheduleInputs, AkitaScheduleLookupKey, CleartextWitnessShape, CommitmentEnvelope,
    DecompositionParams, DirectStep, FoldStep, LevelParams, MRowLayout, Schedule, SisModulusFamily,
    Step,
};

use akita_derive::{schedule_plan_from_table, PlanPolicy};

/// Mode selector for [`find_optimal_schedule`].
///
/// Replaces the previous `allow_table_fast_path: bool` parameter. The
/// boolean had grown to control three intertwined behaviours (offline
/// fast-path consultation, singleton plan seeding, and table-derived
/// envelope flooring), and the name no longer described the contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScheduleSearchMode {
    /// Production runtime mode. The search first consults
    /// `Cfg::schedule_table()` as a fast path; on a miss the DP runs
    /// with the singleton `Cfg::schedule_plan` and the
    /// `Cfg::schedule_table()`-floored envelope as seed inputs.
    RuntimeTableSeeded,
    /// Offline regeneration mode used by `gen_schedule_tables`. The DP
    /// ignores `Cfg::schedule_table()` and any singleton plan derived
    /// from it, and reconstructs the envelope from
    /// `Cfg::audited_root_rank` alone. This guarantees the output is a
    /// pure function of the `Cfg` itself, so the generator is
    /// idempotent against the table it just emitted.
    RegenerateFromScratch,
}

impl ScheduleSearchMode {
    #[inline]
    fn consults_offline_tables(self) -> bool {
        matches!(self, Self::RuntimeTableSeeded)
    }
}

/// Build a `SearchOptions` value from a `CommitmentConfig`.
///
/// This is the bridge between the runtime config trait and the internal
/// schedule-search machinery. Public callers reach the planner through the
/// `<Cfg>`-generic [`find_optimal_schedule`] wrapper below; `SearchOptions`
/// itself is crate-private.
fn search_options_for_cfg<Cfg: CommitmentConfig>(
    key: AkitaScheduleLookupKey,
    mode: ScheduleSearchMode,
) -> Result<SearchOptions, AkitaError> {
    // The singleton schedule plan is fixed for the whole search (it only
    // depends on `key.num_vars`); precompute it once instead of having
    // every DP candidate evaluation reach back into `Cfg::schedule_plan`.
    //
    // When the caller is regenerating tables
    // (`ScheduleSearchMode::RegenerateFromScratch`), the singleton plan
    // and offline table are both dropped so the DP cannot read back any
    // value that itself came from a stored schedule entry — otherwise
    // `gen_schedule_tables` would not be idempotent: re-running it
    // against the table it just emitted would be free to oscillate
    // between equally-priced candidates whose tie ordering depends on
    // which prior table seeded `opts.schedule_plan`.
    let (schedule_plan, table, envelope) = if mode.consults_offline_tables() {
        (
            Cfg::schedule_plan(AkitaScheduleLookupKey::singleton(key.num_vars))?,
            Cfg::schedule_table(),
            Cfg::envelope(key.num_vars),
        )
    } else {
        // Production `Cfg::envelope` floors the SIS-audited rank against
        // the historical maxima recorded in `Cfg::schedule_table()`. That
        // monotone floor is appropriate for runtime consumers, but during
        // table regeneration it would feed the previous table's widths
        // back into the DP and prevent the binary from being idempotent.
        // Reconstruct the envelope from the audited ranks alone, matching
        // what a from-scratch search would compute on an empty table.
        let inner_floor = Cfg::audited_root_rank(AjtaiRole::Inner, key.num_vars);
        let outer_floor = Cfg::audited_root_rank(AjtaiRole::Outer, key.num_vars);
        (
            None,
            None,
            CommitmentEnvelope {
                max_n_a: inner_floor,
                max_n_b: outer_floor,
                max_n_d: outer_floor,
            },
        )
    };
    Ok(SearchOptions {
        key,
        ring_dimension: Cfg::D,
        decomposition: Cfg::decomposition(),
        sis_modulus_family: Cfg::sis_modulus_family(),
        challenge_field_bits: Cfg::decomposition().field_bits() * Cfg::CHAL_EXT_DEGREE as u32,
        extension_opening_width: Cfg::CLAIM_EXT_DEGREE,
        recursive_witness_expansion: 1,
        recursive_public_rows: 1,
        table,
        stage1_challenge_config: Cfg::stage1_challenge_config,
        schedule_plan,
        envelope,
        ring_subfield_embedding_norm_bound: Cfg::ring_subfield_embedding_norm_bound(),
        log_basis_search_range: Cfg::log_basis_search_range,
        fold_challenge_shape: Cfg::fold_challenge_shape_at_level,
    })
}

const MAX_RECURSION_DEPTH: usize = 12;

/// Value-typed planner inputs.
///
/// Replaces the deleted `PlannerConfig` trait. Callers (today, `akita-config`)
/// construct one of these per search call by translating their
/// `CommitmentConfig` shape into plain values + named function pointers.
///
/// Function-pointer fields (instead of generic `Fn` closures) keep
/// `SearchOptions` `Clone`-able and `'static`. Every current call site is a
/// `CommitmentConfig` associated function, so no captures are needed. If a
/// future config needs closures, switch the field to a
/// `Box<dyn Fn(...) -> _ + Send + Sync + 'static>` — the change is internal
/// to this module and a handful of callers.
pub(crate) struct SearchOptions {
    /// Public schedule lookup key the search is solving for.
    pub key: AkitaScheduleLookupKey,
    /// Cyclotomic ring dimension `D` mirroring `Cfg::D`. Needed alongside
    /// `decomposition` so root-direct commit-layout derivation can read it
    /// without `<Cfg>` plumbing through every internal function.
    pub ring_dimension: usize,
    /// Root decomposition parameters; `field_bits()` is the effective
    /// field bit width consumed by sizing formulas.
    pub decomposition: DecompositionParams,
    /// SIS modulus family used by every generated SIS-floor lookup along
    /// this search path.
    pub sis_modulus_family: SisModulusFamily,
    /// Challenge-field bit width consumed by Fiat-Shamir scalar bytes in
    /// proof-size accounting.
    pub challenge_field_bits: u32,
    /// Logical extension-opening width (base-field path uses `1`).
    pub extension_opening_width: usize,
    /// Multiplicative expansion factor on recursive witnesses carried at
    /// extension-opening boundaries.
    pub recursive_witness_expansion: usize,
    /// Number of public opening rows used by each recursive fold.
    pub recursive_public_rows: usize,
    /// Optional generated schedule table to consult before DP search,
    /// gated by the [`ScheduleSearchMode`] argument on
    /// [`find_optimal_schedule`].
    pub table: Option<GeneratedScheduleTable>,
    /// Stage-1 sparse challenge selector for a given ring dimension. The
    /// hook is Result-returning so config-side validation errors propagate
    /// instead of panicking on the verifier replay path.
    pub stage1_challenge_config: fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    /// Pre-computed singleton schedule plan
    /// (`Cfg::schedule_plan(singleton_key(num_vars))`). Consumed by
    /// [`akita_derive::current_level_layout_with_log_basis`] as the
    /// table-lookup shortcut; precomputed once per search since
    /// `num_vars` is fixed.
    pub schedule_plan: Option<AkitaSchedulePlan>,
    /// Pre-computed commitment envelope for `key.num_vars`. Consumed by
    /// [`akita_derive::direct_level_params_with_log_basis`] and
    /// [`akita_derive::current_level_layout_with_log_basis`] at
    /// candidate-evaluation sites.
    pub envelope: CommitmentEnvelope,
    /// Infinity-norm expansion introduced when claim-field coordinates are
    /// embedded into the ring subfield via `psi`.
    pub ring_subfield_embedding_norm_bound: u32,
    /// Inclusive `(min, max)` log-basis search range at a state.
    pub log_basis_search_range: fn(AkitaScheduleInputs) -> (u32, u32),
    /// Per-level fold-round challenge shape; mirrors
    /// `CommitmentConfig::fold_challenge_shape_at_level`. The table fast path
    /// uses this through schedule materialization; the from-scratch root DP
    /// search still derives candidates from flat default params.
    pub fold_challenge_shape: fn(AkitaScheduleInputs) -> akita_challenges::TensorChallengeShape,
}

impl SearchOptions {
    /// Effective field bit width for this search (from `decomposition`).
    fn field_bits(&self) -> u32 {
        self.decomposition.field_bits()
    }
}

/// Root `z` protocol vectors represented by a schedule lookup key.
fn num_z_vectors(key: AkitaScheduleLookupKey) -> usize {
    key.num_z_vectors
}

fn derive_batched_root_level_derivation(
    opts: &SearchOptions,
    num_vars: usize,
    root_lp: &LevelParams,
    num_claims: usize,
) -> Result<(LevelParams, LevelParams), AkitaError> {
    let inputs = AkitaScheduleInputs {
        num_vars,
        level: 0,
        current_w_len: root_current_w_len(root_lp),
    };
    let level_lp = scale_batched_root_layout(
        root_lp,
        num_claims,
        (opts.stage1_challenge_config)(root_lp.ring_dimension)?.l1_norm(),
        opts.field_bits(),
    )?;
    let derived_root_lp = akita_derive::sis_derived_root_params_for_layout(
        opts.sis_modulus_family,
        level_lp.ring_dimension,
        opts.decomposition,
        (opts.stage1_challenge_config)(level_lp.ring_dimension)?,
        opts.ring_subfield_embedding_norm_bound,
        inputs,
        &level_lp,
    )?
    .with_layout(&level_lp);
    Ok((level_lp, derived_root_lp))
}

// -----------------------------------------------------------------------
// Single-level evaluation
// -----------------------------------------------------------------------

/// All layout data for one candidate fold level.
struct CandidateLevelParams {
    proof_lp: LevelParams,
    lp: LevelParams,
    next_w_len: usize,
}

/// Derive the layout for folding at `(level, w_len, log_basis)`.
/// Returns `None` if the layout is infeasible or doesn't shrink the witness.
fn derive_candidate_level_params(
    opts: &SearchOptions,
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
        opts.sis_modulus_family,
        opts.ring_dimension,
        opts.decomposition,
        opts.ring_subfield_embedding_norm_bound,
        opts.schedule_plan.as_ref(),
        &opts.envelope,
        opts.stage1_challenge_config,
        inputs,
        log_basis,
    ) {
        Ok(level_lp) => level_lp,
        Err(_) => return Ok(None),
    };

    let fb = opts.field_bits();
    if opts.recursive_public_rows != 1 {
        return Err(AkitaError::InvalidSetup(
            "recursive schedule planning currently requires exactly one public row".to_string(),
        ));
    }
    // Recursive folds carry one recursive witness and open it at one prepared
    // recursive point. Root batching is reflected only at level 0.
    let w_ring_elements = w_ring_element_count_with_counts_bits(fb, &level_lp, 1, 1, 1, 1)?;
    let next_w_len = w_ring_elements
        .checked_mul(level_lp.ring_dimension)
        .and_then(|len| len.checked_mul(opts.recursive_witness_expansion))
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
        proof_lp: level_lp.clone(),
        lp: level_lp,
        next_w_len,
    }))
}

fn compute_level_proof_size(
    opts: &SearchOptions,
    candidate: &CandidateLevelParams,
    next_level_params: &LevelParams,
    num_public_outputs: usize,
) -> usize {
    level_proof_bytes(
        opts.field_bits(),
        opts.challenge_field_bits,
        &candidate.proof_lp,
        &candidate.lp,
        next_level_params,
        candidate.next_w_len,
        num_public_outputs,
    )
}

fn compute_terminal_level_proof_size(
    opts: &SearchOptions,
    candidate: &CandidateLevelParams,
    terminal_next_w_len: usize,
    num_public_outputs: usize,
) -> usize {
    terminal_level_proof_bytes(
        opts.field_bits(),
        opts.challenge_field_bits,
        &candidate.proof_lp,
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

fn extension_opening_reduction_level_bytes(
    opts: &SearchOptions,
    key: AkitaScheduleLookupKey,
    fold_level: usize,
    current_w_len: usize,
) -> Result<usize, AkitaError> {
    let width = opts.extension_opening_width;
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
        opts.challenge_field_bits,
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
/// shape (computed from the last fold's `lp` under [`MRowLayout::WithoutDBlock`])
/// overwrites this once the enclosing fold candidate is known.
fn to_direct_step(opts: &SearchOptions, current_w_len: usize, log_basis: u32) -> Step {
    let expansion = opts.recursive_witness_expansion;
    assert!(expansion > 0, "recursive witness expansion must be nonzero");
    assert_eq!(
        current_w_len % expansion,
        0,
        "terminal recursive witness length must be divisible by the extension expansion"
    );
    let witness_shape = CleartextWitnessShape::PackedDigits((current_w_len / expansion, log_basis));
    let direct_bytes = direct_witness_bytes(opts.field_bits(), &witness_shape);
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
fn finalize_terminal_direct_witness_shape(
    opts: &SearchOptions,
    suffix_steps: &mut [Step],
    candidate: &CandidateLevelParams,
    num_points: usize,
    num_t_vectors: usize,
    num_w_vectors: usize,
    num_public_rows: usize,
    num_vars: usize,
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
    let CleartextWitnessShape::PackedDigits((_, log_basis)) = direct.witness_shape else {
        return Err(AkitaError::InvalidSetup(
            "terminal direct finalizer expected a packed-digit witness".to_string(),
        ));
    };
    let ring_count = w_ring_element_count_with_counts_for_layout_bits(
        opts.field_bits(),
        &candidate.lp,
        num_points,
        num_t_vectors,
        num_w_vectors,
        num_public_rows,
        MRowLayout::WithoutDBlock,
    )
    .expect("terminal recursive witness length overflow");
    let terminal_field_len = ring_count
        .checked_mul(candidate.lp.ring_dimension)
        .expect("terminal recursive witness length overflow");
    let witness_shape = CleartextWitnessShape::PackedDigits((terminal_field_len, log_basis));
    let direct_bytes = direct_witness_bytes(opts.field_bits(), &witness_shape);
    // Bake the SIS-secure terminal-direct level params onto the step so
    // prover/verifier (and the materializer, when this candidate is
    // emitted via the offline table) can read them straight from the
    // schedule. This mirrors what the prover's
    // `scheduled_next_level_params` callback used to compute on the fly.
    let level_params = akita_derive::direct_level_params_with_log_basis(
        opts.sis_modulus_family,
        opts.ring_dimension,
        opts.decomposition,
        (opts.stage1_challenge_config)(opts.ring_dimension)?,
        opts.ring_subfield_embedding_norm_bound,
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

fn basis_range(
    opts: &SearchOptions,
    num_vars: usize,
    level: usize,
    current_w_len: usize,
) -> std::ops::RangeInclusive<u32> {
    let (lo, hi) = (opts.log_basis_search_range)(AkitaScheduleInputs {
        num_vars,
        level,
        current_w_len,
    });
    lo..=hi
}

fn level_params_from_fold_step(opts: &SearchOptions, step: &FoldStep) -> LevelParams {
    if let Ok(config) = (opts.stage1_challenge_config)(step.params.ring_dimension) {
        debug_assert_eq!(config.l1_norm(), step.params.challenge_l1_mass());
    }
    step.params.clone()
}

fn successor_level_params_from_schedule(
    opts: &SearchOptions,
    num_vars: usize,
    level: usize,
    current_w_len: usize,
    suffix_steps: &[Step],
) -> Result<LevelParams, AkitaError> {
    match suffix_steps
        .first()
        .expect("optimal suffix schedule must contain at least one step")
    {
        Step::Fold(step) => Ok(level_params_from_fold_step(opts, step)),
        Step::Direct(step) => akita_derive::direct_level_params_with_log_basis(
            opts.sis_modulus_family,
            opts.ring_dimension,
            opts.decomposition,
            (opts.stage1_challenge_config)(opts.ring_dimension)?,
            opts.ring_subfield_embedding_norm_bound,
            &opts.envelope,
            AkitaScheduleInputs {
                num_vars,
                level,
                current_w_len,
            },
            step.log_basis(opts.field_bits()),
        ),
    }
}

// -----------------------------------------------------------------------
// DP — suffix search
// -----------------------------------------------------------------------

type ScheduleMemo = HashMap<(usize, usize, u32), (usize, Vec<Step>)>;

fn derive_optimal_suffix_schedule(
    opts: &SearchOptions,
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
            opts.sis_modulus_family,
            opts.ring_dimension,
            opts.decomposition,
            match (opts.stage1_challenge_config)(opts.ring_dimension) {
                Ok(s) => s,
                Err(_) => return Ok((usize::MAX, Vec::new())),
            },
            opts.ring_subfield_embedding_norm_bound,
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
        let placeholder = to_direct_step(opts, current_w_len, current_lb);
        let Step::Direct(direct) = &placeholder else {
            unreachable!("to_direct_step returns Step::Direct");
        };
        best_cost = direct.direct_bytes;
        best_schedule = vec![placeholder];
    }

    if depth <= MAX_RECURSION_DEPTH {
        for lb in basis_range(opts, num_vars, level, current_w_len) {
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
                    &mut suffix_steps,
                    &candidate,
                    1,
                    1,
                    1,
                    1,
                    num_vars,
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
            let Ok(eor_bytes) = extension_opening_reduction_level_bytes(
                opts,
                AkitaScheduleLookupKey::singleton(num_vars),
                level,
                current_w_len,
            ) else {
                continue;
            };
            let level_proof_size = if suffix_is_terminal {
                let terminal_next_w_len = next_w_len_override
                    .expect("suffix_is_terminal branch populates next_w_len_override above");
                compute_terminal_level_proof_size(
                    opts,
                    &candidate,
                    terminal_next_w_len,
                    opts.recursive_public_rows,
                ) + eor_bytes
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
                compute_level_proof_size(
                    opts,
                    &candidate,
                    &next_level_params,
                    opts.recursive_public_rows,
                ) + eor_bytes
            };

            let total = level_proof_size + suffix_cost;
            if total < best_cost {
                best_cost = total;
                let mut steps = Vec::with_capacity(1 + suffix_steps.len());
                steps.push(to_fold_step(
                    &candidate,
                    current_w_len,
                    level_proof_size,
                    opts.field_bits(),
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

fn root_w_ring_element_count(
    opts: &SearchOptions,
    lp: &LevelParams,
    key: AkitaScheduleLookupKey,
) -> Result<usize, AkitaError> {
    let fb = opts.field_bits();
    let r_decomp = compute_num_digits_full_field(fb, lp.log_basis);

    let t_vectors = key.num_t_vectors;
    let w_vectors = key.num_w_vectors;
    let z_vectors = num_z_vectors(key);
    let num_points = key.num_points;

    let w_hat = w_vectors * lp.num_blocks * lp.num_digits_open;
    let t_hat = t_vectors * lp.num_blocks * lp.a_key.row_len() * lp.num_digits_open;
    let z_folded_rings = z_vectors * lp.inner_width() * lp.num_digits_fold;
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
        Ok(w_hat + t_hat + b_blinding + d_blinding + z_folded_rings + r)
    }
    #[cfg(not(feature = "zk"))]
    {
        Ok(w_hat + t_hat + z_folded_rings + r)
    }
}

// -----------------------------------------------------------------------
// Key-driven root candidate + entry point
// -----------------------------------------------------------------------

fn derive_root_candidate(
    opts: &SearchOptions,
    num_vars: usize,
    root_w_len: usize,
    log_basis: u32,
    key: AkitaScheduleLookupKey,
) -> Result<Option<CandidateLevelParams>, AkitaError> {
    let inputs = AkitaScheduleInputs {
        num_vars,
        level: 0,
        current_w_len: root_w_len,
    };

    let root_lp = match akita_derive::root_level_layout_with_log_basis(
        opts.sis_modulus_family,
        opts.ring_dimension,
        opts.decomposition,
        (opts.stage1_challenge_config)(opts.ring_dimension)?,
        opts.ring_subfield_embedding_norm_bound,
        inputs,
        log_basis,
    ) {
        Ok(root_lp) => root_lp,
        Err(_) => return Ok(None),
    };
    let fb = opts.field_bits();

    let alpha = root_lp.ring_dimension.trailing_zeros() as usize;
    let Some(reduced_vars) = num_vars.checked_sub(alpha) else {
        return Ok(None);
    };
    if reduced_vars < 1 {
        return Ok(None);
    }

    let mut best: Option<CandidateLevelParams> = None;

    let r_lo: usize = if reduced_vars >= 3 { 1 } else { 0 };
    let r_hi: usize = reduced_vars.saturating_sub(1).max(r_lo);

    for r_vars in r_lo..=r_hi {
        let m_vars = reduced_vars - r_vars;
        let per_poly_fold = compute_num_digits_fold_with_claims(
            r_vars,
            root_lp.challenge_l1_mass(),
            root_lp.log_basis,
            1,
            fb,
        );

        let Some(num_blocks) = 1usize.checked_shl(r_vars as u32) else {
            continue;
        };
        let Some(block_len) = 1usize.checked_shl(m_vars as u32) else {
            continue;
        };
        let Some(inner_width) = block_len.checked_mul(root_lp.num_digits_commit) else {
            continue;
        };
        let Some(outer_width) = root_lp
            .a_key
            .row_len()
            .checked_mul(root_lp.num_digits_open)
            .and_then(|x| x.checked_mul(num_blocks))
        else {
            continue;
        };
        let Some(d_matrix_width) = root_lp.num_digits_open.checked_mul(num_blocks) else {
            continue;
        };

        let d = root_lp.ring_dimension;
        let Ok(a_key) = AjtaiKeyParams::try_new(
            opts.sis_modulus_family,
            root_lp.a_key.row_len(),
            inner_width,
            root_lp.a_key.collision_inf(),
            d,
        ) else {
            continue;
        };
        let Ok(b_key) = AjtaiKeyParams::try_new(
            opts.sis_modulus_family,
            root_lp.b_key.row_len(),
            outer_width,
            root_lp.b_key.collision_inf(),
            d,
        ) else {
            continue;
        };
        let Ok(d_key) = AjtaiKeyParams::try_new(
            opts.sis_modulus_family,
            root_lp.d_key.row_len(),
            d_matrix_width,
            root_lp.d_key.collision_inf(),
            d,
        ) else {
            continue;
        };

        let candidate_lp = LevelParams {
            ring_dimension: d,
            log_basis: root_lp.log_basis,
            a_key,
            b_key,
            d_key,
            num_blocks,
            block_len,
            m_vars,
            r_vars,
            stage1_config: root_lp.stage1_config.clone(),
            fold_challenge_shape: root_lp.fold_challenge_shape,
            num_digits_commit: root_lp.num_digits_commit,
            num_digits_open: root_lp.num_digits_open,
            num_digits_fold: per_poly_fold,
        };

        let Ok((level_lp, proof_lp)) =
            derive_batched_root_level_derivation(opts, num_vars, &candidate_lp, key.num_t_vectors)
        else {
            continue;
        };
        let raw_w_ring = root_w_ring_element_count(opts, &level_lp, key)?;
        let next_w_len = raw_w_ring
            .checked_mul(level_lp.ring_dimension)
            .and_then(|len| len.checked_mul(opts.recursive_witness_expansion))
            .ok_or_else(|| {
                AkitaError::InvalidSetup("root recursive witness length overflow".into())
            })?;

        let next_bits = next_w_len.checked_mul(log_basis as usize).ok_or_else(|| {
            AkitaError::InvalidSetup("root next witness bit length overflow".into())
        })?;
        let root_bits = root_w_len
            .checked_mul(fb as usize)
            .ok_or_else(|| AkitaError::InvalidSetup("root witness bit length overflow".into()))?;
        if next_bits >= root_bits {
            continue;
        }

        if best.as_ref().is_none_or(|b| next_w_len < b.next_w_len) {
            best = Some(CandidateLevelParams {
                proof_lp,
                lp: level_lp,
                next_w_len,
            });
        }
    }

    Ok(best)
}

/// Consult the offline schedule tables for a pre-computed answer.
fn offline_schedule_for_key(opts: &SearchOptions) -> Result<Option<Schedule>, AkitaError> {
    let Some(table) = opts.table else {
        return Ok(None);
    };
    // Use `Cfg::Field`-agnostic materialization: the materializer takes F
    // only as a marker (its only use is bit-width sizing, which we already
    // supply via decomposition.field_bits()). We thread the base-field
    // marker through with `()` -> impossible; instead pick a small canonical
    // field that satisfies the trait bound. Materialization arithmetic is
    // bit-width driven and uses `root_decomp.field_bits()` everywhere, so
    // this is just a phantom.
    use akita_field::Prime128OffsetA7F7 as PhantomField;
    let plan = schedule_plan_from_table::<PhantomField, _>(
        opts.key,
        table,
        PlanPolicy {
            sis_family: opts.sis_modulus_family,
            ring_dimension: opts.ring_dimension,
            root_decomp: opts.decomposition,
            challenge_field_bits: opts.challenge_field_bits,
            recursive_public_rows: opts.recursive_public_rows,
            extension_opening_width: opts.extension_opening_width,
            stage1_challenge_config: opts.stage1_challenge_config,
            envelope: opts.envelope,
            ring_subfield_norm_bound: opts.ring_subfield_embedding_norm_bound,
            fold_challenge_shape: opts.fold_challenge_shape,
        },
    )?;
    Ok(plan.map(|plan| schedule_from_plan(&plan, opts.field_bits())))
}

/// Find the optimal schedule for a root schedule lookup key under `Cfg`.
///
/// Behaviour is driven by the [`ScheduleSearchMode`] argument; see its
/// docs for the runtime-vs-regeneration contract.
///
/// # Errors
///
/// Returns an error if vector counts are invalid, if the witness length
/// overflows, or if generated-table materialization fails.
pub fn find_optimal_schedule<Cfg: CommitmentConfig>(
    key: AkitaScheduleLookupKey,
    mode: ScheduleSearchMode,
) -> Result<Schedule, AkitaError> {
    let opts = search_options_for_cfg::<Cfg>(key, mode)?;
    let opts = &opts;
    let t_vectors = key.num_t_vectors;
    let w_vectors = key.num_w_vectors;
    let z_vectors = num_z_vectors(key);
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
    let num_vars = key.num_vars;

    if mode.consults_offline_tables() {
        if let Some(schedule) = offline_schedule_for_key(opts)? {
            tracing::debug!(
                num_vars,
                num_points,
                num_t_vectors = t_vectors,
                num_w_vectors = w_vectors,
                num_z_vectors = z_vectors,
                total_bytes = schedule.total_bytes,
                "schedule planner: served from offline schedule tables"
            );
            return Ok(schedule);
        }
    }

    let root_w_len = 1usize
        .checked_shl(num_vars as u32)
        .ok_or_else(|| AkitaError::InvalidSetup("witness too large".into()))?;

    let fb = opts.field_bits();
    let root_direct_shape = CleartextWitnessShape::FieldElements(root_w_len);
    let mut best_cost = direct_witness_bytes(fb, &root_direct_shape);
    // Populate `commit_params` so consumers don't have to re-derive the
    // root commit layout from the schedule shape. Uses the same primitives
    // as the (deleted) `akita_root_commitment_layout` fallback. Large
    // `num_vars` baselines whose root layout exceeds the audited SIS-floor
    // are still valid as a proof-bytes upper bound for the DP comparator
    // (no production caller asks for their singleton commit params); they
    // record `commit_params: None`.
    let root_direct_commit_params = {
        let singleton = akita_derive::root_direct_commit_layout(
            opts.sis_modulus_family,
            Cfg::D,
            opts.decomposition,
            (opts.stage1_challenge_config)(Cfg::D)?,
            opts.ring_subfield_embedding_norm_bound,
            num_vars,
            opts.decomposition.log_basis,
        )
        .ok();
        // Apply the same batched-root scaling as the Fold-root path so
        // `commit_params` carries the correct width for non-singleton
        // incidences. Mirrors the materializer's Fold/Direct symmetry.
        let root_is_batched = key.num_points != 1
            || key.num_t_vectors != 1
            || key.num_w_vectors != 1
            || key.num_z_vectors != 1;
        match singleton {
            Some(lp) if root_is_batched => Some(scale_batched_root_layout(
                &lp,
                key.num_t_vectors,
                (opts.stage1_challenge_config)(Cfg::D)?.l1_norm(),
                opts.field_bits(),
            )?),
            other => other,
        }
    };
    let mut best_steps: Vec<Step> = vec![Step::Direct(DirectStep {
        current_w_len: root_w_len,
        witness_shape: root_direct_shape,
        direct_bytes: best_cost,
        commit_params: root_direct_commit_params,
        // Root-direct never has a next level after itself; the prover
        // walk stops at this direct step.
        level_params: None,
    })];
    let mut memo = ScheduleMemo::new();

    for root_lb in basis_range(opts, num_vars, 0, root_w_len) {
        let Some(candidate) = derive_root_candidate(opts, num_vars, root_w_len, root_lb, key)?
        else {
            continue;
        };
        let (mut suffix_cost, mut suffix_steps) = derive_optimal_suffix_schedule(
            opts,
            &mut memo,
            num_vars,
            1,
            candidate.next_w_len,
            root_lb,
            0,
        )?;
        if suffix_steps.is_empty() {
            continue;
        }
        let suffix_is_terminal = matches!(suffix_steps.first(), Some(Step::Direct(_)));
        let Ok(eor_bytes) = extension_opening_reduction_level_bytes(opts, key, 0, root_w_len)
        else {
            continue;
        };
        let next_w_len_override = if suffix_is_terminal {
            let old_direct_bytes = match suffix_steps.first().expect("suffix non-empty") {
                Step::Direct(direct) => direct.direct_bytes,
                Step::Fold(_) => unreachable!("suffix_is_terminal guard"),
            };
            finalize_terminal_direct_witness_shape(
                opts,
                &mut suffix_steps,
                &candidate,
                num_points,
                t_vectors,
                w_vectors,
                z_vectors,
                num_vars,
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
            compute_terminal_level_proof_size(opts, &candidate, terminal_next_w_len, z_vectors)
                + eor_bytes
        } else {
            let Ok(next_level_params) = successor_level_params_from_schedule(
                opts,
                num_vars,
                1,
                candidate.next_w_len,
                &suffix_steps,
            ) else {
                continue;
            };
            compute_level_proof_size(opts, &candidate, &next_level_params, z_vectors) + eor_bytes
        };

        let total = root_proof_size + suffix_cost;
        if total < best_cost {
            best_cost = total;
            let mut steps = Vec::with_capacity(1 + suffix_steps.len());
            steps.push(to_fold_step(
                &candidate,
                root_w_len,
                root_proof_size,
                opts.field_bits(),
                next_w_len_override,
            ));
            steps.extend(suffix_steps);
            best_steps = steps;
        }
    }

    let num_folds = best_steps
        .iter()
        .filter(|s| matches!(s, Step::Fold(_)))
        .count();
    tracing::info!(
        num_vars,
        num_points,
        num_t_vectors = t_vectors,
        num_w_vectors = w_vectors,
        num_z_vectors = z_vectors,
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
        assert_eq!(num_z_vectors(key), 1);
    }

    #[test]
    #[cfg(not(feature = "zk"))]
    fn planner_uses_generated_schedule_fast_path() {
        // Pick a real preset and a singleton key its table covers. The fast
        // path should short-circuit before DP runs.
        use akita_config::proof_optimized::fp128;
        let key = AkitaScheduleLookupKey::singleton(8);
        let schedule =
            find_optimal_schedule::<fp128::D64Full>(key, ScheduleSearchMode::RuntimeTableSeeded)
                .expect("offline schedule lookup");
        assert!(!schedule.steps.is_empty());
    }
}
