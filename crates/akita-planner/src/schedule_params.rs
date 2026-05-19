//! Schedule planner that finds the global minimum proof size.
//!
//! A single exhaustive DP over `(level, w_len, log_basis)` states.  At each
//! state, every feasible basis is tried; `level_proof_bytes` uses the
//! smallest `next_commit` across all next-level bases; the suffix is
//! recursed into unconstrained.
//!
//! Uses config-supplied protocol layout derivation and the same proof-size
//! formulas as runtime generated-schedule validation.

use std::collections::HashMap;

use crate::proof_size::{stage1_bytes_optimized, sumcheck_rounds};
use crate::PlannerConfig;
use akita_field::AkitaError;
use akita_types::layout::digit_math::{
    compute_num_digits_fold_with_claims, compute_num_digits_full_field,
};
use akita_types::{
    direct_witness_bytes, field_bytes, level_proof_bytes,
    planned_joint_next_w_len_with_setup_group, planned_joint_next_w_len_with_setup_group_tiered,
    planned_setup_claim_reduction_rounds, planned_setup_field_len,
    planned_verifier_setup_storage_field_len_for_setup, planned_w_ring_element_count_with_claims,
    root_current_w_len, scale_batched_root_layout, schedule_from_plan, tiered_setup_group_lp,
    AjtaiKeyParams, AkitaRootBatchSummary, AkitaScheduleInputs, AkitaScheduleLookupKey, DirectStep,
    DirectWitnessShape, FoldStep, LevelParams, Schedule, Step, TieredSetupParams, WitnessShape,
};

const MAX_RECURSION_DEPTH: usize = 12;

fn derive_batched_root_level_derivation<Cfg>(
    max_num_vars: usize,
    root_lp: &LevelParams,
    num_claims: usize,
) -> Result<(LevelParams, LevelParams), AkitaError>
where
    Cfg: PlannerConfig,
{
    let inputs = AkitaScheduleInputs {
        max_num_vars,
        level: 0,
        current_w_len: root_current_w_len(root_lp),
    };
    let level_lp = scale_batched_root_layout(root_lp, num_claims, Cfg::planner_field_bits())?;
    let derived_root_lp =
        Cfg::planner_root_level_params_for_layout_with_log_basis(inputs, &level_lp)?;
    Ok((level_lp, derived_root_lp))
}

// -----------------------------------------------------------------------
// Single-level evaluation
// -----------------------------------------------------------------------

/// All layout data for one candidate fold level.
struct CandidateLevelParams {
    proof_lp: LevelParams,
    lp: LevelParams,
    /// Field-element length of the W output this level produces, which
    /// becomes the next level's `current_w_len`. Under un-tiered cascade
    /// (book §5.3 lines 627-660, `f = 1`) this is the multi-group joint
    /// fold output when `s_field_len_in > 0`; otherwise the single-claim
    /// fold output. The runtime's `handle[0].w.len()` at the next level
    /// matches this.
    next_w_len: usize,
    w_ring: usize,
    /// Field-element length of the `S` polynomial routed from THIS level
    /// to the next as the second commitment-group handle (book §5.3
    /// "split commitment"). Zero when this level does not route `S`
    /// recursively (no `use_setup_claim_reduction` or no fold suffix
    /// to discharge through).
    s_field_len_emitted: usize,
    /// Phase D-full Slice G tiered shape selected by the planner for the
    /// `S` group routed from THIS level. `un_tiered()` (`f = 1`) for the
    /// Slice F baseline; `Cfg::planner_setup_shrink_factor()` packaged as
    /// `TieredSetupParams::new(f)?` when the planner enables the tiered
    /// `|S|/f` cascade.
    tier_setup_params: TieredSetupParams,
    /// Number of distinct opening points this level joint-opens (= the
    /// number of `y_ring` slots in the emitted level proof). Set by the
    /// incoming cascade state:
    ///
    /// * `1` — singleton recursive level (no incoming `S`).
    /// * `2` — un-tiered cascade incoming (W, S).
    /// * `3` — tiered cascade incoming (W, chunks, meta).
    ///
    /// Mirrors `num_groups_for_y` in
    /// `prove_recursive_multi_fold_with_params`; the on-wire proof
    /// carries one `y_ring` per group.
    proof_num_eval_rows: usize,
    /// Optional setup-side claim-reduction sumcheck round count for the
    /// proof emitted at THIS level (book §5.3 line 658, §5.4 line 752).
    /// `Some(rounds)` whenever `lp.use_setup_claim_reduction == true`
    /// (CR fires unconditionally on CR-on levels; the cleartext-S
    /// discharge path still emits the same payload). `None` otherwise.
    setup_claim_reduction_rounds: Option<usize>,
    /// Setup-polynomial length this CR-on level binds, in field elements.
    /// The planner uses this both for routed setup-storage precompute and
    /// for cleartext-discharge work. Zero when CR is off.
    setup_field_len: usize,
}

/// Derive the layout for folding at `(level, w_len, log_basis)`.
/// Returns `None` if the layout is infeasible or doesn't shrink the witness.
///
/// `s_field_len_in` is the field-element length of the `S` polynomial
/// routed INTO this level by the previous level's cascade (book §5.3).
/// Zero means no cascade is active at this level (single-claim view);
/// non-zero activates the multi-group `(W, S)` joint-open path here and
/// uses cascade-aware fold-output sizing. `routes_setup_recursively` is
/// `true` when this level should push `S` to the suffix (i.e., the next
/// level is a fold step and `use_setup_claim_reduction` is on for this
/// level).
fn derive_candidate_level_params<Cfg: PlannerConfig>(
    max_num_vars: usize,
    level: usize,
    current_w_len: usize,
    log_basis: u32,
    s_field_len_in: usize,
    routes_setup_recursively: bool,
) -> Option<CandidateLevelParams> {
    let inputs = AkitaScheduleInputs {
        max_num_vars,
        level,
        current_w_len,
    };

    let level_lp = if level == 0 {
        Cfg::planner_root_level_layout_with_log_basis(inputs, log_basis).ok()?
    } else {
        Cfg::planner_current_level_layout_with_log_basis(inputs, log_basis).ok()?
    };

    let fb = Cfg::planner_field_bits();
    // Phase D-full Slice G tier shape selected by the config (book §5.4).
    //
    // Two tiers participate at every level beyond the root:
    //
    // * `incoming_tier` is the tier the *previous* level used to emit
    //   `S`. It dictates how the runtime structures the routed
    //   handle at this level's input, and therefore the per-chunk LP
    //   used by the joint `(W, S)` opening sizing here. The DP only
    //   reaches `s_field_len_in > 0` through a chain whose immediate
    //   predecessor routed with `_at_level(level - 1)`, so we can
    //   recover the incoming tier from the configured per-level
    //   policy without threading extra state through the memo key.
    //
    // * `outgoing_tier` is the tier *this* level uses when it emits
    //   `S` to the next level (`_at_level(level)`). Carried on the
    //   `FoldStep::tier_setup_params` so the runtime/cascade matrix
    //   sizing know how to chunk the emitted handle.
    //
    // Book §5.8 line 1170 prescribes per-level tier `f_{L0} = 8`,
    // `f_{L1} = 4`. Configs that want a single uniform `f` get it via
    // the default impl of `planner_setup_shrink_factor_at_level`
    // (which delegates to `planner_setup_shrink_factor`); cascade
    // configs override the per-level hook to return different `f`
    // per recursion level.
    let outgoing_tier =
        TieredSetupParams::new(Cfg::planner_setup_shrink_factor_at_level(level).max(1))
            .unwrap_or_else(|_| TieredSetupParams::un_tiered());
    let incoming_tier = if level == 0 {
        TieredSetupParams::un_tiered()
    } else {
        TieredSetupParams::new(Cfg::planner_setup_shrink_factor_at_level(level - 1).max(1))
            .unwrap_or_else(|_| TieredSetupParams::un_tiered())
    };
    // Phase D-full v2 cascade: when the previous level routed `S`
    // (`s_field_len_in > 0`), this level joint-opens `(W, S)` as
    // multi-group. Under un-tiered (`f = 1`, book §5.3 lines 627-660)
    // it is 2 groups; under tiered (`f > 1`, book §5.4 lines 709-754)
    // it is `k + 2` groups (W + k chunks + meta). The `S`-group LP
    // is derived from the incoming `S` size, the outer LP, and the
    // *incoming* tier via `tiered_setup_group_lp` (degenerates to
    // the un-tiered helper when `f = 1`).
    let s_lp_in = if s_field_len_in > 0 {
        tiered_setup_group_lp(&level_lp, s_field_len_in, incoming_tier).ok()
    } else {
        None
    };
    let (w_ring, natural_next_w_len) = if let Some(s_lp) = &s_lp_in {
        let num_eval_rows = if incoming_tier.is_tiered() {
            // Phase 5 grouping merges k chunks into ONE commitment
            // group (sharing chunk_lp + tier marker) so the next-level
            // multi-claim infra sees three opening points: W, the
            // tiered chunks group (claim_count = k, one shared point),
            // and meta. `num_eval_rows = 3` per the merged grouping.
            3
        } else {
            2
        };
        let joint_field = if incoming_tier.is_tiered() {
            planned_joint_next_w_len_with_setup_group_tiered(
                fb,
                &level_lp,
                s_lp,
                incoming_tier,
                num_eval_rows,
            )
        } else {
            planned_joint_next_w_len_with_setup_group(fb, &level_lp, s_lp, num_eval_rows)
        };
        let w_ring = joint_field / level_lp.ring_dimension;
        (w_ring, joint_field)
    } else {
        let w_ring = planned_w_ring_element_count_with_claims(fb, &level_lp, 1);
        (w_ring, w_ring * level_lp.ring_dimension)
    };

    // Cascade-aware M-table shape at THIS level. The number of distinct
    // opening points (and therefore commitment groups in the joint open)
    // is set by the incoming cascade:
    //
    // * un-tiered cascade (`incoming_tier.shrink_factor == 1`, book
    //   §5.3): 2 groups `(W, S)`;
    // * tiered cascade (`incoming_tier.is_tiered()`, book §5.4): 3
    //   groups `(W, chunks, meta)` — the tiered chunks expand into a
    //   `claim_count = k` group and the meta commit sits alongside as
    //   a third opening point;
    // * singleton: 1 group (only `W`).
    let (num_eval_rows, num_commitment_groups) =
        match (s_lp_in.is_some(), incoming_tier.is_tiered()) {
            (true, true) => (3, 3),
            (true, false) => (2, 2),
            _ => (1, 1),
        };
    let setup_field_len = if level_lp.use_setup_claim_reduction {
        planned_setup_field_len(
            &level_lp,
            s_lp_in.as_ref(),
            incoming_tier,
            num_eval_rows,
            num_commitment_groups,
        )
    } else {
        0
    };
    // `s_field_len_emitted` is the `S` polynomial size this level pushes
    // to the next as a separate commitment-group handle.
    let s_field_len_emitted = if routes_setup_recursively {
        setup_field_len
    } else {
        0
    };
    // Book §5.3 line 658 / §5.4 line 752 setup-claim-reduction
    // sumcheck round count. Fires unconditionally on CR-on levels
    // regardless of whether `S` is routed recursively or discharged
    // via cleartext mle: the runtime emits the same payload either
    // way (see `crates/akita-prover/src/protocol/flow.rs` line 1315).
    // Sized against the same `(s_lp_in, incoming_tier, num_eval_rows,
    // num_commitment_groups)` shape that `planned_setup_field_len`
    // consumes so the cost model tracks the joint-open envelope
    // exactly, including the tiered chunks + meta expansion.
    let setup_claim_reduction_rounds = if level_lp.use_setup_claim_reduction {
        Some(planned_setup_claim_reduction_rounds(
            &level_lp,
            s_lp_in.as_ref(),
            incoming_tier,
            num_eval_rows,
            num_commitment_groups,
        ))
    } else {
        None
    };
    // Carry the OUTGOING tier shape only when this level actually
    // routes the S group recursively. Otherwise the FoldStep is the
    // un-tiered baseline and the runtime keeps the existing
    // single-claim path.
    let tier_setup_params = if s_field_len_emitted > 0 {
        outgoing_tier
    } else {
        TieredSetupParams::un_tiered()
    };

    let input_elem_bits = if level == 0 {
        fb as usize
    } else {
        log_basis as usize
    };
    // Shrinkage check: with cascade active at this level, `S` inflates
    // the fold output beyond the W-only size, so compare against the
    // joint input `current_w_len + s_field_len_in`. Without cascade,
    // this reduces to the historical W-only check.
    //
    // Note: book §5.4 Table 1 shows tiered routing grows the next-level
    // witness by the T2 ratio (1.0-3.0× at the book's intended shape
    // with shared `D_chunk/B_chunk` and smaller per-chunk SIS ranks).
    // The current Phase 3 architecture treats chunks as `k` separate
    // commitment groups (no shared-matrix collapse, no smaller
    // per-chunk SIS rank), so the witness grows by `~k` per level
    // instead of `~T2 ratio`. The shrinkage check therefore correctly
    // rejects most tiered candidates until Phase 5 (block-diagonal
    // `D_chunk/B_chunk` MLE collapse + per-chunk SIS rank shrink) is
    // implemented.
    let joint_input_len = current_w_len.saturating_add(s_field_len_in);
    if natural_next_w_len * (log_basis as usize) >= joint_input_len * input_elem_bits {
        return None;
    }

    Some(CandidateLevelParams {
        proof_lp: level_lp.clone(),
        lp: level_lp,
        next_w_len: natural_next_w_len,
        w_ring,
        s_field_len_emitted,
        tier_setup_params,
        proof_num_eval_rows: num_eval_rows,
        setup_claim_reduction_rounds,
        setup_field_len,
    })
}

/// Compute the proof bytes for this fold level against a concrete successor.
///
/// `num_public_outputs` is the number of `y_ring` slots on the wire for
/// THIS level (= number of distinct opening points in the joint open).
/// The setup-claim-reduction payload bytes (book §5.3 line 658, §5.4
/// line 752) are derived inside [`level_proof_bytes`] from `lp`,
/// `s_field_len_emitted`, and `tier_setup_params` carried on the
/// candidate so the DP scores CR-on shapes against the actual on-wire
/// payload instead of the single-claim baseline.
fn compute_level_proof_size<Cfg: PlannerConfig>(
    candidate: &CandidateLevelParams,
    next_level_params: &LevelParams,
    num_public_outputs: usize,
) -> usize {
    let fb = Cfg::planner_field_bits();
    level_proof_bytes(
        fb,
        &candidate.proof_lp,
        &candidate.lp,
        next_level_params,
        candidate.next_w_len,
        num_public_outputs,
        candidate.setup_claim_reduction_rounds,
    )
}

fn setup_storage_objective_cost<Cfg: PlannerConfig>(storage_field_len: usize) -> usize {
    if storage_field_len == 0 || Cfg::planner_setup_storage_weight() == 0 {
        return 0;
    }
    let storage_bytes = storage_field_len.saturating_mul(field_bytes(Cfg::planner_field_bits()));
    let weighted = storage_bytes.saturating_mul(Cfg::planner_setup_storage_weight());
    weighted.div_ceil(Cfg::planner_setup_storage_amortization_proofs().max(1))
}

fn cleartext_discharge_objective_cost<Cfg: PlannerConfig>(discharge_field_len: usize) -> usize {
    if discharge_field_len == 0 || Cfg::planner_cleartext_discharge_weight() == 0 {
        return 0;
    }
    discharge_field_len
        .saturating_mul(field_bytes(Cfg::planner_field_bits()))
        .saturating_mul(Cfg::planner_cleartext_discharge_weight())
}

// -----------------------------------------------------------------------
// Step construction
// -----------------------------------------------------------------------

fn to_fold_step(
    c: &CandidateLevelParams,
    current_w_len: usize,
    level_bytes: usize,
    field_bits: u32,
) -> Step {
    let per_poly_fold = compute_num_digits_fold_with_claims(
        c.lp.r_vars,
        c.lp.challenge_l1_mass(),
        c.lp.log_basis,
        1,
        field_bits,
    );
    Step::Fold(FoldStep {
        params: c.lp.clone(),
        current_w_len,
        delta_fold_per_poly: per_poly_fold,
        w_ring: c.w_ring,
        next_w_len: c.next_w_len,
        level_bytes,
        s_field_len_emitted: c.s_field_len_emitted,
        tier_setup_params: c.tier_setup_params,
    })
}

fn to_direct_step(current_w_len: usize, log_basis: u32) -> Step {
    Step::Direct(DirectStep {
        current_w_len,
        bits_per_elem: log_basis,
        direct_bytes: (current_w_len * log_basis as usize).div_ceil(8),
    })
}

/// Inclusive range of `log_basis` values to search at a given state.
fn basis_range<Cfg: PlannerConfig>(
    max_num_vars: usize,
    level: usize,
    current_w_len: usize,
) -> std::ops::RangeInclusive<u32> {
    let (lo, hi) = Cfg::planner_log_basis_search_range(AkitaScheduleInputs {
        max_num_vars,
        level,
        current_w_len,
    });
    lo..=hi
}

fn level_params_from_fold_step<Cfg: PlannerConfig>(step: &FoldStep) -> LevelParams {
    let stage1_config = Cfg::planner_stage1_challenge_config(step.params.ring_dimension);
    debug_assert_eq!(
        step.params
            .stage1_challenge_shape
            .effective_l1_mass(&stage1_config),
        step.params.challenge_l1_mass()
    );
    step.params.clone()
}

fn successor_level_params_from_schedule<Cfg: PlannerConfig>(
    max_num_vars: usize,
    level: usize,
    current_w_len: usize,
    suffix_steps: &[Step],
) -> Result<LevelParams, AkitaError> {
    match suffix_steps
        .first()
        .expect("optimal suffix schedule must contain at least one step")
    {
        Step::Fold(step) => Ok(level_params_from_fold_step::<Cfg>(step)),
        Step::Direct(step) => Cfg::planner_current_level_layout_with_log_basis(
            AkitaScheduleInputs {
                max_num_vars,
                level,
                current_w_len,
            },
            step.bits_per_elem,
        ),
    }
}

// -----------------------------------------------------------------------
// DP — suffix search
// -----------------------------------------------------------------------

#[derive(Clone)]
struct PlannedSuffix {
    objective_cost: usize,
    proof_bytes: usize,
    steps: Vec<Step>,
}

/// Memo key: `(level, w_len, log_basis, s_field_len_in)`. The last
/// component disambiguates cascade-routed multi-claim states from
/// singleton states: at the same `(level, w_len, log_basis)` a post-
/// cascade caller has `s_field_len_in > 0`, which produces a larger
/// joint fold output and a larger emitted `S` than the singleton path.
type ScheduleMemo = HashMap<(usize, usize, u32, usize), PlannedSuffix>;

fn stage1_prover_penalty<Cfg: PlannerConfig>(lp: &LevelParams, next_w_len: usize) -> usize {
    let weight = Cfg::planner_stage1_prover_weight();
    if weight == 0 {
        return 0;
    }
    let rounds = sumcheck_rounds(lp.ring_dimension as u32, next_w_len);
    weight.saturating_mul(stage1_bytes_optimized(
        rounds,
        lp.log_basis,
        Cfg::planner_field_bits(),
    ))
}

/// Find the minimum-cost suffix starting at `(level, current_w_len, current_lb)`,
/// returning both the total bytes and the step-by-step schedule.
///
/// `s_field_len_in` is the field-element length of the `S` polynomial
/// routed INTO this level by the previous level's cascade (book §5.3
/// "split commitment", un-tiered `f = 1`). For the root or a non-
/// cascade recursive level it is `0`; immediately after a parent level
/// emitted a setup-claim-reduction payload AND chose to route `S`
/// recursively it carries the parent's emitted S field length.
fn derive_optimal_suffix_schedule<Cfg: PlannerConfig>(
    memo: &mut ScheduleMemo,
    max_num_vars: usize,
    level: usize,
    current_w_len: usize,
    current_lb: u32,
    s_field_len_in: usize,
    depth: usize,
) -> PlannedSuffix {
    let key = (level, current_w_len, current_lb, s_field_len_in);
    if depth <= MAX_RECURSION_DEPTH {
        if let Some(cached) = memo.get(&key) {
            return cached.clone();
        }
    }

    // Per-level cascade gate: configs that prescribe a tiered shrink at
    // this level (book §5.8 line 1170 `f_{L0}=8`, `f_{L1}=4`) make
    // routing mandatory here. After Drift 3 γ-aggregation the runtime
    // cost model is honest per claim_count = 1 chunks group, but the
    // verifier-side cleartext MLE discharge at a direct-terminating
    // L+1 is not yet priced into the planner's objective. Until that
    // wire-cost gap closes, the cascade still requires the explicit
    // tier policy to materialise — keep the gate. The Drift 4 audit
    // closure is tracked in specs/section5_protocol_drift_audit.md.
    let level_tier =
        TieredSetupParams::new(Cfg::planner_setup_shrink_factor_at_level(level).max(1))
            .unwrap_or_else(|_| TieredSetupParams::un_tiered());
    let route_choices: &[bool] = if level_tier.is_tiered() {
        &[true]
    } else {
        &[false, true]
    };

    // Baseline: send the witness directly without folding. The cascade
    // chain terminates here; any pending `S` at the direct step is
    // discharged via the cleartext mle check in
    // `verify_setup_claim_reduction`. Levels that prescribe a tiered
    // shrink forbid the direct shortcut so the cascade can materialise.
    let fb = Cfg::planner_field_bits();
    let direct_bytes = direct_witness_bytes(
        fb,
        &DirectWitnessShape::PackedDigits((current_w_len, current_lb)),
    );
    let direct_objective_cost =
        direct_bytes.saturating_add(cleartext_discharge_objective_cost::<Cfg>(s_field_len_in));
    let mut best = if level_tier.is_tiered() {
        PlannedSuffix {
            objective_cost: usize::MAX,
            proof_bytes: usize::MAX,
            steps: Vec::new(),
        }
    } else {
        PlannedSuffix {
            objective_cost: direct_objective_cost,
            proof_bytes: direct_bytes,
            steps: vec![to_direct_step(current_w_len, current_lb)],
        }
    };

    // Try each feasible basis for one more fold level.
    if depth <= MAX_RECURSION_DEPTH {
        for lb in basis_range::<Cfg>(max_num_vars, level, current_w_len) {
            if lb < current_lb {
                continue;
            }
            for &routes_setup_recursively in route_choices {
                let Some(candidate) = derive_candidate_level_params::<Cfg>(
                    max_num_vars,
                    level,
                    current_w_len,
                    lb,
                    s_field_len_in,
                    routes_setup_recursively,
                ) else {
                    continue;
                };
                if routes_setup_recursively && candidate.s_field_len_emitted == 0 {
                    continue;
                }

                let suffix = derive_optimal_suffix_schedule::<Cfg>(
                    memo,
                    max_num_vars,
                    level + 1,
                    candidate.next_w_len,
                    lb,
                    candidate.s_field_len_emitted,
                    depth + 1,
                );
                // The suffix DP returns empty steps when a force-routed
                // sub-level had no viable fold candidate; in that case
                // the current candidate has no valid continuation.
                let Some(first_step) = suffix.steps.first() else {
                    continue;
                };
                // When this level claims to route `S` recursively,
                // the next step must actually be a fold (cleartext
                // mle handles the terminal-direct case instead).
                if routes_setup_recursively && !matches!(first_step, Step::Fold(_)) {
                    continue;
                }

                let Ok(next_level_params) = successor_level_params_from_schedule::<Cfg>(
                    max_num_vars,
                    level + 1,
                    candidate.next_w_len,
                    &suffix.steps,
                ) else {
                    continue;
                };
                // Pass the candidate's joint-open `num_eval_rows` (1 for
                // singleton, 2 for un-tiered cascade, 3 for tiered) so
                // `y_bytes` reflects the actual on-wire y-ring count.
                let level_proof_size = compute_level_proof_size::<Cfg>(
                    &candidate,
                    &next_level_params,
                    candidate.proof_num_eval_rows,
                );
                let Ok(setup_storage_field_len) =
                    planned_verifier_setup_storage_field_len_for_setup(
                        &next_level_params,
                        candidate.setup_field_len,
                        candidate.tier_setup_params,
                    )
                else {
                    continue;
                };

                let proof_bytes = level_proof_size.saturating_add(suffix.proof_bytes);
                let objective_cost = level_proof_size
                    .saturating_add(setup_storage_objective_cost::<Cfg>(setup_storage_field_len))
                    .saturating_add(stage1_prover_penalty::<Cfg>(
                        &candidate.lp,
                        candidate.next_w_len,
                    ))
                    .saturating_add(suffix.objective_cost);
                if objective_cost < best.objective_cost {
                    let mut steps = Vec::with_capacity(1 + suffix.steps.len());
                    steps.push(to_fold_step(
                        &candidate,
                        current_w_len,
                        level_proof_size,
                        Cfg::planner_field_bits(),
                    ));
                    steps.extend(suffix.steps);
                    best = PlannedSuffix {
                        objective_cost,
                        proof_bytes,
                        steps,
                    };
                }
            }
        }

        memo.insert(key, best.clone());
    }

    best
}

// -----------------------------------------------------------------------
// Witness shape
// -----------------------------------------------------------------------

/// Root-level witness ring-element count parameterized by aggregate shape.
///
/// Implements the single shape-agnostic formula
///
/// ```text
///   W(lp; K, G, P) = K · 2^r · δ_open                       // |ŵ|
///                  + K · 2^r · n_A · δ_open                 // |t̂|
///                  + P · 2^m · δ_commit · δ_fold            // |z_pre|
///                  + (n_D + n_B·G + P + 1 + n_A) · δ_R(b)   // |r|
/// ```
///
/// Mirrors `w_ring_element_count_with_counts` in `ring_switch.rs` but uses
/// field-erased helpers already available to the planner. The singleton
/// case is just `(K, G, P) = (1, 1, 1)` of this formula — there is no
/// branching on "batched vs non-batched".
fn root_w_ring_element_count<Cfg: PlannerConfig>(lp: &LevelParams, shape: &WitnessShape) -> usize {
    let fb = Cfg::planner_field_bits();
    let r_decomp = compute_num_digits_full_field(fb, lp.log_basis);

    let w_hat = shape.num_claims * lp.num_blocks * lp.num_digits_open;
    let t_hat = shape.num_claims * lp.num_blocks * lp.a_key.row_len() * lp.num_digits_open;
    let z_pre = shape.num_points * lp.inner_width() * lp.num_digits_fold;
    // One public y-row per distinct opening point.
    let r_rows = lp.m_row_count(shape.num_commitment_groups, shape.num_points);
    let r = r_rows * r_decomp;

    w_hat + t_hat + z_pre + r
}

// -----------------------------------------------------------------------
// Shape-agnostic root candidate + entry point
// -----------------------------------------------------------------------

/// Derive the optimal root candidate at `log_basis` for any witness shape.
///
/// Runs the full `(m, r)` block-split search using the shape-agnostic
/// witness sizing formula `root_w_ring_element_count`. Singleton openings
/// are `WitnessShape::singleton()`.
fn derive_root_candidate<Cfg: PlannerConfig>(
    max_num_vars: usize,
    root_w_len: usize,
    log_basis: u32,
    shape: &WitnessShape,
) -> Option<CandidateLevelParams> {
    let inputs = AkitaScheduleInputs {
        max_num_vars,
        level: 0,
        current_w_len: root_w_len,
    };

    let root_lp = Cfg::planner_root_level_layout_with_log_basis(inputs, log_basis).ok()?;
    let fb = Cfg::planner_field_bits();

    let alpha = Cfg::PLANNER_D.trailing_zeros() as usize;
    let reduced_vars = max_num_vars.checked_sub(alpha)?;
    if reduced_vars < 1 {
        return None;
    }

    let mut best: Option<CandidateLevelParams> = None;

    // Try every feasible (m, r) split with m + r = reduced_vars.
    //
    // We allow r_vars = 0 as a candidate: this corresponds to a single
    // block (no row-direction folding). It is the natural fallback for
    // very small witnesses where `optimal_m_r_split` would otherwise pick
    // r = 0 in the singleton path (`reduced_vars <= 2`). For larger
    // witnesses the loop also covers all r in [1, reduced_vars - 1].
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
            root_lp.a_key.row_len(),
            inner_width,
            root_lp.a_key.collision_inf(),
            d,
        ) else {
            continue;
        };
        let Ok(b_key) = AjtaiKeyParams::try_new(
            root_lp.b_key.row_len(),
            outer_width,
            root_lp.b_key.collision_inf(),
            d,
        ) else {
            continue;
        };
        let Ok(d_key) = AjtaiKeyParams::try_new(
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
            stage1_challenge_shape: root_lp.stage1_challenge_shape,
            use_setup_claim_reduction: root_lp.use_setup_claim_reduction,
            num_digits_commit: root_lp.num_digits_commit,
            num_digits_open: root_lp.num_digits_open,
            num_digits_fold: per_poly_fold,
            groups: None,
        };

        let Ok((level_lp, proof_lp)) = derive_batched_root_level_derivation::<Cfg>(
            max_num_vars,
            &candidate_lp,
            shape.num_claims,
        ) else {
            continue;
        };
        let w_ring = root_w_ring_element_count::<Cfg>(&level_lp, shape);
        let natural_next_w_len = w_ring * level_lp.ring_dimension;

        if natural_next_w_len * (log_basis as usize) >= root_w_len * (fb as usize) {
            continue;
        }

        if best
            .as_ref()
            .is_none_or(|b| natural_next_w_len < b.next_w_len)
        {
            // Root candidate before cascade routing is selected: this
            // helper is consumed by `derive_root_candidate` callers
            // that then re-derive the routed shape via
            // `find_optimal_schedule_with_max`'s root cascade block,
            // which constructs a fresh `CandidateLevelParams` with the
            // proper CR rounds and num_eval_rows. Keep this snapshot
            // single-claim until the routed candidate replaces it.
            best = Some(CandidateLevelParams {
                proof_lp,
                lp: level_lp,
                next_w_len: natural_next_w_len,
                w_ring,
                s_field_len_emitted: 0,
                tier_setup_params: TieredSetupParams::un_tiered(),
                proof_num_eval_rows: shape.num_points,
                setup_claim_reduction_rounds: None,
                setup_field_len: 0,
            });
        }
    }

    best
}

/// Consult the offline schedule tables for a pre-computed answer.
///
/// Returns `Ok(Some(schedule))` when the config ships an offline entry
/// whose [`AkitaScheduleLookupKey`] exactly matches the requested
/// `(max_num_vars=num_vars, num_vars, layout_num_claims, batch)` case.
/// A [`WitnessShape`] is invalid as a [`AkitaRootBatchSummary`] when
/// `G > K` or `P > K`, in which case no offline entry can exist and we
/// return `Ok(None)` so the caller falls through to the DP.
fn offline_schedule_for_shape<Cfg: PlannerConfig>(
    max_num_vars: usize,
    num_vars: usize,
    shape: WitnessShape,
) -> Result<Option<Schedule>, AkitaError> {
    let batch = match AkitaRootBatchSummary::new(
        shape.num_claims,
        shape.num_commitment_groups,
        shape.num_points,
    ) {
        Ok(batch) => batch,
        Err(_) => return Ok(None),
    };
    let key = AkitaScheduleLookupKey::with_batch(max_num_vars, num_vars, shape.num_claims, batch);
    Ok(Cfg::planner_schedule_plan(key)?
        .map(|plan| schedule_from_plan(&plan, Cfg::planner_field_bits())))
}

/// Find the optimal schedule for any root opening shape.
///
/// The planner is shape-agnostic: it takes the aggregate witness shape
/// `(K, G, P)` (or, equivalently, per-group point counts) and always
/// returns the same optimum. Singleton openings are
/// `WitnessShape::singleton()`; there is no separate code path for
/// batched vs non-batched.
///
/// **Offline fast path.** Each `(Cfg, num_vars, shape)` that ships with
/// the crate has a pre-computed entry in `Cfg::schedule_plan` (the
/// generated schedule tables in `akita-types`). Every such entry is keyed
/// on the full [`WitnessShape`] — i.e. each batching case is a distinct
/// row — so this function just returns the stored answer in O(1). Only
/// shapes without an offline entry fall back to the DP search.
///
/// # Errors
///
/// Returns an error if any of `K`, `G`, `P` is zero, if the witness
/// length overflows, or if the config's offline-table lookup fails.
pub fn find_optimal_schedule<Cfg: PlannerConfig>(
    num_vars: usize,
    shape: WitnessShape,
) -> Result<Schedule, AkitaError> {
    find_optimal_schedule_with_max::<Cfg>(num_vars, num_vars, shape)
}

/// Find the optimal schedule for an opening with distinct setup capacity and
/// actual witness size.
///
/// `max_num_vars` is the setup/config capacity used for SIS-secure matrix
/// sizing; `num_vars` is the opened polynomial size and therefore determines
/// the root witness length.
///
/// # Errors
///
/// Returns an error if any of `K`, `G`, `P` is zero, if the witness
/// length overflows, or if the config's offline-table lookup fails.
pub fn find_optimal_schedule_with_max<Cfg: PlannerConfig>(
    max_num_vars: usize,
    num_vars: usize,
    shape: WitnessShape,
) -> Result<Schedule, AkitaError> {
    if shape.num_claims == 0 || shape.num_commitment_groups == 0 || shape.num_points == 0 {
        return Err(AkitaError::InvalidSetup(
            "witness shape dimensions must be at least 1".into(),
        ));
    }

    if let Some(schedule) = offline_schedule_for_shape::<Cfg>(max_num_vars, num_vars, shape)? {
        tracing::debug!(
            max_num_vars,
            num_vars,
            num_claims = shape.num_claims,
            num_commitment_groups = shape.num_commitment_groups,
            num_points = shape.num_points,
            total_bytes = schedule.total_bytes,
            "schedule planner: served from offline schedule tables"
        );
        return Ok(schedule);
    }

    let root_w_len = 1usize
        .checked_shl(num_vars as u32)
        .ok_or_else(|| AkitaError::InvalidSetup("witness too large".into()))?;

    let fb = Cfg::planner_field_bits();
    let direct_bytes = direct_witness_bytes(fb, &DirectWitnessShape::FieldElements(root_w_len));
    let tier_shrink = Cfg::planner_setup_shrink_factor_at_level(0).max(1);
    // Tiered configs (book §5.4) cannot use the root-direct shortcut:
    // they require a routed fold schedule so the per-chunk + meta
    // material is bound recursively. The Drift 4 audit closure to
    // retire this gate is tracked in
    // `specs/section5_protocol_drift_audit.md`; Drift 3 alone is not
    // sufficient because the planner's objective does not yet price
    // the verifier-side cleartext MLE discharge at a direct-terminating
    // L+1 symmetrically with the root's CR sumcheck discharge.
    let mut best = if tier_shrink > 1 {
        PlannedSuffix {
            objective_cost: usize::MAX,
            proof_bytes: usize::MAX,
            steps: Vec::new(),
        }
    } else {
        PlannedSuffix {
            objective_cost: direct_bytes,
            proof_bytes: direct_bytes,
            steps: vec![to_direct_step(root_w_len, fb)],
        }
    };
    let mut memo = ScheduleMemo::new();

    for root_lb in basis_range::<Cfg>(max_num_vars, 0, root_w_len) {
        let Some(candidate) =
            derive_root_candidate::<Cfg>(max_num_vars, root_w_len, root_lb, &shape)
        else {
            continue;
        };

        // Phase D-full cascade: root may route `S` recursively, in
        // which case the suffix entry sees `s_field_len_in =
        // candidate.s_field_len_emitted`. Under un-tiered (`f = 1`,
        // book §5.3 lines 627-660) the suffix discharges via a
        // standard joint W+S open at L+1; under tiered (`f > 1`,
        // book §5.4) the suffix's first level expands the S group
        // into per-chunk + meta-tier rows via Slice G's
        // `tiered_setup_group_lp`. The tier shape itself is carried
        // alongside `s_field_len_emitted` on the FoldStep.
        let root_tier = TieredSetupParams::new(Cfg::planner_setup_shrink_factor_at_level(0).max(1))
            .unwrap_or_else(|_| TieredSetupParams::un_tiered());
        // Tiered configs (book §5.4) require routed root schedules so
        // the per-chunk + meta tiered material is bound recursively;
        // see Drift 4 audit closure in
        // `specs/section5_protocol_drift_audit.md`.
        let route_choices: &[bool] = if root_tier.is_tiered() {
            &[true]
        } else {
            &[false, true]
        };
        for &routes_setup_recursively in route_choices {
            // Root's M-table never carries an incoming `S` group: any
            // tier shape applies at the next level when chunks expand.
            // Pass `un_tiered()` for the incoming tier and use the
            // root's `(num_commitment_groups, num_points)` shape.
            let root_num_eval_rows = shape.num_points;
            let root_num_commitment_groups = shape.num_commitment_groups;
            let root_setup_field_len = if candidate.lp.use_setup_claim_reduction {
                planned_setup_field_len(
                    &candidate.lp,
                    None,
                    TieredSetupParams::un_tiered(),
                    root_num_eval_rows,
                    root_num_commitment_groups,
                )
            } else {
                0
            };
            let root_s_field_len_emitted = if routes_setup_recursively {
                root_setup_field_len
            } else {
                0
            };
            if routes_setup_recursively && root_s_field_len_emitted == 0 {
                continue;
            }

            let suffix = derive_optimal_suffix_schedule::<Cfg>(
                &mut memo,
                max_num_vars,
                1,
                candidate.next_w_len,
                root_lb,
                root_s_field_len_emitted,
                0,
            );
            let Some(first_step) = suffix.steps.first() else {
                continue;
            };
            if routes_setup_recursively && !matches!(first_step, Step::Fold(_)) {
                continue;
            }

            let Ok(next_level_params) = successor_level_params_from_schedule::<Cfg>(
                max_num_vars,
                1,
                candidate.next_w_len,
                &suffix.steps,
            ) else {
                continue;
            };
            // Root-level proofs carry one public y-ring per distinct
            // opening point AND the CR payload for CR-on configs (book
            // §5.3 line 658, §5.4 line 752). The CR sumcheck rounds
            // are computed from the root's M-table shape with no
            // incoming `S` (root never has incoming cascade) and the
            // batched root's `(num_eval_rows, num_commitment_groups)`
            // taken straight from the input shape.
            let root_cr_rounds = if candidate.lp.use_setup_claim_reduction {
                Some(planned_setup_claim_reduction_rounds(
                    &candidate.lp,
                    None,
                    TieredSetupParams::un_tiered(),
                    shape.num_points,
                    shape.num_commitment_groups,
                ))
            } else {
                None
            };
            let root_storage_tier = if routes_setup_recursively {
                root_tier
            } else {
                TieredSetupParams::un_tiered()
            };
            let Ok(root_setup_storage_field_len) =
                planned_verifier_setup_storage_field_len_for_setup(
                    &next_level_params,
                    root_setup_field_len,
                    root_storage_tier,
                )
            else {
                continue;
            };
            let root_cleartext_discharge_field_len =
                if candidate.lp.use_setup_claim_reduction && !routes_setup_recursively {
                    root_setup_field_len
                } else {
                    0
                };
            let root_proof_size = level_proof_bytes(
                fb,
                &candidate.proof_lp,
                &candidate.lp,
                &next_level_params,
                candidate.next_w_len,
                shape.num_points,
                root_cr_rounds,
            );

            let proof_bytes = root_proof_size.saturating_add(suffix.proof_bytes);
            let objective_cost = root_proof_size
                .saturating_add(setup_storage_objective_cost::<Cfg>(
                    root_setup_storage_field_len,
                ))
                .saturating_add(cleartext_discharge_objective_cost::<Cfg>(
                    root_cleartext_discharge_field_len,
                ))
                .saturating_add(stage1_prover_penalty::<Cfg>(
                    &candidate.lp,
                    candidate.next_w_len,
                ))
                .saturating_add(suffix.objective_cost);
            if objective_cost < best.objective_cost {
                let root_tier_for_step = if root_s_field_len_emitted > 0 {
                    root_tier
                } else {
                    TieredSetupParams::un_tiered()
                };
                let root_candidate = CandidateLevelParams {
                    proof_lp: candidate.proof_lp.clone(),
                    lp: candidate.lp.clone(),
                    next_w_len: candidate.next_w_len,
                    w_ring: candidate.w_ring,
                    s_field_len_emitted: root_s_field_len_emitted,
                    tier_setup_params: root_tier_for_step,
                    proof_num_eval_rows: shape.num_points,
                    setup_claim_reduction_rounds: root_cr_rounds,
                    setup_field_len: root_setup_field_len,
                };
                let mut steps = Vec::with_capacity(1 + suffix.steps.len());
                steps.push(to_fold_step(
                    &root_candidate,
                    root_w_len,
                    root_proof_size,
                    Cfg::planner_field_bits(),
                ));
                steps.extend(suffix.steps);
                best = PlannedSuffix {
                    objective_cost,
                    proof_bytes,
                    steps,
                };
            }
        }
    }

    if best.steps.is_empty() {
        // Tiered configs that could not find any routed fold schedule
        // (the only feasible shape under `f > 1` per book §5.4) fail
        // loudly here. Common causes: NV too small for the cascade
        // sizing, SIS-floor rejection at chunk_lp/meta_lp shape, or
        // tier shape incompatible with the level's `(m, r)` axes.
        return Err(AkitaError::InvalidSetup(format!(
            "tiered claim-reduction planner could not find a routed fold schedule \
             for max_num_vars={max_num_vars}, num_vars={num_vars}, shape={shape:?}, \
             tier_shrink={tier_shrink}; try a smaller `f`, larger `num_vars`, or \
             a config that supports the required SIS rank"
        )));
    }
    let num_folds = best
        .steps
        .iter()
        .filter(|s| matches!(s, Step::Fold(_)))
        .count();
    if tier_shrink > 1
        && !best
            .steps
            .iter()
            .any(|s| matches!(s, Step::Fold(f) if f.tier_setup_params.is_tiered() && f.s_field_len_emitted > 0))
    {
        return Err(AkitaError::InvalidSetup(format!(
            "tiered claim-reduction planner produced a schedule without a routed \
             tiered fold step for max_num_vars={max_num_vars}, num_vars={num_vars}, \
             shape={shape:?}, tier_shrink={tier_shrink}: the runtime requires at \
             least one fold step with tier_setup_params.is_tiered() && \
             s_field_len_emitted > 0 to bind the per-chunk + meta material"
        )));
    }
    tracing::info!(
        max_num_vars,
        num_vars,
        num_claims = shape.num_claims,
        num_commitment_groups = shape.num_commitment_groups,
        num_points = shape.num_points,
        total_bytes = best.proof_bytes,
        objective_cost = best.objective_cost,
        fold_levels = num_folds,
        "schedule planner: computed from scratch (no offline entry)"
    );

    Ok(Schedule {
        steps: best.steps,
        total_bytes: best.proof_bytes,
    })
}
