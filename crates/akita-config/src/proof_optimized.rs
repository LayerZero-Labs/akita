//! Concrete proof-optimized commitment configs for the default fp128 protocol.
//!
//! Each config is a plain unit struct that wires its required
//! [`CommitmentConfig`] hooks to the policy-agnostic SIS primitives in
//! the crate-internal `config::sis_policy` module and the
//! generated schedule tables in `akita-types`. A preset only
//! declares its `(D, LOG_COMMIT_BOUND)` decomposition, its sparse stage-1
//! family, the generated schedule table that backs it, and (when applicable)
//! the audited root-rank floor.

use super::{AjtaiRole, CommitmentConfig, CommitmentEnvelope, DecompositionParams};
use crate::schedule_policy::{fallback_batched_root_split, generated_schedule_plan_from_table};
use crate::sis_policy::{
    derived_root_commitment_layout_from_params, sis_derived_recursive_params,
    sis_derived_root_params_for_layout,
};
use akita_challenges::SparseChallengeConfig;
use akita_field::AkitaError;
use akita_field::{
    Ext2, Prime128OffsetA7F7, Prime16Offset99, Prime32Offset99, Prime64Offset59, RingSubfieldFp4,
    RingSubfieldFp8,
};
use akita_types::generated::table_entry_envelope_up_to_num_vars;
use akita_types::ClaimIncidenceSummary;
#[cfg(feature = "planner")]
use akita_types::Step;
use akita_types::{
    exact_planned_level_execution, planned_log_basis_at_level_from_schedule,
    planned_schedule_key_from_schedule, AkitaPlannedStep, AkitaScheduleInputs,
    AkitaScheduleLookupKey, AkitaSchedulePlan, LevelParams,
};

// ---------------------------------------------------------------------------
// fp128 family policy
// ---------------------------------------------------------------------------

/// Inclusive minimum of the proof-optimized log-basis search range.
const PROOF_OPTIMIZED_LOG_BASIS_MIN: u32 = 2;
/// Inclusive maximum of the proof-optimized log-basis search range.
const PROOF_OPTIMIZED_LOG_BASIS_MAX: u32 = 6;

/// Decomposition parameters used by every fp128 preset, keyed by
/// `LOG_COMMIT_BOUND`.
pub(crate) fn fp128_decomposition(log_commit_bound: u32, log_basis: u32) -> DecompositionParams {
    DecompositionParams {
        log_basis,
        log_commit_bound,
        log_open_bound: if log_commit_bound < 128 {
            Some(128)
        } else {
            None
        },
    }
}

/// Sparse stage-1 challenge family for a given fp128 ring degree.
pub(crate) fn fp128_stage1_challenge_config(d: usize) -> SparseChallengeConfig {
    match d {
        32 => SparseChallengeConfig::BoundedL1Norm,
        64 => SparseChallengeConfig::ExactShell {
            count_mag1: 30,
            count_mag2: 12,
        },
        128 => SparseChallengeConfig::Uniform {
            weight: 31,
            nonzero_coeffs: vec![-1, 1],
        },
        _ => panic!("unsupported fp128 ring dim {d}"),
    }
}

/// Audited root-rank policy used by every fp128 preset.
///
/// Returns `1`, escalating to `2` once `max_num_vars` crosses the threshold
/// for the audited `(D, log_commit_bound, role)` cell.
pub(crate) fn fp128_audited_root_rank<Cfg: CommitmentConfig>(
    role: AjtaiRole,
    max_num_vars: usize,
) -> usize {
    let log_commit_bound = Cfg::decomposition().log_commit_bound;
    let threshold: Option<usize> = match (Cfg::D, log_commit_bound, role) {
        // `D=128` full-field A escalates to 2 from `max_num_vars=59` onward.
        (128, lcb, AjtaiRole::Inner) if lcb != 1 => Some(59),
        // `D=128` outer (B/D) escalates from `max_num_vars=54` onward.
        (128, _, AjtaiRole::Outer) => Some(54),
        // `D=64` onehot outer (B/D) escalates from `max_num_vars=38` onward.
        (64, 1, AjtaiRole::Outer) => Some(38),
        _ => None,
    };
    1 + usize::from(threshold.is_some_and(|t| max_num_vars >= t))
}

// ---------------------------------------------------------------------------
// Trait-shaped wrappers consumed by the macro below.
//
// Each wrapper implements one required `CommitmentConfig` method by routing
// through the planned schedule table when available and falling back to the
// SIS primitives in `config::sis_policy` otherwise.
// ---------------------------------------------------------------------------

/// Inclusive `(min, max)` log-basis search range used by every fp128 preset.
pub(crate) fn proof_optimized_log_basis_search_range() -> (u32, u32) {
    (PROOF_OPTIMIZED_LOG_BASIS_MIN, PROOF_OPTIMIZED_LOG_BASIS_MAX)
}

/// Proof-optimized `schedule_plan` impl.
pub(crate) fn proof_optimized_schedule_plan<Cfg>(
    key: AkitaScheduleLookupKey,
) -> Result<Option<AkitaSchedulePlan>, AkitaError>
where
    Cfg: CommitmentConfig,
{
    let Some(table) = Cfg::schedule_table() else {
        return Ok(None);
    };
    generated_schedule_plan_from_table::<Cfg>(key, table)
}

/// Proof-optimized `schedule_key` impl: derive a stable identifier from the
/// planned schedule (or from the lookup key when no entry exists).
pub(crate) fn proof_optimized_schedule_key<Cfg: CommitmentConfig>(
    key: AkitaScheduleLookupKey,
) -> String {
    match proof_optimized_schedule_plan::<Cfg>(key) {
        Ok(Some(plan)) => planned_schedule_key_from_schedule(key, &plan),
        _ => format!(
            "generated-miss/d{}/num{}/g{}t{}w{}z{}",
            Cfg::D,
            key.num_vars,
            key.num_points,
            key.num_t_vectors,
            key.num_w_vectors,
            key.num_z_vectors,
        ),
    }
}

/// Proof-optimized `log_basis_at_level` impl: read from the planned schedule
/// when available; otherwise fall back to the root decomposition's basis.
pub(crate) fn proof_optimized_log_basis_at_level<Cfg: CommitmentConfig>(
    inputs: AkitaScheduleInputs,
) -> u32 {
    let key = AkitaScheduleLookupKey::singleton(inputs.num_vars);
    match proof_optimized_schedule_plan::<Cfg>(key) {
        Ok(Some(plan)) => planned_log_basis_at_level_from_schedule(&plan, inputs)
            .expect("generated proof-optimized schedule must be derivable from public inputs"),
        _ => Cfg::decomposition().log_basis,
    }
}

/// Proof-optimized `level_params_with_log_basis` impl: prefer the exact
/// planned level when the public inputs match; otherwise derive SIS-secure
/// recursive params (or fall back to the envelope for level 0).
pub(crate) fn proof_optimized_level_params_with_log_basis<Cfg: CommitmentConfig>(
    inputs: AkitaScheduleInputs,
    log_basis: u32,
) -> LevelParams {
    let singleton_key = AkitaScheduleLookupKey::singleton(inputs.num_vars);
    if let Ok(Some(plan)) = proof_optimized_schedule_plan::<Cfg>(singleton_key) {
        if let Ok(Some(planned_level)) =
            exact_planned_level_execution(&plan, inputs, log_basis, Cfg::stage1_challenge_config)
        {
            return planned_level.level.lp.clone();
        }
    }
    let envelope = Cfg::envelope(inputs.num_vars);
    let d = Cfg::D;
    let stage1_config = Cfg::stage1_challenge_config(d);

    if inputs.level > 0 {
        if let Some(params) = sis_derived_recursive_params::<Cfg>(
            d,
            log_basis,
            inputs.current_w_len,
            &stage1_config,
            &envelope,
        ) {
            if let Ok(lp) = akita_types::recursive_level_layout_from_params(
                &params,
                inputs.current_w_len,
                Cfg::decomposition(),
            ) {
                return lp;
            }
            return params;
        }
    }

    LevelParams::params_only(
        Cfg::sis_modulus_family(),
        d,
        log_basis,
        envelope.max_n_a,
        envelope.max_n_b,
        envelope.max_n_d,
        stage1_config,
    )
}

/// Proof-optimized `root_level_params_for_layout_with_log_basis` impl.
pub(crate) fn proof_optimized_root_level_params_for_layout_with_log_basis<Cfg: CommitmentConfig>(
    inputs: AkitaScheduleInputs,
    lp: &LevelParams,
) -> Result<LevelParams, AkitaError> {
    let params = sis_derived_root_params_for_layout::<Cfg>(inputs, lp)?;
    Ok(params.with_layout(lp))
}

// ---------------------------------------------------------------------------
// Tiered (`split_factor > 1`) helpers
//
// These mirror the production legacy helpers but post-process the root LP
// to inject the tier-1 / F metadata required by
// `specs/tiered_commit.md` §3. They are exposed to the
// `impl_fp128_tier3_preset!` macro below so a tier-3 preset can share
// almost all of the legacy production scaffolding (SIS-rank
// convergence, schedule plan materialisation, envelope sizing) and
// differ only in the root-LP shape.
// ---------------------------------------------------------------------------

/// Compute `num_digits_outer` so the balanced gadget of basis
/// `b = 2^outer_log_basis` covers the full centered range `[-q/2, q/2)`
/// for a `field_bits`-bit modulus.
///
/// Balanced range for basis `b` and depth `δ`:
/// `max ≈ ((b/2 − 1) / (b − 1)) · b^δ`. We need `max ≥ q/2 = 2^{field_bits−1}`,
/// i.e. `b^δ ≥ ((b − 1)/(b/2 − 1)) · 2^{field_bits−1}`. Setting `c =
/// (b − 1)/(b/2 − 1) ≤ 2`, this gives `δ · outer_log_basis ≥
/// field_bits − 1 + log2(c) ≤ field_bits`. The closed form
/// `δ = ⌈(field_bits + 2) / outer_log_basis⌉` over-provisions by at
/// most one digit (safety margin worth ≪ 1 % of witness bytes) and
/// matches the bench's manually-tuned `(lb=2, δ=65)` choice for Q128.
pub(crate) fn proof_optimized_tier3_num_digits_outer(
    field_bits: u32,
    outer_log_basis: u32,
) -> usize {
    let numerator = (field_bits as usize) + 2;
    numerator.div_ceil(outer_log_basis as usize)
}

/// Layer tier-3 metadata (`split_factor`, `outer_log_basis`,
/// `num_digits_outer`, `f_key`) onto a legacy root `LevelParams`, and
/// shrink `b_key.col_len` from the full outer width to
/// `chunk_width = outer_width / split_factor`.
///
/// The legacy LP is taken as-is for `(n_a, n_d, ring_dimension,
/// log_basis, m_vars, r_vars, block_len, num_blocks,
/// num_digits_{commit,open,fold})`. The tiered fields are derived
/// from the modulus family + SIS floors via
/// [`akita_types::layout::sis_derivation::tiered_b_prime_rank`] and
/// [`akita_types::layout::sis_derivation::tiered_f_rank`].
///
/// # Errors
///
/// Returns an error if the outer width is not divisible by
/// `split_factor`, or if the SIS floor tables don't cover the
/// requested `(family, D, collision, width)` tuple.
pub(crate) fn proof_optimized_tier3_apply_to_root_lp<Cfg: CommitmentConfig>(
    legacy_root: &LevelParams,
    split_factor: usize,
) -> Result<LevelParams, AkitaError> {
    use akita_types::layout::sis_derivation::{
        balanced_digit_delta_bound, tiered_b_prime_rank, tiered_f_rank,
    };
    let family = legacy_root.b_key.sis_family();
    let d = legacy_root.ring_dimension;
    let outer_log_basis = legacy_root.log_basis;
    let field_bits = Cfg::decomposition().field_bits();
    let num_digits_outer = proof_optimized_tier3_num_digits_outer(field_bits, outer_log_basis);
    let full_outer_width = legacy_root.full_outer_width();
    if full_outer_width % split_factor != 0 {
        return Err(AkitaError::InvalidSetup(format!(
            "tier3: outer_width {full_outer_width} not divisible by split_factor {split_factor} \
             (legacy LP shape n_a={}, num_blocks={}, depth_open={}); pick a (n_a, r_vars, \
             depth_open) tuple whose product is divisible by {split_factor}",
            legacy_root.a_key.row_len(),
            legacy_root.num_blocks,
            legacy_root.num_digits_open,
        )));
    }
    let chunk_width = full_outer_width / split_factor;
    let t_inf_bound = legacy_root.b_key.collision_inf();
    let n_b_prime = tiered_b_prime_rank(
        family,
        d as u32,
        t_inf_bound,
        full_outer_width,
        split_factor,
    )?;
    let n_f = tiered_f_rank(
        family,
        d as u32,
        outer_log_basis,
        n_b_prime,
        split_factor,
        num_digits_outer,
    )?;
    let f_width = (n_b_prime as usize)
        .checked_mul(split_factor)
        .and_then(|w| w.checked_mul(num_digits_outer))
        .ok_or_else(|| AkitaError::InvalidSetup("tier3 F width overflow".to_string()))?;
    let f_collision = balanced_digit_delta_bound(outer_log_basis);
    let tiered_b_key = akita_types::AjtaiKeyParams::new_unchecked(
        family,
        n_b_prime as usize,
        chunk_width,
        t_inf_bound,
        d,
    );
    let f_key =
        akita_types::AjtaiKeyParams::new_unchecked(family, n_f as usize, f_width, f_collision, d);
    Ok(LevelParams {
        split_factor,
        outer_log_basis,
        num_digits_outer,
        f_key,
        b_key: tiered_b_key,
        ..legacy_root.clone()
    })
}

/// Tier-3 variant of `proof_optimized_root_level_layout_with_log_basis`.
///
/// Runs the production SIS-rank convergence to derive the legacy root
/// shape, then layers tier-3 metadata on top via
/// [`proof_optimized_tier3_apply_to_root_lp`].
pub(crate) fn proof_optimized_tier3_root_level_layout_with_log_basis<Cfg: CommitmentConfig>(
    inputs: AkitaScheduleInputs,
    log_basis: u32,
    split_factor: usize,
) -> Result<LevelParams, AkitaError> {
    let legacy = proof_optimized_root_level_layout_with_log_basis::<Cfg>(inputs, log_basis)?;
    proof_optimized_tier3_apply_to_root_lp::<Cfg>(&legacy, split_factor)
}

/// Tier-3 variant of
/// `proof_optimized_root_level_params_for_layout_with_log_basis`.
pub(crate) fn proof_optimized_tier3_root_level_params_for_layout_with_log_basis<
    Cfg: CommitmentConfig,
>(
    inputs: AkitaScheduleInputs,
    lp: &LevelParams,
    split_factor: usize,
) -> Result<LevelParams, AkitaError> {
    let legacy = proof_optimized_root_level_params_for_layout_with_log_basis::<Cfg>(inputs, lp)?;
    proof_optimized_tier3_apply_to_root_lp::<Cfg>(&legacy, split_factor)
}

/// Tier-3 variant of `proof_optimized_max_setup_matrix_size`.
///
/// The base function walks every committable sub-shape
/// `(num_vars', num_polys', num_points')` with `1 ≤ num_vars' ≤ max`.
/// For a tier-3 preset the planner errors out at small `num_vars'`
/// values whose root layout cannot meet the tier-3 constraint
/// `outer_width % split_factor == 0`. We swallow those errors as
/// "unsupported shape" (mirroring the `Ok(None)` semantics that the
/// envelope walker already understands), so the envelope just covers
/// the tier-3-feasible shapes. That suffices for setup sizing because
/// the chunk-width tiered B' is strictly narrower than the legacy
/// outer width at every shape it does support, so smaller / infeasible
/// shapes cannot need a wider matrix than the supported ones.
pub(crate) fn proof_optimized_tier3_max_setup_matrix_size<Cfg: CommitmentConfig>(
    max_num_vars: usize,
    max_num_batched_polys: usize,
    max_num_points: usize,
) -> Result<(usize, usize), AkitaError> {
    if max_num_batched_polys == 0 {
        return Err(AkitaError::InvalidSetup(
            "max_num_batched_polys must be at least 1".to_string(),
        ));
    }
    if max_num_points == 0 {
        return Err(AkitaError::InvalidSetup(
            "max_num_points must be at least 1".to_string(),
        ));
    }
    if max_num_points > max_num_batched_polys {
        return Err(AkitaError::InvalidSetup(format!(
            "max_num_points ({max_num_points}) cannot exceed max_num_batched_polys ({max_num_batched_polys})"
        )));
    }

    let mut max_rows: usize = 1;
    let mut max_stride: usize = 1;
    let mut saw_supported_shape = false;
    let setup_envelope = Cfg::envelope(max_num_vars);
    for num_vars in 1..=max_num_vars {
        for num_polys in 1..=max_num_batched_polys {
            let upper_pts = num_polys.min(max_num_points);
            for num_points in 1..=upper_pts {
                let incidence =
                    ClaimIncidenceSummary::from_counts(num_vars, num_polys, num_points)?;
                // Tier-3-tolerant shape walk: planner errors are
                // treated as "not a supported shape" rather than
                // propagated.
                let shape_env =
                    match setup_matrix_envelope_for_shape::<Cfg>(&incidence, &setup_envelope) {
                        Ok(opt) => opt,
                        Err(AkitaError::InvalidSetup(_)) => None,
                        Err(err) => return Err(err),
                    };
                let Some((rows, stride)) = shape_env else {
                    continue;
                };
                saw_supported_shape = true;
                max_rows = max_rows.max(rows);
                max_stride = max_stride.max(stride);
            }
        }
    }

    if !saw_supported_shape {
        return Err(AkitaError::InvalidSetup(format!(
            "tier3 setup matrix sizing found no generated schedules for max_num_vars={max_num_vars}"
        )));
    }

    Ok((max_rows, max_stride))
}

/// Tier-3 variant of `proof_optimized_schedule_plan`.
///
/// Reads from the Cfg's generated schedule table (sized for the tier-3
/// shapes by the offline generator) and post-processes the root step
/// to layer tier-3 metadata on top — the on-disk
/// `GeneratedFoldStep` records only `(ring_d, log_basis, m_vars,
/// r_vars, n_a, n_b, n_d)`, so the per-Cfg tier-3 constants are
/// re-injected here. The `n_b` stored in the table is already `n_b'`
/// (the tier-1 B' rank) because the offline generator ran the planner
/// DP with a tiered root LP.
pub(crate) fn proof_optimized_tier3_schedule_plan<Cfg: CommitmentConfig>(
    key: AkitaScheduleLookupKey,
    split_factor: usize,
) -> Result<Option<AkitaSchedulePlan>, AkitaError> {
    use akita_types::{
        w_ring_element_count_with_vector_counts_bits,
        w_ring_element_count_with_vector_counts_for_layout_bits,
    };
    let Some(mut plan) = proof_optimized_schedule_plan::<Cfg>(key)? else {
        return Ok(None);
    };
    let field_bits = Cfg::decomposition().field_bits();
    // Layer tier-3 metadata onto the root LP. The base materialiser
    // (`schedule_plan_from_generated_entry`) computed the root's
    // `runtime_next_w_len` against the pre-tiered LP, so we re-do it
    // here against the tiered LP — which DOES include the ûhat
    // segment via the tier-aware
    // `w_ring_element_count_with_counts_for_layout`. We then walk the
    // suffix and update each `inputs.current_w_len` /
    // `next_inputs.current_w_len` using the EXACT same `_bits`
    // variants the base materialiser uses, so we don't introduce
    // sizing discrepancies at recursive levels.
    // Compute the "next step is Direct?" mask before taking the
    // mutable borrow on `plan.steps.first_mut()` (borrow checker).
    let suffix_len = plan.steps.len();
    let next_is_direct: Vec<bool> = (1..suffix_len)
        .map(|i| matches!(plan.steps.get(i + 1), Some(AkitaPlannedStep::Direct(_))))
        .collect();
    let Some(AkitaPlannedStep::Fold(root_level)) = plan.steps.first_mut() else {
        return Ok(Some(plan));
    };
    let tiered_lp = proof_optimized_tier3_apply_to_root_lp::<Cfg>(&root_level.lp, split_factor)?;
    let next_w_ring = akita_types::w_ring_element_count_with_counts_for_layout::<Cfg::Field>(
        &tiered_lp,
        key.num_points,
        key.num_t_vectors,
        key.num_w_vectors,
        key.num_z_vectors,
        akita_types::MRowLayout::Intermediate,
    )?;
    let tiered_next_w_len = next_w_ring
        .checked_mul(tiered_lp.ring_dimension)
        .ok_or_else(|| {
            AkitaError::InvalidSetup("tier3 root next witness length overflow".to_string())
        })?;
    root_level.lp = tiered_lp;
    root_level.next_inputs.current_w_len = tiered_next_w_len;

    // Suffix walk: each level after the root re-uses the previous
    // level's exit state as its entry, and computes its own exit
    // using the EXACT bits-variant the base materialiser uses for
    // adjacent-step consistency (Intermediate for non-terminal folds,
    // Terminal for folds whose successor is a Direct — mirrors the
    // terminal-fold cutover logic in
    // `schedule_plan_from_generated_entry`).
    // At each recursive level, `block_len = ceil(num_ring /
    // num_blocks)` where `num_ring = current_w_len / ring_dimension`
    // (see `LevelParams::with_decomp` — the level's `block_len` is
    // sized to actually hold the carried witness, not just `1 <<
    // m_vars`). Since tier-3 changes the root's exit `num_ring`
    // (extra ûhat segment + tier-1/F r-rows), every downstream level
    // must be re-laid-out under the new entry state. We re-derive
    // each level's LP from its `(m_vars, r_vars)` + the cumulative
    // tiered `current_w_len`, mirroring exactly what
    // `schedule_plan_from_generated_entry` does for the legacy path.
    let root_decomp = Cfg::decomposition();
    let mut prev_w_len = tiered_next_w_len;
    for (idx, step) in plan.steps.iter_mut().enumerate().skip(1) {
        match step {
            AkitaPlannedStep::Fold(level) => {
                level.inputs.current_w_len = prev_w_len;
                // Re-derive the level's layout against the tiered
                // entry state. The level's base `params`
                // (a_key/b_key/d_key/ring_d/log_basis/stage1_config)
                // are the same ones the base materialiser put in;
                // only the layout fields driven by `num_ring` change.
                let level_decomp =
                    akita_types::layout::sis_derivation::recursive_level_decomposition_from_root(
                        root_decomp,
                        level.lp.log_basis,
                    );
                let num_ring = prev_w_len / level.lp.ring_dimension;
                let m_vars = level.lp.m_vars;
                let r_vars = level.lp.r_vars;
                let relayed = akita_types::layout::sis_derivation::level_layout_from_params(
                    m_vars,
                    r_vars,
                    &level.lp,
                    level_decomp,
                    num_ring,
                )?;
                level.lp = relayed;
                let next_ring = if next_is_direct[idx - 1] {
                    w_ring_element_count_with_vector_counts_for_layout_bits::<Cfg::Field>(
                        field_bits,
                        &level.lp,
                        1,
                        1,
                        1,
                        1,
                        akita_types::MRowLayout::Terminal,
                    )?
                } else {
                    w_ring_element_count_with_vector_counts_bits::<Cfg::Field>(
                        field_bits, &level.lp, 1, 1, 1, 1,
                    )?
                };
                let next_len = next_ring
                    .checked_mul(level.lp.ring_dimension)
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup("tier3 suffix next-w overflow".to_string())
                    })?;
                level.next_inputs.current_w_len = next_len;
                prev_w_len = next_len;
            }
            AkitaPlannedStep::Direct(direct) => {
                direct.state.current_w_len = prev_w_len;
            }
        }
    }
    Ok(Some(plan))
}

/// Proof-optimized `root_level_layout_with_log_basis` impl.
pub(crate) fn proof_optimized_root_level_layout_with_log_basis<Cfg: CommitmentConfig>(
    inputs: AkitaScheduleInputs,
    log_basis: u32,
) -> Result<LevelParams, AkitaError> {
    let stage1_config = Cfg::stage1_challenge_config(Cfg::D);
    let mut candidate_n_a = 1usize;
    let rank_cap = proof_optimized_root_a_rank_cap::<Cfg>(inputs, log_basis, &stage1_config)?;
    for _ in 0..rank_cap {
        let candidate_params = LevelParams::params_only(
            Cfg::sis_modulus_family(),
            Cfg::D,
            log_basis,
            candidate_n_a,
            1,
            1,
            stage1_config.clone(),
        );
        let root_lp =
            derived_root_commitment_layout_from_params::<Cfg>(inputs, &candidate_params, false)?;
        let derived_params = sis_derived_root_params_for_layout::<Cfg>(inputs, &root_lp)?;
        if derived_params.a_key.row_len() == candidate_n_a {
            return Ok(derived_params.with_layout(&root_lp));
        }
        candidate_n_a = derived_params.a_key.row_len();
    }
    Err(AkitaError::InvalidSetup(format!(
        "failed to converge on self-consistent root A-row rank for D={} lb={log_basis}",
        Cfg::D
    )))
}

fn proof_optimized_root_a_rank_cap<Cfg: CommitmentConfig>(
    inputs: AkitaScheduleInputs,
    log_basis: u32,
    stage1_config: &SparseChallengeConfig,
) -> Result<usize, AkitaError> {
    let bd_collision = 1u32
        .checked_shl(log_basis)
        .and_then(|bound| bound.checked_sub(1))
        .ok_or_else(|| {
            AkitaError::InvalidSetup(format!(
                "root collision bound overflow for D={} lb={log_basis}",
                Cfg::D
            ))
        })?;
    let a_raw = if inputs.level == 0 && Cfg::decomposition().log_commit_bound == 1 {
        2
    } else {
        bd_collision
    };
    let a_collision_raw = a_raw
        .checked_mul(stage1_config.infinity_norm())
        .and_then(|collision| collision.checked_mul(Cfg::ring_subfield_embedding_norm_bound()))
        .ok_or_else(|| {
            AkitaError::InvalidSetup(format!(
                "root A-role collision overflow for family={:?}, D={}",
                Cfg::sis_modulus_family(),
                Cfg::D
            ))
        })?;
    let a_collision = akita_types::generated::sis_floor::ceil_supported_collision(
        Cfg::sis_modulus_family(),
        Cfg::D as u32,
        a_collision_raw,
    )
    .ok_or_else(|| {
        AkitaError::InvalidSetup(format!(
            "missing supported root A-role collision bucket for family={:?}, D={} \
             and raw collision {a_collision_raw}",
            Cfg::sis_modulus_family(),
            Cfg::D
        ))
    })?;
    akita_types::generated::sis_floor::sis_max_widths(
        Cfg::sis_modulus_family(),
        Cfg::D as u32,
        a_collision,
    )
    .map(<[u64]>::len)
    .ok_or_else(|| {
        AkitaError::InvalidSetup(format!(
            "missing root A-role SIS rank table for family={:?}, D={}, collision_inf={a_collision}",
            Cfg::sis_modulus_family(),
            Cfg::D
        ))
    })
}

/// Proof-optimized `envelope` impl: combine the audited rank floor with the
/// maximum rank reached by any planned level for `max_num_vars`.
pub(crate) fn proof_optimized_envelope<Cfg: CommitmentConfig>(
    max_num_vars: usize,
) -> CommitmentEnvelope {
    let inner_floor = Cfg::audited_root_rank(AjtaiRole::Inner, max_num_vars);
    let outer_floor = Cfg::audited_root_rank(AjtaiRole::Outer, max_num_vars);
    let mut envelope = CommitmentEnvelope {
        max_n_a: inner_floor,
        max_n_b: outer_floor,
        max_n_d: outer_floor,
    };
    if let Some(table) = Cfg::schedule_table() {
        if let Some((gen_n_a, gen_n_b, gen_n_d)) =
            table_entry_envelope_up_to_num_vars(table, max_num_vars)
        {
            envelope.max_n_a = envelope.max_n_a.max(gen_n_a);
            envelope.max_n_b = envelope.max_n_b.max(gen_n_b);
            envelope.max_n_d = envelope.max_n_d.max(gen_n_d);
        }
    }
    envelope
}

/// Size the shared setup matrix from the planned schedule.
///
/// The planner can pick non-monotone `(n_a, n_b, n_d)` ranks across
/// `num_vars` and `num_polys`, so the final envelope is the max over every
/// committable sub-shape `(num_vars', num_polys', num_points')` with
/// `1 <= num_vars' <= max_num_vars`,
/// `1 <= num_polys' <= max_num_batched_polys`, and
/// `1 <= num_points' <= num_polys'.min(max_num_points)`. Without this, a
/// runtime commit at a smaller variable count or differently shaped batch
/// can pick a schedule with strictly larger row count than the all-up
/// envelope.
pub(crate) fn proof_optimized_max_setup_matrix_size<Cfg: CommitmentConfig>(
    max_num_vars: usize,
    max_num_batched_polys: usize,
    max_num_points: usize,
) -> Result<(usize, usize), AkitaError> {
    if max_num_batched_polys == 0 {
        return Err(AkitaError::InvalidSetup(
            "max_num_batched_polys must be at least 1".to_string(),
        ));
    }
    if max_num_points == 0 {
        return Err(AkitaError::InvalidSetup(
            "max_num_points must be at least 1".to_string(),
        ));
    }
    if max_num_points > max_num_batched_polys {
        return Err(AkitaError::InvalidSetup(format!(
            "max_num_points ({max_num_points}) cannot exceed max_num_batched_polys ({max_num_batched_polys})"
        )));
    }

    let mut max_rows: usize = 1;
    let mut max_stride: usize = 1;
    let mut saw_supported_shape = false;
    let setup_envelope = Cfg::envelope(max_num_vars);
    for num_vars in 1..=max_num_vars {
        for num_polys in 1..=max_num_batched_polys {
            let upper_pts = num_polys.min(max_num_points);
            for num_points in 1..=upper_pts {
                let incidence =
                    ClaimIncidenceSummary::from_counts(num_vars, num_polys, num_points)?;
                let Some((rows, stride)) =
                    setup_matrix_envelope_for_shape::<Cfg>(&incidence, &setup_envelope)?
                else {
                    continue;
                };
                saw_supported_shape = true;
                max_rows = max_rows.max(rows);
                max_stride = max_stride.max(stride);
            }
        }
    }

    if !saw_supported_shape {
        return Err(AkitaError::InvalidSetup(format!(
            "setup matrix sizing found no generated schedules for max_num_vars={max_num_vars}"
        )));
    }

    Ok((max_rows, max_stride))
}

fn setup_matrix_envelope_for_shape<Cfg: CommitmentConfig>(
    incidence: &ClaimIncidenceSummary,
    setup_envelope: &CommitmentEnvelope,
) -> Result<Option<(usize, usize)>, AkitaError> {
    let num_polys = incidence.num_polynomials();
    let cached_key = AkitaScheduleLookupKey::new_from_incidence(incidence)?;
    #[cfg(not(feature = "planner"))]
    let _ = setup_envelope;

    let fallback = fallback_batched_root_split::<Cfg>(incidence.num_vars(), num_polys)?;

    let setup_levels: Vec<LevelParams> = if let Some(plan) = Cfg::schedule_plan(cached_key)? {
        setup_level_params_from_plan(&plan)
    } else {
        #[cfg(feature = "planner")]
        {
            let schedule = akita_planner::find_optimal_schedule::<Cfg>(cached_key)?;
            setup_level_params_from_runtime_schedule(schedule.steps, setup_envelope)
        }

        #[cfg(not(feature = "planner"))]
        {
            let _ = cached_key;
            return Ok(None);
        }
    };

    Ok(Some(matrix_envelope_for_levels::<Cfg>(
        &fallback,
        &setup_levels,
    )?))
}

fn setup_level_params_from_plan(plan: &AkitaSchedulePlan) -> Vec<LevelParams> {
    plan.steps
        .iter()
        .filter_map(|step| match step {
            AkitaPlannedStep::Fold(level) => Some(level.lp.clone()),
            AkitaPlannedStep::Direct(_) => None,
        })
        .collect()
}

#[cfg(feature = "planner")]
fn setup_level_params_from_runtime_schedule(
    steps: Vec<Step>,
    _setup_envelope: &CommitmentEnvelope,
) -> Vec<LevelParams> {
    steps
        .into_iter()
        .filter_map(|step| match step {
            Step::Fold(fold_step) => Some(fold_step.params),
            Step::Direct(_) => None,
        })
        .collect()
}

fn matrix_envelope_for_levels<Cfg>(
    fallback_root: &LevelParams,
    setup_levels: &[LevelParams],
) -> Result<(usize, usize), AkitaError>
where
    Cfg: CommitmentConfig,
{
    let mut max_rows: usize = 1;
    let mut max_stride: usize = 1;

    accumulate_matrix_envelope_for_level::<Cfg>(fallback_root, &mut max_rows, &mut max_stride)?;
    if let Some((root_level, recursive_levels)) = setup_levels.split_first() {
        accumulate_matrix_envelope_for_level::<Cfg>(root_level, &mut max_rows, &mut max_stride)?;
        for lp in recursive_levels {
            accumulate_matrix_envelope_for_level::<Cfg>(lp, &mut max_rows, &mut max_stride)?;
        }
    }
    Ok((max_rows, max_stride))
}

fn accumulate_matrix_envelope_for_level<Cfg>(
    lp: &LevelParams,
    max_rows: &mut usize,
    max_stride: &mut usize,
) -> Result<(), AkitaError>
where
    Cfg: CommitmentConfig,
{
    let _cfg_marker = core::marker::PhantomData::<Cfg>;
    let outer_width = lp.outer_width();
    #[cfg(feature = "zk")]
    let outer_width = outer_width
        .checked_add(akita_types::zk::blinding_column_count::<Cfg::Field>(
            lp.b_key.row_len(),
            lp.ring_dimension,
            lp.log_basis,
        ))
        .ok_or_else(|| AkitaError::InvalidSetup("ZK outer width overflow".to_string()))?;
    let d_matrix_width = lp.d_matrix_width();
    #[cfg(feature = "zk")]
    let d_matrix_width = d_matrix_width
        .checked_add(akita_types::zk::blinding_column_count::<Cfg::Field>(
            lp.d_key.row_len(),
            lp.ring_dimension,
            lp.log_basis,
        ))
        .ok_or_else(|| AkitaError::InvalidSetup("ZK D width overflow".to_string()))?;
    *max_rows = (*max_rows)
        .max(lp.a_key.row_len())
        .max(lp.b_key.row_len())
        .max(lp.d_key.row_len());
    *max_stride = (*max_stride)
        .max(lp.inner_width())
        .max(outer_width)
        .max(d_matrix_width);
    Ok(())
}

// ---------------------------------------------------------------------------
// Per-preset CommitmentConfig macro
// ---------------------------------------------------------------------------

/// Generate a complete [`CommitmentConfig`] impl for one fp128 preset.
///
/// Each preset only ships its `(D, LOG_COMMIT_BOUND)` decomposition and the
/// optional generated schedule table. Every other trait method is a one-line
/// delegation to the proof-optimized helpers above.
macro_rules! impl_fp128_preset {
    ($cfg:ident, $d:expr, $log_commit_bound:expr, $table:expr) => {
        impl akita_types::ScheduleProvider for $cfg {
            fn schedule_table() -> Option<akita_types::generated::GeneratedScheduleTable> {
                $table
            }

            fn schedule_key(key: akita_types::AkitaScheduleLookupKey) -> String {
                $crate::proof_optimized::proof_optimized_schedule_key::<Self>(key)
            }

            fn schedule_plan(
                key: akita_types::AkitaScheduleLookupKey,
            ) -> Result<Option<akita_types::AkitaSchedulePlan>, akita_field::AkitaError> {
                $crate::proof_optimized::proof_optimized_schedule_plan::<Self>(key)
            }
        }

        impl $crate::CommitmentConfig for $cfg {
            type Field = Field;
            type ClaimField = Field;
            type ChallengeField = Field;
            const D: usize = $d;

            fn decomposition() -> akita_types::DecompositionParams {
                $crate::proof_optimized::fp128_decomposition($log_commit_bound, 3)
            }

            fn stage1_challenge_config(d: usize) -> akita_challenges::SparseChallengeConfig {
                $crate::proof_optimized::fp128_stage1_challenge_config(d)
            }

            fn sis_modulus_family() -> akita_types::SisModulusFamily {
                akita_types::SisModulusFamily::Q128
            }

            fn audited_root_rank(
                role: akita_types::AjtaiRole,
                max_num_vars: usize,
            ) -> usize {
                $crate::proof_optimized::fp128_audited_root_rank::<Self>(
                    role,
                    max_num_vars,
                )
            }

            fn envelope(
                max_num_vars: usize,
            ) -> akita_types::CommitmentEnvelope {
                $crate::proof_optimized::proof_optimized_envelope::<Self>(
                    max_num_vars,
                )
            }

            fn max_setup_matrix_size(
                max_num_vars: usize,
                max_num_batched_polys: usize,
                max_num_points: usize,
            ) -> Result<(usize, usize), akita_field::AkitaError> {
                $crate::proof_optimized::proof_optimized_max_setup_matrix_size::<Self>(
                    max_num_vars,
                    max_num_batched_polys,
                    max_num_points,
                )
            }

            fn level_params_with_log_basis(
                inputs: akita_types::AkitaScheduleInputs,
                log_basis: u32,
            ) -> akita_types::LevelParams {
                $crate::proof_optimized::proof_optimized_level_params_with_log_basis::<Self>(
                    inputs,
                    log_basis,
                )
            }

            fn root_level_params_for_layout_with_log_basis(
                inputs: akita_types::AkitaScheduleInputs,
                lp: &akita_types::LevelParams,
            ) -> Result<akita_types::LevelParams, akita_field::AkitaError> {
                $crate::proof_optimized::
                    proof_optimized_root_level_params_for_layout_with_log_basis::<Self>(inputs, lp)
            }

            fn root_level_layout_with_log_basis(
                inputs: akita_types::AkitaScheduleInputs,
                log_basis: u32,
            ) -> Result<akita_types::LevelParams, akita_field::AkitaError> {
                $crate::proof_optimized::proof_optimized_root_level_layout_with_log_basis::<Self>(
                    inputs,
                    log_basis,
                )
            }

            fn log_basis_at_level(
                inputs: akita_types::AkitaScheduleInputs,
            ) -> u32 {
                $crate::proof_optimized::proof_optimized_log_basis_at_level::<Self>(inputs)
            }

            fn log_basis_search_range(
                _inputs: akita_types::AkitaScheduleInputs,
            ) -> (u32, u32) {
                $crate::proof_optimized::proof_optimized_log_basis_search_range()
            }
        }

        #[cfg(feature = "planner")]
        impl akita_planner::PlannerConfig for $cfg {
            type PlannerField = Field;

            const PLANNER_D: usize = $d;

            fn planner_field_bits() -> u32 {
                <Self as $crate::CommitmentConfig>::decomposition().field_bits()
            }

            fn planner_challenge_field_bits() -> u32 {
                <Self as $crate::CommitmentConfig>::decomposition().field_bits()
                    * (<Self as $crate::CommitmentConfig>::CHAL_EXT_DEGREE as u32)
            }

            fn planner_extension_opening_width() -> usize {
                <Self as $crate::CommitmentConfig>::CLAIM_EXT_DEGREE
            }

            fn planner_recursive_witness_expansion() -> usize {
                1
            }

            fn planner_recursive_public_rows() -> usize {
                1
            }

            fn planner_sis_modulus_family() -> akita_types::SisModulusFamily {
                <Self as $crate::CommitmentConfig>::sis_modulus_family()
            }

            fn planner_stage1_challenge_config(
                d: usize,
            ) -> akita_challenges::SparseChallengeConfig {
                <Self as $crate::CommitmentConfig>::stage1_challenge_config(d)
            }

            fn planner_schedule_plan(
                key: akita_types::AkitaScheduleLookupKey,
            ) -> Result<Option<akita_types::AkitaSchedulePlan>, akita_field::AkitaError> {
                <Self as akita_types::ScheduleProvider>::schedule_plan(key)
            }

            fn planner_root_level_layout_with_log_basis(
                inputs: akita_types::AkitaScheduleInputs,
                log_basis: u32,
            ) -> Result<akita_types::LevelParams, akita_field::AkitaError> {
                <Self as $crate::CommitmentConfig>::root_level_layout_with_log_basis(
                    inputs,
                    log_basis,
                )
            }

            fn planner_current_level_layout_with_log_basis(
                inputs: akita_types::AkitaScheduleInputs,
                log_basis: u32,
            ) -> Result<akita_types::LevelParams, akita_field::AkitaError> {
                $crate::current_level_layout_with_log_basis::<Self>(
                    inputs,
                    log_basis,
                )
            }

            fn planner_direct_level_params_with_log_basis(
                inputs: akita_types::AkitaScheduleInputs,
                log_basis: u32,
            ) -> Result<akita_types::LevelParams, akita_field::AkitaError> {
                $crate::schedule_policy::direct_level_params_with_log_basis::<Self>(
                    inputs,
                    log_basis,
                )
            }

            fn planner_root_level_params_for_layout_with_log_basis(
                inputs: akita_types::AkitaScheduleInputs,
                lp: &akita_types::LevelParams,
            ) -> Result<akita_types::LevelParams, akita_field::AkitaError> {
                <Self as $crate::CommitmentConfig>::root_level_params_for_layout_with_log_basis(
                    inputs,
                    lp,
                )
            }

            fn planner_log_basis_search_range(
                inputs: akita_types::AkitaScheduleInputs,
            ) -> (u32, u32) {
                <Self as $crate::CommitmentConfig>::log_basis_search_range(inputs)
            }
        }
    };
}
pub(crate) use impl_fp128_preset;

/// Tier-3 variant of `impl_fp128_preset`. Shares 99 % of its body
/// with the legacy macro; the only differences are:
///
/// * `root_level_layout_with_log_basis` /
///   `root_level_params_for_layout_with_log_basis` (and their planner
///   twins) call the `_tier3_` helpers that layer the tier-1 / F /
///   ûhat-gadget metadata on the legacy LP they produce.
/// * `schedule_plan` post-processes the materialised plan so the
///   root step's `LevelParams` carries the tiered fields even though
///   the on-disk `GeneratedFoldStep` only records the legacy fields.
/// * `audited_root_rank` / `envelope` are not extended — the tiered
///   `b_key.col_len = chunk_width = legacy_outer_width / split_factor`
///   is strictly smaller than the legacy outer width, so the legacy
///   envelope safely upper-bounds it.
///
/// `$split:expr` is the tier split factor (`3` for `D32OneHotTier3`).
macro_rules! impl_fp128_tier3_preset {
    ($cfg:ident, $d:expr, $log_commit_bound:expr, $split:expr, $table:expr) => {
        impl akita_types::ScheduleProvider for $cfg {
            fn schedule_table() -> Option<akita_types::generated::GeneratedScheduleTable> {
                $table
            }

            fn schedule_key(key: akita_types::AkitaScheduleLookupKey) -> String {
                $crate::proof_optimized::proof_optimized_schedule_key::<Self>(key)
            }

            fn schedule_plan(
                key: akita_types::AkitaScheduleLookupKey,
            ) -> Result<Option<akita_types::AkitaSchedulePlan>, akita_field::AkitaError> {
                $crate::proof_optimized::proof_optimized_tier3_schedule_plan::<Self>(key, $split)
            }
        }

        impl $crate::CommitmentConfig for $cfg {
            type Field = Field;
            type ClaimField = Field;
            type ChallengeField = Field;
            const D: usize = $d;

            fn decomposition() -> akita_types::DecompositionParams {
                $crate::proof_optimized::fp128_decomposition($log_commit_bound, 3)
            }

            fn stage1_challenge_config(d: usize) -> akita_challenges::SparseChallengeConfig {
                $crate::proof_optimized::fp128_stage1_challenge_config(d)
            }

            fn sis_modulus_family() -> akita_types::SisModulusFamily {
                akita_types::SisModulusFamily::Q128
            }

            fn audited_root_rank(
                role: akita_types::AjtaiRole,
                max_num_vars: usize,
            ) -> usize {
                $crate::proof_optimized::fp128_audited_root_rank::<Self>(role, max_num_vars)
            }

            fn envelope(max_num_vars: usize) -> akita_types::CommitmentEnvelope {
                $crate::proof_optimized::proof_optimized_envelope::<Self>(max_num_vars)
            }

            fn max_setup_matrix_size(
                max_num_vars: usize,
                max_num_batched_polys: usize,
                max_num_points: usize,
            ) -> Result<(usize, usize), akita_field::AkitaError> {
                $crate::proof_optimized::proof_optimized_tier3_max_setup_matrix_size::<Self>(
                    max_num_vars,
                    max_num_batched_polys,
                    max_num_points,
                )
            }

            fn level_params_with_log_basis(
                inputs: akita_types::AkitaScheduleInputs,
                log_basis: u32,
            ) -> akita_types::LevelParams {
                // Non-root levels are non-tiered (tiering applies at
                // the root only — `specs/tiered_commit.md` §1). For
                // the root we still emit a non-tiered base shape here
                // because the table-driven materialiser layers tier-3
                // fields on top via `schedule_plan` above. Production
                // code paths that bypass the table and ask for the
                // root LP shape directly use
                // `root_level_layout_with_log_basis`, which DOES
                // apply tiering.
                $crate::proof_optimized::proof_optimized_level_params_with_log_basis::<Self>(
                    inputs,
                    log_basis,
                )
            }

            fn root_level_params_for_layout_with_log_basis(
                inputs: akita_types::AkitaScheduleInputs,
                lp: &akita_types::LevelParams,
            ) -> Result<akita_types::LevelParams, akita_field::AkitaError> {
                $crate::proof_optimized::
                    proof_optimized_tier3_root_level_params_for_layout_with_log_basis::<Self>(
                        inputs, lp, $split,
                    )
            }

            fn root_level_layout_with_log_basis(
                inputs: akita_types::AkitaScheduleInputs,
                log_basis: u32,
            ) -> Result<akita_types::LevelParams, akita_field::AkitaError> {
                $crate::proof_optimized::proof_optimized_tier3_root_level_layout_with_log_basis::<
                    Self,
                >(inputs, log_basis, $split)
            }

            fn log_basis_at_level(inputs: akita_types::AkitaScheduleInputs) -> u32 {
                $crate::proof_optimized::proof_optimized_log_basis_at_level::<Self>(inputs)
            }

            fn log_basis_search_range(
                _inputs: akita_types::AkitaScheduleInputs,
            ) -> (u32, u32) {
                $crate::proof_optimized::proof_optimized_log_basis_search_range()
            }
        }

        #[cfg(feature = "planner")]
        impl akita_planner::PlannerConfig for $cfg {
            type PlannerField = Field;

            const PLANNER_D: usize = $d;

            fn planner_field_bits() -> u32 {
                <Self as $crate::CommitmentConfig>::decomposition().field_bits()
            }

            fn planner_challenge_field_bits() -> u32 {
                <Self as $crate::CommitmentConfig>::decomposition().field_bits()
                    * (<Self as $crate::CommitmentConfig>::CHAL_EXT_DEGREE as u32)
            }

            fn planner_extension_opening_width() -> usize {
                <Self as $crate::CommitmentConfig>::CLAIM_EXT_DEGREE
            }

            fn planner_recursive_witness_expansion() -> usize {
                1
            }

            fn planner_recursive_public_rows() -> usize {
                1
            }

            fn planner_sis_modulus_family() -> akita_types::SisModulusFamily {
                <Self as $crate::CommitmentConfig>::sis_modulus_family()
            }

            fn planner_stage1_challenge_config(
                d: usize,
            ) -> akita_challenges::SparseChallengeConfig {
                <Self as $crate::CommitmentConfig>::stage1_challenge_config(d)
            }

            fn planner_schedule_plan(
                key: akita_types::AkitaScheduleLookupKey,
            ) -> Result<Option<akita_types::AkitaSchedulePlan>, akita_field::AkitaError> {
                <Self as akita_types::ScheduleProvider>::schedule_plan(key)
            }

            fn planner_root_level_layout_with_log_basis(
                inputs: akita_types::AkitaScheduleInputs,
                log_basis: u32,
            ) -> Result<akita_types::LevelParams, akita_field::AkitaError> {
                <Self as $crate::CommitmentConfig>::root_level_layout_with_log_basis(
                    inputs,
                    log_basis,
                )
            }

            fn planner_current_level_layout_with_log_basis(
                inputs: akita_types::AkitaScheduleInputs,
                log_basis: u32,
            ) -> Result<akita_types::LevelParams, akita_field::AkitaError> {
                $crate::current_level_layout_with_log_basis::<Self>(inputs, log_basis)
            }

            fn planner_direct_level_params_with_log_basis(
                inputs: akita_types::AkitaScheduleInputs,
                log_basis: u32,
            ) -> Result<akita_types::LevelParams, akita_field::AkitaError> {
                $crate::schedule_policy::direct_level_params_with_log_basis::<Self>(
                    inputs, log_basis,
                )
            }

            fn planner_root_level_params_for_layout_with_log_basis(
                inputs: akita_types::AkitaScheduleInputs,
                lp: &akita_types::LevelParams,
            ) -> Result<akita_types::LevelParams, akita_field::AkitaError> {
                <Self as $crate::CommitmentConfig>::root_level_params_for_layout_with_log_basis(
                    inputs, lp,
                )
            }

            fn planner_log_basis_search_range(
                inputs: akita_types::AkitaScheduleInputs,
            ) -> (u32, u32) {
                <Self as $crate::CommitmentConfig>::log_basis_search_range(inputs)
            }
        }
    };
}
pub(crate) use impl_fp128_tier3_preset;

macro_rules! impl_small_field_preset {
    ($cfg:ident, $field:ty, $claim_field:ty, $family:expr, $d:expr, $field_bits:expr, $log_commit_bound:expr, $log_basis:expr, $weight:expr, $coeffs:expr, $table:expr) => {
        impl akita_types::ScheduleProvider for $cfg {
            fn schedule_table() -> Option<akita_types::generated::GeneratedScheduleTable> {
                $table
            }

            fn schedule_key(key: akita_types::AkitaScheduleLookupKey) -> String {
                $crate::proof_optimized::proof_optimized_schedule_key::<Self>(key)
            }

            fn schedule_plan(
                key: akita_types::AkitaScheduleLookupKey,
            ) -> Result<Option<akita_types::AkitaSchedulePlan>, akita_field::AkitaError> {
                $crate::proof_optimized::proof_optimized_schedule_plan::<Self>(key)
            }
        }

        impl $crate::CommitmentConfig for $cfg {
            type Field = $field;
            type ClaimField = $claim_field;
            type ChallengeField = $claim_field;
            const D: usize = $d;

            fn decomposition() -> akita_types::DecompositionParams {
                akita_types::DecompositionParams {
                    log_basis: $log_basis,
                    log_commit_bound: $log_commit_bound,
                    log_open_bound: if $log_commit_bound < $field_bits {
                        Some($field_bits)
                    } else {
                        None
                    },
                }
            }

            fn stage1_challenge_config(d: usize) -> akita_challenges::SparseChallengeConfig {
                assert_eq!(d, Self::D);
                akita_challenges::SparseChallengeConfig::Uniform {
                    weight: $weight,
                    nonzero_coeffs: $coeffs,
                }
            }

            fn sis_modulus_family() -> akita_types::SisModulusFamily {
                $family
            }

            fn audited_root_rank(
                role: akita_types::AjtaiRole,
                max_num_vars: usize,
            ) -> usize {
                let _ = (role, max_num_vars);
                1
            }

            fn envelope(
                max_num_vars: usize,
            ) -> akita_types::CommitmentEnvelope {
                $crate::proof_optimized::proof_optimized_envelope::<Self>(
                    max_num_vars,
                )
            }

            fn max_setup_matrix_size(
                max_num_vars: usize,
                max_num_batched_polys: usize,
                max_num_points: usize,
            ) -> Result<(usize, usize), akita_field::AkitaError> {
                $crate::proof_optimized::proof_optimized_max_setup_matrix_size::<Self>(
                    max_num_vars,
                    max_num_batched_polys,
                    max_num_points,
                )
            }

            fn level_params_with_log_basis(
                inputs: akita_types::AkitaScheduleInputs,
                log_basis: u32,
            ) -> akita_types::LevelParams {
                $crate::proof_optimized::proof_optimized_level_params_with_log_basis::<Self>(
                    inputs,
                    log_basis,
                )
            }

            fn root_level_params_for_layout_with_log_basis(
                inputs: akita_types::AkitaScheduleInputs,
                lp: &akita_types::LevelParams,
            ) -> Result<akita_types::LevelParams, akita_field::AkitaError> {
                $crate::proof_optimized::
                    proof_optimized_root_level_params_for_layout_with_log_basis::<Self>(inputs, lp)
            }

            fn root_level_layout_with_log_basis(
                inputs: akita_types::AkitaScheduleInputs,
                log_basis: u32,
            ) -> Result<akita_types::LevelParams, akita_field::AkitaError> {
                $crate::proof_optimized::proof_optimized_root_level_layout_with_log_basis::<Self>(
                    inputs,
                    log_basis,
                )
            }

            fn log_basis_at_level(
                inputs: akita_types::AkitaScheduleInputs,
            ) -> u32 {
                $crate::proof_optimized::proof_optimized_log_basis_at_level::<Self>(inputs)
            }

            fn log_basis_search_range(
                _inputs: akita_types::AkitaScheduleInputs,
            ) -> (u32, u32) {
                $crate::proof_optimized::proof_optimized_log_basis_search_range()
            }
        }

        #[cfg(feature = "planner")]
        impl akita_planner::PlannerConfig for $cfg {
            type PlannerField = $field;

            const PLANNER_D: usize = $d;

            fn planner_field_bits() -> u32 {
                <Self as $crate::CommitmentConfig>::decomposition().field_bits()
            }

            fn planner_challenge_field_bits() -> u32 {
                <Self as $crate::CommitmentConfig>::decomposition().field_bits()
                    * (<Self as $crate::CommitmentConfig>::CHAL_EXT_DEGREE as u32)
            }

            fn planner_extension_opening_width() -> usize {
                <Self as $crate::CommitmentConfig>::CLAIM_EXT_DEGREE
            }

            fn planner_recursive_witness_expansion() -> usize {
                1
            }

            fn planner_recursive_public_rows() -> usize {
                1
            }

            fn planner_sis_modulus_family() -> akita_types::SisModulusFamily {
                <Self as $crate::CommitmentConfig>::sis_modulus_family()
            }

            fn planner_stage1_challenge_config(
                d: usize,
            ) -> akita_challenges::SparseChallengeConfig {
                <Self as $crate::CommitmentConfig>::stage1_challenge_config(d)
            }

            fn planner_schedule_plan(
                key: akita_types::AkitaScheduleLookupKey,
            ) -> Result<Option<akita_types::AkitaSchedulePlan>, akita_field::AkitaError> {
                <Self as akita_types::ScheduleProvider>::schedule_plan(key)
            }

            fn planner_root_level_layout_with_log_basis(
                inputs: akita_types::AkitaScheduleInputs,
                log_basis: u32,
            ) -> Result<akita_types::LevelParams, akita_field::AkitaError> {
                <Self as $crate::CommitmentConfig>::root_level_layout_with_log_basis(
                    inputs,
                    log_basis,
                )
            }

            fn planner_current_level_layout_with_log_basis(
                inputs: akita_types::AkitaScheduleInputs,
                log_basis: u32,
            ) -> Result<akita_types::LevelParams, akita_field::AkitaError> {
                $crate::current_level_layout_with_log_basis::<Self>(
                    inputs,
                    log_basis,
                )
            }

            fn planner_direct_level_params_with_log_basis(
                inputs: akita_types::AkitaScheduleInputs,
                log_basis: u32,
            ) -> Result<akita_types::LevelParams, akita_field::AkitaError> {
                $crate::schedule_policy::direct_level_params_with_log_basis::<Self>(
                    inputs,
                    log_basis,
                )
            }

            fn planner_root_level_params_for_layout_with_log_basis(
                inputs: akita_types::AkitaScheduleInputs,
                lp: &akita_types::LevelParams,
            ) -> Result<akita_types::LevelParams, akita_field::AkitaError> {
                <Self as $crate::CommitmentConfig>::root_level_params_for_layout_with_log_basis(
                    inputs,
                    lp,
                )
            }

            fn planner_log_basis_search_range(
                inputs: akita_types::AkitaScheduleInputs,
            ) -> (u32, u32) {
                <Self as $crate::CommitmentConfig>::log_basis_search_range(inputs)
            }
        }
    };
}

#[cfg(all(test, not(feature = "zk")))]
mod tests {
    use super::*;

    #[test]
    #[cfg(not(feature = "zk"))]
    fn setup_matrix_envelope_covers_grouped_batch_schedules() {
        let incidence =
            ClaimIncidenceSummary::same_point(30, 4).expect("grouped same-point incidence");
        let envelope = fp128::D32Full::envelope(30);
        let grouped_same_point =
            setup_matrix_envelope_for_shape::<fp128::D32Full>(&incidence, &envelope)
                .unwrap()
                .expect("D32 full table must contain the grouped same-point schedule");

        let setup_envelope = proof_optimized_max_setup_matrix_size::<fp128::D32Full>(30, 4, 1)
            .expect("setup envelope should cover generated grouped batch schedules");
        assert!(setup_envelope.0 >= grouped_same_point.0);
        assert!(setup_envelope.1 >= grouped_same_point.1);
    }

    #[test]
    fn presets_select_expected_sis_modulus_family() {
        assert_eq!(
            <fp128::D32Full as CommitmentConfig>::sis_modulus_family(),
            akita_types::SisModulusFamily::Q128
        );
        assert_eq!(
            <fp32::D64Full as CommitmentConfig>::sis_modulus_family(),
            akita_types::SisModulusFamily::Q32
        );
        assert_eq!(
            <fp64::D64Full as CommitmentConfig>::sis_modulus_family(),
            akita_types::SisModulusFamily::Q64
        );
        assert_eq!(
            <fp16::D64Full as CommitmentConfig>::sis_modulus_family(),
            akita_types::SisModulusFamily::Q16
        );
    }

    #[test]
    fn fp16_generated_schedule_tables_are_wired() {
        let onehot_key = AkitaScheduleLookupKey::singleton(32);
        let onehot_plan =
            <fp16::D32OneHot as akita_types::ScheduleProvider>::schedule_plan(onehot_key)
                .unwrap()
                .expect("fp16 D32 onehot nv32 schedule should be generated");
        assert!(!onehot_plan.steps.is_empty());

        let dense_key = AkitaScheduleLookupKey::singleton(27);
        let dense_plan = <fp16::D32Full as akita_types::ScheduleProvider>::schedule_plan(dense_key)
            .unwrap()
            .expect("fp16 D32 full nv27 schedule should be generated");
        assert!(!dense_plan.steps.is_empty());
    }

    #[test]
    fn fp32_d32_generated_schedule_tables_are_wired() {
        let onehot_key = AkitaScheduleLookupKey::singleton(32);
        let onehot_plan =
            <fp32::D32OneHot as akita_types::ScheduleProvider>::schedule_plan(onehot_key)
                .unwrap()
                .expect("fp32 D32 onehot nv32 schedule should be generated");
        assert!(!onehot_plan.steps.is_empty());

        let dense_key = AkitaScheduleLookupKey::singleton(26);
        let dense_plan = <fp32::D32Full as akita_types::ScheduleProvider>::schedule_plan(dense_key)
            .unwrap()
            .expect("fp32 D32 full nv26 schedule should be generated");
        assert!(!dense_plan.steps.is_empty());
    }
}

// ---------------------------------------------------------------------------
// Public preset structs
// ---------------------------------------------------------------------------

/// Default fp128 protocol presets on `p = 2^128 − 2^32 + 22537`
/// (`Prime128OffsetA7F7`).
pub mod fp128 {
    use super::*;

    /// Base field for the default fp128 presets.
    pub type Field = Prime128OffsetA7F7;

    /// Full-field `D=128` preset for planner-backed experiments.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct D128Full;

    /// Full-field adaptive `D=64` preset.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct D64Full;

    /// Binary onehot generated `D=64` preset.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct D64OneHot;

    /// Full-field adaptive `D=32` preset.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct D32Full;

    /// Onehot adaptive `D=32` preset.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct D32OneHot;

    /// Tier-3 onehot `D=32` preset.
    ///
    /// Same `(field, ring dim, decomposition)` shape as
    /// [`D32OneHot`] but with `split_factor = 3` baked into the root
    /// `LevelParams` (`specs/tiered_commit.md` §3). The recursive
    /// suffix is untouched — tiering applies at the root only. The
    /// generated schedule table is sized for the tier-3 root's
    /// `b_key.col_len = chunk_width = legacy_outer_width / 3`, and
    /// the on-the-wire `n_b = n_b'` (tier-1 B' SIS rank).
    #[derive(Clone, Copy, Debug, Default)]
    pub struct D32OneHotTier3;

    /// Binary onehot `D=128` preset for planner-backed experiments.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct D128OneHot;

    impl_fp128_preset!(D128Full, 128, 128, None);
    impl_fp128_preset!(D128OneHot, 128, 1, None);
    impl_fp128_preset!(
        D64Full,
        64,
        128,
        Some(akita_types::generated::fp128_d64_full_table())
    );
    impl_fp128_preset!(
        D64OneHot,
        64,
        1,
        Some(akita_types::generated::fp128_d64_onehot_table())
    );
    impl_fp128_preset!(
        D32Full,
        32,
        128,
        Some(akita_types::generated::fp128_d32_full_table())
    );
    impl_fp128_preset!(
        D32OneHot,
        32,
        1,
        Some(akita_types::generated::fp128_d32_onehot_table())
    );
    impl_fp128_tier3_preset!(
        D32OneHotTier3,
        32,
        1,
        3,
        Some(akita_types::generated::fp128_d32_onehot_tier3_table())
    );

    /// Concrete fp128 preset selected by a schedule-family query.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Fp128Preset {
        /// Full-field adaptive `D=32` preset.
        D32Full,
        /// Full-field adaptive `D=64` preset.
        D64Full,
        /// Onehot adaptive `D=32` preset.
        D32OneHot,
        /// Binary onehot generated `D=64` preset.
        D64OneHot,
    }

    impl Fp128Preset {
        /// Ring dimension used by this preset.
        pub const fn ring_dimension(self) -> usize {
            match self {
                Self::D32Full | Self::D32OneHot => 32,
                Self::D64Full | Self::D64OneHot => 64,
            }
        }

        /// Whether this preset is onehot-oriented.
        pub const fn is_onehot(self) -> bool {
            matches!(self, Self::D32OneHot | Self::D64OneHot)
        }

        /// Stable human-readable preset name.
        pub const fn name(self) -> &'static str {
            match self {
                Self::D32Full => "D32Full",
                Self::D64Full => "D64Full",
                Self::D32OneHot => "D32OneHot",
                Self::D64OneHot => "D64OneHot",
            }
        }
    }

    /// Best generated-schedule plan for one fp128 preset family.
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub struct Fp128ScheduleSelection {
        /// Selected concrete preset.
        pub preset: Fp128Preset,
        /// Generated schedule plan selected for the supplied lookup key.
        pub plan: AkitaSchedulePlan,
    }

    fn candidate<Cfg: CommitmentConfig>(
        preset: Fp128Preset,
        key: AkitaScheduleLookupKey,
    ) -> Result<Option<Fp128ScheduleSelection>, AkitaError> {
        Ok(Cfg::schedule_plan(key)?.map(|plan| Fp128ScheduleSelection { preset, plan }))
    }

    fn best_by_exact_bytes<I>(candidates: I) -> Option<Fp128ScheduleSelection>
    where
        I: IntoIterator<Item = Option<Fp128ScheduleSelection>>,
    {
        candidates.into_iter().flatten().min_by_key(|selection| {
            (
                selection.plan.exact_proof_bytes,
                selection.preset.ring_dimension(),
            )
        })
    }

    /// Select the best full-field fp128 preset for a schedule lookup key.
    ///
    /// The key carries singleton, grouped, and multipoint batch shape data, so
    /// this helper can be used by profile tooling without manually comparing
    /// typed preset schedule tables. Missing generated rows are ignored; the
    /// returned value is `None` only when no full-field preset has a generated
    /// entry for the key.
    ///
    /// # Errors
    ///
    /// Returns an error if a generated table entry is malformed.
    pub fn best_full_schedule(
        key: AkitaScheduleLookupKey,
    ) -> Result<Option<Fp128ScheduleSelection>, AkitaError> {
        Ok(best_by_exact_bytes([
            candidate::<D32Full>(Fp128Preset::D32Full, key)?,
            candidate::<D64Full>(Fp128Preset::D64Full, key)?,
        ]))
    }

    /// Select the best onehot fp128 preset for a schedule lookup key.
    ///
    /// Missing generated rows are ignored; the returned value is `None` only
    /// when no onehot preset has a generated entry for the key.
    ///
    /// # Errors
    ///
    /// Returns an error if a generated table entry is malformed.
    pub fn best_onehot_schedule(
        key: AkitaScheduleLookupKey,
    ) -> Result<Option<Fp128ScheduleSelection>, AkitaError> {
        Ok(best_by_exact_bytes([
            candidate::<D32OneHot>(Fp128Preset::D32OneHot, key)?,
            candidate::<D64OneHot>(Fp128Preset::D64OneHot, key)?,
        ]))
    }
}

/// fp32 presets used for small-field integration and profiling.
pub mod fp32 {
    use super::*;

    /// Base field for the fp32 scaffold presets.
    pub type Field = Prime32Offset99;
    /// ring-subfield used for fp32 public claims and Fiat-Shamir challenges.
    pub type ExtensionField = RingSubfieldFp4<Field>;

    /// Full-field `D=32` preset for the default fp32 schedule path.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct D32Full;

    /// Onehot `D=32` preset for the default fp32 schedule path.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct D32OneHot;

    /// Full-field `D=64` preset for fp32 crossover profiling.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct D64Full;

    /// Onehot `D=64` preset for fp32 crossover profiling.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct D64OneHot;

    /// Full-field `D=128` preset for planner-backed fp32 experiments.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct D128Full;

    /// Onehot `D=128` preset for planner-backed fp32 experiments.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct D128OneHot;

    /// Full-field `D=256` preset for planner-backed fp32 experiments.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct D256Full;

    /// Onehot `D=256` preset for planner-backed fp32 experiments.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct D256OneHot;

    /// Full-field `D=512` preset for planner-backed fp32 experiments.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct D512Full;

    /// Onehot `D=512` preset for planner-backed fp32 experiments.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct D512OneHot;

    impl_small_field_preset!(
        D32Full,
        Field,
        ExtensionField,
        akita_types::SisModulusFamily::Q32,
        32,
        32,
        32,
        3,
        8,
        vec![-1, 1],
        Some(akita_types::generated::fp32_d32_table())
    );
    impl_small_field_preset!(
        D32OneHot,
        Field,
        ExtensionField,
        akita_types::SisModulusFamily::Q32,
        32,
        32,
        1,
        3,
        8,
        vec![-1, 1],
        Some(akita_types::generated::fp32_d32_onehot_table())
    );
    impl_small_field_preset!(
        D64Full,
        Field,
        ExtensionField,
        akita_types::SisModulusFamily::Q32,
        64,
        32,
        32,
        3,
        8,
        vec![-1, 1],
        Some(akita_types::generated::fp32_d64_table())
    );
    impl_small_field_preset!(
        D64OneHot,
        Field,
        ExtensionField,
        akita_types::SisModulusFamily::Q32,
        64,
        32,
        1,
        3,
        8,
        vec![-1, 1],
        Some(akita_types::generated::fp32_d64_onehot_table())
    );
    impl_small_field_preset!(
        D128Full,
        Field,
        ExtensionField,
        akita_types::SisModulusFamily::Q32,
        128,
        32,
        32,
        3,
        8,
        vec![-1, 1],
        None
    );
    impl_small_field_preset!(
        D128OneHot,
        Field,
        ExtensionField,
        akita_types::SisModulusFamily::Q32,
        128,
        32,
        1,
        3,
        8,
        vec![-1, 1],
        None
    );
    impl_small_field_preset!(
        D256Full,
        Field,
        ExtensionField,
        akita_types::SisModulusFamily::Q32,
        256,
        32,
        32,
        3,
        8,
        vec![-1, 1],
        None
    );
    impl_small_field_preset!(
        D256OneHot,
        Field,
        ExtensionField,
        akita_types::SisModulusFamily::Q32,
        256,
        32,
        1,
        3,
        8,
        vec![-1, 1],
        None
    );
    impl_small_field_preset!(
        D512Full,
        Field,
        ExtensionField,
        akita_types::SisModulusFamily::Q32,
        512,
        32,
        32,
        3,
        8,
        vec![-1, 1],
        None
    );
    impl_small_field_preset!(
        D512OneHot,
        Field,
        ExtensionField,
        akita_types::SisModulusFamily::Q32,
        512,
        32,
        1,
        3,
        8,
        vec![-1, 1],
        None
    );
}

/// fp64 presets used for small-field integration and profiling.
pub mod fp64 {
    use super::*;

    /// Base field for the fp64 scaffold presets.
    pub type Field = Prime64Offset59;
    /// ring-subfield used for fp64 public claims and Fiat-Shamir challenges.
    pub type ExtensionField = Ext2<Field>;

    /// Full-field `D=32` preset for fp64 crossover profiling.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct D32Full;

    /// Onehot `D=32` preset for fp64 crossover profiling.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct D32OneHot;

    /// Full-field `D=64` preset.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct D64Full;

    /// Onehot `D=64` preset.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct D64OneHot;

    /// Full-field `D=128` preset for planner-backed fp64 experiments.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct D128Full;

    /// Onehot `D=128` preset for planner-backed fp64 experiments.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct D128OneHot;

    /// Full-field `D=256` preset for planner-backed fp64 experiments.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct D256Full;

    /// Onehot `D=256` preset for planner-backed fp64 experiments.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct D256OneHot;

    impl_small_field_preset!(
        D32Full,
        Field,
        ExtensionField,
        akita_types::SisModulusFamily::Q64,
        32,
        64,
        64,
        3,
        8,
        vec![-1, 1],
        Some(akita_types::generated::fp64_d32_table())
    );
    impl_small_field_preset!(
        D32OneHot,
        Field,
        ExtensionField,
        akita_types::SisModulusFamily::Q64,
        32,
        64,
        1,
        3,
        8,
        vec![-1, 1],
        Some(akita_types::generated::fp64_d32_onehot_table())
    );
    impl_small_field_preset!(
        D64Full,
        Field,
        ExtensionField,
        akita_types::SisModulusFamily::Q64,
        64,
        64,
        64,
        3,
        8,
        vec![-1, 1],
        Some(akita_types::generated::fp64_d64_table())
    );
    impl_small_field_preset!(
        D64OneHot,
        Field,
        ExtensionField,
        akita_types::SisModulusFamily::Q64,
        64,
        64,
        1,
        3,
        8,
        vec![-1, 1],
        Some(akita_types::generated::fp64_d64_onehot_table())
    );
    impl_small_field_preset!(
        D128Full,
        Field,
        ExtensionField,
        akita_types::SisModulusFamily::Q64,
        128,
        64,
        64,
        3,
        8,
        vec![-1, 1],
        None
    );
    impl_small_field_preset!(
        D128OneHot,
        Field,
        ExtensionField,
        akita_types::SisModulusFamily::Q64,
        128,
        64,
        1,
        3,
        8,
        vec![-1, 1],
        None
    );
    impl_small_field_preset!(
        D256Full,
        Field,
        ExtensionField,
        akita_types::SisModulusFamily::Q64,
        256,
        64,
        64,
        3,
        8,
        vec![-1, 1],
        None
    );
    impl_small_field_preset!(
        D256OneHot,
        Field,
        ExtensionField,
        akita_types::SisModulusFamily::Q64,
        256,
        64,
        1,
        3,
        8,
        vec![-1, 1],
        None
    );
}

/// fp16 presets used for production small-field integration and profiling.
pub mod fp16 {
    use super::*;

    /// Base field for the fp16 presets.
    pub type Field = Prime16Offset99;
    /// Degree-8 ring-subfield used for fp16 public claims and Fiat-Shamir challenges.
    pub type ExtensionField = RingSubfieldFp8<Field>;

    /// Full-field `D=32` preset for fp16 production profiling.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct D32Full;

    /// Onehot `D=32` preset for fp16 production profiling.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct D32OneHot;

    /// Full-field `D=64` preset for fp16 comparison profiling.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct D64Full;

    /// Onehot `D=64` preset for fp16 comparison profiling.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct D64OneHot;

    /// Full-field `D=128` preset for planner-backed fp16 experiments.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct D128Full;

    /// Onehot `D=128` preset for planner-backed fp16 experiments.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct D128OneHot;

    /// Full-field `D=256` preset for planner-backed fp16 experiments.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct D256Full;

    /// Onehot `D=256` preset for planner-backed fp16 experiments.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct D256OneHot;

    /// Full-field `D=512` preset for planner-backed fp16 experiments.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct D512Full;

    /// Onehot `D=512` preset for planner-backed fp16 experiments.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct D512OneHot;

    impl_small_field_preset!(
        D32Full,
        Field,
        ExtensionField,
        akita_types::SisModulusFamily::Q16,
        32,
        16,
        16,
        3,
        8,
        vec![-1, 1],
        Some(akita_types::generated::fp16_d32_full_table())
    );
    impl_small_field_preset!(
        D32OneHot,
        Field,
        ExtensionField,
        akita_types::SisModulusFamily::Q16,
        32,
        16,
        1,
        3,
        8,
        vec![-1, 1],
        Some(akita_types::generated::fp16_d32_onehot_table())
    );
    impl_small_field_preset!(
        D64Full,
        Field,
        ExtensionField,
        akita_types::SisModulusFamily::Q16,
        64,
        16,
        16,
        3,
        8,
        vec![-1, 1],
        Some(akita_types::generated::fp16_d64_full_table())
    );
    impl_small_field_preset!(
        D64OneHot,
        Field,
        ExtensionField,
        akita_types::SisModulusFamily::Q16,
        64,
        16,
        1,
        3,
        8,
        vec![-1, 1],
        Some(akita_types::generated::fp16_d64_onehot_table())
    );
    impl_small_field_preset!(
        D128Full,
        Field,
        ExtensionField,
        akita_types::SisModulusFamily::Q16,
        128,
        16,
        16,
        3,
        8,
        vec![-1, 1],
        None
    );
    impl_small_field_preset!(
        D128OneHot,
        Field,
        ExtensionField,
        akita_types::SisModulusFamily::Q16,
        128,
        16,
        1,
        3,
        8,
        vec![-1, 1],
        None
    );
    impl_small_field_preset!(
        D256Full,
        Field,
        ExtensionField,
        akita_types::SisModulusFamily::Q16,
        256,
        16,
        16,
        3,
        8,
        vec![-1, 1],
        None
    );
    impl_small_field_preset!(
        D256OneHot,
        Field,
        ExtensionField,
        akita_types::SisModulusFamily::Q16,
        256,
        16,
        1,
        3,
        8,
        vec![-1, 1],
        None
    );
    impl_small_field_preset!(
        D512Full,
        Field,
        ExtensionField,
        akita_types::SisModulusFamily::Q16,
        512,
        16,
        16,
        3,
        8,
        vec![-1, 1],
        None
    );
    impl_small_field_preset!(
        D512OneHot,
        Field,
        ExtensionField,
        akita_types::SisModulusFamily::Q16,
        512,
        16,
        1,
        3,
        8,
        vec![-1, 1],
        None
    );
}
