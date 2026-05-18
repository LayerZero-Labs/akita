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
use akita_challenges::Stage1ChallengeShape;
use akita_field::AkitaError;
use akita_field::{Pow2Offset32Field, Pow2Offset64Field, Prime128OffsetA7F7};
use akita_types::generated::table_entry_envelope_for_max_num_vars;
#[cfg(feature = "planner")]
use akita_types::WitnessShape;
use akita_types::{
    exact_planned_level_execution, planned_log_basis_at_level_from_schedule,
    planned_schedule_key_from_schedule, AkitaRootBatchSummary, AkitaScheduleInputs,
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
///
/// Each family must provide `|C| >= 2^128` Fiat-Shamir challenge-space
/// entropy so the per-level CWSS knowledge error
/// `eps_tensor = 4 * 2^(r/2) / |C|` stays below `2^-128`. See
/// `specs/security_analysis.md` Section 5 for the derivation and per-family
/// numbers.
///
/// - **D=32**: `BoundedL1Norm` is truncated to exactly `2^128` challenges
///   by the sampler in `crates/akita-challenges/src/sampler/bounded_l1.rs`.
///   `omega = 121`, `||c||_inf = 8`. Used in `Flat` shape only.
/// - **D=64**: `ExactShell{30, 12}` gives 30 magnitude-1 coefficients and 12
///   magnitude-2 coefficients (42 nonzero positions out of 64). `|C| ≈ 2^131.6`,
///   `omega = 30 + 2*12 = 54`. The `4*omega = 216` MSIS extraction degradation
///   matches the figure cited in book §5 (~8 MSIS bits at the 280+ bit floor).
/// - **D=128**: `Uniform{32, ±1}` gives 32 nonzero positions with random
///   signs out of 128 (`|C| ≈ 2^131.7`, `omega = 32`). The book §5 cites
///   `omega = 31` at D=128; one extra weight gives a small margin without
///   meaningfully increasing the MSIS penalty.
pub(crate) fn fp128_stage1_challenge_config(d: usize) -> SparseChallengeConfig {
    match d {
        32 => SparseChallengeConfig::BoundedL1Norm,
        64 => SparseChallengeConfig::ExactShell {
            count_mag1: 30,
            count_mag2: 12,
        },
        128 => SparseChallengeConfig::Uniform {
            weight: 32,
            nonzero_coeffs: vec![-1, 1],
        },
        _ => panic!("unsupported fp128 ring dim {d}"),
    }
}

fn stage1_challenge_shape_for_config(config: &SparseChallengeConfig) -> Stage1ChallengeShape {
    match config {
        SparseChallengeConfig::BoundedL1Norm => Stage1ChallengeShape::Flat,
        SparseChallengeConfig::Uniform { .. } | SparseChallengeConfig::ExactShell { .. } => {
            Stage1ChallengeShape::Tensor
        }
    }
}

fn apply_stage1_challenge_shape(mut params: LevelParams) -> LevelParams {
    params.stage1_challenge_shape = stage1_challenge_shape_for_config(&params.stage1_config);
    params
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

/// Read the planned schedule for `key` from the config's generated table.
fn lookup_planned_schedule<Cfg: CommitmentConfig>(
    key: AkitaScheduleLookupKey,
) -> Result<Option<AkitaSchedulePlan>, AkitaError> {
    let Some(table) = Cfg::schedule_table() else {
        return Ok(None);
    };
    generated_schedule_plan_from_table::<Cfg>(key, table)
}

/// Inclusive `(min, max)` log-basis search range used by every fp128 preset.
pub(crate) fn proof_optimized_log_basis_search_range() -> (u32, u32) {
    (PROOF_OPTIMIZED_LOG_BASIS_MIN, PROOF_OPTIMIZED_LOG_BASIS_MAX)
}

/// Proof-optimized `schedule_plan` impl.
pub(crate) fn proof_optimized_schedule_plan<Cfg: CommitmentConfig>(
    key: AkitaScheduleLookupKey,
) -> Result<Option<AkitaSchedulePlan>, AkitaError> {
    lookup_planned_schedule::<Cfg>(key)
}

/// Proof-optimized `schedule_key` impl: derive a stable identifier from the
/// planned schedule (or from the lookup key when no entry exists).
pub(crate) fn proof_optimized_schedule_key<Cfg: CommitmentConfig>(
    key: AkitaScheduleLookupKey,
) -> String {
    match lookup_planned_schedule::<Cfg>(key) {
        Ok(Some(plan)) => planned_schedule_key_from_schedule(key, &plan),
        _ => format!(
            "generated-miss/d{}/max{}/num{}/claims{}/batch{}g{}p{}",
            Cfg::D,
            key.max_num_vars,
            key.num_vars,
            key.layout_num_claims,
            key.batch.num_claims,
            key.batch.num_commitment_groups,
            key.batch.num_points,
        ),
    }
}

/// Proof-optimized `log_basis_at_level` impl: read from the planned schedule
/// when available; otherwise fall back to the root decomposition's basis.
pub(crate) fn proof_optimized_log_basis_at_level<Cfg: CommitmentConfig>(
    inputs: AkitaScheduleInputs,
) -> u32 {
    let key = AkitaScheduleLookupKey::singleton(inputs.max_num_vars, inputs.max_num_vars, 1);
    match lookup_planned_schedule::<Cfg>(key) {
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
    let singleton_key =
        AkitaScheduleLookupKey::singleton(inputs.max_num_vars, inputs.max_num_vars, 1);
    if let Ok(Some(plan)) = lookup_planned_schedule::<Cfg>(singleton_key) {
        if let Ok(Some(planned_level)) =
            exact_planned_level_execution(&plan, inputs, log_basis, Cfg::stage1_challenge_config)
        {
            return planned_level.level.lp.clone();
        }
    }
    let envelope = Cfg::envelope(inputs.max_num_vars);
    let d = Cfg::D;
    let stage1_config = Cfg::stage1_challenge_config(d);

    if inputs.level > 0 {
        if let Some(mut params) = sis_derived_recursive_params::<Cfg>(
            d,
            log_basis,
            inputs.current_w_len,
            &stage1_config,
            &envelope,
        ) {
            params.stage1_challenge_shape = stage1_challenge_shape_for_config(&stage1_config);
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

    apply_stage1_challenge_shape(LevelParams::params_only(
        d,
        log_basis,
        envelope.max_n_a,
        envelope.max_n_b,
        envelope.max_n_d,
        stage1_config,
    ))
}

/// Proof-optimized `root_level_params_for_layout_with_log_basis` impl.
pub(crate) fn proof_optimized_root_level_params_for_layout_with_log_basis<Cfg: CommitmentConfig>(
    inputs: AkitaScheduleInputs,
    lp: &LevelParams,
) -> Result<LevelParams, AkitaError> {
    let params = sis_derived_root_params_for_layout::<Cfg>(inputs, lp)?;
    Ok(params.with_layout(lp))
}

/// Proof-optimized `root_level_layout_with_log_basis` impl.
///
/// Iterates `candidate_n_a` until the rank derived by
/// `sis_derived_root_params_for_layout` is at most the candidate rank used to
/// build the layout. That's a sufficient fixed-point: the layout was secure
/// under `candidate_n_a`, so the result we return — with the candidate's
/// layout and `derived.a_key.row_len()` ranks — is also SIS-secure.
///
/// Bounded by `MAX_RANK + 1` iterations. Returns `InvalidSetup` if no
/// candidate rank in `1..=MAX_RANK` is sufficient (would indicate the
/// supplied `inputs` exceed the SIS table coverage at any rank).
pub(crate) fn proof_optimized_root_level_layout_with_log_basis<Cfg: CommitmentConfig>(
    inputs: AkitaScheduleInputs,
    log_basis: u32,
) -> Result<LevelParams, AkitaError> {
    let stage1_config = Cfg::stage1_challenge_config(Cfg::D);
    let mut candidate_n_a = 1usize;
    for _ in 0..(akita_types::generated::sis_floor::MAX_RANK + 1) {
        let candidate_params = apply_stage1_challenge_shape(LevelParams::params_only(
            Cfg::D,
            log_basis,
            candidate_n_a,
            1,
            1,
            stage1_config.clone(),
        ));
        let root_lp =
            derived_root_commitment_layout_from_params::<Cfg>(inputs, &candidate_params, false)?;
        let derived_params = sis_derived_root_params_for_layout::<Cfg>(inputs, &root_lp)?;
        if derived_params.a_key.row_len() <= candidate_n_a {
            // The candidate layout is secure at `derived` rank
            // (≤ `candidate_n_a`), hence also at `candidate_n_a`. Return
            // the derived params (which include SIS-secure b/d ranks) but
            // overwrite the a-rank with the candidate we used to lay out
            // the level, so the layout's `inner_width` matches the rank
            // it was sized for.
            let mut result = derived_params;
            result.a_key = akita_types::AjtaiKeyParams::try_new(
                candidate_n_a,
                result.a_key.col_len(),
                result.a_key.collision_inf(),
                Cfg::D,
            )?;
            return Ok(result.with_layout(&root_lp));
        }
        candidate_n_a = derived_params.a_key.row_len();
    }
    Err(AkitaError::InvalidSetup(format!(
        "failed to converge on self-consistent root A-row rank for D={} lb={log_basis}",
        Cfg::D
    )))
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
            table_entry_envelope_for_max_num_vars(table, max_num_vars)
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
/// `num_polys`, so the final envelope is the max over every committable
/// sub-shape `(num_polys', num_commitment_groups', num_points')` with
/// `1 <= num_polys' <= max_num_batched_polys` and
/// `1 <= num_commitment_groups' <= num_polys'` and
/// `1 <= num_points' <= num_polys'.min(max_num_points)`. Without this, a
/// runtime commit at a smaller or differently grouped batch shape can pick a
/// schedule with strictly larger row count than the all-up envelope.
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
    for num_vars in 1..=max_num_vars {
        for num_polys in 1..=max_num_batched_polys {
            let upper_pts = num_polys.min(max_num_points);
            for num_commitment_groups in 1..=num_polys {
                for num_points in 1..=upper_pts {
                    let Some((rows, stride)) = setup_matrix_envelope_for_shape::<Cfg>(
                        num_vars,
                        num_polys,
                        num_commitment_groups,
                        num_points,
                    )?
                    else {
                        continue;
                    };
                    saw_supported_shape = true;
                    max_rows = max_rows.max(rows);
                    max_stride = max_stride.max(stride);
                }
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
    max_num_vars: usize,
    num_polys: usize,
    num_commitment_groups: usize,
    num_points: usize,
) -> Result<Option<(usize, usize)>, AkitaError> {
    let batch_summary = AkitaRootBatchSummary::new(num_polys, num_commitment_groups, num_points)?;
    let cached_key =
        AkitaScheduleLookupKey::with_batch(max_num_vars, max_num_vars, num_polys, batch_summary);

    let fallback = fallback_batched_root_split::<Cfg>(max_num_vars, num_polys)?;

    // Collect every level the prover/verifier actually consults: each `Fold`
    // step plus the `commit_w_for_next` layout the prover uses at the level
    // *after* every Fold step. The latter is `level_params_with_log_basis(
    // level=k+1, current_w_len=next_w_len)`. For schedules that end in
    // `Direct`, this captures the commit layout for the terminal witness
    // which is otherwise invisible to the schedule's `fold_levels()`
    // iterator. The setup matrix must cover the maximum width/rank across
    // all of these so that the prover's `NttSlotCache` never indexes past
    // its actual storage at recursive commit time.
    let mut all_levels: Vec<LevelParams> = vec![fallback];

    let fold_levels: Vec<(LevelParams, usize)> = if let Some(plan) = Cfg::schedule_plan(cached_key)?
    {
        plan.fold_levels()
            .map(|level| (level.lp.clone(), level.next_inputs.current_w_len))
            .collect()
    } else {
        #[cfg(feature = "planner")]
        {
            let schedule = akita_planner::find_optimal_schedule::<Cfg>(
                max_num_vars,
                WitnessShape::new(num_polys, num_commitment_groups, num_points),
            )?;
            schedule
                .steps
                .into_iter()
                .filter_map(|step| match step {
                    akita_types::Step::Fold(level) => Some((level.params, level.next_w_len)),
                    akita_types::Step::Direct(_) => None,
                })
                .collect()
        }

        #[cfg(not(feature = "planner"))]
        {
            let _ = cached_key;
            return Ok(None);
        }
    };

    for (level_idx, (lp, next_w_len)) in fold_levels.iter().enumerate() {
        all_levels.push(lp.clone());
        // After every Fold step the prover commits the next witness using the
        // *next* level's params. Include those params in the envelope so the
        // setup matrix is large enough for the recursive commit, regardless
        // of whether the next step is another Fold or a terminal Direct.
        //
        // If the next step isn't another Fold (it's a terminal Direct that
        // the planner already accounted for), the next-level params don't
        // exist in the schedule plan and `log_basis_at_level` panics. We
        // handle that by reusing the current fold's log_basis: the commit
        // layout the prover builds for a Fold->Direct transition uses the
        // same basis as the current fold, since `recursive_level_decomposition
        // _from_root` keys off `parent.log_basis`.
        let next_inputs = AkitaScheduleInputs {
            max_num_vars,
            level: level_idx + 1,
            current_w_len: *next_w_len,
        };
        let next_log_basis = match Cfg::schedule_plan(cached_key) {
            Ok(Some(plan)) => {
                akita_types::planned_log_basis_at_level_from_schedule(&plan, next_inputs)
                    .unwrap_or(lp.log_basis)
            }
            _ => lp.log_basis,
        };
        let next_lp = Cfg::level_params_with_log_basis(next_inputs, next_log_basis);
        all_levels.push(next_lp);
    }

    Ok(Some(reduce_level_params_to_matrix_size(all_levels.iter())))
}

fn reduce_level_params_to_matrix_size<'a, I>(level_params: I) -> (usize, usize)
where
    I: IntoIterator<Item = &'a LevelParams>,
{
    let mut max_rows: usize = 1;
    let mut max_stride: usize = 1;
    for lp in level_params {
        max_rows = max_rows
            .max(lp.a_key.row_len())
            .max(lp.b_key.row_len())
            .max(lp.d_key.row_len());
        max_stride = max_stride
            .max(lp.inner_width().next_power_of_two())
            .max(lp.outer_width().next_power_of_two())
            .max(lp.d_matrix_width().next_power_of_two());
    }
    (max_rows, max_stride)
}

// ---------------------------------------------------------------------------
// Per-preset CommitmentConfig macro
// ---------------------------------------------------------------------------

/// Generate a complete [`CommitmentConfig`] impl for one fp128 preset.
///
/// Each preset only ships its `(D, LOG_COMMIT_BOUND)` decomposition and the
/// generated schedule table. Every other trait method is a one-line
/// delegation to the proof-optimized helpers above.
macro_rules! impl_fp128_preset {
    ($cfg:ident, $d:expr, $log_commit_bound:expr, $table:ident) => {
        impl akita_types::ScheduleProvider for $cfg {
            fn schedule_table() -> Option<akita_types::generated::GeneratedScheduleTable> {
                Some(akita_types::generated::$table())
            }

            fn allow_tensor_stage1_schedules() -> bool {
                true
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
            const PLANNER_D: usize = $d;

            fn planner_field_bits() -> u32 {
                <Self as $crate::CommitmentConfig>::decomposition().field_bits()
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

            fn planner_stage1_prover_weight() -> usize {
                3
            }
        }
    };
}
pub(crate) use impl_fp128_preset;

macro_rules! impl_small_field_preset {
    ($cfg:ident, $field:ty, $d:expr, $field_bits:expr, $log_basis:expr, $weight:expr, $coeffs:expr) => {
        impl akita_types::ScheduleProvider for $cfg {
            fn schedule_table() -> Option<akita_types::generated::GeneratedScheduleTable> {
                None
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
            type ClaimField = $field;
            type ChallengeField = $field;
            const D: usize = $d;

            fn decomposition() -> akita_types::DecompositionParams {
                akita_types::DecompositionParams {
                    log_basis: $log_basis,
                    log_commit_bound: $field_bits,
                    log_open_bound: None,
                }
            }

            fn stage1_challenge_config(d: usize) -> akita_challenges::SparseChallengeConfig {
                assert_eq!(d, Self::D);
                akita_challenges::SparseChallengeConfig::Uniform {
                    weight: $weight,
                    nonzero_coeffs: $coeffs,
                }
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
                ($log_basis, $log_basis)
            }
        }

        #[cfg(feature = "planner")]
        impl akita_planner::PlannerConfig for $cfg {
            const PLANNER_D: usize = $d;

            fn planner_field_bits() -> u32 {
                <Self as $crate::CommitmentConfig>::decomposition().field_bits()
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

            fn planner_stage1_prover_weight() -> usize {
                0
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn setup_matrix_envelope_covers_grouped_batch_schedules() {
        let grouped_same_point = setup_matrix_envelope_for_shape::<fp128::D32Full>(30, 4, 1, 1)
            .unwrap()
            .expect("D32 full table must contain the grouped same-point schedule");

        let setup_envelope = proof_optimized_max_setup_matrix_size::<fp128::D32Full>(30, 4, 1)
            .expect("setup envelope should cover generated grouped batch schedules");
        assert!(setup_envelope.0 >= grouped_same_point.0);
        assert!(setup_envelope.1 >= grouped_same_point.1);
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

    /// Full-field adaptive `D=128` preset.
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

    /// Binary onehot generated `D=128` preset.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct D128OneHot;

    impl_fp128_preset!(D128Full, 128, 128, fp128_d128_full_table);
    impl_fp128_preset!(D128OneHot, 128, 1, fp128_d128_onehot_table);
    impl_fp128_preset!(D64Full, 64, 128, fp128_d64_full_table);
    impl_fp128_preset!(D64OneHot, 64, 1, fp128_d64_onehot_table);
    impl_fp128_preset!(D32Full, 32, 128, fp128_d32_full_table);
    impl_fp128_preset!(D32OneHot, 32, 1, fp128_d32_onehot_table);

    /// Concrete fp128 preset selected by a schedule-family query.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Fp128Preset {
        /// Full-field adaptive `D=32` preset.
        D32Full,
        /// Full-field adaptive `D=64` preset.
        D64Full,
        /// Full-field adaptive `D=128` preset.
        D128Full,
        /// Onehot adaptive `D=32` preset.
        D32OneHot,
        /// Binary onehot generated `D=64` preset.
        D64OneHot,
        /// Binary onehot generated `D=128` preset.
        D128OneHot,
    }

    impl Fp128Preset {
        /// Ring dimension used by this preset.
        pub const fn ring_dimension(self) -> usize {
            match self {
                Self::D32Full | Self::D32OneHot => 32,
                Self::D64Full | Self::D64OneHot => 64,
                Self::D128Full | Self::D128OneHot => 128,
            }
        }

        /// Whether this preset is onehot-oriented.
        pub const fn is_onehot(self) -> bool {
            matches!(self, Self::D32OneHot | Self::D64OneHot | Self::D128OneHot)
        }

        /// Stable human-readable preset name.
        pub const fn name(self) -> &'static str {
            match self {
                Self::D32Full => "D32Full",
                Self::D64Full => "D64Full",
                Self::D128Full => "D128Full",
                Self::D32OneHot => "D32OneHot",
                Self::D64OneHot => "D64OneHot",
                Self::D128OneHot => "D128OneHot",
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
            candidate::<D128Full>(Fp128Preset::D128Full, key)?,
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
            candidate::<D128OneHot>(Fp128Preset::D128OneHot, key)?,
        ]))
    }
}

/// Static fp32 scaffold presets used for small-field integration coverage.
pub mod fp32 {
    use super::*;

    /// Base field for the fp32 scaffold presets.
    pub type Field = Pow2Offset32Field;

    /// Full-field static `D=32` preset.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct D32Static;

    impl_small_field_preset!(D32Static, Field, 32, 32, 3, 8, vec![-1, 1]);
}

/// Static fp64 scaffold presets used for small-field integration coverage.
pub mod fp64 {
    use super::*;

    /// Base field for the fp64 scaffold presets.
    pub type Field = Pow2Offset64Field;

    /// Full-field static `D=64` preset.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct D64Static;

    impl_small_field_preset!(D64Static, Field, 64, 64, 3, 8, vec![-1, 1]);
}
