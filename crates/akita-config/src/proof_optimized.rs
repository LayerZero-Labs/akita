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
use akita_field::{Ext2, Prime128OffsetA7F7, Prime32Offset99, Prime64Offset59, RingSubfieldFp4};
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
            key.num_commitment_groups,
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

/// Proof-optimized `root_level_layout_with_log_basis` impl.
pub(crate) fn proof_optimized_root_level_layout_with_log_basis<Cfg: CommitmentConfig>(
    inputs: AkitaScheduleInputs,
    log_basis: u32,
) -> Result<LevelParams, AkitaError> {
    let stage1_config = Cfg::stage1_challenge_config(Cfg::D);
    let mut candidate_n_a = 1usize;
    for _ in 0..akita_types::generated::sis_floor::MAX_RANK {
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
/// committable sub-shape `(num_vars', num_polys', num_commitment_groups',
/// num_points')` with `1 <= num_vars' <= max_num_vars`,
/// `1 <= num_polys' <= max_num_batched_polys` and
/// `1 <= num_commitment_groups' <= num_polys'` and
/// `1 <= num_points' <= num_polys'.min(max_num_points)`. Without this, a
/// runtime commit at a smaller variable count or differently grouped batch
/// shape can pick a schedule with strictly larger row count than the all-up
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
            for num_commitment_groups in 1..=num_polys {
                for num_points in 1..=upper_pts {
                    let incidence = ClaimIncidenceSummary::from_counts(
                        num_vars,
                        num_polys,
                        num_commitment_groups,
                        num_points,
                    )?;
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
    let num_polys = incidence.num_polynomials()?;
    let cached_key = AkitaScheduleLookupKey::new_from_incidence(incidence)?;

    let fallback = fallback_batched_root_split::<Cfg>(incidence.num_vars, num_polys)?;

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
/// generated schedule table. Every other trait method is a one-line
/// delegation to the proof-optimized helpers above.
macro_rules! impl_fp128_preset {
    ($cfg:ident, $d:expr, $log_commit_bound:expr, $table:ident) => {
        impl akita_types::ScheduleProvider for $cfg {
            fn schedule_table() -> Option<akita_types::generated::GeneratedScheduleTable> {
                Some(akita_types::generated::$table())
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
            <fp32::D128Full as CommitmentConfig>::sis_modulus_family(),
            akita_types::SisModulusFamily::Q32
        );
        assert_eq!(
            <fp64::D128Full as CommitmentConfig>::sis_modulus_family(),
            akita_types::SisModulusFamily::Q64
        );
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

/// fp32 presets used for small-field integration and profiling.
pub mod fp32 {
    use super::*;

    /// Base field for the fp32 scaffold presets.
    pub type Field = Prime32Offset99;
    /// ring-subfield used for fp32 public claims and Fiat-Shamir challenges.
    pub type ExtensionField = RingSubfieldFp4<Field>;

    /// Full-field `D=32` preset retained for tuning/regression coverage.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct D32Full;

    /// Onehot `D=32` preset retained for tuning/regression coverage.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct D32OneHot;

    /// Full-field `D=64` preset for fp32 crossover profiling.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct D64Full;

    /// Onehot `D=64` preset for fp32 crossover profiling.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct D64OneHot;

    /// Full-field `D=128` preset for security-calibrated fp32 planning.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct D128Full;

    /// Onehot `D=128` preset for security-calibrated fp32 planning.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct D128OneHot;

    /// Full-field `D=256` preset for security-calibrated fp32 planning.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct D256Full;

    /// Onehot `D=256` preset for security-calibrated fp32 planning.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct D256OneHot;

    /// Full-field `D=512` preset for security-calibrated fp32 planning.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct D512Full;

    /// Onehot `D=512` preset for security-calibrated fp32 planning.
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
        None
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
        None
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
        Some(akita_types::generated::fp32_d128_table())
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
        Some(akita_types::generated::fp32_d128_onehot_table())
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
        Some(akita_types::generated::fp32_d256_table())
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
        Some(akita_types::generated::fp32_d256_onehot_table())
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
        Some(akita_types::generated::fp32_d512_table())
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
        Some(akita_types::generated::fp32_d512_onehot_table())
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

    /// Full-field `D=128` preset for security-calibrated fp64 planning.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct D128Full;

    /// Onehot `D=128` preset for security-calibrated fp64 planning.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct D128OneHot;

    /// Full-field `D=256` preset for security-calibrated fp64 planning.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct D256Full;

    /// Onehot `D=256` preset for security-calibrated fp64 planning.
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
        Some(akita_types::generated::fp64_d128_table())
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
        Some(akita_types::generated::fp64_d128_onehot_table())
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
        Some(akita_types::generated::fp64_d256_table())
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
        Some(akita_types::generated::fp64_d256_onehot_table())
    );
}
