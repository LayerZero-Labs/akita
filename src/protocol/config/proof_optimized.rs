//! Concrete proof-optimized commitment configs for the default fp128 protocol.
//!
//! Each config is a plain unit struct that wires its required
//! [`CommitmentConfig`] hooks to the policy-agnostic SIS primitives in
//! the crate-internal `commitment::sis_derivation` module and the
//! planned-schedule tables in `commitment::generated`. A preset only
//! declares its `(D, LOG_COMMIT_BOUND)` decomposition, its sparse stage-1
//! family, the generated schedule table that backs it, and (when applicable)
//! the audited root-rank floor.

use super::{AjtaiRole, CommitmentConfig, CommitmentEnvelope, DecompositionParams};
use crate::protocol::commitment::schedule::{
    exact_planned_level_execution, fallback_batched_root_split, generated_schedule_plan_from_table,
    hachi_recursive_level_layout_from_params, planned_log_basis_at_level_from_schedule,
    planned_schedule_key_from_schedule, HachiRootBatchSummary, HachiScheduleInputs,
    HachiScheduleLookupKey, HachiSchedulePlan,
};
use crate::protocol::commitment::sis_derivation::{
    derived_root_commitment_layout_from_params, sis_derived_recursive_params,
    sis_derived_root_params_for_layout,
};
use akita_algebra::{Prime128OffsetA7F7, SparseChallengeConfig};
use akita_field::HachiError;
use akita_types::generated::table_entry_envelope_for_max_num_vars;
use akita_types::LevelParams;

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
        32 => SparseChallengeConfig::Uniform {
            weight: 32,
            nonzero_coeffs: (-8..=8).filter(|&c| c != 0).collect(),
        },
        64 => SparseChallengeConfig::SplitRing {
            half_weight: 21,
            max_mag2_per_half: 6,
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
// SIS primitives in `commitment::sis_derivation` otherwise.
// ---------------------------------------------------------------------------

/// Read the planned schedule for `key` from the config's generated table.
fn lookup_planned_schedule<Cfg: CommitmentConfig>(
    key: HachiScheduleLookupKey,
) -> Result<Option<HachiSchedulePlan>, HachiError> {
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
    key: HachiScheduleLookupKey,
) -> Result<Option<HachiSchedulePlan>, HachiError> {
    lookup_planned_schedule::<Cfg>(key)
}

/// Proof-optimized `schedule_key` impl: derive a stable identifier from the
/// planned schedule (or from the lookup key when no entry exists).
pub(crate) fn proof_optimized_schedule_key<Cfg: CommitmentConfig>(
    key: HachiScheduleLookupKey,
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
    inputs: HachiScheduleInputs,
) -> u32 {
    let key = HachiScheduleLookupKey::singleton(inputs.max_num_vars, inputs.max_num_vars, 1);
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
    inputs: HachiScheduleInputs,
    log_basis: u32,
) -> LevelParams {
    let singleton_key =
        HachiScheduleLookupKey::singleton(inputs.max_num_vars, inputs.max_num_vars, 1);
    if let Ok(Some(plan)) = lookup_planned_schedule::<Cfg>(singleton_key) {
        if let Ok(Some(planned_level)) =
            exact_planned_level_execution::<Cfg>(&plan, inputs, log_basis)
        {
            return planned_level.level.lp.clone();
        }
    }
    let envelope = Cfg::envelope(inputs.max_num_vars);
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
            if let Ok(lp) =
                hachi_recursive_level_layout_from_params::<Cfg>(&params, inputs.current_w_len)
            {
                return lp;
            }
            return params;
        }
    }

    LevelParams::params_only(
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
    inputs: HachiScheduleInputs,
    lp: &LevelParams,
) -> Result<LevelParams, HachiError> {
    let params = sis_derived_root_params_for_layout::<Cfg>(inputs, lp)?;
    Ok(params.with_layout(lp))
}

/// Proof-optimized `root_level_layout_with_log_basis` impl.
pub(crate) fn proof_optimized_root_level_layout_with_log_basis<Cfg: CommitmentConfig>(
    inputs: HachiScheduleInputs,
    log_basis: u32,
) -> Result<LevelParams, HachiError> {
    let stage1_config = Cfg::stage1_challenge_config(Cfg::D);
    let mut candidate_n_a = 1usize;
    for _ in 0..akita_types::generated::sis_floor::MAX_RANK {
        let candidate_params = LevelParams::params_only(
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
    Err(HachiError::InvalidSetup(format!(
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
pub(crate) fn proof_optimized_max_setup_matrix_size<Cfg: CommitmentConfig>(
    max_num_vars: usize,
    max_num_batched_polys: usize,
    max_num_points: usize,
) -> Result<(usize, usize), HachiError> {
    if max_num_batched_polys == 0 {
        return Err(HachiError::InvalidSetup(
            "max_num_batched_polys must be at least 1".to_string(),
        ));
    }
    if max_num_points == 0 {
        return Err(HachiError::InvalidSetup(
            "max_num_points must be at least 1".to_string(),
        ));
    }
    if max_num_points > max_num_batched_polys {
        return Err(HachiError::InvalidSetup(format!(
            "max_num_points ({max_num_points}) cannot exceed max_num_batched_polys ({max_num_batched_polys})"
        )));
    }

    let batch_summary =
        HachiRootBatchSummary::new(max_num_batched_polys, max_num_batched_polys, max_num_points)?;
    let cached_key = HachiScheduleLookupKey::with_batch(
        max_num_vars,
        max_num_vars,
        max_num_batched_polys,
        batch_summary,
    );

    let fallback = fallback_batched_root_split::<Cfg>(max_num_vars, max_num_batched_polys)?;

    let fold_levels: Vec<LevelParams> = if let Some(plan) = Cfg::schedule_plan(cached_key)? {
        plan.fold_levels().map(|level| level.lp.clone()).collect()
    } else {
        use crate::planner::schedule_params::find_optimal_schedule;
        use crate::protocol::commitment::{Step, WitnessShape};
        let shape = WitnessShape {
            num_claims: max_num_batched_polys,
            num_commitment_groups: max_num_batched_polys,
            num_points: max_num_points,
        };
        let schedule = find_optimal_schedule::<Cfg>(max_num_vars, shape)?;
        schedule
            .steps
            .iter()
            .filter_map(|step| match step {
                Step::Fold(fold) => Some(fold.params.clone()),
                Step::Direct(_) => None,
            })
            .collect()
    };

    Ok(reduce_level_params_to_matrix_size(
        std::iter::once(&fallback).chain(fold_levels.iter()),
    ))
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
            .max(lp.inner_width())
            .max(lp.outer_width())
            .max(lp.d_matrix_width());
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
        impl $crate::protocol::config::CommitmentConfig for $cfg {
            type Field = Field;
            const D: usize = $d;

            fn decomposition() -> $crate::protocol::config::DecompositionParams {
                $crate::protocol::config::proof_optimized::fp128_decomposition(
                    $log_commit_bound,
                    3,
                )
            }

            fn stage1_challenge_config(d: usize) -> akita_algebra::SparseChallengeConfig {
                $crate::protocol::config::proof_optimized::fp128_stage1_challenge_config(d)
            }

            #[allow(private_interfaces)]
            fn schedule_table(
            ) -> Option<akita_types::generated::GeneratedScheduleTable> {
                Some(akita_types::generated::$table())
            }

            fn audited_root_rank(
                role: $crate::protocol::config::AjtaiRole,
                max_num_vars: usize,
            ) -> usize {
                $crate::protocol::config::proof_optimized::fp128_audited_root_rank::<Self>(
                    role,
                    max_num_vars,
                )
            }

            fn envelope(
                max_num_vars: usize,
            ) -> $crate::protocol::config::CommitmentEnvelope {
                $crate::protocol::config::proof_optimized::proof_optimized_envelope::<Self>(
                    max_num_vars,
                )
            }

            fn max_setup_matrix_size(
                max_num_vars: usize,
                max_num_batched_polys: usize,
                max_num_points: usize,
            ) -> Result<(usize, usize), akita_field::HachiError> {
                $crate::protocol::config::proof_optimized::
                    proof_optimized_max_setup_matrix_size::<Self>(
                        max_num_vars,
                        max_num_batched_polys,
                        max_num_points,
                    )
            }

            fn level_params_with_log_basis(
                inputs: $crate::protocol::commitment::HachiScheduleInputs,
                log_basis: u32,
            ) -> akita_types::LevelParams {
                $crate::protocol::config::proof_optimized::
                    proof_optimized_level_params_with_log_basis::<Self>(inputs, log_basis)
            }

            fn root_level_params_for_layout_with_log_basis(
                inputs: $crate::protocol::commitment::HachiScheduleInputs,
                lp: &akita_types::LevelParams,
            ) -> Result<akita_types::LevelParams, akita_field::HachiError> {
                $crate::protocol::config::proof_optimized::
                    proof_optimized_root_level_params_for_layout_with_log_basis::<Self>(inputs, lp)
            }

            fn root_level_layout_with_log_basis(
                inputs: $crate::protocol::commitment::HachiScheduleInputs,
                log_basis: u32,
            ) -> Result<akita_types::LevelParams, akita_field::HachiError> {
                $crate::protocol::config::proof_optimized::
                    proof_optimized_root_level_layout_with_log_basis::<Self>(inputs, log_basis)
            }

            fn log_basis_at_level(
                inputs: $crate::protocol::commitment::HachiScheduleInputs,
            ) -> u32 {
                $crate::protocol::config::proof_optimized::
                    proof_optimized_log_basis_at_level::<Self>(inputs)
            }

            fn log_basis_search_range(
                _inputs: $crate::protocol::commitment::HachiScheduleInputs,
            ) -> (u32, u32) {
                $crate::protocol::config::proof_optimized::
                    proof_optimized_log_basis_search_range()
            }

            fn schedule_key(
                key: $crate::protocol::commitment::HachiScheduleLookupKey,
            ) -> String {
                $crate::protocol::config::proof_optimized::
                    proof_optimized_schedule_key::<Self>(key)
            }

            fn schedule_plan(
                key: $crate::protocol::commitment::HachiScheduleLookupKey,
            ) -> Result<
                Option<$crate::protocol::commitment::HachiSchedulePlan>,
                akita_field::HachiError,
            > {
                $crate::protocol::config::proof_optimized::
                    proof_optimized_schedule_plan::<Self>(key)
            }
        }
    };
}
pub(crate) use impl_fp128_preset;

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
}
