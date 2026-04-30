//! Concrete commitment configs for the default fp128 protocol.
//!
//! Each config is a plain unit struct with a direct [`CommitmentConfig`]
//! impl. Per-level params, log-basis selection, schedule lookup, and root
//! parameterization are routed through the shared planner-backed helpers in
//! [`super::adaptive`]; concrete configs only provide their decomposition
//! and challenge-family choices.

use super::adaptive::adaptive_envelope;
use super::config::{CommitmentConfig, CommitmentEnvelope, DecompositionParams};
use super::generated::GeneratedScheduleTable;
use crate::algebra::{Prime128Offset2355, SparseChallengeConfig};

// ---------------------------------------------------------------------------
// Internal shared helpers (referenced by `impl_fp128_preset!`)
// ---------------------------------------------------------------------------

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

const FP128_D128_AUDITED_ROOT_RANK2_FROM_NV: usize = 54;
const FP128_D128_AUDITED_ROOT_A_RANK2_FROM_NV: usize = 59;

/// Audited root B/D rank policy for the fp128 family.
fn fp128_audited_root_outer_rank(d: usize, level: usize, max_num_vars: usize) -> usize {
    if d == 128 && level == 0 && max_num_vars >= FP128_D128_AUDITED_ROOT_RANK2_FROM_NV {
        2
    } else {
        1
    }
}

/// Audited root A rank policy for the fp128 family, parameterized by
/// `LOG_COMMIT_BOUND`.
fn fp128_audited_root_a_rank(
    d: usize,
    level: usize,
    max_num_vars: usize,
    log_commit_bound: u32,
) -> usize {
    if d == 128
        && log_commit_bound != 1
        && level == 0
        && max_num_vars >= FP128_D128_AUDITED_ROOT_A_RANK2_FROM_NV
    {
        2
    } else {
        1
    }
}

/// `D=64` onehot root rank escalation policy.
fn fp128_onehot_d64_root_rank(level: usize, max_num_vars: usize) -> usize {
    usize::from(max_num_vars >= 38 && level == 0) + 1
}

/// Per-config envelope helper used by `impl_fp128_preset!`.
pub(crate) fn fp128_envelope<Cfg: CommitmentConfig>(
    log_commit_bound: u32,
    table: impl Fn() -> GeneratedScheduleTable,
    max_num_vars: usize,
) -> CommitmentEnvelope {
    let d = Cfg::D;
    let audited_root_rank = if d == 64 && log_commit_bound == 1 {
        fp128_onehot_d64_root_rank(0, max_num_vars)
    } else {
        fp128_audited_root_outer_rank(d, 0, max_num_vars)
    };
    let audited_root_a_rank = if d == 64 && log_commit_bound == 1 {
        1
    } else {
        fp128_audited_root_a_rank(d, 0, max_num_vars, log_commit_bound)
    };
    adaptive_envelope::<Cfg>(table, audited_root_a_rank, audited_root_rank, max_num_vars)
}

// ---------------------------------------------------------------------------
// Per-preset CommitmentConfig macro
// ---------------------------------------------------------------------------

/// Generate a complete [`CommitmentConfig`] impl for one fp128 preset.
///
/// Each preset only ships its `(D, LOG_COMMIT_BOUND)` decomposition and the
/// generated schedule table; everything else delegates to the planner-backed
/// helpers in [`super::adaptive`].
macro_rules! impl_fp128_preset {
    ($cfg:ident, $d:expr, $log_commit_bound:expr, $table:ident) => {
        impl $crate::protocol::commitment::config::CommitmentConfig for $cfg {
            type Field = Field;
            const D: usize = $d;

            fn decomposition() -> $crate::protocol::commitment::config::DecompositionParams {
                $crate::protocol::commitment::presets::fp128_decomposition($log_commit_bound, 3)
            }

            fn stage1_challenge_config(d: usize) -> $crate::algebra::SparseChallengeConfig {
                $crate::protocol::commitment::presets::fp128_stage1_challenge_config(d)
            }

            fn envelope(
                max_num_vars: usize,
            ) -> $crate::protocol::commitment::config::CommitmentEnvelope {
                $crate::protocol::commitment::presets::fp128_envelope::<Self>(
                    $log_commit_bound,
                    || $crate::protocol::commitment::generated::$table(),
                    max_num_vars,
                )
            }

            fn max_setup_matrix_size(
                max_num_vars: usize,
                max_num_batched_polys: usize,
                max_num_points: usize,
            ) -> Result<(usize, usize), $crate::error::HachiError> {
                $crate::protocol::commitment::adaptive::adaptive_max_setup_matrix_size::<Self, { $d }>(
                    max_num_vars,
                    max_num_batched_polys,
                    max_num_points,
                )
            }

            fn log_basis_at_level(
                inputs: $crate::protocol::commitment::HachiScheduleInputs,
            ) -> u32 {
                $crate::protocol::commitment::adaptive::adaptive_log_basis_at_level::<Self>(
                    || $crate::protocol::commitment::generated::$table(),
                    inputs,
                )
            }

            fn log_basis_search_range(
                _inputs: $crate::protocol::commitment::HachiScheduleInputs,
            ) -> (u32, u32) {
                $crate::protocol::commitment::adaptive::adaptive_log_basis_search_range()
            }

            fn schedule_key(
                key: $crate::protocol::commitment::HachiScheduleLookupKey,
            ) -> String {
                $crate::protocol::commitment::adaptive::adaptive_schedule_key::<Self>(
                    || $crate::protocol::commitment::generated::$table(),
                    key,
                )
            }

            fn schedule_plan(
                key: $crate::protocol::commitment::HachiScheduleLookupKey,
            ) -> Result<
                Option<$crate::protocol::commitment::HachiSchedulePlan>,
                $crate::error::HachiError,
            > {
                $crate::protocol::commitment::adaptive::adaptive_schedule_plan::<Self>(
                    || $crate::protocol::commitment::generated::$table(),
                    key,
                )
            }

            fn root_level_params_for_layout_with_log_basis(
                inputs: $crate::protocol::commitment::HachiScheduleInputs,
                lp: &$crate::protocol::params::LevelParams,
            ) -> Result<$crate::protocol::params::LevelParams, $crate::error::HachiError> {
                $crate::protocol::commitment::adaptive::adaptive_root_level_params_for_layout_with_log_basis::<Self>(inputs, lp)
            }

            fn root_level_layout_with_log_basis(
                inputs: $crate::protocol::commitment::HachiScheduleInputs,
                log_basis: u32,
            ) -> Result<$crate::protocol::params::LevelParams, $crate::error::HachiError> {
                $crate::protocol::commitment::adaptive::adaptive_root_level_layout_with_log_basis::<Self>(
                    inputs, log_basis,
                )
            }

            fn level_params_with_log_basis(
                inputs: $crate::protocol::commitment::HachiScheduleInputs,
                log_basis: u32,
            ) -> $crate::protocol::params::LevelParams {
                $crate::protocol::commitment::adaptive::adaptive_level_params_with_log_basis::<Self>(
                    || $crate::protocol::commitment::generated::$table(),
                    inputs,
                    log_basis,
                )
            }
        }
    };
}
pub(crate) use impl_fp128_preset;

// ---------------------------------------------------------------------------
// Public preset structs
// ---------------------------------------------------------------------------

/// Default fp128 protocol presets on `p = 2^128 - 2355`.
pub mod fp128 {
    use super::*;

    /// Base field for the default fp128 presets.
    pub type Field = Prime128Offset2355;

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
