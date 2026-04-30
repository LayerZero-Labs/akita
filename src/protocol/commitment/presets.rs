//! Concrete commitment configs for the default fp128 protocol.
//!
//! Each config is a plain unit struct. Per-level params, log-basis
//! selection, schedule lookup, and root parameterization all flow through
//! the default trait method bodies on [`CommitmentConfig`], which call into
//! the internal `adaptive` module. A preset only declares its
//! `(D, LOG_COMMIT_BOUND)` decomposition, its sparse stage-1 family, the
//! generated schedule table that backs it, and (when applicable) the
//! audited root-rank floor.

use super::config::{AjtaiRole, CommitmentConfig, DecompositionParams};
use crate::algebra::{Prime128OffsetA7F7, SparseChallengeConfig};

// ---------------------------------------------------------------------------
// Internal shared helpers
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
// Per-preset CommitmentConfig macro
// ---------------------------------------------------------------------------

/// Generate a complete [`CommitmentConfig`] impl for one fp128 preset.
///
/// Each preset only ships its `(D, LOG_COMMIT_BOUND)` decomposition and the
/// generated schedule table; everything else flows through the default trait
/// method bodies.
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

            #[allow(private_interfaces)]
            fn schedule_table(
            ) -> Option<$crate::protocol::commitment::generated::GeneratedScheduleTable> {
                Some($crate::protocol::commitment::generated::$table())
            }

            fn audited_root_rank(
                role: $crate::protocol::commitment::config::AjtaiRole,
                max_num_vars: usize,
            ) -> usize {
                $crate::protocol::commitment::presets::fp128_audited_root_rank::<Self>(
                    role,
                    max_num_vars,
                )
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
