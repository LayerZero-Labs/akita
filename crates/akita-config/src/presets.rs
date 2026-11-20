//! Built-in commitment config presets.
//!
//! Presets are unit structs that bind [`CommitmentConfig`](crate::CommitmentConfig)
//! hooks to [`akita_types`] SIS primitives and external schedule families.

use akita_error::AkitaError;
use jolt_field::{Ext2, FpExt4, Prime128Offset6A23801F, Prime32Offset99, Prime64Offset59};

/// Minimum opening log basis used by the presets.
///
/// This is also the fixed **root-fold** basis: `log_basis_search_range_at_level(0)`
/// collapses the root to `opening_basis_range.0`. Pinning the root to `3` (rather than the
/// smallest reachable `2`) keeps the shrink strong enough that every preset — dense
/// and small-field included — supports the full `nv` range, and matches the value
/// the unpinned planner already favored at the root.
pub(crate) const OPENING_LOG_BASIS_MIN: u32 = 3;
/// Maximum opening log basis used by the presets.
pub(crate) const OPENING_LOG_BASIS_MAX: u32 = 6;
/// Maximum A/source log basis for the large-field presets.
///
/// The signed-i16 commitment path supports values through 16. Large-field
/// presets search that complete implementation domain; q32 keeps its tighter
/// field-specific cap below.
pub(crate) const INNER_LOG_BASIS_MAX: u32 = 16;

const fn inner_basis_range(profile: akita_params::SisModulusProfileId) -> (u32, u32) {
    let max = match profile {
        akita_params::SisModulusProfileId::Q32Offset99 => 10,
        akita_params::SisModulusProfileId::Q64Offset59
        | akita_params::SisModulusProfileId::Q128Offset6A23801F => INNER_LOG_BASIS_MAX,
    };
    (OPENING_LOG_BASIS_MIN, max)
}
/// Explicit sparse-binary chunk size used by standard one-hot presets.
///
/// This is an offline sizing-policy input, not runtime group geometry. Akita's
/// built-in external schedule artifacts use K=16; downstream configurations
/// may generate artifacts from another policy-owned chunk size.
pub const STANDARD_ONEHOT_CHUNK_SIZE: usize =
    akita_params::sis::DEFAULT_UNIT_ONEHOT_SOURCE_CHUNK_SIZE;

/// Shared short ring-challenge policy for the built-in presets.
///
/// Fixed-weight sparse families keyed on ring degree `d` via
/// [`akita_challenges::SparseChallengeConfig::production_for_ring_dim`].
/// Offline planning and artifact admission call this hook with each
/// schedule-selected A dimension. The flat public matrix has no generation
/// dimension.
pub(crate) fn ring_challenge_config(
    d: usize,
) -> Result<akita_challenges::SparseChallengeConfig, AkitaError> {
    let cfg =
        akita_challenges::SparseChallengeConfig::production_for_ring_dim(d).ok_or_else(|| {
            AkitaError::InvalidSetup(format!("unsupported preset ring dimension {d}"))
        })?;
    cfg.validate_for_ring_dim(d)
        .map_err(|msg| AkitaError::InvalidSetup(msg.to_string()))?;
    Ok(cfg)
}

// ---------------------------------------------------------------------------
// Per-preset CommitmentConfig macro
// ---------------------------------------------------------------------------

/// Implement [`CommitmentConfig`](crate::CommitmentConfig) for each built-in preset.
/// The field family supplies the base and extension fields, the field width,
/// and the SIS modulus profile. All presets share the ring-challenge policy
/// and opening basis range.
macro_rules! impl_preset {
    (@ring_dimension_schedule $mode:expr) => {
        const RING_DIMENSION_SCHEDULE: akita_schedules::RingDimensionSchedule = $mode;
    };
    (@committed_source_class unit_one_hot) => {
        fn committed_source_class() -> akita_params::sis::CommittedSourceClass {
            akita_params::sis::CommittedSourceClass::UnitOneHot {
                source_chunk_size: STANDARD_ONEHOT_CHUNK_SIZE,
            }
        }
    };
    (@committed_source_class balanced_digits) => {
        fn committed_source_class() -> akita_params::sis::CommittedSourceClass {
            akita_params::sis::CommittedSourceClass::BalancedSignedDigit
        }
    };
    ($cfg:ident, $field:ty, $ext_field:ty, $family:expr, $field_bits:expr, $log_commit_bound:expr, source = $source:ident, schedule_family = $family_name:literal, ring_dimension_schedule = $mode:expr) => {
        impl_preset!(@core $cfg, $field, $ext_field, $family, $field_bits, $log_commit_bound, $source, $family_name, ring_dimension_schedule = $mode);
    };
    (@options ring_dimension_schedule = $mode:expr) => {
        impl_preset!(@ring_dimension_schedule $mode);
    };
    (@core $cfg:ident, $field:ty, $ext_field:ty, $family:expr, $field_bits:expr, $log_commit_bound:expr, $source:ident, $family_name:literal, $($options:tt)*) => {
        impl $crate::CommitmentConfig for $cfg {
            type Field = $field;
            type ExtField = $ext_field;
            fn schedule_family_name() -> &'static str {
                $family_name
            }
            impl_preset!(@options $($options)*);

            fn decomposition() -> akita_params::DecompositionParams {
                akita_params::DecompositionParams {
                    log_basis: 3,
                    log_commit_bound: $log_commit_bound,
                    log_open_bound: if $log_commit_bound < $field_bits {
                        Some($field_bits)
                    } else {
                        None
                    },
                }
            }

            fn ring_challenge_config(
                d: usize,
            ) -> Result<akita_challenges::SparseChallengeConfig, akita_error::AkitaError> {
                $crate::presets::ring_challenge_config(d)
            }

            fn sis_modulus_profile() -> akita_params::SisModulusProfileId {
                $family
            }


            fn opening_basis_range() -> (u32, u32) {
                (
                    $crate::presets::OPENING_LOG_BASIS_MIN,
                    $crate::presets::OPENING_LOG_BASIS_MAX,
                )
            }

            fn inner_basis_range() -> (u32, u32) {
                $crate::presets::inner_basis_range(
                    Self::sis_modulus_profile(),
                )
            }

            impl_preset!(@committed_source_class $source);

        }

    };
}

// ---------------------------------------------------------------------------
// Public preset structs
// ---------------------------------------------------------------------------

pub mod fp128;
pub mod fp32;
pub mod fp64;
