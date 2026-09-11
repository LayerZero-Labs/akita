//! Proof-optimized commitment config presets.
//!
//! Presets are unit structs that bind [`CommitmentConfig`] hooks to
//! [`akita_types`] SIS primitives and external schedule families.

use super::CommitmentConfig;
use akita_error::AkitaError;
use akita_types::{
    setup_matrix_field_elements_for_schedule, verifier_setup_matrix_capacity_for_schedule,
    AkitaExpandedSetup, CommittedGroupParams, FoldSchedule, OpeningClaimsLayout,
};
use jolt_field::{Ext2, FpExt4, Prime128OffsetA7F7, Prime32Offset99, Prime64Offset59};

/// Minimum proof-optimized log-basis.
///
/// This is also the fixed **root-fold** basis: `log_basis_search_range_at_level(0)`
/// collapses the root to `opening_basis_range.0`. Pinning the root to `3` (rather than the
/// smallest reachable `2`) keeps the shrink strong enough that every preset — dense
/// and small-field included — supports the full `nv` range, and matches the value
/// the unpinned planner already favored at the root.
pub(crate) const PROOF_OPTIMIZED_LOG_BASIS_MIN: u32 = 3;
/// Maximum proof-optimized log-basis.
pub(crate) const PROOF_OPTIMIZED_LOG_BASIS_MAX: u32 = 6;
/// Maximum A/source log basis searched by proof-optimized presets.
///
/// The signed-i16 commitment path supports values through 16. Large-field
/// presets search that complete implementation domain; q32 keeps its tighter
/// field-specific cap below.
pub(crate) const PROOF_OPTIMIZED_INNER_LOG_BASIS_MAX: u32 = 16;

const fn proof_optimized_inner_basis_range(
    profile: akita_types::SisModulusProfileId,
) -> (u32, u32) {
    let max = match profile {
        akita_types::SisModulusProfileId::Q32Offset99 => 10,
        akita_types::SisModulusProfileId::Q64Offset59
        | akita_types::SisModulusProfileId::Q128OffsetA7F7 => PROOF_OPTIMIZED_INNER_LOG_BASIS_MAX,
    };
    (PROOF_OPTIMIZED_LOG_BASIS_MIN, max)
}
/// Explicit sparse-binary chunk size used by standard one-hot presets.
///
/// This is an offline sizing-policy input, not runtime group geometry. Akita's
/// built-in external schedule artifacts use K=256; downstream configurations
/// may generate artifacts from another policy-owned chunk size.
pub const STANDARD_ONEHOT_CHUNK_SIZE: usize =
    akita_types::sis::DEFAULT_UNIT_ONEHOT_SOURCE_CHUNK_SIZE;

/// Shared short ring-challenge policy for every proof-optimized preset.
///
/// Fixed-weight sparse families keyed on ring degree `d` via
/// [`akita_challenges::SparseChallengeConfig::production_for_ring_dim`].
/// Offline planning and artifact admission call this hook with each
/// schedule-selected A dimension. The flat public matrix has no generation
/// dimension.
pub(crate) fn proof_optimized_ring_challenge_config(
    d: usize,
) -> Result<akita_challenges::SparseChallengeConfig, AkitaError> {
    let cfg =
        akita_challenges::SparseChallengeConfig::production_for_ring_dim(d).ok_or_else(|| {
            AkitaError::InvalidSetup(format!("unsupported proof-optimized ring dim {d}"))
        })?;
    cfg.validate_for_ring_dim(d)
        .map_err(|msg| AkitaError::InvalidSetup(msg.to_string()))?;
    Ok(cfg)
}

/// Extract setup-level params from a `FoldSchedule`.
///
pub fn setup_level_params_from_schedule(schedule: &FoldSchedule) -> Vec<CommittedGroupParams> {
    std::iter::once(schedule.root.params.clone())
        .chain(
            schedule
                .recursive_folds
                .iter()
                .map(|fold| fold.params.clone()),
        )
        .collect()
}

/// Reject a concrete schedule whose exact matrix footprint exceeds setup.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidSetup`] when sizing overflows or the setup's
/// materialized shared matrix is too short for `schedule` and `layout`.
pub fn ensure_prover_schedule_fits_setup<Cfg>(
    setup: &AkitaExpandedSetup<Cfg::Field>,
    schedule: &FoldSchedule,
    layout: &OpeningClaimsLayout,
) -> Result<(), AkitaError>
where
    Cfg: CommitmentConfig,
{
    // `setup_matrix_field_elements_for_schedule` already maxes over the root
    // level's A/B/D matrices, every frozen precommitted group, the compression maps,
    // and the fold tail, so it dominates any per-level recomputation here.
    schedule.root.params.validate_opening_batch(layout)?;
    ensure_required_setup_field_elements(
        setup_matrix_field_elements_for_schedule(schedule)?,
        setup.shared_matrix.as_field_slice().len(),
    )
}

/// Reject a concrete schedule whose direct verifier matrix uses exceed setup.
///
/// Offloaded producer edges are covered by verifier-visible setup-prefix
/// commitments and do not require their full committed source prefixes here.
pub fn ensure_verifier_schedule_fits_setup(
    setup: &AkitaExpandedSetup<impl jolt_field::Field>,
    schedule: &FoldSchedule,
    layout: &OpeningClaimsLayout,
) -> Result<(), AkitaError> {
    let required = verifier_setup_matrix_capacity_for_schedule(schedule, layout)?;
    ensure_required_setup_field_elements(
        required.num_field_elements,
        setup.shared_matrix.as_field_slice().len(),
    )
}

fn ensure_required_setup_field_elements(
    required_field_elements: usize,
    available_field_elements: usize,
) -> Result<(), AkitaError> {
    if required_field_elements <= available_field_elements {
        return Ok(());
    }
    Err(AkitaError::InvalidSetup(format!(
        "schedule requires {required_field_elements} physical setup field elements, but setup \
         provides {available_field_elements}"
    )))
}

// ---------------------------------------------------------------------------
// Per-preset CommitmentConfig macro
// ---------------------------------------------------------------------------

/// Generate a [`CommitmentConfig`] impl for one proof-optimized preset.
///
/// One macro covers every proof-optimized preset (fp128 and the small-field
/// fp32/fp64 families): the fp128 presets are the special case where the
/// extension field is the base field, `field_bits == 128`, and the SIS
/// family is `Q128`. All proof-optimized presets share `log_basis = 3`, the
/// shared ring-challenge policy, the shared setup-matrix sizer, and the
/// `[PROOF_OPTIMIZED_LOG_BASIS_MIN, MAX]` basis range, so those are not
/// parameters.
macro_rules! impl_proof_optimized_preset {
    (@ring_dimension_schedule_mode $mode:expr) => {
        const RING_DIMENSION_SCHEDULE_MODE: akita_schedules::RingDimensionScheduleMode = $mode;
    };
    (@committed_source_class unit_one_hot) => {
        fn committed_source_class() -> akita_types::sis::CommittedSourceClass {
            akita_types::sis::CommittedSourceClass::UnitOneHot {
                source_chunk_size: STANDARD_ONEHOT_CHUNK_SIZE,
            }
        }
    };
    (@committed_source_class balanced_digits) => {
        fn committed_source_class() -> akita_types::sis::CommittedSourceClass {
            akita_types::sis::CommittedSourceClass::BalancedSignedDigit
        }
    };
    ($cfg:ident, $field:ty, $ext_field:ty, $family:expr, $field_bits:expr, $log_commit_bound:expr, source = $source:ident, schedule_family = $family_name:literal, ring_dimension_schedule_mode = $mode:expr) => {
        impl_proof_optimized_preset!(@core $cfg, $field, $ext_field, $family, $field_bits, $log_commit_bound, $source, $family_name, ring_dimension_schedule_mode = $mode);
    };
    (@options ring_dimension_schedule_mode = $mode:expr) => {
        impl_proof_optimized_preset!(@ring_dimension_schedule_mode $mode);
    };
    (@core $cfg:ident, $field:ty, $ext_field:ty, $family:expr, $field_bits:expr, $log_commit_bound:expr, $source:ident, $family_name:literal, $($options:tt)*) => {
        impl $crate::CommitmentConfig for $cfg {
            type Field = $field;
            type ExtField = $ext_field;
            fn schedule_family_name() -> &'static str {
                $family_name
            }
            impl_proof_optimized_preset!(@options $($options)*);

            fn decomposition() -> akita_types::DecompositionParams {
                akita_types::DecompositionParams {
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
                $crate::proof_optimized::proof_optimized_ring_challenge_config(d)
            }

            fn sis_modulus_profile() -> akita_types::SisModulusProfileId {
                $family
            }


            fn opening_basis_range() -> (u32, u32) {
                (
                    $crate::proof_optimized::PROOF_OPTIMIZED_LOG_BASIS_MIN,
                    $crate::proof_optimized::PROOF_OPTIMIZED_LOG_BASIS_MAX,
                )
            }

            fn inner_basis_range() -> (u32, u32) {
                $crate::proof_optimized::proof_optimized_inner_basis_range(
                    Self::sis_modulus_profile(),
                )
            }

            impl_proof_optimized_preset!(@committed_source_class $source);

        }

    };
}

// ---------------------------------------------------------------------------
// Public preset structs
// ---------------------------------------------------------------------------

pub mod fp128;
pub mod fp32;
pub mod fp64;
