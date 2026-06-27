//! Proof-optimized commitment config presets.
//!
//! Presets are unit structs that bind [`CommitmentConfig`] hooks to
//! [`akita_types`] SIS primitives and generated schedule tables.

use super::CommitmentConfig;
use akita_challenges::MIN_FOLD_CHALLENGE_ENTROPY_BITS;
use akita_field::AkitaError;
use akita_field::{Ext2, FpExt4, Prime128OffsetA7F7, Prime32Offset99, Prime64Offset59};
use akita_types::OpeningBatchShape;
use akita_types::{AkitaScheduleLookupKey, LevelParams, Schedule, SetupMatrixEnvelope};
use std::any::TypeId;
use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

/// Minimum proof-optimized log-basis.
pub(crate) const PROOF_OPTIMIZED_LOG_BASIS_MIN: u32 = 2;
/// Maximum proof-optimized log-basis.
pub(crate) const PROOF_OPTIMIZED_LOG_BASIS_MAX: u32 = 6;

/// Shared short ring-challenge policy for every proof-optimized preset.
///
/// "Short" means bounded norm, which is the property that keeps the folded
/// witness short enough for SIS binding. It does not mean sparse: at `d == 32`
/// the family is `BoundedL1Norm`, a low-norm ball (`||c||_1 <= 121`,
/// `||c||_inf <= 8`) whose elements can be fully dense. The larger degrees use
/// fixed-weight sparse families (`ExactShell` at `d == 64`, `Uniform` at
/// `d >= 128`), where shortness happens to coincide with sparsity.
///
/// The family is keyed only on the ring degree `d`. A preset's `D` is fixed
/// across all schedule levels, so both the planner DP and the generated-table
/// expansion call the per-`Cfg` hook with `d == Cfg::D` (see
/// `akita_planner::find_schedule` and `generated::expand`). Every family
/// returned here has at least 128 bits of Fiat-Shamir support, which is the
/// soundness floor for the witness-folding ring challenge; presets must not
/// pick a lower-support family. fp128 only reaches `d in {32, 64, 128}`; the
/// small-field presets additionally reach `d == 256`.
pub(crate) fn proof_optimized_ring_challenge_config(
    d: usize,
) -> Result<akita_challenges::SparseChallengeConfig, AkitaError> {
    let cfg = match d {
        32 => akita_challenges::SparseChallengeConfig::BoundedL1Norm,
        64 => akita_challenges::SparseChallengeConfig::ExactShell {
            count_mag1: akita_challenges::D64_PRODUCTION_EXACT_SHELL_MAG1,
            count_mag2: akita_challenges::D64_PRODUCTION_EXACT_SHELL_MAG2,
            operator_norm_threshold: akita_challenges::D64_PRODUCTION_OPERATOR_NORM_THRESHOLD,
        },
        128 => akita_challenges::SparseChallengeConfig::Uniform {
            weight: 31,
            nonzero_coeffs: vec![-1, 1],
        },
        256 => akita_challenges::SparseChallengeConfig::Uniform {
            weight: 23,
            nonzero_coeffs: vec![-1, 1],
        },
        _ => {
            return Err(AkitaError::InvalidSetup(format!(
                "unsupported proof-optimized ring dim {d}"
            )));
        }
    };
    validate_proof_optimized_fold_entropy(&cfg, d)?;
    Ok(cfg)
}

fn validate_proof_optimized_fold_entropy(
    cfg: &akita_challenges::SparseChallengeConfig,
    d: usize,
) -> Result<(), AkitaError> {
    match d {
        32 => cfg.validate::<32>(),
        64 => cfg.validate::<64>(),
        128 => cfg.validate::<128>(),
        256 => cfg.validate::<256>(),
        _ => {
            return Err(AkitaError::InvalidSetup(format!(
                "unsupported proof-optimized ring dim {d}"
            )));
        }
    }
    .map_err(|msg| AkitaError::InvalidSetup(msg.to_string()))?;
    cfg.validate_min_entropy_for_ring_dim(d, MIN_FOLD_CHALLENGE_ENTROPY_BITS)
        .map_err(|msg| AkitaError::InvalidSetup(msg.to_string()))
}

// ---------------------------------------------------------------------------
// `<Cfg>`-generic policy helpers for the planner and materializer.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Trait-shaped wrappers consumed by the macros below.
// ---------------------------------------------------------------------------

/// Size the shared setup matrix from the planned schedule.
///
/// Planned role footprints are not monotone across shapes, so scan all
/// supported sub-shapes and keep the largest packed setup length.
type SetupMatrixEnvelopeCache =
    LazyLock<Mutex<HashMap<(TypeId, usize, usize), SetupMatrixEnvelope>>>;

static SETUP_MATRIX_ENVELOPE_CACHE: SetupMatrixEnvelopeCache =
    LazyLock::new(|| Mutex::new(HashMap::new()));

pub(crate) fn proof_optimized_max_setup_matrix_size<Cfg: CommitmentConfig>(
    max_num_vars: usize,
    max_num_batched_polys: usize,
) -> Result<SetupMatrixEnvelope, AkitaError> {
    let cache_key = (TypeId::of::<Cfg>(), max_num_vars, max_num_batched_polys);
    if let Some(cached) = SETUP_MATRIX_ENVELOPE_CACHE
        .lock()
        .expect("setup matrix envelope cache poisoned")
        .get(&cache_key)
        .copied()
    {
        return Ok(cached);
    }

    let envelope =
        proof_optimized_max_setup_matrix_size_uncached::<Cfg>(max_num_vars, max_num_batched_polys)?;

    SETUP_MATRIX_ENVELOPE_CACHE
        .lock()
        .expect("setup matrix envelope cache poisoned")
        .insert(cache_key, envelope);

    Ok(envelope)
}

fn proof_optimized_max_setup_matrix_size_uncached<Cfg: CommitmentConfig>(
    max_num_vars: usize,
    max_num_batched_polys: usize,
) -> Result<SetupMatrixEnvelope, AkitaError> {
    if max_num_batched_polys == 0 {
        return Err(AkitaError::InvalidSetup(
            "max_num_batched_polys must be at least 1".to_string(),
        ));
    }
    let mut max_setup_len: usize = 1;
    let mut saw_supported_shape = false;
    let poly_counts = setup_envelope_poly_counts(max_num_batched_polys);
    for num_vars in 1..=max_num_vars {
        for &num_polys in &poly_counts {
            let opening_batch = worst_case_grouped_opening_batch_for_shape(num_vars, num_polys)?;
            let Some(envelope) = setup_matrix_envelope_for_shape::<Cfg>(&opening_batch)? else {
                continue;
            };
            saw_supported_shape = true;
            max_setup_len = max_setup_len.max(envelope.max_setup_len);
        }
    }

    if !saw_supported_shape {
        return Err(AkitaError::InvalidSetup(format!(
            "setup matrix sizing found no generated schedules for max_num_vars={max_num_vars}"
        )));
    }

    Ok(SetupMatrixEnvelope { max_setup_len })
}

/// Batched polynomial counts scanned by [`proof_optimized_max_setup_matrix_size`].
///
/// Generated schedule tables (and the offline `gen_schedule_tables` emitter)
/// materialize only singleton (`num_polys = 1`) and 4-batched roots. Scanning
/// every intermediate count in `1..=max` forces table misses on `2` and `3` even
/// though setup-matrix footprints are determined by the endpoint batch sizes.
/// Role footprints can be non-monotone in `num_vars`, but not in these skipped
/// intermediate batch counts for the shipped table key shapes.
pub(crate) fn setup_envelope_poly_counts(max_num_batched_polys: usize) -> Vec<usize> {
    if max_num_batched_polys <= 1 {
        vec![1]
    } else {
        vec![1, max_num_batched_polys]
    }
}

/// Worst-case opening batch for a `(num_vars, num_polynomials)` shape.
pub fn worst_case_grouped_opening_batch_for_shape(
    num_vars: usize,
    num_polynomials: usize,
) -> Result<OpeningBatchShape, AkitaError> {
    OpeningBatchShape::new(num_vars, num_polynomials)
}

fn setup_matrix_envelope_for_shape<Cfg: CommitmentConfig>(
    opening_batch: &OpeningBatchShape,
) -> Result<Option<SetupMatrixEnvelope>, AkitaError> {
    let cached_key = AkitaScheduleLookupKey::new_from_opening_batch(opening_batch)?;

    // Setup-matrix sizing scans many candidate sub-shapes. `runtime_schedule`
    // serves the shipped table on a hit and regenerates via the planner DP on
    // a miss; a shape the planner cannot schedule (infeasible — e.g. a witness
    // too large for this preset's SIS floor) can never be committed, so it
    // needs no setup capacity. Skip it (returning `Ok(None)`) and let the
    // caller's `saw_supported_shape` guard error only if *no* shape is
    // feasible. Genuine bugs in opening_batch-key or envelope construction still
    // propagate via `?`.
    let Ok(schedule) = Cfg::runtime_schedule(cached_key) else {
        return Ok(None);
    };

    Ok(Some(matrix_envelope_for_schedule::<Cfg>(
        &schedule,
        opening_batch,
    )?))
}

/// Extract setup-level params from a runtime `Schedule`.
///
/// Uncommittable root-direct entries carry no setup params and are skipped
/// here; `Cfg::get_params_for_batched_commitment` rejects them loudly.
pub fn setup_level_params_from_runtime_schedule(steps: &[akita_types::Step]) -> Vec<LevelParams> {
    steps
        .iter()
        .filter_map(|step| match step {
            akita_types::Step::Fold(fold_step) => Some(fold_step.params.clone()),
            akita_types::Step::Direct(direct) => direct.params.clone(),
        })
        .collect()
}

fn matrix_envelope_for_levels<Cfg>(
    setup_levels: &[LevelParams],
) -> Result<SetupMatrixEnvelope, AkitaError>
where
    Cfg: CommitmentConfig,
{
    let mut max_setup_len: usize = 1;
    for lp in setup_levels {
        accumulate_matrix_envelope_for_level::<Cfg>(lp, &mut max_setup_len)?;
    }
    Ok(SetupMatrixEnvelope { max_setup_len })
}

/// Packed setup envelope spanning every level in `schedule`, including root
/// runtime widening for the requested opening batch.
pub fn matrix_envelope_for_schedule<Cfg>(
    schedule: &Schedule,
    opening_batch: &OpeningBatchShape,
) -> Result<SetupMatrixEnvelope, AkitaError>
where
    Cfg: CommitmentConfig,
{
    let setup_levels: Vec<LevelParams> = setup_level_params_from_runtime_schedule(&schedule.steps);
    let mut envelope = matrix_envelope_for_levels::<Cfg>(&setup_levels)?;
    accumulate_root_matrix_envelope_for_opening_batch(
        schedule,
        opening_batch,
        &mut envelope.max_setup_len,
    )?;
    Ok(envelope)
}

fn accumulate_matrix_envelope_for_level<Cfg: CommitmentConfig>(
    lp: &LevelParams,
    max_setup_len: &mut usize,
) -> Result<(), AkitaError> {
    let _cfg_marker = core::marker::PhantomData::<Cfg>;
    let a_len = lp
        .a_key
        .row_len()
        .checked_mul(lp.inner_width())
        .ok_or_else(|| AkitaError::InvalidSetup("A setup envelope overflow".to_string()))?;
    let b_len = lp
        .b_key
        .row_len()
        .checked_mul(lp.outer_width())
        .ok_or_else(|| AkitaError::InvalidSetup("B setup envelope overflow".to_string()))?;
    let d_len = lp
        .d_key
        .row_len()
        .checked_mul(lp.d_matrix_width())
        .ok_or_else(|| AkitaError::InvalidSetup("D setup envelope overflow".to_string()))?;
    let f_len = match lp.f_key.as_ref() {
        Some(fk) => fk
            .row_len()
            .checked_mul(fk.col_len())
            .ok_or_else(|| AkitaError::InvalidSetup("F setup envelope overflow".to_string()))?,
        None => 0,
    };
    *max_setup_len = (*max_setup_len).max(a_len).max(b_len).max(d_len).max(f_len);
    Ok(())
}

fn accumulate_root_matrix_envelope_for_opening_batch(
    schedule: &Schedule,
    opening_batch: &OpeningBatchShape,
    max_setup_len: &mut usize,
) -> Result<(), AkitaError> {
    let Some(root_params) = root_commit_params_from_schedule(schedule)? else {
        return Ok(());
    };
    let root_len = root_runtime_matrix_len_for_opening_batch(&root_params, opening_batch)?;
    *max_setup_len = (*max_setup_len).max(root_len);
    Ok(())
}

fn root_runtime_matrix_len_for_opening_batch(
    lp: &LevelParams,
    opening_batch: &OpeningBatchShape,
) -> Result<usize, AkitaError> {
    let num_claims = opening_batch.num_polynomials();
    let max_group_poly_count = opening_batch.num_polynomials();
    let d_width = lp
        .num_blocks
        .checked_mul(num_claims)
        .and_then(|n| n.checked_mul(lp.num_digits_open))
        .ok_or_else(|| AkitaError::InvalidSetup("batched D setup width overflow".to_string()))?;
    let t_cols_per_vector = lp
        .a_key
        .row_len()
        .checked_mul(lp.num_digits_open)
        .and_then(|n| n.checked_mul(lp.num_blocks))
        .ok_or_else(|| {
            AkitaError::InvalidSetup("batched B setup vector width overflow".to_string())
        })?;
    let full_b_width = max_group_poly_count
        .checked_mul(t_cols_per_vector)
        .ok_or_else(|| AkitaError::InvalidSetup("batched B setup width overflow".to_string()))?;
    let d_len =
        lp.d_key.row_len().checked_mul(d_width).ok_or_else(|| {
            AkitaError::InvalidSetup("batched D setup envelope overflow".to_string())
        })?;
    let b_len = lp
        .b_key
        .row_len()
        .checked_mul(full_b_width.div_ceil(lp.tier_split.max(1)))
        .ok_or_else(|| AkitaError::InvalidSetup("batched B setup envelope overflow".to_string()))?;
    let f_len = match lp.f_key.as_ref() {
        Some(fk) => fk.row_len().checked_mul(fk.col_len()).ok_or_else(|| {
            AkitaError::InvalidSetup("batched F setup envelope overflow".to_string())
        })?,
        None => 0,
    };
    Ok(b_len.max(d_len).max(f_len))
}

fn root_commit_params_from_schedule(
    schedule: &Schedule,
) -> Result<Option<LevelParams>, AkitaError> {
    match schedule.steps.first() {
        Some(akita_types::Step::Fold(root_step)) => Ok(Some(root_step.params.clone())),
        Some(akita_types::Step::Direct(direct)) => Ok(direct.params.clone()),
        None => Err(AkitaError::InvalidSetup(
            "schedule has no steps".to_string(),
        )),
    }
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
    (@onehot_chunk_size $onehot_chunk_size:expr) => {
        $onehot_chunk_size
    };
    (@onehot_chunk_size) => {
        1
    };
    (@tiered $tiered:expr) => {
        $tiered
    };
    (@tiered) => {
        false
    };
    (@schedule_catalog none) => {};
    (@schedule_catalog ($feat:literal, $family:literal, $table:ident)) => {
        fn schedule_catalog() -> Option<akita_planner::GeneratedScheduleTable> {
            #[cfg(feature = $feat)]
            {
                Some(akita_schedules::$table())
            }
            #[cfg(not(feature = $feat))]
            {
                None
            }
        }
    };
    (@schedule_catalog tiered ($feat:literal, $family:literal, $table:ident)) => {
        fn schedule_catalog() -> Option<akita_planner::GeneratedScheduleTable> {
            #[cfg(feature = $feat)]
            {
                Some(akita_schedules::$table())
            }
            #[cfg(not(feature = $feat))]
            {
                None
            }
        }
    };
    ($cfg:ident, $field:ty, $ext_field:ty, $family:expr, $d:expr, $field_bits:expr, $log_commit_bound:expr) => {
        impl_proof_optimized_preset!(@core $cfg, $field, $ext_field, $family, $d, $field_bits, $log_commit_bound, 1, false, none);
    };
    ($cfg:ident, $field:ty, $ext_field:ty, $family:expr, $d:expr, $field_bits:expr, $log_commit_bound:expr, schedules = ($feat:literal, $family_name:literal, $table:ident)) => {
        impl_proof_optimized_preset!(@core $cfg, $field, $ext_field, $family, $d, $field_bits, $log_commit_bound, 1, false, table, $feat, $family_name, $table);
    };
    ($cfg:ident, $field:ty, $ext_field:ty, $family:expr, $d:expr, $field_bits:expr, $log_commit_bound:expr, $onehot_chunk_size:expr) => {
        impl_proof_optimized_preset!(@core $cfg, $field, $ext_field, $family, $d, $field_bits, $log_commit_bound, $onehot_chunk_size, false, none);
    };
    ($cfg:ident, $field:ty, $ext_field:ty, $family:expr, $d:expr, $field_bits:expr, $log_commit_bound:expr, $onehot_chunk_size:expr, schedules = ($feat:literal, $family_name:literal, $table:ident)) => {
        impl_proof_optimized_preset!(@core $cfg, $field, $ext_field, $family, $d, $field_bits, $log_commit_bound, $onehot_chunk_size, false, table, $feat, $family_name, $table);
    };
    ($cfg:ident, $field:ty, $ext_field:ty, $family:expr, $d:expr, $field_bits:expr, $log_commit_bound:expr, $onehot_chunk_size:expr, true) => {
        impl_proof_optimized_preset!(@core $cfg, $field, $ext_field, $family, $d, $field_bits, $log_commit_bound, $onehot_chunk_size, true, none);
    };
    ($cfg:ident, $field:ty, $ext_field:ty, $family:expr, $d:expr, $field_bits:expr, $log_commit_bound:expr, $onehot_chunk_size:expr, true, schedules = ($feat:literal, $family_name:literal, $table:ident)) => {
        impl_proof_optimized_preset!(@core $cfg, $field, $ext_field, $family, $d, $field_bits, $log_commit_bound, $onehot_chunk_size, true, tiered_sched $feat, $family_name, $table);
    };
    (@core $cfg:ident, $field:ty, $ext_field:ty, $family:expr, $d:expr, $field_bits:expr, $log_commit_bound:expr, $onehot_chunk:expr, $tiered:expr, tiered_sched $feat:literal, $family_name:literal, $table:ident) => {
        impl $crate::CommitmentConfig for $cfg {
            type Field = $field;
            type ExtField = $ext_field;
            const D: usize = $d;

            const TIERED_COMMITMENT: bool = $tiered;

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
            ) -> Result<akita_challenges::SparseChallengeConfig, akita_field::AkitaError> {
                $crate::proof_optimized::proof_optimized_ring_challenge_config(d)
            }

            fn sis_modulus_family() -> akita_types::SisModulusFamily {
                $family
            }

            fn max_setup_matrix_size(
                max_num_vars: usize,
                max_num_batched_polys: usize,
            ) -> Result<akita_types::SetupMatrixEnvelope, akita_field::AkitaError> {
                $crate::proof_optimized::proof_optimized_max_setup_matrix_size::<Self>(
                    max_num_vars,
                    max_num_batched_polys,
                )
            }

            fn basis_range() -> (u32, u32) {
                (
                    $crate::proof_optimized::PROOF_OPTIMIZED_LOG_BASIS_MIN,
                    $crate::proof_optimized::PROOF_OPTIMIZED_LOG_BASIS_MAX,
                )
            }

            fn onehot_chunk_size() -> usize {
                $onehot_chunk
            }

            impl_proof_optimized_preset!(@schedule_catalog tiered ($feat, $family_name, $table));
        }
    };
    (@core $cfg:ident, $field:ty, $ext_field:ty, $family:expr, $d:expr, $field_bits:expr, $log_commit_bound:expr, $onehot_chunk:expr, $tiered:expr, none) => {
        impl $crate::CommitmentConfig for $cfg {
            type Field = $field;
            type ExtField = $ext_field;
            const D: usize = $d;

            const TIERED_COMMITMENT: bool = $tiered;

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
            ) -> Result<akita_challenges::SparseChallengeConfig, akita_field::AkitaError> {
                $crate::proof_optimized::proof_optimized_ring_challenge_config(d)
            }

            fn sis_modulus_family() -> akita_types::SisModulusFamily {
                $family
            }

            fn max_setup_matrix_size(
                max_num_vars: usize,
                max_num_batched_polys: usize,
            ) -> Result<akita_types::SetupMatrixEnvelope, akita_field::AkitaError> {
                $crate::proof_optimized::proof_optimized_max_setup_matrix_size::<Self>(
                    max_num_vars,
                    max_num_batched_polys,
                )
            }

            fn basis_range() -> (u32, u32) {
                (
                    $crate::proof_optimized::PROOF_OPTIMIZED_LOG_BASIS_MIN,
                    $crate::proof_optimized::PROOF_OPTIMIZED_LOG_BASIS_MAX,
                )
            }

            fn onehot_chunk_size() -> usize {
                $onehot_chunk
            }

            impl_proof_optimized_preset!(@schedule_catalog none);
        }
    };
    (@core $cfg:ident, $field:ty, $ext_field:ty, $family:expr, $d:expr, $field_bits:expr, $log_commit_bound:expr, $onehot_chunk:expr, $tiered:expr, table, $feat:literal, $family_name:literal, $table:ident) => {
        impl $crate::CommitmentConfig for $cfg {
            type Field = $field;
            type ExtField = $ext_field;
            const D: usize = $d;

            const TIERED_COMMITMENT: bool = $tiered;

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
            ) -> Result<akita_challenges::SparseChallengeConfig, akita_field::AkitaError> {
                $crate::proof_optimized::proof_optimized_ring_challenge_config(d)
            }

            fn sis_modulus_family() -> akita_types::SisModulusFamily {
                $family
            }

            fn max_setup_matrix_size(
                max_num_vars: usize,
                max_num_batched_polys: usize,
            ) -> Result<akita_types::SetupMatrixEnvelope, akita_field::AkitaError> {
                $crate::proof_optimized::proof_optimized_max_setup_matrix_size::<Self>(
                    max_num_vars,
                    max_num_batched_polys,
                )
            }

            fn basis_range() -> (u32, u32) {
                (
                    $crate::proof_optimized::PROOF_OPTIMIZED_LOG_BASIS_MIN,
                    $crate::proof_optimized::PROOF_OPTIMIZED_LOG_BASIS_MAX,
                )
            }

            fn onehot_chunk_size() -> usize {
                $onehot_chunk
            }

            impl_proof_optimized_preset!(@schedule_catalog ($feat, $family_name, $table));
        }
    };
}

#[cfg(test)]
mod tests;

// ---------------------------------------------------------------------------
// Public preset structs
// ---------------------------------------------------------------------------

pub mod fp128;
pub mod fp32;
pub mod fp64;
