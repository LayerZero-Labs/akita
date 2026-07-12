//! Proof-optimized commitment config presets.
//!
//! Presets are unit structs that bind [`CommitmentConfig`] hooks to
//! [`akita_types`] SIS primitives and generated schedule tables.

use super::CommitmentConfig;
use crate::matrix_envelope::accumulate_matrix_envelope_for_level;
use akita_field::AkitaError;
use akita_field::{Ext2, FpExt4, Prime128OffsetA7F7, Prime32Offset99, Prime64Offset59};
use akita_types::{
    AkitaScheduleLookupKey, LevelParams, OpeningClaimsLayout, PolynomialGroupLayout, Schedule,
    SetupMatrixEnvelope,
};
use std::any::TypeId;
use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

/// Minimum proof-optimized log-basis.
pub(crate) const PROOF_OPTIMIZED_LOG_BASIS_MIN: u32 = 2;
/// Maximum proof-optimized log-basis.
pub(crate) const PROOF_OPTIMIZED_LOG_BASIS_MAX: u32 = 6;

/// Shared short ring-challenge policy for every proof-optimized preset.
///
/// Fixed-weight sparse families keyed on ring degree `d` via
/// [`akita_challenges::SparseChallengeConfig::production_for_ring_dim`].
/// A preset's `D` is fixed across all schedule levels, so both the planner DP
/// and the generated-table expansion call the per-`Cfg` hook with `d == Cfg::D`.
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

    let poly_counts = setup_envelope_poly_counts(max_num_batched_polys);
    let mut envelope = SetupMatrixEnvelope { max_setup_len: 1 };
    let mut saw_supported_shape = false;
    for main_num_vars in 1..=max_num_vars {
        let precommitted = PolynomialGroupLayout::new(main_num_vars, 1);
        for &main_num_polys in &poly_counts {
            let main_group = PolynomialGroupLayout::new(main_num_vars, main_num_polys);
            let layout = OpeningClaimsLayout::from_root_groups(&[], main_group)?;
            if let Some(entry_envelope) = setup_matrix_envelope_for_shape::<Cfg>(&layout)? {
                saw_supported_shape = true;
                envelope.max_setup_len = envelope.max_setup_len.max(entry_envelope.max_setup_len);
            }

            for num_precommitted in 1..=2 {
                let precommitteds = vec![precommitted; num_precommitted];
                saw_supported_shape |= inflate_setup_envelope_for_precommitted_root_batch::<Cfg>(
                    &precommitteds,
                    main_group,
                    &mut envelope,
                )?;
            }
        }
    }

    if !saw_supported_shape {
        return Err(AkitaError::InvalidSetup(format!(
            "setup matrix sizing found no generated schedules for max_num_vars={max_num_vars}"
        )));
    }

    Ok(envelope)
}

fn inflate_setup_envelope_for_precommitted_root_batch<Cfg: CommitmentConfig>(
    precommitteds: &[PolynomialGroupLayout],
    main_group: PolynomialGroupLayout,
    envelope: &mut SetupMatrixEnvelope,
) -> Result<bool, AkitaError> {
    let layout = OpeningClaimsLayout::from_root_groups(precommitteds, main_group)?;
    let entry_envelope = match setup_matrix_envelope_for_shape::<Cfg>(&layout) {
        Ok(Some(entry_envelope)) => entry_envelope,
        Ok(None) => return Ok(false),
        Err(_) => return Ok(false),
    };
    envelope.max_setup_len = envelope.max_setup_len.max(entry_envelope.max_setup_len);
    Ok(true)
}

#[cfg(test)]
fn runtime_key_from_generated_entry(
    entry: &akita_planner::generated::GeneratedScheduleTableEntry,
) -> AkitaScheduleLookupKey {
    AkitaScheduleLookupKey {
        final_group: entry.final_group,
        precommitteds: entry.precommitteds.to_vec(),
    }
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
pub fn worst_case_multi_group_opening_batch_for_shape(
    num_vars: usize,
    num_polynomials: usize,
) -> Result<OpeningClaimsLayout, AkitaError> {
    OpeningClaimsLayout::new(num_vars, num_polynomials)
}

fn setup_matrix_envelope_for_shape<Cfg: CommitmentConfig>(
    layout: &OpeningClaimsLayout,
) -> Result<Option<SetupMatrixEnvelope>, AkitaError> {
    let cached_key = crate::opening_schedule_key::<Cfg>(layout)?;

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
        &schedule, layout,
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

fn matrix_envelope_for_levels(
    setup_levels: &[LevelParams],
) -> Result<SetupMatrixEnvelope, AkitaError> {
    let mut max_setup_len: usize = 1;
    for lp in setup_levels {
        accumulate_matrix_envelope_for_level(lp, &mut max_setup_len)?;
    }
    Ok(SetupMatrixEnvelope { max_setup_len })
}

/// Packed setup envelope spanning every level in `schedule`, including root
/// runtime widening for the requested opening layout.
pub fn matrix_envelope_for_schedule<Cfg>(
    schedule: &Schedule,
    layout: &OpeningClaimsLayout,
) -> Result<SetupMatrixEnvelope, AkitaError>
where
    Cfg: CommitmentConfig,
{
    let setup_levels: Vec<LevelParams> = setup_level_params_from_runtime_schedule(&schedule.steps);
    let mut envelope = matrix_envelope_for_levels(&setup_levels)?;
    accumulate_root_matrix_envelope_for_opening_batch(
        schedule,
        layout,
        &mut envelope.max_setup_len,
    )?;
    Ok(envelope)
}

fn accumulate_root_matrix_envelope_for_opening_batch(
    schedule: &Schedule,
    layout: &OpeningClaimsLayout,
    max_setup_len: &mut usize,
) -> Result<(), AkitaError> {
    let Some(root_params) = root_commit_params_from_schedule(schedule)? else {
        return Ok(());
    };
    let root_len = root_runtime_matrix_len_for_opening_batch(&root_params, layout)?;
    *max_setup_len = (*max_setup_len).max(root_len);
    Ok(())
}

fn root_runtime_matrix_len_for_opening_batch(
    lp: &LevelParams,
    layout: &OpeningClaimsLayout,
) -> Result<usize, AkitaError> {
    let final_group_index = lp.validate_root_opening_batch(layout)?;
    let final_group = layout.group_layout(final_group_index)?;
    let (mut max_a_len, mut max_b_len, mut d_width) = group_setup_footprint(
        lp.a_key.row_len(),
        lp.a_key.col_len(),
        lp.b_key.row_len(),
        final_group.num_polynomials(),
        lp.num_blocks,
        lp.num_digits_open,
    )?;

    for group in &lp.precommitted_groups {
        let (a_len, b_len, group_d_width) = group_setup_footprint(
            group.a_key.row_len(),
            group.a_key.col_len(),
            group.b_key.row_len(),
            group.layout.group.num_polynomials(),
            group.num_blocks,
            group.num_digits_open,
        )?;
        max_a_len = max_a_len.max(a_len);
        max_b_len = max_b_len.max(b_len);
        d_width = d_width.checked_add(group_d_width).ok_or_else(|| {
            AkitaError::InvalidSetup("multi-group D setup width overflow".to_string())
        })?;
    }

    root_setup_len(lp.d_key.row_len(), d_width, max_a_len, max_b_len)
}

fn group_setup_footprint(
    a_rows: usize,
    a_width: usize,
    b_rows: usize,
    num_polys: usize,
    num_blocks: usize,
    num_digits_open: usize,
) -> Result<(usize, usize, usize), AkitaError> {
    let a_len = a_rows.checked_mul(a_width).ok_or_else(|| {
        AkitaError::InvalidSetup("multi-group A setup envelope overflow".to_string())
    })?;
    let d_width = num_polys
        .checked_mul(num_blocks)
        .and_then(|n| n.checked_mul(num_digits_open))
        .ok_or_else(|| {
            AkitaError::InvalidSetup("multi-group D setup width overflow".to_string())
        })?;
    let t_cols_per_vector = a_rows
        .checked_mul(num_digits_open)
        .and_then(|n| n.checked_mul(num_blocks))
        .ok_or_else(|| {
            AkitaError::InvalidSetup("multi-group B setup width overflow".to_string())
        })?;
    let b_width = num_polys.checked_mul(t_cols_per_vector).ok_or_else(|| {
        AkitaError::InvalidSetup("multi-group B setup width overflow".to_string())
    })?;
    let b_len = b_rows.checked_mul(b_width).ok_or_else(|| {
        AkitaError::InvalidSetup("multi-group B setup envelope overflow".to_string())
    })?;
    Ok((a_len, b_len, d_width))
}

fn root_setup_len(
    d_rows: usize,
    d_width: usize,
    max_a_len: usize,
    max_b_len: usize,
) -> Result<usize, AkitaError> {
    let d_len = d_rows
        .checked_mul(d_width)
        .ok_or_else(|| AkitaError::InvalidSetup("root D setup envelope overflow".to_string()))?;
    Ok(d_len.max(max_a_len).max(max_b_len))
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
    ($cfg:ident, $field:ty, $ext_field:ty, $family:expr, $d:expr, $field_bits:expr, $log_commit_bound:expr) => {
        impl_proof_optimized_preset!(@core $cfg, $field, $ext_field, $family, $d, $field_bits, $log_commit_bound, 1, none);
    };
    ($cfg:ident, $field:ty, $ext_field:ty, $family:expr, $d:expr, $field_bits:expr, $log_commit_bound:expr, schedules = ($feat:literal, $family_name:literal, $table:ident)) => {
        impl_proof_optimized_preset!(@core $cfg, $field, $ext_field, $family, $d, $field_bits, $log_commit_bound, 1, table, $feat, $family_name, $table);
    };
    ($cfg:ident, $field:ty, $ext_field:ty, $family:expr, $d:expr, $field_bits:expr, $log_commit_bound:expr, $onehot_chunk_size:expr) => {
        impl_proof_optimized_preset!(@core $cfg, $field, $ext_field, $family, $d, $field_bits, $log_commit_bound, $onehot_chunk_size, none);
    };
    ($cfg:ident, $field:ty, $ext_field:ty, $family:expr, $d:expr, $field_bits:expr, $log_commit_bound:expr, $onehot_chunk_size:expr, schedules = ($feat:literal, $family_name:literal, $table:ident)) => {
        impl_proof_optimized_preset!(@core $cfg, $field, $ext_field, $family, $d, $field_bits, $log_commit_bound, $onehot_chunk_size, table, $feat, $family_name, $table);
    };
    (@core $cfg:ident, $field:ty, $ext_field:ty, $family:expr, $d:expr, $field_bits:expr, $log_commit_bound:expr, $onehot_chunk:expr, none) => {
        impl $crate::CommitmentConfig for $cfg {
            type Field = $field;
            type ExtField = $ext_field;
            const D: usize = $d;

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
    (@core $cfg:ident, $field:ty, $ext_field:ty, $family:expr, $d:expr, $field_bits:expr, $log_commit_bound:expr, $onehot_chunk:expr, table, $feat:literal, $family_name:literal, $table:ident) => {
        impl $crate::CommitmentConfig for $cfg {
            type Field = $field;
            type ExtField = $ext_field;
            const D: usize = $d;

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
