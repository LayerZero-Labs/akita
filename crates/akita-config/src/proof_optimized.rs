//! Proof-optimized commitment config presets.
//!
//! Presets are unit structs that bind [`CommitmentConfig`] hooks to
//! [`akita_types`] SIS primitives and generated schedule tables.

use super::CommitmentConfig;
use akita_field::AkitaError;
use akita_field::{Ext2, FpExt4, Prime128OffsetA7F7, Prime32Offset99, Prime64Offset59};
use akita_types::{
    accumulate_matrix_envelope_for_level, accumulate_terminal_matrix_envelope,
    setup_matrix_envelope_for_schedule, AkitaExpandedSetup, AkitaScheduleLookupKey,
    CommittedGroupParams, FoldSchedule, OpeningClaimsLayout, PolynomialGroupLayout,
    SetupMatrixEnvelope,
};
use std::any::TypeId;
use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

/// Minimum proof-optimized log-basis.
pub(crate) const PROOF_OPTIMIZED_LOG_BASIS_MIN: u32 = 2;
/// Maximum proof-optimized log-basis.
pub(crate) const PROOF_OPTIMIZED_LOG_BASIS_MAX: u32 = 6;

/// Bound setup preprocessing work before schedule resolution.
///
/// This is a verifier-facing allocation/CPU guard for untrusted serialized
/// setup capacity metadata. Production families currently scan at most a few
/// hundred scalar shapes.
const MAX_VERIFIER_SETUP_SCHEDULE_SCANS: usize = 1 << 14;

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

pub(crate) fn proof_optimized_schedule_key<Cfg: CommitmentConfig>(
    layout: &OpeningClaimsLayout,
) -> Result<AkitaScheduleLookupKey, AkitaError> {
    layout.check()?;
    let final_group = layout.root_final_group_layout()?;
    if layout.num_groups() == 1 {
        return Ok(AkitaScheduleLookupKey::single(final_group));
    }
    let precommitteds = layout
        .root_precommitted_group_layouts()?
        .iter()
        .copied()
        .map(crate::conservative_commitment::conservative_precommitted_group_params::<Cfg>)
        .collect::<Result<Vec<_>, _>>()?;
    let key = AkitaScheduleLookupKey {
        final_group,
        precommitteds,
    };
    key.validate()?;
    Ok(key)
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
    validate_setup_capacity_metadata(max_num_vars, max_num_batched_polys)?;
    let cache_key = (TypeId::of::<Cfg>(), max_num_vars, max_num_batched_polys);
    if let Some(cached) = SETUP_MATRIX_ENVELOPE_CACHE
        .lock()
        .map_err(|_| AkitaError::InvalidSetup("setup capacity cache lock poisoned".into()))?
        .get(&cache_key)
        .cloned()
    {
        return Ok(cached);
    }

    let envelope =
        proof_optimized_max_setup_matrix_size_uncached::<Cfg>(max_num_vars, max_num_batched_polys)?;

    SETUP_MATRIX_ENVELOPE_CACHE
        .lock()
        .map_err(|_| AkitaError::InvalidSetup("setup capacity cache lock poisoned".into()))?
        .insert(cache_key, envelope);

    Ok(envelope)
}

fn proof_optimized_max_setup_matrix_size_uncached<Cfg: CommitmentConfig>(
    max_num_vars: usize,
    max_num_batched_polys: usize,
) -> Result<SetupMatrixEnvelope, AkitaError> {
    let layouts = setup_envelope_scan_layouts::<Cfg>(max_num_vars, max_num_batched_polys)?;
    let mut saw_supported_shape = false;
    let mut envelope = SetupMatrixEnvelope::minimum();
    for layout in &layouts {
        let Ok(schedule) = Cfg::get_params_for_prove(layout) else {
            continue;
        };
        let entry_envelope = setup_matrix_envelope_for_schedule(&schedule)?;
        saw_supported_shape = true;
        envelope.max_setup_len = envelope.max_setup_len.max(entry_envelope.max_setup_len);
    }

    // Prefix-slot materialization is driven by these bounded exact recursive
    // keys. Size their shared matrices from the same keys directly: converting
    // through `OpeningClaimsLayout` would discard frozen precommitted params
    // and could resolve a different schedule.
    for key in crate::generated_families::recursive_group_batch_candidates_for_capacity::<Cfg>(
        max_num_vars,
        max_num_batched_polys,
    )? {
        let schedule = Cfg::runtime_schedule(key)?;
        let entry_envelope = setup_matrix_envelope_for_schedule(&schedule)?;
        saw_supported_shape = true;
        envelope.max_setup_len = envelope.max_setup_len.max(entry_envelope.max_setup_len);
    }

    if !saw_supported_shape {
        return Err(AkitaError::InvalidSetup(format!(
            "setup matrix sizing found no generated schedules for max_num_vars={max_num_vars}"
        )));
    }

    Ok(envelope)
}

fn validate_setup_capacity_metadata(
    max_num_vars: usize,
    max_num_batched_polys: usize,
) -> Result<(), AkitaError> {
    if max_num_batched_polys == 0 {
        return Err(AkitaError::InvalidSetup(
            "max_num_batched_polys must be at least 1".to_string(),
        ));
    }
    if max_num_vars >= usize::BITS as usize {
        return Err(AkitaError::InvalidSetup(format!(
            "verifier setup capacity ({max_num_vars} vars, {max_num_batched_polys} polynomials) \
             exceeds preprocessing limits"
        )));
    }
    Ok(())
}

fn setup_envelope_scan_layouts<Cfg: CommitmentConfig>(
    max_num_vars: usize,
    max_num_batched_polys: usize,
) -> Result<Vec<OpeningClaimsLayout>, AkitaError> {
    let mut layouts = Vec::new();
    let supports_multi_group_root = Cfg::decomposition().log_commit_bound == 1;
    let precommitted_group = PolynomialGroupLayout::new(max_num_vars, 1);
    let precommitted_groups = [precommitted_group;
        crate::generated_families::DEFAULT_GROUP_BATCH_MAX_PRECOMMITTED_GROUPS];

    let mut push_layout = |layout| {
        if layouts.len() >= MAX_VERIFIER_SETUP_SCHEDULE_SCANS {
            return Err(AkitaError::InvalidSetup(format!(
                "verifier setup capacity ({max_num_vars} vars, {max_num_batched_polys} polynomials) \
                 exceeds preprocessing limits"
            )));
        }
        layouts.push(layout);
        Ok(())
    };

    for main_num_vars in 1..=max_num_vars {
        for main_num_polys in 1..=max_num_batched_polys {
            let main_group = PolynomialGroupLayout::new(main_num_vars, main_num_polys);
            push_layout(OpeningClaimsLayout::from_root_groups(&[], main_group)?)?;
            if supports_multi_group_root {
                let num_precommitted =
                    crate::generated_families::DEFAULT_GROUP_BATCH_MAX_PRECOMMITTED_GROUPS;
                let Some(total_polynomials) = main_num_polys.checked_add(num_precommitted) else {
                    continue;
                };
                if total_polynomials > max_num_batched_polys {
                    continue;
                }
                push_layout(OpeningClaimsLayout::from_root_groups(
                    &precommitted_groups[..num_precommitted],
                    main_group,
                )?)?;
            }
        }
    }

    Ok(layouts)
}

/// Extract setup-level params from a `FoldSchedule`.
///
pub fn setup_level_params_from_schedule(schedule: &FoldSchedule) -> Vec<CommittedGroupParams> {
    std::iter::once(schedule.root.params.final_group.commitment.clone())
        .chain(
            schedule
                .recursive_folds
                .iter()
                .map(|fold| fold.params.witness.clone()),
        )
        .collect()
}

/// Reject a concrete schedule whose exact matrix footprint exceeds setup.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidSetup`] when sizing overflows or the setup's
/// materialized shared matrix is too short for `schedule` and `layout`.
pub fn ensure_schedule_fits_setup<Cfg>(
    setup: &AkitaExpandedSetup<Cfg::Field>,
    schedule: &FoldSchedule,
    layout: &OpeningClaimsLayout,
) -> Result<(), AkitaError>
where
    Cfg: CommitmentConfig,
{
    for params in setup_level_params_from_schedule(schedule) {
        let mut required_setup_len = 1;
        accumulate_matrix_envelope_for_level(&params, &mut required_setup_len)?;
        let available_setup_len = setup
            .shared_matrix
            .total_ring_elements_at_dyn(params.d_a())?;
        ensure_required_setup_len(required_setup_len, available_setup_len, params.d_a())?;
    }
    let terminal = &schedule.terminal.params.witness;
    let mut required_setup_len = 1;
    accumulate_terminal_matrix_envelope(terminal, &mut required_setup_len)?;
    let available_setup_len = setup
        .shared_matrix
        .total_ring_elements_at_dyn(terminal.d_a())?;
    ensure_required_setup_len(required_setup_len, available_setup_len, terminal.d_a())?;

    let root_params = &schedule.root.params.final_group.commitment;
    let required_setup_len = root_runtime_matrix_len_for_opening_batch(root_params, layout)?;
    let available_setup_len = setup
        .shared_matrix
        .total_ring_elements_at_dyn(root_params.d_a())?;
    ensure_required_setup_len(required_setup_len, available_setup_len, root_params.d_a())?;
    Ok(())
}

fn ensure_required_setup_len(
    required_setup_len: usize,
    available_setup_len: usize,
    ring_dimension: usize,
) -> Result<(), AkitaError> {
    if required_setup_len <= available_setup_len {
        return Ok(());
    }
    Err(AkitaError::InvalidSetup(format!(
        "schedule requires {required_setup_len} setup ring elements at D={ring_dimension}, but \
         setup provides {available_setup_len}"
    )))
}

fn root_runtime_matrix_len_for_opening_batch(
    lp: &CommittedGroupParams,
    layout: &OpeningClaimsLayout,
) -> Result<usize, AkitaError> {
    let final_group_index = lp.validate_opening_batch(layout)?;
    let final_group = layout.group_layout(final_group_index)?;
    let (a_len, b_len, mut d_width) = group_setup_footprint(
        lp.inner_commit_matrix.output_rank(),
        lp.inner_commit_matrix.input_width(),
        lp.outer_commit_matrix.output_rank(),
        final_group.num_polynomials(),
        lp.num_live_blocks,
        lp.num_digits_open,
    )?;
    let mut max_a_coeff_len = a_len
        .checked_mul(lp.inner_commit_matrix.ring_dimension())
        .ok_or_else(|| AkitaError::InvalidSetup("root A setup envelope overflow".into()))?;
    let mut max_b_coeff_len = b_len
        .checked_mul(lp.outer_commit_matrix.ring_dimension())
        .ok_or_else(|| AkitaError::InvalidSetup("root B setup envelope overflow".into()))?;

    for group in &lp.precommitted_groups {
        let (a_len, b_len, group_d_width) = group_setup_footprint(
            group.inner_commit_matrix.output_rank(),
            group.inner_commit_matrix.input_width(),
            group.outer_commit_matrix.output_rank(),
            group.layout.group.num_polynomials(),
            group.layout.num_live_blocks,
            group.num_digits_open,
        )?;
        let a_coeff_len = a_len
            .checked_mul(group.inner_commit_matrix.ring_dimension())
            .ok_or_else(|| AkitaError::InvalidSetup("multi-group A setup overflow".into()))?;
        let b_coeff_len = b_len
            .checked_mul(group.outer_commit_matrix.ring_dimension())
            .ok_or_else(|| AkitaError::InvalidSetup("multi-group B setup overflow".into()))?;
        max_a_coeff_len = max_a_coeff_len.max(a_coeff_len);
        max_b_coeff_len = max_b_coeff_len.max(b_coeff_len);
        d_width = d_width.checked_add(group_d_width).ok_or_else(|| {
            AkitaError::InvalidSetup("multi-group D setup width overflow".to_string())
        })?;
    }

    root_setup_len(
        lp.open_commit_matrix.output_rank(),
        d_width,
        lp.open_commit_matrix.ring_dimension(),
        max_a_coeff_len,
        max_b_coeff_len,
        lp.d_a(),
    )
}

fn group_setup_footprint(
    a_rows: usize,
    a_width: usize,
    b_rows: usize,
    num_polys: usize,
    num_live_blocks: usize,
    num_digits_open: usize,
) -> Result<(usize, usize, usize), AkitaError> {
    let a_len = a_rows.checked_mul(a_width).ok_or_else(|| {
        AkitaError::InvalidSetup("multi-group A setup envelope overflow".to_string())
    })?;
    let d_width = num_polys
        .checked_mul(num_live_blocks)
        .and_then(|n| n.checked_mul(num_digits_open))
        .ok_or_else(|| {
            AkitaError::InvalidSetup("multi-group D setup width overflow".to_string())
        })?;
    let t_vector_width = a_rows
        .checked_mul(num_digits_open)
        .and_then(|n| n.checked_mul(num_live_blocks))
        .ok_or_else(|| {
            AkitaError::InvalidSetup("multi-group B setup width overflow".to_string())
        })?;
    let b_width = num_polys.checked_mul(t_vector_width).ok_or_else(|| {
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
    d_ring_dim: usize,
    max_a_coeff_len: usize,
    max_b_coeff_len: usize,
    envelope_ring_dim: usize,
) -> Result<usize, AkitaError> {
    if envelope_ring_dim == 0 {
        return Err(AkitaError::InvalidSetup(
            "root setup envelope ring dimension is zero".into(),
        ));
    }
    let d_coeff_len = d_rows
        .checked_mul(d_width)
        .and_then(|len| len.checked_mul(d_ring_dim))
        .ok_or_else(|| AkitaError::InvalidSetup("root D setup envelope overflow".to_string()))?;
    Ok(d_coeff_len
        .max(max_a_coeff_len)
        .max(max_b_coeff_len)
        .div_ceil(envelope_ring_dim))
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
    // Root-fold `log_basis` pin. `none` emits nothing (trait default `None`);
    // `($lb)` overrides `root_log_basis()` to `Some($lb)` for this preset only.
    (@root_log_basis none) => {};
    (@root_log_basis ($root_lb:literal)) => {
        fn root_log_basis() -> Option<u32> {
            Some($root_lb)
        }
    };
    // --- Preset arms that pin the root `log_basis` (must precede the unpinned
    // arms so the `root_log_basis = N` token is not misparsed as an expression). ---
    ($cfg:ident, $field:ty, $ext_field:ty, $family:expr, $d:expr, $field_bits:expr, $log_commit_bound:expr, root_log_basis = $root_lb:literal, schedules = ($feat:literal, $family_name:literal, $table:ident)) => {
        impl_proof_optimized_preset!(@core $cfg, $field, $ext_field, $family, $d, $field_bits, $log_commit_bound, 1, ($root_lb), table, $feat, $family_name, $table);
    };
    ($cfg:ident, $field:ty, $ext_field:ty, $family:expr, $d:expr, $field_bits:expr, $log_commit_bound:expr, root_log_basis = $root_lb:literal) => {
        impl_proof_optimized_preset!(@core $cfg, $field, $ext_field, $family, $d, $field_bits, $log_commit_bound, 1, ($root_lb), none);
    };
    ($cfg:ident, $field:ty, $ext_field:ty, $family:expr, $d:expr, $field_bits:expr, $log_commit_bound:expr, $onehot_chunk_size:expr, root_log_basis = $root_lb:literal) => {
        impl_proof_optimized_preset!(@core $cfg, $field, $ext_field, $family, $d, $field_bits, $log_commit_bound, $onehot_chunk_size, ($root_lb), none);
    };
    ($cfg:ident, $field:ty, $ext_field:ty, $family:expr, $d:expr, $field_bits:expr, $log_commit_bound:expr, $onehot_chunk_size:expr, root_log_basis = $root_lb:literal, schedules = ($feat:literal, $family_name:literal, $table:ident)) => {
        impl_proof_optimized_preset!(@core $cfg, $field, $ext_field, $family, $d, $field_bits, $log_commit_bound, $onehot_chunk_size, ($root_lb), table, $feat, $family_name, $table);
    };
    // --- Unpinned preset arms (root `log_basis` left to the planner). ---
    ($cfg:ident, $field:ty, $ext_field:ty, $family:expr, $d:expr, $field_bits:expr, $log_commit_bound:expr) => {
        impl_proof_optimized_preset!(@core $cfg, $field, $ext_field, $family, $d, $field_bits, $log_commit_bound, 1, none, none);
    };
    ($cfg:ident, $field:ty, $ext_field:ty, $family:expr, $d:expr, $field_bits:expr, $log_commit_bound:expr, schedules = ($feat:literal, $family_name:literal, $table:ident)) => {
        impl_proof_optimized_preset!(@core $cfg, $field, $ext_field, $family, $d, $field_bits, $log_commit_bound, 1, none, table, $feat, $family_name, $table);
    };
    ($cfg:ident, $field:ty, $ext_field:ty, $family:expr, $d:expr, $field_bits:expr, $log_commit_bound:expr, $onehot_chunk_size:expr) => {
        impl_proof_optimized_preset!(@core $cfg, $field, $ext_field, $family, $d, $field_bits, $log_commit_bound, $onehot_chunk_size, none, none);
    };
    ($cfg:ident, $field:ty, $ext_field:ty, $family:expr, $d:expr, $field_bits:expr, $log_commit_bound:expr, $onehot_chunk_size:expr, schedules = ($feat:literal, $family_name:literal, $table:ident)) => {
        impl_proof_optimized_preset!(@core $cfg, $field, $ext_field, $family, $d, $field_bits, $log_commit_bound, $onehot_chunk_size, none, table, $feat, $family_name, $table);
    };
    (@core $cfg:ident, $field:ty, $ext_field:ty, $family:expr, $d:expr, $field_bits:expr, $log_commit_bound:expr, $onehot_chunk:expr, $root_lb:tt, none) => {
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

            fn sis_modulus_profile() -> akita_types::SisModulusProfileId {
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

            impl_proof_optimized_preset!(@root_log_basis $root_lb);

            fn get_params_for_prove(
                layout: &akita_types::OpeningClaimsLayout,
            ) -> Result<akita_types::FoldSchedule, akita_field::AkitaError> {
                Self::runtime_schedule($crate::proof_optimized::proof_optimized_schedule_key::<Self>(
                    layout,
                )?)
            }

            impl_proof_optimized_preset!(@schedule_catalog none);
        }
    };
    (@core $cfg:ident, $field:ty, $ext_field:ty, $family:expr, $d:expr, $field_bits:expr, $log_commit_bound:expr, $onehot_chunk:expr, $root_lb:tt, table, $feat:literal, $family_name:literal, $table:ident) => {
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

            fn sis_modulus_profile() -> akita_types::SisModulusProfileId {
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

            impl_proof_optimized_preset!(@root_log_basis $root_lb);

            fn get_params_for_prove(
                layout: &akita_types::OpeningClaimsLayout,
            ) -> Result<akita_types::FoldSchedule, akita_field::AkitaError> {
                Self::runtime_schedule($crate::proof_optimized::proof_optimized_schedule_key::<Self>(
                    layout,
                )?)
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
