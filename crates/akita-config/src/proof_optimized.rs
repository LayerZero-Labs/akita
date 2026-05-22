//! Concrete proof-optimized commitment configs for the default fp128 protocol.
//!
//! Each config is a plain unit struct that wires its required
//! [`CommitmentConfig`] hooks to the policy-agnostic SIS primitives in
//! [`akita_derive`] and the generated schedule tables in [`akita_types`].
//! A preset only declares its `(D, LOG_COMMIT_BOUND)` decomposition, its
//! sparse stage-1 family, the generated schedule table that backs it, and
//! (when applicable) the audited root-rank floor.

use super::{AjtaiRole, CommitmentConfig, CommitmentEnvelope};
use akita_field::AkitaError;
use akita_field::{
    Ext2, Prime128OffsetA7F7, Prime16Offset99, Prime32Offset99, Prime64Offset59, RingSubfieldFp4,
    RingSubfieldFp8,
};
use akita_types::generated::table_entry_envelope_up_to_num_vars;
use akita_types::ClaimIncidenceSummary;
use akita_types::{
    planned_log_basis_at_level_from_schedule, planned_schedule_key_from_schedule, AkitaPlannedStep,
    AkitaScheduleInputs, AkitaScheduleLookupKey, AkitaSchedulePlan, LevelParams,
};

/// Inclusive minimum of the proof-optimized log-basis search range used by
/// every preset.
pub(crate) const PROOF_OPTIMIZED_LOG_BASIS_MIN: u32 = 2;
/// Inclusive maximum of the proof-optimized log-basis search range used by
/// every preset.
pub(crate) const PROOF_OPTIMIZED_LOG_BASIS_MAX: u32 = 6;

// ---------------------------------------------------------------------------
// `<Cfg>`-generic policy helpers consumed by the planner / materializer.
//
// These four free functions are the proof-optimized expression of "given
// `Cfg`'s envelope and schedule table, derive the level params for a fold
// or direct step." They are passed as fn pointers into
// `akita_derive::PlanPolicy` (materializer) and `akita-planner`'s internal
// `SearchOptions` (DP). Their bodies stay here — not in `akita_derive` —
// because they consult `Cfg::envelope` / `Cfg::schedule_plan`, which only
// exist at the `akita-config` layer.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Trait-shaped wrappers consumed by the macros below.
//
// Each wrapper implements one required `CommitmentConfig` method by routing
// through the planned schedule table when available and falling back to the
// `akita_derive` SIS primitives otherwise. Per-preset constants (decomposition,
// stage-1 challenge family, audited root-rank floor) are inlined into the
// macros rather than living as separate helpers.
// ---------------------------------------------------------------------------

/// Proof-optimized `schedule_plan` impl.
///
/// Materializes the matching entry from `Cfg::schedule_table()` through the
/// planner's `schedule_plan_from_table` materializer with a `PlanPolicy`
/// derived directly from the config hooks.
pub(crate) fn proof_optimized_schedule_plan<Cfg>(
    key: AkitaScheduleLookupKey,
) -> Result<Option<AkitaSchedulePlan>, AkitaError>
where
    Cfg: CommitmentConfig,
{
    let Some(table) = Cfg::schedule_table() else {
        return Ok(None);
    };
    akita_derive::schedule_plan_from_table::<<Cfg as CommitmentConfig>::Field, _>(
        key,
        table,
        akita_derive::PlanPolicy {
            sis_family: Cfg::sis_modulus_family(),
            ring_dimension: Cfg::D,
            root_decomp: Cfg::decomposition(),
            challenge_field_bits: Cfg::decomposition().field_bits() * Cfg::CHAL_EXT_DEGREE as u32,
            recursive_public_rows: 1,
            extension_opening_width: Cfg::CLAIM_EXT_DEGREE,
            stage1_challenge_config: Cfg::stage1_challenge_config,
            envelope: Cfg::envelope(key.num_vars),
            ring_subfield_norm_bound: Cfg::ring_subfield_embedding_norm_bound(),
        },
    )
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
            key.num_points,
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
) -> Result<u32, AkitaError> {
    let key = AkitaScheduleLookupKey::singleton(inputs.num_vars);
    match proof_optimized_schedule_plan::<Cfg>(key)? {
        Some(plan) => planned_log_basis_at_level_from_schedule(&plan, inputs),
        None => Ok(Cfg::decomposition().log_basis),
    }
}

/// Canonical "what level params would `Cfg` use here?" lookup: prefer the
/// exact planned level when the public inputs match the offline schedule
/// table, otherwise derive SIS-secure recursive params (or fall back to
/// the envelope for level 0).
///
/// Used to be `Cfg::level_params_with_log_basis`; now a free function
/// because every preset's impl was a no-op delegator to this body and
/// the only "override" semantics needed live in tests, which call this
/// directly with their own `Cfg`.
///
/// # Errors
///
/// Returns an error if the SIS-floor lookup fails or if `Cfg`'s schedule
/// hooks (table lookup, envelope, stage-1 challenge config) reject the
/// inputs.
pub fn level_params_with_log_basis<Cfg: CommitmentConfig>(
    inputs: AkitaScheduleInputs,
    log_basis: u32,
) -> Result<LevelParams, AkitaError> {
    let plan =
        proof_optimized_schedule_plan::<Cfg>(AkitaScheduleLookupKey::singleton(inputs.num_vars))?;
    let envelope = Cfg::envelope(inputs.num_vars);
    akita_derive::level_params_with_log_basis(
        Cfg::sis_modulus_family(),
        Cfg::D,
        Cfg::decomposition(),
        Cfg::ring_subfield_embedding_norm_bound(),
        plan.as_ref(),
        &envelope,
        Cfg::stage1_challenge_config,
        inputs,
        log_basis,
    )
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
/// committable sub-shape `(num_vars', num_polys', num_points')` with
/// `1 <= num_vars' <= max_num_vars`,
/// `1 <= num_polys' <= max_num_batched_polys`, and
/// `1 <= num_points' <= num_polys'.min(max_num_points)`. Without this, a
/// runtime commit at a smaller variable count or differently shaped batch
/// can pick a schedule with strictly larger row count than the all-up
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
            for num_points in 1..=upper_pts {
                let incidence =
                    ClaimIncidenceSummary::from_counts(num_vars, num_polys, num_points)?;
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
    let cached_key = AkitaScheduleLookupKey::new_from_incidence(incidence)?;
    let _ = setup_envelope;

    // Table-only: configs that want a runtime DP fallback override the
    // `max_setup_matrix_size` trait method directly (see `PlannerCfg`).
    let Some(plan) = Cfg::schedule_plan(cached_key)? else {
        return Ok(None);
    };
    let setup_levels = setup_level_params_from_plan(&plan);

    Ok(Some(matrix_envelope_for_levels::<Cfg>(&setup_levels)?))
}

/// Extract setup-level params from a materialized plan: the root step's
/// commit layout plus every subsequent fold step's `lp` and the terminal
/// `level_params` if the run finishes in a packed direct step.
///
/// Both `Fold(lp)` and `Direct(commit_params | level_params)` are
/// commitments that must be covered by setup matrices. Direct steps now
/// carry their participating `LevelParams` inline (root-direct in
/// `commit_params`, terminal-direct-after-fold in `level_params`); a
/// well-formed plan has exactly one populated per Direct step.
pub fn setup_level_params_from_plan(plan: &AkitaSchedulePlan) -> Vec<LevelParams> {
    plan.steps
        .iter()
        .filter_map(|step| match step {
            AkitaPlannedStep::Fold(level) => Some(level.lp.clone()),
            AkitaPlannedStep::Direct(direct) => direct
                .commit_params
                .clone()
                .or_else(|| direct.level_params.clone()),
        })
        .collect()
}

/// Extract setup-level params from a runtime `Schedule`. Mirrors
/// [`setup_level_params_from_plan`] for schedules that did not come from
/// the offline table (e.g. `PlannerCfg`'s DP fallback path).
pub fn setup_level_params_from_runtime_schedule(steps: &[akita_types::Step]) -> Vec<LevelParams> {
    steps
        .iter()
        .filter_map(|step| match step {
            akita_types::Step::Fold(fold_step) => Some(fold_step.params.clone()),
            akita_types::Step::Direct(direct) => direct
                .commit_params
                .clone()
                .or_else(|| direct.level_params.clone()),
        })
        .collect()
}

pub fn matrix_envelope_for_levels<Cfg>(
    setup_levels: &[LevelParams],
) -> Result<(usize, usize), AkitaError>
where
    Cfg: CommitmentConfig,
{
    let mut max_rows: usize = 1;
    let mut max_stride: usize = 1;
    for lp in setup_levels {
        accumulate_matrix_envelope_for_level::<Cfg>(lp, &mut max_rows, &mut max_stride)?;
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
/// optional generated schedule table. Every other trait method is a one-line
/// delegation to the proof-optimized helpers above.
macro_rules! impl_fp128_preset {
    ($cfg:ident, $d:expr, $log_commit_bound:expr, $table:expr) => {
        impl $crate::CommitmentConfig for $cfg {
            type Field = Field;
            type ClaimField = Field;
            type ChallengeField = Field;
            const D: usize = $d;

            fn decomposition() -> akita_types::DecompositionParams {
                // Every fp128 preset uses `log_basis = 3` and sets
                // `log_open_bound = Some(128)` unless the gadget already
                // saturates the field (`log_commit_bound == 128`).
                akita_types::DecompositionParams {
                    log_basis: 3,
                    log_commit_bound: $log_commit_bound,
                    log_open_bound: if $log_commit_bound < 128 {
                        Some(128)
                    } else {
                        None
                    },
                }
            }

            fn stage1_challenge_config(
                d: usize,
            ) -> Result<akita_challenges::SparseChallengeConfig, akita_field::AkitaError> {
                // Sparse stage-1 challenge family for a given fp128 ring degree.
                match d {
                    32 => Ok(akita_challenges::SparseChallengeConfig::BoundedL1Norm),
                    64 => Ok(akita_challenges::SparseChallengeConfig::ExactShell {
                        count_mag1: 30,
                        count_mag2: 12,
                    }),
                    128 => Ok(akita_challenges::SparseChallengeConfig::Uniform {
                        weight: 31,
                        nonzero_coeffs: vec![-1, 1],
                    }),
                    _ => Err(akita_field::AkitaError::InvalidSetup(format!(
                        "unsupported fp128 ring dim {d}"
                    ))),
                }
            }

            fn sis_modulus_family() -> akita_types::SisModulusFamily {
                akita_types::SisModulusFamily::Q128
            }

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

            fn audited_root_rank(role: akita_types::AjtaiRole, max_num_vars: usize) -> usize {
                // Returns `1`, escalating to `2` once `max_num_vars` crosses the
                // threshold for the audited `(D, log_commit_bound, role)` cell.
                let log_commit_bound =
                    <Self as $crate::CommitmentConfig>::decomposition().log_commit_bound;
                let threshold: Option<usize> = match (
                    <Self as $crate::CommitmentConfig>::D,
                    log_commit_bound,
                    role,
                ) {
                    // `D=128` full-field A escalates to 2 from `max_num_vars=59` onward.
                    (128, lcb, akita_types::AjtaiRole::Inner) if lcb != 1 => Some(59),
                    // `D=128` outer (B/D) escalates from `max_num_vars=54` onward.
                    (128, _, akita_types::AjtaiRole::Outer) => Some(54),
                    // `D=64` onehot outer (B/D) escalates from `max_num_vars=38` onward.
                    (64, 1, akita_types::AjtaiRole::Outer) => Some(38),
                    _ => None,
                };
                1 + usize::from(threshold.is_some_and(|t| max_num_vars >= t))
            }

            fn envelope(max_num_vars: usize) -> akita_types::CommitmentEnvelope {
                $crate::proof_optimized::proof_optimized_envelope::<Self>(max_num_vars)
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

            fn log_basis_at_level(
                inputs: akita_types::AkitaScheduleInputs,
            ) -> Result<u32, akita_field::AkitaError> {
                $crate::proof_optimized::proof_optimized_log_basis_at_level::<Self>(inputs)
            }

            fn log_basis_search_range(_inputs: akita_types::AkitaScheduleInputs) -> (u32, u32) {
                (
                    $crate::proof_optimized::PROOF_OPTIMIZED_LOG_BASIS_MIN,
                    $crate::proof_optimized::PROOF_OPTIMIZED_LOG_BASIS_MAX,
                )
            }
        }
    };
}
pub(crate) use impl_fp128_preset;

macro_rules! impl_small_field_preset {
    ($cfg:ident, $field:ty, $claim_field:ty, $family:expr, $d:expr, $field_bits:expr, $log_commit_bound:expr, $log_basis:expr, $weight:expr, $coeffs:expr, $table:expr) => {
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

            fn stage1_challenge_config(
                d: usize,
            ) -> Result<akita_challenges::SparseChallengeConfig, akita_field::AkitaError> {
                if d != Self::D {
                    return Err(akita_field::AkitaError::InvalidSetup(format!(
                        "unsupported D={} for small-field preset (expected {})",
                        d,
                        Self::D,
                    )));
                }
                Ok(akita_challenges::SparseChallengeConfig::Uniform {
                    weight: $weight,
                    nonzero_coeffs: $coeffs,
                })
            }

            fn sis_modulus_family() -> akita_types::SisModulusFamily {
                $family
            }

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

            fn audited_root_rank(role: akita_types::AjtaiRole, max_num_vars: usize) -> usize {
                let _ = (role, max_num_vars);
                1
            }

            fn envelope(max_num_vars: usize) -> akita_types::CommitmentEnvelope {
                $crate::proof_optimized::proof_optimized_envelope::<Self>(max_num_vars)
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

            fn log_basis_at_level(
                inputs: akita_types::AkitaScheduleInputs,
            ) -> Result<u32, akita_field::AkitaError> {
                $crate::proof_optimized::proof_optimized_log_basis_at_level::<Self>(inputs)
            }

            fn log_basis_search_range(_inputs: akita_types::AkitaScheduleInputs) -> (u32, u32) {
                (
                    $crate::proof_optimized::PROOF_OPTIMIZED_LOG_BASIS_MIN,
                    $crate::proof_optimized::PROOF_OPTIMIZED_LOG_BASIS_MAX,
                )
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(not(feature = "zk"))]
    use akita_types::generated::{
        fp128_d32_full_table, fp128_d32_onehot_table, fp128_d64_full_table, fp128_d64_onehot_table,
        fp16_d32_full_table, fp16_d32_onehot_table, fp16_d64_full_table, fp16_d64_onehot_table,
        fp32_d32_onehot_table, fp32_d32_table, fp32_d64_onehot_table, fp32_d64_table,
        fp64_d32_onehot_table, fp64_d32_table, fp64_d64_onehot_table, fp64_d64_table,
        GeneratedScheduleTable,
    };
    #[cfg(test)]
    use akita_types::layout::digit_math::optimal_m_r_split;
    #[cfg(test)]
    use akita_types::level_layout_from_params;
    #[cfg(test)]
    use akita_types::{planned_w_ring_element_count, recursive_level_decomposition_from_root};

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
    #[cfg(not(feature = "zk"))]
    fn presets_select_expected_sis_modulus_family() {
        assert_eq!(
            <fp128::D32Full as CommitmentConfig>::sis_modulus_family(),
            akita_types::SisModulusFamily::Q128
        );
        assert_eq!(
            <fp32::D64Full as CommitmentConfig>::sis_modulus_family(),
            akita_types::SisModulusFamily::Q32
        );
        assert_eq!(
            <fp64::D64Full as CommitmentConfig>::sis_modulus_family(),
            akita_types::SisModulusFamily::Q64
        );
        assert_eq!(
            <fp16::D64Full as CommitmentConfig>::sis_modulus_family(),
            akita_types::SisModulusFamily::Q16
        );
    }

    #[test]
    #[cfg(not(feature = "zk"))]
    fn fp16_generated_schedule_tables_are_wired() {
        let onehot_key = AkitaScheduleLookupKey::singleton(32);
        let onehot_plan = <fp16::D32OneHot as crate::CommitmentConfig>::schedule_plan(onehot_key)
            .unwrap()
            .expect("fp16 D32 onehot nv32 schedule should be generated");
        assert!(!onehot_plan.steps.is_empty());

        let dense_key = AkitaScheduleLookupKey::singleton(27);
        let dense_plan = <fp16::D32Full as crate::CommitmentConfig>::schedule_plan(dense_key)
            .unwrap()
            .expect("fp16 D32 full nv27 schedule should be generated");
        assert!(!dense_plan.steps.is_empty());
    }

    #[test]
    #[cfg(not(feature = "zk"))]
    fn fp32_d32_generated_schedule_tables_are_wired() {
        let onehot_key = AkitaScheduleLookupKey::singleton(32);
        let onehot_plan = <fp32::D32OneHot as crate::CommitmentConfig>::schedule_plan(onehot_key)
            .unwrap()
            .expect("fp32 D32 onehot nv32 schedule should be generated");
        assert!(!onehot_plan.steps.is_empty());

        let dense_key = AkitaScheduleLookupKey::singleton(26);
        let dense_plan = <fp32::D32Full as crate::CommitmentConfig>::schedule_plan(dense_key)
            .unwrap()
            .expect("fp32 D32 full nv26 schedule should be generated");
        assert!(!dense_plan.steps.is_empty());
    }

    // ----- migrated from former `schedule_policy::tests` -------------------

    #[cfg(not(feature = "zk"))]
    fn assert_plan_matches_runtime_w_sizes<Cfg: CommitmentConfig>(num_vars: usize) {
        assert_plan_matches_runtime_w_sizes_for_key::<Cfg>(AkitaScheduleLookupKey::singleton(
            num_vars,
        ));
    }

    #[cfg(not(feature = "zk"))]
    fn assert_plan_matches_runtime_w_sizes_for_key<Cfg: CommitmentConfig>(
        key: AkitaScheduleLookupKey,
    ) {
        let plan = Cfg::schedule_plan(key)
            .expect("planner should succeed")
            .expect("config should provide a planner");
        let num_fold_levels = plan.num_fold_levels();
        for (idx, level) in plan.fold_levels().enumerate() {
            // The last fold in a fold-then-direct schedule is the terminal
            // recursive fold and ships its W in cleartext under
            // MRowLayout::Terminal (drops the D-block from the per-row `r`
            // quotients), so its `next_w_len` is smaller than what the
            // intermediate-layout helper would report.
            let is_terminal_fold = idx + 1 == num_fold_levels;
            let layout = if is_terminal_fold {
                akita_types::MRowLayout::Terminal
            } else {
                akita_types::MRowLayout::Intermediate
            };
            // Root-level batched witnesses fan out over the key's vector
            // counts; recursive levels collapse back to singleton-by-construction.
            let (num_points, num_t_vectors, num_w_vectors, num_public_rows) = if idx == 0 {
                (
                    key.num_points,
                    key.num_t_vectors,
                    key.num_w_vectors,
                    key.num_z_vectors,
                )
            } else {
                (1, 1, 1, 1)
            };
            let runtime_next_w_len =
                akita_types::w_ring_element_count_with_counts_for_layout::<Cfg::Field>(
                    &level.lp,
                    num_points,
                    num_t_vectors,
                    num_w_vectors,
                    num_public_rows,
                    layout,
                )
                .expect("valid planned witness")
                    * level.lp.ring_dimension;
            assert_eq!(
                runtime_next_w_len, level.next_inputs.current_w_len,
                "planner/runtime next_w_len mismatch at level {} for key={key:?}",
                level.inputs.level
            );
        }
    }

    #[cfg(not(feature = "zk"))]
    fn assert_every_table_entry_materializes<Cfg: CommitmentConfig>(table: GeneratedScheduleTable) {
        for entry in table.entries {
            let key = AkitaScheduleLookupKey::new_with_points(
                entry.key.num_vars,
                entry.key.num_commitment_groups,
                entry.key.num_t_vectors,
                entry.key.num_w_vectors,
                entry.key.num_z_vectors,
            );
            Cfg::schedule_plan(key)
                .expect("config schedule should succeed")
                .expect("config should provide a generated schedule");
        }
    }

    #[cfg(not(feature = "zk"))]
    fn assert_generated_batched_roots_are_scaled<Cfg: CommitmentConfig>(
        table: GeneratedScheduleTable,
    ) {
        let mut checked_folded_entry = false;
        for entry in table
            .entries
            .iter()
            .filter(|entry| entry.key.num_t_vectors > 1)
        {
            let key = AkitaScheduleLookupKey::new_with_points(
                entry.key.num_vars,
                entry.key.num_commitment_groups,
                entry.key.num_t_vectors,
                entry.key.num_w_vectors,
                entry.key.num_z_vectors,
            );
            let generated = Cfg::schedule_plan(key)
                .expect("config schedule should succeed")
                .expect("config should provide a generated schedule");
            let Some(root) = generated.fold_levels().next() else {
                continue;
            };
            checked_folded_entry = true;
            let singleton_outer_width =
                root.lp.a_key.row_len() * root.lp.num_digits_open * root.lp.num_blocks;
            let singleton_d_width = root.lp.num_digits_open * root.lp.num_blocks;
            assert_eq!(
                root.lp.outer_width(),
                singleton_outer_width * entry.key.num_t_vectors,
                "generated batched root B width should be claim-scaled for key={key:?}"
            );
            assert_eq!(
                root.lp.d_matrix_width(),
                singleton_d_width * entry.key.num_t_vectors,
                "generated batched root D width should be claim-scaled for key={key:?}"
            );
        }
        assert!(
            checked_folded_entry,
            "generated table should include at least one folded batched entry"
        );
    }

    #[cfg(not(feature = "zk"))]
    fn assert_exact_root_fold_matches_runtime_root_plan<Cfg: CommitmentConfig, const D: usize>(
        num_vars: usize,
    ) {
        let key = AkitaScheduleLookupKey::singleton(num_vars);
        let plan = Cfg::schedule_plan(key)
            .expect("config schedule should succeed")
            .expect("config should provide an exact schedule");
        let planned_root = akita_types::exact_planned_level_execution(
            &plan,
            AkitaScheduleInputs {
                num_vars,
                level: 0,
                current_w_len: 1usize.checked_shl(num_vars as u32).unwrap_or(0),
            },
            plan.fold_levels()
                .next()
                .expect("exact schedule should begin with a fold")
                .lp
                .log_basis,
            Cfg::stage1_challenge_config,
        )
        .expect("exact plan should resolve the root fold")
        .expect("exact plan should contain a matching root fold");
        let incidence =
            ClaimIncidenceSummary::same_point(num_vars, 1).expect("singleton incidence");
        let runtime_root =
            Cfg::get_params_for_prove(&incidence).expect("runtime root plan should succeed");
        let Some(akita_types::Step::Fold(runtime_root_step)) = runtime_root.steps.first() else {
            panic!("runtime root schedule should start with a fold");
        };
        assert_eq!(
            planned_root.level.inputs.current_w_len,
            runtime_root_step.current_w_len,
            "planned/runtime root current_w_len mismatch for {} at num_vars={num_vars}",
            std::any::type_name::<Cfg>()
        );
        assert_eq!(
            planned_root.level.lp,
            runtime_root_step.params,
            "planned/runtime root lp mismatch for {} at num_vars={num_vars}",
            std::any::type_name::<Cfg>()
        );
        assert_eq!(
            planned_root.level.next_inputs.current_w_len,
            runtime_root_step.next_w_len,
            "planned/runtime next_w_len mismatch for {} at num_vars={num_vars}",
            std::any::type_name::<Cfg>()
        );
    }

    #[test]
    #[cfg(not(feature = "zk"))]
    fn generated_fp128_schedule_tables_match_cfg_schedule() {
        assert_every_table_entry_materializes::<fp128::D32Full>(fp128_d32_full_table());
        assert_every_table_entry_materializes::<fp128::D32OneHot>(fp128_d32_onehot_table());
        assert_every_table_entry_materializes::<fp128::D64Full>(fp128_d64_full_table());
        assert_every_table_entry_materializes::<fp128::D64OneHot>(fp128_d64_onehot_table());
    }

    #[test]
    #[cfg(not(feature = "zk"))]
    fn generated_small_field_schedule_tables_match_cfg_schedule() {
        assert_every_table_entry_materializes::<fp16::D32Full>(fp16_d32_full_table());
        assert_every_table_entry_materializes::<fp16::D32OneHot>(fp16_d32_onehot_table());
        assert_every_table_entry_materializes::<fp16::D64Full>(fp16_d64_full_table());
        assert_every_table_entry_materializes::<fp16::D64OneHot>(fp16_d64_onehot_table());
        assert_every_table_entry_materializes::<fp32::D32Full>(fp32_d32_table());
        assert_every_table_entry_materializes::<fp32::D32OneHot>(fp32_d32_onehot_table());
        assert_every_table_entry_materializes::<fp32::D64Full>(fp32_d64_table());
        assert_every_table_entry_materializes::<fp32::D64OneHot>(fp32_d64_onehot_table());
        assert_every_table_entry_materializes::<fp64::D32Full>(fp64_d32_table());
        assert_every_table_entry_materializes::<fp64::D32OneHot>(fp64_d32_onehot_table());
        assert_every_table_entry_materializes::<fp64::D64Full>(fp64_d64_table());
        assert_every_table_entry_materializes::<fp64::D64OneHot>(fp64_d64_onehot_table());
    }

    #[test]
    #[cfg(not(feature = "zk"))]
    fn generated_batched_roots_restore_scaled_widths() {
        assert_generated_batched_roots_are_scaled::<fp128::D32Full>(fp128_d32_full_table());
        assert_generated_batched_roots_are_scaled::<fp128::D32OneHot>(fp128_d32_onehot_table());
        assert_generated_batched_roots_are_scaled::<fp128::D64Full>(fp128_d64_full_table());
        assert_generated_batched_roots_are_scaled::<fp128::D64OneHot>(fp128_d64_onehot_table());
        assert_generated_batched_roots_are_scaled::<fp16::D32Full>(fp16_d32_full_table());
        assert_generated_batched_roots_are_scaled::<fp16::D32OneHot>(fp16_d32_onehot_table());
        assert_generated_batched_roots_are_scaled::<fp16::D64Full>(fp16_d64_full_table());
        assert_generated_batched_roots_are_scaled::<fp16::D64OneHot>(fp16_d64_onehot_table());
    }

    #[test]
    #[cfg(not(feature = "zk"))]
    fn generated_d32_full_root_fold_matches_runtime_root_plan() {
        assert_exact_root_fold_matches_runtime_root_plan::<fp128::D32Full, 32>(26);
    }

    #[test]
    #[cfg(not(feature = "zk"))]
    fn generated_d64_full_table_materializes_valid_plans() {
        let table = fp128_d64_full_table();
        for entry in table.entries {
            let key = AkitaScheduleLookupKey::new(
                entry.key.num_vars,
                entry.key.num_t_vectors,
                entry.key.num_w_vectors,
                entry.key.num_z_vectors,
            );
            <fp128::D64Full as CommitmentConfig>::schedule_plan(key)
                .expect("config schedule should succeed")
                .expect("entry should exist in generated table");
        }
    }

    #[test]
    #[cfg(not(feature = "zk"))]
    fn generated_table_rejects_sis_family_mismatch() {
        type Cfg = fp128::D64Full;
        let table = fp128_d64_full_table();
        let mismatched = GeneratedScheduleTable {
            sis_family: akita_types::SisModulusFamily::Q32,
            entries: table.entries,
        };
        let entry = mismatched
            .entries
            .iter()
            .find(|entry| entry.key.num_t_vectors == 1)
            .expect("fp128 table should contain singleton rows");
        let key = AkitaScheduleLookupKey::new_with_points(
            entry.key.num_vars,
            entry.key.num_commitment_groups,
            entry.key.num_t_vectors,
            entry.key.num_w_vectors,
            entry.key.num_z_vectors,
        );
        // Drive the planner materializer directly with the mismatched table:
        // `Cfg::schedule_plan` would use the unmodified `Cfg::schedule_table()`,
        // so we bypass it to test the SIS-family mismatch rejection path.
        let err = akita_derive::schedule_plan_from_table::<<Cfg as CommitmentConfig>::Field, _>(
            key,
            mismatched,
            akita_derive::PlanPolicy {
                sis_family: Cfg::sis_modulus_family(),
                ring_dimension: Cfg::D,
                root_decomp: Cfg::decomposition(),
                challenge_field_bits: Cfg::decomposition().field_bits()
                    * Cfg::CHAL_EXT_DEGREE as u32,
                recursive_public_rows: 1,
                extension_opening_width: Cfg::CLAIM_EXT_DEGREE,
                stage1_challenge_config: Cfg::stage1_challenge_config,
                envelope: Cfg::envelope(key.num_vars),
                ring_subfield_norm_bound: Cfg::ring_subfield_embedding_norm_bound(),
            },
        )
        .expect_err("mismatched SIS family must be rejected");
        assert!(
            err.to_string().contains("SIS family mismatch"),
            "unexpected error: {err}"
        );
    }

    #[test]
    #[cfg(not(feature = "zk"))]
    fn adaptive_bounded_plan_matches_runtime_next_w_len() {
        for num_vars in [14, 20, 30] {
            assert_plan_matches_runtime_w_sizes::<fp128::D64Full>(num_vars);
        }
    }

    #[test]
    #[cfg(not(feature = "zk"))]
    fn adaptive_onehot_plan_matches_runtime_next_w_len() {
        for num_vars in [15, 30, 44] {
            assert_plan_matches_runtime_w_sizes::<fp128::D64OneHot>(num_vars);
        }
    }

    #[test]
    #[cfg(not(feature = "zk"))]
    fn batched_root_plan_matches_runtime_next_w_len() {
        let table = fp128_d64_onehot_table();
        let entry = table
            .entries
            .iter()
            .find(|entry| {
                entry.key.num_commitment_groups > 1
                    || entry.key.num_t_vectors > 1
                    || entry.key.num_w_vectors > 1
                    || entry.key.num_z_vectors > 1
            })
            .expect("generated table should contain a non-singleton batched-root row");
        let key = AkitaScheduleLookupKey::new_with_points(
            entry.key.num_vars,
            entry.key.num_commitment_groups,
            entry.key.num_t_vectors,
            entry.key.num_w_vectors,
            entry.key.num_z_vectors,
        );

        assert_plan_matches_runtime_w_sizes_for_key::<fp128::D64OneHot>(key);
    }

    #[test]
    #[cfg(not(feature = "zk"))]
    fn singleton_root_runtime_plan_matches_existing_root_layout() {
        type Cfg = fp128::D64OneHot;

        let incidence = ClaimIncidenceSummary::same_point(30, 1).expect("singleton incidence");
        let runtime = Cfg::get_params_for_prove(&incidence).expect("singleton runtime plan");
        let root_inputs = AkitaScheduleInputs {
            num_vars: 30,
            level: 0,
            current_w_len: 1usize << 30,
        };
        let root_lp = akita_derive::root_level_layout_with_log_basis(
            Cfg::sis_modulus_family(),
            Cfg::D,
            Cfg::decomposition(),
            Cfg::stage1_challenge_config(Cfg::D).unwrap(),
            Cfg::ring_subfield_embedding_norm_bound(),
            root_inputs,
            Cfg::log_basis_at_level(root_inputs).unwrap(),
        )
        .unwrap();
        let Some(akita_types::Step::Fold(runtime_root_step)) = runtime.steps.first() else {
            panic!("singleton schedule should start with a fold");
        };

        assert_eq!(runtime_root_step.params, root_lp);
        assert_eq!(runtime_root_step.current_w_len, 1usize << 30);
        assert_eq!(runtime_root_step.next_w_len % Cfg::D, 0);
    }

    #[test]
    fn recursive_onehot_split_matches_open_digit_witness_count() {
        type Cfg = fp128::D64OneHot;

        // Use the root decomposition basis directly: this test exercises the
        // tight (m, r) split optimizer at a recursive state that is not part of
        // the canonical schedule, so we don't rely on `log_basis_at_level`.
        let log_basis = Cfg::decomposition().log_basis;
        let inputs = AkitaScheduleInputs {
            num_vars: 30,
            level: 1,
            current_w_len: 25_974_272,
        };
        let params =
            crate::proof_optimized::level_params_with_log_basis::<Cfg>(inputs, log_basis).unwrap();
        let decomp =
            recursive_level_decomposition_from_root(Cfg::decomposition(), params.log_basis);
        let num_ring = inputs.current_w_len / params.ring_dimension;
        let lp_12_7 = level_layout_from_params(12, 7, &params, decomp, num_ring).unwrap();
        let lp_11_8 = level_layout_from_params(11, 8, &params, decomp, num_ring).unwrap();
        let w_12_7 = planned_w_ring_element_count::<<Cfg as CommitmentConfig>::Field>(
            Cfg::decomposition().field_bits(),
            &lp_12_7,
        )
        .unwrap();
        let w_11_8 = planned_w_ring_element_count::<<Cfg as CommitmentConfig>::Field>(
            Cfg::decomposition().field_bits(),
            &lp_11_8,
        )
        .unwrap();
        let reduced_vars = (inputs.current_w_len / params.ring_dimension)
            .next_power_of_two()
            .trailing_zeros() as usize;

        assert!(w_12_7 < w_11_8);
        assert_eq!(
            optimal_m_r_split(
                params.a_key.row_len() as u32,
                params.challenge_l1_mass(),
                decomp.log_commit_bound,
                decomp.log_basis,
                reduced_vars,
                num_ring,
                decomp.field_bits(),
            ),
            (12, 7)
        );
    }

    #[test]
    #[cfg(not(feature = "zk"))]
    fn tight_block_len_is_no_larger_than_pow2() {
        for num_vars in [14, 20, 30] {
            let plan = fp128::D64Full::schedule_plan(AkitaScheduleLookupKey::singleton(num_vars))
                .expect("planner should succeed")
                .expect("config should provide a planner");
            for level in plan.fold_levels() {
                let pow2_block = 1usize << level.lp.m_vars;
                assert!(
                    level.lp.block_len <= pow2_block,
                    "block_len {} should be <= 2^m_vars {} at level {} (num_vars={})",
                    level.lp.block_len,
                    pow2_block,
                    level.inputs.level,
                    num_vars
                );
                if level.inputs.level > 0 {
                    let num_ring = level.inputs.current_w_len / level.lp.ring_dimension;
                    let expected_tight = num_ring.div_ceil(level.lp.num_blocks);
                    assert_eq!(
                        level.lp.block_len, expected_tight,
                        "recursive level {} should use tight block_len = ceil({num_ring} / {})",
                        level.inputs.level, level.lp.num_blocks
                    );
                }
            }
        }
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

    /// Full-field `D=128` preset for planner-backed experiments.
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

    /// Binary onehot `D=128` preset for planner-backed experiments.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct D128OneHot;

    impl_fp128_preset!(D128Full, 128, 128, None);
    impl_fp128_preset!(D128OneHot, 128, 1, None);
    impl_fp128_preset!(
        D64Full,
        64,
        128,
        Some(akita_types::generated::fp128_d64_full_table())
    );
    impl_fp128_preset!(
        D64OneHot,
        64,
        1,
        Some(akita_types::generated::fp128_d64_onehot_table())
    );
    impl_fp128_preset!(
        D32Full,
        32,
        128,
        Some(akita_types::generated::fp128_d32_full_table())
    );
    impl_fp128_preset!(
        D32OneHot,
        32,
        1,
        Some(akita_types::generated::fp128_d32_onehot_table())
    );

    /// Concrete fp128 preset selected by a schedule-family query.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Fp128Preset {
        /// Full-field adaptive `D=32` preset.
        D32Full,
        /// Full-field adaptive `D=64` preset.
        D64Full,
        /// Onehot adaptive `D=32` preset.
        D32OneHot,
        /// Binary onehot generated `D=64` preset.
        D64OneHot,
    }

    impl Fp128Preset {
        /// Ring dimension used by this preset.
        pub const fn ring_dimension(self) -> usize {
            match self {
                Self::D32Full | Self::D32OneHot => 32,
                Self::D64Full | Self::D64OneHot => 64,
            }
        }

        /// Whether this preset is onehot-oriented.
        pub const fn is_onehot(self) -> bool {
            matches!(self, Self::D32OneHot | Self::D64OneHot)
        }

        /// Stable human-readable preset name.
        pub const fn name(self) -> &'static str {
            match self {
                Self::D32Full => "D32Full",
                Self::D64Full => "D64Full",
                Self::D32OneHot => "D32OneHot",
                Self::D64OneHot => "D64OneHot",
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

    /// Full-field `D=32` preset for the default fp32 schedule path.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct D32Full;

    /// Onehot `D=32` preset for the default fp32 schedule path.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct D32OneHot;

    /// Full-field `D=64` preset for fp32 crossover profiling.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct D64Full;

    /// Onehot `D=64` preset for fp32 crossover profiling.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct D64OneHot;

    /// Full-field `D=128` preset for planner-backed fp32 experiments.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct D128Full;

    /// Onehot `D=128` preset for planner-backed fp32 experiments.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct D128OneHot;

    /// Full-field `D=256` preset for planner-backed fp32 experiments.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct D256Full;

    /// Onehot `D=256` preset for planner-backed fp32 experiments.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct D256OneHot;

    /// Full-field `D=512` preset for planner-backed fp32 experiments.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct D512Full;

    /// Onehot `D=512` preset for planner-backed fp32 experiments.
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
        Some(akita_types::generated::fp32_d32_table())
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
        Some(akita_types::generated::fp32_d32_onehot_table())
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
        None
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
        None
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
        None
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
        None
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
        None
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
        None
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

    /// Full-field `D=128` preset for planner-backed fp64 experiments.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct D128Full;

    /// Onehot `D=128` preset for planner-backed fp64 experiments.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct D128OneHot;

    /// Full-field `D=256` preset for planner-backed fp64 experiments.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct D256Full;

    /// Onehot `D=256` preset for planner-backed fp64 experiments.
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
        None
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
        None
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
        None
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
        None
    );
}

/// fp16 presets used for production small-field integration and profiling.
pub mod fp16 {
    use super::*;

    /// Base field for the fp16 presets.
    pub type Field = Prime16Offset99;
    /// Degree-8 ring-subfield used for fp16 public claims and Fiat-Shamir challenges.
    pub type ExtensionField = RingSubfieldFp8<Field>;

    /// Full-field `D=32` preset for fp16 production profiling.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct D32Full;

    /// Onehot `D=32` preset for fp16 production profiling.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct D32OneHot;

    /// Full-field `D=64` preset for fp16 comparison profiling.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct D64Full;

    /// Onehot `D=64` preset for fp16 comparison profiling.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct D64OneHot;

    /// Full-field `D=128` preset for planner-backed fp16 experiments.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct D128Full;

    /// Onehot `D=128` preset for planner-backed fp16 experiments.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct D128OneHot;

    /// Full-field `D=256` preset for planner-backed fp16 experiments.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct D256Full;

    /// Onehot `D=256` preset for planner-backed fp16 experiments.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct D256OneHot;

    /// Full-field `D=512` preset for planner-backed fp16 experiments.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct D512Full;

    /// Onehot `D=512` preset for planner-backed fp16 experiments.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct D512OneHot;

    impl_small_field_preset!(
        D32Full,
        Field,
        ExtensionField,
        akita_types::SisModulusFamily::Q16,
        32,
        16,
        16,
        3,
        8,
        vec![-1, 1],
        Some(akita_types::generated::fp16_d32_full_table())
    );
    impl_small_field_preset!(
        D32OneHot,
        Field,
        ExtensionField,
        akita_types::SisModulusFamily::Q16,
        32,
        16,
        1,
        3,
        8,
        vec![-1, 1],
        Some(akita_types::generated::fp16_d32_onehot_table())
    );
    impl_small_field_preset!(
        D64Full,
        Field,
        ExtensionField,
        akita_types::SisModulusFamily::Q16,
        64,
        16,
        16,
        3,
        8,
        vec![-1, 1],
        Some(akita_types::generated::fp16_d64_full_table())
    );
    impl_small_field_preset!(
        D64OneHot,
        Field,
        ExtensionField,
        akita_types::SisModulusFamily::Q16,
        64,
        16,
        1,
        3,
        8,
        vec![-1, 1],
        Some(akita_types::generated::fp16_d64_onehot_table())
    );
    impl_small_field_preset!(
        D128Full,
        Field,
        ExtensionField,
        akita_types::SisModulusFamily::Q16,
        128,
        16,
        16,
        3,
        8,
        vec![-1, 1],
        None
    );
    impl_small_field_preset!(
        D128OneHot,
        Field,
        ExtensionField,
        akita_types::SisModulusFamily::Q16,
        128,
        16,
        1,
        3,
        8,
        vec![-1, 1],
        None
    );
    impl_small_field_preset!(
        D256Full,
        Field,
        ExtensionField,
        akita_types::SisModulusFamily::Q16,
        256,
        16,
        16,
        3,
        8,
        vec![-1, 1],
        None
    );
    impl_small_field_preset!(
        D256OneHot,
        Field,
        ExtensionField,
        akita_types::SisModulusFamily::Q16,
        256,
        16,
        1,
        3,
        8,
        vec![-1, 1],
        None
    );
    impl_small_field_preset!(
        D512Full,
        Field,
        ExtensionField,
        akita_types::SisModulusFamily::Q16,
        512,
        16,
        16,
        3,
        8,
        vec![-1, 1],
        None
    );
    impl_small_field_preset!(
        D512OneHot,
        Field,
        ExtensionField,
        akita_types::SisModulusFamily::Q16,
        512,
        16,
        1,
        3,
        8,
        vec![-1, 1],
        None
    );
}
