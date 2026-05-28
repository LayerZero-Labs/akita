//! Proof-optimized commitment config presets.
//!
//! Presets are unit structs that bind [`CommitmentConfig`] hooks to
//! [`akita_derive`] SIS primitives and generated schedule tables.

use super::{AjtaiRole, CommitmentConfig, CommitmentEnvelope};
use akita_field::AkitaError;
use akita_field::{
    Ext2, Prime128OffsetA7F7, Prime16Offset99, Prime32Offset99, Prime64Offset59, RingSubfieldFp4,
    RingSubfieldFp8,
};
use akita_types::generated::table_entry_envelope_up_to_num_vars;
use akita_types::ClaimIncidenceSummary;
#[cfg(feature = "zk")]
use akita_types::Step;
use akita_types::{
    AkitaPlannedStep, AkitaScheduleInputs, AkitaScheduleLookupKey, AkitaSchedulePlan, LevelParams,
    Schedule, SetupMatrixEnvelope,
};

/// Minimum proof-optimized log-basis.
pub(crate) const PROOF_OPTIMIZED_LOG_BASIS_MIN: u32 = 2;
/// Maximum proof-optimized log-basis.
pub(crate) const PROOF_OPTIMIZED_LOG_BASIS_MAX: u32 = 6;

// ---------------------------------------------------------------------------
// `<Cfg>`-generic policy helpers for the planner and materializer.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Trait-shaped wrappers consumed by the macros below.
// ---------------------------------------------------------------------------

/// Proof-optimized `schedule_plan` impl.
pub(crate) fn proof_optimized_schedule_plan<Cfg>(
    key: AkitaScheduleLookupKey,
    envelope: CommitmentEnvelope,
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
            envelope,
            ring_subfield_norm_bound: Cfg::ring_subfield_embedding_norm_bound(),
            fold_challenge_shape: Cfg::fold_challenge_shape_at_level,
        },
    )
}

/// Lookup level params from the table, or derive SIS-secure fallback params.
///
/// # Errors
///
/// Returns plan-materialization or inner-derivation errors. A table error is
/// hard: falling back would silently disagree with the encoded schedule.
pub fn level_params_with_log_basis<Cfg: CommitmentConfig>(
    inputs: AkitaScheduleInputs,
    log_basis: u32,
) -> Result<LevelParams, AkitaError> {
    let envelope = Cfg::envelope(inputs.num_vars);
    let plan = proof_optimized_schedule_plan::<Cfg>(
        AkitaScheduleLookupKey::singleton(inputs.num_vars),
        envelope,
    )?;
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

/// Proof-optimized `envelope` impl.
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
/// Planned role footprints are not monotone across shapes, so scan all
/// supported sub-shapes and keep the largest packed setup length.
pub(crate) fn proof_optimized_max_setup_matrix_size<Cfg: CommitmentConfig>(
    max_num_vars: usize,
    max_num_batched_polys: usize,
    max_num_points: usize,
) -> Result<SetupMatrixEnvelope, AkitaError> {
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

    let mut max_setup_len: usize = 1;
    let mut saw_supported_shape = false;
    for num_vars in 1..=max_num_vars {
        // Envelope only depends on `num_vars`, so compute it once per
        // outer iteration instead of repeating the table scan inside
        // `Cfg::envelope` for every `(num_polys, num_points)`.
        let envelope = Cfg::envelope(num_vars);
        for num_polys in 1..=max_num_batched_polys {
            let upper_pts = num_polys.min(max_num_points);
            for num_points in 1..=upper_pts {
                let incidence =
                    ClaimIncidenceSummary::from_counts(num_vars, num_polys, num_points)?;
                let Some(envelope) = setup_matrix_envelope_for_shape::<Cfg>(&incidence, envelope)?
                else {
                    continue;
                };
                saw_supported_shape = true;
                max_setup_len = max_setup_len.max(envelope.max_setup_len);
            }
        }
    }

    if !saw_supported_shape {
        return Err(AkitaError::InvalidSetup(format!(
            "setup matrix sizing found no generated schedules for max_num_vars={max_num_vars}"
        )));
    }

    Ok(SetupMatrixEnvelope { max_setup_len })
}

fn setup_matrix_envelope_for_shape<Cfg: CommitmentConfig>(
    incidence: &ClaimIncidenceSummary,
    envelope: CommitmentEnvelope,
) -> Result<Option<SetupMatrixEnvelope>, AkitaError> {
    let cached_key = AkitaScheduleLookupKey::new_from_incidence(incidence)?;

    // Table-only: configs that want a runtime DP fallback override the
    // `max_setup_matrix_size` trait method directly (see `PlannerCfg`).
    // The caller hoisted `envelope` out of the (num_polys, num_points)
    // loop so we skip the table scan that `Cfg::envelope` does on every
    // call.
    let Some(plan) = proof_optimized_schedule_plan::<Cfg>(cached_key, envelope)? else {
        return Ok(None);
    };
    let schedule = akita_types::schedule_from_plan(&plan, Cfg::decomposition().field_bits());

    Ok(Some(matrix_envelope_for_schedule::<Cfg>(
        &schedule, incidence,
    )?))
}

/// Extract setup-level params from a materialized plan.
///
/// Uncommittable root-direct entries carry no setup params and are skipped
/// here; `Cfg::get_params_for_batched_commitment` rejects them loudly.
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

/// Extract setup-level params from a runtime `Schedule`.
///
/// Mirrors [`setup_level_params_from_plan`] for fallback schedules.
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

pub fn matrix_envelope_for_schedule<Cfg>(
    schedule: &Schedule,
    _incidence: &ClaimIncidenceSummary,
) -> Result<SetupMatrixEnvelope, AkitaError>
where
    Cfg: CommitmentConfig,
{
    let setup_levels = setup_level_params_from_runtime_schedule(&schedule.steps);
    let envelope = matrix_envelope_for_levels::<Cfg>(&setup_levels)?;
    #[cfg(feature = "zk")]
    {
        let mut envelope = envelope;
        accumulate_zk_hiding_envelope::<Cfg>(schedule, _incidence, &mut envelope.max_setup_len)?;
        Ok(envelope)
    }
    #[cfg(not(feature = "zk"))]
    {
        Ok(envelope)
    }
}

fn accumulate_matrix_envelope_for_level<Cfg: CommitmentConfig>(
    lp: &LevelParams,
    max_setup_len: &mut usize,
) -> Result<(), AkitaError> {
    let _cfg_marker = core::marker::PhantomData::<Cfg>;
    let outer_width = lp.outer_width();
    #[cfg(feature = "zk")]
    let outer_width = outer_width
        .checked_add(akita_types::zk::blinding_column_count::<Cfg::Field>(
            lp.b_key.row_len(),
            lp.ring_dimension,
            lp.log_basis,
        ))
        .ok_or_else(|| AkitaError::InvalidSetup("ZK B setup envelope overflow".to_string()))?;
    let d_matrix_width = lp.d_matrix_width();
    #[cfg(feature = "zk")]
    let d_matrix_width = d_matrix_width
        .checked_add(akita_types::zk::blinding_column_count::<Cfg::Field>(
            lp.d_key.row_len(),
            lp.ring_dimension,
            lp.log_basis,
        ))
        .ok_or_else(|| AkitaError::InvalidSetup("ZK D setup envelope overflow".to_string()))?;
    let a_len = lp
        .a_key
        .row_len()
        .checked_mul(lp.inner_width())
        .ok_or_else(|| AkitaError::InvalidSetup("A setup envelope overflow".to_string()))?;
    let b_len = lp
        .b_key
        .row_len()
        .checked_mul(outer_width)
        .ok_or_else(|| AkitaError::InvalidSetup("B setup envelope overflow".to_string()))?;
    let d_len = lp
        .d_key
        .row_len()
        .checked_mul(d_matrix_width)
        .ok_or_else(|| AkitaError::InvalidSetup("D setup envelope overflow".to_string()))?;
    *max_setup_len = (*max_setup_len).max(a_len).max(b_len).max(d_len);
    Ok(())
}

#[cfg(feature = "zk")]
fn accumulate_zk_hiding_envelope<Cfg: CommitmentConfig>(
    schedule: &Schedule,
    incidence: &ClaimIncidenceSummary,
    max_setup_len: &mut usize,
) -> Result<(), AkitaError> {
    let Some(root_commit_params) = root_commit_params_from_schedule(schedule)? else {
        return Ok(());
    };
    let hiding_len = zk_hiding_witness_len::<Cfg>(schedule, incidence)?;
    let num_ring = hiding_len.div_ceil(Cfg::D).max(1).next_power_of_two();
    let hiding_params = root_commit_params.with_decomp(
        num_ring.trailing_zeros() as usize,
        0,
        root_commit_params.num_digits_commit,
        root_commit_params.num_digits_open,
        root_commit_params.num_digits_fold,
        num_ring,
    )?;
    accumulate_matrix_envelope_for_level::<Cfg>(&hiding_params, max_setup_len)
}

#[cfg(feature = "zk")]
fn root_commit_params_from_schedule(
    schedule: &Schedule,
) -> Result<Option<LevelParams>, AkitaError> {
    match schedule.steps.first() {
        Some(Step::Fold(root_step)) => Ok(Some(root_step.params.clone())),
        Some(Step::Direct(direct)) => Ok(direct.commit_params.clone()),
        None => Err(AkitaError::InvalidSetup(
            "schedule has no steps".to_string(),
        )),
    }
}

#[cfg(feature = "zk")]
fn zk_hiding_witness_len<Cfg: CommitmentConfig>(
    schedule: &Schedule,
    incidence: &ClaimIncidenceSummary,
) -> Result<usize, AkitaError> {
    let fold_steps = schedule
        .steps
        .iter()
        .filter_map(|step| match step {
            Step::Fold(fold) => Some(fold),
            Step::Direct(_) => None,
        })
        .collect::<Vec<_>>();
    let mut len = 0usize;

    if root_tensor_projection_enabled_for_cfg::<Cfg>(incidence.num_vars()) {
        let split_bits = Cfg::CHAL_EXT_DEGREE.trailing_zeros() as usize;
        let rounds = incidence
            .num_vars()
            .checked_sub(split_bits)
            .ok_or_else(|| AkitaError::InvalidSetup("ZK projection round underflow".to_string()))?;
        let partials = incidence
            .num_claims()
            .checked_mul(Cfg::CHAL_EXT_DEGREE)
            .ok_or_else(|| {
                AkitaError::InvalidSetup("ZK projection partial overflow".to_string())
            })?;
        add_zk_extension_reduction_slots::<Cfg>(&mut len, partials, rounds)?;
    }

    len = len
        .checked_add(
            incidence
                .num_points()
                .checked_mul(Cfg::D)
                .ok_or_else(|| AkitaError::InvalidSetup("ZK root mask overflow".to_string()))?,
        )
        .ok_or_else(|| AkitaError::InvalidSetup("ZK hiding witness overflow".to_string()))?;

    if let Some(root_step) = fold_steps.first() {
        let root_has_stage1 = fold_steps.len() > 1;
        add_zk_level_pad_slots::<Cfg>(
            &mut len,
            &root_step.params,
            root_step.next_w_len,
            root_has_stage1,
        )?;
        if root_has_stage1 {
            add_zk_ext_scalar_slots::<Cfg>(&mut len, 1)?;
        }
        let mut current_opening_vars =
            akita_types::sumcheck_rounds(root_step.params.ring_dimension, root_step.next_w_len);
        for (step_idx, step) in fold_steps.iter().enumerate().skip(1) {
            if Cfg::CHAL_EXT_DEGREE > 1 {
                let split_bits = Cfg::CHAL_EXT_DEGREE.trailing_zeros() as usize;
                let rounds = current_opening_vars
                    .checked_sub(split_bits)
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup(
                            "ZK recursive projection round underflow".to_string(),
                        )
                    })?;
                add_zk_extension_reduction_slots::<Cfg>(&mut len, Cfg::CHAL_EXT_DEGREE, rounds)?;
            }
            len = len.checked_add(Cfg::D).ok_or_else(|| {
                AkitaError::InvalidSetup("ZK recursive mask overflow".to_string())
            })?;
            let include_stage1 = step_idx + 1 < fold_steps.len();
            add_zk_level_pad_slots::<Cfg>(&mut len, &step.params, step.next_w_len, include_stage1)?;
            if include_stage1 {
                add_zk_ext_scalar_slots::<Cfg>(&mut len, 1)?;
            }
            current_opening_vars =
                akita_types::sumcheck_rounds(step.params.ring_dimension, step.next_w_len);
        }
    }

    Ok(len)
}

#[cfg(feature = "zk")]
fn root_tensor_projection_enabled_for_cfg<Cfg: CommitmentConfig>(num_vars: usize) -> bool {
    let width = Cfg::CHAL_EXT_DEGREE;
    let Some(double_width) = width.checked_mul(2) else {
        return false;
    };
    width > 1
        && width == Cfg::CLAIM_EXT_DEGREE
        && width.is_power_of_two()
        && Cfg::D.is_power_of_two()
        && Cfg::D >= double_width
        && Cfg::D.is_multiple_of(width)
        && num_vars >= Cfg::D.trailing_zeros() as usize
}

#[cfg(feature = "zk")]
fn add_zk_level_pad_slots<Cfg: CommitmentConfig>(
    len: &mut usize,
    params: &LevelParams,
    next_w_len: usize,
    include_stage1: bool,
) -> Result<(), AkitaError> {
    let rounds = akita_types::sumcheck_rounds(params.ring_dimension, next_w_len);
    if include_stage1 {
        let b = 1usize
            .checked_shl(params.log_basis)
            .ok_or_else(|| AkitaError::InvalidSetup("ZK stage-1 basis overflow".to_string()))?;
        for shape in akita_types::stage1_tree_stage_shapes(rounds, b) {
            let stored_coeffs = shape.sumcheck_proof.1.max(1);
            add_zk_ext_scalar_slots::<Cfg>(
                len,
                shape
                    .sumcheck_proof
                    .0
                    .checked_mul(stored_coeffs)
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup("ZK stage-1 pad overflow".to_string())
                    })?,
            )?;
            add_zk_ext_scalar_slots::<Cfg>(len, shape.child_claims)?;
        }
    }
    add_zk_ext_scalar_slots::<Cfg>(
        len,
        rounds
            .checked_mul(3)
            .ok_or_else(|| AkitaError::InvalidSetup("ZK stage-2 pad overflow".to_string()))?,
    )
}

#[cfg(feature = "zk")]
fn add_zk_extension_reduction_slots<Cfg: CommitmentConfig>(
    len: &mut usize,
    partials: usize,
    rounds: usize,
) -> Result<(), AkitaError> {
    let reduction_scalars = rounds
        .checked_mul(akita_types::EXTENSION_OPENING_REDUCTION_DEGREE)
        .and_then(|n| n.checked_add(partials))
        .ok_or_else(|| AkitaError::InvalidSetup("ZK extension pad overflow".to_string()))?;
    add_zk_ext_scalar_slots::<Cfg>(len, reduction_scalars)
}

#[cfg(feature = "zk")]
fn add_zk_ext_scalar_slots<Cfg: CommitmentConfig>(
    len: &mut usize,
    scalars: usize,
) -> Result<(), AkitaError> {
    let slots = scalars
        .checked_mul(Cfg::CHAL_EXT_DEGREE)
        .ok_or_else(|| AkitaError::InvalidSetup("ZK scalar pad overflow".to_string()))?;
    *len = len
        .checked_add(slots)
        .ok_or_else(|| AkitaError::InvalidSetup("ZK hiding witness overflow".to_string()))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Per-preset CommitmentConfig macro
// ---------------------------------------------------------------------------

/// Generate a [`CommitmentConfig`] impl for one fp128 preset.
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

            fn schedule_plan(
                key: akita_types::AkitaScheduleLookupKey,
            ) -> Result<Option<akita_types::AkitaSchedulePlan>, akita_field::AkitaError> {
                let envelope = <Self as $crate::CommitmentConfig>::envelope(key.num_vars);
                $crate::proof_optimized::proof_optimized_schedule_plan::<Self>(key, envelope)
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
            ) -> Result<akita_types::SetupMatrixEnvelope, akita_field::AkitaError> {
                $crate::proof_optimized::proof_optimized_max_setup_matrix_size::<Self>(
                    max_num_vars,
                    max_num_batched_polys,
                    max_num_points,
                )
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

            fn schedule_plan(
                key: akita_types::AkitaScheduleLookupKey,
            ) -> Result<Option<akita_types::AkitaSchedulePlan>, akita_field::AkitaError> {
                let envelope = <Self as $crate::CommitmentConfig>::envelope(key.num_vars);
                $crate::proof_optimized::proof_optimized_schedule_plan::<Self>(key, envelope)
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
            ) -> Result<akita_types::SetupMatrixEnvelope, akita_field::AkitaError> {
                $crate::proof_optimized::proof_optimized_max_setup_matrix_size::<Self>(
                    max_num_vars,
                    max_num_batched_polys,
                    max_num_points,
                )
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
mod tests;

// ---------------------------------------------------------------------------
// Public preset structs
// ---------------------------------------------------------------------------

pub mod fp128;
pub mod fp16;
pub mod fp32;
pub mod fp64;
